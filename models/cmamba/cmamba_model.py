"""cmamba_model: strain + witness architecture with CMamba-style cross-channel
mixing, replacing before_model.py's plain shared Mamba stack.

Diagram this implements:

  Strain [B,4096,1] --funnel--> h_strain [B,Tp,d]  --\
                                                        > (1+N) channels --> CMamba blocks --> per-channel pool
  Witness [B,4096,N] --funnel (shared)--> h_wit [B,N,Tp,d] --/                (Mamba + GDD-MLP), e_layers deep

  channel 0 (strain) pooled  -> Strain Projection 8D -> Supcon_s + CE_s
  channels 1..N (witness) pooled -> channel-attention pool -> Witness projection 8D -> Supcon_w

  L = supcon_s_weight * Supcon_s + ce_weight * CE_s + supcon_w_weight * Supcon_w

Why CMamba instead of plain Mamba: vanilla Mamba is channel-independent --
it has no mechanism for one channel's representation to be shaped by
another's. That's a problem here specifically because "does the witness
channel correlate with the strain channel" is the entire question this model
needs to answer (glitch vs. blip hinges on exactly that coincidence).

This version ports the official CMamba repo (Zeng et al., arXiv:2406.05316,
cloned locally) module-for-module rather than approximating it from the
paper text:

  - MambaBlock: CMamba's own hand-rolled selective scan (in_proj -> x,z;
    x_proj -> dt,B,C,D; dt_proj; A_log; selective_scan via pscan.py or a
    sequential fallback), ported from layers/CMambaEncoder.py. Unlike
    mamba_ssm.Mamba, there is no depthwise causal conv1d step -- the
    official block doesn't have one. Also note A_log has shape [1, d_state]
    in the official code (one shared set of decay rates, not one per
    d_ff channel) -- that's their actual released code, not a simplification
    made here.
  - CMambaBlock residual: RMSNorm -> mixer -> [GDD-MLP] -> dropout -> fixed
    residual add. The official repo does not use a learned/gated skip
    connection; an earlier version of this file did (as an approximation)
    and it has been removed for fidelity.
  - GDDMLP: ported from layers/GDDMLP.py. Pools each channel's sequence
    over the *feature* dim (d_model) with adaptive avg/max pooling to a
    per-(channel, timestep) scalar, then runs two shared bottleneck MLPs
    (fc_sc for scale, fc_sf for shift) that mix across the channel
    dimension at every timestep, and applies sigmoid(scale)*x +
    sigmoid(shift). This -- not a per-channel time-pooled descriptor fed to
    one big MLP, which is what this file used to do -- is where CMamba's
    cross-channel correlation is actually modeled.

Channel Mixup (optional, OFF by default): the official training loop mixes
each channel with a *different, randomly permuted channel of the same
sample* (batch_x = batch_x + batch_x[:, :, perm] * N(0, sigma)), not across
the batch. That's reproduced here, restricted to witness channels only
(strain, index 0, is never a mixup source or target, so it can't dilute
Supcon_s/CE_s). Because the permutation never crosses the batch dimension,
it also can't leak detector identity between samples, so no extra
same-detector restriction is needed -- witness channel c of a sample only
ever gets mixed with another witness channel of that *same* sample.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


# =====================================================================
# 1. CONV1D FUNNEL ENCODER (unchanged from before_model.py)
# =====================================================================
class ConvDownBlock(nn.Module):
    def __init__(self, d_model, kernel_size=4, stride=2):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvResBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class ConvFunnelEncoder(nn.Module):
    def __init__(self, d_model=32, n_down_layers=2, n_res_blocks=2, kernel_size=5):
        super().__init__()
        self.in_proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.down_blocks = nn.ModuleList([ConvDownBlock(d_model) for _ in range(n_down_layers)])
        self.res_blocks = nn.ModuleList([
            ConvResBlock(d_model, kernel_size=kernel_size, dilation=2 ** i) for i in range(n_res_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B, T, 1]
        x = x.permute(0, 2, 1)
        x = self.in_proj(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.res_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # [B, T', d_model]
        return self.norm(x)


# =====================================================================
# 2. POOLING (unchanged from before_model.py)
# =====================================================================
class TemporalPool(nn.Module):
    """Mean or attention pooling over the temporal dimension: [B, T, d_model] -> [B, d_model]."""
    def __init__(self, d_model, method="mean"):
        super().__init__()
        self.method = method
        if method == "attention":
            self.attn = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, x):
        if self.method == "attention":
            w = torch.softmax(self.attn(x), dim=1)
            return (x * w).sum(dim=1)
        return x.mean(dim=1)


class ChannelAttentionPool(nn.Module):
    """Attention pool across N channel vectors: [B, N, d_model] -> [B, d_model]."""
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            return x.squeeze(1)
        scores = self.attn(x)                 # [B, N, 1]
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)        # [B, d_model]


# =====================================================================
# 3. CMAMBA CORE: selective-scan Mamba block + GDD-MLP, ported from the
#    official repo's layers/CMambaEncoder.py and layers/GDDMLP.py
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    """CMamba's own selective-scan block (no depthwise conv, unlike
    mamba_ssm.Mamba). Operates on [B', L, d_model] where B' folds batch and
    channel together, exactly like the official code, since channels are
    processed independently by the SSM itself (cross-channel mixing is
    GDD-MLP's job, applied around this block in CMambaBlock)."""
    def __init__(self, d_model, d_ff, d_state=16, dt_rank=None,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, bias=True, use_pscan=True):
        super().__init__()
        self.d_state = d_state
        self.d_ff = d_ff
        self.dt_rank = dt_rank if dt_rank is not None else max(d_ff // 4, 1)
        self.use_pscan = use_pscan

        # projects block input from d_model to 2*d_ff (two branches)
        self.in_proj = nn.Linear(d_model, 2 * d_ff, bias=bias)

        # projects x to input-dependent dt, B, C, D
        self.x_proj = nn.Linear(d_ff, self.dt_rank + 2 * d_state + d_ff, bias=False)

        # projects dt from dt_rank to d_ff
        self.dt_proj = nn.Linear(self.dt_rank, d_ff, bias=True)

        # dt initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_ff) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization -- note: shape [1, d_state], shared across
        # all d_ff channels. That's the official code, not a simplification
        # made here (vanilla Mamba instead uses one A per d_ff channel).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))

        # projects block output from d_ff back to d_model
        self.out_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):  # x: [B', L, d_model] -> [B', L, d_model]
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = F.silu(x)
        y = self.ssm(x)

        z = F.silu(z)
        output = y * z
        return self.out_proj(output)

    def ssm(self, x):  # x: [B', L, d_ff]
        A = -torch.exp(self.A_log.float())  # [1, d_state]

        deltaBCD = self.x_proj(x)
        delta, B, C, D = torch.split(
            deltaBCD, [self.dt_rank, self.d_state, self.d_state, self.d_ff], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))

        if self.use_pscan:
            return self.selective_scan(x, delta, A, B, C, D)
        return self.selective_scan_seq(x, delta, A, B, C, D)

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)          # [B', L, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)         # [B', L, d_ff, d_state]
        BX = deltaB * x.unsqueeze(-1)                         # [B', L, d_ff, d_state]

        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)

        h = torch.zeros(x.size(0), self.d_ff, self.d_state, device=deltaA.device)
        hs = []
        for t in range(L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x


class GDDMLP(nn.Module):
    """Global data-dependent channel mixing, ported from layers/GDDMLP.py.
    Pools each (channel, timestep) over the *feature* dimension (d_model)
    down to a scalar (avg and/or max), then runs two shared bottleneck MLPs
    across the channel dimension -- one for scale, one for shift -- so every
    channel's gate at every timestep is a function of every other channel's
    pooled descriptor at that timestep. This is the actual cross-channel
    mixing mechanism in CMamba."""
    def __init__(self, n_channels, reduction=4, avg_flag=True, max_flag=True):
        super().__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden = max(n_channels // reduction, 1)
        self.fc_sc = nn.Sequential(
            nn.Linear(n_channels, hidden, bias=False), nn.GELU(), nn.Linear(hidden, n_channels, bias=False)
        )
        self.fc_sf = nn.Sequential(
            nn.Linear(n_channels, hidden, bias=False), nn.GELU(), nn.Linear(hidden, n_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, C, T, d_model]
        b, n, p, d = x.shape
        scale = torch.zeros_like(x)
        shift = torch.zeros_like(x)
        if self.avg_flag:
            pooled = self.avg_pool(x.reshape(b * n, p, d)).reshape(b, n, p).permute(0, 2, 1)  # [B, T, C]
            scale += self.fc_sc(pooled).permute(0, 2, 1).unsqueeze(-1)
            shift += self.fc_sf(pooled).permute(0, 2, 1).unsqueeze(-1)
        if self.max_flag:
            pooled = self.max_pool(x.reshape(b * n, p, d)).reshape(b, n, p).permute(0, 2, 1)
            scale += self.fc_sc(pooled).permute(0, 2, 1).unsqueeze(-1)
            shift += self.fc_sf(pooled).permute(0, 2, 1).unsqueeze(-1)
        return self.sigmoid(scale) * x + self.sigmoid(shift)


class CMambaBlock(nn.Module):
    """RMSNorm -> Mamba mixer -> [GDD-MLP] -> dropout -> fixed residual add.
    use_gdd_mlp is the only ablation toggle now -- the official repo doesn't
    have a data-dependent/gated residual, so an earlier "dynamic_skip"
    approximation has been removed. With use_gdd_mlp=False this is a plain
    per-channel Mamba block; the difference from that baseline to full
    CMamba is entirely GDD-MLP."""
    def __init__(self, d_model, n_channels, d_ff=None, d_state=16, dt_rank=None,
                 dropout=0.0, use_gdd_mlp=True, reduction=4, gdd_avg=True, gdd_max=True,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, bias=True, use_pscan=True):
        super().__init__()
        d_ff = d_model if d_ff is None else d_ff
        self.use_gdd_mlp = use_gdd_mlp

        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(
            d_model=d_model, d_ff=d_ff, d_state=d_state, dt_rank=dt_rank,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
            dt_init_floor=dt_init_floor, bias=bias, use_pscan=use_pscan,
        )
        if use_gdd_mlp:
            self.gddmlp = GDDMLP(n_channels, reduction=reduction, avg_flag=gdd_avg, max_flag=gdd_max)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T, D] -> [B, C, T, D]
        B, C, T, D = x.shape
        flat = x.reshape(B * C, T, D)

        out = self.mixer(self.norm(flat))
        if self.use_gdd_mlp:
            out = self.gddmlp(out.reshape(B, C, T, D)).reshape(B * C, T, D)
        out = self.dropout(out)
        out = out + flat
        return out.reshape(B, C, T, D)


class CMambaEncoder(nn.Module):
    """Stack of e_layers CMambaBlocks followed by SiLU, matching the
    official CMambaEncoder.forward exactly."""
    def __init__(self, d_model, n_channels, e_layers, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            CMambaBlock(d_model=d_model, n_channels=n_channels, **block_kwargs)
            for _ in range(e_layers)
        ])

    def forward(self, x):  # x: [B, C, T, D]
        for layer in self.layers:
            x = layer(x)
        return F.silu(x)


class ChannelMixup(nn.Module):
    """Official CMamba channel mixup: batch_x = batch_x + batch_x[:, :,
    perm] * N(0, sigma), where perm permutes channels *within each sample*
    (not across the batch). Restricted here to witness channels
    (channel_start=1): strain is excluded from both the permutation and the
    mixing target so it can't dilute Supcon_s/CE_s. Because the permutation
    never crosses the batch dimension, this also can't leak detector
    identity between samples -- no extra same-detector guard is needed."""
    def __init__(self, sigma=1.0, channel_start=1):
        super().__init__()
        self.sigma = sigma
        self.channel_start = channel_start

    def forward(self, x):  # x: [B, C, T, D]
        B, C = x.shape[0], x.shape[1]
        s = self.channel_start
        n_mix = C - s
        if n_mix < 2:
            return x

        perm = s + torch.randperm(n_mix, device=x.device)
        mix_coef = torch.normal(mean=0.0, std=self.sigma, size=(B, n_mix), device=x.device)
        mix_coef = mix_coef.view(B, n_mix, 1, 1)

        mixed = x.clone()
        mixed[:, s:] = x[:, s:] + x[:, perm] * mix_coef
        return mixed


# =====================================================================
# 4. CMAMBA MODEL
# =====================================================================
class CMambaModel(nn.Module):
    def __init__(self, d_model=32, n_wit=5, n_classes=4, e_layers=3,
                 d_state=16, expand=1, dt_rank=None, reduction=4, proj_dim=8,
                 pool_method="mean", classify_from_projection=True,
                 channel_mixup=False, channel_mixup_sigma=1.0,
                 use_gdd_mlp=True, dropout=0.0, use_pscan=True,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, bias=True):
        super().__init__()
        self.n_wit = n_wit
        self.n_channels = 1 + n_wit
        self.d_model = d_model
        self.classify_from_projection = classify_from_projection
        self.use_gdd_mlp = use_gdd_mlp

        self.strain_enc = ConvFunnelEncoder(d_model=d_model)
        self.witness_enc = ConvFunnelEncoder(d_model=d_model)  # shared across all N witness channels

        self.encoder = CMambaEncoder(
            d_model=d_model, n_channels=self.n_channels, e_layers=e_layers,
            d_ff=d_model * expand, d_state=d_state, dt_rank=dt_rank,
            dropout=dropout, use_gdd_mlp=use_gdd_mlp, reduction=reduction,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
            dt_init_floor=dt_init_floor, bias=bias, use_pscan=use_pscan,
        )

        self.channel_mixup = ChannelMixup(sigma=channel_mixup_sigma) if channel_mixup else None

        self.pool = TemporalPool(d_model, method=pool_method)              # per channel, after CMamba blocks
        self.witness_channel_pool = ChannelAttentionPool(d_model)          # N witness pooled vecs -> 1

        self.strain_proj = nn.Linear(d_model, proj_dim)
        self.witness_proj = nn.Linear(d_model, proj_dim)

        classifier_in = proj_dim if classify_from_projection else d_model
        self.classifier = nn.Linear(classifier_in, n_classes)

    def _encode_raw(self, strain, witness):
        h_strain = self.strain_enc(strain)  # [B, Tp, d]

        B, T, N = witness.shape
        assert N == self.n_wit, f"Expected n_wit={self.n_wit} witness channels, got N={N}"
        wit_flat = witness.permute(0, 2, 1).reshape(B * N, T, 1)
        h_wit_flat = self.witness_enc(wit_flat)                # [B*N, Tp, d]
        Tp, d = h_wit_flat.shape[1], h_wit_flat.shape[2]
        h_wit = h_wit_flat.view(B, N, Tp, d)                    # [B, N, Tp, d]
        return h_strain, h_wit

    def forward(self, strain, witness):
        h_strain, h_wit = self._encode_raw(strain, witness)
        B = strain.shape[0]

        streams = torch.cat([h_strain.unsqueeze(1), h_wit], dim=1)  # [B, C, Tp, d]

        if self.channel_mixup is not None and self.training:
            streams = self.channel_mixup(streams)

        streams = self.encoder(streams)

        Tp = streams.shape[2]
        pooled = self.pool(streams.reshape(B * self.n_channels, Tp, self.d_model))
        pooled = pooled.view(B, self.n_channels, self.d_model)   # [B, C, d]

        strain_feat = pooled[:, 0, :]                             # [B, d]
        witness_feats = pooled[:, 1:, :]                          # [B, N, d]
        witness_feat = self.witness_channel_pool(witness_feats)   # [B, d]

        strain_z = self.strain_proj(strain_feat)                  # [B, proj_dim]
        witness_z = self.witness_proj(witness_feat)               # [B, proj_dim]

        cls_in = strain_z if self.classify_from_projection else strain_feat
        logits = self.classifier(cls_in)

        return logits, strain_z, witness_z


# =====================================================================
# 5. ABLATION FACTORY
# =====================================================================
# GDD-MLP is now the only CMamba-specific toggle (the official repo has no
# gated/data-dependent residual), so the ablation is exactly the paper's
# comparison: plain per-channel Mamba vs. full CMamba.
ABLATION_VARIANTS = {
    "plain_mamba": dict(use_gdd_mlp=False),  # baseline: plain per-channel Mamba, no cross-channel mixing
    "cmamba":      dict(use_gdd_mlp=True),   # full CMamba: + GDD-MLP
}


def build_model(variant="cmamba", **kwargs):
    """kwargs are the usual CMambaModel constructor args (d_model, n_wit,
    n_classes, e_layers, ...). variant picks whether GDD-MLP is on; passing
    use_gdd_mlp directly in kwargs also works and overrides the variant's
    default if both are given."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ABLATION_VARIANTS)}")
    settings = dict(ABLATION_VARIANTS[variant])
    settings.update(kwargs)
    return CMambaModel(**settings)
