"""cmamba_mambassm_model: same architecture as cmamba_model.py, except the
per-channel selective-scan mixer is mamba_ssm.Mamba (CUDA kernel, includes a
causal depthwise conv1d) instead of the from-scratch MambaBlock ported from
the official CMamba repo.

This is the "keep mamba_ssm, only fix the surrounding structure" option
described (but not built) when cmamba_model.py was rewritten. Everything
that isn't the mixer itself -- RMSNorm pre-norm, GDD-MLP (faithful port of
layers/GDDMLP.py), the fixed residual (no gated/dynamic skip -- the old
"MMamba" wrapper's learned gate has been removed here too, since official
CMamba doesn't have one either way), channel mixup, pooling, projection
heads, classifier, ablation variants -- is copied unchanged from
cmamba_model.py. See that file's module docstring for the full rationale
behind each piece.

Trade-off vs. cmamba_model.py: this mixer is not what the official CMamba
paper actually runs (it has an extra conv1d step CMamba's own MambaBlock
doesn't have, and per-d_ff-channel A instead of CMamba's shared [1,d_state]
A), but it's faster (CUDA kernel) and has no dependency on pscan.py's
pure-PyTorch parallel scan. Use this file if you want that speed/dependency
trade rather than exact fidelity; use cmamba_model.py if you want to match
the paper's released code as closely as possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================================
# 1. CONV1D FUNNEL ENCODER (unchanged from cmamba_model.py)
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
# 2. POOLING (unchanged from cmamba_model.py)
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
# 3. CMAMBA CORE: mamba_ssm.Mamba mixer + GDD-MLP (GDD-MLP and the fixed
#    residual are ported from the official repo, same as cmamba_model.py --
#    only the mixer itself differs)
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GDDMLP(nn.Module):
    """Global data-dependent channel mixing, ported from layers/GDDMLP.py
    (identical to the copy in cmamba_model.py). Pools each (channel,
    timestep) over the feature dimension (d_model) down to a scalar (avg
    and/or max), then runs two shared bottleneck MLPs across the channel
    dimension -- one for scale, one for shift -- so every channel's gate at
    every timestep is a function of every other channel's pooled
    descriptor at that timestep."""
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


class MambaMixer(nn.Module):
    """Per-channel temporal mixer using mamba_ssm.Mamba in place of
    cmamba_model.py's from-scratch MambaBlock. Note this pulls in a causal
    depthwise conv1d (Mamba's d_conv) that the official CMamba mixer does
    not have, and uses one A per d_ff channel rather than CMamba's shared
    [1, d_state] A -- so this is not a byte-for-byte match to the paper's
    mixer, just a drop-in replacement with the same [B', T, d_model] ->
    [B', T, d_model] interface."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):  # x: [B', T, d_model]
        return self.mamba(x)


class CMambaBlock(nn.Module):
    """RMSNorm -> Mamba mixer -> [GDD-MLP] -> dropout -> fixed residual add.
    Same structure as cmamba_model.py's CMambaBlock (no gated/dynamic skip
    -- that was an approximation in an earlier version of this codebase,
    removed for both variants since official CMamba doesn't have one).
    use_gdd_mlp is the only ablation toggle."""
    def __init__(self, d_model, n_channels, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, use_gdd_mlp=True, reduction=4, gdd_avg=True, gdd_max=True):
        super().__init__()
        self.use_gdd_mlp = use_gdd_mlp

        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
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
    official CMambaEncoder.forward (same as cmamba_model.py)."""
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
    """Official CMamba channel mixup (identical to cmamba_model.py):
    batch_x = batch_x + batch_x[:, :, perm] * N(0, sigma), where perm
    permutes channels within each sample (not across the batch). Restricted
    to witness channels (channel_start=1); strain is never a mixup source
    or target."""
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
                 d_state=16, d_conv=4, expand=2, reduction=4, proj_dim=8,
                 pool_method="mean", classify_from_projection=True,
                 channel_mixup=False, channel_mixup_sigma=1.0,
                 use_gdd_mlp=True, dropout=0.0):
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
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, use_gdd_mlp=use_gdd_mlp, reduction=reduction,
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
ABLATION_VARIANTS = {
    "plain_mamba": dict(use_gdd_mlp=False),  # baseline: plain per-channel mamba_ssm.Mamba, no cross-channel mixing
    "cmamba":      dict(use_gdd_mlp=True),   # + GDD-MLP
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
