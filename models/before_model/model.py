"""before_model: the architecture from the hand-drawn diagram.

Two branches share the same two funnel encoders (strain + shared witness),
then diverge completely:

  1. BYPASS heads (for SupCon): each encoder's raw sequence output goes
     through its own attention pool directly to an 8D embedding --
     strain_z and witness_z -- WITHOUT ever touching the shared Mamba
     encoder. (Witness needs two pooling stages: temporal pool per channel,
     then a channel-wise attention pool across the N channels.)

  2. FUSION path (for classification): the SAME raw encoder outputs are
     folded together as 1+N streams into one shared-weight Mamba encoder,
     pooled per-stream, then the (1+N) pooled vectors are concatenated
     (flattened) into one (1+N)*d_model vector and fed to the classifier --
     there is no strain/witness split or re-projection at this stage, per
     the diagram (straight arrow from "Mean/attention pool" to "Classifier").

The two branches are independent given the shared encoder outputs -- the
classifier does not read strain_z/witness_z, and the SupCon losses do not
touch anything from the Mamba path.

`use_witness=False` disables the witness column entirely (true ablation --
the witness encoder/pools/projection are never built, and the Mamba stack
only ever processes 1 stream instead of 1+N), matching the ablation flag
already added to dual_model.py.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


# =====================================================================
# 1. CONV1D FUNNEL ENCODER
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
# 2. POOLING
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
# 3. BEFORE_MODEL
# =====================================================================
class BeforeModel(nn.Module):
    def __init__(self, d_model=32, mamba_layers=4, d_state=16, d_conv=4, expand=2,
                 proj_dim=8, n_wit=4, n_classes=3, pool_method="mean", use_witness=True):
        super().__init__()
        self.d_model = d_model
        self.n_wit = n_wit
        self.use_witness = use_witness

        # Shared encoders (same two funnels feed BOTH branches below).
        self.strain_enc = ConvFunnelEncoder(d_model=d_model)

        if self.use_witness:
            self.witness_enc = ConvFunnelEncoder(d_model=d_model)  # shared across all N witness channels

        # --- Bypass heads: encoder output -> attention pool -> 8D embedding ---
        self.strain_bypass_pool = TemporalPool(d_model, method=pool_method)
        self.strain_proj = nn.Linear(d_model, proj_dim)

        if self.use_witness:
            self.witness_bypass_temporal_pool = TemporalPool(d_model, method=pool_method)  # per-channel
            self.witness_channel_pool = ChannelAttentionPool(d_model)                      # N channels -> 1
            self.witness_proj = nn.Linear(d_model, proj_dim)

        # --- Fusion path: shared Mamba over (1+N) streams -> pool -> classifier ---
        # With use_witness=False there's only 1 stream (strain) per sample.
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(mamba_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(mamba_layers)])
        self.mamba_pool = TemporalPool(d_model, method=pool_method)  # per stream, after Mamba

        n_streams = (1 + n_wit) if use_witness else 1
        classifier_in = n_streams * d_model  # concat of all pooled stream vectors
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def _encode_raw(self, strain, witness=None):
        """Runs the funnel encoder(s) once; raw sequence outputs are reused
        by both the bypass heads and the fusion path below."""
        h_strain = self.strain_enc(strain)  # [B, Tp, d_model]

        if not self.use_witness:
            return h_strain, None, None

        B, T, N = witness.shape
        assert N == self.n_wit, f"Expected n_wit={self.n_wit} witness channels, got N={N}"
        wit_flat = witness.permute(0, 2, 1).reshape(B * N, T, 1)
        h_wit_flat = self.witness_enc(wit_flat)                 # [B*N, Tp, d_model]
        Tp, d_model = h_wit_flat.shape[1], h_wit_flat.shape[2]
        h_wit = h_wit_flat.view(B, N, Tp, d_model)               # [B, N, Tp, d_model]

        return h_strain, h_wit, h_wit_flat  # h_wit_flat reused for the per-channel bypass pool

    def encode_bypass(self, h_strain, h_wit_flat, B, N):
        """encoder output -> attention pool -> 8D projection, for both modalities."""
        strain_pooled = self.strain_bypass_pool(h_strain)         # [B, d_model]
        strain_z = self.strain_proj(strain_pooled)                # [B, proj_dim]

        if not self.use_witness:
            return strain_z, None

        wit_pooled_flat = self.witness_bypass_temporal_pool(h_wit_flat)  # [B*N, d_model]
        wit_pooled = wit_pooled_flat.view(B, N, -1)                      # [B, N, d_model]
        witness_combined = self.witness_channel_pool(wit_pooled)        # [B, d_model]
        witness_z = self.witness_proj(witness_combined)                 # [B, proj_dim]

        return strain_z, witness_z

    def encode_fusion(self, h_strain, h_wit):
        """Fold (1+N) streams into batch dim -> shared Mamba -> pool per
        stream -> concat all streams into one vector (NOT strain/witness
        split -- straight to the classifier, per the diagram)."""
        B = h_strain.shape[0]

        if self.use_witness:
            _, N, Tp, d_model = h_wit.shape
            all_streams = torch.cat([h_strain.unsqueeze(1), h_wit], dim=1)  # [B, 1+N, Tp, d_model]
            S = 1 + N
        else:
            Tp, d_model = h_strain.shape[1], h_strain.shape[2]
            all_streams = h_strain.unsqueeze(1)  # [B, 1, Tp, d_model]
            S = 1

        streams_flat = all_streams.reshape(B * S, Tp, d_model)

        for norm, block in zip(self.mamba_norms, self.mamba_blocks):
            streams_flat = streams_flat + block(norm(streams_flat))

        pooled_flat = self.mamba_pool(streams_flat)   # [B*S, d_model]
        pooled = pooled_flat.view(B, S * d_model)      # [B, S*d_model] -- flattened, concatenated
        return pooled

    def forward(self, strain, witness=None):
        h_strain, h_wit, h_wit_flat = self._encode_raw(strain, witness)
        B = strain.shape[0]
        N = self.n_wit if self.use_witness else 0

        strain_z, witness_z = self.encode_bypass(h_strain, h_wit_flat, B, N)
        fused = self.encode_fusion(h_strain, h_wit)
        logits = self.classifier(fused)

        return logits, strain_z, witness_z