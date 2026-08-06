import torch
import torch.nn as nn
from mamba_ssm import Mamba

# =====================================================================
# 1. CONV1D FUNNEL ENCODER (unchanged)
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
    def __init__(self, d_model=32, n_down_layers=4, n_res_blocks=2, kernel_size=5):
        super().__init__()
        self.in_proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.down_blocks = nn.ModuleList([ConvDownBlock(d_model) for _ in range(n_down_layers)])
        self.res_blocks = nn.ModuleList([
            ConvResBlock(d_model, kernel_size=kernel_size, dilation=2 ** i) for i in range(n_res_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.in_proj(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.res_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        return self.norm(x)


# =====================================================================
# 2. POOLING
# =====================================================================
class WitnessChannelAttentionPool(nn.Module):
    """Attention pool across the N witness-channel VECTORS (post-Mamba,
    post-temporal-pool) to reduce N vectors down to 1. Operates on
    [B, N, d_model]. Used only on the SupCon side (see note in
    MambaFusionSupConModel)."""
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, witness_vectors):  # [B, N, d_model]
        if witness_vectors.shape[1] == 1:
            return witness_vectors.squeeze(1)
        scores = self.attn(witness_vectors)             # [B, N, 1]
        weights = torch.softmax(scores, dim=1)
        return (witness_vectors * weights).sum(dim=1)    # [B, d_model]


class TemporalPool(nn.Module):
    """Mean or attention pooling over the temporal dimension. Applied to
    EVERY stream independently, since streams are folded into the batch dim
    before this runs."""
    def __init__(self, d_model, method="mean"):
        super().__init__()
        self.method = method
        if method == "attention":
            self.attn = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, x):  # x: [B, T, d_model]
        if self.method == "attention":
            w = torch.softmax(self.attn(x), dim=1)
            return (x * w).sum(dim=1)
        return x.mean(dim=1)


class BiMambaBlock(nn.Module):
    """Bidirectional wrapper: runs forward and backward SSMs in parallel and
    projects the concatenation back down to d_model."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        out_fwd = self.fwd_mamba(x)
        out_bwd = self.bwd_mamba(torch.flip(x, dims=[1]))
        out_bwd = torch.flip(out_bwd, dims=[1])
        merged = torch.cat([out_fwd, out_bwd], dim=-1)
        return self.proj(merged)


# =====================================================================
# 3. FULL MODEL -- shared-trunk architecture (single shared-weight Mamba
# stack processes all 1+N streams via batch folding), with an optional
# witness branch that can be fully disabled for ablation
# (`use_witness=False`), and an optional bidirectional Mamba
# (`bidirectional=True`) also ablatable.
#
# POOLING IS NOW INDEPENDENT PER BRANCH (this is the key change from the
# earlier version): the same post-Mamba `streams_flat` tensor is pooled
# TWICE, by two separately-parameterized TemporalPool instances:
#
#   - `temporal_pool` -> strain_vec / witness_vec -> strain_proj/witness_proj
#     -> strain_z/witness_z. These proj_dim embeddings feed ONLY the SupCon
#     losses (and the auxiliary strain-only CE head).
#   - `cls_temporal_pool` -> per-stream d_model vectors, concatenated
#     (flattened) across all 1+N streams -> fed DIRECTLY to the classifier,
#     at full d_model resolution, with no dependency on strain_z/witness_z.
#
# This mirrors before_model.py's split between its "bypass" heads (SupCon
# only) and its "fusion" path (classifier only) -- the difference is that
# here both branches read the SAME shared-trunk Mamba output, whereas in
# before_model the bypass branch never touches Mamba at all.
# =====================================================================
class MambaFusionSupConModel(nn.Module):
    def __init__(self, d_model=32, mamba_layers=4, d_state=16, d_conv=4, expand=2,
                 proj_dim=8, pool_method="mean", n_wit=1, n_classes=3, use_witness=True,
                 bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.n_wit = n_wit
        self.use_witness = use_witness
        self.bidirectional = bidirectional

        self.strain_enc = ConvFunnelEncoder(d_model=d_model)

        if self.use_witness:
            self.witness_enc = ConvFunnelEncoder(d_model=d_model)  # shared across all N witness channels
            self.witness_channel_pool = WitnessChannelAttentionPool(d_model=d_model)  # SupCon side only
            self.witness_proj = nn.Linear(d_model, proj_dim)

        # ONE shared-weight Mamba stack, applied per-stream via batch folding.
        # With use_witness=False there's only 1 stream (strain) per sample.
        if bidirectional:
            self.mamba_blocks = nn.ModuleList([
                BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(mamba_layers)
            ])
        else:
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(mamba_layers)
            ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(mamba_layers)])

        # --- SupCon-side pooling (-> strain_z / witness_z / aux head) ---
        self.temporal_pool = TemporalPool(d_model, method=pool_method)
        self.strain_proj = nn.Linear(d_model, proj_dim)

        # --- Classifier-side pooling (independent params, no SupCon gradient) ---
        self.cls_temporal_pool = TemporalPool(d_model, method=pool_method)

        n_streams = (1 + n_wit) if use_witness else 1
        classifier_in = n_streams * d_model  # concat of ALL pooled streams, full d_model each
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

        # Auxiliary strain-only classifier head: CE on strain_z alone,
        # trained alongside the main CE on the fused embedding. This is the
        # aux-loss design from the original (pre-SupCon) model. Still reads
        # strain_z (the SupCon-side projection), not the classifier pooling.
        self.aux_strain_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, n_classes),
        )

    def _run_shared_trunk(self, strain, witness=None):
        """Runs both funnel encoders and the single shared-weight Mamba
        stack once. Returns streams_flat [B*S, Tp, d_model], plus B and S,
        so callers can pool it however they need without recomputing
        Mamba."""
        h_strain = self.strain_enc(strain)  # [B, Tp, d_model]
        B = h_strain.shape[0]

        if self.use_witness:
            _, T, N = witness.shape
            assert N == self.n_wit, f"Expected n_wit={self.n_wit} witness channels, got N={N}"

            wit_flat = witness.permute(0, 2, 1).reshape(B * N, T, 1)
            h_wit_flat = self.witness_enc(wit_flat)               # [B*N, Tp, d_model]
            Tp, d_model = h_wit_flat.shape[1], h_wit_flat.shape[2]
            h_wit = h_wit_flat.view(B, N, Tp, d_model)             # [B, N, Tp, d_model]

            all_streams = torch.cat([h_strain.unsqueeze(1), h_wit], dim=1)  # [B, 1+N, Tp, d_model]
            S = 1 + N
        else:
            all_streams = h_strain.unsqueeze(1)  # [B, 1, Tp, d_model]
            S = 1

        Tp, d_model = all_streams.shape[2], all_streams.shape[3]
        streams_flat = all_streams.reshape(B * S, Tp, d_model)

        for norm, block in zip(self.mamba_norms, self.mamba_blocks):
            streams_flat = streams_flat + block(norm(streams_flat))

        return streams_flat, B, S, d_model

    def encode_features(self, strain, witness=None):
        """Returns everything needed downstream from a single shared-trunk
        pass: (strain_vec, witness_vec, cls_embed).

        strain_vec / witness_vec: [B, d_model] each, from the SupCon-side
        pooling (temporal_pool + witness_channel_pool). witness_vec is None
        if use_witness=False.

        cls_embed: [B, n_streams*d_model], from the INDEPENDENT
        cls_temporal_pool -- every stream pooled and concatenated, with no
        path back to strain_vec/witness_vec.
        """
        streams_flat, B, S, d_model = self._run_shared_trunk(strain, witness)

        # --- SupCon-side pooling ---
        pooled_flat = self.temporal_pool(streams_flat)      # [B*S, d_model]
        pooled = pooled_flat.view(B, S, d_model)             # [B, S, d_model]
        strain_vec = pooled[:, 0, :]                         # [B, d_model]
        if self.use_witness:
            witness_vecs = pooled[:, 1:, :]                  # [B, N, d_model]
            witness_vec = self.witness_channel_pool(witness_vecs)  # [B, d_model]
        else:
            witness_vec = None

        # --- Classifier-side pooling (independent instance/params) ---
        cls_pooled_flat = self.cls_temporal_pool(streams_flat)  # [B*S, d_model]
        cls_embed = cls_pooled_flat.view(B, S * d_model)          # [B, S*d_model], all streams concatenated

        return strain_vec, witness_vec, cls_embed

    def forward(self, strain, witness=None):
        """Returns (strain_z, witness_z) -- the proj_dim SupCon embeddings.
        Unchanged signature so corner-plot / embedding-inspection code keeps
        working as-is."""
        strain_vec, witness_vec, _ = self.encode_features(strain, witness)
        strain_z = self.strain_proj(strain_vec)
        witness_z = self.witness_proj(witness_vec) if self.use_witness else None
        return strain_z, witness_z

    def classify(self, strain, witness=None):
        """Classifier reads the INDEPENDENT cls_embed pooling (no SupCon
        gradient dependency). aux_strain_head still reads strain_z (the
        SupCon-side projection), per the original aux-loss design. Returns
        (main_logits, aux_logits, strain_z, witness_z) -- strain_z/witness_z
        are returned so callers (e.g. the Lightning module) can compute the
        SupCon losses without re-running the shared trunk a second time."""
        strain_vec, witness_vec, cls_embed = self.encode_features(strain, witness)
        strain_z = self.strain_proj(strain_vec)
        witness_z = self.witness_proj(witness_vec) if self.use_witness else None

        main_logits = self.classifier(cls_embed)
        aux_logits = self.aux_strain_head(strain_z)
        return main_logits, aux_logits, strain_z, witness_z