"""baseline: the simplest model in this repo. 2 conv down-sampling layers
per funnel encoder, a stack of plain mamba_ssm.Mamba blocks over the
1+n_wit streams (folded into the batch dim, weights shared across
streams), mean pooling, and a single linear classifier -- cross-entropy
only
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba



# 1. CONV1D FUNNEL ENCODER

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

# 2. POOLING

class TemporalPool(nn.Module):
    """Mean pooling over the temporal dimension: [B, T, d_model] -> [B, d_model]."""
    def forward(self, x):
        return x.mean(dim=1)


# 3. BASELINE MODEL

class BaselineModel(nn.Module):
    def __init__(self, d_model=32, mamba_layers=2, d_state=16, d_conv=4, expand=2,
                 n_wit=5, n_classes=3, use_witness=True):
        super().__init__()
        self.d_model = d_model
        self.n_wit = n_wit
        self.use_witness = use_witness

        self.strain_enc = ConvFunnelEncoder(d_model=d_model)
        if use_witness:
            self.witness_enc = ConvFunnelEncoder(d_model=d_model)  # shared across all N witness channels

        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(mamba_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(mamba_layers)])
        self.pool = TemporalPool()

        n_streams = (1 + n_wit) if use_witness else 1
        self.classifier = nn.Linear(n_streams * d_model, n_classes)

    def forward(self, strain, witness=None):
        h_strain = self.strain_enc(strain)  # [B, Tp, d_model]
        B = strain.shape[0]

        if self.use_witness:
            _, T, N = witness.shape
            assert N == self.n_wit, f"Expected n_wit={self.n_wit} witness channels, got N={N}"
            wit_flat = witness.permute(0, 2, 1).reshape(B * N, T, 1)
            h_wit_flat = self.witness_enc(wit_flat)                    # [B*N, Tp, d_model]
            Tp, d_model = h_wit_flat.shape[1], h_wit_flat.shape[2]
            h_wit = h_wit_flat.view(B, N, Tp, d_model)                  # [B, N, Tp, d_model]
            streams = torch.cat([h_strain.unsqueeze(1), h_wit], dim=1)  # [B, 1+N, Tp, d_model]
            S = 1 + N
        else:
            Tp, d_model = h_strain.shape[1], h_strain.shape[2]
            streams = h_strain.unsqueeze(1)  # [B, 1, Tp, d_model]
            S = 1

        streams_flat = streams.reshape(B * S, Tp, d_model)

        for norm, block in zip(self.mamba_norms, self.mamba_blocks):
            streams_flat = streams_flat + block(norm(streams_flat))

        pooled_flat = self.pool(streams_flat)      # [B*S, d_model]
        pooled = pooled_flat.view(B, S * d_model)    # [B, S*d_model] -- concat of all streams
        return self.classifier(pooled)
