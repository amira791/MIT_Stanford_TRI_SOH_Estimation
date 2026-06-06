"""
cnn_mamba_uq_v3.py
===================
Changes from v2:
  1. Larger model: d_model=192, n_layers=3
  2. Multi-task output: future SOH + current SOH + EOL probability
  3. MC dropout increased to 0.25 for better UQ calibration
  4. Positional encoding added to help Mamba with sequence position

Multi-task design:
  - Shared encoder: CNN → Mamba (learns general degradation representations)
  - Head 1 (future_soh):  predicts SOH at t + HORIZON  [main task]
  - Head 2 (current_soh): predicts SOH at t             [auxiliary]
  - Head 3 (eol_prob):    P(SOH < 0.80) within horizon  [binary auxiliary]

Why multi-task helps generalisation:
  The current_soh auxiliary task forces the encoder to accurately represent
  the battery's current state — not just learn a mapping from window to future.
  This prevents the model from taking shortcuts (like memorising trajectory
  shapes) and instead builds physically meaningful internal representations
  that transfer to new cells and chemistries.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training_mlt import (
    N_FEATURES, SEQ_LEN, CNN_CHANNELS, CNN_KERNEL,
    MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS,
    MC_DROPOUT_P, MC_SAMPLES
)


class CNNEncoder(nn.Module):
    def __init__(self, in_features, channels, kernel):
        super().__init__()
        layers, in_ch = [], in_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.net    = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding — helps Mamba know where in the window it is."""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        return x + self.pe(pos)


class MambaLikeBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.B_proj  = nn.Linear(d_model, d_state, bias=False)
        self.C_proj  = nn.Linear(d_model, d_state, bias=False)
        self.A_log   = nn.Parameter(torch.log(torch.rand(d_state) * 0.5 + 0.4))
        self.out_proj = nn.Linear(d_state, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def ssm_scan(self, x):
        B, T, D = x.shape
        A = torch.sigmoid(self.A_log)
        h = x.new_zeros(B, self.d_state)
        outputs = []
        for t in range(T):
            x_t   = x[:, t, :]
            B_t   = torch.sigmoid(self.B_proj(x_t))
            x_proj = self.C_proj(x_t)
            h     = A * h + B_t * x_proj
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        residual = x
        x_n = self.norm1(x)
        gate, x_proj = self.in_proj(x_n).chunk(2, dim=-1)
        h_seq = self.ssm_scan(x_proj)
        y = torch.sigmoid(gate) * self.out_proj(h_seq)
        x = residual + self.drop(y)
        return x + self.drop(self.mlp(self.norm2(x)))


class MambaEncoder(nn.Module):
    def __init__(self, d_model, d_state, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLikeBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RegressionHead(nn.Module):
    """Single regression head with MC dropout."""
    def __init__(self, d_model, dropout, bounded=True):
        super().__init__()
        self.bounded = bounded
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out = self.net(x)
        if self.bounded:
            return torch.sigmoid(out)   # SOH bounded in [0,1]
        return out                      # unbounded


class CNNMambaUQ(nn.Module):
    """
    Multi-task CNN-Mamba-UQ.

    Training forward():   returns dict {future_soh, current_soh, eol_prob}
    Inference mc_predict(): returns dict {mean, std, ci_low, ci_high} for future_soh
    """
    def __init__(
        self,
        n_features  = N_FEATURES,
        seq_len     = SEQ_LEN,
        cnn_channels= CNN_CHANNELS,
        cnn_kernel  = CNN_KERNEL,
        d_model     = MAMBA_D_MODEL,
        d_state     = MAMBA_D_STATE,
        n_layers    = MAMBA_N_LAYERS,
        dropout     = MC_DROPOUT_P,
        mc_samples  = MC_SAMPLES,
    ):
        super().__init__()
        self.mc_samples = mc_samples

        self.cnn  = CNNEncoder(n_features, cnn_channels, cnn_kernel)
        self.proj = nn.Linear(cnn_channels[-1], d_model)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=seq_len + 10)
        self.mamba = MambaEncoder(d_model, d_state, n_layers, dropout)

        # ── three task heads ─────────────────────────────────────────
        self.head_future  = RegressionHead(d_model, dropout, bounded=True)
        self.head_current = RegressionHead(d_model, dropout, bounded=True)
        self.head_eol     = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()    # probability in [0,1]
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  CNN-Mamba-UQ mlt |  parameters: {n_params:,}")

    def encode(self, x):
        """Shared encoder: (B, T, F) → (B, d_model)"""
        x = self.cnn(x)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.mamba(x)
        return x[:, -1, :]     # last timestep summary

    def forward(self, x):
        """
        Training forward.
        Returns dict — all three heads.
        """
        h = self.encode(x)
        return {
            "future_soh"  : self.head_future(h),    # (B,1) — primary target
            "current_soh" : self.head_current(h),   # (B,1) — auxiliary
            "eol_prob"    : self.head_eol(h),        # (B,1) — auxiliary
        }

    def predict_future(self, x):
        """Single forward pass, future SOH only. Used in eval_epoch."""
        return self.head_future(self.encode(x))

    def mc_predict(self, x):
        """
        MC dropout inference for future SOH uncertainty.
        Returns mean, std, and 95% CI over MC_SAMPLES passes.
        """
        self.train()   # dropout active
        preds = []
        for _ in range(self.mc_samples):
            preds.append(self.head_future(self.encode(x)))
        preds = torch.cat(preds, dim=-1)   # (B, mc_samples)
        self.eval()

        mean    = preds.mean(dim=-1)
        std     = preds.std(dim=-1)
        return {
            "mean"    : mean.cpu().numpy(),
            "std"     : std.cpu().numpy(),
            "ci_low"  : (mean - 1.96 * std).cpu().numpy(),
            "ci_high" : (mean + 1.96 * std).cpu().numpy(),
        }
