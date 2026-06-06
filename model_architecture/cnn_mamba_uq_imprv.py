# model_architecture/cnn_mamba_uq.py
"""
CNN-Mamba-UQ Architecture (FIXED VERSION)
------------------------------------------
Fixes applied:
  1. Removed @torch.no_grad() from mc_predict (fixes UQ zero uncertainty)
  2. Fixed Mamba scan information collapse (preserves feature dimension)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training_50 import (
    N_FEATURES, SEQ_LEN, CNN_CHANNELS, CNN_KERNEL,
    MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS,
    MC_DROPOUT_P, MC_SAMPLES
)

class CNNEncoder(nn.Module):
    """1-D temporal CNN encoder."""
    def __init__(self, in_features: int, channels: list, kernel: int):
        super().__init__()
        layers = []
        in_ch = in_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)

class MambaLikeBlock(nn.Module):
    """Hardware-portable selective SSM block with fixed information flow."""
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)  # FIX: Added projection to preserve features
        self.A_log = nn.Parameter(torch.log(torch.rand(d_state) * 0.5 + 0.4))
        self.out_proj = nn.Linear(d_state, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective linear recurrence over time dimension.
        FIX: No information collapse - preserves d_state dimension throughout.
        """
        B, T, D = x.shape
        A = torch.sigmoid(self.A_log)
        h = x.new_zeros(B, self.d_state)
        outputs = []
        
        for t in range(T):
            x_t = x[:, t, :]                    # (B, D)
            B_t = torch.sigmoid(self.B_proj(x_t))  # (B, d_state)
            x_proj = self.C_proj(x_t)           # (B, d_state) - FIX: preserves information
            h = A * h + B_t * x_proj            # (B, d_state)
            outputs.append(h)
            
        return torch.stack(outputs, dim=1)       # (B, T, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n = self.norm1(x)
        gate, x_proj = self.in_proj(x_n).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        h_seq = self.ssm_scan(x_proj)
        y = self.out_proj(h_seq)
        y = gate * y
        x = residual + self.drop(y)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class MambaEncoder(nn.Module):
    """Stack of MambaLikeBlock layers."""
    def __init__(self, d_model: int, d_state: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLikeBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class UQHead(nn.Module):
    """Regression head with MC Dropout."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # No sigmoid here - SOH is already in [0,1] from preprocessing
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNNMambaUQ(nn.Module):
    """Full CNN-Mamba-UQ model with fixed UQ and Mamba scan."""
    def __init__(
        self,
        n_features: int = N_FEATURES,
        seq_len: int = SEQ_LEN,
        cnn_channels: list = CNN_CHANNELS,
        cnn_kernel: int = CNN_KERNEL,
        d_model: int = MAMBA_D_MODEL,
        d_state: int = MAMBA_D_STATE,
        n_layers: int = MAMBA_N_LAYERS,
        dropout: float = MC_DROPOUT_P,
        mc_samples: int = MC_SAMPLES,
    ):
        super().__init__()
        self.mc_samples = mc_samples

        self.cnn = CNNEncoder(n_features, cnn_channels, cnn_kernel)
        cnn_out_dim = cnn_channels[-1]
        self.proj = nn.Linear(cnn_out_dim, d_model)
        self.mamba = MambaEncoder(d_model, d_state, n_layers, dropout)
        self.head = UQHead(d_model, dropout)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  CNN-Mamba-UQ  |  parameters: {n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        x: (B, T, F) -> SOH: (B, 1)
        """
        x = self.cnn(x)
        x = self.proj(x)
        x = self.mamba(x)
        x = x[:, -1, :]  # Take last timestep
        return self.head(x)

    def mc_predict(self, x: torch.Tensor) -> dict:
        """
        Monte Carlo prediction for uncertainty quantification.
        FIX: Removed @torch.no_grad() decorator to allow proper dropout.
        Runs MC_SAMPLES stochastic forward passes with dropout active.
        """
        self.train()  # Keep dropout active for MC sampling
        
        preds_list = []
        for _ in range(self.mc_samples):
            pred = self.forward(x)
            preds_list.append(pred)
        
        preds = torch.cat(preds_list, dim=-1)  # (B, mc_samples)
        self.eval()

        mean = preds.mean(dim=-1)
        std = preds.std(dim=-1)
        ci_low = mean - 1.96 * std
        ci_high = mean + 1.96 * std

        return {
            "mean": mean.cpu().numpy(),
            "std": std.cpu().numpy(),
            "ci_low": ci_low.cpu().numpy(),
            "ci_high": ci_high.cpu().numpy(),
        }