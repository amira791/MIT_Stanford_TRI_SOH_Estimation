# ncm_to_nca_transfer.py
# Apply the NCM-trained BEM-SOH model directly to the NCA dataset
# (no retraining, no fine-tuning) to test zero-shot cross-chemistry transfer.

import os
import math
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Paths ───
NCM_CHECKPOINT = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\ncm_kfold_results_v2\bem_soh_W50.pt"  # adjust to your NCM checkpoint
NCA_DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\nca_with_temp_processed_v3_fixed.csv"
SAVE_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\ncm_to_nca_transfer"

os.makedirs(SAVE_DIR, exist_ok=True)

# ─── SNL feature columns (9 features, same for NCM and NCA) ───
SNL_FEAT_COLS = [
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    "cap_rel",
    "energy_rel",
    "cycle_pos",
    "temperature_avg",
    "voltage_range",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model definitions (same as training scripts)
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, channels, kernels, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for ch, k in zip(channels, kernels):
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
            ))
        self.out_dim = sum(channels)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1).permute(0, 2, 1)
        return self.dropout(F.gelu(self.proj(x)))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                 kernel_size=d_conv, padding=d_conv - 1,
                                 groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def ssm(self, x):
        B, L, D = x.shape
        N = self.d_state
        dBC = self.x_proj(x)
        delta = F.softplus(self.dt_proj(dBC[..., :1]))
        B_ssm = dBC[..., 1:N + 1]
        C_ssm = dBC[..., N + 1:]
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_u = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]
            ys.append((h * C_ssm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_c = self.conv1d(x_.permute(0, 2, 1))[..., :x_.shape[1]].permute(0, 2, 1)
        y = self.ssm(F.silu(x_c)) * F.silu(z)
        return self.dropout(self.out_proj(y)) + res


class MambaEncoder(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, n_layers, dropout):
        super().__init__()
        self.fwd = MambaEncoder(d_model, d_state, d_conv, expand, n_layers, dropout)
        self.bwd = MambaEncoder(d_model, d_state, d_conv, expand, n_layers, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        z_f = self.fwd(x)
        z_b = self.bwd(torch.flip(x, dims=[1]))
        z_b = torch.flip(z_b, dims=[1])
        return self.fuse(torch.cat([z_f, z_b], dim=-1))


class EvidentialHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(self, z):
        out = self.net(z)
        gamma = torch.sigmoid(out[:, 0])
        nu    = F.softplus(out[:, 1]).clamp(max=50.0) + 1e-6
        alpha = F.softplus(out[:, 2]).clamp(max=50.0) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


class BEM_SOH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        self.cfg = cfg

        self.cnn = MultiScaleCNN(C["input_dim"], C["cnn_channels"],
                                  C["cnn_kernels"], C["dropout"])
        cnn_out = sum(C["cnn_channels"])

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, C["d_model"]),
            nn.LayerNorm(C["d_model"]), nn.GELU(),
            nn.Dropout(C["dropout"]),
        )

        if C["bidirectional"]:
            self.encoder = BiMambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                           C["expand"], C["n_mamba_layers"], C["dropout"])
        else:
            self.encoder = MambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                         C["expand"], C["n_mamba_layers"], C["dropout"])

        self.attn_pool = nn.Linear(C["d_model"], 1)

        if C["evidential"]:
            self.head = EvidentialHead(C["d_model"], C["dropout"])
        else:
            self.head = GaussianHead(C["d_model"], C["dropout"])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def encode(self, x):
        z = self.cnn_proj(self.cnn(x))
        z = self.encoder(z)
        attn = F.softmax(self.attn_pool(z), dim=1)
        return (z * attn).sum(dim=1), attn

    def forward(self, x):
        z, attn = self.encode(x)
        return self.head(z), attn


class GaussianHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, z):
        out = self.net(z)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# 3. NCA Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NCASequenceDataset(Dataset):
    def __init__(self, df, window_size, stride=1):
        self.samples = []
        self.cell_ids = []

        for cid, cell_df in df.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[SNL_FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                self.samples.append((X[start:end], y[end - 1]))
                self.cell_ids.append(cid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []

    for x, y, _ in loader:
        x = x.to(DEVICE)
        out, _ = model(x)
        gamma, nu, alpha, beta = out
        aleatoric = (beta / (alpha - 1)).cpu().numpy()
        epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
        sigma = np.sqrt(aleatoric + epistemic)
        mu = gamma.cpu().numpy()
        all_aleatoric.extend(aleatoric)
        all_epistemic.extend(epistemic)
        all_mu.extend(mu)
        all_sigma.extend(sigma)
        all_y.extend(y.numpy())

    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NCM → NCA TRANSFER TEST")
    print("  (NCM-trained model applied directly to NCA, no retraining)")
    print("=" * 60)

    # ─── Load NCM checkpoint ───
    print("\nLoading NCM checkpoint...")
    checkpoint = torch.load(NCM_CHECKPOINT, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["cfg"]
    cfg["window_size"] = 50  # ensure same as training
    print(f"  NCM config: input_dim={cfg['input_dim']}, "
          f"window_size={cfg['window_size']}")

    # ─── Load NCA data ───
    print("\nLoading NCA data...")
    nca_df = pd.read_csv(NCA_DATA_PATH)
    print(f"  NCA cells: {nca_df['cell_id'].nunique()}")
    print(f"  NCA rows:  {len(nca_df):,}")

    # ─── Verify feature match ───
    missing = [f for f in SNL_FEAT_COLS if f not in nca_df.columns]
    if missing:
        raise ValueError(f"NCA data missing features: {missing}")
    print(f"  Features matched: {len(SNL_FEAT_COLS)}/{len(SNL_FEAT_COLS)} ✅")

    # ─── Normalize NCA using NCM scaler ───
    # CRITICAL: use the NCM-trained scaler, NOT a new one
    print("\nApplying NCM scaler to NCA data...")
    scaler_mean = np.array(checkpoint["scaler_mean"])
    scaler_std = np.array(checkpoint["scaler_std"])
    nca_df[SNL_FEAT_COLS] = (nca_df[SNL_FEAT_COLS].values - scaler_mean) / scaler_std

    # ─── Create dataset ───
    W = cfg["window_size"]
    nca_ds = NCASequenceDataset(nca_df, W, cfg["soh_stride"])
    nca_loader = DataLoader(nca_ds, batch_size=cfg["soh_batch"], shuffle=False)
    print(f"  NCA windows: {len(nca_ds):,}")

    # ─── Load model ───
    print("\nBuilding BEM-SOH model...")
    model = BEM_SOH(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ─── Get predictions ───
    print("\nRunning inference on NCA...")
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, nca_loader)

    # ─── Compute metrics ───
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)

    z = 1.645
    y_lo_raw = y_pred - z * sigma_test
    y_hi_raw = y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    # ─── Display results ───
    print("\n" + "=" * 60)
    print("  NCM → NCA TRANSFER RESULTS")
    print("=" * 60)
    print(f"  MAE:              {mae:.4f}%")
    print(f"  RMSE:             {rmse:.4f}%")
    print(f"  R²:               {r2:.4f}")
    print(f"  PICP (raw):       {picp_raw:.4f}")
    print(f"  PINW (raw):       {pinw_raw:.4f}")
    print(f"  Aleatoric var:    {np.nanmean(aleatoric):.6f}")
    print(f"  Epistemic var:    {np.nanmean(epistemic):.6f}")
    print(f"  Test windows:     {len(y_true):,}")
    print("=" * 60)

    # ─── Save results ───
    results = {
        "source": "NCM",
        "target": "NCA",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "picp_raw": picp_raw,
        "pinw_raw": pinw_raw,
        "mean_aleatoric": float(np.nanmean(aleatoric)),
        "mean_epistemic": float(np.nanmean(epistemic)),
        "n_windows": int(len(y_true)),
    }
    save_path = os.path.join(SAVE_DIR, "ncm_to_nca_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved -> {save_path}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)