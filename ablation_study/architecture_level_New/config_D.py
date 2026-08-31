# evaluate_config_D_from_checkpoint.py
# Re-evaluate your saved model WITHOUT calibration to get Config D results
# Standalone version — all dependencies copied from train_final_model.py

import os, math, time, warnings, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Base configuration (same as your main script)
CFG = dict(
    # Paths
    soh_path  = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv",
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt",

    # Features
    input_dim  = 10,
    window_size = 50,
    soh_stride  = 2,

    # Model
    cnn_channels = [32, 64, 128],
    cnn_kernels  = [3, 7, 15],
    d_model      = 128,
    d_state      = 16,
    d_conv       = 4,
    expand       = 2,
    n_mamba_layers = 3,
    dropout      = 0.15,

    bidirectional = True,
    evidential    = True,
    calibrate     = False,   # Will be overridden by checkpoint anyway

    # Training
    soh_epochs   = 120,
    soh_lr       = 2e-4,
    soh_batch    = 256,
    soh_wd       = 1e-4,
    soh_patience = 25,
    tail_weight  = 3.0,
    warmup_epochs = 10,

    nig_mse_warmup_epochs = 10,
    evid_lambda   = 0.01,
    evid_lambda_max = 0.05,
    evid_mse_weight = 1.0,

    latency_batch_sizes = [1, 32, 256],
    latency_reps = 100,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-cell relative features."""
    df = df.copy()
    cap_rel_list, en_rel_list, ir_rel_list, cycle_pos_list = [], [], [], []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index")
        early = cell_df.iloc[:10]

        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        nom_ir = early["dc_internal_resistance"].mean()
        min_cycle = cell_df["cycle_index"].min()
        max_cycle = cell_df["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)

        cap_rel_list.append((cell_df["charge_capacity"] - nom_cap) / (nom_cap + 1e-9))
        en_rel_list.append((cell_df["charge_energy"] - nom_energy) / (nom_energy + 1e-9))
        ir_rel_list.append((cell_df["dc_internal_resistance"] - nom_ir) / (nom_ir + 1e-9))
        cycle_pos_list.append((cell_df["cycle_index"] - min_cycle) / cyc_range)

    df["cap_rel"] = pd.concat(cap_rel_list)
    df["energy_rel"] = pd.concat(en_rel_list)
    df["ir_rel"] = pd.concat(ir_rel_list)
    df["cycle_pos"] = pd.concat(cycle_pos_list)
    return df


FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]


def load_soh_data(soh_path):
    soh = pd.read_csv(soh_path)
    soh = add_relative_features(soh)

    scaler = StandardScaler()
    scaler.fit(soh[soh.split == "train"][FEAT_COLS].values)
    soh[FEAT_COLS] = scaler.transform(soh[FEAT_COLS].values)

    return soh, scaler


class SequenceDataset(Dataset):
    """Sliding-window dataset for SOH."""
    def __init__(self, df, window_size, stride=1, split=None,
                 weighted=False, tail_thr=0.90, tail_weight=1.0):
        self.samples = []
        self.weights = []
        self.cell_ids = []
        subset = df if split is None else df[df.split == split]

        for cid, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                y_last = y[end - 1]
                self.samples.append((X[start:end], y_last))
                self.cell_ids.append(cid)
                if weighted:
                    w = tail_weight if y_last < tail_thr else 1.0
                else:
                    w = 1.0
                self.weights.append(w)

        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(self.weights[idx])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model (Full BEM-SOH architecture)
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
            nn.Linear(64, 4),  # gamma, log_nu, log_alpha, log_beta
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
            nn.Linear(64, 2),  # [mean, log_var]
        )

    def forward(self, z):
        out = self.net(z)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# 4.  get_predictions (for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, cfg):
    """Returns point pred, total sigma, and (if evidential) the
    aleatoric/epistemic decomposition."""
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []

    for x, y, _ in loader:
        x = x.to(DEVICE)
        out, _ = model(x)
        if cfg["evidential"]:
            gamma, nu, alpha, beta = out
            aleatoric = (beta / (alpha - 1)).cpu().numpy()
            epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
            sigma = np.sqrt(aleatoric + epistemic)
            mu = gamma.cpu().numpy()
            all_aleatoric.extend(aleatoric)
            all_epistemic.extend(epistemic)
        else:
            mu, log_var = out
            sigma = torch.exp(0.5 * log_var).cpu().numpy()
            mu = mu.cpu().numpy()
            all_aleatoric.extend([np.nan] * len(mu))
            all_epistemic.extend([np.nan] * len(mu))

        all_mu.extend(mu)
        all_sigma.extend(sigma)
        all_y.extend(y.numpy())

    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main: Load checkpoint and evaluate WITHOUT calibration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  CONFIG D: Bidirectional + Evidential (NO CALIBRATION)")
    print("  Re-evaluating saved model with calibrate=False")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ─── 1. Load checkpoint ───
    checkpoint_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Get configuration from checkpoint
    cfg = checkpoint["cfg"]
    cfg["calibrate"] = False  # ← FORCE CALIBRATION OFF (Config D)

    # ─── 2. Load data ───
    print("\nLoading SOH data...")
    soh_df, scaler = load_soh_data(cfg["soh_path"])
    print(f"  SOH: {soh_df.shape}")
    print(f"  Cells: {soh_df['barcode'].nunique()}")

    W = cfg["window_size"]
    train_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "train")
    val_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "val")
    test_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "test")

    val_loader = DataLoader(val_ds, batch_size=cfg["soh_batch"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)

    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")
    print(f"  Test sequences:  {len(test_ds):,}")

    # ─── 3. Load model ───
    print("\nBuilding BEM-SOH model...")
    model = BEM_SOH(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Config: bidirectional={cfg['bidirectional']}, "
          f"evidential={cfg['evidential']}, calibrate={cfg['calibrate']}")

    # ─── 4. Evaluate WITHOUT calibration ───
    print("\n" + "=" * 60)
    print("  SOH EVALUATION - TEST SET (Config D: No Calibration)")
    print("=" * 60)

    # Get predictions
    y_val, mu_val, sigma_val, _, _ = get_predictions(model, val_loader, cfg)
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, test_loader, cfg)

    # Accuracy metrics
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100

    # Raw intervals (no calibration)
    z = 1.645  # nominal 90% Gaussian z
    y_lo_raw = y_pred - z * sigma_test
    y_hi_raw = y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    print(f"\n  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  MAPE : {mape:.4f}%")
    print(f"  R2   : {r2:.5f}")

    print(f"\n  -- UNCALIBRATED intervals (nominal 90% Gaussian) --")
    print(f"  PICP : {picp_raw:.4f}  (target ~0.90)")
    print(f"  PINW : {pinw_raw:.4f}")

    if cfg["evidential"]:
        print(f"\n  -- Uncertainty decomposition (mean over test set) --")
        print(f"  Mean aleatoric var  : {np.nanmean(aleatoric):.6f}")
        print(f"  Mean epistemic var  : {np.nanmean(epistemic):.6f}")
        ratio = np.nanmean(aleatoric) / (np.nanmean(epistemic) + 1e-8)
        print(f"  Aleatoric/Epistemic ratio: {ratio:.2f}")

    print(f"\n  -- MAE by SOH region --")
    for label, mask in [
        ("SOH < 0.90", y_true < 0.90),
        ("0.90-0.95", (y_true >= 0.90) & (y_true < 0.95)),
        ("SOH > 0.95", y_true >= 0.95)
    ]:
        if mask.sum() > 0:
            rm = mean_absolute_error(y_true[mask], y_pred[mask]) * 100
            print(f"  {label}: MAE = {rm:.4f}%  (n={mask.sum()})")

    # ─── 5. Summary ───
    print("\n" + "=" * 60)
    print("  CONFIG D RESULTS (Bidirectional + Evidential, NO CALIBRATION)")
    print("=" * 60)
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R2   : {r2:.5f}")
    print(f"  PICP : {picp_raw:.4f}")
    print(f"  PINW : {pinw_raw:.4f}")
    print(f"  Aleatoric : {np.nanmean(aleatoric):.6f}")
    print(f"  Epistemic : {np.nanmean(epistemic):.6f}")
    print("=" * 60)

    # ─── 6. Save results ───
    results = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "picp_raw": picp_raw,
        "pinw_raw": pinw_raw,
        "mean_aleatoric": float(np.nanmean(aleatoric)),
        "mean_epistemic": float(np.nanmean(epistemic)),
        "total_params": total_params,
    }

    print("\nResults saved in 'results' variable.")
    print(json.dumps(results, indent=2, default=str))