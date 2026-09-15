# evaluate_L30_full.py
# Load the saved L=30 model and compute all evaluation metrics:
# performance, uncertainty, and deployment.

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
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

WINDOW_SIZE = 30
CHECKPOINT_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\window_ablation\bem_soh_W30.pt"
SOH_DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"
SAVE_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\window_ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
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
    def __init__(self, df, window_size, stride=1, split=None):
        self.samples = []
        subset = df if split is None else df[df.split == split]

        for cid, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                self.samples.append((X[start:end], y[end - 1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model definitions (same as training)
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []
    all_nu, all_alpha, all_beta = [], [], []

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
        all_nu.extend(nu.cpu().numpy())
        all_alpha.extend(alpha.cpu().numpy())
        all_beta.extend(beta.cpu().numpy())

    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic),
            np.array(all_nu), np.array(all_alpha), np.array(all_beta))


def fit_isotonic_calibrator(y_true, mu, sigma, n_q=20):
    quantiles = np.linspace(0.05, 0.95, n_q)
    empirical = []
    for q in quantiles:
        z = norm.ppf(q)
        covered = (y_true <= mu + z * sigma).mean()
        empirical.append(covered)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(quantiles, empirical)
    return iso


def calibrated_interval(mu, sigma, iso_calibrator, conf=0.90):
    target_q_lo, target_q_hi = (1 - conf) / 2, 1 - (1 - conf) / 2
    grid = np.linspace(0.001, 0.999, 400)
    mapped = iso_calibrator.predict(grid)
    q_lo = grid[np.argmin(np.abs(mapped - target_q_lo))]
    q_hi = grid[np.argmin(np.abs(mapped - target_q_hi))]
    z_lo, z_hi = norm.ppf(q_lo), norm.ppf(q_hi)
    return mu + z_lo * sigma, mu + z_hi * sigma


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    tmp_path = "_tmp_size_check.pt"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)
    return size_mb


@torch.no_grad()
def measure_latency(model, L, D_in, batch_sizes=[1, 32, 256], reps=100, warmup=10):
    model.eval()
    results = {}
    for bs in batch_sizes:
        dummy = torch.randn(bs, L, D_in, device=DEVICE)
        for _ in range(warmup):
            _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total = t1 - t0
        per_batch_ms = (total / reps) * 1000
        per_sample_ms = per_batch_ms / bs
        throughput = bs * reps / total
        results[f"batch_{bs}"] = {
            "ms_per_sample": round(per_sample_ms, 4),
            "throughput": round(throughput, 1),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"  FULL EVALUATION FOR L={WINDOW_SIZE}")
    print("=" * 60)

    # ─── Load checkpoint ───
    print("\nLoading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["cfg"]
    cfg["window_size"] = WINDOW_SIZE

    # ─── Load data ───
    print("Loading data...")
    soh_df, scaler = load_soh_data(SOH_DATA_PATH)

    val_ds = SequenceDataset(soh_df, WINDOW_SIZE, cfg["soh_stride"], "val")
    test_ds = SequenceDataset(soh_df, WINDOW_SIZE, cfg["soh_stride"], "test")

    val_loader = DataLoader(val_ds, batch_size=cfg["soh_batch"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)

    print(f"  Val windows:  {len(val_ds):,}")
    print(f"  Test windows: {len(test_ds):,}")

    # ─── Build model ───
    print("Building model...")
    model = BEM_SOH(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ─── Get predictions ───
    print("Getting predictions...")
    y_val, mu_val, sigma_val, _, _, _, _, _ = get_predictions(model, val_loader)
    (y_true, y_pred, sigma_test, aleatoric, epistemic,
     nu_vals, alpha_vals, beta_vals) = get_predictions(model, test_loader)

    # ─── Performance metrics ───
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100
    max_error = np.max(np.abs(y_true - y_pred)) * 100

    # ─── Uncertainty metrics ───
    z = 1.645
    y_lo_raw = y_pred - z * sigma_test
    y_hi_raw = y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    iso = fit_isotonic_calibrator(y_val, mu_val, sigma_val)
    y_lo_cal, y_hi_cal = calibrated_interval(y_pred, sigma_test, iso, conf=0.90)
    picp_cal = np.mean((y_true >= y_lo_cal) & (y_true <= y_hi_cal))
    pinw_cal = np.mean(y_hi_cal - y_lo_cal) / (y_true.max() - y_true.min() + 1e-8)

    # ─── NLL ───
    var = sigma_test ** 2 + 1e-12
    nll = np.mean(0.5 * np.log(2 * np.pi * var) + (y_true - y_pred) ** 2 / (2 * var))

    # ─── Deployment metrics ───
    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)
    latency = measure_latency(model, WINDOW_SIZE, cfg["input_dim"])

    # ─── Display all results ───
    print("\n" + "=" * 60)
    print(f"  PERFORMANCE METRICS (L={WINDOW_SIZE})")
    print("=" * 60)
    print(f"  MAE:        {mae:.4f}%")
    print(f"  RMSE:       {rmse:.4f}%")
    print(f"  MAPE:       {mape:.4f}%")
    print(f"  R²:         {r2:.5f}")
    print(f"  Max Error:  {max_error:.4f}%")

    print("\n" + "=" * 60)
    print(f"  UNCERTAINTY METRICS (L={WINDOW_SIZE})")
    print("=" * 60)
    print(f"  PICP (raw):        {picp_raw:.4f}")
    print(f"  PICP (calibrated): {picp_cal:.4f}")
    print(f"  PINW (raw):        {pinw_raw:.4f}")
    print(f"  PINW (calibrated): {pinw_cal:.4f}")
    print(f"  NLL:               {nll:.6f}")
    print(f"  Aleatoric var:     {np.nanmean(aleatoric):.6e}")
    print(f"  Epistemic var:     {np.nanmean(epistemic):.6e}")
    print(f"  Aleatoric/Epistemic: {np.nanmean(aleatoric) / np.nanmean(epistemic):.2f}")

    print("\n" + "=" * 60)
    print(f"  DEPLOYMENT METRICS (L={WINDOW_SIZE})")
    print("=" * 60)
    print(f"  Parameters:        {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")
    print(f"  Model size (FP32): {size_mb:.3f} MB")
    for bs, stats in latency.items():
        print(f"  {bs:12s} -> {stats['ms_per_sample']:.4f} ms/sample | "
              f"{stats['throughput']:.1f} samples/sec")

    # ─── Save ───
    results = {
        "window_size": WINDOW_SIZE,
        "performance": {
            "mae": mae, "rmse": rmse, "mape": mape,
            "r2": r2, "max_error": max_error,
        },
        "uncertainty": {
            "picp_raw": picp_raw, "picp_cal": picp_cal,
            "pinw_raw": pinw_raw, "pinw_cal": pinw_cal,
            "nll": nll,
            "mean_aleatoric": float(np.nanmean(aleatoric)),
            "mean_epistemic": float(np.nanmean(epistemic)),
        },
        "deployment": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": size_mb,
            "latency": latency,
            "device": str(DEVICE),
        },
    }
    save_path = os.path.join(SAVE_DIR, f"full_evaluation_W{WINDOW_SIZE}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved -> {save_path}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)