

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
# Publication style
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

OUT_PATH = "results.png"

# ─────────────────────────────────────────────────────────────────────────────
# Config
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

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
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
        self.cell_ids = []
        self.end_cycles = []
        subset = df if split is None else df[df.split == split]

        for cid, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)
            cycle_idx = cell_df["cycle_index"].values

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                self.samples.append((X[start:end], y[end - 1]))
                self.cell_ids.append(cid)
                self.end_cycles.append(cycle_idx[end - 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
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
        self.head = EvidentialHead(C["d_model"], C["dropout"])
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
# Prediction extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []
    all_attn = []

    for x, y, _ in loader:
        x = x.to(DEVICE)
        out, attn = model(x)
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
        all_attn.append(attn.cpu().numpy())

    attn_all = np.concatenate(all_attn, axis=0) if all_attn else None
    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic), attn_all)


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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING 4-PANEL FIGURE")
    print("=" * 60)

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["cfg"]
    cfg["window_size"] = WINDOW_SIZE

    # Load data
    print("Loading data...")
    soh_df, scaler = load_soh_data(SOH_DATA_PATH)

    val_ds = SequenceDataset(soh_df, WINDOW_SIZE, cfg["soh_stride"], "val")
    test_ds = SequenceDataset(soh_df, WINDOW_SIZE, cfg["soh_stride"], "test")

    val_loader = DataLoader(val_ds, batch_size=cfg["soh_batch"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)

    # Build model
    print("Building model...")
    model = BEM_SOH(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get predictions
    print("Getting predictions...")
    y_val, mu_val, sigma_val, _, _, _ = get_predictions(model, val_loader)
    y_true, y_pred, sigma_test, aleatoric, epistemic, attn = get_predictions(model, test_loader)

    # Calibration
    iso = fit_isotonic_calibrator(y_val, mu_val, sigma_val)
    y_lo_cal, y_hi_cal = calibrated_interval(y_pred, sigma_test, iso, conf=0.90)

    # Get cell_ids and cycles for trajectory
    cell_ids = test_ds.cell_ids
    end_cycles = test_ds.end_cycles
    cell_array = np.array(cell_ids)
    cycle_array = np.array(end_cycles)

    mae = mean_absolute_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"  MAE: {mae:.4f}%, R²: {r2:.5f}")

    # ═════════════════════════════════════════════════════════════════════
    # Build 2×2 figure
    # ═════════════════════════════════════════════════════════════════════

    print("\nBuilding 2×2 figure...")

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ─── (a) Predicted vs. True ───
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(y_true, y_pred, alpha=0.25, s=4, c='#2874A6', edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax_a.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax_a.set_xlabel('True SOH')
    ax_a.set_ylabel('Predicted SOH')
    ax_a.set_title(f'(a) Predicted vs. True SOH\nMAE = {mae:.4f}%  |  R² = {r2:.5f}')
    ax_a.legend(loc='upper left')
    ax_a.set_aspect('equal')
    ax_a.grid(True, alpha=0.3)

    # ─── (b) Trajectory for one representative cell ───
    # Choose cell with most windows
    unique_cells, counts = np.unique(cell_array, return_counts=True)
    best_cell = unique_cells[np.argmax(counts)]

    mask = cell_array == best_cell
    cycles_cell = cycle_array[mask]
    y_true_cell = y_true[mask]
    y_pred_cell = y_pred[mask]
    y_lo_cell = y_lo_cal[mask]
    y_hi_cell = y_hi_cal[mask]

    sort_idx = np.argsort(cycles_cell)
    cycles_cell = cycles_cell[sort_idx]
    y_true_cell = y_true_cell[sort_idx]
    y_pred_cell = y_pred_cell[sort_idx]
    y_lo_cell = y_lo_cell[sort_idx]
    y_hi_cell = y_hi_cell[sort_idx]

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(cycles_cell, y_true_cell, 'k-', linewidth=1.8, label='Ground truth')
    ax_b.plot(cycles_cell, y_pred_cell, 'b-', linewidth=1.5, label='Predicted')
    ax_b.fill_between(cycles_cell, y_lo_cell, y_hi_cell,
                       alpha=0.25, color='blue', label='90% calibrated interval')
    ax_b.axhline(y=0.80, color='red', linestyle='--', linewidth=1, label='EOL (80%)')
    ax_b.set_xlabel('Cycle index')
    ax_b.set_ylabel('SOH')
    ax_b.set_title(f'(b) SOH Trajectory — Cell {best_cell}')
    ax_b.legend(loc='lower left')
    ax_b.grid(True, alpha=0.3)

    # ─── (c) Calibration curve ───
    quantiles = np.linspace(0.05, 0.95, 19)

    raw_coverage = []
    for q in quantiles:
        z = norm.ppf(q)
        covered = (y_true <= y_pred + z * sigma_test).mean()
        raw_coverage.append(covered)

    cal_coverage = []
    for q in quantiles:
        y_lo_c, y_hi_c = calibrated_interval(y_pred, sigma_test, iso, conf=q)
        covered = (y_true >= y_lo_c) & (y_true <= y_hi_c)
        cal_coverage.append(covered.mean())

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(quantiles, quantiles, 'k--', linewidth=1.5, label='Perfect calibration')
    ax_c.plot(quantiles, raw_coverage, 'o-', color='#E74C3C', linewidth=1.5,
              markersize=4, label='Raw (uncalibrated)')
    ax_c.plot(quantiles, cal_coverage, 's-', color='#27AE60', linewidth=1.5,
              markersize=4, label='Calibrated')
    ax_c.set_xlabel('Nominal coverage')
    ax_c.set_ylabel('Empirical coverage')
    ax_c.set_title('(c) Calibration Curve')
    ax_c.legend(loc='upper left')
    ax_c.set_aspect('equal')
    ax_c.grid(True, alpha=0.3)

    # ─── (d) Attention weights ───
    sorted_idx = np.argsort(y_true)
    low_idx = sorted_idx[0]
    high_idx = sorted_idx[-1]

    attn_low = attn[low_idx].squeeze()
    attn_high = attn[high_idx].squeeze()

    ax_d = fig.add_subplot(gs[1, 1])
    x_axis = np.arange(len(attn_low))
    ax_d.plot(x_axis, attn_high, '-o', color='#27AE60', linewidth=1.5,
              markersize=3, label=f'High SOH = {y_true[high_idx]:.3f}')
    ax_d.plot(x_axis, attn_low, '-s', color='#E74C3C', linewidth=1.5,
              markersize=3, label=f'Low SOH = {y_true[low_idx]:.3f}')
    ax_d.set_xlabel('Cycle position in window')
    ax_d.set_ylabel('Attention weight')
    ax_d.set_title('(d) Attention Weights — Low vs. High SOH')
    ax_d.legend(loc='upper left')
    ax_d.grid(True, alpha=0.3)

    # ─── Save ───
    plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved -> {OUT_PATH}")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)