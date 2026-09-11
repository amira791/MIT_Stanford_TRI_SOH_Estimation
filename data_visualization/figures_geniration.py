# figures_generation.py
# Self-contained script to generate all five figures for MIT-Stanford
# Includes all model definitions and helper functions

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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

# Set style for publication
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['figure.dpi'] = 300

# Output directory
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt"
SOH_DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & preprocessing (from training script)
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
# 2. Model definitions (from training script)
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
# 3. Helper functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions_with_attention(model, loader, cfg):
    """Returns predictions, uncertainties, and attention weights."""
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []
    all_attn = []

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(DEVICE)
        out, attn = model(x)
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
        all_attn.append(attn.cpu().numpy())

    attn_all = np.concatenate(all_attn, axis=0) if all_attn else None
    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic), attn_all)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figure generation functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(y_true, y_pred, save_path):
    """Figure 1: Predicted vs. actual SOH scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.3, s=5, c='#2874A6', edgecolors='none')

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')

    mae = mean_absolute_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel('True SOH')
    ax.set_ylabel('Predicted SOH')
    ax.set_title(f'Predicted vs. Actual SOH\nMAE = {mae:.4f}%  |  R² = {r2:.5f}')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reliability_diagram(y_val, mu_val, sigma_val, y_test, mu_test, sigma_test, save_path):
    """Figure 2: Reliability diagram for raw vs. calibrated intervals."""
    fig, ax = plt.subplots(figsize=(6, 6))

    quantiles = np.linspace(0.05, 0.95, 19)

    raw_coverage = []
    for q in quantiles:
        z = norm.ppf(q)
        covered = (y_test <= mu_test + z * sigma_test).mean()
        raw_coverage.append(covered)

    val_coverage = []
    for q in quantiles:
        z = norm.ppf(q)
        covered = (y_val <= mu_val + z * sigma_val).mean()
        val_coverage.append(covered)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(quantiles, val_coverage)

    cal_coverage = []
    for q in quantiles:
        q_cal = iso.predict([q])[0]
        z_cal = norm.ppf(q_cal)
        covered = (y_test <= mu_test + z_cal * sigma_test).mean()
        cal_coverage.append(covered)

    ax.plot(quantiles, quantiles, 'k--', linewidth=1.5, label='Perfect calibration')
    ax.plot(quantiles, raw_coverage, 'o-', color='#E74C3C', linewidth=1.5,
            markersize=4, label='Raw (uncalibrated)')
    ax.plot(quantiles, cal_coverage, 's-', color='#27AE60', linewidth=1.5,
            markersize=4, label='Calibrated')

    ax.set_xlabel('Nominal Quantile')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Reliability Diagram')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uncertainty_decomposition(aleatoric, epistemic, save_path):
    """Figure 3: Aleatoric vs. epistemic uncertainty scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Filter out NaN values
    mask = ~np.isnan(aleatoric) & ~np.isnan(epistemic)
    aleatoric = aleatoric[mask]
    epistemic = epistemic[mask]

    ax.scatter(aleatoric, epistemic, alpha=0.3, s=5, c='#8E44AD', edgecolors='none')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Aleatoric Variance (Data Noise)')
    ax.set_ylabel('Epistemic Variance (Model Ignorance)')
    ax.set_title('Uncertainty Decomposition')
    ax.grid(True, alpha=0.3, which='both')

    lims = [min(aleatoric.min(), epistemic.min()), max(aleatoric.max(), epistemic.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Aleatoric = Epistemic')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_attention_heatmap(attn_weights, save_path, n_samples=200):
    """Figure 4: Attention weight heatmap over cycles."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if attn_weights.ndim == 3:
        attn_weights = attn_weights.squeeze(-1)

    if attn_weights.shape[0] > n_samples:
        mean_attn = attn_weights.mean(axis=1)
        sorted_idx = np.argsort(mean_attn)
        attn_subset = attn_weights[sorted_idx[:n_samples]]
    else:
        attn_subset = attn_weights

    im = ax.imshow(attn_subset, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    ax.set_xlabel('Cycle Position in Window (0–49)')
    ax.set_ylabel('Test Sample')
    ax.set_title('Attention Weights Over Cycles')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_attention_over_cycles(attn_weights, save_path):
    """Figure 4b: Average attention weight over cycles."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if attn_weights.ndim == 3:
        attn_weights = attn_weights.squeeze(-1)

    mean_attn = attn_weights.mean(axis=0)
    std_attn = attn_weights.std(axis=0)

    cycles = np.arange(len(mean_attn))

    ax.plot(cycles, mean_attn, color='#2874A6', linewidth=2, label='Mean attention')
    ax.fill_between(cycles, mean_attn - std_attn, mean_attn + std_attn,
                     alpha=0.3, color='#2874A6', label='±1 std')

    ax.set_xlabel('Cycle Position in Window')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Average Attention Weight Over Cycles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cross_chemistry_mae(save_path):
    """Figure 5: Cross-chemistry MAE comparison."""
    datasets = ['MIT-Stanford\n(LFP)', 'SNL NCM\n(NMC)', 'SNL NCA\n(NCA)']
    mae_values = [0.116, 1.11, 1.59]
    mae_errors = [0.0, 0.19, 0.25]

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#2874A6', '#27AE60', '#E67E22']
    bars = ax.bar(datasets, mae_values, yerr=mae_errors, capsize=5,
                   color=colors, edgecolor='black', linewidth=1)

    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('MAE (%)')
    ax.set_title('Cross-Chemistry Generalization')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING FIGURES FOR MIT-STANFORD")
    print("=" * 60)

    # ─── Load MIT checkpoint ───
    print("\nLoading MIT checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["cfg"]

    # ─── Load MIT data ───
    print("Loading MIT data...")
    soh_df, scaler = load_soh_data(SOH_DATA_PATH)
    W = cfg["window_size"]

    val_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "val")
    test_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "test")

    val_loader = DataLoader(val_ds, batch_size=cfg["soh_batch"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)

    # ─── Build model and load weights ───
    print("Building BEM-SOH model...")
    model = BEM_SOH(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ─── Get predictions with attention ───
    print("Getting predictions...")
    y_val, mu_val, sigma_val, _, _, _ = get_predictions_with_attention(model, val_loader, cfg)
    y_true, y_pred, sigma_test, aleatoric, epistemic, attn = get_predictions_with_attention(
        model, test_loader, cfg
    )

    print(f"  Test samples: {len(y_true):,}")
    print(f"  MAE: {mean_absolute_error(y_true, y_pred) * 100:.4f}%")
    print(f"  R²: {r2_score(y_true, y_pred):.5f}")
    print(f"  Attention shape: {attn.shape if attn is not None else 'None'}")

    # ─── Figure 1: Predicted vs. Actual SOH ───
    print("\n" + "=" * 60)
    print("  Figure 1: Predicted vs. Actual SOH")
    print("=" * 60)
    plot_predicted_vs_actual(y_true, y_pred, os.path.join(OUT_DIR, "fig1_pred_vs_actual.png"))

    # ─── Figure 2: Reliability Diagram ───
    print("\n" + "=" * 60)
    print("  Figure 2: Reliability Diagram")
    print("=" * 60)
    plot_reliability_diagram(y_val, mu_val, sigma_val, y_true, y_pred, sigma_test,
                              os.path.join(OUT_DIR, "fig2_reliability_diagram.png"))

    # ─── Figure 3: Aleatoric vs. Epistemic ───
    print("\n" + "=" * 60)
    print("  Figure 3: Aleatoric vs. Epistemic Uncertainty")
    print("=" * 60)
    plot_uncertainty_decomposition(aleatoric, epistemic,
                                    os.path.join(OUT_DIR, "fig3_uncertainty_decomposition.png"))

    # ─── Figure 4: Attention Weight Heatmap ───
    print("\n" + "=" * 60)
    print("  Figure 4: Attention Weight Heatmap")
    print("=" * 60)
    if attn is not None:
        plot_attention_heatmap(attn, os.path.join(OUT_DIR, "fig4_attention_heatmap.png"))
        plot_attention_over_cycles(attn, os.path.join(OUT_DIR, "fig4b_attention_over_cycles.png"))
    else:
        print("  Attention weights not available.")

    # ─── Figure 5: Cross-Chemistry MAE ───
    print("\n" + "=" * 60)
    print("  Figure 5: Cross-Chemistry MAE Comparison")
    print("=" * 60)
    plot_cross_chemistry_mae(os.path.join(OUT_DIR, "fig5_cross_chemistry_mae.png"))

    # ─── Done ───
    print("\n" + "=" * 60)
    print("  ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    - {f}")