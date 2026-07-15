# permutation_importance_soh.py
#
# Model-grounded feature importance for the trained CNN-Mamba-UQ SOH model,
# using permutation importance (Breiman/Fisher-style: shuffle one feature,
# measure how much validation error increases).
#
# WHY THIS EXISTS: the FSS filter-score approach (correlation / VIF / MI /
# redundancy) picked a 3-feature subset that performed WORSE than your
# original manually-picked 11-feature subset when actually retrained. Filter
# scores are model-agnostic; they don't know what THIS specific nonlinear
# Mamba-CNN architecture can extract from "redundant"-looking features.
# Permutation importance instead asks the trained model directly: "how much
# do you rely on this feature?" -- by breaking that feature's information
# and watching performance degrade.
#
# SCOPE: this can only score the features the checkpoint was actually
# trained on (whatever is stored in checkpoint["feat_cols"]). Columns that
# exist only in the fuller CSV (discharge_capacity, discharge_energy,
# temperature_maximum/minimum, date_time_iso_numeric, dc_ir_norm,
# cycle_norm, cycle_index) were never seen by the model and cannot be
# scored without retraining with a wider input_dim.
#
# METHOD: "block" / sample-level permutation. For each feature, the ENTIRE
# time series for that feature is shuffled across samples (not shuffled
# independently at every timestep). This preserves each sample's internal
# temporal shape for that feature while destroying its correlation with the
# target and with the sample's other features -- the standard approach for
# sequence/time-series permutation importance, since per-timestep shuffling
# would over-punish any feature the model reads temporally.

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# CONFIG
# ============================================================

DEFAULT_CHECKPOINT_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\soh_best.pt"
DEFAULT_FULL_FEATURES_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_with_split.csv"
DEFAULT_OUTPUT_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\permutation_importance_results"

# NOTE: to evaluate the full 18-feature model instead of the original
# 10-feature one, run:
#   python permutation_importance_soh.py --checkpoint "...\checkpoints\soh_full_features_best.pt" --output_dir "...\permutation_importance_results_full"

N_REPEATS = 10          # repeats per feature, for a mean +/- std importance estimate
EVAL_BATCH_SIZE = 512

# Column name aliases: {name expected by the trained model : name in the fuller CSV}
COLUMN_ALIASES = {
    "temperature_avg": "temperature_average",
}

# ============================================================
# MODEL DEFINITION
# (kept byte-for-byte identical in structure to train_soh_only.py so the
#  saved state_dict loads correctly -- do not rename/reorder layers here)
# ============================================================


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


class CNNMambaSOH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        self.cnn = MultiScaleCNN(C["input_dim"], C["cnn_channels"], C["cnn_kernels"], C["dropout"])
        cnn_out = sum(C["cnn_channels"])
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, C["d_model"]),
            nn.LayerNorm(C["d_model"]), nn.GELU(),
            nn.Dropout(C["dropout"]),
        )
        self.mamba = MambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                   C["expand"], C["n_mamba_layers"], C["dropout"])
        self.attn_pool = nn.Linear(C["d_model"], 1)
        self.soh_head = nn.Sequential(
            nn.Linear(C["d_model"], 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(64, 2),
        )

    def encode(self, x):
        z = self.cnn_proj(self.cnn(x))
        z = self.mamba(z)
        attn = F.softmax(self.attn_pool(z), dim=1)
        return (z * attn).sum(dim=1)

    def forward(self, x):
        z = self.encode(x)
        out = self.soh_head(z)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


# ============================================================
# CHECKPOINT LOADING
# ============================================================


def load_checkpoint_and_model(checkpoint_path):
    print(f"\n[1] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    cfg = ckpt["cfg"]
    feat_cols = ckpt["feat_cols"]
    scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scaler_std = np.array(ckpt["scaler_std"], dtype=np.float32)

    print(f"  Model was trained on {len(feat_cols)} features: {feat_cols}")
    print(f"  window_size={cfg['window_size']}, stride={cfg['soh_stride']}")

    model = CNNMambaSOH(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "results" in ckpt:
        r = ckpt["results"]
        print(f"  Checkpoint's own test metrics: MAE={r.get('mae'):.4f}%  "
              f"RMSE={r.get('rmse'):.4f}%  R2={r.get('r2'):.5f}")

    return model, cfg, feat_cols, scaler_mean, scaler_std


# ============================================================
# DATA LOADING (fuller CSV, aligned to the model's feat_cols)
# ============================================================


def load_and_prepare_data(full_features_path, feat_cols, scaler_mean, scaler_std):
    print(f"\n[2] Loading fuller feature CSV: {full_features_path}")
    df = pd.read_csv(full_features_path)
    print(f"  Raw shape: {df.shape}")

    # Resolve any column-name aliases (e.g. temperature_avg -> temperature_average)
    for model_name, csv_name in COLUMN_ALIASES.items():
        if model_name not in df.columns and csv_name in df.columns:
            df[model_name] = df[csv_name]

    missing = [f for f in feat_cols if f not in df.columns]
    if missing:
        raise ValueError(
            f"These model features are not present (even after alias mapping) "
            f"in the fuller CSV: {missing}"
        )

    if "split" not in df.columns:
        raise ValueError("Expected a 'split' column in the fuller CSV to select the test set.")

    # IMPORTANT: apply the scaler fit during TRAINING (stored mean/std),
    # never refit a new scaler here -- that would silently create a
    # train/eval preprocessing mismatch.
    df[feat_cols] = (df[feat_cols].values - scaler_mean) / scaler_std

    df = df.dropna(subset=feat_cols + ["soh", "cycle_index", "cell_id"])
    print(f"  After NaN drop: {df.shape}")
    print(f"  Test cells: {df.loc[df.split == 'test', 'cell_id'].nunique()}")

    return df


def build_test_arrays(df, feat_cols, window_size, stride):
    """
    Build (X, y) arrays for the test split using the same sliding-window
    logic as SequenceDataset in train_soh_only.py, without going through
    torch Dataset/DataLoader machinery (simpler to permute in bulk).
    """
    print(f"\n[3] Building test sequences (window={window_size}, stride={stride})...")
    X_list, y_list = [], []
    test_df = df[df.split == "test"]

    for cell_id, cell_df in test_df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
        X = cell_df[feat_cols].values.astype(np.float32)
        y = cell_df["soh"].values.astype(np.float32)

        for end in range(window_size, len(X) + 1, stride):
            start = end - window_size
            X_list.append(X[start:end])
            y_list.append(y[end - 1])

    X_arr = np.stack(X_list, axis=0)  # (N, window, n_features)
    y_arr = np.array(y_list, dtype=np.float32)
    print(f"  Test sequences: {X_arr.shape[0]:,}  |  shape per sample: {X_arr.shape[1:]}")
    return X_arr, y_arr


# ============================================================
# PREDICTION / METRICS
# ============================================================


def predict(model, X, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
            mu, _ = model(xb)
            preds.append(mu.cpu().numpy())
    return np.concatenate(preds)


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================


def permutation_importance(model, X, y, feat_cols, n_repeats=N_REPEATS):
    print("\n[4] Computing baseline performance...")
    baseline_pred = predict(model, X)
    base_mae, base_rmse, base_r2 = compute_metrics(y, baseline_pred)
    print(f"  Baseline: MAE={base_mae:.4f}%  RMSE={base_rmse:.4f}%  R2={base_r2:.5f}")

    print(f"\n[5] Running permutation importance ({n_repeats} repeats per feature)...")
    print("  (block/sample-level shuffle: each feature's full time series is")
    print("   reassigned across samples, temporal shape within a sample is kept intact)\n")

    rows = []
    rng = np.random.default_rng(SEED)

    for f_idx, feat in enumerate(feat_cols):
        mae_increases, rmse_increases, r2_drops = [], [], []

        for rep in range(n_repeats):
            X_perm = X.copy()
            perm_order = rng.permutation(len(X_perm))
            X_perm[:, :, f_idx] = X_perm[perm_order, :, f_idx]

            perm_pred = predict(model, X_perm)
            mae_p, rmse_p, r2_p = compute_metrics(y, perm_pred)

            mae_increases.append(mae_p - base_mae)
            rmse_increases.append(rmse_p - base_rmse)
            r2_drops.append(base_r2 - r2_p)

        row = {
            "Feature": feat,
            "Mean_RMSE_Increase": np.mean(rmse_increases),
            "Std_RMSE_Increase": np.std(rmse_increases),
            "Mean_MAE_Increase": np.mean(mae_increases),
            "Std_MAE_Increase": np.std(mae_increases),
            "Mean_R2_Drop": np.mean(r2_drops),
            "Pct_RMSE_Increase": 100 * np.mean(rmse_increases) / base_rmse,
        }
        rows.append(row)

        print(f"  {feat:<32} RMSE +{row['Mean_RMSE_Increase']:.4f}% "
              f"(+/-{row['Std_RMSE_Increase']:.4f})  "
              f"R2 -{row['Mean_R2_Drop']:.5f}")

    results_df = pd.DataFrame(rows).sort_values("Mean_RMSE_Increase", ascending=False).reset_index(drop=True)
    return results_df, (base_mae, base_rmse, base_r2)


# ============================================================
# OUTPUT
# ============================================================


def save_and_plot(results_df, baseline, output_dir):
    print("\n[6] Saving results...")
    results_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    print("   Saved: permutation_importance.csv")

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = range(len(results_df))
    ax.barh(y_pos, results_df["Mean_RMSE_Increase"],
            xerr=results_df["Std_RMSE_Increase"],
            color="#3498db", alpha=0.8, edgecolor="black", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df["Feature"])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE increase (%) when feature is permuted")
    ax.set_title(
        f"Permutation Importance — CNN-Mamba-UQ SOH model\n"
        f"(baseline RMSE = {baseline[1]:.4f}%, higher bar = more important feature)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: permutation_importance.png")

    print("\nRanked feature importance (most -> least important):")
    for i, row in results_df.iterrows():
        print(f"  {i + 1:2d}. {row['Feature']:<32} "
              f"RMSE +{row['Mean_RMSE_Increase']:.4f}%  "
              f"({row['Pct_RMSE_Increase']:.1f}% relative)")


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Permutation importance for a trained CNN-Mamba-UQ SOH checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH,
                         help="Path to the .pt checkpoint to evaluate")
    parser.add_argument("--data", default=DEFAULT_FULL_FEATURES_PATH,
                         help="Path to the fuller feature CSV")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                         help="Where to save the importance CSV/plot")
    parser.add_argument("--n_repeats", type=int, default=N_REPEATS,
                         help="Number of shuffles per feature")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PERMUTATION IMPORTANCE — TRAINED CNN-MAMBA-UQ SOH MODEL")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data}")

    model, cfg, feat_cols, scaler_mean, scaler_std = load_checkpoint_and_model(args.checkpoint)

    df = load_and_prepare_data(args.data, feat_cols, scaler_mean, scaler_std)

    X, y = build_test_arrays(df, feat_cols, cfg["window_size"], cfg["soh_stride"])

    results_df, baseline = permutation_importance(model, X, y, feat_cols, n_repeats=args.n_repeats)

    save_and_plot(results_df, baseline, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nThis scored the {len(feat_cols)} feature(s) the loaded checkpoint was trained on.")
    print("Point --checkpoint at a different .pt file to score a different feature set.")


if __name__ == "__main__":
    main()