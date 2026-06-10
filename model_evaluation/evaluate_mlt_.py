"""
evaluate_mlt.py
================
Complete evaluation for the trained CNN-Mamba-UQ multi-task model.

Tier 1 — Performance metrics:
    MAE, RMSE, MAPE, R², MaxE  (future SOH prediction)
    PICP, PINAW                (uncertainty quality)

Tier 2 — Deployment metrics:
    Inference latency (mean, std, p95) on CPU
    Model size, peak RAM, throughput

Tier 3 — Per-cell analysis:
    R² and MAE per test cell
    SOH trajectory plots (predicted vs true with CI band)

Outputs (all saved to results_mlt/evaluation/):
    metrics_summary.csv
    deployment_metrics.csv
    per_cell_metrics.csv
    plots/trajectory_<cell_id>.png  (one per test cell)
    plots/predicted_vs_true.png
    plots/error_distribution.png
    plots/uncertainty_vs_error.png

Usage:
    python evaluate_mlt.py
"""

import sys
import time
import pickle
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT = True
except ImportError:
    PLOT = False
    print("  matplotlib not found — skipping plots")


sys.path.insert(0, str(Path(__file__).parent.parent / "model_architecture"))

# ── adjust these paths to match your project ──────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent          # ← go up to project root
RESULTS_DIR    = ROOT_DIR / "results_mlt"
CKPT_PATH      = RESULTS_DIR / "checkpoints" / "cnn_mamba_uq_mlt_best.pt"
DATASET_PKL    = RESULTS_DIR / "soh_dataset_mlt.pkl"
SCALER_PKL     = RESULTS_DIR / "scaler_v3.pkl"        # ← was scaler_mlt.pkl, fix name
TEST_CELLS_PKL = RESULTS_DIR / "test_cells_mlt.pkl"
TEST_DS_PT     = RESULTS_DIR / "test_dataset_mlt.pt"
EVAL_DIR       = RESULTS_DIR / "evaluation_"
PLOTS_DIR      = EVAL_DIR / "plots_"
EVAL_DIR.mkdir(parents=True, exist_ok=True)            # ← parents=True
PLOTS_DIR.mkdir(parents=True, exist_ok=True)           # ← parents=True

# ── config (must match training config) ───────────────────────────────────
FEATURE_COLS = [
    "soh_prev", "delta_soh", "coulombic_eff",
    "dc_ir_norm",        
    "temperature_max", "cycle_norm",
]
TARGET_COL         = "soh"
SEQ_LEN            = 50
PREDICTION_HORIZON = 50
BATCH_SIZE         = 64
MC_SAMPLES         = 50
RANDOM_SEED        = 42
TRAIN_FRAC         = 0.70
VAL_FRAC           = 0.15
SOH_EOL_THRESHOLD  = 0.80
N_INFERENCE_RUNS   = 200

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════
# MODEL — paste your exact architecture here
# ══════════════════════════════════════════════════════════════════════════

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
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pe(pos)


class MambaLikeBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, d_model * 2)
        self.B_proj   = nn.Linear(d_model, d_state, bias=False)
        self.C_proj   = nn.Linear(d_model, d_state, bias=False)
        self.A_log    = nn.Parameter(torch.log(torch.rand(d_state) * 0.5 + 0.4))
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
        A       = torch.sigmoid(self.A_log)
        h       = x.new_zeros(B, self.d_state)
        outputs = []
        for t in range(T):
            x_t    = x[:, t, :]
            B_t    = torch.sigmoid(self.B_proj(x_t))
            x_proj = self.C_proj(x_t)
            h      = A * h + B_t * x_proj
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        residual        = x
        x_n             = self.norm1(x)
        gate, x_proj    = self.in_proj(x_n).chunk(2, dim=-1)
        h_seq           = self.ssm_scan(x_proj)
        y               = torch.sigmoid(gate) * self.out_proj(h_seq)
        x               = residual + self.drop(y)
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
        return torch.sigmoid(out) if self.bounded else out


class CNNMambaUQ(nn.Module):
    def __init__(self, n_features=6, seq_len=50,
                 cnn_channels=[32,64,128], cnn_kernel=3,
                 d_model=192, d_state=16, n_layers=3,
                 dropout=0.25, mc_samples=50):
        super().__init__()
        self.mc_samples = mc_samples
        self.cnn     = CNNEncoder(n_features, cnn_channels, cnn_kernel)
        self.proj    = nn.Linear(cnn_channels[-1], d_model)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=seq_len + 10)
        self.mamba   = MambaEncoder(d_model, d_state, n_layers, dropout)
        self.head_future  = RegressionHead(d_model, dropout, bounded=True)
        self.head_current = RegressionHead(d_model, dropout, bounded=True)
        self.head_eol     = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid()
        )
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  CNN-Mamba-UQ mlt  |  parameters: {n_params:,}")

    def encode(self, x):
        x = self.cnn(x)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.mamba(x)
        return x[:, -1, :]

    def forward(self, x):
        h = self.encode(x)
        return {
            "future_soh"  : self.head_future(h),
            "current_soh" : self.head_current(h),
            "eol_prob"    : self.head_eol(h),
        }

    def predict_future(self, x):
        return self.head_future(self.encode(x))

    # def mc_predict(self, x):
    #     self.train()
    #     preds = []
    #     for _ in range(self.mc_samples):
    #         preds.append(self.head_future(self.encode(x)))
    #     preds = torch.cat(preds, dim=-1)   # (B, mc_samples)
    #     self.eval()
    #     mean    = preds.mean(dim=-1)
    #     std     = preds.std(dim=-1)
    #     return {
    #         "mean"    : mean.cpu().numpy(),
    #         "std"     : std.cpu().numpy(),
    #         "ci_low"  : (mean - 1.96 * std).cpu().numpy(),
    #         "ci_high" : (mean + 1.96 * std).cpu().numpy(),
    #         "eol_prob": self.head_eol(self.encode(x)).squeeze().cpu().detach().numpy(),
    #     }
    def mc_predict(self, x):
      self.train()   # dropout active
      preds = []
      with torch.no_grad():              # ← no gradient tracking needed at inference
         for _ in range(self.mc_samples):
            preds.append(self.head_future(self.encode(x)))
         preds = torch.cat(preds, dim=-1)
         eol   = self.head_eol(self.encode(x)).squeeze()
      self.eval()

      mean = preds.mean(dim=-1)
      std  = preds.std(dim=-1)
      return {
        "mean"    : mean.cpu().numpy(),
        "std"     : std.cpu().numpy(),
        "ci_low"  : (mean - 1.96 * std).cpu().numpy(),
        "ci_high" : (mean + 1.96 * std).cpu().numpy(),
        "eol_prob": eol.cpu().numpy(),
    }


# ══════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def rebuild_test_dataset(df, scaler, test_cells,
                         seq_len=SEQ_LEN, horizon=PREDICTION_HORIZON):
    """
    Re-build the test TensorDataset from the raw dataframe.
    Returns TensorDataset(X, y_future, y_current, y_eol)
    and a list of cell_id labels per sequence for per-cell analysis.
    """
    X_list, yf_list, yc_list, ye_list, cell_labels = [], [], [], [], []
    df_test = df[df["cell_id"].isin(test_cells)].copy()
    df_test["cycle_norm"] = df_test["cycle_norm"].clip(0, 1)

    for cell_id, grp in df_test.groupby("cell_id"):
        grp  = grp.sort_values("cycle_index").reset_index(drop=True)
        Xc   = scaler.transform(grp[FEATURE_COLS].values).astype(np.float32)
        soh  = grp[TARGET_COL].values.astype(np.float32)
        n    = len(grp)
        if n <= seq_len + horizon:
            continue
        for i in range(n - seq_len - horizon + 1):
            yf = soh[i + seq_len + horizon - 1]
            yc = soh[i + seq_len - 1]
            ye = 1.0 if yf < SOH_EOL_THRESHOLD else 0.0
            X_list.append(Xc[i:i + seq_len])
            yf_list.append(yf)
            yc_list.append(yc)
            ye_list.append(ye)
            cell_labels.append(cell_id)

    X  = torch.tensor(np.array(X_list),  dtype=torch.float32)
    yf = torch.tensor(np.array(yf_list), dtype=torch.float32)
    yc = torch.tensor(np.array(yc_list), dtype=torch.float32)
    ye = torch.tensor(np.array(ye_list), dtype=torch.float32)
    return TensorDataset(X, yf, yc, ye), cell_labels


# ══════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, ci_low=None, ci_high=None):
    mae  = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    r2   = r2_score(y_true, y_pred)
    maxe = float(np.max(np.abs(y_true - y_pred)) * 100)

    metrics = {
        "MAE  (%)": round(mae,  4),
        "RMSE (%)": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "R2       ": round(r2,   4),
        "MaxE (%)": round(maxe, 4),
    }

    if ci_low is not None and ci_high is not None:
        covered = ((y_true >= ci_low) & (y_true <= ci_high))
        picp    = float(covered.mean() * 100)
        y_range = float(y_true.max() - y_true.min()) + 1e-9
        pinaw   = float(np.mean(ci_high - ci_low) / y_range)
        metrics["PICP (%) [UQ]"] = round(picp,  4)
        metrics["PINAW    [UQ]"] = round(pinaw, 4)

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# TIER 1 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════

def run_performance_evaluation(model, test_ds, cell_labels, device):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    model.to(device)

    all_mean, all_std       = [], []
    all_ci_lo, all_ci_hi   = [], []
    all_true, all_eol_true = [], []
    all_eol_pred           = []

    print("\n  Running MC prediction over test set …")
    for X_batch, yf_batch, _, ye_batch in loader:
        X_batch = X_batch.to(device)
        res     = model.mc_predict(X_batch)
        all_mean.append(res["mean"])
        all_std.append(res["std"])
        all_ci_lo.append(res["ci_low"])
        all_ci_hi.append(res["ci_high"])
        all_true.append(yf_batch.numpy())
        all_eol_true.append(ye_batch.numpy())
        # eol_prob shape can be scalar or array
        ep = res["eol_prob"]
        if ep.ndim == 0:
            ep = np.array([float(ep)])
        all_eol_pred.append(ep.flatten())

    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_mean)
    y_std   = np.concatenate(all_std)
    ci_low  = np.concatenate(all_ci_lo)
    ci_high = np.concatenate(all_ci_hi)
    eol_t   = np.concatenate(all_eol_true)
    eol_p   = np.concatenate(all_eol_pred)

    metrics = compute_metrics(y_true, y_pred, ci_low, ci_high)

    # EOL classification metrics
    eol_pred_bin = (eol_p >= 0.5).astype(float)
    eol_acc = float((eol_pred_bin == eol_t).mean() * 100)
    tp = ((eol_pred_bin == 1) & (eol_t == 1)).sum()
    fp = ((eol_pred_bin == 1) & (eol_t == 0)).sum()
    fn = ((eol_pred_bin == 0) & (eol_t == 1)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    metrics["EOL Accuracy (%)"]  = round(eol_acc, 4)
    metrics["EOL F1 score"]      = round(float(f1), 4)
    metrics["EOL Precision"]     = round(float(precision), 4)
    metrics["EOL Recall"]        = round(float(recall), 4)

    print(f"\n{'='*60}")
    print(f"  TIER 1 — PERFORMANCE METRICS  (test set, {PREDICTION_HORIZON}-cycle horizon)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"    {k:<28}  {v:>10}")
    print(f"{'='*60}\n")

    # save raw predictions
    df_pred = pd.DataFrame({
        "cell_id" : cell_labels,
        "y_true"  : y_true,
        "y_pred"  : y_pred,
        "y_std"   : y_std,
        "ci_low"  : ci_low,
        "ci_high" : ci_high,
        "error"   : y_true - y_pred,
        "eol_true": eol_t,
        "eol_pred": eol_p,
    })
    df_pred.to_csv(EVAL_DIR / "predictions.csv", index=False)

    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric","Value"])
    df_metrics.to_csv(EVAL_DIR / "metrics_summary.csv", index=False)

    return df_pred, metrics


# ══════════════════════════════════════════════════════════════════════════
# PER-CELL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def run_per_cell_analysis(df_pred):
    rows = []
    for cell_id, grp in df_pred.groupby("cell_id"):
        if len(grp) < 5:
            continue
        y_t = grp["y_true"].values
        y_p = grp["y_pred"].values
        ci_l = grp["ci_low"].values
        ci_h = grp["ci_high"].values
        m = compute_metrics(y_t, y_p, ci_l, ci_h)
        rows.append({
            "cell_id"    : cell_id,
            "n_sequences": len(grp),
            "soh_min"    : round(y_t.min(), 4),
            "soh_max"    : round(y_t.max(), 4),
            "MAE (%)"    : m["MAE  (%)"],
            "RMSE (%)"   : m["RMSE (%)"],
            "R2"         : m["R2       "],
            "MaxE (%)"   : m["MaxE (%)"],
            "PICP (%)"   : m.get("PICP (%) [UQ]", None),
        })

    df_cells = pd.DataFrame(rows).sort_values("R2")
    df_cells.to_csv(EVAL_DIR / "per_cell_metrics.csv", index=False)

    print(f"  PER-CELL R² SUMMARY")
    print(f"  {'Cell':<24}  {'n_seq':>5}  {'SOH range':>12}  {'MAE%':>7}  {'R2':>7}  {'PICP%':>7}")
    for _, r in df_cells.iterrows():
        soh_range = f"[{r['soh_min']:.3f},{r['soh_max']:.3f}]"
        print(f"  {str(r['cell_id']):<24}  {r['n_sequences']:>5}  "
              f"{soh_range:>12}  {r['MAE (%)']:>7.4f}  "
              f"{r['R2']:>7.4f}  {str(r['PICP (%)'])[:6]:>7}")

    mean_r2  = df_cells["R2"].mean()
    mean_mae = df_cells["MAE (%)"].mean()
    print(f"\n  Mean per-cell R² : {mean_r2:.4f}")
    print(f"  Mean per-cell MAE: {mean_mae:.4f}%\n")
    return df_cells


# ══════════════════════════════════════════════════════════════════════════
# TIER 2 — DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════

def run_deployment_evaluation(model, test_ds):
    model_cpu = model.to(CPU_DEVICE)
    model_cpu.eval()

    # single-sample inference latency
    X_single = test_ds[0][0].unsqueeze(0).to(CPU_DEVICE)

    # warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model_cpu.predict_future(X_single)

    latencies = []
    for _ in range(N_INFERENCE_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu.predict_future(X_single)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_mean = float(np.mean(latencies))
    lat_std  = float(np.std(latencies))
    lat_p95  = float(np.percentile(latencies, 95))

    # MC predict latency (full UQ)
    mc_lats = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model_cpu.mc_predict(X_single)
        mc_lats.append((time.perf_counter() - t0) * 1000)
    mc_lat_mean = float(np.mean(mc_lats))

    # batch throughput
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)
    t_start = time.perf_counter()
    with torch.no_grad():
        for X_b, *_ in loader:
            _ = model_cpu.predict_future(X_b.to(CPU_DEVICE))
    throughput = len(test_ds) / (time.perf_counter() - t_start)

    # model size
    n_params  = sum(p.numel() for p in model_cpu.parameters())
    disk_mb   = CKPT_PATH.stat().st_size / 1e6 if CKPT_PATH.exists() else 0.0

    # peak RAM
    tracemalloc.start()
    with torch.no_grad():
        _ = model_cpu.predict_future(X_single)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak_mem / 1024

    metrics = {
        "Parameters"              : f"{n_params:,}",
        "Disk size (MB)"          : f"{disk_mb:.2f}",
        "Peak RAM (KB)"           : f"{peak_kb:.1f}",
        "Inference latency mean (ms)"  : f"{lat_mean:.3f}",
        "Inference latency std  (ms)"  : f"{lat_std:.3f}",
        "Inference latency p95  (ms)"  : f"{lat_p95:.3f}",
        "MC-predict latency mean (ms)" : f"{mc_lat_mean:.1f}",
        "Throughput (samples/s)"  : f"{throughput:.0f}",
        "BMS real-time OK (p95<100ms)" : "YES" if lat_p95 < 100 else "NO",
        "Flash budget OK (<2MB)"  : "YES" if disk_mb < 2.0 else "NO",
    }

    print(f"{'='*60}")
    print(f"  TIER 2 — DEPLOYMENT METRICS  (CPU simulation)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"    {k:<38}  {v:>10}")
    print(f"{'='*60}\n")

    pd.DataFrame(list(metrics.items()),
                 columns=["Metric","Value"]).to_csv(
        EVAL_DIR / "deployment_metrics.csv", index=False)

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_global(df_pred):
    if not PLOT: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Predicted vs True
    ax = axes[0]
    ax.scatter(df_pred["y_true"], df_pred["y_pred"],
               alpha=0.25, s=3, color="#2196F3")
    mn = df_pred["y_true"].min(); mx = df_pred["y_true"].max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("True SOH"); ax.set_ylabel("Predicted SOH")
    ax.set_title("(a) Predicted vs True SOH")
    ax.legend(fontsize=8)

    # (b) Error distribution
    ax = axes[1]
    err_pct = df_pred["error"].values * 100
    ax.hist(err_pct, bins=60, color="#4CAF50", edgecolor="white", lw=0.3)
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    ax.axvline(err_pct.mean(), color="orange", lw=1.5,
               linestyle="--", label=f"mean={err_pct.mean():.3f}%")
    ax.set_xlabel("Error (%)"); ax.set_ylabel("Count")
    ax.set_title("(b) Prediction Error Distribution")
    ax.legend(fontsize=8)

    # (c) Uncertainty vs |Error|
    ax = axes[2]
    ax.scatter(df_pred["y_std"] * 100,
               np.abs(df_pred["error"]) * 100,
               alpha=0.2, s=3, color="#9C27B0")
    ax.set_xlabel("Predicted Uncertainty std (%)")
    ax.set_ylabel("|Error| (%)")
    ax.set_title("(c) Uncertainty vs |Error|\n(good UQ: positive correlation)")

    plt.suptitle(f"CNN-Mamba-UQ  —  {PREDICTION_HORIZON}-cycle-ahead SOH Prediction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "global_evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_trajectories(df_pred, df, test_cells, scaler, model, device, max_cells=6):
    """
    For each test cell, reconstruct the full predicted SOH trajectory
    and plot it against the true SOH with CI band.
    """
    if not PLOT: return

    cells_to_plot = list(test_cells)[:max_cells]
    df_test = df[df["cell_id"].isin(cells_to_plot)].copy()
    df_test["cycle_norm"] = df_test["cycle_norm"].clip(0, 1)

    for cell_id in cells_to_plot:
        grp = df_test[df_test["cell_id"] == cell_id].sort_values("cycle_index")
        if len(grp) <= SEQ_LEN + PREDICTION_HORIZON:
            continue

        Xc  = scaler.transform(grp[FEATURE_COLS].values).astype(np.float32)
        soh = grp[TARGET_COL].values.astype(np.float32)
        cyc = grp["cycle_index"].values
        n   = len(grp)

        pred_cycles, pred_mean, pred_low, pred_high = [], [], [], []
        step = 5   # predict every 5 cycles to keep it fast

        model.to(device)
        for i in range(0, n - SEQ_LEN - PREDICTION_HORIZON + 1, step):
            X_w = torch.tensor(
                Xc[i:i + SEQ_LEN], dtype=torch.float32
            ).unsqueeze(0).to(device)
            res = model.mc_predict(X_w)
            pred_cycles.append(cyc[i + SEQ_LEN + PREDICTION_HORIZON - 1])
            pred_mean.append(float(res["mean"][0]))
            pred_low.append(float(res["ci_low"][0]))
            pred_high.append(float(res["ci_high"][0]))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cyc, soh, color="black", lw=1.5, label="True SOH", zorder=3)
        ax.plot(pred_cycles, pred_mean, color="#E91E63", lw=1.5,
                label="Predicted SOH", zorder=4)
        ax.fill_between(pred_cycles, pred_low, pred_high,
                        alpha=0.25, color="#FF9800", label="95% CI")
        ax.axhline(SOH_EOL_THRESHOLD, color="red", lw=1,
                   linestyle=":", label="EOL threshold (80%)")
        ax.set_xlabel("Cycle index"); ax.set_ylabel("SOH")
        ax.set_title(f"SOH Trajectory — Cell {cell_id}\n"
                     f"({PREDICTION_HORIZON}-cycle-ahead prediction)")
        ax.legend(fontsize=8); ax.set_ylim(0.70, 1.05)
        plt.tight_layout()
        safe = str(cell_id).replace("/", "_").replace("\\", "_")
        out  = PLOTS_DIR / f"trajectory_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")


def plot_training_curve():
    if not PLOT: return
    hist_path = RESULTS_DIR / "training_history_mlt.csv"
    if not hist_path.exists():
        return
    hist = pd.read_csv(hist_path)
    plt.figure(figsize=(8, 4))
    plt.plot(hist["epoch"], hist["train"], label="Train loss")
    plt.plot(hist["epoch"], hist["val"],   label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Curve — CNN-Mamba-UQ MLT")
    plt.legend(); plt.tight_layout()
    out = PLOTS_DIR / "training_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  MODEL EVALUATION  —  CNN-Mamba-UQ MLT")
    print("="*60)

    # ── load model ────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model = CNNMambaUQ()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE,
                                     weights_only=False))
    model.to(DEVICE)
    model.eval()
    print(f"  Device: {DEVICE}")

    # ── load data ─────────────────────────────────────────────────────
    print(f"\nLoading dataset: {DATASET_PKL}")
    df = pd.read_pickle(DATASET_PKL)
    print(f"  {len(df):,} rows | {df['cell_id'].nunique()} cells")

    # load scaler
    print(f"Loading scaler: {SCALER_PKL}")
    with open(SCALER_PKL, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    # load test cells
    if TEST_CELLS_PKL.exists():
        with open(TEST_CELLS_PKL, "rb") as f:
            test_cells = pickle.load(f)
        print(f"  Test cells loaded: {len(test_cells)} cells")
    else:
        # reconstruct split deterministically
        rng   = np.random.default_rng(RANDOM_SEED)
        cells = df["cell_id"].unique()
        idx   = rng.permutation(len(cells))
        n_tr  = int(len(cells) * TRAIN_FRAC)
        n_val = int(len(cells) * VAL_FRAC)
        test_cells = cells[idx[n_tr + n_val:]]
        print(f"  Test cells reconstructed: {len(test_cells)} cells")

    # ── build / load test dataset ─────────────────────────────────────
    if TEST_DS_PT.exists():
        print(f"Loading test dataset: {TEST_DS_PT}")
        test_ds     = torch.load(TEST_DS_PT, weights_only=False)
        # rebuild cell_labels for per-cell analysis
        _, cell_labels = rebuild_test_dataset(df, scaler, test_cells)
    else:
        print("Building test dataset from scratch …")
        test_ds, cell_labels = rebuild_test_dataset(df, scaler, test_cells)

    print(f"  {len(test_ds):,} test sequences\n")

    # ── evaluations ───────────────────────────────────────────────────
    df_pred, perf_metrics = run_performance_evaluation(
        model, test_ds, cell_labels, DEVICE)

    df_cells = run_per_cell_analysis(df_pred)

    run_deployment_evaluation(model, test_ds)

    # ── plots ─────────────────────────────────────────────────────────
    print("Generating plots …")
    plot_training_curve()
    plot_global(df_pred)
    plot_trajectories(df_pred, df, test_cells, scaler, model, DEVICE, max_cells=6)

    # ── final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"  All outputs saved to: {EVAL_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()