"""
MODEL EVALUATION (50-CYCLE HORIZON)
-----------------
Two evaluation tiers:

  TIER 1 — PERFORMANCE METRICS
    MAE, RMSE, MAPE, R², MaxE, PICP, PINAW

  TIER 2 — DEPLOYMENT METRICS
    Inference latency, model size, memory footprint, throughput

Outputs (saved to results_50cycle/)
-------
  evaluation_report.csv
  metrics_summary.csv
  deployment_metrics.csv
  plots/
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
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from 50-cycle config
from configurations.config_training_50 import (
    RESULTS_DIR, MODEL_SAVE_DIR, SCALER_PATH,
    FEATURE_COLS, TARGET_COL, SEQ_LEN, PREDICTION_HORIZON,
    BATCH_SIZE, MC_SAMPLES, PLOTS_DIR
)

from model_architecture.cnn_mamba_uq_imprv import CNNMambaUQ

INFERENCE_DEVICE = "cpu"
N_INFERENCE_RUNS = 200

DEVICE = torch.device(INFERENCE_DEVICE)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT = True
except ImportError:
    PLOT = False
    print("matplotlib not found — skipping plots")

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0.01
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))

def picp(y_true: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> float:
    covered = ((y_true >= ci_low) & (y_true <= ci_high)).mean()
    return float(covered)

def pinaw(ci_low: np.ndarray, ci_high: np.ndarray, y_range: float) -> float:
    return float(np.mean(ci_high - ci_low) / y_range)

def evaluate_performance(model: nn.Module, test_ds: TensorDataset, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()

    all_mean, all_std = [], []
    all_ci_lo, all_ci_hi = [], []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            result = model.mc_predict(X_batch)
            all_mean.append(result["mean"])
            all_std.append(result["std"])
            all_ci_lo.append(result["ci_low"])
            all_ci_hi.append(result["ci_high"])
            all_true.append(y_batch.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_mean)
    y_std = np.concatenate(all_std)
    ci_low = np.concatenate(all_ci_lo)
    ci_high = np.concatenate(all_ci_hi)

    df_pred = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_std": y_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "error": y_true - y_pred,
    })
    df_pred.to_csv(RESULTS_DIR / "evaluation_report_50cycle.csv", index=False)

    y_range = float(y_true.max() - y_true.min())
    metrics = {
        "MAE (%)": mean_absolute_error(y_true, y_pred) * 100,
        "RMSE (%)": np.sqrt(mean_squared_error(y_true, y_pred)) * 100,
        "MAPE (%)": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MaxE (%)": max_error(y_true, y_pred) * 100,
        "PICP (%)": picp(y_true, ci_low, ci_high) * 100,
        "PINAW": pinaw(ci_low, ci_high, y_range),
    }

    print(f"\n{'='*60}")
    print(f"TIER 1 — PERFORMANCE METRICS (50-cycle prediction)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:>10.4f}")
    print(f"{'='*60}\n")

    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df_metrics.to_csv(RESULTS_DIR / "metrics_summary_50cycle.csv", index=False)

    return df_pred, metrics

def evaluate_deployment(model: nn.Module, test_ds: TensorDataset) -> pd.DataFrame:
    model_cpu = model.to(torch.device("cpu"))
    model_cpu.eval()

    X_single = test_ds[0][0].unsqueeze(0).cpu()

    latencies = []
    for _ in range(N_INFERENCE_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(X_single)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_mean = float(np.mean(latencies[10:]))
    lat_std = float(np.std(latencies[10:]))
    lat_p95 = float(np.percentile(latencies[10:], 95))

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    t_start = time.perf_counter()
    with torch.no_grad():
        for X_b, _ in loader:
            _ = model_cpu(X_b.cpu())
    t_total = time.perf_counter() - t_start
    throughput = len(test_ds) / t_total

    n_params = sum(p.numel() for p in model_cpu.parameters())
    ckpt_path = MODEL_SAVE_DIR / "cnn_mamba_uq_50cycle_best.pt"
    disk_mb = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0.0

    tracemalloc.start()
    with torch.no_grad():
        _ = model_cpu(X_single)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_kb = peak_mem / 1024

    metrics = {
        "Parameters": f"{n_params:,}",
        "Disk size (MB)": f"{disk_mb:.2f}",
        "Peak RAM (KB)": f"{peak_mem_kb:.1f}",
        "Latency mean (ms)": f"{lat_mean:.3f}",
        "Latency std (ms)": f"{lat_std:.3f}",
        "Latency p95 (ms)": f"{lat_p95:.3f}",
        "Throughput (samp/s)": f"{throughput:.0f}",
        "BMS real-time OK?": "YES" if lat_p95 < 100 else "NO",
    }

    print(f"{'='*60}")
    print(f"TIER 2 — DEPLOYMENT METRICS (CPU simulation)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<18} {v:>12}")
    print(f"{'='*60}\n")

    df_dep = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df_dep.to_csv(RESULTS_DIR / "deployment_metrics_50cycle.csv", index=False)
    return df_dep

def plot_results(df_pred: pd.DataFrame) -> None:
    if not PLOT:
        return

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df_pred["y_true"], df_pred["y_pred"], alpha=0.3, s=4, color="#2196F3")
    mn = df_pred["y_true"].min()
    mx = df_pred["y_true"].max()
    ax1.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
    ax1.set_xlabel("True SOH")
    ax1.set_ylabel("Predicted SOH")
    ax1.set_title("Predicted vs True SOH (50-cycle horizon)")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_pred["error"] * 100, bins=60, color="#4CAF50", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Error Distribution")

    ax3 = fig.add_subplot(gs[1, 0])
    sample = df_pred.sample(min(500, len(df_pred)), random_state=42).sort_values("y_true")
    ax3.fill_between(range(len(sample)), sample["ci_low"], sample["ci_high"], alpha=0.3, color="#FF9800", label="95% CI")
    ax3.scatter(range(len(sample)), sample["y_true"], s=4, color="black", zorder=3, label="True SOH")
    ax3.scatter(range(len(sample)), sample["y_pred"], s=4, color="#E91E63", zorder=3, label="Predicted SOH", alpha=0.7)
    ax3.set_xlabel("Sample (sorted by true SOH)")
    ax3.set_ylabel("SOH")
    ax3.set_title("Prediction Intervals (500 samples) - 50 cycles ahead")
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_pred["y_std"] * 100, np.abs(df_pred["error"]) * 100, alpha=0.3, s=4, color="#9C27B0")
    ax4.set_xlabel("Predicted Uncertainty (std, %)")
    ax4.set_ylabel("|Error| (%)")
    ax4.set_title("Uncertainty vs Absolute Error")

    plt.suptitle("CNN-Mamba-UQ — SOH Estimation Results (50-cycle prediction)", fontsize=13, fontweight="bold")
    out = PLOTS_DIR / "evaluation_results_50cycle.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out}")

def plot_training_curve() -> None:
    if not PLOT:
        return
    hist_path = RESULTS_DIR / "training_history_50cycle.csv"
    if not hist_path.exists():
        return
    hist = pd.read_csv(hist_path)
    plt.figure(figsize=(8, 4))
    plt.plot(hist["epoch"], hist["train_mse"] * 1e4, label="Train MSE x1e-4")
    plt.plot(hist["epoch"], hist["val_mse"] * 1e4, label="Val MSE x1e-4")
    plt.xlabel("Epoch")
    plt.ylabel("MSE x1e-4")
    plt.title("Training Curve — CNN-Mamba-UQ (50-cycle horizon)")
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "training_curve_50cycle.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot -> {out}")

def main():
    print("\n" + "="*60)
    print("MODEL EVALUATION STARTED (50-CYCLE HORIZON)")
    print("="*60)

    print("\nLoading CNN-Mamba-UQ checkpoint...")
    model = CNNMambaUQ()
    ckpt = MODEL_SAVE_DIR / "cnn_mamba_uq_50cycle_best.pt"
    
    if not ckpt.exists():
        print(f"ERROR: Checkpoint not found at {ckpt}")
        print("Please train the 50-cycle model first!")
        return
    
    state_dict = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded from {ckpt}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prediction horizon: {PREDICTION_HORIZON} cycles")

    print("\nLoading test dataset...")
    test_ds_path = RESULTS_DIR / "test_dataset_50cycle.pt"
    if not test_ds_path.exists():
        print(f"ERROR: Test dataset not found at {test_ds_path}")
        print("Please train the 50-cycle model first!")
        return
    
    test_ds = torch.load(test_ds_path, weights_only=False)
    print(f"{len(test_ds):,} test sequences")

    print("\nRunning performance evaluation...")
    df_pred, perf_metrics = evaluate_performance(model, test_ds, DEVICE)

    print("\nRunning deployment evaluation...")
    evaluate_deployment(model, test_ds)

    print("\nGenerating plots...")
    plot_training_curve()
    plot_results(df_pred)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nOutput files saved in: {RESULTS_DIR}")
    print("  - evaluation_report_50cycle.csv (predictions with uncertainties)")
    print("  - metrics_summary_50cycle.csv (performance metrics)")
    print("  - deployment_metrics_50cycle.csv (latency, throughput, etc.)")
    print("  - plots/evaluation_results_50cycle.png")
    print("  - plots/training_curve_50cycle.png")

if __name__ == "__main__":
    main()