# model_evaluation/evaluate_mlt.py
"""
EVALUATION FOR MULTI-TASK CNN-MAMBA-UQ (50-CYCLE HORIZON)
==========================================================
Evaluates the multi-task model on:
  - Future SOH prediction (primary task)
  - Current SOH prediction (auxiliary)
  - EOL probability prediction (auxiliary)
  
Also computes deployment metrics for edge BMS feasibility.
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
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training_mlt import (
    RESULTS_DIR, MODEL_SAVE_DIR, SCALER_PATH,
    FEATURE_COLS, TARGET_COL, SOH_EOL_THRESHOLD,
    BATCH_SIZE, MC_SAMPLES, PREDICTION_HORIZON
)

from cnn_mamba_uq_mlt import CNNMambaUQ

INFERENCE_DEVICE = "cpu"
N_INFERENCE_RUNS = 200

DEVICE = torch.device(INFERENCE_DEVICE)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT = True
except ImportError:
    PLOT = False
    print("matplotlib not found — skipping plots")


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

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


# ============================================================================
# PERFORMANCE EVALUATION (TIER 1)
# ============================================================================

def evaluate_performance(model, test_ds, device):
    """
    Evaluate multi-task model on test set.
    Returns predictions DataFrame and metrics dict.
    """
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.to(device)

    all_future_pred = []
    all_current_pred = []
    all_eol_pred = []
    all_future_true = []
    all_current_true = []
    all_eol_true = []
    all_std = []
    all_ci_low = []
    all_ci_high = []

    for batch in loader:
        if len(batch) == 4:
            X_batch, y_future, y_current, y_eol = batch
        else:
            # Fallback for older dataset format
            X_batch, y_future = batch[0], batch[1]
            y_current = batch[2] if len(batch) > 2 else None
            y_eol = batch[3] if len(batch) > 3 else None

        X_batch = X_batch.to(device)

        # MC predictions for future SOH (uncertainty)
        result = model.mc_predict(X_batch)
        all_future_pred.extend(result["mean"])
        all_std.extend(result["std"])
        all_ci_low.extend(result["ci_low"])
        all_ci_high.extend(result["ci_high"])
        all_future_true.extend(y_future.numpy())

        # Deterministic predictions for auxiliary tasks
        with torch.no_grad():
            outputs = model(X_batch)
            all_current_pred.extend(outputs["current_soh"].cpu().numpy().flatten())
            all_eol_pred.extend(outputs["eol_prob"].cpu().numpy().flatten())

        if y_current is not None:
            all_current_true.extend(y_current.numpy())
        if y_eol is not None:
            all_eol_true.extend(y_eol.numpy())

    # Convert to numpy arrays
    future_true = np.array(all_future_true)
    future_pred = np.array(all_future_pred)
    future_std = np.array(all_std)
    ci_low = np.array(all_ci_low)
    ci_high = np.array(all_ci_high)

    current_true = np.array(all_current_true) if all_current_true else None
    current_pred = np.array(all_current_pred) if all_current_pred else None
    eol_true = np.array(all_eol_true) if all_eol_true else None
    eol_pred = np.array(all_eol_pred) if all_eol_pred else None

    # Future SOH metrics
    y_range = float(future_true.max() - future_true.min())
    metrics = {
        "Future SOH - MAE (%)": mean_absolute_error(future_true, future_pred) * 100,
        "Future SOH - RMSE (%)": np.sqrt(mean_squared_error(future_true, future_pred)) * 100,
        "Future SOH - MAPE (%)": mape(future_true, future_pred),
        "Future SOH - R²": r2_score(future_true, future_pred),
        "Future SOH - MaxE (%)": max_error(future_true, future_pred) * 100,
        "Future SOH - PICP (%)": picp(future_true, ci_low, ci_high) * 100,
        "Future SOH - PINAW": pinaw(ci_low, ci_high, y_range),
    }

    # Current SOH metrics (if available)
    if current_true is not None and len(current_true) > 0:
        metrics["Current SOH - MAE (%)"] = mean_absolute_error(current_true, current_pred) * 100
        metrics["Current SOH - RMSE (%)"] = np.sqrt(mean_squared_error(current_true, current_pred)) * 100
        metrics["Current SOH - R²"] = r2_score(current_true, current_pred)

    # EOL prediction metrics (if available)
    if eol_true is not None and len(eol_true) > 0:
        eol_pred_binary = (eol_pred > 0.5).astype(float)
        metrics["EOL - Accuracy (%)"] = accuracy_score(eol_true, eol_pred_binary) * 100
        metrics["EOL - Precision (%)"] = precision_score(eol_true, eol_pred_binary, zero_division=0) * 100
        metrics["EOL - Recall (%)"] = recall_score(eol_true, eol_pred_binary, zero_division=0) * 100
        metrics["EOL - F1 Score"] = f1_score(eol_true, eol_pred_binary, zero_division=0)
        try:
            metrics["EOL - AUC-ROC"] = roc_auc_score(eol_true, eol_pred)
        except:
            metrics["EOL - AUC-ROC"] = 0.0

    # Save predictions
    df_pred = pd.DataFrame({
        "future_soh_true": future_true,
        "future_soh_pred": future_pred,
        "future_soh_std": future_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "error": future_true - future_pred,
    })
    
    if current_true is not None:
        df_pred["current_soh_true"] = current_true
        df_pred["current_soh_pred"] = current_pred
        
    if eol_true is not None:
        df_pred["eol_true"] = eol_true
        df_pred["eol_pred"] = eol_pred
        df_pred["eol_pred_binary"] = (eol_pred > 0.5).astype(int)

    df_pred.to_csv(RESULTS_DIR / "evaluation_report_mlt.csv", index=False)

    # Print metrics
    print(f"\n{'='*60}")
    print(f"TIER 1 — PERFORMANCE METRICS (50-cycle horizon)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<30} {v:>10.4f}")
    print(f"{'='*60}\n")

    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df_metrics.to_csv(RESULTS_DIR / "metrics_summary_mlt.csv", index=False)

    return df_pred, metrics


# ============================================================================
# DEPLOYMENT EVALUATION (TIER 2)
# ============================================================================

def evaluate_deployment(model, test_ds):
    """Measure deployment-relevant metrics on CPU (edge BMS simulation)."""
    model_cpu = model.to(torch.device("cpu"))
    model_cpu.eval()

    # Single sample for latency test
    X_single = test_ds[0][0].unsqueeze(0).cpu()

    # Inference latency
    latencies = []
    for _ in range(N_INFERENCE_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu.predict_future(X_single)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_mean = float(np.mean(latencies[10:]))
    lat_std = float(np.std(latencies[10:]))
    lat_p95 = float(np.percentile(latencies[10:], 95))

    # Throughput
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].cpu()
            _ = model_cpu.predict_future(X_batch)
    t_total = time.perf_counter() - t_start
    throughput = len(test_ds) / t_total

    # Model size
    n_params = sum(p.numel() for p in model_cpu.parameters())
    ckpt_path = MODEL_SAVE_DIR / "cnn_mamba_uq_mlt_best.pt"
    disk_mb = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0.0

    # Peak memory
    tracemalloc.start()
    with torch.no_grad():
        _ = model_cpu.predict_future(X_single)
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
        print(f"  {k:<20} {v:>15}")
    print(f"{'='*60}\n")

    df_dep = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df_dep.to_csv(RESULTS_DIR / "deployment_metrics_mlt.csv", index=False)
    return df_dep


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(df_pred, metrics):
    """Generate evaluation plots."""
    if not PLOT:
        return

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Predicted vs True (Future SOH)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df_pred["future_soh_true"], df_pred["future_soh_pred"], alpha=0.3, s=4, color="#2196F3")
    mn = df_pred["future_soh_true"].min()
    mx = df_pred["future_soh_true"].max()
    ax1.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
    ax1.set_xlabel("True SOH")
    ax1.set_ylabel("Predicted SOH")
    ax1.set_title(f"Future SOH (50 cycles ahead)\nR² = {metrics['Future SOH - R²']:.4f}")
    ax1.legend()

    # Plot 2: Error distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_pred["error"] * 100, bins=60, color="#4CAF50", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Error Distribution")

    # Plot 3: Current SOH (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if "current_soh_true" in df_pred.columns:
        ax3.scatter(df_pred["current_soh_true"], df_pred["current_soh_pred"], alpha=0.3, s=4, color="#FF9800")
        ax3.plot([0, 1], [0, 1], "r--", lw=1.5)
        ax3.set_xlabel("True Current SOH")
        ax3.set_ylabel("Predicted Current SOH")
        ax3.set_title("Current SOH (Auxiliary Task)")
    else:
        ax3.text(0.5, 0.5, "Current SOH not available", ha="center", va="center")
        ax3.set_title("Current SOH (Auxiliary Task)")

    # Plot 4: Prediction intervals
    ax4 = fig.add_subplot(gs[1, 0])
    sample = df_pred.sample(min(500, len(df_pred)), random_state=42).sort_values("future_soh_true")
    ax4.fill_between(range(len(sample)), sample["ci_low"], sample["ci_high"], alpha=0.3, color="#FF9800", label="95% CI")
    ax4.scatter(range(len(sample)), sample["future_soh_true"], s=4, color="black", zorder=3, label="True SOH")
    ax4.scatter(range(len(sample)), sample["future_soh_pred"], s=4, color="#E91E63", zorder=3, label="Predicted SOH", alpha=0.7)
    ax4.set_xlabel("Sample (sorted by true SOH)")
    ax4.set_ylabel("SOH")
    ax4.set_title("Prediction Intervals (500 samples)")
    ax4.legend(fontsize=7)

    # Plot 5: Uncertainty vs Error
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df_pred["future_soh_std"] * 100, np.abs(df_pred["error"]) * 100, alpha=0.3, s=4, color="#9C27B0")
    ax5.set_xlabel("Predicted Uncertainty (std, %)")
    ax5.set_ylabel("|Error| (%)")
    ax5.set_title("Uncertainty Calibration")

    # Plot 6: EOL prediction (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    if "eol_true" in df_pred.columns:
        eol_true = df_pred["eol_true"].values
        eol_pred = df_pred["eol_pred"].values
        # Separate by true class
        pos_mask = eol_true == 1
        neg_mask = eol_true == 0
        ax6.hist(eol_pred[pos_mask], bins=20, alpha=0.5, color="red", label="EOL reached (true)", density=True)
        ax6.hist(eol_pred[neg_mask], bins=20, alpha=0.5, color="blue", label="EOL not reached", density=True)
        ax6.axvline(0.5, color="green", linestyle="--", label="Decision threshold")
        ax6.set_xlabel("Predicted EOL Probability")
        ax6.set_ylabel("Density")
        ax6.set_title("EOL Probability Distribution")
        ax6.legend(fontsize=7)
    else:
        ax6.text(0.5, 0.5, "EOL predictions not available", ha="center", va="center")
        ax6.set_title("EOL Probability (Auxiliary Task)")

    plt.suptitle("CNN-Mamba-UQ Multi-Task — Evaluation Results (50-cycle horizon)", fontsize=13, fontweight="bold")
    out = PLOTS_DIR / "evaluation_results_mlt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out}")


def plot_training_curve():
    """Plot training history."""
    if not PLOT:
        return
    hist_path = RESULTS_DIR / "training_history_mlt.csv"
    if not hist_path.exists():
        return
    hist = pd.read_csv(hist_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist["epoch"], hist["train"], label="Train Loss")
    plt.plot(hist["epoch"], hist["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(hist["epoch"], hist["lr"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.yscale("log")
    plt.title("Learning Rate Schedule")
    plt.legend()
    
    plt.tight_layout()
    out = PLOTS_DIR / "training_curve_mlt.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot -> {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("MODEL EVALUATION STARTED (Multi-Task, 50-CYCLE HORIZON)")
    print("="*60)

    # Load model
    print("\nLoading CNN-Mamba-UQ checkpoint...")
    model = CNNMambaUQ()
    ckpt = MODEL_SAVE_DIR / "cnn_mamba_uq_mlt_best.pt"
    if not ckpt.exists():
        ckpt = Path(__file__).parent.parent / "results" / "checkpoints" / "cnn_mamba_uq_mlt_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    print(f"Loaded from {ckpt}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prediction horizon: {PREDICTION_HORIZON} cycles")

    # Load test dataset
    print("\nLoading test dataset...")
    test_path = RESULTS_DIR / "test_dataset_mlt.pt"
    if not test_path.exists():
        test_path = Path(__file__).parent.parent / "results" / "test_dataset_mlt.pt"
    test_ds = torch.load(test_path, weights_only=False)
    print(f"{len(test_ds):,} test sequences")

    # Evaluate
    print("\nRunning performance evaluation...")
    df_pred, metrics = evaluate_performance(model, test_ds, DEVICE)

    print("\nRunning deployment evaluation...")
    evaluate_deployment(model, test_ds)

    # Generate plots
    print("\nGenerating plots...")
    plot_training_curve()
    plot_results(df_pred, metrics)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nOutput files saved in: {RESULTS_DIR}")
    print(f"  - evaluation_report_mlt.csv (predictions with uncertainties)")
    print(f"  - metrics_summary_mlt.csv (performance metrics)")
    print(f"  - deployment_metrics_mlt.csv (latency, throughput, etc.)")
    print(f"  - plots/evaluation_results_mlt.png")
    print(f"  - plots/training_curve_mlt.png")


if __name__ == "__main__":
    main()