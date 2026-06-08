# model_evaluation/evaluate_mlt_complete.py
"""
COMPLETE EVALUATION FOR MULTI-TASK CNN-MAMBA-UQ
================================================
Includes:
  - Future SOH metrics (MAE, RMSE, R², MAPE, MaxE)
  - Uncertainty metrics (PICP, PINAW)
  - Auxiliary task metrics (Current SOH, EOL probability)
  - Deployment metrics (latency, throughput, memory)
"""

import sys
import time
import pickle
import tracemalloc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training_mlt import (
    RESULTS_DIR, MODEL_SAVE_DIR, SCALER_PATH,
    FEATURE_COLS, TARGET_COL, SOH_EOL_THRESHOLD,
    BATCH_SIZE, MC_SAMPLES, PREDICTION_HORIZON
)

from model_architecture.cnn_mamba_uq_mlt import CNNMambaUQ

INFERENCE_DEVICE = "cpu"
N_INFERENCE_RUNS = 200

DEVICE = torch.device(INFERENCE_DEVICE)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Reduced MC samples for faster evaluation
EVAL_MC_SAMPLES = 20

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
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def picp(y_true: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (target 95% for 95% CI)."""
    covered = ((y_true >= ci_low) & (y_true <= ci_high)).mean()
    return float(covered) * 100


def pinaw(ci_low: np.ndarray, ci_high: np.ndarray, y_range: float) -> float:
    """Prediction Interval Normalised Average Width (lower = sharper)."""
    return float(np.mean(ci_high - ci_low) / y_range)


# ============================================================================
# MC PREDICTION FUNCTION
# ============================================================================

def mc_predict_future(model, x, mc_samples=EVAL_MC_SAMPLES):
    """
    Monte Carlo prediction for future SOH uncertainty.
    """
    model.train()  # Enable dropout
    all_preds = []
    
    with torch.no_grad():
        for _ in range(mc_samples):
            pred = model.head_future(model.encode(x))
            all_preds.append(pred.cpu().numpy())
    
    model.eval()
    
    all_preds = np.concatenate(all_preds, axis=1)  # (B, mc_samples)
    mean = all_preds.mean(axis=1)
    std = all_preds.std(axis=1)
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std
    
    return mean, std, ci_low, ci_high


# ============================================================================
# PERFORMANCE EVALUATION (TIER 1)
# ============================================================================

def evaluate_performance(model, test_ds, device):
    """Complete performance evaluation with all metrics."""
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()

    all_future_pred = []
    all_future_true = []
    all_future_std = []
    all_ci_low = []
    all_ci_high = []
    all_current_pred = []
    all_current_true = []
    all_eol_pred = []
    all_eol_true = []

    print(f"Evaluating {len(test_ds)} sequences with {EVAL_MC_SAMPLES} MC samples...")
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}")
        
        X_batch = batch[0].to(device)
        y_future = batch[1].numpy()
        y_current = batch[2].numpy() if len(batch) > 2 else None
        y_eol = batch[3].numpy() if len(batch) > 3 else None

        # MC predictions for future SOH
        mean, std, ci_low, ci_high = mc_predict_future(model, X_batch)
        all_future_pred.extend(mean)
        all_future_std.extend(std)
        all_ci_low.extend(ci_low)
        all_ci_high.extend(ci_high)
        all_future_true.extend(y_future)

        # Deterministic predictions for auxiliary tasks
        with torch.no_grad():
            outputs = model(X_batch)
            all_current_pred.extend(outputs["current_soh"].cpu().numpy().flatten())
            all_eol_pred.extend(outputs["eol_prob"].cpu().numpy().flatten())

        if y_current is not None:
            all_current_true.extend(y_current)
        if y_eol is not None:
            all_eol_true.extend(y_eol)

    # Convert to numpy arrays
    future_true = np.array(all_future_true)
    future_pred = np.array(all_future_pred)
    future_std = np.array(all_future_std)
    ci_low = np.array(all_ci_low)
    ci_high = np.array(all_ci_high)
    
    current_true = np.array(all_current_true) if all_current_true else None
    current_pred = np.array(all_current_pred) if all_current_pred else None
    eol_true = np.array(all_eol_true) if all_eol_true else None
    eol_pred = np.array(all_eol_pred) if all_eol_pred else None

    # Future SOH metrics
    y_range = float(future_true.max() - future_true.min())
    
    metrics = {
        # Accuracy metrics
        "Future SOH - R²": r2_score(future_true, future_pred),
        "Future SOH - MAE (%)": mean_absolute_error(future_true, future_pred) * 100,
        "Future SOH - RMSE (%)": np.sqrt(mean_squared_error(future_true, future_pred)) * 100,
        "Future SOH - MAPE (%)": mape(future_true, future_pred),
        "Future SOH - MaxE (%)": max_error(future_true, future_pred) * 100,
        # Uncertainty metrics
        "Future SOH - PICP (%)": picp(future_true, ci_low, ci_high),
        "Future SOH - PINAW": pinaw(ci_low, ci_high, y_range),
    }

    # Current SOH metrics (auxiliary task)
    if current_true is not None and len(current_true) > 0:
        metrics["Current SOH - R²"] = r2_score(current_true, current_pred)
        metrics["Current SOH - MAE (%)"] = mean_absolute_error(current_true, current_pred) * 100
        metrics["Current SOH - RMSE (%)"] = np.sqrt(mean_squared_error(current_true, current_pred)) * 100

    # EOL prediction metrics (auxiliary task)
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
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(eol_true, eol_pred_binary).ravel()
        metrics["EOL - Specificity (%)"] = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0

    # Save detailed predictions
    df_pred = pd.DataFrame({
        "future_soh_true": future_true,
        "future_soh_pred": future_pred,
        "future_soh_std": future_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_error": future_true - future_pred,
        "abs_error_percent": np.abs(future_true - future_pred) * 100,
    })
    
    if current_true is not None:
        df_pred["current_soh_true"] = current_true
        df_pred["current_soh_pred"] = current_pred
        df_pred["current_error"] = current_true - current_pred
        
    if eol_true is not None:
        df_pred["eol_true"] = eol_true
        df_pred["eol_pred_prob"] = eol_pred
        df_pred["eol_pred_binary"] = (eol_pred > 0.5).astype(int)

    df_pred.to_csv(RESULTS_DIR / "evaluation_report_mlt_complete.csv", index=False)

    # Print metrics
    print(f"\n{'='*70}")
    print(f"  COMPLETE EVALUATION RESULTS (Multi-Task, {PREDICTION_HORIZON}-cycle horizon)")
    print(f"{'='*70}")
    
    print(f"\n FUTURE SOH PREDICTION (Primary Task):")
    print(f"  {'─'*50}")
    for k in ["Future SOH - R²", "Future SOH - MAE (%)", "Future SOH - RMSE (%)", 
              "Future SOH - MAPE (%)", "Future SOH - MaxE (%)"]:
        if k in metrics:
            print(f"  {k:<25} {metrics[k]:>12.4f}")
    
    print(f"\n UNCERTAINTY QUANTIFICATION:")
    print(f"  {'─'*50}")
    for k in ["Future SOH - PICP (%)", "Future SOH - PINAW"]:
        if k in metrics:
            print(f"  {k:<25} {metrics[k]:>12.4f}")
    
    if "Current SOH - R²" in metrics:
        print(f"\n CURRENT SOH ESTIMATION (Auxiliary Task):")
        print(f"  {'─'*50}")
        for k in ["Current SOH - R²", "Current SOH - MAE (%)", "Current SOH - RMSE (%)"]:
            if k in metrics:
                print(f"  {k:<25} {metrics[k]:>12.4f}")
    
    if "EOL - Accuracy (%)" in metrics:
        print(f"\n EOL PREDICTION (Auxiliary Task):")
        print(f"  {'─'*50}")
        for k in ["EOL - Accuracy (%)", "EOL - Precision (%)", "EOL - Recall (%)",
                  "EOL - F1 Score", "EOL - AUC-ROC", "EOL - Specificity (%)"]:
            if k in metrics:
                print(f"  {k:<25} {metrics[k]:>12.4f}")
    
    print(f"\n{'='*70}\n")

    # Save metrics
    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df_metrics.to_csv(RESULTS_DIR / "metrics_summary_mlt_complete.csv", index=False)

    return df_pred, metrics


# ============================================================================
# DEPLOYMENT EVALUATION (TIER 2)
# ============================================================================

def evaluate_deployment(model, test_ds):
    """Measure deployment-relevant metrics on CPU."""
    model_cpu = model.to(torch.device("cpu"))
    model_cpu.eval()

    # Single sample for latency test
    X_single = test_ds[0][0].unsqueeze(0).cpu()

    print(f"\nRunning deployment evaluation with {N_INFERENCE_RUNS} runs...")

    # Inference latency
    latencies = []
    for i in range(N_INFERENCE_RUNS):
        if i % 50 == 0:
            print(f"  Latency run {i}/{N_INFERENCE_RUNS}")
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(X_single)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_mean = float(np.mean(latencies[10:]))
    lat_std = float(np.std(latencies[10:]))
    lat_p95 = float(np.percentile(latencies[10:], 95))
    lat_p99 = float(np.percentile(latencies[10:], 99))

    # Throughput (batch processing)
    print("  Measuring throughput...")
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].cpu()
            _ = model_cpu(X_batch)
    t_total = time.perf_counter() - t_start
    throughput = len(test_ds) / t_total

    # Model size
    n_params = sum(p.numel() for p in model_cpu.parameters())
    ckpt_path = MODEL_SAVE_DIR / "cnn_mamba_uq_mlt_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(__file__).parent.parent / "results_mlt" / "checkpoints" / "cnn_mamba_uq_mlt_best.pt"
    disk_mb = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0.0

    # Peak memory
    tracemalloc.start()
    with torch.no_grad():
        _ = model_cpu(X_single)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_kb = peak_mem / 1024
    peak_mem_mb = peak_mem_kb / 1024

    deployment_metrics = {
        "Parameters": n_params,
        "Disk Size (MB)": disk_mb,
        "Peak RAM (KB)": peak_mem_kb,
        "Peak RAM (MB)": peak_mem_mb,
        "Latency Mean (ms)": lat_mean,
        "Latency Std (ms)": lat_std,
        "Latency P95 (ms)": lat_p95,
        "Latency P99 (ms)": lat_p99,
        "Throughput (samples/sec)": throughput,
    }

    print(f"\n{'='*70}")
    print(f"  DEPLOYMENT METRICS (CPU - Edge BMS Simulation)")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Value':>15}")
    print(f"  {'─'*42}")
    for k, v in deployment_metrics.items():
        if "Latency" in k:
            print(f"  {k:<25} {v:>15.3f}")
        elif "Parameters" in k:
            print(f"  {k:<25} {v:>15,}")
        else:
            print(f"  {k:<25} {v:>15.2f}")
    
    real_time_ok = "YES" if lat_p95 < 100 else "NO"
    print(f"  {'BMS Real-Time OK (p95<100ms)':<25} {real_time_ok:>15}")
    print(f"{'='*70}\n")

    df_dep = pd.DataFrame(list(deployment_metrics.items()), columns=["Metric", "Value"])
    df_dep.to_csv(RESULTS_DIR / "deployment_metrics_mlt_complete.csv", index=False)
    
    return deployment_metrics


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(df_pred, metrics):
    """Generate comprehensive evaluation plots."""
    if not PLOT:
        return

    print("\nGenerating plots...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

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
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_pred["abs_error_percent"], bins=50, color="#4CAF50", edgecolor="white", linewidth=0.3)
    ax2.axvline(df_pred["abs_error_percent"].mean(), color="red", lw=2, linestyle="--", 
                label=f'Mean Error: {df_pred["abs_error_percent"].mean():.2f}%')
    ax2.set_xlabel("Absolute Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Error Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Current SOH (auxiliary)
    ax3 = fig.add_subplot(gs[0, 2])
    if "current_soh_true" in df_pred.columns:
        ax3.scatter(df_pred["current_soh_true"], df_pred["current_soh_pred"], alpha=0.3, s=4, color="#FF9800")
        ax3.plot([0, 1], [0, 1], "r--", lw=1.5)
        ax3.set_xlabel("True Current SOH")
        ax3.set_ylabel("Predicted Current SOH")
        r2_current = metrics.get("Current SOH - R²", 0)
        ax3.set_title(f"Current SOH (Auxiliary Task)\nR² = {r2_current:.4f}")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Current SOH not available", ha="center", va="center")
        ax3.set_title("Current SOH (Auxiliary Task)")

    # Plot 4: Prediction intervals (uncertainty)
    ax4 = fig.add_subplot(gs[1, 0])
    sample = df_pred.sample(min(500, len(df_pred)), random_state=42).sort_values("future_soh_true")
    ax4.fill_between(range(len(sample)), sample["ci_low"], sample["ci_high"], 
                     alpha=0.3, color="#FF9800", label="95% CI")
    ax4.scatter(range(len(sample)), sample["future_soh_true"], s=4, color="black", 
                zorder=3, label="True SOH")
    ax4.scatter(range(len(sample)), sample["future_soh_pred"], s=4, color="#E91E63", 
                zorder=3, label="Predicted SOH", alpha=0.7)
    ax4.set_xlabel("Sample (sorted by true SOH)")
    ax4.set_ylabel("SOH")
    picp_val = metrics.get("Future SOH - PICP (%)", 0)
    ax4.set_title(f"Prediction Intervals (500 samples)\nPICP = {picp_val:.1f}%")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Uncertainty vs Error calibration
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df_pred["future_soh_std"] * 100, df_pred["abs_error_percent"], 
                alpha=0.3, s=4, color="#9C27B0")
    ax5.set_xlabel("Predicted Uncertainty (std, %)")
    ax5.set_ylabel("Actual |Error| (%)")
    ax5.set_title("Uncertainty Calibration")
    ax5.grid(True, alpha=0.3)
    
    # Add ideal calibration line
    max_val = max(df_pred["future_soh_std"].max() * 100, df_pred["abs_error_percent"].max())
    ax5.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Ideal calibration")
    ax5.legend(fontsize=7)

    # Plot 6: EOL prediction ROC (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    if "eol_true" in df_pred.columns:
        eol_true = df_pred["eol_true"].values
        eol_pred = df_pred["eol_pred_prob"].values
        
        # Histogram of predictions by class
        pos_mask = eol_true == 1
        neg_mask = eol_true == 0
        ax6.hist(eol_pred[pos_mask], bins=20, alpha=0.5, color="red", 
                 label="EOL reached (true)", density=True)
        ax6.hist(eol_pred[neg_mask], bins=20, alpha=0.5, color="blue", 
                 label="EOL not reached", density=True)
        ax6.axvline(0.5, color="green", linestyle="--", label="Decision threshold")
        ax6.set_xlabel("Predicted EOL Probability")
        ax6.set_ylabel("Density")
        auc_val = metrics.get("EOL - AUC-ROC", 0)
        ax6.set_title(f"EOL Probability Distribution\nAUC-ROC = {auc_val:.4f}")
        ax6.legend(fontsize=7)
    else:
        ax6.text(0.5, 0.5, "EOL predictions not available", ha="center", va="center")
        ax6.set_title("EOL Probability (Auxiliary Task)")

    # Plot 7: Error vs Cycle Norm (degradation stage)
    ax7 = fig.add_subplot(gs[2, 0])
    # Need cycle_norm from test data - this requires loading original dataset
    ax7.text(0.5, 0.5, "Requires cycle_norm from test data", ha="center", va="center")
    ax7.set_title("Error vs Degradation Stage")
    ax7.grid(True, alpha=0.3)

    # Plot 8: Error vs Temperature
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.text(0.5, 0.5, "Requires temperature from test data", ha="center", va="center")
    ax8.set_title("Error vs Temperature")
    ax8.grid(True, alpha=0.3)

    # Plot 9: Error vs Resistance
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.text(0.5, 0.5, "Requires resistance from test data", ha="center", va="center")
    ax9.set_title("Error vs Internal Resistance")
    ax9.grid(True, alpha=0.3)

    plt.suptitle("CNN-Mamba-UQ Multi-Task — Complete Evaluation Results (50-cycle horizon)", 
                 fontsize=14, fontweight="bold")
    out = PLOTS_DIR / "evaluation_results_mlt_complete.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {out}")


def plot_training_curve():
    """Plot training history."""
    if not PLOT:
        return
    hist_path = RESULTS_DIR / "training_history_mlt.csv"
    if not hist_path.exists():
        hist_path = Path(__file__).parent.parent / "results_mlt" / "training_history_mlt.csv"
    if not hist_path.exists():
        return
    
    hist = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(hist["epoch"], hist["train"], label="Train Loss", linewidth=2)
    axes[0].plot(hist["epoch"], hist["val"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Huber)")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(hist["epoch"], hist["lr"], label="Learning Rate", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_yscale("log")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = PLOTS_DIR / "training_curve_mlt_complete.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved plot -> {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  COMPLETE EVALUATION - Multi-Task CNN-Mamba-UQ")
    print(f"  {PREDICTION_HORIZON}-Cycle Horizon Prediction")
    print("="*70)

    # Load model
    print("\n Loading model...")
    model = CNNMambaUQ()
    ckpt = MODEL_SAVE_DIR / "cnn_mamba_uq_mlt_best.pt"
    if not ckpt.exists():
        ckpt = Path(__file__).parent.parent / "results_mlt" / "checkpoints" / "cnn_mamba_uq_mlt_best.pt"
    
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded from {ckpt}")
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test dataset
    print("\n Loading test dataset...")
    test_path = RESULTS_DIR / "test_dataset_mlt.pt"
    if not test_path.exists():
        test_path = Path(__file__).parent.parent / "results_mlt" / "test_dataset_mlt.pt"
    
    test_ds = torch.load(test_path, weights_only=False)
    print(f"  {len(test_ds):,} test sequences")

    # Evaluate performance
    print("\n Running performance evaluation...")
    df_pred, metrics = evaluate_performance(model, test_ds, DEVICE)

    # Evaluate deployment
    print("\n Running deployment evaluation...")
    evaluate_deployment(model, test_ds)

    # Generate plots
    print("\n Generating plots...")
    plot_training_curve()
    plot_results(df_pred, metrics)

    # Final summary
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE ")
    print("="*70)
    print(f"\n Output files saved in: {RESULTS_DIR}")
    print(f"  - evaluation_report_mlt_complete.csv")
    print(f"  - metrics_summary_mlt_complete.csv")
    print(f"  - deployment_metrics_mlt_complete.csv")
    print(f"  - plots/evaluation_results_mlt_complete.png")
    print(f"  - plots/training_curve_mlt_complete.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()