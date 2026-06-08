# model_evaluation/evaluate_mlt_fast.py
"""
FAST EVALUATION FOR MULTI-TASK CNN-MAMBA-UQ
============================================
Reduced MC samples for faster evaluation.
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training_mlt import (
    RESULTS_DIR, MODEL_SAVE_DIR,
    BATCH_SIZE, PREDICTION_HORIZON
)

from model_architecture.cnn_mamba_uq_mlt import CNNMambaUQ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# REDUCED for faster evaluation
MC_SAMPLES_FAST = 10  # Was 50, now 10
BATCH_SIZE = 128  # Increased from 64


def mape(y_true, y_pred):
    mask = y_true > 0.01
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_fast(model, test_ds, device, mc_samples=MC_SAMPLES_FAST):
    """Fast evaluation with fewer MC samples."""
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()

    all_future_pred = []
    all_future_true = []
    all_current_pred = []
    all_eol_pred = []

    print(f"Evaluating {len(test_ds)} sequences with {mc_samples} MC samples...")
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}")
        
        X_batch = batch[0].to(device)
        y_future = batch[1].numpy() if len(batch) > 1 else None
        
        # Single batch prediction with MC
        batch_preds = []
        with torch.no_grad():
            # Get deterministic outputs first
            outputs = model(X_batch)
            all_current_pred.extend(outputs["current_soh"].cpu().numpy().flatten())
            all_eol_pred.extend(outputs["eol_prob"].cpu().numpy().flatten())
            
            # MC sampling for future SOH
            for _ in range(mc_samples):
                pred = model.head_future(model.encode(X_batch))
                batch_preds.append(pred.cpu().numpy().flatten())
        
        # Aggregate MC predictions
        batch_preds = np.array(batch_preds)  # (mc_samples, batch_size)
        mean_pred = batch_preds.mean(axis=0)
        
        all_future_pred.extend(mean_pred)
        if y_future is not None:
            all_future_true.extend(y_future)

    # Convert to numpy
    future_pred = np.array(all_future_pred)
    future_true = np.array(all_future_true) if all_future_true else None
    current_pred = np.array(all_current_pred)
    eol_pred = np.array(all_eol_pred)

    # Calculate metrics
    metrics = {}
    if future_true is not None:
        metrics["R²"] = r2_score(future_true, future_pred)
        metrics["MAE (%)"] = mean_absolute_error(future_true, future_pred) * 100
        metrics["RMSE (%)"] = np.sqrt(mean_squared_error(future_true, future_pred)) * 100
        metrics["MAPE (%)"] = mape(future_true, future_pred)

    # Print results
    print(f"\n{'='*60}")
    print(f"FAST EVALUATION RESULTS (50-cycle horizon)")
    print(f"MC samples: {mc_samples}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<15} {v:>10.4f}")
    print(f"{'='*60}\n")

    return metrics


def main():
    print("\n" + "="*60)
    print("FAST EVALUATION (Multi-Task, 50-CYCLE HORIZON)")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model = CNNMambaUQ()
    ckpt = MODEL_SAVE_DIR / "cnn_mamba_uq_mlt_best.pt"
    if not ckpt.exists():
        ckpt = Path(__file__).parent.parent / "results_mlt" / "checkpoints" / "cnn_mamba_uq_mlt_best.pt"
    
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded from {ckpt}")
    print(f"Device: {DEVICE}")

    # Load test dataset
    print("\nLoading test dataset...")
    test_path = RESULTS_DIR / "test_dataset_mlt.pt"
    if not test_path.exists():
        test_path = Path(__file__).parent.parent / "results_mlt" / "test_dataset_mlt.pt"
    
    test_ds = torch.load(test_path, weights_only=False)
    print(f"{len(test_ds):,} test sequences")

    # Evaluate
    print("\nRunning fast evaluation...")
    metrics = evaluate_fast(model, test_ds, DEVICE)

    # Save results
    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df_metrics.to_csv(RESULTS_DIR / "metrics_summary_mlt_fast.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'metrics_summary_mlt_fast.csv'}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()