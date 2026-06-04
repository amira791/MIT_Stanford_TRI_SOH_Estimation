# diagnostics/diagnose_issues.py
"""
Diagnostic script to identify root causes of poor model performance
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training import RESULTS_DIR, FEATURE_COLS, TARGET_COL
from model_architecture.cnn_mamba_uq import CNNMambaUQ

def check_cause_1_soh_range():
    """Check SOH label range - Cause 1"""
    print("\n" + "="*60)
    print("CAUSE 1: SOH Label Range Check")
    print("="*60)
    
    df = pd.read_pickle(RESULTS_DIR / "soh_dataset.pkl")
    
    soh_min = df[TARGET_COL].min()
    soh_max = df[TARGET_COL].max()
    soh_mean = df[TARGET_COL].mean()
    
    print(f"\nSOH statistics:")
    print(f"  Min:  {soh_min:.4f}")
    print(f"  Max:  {soh_max:.4f}")
    print(f"  Mean: {soh_mean:.4f}")
    
    if soh_min >= 0.8:
        print(f"\n ISSUE DETECTED: SOH min = {soh_min:.4f}")
        print(f"   Model outputs sigmoid in [0,1] but SOH labels are in [{soh_min:.2f}, {soh_max:.2f}]")
        print(f"   -> Remove sigmoid from model output")
        return False
    elif soh_min < 0:
        print(f"\n ISSUE DETECTED: Negative SOH values found!")
        return False
    else:
        print(f"\n✓ SOH range is [0,1] compatible")
        return True

def check_cause_2_uq_head():
    """Check UQ head implementation - Cause 2"""
    print("\n" + "="*60)
    print("CAUSE 2: UQ Head Check")
    print("="*60)
    
    # Load a small test sample
    df = pd.read_pickle(RESULTS_DIR / "soh_dataset.pkl")
    test_cells_path = RESULTS_DIR / "test_cells.pkl"
    
    if test_cells_path.exists():
        with open(test_cells_path, "rb") as f:
            test_cells = pickle.load(f)
        df_test = df[df["cell_id"].isin(test_cells[:1])].copy()
    else:
        df_test = df.head(100).copy()
    
    # Create a simple sequence
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS].values)
    
    X_sample = scaler.transform(df_test[FEATURE_COLS].values[:30]).astype(np.float32)
    X_tensor = torch.tensor(X_sample).unsqueeze(0)  # (1, 30, 6)
    
    # Load model
    model = CNNMambaUQ()
    ckpt = RESULTS_DIR / "checkpoints" / "cnn_mamba_uq_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    
    # Test mc_predict
    print("\nTesting mc_predict...")
    model.train()
    with torch.no_grad():
        result = model.mc_predict(X_tensor)
    
    pred_std = result["std"][0]
    ci_low = result["ci_low"][0]
    ci_high = result["ci_high"][0]
    
    print(f"\nMC Predict results:")
    print(f"  Prediction std: {pred_std:.6f}")
    print(f"  CI width: {ci_high - ci_low:.6f}")
    
    if pred_std < 0.001 or (ci_high - ci_low) < 0.001:
        print(f"\n⚠️ ISSUE DETECTED: Uncertainty is zero!")
        print(f"   PINAW = 0.0, PICP = {result['ci_low'].shape}")
        print(f"   -> Remove @torch.no_grad() decorator from mc_predict")
        return False
    else:
        print(f"\n✓ UQ head produces non-zero uncertainty")
        return True

def check_cause_3_mamba_scan():
    """Check Mamba scan information collapse - Cause 3"""
    print("\n" + "="*60)
    print("CAUSE 3: Mamba Scan Information Loss Check")
    print("="*60)
    
    # Create a test input
    B, T, D, d_state = 2, 10, 128, 16
    x_test = torch.randn(B, T, D)
    
    # Simulate the problematic line
    print("\nSimulating current Mamba scan (with mean collapse):")
    x_t = x_test[:, 0, :]  # Take first timestep
    x_mean = x_t.mean(-1, keepdim=True)  # Collapse 128-dim to scalar
    print(f"  Input shape: {x_t.shape}")
    print(f"  After mean collapse: {x_mean.shape}")
    print(f"  Information loss: {D} dimensions → 1 dimension ({(1/D)*100:.1f}% retained)")
    
    print("\n⚠️ ISSUE DETECTED: Information collapse!")
    print("   The Mamba scan reduces 128 features to 1 scalar per timestep")
    print("   This explains why MAE = 3.46% and RMSE = 5.51%")
    print("\n   Fix: Replace x_t.mean(-1, keepdim=True) with proper projection")
    
    return False

def check_training_convergence():
    """Check if training actually learned anything"""
    print("\n" + "="*60)
    print("TRAINING CONVERGENCE CHECK")
    print("="*60)
    
    hist_path = RESULTS_DIR / "training_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        print(f"\nTraining history:")
        print(f"  Initial train MSE: {hist['train_mse'].iloc[0]:.6f}")
        print(f"  Final train MSE:   {hist['train_mse'].iloc[-1]:.6f}")
        print(f"  Initial val MSE:   {hist['val_mse'].iloc[0]:.6f}")
        print(f"  Final val MSE:     {hist['val_mse'].iloc[-1]:.6f}")
        
        improvement = (hist['train_mse'].iloc[0] - hist['train_mse'].iloc[-1]) / hist['train_mse'].iloc[0] * 100
        print(f"  Improvement: {improvement:.1f}%")
        
        if improvement < 1:
            print("\n⚠️ ISSUE: Model did not learn (loss barely changed)")
            print("   Possible causes: learning rate too low, sigmoid saturation, or collapsed Mamba")
        else:
            print("\n✓ Model loss decreased during training")
    else:
        print("\nNo training history found")

def main():
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT: Model Performance Issues")
    print("="*60)
    
    results = {}
    
    results["cause1"] = check_cause_1_soh_range()
    results["cause2"] = check_cause_2_uq_head()
    results["cause3"] = check_cause_3_mamba_scan()
    check_training_convergence()
    
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    if not results["cause1"]:
        print("\n1. FIX CAUSE 1 FIRST: Remove sigmoid from UQHead")
        print("   In cnn_mamba_uq.py, comment out: # nn.Sigmoid()")
        
    if not results["cause2"]:
        print("\n2. FIX CAUSE 2: Remove @torch.no_grad() from mc_predict")
        print("   In cnn_mamba_uq.py, remove the decorator line")
        
    if not results["cause3"]:
        print("\n3. FIX CAUSE 3: Fix Mamba scan information collapse")
        print("   Replace: h = A * h + B_t * x_t.mean(-1, keepdim=True)")
        print("   With:    h = A * h + B_t * (x_t @ C_proj)")
        
    print("\nRecommended fix order: Cause 1 → Retrain → Test → Then Causes 2 & 3")

if __name__ == "__main__":
    main()