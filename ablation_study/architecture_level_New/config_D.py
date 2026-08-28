# evaluate_config_D_from_checkpoint.py
# Re-evaluate your saved model WITHOUT calibration to get Config D results

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader

# Import your model class and utilities
# (Copy these from your main script or import them)
from train_final_model import (
    BEM_SOH, SequenceDataset, FEAT_COLS, 
    load_soh_data, get_predictions, CFG, DEVICE
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the saved checkpoint
# ─────────────────────────────────────────────────────────────────────────────

checkpoint_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Get the configuration from the checkpoint
cfg = checkpoint["cfg"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Force calibration OFF for Config D evaluation
# ─────────────────────────────────────────────────────────────────────────────

cfg["calibrate"] = False  # ← THIS IS THE KEY CHANGE!

print("=" * 60)
print("  CONFIG D: Bidirectional + Evidential (NO CALIBRATION)")
print("  Re-evaluating saved model with calibrate=False")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load data
# ─────────────────────────────────────────────────────────────────────────────

soh_df, scaler = load_soh_data(cfg["soh_path"])

W = cfg["window_size"]
train_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "train")
val_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "val")
test_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "test")

val_loader = DataLoader(val_ds, batch_size=cfg["soh_batch"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load model
# ─────────────────────────────────────────────────────────────────────────────

model = BEM_SOH(cfg).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\nModel loaded successfully!")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Config: bidirectional={cfg['bidirectional']}, "
      f"evidential={cfg['evidential']}, calibrate={cfg['calibrate']}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate WITHOUT calibration (Config D)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_no_calibration(model, val_loader, test_loader, cfg):
    """Same as evaluate_soh but with calibration forced off."""
    
    # Get predictions
    y_val, mu_val, sigma_val, _, _ = get_predictions(model, val_loader, cfg)
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, test_loader, cfg)
    
    # Accuracy metrics
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100
    
    # Raw intervals (no calibration)
    z = 1.645  # nominal 90% Gaussian z
    y_lo_raw = y_pred - z * sigma_test
    y_hi_raw = y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)
    
    print(f"\n  -- ACCURACY --")
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  MAPE : {mape:.4f}%")
    print(f"  R2   : {r2:.5f}")
    
    print(f"\n  -- UNCALIBRATED intervals (nominal 90% Gaussian) --")
    print(f"  PICP : {picp_raw:.4f}  (target ~0.90)")
    print(f"  PINW : {pinw_raw:.4f}")
    
    if cfg["evidential"]:
        print(f"\n  -- Uncertainty decomposition --")
        print(f"  Mean aleatoric var  : {np.nanmean(aleatoric):.6f}")
        print(f"  Mean epistemic var  : {np.nanmean(epistemic):.6f}")
        ratio = np.nanmean(aleatoric) / (np.nanmean(epistemic) + 1e-8)
        print(f"  Aleatoric/Epistemic ratio: {ratio:.2f}")
    
    print(f"\n  -- MAE by SOH region --")
    for label, mask in [
        ("SOH < 0.90", y_true < 0.90),
        ("0.90-0.95", (y_true >= 0.90) & (y_true < 0.95)),
        ("SOH > 0.95", y_true >= 0.95)
    ]:
        if mask.sum() > 0:
            rm = mean_absolute_error(y_true[mask], y_pred[mask]) * 100
            print(f"  {label}: MAE = {rm:.4f}%  (n={mask.sum()})")
    
    results = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "picp_raw": picp_raw,
        "pinw_raw": pinw_raw,
        "mean_aleatoric": float(np.nanmean(aleatoric)),
        "mean_epistemic": float(np.nanmean(epistemic)),
    }
    
    return results

# Run evaluation
results_config_D = evaluate_no_calibration(model, val_loader, test_loader, cfg)

print("\n" + "=" * 60)
print("  CONFIG D RESULTS (Bidirectional + Evidential, NO CALIBRATION)")
print("=" * 60)
print(f"  MAE  : {results_config_D['mae']:.4f}%")
print(f"  RMSE : {results_config_D['rmse']:.4f}%")
print(f"  R2   : {results_config_D['r2']:.5f}")
print(f"  PICP : {results_config_D['picp_raw']:.4f}")
print(f"  PINW : {results_config_D['pinw_raw']:.4f}")
print("=" * 60)