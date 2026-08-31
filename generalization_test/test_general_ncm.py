# zero_shot_evaluation.py
# Zero-shot evaluation of BEM-SOH on SNL NCM and NCA datasets

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Imports from your main script
# ─────────────────────────────────────────────────────────────────────────────

from train_final_model import (
    BEM_SOH, SequenceDataset, get_predictions, FEAT_COLS, CFG, DEVICE
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt"
NCM_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\ncm_processed.csv"
NCA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\nca_processed.csv"
OUT_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load trained model
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  ZERO-SHOT GENERALIZATION ON SNL DATASETS")
print("=" * 60)

print("\nLoading trained BEM-SOH model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

cfg = checkpoint["cfg"]
model = BEM_SOH(cfg).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"  Model trained on: MIT-Stanford (LFP)")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Get MIT scaler
scaler_mit = StandardScaler()
scaler_mit.mean_ = np.array(checkpoint["scaler_mean"])
scaler_mit.scale_ = np.array(checkpoint["scaler_std"])

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper function for zero-shot evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_zero_shot(df_path, chemistry_name):
    """Evaluate BEM-SOH zero-shot on a processed dataset."""
    
    print("\n" + "=" * 60)
    print(f"  Evaluating on SNL-{chemistry_name}")
    print("=" * 60)
    
    # Load processed data
    df = pd.read_csv(df_path)
    print(f"  Loaded {len(df):,} rows, {df['cell_id'].nunique()} cells")
    
    # ─── Get features that exist ───
    # The NCM/NCA datasets have 8 features (no DCIR, no temperature)
    # We need to map them to match the model's expected features
    
    # Available features in NCM/NCA processed data
    available_feats = [
        'charge_capacity', 'charge_energy',
        'cap_rel', 'energy_rel', 'cycle_pos', 'voltage_range'
    ]
    
    # Our model expects 10 features. We need to create dummy/missing ones.
    # For zero-shot evaluation with missing features, we have two options:
    # Option A: Train a minimal-feature version (recommended for paper)
    # Option B: Use existing features and pad missing ones with zeros
    
    # For now, let's use Option B (quick test):
    # We'll create the full 10-feature set with placeholders for missing features
    
    required_feats = FEAT_COLS
    missing_feats = [f for f in required_feats if f not in df.columns]
    print(f"  Missing features: {missing_feats}")
    
    # Create dummy features (0.0) for missing ones
    for feat in missing_feats:
        df[feat] = 0.0
    
    # Ensure coulombic_efficiency features exist (they should)
    if 'coulombic_efficiency_lagged_1' not in df.columns:
        df['coulombic_efficiency_lagged_1'] = df['charge_capacity'] / (df['discharge_capacity'] + 1e-9)
    if 'coulombic_efficiency_lagged_2' not in df.columns:
        df['coulombic_efficiency_lagged_2'] = df['coulombic_efficiency_lagged_1']
    
    # ─── Normalize using MIT scaler ───
    # Only normalize features that exist in MIT training
    norm_feats = [f for f in FEAT_COLS if f in df.columns]
    df[norm_feats] = scaler_mit.transform(df[norm_feats].values)
    
    # ─── Create dataloader ───
    W = cfg["window_size"]
    
    # Custom dataset for generalization (no split)
    class GenDataset:
        def __init__(self, df, window_size, stride=1):
            self.samples = []
            self.cell_ids = []
            
            for cid, cell_df in df.groupby("cell_id"):
                cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
                X = cell_df[FEAT_COLS].values.astype(np.float32)
                y = cell_df["soh"].values.astype(np.float32)
                
                for end in range(window_size, len(X) + 1, stride):
                    start = end - window_size
                    self.samples.append((X[start:end], y[end - 1]))
                    self.cell_ids.append(cid)
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            x, y = self.samples[idx]
            return torch.tensor(x), torch.tensor(y)
    
    dataset = GenDataset(df, W, cfg["soh_stride"])
    loader = DataLoader(dataset, batch_size=cfg["soh_batch"], shuffle=False)
    
    print(f"  Windows: {len(dataset)}")
    
    # ─── Evaluate ───
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, loader, cfg)
    
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    
    z = 1.645
    y_lo = y_pred - z * sigma_test
    y_hi = y_pred + z * sigma_test
    picp = np.mean((y_true >= y_lo) & (y_true <= y_hi))
    pinw = np.mean(y_hi - y_lo) / (y_true.max() - y_true.min() + 1e-8)
    
    print(f"\n  -- Results --")
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.4f}")
    print(f"  PICP : {picp:.4f}  (target ~0.90)")
    print(f"  PINW : {pinw:.4f}")
    
    if cfg["evidential"]:
        print(f"  Aleatoric : {np.nanmean(aleatoric):.6f}")
        print(f"  Epistemic : {np.nanmean(epistemic):.6f}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "picp": picp,
        "pinw": pinw,
        "mean_aleatoric": float(np.nanmean(aleatoric)) if cfg["evidential"] else None,
        "mean_epistemic": float(np.nanmean(epistemic)) if cfg["evidential"] else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run evaluation on both datasets
# ─────────────────────────────────────────────────────────────────────────────

results = {}

if os.path.exists(NCM_PATH):
    results["NCM"] = evaluate_zero_shot(NCM_PATH, "NCM")

if os.path.exists(NCA_PATH):
    results["NCA"] = evaluate_zero_shot(NCA_PATH, "NCA")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  GENERALIZATION SUMMARY")
print("=" * 60)

print(f"\n  {'Dataset':<15} {'MAE (%)':<12} {'R²':<10} {'PICP':<10} {'PINW':<10}")
print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'MIT (LFP)':<15} {0.116:<12.4f} {0.997:<10.4f} {0.943:<10.4f} {0.023:<10.4f}")

for name, res in results.items():
    print(f"  {name:<15} {res['mae']:<12.4f} {res['r2']:<10.4f} {res['picp']:<10.4f} {res['pinw']:<10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save results
# ─────────────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUT_DIR, "zero_shot_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved -> {out_path}")
print("=" * 60)