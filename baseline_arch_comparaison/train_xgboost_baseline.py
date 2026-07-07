"""
XGBoost Baseline for Battery SOH Estimation
--------------------------------------------
- Uses flattened sequence features (50 cycles × 10 features = 500 inputs)
- Same train/val/test split as CNN-Mamba-UQ
- No data leakage (split by physical cell)
- Hyperparameters tuned for regression
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"
SAVE_DIR = Path(__file__).parent
SAVE_DIR.mkdir(exist_ok=True)

# Feature columns (same as CNN-Mamba-UQ)
FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def add_relative_features(df):
    """Add per-cell relative features (same as CNN-Mamba-UQ)"""
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

def load_and_preprocess():
    """Load data, add features, normalize, and prepare sequences"""
    
    print("Loading SOH data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")
    
    # Add relative features
    df = add_relative_features(df)
    print(f"  Features added: {df.shape}")
    
    # Feature columns to use
    feature_cols = FEAT_COLS
    target_col = "soh"
    
    # Create sliding window sequences (50 cycles)
    window_size = 50
    stride = 2
    
    X_list, y_list, split_list, cell_list = [], [], [], []
    
    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
        X = cell_df[feature_cols].values.astype(np.float32)
        y = cell_df[target_col].values.astype(np.float32)
        split = cell_df["split"].values
        cell = cell_df["barcode"].values[0]
        
        # Slide window over each cell
        for end in range(window_size, len(X) + 1, stride):
            start = end - window_size
            x_flat = X[start:end].flatten()  # Flatten to vector
            y_last = y[end - 1]
            X_list.append(x_flat)
            y_list.append(y_last)
            split_list.append(split[end - 1])
            cell_list.append(cell)
    
    X = np.array(X_list)
    y = np.array(y_list)
    splits = np.array(split_list)
    cells = np.array(cell_list)
    
    print(f"  Sequences created: {X.shape}")
    
    # Split by predefined split
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # Normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, cells

# ============================================================
# TRAINING
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model with hyperparameter tuning"""
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST BASELINE")
    print("="*60)
    
    # XGBoost parameters (tuned for regression)
    params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    print("\nHyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create DMatrix for early stopping
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train with early stopping
    print("\nTraining...")
    evals = [(dtrain, "train"), (dval, "eval")]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose_eval=50,
    )
    
    return model, params

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X_test, y_test, label="Test"):
    """Evaluate model performance"""
    
    y_pred = model.predict(xgb.DMatrix(X_test))
    
    mae = mean_absolute_error(y_test, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  ── {label} Results ──────────────────────────────")
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.5f}")
    
    return {
        "mae_pct": mae,
        "rmse_pct": rmse,
        "r2": r2,
        "y_true": y_test,
        "y_pred": y_pred,
    }

def plot_results(y_true, y_pred, save_path):
    """Create prediction vs true scatter plot"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # Calculate metrics inside the function
    mae = mean_absolute_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=5)
    plt.plot([0.7, 1.0], [0.7, 1.0], "r--", label="Perfect Prediction")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"XGBoost: Predictions vs True SOH\nMAE: {mae:.4f}%, R²: {r2:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved: {save_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("XGBOOST BASELINE FOR BATTERY SOH ESTIMATION")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, cells = load_and_preprocess()
    
    # Train model
    print("\n[2] Training XGBoost...")
    model, params = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Save model
    print("\n[3] Saving model...")
    joblib.dump(model, SAVE_DIR / "xgboost_model.pkl")
    joblib.dump(scaler, SAVE_DIR / "scaler.pkl")
    with open(SAVE_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Model saved to: {SAVE_DIR / 'xgboost_model.pkl'}")
    
    # Evaluate
    print("\n[4] Evaluating...")
    train_results = evaluate(model, X_train, y_train, "Train")
    val_results = evaluate(model, X_val, y_val, "Validation")
    test_results = evaluate(model, X_test, y_test, "Test")
    
    # Combine results
    results = {
        "train": train_results,
        "val": val_results,
        "test": test_results,
        "params": params,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    
    # Save results
    results_to_save = {
        "train_mae_pct": train_results["mae_pct"],
        "train_rmse_pct": train_results["rmse_pct"],
        "train_r2": train_results["r2"],
        "val_mae_pct": val_results["mae_pct"],
        "val_rmse_pct": val_results["rmse_pct"],
        "val_r2": val_results["r2"],
        "test_mae_pct": test_results["mae_pct"],
        "test_rmse_pct": test_results["rmse_pct"],
        "test_r2": test_results["r2"],
        "params": params,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    
    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"  Results saved to: {SAVE_DIR / 'results.json'}")
    
    # Plot
    print("\n[5] Creating plots...")
    plot_results(test_results["y_true"], test_results["y_pred"], SAVE_DIR / "predictions.png")
    
    # Summary
    print("\n" + "="*60)
    print("XGBOOST RESULTS SUMMARY")
    print("="*60)
    print(f"  Train MAE: {train_results['mae_pct']:.4f}%")
    print(f"  Val MAE:   {val_results['mae_pct']:.4f}%")
    print(f"  Test MAE:  {test_results['mae_pct']:.4f}%")
    print(f"  Test R²:   {test_results['r2']:.5f}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()