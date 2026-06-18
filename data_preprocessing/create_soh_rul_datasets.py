# create_separated_datasets.py (UPDATED - RUL LABELED ONLY)

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
FINAL_DATASET_DIR = Path(__file__).parent / "final_dataset"
SOH_DIR = FINAL_DATASET_DIR / "soh"
RUL_DIR = FINAL_DATASET_DIR / "rul"

# Create directories
SOH_DIR.mkdir(exist_ok=True)
RUL_DIR.mkdir(exist_ok=True)

# Feature columns (7 features)
FEATURES = [
    "cycle_index",
    "dc_internal_resistance",
    "temperature_avg",
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2"
]

def load_full_dataset():
    """Load the combined dataset"""
    df = pd.read_pickle(FINAL_DATASET_DIR / "final_dataset.pkl")
    print(f"Loaded {len(df)} rows, {df['barcode'].nunique()} cells")
    return df


# ============================================
# SOH DATASET (Supervised - All cells)
# ============================================

def create_soh_dataset(df):
    """
    Create SOH dataset using ALL cells (134 cells, all cycles)
    Every row has a valid SOH label
    """
    print("\n" + "=" * 60)
    print("CREATING SOH DATASET (SUPERVISED)")
    print("=" * 60)
    
    # Use all data
    soh_df = df.copy()
    
    # Select columns: metadata + features + SOH target
    soh_df = soh_df[["cell_id", "barcode", "channel", "split"] + FEATURES + ["soh"]]
    
    # Remove any rows with NaN
    soh_df = soh_df.dropna()
    
    # Split by predefined split column
    train_df = soh_df[soh_df["split"] == "train"]
    val_df = soh_df[soh_df["split"] == "val"]
    test_df = soh_df[soh_df["split"] == "test"]
    
    # Save datasets
    soh_df.to_pickle(SOH_DIR / "soh_full.pkl")
    train_df.to_pickle(SOH_DIR / "soh_train.pkl")
    val_df.to_pickle(SOH_DIR / "soh_val.pkl")
    test_df.to_pickle(SOH_DIR / "soh_test.pkl")
    
    # Save as CSV for inspection
    soh_df.to_csv(SOH_DIR / "soh_full.csv", index=False)
    
    # Statistics
    stats = {
        "total_rows": len(soh_df),
        "total_cells": int(soh_df["barcode"].nunique()),
        "train_rows": len(train_df),
        "train_cells": int(train_df["barcode"].nunique()),
        "val_rows": len(val_df),
        "val_cells": int(val_df["barcode"].nunique()),
        "test_rows": len(test_df),
        "test_cells": int(test_df["barcode"].nunique()),
        "soh_min": float(soh_df["soh"].min()),
        "soh_max": float(soh_df["soh"].max()),
        "soh_mean": float(soh_df["soh"].mean()),
        "features": FEATURES
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(SOH_DIR / "soh_stats.csv", index=False)
    
    print(f"\nSOH Dataset Summary:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Train: {stats['train_rows']:,} rows ({stats['train_cells']} cells)")
    print(f"  Val:   {stats['val_rows']:,} rows ({stats['val_cells']} cells)")
    print(f"  Test:  {stats['test_rows']:,} rows ({stats['test_cells']} cells)")
    print(f"  SOH range: {stats['soh_min']:.4f} - {stats['soh_max']:.4f}")
    
    return soh_df, train_df, val_df, test_df


# ============================================
# RUL DATASET (Supervised - Labeled only)
# ============================================

def create_rul_dataset_supervised(df):
    """
    Create RUL dataset for SUPERVISED learning
    - ONLY KEEPS cells with true RUL labels (34 cells)
    - Drops all unlabeled cells (100 cells with RUL = -1)
    - Every row has a valid RUL value >= 0
    """
    print("\n" + "=" * 60)
    print("CREATING RUL DATASET (SUPERVISED - LABELED ONLY)")
    print("=" * 60)
    
    # FILTER: Keep only cells with valid RUL labels (rul >= 0)
    rul_df = df[df["rul"] >= 0].copy()
    
    # Select columns: metadata + features + RUL
    rul_df = rul_df[["cell_id", "barcode", "channel", "split"] + FEATURES + ["rul"]]
    
    # Remove any rows with NaN in features
    rul_df = rul_df.dropna(subset=FEATURES)
    
    # Split by predefined split column
    train_df = rul_df[rul_df["split"] == "train"]
    val_df = rul_df[rul_df["split"] == "val"]
    test_df = rul_df[rul_df["split"] == "test"]
    
    # Save datasets
    rul_df.to_pickle(RUL_DIR / "rul_full.pkl")
    train_df.to_pickle(RUL_DIR / "rul_train.pkl")
    val_df.to_pickle(RUL_DIR / "rul_val.pkl")
    test_df.to_pickle(RUL_DIR / "rul_test.pkl")
    
    # Save as CSV for inspection
    rul_df.to_csv(RUL_DIR / "rul_full.csv", index=False)
    
    # Statistics
    stats = {
        "total_rows": int(len(rul_df)),
        "total_cells": int(rul_df["barcode"].nunique()),
        "train_rows": int(len(train_df)),
        "train_cells": int(train_df["barcode"].nunique()),
        "val_rows": int(len(val_df)),
        "val_cells": int(val_df["barcode"].nunique()),
        "test_rows": int(len(test_df)),
        "test_cells": int(test_df["barcode"].nunique()),
        "rul_min": float(rul_df["rul"].min()),
        "rul_max": float(rul_df["rul"].max()),
        "rul_mean": float(rul_df["rul"].mean()),
        "rul_std": float(rul_df["rul"].std()),
        "features": FEATURES,
        "dataset_type": "supervised_labeled_only"
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(RUL_DIR / "rul_stats.csv", index=False)
    
    # Save cell information
    cell_info = {
        "total_labeled_cells": int(stats["total_cells"]),
        "labeled_cells": [str(cell) for cell in rul_df["barcode"].unique()],
        "description": "Only cells with valid RUL labels (rul >= 0) are included"
    }
    
    with open(RUL_DIR / "rul_cell_info.json", "w") as f:
        json.dump(cell_info, f, indent=2)
    
    print(f"\nRUL Supervised Dataset Summary (Labeled Only):")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total labeled cells: {stats['total_cells']} (out of 134 total)")
    print(f"  Train: {stats['train_rows']:,} rows ({stats['train_cells']} cells)")
    print(f"  Val:   {stats['val_rows']:,} rows ({stats['val_cells']} cells)")
    print(f"  Test:  {stats['test_rows']:,} rows ({stats['test_cells']} cells)")
    print(f"  RUL range: {stats['rul_min']:.1f} - {stats['rul_max']:.1f} cycles")
    print(f"  RUL mean: {stats['rul_mean']:.1f} cycles")
    print(f"  RUL std: {stats['rul_std']:.1f} cycles")
    
    return rul_df, train_df, val_df, test_df


# ============================================
# NORMALIZED DATASETS
# ============================================

def create_normalized_datasets(soh_full, rul_full):
    """
    Create normalized versions for neural network training
    """
    print("\n" + "=" * 60)
    print("CREATING NORMALIZED DATASETS")
    print("=" * 60)
    
    # Normalize SOH features
    scaler_soh = StandardScaler()
    soh_features = scaler_soh.fit_transform(soh_full[FEATURES])
    
    soh_normalized = soh_full.copy()
    for i, col in enumerate(FEATURES):
        soh_normalized[col] = soh_features[:, i]
    
    # Normalize RUL features (only labeled data)
    scaler_rul = StandardScaler()
    rul_features = scaler_rul.fit_transform(rul_full[FEATURES])
    
    rul_normalized = rul_full.copy()
    for i, col in enumerate(FEATURES):
        rul_normalized[col] = rul_features[:, i]
    
    # Save normalized versions
    soh_normalized.to_pickle(SOH_DIR / "soh_full_normalized.pkl")
    rul_normalized.to_pickle(RUL_DIR / "rul_full_normalized.pkl")
    
    # Save scalers
    with open(SOH_DIR / "scaler_soh.pkl", "wb") as f:
        pickle.dump(scaler_soh, f)
    with open(RUL_DIR / "scaler_rul.pkl", "wb") as f:
        pickle.dump(scaler_rul, f)
    
    print(f"  SOH normalized: {soh_normalized.shape}")
    print(f"  RUL normalized: {rul_normalized.shape}")
    print(f"  Scalers saved to both directories")
    
    return scaler_soh, scaler_rul


# ============================================
# SEQUENCE DATASETS FOR CNN-MAMBA-UQ
# ============================================

def create_sequences(df, feature_cols, target_col, seq_len=50):
    """
    Create sliding window sequences for CNN-Mamba-UQ
    """
    X, y, metadata = [], [], []
    
    # Group by cell
    for cell_id in df["cell_id"].unique():
        cell_data = df[df["cell_id"] == cell_id].sort_values("cycle_index")
        
        if len(cell_data) < seq_len + 1:
            continue
        
        features = cell_data[feature_cols].values
        targets = cell_data[target_col].values
        
        for i in range(len(cell_data) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len])
            metadata.append({
                "cell_id": str(cell_id),
                "cycle": int(cell_data["cycle_index"].iloc[i+seq_len])
            })
    
    return np.array(X), np.array(y), metadata


def save_metadata_safe(metadata, filepath):
    """
    Save metadata safely as JSON with proper type conversion
    """
    serializable_metadata = []
    for item in metadata:
        serializable_item = {
            "cell_id": str(item["cell_id"]),
            "cycle": int(item["cycle"])
        }
        serializable_metadata.append(serializable_item)
    
    with open(filepath, "w") as f:
        json.dump(serializable_metadata, f, indent=2)


def create_sequence_datasets(sequence_length=50):
    """
    Create sequence datasets for both SOH and RUL
    """
    print("\n" + "=" * 60)
    print(f"CREATING SEQUENCE DATASETS (length={sequence_length})")
    print("=" * 60)
    
    # ===== SOH Sequences =====
    print("\n  Creating SOH sequences...")
    soh_train = pd.read_pickle(SOH_DIR / "soh_train.pkl")
    soh_val = pd.read_pickle(SOH_DIR / "soh_val.pkl")
    soh_test = pd.read_pickle(SOH_DIR / "soh_test.pkl")
    
    X_soh_train, y_soh_train, _ = create_sequences(soh_train, FEATURES, "soh", sequence_length)
    X_soh_val, y_soh_val, _ = create_sequences(soh_val, FEATURES, "soh", sequence_length)
    X_soh_test, y_soh_test, _ = create_sequences(soh_test, FEATURES, "soh", sequence_length)
    
    # Save SOH sequences
    np.save(SOH_DIR / f"X_soh_train_seq{sequence_length}.npy", X_soh_train)
    np.save(SOH_DIR / f"y_soh_train_seq{sequence_length}.npy", y_soh_train)
    np.save(SOH_DIR / f"X_soh_val_seq{sequence_length}.npy", X_soh_val)
    np.save(SOH_DIR / f"y_soh_val_seq{sequence_length}.npy", y_soh_val)
    np.save(SOH_DIR / f"X_soh_test_seq{sequence_length}.npy", X_soh_test)
    np.save(SOH_DIR / f"y_soh_test_seq{sequence_length}.npy", y_soh_test)
    
    print(f"    SOH train sequences: {X_soh_train.shape}")
    print(f"    SOH val sequences: {X_soh_val.shape}")
    print(f"    SOH test sequences: {X_soh_test.shape}")
    
    # ===== RUL Sequences (Supervised - Labeled Only) =====
    print("\n  Creating RUL sequences (supervised - labeled only)...")
    rul_train = pd.read_pickle(RUL_DIR / "rul_train.pkl")
    rul_val = pd.read_pickle(RUL_DIR / "rul_val.pkl")
    rul_test = pd.read_pickle(RUL_DIR / "rul_test.pkl")
    
    X_rul_train, y_rul_train, metadata_train = create_sequences(rul_train, FEATURES, "rul", sequence_length)
    X_rul_val, y_rul_val, metadata_val = create_sequences(rul_val, FEATURES, "rul", sequence_length)
    X_rul_test, y_rul_test, metadata_test = create_sequences(rul_test, FEATURES, "rul", sequence_length)
    
    # Save RUL sequences
    np.save(RUL_DIR / f"X_rul_train_seq{sequence_length}.npy", X_rul_train)
    np.save(RUL_DIR / f"y_rul_train_seq{sequence_length}.npy", y_rul_train)
    np.save(RUL_DIR / f"X_rul_val_seq{sequence_length}.npy", X_rul_val)
    np.save(RUL_DIR / f"y_rul_val_seq{sequence_length}.npy", y_rul_val)
    np.save(RUL_DIR / f"X_rul_test_seq{sequence_length}.npy", X_rul_test)
    np.save(RUL_DIR / f"y_rul_test_seq{sequence_length}.npy", y_rul_test)
    
    # Save metadata
    save_metadata_safe(metadata_train, RUL_DIR / f"rul_metadata_train_seq{sequence_length}.json")
    save_metadata_safe(metadata_val, RUL_DIR / f"rul_metadata_val_seq{sequence_length}.json")
    save_metadata_safe(metadata_test, RUL_DIR / f"rul_metadata_test_seq{sequence_length}.json")
    
    print(f"    RUL train sequences: {X_rul_train.shape}")
    print(f"    RUL val sequences: {X_rul_val.shape}")
    print(f"    RUL test sequences: {X_rul_test.shape}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CREATING SEPARATED DATASETS")
    print("SOH: Supervised (All cells)")
    print("RUL: Supervised (Labeled cells only)")
    print("=" * 60)
    
    # Load full dataset
    df = load_full_dataset()
    
    # Create SOH dataset (supervised, all cells)
    soh_full, soh_train, soh_val, soh_test = create_soh_dataset(df)
    
    # Create RUL dataset (supervised, labeled only)
    rul_full, rul_train, rul_val, rul_test = create_rul_dataset_supervised(df)
    
    # Create normalized versions
    create_normalized_datasets(soh_full, rul_full)
    
    # Create sequence datasets for CNN-Mamba-UQ
    create_sequence_datasets(sequence_length=50)
    
    print("\n" + "=" * 60)
    print("DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nSOH Dataset (Supervised - All cells): {SOH_DIR}")
    print(f"  - Use all {soh_full['barcode'].nunique()} cells")
    print(f"  - Target: 'soh'")
    print(f"\nRUL Dataset (Supervised - Labeled Only): {RUL_DIR}")
    print(f"  - Use ONLY {rul_full['barcode'].nunique()} labeled cells")
    print(f"  - Target: 'rul' (all values >= 0)")
    print(f"  - Dropped {df['barcode'].nunique() - rul_full['barcode'].nunique()} unlabeled cells")
    print("\nReady for CNN-Mamba-UQ training!")


if __name__ == "__main__":
    main()