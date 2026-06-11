# create_separated_datasets.py (FIXED VERSION)

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
# RUL DATASET (Semi-Supervised - All cells)
# ============================================

def create_rul_dataset_semisupervised(df):
    """
    Create RUL dataset for SEMI-SUPERVISED learning
    - KEEPS ALL 134 cells
    - 34 cells have true RUL labels (rul >= 0)
    - 100 cells have RUL = -1 (unlabeled, used for consistency/ranking loss)
    """
    print("\n" + "=" * 60)
    print("CREATING RUL DATASET (SEMI-SUPERVISED)")
    print("=" * 60)
    
    # KEEP ALL CELLS (don't filter!)
    rul_df = df.copy()
    
    # Select columns: metadata + features + RUL
    rul_df = rul_df[["cell_id", "barcode", "channel", "split"] + FEATURES + ["rul"]]
    
    # Add label indicator (1 = has true RUL label, 0 = unlabeled)
    rul_df["has_label"] = (rul_df["rul"] >= 0).astype(int)
    
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
    
    # Semi-supervised metadata
    train_labeled = train_df[train_df["has_label"] == 1]
    train_unlabeled = train_df[train_df["has_label"] == 0]
    
    # Get cell-level statistics (convert numpy types to Python native)
    labeled_cells = train_df[train_df["has_label"] == 1]["barcode"].unique()
    unlabeled_cells = train_df[train_df["has_label"] == 0]["barcode"].unique()
    
    # Statistics (convert all numpy types to Python native)
    stats = {
        "total_rows": int(len(rul_df)),
        "total_cells": int(rul_df["barcode"].nunique()),
        "labeled_cells_total": int(rul_df[rul_df["has_label"] == 1]["barcode"].nunique()),
        "unlabeled_cells_total": int(rul_df[rul_df["has_label"] == 0]["barcode"].nunique()),
        "train_rows": int(len(train_df)),
        "train_labeled_rows": int(len(train_labeled)),
        "train_unlabeled_rows": int(len(train_unlabeled)),
        "train_labeled_cells": int(len(labeled_cells)),
        "train_unlabeled_cells": int(len(unlabeled_cells)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "rul_min": float(rul_df[rul_df["rul"] >= 0]["rul"].min()) if len(rul_df[rul_df["rul"] >= 0]) > 0 else -1.0,
        "rul_max": float(rul_df[rul_df["rul"] >= 0]["rul"].max()) if len(rul_df[rul_df["rul"] >= 0]) > 0 else -1.0,
        "rul_mean": float(rul_df[rul_df["rul"] >= 0]["rul"].mean()) if len(rul_df[rul_df["rul"] >= 0]) > 0 else -1.0,
        "features": FEATURES,
        "semi_supervised_method": "consistency_regularization_and_ranking",
        "unlabeled_placeholder": -1
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(RUL_DIR / "rul_stats.csv", index=False)
    
    # Save semi-supervised info (convert numpy types to Python native)
    semisupervised_info = {
        "labeled_cells_train": [str(cell) for cell in labeled_cells],  # Convert to string
        "unlabeled_cells_train": [str(cell) for cell in unlabeled_cells],
        "total_labeled_cells": int(stats["labeled_cells_total"]),
        "total_unlabeled_cells": int(stats["unlabeled_cells_total"]),
        "description": "Use labeled cells for supervised loss, unlabeled cells for consistency/ranking loss"
    }
    
    with open(RUL_DIR / "rul_semisupervised_info.json", "w") as f:
        json.dump(semisupervised_info, f, indent=2)
    
    print(f"\nRUL Semi-Supervised Dataset Summary:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"    - Labeled cells (true RUL): {stats['labeled_cells_total']}")
    print(f"    - Unlabeled cells (RUL = -1): {stats['unlabeled_cells_total']}")
    print(f"\n  Train split:")
    print(f"    - Labeled rows: {stats['train_labeled_rows']:,} ({stats['train_labeled_cells']} cells)")
    print(f"    - Unlabeled rows: {stats['train_unlabeled_rows']:,} ({stats['train_unlabeled_cells']} cells)")
    print(f"    - Total: {stats['train_rows']:,} rows")
    print(f"\n  RUL range (labeled only): {stats['rul_min']} - {stats['rul_max']} cycles")
    print(f"  RUL mean (labeled only): {stats['rul_mean']:.1f} cycles")
    
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
    
    # Normalize RUL features (using ALL cells, including unlabeled)
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
        has_label = cell_data["has_label"].values if "has_label" in cell_data.columns else np.ones(len(cell_data))
        
        for i in range(len(cell_data) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len])
            # Convert numpy types to Python native for JSON serialization
            metadata.append({
                "cell_id": str(cell_id),  # Convert to string
                "cycle": int(cell_data["cycle_index"].iloc[i+seq_len]),  # Convert to int
                "has_label": int(has_label[i+seq_len])  # Convert to int
            })
    
    return np.array(X), np.array(y), metadata


def save_metadata_safe(metadata, filepath):
    """
    Save metadata safely as JSON with proper type conversion
    """
    # Ensure all values are JSON serializable
    serializable_metadata = []
    for item in metadata:
        serializable_item = {
            "cell_id": str(item["cell_id"]),
            "cycle": int(item["cycle"]),
            "has_label": int(item["has_label"])
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
    
    # Add has_label column for consistency (all SOH data has labels)
    for df in [soh_train, soh_val, soh_test]:
        df["has_label"] = 1
    
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
    
    # ===== RUL Sequences (Semi-Supervised) =====
    print("\n  Creating RUL sequences (semi-supervised)...")
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
    
    # Save metadata safely (with proper JSON serialization)
    save_metadata_safe(metadata_train, RUL_DIR / f"rul_metadata_train_seq{sequence_length}.json")
    save_metadata_safe(metadata_val, RUL_DIR / f"rul_metadata_val_seq{sequence_length}.json")
    save_metadata_safe(metadata_test, RUL_DIR / f"rul_metadata_test_seq{sequence_length}.json")
    
    # Count labeled vs unlabeled in sequences
    train_labeled_count = sum(1 for m in metadata_train if m["has_label"] == 1)
    train_unlabeled_count = sum(1 for m in metadata_train if m["has_label"] == 0)
    
    print(f"    RUL train sequences: {X_rul_train.shape}")
    print(f"      - Labeled sequences (true RUL): {train_labeled_count}")
    print(f"      - Unlabeled sequences (RUL = -1): {train_unlabeled_count}")
    print(f"    RUL val sequences: {X_rul_val.shape}")
    print(f"    RUL test sequences: {X_rul_test.shape}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CREATING SEPARATED DATASETS")
    print("SOH: Supervised | RUL: Semi-Supervised")
    print("=" * 60)
    
    # Load full dataset
    df = load_full_dataset()
    
    # Create SOH dataset (supervised, all cells)
    soh_full, soh_train, soh_val, soh_test = create_soh_dataset(df)
    
    # Create RUL dataset (semi-supervised, all cells)
    rul_full, rul_train, rul_val, rul_test = create_rul_dataset_semisupervised(df)
    
    # Create normalized versions
    create_normalized_datasets(soh_full, rul_full)
    
    # Create sequence datasets for CNN-Mamba-UQ
    create_sequence_datasets(sequence_length=50)
    
    print("\n" + "=" * 60)
    print("DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nSOH Dataset (Supervised): {SOH_DIR}")
    print(f"  - Use all {soh_full['barcode'].nunique()} cells")
    print(f"  - Target: 'soh'")
    print(f"\nRUL Dataset (Semi-Supervised): {RUL_DIR}")
    print(f"  - Use all {rul_full['barcode'].nunique()} cells")
    print(f"  - Labeled cells: {rul_full[rul_full['has_label']==1]['barcode'].nunique()}")
    print(f"  - Unlabeled cells: {rul_full[rul_full['has_label']==0]['barcode'].nunique()}")
    print(f"  - Unlabeled cells have RUL = -1 (use for consistency/ranking loss)")
    print("\nReady for CNN-Mamba-UQ training!")


if __name__ == "__main__":
    main()