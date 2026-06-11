# target_calculation.py
"""
Target Calculation for SOH and RUL
---------------------------------
Loads cleaned cells, calculates SOH and RUL targets, 
engineers features (CE lagged), and saves final dataset ready for training.

Output:
  - final_dataset/final_dataset.pkl    (Complete dataset with SOH + RUL targets)
  - final_dataset/dataset_stats.csv    (Summary statistics)
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configurations.config_features import (
    CLEANED_CELLS_PKL,
    PHYSICAL_CELLS_GROUPING_PKL,
    FINAL_FEATURES,
    SOH_TARGET,
    RUL_TARGET,
    SOH_EOL_THRESHOLD,
)

# Output directory
FINAL_DATASET_DIR = Path(__file__).parent / "final_dataset"
FINAL_DATASET_DIR.mkdir(exist_ok=True)


def calculate_coulombic_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Coulombic Efficiency with lagged values (no leakage)
    """
    df = df.copy()
    
    # Actual CE (needs discharge_capacity)
    df["ce_actual"] = df["charge_capacity"] / df["discharge_capacity"]
    
    # Lagged features (no future information)
    df["coulombic_efficiency_lagged_1"] = df["ce_actual"].shift(1)
    df["coulombic_efficiency_lagged_2"] = df["ce_actual"].shift(2)
    
    return df


def calculate_soh_target(df: pd.DataFrame, initial_capacity: float) -> pd.DataFrame:
    """
    Calculate SOH (State of Health) target
    """
    df = df.copy()
    df[SOH_TARGET] = df["discharge_capacity"] / initial_capacity
    df[SOH_TARGET] = df[SOH_TARGET].clip(0, 1)
    return df


def calculate_rul_target(df: pd.DataFrame, eol_threshold: float = 0.80) -> pd.DataFrame:
    """
    Calculate RUL (Remaining Useful Life) target
    """
    df = df.copy()
    
    eol_mask = df[SOH_TARGET] <= eol_threshold
    if eol_mask.any():
        eol_cycle = df[eol_mask]["cycle_index"].min()
        df[RUL_TARGET] = eol_cycle - df["cycle_index"]
        df[RUL_TARGET] = df[RUL_TARGET].clip(lower=0)
    else:
        df[RUL_TARGET] = -1  # Censored
    
    return df


def process_single_cell(cell_data: Dict, cell_id: int) -> Optional[pd.DataFrame]:
    """Process one cell into DataFrame with targets"""
    
    df = pd.DataFrame({
        "cell_id": cell_id,
        "barcode": cell_data["barcode"],
        "channel": cell_data["channel"],
        "batch": cell_data["batch"],
        "protocol": cell_data["protocol"],
        "cycle_index": cell_data["cycle_index"],
        "discharge_capacity": cell_data["discharge_capacity"],
        "charge_capacity": cell_data["charge_capacity"],
        "dc_internal_resistance": cell_data["dc_internal_resistance"],
        "temperature_avg": cell_data["temperature_avg"],
        "charge_energy": cell_data["charge_energy"],
        "initial_capacity": cell_data["initial_capacity"],
        "total_cycles": cell_data["total_cycles"],
    })
    
    # Calculate features and targets
    df = calculate_coulombic_efficiency(df)
    df = calculate_soh_target(df, cell_data["initial_capacity"])
    df = calculate_rul_target(df, SOH_EOL_THRESHOLD)
    
    # Drop rows with NaN (from lagged calculations)
    df = df.dropna()
    
    return df


def load_and_process_cells(cells: List[Dict]) -> pd.DataFrame:
    """Process all cells into single DataFrame"""
    all_dfs = []
    
    for cell_id, cell_data in enumerate(tqdm(cells, desc="Processing cells")):
        df_cell = process_single_cell(cell_data, cell_id)
        if df_cell is not None and len(df_cell) > 0:
            all_dfs.append(df_cell)
    
    if not all_dfs:
        raise ValueError("No cells were successfully processed")
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df


def create_train_test_split_by_physical_cell(
    df: pd.DataFrame,
    physical_cells_grouping: Dict[str, List[Dict]],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by physical cell (no data leakage)"""
    from sklearn.model_selection import train_test_split
    
    unique_barcodes = df["barcode"].unique()
    
    train_barcodes, temp_barcodes = train_test_split(
        unique_barcodes,
        test_size=test_size + val_size,
        random_state=random_seed
    )
    
    val_barcodes, test_barcodes = train_test_split(
        temp_barcodes,
        test_size=test_size / (test_size + val_size),
        random_state=random_seed
    )
    
    train_df = df[df["barcode"].isin(train_barcodes)]
    val_df = df[df["barcode"].isin(val_barcodes)]
    test_df = df[df["barcode"].isin(test_barcodes)]
    
    # Add split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    return train_df, val_df, test_df


def save_combined_dataset(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save ONE combined dataset with all information
    """
    # Combine all splits with split indicator
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Select final columns
    # final_columns = [
    #     "cell_id", "barcode", "channel", "batch", "protocol",
    #     "split",  # train/val/test indicator
    #     "cycle_index", "total_cycles",
    # ] + FINAL_FEATURES + [SOH_TARGET, RUL_TARGET]

    final_columns = [
    "cell_id", "barcode", "channel", "batch", "protocol",
    "split",
    "cycle_index", "total_cycles",  # Only once!
     ] + [f for f in FINAL_FEATURES if f != "cycle_index"] + [SOH_TARGET, RUL_TARGET]
    
    combined_df = combined_df[final_columns]
    
    # Save combined dataset
    combined_df.to_pickle(FINAL_DATASET_DIR / "final_dataset.pkl")
    combined_df.to_csv(FINAL_DATASET_DIR / "final_dataset.csv", index=False)
    
    print(f"\nSaved to: {FINAL_DATASET_DIR}")
    print(f"  - final_dataset.pkl (Combined dataset with SOH + RUL + split)")
    print(f"  - final_dataset.csv (Same as CSV)")


def generate_stats(df: pd.DataFrame):
    """Generate and save dataset statistics"""
    
    stats = {
        # Overall
        "total_physical_cells": df["barcode"].nunique(),
        "total_channels": df["cell_id"].nunique(),
        "total_rows": len(df),
        
        # Split sizes
        "train_rows": (df["split"] == "train").sum(),
        "val_rows": (df["split"] == "val").sum(),
        "test_rows": (df["split"] == "test").sum(),
        
        "train_cells": df[df["split"] == "train"]["barcode"].nunique(),
        "val_cells": df[df["split"] == "val"]["barcode"].nunique(),
        "test_cells": df[df["split"] == "test"]["barcode"].nunique(),
        
        # Target ranges
        "soh_min": df[SOH_TARGET].min(),
        "soh_max": df[SOH_TARGET].max(),
        "soh_mean": df[SOH_TARGET].mean(),
        
        "rul_min": df[df[RUL_TARGET] >= 0][RUL_TARGET].min(),
        "rul_max": df[df[RUL_TARGET] >= 0][RUL_TARGET].max(),
        "rul_mean": df[df[RUL_TARGET] >= 0][RUL_TARGET].mean(),
        
        # Data quality
        "censored_rows": (df[RUL_TARGET] == -1).sum(),
        "censored_cells": df[df[RUL_TARGET] == -1]["barcode"].nunique(),
        
        # Cycle range
        "cycle_min": df["cycle_index"].min(),
        "cycle_max": df["cycle_index"].max(),
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(FINAL_DATASET_DIR / "dataset_stats.csv", index=False)
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return stats_df


def main():
    print("=" * 60)
    print("TARGET CALCULATION: SOH and RUL")
    print("=" * 60)
    
    # Step 1: Load cleaned cells
    print("\n[1] Loading cleaned cells...")
    with open(CLEANED_CELLS_PKL, "rb") as f:
        cells = pickle.load(f)
    print(f"  Loaded {len(cells)} channels")
    
    # Step 2: Load physical cell grouping
    print("\n[2] Loading physical cell grouping...")
    with open(PHYSICAL_CELLS_GROUPING_PKL, "rb") as f:
        physical_cells = pickle.load(f)
    print(f"  Loaded {len(physical_cells)} physical cells")
    
    # Step 3: Process all cells
    print("\n[3] Processing cells and calculating targets...")
    final_df = load_and_process_cells(cells)
    print(f"  Final dataset shape: {final_df.shape}")
    
    # Step 4: Create train/val/test splits
    print("\n[4] Creating train/val/test splits (by physical cell)...")
    train_df, val_df, test_df = create_train_test_split_by_physical_cell(
        final_df, physical_cells, test_size=0.2, val_size=0.1, random_seed=42
    )
    
    # Step 5: Save combined dataset
    print("\n[5] Saving combined dataset...")
    save_combined_dataset(final_df, train_df, val_df, test_df)
    
    # Step 6: Load saved dataset and show stats
    combined = pd.read_pickle(FINAL_DATASET_DIR / "final_dataset.pkl")
    
    # Step 7: Generate statistics
    print("\n[6] Generating statistics...")
    generate_stats(combined)
    
    # Step 8: Display sample
    print("\n[7] Sample of final dataset (first 10 rows):")
    print("=" * 100)
    display_cols = ["barcode", "split", "cycle_index", "charge_capacity", 
                    "dc_internal_resistance", "coulombic_efficiency_lagged_1", 
                    SOH_TARGET, RUL_TARGET]
    print(combined[display_cols].head(10).to_string())
    
    print("\n" + "=" * 60)
    print("TARGET CALCULATION COMPLETE")
    print("=" * 60)
    print(f"\nReady for training!")
    print(f"Load with: pd.read_pickle('{FINAL_DATASET_DIR / 'final_dataset.pkl'}')")


if __name__ == "__main__":
    main()