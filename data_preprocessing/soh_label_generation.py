"""
step2_soh_labels.py
====================
SOH LABEL GENERATION & DATASET SAVING
---------------------------------------
Reads cleaned_cells.pkl produced by step1, then:

  1. Computes per-cycle SOH = Q_discharge(t) / Q_initial
  2. Detects EOL cycle
  3. Builds feature matrix with lag features
  4. Adds SOH label column
  5. Saves final preprocessed dataset

Output:
  results/soh_dataset.csv   (human-readable)
  results/soh_dataset.pkl   (fast load for training)
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_preprocessing import (
    RESULTS_DIR, SOH_EOL_THRESHOLD, FEATURE_COLS, TARGET_COL, NOMINAL_CAPACITY
)

CLEANED_PKL = RESULTS_DIR / "cleaned_cells.pkl"

def compute_soh(cell: Dict) -> np.ndarray:
    """SOH(t) = discharge_capacity(t) / initial_capacity. Clamped to [0.0, 1.0]."""
    soh = cell["discharge_capacity"] / cell["initial_capacity"]
    return np.clip(soh, 0.0, 1.0).astype(np.float32)

def find_eol(soh: np.ndarray, cycle_index: np.ndarray, threshold: float = SOH_EOL_THRESHOLD) -> int:
    """Return the cycle index at which SOH first drops below threshold."""
    candidates = np.where(soh < threshold)[0]
    if len(candidates) > 0:
        return int(cycle_index[candidates[0]])
    return int(cycle_index[-1])

def build_cell_dataframe(cell: Dict) -> pd.DataFrame:
    """Build feature + label DataFrame for one cell."""
    soh = compute_soh(cell)
    cycle_index = cell["cycle_index"]
    n = len(cycle_index)

    discharge_cap_prev = np.empty(n, dtype=np.float32)
    discharge_cap_prev[0] = np.nan
    discharge_cap_prev[1:] = cell["discharge_capacity"][:-1]

    df = pd.DataFrame({
        "discharge_capacity_prev": discharge_cap_prev,
        "charge_capacity": cell["charge_capacity"],
        "dc_internal_resistance": cell["dc_internal_resistance"],
        "temperature_max": cell["temperature_max"],
        "temperature_avg": cell["temperature_avg"],
        "cycle_index": cycle_index.astype(np.float32),
        "soh": soh,
        "cell_id": cell["barcode"],
        "protocol": cell["protocol"],
        "initial_capacity": cell["initial_capacity"],
        "eol_cycle": find_eol(soh, cycle_index),
    })

    df = df.iloc[1:].reset_index(drop=True)
    return df

def print_dataset_report(df: pd.DataFrame) -> None:
    """Print summary statistics of the final dataset."""
    n_cells = df["cell_id"].nunique()
    n_samples = len(df)
    soh_min = df["soh"].min()
    soh_max = df["soh"].max()
    soh_mean = df["soh"].mean()
    cycles_mean = df.groupby("cell_id")["cycle_index"].count().mean()
    eol_mean = df.groupby("cell_id")["eol_cycle"].first().mean()
    n_eol_reached = (df.groupby("cell_id")["soh"].min() < SOH_EOL_THRESHOLD).sum()

    print(f"\n{'='*60}")
    print(f"SOH LABEL GENERATION - DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Cells           : {n_cells}")
    print(f"  Total samples   : {n_samples:,}")
    print(f"  Avg cycles/cell : {cycles_mean:.0f}")
    print(f"  Avg EOL cycle   : {eol_mean:.0f}")
    print(f"  Cells at EOL    : {n_eol_reached} / {n_cells}")
    print(f"\n  SOH distribution:")
    print(f"    min  = {soh_min:.4f}")
    print(f"    mean = {soh_mean:.4f}")
    print(f"    max  = {soh_max:.4f}")
    print(f"{'='*60}\n")

def save_dataset(df: pd.DataFrame) -> None:
    """Save final dataset to CSV and PKL formats."""
    csv_path = RESULTS_DIR / "soh_dataset.csv"
    pkl_path = RESULTS_DIR / "soh_dataset.pkl"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"  Saved CSV: {csv_path}")
    print(f"  Saved PKL: {pkl_path}")

if __name__ == "__main__":
    print(f"\nLoading cleaned cells from {CLEANED_PKL}...")
    with open(CLEANED_PKL, "rb") as f:
        cells = pickle.load(f)
    print(f"  Loaded {len(cells)} cells")

    print("\nBuilding per-cycle feature and SOH label DataFrame...")
    cell_dfs = []
    for i, cell in enumerate(cells):
        cdf = build_cell_dataframe(cell)
        cell_dfs.append(cdf)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells")

    df_all = pd.concat(cell_dfs, ignore_index=True)

    print_dataset_report(df_all)
    save_dataset(df_all)
    print("\nStep 2 complete. Final dataset ready for training.")