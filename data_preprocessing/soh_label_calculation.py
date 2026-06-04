# data_preprocessing/soh_label_generation.py
"""
SOH LABEL GENERATION & DATASET SAVING (FIXED VERSION)
------------------------------------------------------
Key changes vs original version:
  REMOVED  discharge_capacity_prev  (raw Ah — cell-scale dependent)
  REMOVED  charge_capacity          (raw Ah — cell-scale dependent)
  REMOVED  cycle_index (raw)        (unbounded — different max per cell)

  ADDED    soh_prev                 = discharge_capacity(t-1) / initial_cap
  ADDED    delta_soh                = soh(t) - soh(t-1)
  ADDED    coulombic_eff            = charge_capacity(t-1) / discharge_capacity(t-1)
  KEPT     dc_internal_resistance   (already cell-invariant)
  KEPT     temperature_max          (already cell-invariant)
  REPLACED cycle_index → cycle_norm = cycle_index / eol_cycle ∈ [0,1]

Output:
  results/soh_dataset.csv
  results/soh_dataset.pkl
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

from configurations.config_training_ import (
    RESULTS_DIR, SOH_EOL_THRESHOLD, FEATURE_COLS, TARGET_COL
)

CLEANED_PKL = RESULTS_DIR / "cleaned_cells.pkl"

def compute_soh(cell: Dict) -> np.ndarray:
    """SOH(t) = discharge_capacity(t) / initial_capacity, clamped to [0,1]."""
    soh = cell["discharge_capacity"] / cell["initial_capacity"]
    return np.clip(soh, 0.0, 1.0).astype(np.float32)

def find_eol(soh: np.ndarray, cycle_index: np.ndarray,
             threshold: float = SOH_EOL_THRESHOLD) -> int:
    candidates = np.where(soh < threshold)[0]
    return int(cycle_index[candidates[0]]) if len(candidates) > 0 else int(cycle_index[-1])

def build_cell_dataframe(cell: Dict) -> pd.DataFrame:
    """Build feature + label DataFrame with SOH-normalised features."""
    soh = compute_soh(cell)
    cycle_index = cell["cycle_index"]
    n = len(cycle_index)
    eol = find_eol(soh, cycle_index)
    C0 = cell["initial_capacity"]

    # soh_prev: SOH at t-1
    soh_prev = np.empty(n, dtype=np.float32)
    soh_prev[0] = np.nan
    soh_prev[1:] = soh[:-1]

    # delta_soh: SOH(t) - SOH(t-1)
    delta_soh = np.empty(n, dtype=np.float32)
    delta_soh[0] = np.nan
    delta_soh[1:] = soh[1:] - soh[:-1]

    # coulombic_eff: charge_cap(t-1) / discharge_cap(t-1)
    disch_prev = np.empty(n, dtype=np.float32)
    disch_prev[0] = np.nan
    disch_prev[1:] = cell["discharge_capacity"][:-1]

    charg_prev = np.empty(n, dtype=np.float32)
    charg_prev[0] = np.nan
    charg_prev[1:] = cell["charge_capacity"][:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        coul_eff = np.where(disch_prev > 0.01, charg_prev / disch_prev, np.nan).astype(np.float32)
    coul_eff = np.clip(coul_eff, 0.80, 1.20)

    # cycle_norm: relative life position
    cycle_norm = (cycle_index / max(eol, 1)).astype(np.float32)

    df = pd.DataFrame({
        "soh_prev": soh_prev,
        "delta_soh": delta_soh,
        "coulombic_eff": coul_eff,
        "dc_internal_resistance": cell["dc_internal_resistance"],
        "temperature_max": cell["temperature_max"],
        "cycle_norm": cycle_norm,
        "soh": soh,
        "cycle_index": cycle_index.astype(np.float32),
        "cell_id": cell["barcode"],
        "protocol": cell["protocol"],
        "initial_capacity": C0,
        "eol_cycle": eol,
    })

    df = df.iloc[1:].reset_index(drop=True)

    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
            df[col].fillna(df[col].mean(), inplace=True)

    return df

def print_dataset_report(df: pd.DataFrame) -> None:
    n_cells = df["cell_id"].nunique()
    n_samples = len(df)

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY (fixed features)")
    print(f"{'='*60}")
    print(f"  Cells         : {n_cells}")
    print(f"  Total samples : {n_samples:,}")
    print(f"  Avg cycles    : {n_samples / n_cells:.0f}")
    print(f"\n  SOH distribution:")
    print(f"    min  = {df['soh'].min():.4f}")
    print(f"    mean = {df['soh'].mean():.4f}")
    print(f"    max  = {df['soh'].max():.4f}")
    print(f"\n  Feature ranges:")
    for col in FEATURE_COLS:
        lo = df[col].min()
        hi = df[col].max()
        print(f"    {col:<25}  [{lo:.4f}, {hi:.4f}]")

    corr = df["soh_prev"].corr(df["soh"])
    print(f"\n  Pearson r(soh_prev, soh) = {corr:.4f} (expected > 0.98)")
    print(f"{'='*60}\n")

def save_dataset(df: pd.DataFrame) -> None:
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

    print("\nBuilding fixed feature + SOH label DataFrame...")
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