# step2_soh_labels_v4.py
"""
SOH Dataset Generation V4 - Using ALL Summary Features
=======================================================
Uses the new cleaned cells with ALL 10 summary features from results2/

Features available:
- cycle_index
- discharge_capacity
- charge_capacity
- discharge_energy
- charge_energy
- dc_internal_resistance
- temperature_maximum
- temperature_average
- temperature_minimum
- date_time_iso_numeric

Plus engineered features:
- soh (target)
- soh_prev (lagged SOH)
- delta_soh (SOH change rate)
- coulombic_eff (charge/discharge efficiency)
- dc_ir_norm (normalized resistance)
- cycle_norm (normalized cycle position)
- eol_cycle (end-of-life cycle)
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

# ============================================================
# CONFIGURATION
# ============================================================

# Use results2 (new preprocessing with ALL features)
RESULTS2_DIR = Path(__file__).parent.parent / "results2"
RESULTS2_DIR.mkdir(exist_ok=True)

# Original results directory for comparison
RESULTS_DIR = Path(__file__).parent.parent / "results_mlt"
RESULTS_DIR.mkdir(exist_ok=True)

# Path to new cleaned cells (ALL features)
CLEANED_PKL = RESULTS2_DIR / "cleaned_cells_all_features.pkl"

# Parameters
SOH_EOL_THRESHOLD = 0.80
INIT_CYCLES_AVG = 5
NOMINAL_CAPACITY = 1.1

# ============================================================
# FEATURE COLUMNS FOR THE DATASET
# ============================================================

# All available features from summary data
ALL_SUMMARY_FEATURES = [
    "cycle_index",
    "discharge_capacity",
    "charge_capacity",
    "discharge_energy",
    "charge_energy",
    "dc_internal_resistance",
    "temperature_maximum",
    "temperature_average",
    "temperature_minimum",
    "date_time_iso_numeric",
]

# Engineered features to add
ENGINEERED_FEATURES = [
    "soh",           # target
    "soh_prev",      # SOH at t-1
    "delta_soh",     # SOH(t) - SOH(t-1)
    "coulombic_eff", # charge_capacity / discharge_capacity
    "dc_ir_norm",    # normalized resistance (R - R0) / R0
    "cycle_norm",    # normalized cycle position
    "eol_cycle",     # end-of-life cycle
]

# Combined feature list
FEATURE_COLS = ALL_SUMMARY_FEATURES + ENGINEERED_FEATURES

# ============================================================
# FUNCTIONS
# ============================================================

def compute_soh(cell: Dict) -> np.ndarray:
    """Compute SoH from discharge_capacity and initial_capacity"""
    return np.clip(
        cell["discharge_capacity"] / cell["initial_capacity"], 0.0, 1.0
    ).astype(np.float32)


def find_eol(soh: np.ndarray, cycle_index: np.ndarray,
             threshold: float = SOH_EOL_THRESHOLD) -> int:
    """Find the cycle where SoH drops below threshold"""
    candidates = np.where(soh < threshold)[0]
    return int(cycle_index[candidates[0]]) if len(candidates) > 0 \
           else int(cycle_index[-1])


def build_cell_dataframe(cell: Dict) -> pd.DataFrame:
    """Build a DataFrame for a single cell with ALL features"""
    
    soh = compute_soh(cell)
    cycle_index = cell["cycle_index"]
    n = len(cycle_index)
    eol = find_eol(soh, cycle_index)
    C0 = cell["initial_capacity"]

    # ── initial resistance ─────────────────────────────────────────────
    R0 = float(np.nanmean(cell["dc_internal_resistance"][:INIT_CYCLES_AVG]))
    if R0 <= 0 or np.isnan(R0):
        R0 = float(np.nanmean(cell["dc_internal_resistance"]))
    if R0 <= 0:
        R0 = 0.02   # fallback to typical LFP value

    # ── normalized resistance: (R - R0) / R0 ──────────────────────────
    dc_ir_norm = (cell["dc_internal_resistance"] - R0) / R0
    dc_ir_norm = np.clip(dc_ir_norm, -0.10, 2.0).astype(np.float32)

    # ── lag features ──────────────────────────────────────────────────
    soh_prev = np.empty(n, dtype=np.float32)
    soh_prev[0] = np.nan
    soh_prev[1:] = soh[:-1]

    delta_soh = np.empty(n, dtype=np.float32)
    delta_soh[0] = np.nan
    delta_soh[1:] = soh[1:] - soh[:-1]

    disch_prev = np.empty(n, dtype=np.float32)
    disch_prev[0] = np.nan
    disch_prev[1:] = cell["discharge_capacity"][:-1]

    charg_prev = np.empty(n, dtype=np.float32)
    charg_prev[0] = np.nan
    charg_prev[1:] = cell["charge_capacity"][:-1]

    # ── coulombic efficiency ──────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        coul_eff = np.where(disch_prev > 0.01,
                            charg_prev / disch_prev, np.nan).astype(np.float32)
    coul_eff = np.clip(coul_eff, 0.80, 1.20)

    # ── normalized cycle position ─────────────────────────────────────
    cycle_norm = np.clip(cycle_index / max(eol, 1), 0.0, 1.0).astype(np.float32)

    # ── build DataFrame with ALL features ─────────────────────────────
    df = pd.DataFrame({
        # ALL summary features
        "cycle_index": cycle_index.astype(np.float32),
        "discharge_capacity": cell["discharge_capacity"].astype(np.float32),
        "charge_capacity": cell["charge_capacity"].astype(np.float32),
        "discharge_energy": cell["discharge_energy"].astype(np.float32),
        "charge_energy": cell["charge_energy"].astype(np.float32),
        "dc_internal_resistance": cell["dc_internal_resistance"].astype(np.float32),
        "temperature_maximum": cell["temperature_maximum"].astype(np.float32),
        "temperature_average": cell["temperature_average"].astype(np.float32),
        "temperature_minimum": cell["temperature_minimum"].astype(np.float32),
        "date_time_iso_numeric": cell["date_time_iso_numeric"].astype(np.float32),
        
        # Engineered features
        "soh": soh,
        "soh_prev": soh_prev,
        "delta_soh": delta_soh,
        "coulombic_eff": coul_eff,
        "dc_ir_norm": dc_ir_norm,
        "cycle_norm": cycle_norm,
        
        # Metadata
        "cell_id": cell["barcode"],
        "protocol": cell["protocol"],
        "initial_capacity": C0,
        "initial_resistance": R0,
        "eol_cycle": eol,
    })

    # Remove first row (NaN from lag features)
    df = df.iloc[1:].reset_index(drop=True)

    # Fill any remaining NaNs
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill().fillna(df[col].mean())

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SOH DATASET GENERATION V4 - ALL FEATURES")
    print("=" * 60)
    
    print(f"\n[1] Loading cleaned cells from: {CLEANED_PKL}")
    with open(CLEANED_PKL, "rb") as f:
        cells: List[Dict] = pickle.load(f)
    print(f"  Loaded {len(cells)} cells")

    print(f"\n[2] Available features per cell:")
    sample_cell = cells[0]
    for key in sample_cell.keys():
        print(f"    - {key}")

    print("\n[3] Building cell DataFrames...")
    cell_dfs = []
    for i, cell in enumerate(cells):
        cdf = build_cell_dataframe(cell)
        cell_dfs.append(cdf)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells")

    print("\n[4] Concatenating all cells...")
    df_all = pd.concat(cell_dfs, ignore_index=True)

    print("\n[5] Final dataset statistics:")
    print(f"  Shape: {df_all.shape}")
    print(f"  Cells: {df_all['cell_id'].nunique()}")
    print(f"  Columns: {df_all.columns.tolist()}")

    print("\n[6] Feature summary:")
    print(f"  Total features: {len(df_all.columns)}")
    print(f"  Features from summary data: {len(ALL_SUMMARY_FEATURES)}")
    print(f"  Engineered features: {len(ENGINEERED_FEATURES)}")

    # Summary statistics
    print("\n[7] Summary statistics:")
    print(f"  SoH range: [{df_all['soh'].min():.4f}, {df_all['soh'].max():.4f}]")
    print(f"  SoH mean: {df_all['soh'].mean():.4f}")
    print(f"  dc_ir_norm range: [{df_all['dc_ir_norm'].min():.4f}, {df_all['dc_ir_norm'].max():.4f}]")
    print(f"  cycle_norm range: [{df_all['cycle_norm'].min():.4f}, {df_all['cycle_norm'].max():.4f}]")
    print(f"  coulombic_eff range: [{df_all['coulombic_eff'].min():.4f}, {df_all['coulombic_eff'].max():.4f}]")

    # Save to CSV and pickle
    print("\n[8] Saving datasets...")
    csv_path = RESULTS2_DIR / "soh_dataset_all_features.csv"
    pkl_path = RESULTS2_DIR / "soh_dataset_all_features.pkl"
    
    df_all.to_csv(csv_path, index=False)
    df_all.to_pickle(pkl_path)
    
    print(f"\n  ✅ Saved: {csv_path}")
    print(f"  ✅ Saved: {pkl_path}")

    print("\n" + "=" * 60)
    print("SOH DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset saved to: {RESULTS2_DIR}")
    print(f"  - soh_dataset_all_features.csv")
    print(f"  - soh_dataset_all_features.pkl")
    print("\nReady for feature importance analysis!")


if __name__ == "__main__":
    main()