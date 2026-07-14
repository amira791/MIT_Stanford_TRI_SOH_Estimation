# recreate_full_dataset_with_all_features.py
"""
Recreate Full Dataset with ALL Features for Feature Importance Analysis
-----------------------------------------------------------------------
1. Loads cleaned cells from results2/ (with ALL summary features)
2. Adds lagged CE features (coulombic_efficiency_lagged_1, _2)
3. Adds engineered features (cap_rel, energy_rel, ir_rel, cycle_pos)
4. Saves to results2/soh_dataset_full_features.pkl

Features included:
- Summary (10): cycle_index, discharge_capacity, charge_capacity,
  discharge_energy, charge_energy, dc_internal_resistance,
  temperature_maximum, temperature_average, temperature_minimum,
  date_time_iso_numeric
- Lagged CE (2): coulombic_efficiency_lagged_1, _2
- Engineered (4): cap_rel, energy_rel, ir_rel, cycle_pos
- Target: soh
- Metadata: cell_id, protocol, etc.
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

RESULTS2_DIR = Path(__file__).parent.parent / "results2"
RESULTS2_DIR.mkdir(exist_ok=True)

CLEANED_PKL = RESULTS2_DIR / "cleaned_cells_all_features.pkl"
OUTPUT_PKL = RESULTS2_DIR / "soh_dataset_full_features.pkl"
OUTPUT_CSV = RESULTS2_DIR / "soh_dataset_full_features.csv"

# ============================================================
# CONFIGURATION
# ============================================================

SOH_EOL_THRESHOLD = 0.80
INIT_CYCLES_AVG = 5
NOMINAL_CAPACITY = 1.1

# ============================================================
# ALL FEATURES WE WANT
# ============================================================

# Summary features (10)
SUMMARY_FEATURES = [
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

# Lagged CE features (2) - will be computed
CE_FEATURES = [
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
]

# Engineered features (4) - will be computed
ENGINEERED_FEATURES = [
    "cap_rel",
    "energy_rel",
    "ir_rel",
    "cycle_pos",
]

# All feature columns for the final dataset
ALL_FEAT_COLS = SUMMARY_FEATURES + CE_FEATURES + ENGINEERED_FEATURES

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
    
    # ── initial resistance ─────────────────────────────────────────────
    R0 = float(np.nanmean(cell["dc_internal_resistance"][:INIT_CYCLES_AVG]))
    if R0 <= 0 or np.isnan(R0):
        R0 = float(np.nanmean(cell["dc_internal_resistance"]))
    if R0 <= 0:
        R0 = 0.02  # fallback

    # ── normalized resistance: (R - R0) / R0 ──────────────────────────
    dc_ir_norm = (cell["dc_internal_resistance"] - R0) / R0
    dc_ir_norm = np.clip(dc_ir_norm, -0.10, 2.0).astype(np.float32)

    # ── normalized cycle position ─────────────────────────────────────
    cycle_norm = np.clip(cycle_index / max(eol, 1), 0.0, 1.0).astype(np.float32)

    # ── lagged coulombic efficiency ──────────────────────────────────
    # Coulombic efficiency = charge_capacity / discharge_capacity
    with np.errstate(divide="ignore", invalid="ignore"):
        coul_eff = cell["charge_capacity"] / cell["discharge_capacity"]
        coul_eff = np.where(cell["discharge_capacity"] > 0.01, coul_eff, np.nan)
    coul_eff = np.clip(coul_eff, 0.80, 1.20).astype(np.float32)
    
    # Lagged versions
    ce_lag1 = np.full(n, np.nan, dtype=np.float32)
    ce_lag2 = np.full(n, np.nan, dtype=np.float32)
    ce_lag1[1:] = coul_eff[:-1]  # t-1
    ce_lag2[2:] = coul_eff[:-2]  # t-2

    # ── per-cell relative features (using first 10 cycles) ──────────
    early = cell["cycle_index"][:10]
    early_cap = cell["charge_capacity"][:10]
    early_energy = cell["charge_energy"][:10]
    early_ir = cell["dc_internal_resistance"][:10]
    
    nom_cap = np.nanmean(early_cap)
    nom_energy = np.nanmean(early_energy)
    nom_ir = np.nanmean(early_ir)
    
    # Avoid division by zero
    if nom_cap <= 0 or np.isnan(nom_cap):
        nom_cap = 1.0
    if nom_energy <= 0 or np.isnan(nom_energy):
        nom_energy = 3.0
    if nom_ir <= 0 or np.isnan(nom_ir):
        nom_ir = 0.02
    
    cap_rel = (cell["charge_capacity"] - nom_cap) / nom_cap
    energy_rel = (cell["charge_energy"] - nom_energy) / nom_energy
    ir_rel = (cell["dc_internal_resistance"] - nom_ir) / nom_ir
    
    # ── build DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        # Summary features (10)
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
        
        # Lagged CE features
        "coulombic_efficiency_lagged_1": ce_lag1,
        "coulombic_efficiency_lagged_2": ce_lag2,
        
        # Engineered features
        "cap_rel": cap_rel.astype(np.float32),
        "energy_rel": energy_rel.astype(np.float32),
        "ir_rel": ir_rel.astype(np.float32),
        "cycle_pos": cycle_norm,  # This is the same as cycle_norm
        "dc_ir_norm": dc_ir_norm,
        "cycle_norm": cycle_norm,
        
        # Target
        "soh": soh,
        
        # Metadata
        "cell_id": cell["barcode"],
        "protocol": cell["protocol"],
        "initial_capacity": cell["initial_capacity"],
        "initial_resistance": R0,
        "eol_cycle": eol,
    })

    # Drop first 2 rows (NaN from lagged CE)
    df = df.iloc[2:].reset_index(drop=True)

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
    print("RECREATING FULL DATASET WITH ALL FEATURES")
    print("=" * 60)

    print(f"\n[1] Loading cleaned cells from: {CLEANED_PKL}")
    with open(CLEANED_PKL, "rb") as f:
        cells: List[Dict] = pickle.load(f)
    print(f"  Loaded {len(cells)} cells")

    print("\n[2] Building cell DataFrames...")
    cell_dfs = []
    for i, cell in enumerate(cells):
        try:
            cdf = build_cell_dataframe(cell)
            cell_dfs.append(cdf)
        except Exception as e:
            print(f"  Warning: error processing cell {cell.get('barcode', 'unknown')}: {e}")
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells")

    print(f"\n[3] Concatenating all cells...")
    df_all = pd.concat(cell_dfs, ignore_index=True)

    print(f"\n[4] Final dataset statistics:")
    print(f"  Shape: {df_all.shape}")
    print(f"  Cells: {df_all['cell_id'].nunique()}")
    print(f"  Features: {len(df_all.columns)}")

    print("\n[5] Feature list:")
    for i, col in enumerate(df_all.columns):
        print(f"  {i+1:>2}. {col}")

    print("\n[6] Feature summary:")
    print(f"  Summary features (10): {[c for c in SUMMARY_FEATURES if c in df_all.columns]}")
    print(f"  CE features (2): {[c for c in CE_FEATURES if c in df_all.columns]}")
    print(f"  Engineered features (4): {[c for c in ENGINEERED_FEATURES if c in df_all.columns]}")

    print("\n[7] Target range:")
    print(f"  soh min: {df_all['soh'].min():.4f}")
    print(f"  soh max: {df_all['soh'].max():.4f}")
    print(f"  soh mean: {df_all['soh'].mean():.4f}")

    # Save to CSV and pickle
    print("\n[8] Saving datasets...")
    df_all.to_csv(OUTPUT_CSV, index=False)
    df_all.to_pickle(OUTPUT_PKL)

    print(f"\n  ✅ Saved: {OUTPUT_CSV}")
    print(f"  ✅ Saved: {OUTPUT_PKL}")

    print("\n" + "=" * 60)
    print("DATASET CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nNow run feature importance using:")
    print(f"  DATA_PATH = {OUTPUT_PKL}")
    print("\nAll features are available for importance analysis.")


if __name__ == "__main__":
    main()