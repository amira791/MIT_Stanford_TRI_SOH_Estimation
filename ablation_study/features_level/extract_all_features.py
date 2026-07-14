"""
EXTRACT ALL SUMMARY FEATURES FOR FEATURE IMPORTANCE ANALYSIS
=============================================================
Extracts ALL available summary features from the cleaned cells
to perform complete feature importance analysis.

Features extracted:
- cycle_index
- discharge_capacity
- charge_capacity
- discharge_energy
- charge_energy
- dc_internal_resistance
- temperature_maximum
- temperature_average
- temperature_minimum
- date_time_iso (optional, converted to numeric)
- SoH (target)

Output: full_summary_features.csv
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

CLEANED_PKL = Path(__file__).parent.parent / "results" / "cleaned_cells.pkl"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "all_summary_features.csv"
OUTPUT_PKL = OUTPUT_DIR / "all_summary_features.pkl"

# ============================================================
# CONFIGURATION
# ============================================================

INIT_CYCLES_AVG = 5
NOMINAL_CAPACITY = 1.1

# All summary features (from dataset documentation)
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
    "date_time_iso",
]

# ============================================================
# EXTRACTION
# ============================================================

def compute_soh(cell: Dict) -> np.ndarray:
    """Compute SoH from discharge_capacity and initial_capacity"""
    return np.clip(
        cell["discharge_capacity"] / cell["initial_capacity"], 0.0, 1.0
    ).astype(np.float32)


def parse_date_time(date_str) -> float:
    """Convert date_time_iso to numeric (seconds since epoch)"""
    if pd.isna(date_str) or date_str == "":
        return np.nan
    try:
        import datetime
        if isinstance(date_str, str):
            # Try different formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    dt = datetime.datetime.strptime(date_str, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
        return np.nan
    except:
        return np.nan


def extract_cell_data(cell: Dict) -> pd.DataFrame:
    """Extract ALL summary features from a single cell"""
    
    # Get arrays from cell
    n = len(cell["cycle_index"])
    
    # Extract all available features
    data = {
        "cycle_index": cell["cycle_index"],
        "discharge_capacity": cell["discharge_capacity"],
        "charge_capacity": cell["charge_capacity"],
        "discharge_energy": cell["discharge_energy"],
        "charge_energy": cell["charge_energy"],
        "dc_internal_resistance": cell["dc_internal_resistance"],
        "temperature_maximum": cell["temperature_maximum"],
        "temperature_average": cell["temperature_average"],
        "temperature_minimum": cell["temperature_minimum"],
        "barcode": [cell["barcode"]] * n,
        "protocol": [cell["protocol"]] * n,
    }
    
    # Optional: extract date_time_iso if available
    if "date_time_iso" in cell and cell["date_time_iso"] is not None:
        if isinstance(cell["date_time_iso"], (list, np.ndarray)):
            data["date_time_iso"] = cell["date_time_iso"]
        else:
            data["date_time_iso"] = [cell["date_time_iso"]] * n
    else:
        data["date_time_iso"] = [np.nan] * n
    
    df = pd.DataFrame(data)
    
    # Compute SoH target
    df["soh"] = compute_soh(cell)
    
    # Ensure proper types
    for col in ["cycle_index", "discharge_capacity", "charge_capacity", 
                "discharge_energy", "charge_energy", "dc_internal_resistance",
                "temperature_maximum", "temperature_average", "temperature_minimum"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    
    return df


def load_and_extract_all():
    """Load cleaned cells and extract all features"""
    
    print("=" * 60)
    print("EXTRACTING ALL SUMMARY FEATURES")
    print("=" * 60)
    
    # Load cleaned cells
    print(f"\n[1] Loading cleaned cells from: {CLEANED_PKL}")
    with open(CLEANED_PKL, "rb") as f:
        cells: List[Dict] = pickle.load(f)
    print(f"  Loaded {len(cells)} cells")
    
    # Extract all cells
    print("\n[2] Extracting features from cells...")
    all_dfs = []
    
    for i, cell in enumerate(cells):
        df = extract_cell_data(cell)
        all_dfs.append(df)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells")
    
    # Concatenate all cells
    print("\n[3] Concatenating all cells...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Convert date_time_iso to numeric
    print("\n[4] Converting date_time_iso to numeric...")
    df_all["date_time_iso_numeric"] = df_all["date_time_iso"].apply(parse_date_time)
    df_all = df_all.drop(columns=["date_time_iso"])
    
    # Drop any rows with NaN in feature columns
    print("\n[5] Dropping rows with NaN in feature columns...")
    feature_cols = [c for c in SUMMARY_FEATURES if c != "date_time_iso"]
    before = len(df_all)
    df_all = df_all.dropna(subset=feature_cols + ["soh"])
    after = len(df_all)
    print(f"  Dropped {before - after} rows with NaN")
    
    # Add engineered features for completeness
    print("\n[6] Adding engineered features...")
    df_all = add_engineered_features(df_all)
    
    # Final statistics
    print("\n[7] Final dataset statistics:")
    print(f"  Shape: {df_all.shape}")
    print(f"  Cells: {df_all['barcode'].nunique()}")
    print(f"  Features: {df_all.columns.tolist()}")
    
    # Save
    print("\n[8] Saving datasets...")
    df_all.to_csv(OUTPUT_CSV, index=False)
    df_all.to_pickle(OUTPUT_PKL)
    
    print(f"\n  ✅ Saved: {OUTPUT_CSV}")
    print(f"  ✅ Saved: {OUTPUT_PKL}")
    
    # Show feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print("\nAvailable features for importance analysis:")
    feature_cols = [c for c in df_all.columns if c not in ["barcode", "protocol", "soh"]]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col}")
    
    return df_all


def add_engineered_features(df):
    """Add common engineered features for completeness"""
    df = df.copy()
    
    # Coulombic efficiency
    with np.errstate(divide="ignore", invalid="ignore"):
        coul_eff = df["charge_capacity"] / df["discharge_capacity"]
        coul_eff = np.where(df["discharge_capacity"] > 0.01, coul_eff, np.nan)
    df["coulombic_efficiency"] = np.clip(coul_eff, 0.80, 1.20)
    
    # Per-cell normalized features (using first 10 cycles)
    df["cap_rel"] = np.nan
    df["energy_rel"] = np.nan
    df["ir_rel"] = np.nan
    
    for barcode, group in df.groupby("barcode"):
        group_sorted = group.sort_values("cycle_index")
        early = group_sorted.iloc[:10]
        
        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        nom_ir = early["dc_internal_resistance"].mean()
        
        mask = df["barcode"] == barcode
        df.loc[mask, "cap_rel"] = (df.loc[mask, "charge_capacity"] - nom_cap) / nom_cap
        df.loc[mask, "energy_rel"] = (df.loc[mask, "charge_energy"] - nom_energy) / nom_energy
        df.loc[mask, "ir_rel"] = (df.loc[mask, "dc_internal_resistance"] - nom_ir) / nom_ir
    
    # Cycle position
    for barcode, group in df.groupby("barcode"):
        min_cycle = group["cycle_index"].min()
        max_cycle = group["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)
        mask = df["barcode"] == barcode
        df.loc[mask, "cycle_pos"] = (df.loc[mask, "cycle_index"] - min_cycle) / cyc_range
    
    return df


def show_summary_statistics(df):
    """Show summary statistics for all features"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    feature_cols = [c for c in df.columns if c not in ["barcode", "protocol", "soh"]]
    
    stats = []
    for col in feature_cols:
        if col in df.columns:
            stats.append({
                "Feature": col,
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "NaN%": df[col].isna().mean() * 100
            })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    df_all = load_and_extract_all()
    show_summary_statistics(df_all)
    
    print("\n" + "=" * 60)
    print("✅ ALL SUMMARY FEATURES EXTRACTED")
    print("=" * 60)
    print(f"\nUse this dataset for:")
    print("  1. Correlation analysis with SoH")
    print("  2. Feature redundancy analysis")
    print("  3. VIF analysis")
    print("  4. Full feature importance")
    print("\nDataset saved to:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_PKL}")

if __name__ == "__main__":
    main()