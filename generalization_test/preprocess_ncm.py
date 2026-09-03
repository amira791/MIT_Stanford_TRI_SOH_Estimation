# prepare_ncm_with_temperature.py
# 
# Merges cycle_data with timeseries to extract temperature per cycle
# Produces 9 features for NCM (including temperature_avg)

import os
import glob
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NMC\SNL NMC"
OUT_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results"
OUT_FILE = os.path.join(OUT_DIR, "ncm_with_temp_processed.csv")

N_EARLY_CYCLES = 10
SPLIT_RATIOS = dict(train=0.7, val=0.15, test=0.15)
SPLIT_SEED = 42

# ─── 9 FEATURES (now with temperature!) ───
NCM_FEAT_COLS = [
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    "cap_rel",
    "energy_rel",
    "voltage_range",
    "cycle_pos",
    "temperature_avg",           # ← NEW! From timeseries
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Find all files
# ─────────────────────────────────────────────────────────────────────────────

def find_file_pairs(raw_dir):
    """Find matching cycle_data and timeseries file pairs."""
    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    cycle_data_files = {}
    timeseries_files = {}
    
    for f in all_files:
        filename = os.path.basename(f)
        if '_cycle_data.csv' in filename:
            cell_id = filename.replace('_cycle_data.csv', '')
            cycle_data_files[cell_id] = f
        elif '_timeseries.csv' in filename:
            cell_id = filename.replace('_timeseries.csv', '')
            timeseries_files[cell_id] = f
    
    # Find overlap
    overlap = set(cycle_data_files.keys()) & set(timeseries_files.keys())
    print(f"  Found {len(overlap)} cells with both cycle_data and timeseries")
    
    return overlap, cycle_data_files, timeseries_files

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load cycle_data
# ─────────────────────────────────────────────────────────────────────────────

def load_cycle_data(file_path):
    """Load and standardize cycle_data."""
    df = pd.read_csv(file_path)
    
    # Rename columns
    rename_map = {
        'Cycle_Index': 'cycle_index',
        'Charge_Capacity (Ah)': 'charge_capacity',
        'Discharge_Capacity (Ah)': 'discharge_capacity',
        'Charge_Energy (Wh)': 'charge_energy',
        'Discharge_Energy (Wh)': 'discharge_energy',
        'Min_Voltage (V)': 'min_voltage',
        'Max_Voltage (V)': 'max_voltage',
        'Min_Current (A)': 'min_current',
        'Max_Current (A)': 'max_current',
        'Test_Time (s)': 'test_time',
    }
    
    df = df.rename(columns=rename_map)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Extract temperature from timeseries
# ─────────────────────────────────────────────────────────────────────────────

def extract_temperature_per_cycle(file_path):
    """
    Extract average cell temperature per cycle from timeseries.
    Returns DataFrame with cycle_index and temperature_avg.
    """
    df = pd.read_csv(file_path)
    
    # Get temperature column
    temp_col = None
    if 'Cell_Temperature (C)' in df.columns:
        temp_col = 'Cell_Temperature (C)'
    elif 'Environment_Temperature (C)' in df.columns:
        temp_col = 'Environment_Temperature (C)'
    else:
        print("Warning: No temperature column found in timeseries")
        return None
    
    # Group by cycle and compute average temperature
    temp_per_cycle = df.groupby('Cycle_Index')[temp_col].mean().reset_index()
    temp_per_cycle.columns = ['cycle_index', 'temperature_avg']
    
    return temp_per_cycle

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Merge and process
# ─────────────────────────────────────────────────────────────────────────────

def process_cell(cell_id, cycle_path, timeseries_path):
    """Process a single cell: merge cycle_data with timeseries temperature."""
    
    # Load cycle_data
    df = load_cycle_data(cycle_path)
    
    # Extract temperature from timeseries
    temp_df = extract_temperature_per_cycle(timeseries_path)
    
    if temp_df is not None:
        # Merge temperature with cycle_data
        df = df.merge(temp_df, on='cycle_index', how='left')
        # Fill any missing temperature with forward fill
        df['temperature_avg'] = df['temperature_avg'].fillna(method='ffill')
        # If still missing, use 25°C
        df['temperature_avg'] = df['temperature_avg'].fillna(25.0)
    else:
        df['temperature_avg'] = 25.0
    
    # Add cell_id
    df['cell_id'] = cell_id
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Add derived features
# ─────────────────────────────────────────────────────────────────────────────

def add_derived_features(df):
    """Add SOH, coulombic efficiency, relative features."""
    
    df = df.copy()
    out = []
    
    for cid, g in df.groupby("cell_id"):
        g = g.sort_values("cycle_index").copy()
        
        # ─── SOH from discharge capacity ───
        early = g.iloc[:N_EARLY_CYCLES]
        nominal_capacity = early["discharge_capacity"].mean()
        g["soh"] = g["discharge_capacity"] / (nominal_capacity + 1e-9)
        g["soh"] = g["soh"].clip(0.3, 1.1)
        
        # ─── Coulombic Efficiency ───
        g["coulombic_efficiency"] = g["discharge_capacity"] / (g["charge_capacity"] + 1e-9)
        g["coulombic_efficiency"] = g["coulombic_efficiency"].clip(0.85, 1.05)
        
        # ─── Coulombic Efficiency Lags ───
        g["coulombic_efficiency_lagged_1"] = g["coulombic_efficiency"].shift(1).fillna(g["coulombic_efficiency"])
        g["coulombic_efficiency_lagged_2"] = g["coulombic_efficiency"].shift(2).fillna(g["coulombic_efficiency"])
        
        # ─── Relative Features ───
        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        min_cycle = g["cycle_index"].min()
        max_cycle = g["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)
        
        g["cap_rel"] = (g["charge_capacity"] - nom_cap) / (nom_cap + 1e-9)
        g["energy_rel"] = (g["charge_energy"] - nom_energy) / (nom_energy + 1e-9)
        g["cycle_pos"] = (g["cycle_index"] - min_cycle) / cyc_range
        g["voltage_range"] = g["max_voltage"] - g["min_voltage"]
        
        # ─── Fill any NaN ───
        for col in ["cap_rel", "energy_rel", "cycle_pos", "voltage_range", "temperature_avg"]:
            g[col] = g[col].fillna(0)
        
        out.append(g)
    
    return pd.concat(out, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Split
# ─────────────────────────────────────────────────────────────────────────────

def assign_cell_level_split(df, ratios, seed):
    cells = sorted(df["cell_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    
    n = len(cells)
    n_train = int(round(ratios["train"] * n))
    n_val = int(round(ratios["val"] * n))
    
    train_cells = set(cells[:n_train])
    val_cells = set(cells[n_train:n_train + n_val])
    test_cells = set(cells[n_train + n_val:])
    
    def label(cid):
        if cid in train_cells:
            return "train"
        if cid in val_cells:
            return "val"
        return "test"
    
    df = df.copy()
    df["split"] = df["cell_id"].map(label)
    print(f"  Cells -> train:{len(train_cells)}  val:{len(val_cells)}  test:{len(test_cells)}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PREPARING NCM DATA WITH TEMPERATURE")
    print("  (9 features: including temperature_avg)")
    print("=" * 60)
    
    print("\nFinding file pairs...")
    cells, cycle_files, time_files = find_file_pairs(RAW_DIR)
    
    print("\nProcessing cells...")
    all_dfs = []
    for i, cell_id in enumerate(sorted(cells)):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells...")
        df = process_cell(cell_id, cycle_files[cell_id], time_files[cell_id])
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    
    print("\nAdding derived features...")
    df = add_derived_features(df)
    
    print("\nAssigning cell-level split...")
    df = assign_cell_level_split(df, SPLIT_RATIOS, SPLIT_SEED)
    
    # ─── Sanity checks ───
    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)
    
    n_nan = df[NCM_FEAT_COLS + ["soh"]].isna().sum().sum()
    print(f"  NaN count: {n_nan:,} ({'OK' if n_nan == 0 else 'WARNING'})")
    
    print(f"  SOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
    print(f"  Temperature range: [{df['temperature_avg'].min():.2f}, {df['temperature_avg'].max():.2f}]")
    print(f"  Total rows: {len(df):,}")
    print(f"  Cells: {df['cell_id'].nunique()}")
    
    # ─── Save ───
    print("\n" + "=" * 60)
    print("  SAVING")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    save_cols = ["cell_id", "cycle_index", "split", "soh"] + NCM_FEAT_COLS
    df[save_cols].to_csv(OUT_FILE, index=False)
    
    print(f"  Saved -> {OUT_FILE}")
    print(f"  Features: {len(NCM_FEAT_COLS)} (including temperature_avg)")
    print(f"  Shape: {df.shape[0]:,} rows, {df['cell_id'].nunique()} cells")
    
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"\n  Now you can train BEM-SOH with {len(NCM_FEAT_COLS)} features:")
    print(f"  {NCM_FEAT_COLS}")