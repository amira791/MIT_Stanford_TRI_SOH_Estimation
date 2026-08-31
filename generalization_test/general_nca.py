# prepare_nca_generalization_data.py
# 
# Processes SNL NCA cycle_data files into the same schema
# as load_soh_data() / SequenceDataset for cross-chemistry generalization.

import os
import glob
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NCA\SNL NCA"
OUT_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results"
OUT_FILE = os.path.join(OUT_DIR, "nca_processed.csv")

N_EARLY_CYCLES = 10
SPLIT_RATIOS = dict(train=0.7, val=0.15, test=0.15)
SPLIT_SEED = 42

# 8 features (same as NCM, temperature not available)
NCA_FEAT_COLS = [
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    "cap_rel",
    "energy_rel",
    "voltage_range",
    "cycle_pos",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load ONLY cycle_data files
# ─────────────────────────────────────────────────────────────────────────────

def load_cycle_files(raw_dir):
    """Load only *_cycle_data.csv files (ignore timeseries)."""
    
    pattern = os.path.join(raw_dir, "*_cycle_data.csv")
    paths = glob.glob(pattern)
    paths = sorted(paths)
    
    print(f"  Found {len(paths)} cycle_data files")
    
    if not paths:
        raise FileNotFoundError(f"No cycle_data files found in {raw_dir}")
    
    frames = []
    for i, p in enumerate(paths):
        try:
            filename = os.path.basename(p)
            cell_id = filename.replace('_cycle_data.csv', '')
            
            df = pd.read_csv(p, low_memory=False)
            df["cell_id"] = cell_id
            frames.append(df)
            
            if (i + 1) % 10 == 0:
                print(f"    Loaded {i+1}/{len(paths)} files...")
                
        except Exception as e:
            print(f"    Warning: Could not load {p}: {e}")
            continue
    
    if not frames:
        raise ValueError("No files could be loaded successfully.")
    
    raw = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(frames)} cell files -> {raw.shape[0]:,} rows")
    return raw

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Standardize columns
# ─────────────────────────────────────────────────────────────────────────────

def standardize_columns(df):
    """Rename columns to standard names."""
    
    mapping = {
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
    
    rename_dict = {}
    for old, new in mapping.items():
        if old in df.columns:
            rename_dict[old] = new
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"  Renamed {len(rename_dict)} columns")
    
    # NCA cycle_data does NOT have temperature
    # Use 25°C as default (room temperature)
    print("  No temperature column in cycle_data. Using 25°C.")
    df['temperature_avg'] = 25.0
    
    needed = ['cell_id', 'cycle_index', 'charge_capacity', 'discharge_capacity',
              'charge_energy', 'discharge_energy', 'min_voltage', 'max_voltage',
              'min_current', 'max_current', 'temperature_avg']
    
    available = [col for col in needed if col in df.columns]
    df = df[available].copy()
    
    print(f"  Kept {len(available)} columns")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: SOH label
# ─────────────────────────────────────────────────────────────────────────────

def add_soh_label(df):
    """SOH based on discharge capacity."""
    
    out = []
    
    for cid, g in df.groupby("cell_id"):
        g = g.sort_values("cycle_index").copy()
        
        early = g.iloc[:N_EARLY_CYCLES]
        nominal = early["discharge_capacity"].mean()
        
        if pd.isna(nominal) or nominal <= 0:
            nominal = g["discharge_capacity"].median()
        
        g["soh"] = g["discharge_capacity"] / (nominal + 1e-9)
        g["soh"] = g["soh"].clip(0.3, 1.1)
        
        out.append(g)
    
    return pd.concat(out, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Coulombic efficiency
# ─────────────────────────────────────────────────────────────────────────────

def add_coulombic_efficiency(df):
    """Compute coulombic efficiency and lags."""
    
    df = df.copy()
    df["coulombic_efficiency"] = df["discharge_capacity"] / (df["charge_capacity"] + 1e-9)
    df["coulombic_efficiency"] = df["coulombic_efficiency"].clip(0.85, 1.05)
    
    for lag in [1, 2]:
        col_name = f"coulombic_efficiency_lagged_{lag}"
        df[col_name] = df.groupby("cell_id")["coulombic_efficiency"].shift(lag)
        df[col_name] = df[col_name].fillna(df["coulombic_efficiency"])
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Relative features
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_and_proxy_features(df):
    """cap_rel, energy_rel, cycle_pos, voltage_range."""
    
    df = df.copy()
    cap_rel_list, en_rel_list, cyc_pos_list, vr_list = [], [], [], []
    
    for cid, g in df.groupby("cell_id"):
        g = g.sort_values("cycle_index")
        early = g.iloc[:N_EARLY_CYCLES]
        
        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        min_cycle = g["cycle_index"].min()
        max_cycle = g["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)
        
        if pd.isna(nom_cap) or nom_cap <= 0:
            nom_cap = g["charge_capacity"].median()
        if pd.isna(nom_energy) or nom_energy <= 0:
            nom_energy = g["charge_energy"].median()
        
        cap_rel_list.append((g["charge_capacity"] - nom_cap) / (nom_cap + 1e-9))
        en_rel_list.append((g["charge_energy"] - nom_energy) / (nom_energy + 1e-9))
        cyc_pos_list.append((g["cycle_index"] - min_cycle) / cyc_range)
        vr_list.append(g["max_voltage"] - g["min_voltage"])
    
    df["cap_rel"] = pd.concat(cap_rel_list)
    df["energy_rel"] = pd.concat(en_rel_list)
    df["cycle_pos"] = pd.concat(cyc_pos_list)
    df["voltage_range"] = pd.concat(vr_list)
    
    for col in ["cap_rel", "energy_rel", "cycle_pos", "voltage_range"]:
        df[col] = df[col].fillna(0)
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Train/val/test split
# ─────────────────────────────────────────────────────────────────────────────

def assign_cell_level_split(df, ratios, seed):
    """Assign cell-level split."""
    
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
    print("  PREPARING NCA GENERALIZATION DATA")
    print("  (Cycle Data Only)")
    print("=" * 60)
    
    print("\nLoading NCA cycle_data files...")
    raw = load_cycle_files(RAW_DIR)
    
    print("\nStandardizing columns...")
    df = standardize_columns(raw)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    print("\nComputing SOH label...")
    df = add_soh_label(df)
    print(f"  SOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
    
    print("\nComputing coulombic efficiency...")
    df = add_coulombic_efficiency(df)
    
    print("\nComputing relative features...")
    df = add_relative_and_proxy_features(df)
    
    print("\nAssigning cell-level split...")
    df = assign_cell_level_split(df, SPLIT_RATIOS, SPLIT_SEED)
    
    # ─── Sanity checks ───
    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)
    
    n_nan = df[NCA_FEAT_COLS + ["soh"]].isna().sum().sum()
    print(f"  NaN count: {n_nan:,} ({'OK' if n_nan == 0 else 'WARNING'})")
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Cells: {df['cell_id'].nunique()}")
    
    # ─── Save ───
    print("\n" + "=" * 60)
    print("  SAVING")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    save_cols = ["cell_id", "cycle_index", "split", "soh"] + NCA_FEAT_COLS
    df[save_cols].to_csv(OUT_FILE, index=False)
    
    print(f"  Saved -> {OUT_FILE}")
    print(f"  Shape: {df.shape[0]:,} rows, {df['cell_id'].nunique()} cells")
    
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)