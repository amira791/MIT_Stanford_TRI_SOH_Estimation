# prepare_ncm_kfold_clean.py
# Clean K-Fold by cell

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NMC\SNL NMC"
OUT_DIR = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results"
OUT_FILE = os.path.join(OUT_DIR, "ncm_kfold_clean_processed.csv")

N_FOLDS = 5
K_FOLD_SEED = 42
N_EARLY_CYCLES = 10
RESTRICT_TO_FULL_DOD = True
MIN_VALID_CAPACITY_FRAC = 0.15
SPIKE_DEVIATION_FRAC = 0.35
ROLLING_WINDOW = 7
VOLTAGE_COVERAGE_THRESHOLD = 0.80

BASE_FEAT_COLS = [
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "cycle_pos", "temperature_avg",
]
OPTIONAL_VOLTAGE_FEAT = "voltage_range"

# ─────────────────────────────────────────────────────────────────────────────
# File finding
# ─────────────────────────────────────────────────────────────────────────────

PROTOCOL_RE = re.compile(r"NMC_(\d+C)_(\d+-\d+)_([\d.]+-[\d.]+C)")

def find_file_pairs(raw_dir):
    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    cycle_data_files, timeseries_files = {}, {}
    for f in all_files:
        filename = os.path.basename(f)
        if "_cycle_data.csv" in filename:
            cell_id = filename.replace("_cycle_data.csv", "")
            cycle_data_files[cell_id] = f
        elif "_timeseries.csv" in filename:
            cell_id = filename.replace("_timeseries.csv", "")
            timeseries_files[cell_id] = f
    overlap = set(cycle_data_files.keys()) & set(timeseries_files.keys())
    print(f"  Found {len(overlap)} cells with both cycle_data and timeseries")
    return overlap, cycle_data_files, timeseries_files

def parse_protocol(cell_id):
    m = PROTOCOL_RE.search(cell_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return "unknown"

def parse_dod_window(cell_id):
    m = PROTOCOL_RE.search(cell_id)
    return m.group(2) if m else None

# ─────────────────────────────────────────────────────────────────────────────
# Load and clean
# ─────────────────────────────────────────────────────────────────────────────

RENAME_MAP = {
    "Cycle_Index": "cycle_index",
    "Charge_Capacity (Ah)": "charge_capacity",
    "Discharge_Capacity (Ah)": "discharge_capacity",
    "Charge_Energy (Wh)": "charge_energy",
    "Discharge_Energy (Wh)": "discharge_energy",
    "Min_Voltage (V)": "min_voltage",
    "Max_Voltage (V)": "max_voltage",
    "Min_Current (A)": "min_current",
    "Max_Current (A)": "max_current",
    "Test_Time (s)": "test_time",
}
REQUIRED_ALWAYS = ["cycle_index", "charge_capacity", "discharge_capacity",
                    "charge_energy", "discharge_energy"]

def load_cycle_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns=RENAME_MAP)
    missing_required = [c for c in REQUIRED_ALWAYS if c not in df.columns]
    if missing_required:
        raise KeyError(f"{file_path}: missing required columns {missing_required}")
    has_voltage = ("min_voltage" in df.columns) and ("max_voltage" in df.columns)
    return df, has_voltage

def extract_temperature_per_cycle(file_path):
    df = pd.read_csv(file_path)
    temp_col = None
    if "Cell_Temperature (C)" in df.columns:
        temp_col = "Cell_Temperature (C)"
    elif "Environment_Temperature (C)" in df.columns:
        temp_col = "Environment_Temperature (C)"
    if temp_col is None:
        return None, None
    temp_per_cycle = df.groupby("Cycle_Index")[temp_col].mean().reset_index()
    temp_per_cycle.columns = ["cycle_index", "temperature_avg"]
    return temp_per_cycle, temp_col

def remove_capacity_artifacts(g, cell_id, log):
    g = g.sort_values("cycle_index").reset_index(drop=True)
    n0 = len(g)

    median_charge = g["charge_capacity"].median()
    median_discharge = g["discharge_capacity"].median()
    valid_mask = (g["charge_capacity"] >= (MIN_VALID_CAPACITY_FRAC * median_charge)) & \
                 (g["discharge_capacity"] >= (MIN_VALID_CAPACITY_FRAC * median_discharge))
    n_dropped_zero = (~valid_mask).sum()
    g = g[valid_mask].reset_index(drop=True)

    if len(g) < ROLLING_WINDOW + 1:
        log.append((cell_id, n0, n_dropped_zero, 0, len(g)))
        return g

    def spike_mask_for(col):
        roll_median = g[col].rolling(ROLLING_WINDOW, center=True, min_periods=3).median()
        roll_median = roll_median.bfill().ffill()
        deviation = (g[col] - roll_median).abs() / (roll_median + 1e-9)
        return deviation <= SPIKE_DEVIATION_FRAC

    keep_mask = spike_mask_for("charge_capacity") & spike_mask_for("discharge_capacity")
    n_dropped_spike = (~keep_mask).sum()
    g = g[keep_mask].reset_index(drop=True)

    log.append((cell_id, n0, n_dropped_zero, n_dropped_spike, len(g)))
    return g

def process_cell(cell_id, cycle_path, timeseries_path, artifact_log):
    df, has_voltage = load_cycle_data(cycle_path)
    temp_df, _ = extract_temperature_per_cycle(timeseries_path)

    if temp_df is not None:
        df = df.merge(temp_df, on="cycle_index", how="left")
        df["temperature_avg"] = df["temperature_avg"].ffill().bfill()
        df["temperature_avg"] = df["temperature_avg"].fillna(25.0)
    else:
        df["temperature_avg"] = 25.0

    df["cell_id"] = cell_id
    df["protocol"] = parse_protocol(cell_id)

    df = remove_capacity_artifacts(df, cell_id, artifact_log)
    return df, has_voltage

# ─────────────────────────────────────────────────────────────────────────────
# Derived features
# ─────────────────────────────────────────────────────────────────────────────

def add_derived_features(df, use_voltage):
    out = []
    for cid, g in df.groupby("cell_id"):
        g = g.sort_values("cycle_index").copy()
        if len(g) < N_EARLY_CYCLES + 1:
            continue

        early = g.iloc[:N_EARLY_CYCLES]
        nominal_capacity = early["discharge_capacity"].mean()
        g["soh"] = (g["discharge_capacity"] / (nominal_capacity + 1e-9)).clip(0.0, 1.2)

        g["coulombic_efficiency"] = (g["discharge_capacity"] / (g["charge_capacity"] + 1e-9)).clip(0.5, 1.05)
        g["coulombic_efficiency_lagged_1"] = g["coulombic_efficiency"].shift(1).bfill()
        g["coulombic_efficiency_lagged_2"] = g["coulombic_efficiency"].shift(2).bfill()

        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        min_cycle, max_cycle = g["cycle_index"].min(), g["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)

        g["cap_rel"] = (g["charge_capacity"] - nom_cap) / (nom_cap + 1e-9)
        g["energy_rel"] = (g["charge_energy"] - nom_energy) / (nom_energy + 1e-9)
        g["cycle_pos"] = (g["cycle_index"] - min_cycle) / cyc_range

        if use_voltage:
            g["voltage_range"] = g["max_voltage"] - g["min_voltage"]

        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# K-Fold Splits (SIMPLIFIED & ROBUST)
# ─────────────────────────────────────────────────────────────────────────────

def assign_kfold_splits(df, n_folds=5, seed=42):
    """Assigns K-Fold by cell splits - SIMPLIFIED VERSION."""
    
    # Get unique cells
    cell_info = df[["cell_id"]].drop_duplicates()
    cells = cell_info["cell_id"].tolist()
    
    print(f"\n  Assigning K-Fold by cell (n_folds={n_folds})...")
    print(f"  Total cells: {len(cells)}")
    
    # Shuffle cells
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(cells))
    shuffled_cells = [cells[i] for i in shuffled_idx]
    
    # Create folds
    fold_size = len(shuffled_cells) // n_folds
    fold_assignments = {}
    
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < n_folds - 1 else len(shuffled_cells)
        
        test_cells = shuffled_cells[start:end]
        remaining = [c for c in shuffled_cells if c not in test_cells]
        
        val_split_idx = int(0.8 * len(remaining))
        val_cells = remaining[val_split_idx:]
        train_cells = remaining[:val_split_idx]
        
        fold_assignments[fold_idx] = {
            "train": set(train_cells),
            "val": set(val_cells),
            "test": set(test_cells)
        }
        print(f"    Fold {fold_idx+1}: train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)}")
    
    # Assign fold and split to dataframe
    df = df.copy()
    df["fold"] = -1
    df["fold_split"] = "none"
    
    for fold_idx, assignments in fold_assignments.items():
        for split_name, cell_set in assignments.items():
            mask = df["cell_id"].isin(cell_set)
            df.loc[mask, "fold"] = fold_idx
            df.loc[mask, "fold_split"] = split_name
    
    # Verify assignment worked
    print(f"\n  Verification after assignment:")
    for fold_idx in range(n_folds):
        count = (df["fold"] == fold_idx).sum()
        print(f"    Fold {fold_idx+1}: {count:,} rows")
    
    return df, fold_assignments

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PREPARING NCM DATA (K-Fold by Cell) - CLEAN")
    print("=" * 60)
    print(f"  K-Fold: {N_FOLDS} folds")

    print("\nFinding file pairs...")
    cells, cycle_files, time_files = find_file_pairs(RAW_DIR)

    print("\nProcessing cells (with artifact filtering)...")
    all_dfs = []
    voltage_coverage = {}
    artifact_log = []

    for i, cell_id in enumerate(sorted(cells)):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells...")
        df_cell, has_voltage = process_cell(cell_id, cycle_files[cell_id], time_files[cell_id], artifact_log)
        voltage_coverage[cell_id] = has_voltage
        if len(df_cell) > N_EARLY_CYCLES:
            all_dfs.append(df_cell)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total rows after artifact filtering: {len(df):,}")

    # ─── Artifact report ───
    print("\n" + "=" * 60)
    print("  ARTIFACT-FILTERING REPORT")
    print("=" * 60)
    log_df = pd.DataFrame(artifact_log, columns=["cell_id", "n0", "dropped_zero", "dropped_spike", "n_final"])
    print(f"  Total zero/dead rows dropped : {log_df['dropped_zero'].sum():,}")
    print(f"  Total RPT/spike rows dropped : {log_df['dropped_spike'].sum():,}")

    # ─── Voltage coverage ───
    print("\n" + "=" * 60)
    print("  VOLTAGE-COLUMN COVERAGE CHECK")
    print("=" * 60)
    n_with_voltage = sum(voltage_coverage.values())
    coverage_frac = n_with_voltage / len(voltage_coverage)
    print(f"  Cells with min/max voltage columns: {n_with_voltage}/{len(voltage_coverage)} ({coverage_frac:.1%})")
    use_voltage = coverage_frac >= VOLTAGE_COVERAGE_THRESHOLD
    if use_voltage:
        print(f"  -> Keeping voltage_range as a feature")
        covered_ids = {cid for cid, has_v in voltage_coverage.items() if has_v}
        df = df[df["cell_id"].isin(covered_ids)].reset_index(drop=True)

    feat_cols = BASE_FEAT_COLS + ([OPTIONAL_VOLTAGE_FEAT] if use_voltage else [])

    # ─── DOD filter ───
    if RESTRICT_TO_FULL_DOD:
        df["dod_window"] = df["cell_id"].apply(parse_dod_window)
        n_cells_before = df["cell_id"].nunique()
        n_rows_before = len(df)
        kept_mask = df["dod_window"] == "0-100"
        df = df[kept_mask].reset_index(drop=True)
        print("\n" + "=" * 60)
        print("  DOD-WINDOW FILTER")
        print("=" * 60)
        print(f"  Cells excluded: {n_cells_before - df['cell_id'].nunique()} / {n_cells_before}")
        print(f"  Remaining: {len(df):,} rows, {df['cell_id'].nunique()} cells (full 0-100% DOD)")

    print("\nAdding derived features...")
    df = add_derived_features(df, use_voltage=use_voltage)

    print("\nAssigning K-Fold splits...")
    df, fold_assignments = assign_kfold_splits(df, n_folds=N_FOLDS, seed=K_FOLD_SEED)

    # ─── Final checks ───
    print("\n" + "=" * 60)
    print("  FINAL SANITY CHECKS")
    print("=" * 60)
    n_nan = df[feat_cols + ["soh"]].isna().sum().sum()
    print(f"  NaN count: {n_nan} ({'OK' if n_nan == 0 else 'INVESTIGATE'})")
    print(f"  SOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
    print(f"  Temperature range: [{df['temperature_avg'].min():.2f}, {df['temperature_avg'].max():.2f}]")
    print(f"  Final features ({len(feat_cols)}): {feat_cols}")
    print(f"  Total rows: {len(df):,}  |  Cells: {df['cell_id'].nunique()}")

    # ─── Save ───
    os.makedirs(OUT_DIR, exist_ok=True)
    save_cols = ["cell_id", "cycle_index", "fold", "fold_split", "protocol", "soh"] + feat_cols
    df[save_cols].to_csv(OUT_FILE, index=False)
    print(f"\n  Saved -> {OUT_FILE}")

    print("\n  K-Fold Summary:")
    for fold_idx in range(N_FOLDS):
        fold_df = df[df["fold"] == fold_idx]
        split_counts = fold_df["fold_split"].value_counts()
        print(f"    Fold {fold_idx+1}: train={split_counts.get('train', 0):,}, "
              f"val={split_counts.get('val', 0):,}, test={split_counts.get('test', 0):,}")

if __name__ == "__main__":
    main()