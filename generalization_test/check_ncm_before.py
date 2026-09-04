# prepare_ncm_with_temperature_v2.py
#
# Fixes applied vs. the previous version, based on explore_ncm.py's own
# findings:
#   1. Column-coverage check BEFORE assuming voltage_range is computable -
#      explore_ncm.py showed Min/Max_Voltage/Current are NOT present in all
#      64 files. This script checks coverage per file and either restricts
#      to covered cells or drops the feature, based on actual coverage -
#      never silently crashes or silently fabricates the column.
#   2. Zero/near-zero capacity artifact rows are REMOVED (not clipped) -
#      your exploration showed a "100% fade to 0.0000 Ah" row that is almost
#      certainly a logging artifact, not real end-of-life data.
#   3. RPT-style capacity SPIKES (e.g. cycle 4 at 5.359 Ah vs ~2.6 Ah
#      neighbors in your sample) are detected via a robust rolling-median
#      filter and removed BEFORE the early-cycle nominal-capacity reference
#      is computed, so they can't contaminate cap_rel/energy_rel/soh.
#   4. No blind SOH clip(0.3, 1.1) as a band-aid - clipping is now a loose
#      safety net only (0, 1.2), applied AFTER artifact removal, not instead
#      of it.
#   5. Fixed deprecated fillna(method="ffill") -> .ffill().
#   6. Optional protocol-stratified split (parsed from filename, e.g.
#      "15C_0-100_0.5-1C") so train/val/test each see a mix of C-rates and
#      temperatures rather than risking an accidental single-protocol split.
#   7. Every filtering step prints exactly how many rows/cells were affected
#      and why - use these numbers directly in your Dataset section
#      ("N cycles across M cells excluded due to logging artifacts /
#      reference-performance-test cycles, identified via ...").

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
OUT_FILE = os.path.join(OUT_DIR, "ncm_with_temp_processed_v2.csv")

N_EARLY_CYCLES = 10
SPLIT_RATIOS = dict(train=0.7, val=0.15, test=0.15)
SPLIT_SEED = 42
STRATIFY_SPLIT_BY_PROTOCOL = True  # see Step 6 below

# Artifact-row thresholds (tune only if diagnostics below suggest otherwise -
# these are deliberately conservative defaults, not tuned for a metric)
MIN_VALID_CAPACITY_FRAC = 0.15   # rows below 15% of that cell's median charge_capacity -> dropped as artifacts
SPIKE_DEVIATION_FRAC = 0.35      # rows deviating >35% from a local rolling median -> flagged as RPT/spike, dropped
ROLLING_WINDOW = 7               # window (cycles) for the local median used in spike detection

BASE_FEAT_COLS = [
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "cycle_pos", "temperature_avg",
]
OPTIONAL_VOLTAGE_FEAT = "voltage_range"  # only included if coverage is sufficient (see Step 2)
VOLTAGE_COVERAGE_THRESHOLD = 0.80        # require >=80% of kept cells to have it, else drop the feature entirely


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: find file pairs + PARSE PROTOCOL from filename
# ─────────────────────────────────────────────────────────────────────────────

PROTOCOL_RE = re.compile(r"NMC_(\d+C)_0-100_([\d.]+-[\d.]+C)")

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
    """Extract a coarse protocol group (temperature + C-rate) for stratified
    splitting. Returns 'unknown' if the filename doesn't match the expected
    pattern - check this doesn't dominate your groups before relying on it."""
    m = PROTOCOL_RE.search(cell_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: load cycle_data with COLUMN-COVERAGE CHECK (fixes crash risk)
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: artifact-row filtering (NEW - the core fix)
# ─────────────────────────────────────────────────────────────────────────────

def remove_capacity_artifacts(g, cell_id, log):
    """Removes (a) near-zero/dead-logging rows and (b) RPT-style spikes,
    BEFORE any nominal-capacity reference or relative feature is computed.
    Order matters: spike detection uses a rolling median, so it must run on
    a series that isn't already corrupted by (a)."""
    g = g.sort_values("cycle_index").reset_index(drop=True)
    n0 = len(g)

    # (a) near-zero / dead rows
    median_cap = g["charge_capacity"].median()
    valid_mask = g["charge_capacity"] >= (MIN_VALID_CAPACITY_FRAC * median_cap)
    n_dropped_zero = (~valid_mask).sum()
    g = g[valid_mask].reset_index(drop=True)

    if len(g) < ROLLING_WINDOW + 1:
        log.append((cell_id, n0, n_dropped_zero, 0, len(g)))
        return g  # too short to do rolling-spike detection meaningfully

    # (b) RPT/spike detection via rolling median (robust to a single outlier)
    roll_median = g["charge_capacity"].rolling(ROLLING_WINDOW, center=True,
                                                min_periods=3).median()
    roll_median = roll_median.bfill().ffill()
    deviation = (g["charge_capacity"] - roll_median).abs() / (roll_median + 1e-9)
    spike_mask = deviation <= SPIKE_DEVIATION_FRAC
    n_dropped_spike = (~spike_mask).sum()
    g = g[spike_mask].reset_index(drop=True)

    log.append((cell_id, n0, n_dropped_zero, n_dropped_spike, len(g)))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: process one cell end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def process_cell(cell_id, cycle_path, timeseries_path, artifact_log):
    df, has_voltage = load_cycle_data(cycle_path)
    temp_df, temp_col_used = extract_temperature_per_cycle(timeseries_path)

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
# Step 5: derived features (soh, CE, relative features) - run AFTER cleaning
# ─────────────────────────────────────────────────────────────────────────────

def add_derived_features(df, use_voltage):
    out = []
    for cid, g in df.groupby("cell_id"):
        g = g.sort_values("cycle_index").copy()
        if len(g) < N_EARLY_CYCLES + 1:
            continue  # too few valid cycles left after cleaning to be usable

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
# Step 6: split - optionally stratified by protocol group
# ─────────────────────────────────────────────────────────────────────────────

def assign_cell_level_split(df, ratios, seed, stratify_by_protocol=True):
    rng = np.random.default_rng(seed)
    cell_protocol = df.groupby("cell_id")["protocol"].first()

    train_cells, val_cells, test_cells = set(), set(), set()

    if stratify_by_protocol:
        for proto, group in cell_protocol.groupby(cell_protocol):
            cells = sorted(group.index.tolist())
            rng.shuffle(cells)
            n = len(cells)
            n_train = max(1, int(round(ratios["train"] * n))) if n >= 3 else n
            n_val = max(0, int(round(ratios["val"] * n))) if n >= 3 else 0
            train_cells.update(cells[:n_train])
            val_cells.update(cells[n_train:n_train + n_val])
            test_cells.update(cells[n_train + n_val:])
    else:
        cells = sorted(cell_protocol.index.tolist())
        rng.shuffle(cells)
        n = len(cells)
        n_train = int(round(ratios["train"] * n))
        n_val = int(round(ratios["val"] * n))
        train_cells.update(cells[:n_train])
        val_cells.update(cells[n_train:n_train + n_val])
        test_cells.update(cells[n_train + n_val:])

    def label(cid):
        if cid in train_cells:
            return "train"
        if cid in val_cells:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["cell_id"].map(label)
    print(f"  Cells -> train:{len(train_cells)}  val:{len(val_cells)}  test:{len(test_cells)}"
          f"  (stratified={stratify_by_protocol})")

    protocol_by_split = df.groupby("split")["protocol"].value_counts().unstack(fill_value=0)
    print("  Protocol coverage per split (cell-cycle counts):")
    print(protocol_by_split.to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PREPARING NCM DATA (v2 - artifact-filtered)")
    print("=" * 60)

    print("\nFinding file pairs...")
    cells, cycle_files, time_files = find_file_pairs(RAW_DIR)

    print("\nProcessing cells (with artifact filtering)...")
    all_dfs = []
    voltage_coverage = {}
    artifact_log = []  # (cell_id, n0, n_dropped_zero, n_dropped_spike, n_final)

    for i, cell_id in enumerate(sorted(cells)):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(cells)} cells...")
        df_cell, has_voltage = process_cell(cell_id, cycle_files[cell_id], time_files[cell_id], artifact_log)
        voltage_coverage[cell_id] = has_voltage
        if len(df_cell) > N_EARLY_CYCLES:
            all_dfs.append(df_cell)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total rows after artifact filtering: {len(df):,}")
    print(f"  Cells retained: {df['cell_id'].nunique()} / {len(cells)}")

    # ─── artifact-filtering report ───
    print("\n" + "=" * 60)
    print("  ARTIFACT-FILTERING REPORT")
    print("=" * 60)
    log_df = pd.DataFrame(artifact_log, columns=["cell_id", "n0", "dropped_zero", "dropped_spike", "n_final"])
    print(f"  Total zero/dead rows dropped : {log_df['dropped_zero'].sum():,}")
    print(f"  Total RPT/spike rows dropped : {log_df['dropped_spike'].sum():,}")
    print(f"  Cells most affected (top 5):")
    print(log_df.assign(pct_dropped=lambda d: 100 * (d.dropped_zero + d.dropped_spike) / d.n0)
          .sort_values("pct_dropped", ascending=False).head(5).to_string(index=False))

    # ─── voltage coverage check ───
    print("\n" + "=" * 60)
    print("  VOLTAGE-COLUMN COVERAGE CHECK")
    print("=" * 60)
    n_with_voltage = sum(voltage_coverage.values())
    coverage_frac = n_with_voltage / len(voltage_coverage)
    print(f"  Cells with min/max voltage columns: {n_with_voltage}/{len(voltage_coverage)} "
          f"({coverage_frac:.1%})")
    use_voltage = coverage_frac >= VOLTAGE_COVERAGE_THRESHOLD
    if use_voltage:
        print(f"  -> Coverage >= {VOLTAGE_COVERAGE_THRESHOLD:.0%}: keeping voltage_range as a feature, "
              f"restricting to the {n_with_voltage} covered cells.")
        covered_ids = {cid for cid, has_v in voltage_coverage.items() if has_v}
        df = df[df["cell_id"].isin(covered_ids)].reset_index(drop=True)
    else:
        print(f"  -> Coverage < {VOLTAGE_COVERAGE_THRESHOLD:.0%}: DROPPING voltage_range feature "
              f"to retain all {len(voltage_coverage)} cells instead. Update BEM-SOH's feature "
              f"list accordingly (8 features, no voltage_range).")

    feat_cols = BASE_FEAT_COLS + ([OPTIONAL_VOLTAGE_FEAT] if use_voltage else [])

    print("\nAdding derived features (soh, CE, relative features)...")
    df = add_derived_features(df, use_voltage=use_voltage)

    print("\nAssigning cell-level split...")
    df = assign_cell_level_split(df, SPLIT_RATIOS, SPLIT_SEED, stratify_by_protocol=STRATIFY_SPLIT_BY_PROTOCOL)

    # ─── final sanity checks ───
    print("\n" + "=" * 60)
    print("  FINAL SANITY CHECKS")
    print("=" * 60)
    n_nan = df[feat_cols + ["soh"]].isna().sum().sum()
    print(f"  NaN count in feature+label columns: {n_nan} ({'OK' if n_nan == 0 else 'INVESTIGATE'})")
    print(f"  SOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
    print(f"  Temperature range: [{df['temperature_avg'].min():.2f}, {df['temperature_avg'].max():.2f}]")
    print(f"  Final features ({len(feat_cols)}): {feat_cols}")
    print(f"  Total rows: {len(df):,}  |  Cells: {df['cell_id'].nunique()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_cols = ["cell_id", "cycle_index", "split", "protocol", "soh"] + feat_cols
    df[save_cols].to_csv(OUT_FILE, index=False)
    print(f"\n  Saved -> {OUT_FILE}")
    print(f"  Reminder: set NCM_FEAT_COLS = {feat_cols} and input_dim = {len(feat_cols)} "
          f"in the training script to match this output.")


if __name__ == "__main__":
    main()