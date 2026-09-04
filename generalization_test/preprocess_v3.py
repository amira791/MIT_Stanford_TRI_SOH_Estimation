# prepare_ncm_with_temperature_v3.py
#
# v3 changes vs v2 (motivated by explore_ncm_processed.py findings on the
# v2 output: voltage_range had a max of ~8.84 V against a mean of ~2.20 V
# and std of ~0.08 V - roughly an 80-sigma outlier that survived unfiltered):
#
#   8. NEW: voltage-channel spike filtering. remove_capacity_artifacts()
#      previously only ever inspected charge_capacity/discharge_capacity.
#      A sensor glitch on the voltage channel is independent of whether
#      that row's capacity reading looks fine, so it needs its own check,
#      not a side effect of the capacity check. We now compute a
#      provisional voltage_range = max_voltage - min_voltage at the SAME
#      pipeline stage as the capacity artifact check (i.e. before the real
#      voltage_range is derived in add_derived_features, and before any
#      nominal reference is computed), run it through the same rolling-
#      median spike detector, and drop rows that fail it - using a tighter
#      deviation threshold than capacity, since voltage_range naturally
#      varies far less across a full-DOD cycling protocol.
#
# Everything else (all v2 fixes: column-coverage check, zero/near-zero
# capacity artifact removal, RPT-spike removal, loose safety-net clip,
# ffill fix, protocol-stratified split with per-group val/test guarantee,
# and full diagnostic printing) is unchanged.

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
OUT_FILE = os.path.join(OUT_DIR, "ncm_with_temp_processed_v3.csv")

N_EARLY_CYCLES = 10
SPLIT_RATIOS = dict(train=0.7, val=0.15, test=0.15)
SPLIT_SEED = 42
RESTRICT_TO_FULL_DOD = True  # See discussion: partial-DOD cells (20-80%,
    # 40-60%) report capacity over a reduced SOC window, which is NOT
    # directly comparable to a full-range SOH reference without additional
    # normalization. Keeping this True restricts the study to 0-100% DOD
    # cells only - the scientifically defensible default. Set False only if
    # you've implemented and validated a DOD-window normalization for SOH.
STRATIFY_SPLIT_BY_PROTOCOL = True  # see Step 6 below

# Artifact-row thresholds (tune only if diagnostics below suggest otherwise -
# these are deliberately conservative defaults, not tuned for a metric)
MIN_VALID_CAPACITY_FRAC = 0.15   # rows below 15% of that cell's median charge_capacity -> dropped as artifacts
SPIKE_DEVIATION_FRAC = 0.35      # rows deviating >35% from a local rolling median (capacity) -> flagged as RPT/spike, dropped
VOLTAGE_SPIKE_DEVIATION_FRAC = 0.25  # same idea, applied to voltage_range. Tighter
    # than capacity's 0.35 because voltage_range varies much less across a
    # protocol (full-DOD NMC typically sits in a narrow band); if legitimate
    # temperature-driven voltage_range variation starts getting caught by
    # this, loosen it - check the filtering report's voltage-spike counts
    # per cell first.
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
    """Extract a coarse protocol group (temperature + DOD window + C-rate)
    for stratified splitting. Handles any DOD range (e.g. '0-100', '20-80'),
    not just full-depth cycling. Returns 'unknown' if the filename doesn't
    match - check the printed protocol table for a large 'unknown' bucket,
    which signals this regex needs adjusting to your actual filenames."""
    m = PROTOCOL_RE.search(cell_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return "unknown"


def parse_dod_window(cell_id):
    """Returns the DOD window string (e.g. '0-100', '40-60') or None if
    unparseable. Used to filter to full-DOD-only cells (see
    RESTRICT_TO_FULL_DOD) - partial-DOD cells report capacity over a
    reduced SOC window and are not directly SOH-comparable without
    normalization this pipeline does not implement."""
    m = PROTOCOL_RE.search(cell_id)
    return m.group(2) if m else None


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
# Step 3: artifact-row filtering (capacity AND voltage)
# ─────────────────────────────────────────────────────────────────────────────

def remove_capacity_artifacts(g, cell_id, log, has_voltage=False):
    """Removes (a) near-zero/dead-logging rows, (b) RPT-style capacity
    spikes, and (c) voltage-sensor spikes - ALL before any nominal-capacity
    reference or relative feature is computed. Checks capacity columns
    independently of the voltage columns - a sensor glitch on one channel
    doesn't imply anything about the other, so each gets its own criterion
    (this mirrors the existing charge/discharge-independence fix: a row can
    fail on any ONE of the checked columns and still be dropped).

    (c) is new in v3: v2's filter only ever inspected charge_capacity/
    discharge_capacity, so voltage-channel glitches (e.g. a spurious
    max_voltage reading) passed straight through into the final
    voltage_range feature undetected. We compute a PROVISIONAL
    voltage_range here - before the real one is derived later in
    add_derived_features - purely so it can be spike-filtered at the same
    pipeline stage as capacity, using a rolling median local to each cell's
    own cycle history (not a global threshold).

    Order matters: spike detection uses a rolling median, so it must run on
    a series that isn't already corrupted by (a).
    """
    g = g.sort_values("cycle_index").reset_index(drop=True)
    n0 = len(g)

    # (a) near-zero / dead rows - check both columns, drop if EITHER fails
    median_charge = g["charge_capacity"].median()
    median_discharge = g["discharge_capacity"].median()
    valid_mask = (g["charge_capacity"] >= (MIN_VALID_CAPACITY_FRAC * median_charge)) & \
                 (g["discharge_capacity"] >= (MIN_VALID_CAPACITY_FRAC * median_discharge))
    n_dropped_zero = (~valid_mask).sum()
    g = g[valid_mask].reset_index(drop=True)

    if len(g) < ROLLING_WINDOW + 1:
        log.append((cell_id, n0, n_dropped_zero, 0, 0, len(g)))
        return g  # too short to do rolling-spike detection meaningfully

    # provisional voltage_range, used only for spike detection at this stage
    has_voltage_cols = has_voltage and "min_voltage" in g.columns and "max_voltage" in g.columns
    if has_voltage_cols:
        g["_voltage_range_tmp"] = g["max_voltage"] - g["min_voltage"]

    # (b)/(c) rolling-median spike detection - drop if ANY checked column
    # deviates beyond ITS threshold
    def spike_mask_for(col, thr):
        roll_median = g[col].rolling(ROLLING_WINDOW, center=True, min_periods=3).median()
        roll_median = roll_median.bfill().ffill()
        deviation = (g[col] - roll_median).abs() / (roll_median + 1e-9)
        return deviation <= thr

    cap_keep_mask = spike_mask_for("charge_capacity", SPIKE_DEVIATION_FRAC) & \
                    spike_mask_for("discharge_capacity", SPIKE_DEVIATION_FRAC)
    n_dropped_cap_spike = (~cap_keep_mask).sum()

    if has_voltage_cols:
        volt_keep_mask = spike_mask_for("_voltage_range_tmp", VOLTAGE_SPIKE_DEVIATION_FRAC)
        n_dropped_volt_spike = (~volt_keep_mask).sum()
        keep_mask = cap_keep_mask & volt_keep_mask
    else:
        n_dropped_volt_spike = 0
        keep_mask = cap_keep_mask

    g = g[keep_mask].reset_index(drop=True)
    if "_voltage_range_tmp" in g.columns:
        g = g.drop(columns=["_voltage_range_tmp"])

    log.append((cell_id, n0, n_dropped_zero, n_dropped_cap_spike,
                n_dropped_volt_spike, len(g)))
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

    df = remove_capacity_artifacts(df, cell_id, artifact_log, has_voltage=has_voltage)
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

def _allocate_group(cells, ratios, rng):
    """Allocates one protocol group's cells to train/val/test, guaranteeing
    representation in val/test where the group is large enough to support
    it, rather than silently dumping small groups entirely into train
    (the bug in the previous version - every named protocol had 0 cells in
    val/test because the old code required n>=3 to split at all).

    n==1 -> train only (nothing to split)
    n==2 -> 1 train, 1 test (too small to also carve out val)
    n>=3 -> at least 1 cell guaranteed to val AND test, remainder by ratio
    """
    cells = list(cells)
    rng.shuffle(cells)
    n = len(cells)

    if n == 1:
        return cells, [], []
    if n == 2:
        return cells[:1], [], cells[1:]

    n_test = max(1, round(ratios["test"] * n))
    n_val = max(1, round(ratios["val"] * n))
    n_train = n - n_test - n_val
    if n_train < 1:
        # extremely small group after guaranteeing 1 each to val/test -
        # shrink val/test back down rather than leave train empty
        n_train = 1
        n_val = max(0, n_val - 1) if (n_val + n_test) > (n - 1) else n_val
        n_test = n - n_train - n_val

    return cells[:n_train], cells[n_train:n_train + n_val], cells[n_train + n_val:]


def assign_cell_level_split(df, ratios, seed, stratify_by_protocol=True):
    rng = np.random.default_rng(seed)
    cell_protocol = df.groupby("cell_id")["protocol"].first()

    train_cells, val_cells, test_cells = set(), set(), set()

    if stratify_by_protocol:
        for proto, group in cell_protocol.groupby(cell_protocol):
            tr, va, te = _allocate_group(sorted(group.index.tolist()), ratios, rng)
            train_cells.update(tr)
            val_cells.update(va)
            test_cells.update(te)
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
    print(f"  NOTE: with small per-protocol cell counts, exact 70/15/15 is not always "
          f"achievable - each group is guaranteed >=1 cell in val AND test where n>=3, "
          f"rather than being dumped entirely into train. Groups with only 2 cells will "
          f"STILL have zero val representation under this scheme (1 train, 1 test) - if "
          f"you rely on val for calibration (isotonic fitting), consider k-fold "
          f"calibration instead of a fixed val split; a per-protocol allocation rule alone "
          f"cannot fix that for 2-cell groups.")

    protocol_by_split = df.groupby("split")["protocol"].value_counts().unstack(fill_value=0)
    print("  Protocol coverage per split (cell-cycle counts):")
    print(protocol_by_split.to_string())

    # Row-count share per split - cell-count-balanced allocation can still
    # produce row-count-imbalanced splits when cells vary a lot in cycle
    # life (this is what produced a 43.6%-of-all-rows test set previously).
    # Make this impossible to miss.
    row_counts = df["split"].value_counts()
    row_pct = (100 * row_counts / row_counts.sum()).round(1)
    print(f"\n  Row-count share per split: "
          f"train={row_pct.get('train',0)}%  val={row_pct.get('val',0)}%  test={row_pct.get('test',0)}%")
    print(f"  (target was train={100*ratios['train']:.0f}% val={100*ratios['val']:.0f}% "
          f"test={100*ratios['test']:.0f}% - cell-count-balanced allocation does not "
          f"guarantee row-count balance when cells vary greatly in cycle life)")
    if row_pct.get("test", 0) > 25 or row_pct.get("val", 0) > 25:
        print("  WARNING: test or val row-share is much larger than the target ratio - "
              "likely caused by one or two long-cycle-life cells landing in that split. "
              "Consider changing SPLIT_SEED and re-checking, or manually reassigning "
              "the specific oversized cell(s) shown in the protocol table above.")

    n_unknown = (cell_protocol == "unknown").sum()
    if n_unknown > 0:
        print(f"\n  WARNING: {n_unknown}/{len(cell_protocol)} cells fell into 'unknown' "
              f"protocol (PROTOCOL_RE did not match their filename) - inspect these "
              f"filenames and extend PROTOCOL_RE if this is a large fraction.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PREPARING NCM DATA (v3 - artifact-filtered incl. voltage spikes)")
    print("=" * 60)

    print("\nFinding file pairs...")
    cells, cycle_files, time_files = find_file_pairs(RAW_DIR)

    print("\nProcessing cells (with artifact filtering)...")
    all_dfs = []
    voltage_coverage = {}
    artifact_log = []  # (cell_id, n0, n_dropped_zero, n_dropped_cap_spike, n_dropped_volt_spike, n_final)

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
    log_df = pd.DataFrame(artifact_log, columns=["cell_id", "n0", "dropped_zero",
                                                   "dropped_cap_spike", "dropped_volt_spike", "n_final"])
    print(f"  Total zero/dead rows dropped         : {log_df['dropped_zero'].sum():,}")
    print(f"  Total RPT/capacity-spike rows dropped : {log_df['dropped_cap_spike'].sum():,}")
    print(f"  Total voltage-spike rows dropped      : {log_df['dropped_volt_spike'].sum():,}")
    print(f"  Cells most affected (top 5):")
    print(log_df.assign(pct_dropped=lambda d: 100 * (d.dropped_zero + d.dropped_cap_spike + d.dropped_volt_spike) / d.n0)
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

    # ─── DOD-window filter (fixes the critical SOH-comparability issue) ───
    if RESTRICT_TO_FULL_DOD:
        df["dod_window"] = df["cell_id"].apply(parse_dod_window)
        n_cells_before = df["cell_id"].nunique()
        n_rows_before = len(df)
        kept_mask = df["dod_window"] == "0-100"
        excluded_protocols = sorted(df.loc[~kept_mask, "protocol"].unique())
        df = df[kept_mask].reset_index(drop=True)
        print("\n" + "=" * 60)
        print("  DOD-WINDOW FILTER (RESTRICT_TO_FULL_DOD=True)")
        print("=" * 60)
        print(f"  Cells excluded (partial-DOD protocols): "
              f"{n_cells_before - df['cell_id'].nunique()} / {n_cells_before}")
        print(f"  Rows excluded: {n_rows_before - len(df):,} / {n_rows_before:,} "
              f"({100*(n_rows_before-len(df))/n_rows_before:.1f}%)")
        print(f"  Excluded protocol groups: {excluded_protocols}")
        print(f"  Remaining: {len(df):,} rows, {df['cell_id'].nunique()} cells "
              f"(full 0-100% DOD only)")
        if df["cell_id"].nunique() < 8:
            print("  WARNING: very few cells remain - check PROTOCOL_RE is matching "
                  "your 0-100 DOD filenames correctly before proceeding.")

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
    n_at_floor = (df["soh"] <= 0.001).sum()
    n_at_ceiling = (df["soh"] >= 1.199).sum()
    print(f"  Rows at clip floor (<=0.001): {n_at_floor}  |  at clip ceiling (>=1.199): {n_at_ceiling}")
    if n_at_floor > 0 or n_at_ceiling > 0:
        print("  NOTE: nonzero counts here mean some rows are hitting the safety-net clip "
              "bounds directly - inspect these specific rows/cells before trusting them; "
              "a few near-ceiling rows from early-life capacity rise is plausible and "
              "normal, but a floor-hit should not happen after artifact filtering.")
    print(f"  Temperature range: [{df['temperature_avg'].min():.2f}, {df['temperature_avg'].max():.2f}]")
    if use_voltage:
        print(f"  voltage_range: min={df['voltage_range'].min():.4f}  "
              f"max={df['voltage_range'].max():.4f}  mean={df['voltage_range'].mean():.4f}  "
              f"std={df['voltage_range'].std():.4f}")
        vr_z = (df["voltage_range"] - df["voltage_range"].mean()) / (df["voltage_range"].std() + 1e-9)
        n_extreme = (vr_z.abs() > 10).sum()
        print(f"  voltage_range rows with |z| > 10: {n_extreme} "
              f"({'OK' if n_extreme == 0 else 'INVESTIGATE - spike filter may need a tighter threshold'})")
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