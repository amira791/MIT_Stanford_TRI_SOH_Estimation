# explore_ncm_processed.py
#
# Purpose: give you the concrete numbers needed to decide how to resolve
# the MIT (10-feature, has IR) vs SNL (9-feature, no IR, has voltage_range)
# schema mismatch, and to sanity-check the split/calibration coverage
# issues flagged in review. This script does NOT retrain or run the model -
# it only inspects data. Run it locally where the CSV/raw files live.
#
# Sections:
#   1. Schema diff vs the MIT feature list the checkpoint expects
#   2. Raw-file scan for ANY IR/resistance-adjacent columns SNL might have
#      under a different name (HPPC pulse tests, etc.) that the prepare
#      script didn't pick up
#   3. Per-feature distribution summary (SNL) - range/mean/std per split
#   4. Temperature-range overlap check against the MIT training range
#   5. Calibration-set (val) protocol coverage vs test coverage - flags
#      whether isotonic calibration is being fit on a representative subset
#   6. SOH distribution + tail coverage per split (matters for tail_weight
#      and PINW comparisons downstream)

import os
import glob
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Config - EDIT THESE PATHS
# ─────────────────────────────────────────────────────────────────────────────

SNL_PROCESSED_CSV = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\ncm_with_temp_processed_v2.csv"
SNL_RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NMC\SNL NMC"

# The exact feature list/order the MIT checkpoint was trained on
# (from train_soh_bem.py FEAT_COLS) - used only for the schema diff below.
MIT_FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]

# Fill in the MIT/Stanford/TRI training temperature range if you know it
# (check your soh_full.csv the same way this script checks the SNL one -
# see the standalone snippet printed at the bottom if you want to reuse
# this script on that file too). Leave as None to skip the overlap check.
MIT_TEMP_RANGE = None  # e.g. (20.0, 40.0)

# Any column-name fragment that might indicate resistance/impedance data
# lurking in the raw SNL files under a name the prepare script didn't
# already map (e.g. "Internal_Resistance", "DCIR", "HPPC", "Impedance").
RESISTANCE_KEYWORDS = ["resist", "dcir", "impedance", "hppc", "ir_", "_ir"]


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema diff
# ─────────────────────────────────────────────────────────────────────────────

def schema_diff(df):
    section("1. SCHEMA DIFF vs MIT checkpoint's expected FEAT_COLS")
    snl_cols = [c for c in df.columns if c not in
                ("cell_id", "cycle_index", "split", "protocol", "soh")]
    mit_set, snl_set = set(MIT_FEAT_COLS), set(snl_cols)

    print(f"  MIT FEAT_COLS ({len(MIT_FEAT_COLS)}): {MIT_FEAT_COLS}")
    print(f"  SNL feature cols ({len(snl_cols)}): {snl_cols}")
    print(f"\n  In MIT but NOT in SNL: {sorted(mit_set - snl_set)}")
    print(f"  In SNL but NOT in MIT: {sorted(snl_set - mit_set)}")

    if mit_set - snl_set:
        print("\n  -> These columns are MISSING from SNL. The checkpoint cannot run "
              "as-is against this file. See section 2 for whether IR is recoverable "
              "from the raw files at all.")
    if snl_set - mit_set:
        print("\n  -> These SNL columns have no slot in the trained model and would "
              "need to be dropped (or the model retrained) before inference.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Raw-file scan for hidden IR/resistance columns
# ─────────────────────────────────────────────────────────────────────────────

def scan_raw_for_resistance_columns(raw_dir):
    section("2. RAW SNL FILES - scan for any resistance/impedance-adjacent columns")
    if not os.path.isdir(raw_dir):
        print(f"  Raw dir not found: {raw_dir} (skipping - edit SNL_RAW_DIR)")
        return

    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    print(f"  Scanning {len(all_files)} files for header matches to {RESISTANCE_KEYWORDS} ...")

    hits = {}  # filename -> matching columns
    for f in all_files:
        try:
            header = pd.read_csv(f, nrows=0).columns.tolist()
        except Exception as e:
            print(f"    could not read header of {os.path.basename(f)}: {e}")
            continue
        matches = [c for c in header if any(k in c.lower() for k in RESISTANCE_KEYWORDS)]
        if matches:
            hits[os.path.basename(f)] = matches

    if not hits:
        print("  No resistance/impedance-like columns found in ANY raw file header. "
              "This is fairly strong evidence SNL's cycle_data/timeseries files simply "
              "don't carry an IR signal (SNL protocols weren't run with in-line HPPC "
              "pulses the way MIT/Stanford/TRI's were) -> option 1 from the review "
              "(reconstruct IR) is likely not viable, option 2 or 3 are the realistic paths.")
    else:
        print(f"  Found matches in {len(hits)} file(s):")
        for fname, cols in list(hits.items())[:15]:
            print(f"    {fname}: {cols}")
        if len(hits) > 15:
            print(f"    ... and {len(hits) - 15} more files")
        print("\n  -> IR-adjacent data DOES exist somewhere in the raw files. Worth "
              "checking whether it's per-cycle (usable) or only in a separate HPPC "
              "test file with its own cycling schedule (harder to align to cycle_index).")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-feature distribution summary
# ─────────────────────────────────────────────────────────────────────────────

def feature_distributions(df):
    section("3. PER-FEATURE DISTRIBUTION SUMMARY (by split)")
    feat_cols = [c for c in df.columns if c not in
                 ("cell_id", "cycle_index", "split", "protocol", "soh")]
    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        print(f"\n  -- split = {split}  (n={len(sub):,}) --")
        desc = sub[feat_cols].describe().T[["mean", "std", "min", "max"]]
        print(desc.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temperature range overlap vs MIT training range
# ─────────────────────────────────────────────────────────────────────────────

def temperature_overlap(df, mit_range):
    section("4. TEMPERATURE RANGE OVERLAP CHECK (SNL vs MIT training)")
    snl_min, snl_max = df["temperature_avg"].min(), df["temperature_avg"].max()
    print(f"  SNL temperature range: [{snl_min:.2f}, {snl_max:.2f}] C")

    if mit_range is None:
        print("  MIT_TEMP_RANGE not set - fill it in at the top of this script "
              "(pull min/max of 'temperature_avg' from soh_full.csv for the train split) "
              "to get an automatic overlap verdict. For now, compare the SNL range above "
              "against your MIT dataset's known cycling temperature by hand.")
        return

    mit_min, mit_max = mit_range
    print(f"  MIT training temperature range: [{mit_min:.2f}, {mit_max:.2f}] C")
    overlap_lo, overlap_hi = max(snl_min, mit_min), min(snl_max, mit_max)
    if overlap_lo > overlap_hi:
        print("  -> NO OVERLAP. SNL temperatures fall entirely outside the MIT training "
              "range. Any generalization-test result will be confounded by extrapolation, "
              "not just chemistry/protocol shift - flag this explicitly if you report results.")
    else:
        frac_snl_covered = ((df["temperature_avg"] >= mit_min) & (df["temperature_avg"] <= mit_max)).mean()
        print(f"  -> Overlap region: [{overlap_lo:.2f}, {overlap_hi:.2f}] C")
        print(f"  -> {frac_snl_covered:.1%} of SNL rows fall inside the MIT training range; "
              f"the rest are outside it (extrapolation for the model, not interpolation).")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Calibration (val) protocol coverage vs test coverage
# ─────────────────────────────────────────────────────────────────────────────

def calibration_coverage(df):
    section("5. VAL (calibration) PROTOCOL COVERAGE vs TEST PROTOCOL COVERAGE")
    val_protocols = set(df[df.split == "val"]["protocol"].unique())
    test_protocols = set(df[df.split == "test"]["protocol"].unique())
    train_protocols = set(df[df.split == "train"]["protocol"].unique())

    print(f"  Protocols in val:   {sorted(val_protocols)}")
    print(f"  Protocols in test:  {sorted(test_protocols)}")
    missing = test_protocols - val_protocols
    print(f"\n  Test protocols with NO representation in val: {sorted(missing)}")
    if missing:
        n_affected = df[(df.split == "test") & (df.protocol.isin(missing))].shape[0]
        n_test_total = (df.split == "test").sum()
        print(f"  -> {n_affected:,}/{n_test_total:,} test rows "
              f"({100*n_affected/n_test_total:.1f}%) belong to protocols the isotonic "
              f"calibrator never saw during fitting. Calibrated PICP/PINW on test is "
              f"likely to be less reliable for exactly these rows. Consider a different "
              f"SPLIT_SEED, or evaluating calibration quality separately for covered vs "
              f"uncovered protocols instead of one pooled number.")
    else:
        print("  -> Every test protocol has at least some val representation. Good.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SOH distribution + tail coverage per split
# ─────────────────────────────────────────────────────────────────────────────

def soh_distribution(df):
    section("6. SOH DISTRIBUTION + TAIL COVERAGE PER SPLIT")
    for split in ["train", "val", "test"]:
        sub = df[df.split == split]["soh"]
        n_tail = (sub < 0.90).sum()
        print(f"  {split:5s}: n={len(sub):,}  "
              f"mean={sub.mean():.4f}  min={sub.min():.4f}  max={sub.max():.4f}  "
              f"| SOH<0.90 rows: {n_tail} ({100*n_tail/len(sub):.1f}%)")
    print("\n  If any split has ~0 tail rows, MAE/PINW reported for that split says "
        "nothing about low-SOH (end-of-life) performance, which is usually the region "
        "that matters most for a SOH estimator.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  EXPLORING PREPROCESSED SNL DATASET")
    print("=" * 70)
    print(f"  File: {SNL_PROCESSED_CSV}")

    df = pd.read_csv(SNL_PROCESSED_CSV)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols, "
          f"{df['cell_id'].nunique()} cells")

    schema_diff(df)
    scan_raw_for_resistance_columns(SNL_RAW_DIR)
    feature_distributions(df)
    temperature_overlap(df, MIT_TEMP_RANGE)
    calibration_coverage(df)
    soh_distribution(df)

    section("DONE")
    print("  Use sections 1-2 to decide the feature-schema path (reconstruct IR / "
          "retrain SNL-specific / drop IR from MIT too).")
    print("  Use sections 4-6 to decide whether the current split/seed is trustworthy "
          "enough to report, or needs a reshuffle first.")


if __name__ == "__main__":
    main()