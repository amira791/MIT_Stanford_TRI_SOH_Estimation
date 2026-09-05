# fix_nca_soh_artifacts.py
# Simple script to remove SOH < 0.30 artifacts from NCA processed data

import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\nca_with_temp_processed_v3.csv"
OUTPUT_FILE = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\nca_with_temp_processed_v3_fixed.csv"

SOH_MIN_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  FIXING NCA SOH ARTIFACTS")
print("=" * 60)

# Load data
df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded: {len(df):,} rows, {df['cell_id'].nunique()} cells")
print(f"Current SOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")

# Check how many rows below threshold
n_below = (df["soh"] < SOH_MIN_THRESHOLD).sum()
print(f"\nRows with SOH < {SOH_MIN_THRESHOLD}: {n_below}")

if n_below > 0:
    # Show affected cells
    affected_cells = df[df["soh"] < SOH_MIN_THRESHOLD]["cell_id"].unique()
    print(f"Affected cells: {affected_cells.tolist()}")
    
    # Drop rows below threshold
    df_fixed = df[df["soh"] >= SOH_MIN_THRESHOLD].copy()
    print(f"\nRows after fix: {len(df_fixed):,}")
    print(f"Cells after fix: {df_fixed['cell_id'].nunique()}")
    print(f"New SOH range: [{df_fixed['soh'].min():.4f}, {df_fixed['soh'].max():.4f}]")
    
    # Save
    df_fixed.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
else:
    print(f"\n✅ No rows below SOH < {SOH_MIN_THRESHOLD}. Data is clean.")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved copy to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)