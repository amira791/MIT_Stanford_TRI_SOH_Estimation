# explore_snl_ncm_data.py
# Comprehensive exploration of the processed SNL NCM dataset
# v3 output analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the processed data
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\ncm_with_temp_processed_v3.csv"

print("=" * 60)
print("  SNL NCM DATASET EXPLORATION (v3 Processed)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic statistics
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  BASIC STATISTICS")
print("=" * 60)

print(f"\nCells: {df['cell_id'].nunique()}")
print(f"Protocols: {df['protocol'].nunique()}")
print(f"Cycles total: {len(df):,}")

print("\nFeature statistics:")
print(df[["charge_capacity", "charge_energy", "temperature_avg", "voltage_range", "soh"]].describe())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Check SOH distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SOH DISTRIBUTION")
print("=" * 60)

print(f"\nSOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
print(f"SOH mean: {df['soh'].mean():.4f}")
print(f"SOH std: {df['soh'].std():.4f}")

# SOH bins
bins = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
df['soh_bin'] = pd.cut(df['soh'], bins=bins)
print("\nSOH distribution per bin:")
print(df['soh_bin'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Check by protocol
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PROTOCOL GROUPS")
print("=" * 60)

protocol_stats = df.groupby("protocol").agg({
    "cell_id": "nunique",
    "cycle_index": "count",
    "soh": ["min", "max", "mean"],
    "temperature_avg": ["min", "max", "mean"]
}).round(4)
protocol_stats.columns = ["cells", "cycles", "soh_min", "soh_max", "soh_mean", "temp_min", "temp_max", "temp_mean"]
print(protocol_stats)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check by cell
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PER-CELL SUMMARY")
print("=" * 60)

cell_stats = df.groupby("cell_id").agg({
    "cycle_index": "count",
    "protocol": "first",
    "soh": ["min", "max", "mean", "std"],
    "temperature_avg": ["mean", "std"],
    "voltage_range": ["mean", "std"]
}).round(4)
cell_stats.columns = ["cycles", "protocol", "soh_min", "soh_max", "soh_mean", "soh_std", 
                       "temp_mean", "temp_std", "vr_mean", "vr_std"]
print(cell_stats)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Check split distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SPLIT DISTRIBUTION")
print("=" * 60)

split_stats = df.groupby("split").agg({
    "cell_id": "nunique",
    "cycle_index": "count",
    "soh": ["min", "max", "mean"],
}).round(4)
split_stats.columns = ["cells", "cycles", "soh_min", "soh_max", "soh_mean"]
print(split_stats)

# Row percentage
row_pct = df["split"].value_counts(normalize=True) * 100
print(f"\nRow percentage:\n{row_pct.round(1)}")

# Cell count per split
cell_pct = df.groupby("split")["cell_id"].nunique() / df["cell_id"].nunique() * 100
print(f"\nCell percentage:\n{cell_pct.round(1)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Check feature correlations
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  FEATURE CORRELATIONS WITH SOH")
print("=" * 60)

feature_cols = ["charge_capacity", "charge_energy", "temperature_avg", "voltage_range", 
                "cap_rel", "energy_rel", "cycle_pos", "soh"]
corr_matrix = df[feature_cols].corr()
soh_corr = corr_matrix["soh"].sort_values(ascending=False)
print("\nCorrelation with SOH:")
print(soh_corr)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Check temperature distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  TEMPERATURE DISTRIBUTION")
print("=" * 60)

print(f"\nTemperature range: [{df['temperature_avg'].min():.2f}, {df['temperature_avg'].max():.2f}]")
print(f"Temperature mean: {df['temperature_avg'].mean():.2f}")
print(f"Temperature std: {df['temperature_avg'].std():.2f}")

# Temperature by protocol
temp_by_protocol = df.groupby("protocol")["temperature_avg"].agg(["min", "max", "mean", "std"]).round(2)
print("\nTemperature by protocol:")
print(temp_by_protocol)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Check voltage_range distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  VOLTAGE_RANGE DISTRIBUTION")
print("=" * 60)

print(f"\nvoltage_range range: [{df['voltage_range'].min():.4f}, {df['voltage_range'].max():.4f}]")
print(f"voltage_range mean: {df['voltage_range'].mean():.4f}")
print(f"voltage_range std: {df['voltage_range'].std():.4f}")

# Check for outliers
vr_z = (df["voltage_range"] - df["voltage_range"].mean()) / (df["voltage_range"].std() + 1e-9)
n_extreme = (vr_z.abs() > 5).sum()
print(f"\nRows with |z| > 5: {n_extreme} ({100*n_extreme/len(df):.3f}%)")

if n_extreme > 0:
    print("\nExtreme voltage_range rows:")
    print(df[vr_z.abs() > 5][["cell_id", "cycle_index", "voltage_range", "protocol", "temperature_avg"]].head(10))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Check SOH vs Cycles (degradation patterns)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  DEGRADATION PATTERNS")
print("=" * 60)

# For each cell, check if SOH decreases with cycles
def check_monotonic_decrease(group):
    group = group.sort_values("cycle_index")
    soh = group["soh"].values
    # Check if SOH generally decreases
    decreasing = np.mean(np.diff(soh)) < -0.001  # at least 0.1% decrease per cycle on average
    return decreasing

monotonic = df.groupby("cell_id").apply(check_monotonic_decrease)
print(f"\nCells with decreasing SOH trend: {monotonic.sum()}/{len(monotonic)}")

if monotonic.sum() < len(monotonic):
    non_monotonic = monotonic[~monotonic].index.tolist()
    print(f"Cells with non-monotonic SOH: {non_monotonic}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Recommendations
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  RECOMMENDATIONS")
print("=" * 60)

# print("""
# Based on the exploration:

# 1. DATA QUALITY:
#    - 22 cells, 14,626 rows (full DOD only)
#    - SOH range: 0.60 – 1.04 (healthy degradation range)
#    - Temperature: 15.9 – 41.5°C (diverse)
#    - voltage_range is clean (mean=2.20, std=0.003)

# 2. SPLIT ISSUE:
#    - Row-count split: train=47.3%, val=12.5%, test=40.1%
#    - Cell-count split: train=50%, val=14%, test=36%
#    - Test set is too large → consider K-Fold by cell

# 3. RECOMMENDED NEXT STEPS:
#    Option A: Use K-Fold by cell (5 folds) for robust results
#    Option B: Re-split with different seed to balance row counts
#    Option C: Use the current split and note the imbalance as a limitation

# 4. FEATURES (9):
#    - charge_capacity, charge_energy, coulombic_efficiency_lagged_1,
#      coulombic_efficiency_lagged_2, cap_rel, energy_rel, cycle_pos,
#      temperature_avg, voltage_range

# 5. TRAINING DECISION:
#    - Use K-Fold by cell to get mean ± std results
#    - This will close the row-imbalance issue
#    - Report results as: MAE = X.XX% ± X.XX%
# """)

print("=" * 60)
print("  EXPLORATION COMPLETE")
print("=" * 60)