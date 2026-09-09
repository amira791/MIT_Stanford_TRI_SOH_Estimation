# final_dataset_statistics.py
"""
Generate Final Dataset Statistics
--------------------------------
This script loads the final merged dataset and computes all statistics
reported in Section 2.5 of the documentation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv")
# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("FINAL DATASET STATISTICS")
print("=" * 70)

print("\n[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================
# 2.5 FINAL DATASET STATISTICS
# ============================================================

print("\n[2.5] Final Dataset Statistics:")
print("-" * 50)

# Valid cells
valid_cells = df['cell_id'].nunique()
print(f"  Valid cells: {valid_cells}")

# Total cycles
total_cycles = len(df)
print(f"  Total cycles: {total_cycles:,}")

# Cycle range
cycle_min = df['cycle_index'].min()
cycle_max = df['cycle_index'].max()
print(f"  Cycle range: {cycle_min:.0f} - {cycle_max:.0f}")

# SOH statistics
soh_min = df['soh'].min()
soh_max = df['soh'].max()
soh_mean = df['soh'].mean()
print(f"  SOH range: {soh_min:.3f} - {soh_max:.3f}")
print(f"  SOH mean: {soh_mean:.3f}")

# Number of features
n_features = len(df.columns)
print(f"  Features: {n_features} (raw + engineered)")

# ============================================================
# SPLIT STATISTICS
# ============================================================

print("\n[Split Distribution]")
print("-" * 50)

for split in ['train', 'val', 'test']:
    split_df = df[df['split'] == split]
    rows = len(split_df)
    cells = split_df['cell_id'].nunique()
    pct = rows / total_cycles * 100
    print(f"  {split.capitalize()}: {cells} cells ({rows:,} rows, {pct:.1f}%)")

# ============================================================
# DETAILED FEATURE STATISTICS
# ============================================================

print("\n[Feature Statistics]")
print("-" * 50)

# Select key features
key_features = [
    'cycle_index',
    'discharge_capacity',
    'charge_capacity',
    'discharge_energy',
    'charge_energy',
    'dc_internal_resistance',
    'temperature_maximum',
    'temperature_average',
    'temperature_minimum',
    'coulombic_efficiency_lagged_1',
    'coulombic_efficiency_lagged_2',
    'cap_rel',
    'energy_rel',
    'ir_rel',
    'cycle_pos',
    'soh'
]

for feat in key_features:
    if feat in df.columns:
        stats = df[feat].describe()
        print(f"\n  {feat}:")
        print(f"    Count: {stats['count']:,.0f}")
        print(f"    Mean:  {stats['mean']:.4f}")
        print(f"    Std:   {stats['std']:.4f}")
        print(f"    Min:   {stats['min']:.4f}")
        print(f"    25%:   {stats['25%']:.4f}")
        print(f"    50%:   {stats['50%']:.4f}")
        print(f"    75%:   {stats['75%']:.4f}")
        print(f"    Max:   {stats['max']:.4f}")

# ============================================================
# EXPORT STATISTICS
# ============================================================

print("\n[Exporting Statistics]")
print("-" * 50)

# Create summary DataFrame
summary_data = {
    'Parameter': [
        'Valid cells',
        'Total cycles',
        'Cycle range',
        'SOH range',
        'SOH mean',
        'Features',
        'Train cells',
        'Train rows',
        'Val cells',
        'Val rows',
        'Test cells',
        'Test rows'
    ],
    'Value': [
        valid_cells,
        f"{total_cycles:,}",
        f"{cycle_min:.0f} - {cycle_max:.0f}",
        f"{soh_min:.3f} - {soh_max:.3f}",
        f"{soh_mean:.3f}",
        n_features,
        df[df['split'] == 'train']['cell_id'].nunique(),
        len(df[df['split'] == 'train']),
        df[df['split'] == 'val']['cell_id'].nunique(),
        len(df[df['split'] == 'val']),
        df[df['split'] == 'test']['cell_id'].nunique(),
        len(df[df['split'] == 'test'])
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save to CSV
output_path = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\dataset_statistics.csv")
summary_df.to_csv(output_path, index=False)
print(f"\n  Statistics saved to: {output_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✅ FINAL DATASET STATISTICS COMPLETE")
print("=" * 70)

print(f"\nDataset ready for CNN-Mamba-UQ training.")
print(f"  - {valid_cells} cells")
print(f"  - {total_cycles:,} cycles")
print(f"  - {n_features} features")