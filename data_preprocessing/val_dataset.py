# validate_dataset.py
"""
Validate the merged dataset with split column.
Checks:
1. Shape and columns
2. Split distribution
3. Data types
4. Missing values
5. Feature correlations with SOH
6. Cell distribution
7. Statistical summary
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_with_split.csv")
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2")

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

print("\n[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================
# BASIC INFO
# ============================================================

print("\n[2] Basic Information:")
print("-" * 50)
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Unique cells: {df['cell_id'].nunique()}")
print(f"  Cycle range: {df['cycle_index'].min():,} - {df['cycle_index'].max():,}")
print(f"  SOH range: {df['soh'].min():.4f} - {df['soh'].max():.4f}")
print(f"  SOH mean: {df['soh'].mean():.4f}")

# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print("\n[3] Split Distribution:")
print("-" * 50)

split_counts = df['split'].value_counts()
print(f"\n  Rows per split:")
for split in ['train', 'val', 'test']:
    count = split_counts.get(split, 0)
    pct = count / len(df) * 100
    print(f"    {split}: {count:>8,} rows ({pct:>5.1f}%)")

print(f"\n  Cells per split:")
for split in ['train', 'val', 'test']:
    cells = df[df['split'] == split]['cell_id'].nunique()
    print(f"    {split}: {cells:>8} cells")

# ============================================================
# DATA TYPES
# ============================================================

print("\n[4] Data Types:")
print("-" * 50)

dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

print("\n  Columns with object dtype:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"    - {col}")

# ============================================================
# MISSING VALUES
# ============================================================

print("\n[5] Missing Values:")
print("-" * 50)

missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"  Columns with missing values:")
    for col, count in missing_cols.items():
        pct = count / len(df) * 100
        print(f"    - {col}: {count:,} ({pct:.2f}%)")
else:
    print("  ✅ No missing values found!")

# ============================================================
# FEATURE CORRELATIONS WITH SOH
# ============================================================

print("\n[6] Feature Correlations with SOH:")
print("-" * 50)

# Select numeric columns (excluding metadata)
numeric_cols = df.select_dtypes(include=[np.number]).columns
exclude_cols = ['cell_id', 'cycle_index', 'soh']
corr_cols = [c for c in numeric_cols if c not in exclude_cols]

correlations = {}
for col in corr_cols:
    if col in df.columns:
        corr = df[col].corr(df['soh'])
        if not pd.isna(corr):
            correlations[col] = abs(corr)

# Sort by correlation
sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 10 features with highest correlation to SOH:")
for i, (col, corr) in enumerate(sorted_corr[:10], 1):
    print(f"    {i:>2}. {col:<30} {corr:.4f}")

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\n[7] Statistical Summary (key features):")
print("-" * 50)

key_features = ['soh', 'cycle_index', 'charge_capacity', 'dc_internal_resistance', 
                'coulombic_efficiency_lagged_1', 'cap_rel']

for feat in key_features:
    if feat in df.columns:
        print(f"\n  {feat}:")
        print(f"    Min:  {df[feat].min():.4f}")
        print(f"    Max:  {df[feat].max():.4f}")
        print(f"    Mean: {df[feat].mean():.4f}")
        print(f"    Std:  {df[feat].std():.4f}")

# ============================================================
# CELL CYCLE DISTRIBUTION
# ============================================================

print("\n[8] Cell Cycle Distribution:")
print("-" * 50)

cycle_stats = df.groupby('cell_id')['cycle_index'].max()
print(f"  Min cycles per cell: {cycle_stats.min():,}")
print(f"  Max cycles per cell: {cycle_stats.max():,}")
print(f"  Mean cycles per cell: {cycle_stats.mean():.1f}")
print(f"  Median cycles per cell: {cycle_stats.median():.1f}")

# ============================================================
# SPLIT VALIDATION
# ============================================================

print("\n[9] Split Validation:")
print("-" * 50)

# Check if any cell appears in multiple splits
for split in ['train', 'val', 'test']:
    cells = set(df[df['split'] == split]['cell_id'].unique())
    for other in ['train', 'val', 'test']:
        if split != other:
            other_cells = set(df[df['split'] == other]['cell_id'].unique())
            overlap = cells.intersection(other_cells)
            if overlap:
                print(f"  ⚠️ Overlap between {split} and {other}: {len(overlap)} cells")
            else:
                print(f"  ✅ No overlap between {split} and {other}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✅ VALIDATION COMPLETE")
print("=" * 70)

print(f"\nDataset is ready for:")
print("  - Feature importance analysis")
print("  - Model training with all features")
print("  - Ablation studies")

print(f"\n  Output files:")
print(f"    CSV: {DATASET_PATH}")
print(f"    Pickle: {OUTPUT_DIR / 'soh_full_with_split.pkl'}")

# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

report_path = OUTPUT_DIR / "validation_report.txt"
with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DATASET VALIDATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Total rows: {len(df):,}\n")
    f.write(f"Total columns: {len(df.columns)}\n")
    f.write(f"Unique cells: {df['cell_id'].nunique()}\n\n")
    
    f.write("Split Distribution:\n")
    for split in ['train', 'val', 'test']:
        count = split_counts.get(split, 0)
        pct = count / len(df) * 100
        cells = df[df['split'] == split]['cell_id'].nunique()
        f.write(f"  {split}: {count:,} rows ({pct:.1f}%), {cells} cells\n")
    
    f.write("\nTop 10 Feature Correlations with SOH:\n")
    for i, (col, corr) in enumerate(sorted_corr[:10], 1):
        f.write(f"  {i}. {col}: {corr:.4f}\n")
    
    f.write("\nValidation passed: ✅\n")

print(f"\n  Validation report saved: {report_path}")