# merge_datasets_fixed.py
"""
Merge the original SOH dataset (with split column) with the full features dataset.
- Converts cell_id to string type for consistent merging
- Uses cell_id and cycle_index as merge keys
- Preserves the split column from the original dataset
"""

import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SOH_FULL_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv")
FULL_FEATURES_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_dataset_full_features.csv")
OUTPUT_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_merged.csv")
OUTPUT_PKL_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_merged.pkl")

# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("MERGING DATASETS (FIXED)")
print("=" * 60)

print("\n[1] Loading datasets...")

soh_full = pd.read_csv(SOH_FULL_PATH)
print(f"  Original SOH shape: {soh_full.shape}")
print(f"  Original SOH cell_id type: {soh_full['cell_id'].dtype}")
print(f"  Original SOH cell_id sample: {soh_full['cell_id'].iloc[0]}")

full_features = pd.read_csv(FULL_FEATURES_PATH)
print(f"  Full features shape: {full_features.shape}")
print(f"  Full features cell_id type: {full_features['cell_id'].dtype}")
print(f"  Full features cell_id sample: {full_features['cell_id'].iloc[0]}")

# ============================================================
# CONVERT CELL_ID TO STRING (BOTH DATASETS)
# ============================================================

print("\n[2] Converting cell_id to string type...")

# Convert both to string for consistent merging
soh_full['cell_id'] = soh_full['cell_id'].astype(str)
full_features['cell_id'] = full_features['cell_id'].astype(str)

print(f"  ✅ Both cell_id columns converted to string")
print(f"  Original SOH cell_id sample: {soh_full['cell_id'].iloc[0]}")
print(f"  Full features cell_id sample: {full_features['cell_id'].iloc[0]}")

# ============================================================
# VERIFY MATCHING
# ============================================================

print("\n[3] Verifying cell_id matching...")

soh_cells = set(soh_full['cell_id'].unique())
full_cells = set(full_features['cell_id'].unique())

print(f"  Cells in SOH: {len(soh_cells)}")
print(f"  Cells in full features: {len(full_cells)}")

overlap = soh_cells.intersection(full_cells)
print(f"  Overlap: {len(overlap)} cells")

if len(overlap) == 0:
    print("  ❌ No matching cell_ids found!")
    print(f"  Sample SOH cell_ids: {list(soh_cells)[:5]}")
    print(f"  Sample Full cell_ids: {list(full_cells)[:5]}")
    exit()

missing = soh_cells - full_cells
if missing:
    print(f"  ⚠️ Cells in SOH but not in full features: {len(missing)}")
else:
    print(f"  ✅ All cells match!")

# ============================================================
# MERGE DATASETS
# ============================================================

print("\n[4] Merging datasets...")

# Select columns from original SOH that we want to keep
soh_keep_cols = ['cell_id', 'cycle_index', 'split', 'barcode', 'channel']
soh_subset = soh_full[soh_keep_cols]

print(f"  SOH subset shape: {soh_subset.shape}")
print(f"  SOH subset columns: {soh_subset.columns.tolist()}")

# Merge: left join on cell_id and cycle_index
merged = pd.merge(
    full_features,
    soh_subset,
    on=['cell_id', 'cycle_index'],
    how='left'
)

print(f"\n  Merged shape: {merged.shape}")

# Check for any rows without split
missing_split = merged[merged['split'].isna()]
if len(missing_split) > 0:
    print(f"  ⚠️ {len(missing_split):,} rows have no split (shouldn't happen)")
else:
    print(f"  ✅ All rows have split column")

# ============================================================
# VERIFY SPLIT DISTRIBUTION
# ============================================================

print("\n[5] Verifying split distribution...")

print("\n  Split distribution in merged dataset:")
for split, count in merged['split'].value_counts().items():
    print(f"    {split}: {count:,} rows")

print(f"\n  Unique cells per split:")
for split in ['train', 'val', 'test']:
    cells = merged[merged['split'] == split]['cell_id'].nunique()
    print(f"    {split}: {cells} cells")

# ============================================================
# REORDER COLUMNS
# ============================================================

print("\n[6] Reordering columns...")

# Move important columns to the front
front_cols = ['cell_id', 'barcode', 'channel', 'split', 'cycle_index', 'soh']
other_cols = [col for col in merged.columns if col not in front_cols]
new_order = front_cols + other_cols

merged = merged[new_order]

print(f"  Columns reordered: {len(merged.columns)} columns")

# ============================================================
# SAVE MERGED DATASET
# ============================================================

print("\n[7] Saving merged dataset...")

merged.to_csv(OUTPUT_PATH, index=False)
print(f"  ✅ Saved CSV: {OUTPUT_PATH}")

merged.to_pickle(OUTPUT_PKL_PATH)
print(f"  ✅ Saved Pickle: {OUTPUT_PKL_PATH}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅ DATASETS MERGED SUCCESSFULLY!")
print("=" * 60)

print(f"\nOriginal SOH: {soh_full.shape[0]:,} rows, {soh_full.shape[1]} columns")
print(f"Full Features: {full_features.shape[0]:,} rows, {full_features.shape[1]} columns")
print(f"Merged: {merged.shape[0]:,} rows, {merged.shape[1]} columns")

print(f"\nAll features now available with split column!")
print(f"\nOutput files:")
print(f"  CSV: {OUTPUT_PATH}")
print(f"  Pickle: {OUTPUT_PKL_PATH}")

print("\nFeatures in merged dataset:")
for i, col in enumerate(merged.columns, 1):
    print(f"  {i:>2}. {col}")
print("=" * 60)