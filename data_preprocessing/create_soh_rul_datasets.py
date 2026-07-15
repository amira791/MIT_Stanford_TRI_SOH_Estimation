# add_split_to_full_features_v2.py
"""
Add split column to full features dataset by creating a mapping.
- The full features dataset has cell_id as string (barcode-like)
- The original SOH dataset has cell_id as numeric and barcode as string
- We create a mapping from barcode to split
- Then we add the split column to full features using its cell_id (which is the barcode)
"""

import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SOH_FULL_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv")
FULL_FEATURES_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_dataset_full_features.csv")
OUTPUT_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_with_split.csv")
OUTPUT_PKL_PATH = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_with_split.pkl")

# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("ADDING SPLIT COLUMN USING CELL_ID MAPPING")
print("=" * 60)

print("\n[1] Loading datasets...")

soh_full = pd.read_csv(SOH_FULL_PATH)
print(f"  Original SOH shape: {soh_full.shape}")
print(f"  Original SOH columns: {soh_full.columns.tolist()}")
print(f"  Original SOH cell_id (numeric) sample: {soh_full['cell_id'].iloc[0]}")
print(f"  Original SOH barcode sample: {soh_full['barcode'].iloc[0]}")

full_features = pd.read_csv(FULL_FEATURES_PATH)
print(f"  Full features shape: {full_features.shape}")
print(f"  Full features columns: {full_features.columns.tolist()}")
print(f"  Full features cell_id sample: {full_features['cell_id'].iloc[0]}")

# ============================================================
# CREATE MAPPING: barcode -> split
# ============================================================

print("\n[2] Creating mapping from barcode to split...")

# Get unique barcode-split mapping from original SOH
split_mapping = soh_full[['barcode', 'split']].drop_duplicates().set_index('barcode')['split'].to_dict()

print(f"  Created mapping with {len(split_mapping)} barcodes")
print(f"  Sample mapping: {list(split_mapping.items())[:3]}")

# ============================================================
# ADD SPLIT COLUMN TO FULL FEATURES USING cell_id (which is the barcode)
# ============================================================

print("\n[3] Adding split column to full features...")

# The full features dataset uses cell_id as the barcode (string)
# Map the split using cell_id
full_features['split'] = full_features['cell_id'].map(split_mapping)

# Check for unmapped rows
unmapped = full_features[full_features['split'].isna()]
if len(unmapped) > 0:
    print(f"  ⚠️ {len(unmapped):,} rows with unmapped cell_ids!")
    print(f"  Unmapped cell_ids sample: {unmapped['cell_id'].unique()[:5].tolist()}")
    
    # Try to clean the cell_id (remove 'EL' prefix, lowercase, etc.)
    print("\n  Attempting to clean cell_id for matching...")
    
    # Create a cleaned version for matching
    # The full features cell_id might have 'EL' prefix while barcode has 'el'
    full_features['cell_id_clean'] = full_features['cell_id'].str.lower()
    
    # Create mapping with lowercase barcodes
    split_mapping_lower = soh_full[['barcode', 'split']].drop_duplicates()
    split_mapping_lower['barcode'] = split_mapping_lower['barcode'].str.lower()
    split_mapping_lower = split_mapping_lower.set_index('barcode')['split'].to_dict()
    
    # Try mapping with cleaned cell_id
    full_features['split'] = full_features['cell_id_clean'].map(split_mapping_lower)
    
    # Check again
    unmapped_again = full_features[full_features['split'].isna()]
    if len(unmapped_again) < len(unmapped):
        print(f"  ✅ Cleaned mapping helped! Now {len(unmapped_again):,} rows unmapped")
    else:
        print(f"  ⚠️ Still {len(unmapped_again):,} rows unmapped")
    
    # For remaining unmapped, use test split as fallback
    if len(unmapped_again) > 0:
        print(f"  Using 'test' split for remaining {len(unmapped_again):,} unmapped rows")
        full_features['split'] = full_features['split'].fillna('test')
else:
    print(f"  ✅ All rows mapped successfully!")

# ============================================================
# VERIFY SPLIT DISTRIBUTION
# ============================================================

print("\n[4] Verifying split distribution...")

print("\n  Split distribution in full features:")
for split, count in full_features['split'].value_counts().items():
    print(f"    {split}: {count:,} rows")

print(f"\n  Unique cells per split:")
for split in ['train', 'val', 'test']:
    cells = full_features[full_features['split'] == split]['cell_id'].nunique()
    print(f"    {split}: {cells} cells")

# ============================================================
# CLEAN UP
# ============================================================

print("\n[5] Cleaning up...")

# Remove temporary columns
if 'cell_id_clean' in full_features.columns:
    full_features = full_features.drop(columns=['cell_id_clean'])

# ============================================================
# REORDER COLUMNS
# ============================================================

print("\n[6] Reordering columns...")

# Move important columns to the front
front_cols = ['cell_id', 'split', 'cycle_index', 'soh']
other_cols = [col for col in full_features.columns if col not in front_cols]
new_order = front_cols + other_cols

full_features = full_features[new_order]

print(f"  Columns reordered: {len(full_features.columns)} columns")

# ============================================================
# SAVE
# ============================================================

print("\n[7] Saving datasets...")

full_features.to_csv(OUTPUT_PATH, index=False)
print(f"  ✅ Saved CSV: {OUTPUT_PATH}")

full_features.to_pickle(OUTPUT_PKL_PATH)
print(f"  ✅ Saved Pickle: {OUTPUT_PKL_PATH}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅ SPLIT COLUMN ADDED SUCCESSFULLY!")
print("=" * 60)

print(f"\nDataset shape: {full_features.shape[0]:,} rows, {full_features.shape[1]} columns")

print(f"\nOutput files:")
print(f"  CSV: {OUTPUT_PATH}")
print(f"  Pickle: {OUTPUT_PKL_PATH}")

print("\nFeatures in dataset:")
for i, col in enumerate(full_features.columns, 1):
    print(f"  {i:>2}. {col}")
print("=" * 60)