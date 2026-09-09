# count_physical_cells_raw.py
"""
Analyze raw JSON files to count unique physical cells and channels.
This script scans the raw dataset directory and extracts:
1. Total number of JSON files
2. Unique physical cells (by barcode)
3. Channel distribution per cell
4. Invalid/corrupted files
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")

# ============================================================
# ANALYZE RAW JSON FILES
# ============================================================

print("=" * 70)
print("ANALYZING RAW JSON FILES")
print("=" * 70)

print(f"\n[1] Scanning directory: {DATA_DIR}")

# Get all JSON files
json_files = sorted(DATA_DIR.glob("*_structure.json"))
print(f"  Found {len(json_files)} JSON files")

# Analyze each file
barcodes = []
channels = []
valid_files = 0
invalid_files = 0
cell_channel_map = defaultdict(list)

print("\n[2] Parsing JSON files...")

for fp in json_files:
    try:
        with open(fp, "r") as f:
            data = json.load(f)
        
        # Extract barcode
        barcode = data.get("barcode", "UNKNOWN")
        barcodes.append(barcode)
        
        # Extract channel from filename
        filename = fp.name
        channel = "UNKNOWN"
        if "_CH" in filename:
            parts = filename.split('_')
            for part in parts:
                if part.startswith("CH"):
                    channel = part
                    break
        
        channels.append(channel)
        cell_channel_map[barcode].append(channel)
        valid_files += 1
        
    except Exception as e:
        print(f"  ERROR reading {fp.name}: {e}")
        invalid_files += 1

# ============================================================
# RESULTS
# ============================================================

print("\n[3] Results:")
print("-" * 50)

print(f"  Total JSON files: {len(json_files)}")
print(f"  Valid files: {valid_files}")
print(f"  Invalid files: {invalid_files}")

# Unique barcodes (physical cells)
unique_barcodes = set(barcodes)
print(f"\n  Unique physical cells (barcodes): {len(unique_barcodes)}")

# Distribution of channels per cell
channel_counts = {barcode: len(channels) for barcode, channels in cell_channel_map.items()}
avg_channels = sum(channel_counts.values()) / len(channel_counts) if channel_counts else 0

print(f"\n  Channel distribution:")
print(f"    Average channels per cell: {avg_channels:.2f}")
print(f"    Min channels per cell: {min(channel_counts.values()) if channel_counts else 0}")
print(f"    Max channels per cell: {max(channel_counts.values()) if channel_counts else 0}")

# Count cells by number of channels
channel_distribution = defaultdict(int)
for count in channel_counts.values():
    channel_distribution[count] += 1

print(f"\n  Cells by number of channels:")
for count in sorted(channel_distribution.keys()):
    cells = channel_distribution[count]
    print(f"    {count} channel(s): {cells} cells")

# ============================================================
# SHOW EXAMPLES OF MULTI-CHANNEL CELLS
# ============================================================

print("\n[4] Examples of cells with multiple channels:")
print("-" * 50)

multi_channel_cells = {barcode: ch for barcode, ch in cell_channel_map.items() if len(ch) > 1}
sorted_multi = sorted(multi_channel_cells.items(), key=lambda x: len(x[1]), reverse=True)

for i, (barcode, channels_list) in enumerate(sorted_multi[:10], 1):
    print(f"  {i}. Barcode: {barcode} ({len(channels_list)} channels)")
    print(f"     Channels: {', '.join(channels_list[:5])}{'...' if len(channels_list) > 5 else ''}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATASET SUMMARY                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Total JSON files: {len(json_files)}                        │
│  Valid files:      {valid_files}                           │
│  Invalid files:    {invalid_files}                         │
│                                                             │
│  Unique physical cells (barcodes): {len(unique_barcodes)}   │
│                                                             │
│  Channel distribution:                                     │
│    Average: {avg_channels:.2f} channels per cell            │
│    Min:     {min(channel_counts.values()) if channel_counts else 0} channels per cell│
│    Max:     {max(channel_counts.values()) if channel_counts else 0} channels per cell│
│                                                             │
│  Multi-channel cells: {len(multi_channel_cells)} cells      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

# ============================================================
# VERIFY WITH DOCUMENTATION
# ============================================================

print("\n[5] Verification with Documentation:")
print("-" * 50)

print(f"  Dataset documentation states: 124 physical cells")
print(f"  Our analysis found:           {len(unique_barcodes)} physical cells")

if len(unique_barcodes) != 124:
    print(f"\n  ⚠️ Difference: {len(unique_barcodes) - 124} cells")
    print(f"  Possible reasons:")
    print(f"    - Some cells tested on multiple channels are counted separately")
    print(f"    - Additional cells in the dataset beyond the original 124")
    print(f"    - Different interpretation of 'physical cell'")
else:
    print(f"\n  ✅ Confirmed: {len(unique_barcodes)} physical cells match documentation")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "total_json_files": len(json_files),
    "valid_files": valid_files,
    "invalid_files": invalid_files,
    "unique_physical_cells": len(unique_barcodes),
    "avg_channels_per_cell": avg_channels,
    "min_channels_per_cell": min(channel_counts.values()) if channel_counts else 0,
    "max_channels_per_cell": max(channel_counts.values()) if channel_counts else 0,
    "multi_channel_cells": len(multi_channel_cells),
    "channel_distribution": dict(channel_distribution),
}

# Create summary DataFrame
summary_df = pd.DataFrame([results])
summary_df.to_csv("raw_dataset_analysis.csv", index=False)
print(f"\n  Results saved to: raw_dataset_analysis.csv")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)