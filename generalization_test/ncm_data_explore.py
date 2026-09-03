# explore_ncm_files.py
# Simple script to explore the NCM folder structure
# Understand what files exist and how they're organized

import os
import glob
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NMC\SNL NMC"

# ─────────────────────────────────────────────────────────────────────────────
# 2. List all files
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SNL NCM FOLDER EXPLORATION")
print("=" * 60)

# Find all CSV files
all_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
all_files = sorted(all_files)

print(f"\nTotal files: {len(all_files)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Classify files by type
# ─────────────────────────────────────────────────────────────────────────────

cycle_data_files = []
timeseries_files = []
other_files = []

for f in all_files:
    filename = os.path.basename(f)
    if 'cycle_data' in filename.lower():
        cycle_data_files.append(f)
    elif 'timeseries' in filename.lower():
        timeseries_files.append(f)
    else:
        other_files.append(f)

print(f"\nCycle data files:   {len(cycle_data_files)}")
print(f"Timeseries files:   {len(timeseries_files)}")
print(f"Other files:        {len(other_files)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Show first 20 files
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  FIRST 20 FILES")
print("=" * 60)

for i, f in enumerate(all_files[:20]):
    filename = os.path.basename(f)
    file_type = "CYCLE" if 'cycle_data' in filename.lower() else "TIME" if 'timeseries' in filename.lower() else "OTHER"
    print(f"  {i+1:2d}. [{file_type}] {filename}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Identify unique cell IDs from filenames
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  UNIQUE CELL IDs")
print("=" * 60)

# Try to extract cell IDs from cycle_data files
cell_ids = set()
for f in cycle_data_files:
    filename = os.path.basename(f)
    # Remove '_cycle_data.csv' to get cell ID
    cell_id = filename.replace('_cycle_data.csv', '')
    # Alternative: remove everything after last '_cycle_data'
    if '_cycle_data' in filename:
        cell_id = filename.split('_cycle_data')[0]
    cell_ids.add(cell_id)

print(f"\nUnique cell IDs from cycle_data files: {len(cell_ids)}")
print(f"Cell IDs (first 10): {sorted(list(cell_ids))[:10]}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Check if cycle_data and timeseries files have matching cell IDs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  MATCHING CYCLE DATA vs TIMESERIES")
print("=" * 60)

# Extract cell IDs from timeseries files
ts_cell_ids = set()
for f in timeseries_files:
    filename = os.path.basename(f)
    if '_timeseries' in filename:
        cell_id = filename.split('_timeseries')[0]
        ts_cell_ids.add(cell_id)

print(f"\nCell IDs from cycle_data: {len(cell_ids)}")
print(f"Cell IDs from timeseries: {len(ts_cell_ids)}")

# Find overlapping
overlap = cell_ids & ts_cell_ids
only_cycle = cell_ids - ts_cell_ids
only_ts = ts_cell_ids - cell_ids

print(f"Overlap (both):          {len(overlap)}")
print(f"Only in cycle_data:      {len(only_cycle)}")
print(f"Only in timeseries:      {len(only_ts)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Inspect one cycle_data file in detail
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  INSPECTING A CYCLE DATA FILE")
print("=" * 60)

if cycle_data_files:
    sample_file = cycle_data_files[0]
    print(f"\nFile: {os.path.basename(sample_file)}")
    
    try:
        df = pd.read_csv(sample_file)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns:\n{df.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check for SOH-related columns
        print(f"\nSOH-related columns found:")
        for col in df.columns:
            if 'capacity' in col.lower() or 'energy' in col.lower() or 'soh' in col.lower():
                print(f"  - {col}")
                
        # Check data types
        print(f"\nData types:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"Error loading file: {e}")
else:
    print("No cycle_data files found!")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Check for temperature data in timeseries files
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  CHECKING TIMESERIES FOR TEMPERATURE")
print("=" * 60)

if timeseries_files:
    sample_ts = timeseries_files[0]
    print(f"\nFile: {os.path.basename(sample_ts)}")
    
    try:
        df_ts = pd.read_csv(sample_ts)
        print(f"\nShape: {df_ts.shape}")
        print(f"\nColumns:\n{df_ts.columns.tolist()}")
        
        # Check for temperature columns
        temp_cols = [col for col in df_ts.columns if 'temp' in col.lower()]
        if temp_cols:
            print(f"\nTemperature columns found: {temp_cols}")
            print(f"Sample temperature data:")
            print(df_ts[temp_cols].head())
        else:
            print("\nNo temperature columns found in timeseries data.")
            
    except Exception as e:
        print(f"Error loading file: {e}")
else:
    print("No timeseries files found!")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"""
Total files:               {len(all_files)}
Cycle data files:          {len(cycle_data_files)}
Timeseries files:          {len(timeseries_files)}
Unique cell IDs (cycle):   {len(cell_ids)}
Unique cell IDs (time):    {len(ts_cell_ids)}
Cells with both files:     {len(overlap)}
""")

print("✅ Exploration complete! Now you know:")
print(f"  - There are {len(cell_ids)} unique cells in cycle_data")
print(f"  - There are {len(ts_cell_ids)} unique cells in timeseries")
print(f"  - {len(overlap)} cells have BOTH cycle_data AND timeseries")

if not temp_cols:
    print("  - No temperature data in cycle_data files")

print("\n" + "=" * 60)
print("  NEXT STEPS")
print("=" * 60)
# print("""
# 1. If cells have both cycle_data AND timeseries:
#    - cycle_data has cycle-level summaries (SOH)
#    - timeseries has raw data with temperature
   
# 2. We can merge them to get:
#    - All features from cycle_data (capacity, energy, voltage, current)
#    - Temperature from timeseries (average per cycle)
   
# 3. Then we'll have 9 features:
#    - charge_capacity, charge_energy, coulombic_efficiency_lagged_1, 
#      coulombic_efficiency_lagged_2, cap_rel, energy_rel, voltage_range, 
#      cycle_pos, temperature_avg
# """)