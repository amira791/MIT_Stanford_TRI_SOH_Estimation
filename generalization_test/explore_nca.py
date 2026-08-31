# explore_snl_nca.py
# Explore the SNL NCA dataset before processing

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NCA\SNL NCA"

# ─────────────────────────────────────────────────────────────────────────────
# 1. List all files
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SNL NCA DATASET EXPLORATION")
print("=" * 60)

# Find all CSV files
csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
print(f"\nFound {len(csv_files)} CSV files")

# Show first 5 files
print("\nFirst 5 files:")
for f in csv_files[:5]:
    print(f"  {os.path.basename(f)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Separate cycle_data vs timeseries
# ─────────────────────────────────────────────────────────────────────────────

cycle_files = [f for f in csv_files if 'cycle_data' in f.lower()]
timeseries_files = [f for f in csv_files if 'timeseries' in f.lower()]

print(f"\nCycle data files: {len(cycle_files)}")
print(f"Timeseries files: {len(timeseries_files)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inspect a cycle_data file
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  INSPECTING CYCLE DATA FILE")
print("=" * 60)

if cycle_files:
    sample_file = cycle_files[0]
    print(f"\nFile: {os.path.basename(sample_file)}")
    
    try:
        df_cycle = pd.read_csv(sample_file)
        print(f"\nShape: {df_cycle.shape}")
        print(f"\nColumns:\n{df_cycle.columns.tolist()}")
        
        print(f"\nFirst 5 rows:")
        print(df_cycle.head())
        
        print(f"\nData types:")
        print(df_cycle.dtypes)
        
        print(f"\nBasic statistics:")
        print(df_cycle.describe())
        
    except Exception as e:
        print(f"Error loading: {e}")
else:
    print("No cycle_data files found!")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Inspect a timeseries file (for reference)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  INSPECTING TIMESERIES FILE (Reference)")
print("=" * 60)

if timeseries_files:
    sample_file = timeseries_files[0]
    print(f"\nFile: {os.path.basename(sample_file)}")
    
    try:
        df_ts = pd.read_csv(sample_file)
        print(f"\nShape: {df_ts.shape}")
        print(f"\nColumns:\n{df_ts.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(df_ts.head(3))
        
    except Exception as e:
        print(f"Error loading: {e}")
else:
    print("No timeseries files found!")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check for missing values (cycle_data)
# ─────────────────────────────────────────────────────────────────────────────

if cycle_files:
    print("\n" + "=" * 60)
    print("  MISSING VALUES (Cycle Data)")
    print("=" * 60)
    
    missing = df_cycle.isnull().sum()
    missing_pct = (missing / len(df_cycle)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Check value ranges for key columns (cycle_data)
# ─────────────────────────────────────────────────────────────────────────────

if cycle_files:
    print("\n" + "=" * 60)
    print("  VALUE RANGES (Cycle Data)")
    print("=" * 60)
    
    key_columns = ['Cycle_Index', 'Charge_Capacity (Ah)', 'Discharge_Capacity (Ah)', 
                   'Charge_Energy (Wh)', 'Discharge_Energy (Wh)', 
                   'Min_Voltage (V)', 'Max_Voltage (V)']
    
    for col in key_columns:
        if col in df_cycle.columns:
            print(f"\n{col}:")
            print(f"  Min: {df_cycle[col].min():.4f}")
            print(f"  Max: {df_cycle[col].max():.4f}")
            print(f"  Mean: {df_cycle[col].mean():.4f}")
            print(f"  Std: {df_cycle[col].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Check multiple cycle_data files for consistency
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  CHECKING MULTIPLE CYCLE DATA FILES")
print("=" * 60)

if cycle_files:
    all_columns = set()
    file_shapes = []
    columns_per_file = []
    
    for i, f in enumerate(cycle_files[:10]):  # Check first 10 files
        try:
            df = pd.read_csv(f)
            file_shapes.append(df.shape)
            columns_per_file.append(set(df.columns))
            all_columns.update(df.columns)
            
            has_capacity = 'Charge_Capacity (Ah)' in df.columns
            has_energy = 'Charge_Energy (Wh)' in df.columns
            has_temp = 'Cell_Temperature (C)' in df.columns or 'Environment_Temperature (C)' in df.columns
            
            print(f"\nFile {i+1}: {os.path.basename(f)}")
            print(f"  Shape: {df.shape}")
            print(f"  Has Charge_Capacity: {has_capacity}")
            print(f"  Has Charge_Energy: {has_energy}")
            print(f"  Has Temperature: {has_temp}")
            print(f"  Columns: {len(df.columns)}")
            
        except Exception as e:
            print(f"\nFile {i+1}: {os.path.basename(f)} - Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Check column consistency across cycle_data files
# ─────────────────────────────────────────────────────────────────────────────

if cycle_files and columns_per_file:
    print("\n" + "=" * 60)
    print("  COLUMN CONSISTENCY")
    print("=" * 60)
    
    common_columns = set.intersection(*columns_per_file)
    print(f"\nColumns present in all cycle_data files:\n{sorted(list(common_columns))}")
    
    all_cols = set.union(*columns_per_file)
    varying = all_cols - common_columns
    if varying:
        print(f"\nColumns that vary across files:\n{sorted(list(varying))}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Check SOH calculation feasibility (cycle_data)
# ─────────────────────────────────────────────────────────────────────────────

if cycle_files:
    print("\n" + "=" * 60)
    print("  SOH CALCULATION FEASIBILITY")
    print("=" * 60)
    
    if 'Discharge_Capacity (Ah)' in df_cycle.columns:
        cycle_col = 'Cycle_Index' if 'Cycle_Index' in df_cycle.columns else None
        if cycle_col:
            df_sorted = df_cycle.sort_values(cycle_col)
            cap = df_sorted['Discharge_Capacity (Ah)'].values
            
            decreasing = np.mean(np.diff(cap)) < 0
            print(f"\nDischarge capacity trend: {'Decreasing' if decreasing else 'Increasing or flat'}")
            print(f"  Initial capacity: {cap[0]:.4f} Ah")
            print(f"  Final capacity: {cap[-1]:.4f} Ah")
            print(f"  Capacity fade: {(1 - cap[-1]/cap[0])*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\nTotal files: {len(csv_files)}")
print(f"Cycle data files: {len(cycle_files)}")
print(f"Timeseries files: {len(timeseries_files)}")

if cycle_files:
    print(f"Sample cycle_data shape: {df_cycle.shape}")
    
    required = ['Cycle_Index', 'Charge_Capacity (Ah)', 'Discharge_Capacity (Ah)', 
                'Charge_Energy (Wh)', 'Discharge_Energy (Wh)']
    available = [col for col in required if col in df_cycle.columns]
    missing = [col for col in required if col not in df_cycle.columns]
    
    print(f"\nRequired columns present: {len(available)}/{len(required)}")
    if missing:
        print(f"  Missing: {missing}")
    
    # Check for temperature
    has_temp = any(col in df_cycle.columns for col in ['Cell_Temperature (C)', 'Environment_Temperature (C)'])
    print(f"  Temperature column: {' Yes' if has_temp else ' No (use 25°C)'}")

print("\n Exploration complete! You can now run the processing script.")