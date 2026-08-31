# explore_snl_ncm.py
# Explore the SNL NCM dataset before processing

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL\SNL NMC\SNL NMC"

# ─────────────────────────────────────────────────────────────────────────────
# 1. List all files
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SNL NCM DATASET EXPLORATION")
print("=" * 60)

# Find all CSV files
csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
print(f"\nFound {len(csv_files)} CSV files")

# Show first 5 files
print("\nFirst 5 files:")
for f in csv_files[:5]:
    print(f"  {os.path.basename(f)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load one file to inspect columns
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  INSPECTING FIRST FILE")
print("=" * 60)

sample_file = csv_files[0]
print(f"\nFile: {os.path.basename(sample_file)}")

# Load first file with different options to see what works
try:
    df_sample = pd.read_csv(sample_file)
    print(f"\nShape: {df_sample.shape}")
    print(f"\nColumns:\n{df_sample.columns.tolist()}")
    
    print(f"\nFirst 5 rows:")
    print(df_sample.head())
    
    print(f"\nData types:")
    print(df_sample.dtypes)
    
    print(f"\nBasic statistics:")
    print(df_sample.describe())
    
except Exception as e:
    print(f"Error loading: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Check for missing values
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  MISSING VALUES (First File)")
print("=" * 60)

missing = df_sample.isnull().sum()
missing_pct = (missing / len(df_sample)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing'] > 0])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Check value ranges for key columns
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  VALUE RANGES (First File)")
print("=" * 60)

key_columns = ['Cycle_Index', 'Charge_Capacity (Ah)', 'Discharge_Capacity (Ah)', 
               'Charge_Energy (Wh)', 'Discharge_Energy (Wh)', 
               'Min_Voltage (V)', 'Max_Voltage (V)']

for col in key_columns:
    if col in df_sample.columns:
        print(f"\n{col}:")
        print(f"  Min: {df_sample[col].min():.4f}")
        print(f"  Max: {df_sample[col].max():.4f}")
        print(f"  Mean: {df_sample[col].mean():.4f}")
        print(f"  Std: {df_sample[col].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check multiple files for consistency
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  CHECKING MULTIPLE FILES")
print("=" * 60)

all_columns = set()
file_shapes = []
columns_per_file = []

for i, f in enumerate(csv_files[:10]):  # Check first 10 files
    try:
        df = pd.read_csv(f)
        file_shapes.append(df.shape)
        columns_per_file.append(set(df.columns))
        all_columns.update(df.columns)
        
        # Check for key columns
        has_capacity = 'Charge_Capacity (Ah)' in df.columns
        has_energy = 'Charge_Energy (Wh)' in df.columns
        
        print(f"\nFile {i+1}: {os.path.basename(f)}")
        print(f"  Shape: {df.shape}")
        print(f"  Has Charge_Capacity: {has_capacity}")
        print(f"  Has Charge_Energy: {has_energy}")
        print(f"  Columns: {len(df.columns)}")
        
    except Exception as e:
        print(f"\nFile {i+1}: {os.path.basename(f)} - Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Check column consistency across files
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  COLUMN CONSISTENCY")
print("=" * 60)

# Show common columns
if columns_per_file:
    common_columns = set.intersection(*columns_per_file)
    print(f"\nColumns present in all files:\n{sorted(list(common_columns))}")
    
    # Show columns that vary
    all_cols = set.union(*columns_per_file)
    varying = all_cols - common_columns
    if varying:
        print(f"\nColumns that vary across files:\n{sorted(list(varying))}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Check SOH calculation feasibility
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SOH CALCULATION FEASIBILITY")
print("=" * 60)

# Check if we can compute SOH from discharge capacity
if 'Discharge_Capacity (Ah)' in df_sample.columns:
    # Check if discharge capacity decreases over time (expected)
    cycle_col = 'Cycle_Index' if 'Cycle_Index' in df_sample.columns else None
    if cycle_col:
        # Sort by cycle
        df_sorted = df_sample.sort_values(cycle_col)
        cap = df_sorted['Discharge_Capacity (Ah)'].values
        
        # Check monotonic trend
        decreasing = np.mean(np.diff(cap)) < 0
        print(f"\nDischarge capacity trend: {'Decreasing' if decreasing else 'Increasing or flat'}")
        print(f"  Initial capacity: {cap[0]:.4f} Ah")
        print(f"  Final capacity: {cap[-1]:.4f} Ah")
        print(f"  Capacity fade: {(1 - cap[-1]/cap[0])*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\nTotal files: {len(csv_files)}")
print(f"Common columns: {len(common_columns) if columns_per_file else 'N/A'}")
print(f"Sample file shape: {df_sample.shape}")

# Check if we have the necessary columns
required = ['Cycle_Index', 'Charge_Capacity (Ah)', 'Discharge_Capacity (Ah)', 
            'Charge_Energy (Wh)', 'Discharge_Energy (Wh)']
available = [col for col in required if col in df_sample.columns]
missing = [col for col in required if col not in df_sample.columns]

print(f"\nRequired columns present: {len(available)}/{len(required)}")
if missing:
    print(f"  Missing: {missing}")

print("\n Exploration complete! You can now run the processing script.")