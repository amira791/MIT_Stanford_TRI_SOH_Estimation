"""
RECREATE FULL CSV WITH ALL SUMMARY FEATURES
-------------------------------------------
Loads the original pickle files (which contain all features) and exports
them to CSV for feature selection analysis.
"""

import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

# Paths to your original pickle files (with ALL features)
SOH_PKL_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.pkl"
RUL_PKL_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\rul\rul_full.pkl"

# Output paths
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\full_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOH_CSV_PATH = OUTPUT_DIR / "soh_full_features.csv"

# ============================================================
# LOAD AND EXPORT
# ============================================================

print("="*60)
print("RECREATING FULL CSV WITH ALL FEATURES")
print("="*60)

# Load pickle files
print("\n[1] Loading pickle files...")
soh_df = pd.read_pickle(SOH_PKL_PATH)
print(f"  SOH shape: {soh_df.shape}")
print(f"  SOH columns: {soh_df.columns.tolist()}")

# Export to CSV
print("\n[2] Exporting to CSV...")
soh_df.to_csv(SOH_CSV_PATH, index=False)
print(f"  ✅ Saved: {SOH_CSV_PATH}")

# Show all available features
print("\n[3] Available features in SOH dataset:")
print("-" * 50)
for i, col in enumerate(soh_df.columns):
    print(f"  {i+1}. {col}")

print("\n" + "="*60)
print("✅ DONE! Use this CSV for feature selection.")
print(f"   Path: {SOH_CSV_PATH}")
print("="*60)