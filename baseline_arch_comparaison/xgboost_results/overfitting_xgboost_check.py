# diagnostic_xgboost.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# Load the saved model and data
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load original data
df = pd.read_csv(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv")

# Check if test data is actually different from train
print("Split distribution:")
print(df['split'].value_counts())

# Load test predictions from results.json
import json
with open("results.json", "r") as f:
    results = json.load(f)

print("\nXGBoost Results:")
print(f"  Test MAE: {results['test_mae_pct']:.4f}%")
print(f"  Test R²:  {results['test_r2']:.4f}")

# Check if predictions are too close to true values
# This would indicate leakage
print("\nChecking for data leakage...")
print(f"  Train shape: {results['n_train']}")
print(f"  Val shape:   {results['n_val']}")
print(f"  Test shape:  {results['n_test']}")

# Check if test and train have different cells
train_cells = df[df['split'] == 'train']['barcode'].unique()
test_cells = df[df['split'] == 'test']['barcode'].unique()
val_cells = df[df['split'] == 'val']['barcode'].unique()

print(f"\nCell distribution:")
print(f"  Train cells: {len(train_cells)}")
print(f"  Val cells:   {len(val_cells)}")
print(f"  Test cells:  {len(test_cells)}")

# Check if any cell appears in multiple splits
train_set = set(train_cells)
test_set = set(test_cells)
val_set = set(val_cells)

overlap_train_test = train_set.intersection(test_set)
overlap_train_val = train_set.intersection(val_set)
overlap_val_test = val_set.intersection(test_set)

print(f"\nData leakage check:")
print(f"  Train ∩ Test: {len(overlap_train_test)} cells")
print(f"  Train ∩ Val:  {len(overlap_train_val)} cells")
print(f"  Val ∩ Test:   {len(overlap_val_test)} cells")

if len(overlap_train_test) > 0 or len(overlap_train_val) > 0 or len(overlap_val_test) > 0:
    print("⚠️ WARNING: Data leakage detected!")
else:
    print("✅ No data leakage detected.")