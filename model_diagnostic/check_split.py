# check_split_column.py
import pandas as pd

# Path to your CSV
csv_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Check columns
print("=" * 60)
print("CHECKING CSV FILE FOR 'split' COLUMN")
print("=" * 60)

print(f"\nFile: {csv_path}")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Check if 'split' exists
if 'split' in df.columns:
    print("\n✅ SUCCESS: 'split' column EXISTS!")
    print(f"\nSplit values: {df['split'].unique()}")
    print(f"Split counts:")
    print(df['split'].value_counts())
else:
    print("\n❌ ERROR: 'split' column MISSING!")
    print("Your CSV does not have the required 'split' column.")
    print("The training code will FAIL.")