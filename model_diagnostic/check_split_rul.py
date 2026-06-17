# check_split_rul.py
import pandas as pd

rul_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\rul\rul_full.csv"

rul = pd.read_csv(rul_path)

print("=" * 60)
print("CHECKING RUL CSV FILE")
print("=" * 60)

print(f"\nShape: {rul.shape}")
print(f"Columns: {rul.columns.tolist()}")

if 'split' in rul.columns:
    print("\n✅ SUCCESS: 'split' column EXISTS in RUL!")
    print(f"\nSplit values: {rul['split'].unique()}")
    print(f"Split counts:")
    print(rul['split'].value_counts())
else:
    print("\n❌ ERROR: 'split' column MISSING in RUL!")

# Also check for 'has_label' column (needed for semi-supervised RUL)
if 'has_label' in rul.columns:
    print(f"\n✅ 'has_label' column EXISTS!")
    print(f"Labeled (has_label=1): {(rul['has_label'] == 1).sum()}")
    print(f"Unlabeled (has_label=0): {(rul['has_label'] == 0).sum()}")
else:
    print("\n⚠️ WARNING: 'has_label' column MISSING")
    print("This is needed for semi-supervised RUL training")