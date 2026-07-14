# Check your column names
import pandas as pd

df = pd.read_csv(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv")
print(df.columns.tolist())