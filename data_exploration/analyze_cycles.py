# verify_cycles.py
import json
from pathlib import Path
import pandas as pd

def verify_cell_cycles(json_file):
    """Extract cycles and capacities from a single cell"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'summary' in data:
        summary = data['summary']
        if 'cycle_index' in summary and 'discharge_capacity' in summary:
            df = pd.DataFrame({
                'cycle': summary['cycle_index'],
                'capacity': summary['discharge_capacity']
            })
            return df
    
    return None

# Your dataset path
dataset_path = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")

# Find all JSON files
json_files = list(dataset_path.rglob("*_structure.json"))
print(f"Found {len(json_files)} cells\n")

# Analyze first 15 cells
for i, json_file in enumerate(json_files[:15]):
    df = verify_cell_cycles(json_file)
    
    if df is not None:
        print(f"\n{'='*60}")
        print(f"Cell {i+1}: {json_file.stem}")
        print(f"{'='*60}")
        
        # Show first 10 cycles
        print("\nFirst 10 cycles:")
        print(df.head(10).to_string(index=False))
        
        # Check cycle 0 anomaly
        cycle0 = df[df['cycle'] == 0]
        if not cycle0.empty:
            cap0 = cycle0['capacity'].values[0]
            if cap0 > 1.3:
                print(f"\n  Cycle 0 capacity: {cap0:.3f}Ah (ANOMALOUS - >1.3Ah)")
            else:
                print(f"\n✓ Cycle 0 capacity: {cap0:.3f}Ah (normal range)")
        
        # Show cycles 1-5 statistics
        cycles_1_5 = df[(df['cycle'] >= 1) & (df['cycle'] <= 5)]['capacity']
        if len(cycles_1_5) == 5:
            print(f"\nCycles 1-5 capacities: {[round(c, 3) for c in cycles_1_5.values]}")
            print(f"Mean (cycles 1-5): {cycles_1_5.mean():.3f}Ah")
            print(f"Std deviation: {cycles_1_5.std():.4f}Ah")
        
        print(f"\nTotal cycles: {len(df)}")