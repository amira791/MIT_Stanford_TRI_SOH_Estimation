# verify_unique_cells.py
from pathlib import Path
from collections import defaultdict

dataset_path = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")

# Find all JSON files
json_files = list(dataset_path.rglob("*_structure.json"))

# Extract unique cell IDs
cell_ids = set()
cell_channels = defaultdict(list)

for file in json_files:
    name = file.stem  # FastCharge_000069_CH25_structure
    parts = name.split('_')
    
    if len(parts) >= 3:
        cell_id = parts[1]  # 000069
        channel = parts[2]  # CH25
        
        cell_ids.add(cell_id)
        cell_channels[cell_id].append(channel)

print(f"Total JSON files: {len(json_files)}")
print(f"Unique cell IDs: {len(cell_ids)}")
print(f"\nOfficial count: 124 cells")
print(f"Difference: {len(cell_ids) - 124}")

if len(cell_ids) != 124:
    print(f"\n Found {len(cell_ids)} unique cells, not 124")
    print("\nCells with multiple channels:")
    for cell_id, channels in sorted(cell_channels.items()):
        if len(channels) > 1:
            print(f"  Cell {cell_id}: {len(channels)} channels ({', '.join(channels)})")