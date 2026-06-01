import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

def explore_dataset_structure(root_path):
    """
    Explore the MIT-Stanford battery dataset structure
    """
    root_dir = Path(root_path)
    
    # Find all JSON files
    json_files = list(root_dir.rglob("*_structure.json"))
    
    print(f"Found {len(json_files)} JSON files")
    print("="*80)
    
    # Store structures from different files to understand variations
    structures = []
    
    # Analyze first few files from each major subdirectory
    files_to_analyze = []
    
    # Get representative files from each subdirectory
    for subdir in root_dir.rglob("*"):
        if subdir.is_dir():
            subdir_files = list(subdir.glob("*_structure.json"))
            if subdir_files:
                files_to_analyze.append(subdir_files[0])  # Take first file from each subdir
    
    # Also include some random files
    import random
    if len(json_files) > 20:
        files_to_analyze.extend(random.sample(json_files, min(20, len(json_files))))
    
    files_to_analyze = list(set(files_to_analyze))  # Remove duplicates
    
    print(f"\nAnalyzing {len(files_to_analyze)} representative files...\n")
    
    for json_file in files_to_analyze[:10]:  # Limit to 10 for initial analysis
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                structures.append({
                    'file': str(json_file.relative_to(root_dir)),
                    'data': data
                })
                print(f"✓ Loaded: {json_file.name}")
        except Exception as e:
            print(f"✗ Error loading {json_file.name}: {e}")
    
    print("\n" + "="*80)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*80)
    
    # Analyze structure of first file in detail
    if structures:
        first_structure = structures[0]['data']
        print("\n FIRST FILE STRUCTURE:")
        print(f"File: {structures[0]['file']}")
        print(f"Type: {type(first_structure)}")
        
        if isinstance(first_structure, dict):
            print(f"\nTop-level keys: {list(first_structure.keys())}")
            
            for key, value in first_structure.items():
                print(f"\n🔑 '{key}':")
                print(f"   Type: {type(value)}")
                if isinstance(value, dict):
                    print(f"   Sub-keys: {list(value.keys())[:10]}")  # First 10 subkeys
                    if len(value) > 10:
                        print(f"   ... and {len(value)-10} more keys")
                elif isinstance(value, list):
                    print(f"   Length: {len(value)}")
                    if len(value) > 0:
                        print(f"   First element type: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"   First element keys: {list(value[0].keys())[:5]}")
        else:
            print(f"Data is not a dictionary, it's a {type(first_structure)}")
    
    # Collect all column names across files
    print("\n" + "="*80)
    print("COLUMN ANALYSIS")
    print("="*80)
    
    all_columns = defaultdict(int)
    column_examples = {}
    
    for structure_info in structures[:5]:  # Analyze first 5 files for columns
        data = structure_info['data']
        
        if isinstance(data, dict):
            # Look for time-series data or cycle data
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # This might be the main data array
                    if isinstance(value[0], dict):
                        columns = list(value[0].keys())
                        for col in columns:
                            all_columns[col] += 1
                            if col not in column_examples:
                                column_examples[col] = value[0][col]
                    
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                columns = list(data[0].keys())
                for col in columns:
                    all_columns[col] += 1
                    if col not in column_examples:
                        column_examples[col] = data[0][col]
    
    if all_columns:
        print("\n Columns found in the dataset:")
        print("-" * 40)
        for col, count in sorted(all_columns.items()):
            example = column_examples.get(col, "N/A")
            print(f"  • {col:<20} (found in {count} files) - Example: {example}")
    else:
        print("\nNo column-based data structure found. Analyzing raw structure...")
        
        # Show structure of first file completely
        if structures:
            print("\n Complete structure of first file:")
            # Try to print more of the structure
            structure_str = json.dumps(structures[0]['data'], indent=2)
            if len(structure_str) > 2000:
                print(structure_str[:2000] + "\n... (truncated)")
            else:
                print(structure_str)
    
    # Check for metadata
    print("\n" + "="*80)
    print("METADATA ANALYSIS")
    print("="*80)
    
    metadata_keys = set()
    for structure_info in structures:
        data = structure_info['data']
        if isinstance(data, dict):
            for key in data.keys():
                if not isinstance(data[key], (list, dict)) or len(str(data[key])) < 100:
                    metadata_keys.add(key)
    
    if metadata_keys:
        print("\n  Potential metadata fields:")
        for key in sorted(metadata_keys):
            print(f"  • {key}")
    
    # File naming convention analysis
    print("\n" + "="*80)
    print("FILE NAMING CONVENTION")
    print("="*80)
    
    naming_patterns = defaultdict(list)
    for json_file in json_files[:20]:
        name = json_file.stem
        parts = name.split('_')
        naming_patterns[len(parts)].append(name)
    
    print("\n File naming patterns:")
    for pattern_len, examples in naming_patterns.items():
        print(f"  {pattern_len} parts: {examples[0]} (and {len(examples)-1} more)")
    
    # Identify main data structure from a sample file
    print("\n" + "="*80)
    print("SAMPLE DATA PREVIEW")
    print("="*80)
    
    if structures:
        sample_file = structures[0]['file']
        sample_data = structures[0]['data']
        
        print(f"\n Sample from: {sample_file}")
        
        if isinstance(sample_data, dict):
            for key, value in sample_data.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"\n  Key '{key}': Array of {len(value)} elements")
                    if len(value) <= 5:
                        print(f"    Preview: {value}")
                    else:
                        print(f"    First 3 elements: {value[:3]}")
                elif isinstance(value, dict):
                    print(f"\n  Key '{key}': Dictionary with {len(value)} keys")
                    print(f"    Keys: {list(value.keys())[:5]}")
                    # If there are nested structures, show first few
                    for sub_key, sub_value in list(value.items())[:3]:
                        if isinstance(sub_value, list) and len(sub_value) > 0:
                            print(f"      - {sub_key}: list with {len(sub_value)} items")
                        else:
                            print(f"      - {sub_key}: {type(sub_value).__name__}")
                else:
                    print(f"\n  Key '{key}': {value}")
    
    return structures

def create_dataframe_from_structure(json_file):
    """
    Convert a JSON structure file to a pandas DataFrame
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Try to find the main data array
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        # Look for arrays in the dictionary
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    print(f"Found data array under key: '{key}'")
                    return pd.DataFrame(value)
        
        # If no array found, return the whole dict as single row
        return pd.DataFrame([data])
    else:
        print(f"Unexpected data type: {type(data)}")
        return None

# Main execution
if __name__ == "__main__":
    # CORRECTED PATH - remove the "PS:\" prefix
    dataset_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website"
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Path not found: {dataset_path}")
        print("\nTrying alternative paths...")
        
        # Try alternative path formats
        alt_paths = [
            r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website",
            r".\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website",
            os.path.join(os.getcwd(), "04 MIT–Stanford–TRI Fast-Charging Dataset", "Main Website")
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                dataset_path = alt_path
                print(f"Found path: {dataset_path}")
                break
        else:
            print("\nPlease enter the correct path to your dataset:")
            dataset_path = input("Dataset path: ").strip()
            
            if not os.path.exists(dataset_path):
                print("Path still not found. Please check and update the path in the script.")
                exit(1)
    
    print(f"\nUsing dataset path: {dataset_path}\n")
    
    # Explore the dataset structure
    structures = explore_dataset_structure(dataset_path)
    
    # Optional: Load one file into a DataFrame
    print("\n" + "="*80)
    print("LOADING SAMPLE DATAFRAME")
    print("="*80)
    
    # Find one JSON file to load
    root_dir = Path(dataset_path)
    json_files = list(root_dir.rglob("*_structure.json"))
    
    if json_files:
        sample_file = json_files[0]
        print(f"\nLoading: {sample_file.name}")
        
        df = create_dataframe_from_structure(sample_file)
        
        if df is not None:
            print(f"\n DataFrame shape: {df.shape}")
            print(f"\nColumns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nData types:")
            print(df.dtypes)
            print(f"\nBasic statistics for numeric columns:")
            print(df.describe())
    else:
        print("No JSON files found in the specified directory.")