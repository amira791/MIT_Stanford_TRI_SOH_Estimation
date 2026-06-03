import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MITStanfordBatteryDataset:
    """Loader for MIT-Stanford battery dataset - Handles both dataset formats"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        # Look for the FastCharge directory (Data-driven prediction dataset)
        self.fastcharge_dir = self.data_path / "Data-driven prediction of battery cycle life before capacity degradation" / "FastCharge"
        
        # If not found, try alternative paths
        if not self.fastcharge_dir.exists():
            # Try the other dataset (Closed-loop optimization)
            self.fastcharge_dir = self.data_path / "Closed-loop optimization of extreme fast charging for batteries using machine learning"
            print(f"Using Closed-loop optimization dataset at: {self.fastcharge_dir}")
        
        print(f"Dataset directory: {self.fastcharge_dir}")
        
    def load_cell(self, cell_id):
        """Load a specific cell by its ID (e.g., '000000')"""
        # Try different patterns
        patterns = [
            f"FastCharge_{cell_id}_CH*_structure.json",
            f"*_{cell_id}_CH*_structure.json", 
            f"*_CH*_structure.json"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(list(self.fastcharge_dir.rglob(pattern)))
            if files:
                break
        
        if not files:
            # Try to find any file with this cell_id
            all_files = list(self.fastcharge_dir.rglob("*_structure.json"))
            files = [f for f in all_files if cell_id in str(f)]
            
        if not files:
            raise ValueError(f"Cell {cell_id} not found in {self.fastcharge_dir}")
        
        print(f"Loading: {files[0].name}")
        with open(files[0], 'r') as f:
            data = json.load(f)
        
        return self._parse_cell_data(data)
    
    def _parse_cell_data(self, data):
        """Parse JSON into usable DataFrames - Handles multiple formats"""
        
        result = {
            'metadata': {},
            'summary': None,
            'raw': None,
            'interpolated': None,
            'cycles': None  # Alternative name for summary
        }
        
        # Extract metadata
        result['metadata'] = {
            'barcode': data.get('barcode', 'Unknown'),
            'channel': data.get('channel_id', 'Unknown'),
            'protocol': data.get('protocol', 'Unknown'),
            'module': data.get('@module', 'Unknown'),
            'class': data.get('@class', 'Unknown')
        }
        
        # Try different possible keys for summary data
        summary_keys = ['summary', 'cycles', 'cycle_summary', 'cycle_data']
        for key in summary_keys:
            if key in data and data[key] is not None:
                if isinstance(data[key], dict):
                    # Check if it's a dictionary of lists
                    if any(isinstance(v, list) for v in data[key].values()):
                        result['summary'] = pd.DataFrame(data[key])
                        break
                elif isinstance(data[key], list) and len(data[key]) > 0:
                    result['summary'] = pd.DataFrame(data[key])
                    break
        
        # Try different keys for raw data
        raw_keys = ['raw_data', 'raw', 'data', 'timeseries']
        for key in raw_keys:
            if key in data and data[key] is not None:
                if isinstance(data[key], dict):
                    if any(isinstance(v, list) for v in data[key].values()):
                        # Convert dict of lists to DataFrame
                        result['raw'] = pd.DataFrame(data[key])
                        break
                elif isinstance(data[key], list) and len(data[key]) > 0:
                    result['raw'] = pd.DataFrame(data[key])
                    break
        
        # Try different keys for interpolated data
        interp_keys = ['cycles_interpolated', 'interpolated', 'interp_data']
        for key in interp_keys:
            if key in data and data[key] is not None:
                if isinstance(data[key], dict):
                    if any(isinstance(v, list) for v in data[key].values()):
                        result['interpolated'] = pd.DataFrame(data[key])
                        break
        
        # If we have a 'summary' that's a list of cycles with nested data
        if result['summary'] is None and 'summary' in data:
            summary_data = data['summary']
            if isinstance(summary_data, dict) and 'cycle_index' in summary_data:
                # Already handled above
                pass
            elif isinstance(summary_data, list):
                result['summary'] = pd.DataFrame(summary_data)
        
        # Print what we found
        print(f"  ✓ Found summary data: {result['summary'] is not None}")
        print(f"  ✓ Found raw data: {result['raw'] is not None}")
        print(f"  ✓ Found interpolated data: {result['interpolated'] is not None}")
        
        return result
    
    def get_available_cells(self):
        """List all available cells in the dataset"""
        json_files = list(self.fastcharge_dir.rglob("*_structure.json"))
        
        cells = []
        for file in json_files:
            # Extract cell ID from filename
            name = file.stem
            if 'FastCharge_' in name:
                cell_id = name.split('_')[1]
            elif 'CH' in name:
                # Try to extract from pattern like "2018-08-28_oed_0_CH10"
                parts = name.split('_')
                cell_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            else:
                cell_id = name
            
            cells.append({
                'file': file.name,
                'cell_id': cell_id,
                'path': str(file)
            })
        
        return pd.DataFrame(cells)
    
    def get_cycle_life_data(self):
        """Extract cycle life for all cells"""
        all_cells = []
        json_files = list(self.fastcharge_dir.rglob("*_structure.json"))
        
        print(f"Processing {len(json_files)} files...")
        
        for i, json_file in enumerate(json_files[:10]):  # Limit to 10 for testing
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse the data
                parsed = self._parse_cell_data(data)
                
                if parsed['summary'] is not None:
                    summary = parsed['summary']
                    
                    # Check if discharge_capacity exists
                    if 'discharge_capacity' in summary.columns:
                        initial_cap = summary['discharge_capacity'].iloc[0]
                        
                        # Calculate end of life (80% capacity)
                        eol_mask = summary['discharge_capacity'] <= initial_cap * 0.8
                        
                        cell_info = {
                            'cell_id': json_file.stem,
                            'barcode': parsed['metadata']['barcode'],
                            'protocol': parsed['metadata']['protocol'],
                            'total_cycles': len(summary),
                            'initial_capacity': initial_cap,
                            'final_capacity': summary['discharge_capacity'].iloc[-1],
                            'eol_cycle': summary[eol_mask]['cycle_index'].iloc[0] if eol_mask.any() else len(summary),
                            'capacity_fade_pct': (1 - summary['discharge_capacity'].iloc[-1] / initial_cap) * 100
                        }
                        all_cells.append(cell_info)
                        print(f"  ✓ {json_file.name}: {cell_info['total_cycles']} cycles")
                    else:
                        print(f"  ✗ No discharge_capacity in {json_file.name}")
                        print(f"    Columns: {list(summary.columns)[:5]}...")
                else:
                    print(f"  ✗ No summary data in {json_file.name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {json_file.name}: {e}")
        
        return pd.DataFrame(all_cells)
    
    def extract_features_for_early_prediction(self, n_cycles=100):
        """
        Extract features from first n cycles for early prediction
        """
        features_list = []
        json_files = list(self.fastcharge_dir.rglob("*_structure.json"))
        
        print(f"Extracting features from {len(json_files)} files...")
        
        for json_file in json_files[:10]:  # Limit to 10 for testing
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                parsed = self._parse_cell_data(data)
                
                if parsed['summary'] is not None:
                    summary = parsed['summary']
                    
                    # Use only first n cycles
                    early_data = summary[summary['cycle_index'] <= n_cycles]
                    
                    if len(early_data) > 0 and 'discharge_capacity' in early_data.columns:
                        # Extract statistical features
                        features = {
                            'cell_id': json_file.stem,
                            'barcode': parsed['metadata']['barcode'],
                            'protocol': parsed['metadata']['protocol']
                        }
                        
                        # Capacity features
                        capacities = early_data['discharge_capacity']
                        features.update({
                            'cap_initial': capacities.iloc[0],
                            'cap_mean': capacities.mean(),
                            'cap_std': capacities.std(),
                            'cap_min': capacities.min(),
                            'cap_slope': np.polyfit(range(len(capacities)), capacities, 1)[0] if len(capacities) > 1 else 0,
                            'cap_fade_rate': (capacities.iloc[0] - capacities.iloc[-1]) / len(capacities) if len(capacities) > 1 else 0
                        })
                        
                        # Resistance features (if available)
                        if 'dc_internal_resistance' in early_data.columns:
                            resistance = early_data['dc_internal_resistance'].dropna()
                            if len(resistance) > 1:
                                features.update({
                                    'res_initial': resistance.iloc[0],
                                    'res_final': resistance.iloc[-1],
                                    'res_increase_rate': (resistance.iloc[-1] - resistance.iloc[0]) / len(resistance)
                                })
                        
                        # Temperature features (if available)
                        if 'temperature_average' in early_data.columns:
                            temp = early_data['temperature_average'].dropna()
                            if len(temp) > 0:
                                features.update({
                                    'temp_mean': temp.mean(),
                                    'temp_std': temp.std(),
                                    'temp_max': temp.max()
                                })
                        
                        features_list.append(features)
                        print(f"  ✓ {json_file.name}")
                        
            except Exception as e:
                print(f"  ✗ Error processing {json_file.name}: {e}")
        
        return pd.DataFrame(features_list)
    
    def inspect_file_structure(self, sample_file=None):
        """Inspect the structure of a sample file to understand available keys"""
        if sample_file is None:
            json_files = list(self.fastcharge_dir.rglob("*_structure.json"))
            if not json_files:
                print("No files found!")
                return
            sample_file = json_files[0]
        
        print(f"\n{'='*60}")
        print(f"INSPECTING: {sample_file.name}")
        print(f"{'='*60}")
        
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        print(f"\nTop-level keys: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n🔑 '{key}':")
            print(f"   Type: {type(value).__name__}")
            
            if isinstance(value, dict):
                print(f"   Keys: {list(value.keys())[:10]}")
                # Show sample of nested structure
                for sub_key, sub_value in list(value.items())[:3]:
                    if isinstance(sub_value, list) and len(sub_value) > 0:
                        print(f"      - {sub_key}: list with {len(sub_value)} items")
                        if isinstance(sub_value[0], (int, float, str)):
                            print(f"        Sample: {sub_value[:3]}")
                    elif isinstance(sub_value, dict):
                        print(f"      - {sub_key}: dict with {len(sub_value)} keys")
                    else:
                        print(f"      - {sub_key}: {type(sub_value).__name__} = {sub_value}")
                        
            elif isinstance(value, list):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First element type: {type(value[0]).__name__}")
                    if isinstance(value[0], dict):
                        print(f"   First element keys: {list(value[0].keys())[:5]}")
                        
            else:
                print(f"   Value: {value}")

# Main execution
if __name__ == "__main__":
    # Update this path to your dataset location
    dataset_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website"
    
    # Initialize the dataset loader
    dataset = MITStanfordBatteryDataset(dataset_path)
    
    # First, inspect the structure of a sample file
    print("\n INSPECTING DATASET STRUCTURE")
    dataset.inspect_file_structure()
    
    # Get available cells
    print("\n\n AVAILABLE CELLS")
    cells_df = dataset.get_available_cells()
    print(f"Found {len(cells_df)} cells")
    print(cells_df.head())
    
    # Try to load a specific cell
    print("\n\n LOADING SAMPLE CELL")
    try:
        # Try different cell IDs that might exist
        cell_ids_to_try = ["000000", "000001", "000002", "CH1", "CH10"]
        
        for cell_id in cell_ids_to_try:
            try:
                print(f"\nTrying cell ID: {cell_id}")
                cell_data = dataset.load_cell(cell_id)
                
                if cell_data['summary'] is not None:
                    print(f"\n Successfully loaded cell {cell_id}")
                    print(f"\nSummary data shape: {cell_data['summary'].shape}")
                    print(f"Summary columns: {list(cell_data['summary'].columns)}")
                    print(f"\nFirst 5 cycles:")
                    print(cell_data['summary'].head())
                    break
            except ValueError as e:
                print(f"  Cell {cell_id} not found: {e}")
                
    except Exception as e:
        print(f"Error loading cell: {e}")
    
    # Get cycle life data for all cells
    print("\n\n CYCLE LIFE STATISTICS")
    cycle_life_df = dataset.get_cycle_life_data()
    if len(cycle_life_df) > 0:
        print(f"\nFound {len(cycle_life_df)} cells with cycle data")
        print(cycle_life_df.describe())
    else:
        print("No cycle life data found. Check the file structure.")
    
    # Extract features for early prediction
    print("\n\n EARLY PREDICTION FEATURES")
    features_df = dataset.extract_features_for_early_prediction(n_cycles=100)
    if len(features_df) > 0:
        print(f"\nFeature matrix shape: {features_df.shape}")
        print(features_df.head())
    else:
        print("No features extracted. Check the data structure.")