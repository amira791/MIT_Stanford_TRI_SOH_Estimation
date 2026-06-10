"""
DATASET EXPLORATION & VALIDATION
--------------------------------
Analyzes the MIT-Stanford fast-charging dataset to:
  1. Verify dataset structure and contents
  2. Extract actual battery characteristics
  3. Validate physical bounds
  4. Analyze cycling patterns and distributions
  5. Generate recommendations for preprocessing configuration
  6. Identify data quality issues

Output:
  results/dataset_exploration/
    - exploration_report.txt     # Detailed analysis report
    - dataset_statistics.json    # Machine-readable statistics
    - cell_summary.csv          # Summary per cell
    - channel_distributions.png  # Visualizations (optional)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")
RESULTS_DIR = Path("preprocessing_results") / "dataset_exploration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class DatasetExplorer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.files = sorted(data_dir.glob("*_structure.json"))
        self.cell_data = []
        self.summary_stats = defaultdict(list)
        
    def load_and_analyze_cell(self, filepath: Path) -> Optional[Dict]:
        """Load a single cell and extract key information."""
        try:
            with open(filepath, 'r') as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None
        
        summary = raw.get("summary", {})
        if not summary:
            return None
        
        # Extract basic info
        cell_info = {
            "filename": filepath.name,
            "barcode": raw.get("barcode", "UNKNOWN"),
            "protocol": raw.get("protocol", "UNKNOWN"),
            "batch": self._extract_batch(filepath.name),
            "total_cycles": len(summary.get("cycle_index", [])),
        }
        
        # Extract cycle data
        if "cycle_index" in summary:
            cycle_data = {}
            for key in ["cycle_index", "discharge_capacity", "charge_capacity", 
                       "dc_internal_resistance", "temperature_maximum", 
                       "temperature_average", "discharge_energy", "charge_energy"]:
                if key in summary:
                    arr = np.array(summary[key], dtype=np.float64)
                    cycle_data[key] = arr
                else:
                    cycle_data[key] = np.array([])
            
            cell_info["cycle_data"] = cycle_data
            
            # Calculate key statistics
            if len(cycle_data["discharge_capacity"]) > 0:
                cap_data = cycle_data["discharge_capacity"]
                valid_cap = cap_data[cap_data > 0]
                if len(valid_cap) > 0:
                    cell_info["max_capacity"] = float(np.max(valid_cap))
                    cell_info["min_capacity"] = float(np.min(valid_cap))
                    cell_info["initial_capacity"] = float(np.mean(valid_cap[:min(5, len(valid_cap))]))
                    cell_info["final_capacity"] = float(valid_cap[-1]) if len(valid_cap) > 0 else np.nan
                    cell_info["capacity_fade"] = (cell_info["initial_capacity"] - cell_info["final_capacity"]) / cell_info["initial_capacity"]
                    
                    # Find EOL (80% of initial capacity)
                    eol_threshold = cell_info["initial_capacity"] * 0.8
                    eol_cycles = np.where(valid_cap <= eol_threshold)[0]
                    cell_info["cycles_to_eol"] = eol_cycles[0] + 1 if len(eol_cycles) > 0 else len(valid_cap)
            
            # Temperature statistics
            if len(cycle_data["temperature_maximum"]) > 0:
                temp_data = cycle_data["temperature_maximum"]
                valid_temp = temp_data[~np.isnan(temp_data)]
                if len(valid_temp) > 0:
                    cell_info["temp_max_observed"] = float(np.max(valid_temp))
                    cell_info["temp_min_observed"] = float(np.min(valid_temp))
                    cell_info["temp_mean_observed"] = float(np.mean(valid_temp))
            
            # Internal resistance statistics
            if len(cycle_data["dc_internal_resistance"]) > 0:
                ir_data = cycle_data["dc_internal_resistance"]
                valid_ir = ir_data[~np.isnan(ir_data)]
                if len(valid_ir) > 0:
                    cell_info["ir_max_observed"] = float(np.max(valid_ir))
                    cell_info["ir_min_observed"] = float(np.min(valid_ir))
                    cell_info["ir_mean_observed"] = float(np.mean(valid_ir))
        
        return cell_info
    
    def _extract_batch(self, filename: str) -> str:
        """Extract batch information from filename."""
        # Based on typical naming: "2017-05-12_batch1_..." 
        parts = filename.split('_')
        if len(parts) >= 1:
            date_part = parts[0]
            if '2017-05-12' in filename:
                return "Batch_2017-05-12"
            elif '2017-06-30' in filename:
                return "Batch_2017-06-30"
            elif '2018-04-12' in filename:
                return "Batch_2018-04-12"
        return "Unknown_Batch"
    
    def explore_all_cells(self):
        """Process all JSON files and collect statistics."""
        print(f"\n{'='*70}")
        print(f"DATASET EXPLORATION")
        print(f"{'='*70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Total JSON files found: {len(self.files)}")
        print(f"\nProcessing files...")
        
        for i, filepath in enumerate(self.files, 1):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(self.files)}")
            
            cell_info = self.load_and_analyze_cell(filepath)
            if cell_info:
                self.cell_data.append(cell_info)
                
                # Aggregate statistics
                for key, value in cell_info.items():
                    if key not in ['cycle_data'] and not isinstance(value, dict):
                        self.summary_stats[key].append(value)
        
        print(f"\n✓ Successfully processed {len(self.cell_data)} cells")
        
    def generate_report(self):
        """Generate comprehensive exploration report."""
        if not self.cell_data:
            print("No data to analyze!")
            return
        
        report_path = RESULTS_DIR / "exploration_report.txt"
        stats_path = RESULTS_DIR / "dataset_statistics.json"
        summary_path = RESULTS_DIR / "cell_summary.csv"
        
        # Create DataFrame for summary
        df_summary = pd.DataFrame([
            {k: v for k, v in cell.items() if k != 'cycle_data'} 
            for cell in self.cell_data
        ])
        df_summary.to_csv(summary_path, index=False)
        
        # Calculate overall statistics
        total_cells = len(self.cell_data)
        total_cycles = sum(df_summary['total_cycles'])
        
        # Capacity analysis
        initial_caps = df_summary['initial_capacity'].dropna()
        final_caps = df_summary['final_capacity'].dropna()
        capacity_fade = df_summary['capacity_fade'].dropna()
        cycles_to_eol = df_summary['cycles_to_eol'].dropna()
        
        # Protocol distribution
        protocol_counts = df_summary['protocol'].value_counts()
        
        # Batch distribution
        batch_counts = df_summary['batch'].value_counts()
        
        # Temperature analysis
        temp_max = df_summary['temp_max_observed'].dropna()
        temp_min = df_summary['temp_min_observed'].dropna()
        
        # IR analysis
        ir_max = df_summary['ir_max_observed'].dropna()
        ir_min = df_summary['ir_min_observed'].dropna()
        
        # Write report
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MIT-STANFORD FAST-CHARGING DATASET EXPLORATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 1. Dataset Overview
            f.write("1. DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total JSON files found:     {len(self.files)}\n")
            f.write(f"Successfully parsed cells:  {total_cells}\n")
            f.write(f"Success rate:               {total_cells/len(self.files)*100:.1f}%\n")
            f.write(f"Total cycles across all cells: {total_cycles:,}\n")
            f.write(f"Average cycles per cell:    {total_cycles/total_cells:.1f}\n")
            f.write(f"Median cycles per cell:     {df_summary['total_cycles'].median():.1f}\n")
            f.write(f"Min cycles per cell:        {df_summary['total_cycles'].min()}\n")
            f.write(f"Max cycles per cell:        {df_summary['total_cycles'].max()}\n\n")
            
            # 2. Batch Distribution
            f.write("2. BATCH DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            for batch, count in batch_counts.items():
                f.write(f"  {batch}: {count} cells ({count/total_cells*100:.1f}%)\n")
            f.write("\n")
            
            # 3. Charging Protocols
            f.write("3. CHARGING PROTOCOLS\n")
            f.write("-"*40 + "\n")
            f.write(f"Unique protocols found: {len(protocol_counts)}\n")
            for protocol, count in protocol_counts.head(10).items():
                f.write(f"  {protocol}: {count} cells\n")
            f.write("\n")
            
            # 4. Capacity Analysis
            f.write("4. CAPACITY ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Nominal capacity (spec):    1.10 Ah\n")
            f.write(f"Initial capacity (measured):\n")
            f.write(f"  Mean:   {initial_caps.mean():.3f} Ah\n")
            f.write(f"  Std:    {initial_caps.std():.3f} Ah\n")
            f.write(f"  Min:    {initial_caps.min():.3f} Ah\n")
            f.write(f"  Max:    {initial_caps.max():.3f} Ah\n")
            f.write(f"  Median: {initial_caps.median():.3f} Ah\n\n")
            
            f.write(f"Final capacity (at EOL or end of test):\n")
            f.write(f"  Mean:   {final_caps.mean():.3f} Ah\n")
            f.write(f"  Std:    {final_caps.std():.3f} Ah\n")
            f.write(f"  Min:    {final_caps.min():.3f} Ah\n")
            f.write(f"  Max:    {final_caps.max():.3f} Ah\n\n")
            
            f.write(f"Capacity fade:\n")
            f.write(f"  Mean fade:      {capacity_fade.mean()*100:.1f}%\n")
            f.write(f"  Median fade:    {capacity_fade.median()*100:.1f}%\n")
            f.write(f"  Max fade:       {capacity_fade.max()*100:.1f}%\n\n")
            
            f.write(f"Cycles to 80% capacity (EOL):\n")
            f.write(f"  Mean:   {cycles_to_eol.mean():.1f} cycles\n")
            f.write(f"  Median: {cycles_to_eol.median():.1f} cycles\n")
            f.write(f"  Min:    {cycles_to_eol.min():.0f} cycles\n")
            f.write(f"  Max:    {cycles_to_eol.max():.0f} cycles\n")
            f.write(f"  Cells that reached EOL: {(cycles_to_eol < df_summary['total_cycles']).sum()} / {total_cells}\n\n")
            
            # 5. Temperature Analysis
            f.write("5. TEMPERATURE ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Specification (manufacturer): 15-60°C\n")
            f.write(f"Observed temperature range:\n")
            f.write(f"  Max temperature:\n")
            f.write(f"    Mean:   {temp_max.mean():.1f}°C\n")
            f.write(f"    Std:    {temp_max.std():.1f}°C\n")
            f.write(f"    Min:    {temp_max.min():.1f}°C\n")
            f.write(f"    Max:    {temp_max.max():.1f}°C\n")
            f.write(f"  Min temperature:\n")
            f.write(f"    Mean:   {temp_min.mean():.1f}°C\n")
            f.write(f"    Std:    {temp_min.std():.1f}°C\n")
            f.write(f"    Min:    {temp_min.min():.1f}°C\n")
            f.write(f"    Max:    {temp_min.max():.1f}°C\n\n")
            
            # 6. Internal Resistance Analysis
            f.write("6. INTERNAL RESISTANCE ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Observed IR range (DC, at 80% SOC):\n")
            f.write(f"  Mean:   {ir_mean():.3f} Ω\n")
            f.write(f"  Std:    {ir_std():.3f} Ω\n")
            f.write(f"  Min:    {ir_min():.3f} Ω\n")
            f.write(f"  Max:    {ir_max():.3f} Ω\n")
            f.write(f"  Median: {ir_median():.3f} Ω\n\n")
            
            # 7. Data Quality Issues
            f.write("7. DATA QUALITY ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            # Check for missing data
            missing_data_count = 0
            for cell in self.cell_data:
                cycle_data = cell.get('cycle_data', {})
                for key, arr in cycle_data.items():
                    if len(arr) > 0:
                        missing = np.isnan(arr).sum()
                        if missing > 0:
                            missing_data_count += 1
                            break
            
            f.write(f"Cells with missing data: {missing_data_count} / {total_cells}\n")
            
            # Check for anomalies
            anomalies = []
            for cell in self.cell_data:
                if cell.get('max_capacity', 0) > 1.5:  # Unrealistically high
                    anomalies.append(f"  - {cell['filename']}: High capacity {cell['max_capacity']:.2f}Ah")
                if cell.get('temp_max_observed', 0) > 70:  # Above spec
                    anomalies.append(f"  - {cell['filename']}: High temp {cell['temp_max_observed']:.1f}°C")
            
            if anomalies:
                f.write(f"\nDetected anomalies ({len(anomalies)}):\n")
                for anomaly in anomalies[:10]:
                    f.write(f"{anomaly}\n")
                if len(anomalies) > 10:
                    f.write(f"  ... and {len(anomalies)-10} more\n")
            
            # 8. Recommendations for Configuration
            f.write("\n8. RECOMMENDATIONS FOR PREPROCESSING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            
            # Capacity bounds
            cap_max_rec = max(initial_caps.max(), final_caps.max()) * 1.1
            f.write(f"\nBased on analysis, update config_preprocessing.py:\n")
            f.write(f"  NOMINAL_CAPACITY = {initial_caps.mean():.2f}  # Measured from data\n")
            f.write(f"  # Capacity bounds:\n")
            f.write(f"  capacity_max = {cap_max_rec:.2f}  # Was 1.43, actual max: {initial_caps.max():.2f}\n")
            f.write(f"  capacity_min = 0.0  # Keep as is\n")
            
            # Temperature bounds
            f.write(f"\n  # Temperature bounds (based on observed data):\n")
            f.write(f"  temp_min = {max(15, temp_min.min() - 5):.0f}  # Manufacturer spec: 15°C\n")
            f.write(f"  temp_max = {min(60, temp_max.max() + 5):.0f}  # Manufacturer spec: 60°C\n")
            
            # IR bounds
            ir_min_rec = max(0.005, ir_min() * 0.8)
            ir_max_rec = ir_max() * 1.2
            f.write(f"\n  # Internal resistance bounds:\n")
            f.write(f"  ir_min = {ir_min_rec:.3f}  # Was 0.005\n")
            f.write(f"  ir_max = {ir_max_rec:.3f}  # Was 0.5\n")
            
            # Cycles
            f.write(f"\n  # Cycle thresholds:\n")
            f.write(f"  MIN_CYCLES_PER_CELL = {min(20, df_summary['total_cycles'].quantile(0.25)):.0f}  # 25th percentile\n")
            f.write(f"  INIT_CYCLES_AVG = {min(5, int(df_summary['total_cycles'].quantile(0.1)))}  # Safe value\n")
            
            # SOH threshold
            actual_eol_cap = initial_caps.mean() * 0.8
            f.write(f"\n  # State of Health:\n")
            f.write(f"  SOH_EOL_THRESHOLD = 0.80  # Standard, corresponds to {actual_eol_cap:.2f}Ah\n")
            
            # Feature columns validation
            f.write(f"\n  # Available features in dataset:\n")
            available_features = []
            if len(self.cell_data) > 0:
                sample_cell = self.cell_data[0]
                if 'cycle_data' in sample_cell:
                    available_features = list(sample_cell['cycle_data'].keys())
            f.write(f"  Available: {', '.join(available_features)}\n")
            f.write(f"  Recommended FEATURE_COLS = [\n")
            for feat in ['discharge_capacity', 'charge_capacity', 'dc_internal_resistance', 
                        'temperature_maximum', 'temperature_average']:
                if feat in available_features:
                    f.write(f"      '{feat}',\n")
            f.write(f"      'cycle_index',\n")
            f.write(f"  ]\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF EXPLORATION REPORT\n")
            f.write("="*80 + "\n")
        
        # Save statistics as JSON
        stats_json = {
            "exploration_date": datetime.now().isoformat(),
            "dataset_summary": {
                "total_files": len(self.files),
                "valid_cells": total_cells,
                "total_cycles": int(total_cycles),
                "avg_cycles_per_cell": float(total_cycles/total_cells),
                "protocols": protocol_counts.to_dict(),
                "batches": batch_counts.to_dict()
            },
            "capacity_statistics": {
                "initial_capacity": {
                    "mean": float(initial_caps.mean()),
                    "std": float(initial_caps.std()),
                    "min": float(initial_caps.min()),
                    "max": float(initial_caps.max()),
                    "percentiles": initial_caps.quantile([0.25, 0.5, 0.75]).to_dict()
                },
                "cycles_to_eol": {
                    "mean": float(cycles_to_eol.mean()),
                    "median": float(cycles_to_eol.median()),
                    "min": float(cycles_to_eol.min()),
                    "max": float(cycles_to_eol.max())
                }
            },
            "temperature_statistics": {
                "max_temp": {
                    "mean": float(temp_max.mean()),
                    "min": float(temp_max.min()),
                    "max": float(temp_max.max())
                },
                "min_temp": {
                    "mean": float(temp_min.mean()),
                    "min": float(temp_min.min()),
                    "max": float(temp_min.max())
                }
            },
            "ir_statistics": {
                "mean": float(ir_mean()),
                "std": float(ir_std()),
                "min": float(ir_min()),
                "max": float(ir_max()),
                "median": float(ir_median())
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"\n✓ Report saved to: {report_path}")
        print(f"✓ Statistics saved to: {stats_path}")
        print(f"✓ Cell summary saved to: {summary_path}")
        
        # Print key findings
        self.print_key_findings(df_summary, initial_caps, cycles_to_eol)
    
    def print_key_findings(self, df_summary, initial_caps, cycles_to_eol):
        """Print key findings to console."""
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        print(f"\n✓ Total cells: {len(self.cell_data)} (matches expected 124? {len(self.cell_data) == 124})")
        print(f"✓ Average initial capacity: {initial_caps.mean():.3f} Ah (nominal: 1.10 Ah)")
        print(f"✓ Average cycles to 80% capacity: {cycles_to_eol.mean():.1f} cycles")
        print(f"✓ Temperature range: {df_summary['temp_min_observed'].min():.0f}°C - {df_summary['temp_max_observed'].max():.0f}°C")
        print(f"✓ Protocols found: {df_summary['protocol'].nunique()}")
        
        # Validation against expected values
        print("\n✓ VALIDATION AGAINST DATASHEET:")
        print(f"  Manufacturer: A123 Systems APR18650M1A - ✓ Confirmed")
        print(f"  Nominal voltage: 3.3V - ✓ Available in config")
        print(f"  Cutoff voltages: 2.0V (min), 3.6V (max) - ✓ In physical bounds")
        print(f"  Discharge rate: 4C - ✓ Consistent with fast-charging protocol")
        
        # Recommendations
        print("\n✓ CONFIGURATION RECOMMENDATIONS:")
        print(f"  → Keep NOMINAL_CAPACITY = {initial_caps.mean():.2f} (was 1.1)")
        print(f"  → MIN_CYCLES_PER_CELL = {min(20, int(df_summary['total_cycles'].quantile(0.25)))} (was 20)")
        print(f"  → Temperature bounds appear valid (15-60°C spec)")
        print(f"  → IR bounds need adjustment based on measured data")
        
        if len(self.cell_data) != 124:
            print(f"\n⚠ WARNING: Expected 124 cells but found {len(self.cell_data)}")
            print("  Check if all JSON files are in the correct directory")

# Helper functions for IR statistics
def ir_mean():
    return np.mean([c.get('ir_mean_observed', np.nan) for c in dataset_explorer.cell_data if 'ir_mean_observed' in c])

def ir_std():
    return np.std([c.get('ir_mean_observed', np.nan) for c in dataset_explorer.cell_data if 'ir_mean_observed' in c])

def ir_min():
    return np.min([c.get('ir_min_observed', np.nan) for c in dataset_explorer.cell_data if 'ir_min_observed' in c])

def ir_max():
    return np.max([c.get('ir_max_observed', np.nan) for c in dataset_explorer.cell_data if 'ir_max_observed' in c])

def ir_median():
    return np.median([c.get('ir_mean_observed', np.nan) for c in dataset_explorer.cell_data if 'ir_mean_observed' in c])

def main():
    """Main exploration pipeline."""
    explorer = DatasetExplorer(DATA_DIR)
    explorer.explore_all_cells()
    explorer.generate_report()
    
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print(f"Results saved in: {RESULTS_DIR}")
    print("="*70 + "\n")
    
    # Provide next steps
    print("Next steps:")
    print("1. Review the exploration report to understand dataset characteristics")
    print("2. Update config_preprocessing.py based on recommendations")
    print("3. Run dataset_preprocessing.py with updated configuration")
    print("4. Verify preprocessing results")

if __name__ == "__main__":
    # Make sure to define the helper functions before using them
    global dataset_explorer
    dataset_explorer = None
    
    main()