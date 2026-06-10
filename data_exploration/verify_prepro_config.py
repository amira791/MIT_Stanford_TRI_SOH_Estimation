# verify_config_params_fast.py
"""
Optimized verification script - only loads summary data
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

sys.path.insert(0, str(Path(__file__).parent))
DATA_DIR = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")

def analyze_cycle_distribution_fast(data_dir: Path):
    """Only load summary data from JSON files (much faster)"""
    
    files = sorted(data_dir.glob("*_structure.json"))
    
    results = []
    
    for fp in tqdm(files, desc="Processing files"):
        try:
            # Method 1: Load only first chunk if possible
            with open(fp, "r") as f:
                # Use ijson for streaming parse (even faster)
                # But simpler: load full and extract summary
                raw = json.load(f)
            
            # Extract ONLY summary (discard raw and interpolated)
            summary = raw.get("summary", {})
            
            if not summary or "discharge_capacity" not in summary:
                continue
                
            cycles = np.array(summary.get("cycle_index", []), dtype=int)
            capacities = np.array(summary.get("discharge_capacity", []), dtype=float)
            
            # Keep cycles >= 1
            mask = cycles >= 1
            cycles_clean = cycles[mask]
            caps_clean = capacities[mask]
            
            if len(caps_clean) == 0:
                continue
                
            total_cycles = len(caps_clean)
            
            # Calculate stats only for needed n values
            init_stats = {}
            for n in [3, 5, 10]:
                if total_cycles >= n:
                    init_cap = np.nanmean(caps_clean[:n])
                    init_std = np.nanstd(caps_clean[:n])
                    init_stats[f'init_{n}_mean'] = init_cap
                    init_stats[f'init_{n}_std'] = init_std
                    init_stats[f'init_{n}_cv'] = init_std / init_cap if init_cap > 0 else np.nan
                else:
                    init_stats[f'init_{n}_mean'] = np.nan
                    init_stats[f'init_{n}_std'] = np.nan
                    init_stats[f'init_{n}_cv'] = np.nan
            
            results.append({
                'filename': fp.name,
                'total_cycles': total_cycles,
                **init_stats
            })
            
            # Free memory
            del raw
            del summary
            
        except Exception as e:
            print(f"Error with {fp.name}: {e}")
            continue
    
    return pd.DataFrame(results)

def main():
    print(f"Data directory: {DATA_DIR}")
    print("Loading only summary data (much faster)...")
    
    df = analyze_cycle_distribution_fast(DATA_DIR)
    
    if len(df) == 0:
        print("No valid cells found!")
        return
    
    print(f"\nFound {len(df)} valid cells")
    print(f"Total cycles - Min: {df['total_cycles'].min()}, Max: {df['total_cycles'].max()}, Mean: {df['total_cycles'].mean():.1f}")
    
    # Quick verification for INIT_CYCLES_AVG
    cv_5 = df['init_5_cv'].dropna().mean()
    print(f"\nINIT_CYCLES_AVG=5: Mean CV = {cv_5:.4f}")
    
    # Quick verification for MIN_CYCLES_PER_CELL
    discarded_pct = 100 * (df['total_cycles'] < 20).sum() / len(df)
    print(f"MIN_CYCLES_PER_CELL=20: Discards {discarded_pct:.1f}% of cells")
    
    report_path = Path(__file__).parent / "config_verification_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()