# analyze_dataset.py
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def load_json_file(filepath):
    """Safely load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def get_summary_data(data):
    """Extract summary data from JSON"""
    if data and 'summary' in data:
        summary = data['summary']
        if all(k in summary for k in ['cycle_index', 'discharge_capacity']):
            return pd.DataFrame({
                'cycle': summary['cycle_index'],
                'capacity': summary['discharge_capacity'],
                'resistance': summary.get('dc_internal_resistance', [np.nan]*len(summary['cycle_index'])),
                'temp_max': summary.get('temperature_maximum', [np.nan]*len(summary['cycle_index'])),
                'temp_avg': summary.get('temperature_average', [np.nan]*len(summary['cycle_index']))
            })
    return None

def analyze_dataset(data_path):
    """Main analysis function"""
    data_dir = Path(data_path)
    json_files = list(data_dir.rglob("*_structure.json"))
    
    print(f" Found {len(json_files)} JSON files\n")
    
    # Group by cell ID
    cell_groups = defaultdict(list)
    for filepath in json_files:
        cell_id = filepath.stem.split('_')[1]
        cell_groups[cell_id].append(filepath)
    
    print(f" Unique cell IDs: {len(cell_groups)}")
    print(f" Expected: 124 cells (official)\n")
    
    # Show cells with multiple channels
    multi_channel = {k: v for k, v in cell_groups.items() if len(v) > 1}
    print(f" Cells with multiple channels: {len(multi_channel)}")
    
    if multi_channel:
        print("\n Multi-channel cells:")
        for cell_id, files in list(multi_channel.items())[:10]:
            channels = [f.stem.split('_')[2] for f in files]
            print(f"   Cell {cell_id}: {len(files)} channels ({', '.join(channels)})")
        if len(multi_channel) > 10:
            print(f"   ... and {len(multi_channel) - 10} more")
    
    # Analyze each multi-channel cell for continuity
    print("\n" + "="*70)
    print(" CONTINUITY ANALYSIS (Multi-channel cells)")
    print("="*70)
    
    continuity_issues = []
    for cell_id, files in list(multi_channel.items())[:5]:  # Check first 5
        print(f"\n Cell {cell_id}:")
        
        channels_data = []
        for filepath in files:
            data = load_json_file(filepath)
            df = get_summary_data(data)
            if df is not None and len(df) > 0:
                channels_data.append({
                    'channel': filepath.stem.split('_')[2],
                    'cycle_start': int(df['cycle'].min()),
                    'cycle_end': int(df['cycle'].max()),
                    'n_cycles': len(df),
                    'capacity_start': float(df['capacity'].iloc[0]),
                    'capacity_end': float(df['capacity'].iloc[-1])
                })
        
        if len(channels_data) > 1:
            sorted_channels = sorted(channels_data, key=lambda x: x['cycle_start'])
            for i, ch in enumerate(sorted_channels):
                print(f"   {ch['channel']}: cycles {ch['cycle_start']}-{ch['cycle_end']} ({ch['n_cycles']} cycles)")
            
            # Check for gaps/overlaps
            for i in range(len(sorted_channels)-1):
                current_end = sorted_channels[i]['cycle_end']
                next_start = sorted_channels[i+1]['cycle_start']
                if next_start <= current_end:
                    issue = f"   ⚠️ OVERLAP: {sorted_channels[i]['channel']} ends at {current_end}, {sorted_channels[i+1]['channel']} starts at {next_start}"
                    print(issue)
                    continuity_issues.append(issue)
                elif next_start > current_end + 5:
                    issue = f"   ⚠️ GAP: {current_end} → {next_start} ({next_start - current_end} cycles missing)"
                    print(issue)
                    continuity_issues.append(issue)
                else:
                    print(f"   ✓ Continuous: {sorted_channels[i]['channel']} → {sorted_channels[i+1]['channel']}")
    
    # Detect anomalies
    print("\n" + "="*70)
    print(" ANOMALY DETECTION")
    print("="*70)
    
    anomaly_count = 0
    cycle0_anomalies = 0
    sudden_drops = 0
    
    for filepath in json_files[:30]:  # Check first 30 files
        cell_id = filepath.stem.split('_')[1]
        channel = filepath.stem.split('_')[2]
        
        data = load_json_file(filepath)
        df = get_summary_data(data)
        
        if df is not None and len(df) > 0:
            anomalies = []
            
            # Check cycle 0
            cycle0 = df[df['cycle'] == 0]
            if not cycle0.empty:
                cap0 = cycle0['capacity'].iloc[0]
                if cap0 > 1.3:
                    anomalies.append(f'cycle_0_anomaly: {cap0:.3f}Ah')
                    cycle0_anomalies += 1
            
            # Check sudden drops
            capacity = df['capacity'].values
            for i in range(1, min(50, len(capacity))):
                drop = (capacity[i-1] - capacity[i]) / capacity[i-1]
                if drop > 0.15:
                    anomalies.append(f'sudden_drop at cycle {df["cycle"].iloc[i]}: {drop*100:.1f}%')
                    sudden_drops += 1
                    break
            
            if anomalies:
                anomaly_count += 1
                if anomaly_count <= 10:  # Show first 10
                    print(f"\n {cell_id}_{channel}:")
                    for a in anomalies:
                        print(f"    {a}")
    
    print(f"\n Summary: {anomaly_count} files with anomalies (out of 30 checked)")
    print(f"   - Cycle 0 anomalies: {cycle0_anomalies}")
    print(f"   - Sudden capacity drops: {sudden_drops}")
    
    # Final recommendations
    print("\n" + "="*70)
    print(" RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. CYCLE INDEXING:")
    if cycle0_anomalies > 0:
        print("    Skip cycle 0 entirely (it's corrupted formation data)")
        print("   → Set START_CYCLE_IDX = 1 in config")
    else:
        print("   → Cycle 0 appears normal, can include in baseline")
    
    print("\n2. MULTI-CHANNEL CELLS:")
    if continuity_issues:
        print("    Found gaps/overlaps - need to merge channels carefully")
        print("   → Merge channels in chronological order")
        print("   → Or keep only the channel with most cycles")
    else:
        print("   → Channels appear continuous - safe to merge")
    
    print("\n3. INIT_CYCLES_AVG:")
    print("   → Use 5 cycles (cycles 1-5) for initial capacity baseline")
    print("   → These cycles are stable and exclude cycle 0")
    
    print("\n4. MIN_CYCLES_PER_CELL:")
    print("   → Set to 20 (minimum cycles after removing cycle 0)")
    
    return {
        'total_files': len(json_files),
        'unique_cells': len(cell_groups),
        'multi_channel_cells': len(multi_channel),
        'has_cycle_0_anomaly': cycle0_anomalies > 0,
        'continuity_issues': len(continuity_issues),
        'recommendations': {
            'skip_cycle_0': cycle0_anomalies > 0,
            'init_cycles_avg': 5,
            'start_cycle_idx': 1,
            'min_cycles_per_cell': 20
        }
    }

# Run analysis
if __name__ == "__main__":
    dataset_path = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")
    
    if not dataset_path.exists():
        print(f" Path not found: {dataset_path}")
    else:
        results = analyze_dataset(dataset_path)
        
        print("\n" + "="*70)
        print(" FINAL SUMMARY")
        print("="*70)
        print(f"\nTotal files: {results['total_files']}")
        print(f"Unique cells: {results['unique_cells']}")
        print(f"Cells with multiple channels: {results['multi_channel_cells']}")
        print(f"Cycle 0 anomaly detected: {results['has_cycle_0_anomaly']}")
        
        print("\n RECOMMENDED CONFIG SETTINGS:")
        print(f"   START_CYCLE_IDX = {results['recommendations']['start_cycle_idx']}")
        print(f"   INIT_CYCLES_AVG = {results['recommendations']['init_cycles_avg']}")
        print(f"   MIN_CYCLES_PER_CELL = {results['recommendations']['min_cycles_per_cell']}")