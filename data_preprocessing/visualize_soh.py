# visualize_soh.py
"""
VISUALIZE SOH DEGRADATION CURVES
--------------------------------
Loads the final preprocessed dataset and generates visualizations of
SOH degradation trajectories for multiple cells.

Output:
  results/figures/soh_curves.png
  results/figures/soh_curves_selected.png
  results/figures/eol_distribution.png
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_preprocessing import RESULTS_DIR

def load_dataset() -> pd.DataFrame:
    """Load the preprocessed SOH dataset."""
    pkl_path = RESULTS_DIR / "soh_dataset.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {pkl_path}. Run step2 first.")
    
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    
    print(f"Loaded dataset: {len(df)} rows, {df['cell_id'].nunique()} cells")
    return df

def plot_all_soh_curves(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot SOH degradation curves for all cells on a single figure."""
    plt.figure(figsize=(14, 8))
    
    cell_ids = df['cell_id'].unique()
    
    for cell_id in cell_ids:
        cell_data = df[df['cell_id'] == cell_id].sort_values('cycle_index')
        soh = cell_data['soh'].values
        cycles = cell_data['cycle_index'].values
        
        plt.plot(cycles, soh, linewidth=0.5, alpha=0.6, color='blue')
    
    plt.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='EOL Threshold (80%)')
    plt.xlabel('Cycle Index', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.title(f'All Cells SOH Degradation Curves (n={len(cell_ids)})', fontsize=14)
    plt.ylim(0.70, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_selected_soh_curves(df: pd.DataFrame, n_cells: int = 10, save_path: Optional[Path] = None):
    """Plot SOH degradation curves for a random selection of cells."""
    plt.figure(figsize=(12, 7))
    
    cell_ids = df['cell_id'].unique()
    selected_cells = np.random.choice(cell_ids, min(n_cells, len(cell_ids)), replace=False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cells)))
    
    for cell_id, color in zip(selected_cells, colors):
        cell_data = df[df['cell_id'] == cell_id].sort_values('cycle_index')
        soh = cell_data['soh'].values
        cycles = cell_data['cycle_index'].values
        
        plt.plot(cycles, soh, linewidth=1.5, alpha=0.8, color=color, label=cell_id[:15])
    
    plt.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='EOL Threshold (80%)')
    plt.xlabel('Cycle Index', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.title(f'SOH Degradation Curves (Sample of {len(selected_cells)} Cells)', fontsize=14)
    plt.ylim(0.70, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=8, ncol=2)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_eol_distribution(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot histogram of EOL cycle distribution for cells that reached EOL."""
    cell_eol = df.groupby('cell_id')['eol_cycle'].first().reset_index()
    
    cells_reached_eol = cell_eol[cell_eol['eol_cycle'] < cell_eol['eol_cycle'].max()]
    cells_no_eol = cell_eol[cell_eol['eol_cycle'] == cell_eol['eol_cycle'].max()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(cells_reached_eol['eol_cycle'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('EOL Cycle (SOH < 0.80)', fontsize=12)
    ax1.set_ylabel('Number of Cells', fontsize=12)
    ax1.set_title(f'Distribution of EOL Cycles (n={len(cells_reached_eol)})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    labels = ['Reached EOL', 'Did Not Reach EOL']
    sizes = [len(cells_reached_eol), len(cells_no_eol)]
    colors = ['steelblue', 'lightcoral']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cells that Reached End of Life (80% Capacity)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_soh_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot heatmap of SOH values across cycles and cells."""
    cell_data = []
    cell_ids = df['cell_id'].unique()[:50]
    
    for cell_id in cell_ids:
        cell_df = df[df['cell_id'] == cell_id].sort_values('cycle_index')
        soh_values = cell_df['soh'].values
        cycles = cell_df['cycle_index'].values
        
        for cycle, soh in zip(cycles, soh_values):
            cell_data.append({'cell': cell_id, 'cycle': cycle, 'soh': soh})
    
    heatmap_df = pd.DataFrame(cell_data)
    pivot_df = heatmap_df.pivot(index='cell', columns='cycle', values='soh')
    
    plt.figure(figsize=(16, 8))
    im = plt.imshow(pivot_df.values, aspect='auto', cmap='RdYlGn_r', vmin=0.7, vmax=1.0)
    plt.colorbar(im, label='SOH')
    plt.xlabel('Cycle Index', fontsize=12)
    plt.ylabel('Cell ID (sample of 50)', fontsize=12)
    plt.title('SOH Heatmap - Cycles vs Cells', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def print_soh_statistics(df: pd.DataFrame):
    """Print detailed SOH statistics."""
    cell_stats = df.groupby('cell_id').agg({
        'soh': ['min', 'mean', 'max'],
        'cycle_index': 'count',
        'eol_cycle': 'first'
    }).reset_index()
    
    cell_stats.columns = ['cell_id', 'soh_min', 'soh_mean', 'soh_max', 'total_cycles', 'eol_cycle']
    
    print("\n" + "="*60)
    print("SOH STATISTICS BY CELL")
    print("="*60)
    print(f"\nTotal cells: {len(cell_stats)}")
    print(f"Cells reaching EOL (SOH < 0.80): {(cell_stats['soh_min'] < 0.80).sum()}")
    print(f"Cells never reaching EOL: {(cell_stats['soh_min'] >= 0.80).sum()}")
    
    print(f"\nCycle range:")
    print(f"  Min cycles per cell: {cell_stats['total_cycles'].min()}")
    print(f"  Max cycles per cell: {cell_stats['total_cycles'].max()}")
    print(f"  Mean cycles per cell: {cell_stats['total_cycles'].mean():.0f}")
    
    print(f"\nSOH range across all cycles:")
    print(f"  Minimum SOH: {df['soh'].min():.4f}")
    print(f"  Maximum SOH: {df['soh'].max():.4f}")
    print(f"  Mean SOH: {df['soh'].mean():.4f}")
    print(f"  Median SOH: {df['soh'].median():.4f}")
    
    eol_reached = cell_stats[cell_stats['soh_min'] < 0.80]
    if len(eol_reached) > 0:
        print(f"\nEOL statistics for cells that reached threshold:")
        print(f"  Mean EOL cycle: {eol_reached['eol_cycle'].mean():.0f}")
        print(f"  Median EOL cycle: {eol_reached['eol_cycle'].median():.0f}")
        print(f"  Min EOL cycle: {eol_reached['eol_cycle'].min()}")
        print(f"  Max EOL cycle: {eol_reached['eol_cycle'].max()}")

def main():
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    df = load_dataset()
    
    print_soh_statistics(df)
    
    plot_all_soh_curves(df, save_path=figures_dir / "soh_curves_all.png")
    
    plot_selected_soh_curves(df, n_cells=10, save_path=figures_dir / "soh_curves_selected.png")
    
    plot_eol_distribution(df, save_path=figures_dir / "eol_distribution.png")
    
    plot_soh_heatmap(df, save_path=figures_dir / "soh_heatmap.png")
    
    print(f"\nAll figures saved to: {figures_dir}")

if __name__ == "__main__":
    main()