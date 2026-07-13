"""
Visualize Degradation: EOL vs Censored Cells
--------------------------------------------
- Automatically finds 8 EOL cells and 2 censored cells from your dataset
- Red lines: cells that reached EOL
- Blue lines: cells that did NOT reach EOL
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"

# How many cells to show
N_EOL = 1
N_CENSORED = 1

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['barcode'].nunique()} cells")
    return df

# ============================================================
# FIND CELLS
# ============================================================

def find_cells(df, n_eol=1, n_censored=1, skip_first_eol=False):
    """Find cells that reached EOL and cells that didn't
    
    Args:
        df: DataFrame
        n_eol: Number of EOL cells to select
        n_censored: Number of censored cells to select
        skip_first_eol: If True, skip the EOL cell with the most cycles
    """
    
    eol_cells = []
    censored_cells = []
    
    for cell in df['barcode'].unique():
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        final_soh = cell_data['soh'].iloc[-1]
        total_cycles = cell_data['cycle_index'].iloc[-1]
        
        # Only consider cells with enough cycles
        if total_cycles < 100:
            continue
        
        if final_soh <= 0.80:
            eol_cells.append((cell, total_cycles, final_soh))
        else:
            censored_cells.append((cell, total_cycles, final_soh))
    
    # Sort by total cycles (descending)
    eol_cells.sort(key=lambda x: x[1], reverse=True)
    censored_cells.sort(key=lambda x: x[1], reverse=True)
    
    # Skip first EOL cell if requested
    if skip_first_eol and len(eol_cells) > 0:
        print(f"  ⏭️ Skipping first EOL cell: {eol_cells[0][0]} ({eol_cells[0][1]} cycles)")
        eol_cells = eol_cells[1:]  # Remove the first one
        eol_cells = eol_cells[2:]
    
    # Select N cells
    selected_eol = eol_cells[:n_eol]
    selected_censored = censored_cells[:n_censored]
    
    return selected_eol, selected_censored

# ============================================================
# PLOT
# ============================================================

def plot_cells(df, eol_cells, censored_cells):
    """Plot all cells in one figure"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot EOL cells (Red)
    for cell, cycles, final_soh in eol_cells:
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        ax.plot(cell_data['cycle_index'], cell_data['soh'], 
                color='red', alpha=0.7, linewidth=2)
    
    # Plot Censored cells (Blue)
    for cell, cycles, final_soh in censored_cells:
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        ax.plot(cell_data['cycle_index'], cell_data['soh'], 
                color='blue', alpha=0.9, linewidth=2.5, linestyle='--')
        # Mark final point
        ax.scatter(cycles, final_soh, color='blue', s=80, zorder=5)
        ax.annotate(f'{cell[:10]}...\nStopped at {final_soh*100:.1f}%',
                   xy=(cycles, final_soh),
                   xytext=(cycles + 20, final_soh + 0.02),
                   fontsize=9, color='blue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # EOL threshold
    ax.axhline(y=0.80, color='black', linestyle=':', linewidth=2, label='EOL Threshold (80%)')
    
    # Labels
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('SOH', fontsize=12)
    ax.set_title(f'Degradation: {len(eol_cells)} EOL Cells (Red) vs {len(censored_cells)} Censored Cells (Blue)', fontsize=14)
    ax.set_ylim(0.65, 1.05)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label=f'EOL Cells ({len(eol_cells)})'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label=f'Censored Cells ({len(censored_cells)})'),
        Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='EOL Threshold (80%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('degradation_eol_vs_censored.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Plot saved: degradation_eol_vs_censored.png")


# ============================================================
# PRINT INFO
# ============================================================

def print_cell_info(eol_cells, censored_cells):
    print("\n" + "="*60)
    print("FOUND CELLS")
    print("="*60)
    
    print(f"\n--- EOL Cells (Red) - {len(eol_cells)} cells ---")
    for cell, cycles, final_soh in eol_cells:
        print(f"  {cell[:25]:25s} | Cycles: {cycles:4d} | Final SOH: {final_soh*100:5.1f}%")
    
    print(f"\n--- Censored Cells (Blue) - {len(censored_cells)} cells ---")
    for cell, cycles, final_soh in censored_cells:
        print(f"  {cell[:25]:25s} | Cycles: {cycles:4d} | Final SOH: {final_soh*100:5.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("DEGRADATION: EOL vs Censored Cells")
    print("="*60)
    
    df = load_data()
    
    # Find cells
    eol_cells, censored_cells = find_cells(df, N_EOL, N_CENSORED, skip_first_eol=True)
    
    if len(eol_cells) == 0:
        print("\n⚠️ No EOL cells found in dataset!")
        return
    
    if len(censored_cells) == 0:
        print("\n⚠️ No censored cells found in dataset!")
        return
    
    print_cell_info(eol_cells, censored_cells)
    
    # Create plot
    plot_cells(df, eol_cells, censored_cells)
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()