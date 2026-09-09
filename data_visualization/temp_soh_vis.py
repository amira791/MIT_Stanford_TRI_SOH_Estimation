# preprocessing_figures.py
"""
Generate figures for the data preprocessing subsection:
1. Temperature smoothing (before vs after)
2. SOH trajectories (EOL vs censored)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"

df = pd.read_csv(DATA_PATH)

# ============================================================
# FIGURE 1: TEMPERATURE SMOOTHING
# ============================================================

def add_relative_features(df):
    # Same as your existing function
    df = df.copy()
    cap_rel_list, en_rel_list, ir_rel_list, cycle_pos_list = [], [], [], []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index")
        early = cell_df.iloc[:10]

        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        nom_ir = early["dc_internal_resistance"].mean()
        min_cycle = cell_df["cycle_index"].min()
        max_cycle = cell_df["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)

        cap_rel_list.append((cell_df["charge_capacity"] - nom_cap) / (nom_cap + 1e-9))
        en_rel_list.append((cell_df["charge_energy"] - nom_energy) / (nom_energy + 1e-9))
        ir_rel_list.append((cell_df["dc_internal_resistance"] - nom_ir) / (nom_ir + 1e-9))
        cycle_pos_list.append((cell_df["cycle_index"] - min_cycle) / cyc_range)

    df["cap_rel"] = pd.concat(cap_rel_list)
    df["energy_rel"] = pd.concat(en_rel_list)
    df["ir_rel"] = pd.concat(ir_rel_list)
    df["cycle_pos"] = pd.concat(cycle_pos_list)
    return df

df = add_relative_features(df)

# Pick a representative cell
cell_id = df['barcode'].iloc[0]
cell_data = df[df['barcode'] == cell_id].sort_values('cycle_index')

# Raw temperature (from CSV)
temp_raw = cell_data['temperature_avg'].values

# Simulate smoothed temperature (median filter, window=5)
temp_smoothed = signal.medfilt(temp_raw, kernel_size=5)

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Raw temperature
axes[0].plot(cell_data['cycle_index'], temp_raw, 'b-', linewidth=1.5)
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('(a) Raw Temperature Measurements')
axes[0].grid(True, alpha=0.3)

# Smoothed temperature
axes[1].plot(cell_data['cycle_index'], temp_smoothed, 'g-', linewidth=1.5)
axes[1].set_xlabel('Cycle Number')
axes[1].set_ylabel('Temperature (°C)')
axes[1].set_title('(b) After Median Filtering (window=5)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_smoothing.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 2: SOH TRAJECTORIES
# ============================================================

# Find EOL and censored cells
eol_cells = []
censored_cells = []

for cell in df['barcode'].unique():
    cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
    if len(cell_data) < 100:
        continue
    final_soh = cell_data['soh'].iloc[-1]
    if final_soh <= 0.80:
        eol_cells.append(cell)
    else:
        censored_cells.append(cell)

# Select 8 EOL and 2 censored cells
selected_eol = eol_cells[:8]
selected_censored = censored_cells[:2]
selected_cells = selected_eol + selected_censored

# Create figure
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

colors = ['red'] * 8 + ['blue'] * 2

for idx, cell in enumerate(selected_cells):
    ax = axes[idx]
    cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
    
    ax.plot(cell_data['cycle_index'], cell_data['soh'], color=colors[idx], linewidth=2)
    ax.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('SOH')
    ax.set_title(f'{cell[:15]}...', fontsize=10)
    ax.set_ylim(0.70, 1.05)
    ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='EOL (8 cells)'),
    Line2D([0], [0], color='blue', linewidth=2, label='Censored (2 cells)'),
    Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='EOL Threshold (80%)'),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.suptitle('SOH Degradation Trajectories: 8 EOL vs 2 Censored Cells', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('soh_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()