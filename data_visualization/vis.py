"""
Battery Dataset Visualization
-----------------------------
Visualizes the MIT-Stanford SOH dataset used for training CNN-Mamba-UQ.
Handles both raw and preprocessed datasets (adds relative features if missing).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_visualization\visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def add_relative_features(df):
    """Add per-cell relative features if they don't exist"""
    df = df.copy()
    
    # Check if relative features already exist
    if all(col in df.columns for col in ["cap_rel", "energy_rel", "ir_rel", "cycle_pos"]):
        print("  Relative features already exist.")
        return df
    
    print("  Adding relative features...")
    
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


def load_data():
    """Load and prepare the SOH dataset"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Cells: {df['barcode'].nunique()}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")
    
    # Add relative features if missing
    df = add_relative_features(df)
    
    return df


# Feature columns (will be determined dynamically)
def get_feat_cols(df):
    """Get available feature columns"""
    raw_feats = [
        "dc_internal_resistance", "temperature_avg",
        "charge_capacity", "charge_energy",
        "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    ]
    rel_feats = ["cap_rel", "energy_rel", "ir_rel", "cycle_pos"]
    
    available = [f for f in raw_feats if f in df.columns]
    if all(f in df.columns for f in rel_feats):
        available += rel_feats
    
    return available


FEAT_NAMES = {
    "dc_internal_resistance": "Internal Resistance (Ω)",
    "temperature_avg": "Temperature (°C)",
    "charge_capacity": "Charge Capacity (Ah)",
    "charge_energy": "Charge Energy (Wh)",
    "coulombic_efficiency_lagged_1": "CE (t-1)",
    "coulombic_efficiency_lagged_2": "CE (t-2)",
    "cap_rel": "Relative Capacity",
    "energy_rel": "Relative Energy",
    "ir_rel": "Relative Resistance",
    "cycle_pos": "Cycle Position",
    "soh": "SOH",
}

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_soh_trajectories(df, n_cells=20):
    """Plot SOH trajectories for multiple cells"""
    print("\n[1] Plotting SOH trajectories...")
    
    cells = df['barcode'].unique()[:n_cells]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Multiple cell trajectories
    ax1 = axes[0, 0]
    for cell in cells:
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        ax1.plot(cell_data['cycle_index'], cell_data['soh'], alpha=0.6, linewidth=1)
    ax1.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='EOL (80%)')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('SOH')
    ax1.set_title(f'SOH Trajectories (First {n_cells} Cells)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SOH distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(df['soh'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0.80, color='red', linestyle='--', linewidth=2, label='EOL (80%)')
    ax2.axvline(x=df['soh'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["soh"].mean():.3f}')
    ax2.set_xlabel('SOH')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SOH Distribution')
    ax2.legend()
    
    # Plot 3: Example cell with degradation path
    ax3 = axes[1, 0]
    example_cell = cells[0]
    cell_data = df[df['barcode'] == example_cell].sort_values('cycle_index')
    ax3.plot(cell_data['cycle_index'], cell_data['soh'], 'b-', linewidth=2)
    ax3.fill_between(cell_data['cycle_index'], 0, cell_data['soh'], alpha=0.2)
    ax3.axhline(y=0.80, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Cycle Number')
    ax3.set_ylabel('SOH')
    ax3.set_title(f'SOH Degradation - Cell: {example_cell[:15]}...')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Degradation rate per cell
    ax4 = axes[1, 1]
    degradation_rates = []
    for cell in df['barcode'].unique():
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        if len(cell_data) > 10:
            start_soh = cell_data['soh'].iloc[0]
            end_soh = cell_data['soh'].iloc[-1]
            end_cycle = cell_data['cycle_index'].iloc[-1]
            rate = (start_soh - end_soh) / end_cycle * 100
            degradation_rates.append(rate)
    
    ax4.hist(degradation_rates, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax4.set_xlabel('Degradation Rate (% per 100 cycles)')
    ax4.set_ylabel('Number of Cells')
    ax4.set_title('Degradation Rate Distribution')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'soh_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'soh_trajectories.png'}")


def plot_feature_distributions(df):
    """Plot feature distributions"""
    print("\n[2] Plotting feature distributions...")
    
    feat_cols = get_feat_cols(df)
    
    # Select features to show (up to 6)
    features_to_plot = [f for f in feat_cols if f in [
        'dc_internal_resistance', 'temperature_avg', 'charge_capacity', 
        'coulombic_efficiency_lagged_1', 'cap_rel', 'cycle_pos'
    ]]
    
    if len(features_to_plot) < 6:
        # Fill with available features
        for f in feat_cols[:6]:
            if f not in features_to_plot:
                features_to_plot.append(f)
    
    features_to_plot = features_to_plot[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]
        data = df[feat].dropna()
        if len(data) > 0:
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {data.mean():.4f}')
            ax.set_xlabel(FEAT_NAMES.get(feat, feat))
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {FEAT_NAMES.get(feat, feat)}')
            ax.legend()
    
    # Hide empty subplots
    for idx in range(len(features_to_plot), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'feature_distributions.png'}")


def plot_feature_correlations(df):
    """Plot feature correlation matrix"""
    print("\n[3] Plotting feature correlations...")
    
    feat_cols = get_feat_cols(df)
    corr_cols = [f for f in feat_cols if f in df.columns] + ['soh']
    corr_df = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    labels = [FEAT_NAMES.get(c, c) for c in corr_df.columns]
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'},
                xticklabels=labels, yticklabels=labels)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'feature_correlations.png'}")


def plot_split_distribution(df):
    """Plot train/val/test split distribution"""
    print("\n[4] Plotting split distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Cycle count per split
    ax1 = axes[0]
    split_counts = df['split'].value_counts()
    colors = {'train': '#2ecc71', 'val': '#f39c12', 'test': '#e74c3c'}
    ax1.bar(split_counts.index, split_counts.values, 
            color=[colors.get(s, 'gray') for s in split_counts.index], 
            edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Split')
    ax1.set_ylabel('Number of Cycles')
    ax1.set_title('Cycles per Split')
    for i, (label, count) in enumerate(split_counts.items()):
        ax1.text(i, count + 1000, f'{count:,}', ha='center', fontsize=10)
    
    # Plot 2: Cell count per split
    ax2 = axes[1]
    cell_counts = df.groupby('split')['barcode'].nunique()
    ax2.bar(cell_counts.index, cell_counts.values, 
            color=[colors.get(s, 'gray') for s in cell_counts.index], 
            edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Split')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Cells per Split')
    for i, (label, count) in enumerate(cell_counts.items()):
        ax2.text(i, count + 1, f'{count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'split_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'split_distribution.png'}")


def plot_feature_evolution(df, n_cells=5):
    """Plot how features evolve with cycles"""
    print("\n[5] Plotting feature evolution...")
    
    cells = df['barcode'].unique()[:n_cells]
    feat_cols = get_feat_cols(df)
    
    features_to_plot = ['dc_internal_resistance', 'temperature_avg', 
                        'charge_capacity', 'coulombic_efficiency_lagged_1', 
                        'soh']
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    
    # Add cycle_pos if available
    if 'cycle_pos' in df.columns:
        features_to_plot.append('cycle_pos')
    
    # Ensure we have enough features
    if len(features_to_plot) < 6:
        extra = [f for f in feat_cols if f not in features_to_plot][:6-len(features_to_plot)]
        features_to_plot += extra
    
    features_to_plot = features_to_plot[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]
        for cell in cells:
            cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
            if len(cell_data) > 100:
                cell_data = cell_data.iloc[::5]
            ax.plot(cell_data['cycle_index'], cell_data[feat], alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel(FEAT_NAMES.get(feat, feat))
        ax.set_title(f'{FEAT_NAMES.get(feat, feat)} vs Cycles')
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(features_to_plot), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'feature_evolution.png'}")


def plot_sample_sequences(df, n_samples=4):
    """Plot sample sequences (50-cycle windows)"""
    print("\n[6] Plotting sample sequences...")
    
    cells = df['barcode'].unique()
    valid_cells = []
    for cell in cells:
        if len(df[df['barcode'] == cell]) > 100:
            valid_cells.append(cell)
    
    selected_cells = valid_cells[:n_samples]
    
    if len(selected_cells) == 0:
        print("  No cells with enough cycles.")
        return
    
    fig, axes = plt.subplots(len(selected_cells), 1, figsize=(14, 3*len(selected_cells)))
    if len(selected_cells) == 1:
        axes = [axes]
    
    for idx, cell in enumerate(selected_cells):
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        cell_data = cell_data.iloc[:100]
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        ax.plot(cell_data['cycle_index'], cell_data['soh'], 'b-', linewidth=2, label='SOH')
        ax2.plot(cell_data['cycle_index'], cell_data['dc_internal_resistance'], 
                'r-', linewidth=1.5, alpha=0.7, label='Resistance')
        ax2.plot(cell_data['cycle_index'], cell_data['temperature_avg'], 
                'g-', linewidth=1.5, alpha=0.7, label='Temperature')
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('SOH')
        ax2.set_ylabel('Resistance / Temperature')
        ax.set_title(f'Cell: {cell[:15]}... (First 100 cycles)')
        ax.grid(True, alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_sequences.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'sample_sequences.png'}")


def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    print("\n[7] Creating summary dashboard...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. SOH Distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(df['soh'], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0.80, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('SOH')
    ax1.set_ylabel('Frequency')
    ax1.set_title('SOH Distribution')
    
    # 2. Feature Correlation Heatmap
    ax2 = plt.subplot(3, 3, 2)
    corr_cols = []
    for c in ['dc_internal_resistance', 'temperature_avg', 'charge_capacity', 
              'coulombic_efficiency_lagged_1', 'soh']:
        if c in df.columns:
            corr_cols.append(c)
    if len(corr_cols) < 5:
        # Add available features
        for c in get_feat_cols(df):
            if c not in corr_cols and c != 'soh':
                corr_cols.append(c)
                if len(corr_cols) >= 5:
                    break
    
    corr_df = df[corr_cols].corr()
    im = ax2.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(corr_cols)))
    ax2.set_yticks(range(len(corr_cols)))
    labels = [FEAT_NAMES.get(c, c[:3]) for c in corr_cols]
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yticklabels(labels)
    ax2.set_title('Feature Correlations')
    plt.colorbar(im, ax=ax2)
    
    # 3. Split Distribution
    ax3 = plt.subplot(3, 3, 3)
    split_counts = df['split'].value_counts()
    colors_split = {'train': '#2ecc71', 'val': '#f39c12', 'test': '#e74c3c'}
    ax3.bar(split_counts.index, split_counts.values, 
            color=[colors_split.get(s, 'gray') for s in split_counts.index],
            edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Split')
    ax3.set_ylabel('Cycles')
    ax3.set_title('Data Split')
    
    # 4. SOH Trajectories
    ax4 = plt.subplot(3, 3, 4)
    cells = df['barcode'].unique()[:10]
    for cell in cells:
        cell_data = df[df['barcode'] == cell].sort_values('cycle_index')
        ax4.plot(cell_data['cycle_index'], cell_data['soh'], alpha=0.5, linewidth=0.8)
    ax4.axhline(y=0.80, color='red', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('SOH')
    ax4.set_title('SOH Trajectories')
    
    # 5. Resistance vs SOH
    ax5 = plt.subplot(3, 3, 5)
    sample = df.sample(min(5000, len(df)))
    ax5.scatter(sample['dc_internal_resistance'], sample['soh'], alpha=0.3, s=5)
    ax5.set_xlabel('Internal Resistance (Ω)')
    ax5.set_ylabel('SOH')
    ax5.set_title('Resistance vs SOH')
    
    # 6. Temperature Distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df['temperature_avg'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax6.set_xlabel('Temperature (°C)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Temperature Distribution')
    
    # 7. Capacity Distribution
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(df['charge_capacity'], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax7.set_xlabel('Charge Capacity (Ah)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Charge Capacity Distribution')
    
    # 8. CE Distribution
    ax8 = plt.subplot(3, 3, 8)
    ce_vals = df['coulombic_efficiency_lagged_1'].dropna()
    if len(ce_vals) > 0:
        ax8.hist(ce_vals, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax8.set_xlabel('Coulombic Efficiency')
        ax8.set_ylabel('Frequency')
        ax8.set_title('CE Distribution')
    
    # 9. Cycle Count per Cell
    ax9 = plt.subplot(3, 3, 9)
    cycle_counts = df.groupby('barcode')['cycle_index'].max().sort_values()
    ax9.hist(cycle_counts, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax9.set_xlabel('Max Cycle Count')
    ax9.set_ylabel('Number of Cells')
    ax9.set_title('Cycle Count per Cell')
    
    plt.suptitle('Dataset Summary Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'summary_dashboard.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("BATTERY DATASET VISUALIZATION")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Generate all visualizations
    plot_soh_trajectories(df, n_cells=20)
    plot_feature_distributions(df)
    plot_feature_correlations(df)
    plot_split_distribution(df)
    plot_feature_evolution(df, n_cells=5)
    plot_sample_sequences(df, n_samples=4)
    create_summary_dashboard(df)
    
    print("\n" + "="*60)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()