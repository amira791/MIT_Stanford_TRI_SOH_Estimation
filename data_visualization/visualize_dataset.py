# visualize_dataset.py
"""
Comprehensive Visualization for Battery Dataset
----------------------------------------------
Creates visualizations for:
1. SOH degradation trajectories
2. RUL distribution
3. Feature correlations
4. Training/validation/test splits
5. Capacity fade patterns
6. Resistance increase patterns
7. Temperature effects
8. Coulombic efficiency trends
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



BASE_DIR = Path(r'C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation')

FINAL_DATASET_DIR = BASE_DIR / "data_preprocessing" / "final_dataset"
SOH_DIR = FINAL_DATASET_DIR / "soh"
RUL_DIR = FINAL_DATASET_DIR / "rul"
VISUALIZATION_DIR = BASE_DIR / "data_visualization" / "visualizations"

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# Feature names for plotting
FEATURES = [
    "cycle_index",
    "dc_internal_resistance",
    "temperature_avg",
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2"
]

FEATURE_NAMES_PRETTY = {
    "cycle_index": "Cycle Number",
    "dc_internal_resistance": "DC Internal Resistance (Ω)",
    "temperature_avg": "Average Temperature (°C)",
    "charge_capacity": "Charge Capacity (Ah)",
    "charge_energy": "Charge Energy (Wh)",
    "coulombic_efficiency_lagged_1": "Coulombic Efficiency (t-1)",
    "coulombic_efficiency_lagged_2": "Coulombic Efficiency (t-2)",
    "soh": "State of Health (SOH)",
    "rul": "Remaining Useful Life (cycles)"
}


def load_datasets():
    """Load SOH and RUL datasets"""
    print("Loading datasets...")
    
    soh_full = pd.read_pickle(SOH_DIR / "soh_full.pkl")
    rul_full = pd.read_pickle(RUL_DIR / "rul_full.pkl")
    
    print(f"  SOH dataset: {len(soh_full)} rows, {soh_full['barcode'].nunique()} cells")
    print(f"  RUL dataset: {len(rul_full)} rows, {rul_full['barcode'].nunique()} cells")
    
    return soh_full, rul_full


def plot_soh_trajectories(soh_full, num_cells=20):
    """Plot SOH degradation trajectories for multiple cells"""
    print("\n[1] Plotting SOH trajectories...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Multiple cell trajectories
    ax1 = axes[0, 0]
    cells = soh_full['barcode'].unique()[:num_cells]
    
    for cell in cells:
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        ax1.plot(cell_data['cycle_index'], cell_data['soh'], alpha=0.7, linewidth=1)
    
    ax1.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='EOL Threshold (80%)')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('State of Health (SOH)')
    ax1.set_title(f'SOH Trajectories (First {num_cells} Cells)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SOH distribution
    ax2 = axes[0, 1]
    ax2.hist(soh_full['soh'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.80, color='red', linestyle='--', linewidth=2, label='EOL Threshold')
    ax2.set_xlabel('SOH')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SOH Distribution')
    ax2.legend()
    
    # Plot 3: SOH vs Cycle (heatmap style for one cell)
    ax3 = axes[1, 0]
    example_cell = cells[0]
    cell_data = soh_full[soh_full['barcode'] == example_cell].sort_values('cycle_index')
    ax3.plot(cell_data['cycle_index'], cell_data['soh'], 'b-', linewidth=2)
    ax3.fill_between(cell_data['cycle_index'], 0, cell_data['soh'], alpha=0.3)
    ax3.axhline(y=0.80, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Cycle Number')
    ax3.set_ylabel('SOH')
    ax3.set_title(f'SOH Degradation - Cell: {example_cell[:15]}...')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Degradation rate distribution
    ax4 = axes[1, 1]
    degradation_rates = []
    for cell in soh_full['barcode'].unique():
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        if len(cell_data) > 10:
            start_soh = cell_data['soh'].iloc[0]
            end_soh = cell_data['soh'].iloc[-1]
            end_cycle = cell_data['cycle_index'].iloc[-1]
            rate = (start_soh - end_soh) / end_cycle * 100
            degradation_rates.append(rate)
    
    ax4.hist(degradation_rates, bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Degradation Rate (% per 100 cycles)')
    ax4.set_ylabel('Number of Cells')
    ax4.set_title('Degradation Rate Distribution')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "soh_trajectories.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'soh_trajectories.png'}")


def plot_rul_distribution(rul_full):
    """Plot RUL distribution for labeled cells"""
    print("\n[2] Plotting RUL distribution...")
    
    # Filter only labeled cells (RUL >= 0)
    labeled = rul_full[rul_full['rul'] >= 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: RUL histogram
    ax1 = axes[0]
    ax1.hist(labeled['rul'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax1.set_xlabel('Remaining Useful Life (cycles)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('RUL Distribution (Labeled Cells)')
    ax1.axvline(x=labeled['rul'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {labeled['rul'].mean():.1f}")
    ax1.legend()
    
    # Plot 2: RUL by cell
    ax2 = axes[1]
    cell_rul = labeled.groupby('barcode')['rul'].max().sort_values()
    ax2.barh(range(len(cell_rul)), cell_rul.values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Maximum RUL (cycles)')
    ax2.set_ylabel('Cell Index')
    ax2.set_title('RUL per Cell (Sorted)')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "rul_distribution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'rul_distribution.png'}")


def plot_feature_correlations(soh_full):
    """Plot feature correlation matrix"""
    print("\n[3] Plotting feature correlations...")
    
    # Select features and target
    corr_df = soh_full[FEATURES + ['soh']].copy()
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, 
                cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "feature_correlations.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'feature_correlations.png'}")


def plot_feature_evolution(soh_full, num_cells=5):
    """Plot how features evolve with cycles"""
    print("\n[4] Plotting feature evolution...")
    
    cells = soh_full['barcode'].unique()[:num_cells]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    features_to_plot = ['dc_internal_resistance', 'temperature_avg', 
                        'charge_capacity', 'coulombic_efficiency_lagged_1', 'soh']
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        
        for cell in cells:
            cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
            # Take every 10th point for cleaner plot
            if len(cell_data) > 100:
                cell_data = cell_data.iloc[::10]
            ax.plot(cell_data['cycle_index'], cell_data[feature], alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel(FEATURE_NAMES_PRETTY.get(feature, feature))
        ax.set_title(f'{FEATURE_NAMES_PRETTY.get(feature, feature)} vs Cycles')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Box plot of features
    ax = axes[5]
    features_for_box = ['dc_internal_resistance', 'temperature_avg', 'charge_capacity']
    data_to_plot = [soh_full[f].values for f in features_for_box]
    bp = ax.boxplot(data_to_plot, labels=features_for_box, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_ylabel('Value')
    ax.set_title('Feature Distribution Overview')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "feature_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'feature_evolution.png'}")


def plot_split_distribution(soh_full):
    """Plot train/val/test split distribution"""
    print("\n[5] Plotting split distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Number of cycles per split
    ax1 = axes[0]
    split_counts = soh_full['split'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax1.bar(split_counts.index, split_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Split')
    ax1.set_ylabel('Number of Cycles')
    ax1.set_title('Cycles per Split')
    for i, (label, count) in enumerate(split_counts.items()):
        ax1.text(i, count + 1000, f'{count:,}', ha='center', fontsize=10)
    
    # Plot 2: Number of cells per split
    ax2 = axes[1]
    cell_counts = soh_full.groupby('split')['barcode'].nunique()
    ax2.bar(cell_counts.index, cell_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Split')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Cells per Split')
    for i, (label, count) in enumerate(cell_counts.items()):
        ax2.text(i, count + 1, f'{count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "split_distribution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'split_distribution.png'}")


def plot_resistance_vs_soh(soh_full):
    """Plot relationship between resistance and SOH"""
    print("\n[6] Plotting resistance vs SOH relationship...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Scatter plot
    ax1 = axes[0]
    # Sample for performance
    sample = soh_full.sample(min(10000, len(soh_full)))
    ax1.scatter(sample['dc_internal_resistance'], sample['soh'], alpha=0.3, s=1)
    ax1.set_xlabel('DC Internal Resistance (Ω)')
    ax1.set_ylabel('SOH')
    ax1.set_title('Resistance vs SOH Relationship')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Resistance increase over time for multiple cells
    ax2 = axes[1]
    cells = soh_full['barcode'].unique()[:10]
    for cell in cells:
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        normalized_r = cell_data['dc_internal_resistance'] / cell_data['dc_internal_resistance'].iloc[0]
        ax2.plot(cell_data['cycle_index'], normalized_r, alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Cycle Number')
    ax2.set_ylabel('Normalized Resistance (R/R₀)')
    ax2.set_title('Resistance Increase Over Time (10 Cells)')
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='2× Initial')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "resistance_vs_soh.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'resistance_vs_soh.png'}")


def plot_temperature_effects(soh_full):
    """Plot temperature effects on degradation"""
    print("\n[7] Plotting temperature effects...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Temperature distribution
    ax1 = axes[0]
    ax1.hist(soh_full['temperature_avg'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax1.set_xlabel('Average Temperature (°C)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Temperature Distribution')
    ax1.axvline(x=soh_full['temperature_avg'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {soh_full['temperature_avg'].mean():.1f}°C")
    ax1.legend()
    
    # Plot 2: Temperature vs SOH
    ax2 = axes[1]
    sample = soh_full.sample(min(5000, len(soh_full)))
    scatter = ax2.scatter(sample['temperature_avg'], sample['soh'], 
                          c=sample['cycle_index'], cmap='viridis', alpha=0.5, s=10)
    ax2.set_xlabel('Average Temperature (°C)')
    ax2.set_ylabel('SOH')
    ax2.set_title('Temperature vs SOH (colored by cycle)')
    plt.colorbar(scatter, ax=ax2, label='Cycle Number')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "temperature_effects.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'temperature_effects.png'}")


def plot_coulombic_efficiency(soh_full):
    """Plot Coulombic efficiency trends"""
    print("\n[8] Plotting Coulombic efficiency...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: CE over cycles
    ax1 = axes[0]
    cells = soh_full['barcode'].unique()[:10]
    for cell in cells:
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        # Take every 10th point
        if len(cell_data) > 100:
            cell_data = cell_data.iloc[::10]
        ax1.plot(cell_data['cycle_index'], cell_data['coulombic_efficiency_lagged_1'], 
                alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('Coulombic Efficiency')
    ax1.set_title('Coulombic Efficiency Over Time (10 Cells)')
    ax1.axhline(y=0.999, color='green', linestyle='--', alpha=0.5, label='99.9%')
    ax1.axhline(y=0.995, color='orange', linestyle='--', alpha=0.5, label='99.5%')
    ax1.axhline(y=0.990, color='red', linestyle='--', alpha=0.5, label='99.0%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CE distribution
    ax2 = axes[1]
    ce_values = soh_full['coulombic_efficiency_lagged_1'].dropna()
    ax2.hist(ce_values, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax2.set_xlabel('Coulombic Efficiency')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Coulombic Efficiency Distribution')
    ax2.axvline(x=ce_values.mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {ce_values.mean():.5f}")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "coulombic_efficiency.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'coulombic_efficiency.png'}")


def plot_labeled_vs_unlabeled(rul_full):
    """Visualize labeled vs unlabeled cells for RUL"""
    print("\n[9] Plotting labeled vs unlabeled cells...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Pie chart
    ax1 = axes[0]
    labeled_count = len(rul_full[rul_full['has_label'] == 1]['barcode'].unique())
    unlabeled_count = len(rul_full[rul_full['has_label'] == 0]['barcode'].unique())
    
    ax1.pie([labeled_count, unlabeled_count], 
            labels=[f'Labeled ({labeled_count} cells)', f'Unlabeled ({unlabeled_count} cells)'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('RUL Data: Labeled vs Unlabeled Cells')
    
    # Plot 2: SOH range comparison
    ax2 = axes[1]
    labeled_cells = rul_full[rul_full['has_label'] == 1]['barcode'].unique()
    unlabeled_cells = rul_full[rul_full['has_label'] == 0]['barcode'].unique()
    
    labeled_min_soh = rul_full[rul_full['barcode'].isin(labeled_cells)].groupby('barcode')['soh'].min()
    unlabeled_min_soh = rul_full[rul_full['barcode'].isin(unlabeled_cells)].groupby('barcode')['soh'].min()
    
    ax2.hist(labeled_min_soh, bins=20, alpha=0.5, label='Labeled Cells', color='green', edgecolor='black')
    ax2.hist(unlabeled_min_soh, bins=20, alpha=0.5, label='Unlabeled Cells', color='red', edgecolor='black')
    ax2.axvline(x=0.80, color='blue', linestyle='--', linewidth=2, label='EOL Threshold (80%)')
    ax2.set_xlabel('Minimum SOH Achieved')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Minimum SOH: Labeled vs Unlabeled Cells')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "labeled_vs_unlabeled.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'labeled_vs_unlabeled.png'}")


def plot_summary_dashboard(soh_full, rul_full):
    """Create a comprehensive summary dashboard"""
    print("\n[10] Creating summary dashboard...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. SOH Trajectories (top left)
    ax1 = plt.subplot(3, 3, 1)
    cells = soh_full['barcode'].unique()[:15]
    for cell in cells:
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        ax1.plot(cell_data['cycle_index'], cell_data['soh'], alpha=0.5, linewidth=0.8)
    ax1.axhline(y=0.80, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('SOH')
    ax1.set_title('SOH Trajectories')
    ax1.grid(True, alpha=0.3)
    
    # 2. RUL Distribution (top middle)
    ax2 = plt.subplot(3, 3, 2)
    labeled = rul_full[rul_full['rul'] >= 0]
    ax2.hist(labeled['rul'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('RUL (cycles)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('RUL Distribution')
    
    # 3. Feature Correlation (top right)
    ax3 = plt.subplot(3, 3, 3)
    corr_df = soh_full[['dc_internal_resistance', 'temperature_avg', 'charge_capacity', 'soh']].corr()
    im = ax3.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    ax3.set_xticklabels(['R', 'T', 'C', 'SOH'], rotation=45)
    ax3.set_yticklabels(['R', 'T', 'C', 'SOH'])
    ax3.set_title('Feature Correlations')
    plt.colorbar(im, ax=ax3)
    
    # 4. Resistance vs Cycle (middle left)
    ax4 = plt.subplot(3, 3, 4)
    for cell in cells[:5]:
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        if len(cell_data) > 100:
            cell_data = cell_data.iloc[::20]
        ax4.plot(cell_data['cycle_index'], cell_data['dc_internal_resistance'], alpha=0.7)
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('Resistance (Ω)')
    ax4.set_title('Resistance Increase')
    ax4.grid(True, alpha=0.3)
    
    # 5. Temperature Distribution (middle center)
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(soh_full['temperature_avg'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Temperature Distribution')
    
    # 6. CE Distribution (middle right)
    ax6 = plt.subplot(3, 3, 6)
    ce_vals = soh_full['coulombic_efficiency_lagged_1'].dropna()
    ax6.hist(ce_vals, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax6.set_xlabel('Coulombic Efficiency')
    ax6.set_ylabel('Frequency')
    ax6.set_title('CE Distribution')
    
    # 7. Split Distribution (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    split_counts = soh_full['split'].value_counts()
    colors_split = ['#2ecc71', '#f39c12', '#e74c3c']
    ax7.bar(split_counts.index, split_counts.values, color=colors_split, alpha=0.7)
    ax7.set_xlabel('Split')
    ax7.set_ylabel('Cycles')
    ax7.set_title('Data Split Distribution')
    
    # 8. Labeled vs Unlabeled (bottom center)
    ax8 = plt.subplot(3, 3, 8)
    labeled_count = len(rul_full[rul_full['has_label'] == 1]['barcode'].unique())
    unlabeled_count = len(rul_full[rul_full['has_label'] == 0]['barcode'].unique())
    ax8.bar(['Labeled', 'Unlabeled'], [labeled_count, unlabeled_count], 
            color=['green', 'red'], alpha=0.7)
    ax8.set_ylabel('Number of Cells')
    ax8.set_title('RUL Data: Labeled vs Unlabeled')
    
    # 9. Capacity Fade (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    for cell in cells[:5]:
        cell_data = soh_full[soh_full['barcode'] == cell].sort_values('cycle_index')
        if len(cell_data) > 100:
            cell_data = cell_data.iloc[::20]
        ax9.plot(cell_data['cycle_index'], cell_data['charge_capacity'], alpha=0.7)
    ax9.set_xlabel('Cycle')
    ax9.set_ylabel('Charge Capacity (Ah)')
    ax9.set_title('Capacity Fade')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Battery Dataset Summary Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved to: {VISUALIZATION_DIR / 'summary_dashboard.png'}")


def main():
    print("=" * 60)
    print("BATTERY DATASET VISUALIZATION")
    print("=" * 60)
    
    # Load datasets
    soh_full, rul_full = load_datasets()
    
    # Generate all visualizations
    plot_soh_trajectories(soh_full, num_cells=20)
    plot_rul_distribution(rul_full)
    plot_feature_correlations(soh_full)
    plot_feature_evolution(soh_full, num_cells=5)
    plot_split_distribution(soh_full)
    plot_resistance_vs_soh(soh_full)
    plot_temperature_effects(soh_full)
    plot_coulombic_efficiency(soh_full)
    plot_labeled_vs_unlabeled(rul_full)
    plot_summary_dashboard(soh_full, rul_full)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {VISUALIZATION_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()