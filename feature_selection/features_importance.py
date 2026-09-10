# feature_importance_analysis.py
"""
Feature Importance Analysis for SOH Estimation
----------------------------------------------
Calculates feature importance using:
1. Pearson correlation with SOH
2. Spearman correlation with SOH
3. Mutual Information (non-linear)
4. Permutation Importance (using a trained model)
5. VIF (multicollinearity)

Features: The 10 features used in the model
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\feature_importance_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FEATURE COLUMNS (The 10 features used in the model)
# ============================================================

FEAT_COLS = [
    "dc_internal_resistance",
    "temperature_avg",
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    "cap_rel",
    "energy_rel",
    "ir_rel",
    "cycle_pos",
]

FEAT_NAMES = {
    "dc_internal_resistance": "Internal Resistance",
    "temperature_avg": "Average Temperature",
    "charge_capacity": "Charge Capacity",
    "charge_energy": "Charge Energy",
    "coulombic_efficiency_lagged_1": "CE (t-1)",
    "coulombic_efficiency_lagged_2": "CE (t-2)",
    "cap_rel": "Relative Capacity",
    "energy_rel": "Relative Energy",
    "ir_rel": "Relative Resistance",
    "cycle_pos": "Cycle Position",
}

# ============================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================

def add_relative_features(df):
    """Add per-cell relative features"""
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


def load_data():
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = add_relative_features(df)
    print(f"  Shape: {df.shape}")
    print(f"  Cells: {df['cell_id'].nunique()}")
    return df

# ============================================================
# METHOD 1: PEARSON CORRELATION (Linear)
# ============================================================

def compute_pearson(df):
    print("\n[2] Computing Pearson correlation...")
    results = {}
    for feat in FEAT_COLS:
        corr, _ = pearsonr(df[feat], df['soh'])
        results[feat] = abs(corr)
    return results

# ============================================================
# METHOD 2: SPEARMAN CORRELATION (Monotonic)
# ============================================================

def compute_spearman(df):
    print("\n[3] Computing Spearman correlation...")
    results = {}
    for feat in FEAT_COLS:
        corr, _ = spearmanr(df[feat], df['soh'])
        results[feat] = abs(corr)
    return results

# ============================================================
# METHOD 3: MUTUAL INFORMATION (Non-linear)
# ============================================================

def compute_mutual_information(df):
    print("\n[4] Computing Mutual Information...")
    X = df[FEAT_COLS].values
    y = df['soh'].values
    mi = mutual_info_regression(X, y, random_state=42)
    mi_dict = {feat: m for feat, m in zip(FEAT_COLS, mi)}
    # Normalize
    mi_max = max(mi_dict.values())
    mi_norm = {f: m / mi_max for f, m in mi_dict.items()}
    return mi_norm

# ============================================================
# METHOD 4: PERMUTATION IMPORTANCE (Model-based)
# ============================================================

def compute_permutation_importance(df):
    print("\n[5] Computing Permutation Importance...")
    
    # Use train split only
    train_df = df[df['split'] == 'train']
    X_train = train_df[FEAT_COLS].values
    y_train = train_df['soh'].values
    
    # Train a Random Forest (proxy model)
    print("  Training Random Forest proxy model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Compute permutation importance
    print("  Computing permutation importance...")
    result = permutation_importance(
        model, X_train, y_train,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    
    imp_dict = {feat: imp for feat, imp in zip(FEAT_COLS, result.importances_mean)}
    return imp_dict

# ============================================================
# METHOD 5: VIF (Multicollinearity)
# ============================================================

def compute_vif(df):
    print("\n[6] Computing VIF...")
    X = df[FEAT_COLS].values
    X_with_const = np.column_stack([np.ones(len(df)), X])
    
    vif_dict = {}
    for i, feat in enumerate(FEAT_COLS):
        vif = variance_inflation_factor(X_with_const, i + 1)
        vif_dict[feat] = vif
    return vif_dict

# ============================================================
# COMBINE RESULTS
# ============================================================

def combine_results(pearson, spearman, mi, perm, vif):
    print("\n[7] Combining results...")
    
    results = []
    for feat in FEAT_COLS:
        results.append({
            'Feature': feat,
            'Feature_Name': FEAT_NAMES.get(feat, feat),
            'Pearson': pearson.get(feat, 0),
            'Spearman': spearman.get(feat, 0),
            'Mutual_Information': mi.get(feat, 0),
            'Permutation_Importance': perm.get(feat, 0),
            'VIF': vif.get(feat, 0),
        })
    
    df_results = pd.DataFrame(results)
    
    # Compute composite score
    df_results['Correlation_Score'] = (df_results['Pearson'] + df_results['Spearman']) / 2
    df_results['Composite_Score'] = (
        0.25 * df_results['Pearson'] +
        0.25 * df_results['Spearman'] +
        0.20 * df_results['Mutual_Information'] +
        0.20 * df_results['Permutation_Importance'] +
        0.10 * (1 / (1 + df_results['VIF']))
    )
    
    # Sort by composite score
    df_results = df_results.sort_values('Composite_Score', ascending=False)
    
    return df_results

# ============================================================
# VISUALIZATION
# ============================================================

def create_visualizations(df_results, output_dir):
    print("\n[8] Creating visualizations...")
    
    # 1. Correlation bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_results))
    width = 0.35
    
    ax.bar(x - width/2, df_results['Pearson'], width, label='Pearson', alpha=0.8)
    ax.bar(x + width/2, df_results['Spearman'], width, label='Spearman', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['Feature_Name'], rotation=45, ha='right')
    ax.set_ylabel('Correlation with SOH')
    ax.set_title('Feature Correlation with SOH')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png', dpi=150)
    plt.close()
    
    # 2. Composite score bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(df_results['Composite_Score'] / df_results['Composite_Score'].max())
    ax.barh(df_results['Feature_Name'], df_results['Composite_Score'], color=colors)
    ax.set_xlabel('Composite Importance Score')
    ax.set_title('Feature Importance (Composite Score)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_composite.png', dpi=150)
    plt.close()
    
    # 3. Mutual Information bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(df_results['Feature_Name'], df_results['Mutual_Information'], color='steelblue')
    ax.set_xlabel('Mutual Information (normalized)')
    ax.set_title('Feature Mutual Information with SOH')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_mutual_information.png', dpi=150)
    plt.close()
    
    # 4. Permutation Importance
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(df_results['Feature_Name'], df_results['Permutation_Importance'], color='coral')
    ax.set_xlabel('Permutation Importance (ΔMAE)')
    ax.set_title('Feature Permutation Importance')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_permutation_importance.png', dpi=150)
    plt.close()
    
    # 5. VIF bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' for v in df_results['VIF']]
    ax.barh(df_results['Feature_Name'], df_results['VIF'], color=colors)
    ax.axvline(x=5, color='orange', linestyle='--', label='Moderate (5)')
    ax.axvline(x=10, color='red', linestyle='--', label='High (10)')
    ax.set_xlabel('VIF')
    ax.set_title('Feature Multicollinearity (VIF)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_vif.png', dpi=150)
    plt.close()
    
    print(f"  Visualizations saved to {output_dir}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Compute all importance metrics
    pearson = compute_pearson(df)
    spearman = compute_spearman(df)
    mi = compute_mutual_information(df)
    perm = compute_permutation_importance(df)
    vif = compute_vif(df)
    
    # Combine
    df_results = combine_results(pearson, spearman, mi, perm, vif)
    
    # Print results
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RESULTS")
    print("=" * 70)
    
    print("\n" + "-" * 80)
    print(f"{'Feature':<25} {'Pearson':>10} {'Spearman':>10} {'MI':>10} {'Perm':>10} {'VIF':>10}")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"{row['Feature_Name']:<25} {row['Pearson']:>10.4f} {row['Spearman']:>10.4f} "
              f"{row['Mutual_Information']:>10.4f} {row['Permutation_Importance']:>10.4f} "
              f"{row['VIF']:>10.2f}")
    print("-" * 80)
    
    # Save results
    output_path = OUTPUT_DIR / "feature_importance_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")
    
    # Create visualizations
    create_visualizations(df_results, OUTPUT_DIR)
    
    # Print top features
    print("\n" + "=" * 70)
    print("TOP FEATURES BY COMPOSITE SCORE")
    print("=" * 70)
    for i, (_, row) in enumerate(df_results.iterrows(), 1):
        print(f"  {i:>2}. {row['Feature_Name']:<25} {row['Composite_Score']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()