"""
================================================================================
FEATURE IMPORTANCE ANALYSIS - ALL 16 SPECIFIED FEATURES
================================================================================

This script calculates feature importance for:
1. Summary features (10): cycle_index, discharge_capacity, charge_capacity,
   discharge_energy, charge_energy, dc_internal_resistance, temperature_maximum,
   temperature_average, temperature_minimum, date_time_iso_numeric
2. Lagged CE features (2): coulombic_efficiency_lagged_1, coulombic_efficiency_lagged_2
3. Engineered features (4): cap_rel, energy_rel, ir_rel, cycle_pos

Uses the new dataset from results2/soh_dataset_full_features.pkl
================================================================================
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & CONFIGURATION
# ============================================================

# USE THE NEW DATASET WITH ALL FEATURES
DATA_PATH = r'C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_dataset_full_features.pkl'
OUTPUT_DIR = Path(__file__).parent.parent / "feature_selection_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FSS_KEEP_THRESHOLD = 0.50
LEAKAGE_THRESHOLD = 0.5

# ============================================================
# FEATURE DEFINITIONS - ALL 16 FEATURES
# ============================================================

FEAT_COLS = [
    # Summary features (10)
    "cycle_index",
    "discharge_capacity",
    "charge_capacity",
    "discharge_energy",
    "charge_energy",
    "dc_internal_resistance",
    "temperature_maximum",
    "temperature_average",
    "temperature_minimum",
    "date_time_iso_numeric",
    # Lagged CE features (2)
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    # Engineered features (4)
    "cap_rel",
    "energy_rel",
    "ir_rel",
    "cycle_pos",
]

FEAT_NAMES = {
    "cycle_index": "Cycle Index",
    "discharge_capacity": "Discharge Capacity",
    "charge_capacity": "Charge Capacity",
    "discharge_energy": "Discharge Energy",
    "charge_energy": "Charge Energy",
    "dc_internal_resistance": "Internal Resistance",
    "temperature_maximum": "Max Temperature",
    "temperature_average": "Average Temperature",
    "temperature_minimum": "Min Temperature",
    "date_time_iso_numeric": "Timestamp",
    "coulombic_efficiency_lagged_1": "CE (t-1)",
    "coulombic_efficiency_lagged_2": "CE (t-2)",
    "cap_rel": "Relative Capacity",
    "energy_rel": "Relative Energy",
    "ir_rel": "Relative Resistance",
    "cycle_pos": "Cycle Position",
}

# ============================================================
# TARGET LEAKAGE SCORES
# ============================================================

def get_target_leakage_score(feature):
    """Target Leakage Score for each feature."""
    if feature == "discharge_capacity":
        return 1.0
    elif feature == "discharge_energy":
        return 0.8
    else:
        return 0.0

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    print("\n[1] Loading data...")
    df = pd.read_pickle(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Cells: {df['cell_id'].nunique()}")
    
    # Check which features are present
    missing = [f for f in FEAT_COLS if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    
    # Drop rows with NaN
    df = df.dropna(subset=FEAT_COLS + ["soh"])
    print(f"  After dropping NaN: {df.shape}")
    
    return df

def split_cells(df):
    cell_ids = df["cell_id"].unique()
    train_ids, test_ids = train_test_split(
        cell_ids, test_size=0.2, random_state=RANDOM_STATE
    )
    train_df = df[df["cell_id"].isin(train_ids)].copy()
    test_df = df[df["cell_id"].isin(test_ids)].copy()
    print(f"\n  Cell split: {len(train_ids)} train / {len(test_ids)} test")
    return train_df, test_df

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_correlation_soh(df):
    print("\n[2] Computing correlation with SOH...")
    pearson_corr = {}

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index")
        for feat in FEAT_COLS:
            if feat in cell_df.columns:
                corr = cell_df[feat].corr(cell_df["soh"])
                if not np.isnan(corr):
                    pearson_corr.setdefault(feat, []).append(abs(corr))

    avg_corr = {f: np.mean(pearson_corr[f]) for f in FEAT_COLS if f in pearson_corr}

    print("\n  Correlation with SOH:")
    for feat in FEAT_COLS:
        name = FEAT_NAMES.get(feat, feat)
        print(f"    {name:<30} {avg_corr.get(feat, 0):>10.4f}")

    return avg_corr

def compute_mutual_information(df):
    print("\n[3] Computing Mutual Information...")
    X = df[FEAT_COLS].values
    y = df["soh"].values

    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    mi_dict = {feat: mi for feat, mi in zip(FEAT_COLS, mi_scores)}

    mi_max = max(mi_dict.values()) if mi_dict and max(mi_dict.values()) > 0 else 1
    mi_normalized = {f: mi / mi_max for f, mi in mi_dict.items()}

    print("\n  Mutual Information (normalized):")
    for feat in FEAT_COLS:
        name = FEAT_NAMES.get(feat, feat)
        print(f"    {name:<30} {mi_normalized.get(feat, 0):>10.4f}")

    return mi_normalized

def compute_redundancy(df):
    print("\n[4] Computing redundancy...")
    corr_matrix = df[FEAT_COLS].corr().abs()

    max_corr = {}
    for feat in FEAT_COLS:
        corr_values = corr_matrix[feat].drop(feat)
        max_corr[feat] = corr_values.max()

    print("\n  Maximum correlation with other features:")
    for feat in FEAT_COLS:
        name = FEAT_NAMES.get(feat, feat)
        print(f"    {name:<30} {max_corr[feat]:>10.4f}")

    return max_corr, corr_matrix

def compute_vif(df):
    print("\n[5] Computing VIF...")
    X = df[FEAT_COLS].values
    X_with_const = np.column_stack([np.ones(len(df)), X])

    vif_dict = {}
    for i, feat in enumerate(FEAT_COLS):
        vif = variance_inflation_factor(X_with_const, i + 1)
        vif_dict[feat] = vif

    print("\n  VIF Scores:")
    for feat in FEAT_COLS:
        name = FEAT_NAMES.get(feat, feat)
        status = " OK" if vif_dict[feat] < 5 else "Moderate" if vif_dict[feat] < 10 else "High"
        print(f"    {name:<30} {vif_dict[feat]:>10.2f}  {status}")

    return vif_dict

def compute_fss(pearson_corr, mi_normalized, max_corr, vif_dict):
    print("\n" + "=" * 60)
    print("FSS CALCULATION")
    print("=" * 60)

    w1, w2, w3, w5 = 0.35, 0.30, 0.20, 0.15

    fss_scores = {}
    gated_features = []

    for feat in FEAT_COLS:
        leakage = get_target_leakage_score(feat)

        if leakage >= LEAKAGE_THRESHOLD:
            fss_scores[feat] = -np.inf
            gated_features.append(feat)
            continue

        c1 = pearson_corr.get(feat, 0)
        c2 = 1 - max_corr.get(feat, 1)
        vif = vif_dict.get(feat, 100)
        c3 = 1 / (1 + vif)
        c5 = mi_normalized.get(feat, 0)

        fss = w1 * c1 + w2 * c2 + w3 * c3 + w5 * c5
        fss_scores[feat] = fss

    sorted_fss = sorted(fss_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nWeights:")
    print(f"  Correlation with SOH: {w1:.2f}")
    print(f"  Non-Redundancy:        {w2:.2f}")
    print(f"  Low VIF:               {w3:.2f}")
    print(f"  Mutual Information:    {w5:.2f}")
    print(f"\nLeakage gate: features with leakage >= {LEAKAGE_THRESHOLD} are removed")
    print(f"Threshold: FSS >= {FSS_KEEP_THRESHOLD} -> Keep")

    if gated_features:
        print(f"\nGated out due to target leakage: {[FEAT_NAMES.get(f, f) for f in gated_features]}")

    print("\nResults:")
    print("-" * 70)
    print(f"  {'Rank':<5} {'Feature':<30} {'FSS':>10} {'Leakage':>10} {'Decision':>10}")
    print("-" * 70)

    for i, (feat, score) in enumerate(sorted_fss):
        name = FEAT_NAMES.get(feat, feat)
        leakage = get_target_leakage_score(feat)
        if leakage >= LEAKAGE_THRESHOLD:
            decision = "Remove (leakage)"
        elif score >= FSS_KEEP_THRESHOLD:
            decision = "Keep"
        else:
            decision = "Remove"
        score_str = f"{score:>10.4f}" if np.isfinite(score) else f"{'--':>10}"
        print(f"  {i+1:<5} {name:<30} {score_str} {leakage:>10.1f} {decision:>16}")

    return fss_scores, sorted_fss

def create_visualizations(sorted_fss, corr_matrix, output_dir):
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    plot_items = [(f, s) for f, s in sorted_fss if np.isfinite(s)]
    gated_items = [(f, s) for f, s in sorted_fss if not np.isfinite(s)]

    fig, ax = plt.subplots(figsize=(12, 8))

    features = [FEAT_NAMES.get(f, f) for f, _ in plot_items]
    scores = [s for _, s in plot_items]
    colors = ["#2ecc71" if s >= FSS_KEEP_THRESHOLD else "#e74c3c" for s in scores]

    ax.barh(range(len(features)), scores, color=colors, alpha=0.7, edgecolor="black")
    ax.axvline(x=FSS_KEEP_THRESHOLD, color="black", linestyle="--", linewidth=2, label=f"Threshold ({FSS_KEEP_THRESHOLD})")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("FSS Score", fontsize=12)
    title = "Feature Selection Score (FSS) Ranking"
    if gated_items:
        title += f"\n(excludes {len(gated_items)} feature(s) removed for target leakage)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "fss_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: fss_ranking.png")

    fig, ax = plt.subplots(figsize=(14, 12))

    if "soh" in corr_matrix.columns:
        soh_corr = corr_matrix["soh"].drop("soh").sort_values(ascending=False)
        sorted_cols = list(soh_corr.index) + ["soh"]
        corr_matrix = corr_matrix.loc[sorted_cols, sorted_cols]

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"label": "Correlation"},
        ax=ax,
        mask=mask,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: correlation_matrix.png")

def export_results(pearson_corr, mi_normalized, max_corr, vif_dict, fss_scores, output_dir):
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    df_results = pd.DataFrame(
        {
            "Feature": FEAT_COLS,
            "Feature_Name": [FEAT_NAMES.get(f, f) for f in FEAT_COLS],
            "Pearson_Correlation": [pearson_corr.get(f, 0) for f in FEAT_COLS],
            "Mutual_Information": [mi_normalized.get(f, 0) for f in FEAT_COLS],
            "Max_Correlation": [max_corr.get(f, 1) for f in FEAT_COLS],
            "VIF": [vif_dict.get(f, 100) for f in FEAT_COLS],
            "Target_Leakage": [get_target_leakage_score(f) for f in FEAT_COLS],
            "FSS_Score": [fss_scores.get(f, 0) for f in FEAT_COLS],
        }
    )

    def decide(row):
        if row["Target_Leakage"] >= LEAKAGE_THRESHOLD:
            return "Remove (leakage)"
        return "Keep" if row["FSS_Score"] >= FSS_KEEP_THRESHOLD else "Remove"

    df_results["Decision"] = df_results.apply(decide, axis=1)
    df_results = df_results.sort_values("FSS_Score", ascending=False)

    df_results.to_csv(output_dir / "feature_selection_results.csv", index=False)
    print("   Saved: feature_selection_results.csv")

    fss_df = df_results[["Feature_Name", "FSS_Score", "Target_Leakage", "Decision"]]
    fss_df.to_csv(output_dir / "fss_ranking.csv", index=False)
    print("   Saved: fss_ranking.csv")

    print("\nSummary:")
    print("-" * 50)
    keep_count = (df_results["Decision"] == "Keep").sum()
    leak_count = (df_results["Decision"] == "Remove (leakage)").sum()
    other_remove_count = (df_results["Decision"] == "Remove").sum()
    print(f"  Features Kept: {keep_count}")
    print(f"  Features Removed (leakage): {leak_count}")
    print(f"  Features Removed (other): {other_remove_count}")
    print(f"  Total Features: {len(FEAT_COLS)}")

    print("\nFinal Selected Features:")
    for _, row in df_results[df_results["Decision"] == "Keep"].iterrows():
        print(f"   {row['Feature_Name']}: FSS = {row['FSS_Score']:.4f}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS - 16 FEATURES")
    print("=" * 60)

    df = load_data()
    train_df, test_df = split_cells(df)

    pearson_corr = compute_correlation_soh(train_df)
    mi_normalized = compute_mutual_information(train_df)
    max_corr, corr_matrix = compute_redundancy(train_df)
    vif_dict = compute_vif(train_df)

    fss_scores, sorted_fss = compute_fss(pearson_corr, mi_normalized, max_corr, vif_dict)

    create_visualizations(sorted_fss, corr_matrix, OUTPUT_DIR)

    export_results(pearson_corr, mi_normalized, max_corr, vif_dict, fss_scores, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()