"""
step2_soh_labels_v3.py
=======================
Change from v2: dc_internal_resistance → dc_ir_norm = (R - R_initial) / R_initial

Why: dc_internal_resistance absolute values differ across chemistries
  LFP 18650:   ~0.017 Ω
  NMC 18650:   ~0.030–0.080 Ω  
  LCO pouch:   ~0.005–0.015 Ω
  NCA 21700:   ~0.025–0.060 Ω

If you train on LFP and test on NMC, the scaler maps wrong because
the resistance distributions are completely different ranges.

After normalisation: dc_ir_norm starts at 0.0 for every cell
regardless of chemistry, and grows positively as the cell ages.
A value of 0.20 means "resistance has grown 20% above initial" —
this has the same meaning for every chemistry.

R_initial = mean of first INIT_CYCLES_AVG cycles (same as capacity).
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from configurations.config_training_mlt import (
    RESULTS_DIR, SOH_EOL_THRESHOLD, FEATURE_COLS, TARGET_COL, INIT_CYCLES_AVG
)

CLEANED_PKL = Path(__file__).parent.parent / "results" / "cleaned_cells.pkl"


def compute_soh(cell: Dict) -> np.ndarray:
    return np.clip(
        cell["discharge_capacity"] / cell["initial_capacity"], 0.0, 1.0
    ).astype(np.float32)


def find_eol(soh: np.ndarray, cycle_index: np.ndarray,
             threshold: float = SOH_EOL_THRESHOLD) -> int:
    candidates = np.where(soh < threshold)[0]
    return int(cycle_index[candidates[0]]) if len(candidates) > 0 \
           else int(cycle_index[-1])


def build_cell_dataframe(cell: Dict) -> pd.DataFrame:
    soh         = compute_soh(cell)
    cycle_index = cell["cycle_index"]
    n           = len(cycle_index)
    eol         = find_eol(soh, cycle_index)
    C0          = cell["initial_capacity"]

    # ── initial resistance (mean of first INIT_CYCLES_AVG cycles) ─────
    R0 = float(np.nanmean(cell["dc_internal_resistance"][:INIT_CYCLES_AVG]))
    if R0 <= 0 or np.isnan(R0):
        R0 = float(np.nanmean(cell["dc_internal_resistance"]))
    if R0 <= 0:
        R0 = 0.02   # fallback to typical LFP value

    # ── normalised resistance: (R - R0) / R0 ─────────────────────────
    dc_ir_norm = (cell["dc_internal_resistance"] - R0) / R0
    # clip to [-0.1, 2.0]: negative allowed (measurement noise early life)
    dc_ir_norm = np.clip(dc_ir_norm, -0.10, 2.0).astype(np.float32)

    # ── lag features ──────────────────────────────────────────────────
    soh_prev = np.empty(n, dtype=np.float32)
    soh_prev[0] = np.nan; soh_prev[1:] = soh[:-1]

    delta_soh = np.empty(n, dtype=np.float32)
    delta_soh[0] = np.nan; delta_soh[1:] = soh[1:] - soh[:-1]

    disch_prev = np.empty(n, dtype=np.float32)
    disch_prev[0] = np.nan; disch_prev[1:] = cell["discharge_capacity"][:-1]

    charg_prev = np.empty(n, dtype=np.float32)
    charg_prev[0] = np.nan; charg_prev[1:] = cell["charge_capacity"][:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        coul_eff = np.where(disch_prev > 0.01,
                            charg_prev / disch_prev, np.nan).astype(np.float32)
    coul_eff = np.clip(coul_eff, 0.80, 1.20)

    cycle_norm = np.clip(cycle_index / max(eol, 1), 0.0, 1.0).astype(np.float32)

    df = pd.DataFrame({
        "soh_prev"    : soh_prev,
        "delta_soh"   : delta_soh,
        "coulombic_eff": coul_eff,
        "dc_ir_norm"  : dc_ir_norm,      # ← chemistry-invariant resistance
        "temperature_max": cell["temperature_max"],
        "cycle_norm"  : cycle_norm,
        "soh"         : soh,
        "cycle_index" : cycle_index.astype(np.float32),
        "cell_id"     : cell["barcode"],
        "protocol"    : cell["protocol"],
        "initial_capacity": C0,
        "initial_resistance": R0,        # keep for reference
        "eol_cycle"   : eol,
    })

    df = df.iloc[1:].reset_index(drop=True)
    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill().fillna(df[col].mean())

    return df


if __name__ == "__main__":
    print(f"Loading cleaned cells from {CLEANED_PKL} …")
    with open(CLEANED_PKL, "rb") as f:
        cells: List[Dict] = pickle.load(f)
    print(f"  Loaded {len(cells)} cells")

    cell_dfs = []
    for i, cell in enumerate(cells):
        cdf = build_cell_dataframe(cell)
        cell_dfs.append(cdf)
        print(f"  [{i+1:>3}/{len(cells)}] {cell['barcode']:<20} "
              f"{len(cdf):>5} rows  "
              f"dc_ir_norm=[{cdf['dc_ir_norm'].min():.3f},{cdf['dc_ir_norm'].max():.3f}]")

    df_all = pd.concat(cell_dfs, ignore_index=True)

    # Verify dc_ir_norm is chemistry-invariant
    print(f"\n  dc_ir_norm global range: [{df_all['dc_ir_norm'].min():.3f}, "
          f"{df_all['dc_ir_norm'].max():.3f}]")
    print(f"  dc_ir_norm mean: {df_all['dc_ir_norm'].mean():.4f}  "
          f"(should be close to 0.0 at start of life)")

    csv_path = RESULTS_DIR / "soh_dataset_mlt.csv"
    pkl_path = RESULTS_DIR / "soh_dataset_mlt.pkl"
    df_all.to_csv(csv_path, index=False)
    df_all.to_pickle(pkl_path)
    print(f"\n  Saved → {pkl_path}")
    print("Step 2 v3 complete.")
