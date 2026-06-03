# dataset_preprocessing.py
"""
DATASET PREPROCESSING & CLEANING
--------------------------------
Reads all JSON cells from the MIT-Stanford dataset and applies:

  1. Formation-cycle removal (cycle 0 is artefact)
  2. Physical plausibility checks
  3. Outlier removal per channel (IQR-based)
  4. Missing-value imputation (forward-fill then column mean)
  5. Cells with < MIN_CYCLES_PER_CELL valid cycles are discarded

Output
------
  results/cleaned_cells.pkl     - List of cleaned cell dictionaries
  results/cleaning_report.csv   - Summary of which cells passed/failed
"""




import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_preprocessing import (
    DATA_DIR, RESULTS_DIR,
    NOMINAL_CAPACITY, INIT_CYCLES_AVG, MIN_CYCLES_PER_CELL,
)

# Physical bounds (A123 APR18650M1A datasheet)
BOUNDS = dict(
    voltage_min   = 2.0,
    voltage_max   = 3.6,
    current_min   = -5.5,
    current_max   = 5.5,
    capacity_min  = 0.0,
    capacity_max  = NOMINAL_CAPACITY * 1.3,
    temp_min      = 15.0,
    temp_max      = 60.0,
    ir_min        = 0.005,
    ir_max        = 0.5,
)

IQR_K = 3.0

def _iqr_mask(values: np.ndarray, k: float = IQR_K) -> np.ndarray:
    q1, q3 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (values >= lo) & (values <= hi)

def _forward_fill(arr: np.ndarray) -> np.ndarray:
    df = pd.Series(arr)
    return df.ffill().bfill().to_numpy(dtype=np.float32)

def _clip_and_nan(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = values.astype(np.float64).copy()
    out[(out < lo) | (out > hi)] = np.nan
    return out

def load_and_clean_cell(filepath: str) -> Optional[Dict]:
    filepath = Path(filepath)

    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
    except Exception:
        return None

    summary = raw.get("summary", {})
    if not summary or "discharge_capacity" not in summary:
        return None

    n = len(summary["cycle_index"])

    def _get(key, default=np.nan):
        arr = summary.get(key)
        if arr is None:
            return np.full(n, default, dtype=np.float64)
        return np.array(arr, dtype=np.float64)

    df = pd.DataFrame({
        "cycle_index"            : _get("cycle_index").astype(int),
        "discharge_capacity"     : _get("discharge_capacity"),
        "charge_capacity"        : _get("charge_capacity"),
        "dc_internal_resistance" : _get("dc_internal_resistance"),
        "temperature_max"        : _get("temperature_maximum"),
        "temperature_avg"        : _get("temperature_average"),
        "discharge_energy"       : _get("discharge_energy"),
        "charge_energy"          : _get("charge_energy"),
    })

    df = df[df["cycle_index"] >= 1].copy()
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_CYCLES_PER_CELL:
        return None

    df["discharge_capacity"] = _clip_and_nan(
        df["discharge_capacity"].values,
        BOUNDS["capacity_min"], BOUNDS["capacity_max"])
    df["charge_capacity"] = _clip_and_nan(
        df["charge_capacity"].values,
        BOUNDS["capacity_min"], BOUNDS["capacity_max"])
    df["dc_internal_resistance"] = _clip_and_nan(
        df["dc_internal_resistance"].values,
        BOUNDS["ir_min"], BOUNDS["ir_max"])
    df["temperature_max"] = _clip_and_nan(
        df["temperature_max"].values,
        BOUNDS["temp_min"], BOUNDS["temp_max"])
    df["temperature_avg"] = _clip_and_nan(
        df["temperature_avg"].values,
        BOUNDS["temp_min"], BOUNDS["temp_max"])

    for col in ["discharge_capacity", "charge_capacity", "dc_internal_resistance"]:
        vals = df[col].values.copy()
        non_nan_mask = ~np.isnan(vals)
        if non_nan_mask.sum() > 0:
            keep = _iqr_mask(vals[non_nan_mask])
            non_nan_idx = np.where(non_nan_mask)[0]
            for idx, k in zip(non_nan_idx, keep):
                if not k:
                    vals[idx] = np.nan
        df[col] = vals

    for col in ["discharge_capacity", "charge_capacity",
                "dc_internal_resistance",
                "temperature_max", "temperature_avg",
                "discharge_energy", "charge_energy"]:
        df[col] = _forward_fill(df[col].values)
        if df[col].isna().any():
            mean_val = np.nanmean(df[col].values)
            df[col].fillna(mean_val, inplace=True)

    if len(df) < MIN_CYCLES_PER_CELL:
        return None

    init_cap = float(np.nanmean(
        df["discharge_capacity"].values[:INIT_CYCLES_AVG]))
    if init_cap <= 0:
        return None

    return {
        "filename": filepath.name,
        "barcode": raw.get("barcode", ""),
        "protocol": raw.get("protocol", ""),
        "cycle_index": df["cycle_index"].values.astype(np.int32),
        "discharge_capacity": df["discharge_capacity"].values.astype(np.float32),
        "charge_capacity": df["charge_capacity"].values.astype(np.float32),
        "dc_internal_resistance": df["dc_internal_resistance"].values.astype(np.float32),
        "temperature_max": df["temperature_max"].values.astype(np.float32),
        "temperature_avg": df["temperature_avg"].values.astype(np.float32),
        "discharge_energy": df["discharge_energy"].values.astype(np.float32),
        "charge_energy": df["charge_energy"].values.astype(np.float32),
        "initial_capacity": init_cap,
        "total_cycles": len(df),
    }

def load_all_cells(data_dir: Path, verbose: bool = True) -> Tuple[List[Dict], pd.DataFrame]:
    files = sorted(data_dir.glob("*_structure.json"))

    cells = []
    report_rows = []

    for fp in files:
        cell = load_and_clean_cell(str(fp))

        if cell is not None:
            cells.append(cell)
            status = "OK"
            cycles = cell["total_cycles"]
            ic = cell["initial_capacity"]
        else:
            status = "DISCARDED"
            cycles = 0
            ic = 0.0

        report_rows.append({
            "filename": fp.name,
            "status": status,
            "total_cycles": cycles,
            "initial_capacity": ic,
        })

        if verbose:
            tag = "OK" if status == "OK" else "DISCARD"
            print(f"{tag}|{fp.name}|{cycles}|{ic:.4f}")

    report = pd.DataFrame(report_rows)

    if verbose:
        n_ok = (report["status"] == "OK").sum()
        n_total = len(files)
        print(f"SUMMARY|TOTAL_FILES|{n_total}")
        print(f"SUMMARY|VALID_CELLS|{n_ok}")
        print(f"SUMMARY|DISCARDED_CELLS|{n_total - n_ok}")

    return cells, report

def save_cleaned_cells(cells: List[Dict], report: pd.DataFrame) -> None:
    out_pkl = RESULTS_DIR / "cleaned_cells.pkl"
    out_report = RESULTS_DIR / "cleaning_report.csv"

    with open(out_pkl, "wb") as f:
        pickle.dump(cells, f)

    report.to_csv(out_report, index=False)

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cells, report = load_all_cells(DATA_DIR, verbose=True)
    save_cleaned_cells(cells, report)

if __name__ == "__main__":
    main()