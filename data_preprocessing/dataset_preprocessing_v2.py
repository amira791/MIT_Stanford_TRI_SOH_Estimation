# dataset_preprocessing_v3_fixed.py
"""
DATASET PREPROCESSING V3 - FIXED DATE_TIME_ISO
------------------------------------------------
Fixes the date_time_iso conversion error by handling strings properly.
"""

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy import signal
import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS2_DIR = Path(__file__).parent.parent / "results2"
RESULTS2_DIR.mkdir(exist_ok=True)

NOMINAL_CAPACITY = 1.1
INIT_CYCLES_AVG = 5
MIN_CYCLES_PER_CELL = 20

DATA_DIR = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")

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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def _smooth_temperature(temp_array: np.ndarray, window_length: int = 5) -> np.ndarray:
    if len(temp_array) < window_length:
        return temp_array
    return signal.medfilt(temp_array, kernel_size=window_length)


def _normalize_internal_resistance(ir_array: np.ndarray, batch: str) -> np.ndarray:
    if batch == "batch3":
        correction_factor = 1.02
        return ir_array * correction_factor
    return ir_array


def _detect_batch_from_filename(filename: str) -> str:
    if "2017-05" in filename or "batch1" in filename.lower():
        return "batch1"
    elif "2017-06" in filename or "batch2" in filename.lower():
        return "batch2"
    else:
        return "batch3"


def _parse_date_time(date_value) -> float:
    """Convert date_time_iso to numeric (seconds since epoch)"""
    if pd.isna(date_value) or date_value is None or date_value == "":
        return np.nan
    try:
        if isinstance(date_value, str):
            # Handle ISO format with timezone
            # Example: '2017-07-01T03:52:32+00:00'
            date_value = date_value.replace('+00:00', '').replace('Z', '')
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    dt = datetime.datetime.strptime(date_value, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
        elif isinstance(date_value, (int, float)):
            return float(date_value)
        return np.nan
    except:
        return np.nan


def _get_array_from_summary(summary, key, n, default=np.nan):
    """Get array from summary, handling date_time_iso specially"""
    arr = summary.get(key)
    if arr is None:
        return np.full(n, default, dtype=np.float64)
    
    # Special handling for date_time_iso
    if key == "date_time_iso":
        # Convert each date string to numeric
        numeric_dates = []
        for val in arr:
            numeric_dates.append(_parse_date_time(val))
        return np.array(numeric_dates, dtype=np.float64)
    
    # For other features, convert directly
    return np.array(arr, dtype=np.float64)


# ============================================================
# LOAD AND CLEAN CELL - EXTRACT ALL 10 FEATURES
# ============================================================

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

    # Helper to get arrays
    def _get(key, default=np.nan):
        arr = summary.get(key)
        if arr is None:
            return np.full(n, default, dtype=np.float64)
        
        # Special handling for date_time_iso
        if key == "date_time_iso":
            numeric_dates = []
            for val in arr:
                numeric_dates.append(_parse_date_time(val))
            return np.array(numeric_dates, dtype=np.float64)
        
        return np.array(arr, dtype=np.float64)

    # ============================================================
    # EXTRACT ALL 10 SUMMARY FEATURES
    # ============================================================
    df = pd.DataFrame({
        # 1. Cycle information
        "cycle_index": _get("cycle_index").astype(int),
        
        # 2-5. Capacity and Energy
        "discharge_capacity": _get("discharge_capacity"),
        "charge_capacity": _get("charge_capacity"),
        "discharge_energy": _get("discharge_energy"),
        "charge_energy": _get("charge_energy"),
        
        # 6. Resistance
        "dc_internal_resistance": _get("dc_internal_resistance"),
        
        # 7-9. Temperature (ALL THREE)
        "temperature_maximum": _get("temperature_maximum"),
        "temperature_average": _get("temperature_average"),
        "temperature_minimum": _get("temperature_minimum"),
        
        # 10. Timestamp (already numeric from _get)
        "date_time_iso_numeric": _get("date_time_iso"),
    })

    # Remove cycle 0
    df = df[df["cycle_index"] >= 1].copy()
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_CYCLES_PER_CELL:
        return None

    # Physical bounds
    df["discharge_capacity"] = _clip_and_nan(df["discharge_capacity"].values, BOUNDS["capacity_min"], BOUNDS["capacity_max"])
    df["charge_capacity"] = _clip_and_nan(df["charge_capacity"].values, BOUNDS["capacity_min"], BOUNDS["capacity_max"])
    df["dc_internal_resistance"] = _clip_and_nan(df["dc_internal_resistance"].values, BOUNDS["ir_min"], BOUNDS["ir_max"])
    df["temperature_maximum"] = _clip_and_nan(df["temperature_maximum"].values, BOUNDS["temp_min"], BOUNDS["temp_max"])
    df["temperature_average"] = _clip_and_nan(df["temperature_average"].values, BOUNDS["temp_min"], BOUNDS["temp_max"])
    df["temperature_minimum"] = _clip_and_nan(df["temperature_minimum"].values, BOUNDS["temp_min"], BOUNDS["temp_max"])

    # IQR outlier removal
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

    # Global anomaly fixes
    batch = _detect_batch_from_filename(filepath.name)
    df["temperature_average"] = _smooth_temperature(df["temperature_average"].values)
    df["temperature_maximum"] = _smooth_temperature(df["temperature_maximum"].values)
    df["temperature_minimum"] = _smooth_temperature(df["temperature_minimum"].values)
    df["dc_internal_resistance"] = _normalize_internal_resistance(df["dc_internal_resistance"].values, batch)

    # Missing value imputation (ALL columns)
    for col in ["discharge_capacity", "charge_capacity", "dc_internal_resistance",
                "temperature_maximum", "temperature_average", "temperature_minimum",
                "discharge_energy", "charge_energy", "date_time_iso_numeric"]:
        df[col] = _forward_fill(df[col].values)
        if df[col].isna().any():
            mean_val = np.nanmean(df[col].values)
            df[col].fillna(mean_val, inplace=True)

    if len(df) < MIN_CYCLES_PER_CELL:
        return None

    init_cap = float(np.nanmean(df["discharge_capacity"].values[:INIT_CYCLES_AVG]))
    if init_cap <= 0:
        return None

    # Extract channel ID
    channel = "UNKNOWN"
    if "_CH" in filepath.name:
        parts = filepath.name.split('_')
        for part in parts:
            if part.startswith("CH"):
                channel = part
                break

    # ============================================================
    # RETURN ALL FEATURES
    # ============================================================
    return {
        "filename": filepath.name,
        "barcode": raw.get("barcode", ""),
        "channel": channel,
        "batch": batch,
        "protocol": raw.get("protocol", ""),
        
        # ALL 10 summary features
        "cycle_index": df["cycle_index"].values.astype(np.int32),
        "discharge_capacity": df["discharge_capacity"].values.astype(np.float32),
        "charge_capacity": df["charge_capacity"].values.astype(np.float32),
        "discharge_energy": df["discharge_energy"].values.astype(np.float32),
        "charge_energy": df["charge_energy"].values.astype(np.float32),
        "dc_internal_resistance": df["dc_internal_resistance"].values.astype(np.float32),
        "temperature_maximum": df["temperature_maximum"].values.astype(np.float32),
        "temperature_average": df["temperature_average"].values.astype(np.float32),
        "temperature_minimum": df["temperature_minimum"].values.astype(np.float32),
        "date_time_iso_numeric": df["date_time_iso_numeric"].values.astype(np.float32),
        
        # Metadata
        "initial_capacity": init_cap,
        "total_cycles": len(df),
    }


# ============================================================
# LOAD ALL CELLS
# ============================================================

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
    
    return cells, report


# ============================================================
# GROUP AND SAVE
# ============================================================

def group_by_physical_cell(cells: List[Dict]) -> Dict[str, List[Dict]]:
    physical_cells = {}
    for cell_data in cells:
        barcode = cell_data["barcode"]
        if barcode not in physical_cells:
            physical_cells[barcode] = []
        physical_cells[barcode].append(cell_data)
    return physical_cells


def save_cleaned_cells(cells: List[Dict], report: pd.DataFrame) -> None:
    RESULTS2_DIR.mkdir(exist_ok=True)
    
    out_pkl = RESULTS2_DIR / "cleaned_cells_all_features.pkl"
    out_report = RESULTS2_DIR / "cleaning_report.csv"
    
    with open(out_pkl, "wb") as f:
        pickle.dump(cells, f)
    report.to_csv(out_report, index=False)
    
    # Save physical cell grouping
    physical_cells = group_by_physical_cell(cells)
    physical_summary = []
    for barcode, channels in physical_cells.items():
        physical_summary.append({
            "barcode": barcode,
            "num_channels": len(channels),
            "channels": [c["filename"] for c in channels],
            "total_cycles_max": max(c["total_cycles"] for c in channels),
            "batches": list(set(c["batch"] for c in channels))
        })
    
    physical_df = pd.DataFrame(physical_summary)
    physical_df.to_csv(RESULTS2_DIR / "physical_cells_summary.csv", index=False)
    
    with open(RESULTS2_DIR / "physical_cells_grouping.pkl", "wb") as f:
        pickle.dump(physical_cells, f)
    
    print(f"\n{'='*60}")
    print(f"SAVED FILES (to results2/):")
    print(f"  {out_pkl}")
    print(f"  {out_report}")
    print(f"  {RESULTS2_DIR / 'physical_cells_summary.csv'}")
    print(f"  {RESULTS2_DIR / 'physical_cells_grouping.pkl'}")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS2_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("PREPROCESSING V3: MIT-Stanford Battery Dataset")
    print("="*60)
    print("\nEXTRACTING ALL 10 SUMMARY FEATURES")
    print("  ✅ cycle_index")
    print("  ✅ discharge_capacity")
    print("  ✅ charge_capacity")
    print("  ✅ discharge_energy")
    print("  ✅ charge_energy")
    print("  ✅ dc_internal_resistance")
    print("  ✅ temperature_maximum")
    print("  ✅ temperature_average")
    print("  ✅ temperature_minimum")
    print("  ✅ date_time_iso (converted to numeric)")
    print("="*60 + "\n")

    cells, report = load_all_cells(DATA_DIR, verbose=True)
    save_cleaned_cells(cells, report)
    
    n_ok = (report["status"] == "OK").sum()
    n_total = len(report)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total files: {n_total}")
    print(f"  Valid cells: {n_ok}")
    print(f"  Discarded: {n_total - n_ok}")
    print(f"  Unique physical cells: {len(group_by_physical_cell(cells))}")
    print(f"{'='*60}")
    print("\nPREPROCESSING COMPLETE")
    print(f"Results saved to: {RESULTS2_DIR}")


if __name__ == "__main__":
    main()