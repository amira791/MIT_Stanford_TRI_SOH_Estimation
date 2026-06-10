# configurations/config_preprocessing.py
"""
Configuration for preprocessing only (no torch dependency)
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")
RESULTS_DIR = ROOT_DIR / "preprocessing_results_2"
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATASET PARAMETERS
# ─────────────────────────────────────────────────────────────
NOMINAL_CAPACITY = 1.1
INIT_CYCLES_AVG = 5
MIN_CYCLES_PER_CELL = 20