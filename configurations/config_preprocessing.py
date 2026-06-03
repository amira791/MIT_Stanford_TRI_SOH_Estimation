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
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROCESSED_CSV = RESULTS_DIR / "soh_dataset.csv"
PROCESSED_PKL = RESULTS_DIR / "soh_dataset.pkl"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"
MODEL_SAVE_DIR = ROOT_DIR / "checkpoints"
MODEL_SAVE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATASET PARAMETERS
# ─────────────────────────────────────────────────────────────
NOMINAL_CAPACITY = 1.1
SOH_EOL_THRESHOLD = 0.80
START_CYCLE_IDX = 1
INIT_CYCLES_AVG = 5
MIN_CYCLES_PER_CELL = 20
TOTAL_TRAINING_SAMPLES = 140
TREAT_CHANNELS_AS_SEPARATE = True

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITION
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "discharge_capacity_prev",
    "charge_capacity",
    "dc_internal_resistance",
    "temperature_max",
    "temperature_avg",
    "cycle_index",
]
TARGET_COL = "soh"
N_FEATURES = len(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42
SEQ_LEN = 30
SPLIT_BY_CELL_ID = False