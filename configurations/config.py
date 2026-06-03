# config.py - Updated for Option B (use all channels)
"""
Central configuration for SOH estimation project using ALL 140 JSON files
Each channel treated as separate battery cell
"""

import os
from pathlib import Path
import torch

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website\Data-driven prediction of battery cycle life before capacity degradation\FastCharge")
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROCESSED_CSV   = RESULTS_DIR / "soh_dataset.csv"
PROCESSED_PKL   = RESULTS_DIR / "soh_dataset.pkl"
SCALER_PATH     = RESULTS_DIR / "scaler.pkl"
MODEL_SAVE_DIR  = RESULTS_DIR / "checkpoints"
MODEL_SAVE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATASET PARAMETERS - Option B (all 140 files)
# ─────────────────────────────────────────────────────────────
NOMINAL_CAPACITY    = 1.1      # Ah (A123 APR18650M1A datasheet)
SOH_EOL_THRESHOLD   = 0.80     # 80% = end of life

# CRITICAL: Cycle 0 is corrupted (formation cycle with wrong capacity)
# From analysis: Cycle 0 = 1.5-1.9Ah (should be ~1.05Ah)
START_CYCLE_IDX     = 1        # Skip cycle 0 entirely
INIT_CYCLES_AVG     = 5        # Average cycles 1-5 for initial capacity
MIN_CYCLES_PER_CELL = 20       # Discard files with <20 cycles (after skipping cycle 0)

# Option B specific: We keep ALL 140 files
# Each channel (CH16, CH30, CH38, etc.) treated as separate cell
TOTAL_TRAINING_SAMPLES = 140   # Total JSON files to use
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
TARGET_COL   = "soh"
N_FEATURES   = len(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
RANDOM_SEED = 42
SEQ_LEN     = 30

# For Option B: Split by FILE (not by cell ID since we treat as separate)
SPLIT_BY_CELL_ID = False  # Different from original recommendation

# ─────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
CNN_CHANNELS   = [32, 64, 128]
CNN_KERNEL     = 3
MAMBA_D_MODEL  = 128
MAMBA_D_STATE  = 16
MAMBA_N_LAYERS = 2
MC_DROPOUT_P   = 0.15
MC_SAMPLES     = 50

# ─────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 100
PATIENCE      = 15
LR_PATIENCE   = 7

# ─────────────────────────────────────────────────────────────
# GPU CONFIGURATION
# ─────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

TRAINING_DEVICE = get_device()
INFERENCE_DEVICE = "cpu"
USE_MIXED_PRECISION = True
NUM_WORKERS = 4

# ─────────────────────────────────────────────────────────────
# DEPLOYMENT
# ─────────────────────────────────────────────────────────────
N_INFERENCE_RUNS = 200