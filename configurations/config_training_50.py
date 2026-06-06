"""
Configuration for training with 50-cycle prediction horizon
Separate results directory to avoid overwriting 30-cycle model
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CUSTOM OUTPUT PATHS FOR 50-CYCLE MODEL (prevents overwriting)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# Custom results directory for 50-cycle model
RESULTS_DIR = PROJECT_ROOT / "results_50cycle"
MODEL_SAVE_DIR = RESULTS_DIR / "checkpoints"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Now import preprocessing config (for dataset constants)
from configurations.config_preprocessing import *

# Override any RESULTS_DIR from preprocessing if needed
# (This ensures all outputs go to our custom directory)

# ─────────────────────────────────────────────────────────────
# DATASET CONSTANTS
# ─────────────────────────────────────────────────────────────
NOMINAL_CAPACITY    = 1.1
SOH_EOL_THRESHOLD   = 0.80
INIT_CYCLES_AVG     = 5
MIN_CYCLES_PER_CELL = 20

# ─────────────────────────────────────────────────────────────
# FIXED FEATURE LIST - SOH-normalised features
# ─────────────────────────────────────────────────────────────
# All capacity-based features are now dimensionless ratios
# invariant to cell-to-cell capacity differences
# 
# IMPORTANT: delta_soh is REMOVED because:
#   soh(t) = soh_prev(t) + delta_soh(t) is a mathematical identity
#   The model would cheat by learning addition instead of degradation patterns

FEATURE_COLS = [
    "soh_prev",
    "coulombic_eff",
    "dc_internal_resistance",
    "temperature_max",
    "cycle_norm",
]

TARGET_COL = "soh"
N_FEATURES = len(FEATURE_COLS)  # Now 5 features (was 6)

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
RANDOM_SEED  = 42
SEQ_LEN      = 50                    # INCREASED from 30 to 50 (more history for long-term prediction)
PREDICTION_HORIZON = 50              # NEW: predict 50 cycles ahead (meaningful task)

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
MAX_EPOCHS    = 150
PATIENCE      = 20
LR_PATIENCE   = 8

# ─────────────────────────────────────────────────────────────
# DEPLOYMENT EVALUATION
# ─────────────────────────────────────────────────────────────
INFERENCE_DEVICE = "cpu"
N_INFERENCE_RUNS = 200

print(f"\n[CONFIG] 50-CYCLE MODEL CONFIGURATION LOADED")
print(f"[CONFIG] Results directory: {RESULTS_DIR}")
print(f"[CONFIG] Prediction horizon: {PREDICTION_HORIZON} cycles")
print(f"[CONFIG] Sequence length: {SEQ_LEN} cycles\n")