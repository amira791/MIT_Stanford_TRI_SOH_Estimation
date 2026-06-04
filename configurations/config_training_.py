# configurations/config_training.py
"""
Configuration for training only (includes model hyperparameters)
FIXED: Uses SOH-normalised features instead of raw capacity values
"""

from configurations.config_preprocessing import *

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
FEATURE_COLS = [
    "soh_prev",                 # SOH at cycle t-1 (dimensionless)
    "delta_soh",                # SOH(t) - SOH(t-1) - degradation rate
    "coulombic_eff",            # charge_cap / discharge_cap(t-1) - efficiency
    "dc_internal_resistance",   # resistance (Ohms) - already cell-invariant
    "temperature_max",          # peak temperature (°C) - already cell-invariant
    "cycle_norm",               # cycle / eol_cycle ∈ [0,1] - relative position
]

TARGET_COL = "soh"
N_FEATURES = len(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
RANDOM_SEED  = 42
SEQ_LEN      = 30

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