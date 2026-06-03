# configurations/config_training.py
"""
Configuration for training only (includes model hyperparameters)
"""

from configurations.config_preprocessing import *

# ─────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
CNN_CHANNELS = [32, 64, 128]
CNN_KERNEL = 3
MAMBA_D_MODEL = 128
MAMBA_D_STATE = 16
MAMBA_N_LAYERS = 2
MC_DROPOUT_P = 0.15
MC_SAMPLES = 50

# ─────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15
LR_PATIENCE = 7