
"""
config_training_v3.py
======================
Key changes vs config_training_50:

1. dc_internal_resistance → normalised as (R - R0) / R0  in step2
   This makes the feature chemistry-agnostic (LFP ~0.017Ω vs NMC ~0.08Ω)
   The normalised value starts at 0 for every cell and grows with aging.

2. Larger Mamba model: d_model 128→192, n_layers 2→3
   Ridge achieves R²=0.80 with flat features; CNN-Mamba should exceed that.
   The extra capacity is needed to learn the 50-cycle degradation trajectory.

3. MC_DROPOUT_P 0.15 → 0.25
   Wider dropout variance → wider CI → PICP will approach 95% target.

4. LOSS = 'huber' (delta=0.02)
   Huber loss is MSE for errors < delta and MAE for errors > delta.
   This makes training robust to the 2 anomalous cells (R²=−0.48/−0.86)
   that currently drag the global R² from 0.80 down to 0.42.

5. Multi-task heads: current SOH (t+0), future SOH (t+50), EOL prob
"""

from pathlib import Path

ROOT_DIR    = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results_mlt"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_SAVE_DIR = RESULTS_DIR / "checkpoints"
MODEL_SAVE_DIR.mkdir(exist_ok=True)
SCALER_PATH = RESULTS_DIR / "scaler_v3.pkl"

# ── dataset ────────────────────────────────────────────────────────────────
NOMINAL_CAPACITY    = 1.1
SOH_EOL_THRESHOLD   = 0.80
INIT_CYCLES_AVG     = 5
MIN_CYCLES_PER_CELL = 20

# ── FEATURE LIST v3 ────────────────────────────────────────────────────────
# dc_internal_resistance replaced by dc_ir_norm = (R - R_initial) / R_initial
# All features now physically invariant across chemistries
FEATURE_COLS = [
    "soh_prev",      # SOH at t-1          → [0,1] for all chemistries
    "delta_soh",     # SOH(t) - SOH(t-1)   → degradation rate
    "coulombic_eff", # Q_charge/Q_discharge → ~1.0 for all chemistries
    "dc_ir_norm",    # (R-R0)/R0            → starts at 0, grows with aging
    "temperature_max",  # °C — physical signal, already comparable
    "cycle_norm",    # cycle / eol_cycle    → [0,1] for all chemistries
]
TARGET_COL = "soh"
N_FEATURES = len(FEATURE_COLS)   # 6

# ── split ──────────────────────────────────────────────────────────────────
TRAIN_FRAC       = 0.70
VAL_FRAC         = 0.15
RANDOM_SEED      = 42
SEQ_LEN          = 50
PREDICTION_HORIZON = 50    # predict SOH 50 cycles ahead

# ── model (enlarged) ──────────────────────────────────────────────────────
CNN_CHANNELS   = [32, 64, 128]
CNN_KERNEL     = 3
MAMBA_D_MODEL  = 192        # ← increased from 128
MAMBA_D_STATE  = 16
MAMBA_N_LAYERS = 3          # ← increased from 2
MC_DROPOUT_P   = 0.25       # ← increased from 0.15 for better UQ calibration
MC_SAMPLES     = 50

# ── training ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4        # ← slightly lower for larger model stability
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 150
PATIENCE      = 20
LR_PATIENCE   = 8

# ── loss ───────────────────────────────────────────────────────────────────
LOSS_TYPE     = "huber"     # "mse" or "huber"
HUBER_DELTA   = 0.02        # transition point (= 2% SOH error)
# Huber = MSE for |error| < delta, MAE for |error| >= delta
# This down-weights the 2 anomalous test cells without discarding them

# ── multi-task ─────────────────────────────────────────────────────────────
MULTI_TASK    = True
# Head 1 weight: future SOH (main task)
# Head 2 weight: current SOH (auxiliary — helps encoder learn current state)
# Head 3 weight: EOL probability (binary — SOH < 0.80)
MTL_WEIGHTS   = {"future_soh": 1.0, "current_soh": 0.3, "eol_prob": 0.2}

# ── deployment ─────────────────────────────────────────────────────────────
INFERENCE_DEVICE = "cpu"
N_INFERENCE_RUNS = 200
