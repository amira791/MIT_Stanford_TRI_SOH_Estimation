# configurations/feature_config.py
"""
Feature Configuration for SOH and RUL Prediction
Simple feature set - calculations done in training script
"""








# ─────────────────────────────────────────────────────────────
# PATHS TO CLEANED DATASET (from preprocessing)
# ─────────────────────────────────────────────────────────────

from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Results directory (where preprocessing saved the cleaned data)
RESULTS_DIR = ROOT_DIR / "results"

# Cleaned dataset files
CLEANED_CELLS_PKL = RESULTS_DIR / "cleaned_cells.pkl"
CLEANING_REPORT_CSV = RESULTS_DIR / "cleaning_report.csv"
PHYSICAL_CELLS_SUMMARY_CSV = RESULTS_DIR / "physical_cells_summary.csv"
PHYSICAL_CELLS_GROUPING_PKL = RESULTS_DIR / "physical_cells_grouping.pkl"


# ─────────────────────────────────────────────────────────────
# FINAL FEATURE SET (7 features only)
# ─────────────────────────────────────────────────────────────

FINAL_FEATURES = [
    "cycle_index",                      # Temporal position
    "dc_internal_resistance",           # Health indicator
    "temperature_avg",                  # Stress factor
    "charge_capacity",                  # Direct measurement
    "charge_energy",                    # Energy measurement
    "coulombic_efficiency_lagged_1",    # CE from previous cycle
    "coulombic_efficiency_lagged_2",    # CE from 2 cycles ago (trend)
]

# ─────────────────────────────────────────────────────────────
# TARGETS
# ─────────────────────────────────────────────────────────────

SOH_TARGET = "soh"           # State of Health (0-1 or 0-100%)
RUL_TARGET = "rul"           # Remaining Useful Life (cycles)

# ─────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────

SOH_EOL_THRESHOLD = 0.80     # 80% of nominal capacity
NOMINAL_CAPACITY = 1.1       # Ah

# ─────────────────────────────────────────────────────────────
# SEQUENCE PARAMETERS (for CNN-Mamba-UQ)
# ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 50         # Number of past cycles for sequence input

# ─────────────────────────────────────────────────────────────
# CONFIG DICTIONARY
# ─────────────────────────────────────────────────────────────

FEATURE_CONFIG = {
    "features": FINAL_FEATURES,
    "soh_target": SOH_TARGET,
    "rul_target": RUL_TARGET,
    "eol_threshold": SOH_EOL_THRESHOLD,
    "nominal_capacity": NOMINAL_CAPACITY,
    "sequence_length": SEQUENCE_LENGTH,
}

