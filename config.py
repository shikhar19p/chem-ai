"""
config.py — Global constants and hyperparameters for the reactive extraction ML pipeline.
System: Propionic Acid + TBA / DES (Thymol:Menthol)
"""

import os

# ─── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Experimental factor levels ───────────────────────────────────────────────
CIN_LEVELS = [0.05, 0.10, 0.15, 0.20]    # Initial acid concentration (Normal)
TBA_LEVELS = [5, 10, 15, 20]              # Extractant TBA (wt %)
# DES molar ratio Thymol:Menthol encoded as a single float:
#   1:1   -> 1.0
#   1:1.5 -> 1.5
#   2:1   -> 2.0
DES_RATIO_LEVELS = [1.0, 1.5, 2.0]

TEMPERATURE_K = 306.0
PRESSURE_KPA = 101.32

# ─── Targets ──────────────────────────────────────────────────────────────────
TARGETS = ['KD', 'E_pct', 'Z', 'SF_min']

# ─── Feature sets ─────────────────────────────────────────────────────────────
BASE_FEATURES  = ['Cin', 'TBA_pct', 'DES_ratio_num']
GAMMA_FEATURES = BASE_FEATURES + ['gamma_aq', 'gamma_org', 'C_TBA_molar']
POLY_FEATURES  = BASE_FEATURES + [
    'Cin_sq', 'TBA_sq', 'DES_sq',
    'Cin_x_TBA', 'Cin_x_DES', 'TBA_x_DES'
]
FULL_FEATURES  = POLY_FEATURES + ['gamma_aq', 'gamma_org', 'C_TBA_molar']

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
RAW_DATA_CSV = os.path.join(DATA_DIR, 'raw', 'thesis_data.csv')
SYN_DATA_CSV = os.path.join(DATA_DIR, 'processed', 'synthetic_dataset.csv')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR  = os.path.join(OUTPUT_DIR, 'figures')
MODELS_DIR   = os.path.join(OUTPUT_DIR, 'models')
REPORTS_DIR  = os.path.join(OUTPUT_DIR, 'reports')

# ─── Cross-validation ─────────────────────────────────────────────────────────
CV_FOLDS   = 5
TEST_SIZE  = 0.2

# ─── Random Forest hyperparameter grid ────────────────────────────────────────
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
}

# ─── XGBoost hyperparameters ──────────────────────────────────────────────────
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'tree_method': 'hist',
    'n_jobs': -1,
}

# ─── GPR hyperparameters ──────────────────────────────────────────────────────
GPR_N_RESTARTS = 10

# ─── Bayesian Optimisation ────────────────────────────────────────────────────
BAYES_N_INIT   = 5
BAYES_N_ITER   = 30
BAYES_NEXT_N   = 5
BAYES_PBOUNDS  = {
    'Cin':           (0.03, 0.25),
    'TBA_pct':       (3.0,  25.0),
    'DES_ratio_num': (0.8,  2.2),
}

# ─── ANN hyperparameters ──────────────────────────────────────────────────────
ANN_HIDDEN_LAYERS  = [64, 32, 16]
ANN_EPOCHS         = 500
ANN_BATCH_SIZE     = 8
ANN_LEARNING_RATE  = 0.001
ANN_DROPOUT        = 0.2
ANN_L2             = 1e-4
ANN_PATIENCE       = 50

# ─── NRTL estimation parameters (literature-based, carboxylic acid/amine) ─────
NRTL_TAU_AW    = 1.8    # interaction param acid-water (at 298 K)
NRTL_TAU_WA    = 0.5    # interaction param water-acid
NRTL_ALPHA     = 0.3    # non-randomness factor
NRTL_K_EX_BASE = 28.0   # extraction equilibrium constant base (L/mol)
TBA_MW         = 185.35  # g/mol
DES_BASE_DENSITY = 0.92  # g/mL (approximate, varies with DES ratio)
