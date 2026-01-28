"""
Centralized configuration for cardiovascular risk prediction pipeline.

This module defines all shared constants across the three-stage pipeline:
  1. Preprocessing (cardiovascular_preprocessing.py)
  2. HPC Modeling (cardiovascular_optuna_gpu.py)
  3. Interpretability (cardio_SHAP.ipynb)

By centralizing these values, we ensure consistency and make it easy
to adjust parameters without searching through multiple files.

Usage:
------
    from config import RANDOM_STATE, N_TRIALS, CV_FOLDS
    
    np.random.seed(RANDOM_STATE)
    study = optuna.create_study(...)
    study.optimize(objective, n_trials=N_TRIALS)
"""

import os
from pathlib import Path

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Seed for numpy, scikit-learn, XGBoost, CatBoost random number generators
RANDOM_STATE = 42

# =============================================================================
# HYPERPARAMETER OPTIMIZATION (Optuna)
# =============================================================================
# Number of trials per model (total: N_TRIALS × 3 models)
N_TRIALS = 100

# Cross-validation folds for model evaluation during tuning
CV_FOLDS = 5

# =============================================================================
# DATA COMPRESSION
# =============================================================================
# joblib compression level (0-9):
#   0 = no compression (fastest, largest files)
#   3 = optimal balance of speed and size (recommended)
#   9 = maximum compression (slowest, smallest files)
# For this project: level 3 is recommended (typical file reduction ~60-70%)
COMPRESSION_LEVEL = 3

# =============================================================================
# HPC CONFIGURATION (Alex Cluster at FAU)
# =============================================================================
# Base directory on HPC cluster
BASE_DIR_HPC = Path("/home/vault/b192aa/b192aa44/cardio_project")

# Detect if running on HPC (via Slurm environment variable)
HPC_ENABLED = "SLURM_JOB_ID" in os.environ

# =============================================================================
# CLINICAL THRESHOLDING
# =============================================================================
# Minimum recall (sensitivity) threshold for clinical safety
# In medical screening, we tolerate lower precision (more false positives)
# to minimize false negatives (missed diagnoses)
MIN_RECALL_THRESHOLD = 0.85

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================
# Train/test split ratio
TEST_SIZE = 0.20

# =============================================================================
# OPTUNA SAMPLING
# =============================================================================
# Optuna sampler (TPESampler = Tree-Parzen Estimator — Bayesian optimization)
OPTUNA_SAMPLER = "TPE"

# =============================================================================
# GPU CONFIGURATION (for HPC)
# =============================================================================
# XGBoost settings
XGBOOST_GPU_DEVICE = "cuda"
XGBOOST_TREE_METHOD = "hist"

# CatBoost settings
CATBOOST_GPU_DEVICE = "0"  # GPU index
CATBOOST_TASK_TYPE = "GPU"

# =============================================================================
# PATHS (Relative or Absolute)
# =============================================================================
# These are typically resolved at runtime using Path('.').resolve()
# to ensure portability across different execution environments
DATA_SUBDIRS = {
    "processed": "processed",           # Preprocessed data
    "models": "models_/outputs/models", # Trained models
    "outputs": "outputs",               # Results
    "plots": "outputs/plots",           # Visualizations
    "logs": "outputs/logs",             # Execution logs
    "shap": "outputs/shap",             # SHAP artifacts
}
