# Technical Execution Guide: Cardiovascular Risk Prediction Pipeline

## Overview

This guide provides step-by-step instructions for executing a three-stage cardiovascular risk prediction pipeline:

1. **Preprocessing Stage**: `cardiovascular_preprocessing.py` (generates model-ready artifacts)
2. **HPC Modeling Stage**: `cardiovascular_optuna_gpu.py` (hyperparameter optimization on GPU cluster)
3. **Validation/Inference Stage**: `cardiovascular_models.ipynb` (local evaluation and model validation)

The project uses centralized configuration (`config.py`) and shared utilities (`cardio_utils.py`) to ensure consistency across all stages.

---

## 1. Environment Setup & Dependencies

### 1.1 Python Version Requirement

**Required:** Python 3.12 (as specified in the Slurm job script)

### 1.2 Core Dependencies

Install all required libraries:

```bash
pip install scikit-learn optuna xgboost catboost joblib pandas numpy matplotlib seaborn
```

**Recommended:** Create a virtual environment or conda environment:

```bash
# Using conda
conda create -n cardio python=3.12
conda activate cardio
pip install -r requirements.txt

# Using venv
python3.12 -m venv cardio_env
source cardio_env/bin/activate  # Linux/Mac
cardio_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 1.3 Critical Environment Variable

**PYTHONPATH Configuration:**

The `cardio_utils.py` and `config.py` modules must be importable from any script or notebook in the pipeline.

```bash
# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"

# Windows Command Prompt
set PYTHONPATH=%PYTHONPATH%;%CD%
```

**Why This Matters:**
- `cardio_utils.py` contains the essential `SklearnCompatibleWrapper` class
- `config.py` centralizes all hyperparameters (RANDOM_STATE, N_TRIALS, CV_FOLDS)
- Without proper PYTHONPATH, you'll get `ModuleNotFoundError: No module named 'cardio_utils'`

---

## 2. Prerequisites: Data Artifacts from Preprocessing

### 2.1 Run Preprocessing Stage

**Before training any models**, you must run the preprocessing script to generate model-ready artifacts:

```bash
python cardiovascular_preprocessing.py
```

### 2.2 Expected Output Structure

The preprocessing script generates the following directory structure:

```
processed/
├── X_train_ready.joblib         # Preprocessed training features (scaled, encoded)
├── X_test_ready.joblib          # Preprocessed test features
├── y_train_ready.joblib         # Training labels (0/1)
├── y_test_ready.joblib          # Test labels (0/1)
├── feature_names.joblib         # Feature names after transformation
└── preprocessor.joblib          # Fitted ColumnTransformer (for inference)
```

### 2.3 Data Compression

All `.joblib` files use compression level 3 (defined in `config.py`) for:
- **Size Reduction:** ~60-70% smaller than uncompressed
- **Speed:** Optimal balance between I/O and decompression time
- **Storage Efficiency:** Important for large datasets on HPC clusters

### 2.4 Verification

Verify that all artifacts were created successfully:

```bash
# Linux/Mac
ls -lh processed/*.joblib

# Windows PowerShell
Get-ChildItem processed\*.joblib | Format-Table Name, Length
```

You should see 6 `.joblib` files totaling several MB.

---

## 3. Execution: HPC Modeling with `cardiovascular_optuna_gpu.py`

### 3.1 GPU Requirements

The script is optimized for GPU-accelerated training with these requirements:

**XGBoost Configuration:**
- `tree_method="hist"` (GPU-accelerated histogram method)
- `device="cuda"` (CUDA-enabled GPU)

**CatBoost Configuration:**
- `task_type="GPU"` (GPU training mode)
- `devices="0"` (GPU device index)

### 3.2 Slurm Job Submission (HPC Clusters)

#### 3.2.1 Slurm Script Template

Create or modify `run_optuna_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cardio_optuna
#SBATCH --output=/path/to/project/outputs/logs/slurm_%j.out
#SBATCH --error=/path/to/project/outputs/logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a40:1        # Request 1 A40 GPU (adjust as needed)
#SBATCH --partition=a40         # GPU partition (cluster-specific)
#SBATCH --cpus-per-task=8       # 8 CPU cores

source /etc/profile

# Strict error handling
set -euo pipefail

# Fix unbound LD_LIBRARY_PATH variable warning
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

module purge
module unload python
module load python/3.12-conda

cd /path/to/project

# Ensure shared utilities are importable
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python /path/to/project/cardiovascular_optuna_gpu.py
```

#### 3.2.2 Key Slurm Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--gres=gpu:a40:1` | 1 A40 GPU | GPU resource allocation |
| `--cpus-per-task=8` | 8 CPU cores | Parallel data loading/preprocessing |
| `--time=04:00:00` | 4 hours | Job time limit (adjust based on N_TRIALS) |
| `--partition=a40` | a40 | GPU partition (cluster-specific) |

#### 3.2.3 Submit Job

```bash
sbatch run_optuna_job.sh
```

**Monitor Job Status:**

```bash
# Check job queue
squeue -u $USER

# View live output (replace JOB_ID)
tail -f outputs/logs/slurm_JOB_ID.out

# Check for errors
tail -f outputs/logs/slurm_JOB_ID.err
```

### 3.3 Output Locations

Based on `DATA_SUBDIRS` in `config.py`, the script generates outputs in:

```
outputs/
├── models/
│   └── best_model/
│       ├── best_model.joblib        # Calibrated best model (ready for deployment)
│       └── metadata.json            # Hyperparameters + performance metrics
├── plots/
│   ├── roc_curve_TIMESTAMP.png
│   ├── pr_curve_TIMESTAMP.png
│   ├── calibration_plot_TIMESTAMP.png
│   └── confusion_matrix_TIMESTAMP.png
└── logs/
    ├── run_YYYYMMDD_HHMMSS.log      # Detailed execution log
    └── slurm_JOBID.out/err          # Slurm job logs
```

### 3.4 Execution Flow (What Happens Internally)

1. **Load Preprocessed Data**: Loads `X_train_ready.joblib`, `X_test_ready.joblib`, etc.

2. **Optuna Optimization**: Runs 100 trials per model (configurable via `N_TRIALS`):
   - Logistic Regression with L1/L2 regularization
   - XGBoost with GPU acceleration
   - CatBoost with GPU acceleration

3. **Cross-Validation**: Each trial evaluated with 5-fold stratified CV (configurable via `CV_FOLDS`)

4. **Model Selection**: Best model selected by highest **PR-AUC** (Precision-Recall AUC)
   - **Why PR-AUC?** With only ~8% positive cases (disease), traditional accuracy and ROC-AUC are misleading. PR-AUC focuses on the minority class.

5. **Isotonic Calibration**: Best model calibrated using `CalibratedClassifierCV` with isotonic regression

6. **Clinical Thresholding**: Decision threshold optimized to maximize precision while maintaining recall ≥ 0.85 (configurable via `MIN_RECALL_THRESHOLD`)

7. **Visualization & Logging**: Generates ROC curves, PR curves, calibration plots, and confusion matrices

### 3.5 Local Execution (No HPC)

If running locally without Slurm:

```bash
cd /path/to/project
export PYTHONPATH="$(pwd):$PYTHONPATH"
python cardiovascular_optuna_gpu.py
```

**Note:** GPU acceleration requires CUDA-compatible GPU and drivers. For CPU-only execution, modify `config.py`:

```python
# CPU Configuration
XGBOOST_GPU_DEVICE = "cpu"
XGBOOST_TREE_METHOD = "hist"
CATBOOST_TASK_TYPE = "CPU"
```

---

## 4. Execution: Local Validation with `cardiovascular_models.ipynb`

### 4.1 Open Notebook

```bash
jupyter notebook cardiovascular_models.ipynb
```

Or use VS Code with Jupyter extension.

### 4.2 Load Model-Ready Artifacts

Execute the first cell to load preprocessed data:

```python
import joblib
from pathlib import Path

data_path = Path("processed")
X_train = joblib.load(data_path / "X_train_ready.joblib")
X_test = joblib.load(data_path / "X_test_ready.joblib")
y_train = joblib.load(data_path / "y_train_ready.joblib")
y_test = joblib.load(data_path / "y_test_ready.joblib")
feature_names = joblib.load(data_path / "feature_names.joblib")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Number of features: {len(feature_names)}")
```

### 4.3 Model Compatibility Layer (CRITICAL)

**Always use `SklearnCompatibleWrapper`** from `cardio_utils.py` for XGBoost and CatBoost models:

```python
from cardio_utils import SklearnCompatibleWrapper
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Example: XGBoost
xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
wrapped_xgb = SklearnCompatibleWrapper(xgb)

# Example: CatBoost
cat = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)
wrapped_cat = SklearnCompatibleWrapper(cat)
```

#### Why the Wrapper is Essential

The `SklearnCompatibleWrapper` solves critical compatibility issues:

1. **sklearn Calibration Compatibility**:
   - Implements `__sklearn_tags__()` method required by `CalibratedClassifierCV` in sklearn >= 1.3
   - Without wrapper: `TypeError: object has no attribute '__sklearn_tags__'`

2. **Attribute Delegation**:
   - Delegates `feature_importances_`, `get_booster()`, and other attributes via `__getattr__()`
   - Enables access to model-specific attributes and methods

3. **sklearn Cloning Support**:
   - Inherits from `BaseEstimator` for proper `sklearn.base.clone()` compatibility
   - Required for cross-validation and ensemble methods

**Without Wrapper:**
```python
# This will FAIL with CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
xgb = XGBClassifier(n_estimators=100)
calibrated = CalibratedClassifierCV(xgb)  # ERROR!
```

**With Wrapper:**
```python
# This works correctly
wrapped_xgb = SklearnCompatibleWrapper(XGBClassifier(n_estimators=100))
calibrated = CalibratedClassifierCV(wrapped_xgb)  # Success!
calibrated.fit(X_train, y_train)
```

### 4.4 Cross-Validation with Unified Function

Use the `cross_validate_model()` function from `cardio_utils.py`:

```python
from cardio_utils import cross_validate_model
from sklearn.model_selection import StratifiedKFold
from config import RANDOM_STATE, CV_FOLDS

# Create stratified CV strategy
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Evaluate model
logloss, pr_auc = cross_validate_model(wrapped_xgb, X_train, y_train, cv)
print(f"Cross-Validation Results:")
print(f"  PR-AUC:    {pr_auc:.4f}")
print(f"  Log Loss:  {logloss:.4f}")
```

**What This Function Does:**
- Performs stratified k-fold cross-validation (preserves class distribution)
- Handles early stopping for XGBoost/CatBoost automatically
- Works with both DataFrames and numpy arrays via `_subset_rows()` helper
- Returns mean Log Loss and mean PR-AUC across folds

### 4.5 Model Training and Calibration

Train the wrapped model on full training set and apply isotonic calibration:

```python
from sklearn.calibration import CalibratedClassifierCV

# Train base model
wrapped_xgb.fit(X_train, y_train)

# Apply isotonic calibration with 5-fold CV
calibrated_model = CalibratedClassifierCV(
    estimator=wrapped_xgb, 
    method="isotonic", 
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Generate predictions
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
```

### 4.6 Evaluate Performance

```python
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    log_loss, 
    brier_score_loss
)

print("Test Set Performance:")
print(f"  ROC-AUC:      {roc_auc_score(y_test, y_proba):.4f}")
print(f"  PR-AUC:       {average_precision_score(y_test, y_proba):.4f}")
print(f"  Log Loss:     {log_loss(y_test, y_proba):.4f}")
print(f"  Brier Score:  {brier_score_loss(y_test, y_proba):.4f}")
```

---

## 5. Troubleshooting Common Issues

### Issue 1: `processed/` Directory Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'processed/X_train_ready.joblib'
```

**Cause:** Preprocessing stage was not run or failed.

**Solution:**
```bash
python cardiovascular_preprocessing.py
```

**Verification:**
```bash
ls -lh processed/*.joblib  # Should show 6 files
```

**Fallback (Manual Recovery):**

If preprocessing partially failed, you can load raw splits and reapply the preprocessor:

```python
import pandas as pd
import joblib
from pathlib import Path

data_path = Path("processed")

# Load raw splits
X_train = pd.read_csv(data_path / "X_train_raw.csv")
X_test = pd.read_csv(data_path / "X_test_raw.csv")
y_train = pd.read_csv(data_path / "y_train_raw.csv").squeeze()
y_test = pd.read_csv(data_path / "y_test_raw.csv").squeeze()

# Load and apply fitted preprocessor
preprocessor = joblib.load(data_path / "preprocessor.joblib")
X_train_ready = preprocessor.transform(X_train)
X_test_ready = preprocessor.transform(X_test)

# Save processed arrays
joblib.dump(X_train_ready, data_path / "X_train_ready.joblib", compress=3)
joblib.dump(X_test_ready, data_path / "X_test_ready.joblib", compress=3)
```

---

### Issue 2: `ModuleNotFoundError: No module named 'cardio_utils'`

**Error:**
```
ModuleNotFoundError: No module named 'cardio_utils'
```

**Cause:** Python cannot find the `cardio_utils.py` or `config.py` modules.

**Solution 1 - Set PYTHONPATH:**
```bash
# Linux/Mac
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Windows PowerShell
$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"
```

**Solution 2 - Run from Project Root:**
```bash
cd /path/to/project
python cardiovascular_optuna_gpu.py
```

**Solution 3 - Add to Jupyter Kernel:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Now you can import
from cardio_utils import SklearnCompatibleWrapper
from config import RANDOM_STATE, N_TRIALS
```

**Verification:**
```python
import sys
print("Python search paths:")
for path in sys.path:
    print(f"  {path}")

# Should include your project directory
```

---

### Issue 3: GPU Not Detected on HPC

**Error:**
```
RuntimeError: No GPU detected. Please check CUDA installation.
```

**Diagnostic Steps:**

1. **Verify GPU Allocation:**
```bash
nvidia-smi
```
Expected output: Shows allocated GPU (e.g., A40 with memory usage)

2. **Check Slurm Logs:**
```bash
cat outputs/logs/slurm_<JOB_ID>.err
grep -i "error\|warning" outputs/logs/slurm_<JOB_ID>.err
```

3. **Verify CUDA Module:**
```bash
module list
# Should show cuda module loaded
```

4. **Test CUDA in Python:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

**Solution:**

If GPU is not available, modify `config.py` for CPU execution:

```python
# Fallback to CPU
XGBOOST_GPU_DEVICE = "cpu"
XGBOOST_TREE_METHOD = "hist"
CATBOOST_TASK_TYPE = "CPU"
```

---

### Issue 4: Calibration Fails with XGBoost/CatBoost

**Error:**
```
AttributeError: 'XGBClassifier' object has no attribute '__sklearn_tags__'
```
or
```
TypeError: clone() missing required positional argument
```

**Cause:** Missing `SklearnCompatibleWrapper` around tree-based models.

**Incorrect Usage:**
```python
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

xgb = XGBClassifier(n_estimators=100)
calibrated = CalibratedClassifierCV(xgb)  # FAILS!
```

**Correct Usage:**
```python
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from cardio_utils import SklearnCompatibleWrapper

xgb = XGBClassifier(n_estimators=100)
wrapped_xgb = SklearnCompatibleWrapper(xgb)  # Wrap first!
calibrated = CalibratedClassifierCV(wrapped_xgb)  # Now works!
calibrated.fit(X_train, y_train)
```

**Remember:** Always wrap XGBoost and CatBoost models before:
- Using `CalibratedClassifierCV`
- Using `cross_validate_model()` from `cardio_utils`
- Performing cross-validation or calibration

---

### Issue 5: Memory Error During Training

**Error:**
```
MemoryError: Unable to allocate array with shape (70000, 50000)
```

**Cause:** Insufficient RAM for large datasets or high-dimensional feature spaces.

**Solution 1 - Reduce Batch Size:**

Modify XGBoost parameters:
```python
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    max_bin=256,  # Reduce histogram bins (default: 256)
    tree_method="hist",  # Memory-efficient histogram method
)
```

**Solution 2 - Use Sparse Matrices:**

If data has many zero values after one-hot encoding:
```python
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

# Use sparse output in preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=True), categorical_cols),
        # ... other transformers
    ]
)
```

**Solution 3 - Increase HPC Memory:**

Modify Slurm script:
```bash
#SBATCH --mem=64G  # Request 64GB RAM (adjust as needed)
```

---

### Issue 6: Early Stopping Not Working

**Error:** Model trains for all epochs despite validation loss not improving.

**Cause:** Early stopping callback not properly configured.

**Solution for XGBoost:**
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    eval_metric="logloss",
    random_state=42
)

# Must provide eval_set during fit
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

**Solution for CatBoost:**
```python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=1000,
    early_stopping_rounds=50,
    eval_metric="Logloss",
    random_state=42,
    verbose=False
)

# Must provide eval_set during fit
cat.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)
)
```

**Note:** The `cross_validate_model()` function in `cardio_utils.py` automatically handles early stopping for both XGBoost and CatBoost.

---

## 6. Configuration Reference

All hyperparameters are centralized in `config.py` for consistency across the pipeline.

### 6.1 Core Configuration

| Parameter | Default Value | Purpose |
|-----------|---------------|---------|
| `RANDOM_STATE` | 42 | Reproducibility seed for all random operations |
| `N_TRIALS` | 100 | Number of Optuna trials per model (total: 300 trials) |
| `CV_FOLDS` | 5 | Number of cross-validation folds |
| `MIN_RECALL_THRESHOLD` | 0.85 | Minimum recall for clinical decision threshold |
| `COMPRESSION_LEVEL` | 3 | joblib compression level (0-9) |
| `TEST_SIZE` | 0.20 | Train/test split ratio (80/20) |

### 6.2 HPC Configuration

| Parameter | Default Value | Purpose |
|-----------|---------------|---------|
| `BASE_DIR_HPC` | `/path/to/project` | Base directory on HPC cluster |
| `HPC_ENABLED` | Auto-detected | True if `SLURM_JOB_ID` in environment |

### 6.3 GPU Configuration

| Parameter | XGBoost Value | CatBoost Value | Purpose |
|-----------|---------------|----------------|---------|
| Device/Task Type | `"cuda"` | `"GPU"` | Enable GPU acceleration |
| Tree Method | `"hist"` | N/A | GPU-optimized histogram method |
| Device Index | N/A | `"0"` | GPU device to use |

### 6.4 Data Paths

| Key | Default Path | Contents |
|-----|-------------|----------|
| `processed` | `processed/` | Preprocessed `.joblib` artifacts |
| `models` | `models_/outputs/models/` | Trained model checkpoints |
| `outputs` | `outputs/` | General outputs |
| `plots` | `outputs/plots/` | Visualizations (ROC, PR curves, etc.) |
| `logs` | `outputs/logs/` | Execution logs |

### 6.5 Modifying Configuration

To adjust parameters, edit `config.py`:

```python
# Example: Increase trials for better optimization
N_TRIALS = 200  # More thorough search (slower)

# Example: Reduce CV folds for faster iteration
CV_FOLDS = 3  # Faster but less robust estimates

# Example: Stricter recall constraint
MIN_RECALL_THRESHOLD = 0.90  # Fewer false negatives (more false positives)
```

**Important:** After modifying `config.py`, re-run affected pipeline stages:
- Changed `N_TRIALS` or `CV_FOLDS`? → Re-run `cardiovascular_optuna_gpu.py`
- Changed `TEST_SIZE` or `RANDOM_STATE`? → Re-run entire pipeline from preprocessing

---

## 7. Execution Checklist

Use this checklist to ensure all prerequisites are met before running the pipeline.

### 7.1 Environment Setup
- [ ] Python 3.12 environment active
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `PYTHONPATH` includes project root directory
- [ ] Jupyter kernel configured (for notebook execution)

### 7.2 Data Preparation
- [ ] `HeartDisease.csv` exists in project root
- [ ] `processed/` directory exists
- [ ] All 6 `.joblib` artifacts present in `processed/` directory:
  - [ ] `X_train_ready.joblib`
  - [ ] `X_test_ready.joblib`
  - [ ] `y_train_ready.joblib`
  - [ ] `y_test_ready.joblib`
  - [ ] `feature_names.joblib`
  - [ ] `preprocessor.joblib`

### 7.3 HPC Configuration (If Applicable)
- [ ] Slurm script (`run_optuna_job.sh`) configured with correct paths
- [ ] GPU partition and resources specified correctly
- [ ] Output directories exist on HPC cluster:
  - [ ] `outputs/logs/`
  - [ ] `outputs/models/`
  - [ ] `outputs/plots/`
- [ ] Python module loaded (e.g., `module load python/3.12-conda`)
- [ ] CUDA/GPU drivers available (`nvidia-smi` works)

### 7.4 Local Configuration (If Applicable)
- [ ] Jupyter notebook server running or VS Code Jupyter extension installed
- [ ] Local output directories exist
- [ ] Sufficient disk space (~500MB for outputs)
- [ ] (Optional) GPU drivers installed if using local GPU

### 7.5 Code Dependencies
- [ ] `cardio_utils.py` present and importable
- [ ] `config.py` present and importable
- [ ] `cardiovascular_preprocessing.py` executable
- [ ] `cardiovascular_optuna_gpu.py` executable
- [ ] `cardiovascular_models.ipynb` openable

---

## 8. Execution Order and Workflow

Follow this workflow for a complete pipeline run:

### Stage 1: Preprocessing (One-Time Setup)

```bash
# Verify data file exists
ls HeartDisease.csv

# Run preprocessing
python cardiovascular_preprocessing.py

# Verify outputs
ls -lh processed/*.joblib
```

**Expected Duration:** 1-2 minutes

**Outputs:**
- `processed/X_train_ready.joblib` (~5-10 MB)
- `processed/X_test_ready.joblib` (~1-2 MB)
- `processed/y_train_ready.joblib` (~100 KB)
- `processed/y_test_ready.joblib` (~25 KB)
- `processed/feature_names.joblib` (~5 KB)
- `processed/preprocessor.joblib` (~50 KB)

---

### Stage 2: HPC Hyperparameter Optimization (Primary Training)

```bash
# Submit Slurm job
sbatch run_optuna_job.sh

# Monitor job
squeue -u $USER

# Watch live output
tail -f outputs/logs/slurm_JOBID.out
```

**Expected Duration:** 2-4 hours (depends on N_TRIALS, GPU speed)

**Outputs:**
- `outputs/models/best_model/best_model.joblib` (calibrated model)
- `outputs/models/best_model/metadata.json` (hyperparameters + metrics)
- `outputs/plots/*.png` (ROC curves, PR curves, calibration plots)
- `outputs/logs/run_TIMESTAMP.log` (detailed execution log)

---

### Stage 3: Local Validation and Interpretability

```bash
# Launch Jupyter notebook
jupyter notebook cardiovascular_models.ipynb

# Or use VS Code
code cardiovascular_models.ipynb
```

**Execute cells sequentially:**

1. **Load Data** → Verify shapes and distributions
2. **Build Models** → Train baseline and tuned models
3. **Cross-Validation** → Compare PR-AUC across models
4. **Calibration** → Apply isotonic calibration
5. **Threshold Optimization** → Find optimal decision threshold
6. **Evaluation** → Generate metrics, plots, confusion matrices

**Expected Duration:** 10-30 minutes (depends on models trained)

**Outputs:**
- Model comparison dataframe
- Performance visualizations
- Model evaluation metrics

---

## 9. Advanced Topics

### 9.1 Parallel Optuna Trials

For faster hyperparameter search, enable parallel trials:

**Modify `cardiovascular_optuna_gpu.py`:**
```python
# Single GPU (sequential trials)
study.optimize(objective_xgb, n_trials=100, n_jobs=1)

# Multiple CPUs (parallel trials, CPU-based models only)
study.optimize(objective_lr, n_trials=100, n_jobs=4)
```

**Note:** GPU models (XGBoost, CatBoost) typically run sequentially on a single GPU. For multi-GPU parallelization, see Optuna's distributed optimization documentation.

### 9.2 Custom Metrics

To optimize for a different metric, modify the objective function:

```python
def objective_xgb(trial):
    # ... parameter sampling ...
    
    # Replace PR-AUC with custom metric
    from sklearn.metrics import f1_score
    
    f1_scores = []
    for train_idx, val_idx in cv_strategy.split(X, y):
        # ... training ...
        y_pred = model.predict(X_val_fold)
        f1_scores.append(f1_score(y_val_fold, y_pred))
    
    return float(np.mean(f1_scores))  # Maximize F1 score
```

### 9.3 Model Stacking/Ensembling

Combine multiple models for improved performance:

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base models (wrapped)
xgb = SklearnCompatibleWrapper(XGBClassifier(...))
cat = SklearnCompatibleWrapper(CatBoostClassifier(...))
lr = LogisticRegression(...)

# Stacking ensemble
stacked = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lr', lr)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacked.fit(X_train, y_train)
```

---

## 10. Performance Optimization Tips

### 10.1 Speed Optimization

**Preprocessing:**
- Use `n_jobs=-1` in `ColumnTransformer` for parallel transformation
- Enable `sparse_output=True` for one-hot encoding if data is high-dimensional

**Training:**
- Use GPU acceleration for XGBoost/CatBoost (30-100x speedup)
- Enable early stopping to avoid unnecessary epochs
- Reduce `n_trials` during development (e.g., 20 trials for prototyping)

**Cross-Validation:**
- Reduce `CV_FOLDS` from 5 to 3 for faster iteration
- Use `n_jobs=-1` in cross-validation for parallel fold processing

### 10.2 Memory Optimization

**Data Loading:**
- Use `compress=3` for joblib files (already configured)
- Load only necessary columns during preprocessing
- Use sparse matrices for high-dimensional one-hot encoded data

**Model Training:**
- Reduce `max_bin` in XGBoost (e.g., 128 instead of 256)
- Use `hist` tree method instead of `exact` (memory-efficient)
- Clear old trial models in Optuna: `study.trials_dataframe().drop_duplicates()`

### 10.3 Disk Space Optimization

**Compression Levels:**
```python
# Fast but larger files
joblib.dump(model, path, compress=0)  # No compression

# Balanced (recommended)
joblib.dump(model, path, compress=3)  # 60-70% reduction

# Maximum compression (slower)
joblib.dump(model, path, compress=9)  # 75-85% reduction
```

**Cleanup:**
```bash
# Remove intermediate plots
rm outputs/plots/*_TIMESTAMP.png

# Remove old Slurm logs
find outputs/logs -name "slurm_*.out" -mtime +30 -delete
```

---

## 11. Validation and Testing

### 11.1 Unit Testing Utilities

Verify `cardio_utils.py` functions work correctly:

```python
# Test SklearnCompatibleWrapper
from cardio_utils import SklearnCompatibleWrapper
from xgboost import XGBClassifier
from sklearn.base import clone

xgb = XGBClassifier(n_estimators=10)
wrapped = SklearnCompatibleWrapper(xgb)

# Test cloning
cloned = clone(wrapped)
assert cloned is not wrapped
assert cloned.model is not wrapped.model

# Test sklearn tags
tags = wrapped.__sklearn_tags__()
assert tags is not None

print("✓ SklearnCompatibleWrapper tests passed")
```

### 11.2 Data Integrity Checks

Verify preprocessing outputs:

```python
import joblib
import numpy as np

# Load artifacts
X_train = joblib.load("processed/X_train_ready.joblib")
y_train = joblib.load("processed/y_train_ready.joblib")

# Check for NaN values
assert not np.isnan(X_train).any(), "NaN values found in X_train"
assert not np.isnan(y_train).any(), "NaN values found in y_train"

# Check label distribution
unique, counts = np.unique(y_train, return_counts=True)
assert len(unique) == 2, f"Expected 2 classes, got {len(unique)}"
assert all(unique == [0, 1]), f"Expected labels [0, 1], got {unique}"

print(f"✓ Data integrity verified")
print(f"  Train samples: {len(y_train)}")
print(f"  Class distribution: {dict(zip(unique, counts))}")
```

### 11.3 Model Sanity Checks

Verify models are performing better than baseline:

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Dummy classifier (predicts majority class)
dummy_pred = np.ones(len(y_test)) * y_train.mean()
dummy_roc_auc = roc_auc_score(y_test, dummy_pred)
dummy_pr_auc = average_precision_score(y_test, dummy_pred)

# Your model
y_proba = model.predict_proba(X_test)[:, 1]
model_roc_auc = roc_auc_score(y_test, y_proba)
model_pr_auc = average_precision_score(y_test, y_proba)

# Sanity check
assert model_roc_auc > dummy_roc_auc, "Model worse than dummy classifier!"
assert model_pr_auc > dummy_pr_auc, "Model worse than dummy classifier!"

print(f"✓ Model sanity checks passed")
print(f"  Dummy ROC-AUC: {dummy_roc_auc:.4f}, Model ROC-AUC: {model_roc_auc:.4f}")
print(f"  Dummy PR-AUC:  {dummy_pr_auc:.4f}, Model PR-AUC:  {model_pr_auc:.4f}")
```

---

## 12. Deployment Considerations

### 12.1 Model Serialization

Save the calibrated model and preprocessor for production:

```python
import joblib

# Save model
joblib.dump(calibrated_model, "best_model.joblib", compress=3)

# Save preprocessor
joblib.dump(preprocessor, "preprocessor.joblib", compress=3)

# Save metadata
metadata = {
    "model_type": "CalibratedXGBClassifier",
    "train_date": "2026-01-28",
    "performance": {
        "roc_auc": 0.8543,
        "pr_auc": 0.6721,
        "threshold": 0.3214
    },
    "hyperparameters": {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05
    }
}

import json
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### 12.2 Inference Pipeline

Create a prediction function for new data:

```python
def predict_cardiovascular_risk(patient_data):
    """
    Predict cardiovascular disease risk for new patient data.
    
    Args:
        patient_data: DataFrame with raw patient features
        
    Returns:
        dict: Prediction probability and risk category
    """
    import joblib
    
    # Load artifacts
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("best_model.joblib")
    
    # Preprocess
    X = preprocessor.transform(patient_data)
    
    # Predict
    proba = model.predict_proba(X)[:, 1][0]
    
    # Apply clinical threshold
    threshold = 0.3214  # From metadata.json
    prediction = "High Risk" if proba >= threshold else "Low Risk"
    
    return {
        "probability": float(proba),
        "risk_category": prediction,
        "confidence": abs(proba - threshold)
    }
```

### 12.3 Model Monitoring

Track model performance over time:

```python
# Log predictions for monitoring
import pandas as pd
from datetime import datetime

def log_prediction(patient_id, prediction, actual=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "predicted_proba": prediction["probability"],
        "predicted_class": prediction["risk_category"],
        "actual_class": actual  # Fill in after diagnosis
    }
    
    # Append to monitoring log
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv("predictions_log.csv", mode="a", header=False, index=False)
```

---

## 13. References and Resources

### 13.1 Key Documentation

- **Scikit-learn**: https://scikit-learn.org/stable/
- **XGBoost**: https://xgboost.readthedocs.io/
- **CatBoost**: https://catboost.ai/docs/
- **Optuna**: https://optuna.readthedocs.io/

### 13.2 Pipeline Architecture

```
HeartDisease.csv
      ↓
cardiovascular_preprocessing.py
      ↓
processed/*.joblib artifacts
      ↓
      ├─→ cardiovascular_optuna_gpu.py (HPC) → best_model.joblib
      └─→ cardiovascular_models.ipynb (Local) → Validation
```

### 13.3 Key Design Decisions

1. **PR-AUC over ROC-AUC**: With 8% positive class prevalence, PR-AUC is more informative
2. **Isotonic Calibration**: Better probability estimates for clinical decision-making
3. **SklearnCompatibleWrapper**: Ensures consistent behavior across sklearn ecosystem
4. **Centralized Config**: Single source of truth for all hyperparameters
5. **Compressed Joblib**: Balances disk space and I/O speed

---

## 14. Support and Troubleshooting

If you encounter issues not covered in this guide:

1. **Check Logs**: Review `outputs/logs/run_TIMESTAMP.log` for detailed error messages
2. **Verify Environment**: Ensure Python version, dependencies, and PYTHONPATH are correct
3. **Validate Data**: Use the data integrity checks in Section 11.2
4. **Test Utilities**: Run unit tests for `cardio_utils.py` as shown in Section 11.1
5. **Consult Documentation**: Refer to library-specific docs for advanced troubleshooting
