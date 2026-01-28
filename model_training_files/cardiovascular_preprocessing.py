# =============================================================================
# IMPORTS: Core libraries for preprocessing
# =============================================================================
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Scikit-learn preprocessing components
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True)
full_data_path = Path("HeartDisease.csv")

print("=" * 60)
print("PREPROCESSING NOTEBOOK INITIALIZED")
print("=" * 60)
print(f"Output directory: {PROCESSED_DIR.resolve()}")
print(f"Random state: {RANDOM_STATE}")

# =============================================================================
# STEP 1: Load raw data and ensure stratified split exists
# =============================================================================

# Define expected split files
x_train_path = PROCESSED_DIR / "X_train_raw.csv"
x_test_path = PROCESSED_DIR / "X_test_raw.csv"
y_train_path = PROCESSED_DIR / "y_train_raw.csv"
y_test_path = PROCESSED_DIR / "y_test_raw.csv"

# Global fallback path - define it here to be safe
full_data_path = Path("HeartDisease.csv")

if x_train_path.exists() and x_test_path.exists() and y_train_path.exists() and y_test_path.exists():
    # OPTION A: Load existing stratified train/test splits
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")
    print("Loaded existing stratified train/test splits from /processed")
else:
    # OPTION B: Create stratified split from the cleaned full dataset
    if not full_data_path.exists():
        raise FileNotFoundError(
            f"Could not find pre-split data OR {full_data_path} in the project root."
        )

    df = pd.read_csv(full_data_path)
    
    # Map target immediately during initial load
    if df["Heart_Disease"].dtype == "object":
        df["Heart_Disease"] = df["Heart_Disease"].map({"No": 0, "Yes": 1})

    y = df["Heart_Disease"]
    X = df.drop(columns=["Heart_Disease"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Save splits for reproducibility
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print("Created and saved new stratified train/test splits to /processed")

# --- POST-LOAD CLEANUP ---

# Force targets to numeric 0/1 regardless of how they were loaded
target_map = {"No": 0, "Yes": 1, 0: 0, 1: 1}
y_train = y_train.map(target_map).astype(int)
y_test = y_test.map(target_map).astype(int)

# Create Age_num if missing (fallback safety)
def _age_category_to_num(value: str) -> float:
    if pd.isna(value): return np.nan
    text = str(value).strip()
    if text.lower() in {"80 or older", "80+", "80+ years"}: return 80.0
    if "-" in text:
        parts = text.split("-")
        try: return (float(parts[0]) + float(parts[1])) / 2
        except: return np.nan
    return np.nan

if "Age_num" not in X_train.columns and "Age_Category" in X_train.columns:
    X_train["Age_num"] = X_train["Age_Category"].apply(_age_category_to_num)
    X_test["Age_num"] = X_test["Age_Category"].apply(_age_category_to_num)
    print("Created Age_num from Age_Category")

# Display shapes for verification
print("=" * 60)
print("DATA LOADED AND VERIFIED")
print("=" * 60)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape} (dtype: {y_train.dtype})")
print(f"Label distribution: {dict(y_train.value_counts())}")

# =============================================================================
# STEP 2: Verify class imbalance in train/test splits
# =============================================================================

def display_class_distribution(y, set_name):
    total = len(y)
    # Count how many times "Yes" appears
    positive_count = (y == 1).sum() 
    negative_count = total - positive_count
    
    # Calculate percentage
    pos_pct = (positive_count / total) * 100
    print(f"{set_name}: {positive_count} (Heart Disease), {negative_count} (Healthy) [{pos_pct:.2f}% Positive]")
    return pos_pct

print("=" * 60)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 60)

train_pct = display_class_distribution(y_train, "Training Set")
test_pct = display_class_distribution(y_test, "Test Set")

# Verify stratification worked
print(f"\nStratification check: Train={train_pct:.2f}%, Test={test_pct:.2f}%")
if abs(train_pct - test_pct) < 1.0:
    print("   → Class ratios are consistent (good stratification)")

# =============================================================================
# STEP 3A: Identify binary columns automatically
# =============================================================================

def find_binary_columns(df):
    """Identify columns with exactly 2 unique values (excluding NaN)."""
    binary_cols = []
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) == 2:
            binary_cols.append(col)
    return binary_cols

detected_binary = find_binary_columns(X_train)
print("Detected binary columns:")
for col in detected_binary:
    unique_vals = X_train[col].dropna().unique().tolist()
    print(f"  • {col}: {unique_vals}")

# =============================================================================
# STEP 3B: Define feature groups explicitly
# =============================================================================

# Categorical features: Require one-hot encoding for XGBoost/LogReg
# CatBoost will use these directly via cat_features parameter
CATEGORICAL_COLS = [
    "General_Health",    # Ordinal health rating (Excellent → Poor)
    "Checkup",           # Time since last medical checkup
    "Diabetes",          # Diabetes status (Yes, No, Pre-diabetes, etc.)
]

# Numeric features: Require scaling for gradient-based models
NUMERIC_COLS = [
    "Height_(cm)",                  # Patient height
    "Weight_(kg)",                  # Patient weight
    "BMI",                          # Body Mass Index (derived)
    "Alcohol_Consumption",          # Alcohol intake score
    "Fruit_Consumption",            # Fruit intake score
    "Green_Vegetables_Consumption", # Vegetable intake score
    "FriedPotato_Consumption",      # Fried food intake score
    "Age_num",                      # Numeric age (from Age_Category)
]

# Binary features: Will be converted to 0/1 before preprocessing
BINARY_COLS = [
    "Exercise",          # Regular exercise (Yes=1, No=0)
    "Skin_Cancer",       # History of skin cancer
    "Other_Cancer",      # History of other cancer
    "Depression",        # Depression diagnosis
    "Arthritis",         # Arthritis diagnosis
    "Sex",               # Male=1, Female=0
    "Smoking_History",   # Ever smoked
]

# Columns to drop (redundant with Age_num)
DROP_COLS = ["Age_Category"]

# Validate that required columns exist
missing_categorical = [col for col in CATEGORICAL_COLS if col not in X_train.columns]
missing_numeric = [col for col in NUMERIC_COLS if col not in X_train.columns]
missing_binary = [col for col in BINARY_COLS if col not in X_train.columns]

if missing_categorical or missing_numeric or missing_binary:
    missing_report = {
        "categorical": missing_categorical,
        "numeric": missing_numeric,
        "binary": missing_binary,
    }
    raise ValueError(
        "Missing expected columns in X_train. "
        f"Please check preprocessing inputs. Details: {missing_report}"
    )

# Display summary
print("=" * 60)
print("FEATURE GROUP DEFINITIONS")
print("=" * 60)
print(f"Categorical columns: {len(CATEGORICAL_COLS)}")
print(f"Numeric columns:     {len(NUMERIC_COLS)}")
print(f"Binary columns:      {len(BINARY_COLS)}")
print(f"Columns to drop:     {DROP_COLS}")

# =============================================================================
# STEP 3C: Drop redundant columns + normalize binary fields
# =============================================================================

# Check if columns exist before dropping (defensive coding)
cols_to_drop = [col for col in DROP_COLS if col in X_train.columns]

if cols_to_drop:
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
else:
    print("No columns to drop (already removed)")

# Normalize binary fields to 0/1 for model-ready arrays
_yes_no_map = {"Yes": 1, "No": 0}
_sex_map = {"Male": 1, "Female": 0}

for col in BINARY_COLS:
    if col not in X_train.columns:
        continue

    if X_train[col].dtype == "object":
        if set(X_train[col].dropna().unique()).issubset({"Yes", "No"}):
            X_train[col] = X_train[col].map(_yes_no_map)
            X_test[col] = X_test[col].map(_yes_no_map)
        elif set(X_train[col].dropna().unique()).issubset({"Male", "Female"}):
            X_train[col] = X_train[col].map(_sex_map)
            X_test[col] = X_test[col].map(_sex_map)

    # Ensure numeric dtype for binary columns
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

print(f"Remaining features: {X_train.shape[1]}")
print(f"Feature names: {X_train.columns.tolist()}")

# Verify training set structure
X_train.head()

# Verify test set structure matches training set
X_test.head()

# =============================================================================
# STEP 4: Build the preprocessing pipeline
# =============================================================================

# --- Numeric feature pipeline ---
# 1. Fill missing values with the median (robust to outliers)
# 2. Standardize to zero mean and unit variance
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# --- Categorical feature pipeline ---
# 1. Fill missing values with the most frequent category
# 2. One-hot encode (creates dummy columns for each category)
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# --- Binary feature pipeline (Solves the NaN issue) ---
# We use 'most_frequent' to fill missing binary values (0 or 1)
binary_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# --- Combine into a single ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("bin", binary_pipeline, BINARY_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=True
)

print("=" * 60)
print("PREPROCESSING PIPELINE CREATED")
print("=" * 60)
print("Pipeline structure:")
print("  Numeric:     Imputer(median) → StandardScaler")
print("  Categorical: Imputer(mode) → OneHotEncoder")
print("  Binary:      Imputer(mode)")
print()
print("Pipeline is NOT yet fitted. Fitting happens in the next step.")

# =============================================================================
# STEP 5: Fit on training data ONLY, then transform both sets
# =============================================================================

print("Fitting preprocessor on training data...")

# FIT on training data (learns parameters: means, stds, categories)
# TRANSFORM training data
X_train_pre = preprocessor.fit_transform(X_train)

# TRANSFORM test data (using parameters learned from training)
X_test_pre = preprocessor.transform(X_test)

# Extract feature names after transformation
feature_names = list(preprocessor.get_feature_names_out())

print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"X_train_pre shape: {X_train_pre.shape}")
print(f"X_test_pre shape:  {X_test_pre.shape}")
print(f"Number of features (after OHE): {len(feature_names)}")
print(f"Output type: {type(X_train_pre).__name__}")

# Display all feature names (important for SHAP/LIME interpretability)
print("Feature names after preprocessing:")
for i, name in enumerate(feature_names):
    print(f"  [{i:2d}] {name}")

# =============================================================================
# CONVERT TARGET TO NUMERIC (Do this before Sanity Checks!)
# =============================================================================
if y_train.dtype == 'object':
    y_train = y_train.map({'No': 0, 'Yes': 1}).astype(int)
    y_test = y_test.map({'No': 0, 'Yes': 1}).astype(int)    

# =============================================================================
# STEP 6: Sanity checks before export
# =============================================================================

print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# Check 1: No NaN values remain
nan_train = np.isnan(X_train_pre).sum()
nan_test = np.isnan(X_test_pre).sum()
print(f"\n✓ NaN values in X_train_pre: {nan_train}")
print(f"✓ NaN values in X_test_pre:  {nan_test}")

if nan_train == 0 and nan_test == 0:
    print("  → All missing values successfully imputed!")
else:
    print("WARNING: NaN values detected after preprocessing!")

# Check 2: Shape consistency
print(f"\n✓ Feature count matches: {X_train_pre.shape[1] == X_test_pre.shape[1]}")
print(f"  Train features: {X_train_pre.shape[1]}")
print(f"  Test features:  {X_test_pre.shape[1]}")

# Check 3: Target balance preserved
train_positive_rate = y_train.mean() * 100
test_positive_rate = y_test.mean() * 100
print(f"\n✓ Train positive rate: {train_positive_rate:.2f}%")
print(f"✓ Test positive rate:  {test_positive_rate:.2f}%")
print(f"  → Stratification preserved: {abs(train_positive_rate - test_positive_rate) < 1.0}")

import joblib
from pathlib import Path
import os

# =============================================================================
# STEP 7: EXPORT WITH COMPRESSION
# =============================================================================
# Compression level 3: optimal balance between file size and speed
# Prevents duplication of uncompressed files in the repo
# =============================================================================

PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# Define compression parameter
COMPRESSION = 3

# Save all model-ready data with compression
joblib.dump(X_train_pre, PROCESSED_DIR / 'X_train_ready.joblib', compress=COMPRESSION)
joblib.dump(y_train,     PROCESSED_DIR / 'y_train_ready.joblib', compress=COMPRESSION)
joblib.dump(X_test_pre,  PROCESSED_DIR / 'X_test_ready.joblib',  compress=COMPRESSION)
joblib.dump(y_test,      PROCESSED_DIR / 'y_test_ready.joblib',  compress=COMPRESSION)
joblib.dump(feature_names, PROCESSED_DIR / 'feature_names.joblib', compress=COMPRESSION)
joblib.dump(preprocessor, PROCESSED_DIR / 'preprocessor.joblib', compress=COMPRESSION)

print("=" * 60)
print("EXPORT COMPLETE (Compressed)")
print("=" * 60)
print(f"✓ X_train_ready.joblib (compressed)")
print(f"✓ X_test_ready.joblib  (compressed)")
print(f"✓ y_train_ready.joblib (compressed)")
print(f"✓ y_test_ready.joblib  (compressed)")
print(f"✓ feature_names.joblib (compressed)")
print(f"✓ preprocessor.joblib  (compressed)")

# Cleanup: Delete intermediate raw CSV files
raw_files = [x_train_path, x_test_path, y_train_path, y_test_path]
for f in raw_files:
    if f.exists():
        os.remove(f)
        print(f"  Removed intermediate: {f.name}")

print("=" * 60)