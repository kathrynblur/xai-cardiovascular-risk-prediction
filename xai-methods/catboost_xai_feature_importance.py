# ============================================================
# xAI METHOD: FEATURE IMPORTANCE for CatBoost (Global Explanation)
# ============================================================
# This script explains a trained CatBoost classifier using:
# 1) CatBoost built-in feature importance (PredictionValuesChange)
# 2) Permutation feature importance
#
# Why this is xAI:
# - Feature importance is a GLOBAL explanation: it shows which variables
#   the model relies on most across all test samples.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# -----------------------------
# ASSUMPTION:
# You already have these objects defined in your environment:
# - model         -> trained CatBoostClassifier
# - X_test_ready  -> pandas DataFrame of test features
# - y_test_ready  -> true labels for test data
# -----------------------------

estimator = model
X_test = X_test_ready
y_test = y_test_ready

print("Model:", type(estimator))
print("X_test shape:", X_test.shape, "| y_test shape:", y_test.shape)

# -----------------------------
# Quick performance context
# -----------------------------
# Report ROC-AUC (good for binary classification, especially imbalanced data).
# Feature importance is meaningful only if the model performs reasonably.

y_proba = estimator.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC:", round(auc, 4))

print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ============================================================
# PART A: CatBoost Built-in Feature Importance
# ============================================================

cb_importances = estimator.get_feature_importance(type="PredictionValuesChange")

cb_imp_df = pd.DataFrame({
    "feature": X_test.columns,
    "catboost_importance": cb_importances
}).sort_values("catboost_importance", ascending=False)

print("\nTop 15 CatBoost built-in important features:")
print(cb_imp_df.head(15).to_string(index=False))

# Plot Top N
top_n = 15
plot_df = cb_imp_df.head(top_n).iloc[::-1]

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["catboost_importance"])
plt.title(f"CatBoost Built-in Feature Importance (Top {top_n})")
plt.xlabel("Importance (PredictionValuesChange)")
plt.tight_layout()
plt.show()

# ============================================================
# PART B: Permutation Feature Importance
# ============================================================

perm = permutation_importance(
    estimator,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

perm_imp_df = pd.DataFrame({
    "feature": X_test.columns,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std
}).sort_values("perm_importance_mean", ascending=False)

print("\nTop 15 Permutation important features:")
print(perm_imp_df.head(15).to_string(index=False))

# Plot Top N with uncertainty (std)
plot_df = perm_imp_df.head(top_n).iloc[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    plot_df["feature"],
    plot_df["perm_importance_mean"],
    xerr=plot_df["perm_importance_std"]
)
plt.title(f"Permutation Feature Importance (Top {top_n})")
plt.xlabel("Mean decrease in ROC-AUC after shuffling (± std)")
plt.tight_layout()
plt.show()

# ============================================================
# PART C: Compare both rankings
# ============================================================

compare_df = cb_imp_df.merge(perm_imp_df, on="feature", how="inner")
compare_df = compare_df.sort_values("perm_importance_mean", ascending=False)

print("\nComparison (Top 20 by permutation importance):")
print(compare_df.head(20).to_string(index=False))

# Save results as CSV files
cb_imp_df.to_csv("catboost_feature_importance.csv", index=False)
perm_imp_df.to_csv("permutation_feature_importance.csv", index=False)
compare_df.to_csv("feature_importance_comparison.csv", index=False)

print("\nSaved CSV files:")
print("- catboost_feature_importance.csv")
print("- permutation_feature_importance.csv")
print("- feature_importance_comparison.csv")
