"""
Shared utilities for cardiovascular risk prediction pipeline.

This module provides a single source of truth for reusable components
across the three-stage pipeline:
  1. Preprocessing (cardiovascular_preprocessing.py)
  2. HPC Modeling (cardiovascular_optuna_gpu.py)
  3. Interpretability

Components:
  - SklearnCompatibleWrapper: Ensures XGBoost/CatBoost work with
    sklearn's CalibratedClassifierCV and SHAP TreeExplainer
  - _subset_rows(): Helper for subsetting DataFrames and arrays
  - cross_validate_model(): Unified CV function for all models
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, average_precision_score
from sklearn.base import clone


class SklearnCompatibleWrapper(ClassifierMixin, BaseEstimator):
    """
    Wrapper to make XGBoost/CatBoost fully sklearn-compatible.
    
    This wrapper is ESSENTIAL for:
    - CalibratedClassifierCV compatibility via __sklearn_tags__
    - SHAP TreeExplainer access via __getattr__ delegation
    - Proper cloning via BaseEstimator inheritance
    - sklearn's calibration pipeline (isotonic regression)
    
    Why This Wrapper?
    -----------------
    XGBoost and CatBoost are not true sklearn estimators. They don't implement
    the full sklearn estimator protocol, which causes issues when:
    - Using CalibratedClassifierCV (needs __sklearn_tags__ for version >= 1.1)
    - Cloning models with sklearn.base.clone()
    - Using xAI methods like SHAP TreeExplainer (needs feature_importances_ delegation)
    
    This wrapper bridges that gap by:
    1. Inheriting from ClassifierMixin and BaseEstimator
    2. Implementing __sklearn_tags__() for newer sklearn versions
    3. Delegating attribute access via __getattr__() to the underlying model
    
    Example:
    --------
    >>> from xgboost import XGBClassifier
    >>> xgb = XGBClassifier(n_estimators=100)
    >>> wrapped_xgb = SklearnCompatibleWrapper(xgb)
    >>> wrapped_xgb.fit(X_train, y_train)
    >>> calibrated = CalibratedClassifierCV(wrapped_xgb)
    >>> calibrated.fit(X_train, y_train)
    """
    
    def __init__(self, model):
        """Initialize wrapper with a model instance."""
        self.model = model
        self._estimator_type = "classifier"

    @property
    def classes_(self):
        """Extract classes from model or provide default binary classes."""
        return getattr(self.model, "classes_", np.array([0, 1]))

    def fit(self, X, y, **kwargs):
        """Fit the underlying model and ensure classes_ are set."""
        self.model.fit(X, y, **kwargs)
        # Ensure classes_ are available for sklearn compatibility
        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Delegate prediction to underlying model."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Delegate probability prediction to underlying model."""
        return self.model.predict_proba(X)

    def __sklearn_tags__(self):
        """
        Set sklearn tags for compatibility with CalibratedClassifierCV.
        
        This method is called by sklearn's internal validation functions.
        For sklearn >= 1.3, it must return a Tags object with proper classifier tags.
        Handles multiple sklearn versions gracefully.
        """
        try:
            # Try the standard approach (sklearn >= 1.3)
            tags = super().__sklearn_tags__()
            if hasattr(tags, 'classifier_tags'):
                try:
                    tags.classifier_tags.target_is_not_sparse = True
                except AttributeError:
                    try:
                        # Fallback for sklearn 1.3/1.4 versions
                        tags.classifier_tags.target_is_sparse = False
                    except AttributeError:
                        pass
            return tags
        except (AttributeError, TypeError):
            # Fallback for older sklearn versions (dict-based tags)
            return {"estimator_type": "classifier"}

    def __getattr__(self, name):
        """
        Delegate attribute access to underlying model.
        
        This allows access to:
        - feature_importances_ (for tree-based feature importance)
        - booster (for XGBoost's internal tree structure)
        - get_booster() (for SHAP TreeExplainer)
        - Any other model-specific attributes
        """
        return getattr(self.model, name)


def _subset_rows(data, indices):
    """
    Helper function to subset both pandas DataFrames and numpy arrays.
    
    This function is used during cross-validation to extract fold subsets.
    It handles both data types transparently.
    
    Args:
        data: Either a pandas DataFrame or numpy array
        indices: Array of integer indices to extract
        
    Returns:
        Subset of data with the specified indices (DataFrame or array)
        
    Example:
    --------
    >>> X_train_fold = _subset_rows(X_train, train_indices)
    >>> y_val_fold = _subset_rows(y_val, val_indices)
    """
    if hasattr(data, 'iloc'):
        # pandas DataFrame — use iloc for integer-location based indexing
        return data.iloc[indices]
    # numpy array — use standard indexing
    return data[indices]


def cross_validate_model(model, X, y, cv_strategy):
    """
    Unified cross-validation function for all models.
    
    This function is used in both:
    1. cardiovascular_optuna_gpu.py (HPC tuning with 5-fold CV)
    2. cardiovascular_modelsREAL.ipynb (validation suite)
    
    It evaluates model performance using:
    - Log Loss: For probability calibration quality
    - PR-AUC (Precision-Recall AUC): Primary metric for imbalanced data
    
    Why PR-AUC for imbalanced data?
    -------------------------------
    With only ~8% positive cases (disease), traditional accuracy and ROC-AUC
    are misleading. A trivial classifier predicting "no disease" for everyone
    achieves 92% accuracy. PR-AUC focuses specifically on the minority class
    (disease cases) and is the recommended metric for imbalanced binary classification.
    
    Args:
        model: Unfitted sklearn-compatible model
        X: Feature matrix (pandas DataFrame or numpy array)
        y: Target vector (pandas Series or numpy array)
        cv_strategy: StratifiedKFold instance for fold generation
        
    Returns:
        tuple: (mean_log_loss, mean_pr_auc)
        
    Example:
    --------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn.linear_model import LogisticRegression
    >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> lr = LogisticRegression()
    >>> logloss, pr_auc = cross_validate_model(lr, X_train, y_train, cv)
    >>> print(f"Mean Log Loss: {logloss:.4f}, Mean PR-AUC: {pr_auc:.4f}")
    """
    log_losses = []
    pr_aucs = []
    
    # Iterate through stratified folds
    for train_idx, val_idx in cv_strategy.split(X, y):
        # Extract fold data using universal subset function
        X_train_fold = _subset_rows(X, train_idx)
        y_train_fold = _subset_rows(y, train_idx)
        X_val_fold = _subset_rows(X, val_idx)
        y_val_fold = _subset_rows(y, val_idx)
        
        # Clone the model to ensure fresh parameters for each fold
        model_clone = clone(model)
        
        # Train on fold training set with early stopping support
        fit_kwargs = {}
        
        # Detect XGBoost/CatBoost models (via wrapper or direct)
        underlying_model = getattr(model_clone, 'model', model_clone)
        model_class_name = type(underlying_model).__name__
        
        # For tree-based models with early stopping, provide validation set
        if model_class_name in ('XGBClassifier', 'CatBoostClassifier'):
            fit_kwargs['eval_set'] = [(X_val_fold, y_val_fold)]
            if model_class_name == 'XGBClassifier':
                fit_kwargs['verbose'] = False
        
        model_clone.fit(X_train_fold, y_train_fold, **fit_kwargs)
        
        # Predict probabilities on fold validation set
        y_proba = model_clone.predict_proba(X_val_fold)[:, 1]
        
        # Compute metrics
        log_losses.append(log_loss(y_val_fold, y_proba))
        pr_aucs.append(average_precision_score(y_val_fold, y_proba))
    
    # Return mean metrics across folds
    return float(np.mean(log_losses)), float(np.mean(pr_aucs))
