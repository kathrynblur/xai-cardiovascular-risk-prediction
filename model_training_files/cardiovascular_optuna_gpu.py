import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Import shared utilities from centralized modules
from cardio_utils import SklearnCompatibleWrapper, _subset_rows, cross_validate_model
from config import (
    RANDOM_STATE,
    N_TRIALS,
    CV_FOLDS,
    COMPRESSION_LEVEL,
    BASE_DIR_HPC,
    HPC_ENABLED,
    MIN_RECALL_THRESHOLD,
)

def is_hpc() -> bool:
    """Check if running on HPC cluster via Slurm."""
    return HPC_ENABLED or "SLURM_JOB_ID" in os.environ


def get_base_dir() -> Path:
    """Get base directory based on environment."""
    if is_hpc():
        return BASE_DIR_HPC
    return Path.cwd()


def ensure_output_dirs(base_dir: Path) -> dict:
    outputs = {
        "models": base_dir / "outputs" / "models",
        "plots": base_dir / "outputs" / "plots",
        "logs": base_dir / "outputs" / "logs",
    }
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def configure_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("cardio_optuna")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized")
    logger.info("Log file: %s", log_file)
    return logger



def save_plot(fig, output_dir: Path, filename: str, logger: logging.Logger) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{Path(filename).stem}_{timestamp}.png"
    out_path = output_dir / safe_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)
    return out_path


def optimize_clinical_threshold(y_true, y_proba, min_recall=None):
    """Find threshold maximizing precision while maintaining minimum recall."""
    if min_recall is None:
        min_recall = MIN_RECALL_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]
    feasible = recall >= min_recall
    if not np.any(feasible):
        best_idx = int(np.argmax(recall))
    else:
        feasible_indices = np.where(feasible)[0]
        best_idx = feasible_indices[int(np.argmax(precision[feasible]))]
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def main():
    base_dir = get_base_dir()
    outputs = ensure_output_dirs(base_dir)
    logger = configure_logging(outputs["logs"])

    logger.info("Base directory: %s", base_dir)
    logger.info("Running on HPC: %s", is_hpc())

    try:
        np.random.seed(RANDOM_STATE)
        sns.set_palette("husl")

        data_path = base_dir / "processed"
        required_files = [
            data_path / "X_train_ready.joblib",
            data_path / "X_test_ready.joblib",
            data_path / "y_train_ready.joblib",
            data_path / "y_test_ready.joblib",
            data_path / "feature_names.joblib",
        ]
        if not all(path.exists() for path in required_files):
            raise FileNotFoundError("Missing model-ready artifacts in processed/ directory")

        X_train = joblib.load(data_path / "X_train_ready.joblib")
        X_test = joblib.load(data_path / "X_test_ready.joblib")
        y_train = joblib.load(data_path / "y_train_ready.joblib")
        y_test = joblib.load(data_path / "y_test_ready.joblib")
        feature_names = joblib.load(data_path / "feature_names.joblib")

        class_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

        logger.info("Loaded model-ready artifacts from %s", data_path)
        logger.info("X_train shape: %s", getattr(X_train, "shape", None))
        logger.info("X_test shape: %s", getattr(X_test, "shape", None))

        def build_lr_model(params):
            # Create a copy to avoid mutating the input
            model_params = params.copy()
            
            # If l1_ratio is present, we're using elasticnet with saga solver
            if 'l1_ratio' in model_params:
                model_params['penalty'] = 'elasticnet'
                model_params['solver'] = 'saga'
            else:
                # Default to l2 with lbfgs if no l1_ratio specified
                model_params.setdefault('penalty', 'l2')
                model_params.setdefault('solver', 'lbfgs')
            
            return LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
                **model_params,
            )

        def build_xgb_model(params):
            xgb = XGBClassifier(
                scale_pos_weight=class_ratio,
                random_state=RANDOM_STATE,
                tree_method="hist",
                device="cuda",
                early_stopping_rounds=50,
                **params,
            )
            return SklearnCompatibleWrapper(xgb) # Wrap XGBClassifier

        def build_cat_model(params):
            base_params = {
                "auto_class_weights": "Balanced",
                "random_state": RANDOM_STATE,
                "verbose": 0,
                "task_type": "GPU",
                "devices": "0",
                "early_stopping_rounds": 50,
            }
            base_params.update(params)
            cat = CatBoostClassifier(**base_params)
            return SklearnCompatibleWrapper(cat)  # Wrap CatBoostClassifier

        # Use imported constants from config.py
        cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        def objective_lr(trial):
            # Suggest hyperparameters (solver/penalty handled by build_lr_model)
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
            model = build_lr_model(params)
            logloss, pr_auc = cross_validate_model(
                model, X_train, y_train, cv_strategy
            )
            trial.set_user_attr("logloss", logloss)
            return pr_auc

        def objective_xgb(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 700),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            model = build_xgb_model(params)
            logloss, pr_auc = cross_validate_model(
                model, X_train, y_train, cv_strategy
            )
            trial.set_user_attr("logloss", logloss)
            return pr_auc

        def objective_cat(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 700),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }
            model = build_cat_model(params)
            logloss, pr_auc = cross_validate_model(
                model, X_train, y_train, cv_strategy
            )
            trial.set_user_attr("logloss", logloss)
            return pr_auc

        # Enable parallel trials with shared storage
        sampler = TPESampler(seed=RANDOM_STATE)
        storage = InMemoryStorage()  # Shared storage for parallel execution
        
        study_lr = optuna.create_study(direction="maximize", sampler=sampler, storage=storage)
        study_xgb = optuna.create_study(direction="maximize", sampler=sampler, storage=storage)
        study_cat = optuna.create_study(direction="maximize", sampler=sampler, storage=storage)

        logger.info("Optimizing Logistic Regression (sequential trials)...")
        study_lr.optimize(objective_lr, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
        logger.info("Optimizing XGBoost (sequential trials)...")
        study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
        logger.info("Optimizing CatBoost (sequential trials)...")
        study_cat.optimize(objective_cat, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

        logger.info("Best PR-AUC (LR): %.4f | Log Loss: %.4f", study_lr.best_value, study_lr.best_trial.user_attrs.get("logloss", 0.0))
        logger.info("Best PR-AUC (XGB): %.4f | Log Loss: %.4f", study_xgb.best_value, study_xgb.best_trial.user_attrs.get("logloss", 0.0))
        logger.info("Best PR-AUC (CAT): %.4f | Log Loss: %.4f", study_cat.best_value, study_cat.best_trial.user_attrs.get("logloss", 0.0))

        best_lr = build_lr_model(study_lr.best_params)
        best_xgb = build_xgb_model(study_xgb.best_params)
        best_cat = build_cat_model(study_cat.best_params)

        tuned_models = {
            "Logistic Regression (Tuned)": best_lr,
            "XGBoost (Tuned)": best_xgb,
            "CatBoost (Tuned, OHE)": best_cat,
        }

        model_outputs = {}
        for name, model in tuned_models.items():
            logger.info("Training %s...", name)
            
            # For final fit on full training data, rebuild without early_stopping
            # (early stopping is only used during hyperparameter tuning cross-validation)
            if "Logistic Regression" in name:
                model_for_fit = build_lr_model(study_lr.best_params)
            elif "XGBoost" in name:
                # Rebuild XGBoost without early_stopping_rounds
                best_params = study_xgb.best_params.copy()
                xgb = XGBClassifier(
                    scale_pos_weight=class_ratio,
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                    device="cuda",
                    **best_params,
                )
                model_for_fit = SklearnCompatibleWrapper(xgb)
            elif "CatBoost" in name:
                # Rebuild CatBoost without early_stopping_rounds
                best_params = study_cat.best_params.copy()
                base_params = {
                    "auto_class_weights": "Balanced",
                    "random_state": RANDOM_STATE,
                    "verbose": 0,
                    "task_type": "GPU",
                    "devices": "0",
                }
                base_params.update(best_params)
                cat = CatBoostClassifier(**base_params)
                model_for_fit = SklearnCompatibleWrapper(cat)
            
            model_for_fit.fit(X_train, y_train)
            y_proba = model_for_fit.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            model_outputs[name] = {
                "model": model_for_fit,
                "y_pred_default": y_pred,
                "y_proba": y_proba,
            }

        results = []
        for name, m_outp in model_outputs.items():
            y_proba = m_outp["y_proba"]
            threshold, tuned_precision, tuned_recall = optimize_clinical_threshold(
                y_test, y_proba, min_recall=0.85
            )

            y_pred_tuned = (y_proba >= threshold).astype(int)
            report = classification_report(
                y_test, 
                y_pred_tuned, 
                target_names=["No Disease", "Disease"],
                digits=4
            )
            logger.info(f"\n[METRICS] Classification Report for {name} (Threshold: {threshold:.4f}):\n{report}")

            m_outp["threshold"] = threshold
            m_outp["tuned_precision"] = tuned_precision
            m_outp["tuned_recall"] = tuned_recall
            results.append(
                {
                    "Model": name,
                    "ROC-AUC": roc_auc_score(y_test, y_proba),
                    "PR-AUC": average_precision_score(y_test, y_proba),
                    "Log Loss": log_loss(y_test, y_proba),
                    "Brier Score": brier_score_loss(y_test, y_proba),
                    "Threshold": threshold,
                    "Tuned Precision": tuned_precision,
                    "Tuned Recall": tuned_recall,
                }
            )

        results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)

        best_model_name = results_df.iloc[0]["Model"]
        best_model = model_outputs[best_model_name]["model"]
        best_proba = model_outputs[best_model_name]["y_proba"]

        calibrated_model = CalibratedClassifierCV(
            estimator=clone(best_model), method="isotonic", cv=5
        )
        calibrated_model.fit(X_train, y_train)
        calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]

        cal_threshold, cal_precision, cal_recall = optimize_clinical_threshold(
            y_test, calibrated_proba, min_recall=0.85
        )
        calibrated_row = {
            "Model": f"Calibrated {best_model_name}",
            "ROC-AUC": roc_auc_score(y_test, calibrated_proba),
            "PR-AUC": average_precision_score(y_test, calibrated_proba),
            "Log Loss": log_loss(y_test, calibrated_proba),
            "Brier Score": brier_score_loss(y_test, calibrated_proba),
            "Threshold": cal_threshold,
            "Tuned Precision": cal_precision,
            "Tuned Recall": cal_recall,
        }
        results_df = pd.concat([results_df, pd.DataFrame([calibrated_row])], ignore_index=True)

        overall_winner_name = f"Calibrated {best_model_name}"
        logger.info("Overall winner (calibrated): %s", overall_winner_name)

        # Generate predictions using the optimized clinical threshold
        y_pred_final = (calibrated_proba >= cal_threshold).astype(int)
        
        # Create the report
        final_report = classification_report(
            y_test, 
            y_pred_final, 
            target_names=["No Disease", "Disease"],
            digits=4
        )
        
        # Log the report so it appears in your .log file and terminal
        logger.info(f"\n[FINAL RESULTS] Classification Report for {overall_winner_name}:")
        logger.info(f"Using Clinical Threshold: {cal_threshold:.4f}")
        logger.info(f"\n{final_report}")

        # Calibration plot
        frac_pos_uncal, mean_pred_uncal = calibration_curve(
            y_test, best_proba, n_bins=10, strategy="quantile"
        )
        frac_pos_cal, mean_pred_cal = calibration_curve(
            y_test, calibrated_proba, n_bins=10, strategy="quantile"
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(mean_pred_uncal, frac_pos_uncal, marker="o", label=f"Uncalibrated: {best_model_name}")
        ax.plot(mean_pred_cal, frac_pos_cal, marker="o", label=f"Calibrated: {best_model_name}")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(loc="upper left")
        save_plot(fig, outputs["plots"], "calibration_curve.png", logger)

        # ROC plot (calibrated winner)
        proba_for_roc = calibrated_proba
        model_label = f"Calibrated {best_model_name}"

        fpr, tpr, _ = roc_curve(y_test, proba_for_roc)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"{model_label}")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        save_plot(fig, outputs["plots"], "roc_curve.png", logger)

        # Precision-Recall plot (calibrated winner)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, calibrated_proba)
        avg_prec = average_precision_score(y_test, calibrated_proba)
        
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        ax_pr.plot(recall_vals, precision_vals, color='teal', 
                   label=f"PR Curve (AP = {avg_prec:.4f})")
        
        # Plot the operating point (your 85% recall target)
        ax_pr.plot(cal_recall, cal_precision, 'ro', 
                   label=f"Clinical Threshold ({cal_threshold:.4f})")
        
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel("Recall (Sensitivity)")
        ax_pr.set_ylabel("Precision (Positive Predictive Value)")
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.legend(loc="upper right")
        ax_pr.grid(alpha=0.3)
        
        save_plot(fig_pr, outputs["plots"], "precision_recall_curve.png", logger)

        # Confusion matrix plot (calibrated winner, clinical threshold)
        threshold = cal_threshold
        y_pred = (calibrated_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix")
        save_plot(fig, outputs["plots"], "confusion_matrix.png", logger)

        # Attach feature names for xAI (calibrated winner for final evaluation)
        final_model = calibrated_model
        final_threshold = cal_threshold

        if hasattr(X_train, "columns"):
            final_model.feature_names_ = X_train.columns.tolist()
        else:
            final_model.feature_names_ = list(feature_names)

        model_artifact = {
            "model_name": overall_winner_name,
            "model": final_model,
            "threshold": final_threshold,
            "feature_names": feature_names,
            "categorical_features": [],
            "numeric_features": list(feature_names),
        }

        model_path = outputs["models"] / "best_model.joblib"
        joblib.dump(model_artifact, model_path)

        metadata = {
            "random_state": RANDOM_STATE,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "best_hyperparameters": {
                "logistic_regression": study_lr.best_params,
                "xgboost": study_xgb.best_params,
                "catboost": study_cat.best_params,
            },
            "optimized_clinical_threshold": final_threshold,
            "overall_winner": overall_winner_name,
            "run_timestamp": datetime.now().isoformat(),
        }

        metadata_path = outputs["models"] / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved model artifact: %s", model_path)
        logger.info("Saved metadata: %s", metadata_path)

    except Exception:
        logger.exception("Fatal error in pipeline execution")
        raise


if __name__ == "__main__":
    main()