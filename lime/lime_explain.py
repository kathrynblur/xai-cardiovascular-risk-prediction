import argparse
import io
import os
import sys
import time

import joblib
import pickle
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


# Ensure project root is on sys.path for pickle imports (e.g., cardio_utils).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import cardio_utils  # noqa: F401
except Exception:
    pass


# Compatibility shim for models pickled from notebooks that referenced this class
# in __main__. It safely delegates to the wrapped object when present.
class _sklearnCompatNoneWrapper:
    def __init__(self, value=None):
        self.value = value

    def __call__(self, *args, **kwargs):
        if self.value is None:
            return None
        return self.value(*args, **kwargs)

    def __getattr__(self, name):
        if self.value is None:
            raise AttributeError(name)
        return getattr(self.value, name)


# Ensure pickle can resolve the class from __main__ or builtins if needed.
setattr(sys.modules.get("__main__"), "_sklearnCompatNoneWrapper", _sklearnCompatNoneWrapper)
setattr(sys, "_sklearnCompatNoneWrapper", _sklearnCompatNoneWrapper)


def _read_csv(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _age_category_to_num(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    extracted = values.str.extract(r"^(\\d{2})", expand=False)
    extracted = extracted.fillna(values.str.extract(r"^(\\d{1})", expand=False))
    return pd.to_numeric(extracted, errors="coerce")


def _coerce_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapped = df.copy()
    yes_no = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    sex_map = {"male": 1, "m": 1, "female": 0, "f": 0}
    for col in mapped.columns:
        if mapped[col].dtype == object:
            lower = mapped[col].astype(str).str.strip().str.lower()
            if lower.isin(yes_no.keys()).all():
                mapped[col] = lower.map(yes_no).astype(float)
            elif lower.isin(sex_map.keys()).all():
                mapped[col] = lower.map(sex_map).astype(float)
    return mapped


def _align_to_preprocessor(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    if not hasattr(preprocessor, "feature_names_in_"):
        return df

    expected = list(preprocessor.feature_names_in_)
    aligned = df.copy()

    if "Age_num" in expected and "Age_num" not in aligned.columns:
        if "Age_Category" in aligned.columns:
            aligned["Age_num"] = _age_category_to_num(aligned["Age_Category"])
        else:
            aligned["Age_num"] = np.nan

    missing = [col for col in expected if col not in aligned.columns]
    for col in missing:
        aligned[col] = np.nan

    aligned = aligned[expected]
    aligned = _coerce_binary_columns(aligned)
    return aligned


def _load_class_names(y_path: str) -> list[str]:
    if not y_path or not os.path.exists(y_path):
        return ["class_0", "class_1"]
    y = _read_csv(y_path)
    if y.shape[1] > 1:
        y = y.iloc[:, 0]
    values = pd.Series(y.squeeze()).dropna().unique().tolist()
    values = sorted(values)
    return [str(v) for v in values]


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_sklearnCompatNoneWrapper":
            return _sklearnCompatNoneWrapper
        return super().find_class(module, name)


class _CompatPickleModule:
    Unpickler = _CompatUnpickler

    @staticmethod
    def load(file_obj):
        return _CompatUnpickler(file_obj).load()

    @staticmethod
    def loads(data):
        return _CompatUnpickler(io.BytesIO(data)).load()

    dump = pickle.dump
    dumps = pickle.dumps


def _joblib_load_compat(path: str):
    try:
        return joblib.load(path)
    except AttributeError as err:
        if "_sklearnCompatNoneWrapper" not in str(err):
            raise

    try:
        return joblib.load(path, pickle_module=_CompatPickleModule)
    except TypeError:
        pass

    original_unpickler = pickle.Unpickler
    pickle.Unpickler = _CompatUnpickler
    try:
        return joblib.load(path)
    finally:
        pickle.Unpickler = original_unpickler


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain a model prediction with LIME.")
    parser.add_argument("--model-path", default="models/outputs/best_model/best_model.joblib")
    parser.add_argument("--preprocessor-path", default="processed/preprocessor.joblib")
    parser.add_argument("--feature-names-path", default="processed/feature_names.joblib")
    parser.add_argument("--train-path", default="processed/X_train_raw.csv.gz")
    parser.add_argument("--test-path", default="processed/X_test_raw.csv.gz")
    parser.add_argument("--y-train-path", default="processed/y_train_raw.csv")
    parser.add_argument("--index", type=int, default=0, help="Row index in test set to explain")
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument(
        "--output-html", default="explainability/lime/lime_explanation.html"
    )
    parser.add_argument(
        "--output-txt", default="explainability/lime/lime_explanation.txt"
    )
    parser.add_argument("--skip-preprocessor", action="store_true")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate LIME over multiple instances and save a global summary.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Limit number of instances for aggregation (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N instances during aggregation (0 to disable).",
    )
    parser.add_argument(
        "--output-agg-csv",
        default="explainability/lime/lime_global_summary.csv",
    )
    parser.add_argument(
        "--output-agg-plot",
        default="explainability/lime/lime_global_summary.png",
    )
    args = parser.parse_args()

    model_bundle = _joblib_load_compat(args.model_path)
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        bundle_feature_names = model_bundle.get("feature_names")
    else:
        model = model_bundle
        bundle_feature_names = None

    X_train = _read_csv(args.train_path)
    X_test = _read_csv(args.test_path)

    has_non_numeric = not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes
    )
    if args.skip_preprocessor:
        preprocessor = None
        if has_non_numeric and os.path.exists(args.preprocessor_path):
            preprocessor = _joblib_load_compat(args.preprocessor_path)
            print(
                "Detected non-numeric features; using preprocessor despite --skip-preprocessor."
            )
    else:
        preprocessor = _joblib_load_compat(args.preprocessor_path)

    raw_feature_names = list(X_train.columns)
    if os.path.exists(args.feature_names_path):
        try:
            _ = _joblib_load_compat(args.feature_names_path)
        except Exception:
            pass

    use_preprocessor = preprocessor is not None
    if use_preprocessor:
        X_train_aligned = _align_to_preprocessor(X_train, preprocessor)
        X_test_aligned = _align_to_preprocessor(X_test, preprocessor)
        X_train_transformed = preprocessor.transform(X_train_aligned)
        X_test_transformed = preprocessor.transform(X_test_aligned)
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()
        try:
            X_train_transformed = np.asarray(X_train_transformed, dtype=float)
            X_test_transformed = np.asarray(X_test_transformed, dtype=float)
        except Exception as exc:
            raise SystemExit(
                f"Preprocessed data is not numeric; cannot run LIME. ({exc})"
            )
        if bundle_feature_names:
            feature_names = list(bundle_feature_names)
        elif hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"f{i}" for i in range(X_train_transformed.shape[1])]
        else:
            feature_names = [f"f{i}" for i in range(X_train_transformed.shape[1])]
        categorical_features = []
        categorical_names = None
    else:
        X_train_transformed = X_train.values
        X_test_transformed = X_test.values
        feature_names = raw_feature_names
        categorical_mask = X_train.dtypes.isin(["object", "category", "bool"])
        categorical_features = [i for i, is_cat in enumerate(categorical_mask) if is_cat]
        categorical_names = {}
        for i in categorical_features:
            categorical_names[i] = (
                X_train.iloc[:, i].astype(str).fillna("<NA>").unique().tolist()
            )

    class_names = _load_class_names(args.y_train_path)

    def predict_proba(x: np.ndarray) -> np.ndarray:
        if use_preprocessor:
            return model.predict_proba(x)
        df = pd.DataFrame(x, columns=raw_feature_names)
        return model.predict_proba(df)

    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        discretize_continuous=True,
        mode="classification",
        random_state=42,
    )

    if args.index < 0 or args.index >= len(X_test):
        raise SystemExit(f"index {args.index} is out of range for test set")

    instance = X_test_transformed[args.index]
    if args.aggregate:
        rng = np.random.default_rng(args.seed)
        total = len(X_test_transformed)
        if args.max_instances and args.max_instances < total:
            indices = rng.choice(total, size=args.max_instances, replace=False)
        else:
            indices = np.arange(total)

        pos_label = 1 if len(class_names) > 1 else 0
        agg_sum = {}
        agg_abs = {}
        agg_count = {}

        start_time = time.time()
        for i, idx in enumerate(indices, start=1):
            exp = explainer.explain_instance(
                X_test_transformed[idx],
                predict_proba,
                num_features=args.num_features,
                num_samples=args.num_samples,
            )
            for feat, weight in exp.as_list(label=pos_label):
                agg_sum[feat] = agg_sum.get(feat, 0.0) + weight
                agg_abs[feat] = agg_abs.get(feat, 0.0) + abs(weight)
                agg_count[feat] = agg_count.get(feat, 0) + 1

            if args.progress_every and (i % args.progress_every == 0 or i == len(indices)):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = (len(indices) - i) / rate if rate > 0 else 0.0
                print(
                    f"Progress: {i}/{len(indices)} | {rate:.2f} it/s | ETA {remaining/60:.1f} min",
                    flush=True,
                )

        rows = []
        for feat in agg_sum:
            rows.append(
                {
                    "feature": feat,
                    "mean_weight": agg_sum[feat] / agg_count[feat],
                    "mean_abs_weight": agg_abs[feat] / agg_count[feat],
                    "frequency": agg_count[feat],
                }
            )
        df = pd.DataFrame(rows).sort_values("mean_abs_weight", ascending=False)

        os.makedirs(os.path.dirname(args.output_agg_csv), exist_ok=True)
        df.to_csv(args.output_agg_csv, index=False)

        try:
            import matplotlib.pyplot as plt

            top = df.head(20).iloc[::-1]
            plt.figure(figsize=(10, 8))
            plt.barh(top["feature"], top["mean_abs_weight"])
            plt.title("LIME global summary (mean |weight|)")
            plt.tight_layout()
            plt.savefig(args.output_agg_plot, dpi=150)
            plt.close()
        except Exception as exc:
            print(f"Failed to plot global summary: {exc}")

        print(f"Saved global summary CSV to {args.output_agg_csv}")
        if os.path.exists(args.output_agg_plot):
            print(f"Saved global summary plot to {args.output_agg_plot}")
    else:
        exp = explainer.explain_instance(
            instance,
            predict_proba,
            num_features=args.num_features,
            num_samples=args.num_samples,
        )

        os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
        with open(args.output_html, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

        with open(args.output_txt, "w", encoding="utf-8") as f:
            for label, score in exp.as_list():
                f.write(f"{label}: {score:.6f}\n")

        print(f"Saved HTML explanation to {args.output_html}")
        print(f"Saved text explanation to {args.output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
