"""
evaluate.py — Evaluate all three trained sentiment classifiers on the held-out
              test set and persist results for the frontend dashboard.

Outputs written to data/processed/
  metrics.json      Per-model accuracy, precision, recall, F1, ROC-AUC,
                    confusion matrix, and per-class report.
  roc_curves.json   FPR / TPR / threshold arrays for each model's ROC curve.

Usage
-----
  python src/evaluate.py
  python src/evaluate.py --output-dir /custom/path

  # From another script:
  from evaluate import evaluate_all
  metrics = evaluate_all()
"""

import os
import sys
import json
import time
import logging
import argparse

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
_MODELS_DIR    = os.path.join(_PROJECT_ROOT, "models")

_MODEL_FILES: dict[str, str] = {
    "naive_bayes":   "naive_bayes.pkl",
    "svm":           "svm.pkl",
    "random_forest": "random_forest.pkl",
}

_MODEL_LABELS: dict[str, str] = {
    "naive_bayes":   "Naive Bayes",
    "svm":           "SVM (LinearSVC)",
    "random_forest": "Random Forest",
}

_METRICS_FILE    = "metrics.json"
_ROC_CURVES_FILE = "roc_curves.json"

# Number of ROC curve points to store (keeps JSON small; frontend interpolates)
_ROC_DOWNSAMPLE = 200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(message: str, char: str = "─", width: int = 62) -> None:
    print(f"\n{char * width}")
    print(f"  {message}")
    print(f"{char * width}")


def _load_test_data() -> tuple:
    """
    Load X_test (sparse TF-IDF matrix) and y_test (binary array) from
    ``data/processed/``.

    Returns
    -------
    tuple[sparse_matrix, np.ndarray]

    Raises
    ------
    FileNotFoundError
        If preprocessing has not been run yet.
    """
    x_path = os.path.join(_PROCESSED_DIR, "X_test.pkl")
    y_path = os.path.join(_PROCESSED_DIR, "y_test.pkl")

    for path in (x_path, y_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed test data not found: {path}\n"
                "Run 'python src/preprocess.py' first."
            )

    logger.info("Loading test data …")
    X_test = joblib.load(x_path)
    y_test = joblib.load(y_path)

    print(f"  X_test shape : {X_test.shape}")
    print(f"  y_test shape : {y_test.shape}")
    print(f"  Class balance: pos={y_test.sum():,}  neg={(y_test == 0).sum():,}")
    return X_test, y_test


def _load_model(model_name: str):
    """
    Load a trained classifier from ``models/``.

    Raises
    ------
    FileNotFoundError
        If the ``.pkl`` file is absent (model not yet trained).
    """
    path = os.path.join(_MODELS_DIR, _MODEL_FILES[model_name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found: {path}\n"
            "Run 'python src/train.py' first."
        )
    logger.info("Loading %s from %s", model_name, path)
    return joblib.load(path)


def _get_probabilities(model, X_test) -> np.ndarray:
    """
    Return predicted positive-class probabilities for ROC-AUC computation.

    Falls back to decision function scores when ``predict_proba`` is
    unavailable (e.g. bare LinearSVC without calibration).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    # Bare LinearSVC path (should not normally be hit with CalibratedClassifierCV)
    scores = model.decision_function(X_test)
    # Normalise to [0, 1] via min-max for comparability
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min == 0:
        return np.full_like(scores, 0.5)
    return (scores - s_min) / (s_max - s_min)


def _downsample_roc(fpr: np.ndarray, tpr: np.ndarray, n: int) -> tuple:
    """
    Uniformly subsample *n* points from a ROC curve so the JSON stays small
    while the shape is fully preserved.  The first and last points are always
    kept to anchor the curve at (0, 0) and (1, 1).
    """
    total = len(fpr)
    if total <= n:
        return fpr, tpr
    indices = np.unique(
        np.concatenate([
            [0],
            np.linspace(1, total - 2, n - 2, dtype=int),
            [total - 1],
        ])
    )
    return fpr[indices], tpr[indices]


def _round4(value: float) -> float:
    """Round to 4 decimal places for clean JSON output."""
    return round(float(value), 4)


def _confusion_matrix_dict(cm: np.ndarray) -> dict:
    """Convert a 2×2 confusion matrix to a labelled dict."""
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negative":  int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive":  int(tp),
        # Derived rates (useful for the frontend)
        "false_positive_rate": _round4(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": _round4(fn / (fn + tp)) if (fn + tp) else 0.0,
    }

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    X_test,
    y_test: np.ndarray,
) -> tuple[dict, dict]:
    """
    Evaluate one trained classifier on the test set.

    Parameters
    ----------
    model_name : str
        Key in ``_MODEL_FILES``.
    X_test     : sparse matrix
        TF-IDF feature matrix for test samples.
    y_test     : np.ndarray
        Ground-truth binary labels.

    Returns
    -------
    tuple[dict, dict]
        (metrics_dict, roc_dict)

        metrics_dict keys:
          label, accuracy, precision, recall, f1, roc_auc,
          confusion_matrix, classification_report, inference_time_s

        roc_dict keys:
          label, fpr (list), tpr (list), auc
    """
    model = _load_model(model_name)

    # ── inference ────────────────────────────────────────────────────────────
    t0        = time.perf_counter()
    y_pred    = model.predict(X_test)
    infer_sec = time.perf_counter() - t0
    y_prob    = _get_probabilities(model, X_test)

    # ── scalar metrics ────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # ── confusion matrix ─────────────────────────────────────────────────────
    cm     = confusion_matrix(y_test, y_pred)
    cm_dict = _confusion_matrix_dict(cm)

    # ── per-class report ─────────────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    # Round all floats inside report for clean storage
    report_clean: dict = {}
    for key, val in report.items():
        if isinstance(val, dict):
            report_clean[key] = {k: _round4(v) if isinstance(v, float) else v
                                 for k, v in val.items()}
        else:
            report_clean[key] = _round4(val) if isinstance(val, float) else val

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
    fpr_ds, tpr_ds      = _downsample_roc(fpr_arr, tpr_arr, _ROC_DOWNSAMPLE)

    metrics_dict = {
        "label":                  _MODEL_LABELS[model_name],
        "accuracy":               _round4(accuracy),
        "precision":              _round4(precision),
        "recall":                 _round4(recall),
        "f1":                     _round4(f1),
        "roc_auc":                _round4(roc_auc),
        "confusion_matrix":       cm_dict,
        "classification_report":  report_clean,
        "inference_time_s":       _round4(infer_sec),
        "n_test_samples":         int(len(y_test)),
    }

    roc_dict = {
        "label": _MODEL_LABELS[model_name],
        "fpr":   [_round4(v) for v in fpr_ds.tolist()],
        "tpr":   [_round4(v) for v in tpr_ds.tolist()],
        "auc":   _round4(roc_auc),
    }

    return metrics_dict, roc_dict


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(all_metrics: dict[str, dict]) -> None:
    """Print a padded ASCII table comparing all model metrics side-by-side."""

    metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels   = {
        "accuracy":  "Accuracy",
        "precision": "Precision (macro)",
        "recall":    "Recall (macro)",
        "f1":        "F1 (macro)",
        "roc_auc":   "ROC-AUC",
    }

    model_names  = list(all_metrics.keys())
    model_labels = [all_metrics[m]["label"] for m in model_names]

    # Column widths
    metric_col = 20
    value_col  = 18

    sep   = "  " + "─" * (metric_col + value_col * len(model_names) + 2)
    print(sep)

    # Header
    header = f"  {'Metric':<{metric_col}}"
    for lbl in model_labels:
        header += f"{lbl:^{value_col}}"
    print(header)
    print(sep)

    # Rows
    for key in metrics_to_show:
        row = f"  {metric_labels[key]:<{metric_col}}"
        values = [all_metrics[m][key] for m in model_names]
        best   = max(values)
        for v in values:
            cell = f"{v:.4f}"
            if v == best:
                cell = f"[{cell}]"   # bracket the winner in each row
            row += f"{cell:^{value_col}}"
        print(row)

    print(sep)

    # Inference time row
    row = f"  {'Inference time (s)':<{metric_col}}"
    for m in model_names:
        row += f"{all_metrics[m]['inference_time_s']:.4f}s".center(value_col)
    print(row)
    print(sep)

    # Confusion matrices
    print("\n  Confusion matrices  (TN | FP | FN | TP)")
    print("  " + "─" * (metric_col + value_col * len(model_names) + 2))
    for m, lbl in zip(model_names, model_labels):
        cm = all_metrics[m]["confusion_matrix"]
        print(
            f"  {lbl:<{metric_col + 4}}"
            f"TN={cm['true_negative']:>5,}  "
            f"FP={cm['false_positive']:>5,}  "
            f"FN={cm['false_negative']:>5,}  "
            f"TP={cm['true_positive']:>5,}"
        )
    print()


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_json(obj: dict, output_dir: str, filename: str) -> str:
    """Serialise *obj* to ``output_dir/filename`` and return the full path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved  {filename:<25}  ({size_kb:.1f} KB)  →  {path}")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all(
    model_names: list[str] | None = None,
    output_dir:  str = _PROCESSED_DIR,
) -> dict:
    """
    Evaluate all trained classifiers and persist results.

    Parameters
    ----------
    model_names : list[str] | None
        Models to evaluate.  Defaults to all three when *None*.
    output_dir  : str
        Directory to write ``metrics.json`` and ``roc_curves.json``.

    Returns
    -------
    dict
        ``{ model_name: metrics_dict }`` — the same structure written to
        ``metrics.json``, useful for programmatic access from
        ``run_pipeline.py``.

    Raises
    ------
    FileNotFoundError
        If test data or a trained model is missing.
    """
    if model_names is None:
        model_names = list(_MODEL_FILES.keys())

    unknown = set(model_names) - set(_MODEL_FILES.keys())
    if unknown:
        raise ValueError(
            f"Unknown model names: {unknown}. "
            f"Valid options: {list(_MODEL_FILES.keys())}"
        )

    _banner("SENTIMENT ANALYSIS — MODEL EVALUATION", char="═")
    print(f"  Models     : {model_names}")
    print(f"  Output dir : {output_dir}")

    # ── load test data once ──────────────────────────────────────────────────
    _banner("Loading test data")
    X_test, y_test = _load_test_data()

    # ── evaluate each model ──────────────────────────────────────────────────
    all_metrics:    dict[str, dict] = {}
    all_roc_curves: dict[str, dict] = {}

    for i, name in enumerate(model_names, 1):
        _banner(f"[{i}/{len(model_names)}]  Evaluating {_MODEL_LABELS[name]}")
        metrics, roc = evaluate_model(name, X_test, y_test)
        all_metrics[name]    = metrics
        all_roc_curves[name] = roc

        # Quick per-model summary
        print(
            f"  accuracy={metrics['accuracy']:.4f}  "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    # ── comparison table ─────────────────────────────────────────────────────
    _banner("Comparison Table", char="═")
    _print_comparison_table(all_metrics)

    # ── persist results ──────────────────────────────────────────────────────
    _banner("Saving results")
    _save_json(all_metrics,    output_dir, _METRICS_FILE)
    _save_json(all_roc_curves, output_dir, _ROC_CURVES_FILE)

    # ── final summary ────────────────────────────────────────────────────────
    best_model = max(all_metrics, key=lambda m: all_metrics[m]["f1"])
    print(f"\n  Best model by F1 : {_MODEL_LABELS[best_model]}")
    print(f"  F1               : {all_metrics[best_model]['f1']:.4f}")
    print(f"  ROC-AUC          : {all_metrics[best_model]['roc_auc']:.4f}")
    print()

    return all_metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained sentiment classifiers on the held-out test set."
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_FILES.keys()) + ["all"],
        default="all",
        help="Which model(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=_PROCESSED_DIR,
        help=f"Directory for metrics.json and roc_curves.json  (default: {_PROCESSED_DIR})",
    )
    args = parser.parse_args()

    names = None if args.model == "all" else [args.model]

    try:
        evaluate_all(model_names=names, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during evaluation: %s", exc)
        sys.exit(1)