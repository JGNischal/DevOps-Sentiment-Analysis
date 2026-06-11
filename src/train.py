"""
train.py — Train and persist three sentiment classifiers on the preprocessed
           IMDb TF-IDF features.

Models trained
--------------
  1. Multinomial Naive Bayes   →  models/naive_bayes.pkl
  2. LinearSVC (SVM)           →  models/svm.pkl
  3. Random Forest             →  models/random_forest.pkl

Usage
-----
  python src/train.py                  # train all three
  python src/train.py --model svm      # train a single model

  # From another script:
  from train import train_all
  results = train_all()
"""

import os
import sys
import time
import argparse
import logging

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

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

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
_MODELS_DIR    = os.path.join(_PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry defines the estimator and output filename.
# LinearSVC is wrapped in CalibratedClassifierCV so it exposes
# predict_proba() — needed for ROC-AUC scoring in evaluate.py.

_MODEL_REGISTRY: dict[str, dict] = {
    "naive_bayes": {
        "label":     "Multinomial Naive Bayes",
        "estimator": MultinomialNB(alpha=0.1),
        "filename":  "naive_bayes.pkl",
        "notes":     "alpha=0.1 (mild Laplace smoothing)",
    },
    "svm": {
        "label":     "LinearSVC  (SVM)",
        "estimator": CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                max_iter=2000,
                dual=True,
                random_state=42,
            ),
            cv=3,
            method="sigmoid",
        ),
        "filename":  "svm.pkl",
        "notes":     "C=1.0, calibrated for probability output (3-fold sigmoid)",
    },
    "random_forest": {
        "label":     "Random Forest",
        "estimator": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        ),
        "filename":  "random_forest.pkl",
        "notes":     "200 trees, max_features=sqrt, class_weight=balanced",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(message: str, char: str = "─", width: int = 62) -> None:
    print(f"\n{char * width}")
    print(f"  {message}")
    print(f"{char * width}")


def _load_train_data() -> tuple:
    """
    Load X_train (sparse TF-IDF matrix) and y_train (binary array)
    from ``data/processed/``.

    Returns
    -------
    tuple[sparse_matrix, np.ndarray]

    Raises
    ------
    FileNotFoundError
        If preprocessing has not been run yet.
    """
    x_path = os.path.join(_PROCESSED_DIR, "X_train.pkl")
    y_path = os.path.join(_PROCESSED_DIR, "y_train.pkl")

    for path in (x_path, y_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed data not found: {path}\n"
                "Run 'python src/preprocess.py' (or 'python run_pipeline.py') first."
            )

    logger.info("Loading X_train from %s", x_path)
    X_train = joblib.load(x_path)

    logger.info("Loading y_train from %s", y_path)
    y_train = joblib.load(y_path)

    print(f"  X_train shape : {X_train.shape}  (sparse TF-IDF)")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  Class balance : pos={y_train.sum():,}  "
          f"neg={(y_train == 0).sum():,}")
    return X_train, y_train


def _save_model(model, filename: str) -> str:
    """
    Save *model* to ``models/<filename>`` with compression.

    Returns the full path written.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, filename)
    joblib.dump(model, path, compress=3)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  Saved  →  {path}  ({size_mb:.1f} MB)")
    return path


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"

# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    model_name:  str,
    X_train,
    y_train:     np.ndarray,
    *,
    force:       bool = False,
) -> dict:
    """
    Train a single classifier and persist it to ``models/``.

    Parameters
    ----------
    model_name : str
        Key in ``_MODEL_REGISTRY`` — one of ``"naive_bayes"``,
        ``"svm"``, ``"random_forest"``.
    X_train    : sparse matrix
        TF-IDF feature matrix for training samples.
    y_train    : np.ndarray
        Binary labels (1 = positive, 0 = negative).
    force      : bool
        If *False* (default), skip training when the ``.pkl`` file
        already exists and print a notice instead.

    Returns
    -------
    dict with keys:
      model_name, label, duration_s, path, skipped (bool)

    Raises
    ------
    ValueError
        If *model_name* is not in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}"
        )

    cfg  = _MODEL_REGISTRY[model_name]
    path = os.path.join(_MODELS_DIR, cfg["filename"])

    # ── skip guard ───────────────────────────────────────────────────────────
    if not force and os.path.exists(path):
        print(f"\n  [SKIP]  {cfg['label']} already trained at {path}")
        print(f"          Pass --force to retrain.")
        return {
            "model_name": model_name,
            "label":      cfg["label"],
            "duration_s": 0.0,
            "path":       path,
            "skipped":    True,
        }

    # ── train ────────────────────────────────────────────────────────────────
    print(f"\n  Model    : {cfg['label']}")
    print(f"  Settings : {cfg['notes']}")
    print(f"  Training on {X_train.shape[0]:,} samples …")

    t0    = time.perf_counter()
    model = cfg["estimator"]
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"  Training time : {_format_duration(elapsed)}")

    # ── save ─────────────────────────────────────────────────────────────────
    saved_path = _save_model(model, cfg["filename"])

    return {
        "model_name": model_name,
        "label":      cfg["label"],
        "duration_s": elapsed,
        "path":       saved_path,
        "skipped":    False,
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_all(
    model_names: list[str] | None = None,
    *,
    force: bool = False,
) -> list[dict]:
    """
    Train (and save) all three classifiers — or a specific subset.

    This function is the primary entry-point for ``run_pipeline.py``
    and any other orchestration script.

    Parameters
    ----------
    model_names : list[str] | None
        Models to train.  Defaults to all three when *None*.
    force : bool
        Re-train and overwrite even if the ``.pkl`` already exists.

    Returns
    -------
    list[dict]
        One result dict per model (see ``train_model()`` return value).
    """
    if model_names is None:
        model_names = list(_MODEL_REGISTRY.keys())

    # validate early
    unknown = set(model_names) - set(_MODEL_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown model names: {unknown}. "
            f"Valid options: {list(_MODEL_REGISTRY.keys())}"
        )

    _banner("SENTIMENT ANALYSIS — MODEL TRAINING", char="═")
    print(f"  Models to train : {model_names}")
    print(f"  Output dir      : {_MODELS_DIR}")
    print(f"  Force retrain   : {force}")

    # ── load shared training data once ───────────────────────────────────────
    _banner("Loading preprocessed training data")
    X_train, y_train = _load_train_data()

    # ── train each model ─────────────────────────────────────────────────────
    results      = []
    pipeline_t0  = time.perf_counter()

    for i, name in enumerate(model_names, 1):
        _banner(f"[{i}/{len(model_names)}]  {_MODEL_REGISTRY[name]['label']}")
        result = train_model(name, X_train, y_train, force=force)
        results.append(result)

    # ── summary table ─────────────────────────────────────────────────────────
    pipeline_elapsed = time.perf_counter() - pipeline_t0
    _banner("Training Summary", char="═")

    col_w = [20, 18, 10]
    header = (
        f"  {'Model':<{col_w[0]}}"
        f"{'Status':<{col_w[1]}}"
        f"{'Time':>{col_w[2]}}"
    )
    print(header)
    print(f"  {'─' * (sum(col_w) + 2)}")

    for r in results:
        status = "skipped" if r["skipped"] else "trained"
        dur    = "—" if r["skipped"] else _format_duration(r["duration_s"])
        print(
            f"  {r['label']:<{col_w[0]}}"
            f"{status:<{col_w[1]}}"
            f"{dur:>{col_w[2]}}"
        )

    print(f"\n  Total wall time : {_format_duration(pipeline_elapsed)}")
    print(f"  Models saved to : {_MODELS_DIR}\n")

    return results

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train sentiment classifiers on preprocessed IMDb TF-IDF data."
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and overwrite existing .pkl files",
    )
    args = parser.parse_args()

    names = None if args.model == "all" else [args.model]

    try:
        train_all(model_names=names, force=args.force)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during training: %s", exc)
        sys.exit(1)