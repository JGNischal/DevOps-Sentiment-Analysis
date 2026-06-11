"""
predict.py — Real-time sentiment inference for the IMDb sentiment-analysis project.

Primary entry-point
-------------------
  predict_sentiment(review_text, model_name) -> dict

  Returns
  -------
  {
      "sentiment":        "positive" | "negative",
      "confidence":       float  (0.0 – 1.0),
      "confidence_pct":   float  (0.0 – 100.0),
      "model_used":       str    (canonical name, e.g. "svm"),
      "model_label":      str    (human label, e.g. "SVM (LinearSVC)"),
      "clean_text":       str    (what the model actually saw),
      "raw_text_length":  int    (character count of original input),
  }

Caching
-------
  Both the TF-IDF vectorizer and every loaded classifier are cached in
  module-level dicts after their first load.  A long-running Flask server
  therefore only pays disk I/O once per model per process lifetime.

  call clear_cache() to evict everything (useful in tests or after retraining).

Usage
-----
  # Direct
  python src/predict.py --review "This film was absolutely brilliant!" --model svm

  # As a module
  from predict import predict_sentiment
  result = predict_sentiment("Terrible plot, wasted two hours.", "naive_bayes")
"""

import os
import sys
import logging
import argparse
import json

import joblib
import numpy as np

from utils import clean_text, load_model, load_vectorizer

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
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR    = os.path.join(_PROJECT_ROOT, "models")
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

# Canonical model names → human-readable labels
_MODEL_LABELS: dict[str, str] = {
    "naive_bayes":   "Naive Bayes",
    "svm":           "SVM (LinearSVC)",
    "random_forest": "Random Forest",
}

# Binary label → string sentiment
_LABEL_TO_SENTIMENT: dict[int, str] = {1: "positive", 0: "negative"}

# Minimum non-trivial token count after cleaning; below this we warn the user.
_MIN_TOKEN_WARNING = 3

# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

def _extract_confidence(model, X_vec) -> tuple[int, float]:
    """
    Return (predicted_label, confidence_score ∈ [0, 1]).

    Strategy
    --------
    1. ``predict_proba``  — preferred; max probability of the winning class.
    2. ``decision_function`` — LinearSVC fallback; min-max normalised to [0,1].
    3. Hard ``predict`` only — last resort; confidence fixed at 0.5 (unknown).

    All three paths still return a hard label consistent with ``predict()``.
    """
    # ── path 1: predict_proba ────────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X_vec)[0]   # shape (2,)
        pred_label = int(np.argmax(proba))
        confidence = float(proba[pred_label])
        return pred_label, confidence

    # ── path 2: decision_function (raw LinearSVC) ────────────────────────────
    if hasattr(model, "decision_function"):
        score      = float(model.decision_function(X_vec)[0])
        pred_label = int(model.predict(X_vec)[0])
        # Sigmoid transform: maps any real score to (0, 1),
        # with 0 → 0.5 (maximum uncertainty at the decision boundary).
        confidence = float(1.0 / (1.0 + np.exp(-score)))
        # Ensure confidence reflects the *winning* class, not always "positive"
        if pred_label == 0:
            confidence = 1.0 - confidence
        return pred_label, confidence

    # ── path 3: hard predict only ────────────────────────────────────────────
    pred_label = int(model.predict(X_vec)[0])
    logger.warning(
        "Model exposes neither predict_proba nor decision_function. "
        "Confidence fixed at 0.5."
    )
    return pred_label, 0.5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(review_text: str, model_name: str) -> None:
    """
    Raise descriptive exceptions for bad inputs before any expensive work.

    Raises
    ------
    TypeError   : non-string review_text
    ValueError  : empty / whitespace-only text, or unrecognised model_name
    """
    if not isinstance(review_text, str):
        raise TypeError(
            f"review_text must be a str, got {type(review_text).__name__}."
        )

    if not review_text.strip():
        raise ValueError(
            "review_text is empty or contains only whitespace. "
            "Please provide a non-empty movie review."
        )

    if model_name not in _MODEL_LABELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose one of: {sorted(_MODEL_LABELS.keys())}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_sentiment(review_text: str, model_name: str) -> dict:
    """
    Predict the sentiment of a single movie review.

    Parameters
    ----------
    review_text : str
        Raw review text (may contain HTML, mixed case, punctuation, etc.).
        The same preprocessing pipeline used during training is applied
        automatically.
    model_name : str
        One of ``"naive_bayes"``, ``"svm"``, ``"random_forest"``.

    Returns
    -------
    dict
        Keys:
          sentiment        "positive" or "negative"
          confidence       float ∈ [0.0, 1.0] — certainty of prediction
          confidence_pct   confidence × 100, rounded to 2 d.p.
          model_used       canonical model name (e.g. "svm")
          model_label      human-readable label  (e.g. "SVM (LinearSVC)")
          clean_text       preprocessed text shown to the model
          raw_text_length  character count of the original input

    Raises
    ------
    TypeError
        If ``review_text`` is not a string.
    ValueError
        If ``review_text`` is empty, or ``model_name`` is unrecognised.
    FileNotFoundError
        If the vectorizer or model ``.pkl`` files have not been created yet.
    RuntimeError
        If the cleaned text is empty (e.g. review contained only HTML/digits).
    """
    # ── 1. Validate inputs ────────────────────────────────────────────────────
    _validate_inputs(review_text, model_name)

    raw_length = len(review_text)
    logger.debug(
        "predict_sentiment called: model=%s  input_len=%d",
        model_name, raw_length,
    )

    # ── 2. Load vectorizer (cached after first call) ─────────────────────────
    vectorizer = load_vectorizer()

    # ── 3. Load model (cached after first call) ───────────────────────────────
    model = load_model(model_name)

    # ── 4. Apply preprocessing pipeline ──────────────────────────────────────
    #       This is the *identical* function used in preprocess.py, imported
    #       from utils.py, so train/inference transformations are guaranteed
    #       to match.
    cleaned = clean_text(review_text)

    if not cleaned.strip():
        raise RuntimeError(
            "The review was empty after preprocessing. "
            "It may have contained only HTML, numbers, or stopwords. "
            "Please provide a more descriptive review."
        )

    token_count = len(cleaned.split())
    if token_count < _MIN_TOKEN_WARNING:
        logger.warning(
            "Very short review after cleaning (%d tokens). "
            "Prediction may be unreliable.",
            token_count,
        )

    # ── 5. TF-IDF transform ───────────────────────────────────────────────────
    #       transform() (not fit_transform) — we must use the vocabulary
    #       learned during training; unseen tokens are silently ignored.
    X_vec = vectorizer.transform([cleaned])   # shape: (1, n_features)

    # ── 6. Predict + extract confidence ───────────────────────────────────────
    pred_label, confidence = _extract_confidence(model, X_vec)

    sentiment  = _LABEL_TO_SENTIMENT[pred_label]
    conf_pct   = round(confidence * 100, 2)

    result = {
        "sentiment":       sentiment,
        "confidence":      round(confidence, 6),
        "confidence_pct":  conf_pct,
        "model_used":      model_name,
        "model_label":     _MODEL_LABELS[model_name],
        "clean_text":      cleaned,
        "raw_text_length": raw_length,
    }

    logger.debug("Result: %s", result)
    return result


def predict_all_models(review_text: str) -> dict[str, dict]:
    """
    Run ``predict_sentiment`` with every available model and return all
    results in a single dict keyed by model name.

    Useful for the "compare models" view in the frontend.

    Parameters
    ----------
    review_text : str
        Raw review text.

    Returns
    -------
    dict[str, dict]
        ``{ "naive_bayes": {...}, "svm": {...}, "random_forest": {...} }``
        Models whose ``.pkl`` files do not exist are silently skipped and
        their key is absent from the result.
    """
    results: dict[str, dict] = {}
    for name in _MODEL_LABELS:
        try:
            results[name] = predict_sentiment(review_text, name)
        except FileNotFoundError:
            logger.warning("Model '%s' not found — skipping.", name)
        except Exception as exc:
            logger.error("Error predicting with '%s': %s", name, exc)
    return results


def clear_cache() -> None:
    """
    Evict all in-memory cached models and the vectorizer.

    Delegates to ``utils.clear_cache()`` so both modules share the same
    singleton store.
    """
    from utils import clear_cache as _clear
    _clear()
    logger.info("Prediction cache cleared.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict sentiment for a single movie review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python src/predict.py --review "Absolutely loved this film!" --model svm
  python src/predict.py --review "Terrible waste of time." --model all
  echo "Great performances all round." | python src/predict.py --stdin --model naive_bayes
        """,
    )
    parser.add_argument(
        "--review", "-r",
        type=str,
        default=None,
        help="Review text to classify (wrap in quotes).",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(_MODEL_LABELS.keys()) + ["all"],
        default="svm",
        help="Model to use for prediction (default: svm).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read review text from standard input instead of --review.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output result as raw JSON (useful for piping).",
    )
    return parser


def _print_result(result: dict) -> None:
    """Pretty-print a single prediction result to stdout."""
    sentiment_icon = "✔" if result["sentiment"] == "positive" else "✘"
    bar_filled     = int(result["confidence"] * 30)
    bar            = "█" * bar_filled + "░" * (30 - bar_filled)

    print(f"\n  {'─' * 52}")
    print(f"  Sentiment   : {sentiment_icon}  {result['sentiment'].upper()}")
    print(f"  Confidence  : {result['confidence_pct']:>6.2f}%  [{bar}]")
    print(f"  Model       : {result['model_label']}")
    print(f"  Input chars : {result['raw_text_length']}")
    print(f"  Clean text  : {result['clean_text'][:80]}{'…' if len(result['clean_text']) > 80 else ''}")
    print(f"  {'─' * 52}\n")


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    # Resolve review text source
    if args.stdin:
        review = sys.stdin.read().strip()
    elif args.review:
        review = args.review
    else:
        parser.error("Provide --review TEXT or --stdin.")

    try:
        if args.model == "all":
            results = predict_all_models(review)
            if args.output_json:
                print(json.dumps(results, indent=2))
            else:
                for name, res in results.items():
                    print(f"\n  ── {res['model_label']} ──")
                    _print_result(res)
        else:
            result = predict_sentiment(review, args.model)
            if args.output_json:
                print(json.dumps(result, indent=2))
            else:
                _print_result(result)

    except (TypeError, ValueError) as exc:
        logger.error("Input error: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Missing artefact: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)