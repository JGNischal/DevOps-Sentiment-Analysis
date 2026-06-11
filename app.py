"""
app.py — Flask REST API for the IMDb Sentiment Analysis project.

Endpoints
---------
  GET  /api/health          Liveness check — always returns 200.
  GET  /api/models          List of trained models available on disk.
  GET  /api/metrics         Per-model evaluation metrics (metrics.json).
  GET  /api/roc-curves      ROC curve FPR/TPR arrays  (roc_curves.json).
  POST /api/predict         Predict sentiment for a single review.
  POST /api/predict/all     Predict with every available model at once.

Usage
-----
  python app.py                        # debug mode, port 5000
  python app.py --port 8080            # custom port
  python app.py --no-debug             # production-ish (still single-threaded)
  FLASK_ENV=production python app.py   # respects FLASK_ENV
"""

import os
import sys
import json
import time
import logging
import argparse
from functools import wraps
from datetime import datetime, timezone

from flask import Flask, jsonify, request, g
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging — set up before importing project modules so their loggers inherit
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports  (src/ is the working directory when run from project root;
# add it to sys.path so imports resolve regardless of how the script is run)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from predict import predict_sentiment, predict_all_models   # noqa: E402
from utils   import get_available_models                    # noqa: E402

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PROCESSED_DIR   = os.path.join(_PROJECT_ROOT, "data", "processed")
_METRICS_FILE    = os.path.join(_PROCESSED_DIR, "metrics.json")
_ROC_CURVES_FILE = os.path.join(_PROCESSED_DIR, "roc_curves.json")

_VALID_MODELS = {"naive_bayes", "svm", "random_forest"}

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """
    Construct and configure the Flask application.

    Returns
    -------
    Flask
        Fully configured app instance.
    """
    app = Flask(__name__)

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Allow all origins in development.  In production, restrict to your
    # frontend domain:  CORS(app, origins=["https://yourfrontend.com"])
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Request timing ────────────────────────────────────────────────────────
    @app.before_request
    def _start_timer():
        g.start_time = time.perf_counter()

    @app.after_request
    def _add_timing_header(response):
        if hasattr(g, "start_time"):
            elapsed_ms = (time.perf_counter() - g.start_time) * 1000
            response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Powered-By"] = "sentiment-analysis-api"
        return response

    # ── Error handlers ────────────────────────────────────────────────────────

    @app.errorhandler(400)
    def bad_request(exc):
        return _error_response(400, "Bad Request", str(exc))

    @app.errorhandler(404)
    def not_found(exc):
        return _error_response(404, "Not Found", str(exc))

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return _error_response(405, "Method Not Allowed", str(exc))

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("Unhandled server error: %s", exc)
        return _error_response(500, "Internal Server Error",
                               "An unexpected error occurred.")

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        """
        Liveness probe — returns 200 with basic runtime info.

        Response JSON
        -------------
        {
          "status":    "ok",
          "timestamp": "2024-01-15T10:30:00Z",
          "models_available": ["naive_bayes", "svm", "random_forest"]
        }
        """
        return jsonify({
            "status":           "ok",
            "timestamp":        _utc_now(),
            "models_available": get_available_models(),
        }), 200

    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/models")
    def list_models():
        """
        Return the list of model names whose ``.pkl`` files exist on disk.

        Response JSON
        -------------
        {
          "models": [
            { "name": "naive_bayes",   "label": "Naive Bayes" },
            { "name": "svm",           "label": "SVM (LinearSVC)" },
            { "name": "random_forest", "label": "Random Forest" }
          ],
          "count": 3
        }
        """
        label_map = {
            "naive_bayes":   "Naive Bayes",
            "svm":           "SVM (LinearSVC)",
            "random_forest": "Random Forest",
        }
        available = get_available_models()
        models    = [
            {"name": name, "label": label_map.get(name, name)}
            for name in available
        ]
        return jsonify({"models": models, "count": len(models)}), 200

    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/metrics")
    def get_metrics():
        """
        Return the full evaluation metrics dict generated by evaluate.py.

        Response JSON  (structure mirrors data/processed/metrics.json)
        -------------
        {
          "naive_bayes":   { "accuracy": 0.85, "f1": 0.85, ... },
          "svm":           { ... },
          "random_forest": { ... }
        }

        Errors
        ------
        404  metrics.json not found (evaluate.py has not been run yet).
        500  File is corrupt / unreadable.
        """
        return _serve_json_file(
            _METRICS_FILE,
            not_found_hint="Run 'python src/evaluate.py' to generate metrics.",
        )

    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/roc-curves")
    def get_roc_curves():
        """
        Return the ROC curve data (FPR / TPR arrays) for all models.

        Response JSON  (structure mirrors data/processed/roc_curves.json)
        -------------
        {
          "naive_bayes":   { "fpr": [...], "tpr": [...], "auc": 0.93 },
          "svm":           { ... },
          "random_forest": { ... }
        }

        Errors
        ------
        404  roc_curves.json not found.
        500  File is corrupt / unreadable.
        """
        return _serve_json_file(
            _ROC_CURVES_FILE,
            not_found_hint="Run 'python src/evaluate.py' to generate ROC data.",
        )

    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/predict")
    def predict():
        """
        Predict the sentiment of a single movie review.

        Request JSON
        ------------
        {
          "review": "This film was absolutely brilliant!",
          "model":  "svm"          (optional; defaults to "svm")
        }

        Response JSON  (200 OK)
        -------------
        {
          "sentiment":       "positive",
          "confidence":      0.924531,
          "confidence_pct":  92.45,
          "model_used":      "svm",
          "model_label":     "SVM (LinearSVC)",
          "clean_text":      "film absolut brilliant",
          "raw_text_length": 36
        }

        Errors
        ------
        400  Missing / empty review, unknown model name, non-JSON body.
        404  Model .pkl not found on disk.
        422  Review becomes empty after preprocessing.
        500  Unexpected inference error.
        """
        # ── parse body ────────────────────────────────────────────────────────
        body = _require_json(request)
        if isinstance(body, tuple):          # error response tuple
            return body

        review     = body.get("review", "")
        model_name = body.get("model", "svm")

        # ── validate ──────────────────────────────────────────────────────────
        if not isinstance(review, str) or not review.strip():
            return _error_response(
                400, "Bad Request",
                "Field 'review' is required and must be a non-empty string."
            )

        if model_name not in _VALID_MODELS:
            return _error_response(
                400, "Bad Request",
                f"Unknown model '{model_name}'. "
                f"Valid options: {sorted(_VALID_MODELS)}."
            )

        # ── check model is trained ────────────────────────────────────────────
        available = get_available_models()
        if model_name not in available:
            return _error_response(
                404, "Not Found",
                f"Model '{model_name}' has not been trained yet. "
                f"Available models: {available}. "
                "Run 'python src/train.py' to train all models."
            )

        # ── infer ─────────────────────────────────────────────────────────────
        try:
            result = predict_sentiment(review, model_name)
            return jsonify(result), 200

        except RuntimeError as exc:
            # Empty after preprocessing
            return _error_response(
                422, "Unprocessable Entity",
                str(exc)
            )
        except FileNotFoundError as exc:
            return _error_response(404, "Not Found", str(exc))
        except (TypeError, ValueError) as exc:
            return _error_response(400, "Bad Request", str(exc))
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            return _error_response(500, "Internal Server Error",
                                   "Prediction failed unexpectedly.")

    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/predict/all")
    def predict_all():
        """
        Run prediction with every available trained model.

        Request JSON
        ------------
        { "review": "The acting was wooden and the plot made no sense." }

        Response JSON  (200 OK)
        -------------
        {
          "naive_bayes":   { "sentiment": "negative", "confidence": 0.91, ... },
          "svm":           { ... },
          "random_forest": { ... }
        }

        Errors
        ------
        400  Missing / empty review or non-JSON body.
        404  No trained models found at all.
        """
        body = _require_json(request)
        if isinstance(body, tuple):
            return body

        review = body.get("review", "")
        if not isinstance(review, str) or not review.strip():
            return _error_response(
                400, "Bad Request",
                "Field 'review' is required and must be a non-empty string."
            )

        available = get_available_models()
        if not available:
            return _error_response(
                404, "Not Found",
                "No trained models found. Run 'python src/train.py' first."
            )

        try:
            results = predict_all_models(review)
            if not results:
                return _error_response(
                    500, "Internal Server Error",
                    "All model predictions failed. Check server logs."
                )
            return jsonify(results), 200

        except Exception as exc:
            logger.exception("predict_all failed: %s", exc)
            return _error_response(500, "Internal Server Error",
                                   "Bulk prediction failed unexpectedly.")

    # ─────────────────────────────────────────────────────────────────────────
    # 404 catch-all for unknown /api/* paths
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/")
    @app.get("/api")
    def api_index():
        """Return a simple API route map."""
        return jsonify({
            "api": "sentiment-analysis",
            "version": "1.0.0",
            "endpoints": [
                {"method": "GET",  "path": "/api/health"},
                {"method": "GET",  "path": "/api/models"},
                {"method": "GET",  "path": "/api/metrics"},
                {"method": "GET",  "path": "/api/roc-curves"},
                {"method": "POST", "path": "/api/predict"},
                {"method": "POST", "path": "/api/predict/all"},
            ],
        }), 200

    return app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_response(status: int, error: str, message: str):
    """
    Construct a consistent JSON error payload.

    Response body
    -------------
    {
      "error":     "Bad Request",
      "message":   "Field 'review' is required …",
      "status":    400,
      "timestamp": "2024-01-15T10:30:00Z"
    }
    """
    payload = {
        "error":     error,
        "message":   message,
        "status":    status,
        "timestamp": _utc_now(),
    }
    return jsonify(payload), status


def _require_json(req):
    """
    Extract the JSON body from *req*, returning an error tuple on failure.

    Returns either the parsed dict or a (Response, status_code) tuple.
    """
    if not req.is_json:
        return _error_response(
            400, "Bad Request",
            "Request Content-Type must be 'application/json'."
        )
    try:
        body = req.get_json(force=False, silent=False)
    except Exception:
        return _error_response(
            400, "Bad Request",
            "Request body is not valid JSON."
        )
    if body is None:
        return _error_response(
            400, "Bad Request",
            "Request body is empty or could not be parsed as JSON."
        )
    return body


def _serve_json_file(path: str, not_found_hint: str = ""):
    """
    Read *path* from disk and return its contents as a JSON response.

    Parameters
    ----------
    path            : str   Absolute path to a ``.json`` file.
    not_found_hint  : str   Extra guidance appended to 404 messages.
    """
    if not os.path.exists(path):
        msg = f"File not found: {os.path.basename(path)}."
        if not_found_hint:
            msg += f"  {not_found_hint}"
        return _error_response(404, "Not Found", msg)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify(data), 200
    except json.JSONDecodeError as exc:
        logger.error("Corrupt JSON at %s: %s", path, exc)
        return _error_response(
            500, "Internal Server Error",
            f"{os.path.basename(path)} exists but could not be parsed. "
            "Re-run the evaluation step to regenerate it."
        )
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return _error_response(500, "Internal Server Error",
                               "File read error — check server logs.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Sentiment Analysis Flask API server.",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("PORT", 5000)),
        help="Port to listen on (default: 5000, or $PORT env var).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable Flask debug mode (enabled by default).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    debug_mode = not args.no_debug
    # Honour FLASK_ENV=production if set
    if os.environ.get("FLASK_ENV") == "production":
        debug_mode = False

    app = create_app()

    print(f"\n{'═' * 56}")
    print(f"  Sentiment Analysis API")
    print(f"{'─' * 56}")
    print(f"  Host   : {args.host}")
    print(f"  Port   : {args.port}")
    print(f"  Debug  : {debug_mode}")
    print(f"  Models : {get_available_models() or '(none trained yet)'}")
    print(f"{'─' * 56}")
    print(f"  Endpoints:")
    print(f"    GET  http://{args.host}:{args.port}/api/health")
    print(f"    GET  http://{args.host}:{args.port}/api/models")
    print(f"    GET  http://{args.host}:{args.port}/api/metrics")
    print(f"    GET  http://{args.host}:{args.port}/api/roc-curves")
    print(f"    POST http://{args.host}:{args.port}/api/predict")
    print(f"    POST http://{args.host}:{args.port}/api/predict/all")
    print(f"{'═' * 56}\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=debug_mode,
    )