"""
run_pipeline.py — End-to-end orchestration for the IMDb Sentiment Analysis pipeline.

Stages
------
  1. Preprocess   — clean, split, and TF-IDF vectorise the raw dataset
  2. Train        — fit Naive Bayes, SVM, and Random Forest classifiers
  3. Evaluate     — compute metrics and persist metrics.json + roc_curves.json

Usage
-----
  python run_pipeline.py                        # full pipeline
  python run_pipeline.py --stages train eval    # skip preprocessing
  python run_pipeline.py --force                # retrain even if models exist
  python run_pipeline.py --data-path /custom/IMDB_Dataset.csv
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Path setup — make src/ importable regardless of working directory
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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

_METRICS_FILE  = os.path.join(_PROJECT_ROOT, "data", "processed", "metrics.json")
_STAGE_NAMES   = ("preprocess", "train", "evaluate")
_TERM_WIDTH    = 64

# Metric columns shown in the final summary table
_SUMMARY_METRICS = [
    ("accuracy",  "Accuracy"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("f1",        "F1"),
    ("roc_auc",   "ROC-AUC"),
]

# ---------------------------------------------------------------------------
# Stage result tracking
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name:       str
    label:      str
    status:     str = "pending"   # "ok" | "failed" | "skipped"
    duration_s: float = 0.0
    error:      str = ""
    details:    dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _rule(char: str = "─", width: int = _TERM_WIDTH) -> str:
    return char * width


def _banner(title: str, char: str = "═") -> None:
    rule = _rule(char)
    print(f"\n{rule}")
    print(f"  {title}")
    print(rule)


def _stage_header(index: int, total: int, label: str) -> None:
    print(f"\n{_rule()}")
    print(f"  Stage {index}/{total}  ·  {label}")
    print(_rule())


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _status_icon(status: str) -> str:
    return {"ok": "✔", "failed": "✘", "skipped": "—", "pending": "·"}.get(status, "?")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_preprocess(data_path: str, **_kwargs) -> dict:
    """Import and call preprocess.main()."""
    from preprocess import main as preprocess_main

    preprocess_main(csv_path=data_path)
    return {}


def _run_train(force: bool = False, **_kwargs) -> dict:
    """Import and call train.train_all()."""
    from train import train_all

    results = train_all(force=force)
    trained  = [r["label"] for r in results if not r["skipped"]]
    skipped  = [r["label"] for r in results if r["skipped"]]
    return {"trained": trained, "skipped": skipped}


def _run_evaluate(**_kwargs) -> dict:
    """Import and call evaluate.evaluate_all()."""
    from evaluate import evaluate_all

    metrics = evaluate_all()
    best    = max(metrics, key=lambda m: metrics[m]["f1"])
    return {
        "best_model": metrics[best]["label"],
        "best_f1":    metrics[best]["f1"],
    }


# Map stage name → runner function
_STAGE_RUNNERS: dict[str, Callable] = {
    "preprocess": _run_preprocess,
    "train":      _run_train,
    "evaluate":   _run_evaluate,
}

_STAGE_LABELS: dict[str, str] = {
    "preprocess": "Preprocess  (clean · split · vectorise)",
    "train":      "Train       (Naive Bayes · SVM · Random Forest)",
    "evaluate":   "Evaluate    (metrics · ROC curves)",
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[StageResult]) -> None:
    """Print the pipeline stage summary and, if available, the metrics table."""

    _banner("PIPELINE SUMMARY", char="═")

    # ── Stage status table ───────────────────────────────────────────────────
    col_label  = 42
    col_status = 10
    col_time   = 10

    header = (
        f"  {'Stage':<{col_label}}"
        f"{'Status':<{col_status}}"
        f"{'Duration':>{col_time}}"
    )
    print(header)
    print(f"  {_rule('─', col_label + col_status + col_time)}")

    for r in results:
        icon     = _status_icon(r.status)
        status   = f"{icon}  {r.status}"
        duration = _format_duration(r.duration_s) if r.status == "ok" else "—"
        print(
            f"  {r.label:<{col_label}}"
            f"{status:<{col_status + 3}}"
            f"{duration:>{col_time}}"
        )
        if r.status == "failed" and r.error:
            # Indent error message under the stage row
            for line in r.error.splitlines()[:3]:
                print(f"      ↳ {line.strip()}")

    print()

    # ── Per-model metrics (if evaluate succeeded) ────────────────────────────
    if not os.path.exists(_METRICS_FILE):
        print("  (metrics.json not found — evaluate stage may have failed)")
        return

    try:
        with open(_METRICS_FILE, "r", encoding="utf-8") as fh:
            all_metrics: dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Could not read metrics.json: {exc}")
        return

    if not all_metrics:
        print("  (metrics.json is empty)")
        return

    _banner("MODEL METRICS", char="─")

    model_names  = list(all_metrics.keys())
    model_labels = [all_metrics[m]["label"] for m in model_names]

    metric_col = 18
    value_col  = 17

    sep = "  " + _rule("─", metric_col + value_col * len(model_names))

    # Header row
    header = f"  {'Metric':<{metric_col}}"
    for lbl in model_labels:
        header += f"{lbl:^{value_col}}"
    print(header)
    print(sep)

    # One row per metric
    for key, label in _SUMMARY_METRICS:
        values = [all_metrics[m].get(key, 0.0) for m in model_names]
        best   = max(values)
        row    = f"  {label:<{metric_col}}"
        for v in values:
            cell = f"{v:.4f}"
            cell = f"[{cell}]" if v == best else f" {cell} "
            row += f"{cell:^{value_col}}"
        print(row)

    print(sep)

    # Inference time row
    row = f"  {'Inference (s)':<{metric_col}}"
    for m in model_names:
        t = all_metrics[m].get("inference_time_s", 0.0)
        row += f"{t:.4f}s".center(value_col)
    print(row)
    print(sep)

    # Best model callout
    best_model = max(all_metrics, key=lambda m: all_metrics[m]["f1"])
    best_f1    = all_metrics[best_model]["f1"]
    best_auc   = all_metrics[best_model]["roc_auc"]
    print(
        f"\n  Best model (F1): {all_metrics[best_model]['label']}"
        f"  —  F1={best_f1:.4f}  ROC-AUC={best_auc:.4f}"
    )
    print()


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    stages:    list[str],
    data_path: str,
    force:     bool = False,
) -> list[StageResult]:
    """
    Execute the requested pipeline stages in order.

    Each stage is called via a direct Python import (no subprocess).
    Exceptions are caught, logged, and recorded in the StageResult — the
    pipeline always continues to the next stage.

    Parameters
    ----------
    stages    : list[str]   Ordered list of stage names to run.
    data_path : str         Path to the raw IMDb CSV.
    force     : bool        Pass --force to the train stage.

    Returns
    -------
    list[StageResult]
        One entry per requested stage.
    """
    kwargs = {"data_path": data_path, "force": force}

    results    = []
    total      = len(stages)
    pipeline_t = time.perf_counter()

    for i, stage_name in enumerate(stages, 1):
        label  = _STAGE_LABELS[stage_name]
        result = StageResult(name=stage_name, label=label)
        results.append(result)

        _stage_header(i, total, label)

        t0 = time.perf_counter()
        try:
            details         = _STAGE_RUNNERS[stage_name](**kwargs)
            result.status   = "ok"
            result.details  = details or {}
            result.duration_s = time.perf_counter() - t0

            print(
                f"\n  {_status_icon('ok')}  {label.split('(')[0].strip()} "
                f"completed in {_format_duration(result.duration_s)}"
            )

            # Print any extra details returned by the runner
            if result.details:
                for k, v in result.details.items():
                    print(f"     {k}: {v}")

        except FileNotFoundError as exc:
            result.status    = "failed"
            result.duration_s = time.perf_counter() - t0
            result.error     = str(exc)
            logger.error("[%s] FileNotFoundError: %s", stage_name, exc)
            print(f"\n  {_status_icon('failed')}  Stage FAILED — {exc}")
            print("     Pipeline continuing to next stage …")

        except Exception as exc:
            result.status    = "failed"
            result.duration_s = time.perf_counter() - t0
            result.error     = str(exc)
            tb_lines = traceback.format_exc().strip().splitlines()
            # Print last 5 lines of traceback for context without flooding
            short_tb = "\n".join(tb_lines[-5:])
            logger.error("[%s] Unexpected error:\n%s", stage_name, short_tb)
            print(f"\n  {_status_icon('failed')}  Stage FAILED — {exc}")
            print(f"  Traceback (last 5 lines):\n{short_tb}")
            print("  Pipeline continuing to next stage …")

    total_elapsed = time.perf_counter() - pipeline_t
    print(f"\n  Total pipeline time: {_format_duration(total_elapsed)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    default_csv = os.path.join(_PROJECT_ROOT, "data", "IMDB Dataset.csv")

    parser = argparse.ArgumentParser(
        description="Run the IMDb Sentiment Analysis ML pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python run_pipeline.py                          # full pipeline
  python run_pipeline.py --stages train evaluate  # skip preprocessing
  python run_pipeline.py --stages evaluate        # re-evaluate only
  python run_pipeline.py --force                  # force model retrain
  python run_pipeline.py --data-path /data/imdb.csv
        """,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=_STAGE_NAMES,
        default=list(_STAGE_NAMES),
        metavar="STAGE",
        help=(
            "Which stages to run (default: all three).  "
            f"Choices: {_STAGE_NAMES}."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=default_csv,
        help=f"Path to IMDB Dataset.csv  (default: {default_csv})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force model retraining even if .pkl files already exist.",
    )
    return parser


def main() -> int:
    """
    Entry-point for CLI use.

    Returns
    -------
    int
        Exit code — 0 if all stages passed, 1 if any stage failed.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # Validate stage order — always run in canonical order even if the user
    # specifies them out of order.
    requested = [s for s in _STAGE_NAMES if s in args.stages]

    _banner("IMDb SENTIMENT ANALYSIS — ML PIPELINE", char="═")
    print(f"  Stages     : {requested}")
    print(f"  Data path  : {args.data_path}")
    print(f"  Force      : {args.force}")
    print(f"  Project    : {_PROJECT_ROOT}")

    # Run
    results = run_pipeline(
        stages    = requested,
        data_path = args.data_path,
        force     = args.force,
    )

    # Summary
    _print_summary(results)

    # Exit code — non-zero if any stage failed
    failed = [r for r in results if r.status == "failed"]
    if failed:
        names = ", ".join(r.name for r in failed)
        print(f"  ⚠  {len(failed)} stage(s) failed: {names}")
        print("  Check the output above for details.\n")
        return 1

    print(f"  All {len(results)} stage(s) completed successfully.\n")
    return 0


# ---------------------------------------------------------------------------
# Direct import API (for use from app.py or notebooks)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    data_path: str | None = None,
    force:     bool       = False,
) -> list[StageResult]:
    """
    Run all three pipeline stages programmatically without touching sys.argv.

    Parameters
    ----------
    data_path : str | None
        Path to the raw CSV.  Defaults to ``data/IMDB Dataset.csv``.
    force : bool
        Force model retrain.

    Returns
    -------
    list[StageResult]
    """
    if data_path is None:
        data_path = os.path.join(_PROJECT_ROOT, "data", "IMDB Dataset.csv")

    results = run_pipeline(
        stages    = list(_STAGE_NAMES),
        data_path = data_path,
        force     = force,
    )
    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())