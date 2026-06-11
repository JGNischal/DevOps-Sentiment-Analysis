"""
preprocess.py — Data loading, cleaning, splitting, and TF-IDF vectorisation
             for the IMDb 50 k sentiment-analysis dataset.

Outputs saved to data/processed/
  tfidf_vectorizer.pkl   fitted TfidfVectorizer
  X_train.pkl            sparse TF-IDF matrix  (train)
  X_test.pkl             sparse TF-IDF matrix  (test)
  y_train.pkl            binary label array     (train)
  y_test.pkl             binary label array     (test)
  label_map.pkl          {"positive": 1, "negative": 0}

Usage
-----
  python src/preprocess.py
  python src/preprocess.py --data-path path/to/IMDB_Dataset.csv
"""

import os
import time
import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Re-use the shared cleaning pipeline from utils so training and inference
# are guaranteed to apply identical transformations.
from utils import clean_text

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
_DEFAULT_CSV  = os.path.join(_PROJECT_ROOT, "data", "IMDB Dataset.csv")
_OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "data", "processed")

# TF-IDF hyper-parameters
_TFIDF_MAX_FEATURES = 50_000
_TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams
_TFIDF_MIN_DF       = 2        # ignore extremely rare terms
_TFIDF_SUBLINEAR_TF = True     # apply 1 + log(tf) scaling

# Train / test split
_TEST_SIZE    = 0.20
_RANDOM_STATE = 42

# Label encoding
_LABEL_MAP = {"positive": 1, "negative": 0}

# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _banner(step: int, total: int, message: str) -> None:
    """Print a clearly visible step header."""
    print(f"\n{'─' * 60}")
    print(f"  Step {step}/{total} — {message}")
    print(f"{'─' * 60}")


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the IMDb CSV and perform basic sanity checks.

    Parameters
    ----------
    csv_path : str
        Path to the raw ``IMDB Dataset.csv`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``review`` (str) and ``sentiment`` (str).

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing or unexpected label values appear.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download 'IMDB Dataset.csv' from Kaggle and place it in data/."
        )

    logger.info("Reading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)

    # --- column validation ---
    required = {"review", "sentiment"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["review", "sentiment"]].copy()

    # --- label validation ---
    unexpected_labels = set(df["sentiment"].unique()) - set(_LABEL_MAP.keys())
    if unexpected_labels:
        raise ValueError(
            f"Unexpected sentiment values: {unexpected_labels}. "
            f"Expected only {set(_LABEL_MAP.keys())}."
        )

    # --- null check ---
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Dropping %d rows with null values.", null_counts.sum())
        df.dropna(inplace=True)

    # --- duplicate check ---
    n_dupes = df.duplicated(subset="review").sum()
    if n_dupes:
        logger.warning("Dropping %d duplicate reviews.", n_dupes)
        df.drop_duplicates(subset="review", inplace=True)

    df.reset_index(drop=True, inplace=True)

    print(f"  Loaded  : {len(df):,} reviews")
    print(f"  Balance : {df['sentiment'].value_counts().to_dict()}")
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Convert string sentiment labels to binary integers in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``sentiment`` column with values ``"positive"``
        or ``"negative"``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Modified DataFrame and the label map used for encoding.
    """
    df["label"] = df["sentiment"].map(_LABEL_MAP)
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    print(f"  Encoded : positive={pos:,}  negative={neg:,}")
    return df, _LABEL_MAP


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ``clean_text()`` to every review, storing results in
    ``df['clean_review']``.

    Progress is logged every 10 000 rows so long runs feel responsive.
    """
    total   = len(df)
    cleaned = []

    logger.info("Cleaning %d reviews — this may take a few minutes …", total)
    t0 = time.time()

    for i, text in enumerate(df["review"], 1):
        cleaned.append(clean_text(text))
        if i % 10_000 == 0 or i == total:
            elapsed = time.time() - t0
            pct     = 100 * i / total
            logger.info("  %d / %d  (%.0f%%)  —  %.1fs elapsed", i, total, pct, elapsed)

    df["clean_review"] = cleaned
    elapsed_total = time.time() - t0
    print(f"  Cleaned {total:,} reviews in {elapsed_total:.1f}s")

    # Show a before/after sample
    sample_idx = df.index[0]
    print(f"\n  [Sample] original  : {df.loc[sample_idx, 'review'][:120].strip()} …")
    print(f"  [Sample] cleaned   : {df.loc[sample_idx, 'clean_review'][:120].strip()} …")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratified 80 / 20 train-test split on the cleaned reviews.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.Series
    """
    X = df["clean_review"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = _TEST_SIZE,
        random_state = _RANDOM_STATE,
        stratify     = y,
    )

    print(f"  Train   : {len(X_train):,} reviews  "
          f"(pos={y_train.sum():,}  neg={(y_train == 0).sum():,})")
    print(f"  Test    : {len(X_test):,}  reviews  "
          f"(pos={y_test.sum():,}  neg={(y_test == 0).sum():,})")
    return X_train, X_test, y_train, y_test


def vectorise(
    X_train: pd.Series,
    X_test:  pd.Series,
) -> tuple:
    """
    Fit a TF-IDF vectorizer on the training set and transform both splits.

    Parameters
    ----------
    X_train, X_test : pd.Series
        Pre-cleaned review text.

    Returns
    -------
    tuple[TfidfVectorizer, sparse_matrix, sparse_matrix]
        (fitted vectorizer, X_train_tfidf, X_test_tfidf)
    """
    logger.info(
        "Fitting TF-IDF (max_features=%d, ngram_range=%s) …",
        _TFIDF_MAX_FEATURES,
        _TFIDF_NGRAM_RANGE,
    )

    vectorizer = TfidfVectorizer(
        max_features = _TFIDF_MAX_FEATURES,
        ngram_range  = _TFIDF_NGRAM_RANGE,
        min_df       = _TFIDF_MIN_DF,
        sublinear_tf = _TFIDF_SUBLINEAR_TF,
        strip_accents = "unicode",
        analyzer     = "word",
        token_pattern= r"\b[a-z]{2,}\b",   # only alpha tokens ≥ 2 chars
    )

    t0            = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    elapsed       = time.time() - t0

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Vocabulary  : {vocab_size:,} tokens  (max allowed: {_TFIDF_MAX_FEATURES:,})")
    print(f"  Train shape : {X_train_tfidf.shape}  (sparse)")
    print(f"  Test shape  : {X_test_tfidf.shape}   (sparse)")
    print(f"  Vectorised in {elapsed:.1f}s")

    return vectorizer, X_train_tfidf, X_test_tfidf


def save_artifacts(
    vectorizer,
    X_train_tfidf,
    X_test_tfidf,
    y_train: pd.Series,
    y_test:  pd.Series,
    output_dir: str,
) -> None:
    """
    Persist all preprocessed artefacts to *output_dir* as ``.pkl`` files.

    Parameters
    ----------
    vectorizer           : fitted TfidfVectorizer
    X_train_tfidf        : sparse matrix
    X_test_tfidf         : sparse matrix
    y_train, y_test      : pd.Series of binary labels
    output_dir           : directory path (created if absent)
    """
    os.makedirs(output_dir, exist_ok=True)

    artefacts = {
        "tfidf_vectorizer.pkl": vectorizer,
        "X_train.pkl":          X_train_tfidf,
        "X_test.pkl":           X_test_tfidf,
        "y_train.pkl":          y_train.values,
        "y_test.pkl":           y_test.values,
        "label_map.pkl":        _LABEL_MAP,
    }

    for filename, obj in artefacts.items():
        path = os.path.join(output_dir, filename)
        joblib.dump(obj, path, compress=3)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  Saved  {filename:<30}  ({size_mb:.1f} MB)  →  {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str = _DEFAULT_CSV, output_dir: str = _OUTPUT_DIR) -> None:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    csv_path   : str  Path to ``IMDB Dataset.csv``.
    output_dir : str  Directory where processed artefacts are saved.
    """
    TOTAL_STEPS = 6
    pipeline_start = time.time()

    # ── Step 1: Load ────────────────────────────────────────────────────────
    _banner(1, TOTAL_STEPS, "Load raw data")
    df = load_data(csv_path)

    # ── Step 2: Encode labels ───────────────────────────────────────────────
    _banner(2, TOTAL_STEPS, "Encode labels")
    df, label_map = encode_labels(df)

    # ── Step 3: Clean text ──────────────────────────────────────────────────
    _banner(3, TOTAL_STEPS, "Clean text")
    df = apply_cleaning(df)

    # ── Step 4: Train / test split ──────────────────────────────────────────
    _banner(4, TOTAL_STEPS, "Train / test split  (80 / 20, stratified)")
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Step 5: TF-IDF vectorisation ────────────────────────────────────────
    _banner(5, TOTAL_STEPS, "TF-IDF vectorisation")
    vectorizer, X_train_tfidf, X_test_tfidf = vectorise(X_train, X_test)

    # ── Step 6: Save artefacts ──────────────────────────────────────────────
    _banner(6, TOTAL_STEPS, f"Save artefacts → {output_dir}")
    save_artifacts(
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        output_dir,
    )

    elapsed_total = time.time() - pipeline_start
    print(f"\n{'═' * 60}")
    print(f"  Preprocessing complete in {elapsed_total:.1f}s")
    print(f"  Artefacts saved to: {output_dir}")
    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the IMDb 50k dataset for sentiment analysis."
    )
    parser.add_argument(
        "--data-path",
        default=_DEFAULT_CSV,
        help=f"Path to IMDB Dataset.csv  (default: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        default=_OUTPUT_DIR,
        help=f"Directory to write processed artefacts  (default: {_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    main(csv_path=args.data_path, output_dir=args.output_dir)