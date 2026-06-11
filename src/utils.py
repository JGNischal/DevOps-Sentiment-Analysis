"""
utils.py — Shared utilities for the sentiment analysis pipeline.

Provides:
  - clean_text(text)          : full preprocessing pipeline
  - load_model(model_name)    : load + cache a trained classifier
  - load_vectorizer()         : load + cache the TF-IDF vectorizer
  - get_available_models()    : list models that have been trained
"""

import os
import re
import string
import logging

import joblib
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
# NLTK resource bootstrap — download quietly if not already present
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download required NLTK data packages if they are missing."""
    resources = {
        "corpora/stopwords":        "stopwords",
        "tokenizers/punkt":         "punkt",
        "tokenizers/punkt_tab":     "punkt_tab",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", package)
            nltk.download(package, quiet=True)

_ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded, cached after first access)
# ---------------------------------------------------------------------------

_stemmer: PorterStemmer | None = None
_stop_words: frozenset | None = None
_vectorizer = None                  # TF-IDF vectorizer
_model_cache: dict = {}             # { model_name: fitted_classifier }

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve paths relative to the project root (two levels above this file)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "models")
_DATA_DIR     = os.path.join(_PROJECT_ROOT, "data", "processed")

# Canonical names → file names
_MODEL_FILES: dict[str, str] = {
    "naive_bayes":    "naive_bayes.pkl",
    "svm":            "svm.pkl",
    "random_forest":  "random_forest.pkl",
}

_VECTORIZER_FILE = os.path.join(_DATA_DIR, "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_stemmer() -> PorterStemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer


def _get_stop_words() -> frozenset:
    global _stop_words
    if _stop_words is None:
        _stop_words = frozenset(stopwords.words("english"))
    return _stop_words

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Full text-cleaning pipeline used by both training and inference.

    Steps
    -----
    1. Coerce to string and lowercase.
    2. Strip HTML tags (BeautifulSoup lxml/html.parser).
    3. Remove URLs.
    4. Remove punctuation and digits.
    5. Collapse whitespace.
    6. Tokenise on whitespace.
    7. Remove stopwords.
    8. Stem each token with PorterStemmer.
    9. Rejoin tokens into a single string.

    Parameters
    ----------
    text : str
        Raw review text (may contain HTML).

    Returns
    -------
    str
        Cleaned, stemmed text ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Strip HTML
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6-8. Tokenise → remove stopwords → stem
    stemmer   = _get_stemmer()
    stop_words = _get_stop_words()

    tokens = [
        stemmer.stem(token)
        for token in text.split()
        if token and token not in stop_words and len(token) > 1
    ]

    return " ".join(tokens)


def load_model(model_name: str):
    """
    Load a trained classifier from ``models/`` and cache it in memory.

    Parameters
    ----------
    model_name : str
        One of ``"naive_bayes"``, ``"svm"``, or ``"random_forest"``.

    Returns
    -------
    Fitted scikit-learn estimator.

    Raises
    ------
    ValueError
        If *model_name* is not a recognised key.
    FileNotFoundError
        If the corresponding ``.pkl`` file does not exist on disk.
    """
    if model_name not in _MODEL_FILES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available options: {list(_MODEL_FILES.keys())}"
        )

    if model_name not in _model_cache:
        path = os.path.join(_MODELS_DIR, _MODEL_FILES[model_name])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run 'python src/train.py' (or 'python run_pipeline.py') first."
            )
        logger.info("Loading model '%s' from %s", model_name, path)
        _model_cache[model_name] = joblib.load(path)

    return _model_cache[model_name]


def load_vectorizer():
    """
    Load the TF-IDF vectorizer from ``data/processed/`` and cache it.

    Returns
    -------
    Fitted ``TfidfVectorizer``.

    Raises
    ------
    FileNotFoundError
        If ``tfidf_vectorizer.pkl`` is not present.
    """
    global _vectorizer
    if _vectorizer is None:
        if not os.path.exists(_VECTORIZER_FILE):
            raise FileNotFoundError(
                f"Vectorizer not found: {_VECTORIZER_FILE}\n"
                "Run 'python src/preprocess.py' (or 'python run_pipeline.py') first."
            )
        logger.info("Loading TF-IDF vectorizer from %s", _VECTORIZER_FILE)
        _vectorizer = joblib.load(_VECTORIZER_FILE)
    return _vectorizer


def get_available_models() -> list[str]:
    """
    Return a list of model names whose ``.pkl`` files exist in ``models/``.

    Iterates over the canonical model registry; only names whose files are
    present on disk are included, so the list reflects what has actually been
    trained.

    Returns
    -------
    list[str]
        Subset of ``["naive_bayes", "svm", "random_forest"]``.
    """
    available = []
    for name, filename in _MODEL_FILES.items():
        path = os.path.join(_MODELS_DIR, filename)
        if os.path.exists(path):
            available.append(name)
    return available


# ---------------------------------------------------------------------------
# Cache management (useful for long-running servers)
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Evict all in-memory cached models and the vectorizer."""
    global _vectorizer
    _model_cache.clear()
    _vectorizer = None
    logger.info("Model and vectorizer cache cleared.")


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "<br />This is a <b>GREAT</b> movie! I loved it 100%. "
        "Highly recommended for everyone in 2024."
    )
    print("Raw text  :", sample)
    print("Cleaned   :", clean_text(sample))
    print()
    print("Available models:", get_available_models() or "(none trained yet)")