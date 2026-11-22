
"""Reusable text-cleaning and NLP preprocessing functions."""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Preserve words that can reverse or qualify sentiment.
NEGATION_WORDS = {
    "not", "no", "nor", "never", "neither", "without"
}

# Normalize common contractions before removing punctuation.
CONTRACTIONS = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "won't": "will not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
}


def ensure_nltk_resources(download: bool = False) -> bool:
    """Check NLTK resources and optionally download them."""

    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    missing = []

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(package_name)

    if missing and download:
        for package_name in missing:
            nltk.download(package_name, quiet=True)
        return ensure_nltk_resources(download=False)

    return not missing


@lru_cache(maxsize=1)
def _language_tools():
    """Load stopwords and the available morphology tool."""

    if ensure_nltk_resources():
        return set(stopwords.words("english")), WordNetLemmatizer(), True

    return set(ENGLISH_STOP_WORDS), PorterStemmer(), False


def clean_text(text: object) -> str:
    """Lowercase text, normalize contractions and remove punctuation."""

    if text is None:
        return ""

    value = str(text).lower()

    # Expand common contractions before removing apostrophes.
    for contraction, expansion in CONTRACTIONS.items():
        value = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            value,
        )

    value = re.sub(r"[^a-z\s]", " ", value)

    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: object) -> list[str]:
    """Tokenize normalized English text."""

    return re.findall(r"\b[a-z]+\b", clean_text(text))


def preprocess_text(text: object) -> str:
    """Remove stopwords while preserving negation words."""

    if not clean_text(text):
        return ""

    english_stopwords, morphology_tool, has_wordnet = _language_tools()

    # Do not remove negation words that can reverse sentiment.
    english_stopwords = english_stopwords - NEGATION_WORDS

    tokens = tokenize(text)

    processed_tokens = [
        (
            morphology_tool.lemmatize(token)
            if has_wordnet
            else morphology_tool.stem(token)
        )
        for token in tokens
        if token not in english_stopwords and len(token) > 1
    ]

    return " ".join(processed_tokens)
