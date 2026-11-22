"""Bag-of-Words and TF-IDF feature construction."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def make_bow_vectorizer(max_features: int = 3000) -> CountVectorizer:
    """Create a Bag-of-Words vectorizer for syllabus comparison."""
    return CountVectorizer(max_features=max_features, ngram_range=(1, 2))


def make_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create the primary TF-IDF feature extractor."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
