"""Reliable tests for the core Student Feedback NLP workflow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_dataset, lookup_student
from src.feature_extraction import make_tfidf_vectorizer
from src.model import make_logistic_regression
from src.preprocessing import ensure_nltk_resources, preprocess_text


def test_dataset_loading():
    data = load_dataset()
    assert {"Student_ID", "Feedback", "Sentiment"}.issubset(data.columns)
    assert len(data) == 5000


def test_preprocessing_and_empty_text():
    ensure_nltk_resources(download=True)
    assert preprocess_text("I Really Enjoy the Course!!!")
    assert preprocess_text(None) == ""
    assert preprocess_text("   !!! ") == ""


def test_tfidf_and_model_prediction():
    documents = ["helpful practical session", "difficult confusing topic", "average normal course"]
    labels = ["Positive", "Negative", "Neutral"]
    vectorizer = make_tfidf_vectorizer(); matrix = vectorizer.fit_transform(documents)
    assert matrix.shape[0] == 3
    model = make_logistic_regression().fit(matrix, labels)
    assert model.predict(vectorizer.transform(["helpful session"]))[0] == "Positive"


def test_student_lookup():
    data = load_dataset()
    assert lookup_student(data, "s1001")["Student_ID"] == "S1001"
    assert lookup_student(data, "missing-id") is None
