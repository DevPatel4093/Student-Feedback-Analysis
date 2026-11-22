"""Model creation and prediction helpers."""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=42)


def make_naive_bayes() -> MultinomialNB:
    return MultinomialNB()


def predict_with_confidence(model, vectorizer, processed_text: str) -> tuple[str, float]:
    """Return predicted label and highest class probability as a percentage."""
    if not processed_text.strip():
        raise ValueError("Please enter feedback containing meaningful words.")
    features = vectorizer.transform([processed_text])
    label = str(model.predict(features)[0])
    confidence = float(model.predict_proba(features).max() * 100)
    return label, confidence
