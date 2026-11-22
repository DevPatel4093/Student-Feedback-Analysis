
"""Train and evaluate Student Feedback NLP classifiers.

Run from the repository root: python train.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib")
)

import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import dataset_summary, load_dataset
from src.evaluation import comparison_table, evaluate_predictions
from src.feature_extraction import make_tfidf_vectorizer
from src.model import make_logistic_regression, make_naive_bayes
from src.preprocessing import ensure_nltk_resources, preprocess_text
from src.visualization import (
    plot_confusion_matrix,
    plot_sentiment_distribution,
    plot_word_frequency,
    plot_wordclouds,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def main() -> None:
    # Load the dataset.
    ensure_nltk_resources(download=False)
    data = load_dataset()

    print(
        "Dataset summary:",
        json.dumps(dataset_summary(data), indent=2)
    )

    # Preprocess feedback.
    data = data.copy()
    data["Feedback"] = data["Feedback"].fillna("").astype(str)
    data["Processed_Feedback"] = data["Feedback"].map(preprocess_text)

    data = data[data["Processed_Feedback"].str.len() > 0].copy()

    # Check for identical processed feedback with conflicting labels.
    label_counts = data.groupby("Processed_Feedback")["Sentiment"].nunique()
    conflicting_texts = label_counts[label_counts > 1]

    if not conflicting_texts.empty:
        raise ValueError(
            f"Found {len(conflicting_texts)} feedback texts with conflicting "
            "sentiment labels. Resolve these labels before training."
        )

    # Remove duplicate feedback before splitting.
    before_deduplication = len(data)

    data = data.drop_duplicates(
        subset=["Processed_Feedback"]
    ).copy()

    print(f"Records before deduplication: {before_deduplication}")
    print(f"Unique feedback records: {len(data)}")
    print(
        "Duplicate feedback records removed:",
        before_deduplication - len(data)
    )

    # Split unique feedback into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(
        data["Processed_Feedback"],
        data["Sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=data["Sentiment"],
    )

    # Fit TF-IDF only on training feedback.
    vectorizer = make_tfidf_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    # Train and evaluate both classifiers.
    candidates = {
        "Logistic Regression": make_logistic_regression(),
        "Multinomial Naive Bayes": make_naive_bayes(),
    }

    results = {}
    trained_models = {}

    for name, classifier in candidates.items():
        classifier.fit(x_train_tfidf, y_train)

        predictions = classifier.predict(x_test_tfidf)

        results[name] = evaluate_predictions(
            y_test,
            predictions
        )

        trained_models[name] = classifier

    # Compare models and select the best according to the existing
    # comparison_table() implementation.
    comparison = comparison_table(results)

    print("\nModel comparison:\n", comparison.to_string(index=False))

    best_name = comparison.iloc[0]["Model"]
    best_metrics = results[best_name]

    print(f"\nBest model: {best_name}\n")
    print(best_metrics["report"])

    # Save the selected model and vectorizer.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        trained_models[best_name],
        MODELS_DIR / "sentiment_model.pkl"
    )

    joblib.dump(
        vectorizer,
        MODELS_DIR / "tfidf_vectorizer.pkl"
    )

    # Save evaluation results.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        OUTPUTS_DIR / "model_comparison.csv",
        index=False
    )

    with (OUTPUTS_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                key: value
                for key, value in best_metrics.items()
                if key not in {"confusion_matrix", "report"}
            },
            file,
            indent=2,
        )

    # Regenerate visualizations.
    plot_sentiment_distribution(
        data, OUTPUTS_DIR / "sentiment_distribution.png"
    )

    plot_word_frequency(
        data["Processed_Feedback"],
        OUTPUTS_DIR / "word_frequency.png"
    )

    plot_wordclouds(
        data, OUTPUTS_DIR / "wordcloud.png"
    )

    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        OUTPUTS_DIR / "confusion_matrix.png"
    )

    print(
        f"Saved {best_name} and TF-IDF vectorizer "
        f"to {MODELS_DIR}"
    )


if __name__ == "__main__":
    main()
