"""Classification metrics and model-comparison helpers."""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

LABELS = ["Positive", "Neutral", "Negative"]


def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculate overall and class-wise evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Format comparable weighted metrics for each candidate model."""
    return pd.DataFrame(
        [
            {"Model": name, **{metric.title(): values[metric] for metric in ("accuracy", "precision", "recall", "f1")}}
            for name, values in results.items()
        ]
    ).sort_values("F1", ascending=False).reset_index(drop=True)
