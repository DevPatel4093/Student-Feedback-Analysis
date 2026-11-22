"""Dataset loading, validation, and student-record lookup utilities."""
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Student_ID", "Feedback", "Sentiment"}


def project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[1]


def default_dataset_path() -> Path:
    return project_root() / "Student_Feedback_NLP.csv"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the student-feedback CSV without altering its source file."""
    dataset_path = Path(path) if path else default_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place Student_Feedback_NLP.csv in the project root."
        )
    data = pd.read_csv(dataset_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Dataset is empty; training cannot continue.")
    return data


def dataset_summary(data: pd.DataFrame) -> dict:
    """Return inspection values used by the training script and Streamlit dashboard."""
    return {
        "records": len(data),
        "columns": len(data.columns),
        "missing_values": data.isna().sum().to_dict(),
        "duplicate_records": int(data.duplicated().sum()),
        "sentiment_distribution": data["Sentiment"].value_counts().to_dict(),
    }


def lookup_student(data: pd.DataFrame, student_id: str) -> pd.Series | None:
    """Find one student record using a case-insensitive Student_ID lookup."""
    normalized_id = str(student_id).strip().upper()
    records = data[data["Student_ID"].astype(str).str.upper() == normalized_id]
    return None if records.empty else records.iloc[0]
