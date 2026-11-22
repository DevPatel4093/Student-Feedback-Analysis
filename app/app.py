"""Streamlit interface for Student Feedback Analysis Using NLP."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import dataset_summary, load_dataset, lookup_student
from src.model import predict_with_confidence
from src.preprocessing import preprocess_text

MODEL_PATH = ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"
OUTPUTS = ROOT / "outputs"
st.markdown(
    """
    <style>
        .stApp { background: #0e1626; color: #eef3fb; }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { color: #f7fbff; }
        [data-testid="stMetric"] { background: #17243a; border: 1px solid #2c4263; border-radius: 10px; padding: 16px; }
        [data-testid="stDataFrame"] { border: 1px solid #2c4263; border-radius: 8px; }
        button[data-baseweb="tab"] { color: #b9c7da; font-size: 0.94rem; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #68b5ff; border-bottom-color: #68b5ff; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.set_page_config(page_title="Student Feedback NLP", page_icon="🎓", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def get_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Run `python train.py` from the project root first.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)



def image_or_message(filename: str, label: str) -> None:
    path = OUTPUTS / filename

    if path.exists():
        # Keep charts compact instead of stretching them across the page.
        image_widths = {
            "sentiment_distribution.png": 700,
            "word_frequency.png": 650,
            "wordcloud.png": 800,
            "confusion_matrix.png": 550,
        }

        display_width = image_widths.get(filename, 650)

        st.image(
            str(path),
            caption=label,
            width=display_width,
        )
    else:
        st.info(
            f"{label} is not available yet. "
            "Run the training script to generate it."
        )



def main() -> None:
    st.title("🎓 Student Feedback Analysis Using NLP")
    st.caption("Academic demonstration using a synthetically generated student-feedback dataset.")
    try:
        data = get_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error)); return
    summary = dataset_summary(data)
    dashboard_tab, overview_tab, distribution_tab, analysis_tab, lookup_tab, performance_tab = st.tabs(
        ["Dashboard", "Dataset Overview", "Sentiment Distribution", "Analyze Feedback", "Student Lookup", "Model Performance"]
    )
    with dashboard_tab:
        st.header("Dashboard")
        columns = st.columns(4)
        columns[0].metric("Total feedback", summary["records"])
        for column, label in zip(columns[1:], ["Positive", "Neutral", "Negative"]):
            column.metric(label, summary["sentiment_distribution"].get(label, 0))
        st.info("Student_ID is used only for record lookup; it is never used as a model feature.")

    with overview_tab:
        st.header("Dataset Overview")
        st.write(f"**Rows:** {summary['records']}  |  **Columns:** {summary['columns']}  |  **Duplicate rows:** {summary['duplicate_records']}")
        st.subheader("Missing values"); st.json(summary["missing_values"])
        st.subheader("Sample records"); st.dataframe(data.head(10), use_container_width=True)
    with distribution_tab:
        st.header("Exploratory Text Analysis")
        image_or_message("sentiment_distribution.png", "Sentiment distribution")
        image_or_message("word_frequency.png", "Most frequent processed words")
        image_or_message("wordcloud.png", "Word clouds by sentiment")
    with analysis_tab:
        st.header("Analyze Feedback")
        feedback = st.text_area("Enter student feedback", placeholder="The practical sessions are very helpful and I understand the concepts well.")
        if st.button("Predict Sentiment", type="primary"):
            if not feedback.strip(): st.warning("Please enter feedback before requesting a prediction.")
            else:
                try:
                    model, vectorizer = get_artifacts()
                    label, confidence = predict_with_confidence(model, vectorizer, preprocess_text(feedback))
                    st.success(f"Predicted Sentiment: **{label}**"); st.write(f"Confidence: **{confidence:.1f}%**")
                except (RuntimeError, ValueError, FileNotFoundError) as error: st.error(str(error))
    with lookup_tab:
        st.header("Student Lookup")
        student_id = st.text_input("Enter Student ID", placeholder="S1001")
        if st.button("Find Student"):
            record = lookup_student(data, student_id)
            if record is None: st.error("Student ID not found. Please check the ID and try again.")
            else:
                st.write(f"**Student ID:** {record['Student_ID']}"); st.write(f"**Feedback:** {record['Feedback']}"); st.write(f"**Actual Sentiment:** {record['Sentiment']}")
                try:
                    model, vectorizer = get_artifacts()
                    predicted, confidence = predict_with_confidence(model, vectorizer, preprocess_text(record["Feedback"]))
                    st.write(f"**Predicted Sentiment:** {predicted} ({confidence:.1f}% confidence)")
                except (RuntimeError, ValueError, FileNotFoundError) as error: st.warning(str(error))
    with performance_tab:
        st.header("Model Performance")
        comparison_path, metrics_path = OUTPUTS / "model_comparison.csv", OUTPUTS / "metrics.json"
        if comparison_path.exists(): st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
        else: st.info("Run `python train.py` to calculate model performance.")
        image_or_message("confusion_matrix.png", "Best model confusion matrix")
        if metrics_path.exists(): st.json(json.loads(metrics_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
