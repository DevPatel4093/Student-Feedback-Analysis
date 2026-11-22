# Student Feedback Analysis Using Natural Language Processing

An academic NLP project that classifies textual student feedback as **Positive**, **Neutral**, or **Negative**. It includes a modular preprocessing pipeline, TF-IDF and Bag-of-Words features, classifier comparison, visual analysis, automated tests, and a Streamlit dashboard.

> **Synthetic-data disclaimer:** The student feedback dataset used in this project is synthetically generated for academic demonstration of NLP preprocessing, feature extraction, and sentiment classification techniques. It must not be treated as authentic student opinion data or evidence of real-world model performance.

## Problem and objectives

Reading thousands of open-ended comments manually is difficult. This project turns feedback text into usable sentiment categories while demonstrating tokenization, stopword removal, English lemmatization, vectorization, text classification, evaluation, and visualization.

`Feedback` is the input feature and `Sentiment` is the target. `Student_ID` is used only to find a record in the app; it is never a machine-learning feature.

## NLP pipeline

`Raw feedback → lowercase/cleaning → tokenization → stopword removal → lemmatization → TF-IDF → classifier → sentiment + evaluation`

Bag of Words (`CountVectorizer`) is included for syllabus demonstration. TF-IDF (`TfidfVectorizer`) is the final model representation. The comparison models are Logistic Regression and Multinomial Naive Bayes, using an 80/20 stratified split with `random_state=42`.

## Project structure

```text
Student_Feedback_NLP.csv       Original 5,000-row synthetic dataset
src/                           Modular loading, NLP, ML, evaluation and plots
train.py                       Trains models and creates saved artifacts/plots
app/app.py                     Streamlit user interface
notebooks/NLP_Analysis.ipynb   Academic walkthrough notebook
tests/test_pipeline.py         Core workflow tests
models/                        Saved best model and TF-IDF vectorizer (after training)
outputs/                       Metrics, charts and confusion matrix (after training)
```

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app/app.py
```

The first training run attempts to download NLTK `stopwords`, `wordnet`, and `omw-1.4` if needed. When an offline lab network blocks this, the pipeline safely uses scikit-learn stopwords plus NLTK Porter stemming; with the resources installed it uses the preferred WordNet lemmatization.

## Outputs and application

Training saves `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`, plus sentiment distribution, word frequency, word-cloud, confusion-matrix, comparison CSV, and metrics JSON files under `outputs/`. The app provides Dashboard, Dataset Overview, Sentiment Distribution, Analyze Feedback, Student Lookup, and Model Performance pages. It does not retrain for individual predictions.

## Limitations and future scope

The feedback is synthetic, likely repetitive, English-only, and limited to three broad sentiment labels. Results should therefore be presented as a demonstration, not a production benchmark. Natural extensions include genuine consented feedback, aspect-based sentiment, multilingual data, topics, and transformer models.

## Conclusion

This project provides an explainable end-to-end NLP text-classification workflow suitable for an academic demonstration and viva.
