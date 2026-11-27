# Student Feedback Analysis 📊 Using Natural Language Processing (NLP)

An NLP-based application that analyzes student feedback and automatically classifies it into **Positive, Negative, or Neutral** sentiment categories using Machine Learning algorithms. It includes a modular preprocessing pipeline, TF-IDF and Bag-of-Words features, classifier comparison, visual analysis, automated tests, and a Streamlit dashboard.

## 📌 Project Overview

Student feedback plays an important role in evaluating teaching quality, course effectiveness, and the overall learning experience. However, manually analyzing a large number of feedback responses can be time-consuming and difficult.

> **Synthetic-data disclaimer:** The student feedback dataset used in this project is synthetically generated for academic demonstration of NLP preprocessing, feature extraction, and sentiment classification techniques. It must not be treated as authentic student opinion data or evidence of real-world model performance.

This project aims to automate the feedback analysis process using Natural Language Processing (NLP) and Machine Learning techniques.

The system processes textual feedback, extracts meaningful features, and predicts its sentiment using two classification algorithms:

- Logistic Regression
- Multinomial Naive Bayes

A user-friendly Streamlit web application allows users to enter student feedback and receive real-time sentiment predictions along with the model's confidence score.

## Problem

Reading thousands of open-ended comments manually is difficult. This project turns feedback text into usable sentiment categories while demonstrating tokenization, stopword removal, English lemmatization, vectorization, text classification, evaluation, and visualization.

`Feedback` is the input feature and `Sentiment` is the target. `Student_ID` is used only to find a record in the app; it is never a machine-learning feature.

## 🎯 Project Objectives

- Automate the analysis of student feedback.
- Classify feedback into Positive, Negative, and Neutral categories.
- Apply NLP techniques to clean and process textual data.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple Machine Learning classification algorithms.
- Evaluate model performance using standard classification metrics.
- Develop an interactive web application for real-time sentiment prediction.

## NLP pipeline

`Raw feedback → lowercase/cleaning → tokenization → stopword removal → lemmatization → TF-IDF → classifier → sentiment + evaluation`

Bag of Words (`CountVectorizer`) is included for syllabus demonstration. TF-IDF (`TfidfVectorizer`) is the final model representation. The comparison models are Logistic Regression and Multinomial Naive Bayes, using an 80/20 stratified split with `random_state=42`.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Natural Language Processing (NLP) | Text processing and analysis |
| Pandas | Dataset handling and manipulation |
| NumPy | Numerical operations |
| NLTK | Tokenization, stopword removal, and stemming/lemmatization |
| Scikit-learn | Machine Learning and model evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Sentiment classification |
| Multinomial Naive Bayes | Sentiment classification |
| Matplotlib | Data visualization |
| WordCloud | Visualization of frequently occurring words |
| Streamlit | Interactive web application |
| Joblib | Saving and loading trained models |

## 🧠 Project Workflow

The project follows these major steps:

1. **Data Collection:** Load the student feedback dataset containing textual feedback and corresponding sentiment labels.

2. **Data Preprocessing:** Handle missing values, convert text into lowercase, remove unnecessary characters, tokenize sentences, remove stopwords, and apply stemming or lemmatization.

3. **Duplicate Handling:** Identify and remove duplicate processed feedback records to reduce repeated entries during model evaluation.

4. **Feature Extraction:** Apply TF-IDF (Term Frequency-Inverse Document Frequency) to convert textual feedback into numerical feature vectors.

5. **Model Training:** Train Logistic Regression and Multinomial Naive Bayes classifiers using the processed dataset.

6. **Model Evaluation:** Compare the trained models using accuracy, precision, recall, and F1-score.

7. **Model Saving:** Save the trained classifier and TF-IDF vectorizer using Joblib.

8. **Prediction:** Use the saved model to classify newly entered student feedback through the Streamlit application.

## ⚙️ Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is a supervised Machine Learning classification algorithm used to predict the sentiment category of student feedback.

It learns the relationship between TF-IDF features and sentiment labels and predicts whether the given feedback is Positive, Negative, or Neutral.

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks.

It calculates the probability of a feedback statement belonging to each sentiment category based on the extracted text features.

### 3. TF-IDF Feature Extraction

TF-IDF assigns numerical importance to words based on their frequency in a document and their distribution across the dataset.

It helps convert textual feedback into numerical representations that Machine Learning algorithms can process.

## 📊 Model Evaluation

The project compares both classification algorithms using the following evaluation metrics:

- **Accuracy:** Measures the proportion of correctly classified feedback records.
- **Precision:** Measures how many predicted instances of a sentiment category are actually correct.
- **Recall:** Measures how many actual instances of a sentiment category are correctly identified.
- **F1-Score:** Represents the harmonic mean of precision and recall.

### Current Experimental Results

The current training run produced the following results after duplicate processed feedback was removed:

| Evaluation Details | Result |
|---|---:|
| Original dataset records | 5,000 |
| Unique feedback records | 882 |
| Duplicate records removed | 4,118 |
| Test dataset records | 177 |

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 100% | 100% | 100% | 100% |
| Multinomial Naive Bayes | 100% | 100% | 100% | 100% |

**Note:** These are the results from the current random train-test split, not a guarantee of performance on unseen real-world student feedback. The high scores require further validation using independently collected and human-labelled feedback. Duplicate removal does not establish that the dataset is representative of genuine student responses.

## 🌐 Web Application

The project includes an interactive Streamlit application that allows users to analyze student feedback without running the training process manually.

### Application Features

- Simple and user-friendly interface.
- Text input for student feedback.
- Real-time sentiment prediction.
- Classification into Positive, Negative, and Neutral.
- Display of the predicted sentiment.
- Confidence score associated with the prediction.
- Uses previously trained and saved Machine Learning models.

### Example Feedback

| Student Feedback | Expected Sentiment |
|---|---|
| The professor explains every concept clearly, and the lectures are very helpful. | Positive |
| The teaching is poor, and the lectures are not helpful. | Negative |
| The course contains five units and three assignments. | Neutral |

*These are expected human classifications; actual predictions depend on the trained model.*

## 📁 Project Structure

```text
Student-Feedback-NLP/
│
├── app/
│   ├── app.py
│   └── models/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── notebooks/
│   └── NLP_Analysis.ipynb
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/
│   ├── model_comparison.csv
│   ├── metrics.json
│   ├── sentiment_distribution.png
│   ├── word_frequency.png
│   ├── wordcloud.png
│   └── confusion_matrix.png
│
├── tests/
│
├── Student_Feedback_NLP.csv
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

- Python 3.10 or later
- pip package manager
- Visual Studio Code or another Python IDE

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Machine Learning Models

Run the following command from the project root directory:

```bash
python train.py
```

This command processes the dataset, trains both classification algorithms, compares their performance, and saves the selected model and TF-IDF vectorizer.

### Step 3: Launch the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open in your default web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

The first training run attempts to download NLTK `stopwords`, `wordnet`, and `omw-1.4` if needed. When an offline lab network blocks this, the pipeline safely uses scikit-learn stopwords plus NLTK Porter stemming; with the resources installed it uses the preferred WordNet lemmatization.

## Outputs and application

Training saves `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`, plus sentiment distribution, word frequency, word-cloud, confusion-matrix, comparison CSV, and metrics JSON files under `outputs/`. The app provides Dashboard, Dataset Overview, Sentiment Distribution, Analyze Feedback, Student Lookup, and Model Performance pages. It does not retrain for individual predictions.

## 📈 Generated Outputs

The training process generates the following evaluation and visualization files:

| Output File | Description |
|---|---|
| model_comparison.csv | Comparison of classification algorithms |
| metrics.json | Evaluation metrics of the selected model |
| sentiment_distribution.png | Distribution of sentiment categories |
| word_frequency.png | Most frequently occurring words |
| wordcloud.png | Word cloud visualization |
| confusion_matrix.png | Classification performance visualization |

## ⚠️ Limitations

- The quality of predictions depends on the quality and diversity of the training dataset.
- Repeated or similar feedback can influence evaluation results.
- Negation, sarcasm, informal language, and mixed sentiment may affect classification.
- Predictions on genuine student feedback require independent validation.
- The confidence score represents the model's estimated prediction probability or score and does not guarantee correctness.
- The current evaluation results should not be interpreted as guaranteed real-world accuracy.

## 🔮 Future Scope

The project can be further improved through:

- Integration of larger and more diverse student feedback datasets.
- Validation using independently collected and human-labelled feedback.
- Improved handling of negation, sarcasm, and informal expressions.
- Experimentation with advanced NLP techniques and deep learning models.
- Addition of feedback analytics and interactive dashboards.
- Department-wise and subject-wise sentiment analysis.
- Generation of automated reports to support academic decision-making.

## 🎓 Applications

- Educational institutions and universities.
- Student feedback management systems.
- Teaching quality assessment.
- Course and curriculum evaluation.
- Academic administration and decision-making.
- Analysis of student satisfaction and learning experiences.

**If you find this project useful, consider giving the repository a ⭐ on GitHub!**
