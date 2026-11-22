# Project Documentation

## Architecture

The project loads and validates the CSV, preprocesses only the `Feedback` column, splits the data into stratified train and test sets, fits the vectorizer on training text only, trains two classifiers, evaluates them, saves the best one, and exposes it through Streamlit. This order prevents test-data leakage. `Student_ID` is retained solely for student lookup.

## Core NLP concepts

**Natural Language Processing (NLP)** enables computers to work with human language. It is useful here because written feedback is unstructured and cannot be directly used by a standard machine-learning algorithm.

**Tokenization** separates text into words. **Stopwords** are common words such as “the” and “is” that often add little discrimination. **Stemming** cuts a word to a rough root; **lemmatization**, selected when WordNet resources are available, maps words toward dictionary base forms and is easier to explain in a viva. The pipeline lowercases text, removes non-alphabetic characters, tokenizes with a regular expression, removes English stopwords, and applies WordNet lemmatization. For offline labs where those NLTK data packages cannot be downloaded, it transparently falls back to Porter stemming, so the pipeline remains runnable.

**Bag of Words** counts term occurrences. It is a simple baseline representation and is demonstrated through `CountVectorizer`. **TF-IDF** weights terms by how informative they are across the corpus; common terms receive less emphasis. `TfidfVectorizer` is the primary representation.

**Sentiment analysis** identifies opinion polarity. **Text classification** trains a model from labelled examples to assign a class to unseen feedback. Logistic Regression is a strong, interpretable baseline for sparse TF-IDF data. Multinomial Naive Bayes is a fast probabilistic text-classification baseline; the project compares both rather than assuming one is better.

## Training and evaluation

The dataset is split 80% training and 20% testing with stratification and fixed `random_state=42`. The TF-IDF vocabulary is learned only from training records. Accuracy is the share of correct predictions. Precision measures whether predicted class labels are correct. Recall measures whether real class examples are found. F1-score balances precision and recall. The confusion matrix shows actual labels against predicted labels for each class.

The model-comparison CSV reports weighted precision, recall, and F1 along with accuracy. Class-wise values are printed as a scikit-learn classification report during training. Charts cover label balance, common processed words, three sentiment-specific word clouds, and the best-model confusion matrix.

## How to use

1. Install requirements and run `python train.py` from the project root.
2. Inspect generated artifacts under `outputs/` and the saved model/vectorizer under `models/`.
3. Start the application with `streamlit run app/app.py`.
4. Enter feedback to obtain a prediction, or search an existing Student_ID such as `S1001`.

## Academic limitation

The student feedback dataset used in this project is synthetically generated for academic demonstration of NLP preprocessing, feature extraction, and sentiment classification techniques. Its scores do not establish real-world educational sentiment performance. A real deployment needs consented, representative, carefully labelled data and fairness/privacy review.
