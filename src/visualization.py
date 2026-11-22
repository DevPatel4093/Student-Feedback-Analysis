"""Visualization functions used by training and the notebook."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from src.evaluation import LABELS


def _prepare_output(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_sentiment_distribution(data, output_path: str | Path) -> None:
    target = _prepare_output(output_path)
    plt.figure(figsize=(7, 3.5))
    ax = sns.countplot(data=data, x="Sentiment", order=LABELS, hue="Sentiment", palette="Set2", legend=False)
    ax.set(title="Student Feedback Sentiment Distribution", xlabel="Sentiment", ylabel="Feedback records")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(target, dpi=130, bbox_inches="tight")
    plt.close()

def plot_word_frequency(processed_texts, output_path: str | Path, top_n: int = 15) -> None:
    target = _prepare_output(output_path)
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(processed_texts)
    frequencies = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    top = sorted(zip(words, frequencies), key=lambda item: item[1], reverse=True)[:top_n]
    labels, values = zip(*reversed(top))
    plt.figure(figsize=(7, 4))
    plt.barh(labels, values, color="#4C78A8")
    plt.title("Most Frequent Processed Words")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(target, dpi=130, bbox_inches="tight")
    plt.close()

def plot_wordclouds(data, output_path: str | Path) -> None:
    """Save three sentiment-specific word clouds in a single clear figure."""
    target = _prepare_output(output_path)
    colors = {"Positive": "Greens", "Neutral": "Blues", "Negative": "Reds"}
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, sentiment in zip(axes, LABELS):
        text = " ".join(data.loc[data["Sentiment"] == sentiment, "Processed_Feedback"])
        cloud = WordCloud(width=700, height=400, background_color="white", colormap=colors[sentiment]).generate(text or "no data")
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_title(f"{sentiment} feedback")
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(target, dpi=130, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(matrix, output_path: str | Path) -> None:
    target = _prepare_output(output_path)
    plt.figure(figsize=(5, 3.8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted sentiment"); plt.ylabel("Actual sentiment")
    plt.tight_layout()
    plt.savefig(target, dpi=130, bbox_inches="tight")
    plt.close()
