"""
End-to-end sentiment analysis and ethical-dimension mining for tourism reviews
of Kuburan Batu Lemo (Toraja).

Features
--------
- CSV input with columns: review_text, rating (optional), date (optional)
- Indonesian + English preprocessing
- TF-IDF + Multinomial Naive Bayes baseline
- Lexicon-based sentiment (VADER for English, rule-based lexicon for Indonesian)
- IndoBERT fine-tuning and 5-fold cross-validation
- Ethical dimension classification via keyword rules
- Evaluation, visualization, CSV export, and auto-generated insights

Recommended CSV columns
-----------------------
- review_text          : required
- rating               : optional; used to derive weak sentiment labels if no
                         sentiment_label is available
- sentiment_label      : optional; preferred supervised label column
- date                 : optional

Usage example
-------------
python sentiment_analysis_kuburan_batu.py \
    --input reviews.csv \
    --output-dir outputs \
    --bert-model indobenchmark/indobert-base-p1
"""

from __future__ import annotations

import argparse
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from wordcloud import WordCloud

if TYPE_CHECKING:  # pragma: no cover
    from transformers import Trainer


RANDOM_STATE = 42
TEXT_COLUMN = "review_text"
LABEL_COLUMN = "sentiment_label"
SUPPORTED_LABELS = ["negative", "neutral", "positive"]
METHOD_COLUMN_MAP = {
    "VADER": "vader_sentiment",
    "Naive Bayes": "naive_bayes_sentiment",
    "IndoBERT": "indobert_sentiment",
}

ETHICAL_KEYWORDS = {
    "Respect": ["respect", "hormat", "sopan", "etika"],
    "Behavior": ["behavior", "perilaku", "attitude"],
    "Cultural Value": ["culture", "budaya", "tradisi"],
    "Commercialization": ["ticket", "harga", "mahal", "komersial"],
}

ID_POSITIVE_WORDS = {
    "bagus": 2.0,
    "indah": 2.5,
    "menarik": 1.8,
    "luar biasa": 2.7,
    "rekomendasi": 2.2,
    "bersih": 1.5,
    "unik": 2.0,
    "baik": 1.5,
    "keren": 1.8,
    "suka": 1.8,
    "nyaman": 1.6,
    "cantik": 2.0,
    "hebat": 1.8,
}

ID_NEGATIVE_WORDS = {
    "buruk": -2.2,
    "jelek": -2.0,
    "kotor": -1.7,
    "mahal": -1.5,
    "ramai": -0.8,
    "kecewa": -2.5,
    "kurang": -1.4,
    "parah": -2.2,
    "rusak": -1.8,
    "tidak sopan": -2.0,
    "tidak nyaman": -2.0,
    "susah": -1.5,
    "buram": -1.0,
}

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "tidak",
    "tak",
    "bukan",
    "jangan",
    "kurang",
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_nltk_resources() -> None:
    resources: List[Tuple[str, List[str]]] = [
        ("punkt", ["tokenizers/punkt"]),
        ("punkt_tab", ["tokenizers/punkt_tab"]),
        ("stopwords", ["corpora/stopwords"]),
        ("vader_lexicon", ["sentiment/vader_lexicon", "sentiment/vader_lexicon.zip"]),
    ]
    for package, resource_paths in resources:
        found = False
        for resource_path in resource_paths:
            try:
                nltk.data.find(resource_path)
                found = True
                break
            except LookupError:
                continue
        if not found:
            nltk.download(package, quiet=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str, id_stopwords: set[str], en_stopwords: set[str]) -> str:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return "unknown"
    id_hits = sum(token in id_stopwords for token in tokens)
    en_hits = sum(token in en_stopwords for token in tokens)
    common_id_markers = {"yang", "dan", "di", "ke", "tidak", "dengan", "untuk", "wisata", "budaya"}
    common_en_markers = {"the", "and", "was", "were", "very", "beautiful", "place", "ticket"}
    id_hits += sum(token in common_id_markers for token in tokens)
    en_hits += sum(token in common_en_markers for token in tokens)
    if id_hits > en_hits:
        return "id"
    if en_hits > id_hits:
        return "en"
    return "mixed"


def build_stopword_sets() -> Tuple[set[str], set[str]]:
    id_factory = StopWordRemoverFactory()
    id_stopwords = set(id_factory.get_stop_words())
    en_stopwords = set(stopwords.words("english"))
    return id_stopwords, en_stopwords


def preprocess_text(
    text: str,
    stemmer,
    id_stopwords: set[str],
    en_stopwords: set[str],
) -> str:
    """
    Lowercase, remove punctuation, tokenize, remove Indonesian and English
    stopwords, then apply Sastrawi stemming.
    """
    if pd.isna(text):
        return ""

    raw_text = str(text)
    text = raw_text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = normalize_whitespace(text)

    try:
        tokens = word_tokenize(text)
    except LookupError:
        ensure_nltk_resources()
        tokens = word_tokenize(text)

    cleaned_tokens = [
        token
        for token in tokens
        if token.isalpha() and token not in id_stopwords and token not in en_stopwords
    ]
    lang = detect_language(raw_text, id_stopwords=id_stopwords, en_stopwords=en_stopwords)
    if lang == "en":
        return " ".join(cleaned_tokens)
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    return " ".join(stemmed_tokens)


def classify_ethics(text: str) -> str:
    text_lower = str(text).lower()
    matched_categories = []
    for category, keywords in ETHICAL_KEYWORDS.items():
        if any(
            (keyword in text_lower) if (" " in keyword) else (re.search(rf"\b{re.escape(keyword)}\b", text_lower) is not None)
            for keyword in keywords
        ):
            matched_categories.append(category)
    return ", ".join(matched_categories) if matched_categories else "General"


def derive_label_from_rating(rating: float) -> str:
    if pd.isna(rating):
        return "neutral"
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def prepare_labels(df: pd.DataFrame) -> pd.Series:
    if LABEL_COLUMN in df.columns:
        labels = df[LABEL_COLUMN].astype(str).str.lower().str.strip()
    elif "rating" in df.columns:
        labels = df["rating"].apply(derive_label_from_rating)
    else:
        warnings.warn(
            "Kolom sentiment_label tidak ditemukan dan rating tidak tersedia. "
            "Label lemah akan dibangun dari metode lexicon sehingga evaluasi "
            "bersifat pseudo-supervised."
        )
        labels = pd.Series(["neutral"] * len(df), index=df.index)

    labels = labels.replace(
        {
            "negatif": "negative",
            "netral": "neutral",
            "positif": "positive",
            "neg": "negative",
            "neu": "neutral",
            "pos": "positive",
        }
    )
    return labels.where(labels.isin(SUPPORTED_LABELS), "neutral")


def _score_indonesian_lexicon(text: str, stemmer) -> float:
    text_lower = str(text).lower()
    score = 0.0

    for phrase, value in ID_POSITIVE_WORDS.items():
        if " " in phrase and phrase in text_lower:
            score += value
    for phrase, value in ID_NEGATIVE_WORDS.items():
        if " " in phrase and phrase in text_lower:
            score += value

    tokens = text_lower.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    for idx, token in enumerate(stemmed_tokens):
        token_score = ID_POSITIVE_WORDS.get(token, 0.0) + ID_NEGATIVE_WORDS.get(token, 0.0)
        if idx > 0 and stemmed_tokens[idx - 1] in NEGATION_WORDS:
            token_score *= -1
        score += token_score

    if not stemmed_tokens:
        return 0.0
    return score / max(len(stemmed_tokens), 1)


def sentiment_lexicon(
    text: str,
    vader_analyzer: SentimentIntensityAnalyzer,
    stemmer,
    id_stopwords: set[str],
    en_stopwords: set[str],
) -> str:
    lang = detect_language(text, id_stopwords=id_stopwords, en_stopwords=en_stopwords)
    text = str(text)

    if lang == "en":
        compound = vader_analyzer.polarity_scores(text)["compound"]
    elif lang == "id":
        compound = _score_indonesian_lexicon(text, stemmer=stemmer)
    else:
        vader_score = vader_analyzer.polarity_scores(text)["compound"]
        id_score = _score_indonesian_lexicon(text, stemmer=stemmer)
        compound = (vader_score + id_score) / 2

    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def build_nb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        (
                            "tfidf",
                            TfidfVectorizer(max_features=5000, sublinear_tf=True),
                            "processed_text",
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", MultinomialNB()),
        ]
    )


def train_ml_model(
    df: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
) -> Tuple[Pipeline, Dict[str, float], pd.Series, Dict[str, object]]:
    """
    Train and tune MultinomialNB using GridSearchCV, then produce cross-validated
    predictions and aggregate metrics.
    """
    pipeline = build_nb_pipeline()
    grid = {
        "features__tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__tfidf__min_df": [1, 2],
        "model__alpha": [0.1, 0.5, 1.0],
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(df[["processed_text"]], labels)
    best_model = search.best_estimator_
    cv_predictions = cross_val_predict(
        best_model,
        df[["processed_text"]],
        labels,
        cv=cv,
        n_jobs=-1,
        method="predict",
    )
    metrics = compute_metrics(labels, cv_predictions)
    metadata = {
        "best_params": search.best_params_,
        "classification_report": classification_report(labels, cv_predictions, zero_division=0),
    }
    best_model.fit(df[["processed_text"]], labels)
    return best_model, metrics, pd.Series(cv_predictions, index=df.index), metadata


def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def validate_and_adjust_folds(labels: pd.Series, requested_folds: int) -> int:
    class_counts = labels.value_counts()
    if class_counts.nunique() == 0 or len(class_counts) < 2:
        raise ValueError(
            "Model Naive Bayes dan IndoBERT membutuhkan minimal 2 kelas sentimen. "
            "Tambahkan variasi label pada kolom sentiment_label atau rating."
        )
    max_valid_folds = int(class_counts.min())
    adjusted_folds = max(2, min(requested_folds, max_valid_folds))
    if adjusted_folds < requested_folds:
        warnings.warn(
            f"Jumlah fold diturunkan dari {requested_folds} menjadi {adjusted_folds} "
            "karena distribusi kelas dataset terlalu kecil."
        )
    return adjusted_folds


class ReviewDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


@dataclass
class BertConfig:
    model_name: str = "indobenchmark/indobert-base-p1"
    learning_rates: Tuple[float, ...] = (2e-5, 3e-5)
    epochs: Tuple[int, ...] = (2, 3)
    batch_size: int = 8
    weight_decay: float = 0.01
    max_length: int = 256


def _transformers_compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _train_bert_once(
    train_texts: List[str],
    train_labels: List[int],
    valid_texts: List[str],
    valid_labels: List[int],
    label_encoder: LabelEncoder,
    config: BertConfig,
    run_name: str,
    output_dir: Path,
    learning_rate: float,
    epochs: int,
) -> Tuple["Trainer", Dict[str, float]]:
    # Avoid importing TF/Keras; we only use PyTorch backend.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)},
    )

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_length=config.max_length)
    valid_dataset = ReviewDataset(valid_texts, valid_labels, tokenizer, max_length=config.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / run_name),
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=epochs,
        weight_decay=config.weight_decay,
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_STATE,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=_transformers_compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    cleaned_metrics = {
        "accuracy": eval_metrics.get("eval_accuracy", 0.0),
        "precision": eval_metrics.get("eval_precision", 0.0),
        "recall": eval_metrics.get("eval_recall", 0.0),
        "f1": eval_metrics.get("eval_f1", 0.0),
    }
    return trainer, cleaned_metrics


def cross_validate_indobert(
    texts: List[str],
    labels: pd.Series,
    config: BertConfig,
    output_dir: Path,
    n_splits: int = 5,
) -> Tuple[Dict[str, float], pd.Series, Dict[str, object]]:
    """
    Perform simple hyperparameter tuning on the first split, then 5-fold
    cross-validation with the best configuration.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(texts, y_encoded))

    tune_train_idx, tune_valid_idx = splits[0]
    best_setting = None
    best_score = -1.0
    tuning_log = []

    for learning_rate in config.learning_rates:
        for epochs in config.epochs:
            trainer, metrics = _train_bert_once(
                [texts[i] for i in tune_train_idx],
                [int(y_encoded[i]) for i in tune_train_idx],
                [texts[i] for i in tune_valid_idx],
                [int(y_encoded[i]) for i in tune_valid_idx],
                label_encoder=label_encoder,
                config=config,
                run_name=f"tuning_lr_{learning_rate}_ep_{epochs}",
                output_dir=output_dir,
                learning_rate=learning_rate,
                epochs=epochs,
            )
            tuning_log.append({"learning_rate": learning_rate, "epochs": epochs, **metrics})
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_setting = {"learning_rate": learning_rate, "epochs": epochs}
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_setting is None:
        raise RuntimeError("Gagal menemukan hyperparameter terbaik untuk IndoBERT.")

    fold_predictions = np.empty(len(texts), dtype=object)
    fold_metrics = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = [int(y_encoded[i]) for i in train_idx]
        test_labels = [int(y_encoded[i]) for i in test_idx]

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(label_encoder.classes_),
            id2label={i: label for i, label in enumerate(label_encoder.classes_)},
            label2id={label: i for i, label in enumerate(label_encoder.classes_)},
        )
        train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_length=config.max_length)
        test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, max_length=config.max_length)

        training_args = TrainingArguments(
            output_dir=str(output_dir / f"bert_fold_{fold_number}"),
            evaluation_strategy="epoch",
            save_strategy="no",
            learning_rate=best_setting["learning_rate"],
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=best_setting["epochs"],
            weight_decay=config.weight_decay,
            logging_strategy="epoch",
            report_to="none",
            seed=RANDOM_STATE,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=_transformers_compute_metrics,
        )
        trainer.train()
        predictions = trainer.predict(test_dataset)
        pred_ids = np.argmax(predictions.predictions, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_ids)
        fold_predictions[test_idx] = pred_labels
        fold_metrics.append(compute_metrics(label_encoder.inverse_transform(test_labels), pred_labels))

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregated_metrics = {
        metric: float(np.mean([fold_metric[metric] for fold_metric in fold_metrics]))
        for metric in ["accuracy", "precision", "recall", "f1"]
    }
    metadata = {
        "best_params": best_setting,
        "tuning_log": pd.DataFrame(tuning_log),
        "label_encoder": label_encoder,
    }
    return aggregated_metrics, pd.Series(fold_predictions), metadata


def fit_final_indobert(
    texts: List[str],
    labels: pd.Series,
    config: BertConfig,
    best_params: Dict[str, float],
    output_dir: Path,
) -> "Trainer":
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)},
    )
    dataset = ReviewDataset(texts, list(map(int, y_encoded)), tokenizer, max_length=config.max_length)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "final_indobert"),
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=best_params["epochs"],
        weight_decay=config.weight_decay,
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_STATE,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=_transformers_compute_metrics,
    )
    trainer.train()
    trainer.model.config.id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    trainer.model.config.label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    trainer.save_model(str(output_dir / "final_indobert"))
    tokenizer.save_pretrained(str(output_dir / "final_indobert"))
    return trainer


def predict_with_indobert(trainer: Trainer, texts: List[str], max_length: int = 256) -> List[str]:
    tokenizer = trainer.tokenizer
    id2label = trainer.model.config.id2label
    dummy_labels = [0] * len(texts)
    dataset = ReviewDataset(texts, dummy_labels, tokenizer, max_length=max_length)
    predictions = trainer.predict(dataset)
    pred_ids = np.argmax(predictions.predictions, axis=1)
    return [id2label[int(idx)] for idx in pred_ids]


def aggregate_lexicon_predictions(
    df: pd.DataFrame,
    vader_analyzer: SentimentIntensityAnalyzer,
    stemmer,
    id_stopwords: set[str],
    en_stopwords: set[str],
) -> pd.Series:
    return df[TEXT_COLUMN].apply(
        lambda text: sentiment_lexicon(text, vader_analyzer, stemmer, id_stopwords, en_stopwords)
    )


def choose_best_method(metrics_table: pd.DataFrame) -> str:
    ranked = metrics_table.sort_values(by=["f1", "accuracy"], ascending=False)
    return str(ranked.iloc[0]["method"])


def generate_auto_insights(
    df: pd.DataFrame,
    metrics_table: pd.DataFrame,
    best_method_name: str,
    best_method_column: str,
) -> List[str]:
    insights = []
    sentiment_counts = df[best_method_column].value_counts(normalize=True).mul(100).round(1)
    dominant_sentiment = sentiment_counts.idxmax()
    insights.append(
        f"Metode terbaik adalah {best_method_name} dengan weighted F1 {metrics_table.set_index('method').loc[best_method_name, 'f1']:.3f}."
    )
    insights.append(
        f"Sentimen dominan menurut {best_method_name} adalah {dominant_sentiment} sebesar {sentiment_counts.iloc[0]:.1f}%."
    )

    ethics_counts = df["ethical_category"].value_counts()
    top_ethic = ethics_counts.idxmax()
    insights.append(
        f"Dimensi etika yang paling sering muncul adalah {top_ethic} sebanyak {int(ethics_counts.iloc[0])} ulasan."
    )

    ethics_sentiment = (
        df.groupby(["ethical_category", best_method_column])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if not ethics_sentiment.empty:
        top_row = ethics_sentiment.iloc[0]
        insights.append(
            f"Kombinasi paling dominan adalah kategori {top_row['ethical_category']} dengan sentimen {top_row[best_method_column]} ({int(top_row['count'])} ulasan)."
        )
    return insights


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    output_path: Path,
    show_plots: bool = False,
) -> None:
    labels = SUPPORTED_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()


def visualize_results(
    df: pd.DataFrame,
    metrics_table: pd.DataFrame,
    best_method_name: str,
    best_method_column: str,
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    sentiment_col = best_method_column
    sentiment_counts = df[sentiment_col].value_counts()
    ethics_counts = df["ethical_category"].value_counts()

    plt.figure(figsize=(7, 7))
    plt.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2", len(sentiment_counts)),
    )
    plt.title(f"Distribusi Sentimen ({best_method_name})")
    plt.tight_layout()
    plt.savefig(output_dir / "pie_sentiment_distribution.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=ethics_counts.index, y=ethics_counts.values, palette="crest")
    plt.title("Distribusi Dimensi Etika")
    plt.xlabel("Ethical Category")
    plt.ylabel("Jumlah Review")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_ethics_distribution.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    wordcloud_text = " ".join(df["processed_text"].dropna().astype(str))
    if wordcloud_text.strip():
        wc = WordCloud(width=1400, height=700, background_color="white", colormap="viridis").generate(wordcloud_text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud Review")
        plt.tight_layout()
        plt.savefig(output_dir / "wordcloud_reviews.png", dpi=300)
        if show_plots:
            plt.show()
        plt.close()

    ethics_sentiment = (
        df.groupby(["ethical_category", sentiment_col]).size().reset_index(name="count")
    )
    pivot_table = ethics_sentiment.pivot(
        index="ethical_category", columns=sentiment_col, values="count"
    ).fillna(0)
    pivot_table = pivot_table.reindex(columns=SUPPORTED_LABELS, fill_value=0)
    pivot_table.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="Spectral")
    plt.title(f"Distribusi Sentimen per Dimensi Etika ({best_method_name})")
    plt.xlabel("Ethical Category")
    plt.ylabel("Jumlah Review")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "stacked_sentiment_by_ethics.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    comparison = metrics_table.melt(id_vars="method", value_vars=["accuracy", "precision", "recall", "f1"])
    sns.barplot(data=comparison, x="metric", y="value", hue="method", palette="tab10")
    plt.title("Perbandingan Kinerja Metode")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close()


def load_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = {TEXT_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset harus memiliki kolom: {sorted(required_columns)}")
    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    return df


def maybe_bootstrap_labels_from_lexicon(
    df: pd.DataFrame,
    labels: pd.Series,
    vader_analyzer: SentimentIntensityAnalyzer,
    stemmer,
    id_stopwords: set[str],
    en_stopwords: set[str],
) -> pd.Series:
    if LABEL_COLUMN not in df.columns and "rating" not in df.columns:
        return aggregate_lexicon_predictions(df, vader_analyzer, stemmer, id_stopwords, en_stopwords)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis sentimen ulasan Kuburan Batu Lemo.")
    parser.add_argument("--input", required=True, help="Path ke dataset CSV input.")
    parser.add_argument("--output-dir", default="outputs", help="Folder untuk hasil CSV dan grafik.")
    parser.add_argument(
        "--bert-model",
        default="indobenchmark/indobert-base-p1",
        help="Nama model Hugging Face untuk fine-tuning IndoBERT.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Jumlah fold cross-validation.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length untuk IndoBERT.")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Tampilkan plot (plt.show) selain menyimpan file PNG. Cocok untuk environment interaktif.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    ensure_nltk_resources()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stemmer = StemmerFactory().create_stemmer()
    id_stopwords, en_stopwords = build_stopword_sets()
    vader_analyzer = SentimentIntensityAnalyzer()

    df = load_dataset(Path(args.input))
    df["ethical_category"] = df[TEXT_COLUMN].apply(classify_ethics)
    df["processed_text"] = df[TEXT_COLUMN].apply(
        lambda text: preprocess_text(text, stemmer=stemmer, id_stopwords=id_stopwords, en_stopwords=en_stopwords)
    )

    labels = prepare_labels(df)
    labels = maybe_bootstrap_labels_from_lexicon(
        df,
        labels,
        vader_analyzer=vader_analyzer,
        stemmer=stemmer,
        id_stopwords=id_stopwords,
        en_stopwords=en_stopwords,
    )
    folds = validate_and_adjust_folds(labels, args.folds)

    df["vader_sentiment"] = aggregate_lexicon_predictions(
        df,
        vader_analyzer=vader_analyzer,
        stemmer=stemmer,
        id_stopwords=id_stopwords,
        en_stopwords=en_stopwords,
    )
    lexicon_metrics = compute_metrics(labels, df["vader_sentiment"])

    nb_model, nb_metrics, nb_cv_predictions, nb_metadata = train_ml_model(df, labels, n_splits=folds)
    df["naive_bayes_sentiment"] = nb_model.predict(df[["processed_text"]])

    bert_config = BertConfig(model_name=args.bert_model, max_length=args.max_length)
    bert_metrics, bert_cv_predictions, bert_metadata = cross_validate_indobert(
        texts=df[TEXT_COLUMN].tolist(),
        labels=labels,
        config=bert_config,
        output_dir=output_dir / "indobert_runs",
        n_splits=folds,
    )
    final_bert_trainer = fit_final_indobert(
        texts=df[TEXT_COLUMN].tolist(),
        labels=labels,
        config=bert_config,
        best_params=bert_metadata["best_params"],
        output_dir=output_dir / "indobert_runs",
    )
    df["indobert_sentiment"] = predict_with_indobert(
        final_bert_trainer,
        texts=df[TEXT_COLUMN].tolist(),
        max_length=args.max_length,
    )

    metrics_table = pd.DataFrame(
        [
            {"method": "VADER", **lexicon_metrics},
            {"method": "Naive Bayes", **nb_metrics},
            {"method": "IndoBERT", **bert_metrics},
        ]
    )
    metrics_table.to_csv(output_dir / "model_metrics.csv", index=False)
    bert_metadata["tuning_log"].to_csv(output_dir / "indobert_tuning_log.csv", index=False)
    with open(output_dir / "naive_bayes_classification_report.txt", "w", encoding="utf-8") as file:
        file.write(nb_metadata["classification_report"])

    best_method = choose_best_method(metrics_table)
    best_method_column = METHOD_COLUMN_MAP[best_method]

    df["sentiment"] = df[best_method_column]
    export_columns = [TEXT_COLUMN, "sentiment", "ethical_category"]
    optional_columns = [col for col in ["rating", "date", "vader_sentiment", "naive_bayes_sentiment", "indobert_sentiment"] if col in df.columns]
    df[export_columns + optional_columns].to_csv(output_dir / "reviews_with_sentiment_ethics.csv", index=False)

    plot_confusion_matrix(
        labels,
        df["vader_sentiment"],
        "Confusion Matrix - VADER",
        output_dir / "cm_vader.png",
        show_plots=args.show_plots,
    )
    plot_confusion_matrix(
        labels,
        nb_cv_predictions,
        "Confusion Matrix - Naive Bayes (CV)",
        output_dir / "cm_naive_bayes.png",
        show_plots=args.show_plots,
    )
    plot_confusion_matrix(
        labels,
        bert_cv_predictions,
        "Confusion Matrix - IndoBERT (CV)",
        output_dir / "cm_indobert.png",
        show_plots=args.show_plots,
    )
    visualize_results(
        df,
        metrics_table,
        best_method,
        best_method_column,
        output_dir,
        show_plots=args.show_plots,
    )

    insights = generate_auto_insights(
        df,
        metrics_table,
        best_method,
        best_method_column,
    )
    with open(output_dir / "insights.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(insights))

    print("\n=== MODEL METRICS ===")
    print(metrics_table.to_string(index=False))
    print("\n=== NAIVE BAYES BEST PARAMS ===")
    print(nb_metadata["best_params"])
    print("\n=== INDOBERT BEST PARAMS ===")
    print(bert_metadata["best_params"])
    print("\n=== AUTO INSIGHTS ===")
    for insight in insights:
        print(f"- {insight}")
    print(f"\nHasil tersimpan di: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
