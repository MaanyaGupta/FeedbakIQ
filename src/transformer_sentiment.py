"""
Transformer-based Sentiment Analysis Pipeline (Phase 3): DistilBERT Classifier.
Fine-tunes DistilBERT (distilbert-base-uncased) for 3-class sentiment classification:
    0 = negative
    1 = neutral
    2 = positive

Supports:
- Reproducible stratified train/val/test splitting (70/15/15) matching Phase 2 baseline.
- Tokenization with customizable max_length.
- Class-weighted CrossEntropyLoss to address severe class imbalance (especially neutral class).
- Evaluation across Electronics, Fashion, and Combined datasets.
- Calculation of Accuracy, Precision, Recall, F1 (weighted & macro), Confusion Matrix,
  and detailed Neutral Class diagnostics.
- Serializes trained model and tokenizer under models/distilbert_sentiment/.
"""
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from src.config import (
        PROCESSED_REVIEWS_FILE,
        MODELS_DIR,
        RANDOM_STATE,
    )
    from src.preprocessing import clean_text
except ImportError:
    from config import (
        PROCESSED_REVIEWS_FILE,
        MODELS_DIR,
        RANDOM_STATE,
    )
    from preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Class and label mappings
LABEL2ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
CLASS_NAMES: List[str] = ["negative", "neutral", "positive"]

# Model configurations
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DISTILBERT_DIR = MODELS_DIR / "distilbert_sentiment"


class ReviewsDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for tokenized review texts and sentiment labels."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class WeightedTrainer(Trainer):
    """
    Custom Hugging Face Trainer incorporating class weights in CrossEntropyLoss
    to prevent majority-class collapse on imbalanced datasets.
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_and_prepare_data(
    data_path: Optional[Path] = None,
    domain: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Load processed reviews from parquet, optionally filter by domain,
    combine review title and text, clean text, and encode sentiment labels.
    """
    target_path = data_path or PROCESSED_REVIEWS_FILE
    if not target_path.exists():
        raise FileNotFoundError(
            f"Processed reviews file not found at '{target_path}'. "
            "Please run metadata and preprocessing pipeline first."
        )

    df = pd.read_parquet(target_path)
    logger.info("Loaded %d reviews from '%s'.", len(df), target_path.name)

    if domain and domain.lower() != "combined":
        df = df[df["domain"].str.lower() == domain.lower()].copy().reset_index(drop=True)
        logger.info("Filtered to domain '%s': %d reviews remaining.", domain, len(df))

    # Combine title and text
    titles = df["title"].fillna("").astype(str).str.strip()
    texts = df["text"].fillna("").astype(str).str.strip()
    combined_text = titles.where(titles == "", titles + ". ") + texts
    cleaned_texts = combined_text.apply(lambda t: clean_text(t) if pd.notna(t) else "")

    # Map sentiment labels to integer IDs
    sentiments = df["sentiment"].astype(str).str.strip().str.lower()
    encoded_labels = sentiments.map(LABEL2ID)

    if encoded_labels.isna().any():
        unmapped = sentiments[encoded_labels.isna()].unique()
        raise ValueError(f"Unknown sentiment values found: {unmapped}")

    return cleaned_texts, encoded_labels.astype(int)


def split_data(
    texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Stratified split into Train (70%), Validation (15%), and Test (15%).
    Matches the baseline split in src/sentiment.py for exact evaluation parity.
    """
    temp_size = test_size + val_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=temp_size,
        random_state=random_state,
        stratify=labels,
    )

    relative_test_ratio = test_size / temp_size
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=temp_labels,
    )

    logger.info(
        "Stratified split: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_texts),
        len(val_texts),
        len(test_texts),
        len(texts),
    )

    return {
        "train_texts": train_texts.tolist(),
        "train_labels": train_labels.tolist(),
        "val_texts": val_texts.tolist(),
        "val_labels": val_labels.tolist(),
        "test_texts": test_texts.tolist(),
        "test_labels": test_labels.tolist(),
    }


def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    """Compute Accuracy, Weighted/Macro Precision, Recall, and F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
    }


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    split_name: str = "Test",
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics, confusion matrix, and neutral class analysis."""
    acc = float(accuracy_score(y_true, y_pred))
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    target_names = [CLASS_NAMES[i] for i in [0, 1, 2]]
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=target_names, zero_division=0
    )

    neutral_stats = report_dict.get("neutral", {})
    neutral_precision = float(neutral_stats.get("precision", 0.0))
    neutral_recall = float(neutral_stats.get("recall", 0.0))
    neutral_f1 = float(neutral_stats.get("f1-score", 0.0))
    neutral_support = int(neutral_stats.get("support", 0))

    return {
        "split": split_name,
        "sample_count": len(y_true),
        "accuracy": acc,
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "neutral_precision": neutral_precision,
        "neutral_recall": neutral_recall,
        "neutral_f1": neutral_f1,
        "neutral_support": neutral_support,
        "confusion_matrix": cm.tolist(),
        "classes": CLASS_NAMES,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }


def fine_tune_distilbert(
    splits: Dict[str, Any],
    domain: str,
    output_dir: Path,
    model_name: str = PRETRAINED_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    random_state: int = RANDOM_STATE,
    use_class_weights: bool = True,
) -> Tuple[Trainer, AutoTokenizer, Dict[str, Any]]:
    """
    Fine-tune DistilBERT for sequence classification with Hugging Face Trainer.
    Evaluates on validation split each epoch and retains the best checkpoint.
    """
    set_seed(random_state)
    logger.info("Initializing tokenizer and DistilBERT model from '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    logger.info("Tokenizing train, val, and test splits (max_length=%d)...", max_length)
    train_encodings = tokenizer(
        splits["train_texts"], truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    val_encodings = tokenizer(
        splits["val_texts"], truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    test_encodings = tokenizer(
        splits["test_texts"], truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    train_dataset = ReviewsDataset(train_encodings, splits["train_labels"])
    val_dataset = ReviewsDataset(val_encodings, splits["val_labels"])
    test_dataset = ReviewsDataset(test_encodings, splits["test_labels"])

    # Compute balanced class weights if requested
    class_weights_tensor = None
    if use_class_weights:
        y_train = np.array(splits["train_labels"])
        classes = np.array([0, 1, 2])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        logger.info(
            "Balanced Class Weights for domain '%s': Negative=%.2f, Neutral=%.2f, Positive=%.2f",
            domain,
            weights[0],
            weights[1],
            weights[2],
        )

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=random_state,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights_tensor,
    )

    logger.info("Starting DistilBERT fine-tuning for domain '%s' (%d epochs)...", domain, num_epochs)
    start_time = time.time()
    train_result = trainer.train()
    training_duration = time.time() - start_time
    logger.info(
        "Finished training domain '%s' in %.2f seconds (%.2f min).",
        domain,
        training_duration,
        training_duration / 60.0,
    )

    # Save best model and tokenizer
    final_model_dir = output_dir / "best_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    logger.info("Saved fine-tuned DistilBERT model and tokenizer to: %s", final_model_dir)

    # Evaluate on Validation set
    val_pred_output = trainer.predict(val_dataset)
    val_preds = np.argmax(val_pred_output.predictions, axis=-1).tolist()
    val_metrics = evaluate_predictions(splits["val_labels"], val_preds, split_name="Validation")

    # Evaluate on Test set
    test_pred_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_pred_output.predictions, axis=-1).tolist()
    test_metrics = evaluate_predictions(splits["test_labels"], test_preds, split_name="Test")

    # Cleanup temporary checkpoints
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    training_meta = {
        "domain": domain,
        "model_name": model_name,
        "max_length": max_length,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": round(training_duration, 2),
        "train_samples": len(splits["train_texts"]),
        "val_samples": len(splits["val_texts"]),
        "test_samples": len(splits["test_texts"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": str(final_model_dir),
    }

    return trainer, tokenizer, training_meta


def run_transformer_sentiment_pipeline(
    data_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    domains: Optional[List[str]] = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Dict[str, Any]]:
    """
    Run DistilBERT training and evaluation across specified domains:
    Electronics, Fashion, and Combined.
    """
    target_domains = domains or ["Electronics", "Fashion", "Combined"]
    base_output_dir = save_dir or DISTILBERT_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Any]] = {}

    for domain in target_domains:
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING DOMAIN: %s", domain)
        logger.info("=" * 60)

        domain_texts, domain_labels = load_and_prepare_data(data_path, domain=domain)
        splits = split_data(domain_texts, domain_labels, test_size=0.15, val_size=0.15)

        domain_dir = base_output_dir / domain.lower()
        domain_dir.mkdir(parents=True, exist_ok=True)

        _, _, meta = fine_tune_distilbert(
            splits=splits,
            domain=domain,
            output_dir=domain_dir,
            model_name=PRETRAINED_MODEL_NAME,
            max_length=max_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=3e-5,
            random_state=RANDOM_STATE,
            use_class_weights=True,
        )
        all_results[domain] = meta

    # Save summary of all evaluations to JSON
    summary_path = base_output_dir / "distilbert_evaluation_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        exportable = {}
        for d, res in all_results.items():
            exportable[d] = {
                "training_time_seconds": res["training_time_seconds"],
                "train_samples": res["train_samples"],
                "val_samples": res["val_samples"],
                "test_samples": res["test_samples"],
                "test_accuracy": res["test_metrics"]["accuracy"],
                "test_weighted_precision": res["test_metrics"]["weighted_precision"],
                "test_weighted_recall": res["test_metrics"]["weighted_recall"],
                "test_weighted_f1": res["test_metrics"]["weighted_f1"],
                "test_macro_precision": res["test_metrics"]["macro_precision"],
                "test_macro_recall": res["test_metrics"]["macro_recall"],
                "test_macro_f1": res["test_metrics"]["macro_f1"],
                "neutral_precision": res["test_metrics"]["neutral_precision"],
                "neutral_recall": res["test_metrics"]["neutral_recall"],
                "neutral_f1": res["test_metrics"]["neutral_f1"],
                "neutral_support": res["test_metrics"]["neutral_support"],
                "confusion_matrix": res["test_metrics"]["confusion_matrix"],
                "model_path": res["model_path"],
            }
        json.dump(exportable, f, indent=2)

    logger.info("Saved DistilBERT evaluation metrics summary to: %s", summary_path)

    # Print comprehensive evaluation results
    print_transformer_evaluation_summary(all_results)

    return all_results


def print_transformer_evaluation_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print classification reports, confusion matrices, and baseline comparison tables."""
    baseline_metrics = {}
    baseline_file = MODELS_DIR / "baseline_evaluation_metrics.json"
    if baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    for domain, res in results.items():
        tm = res["test_metrics"]
        print("\n" + "=" * 68)
        print(f"DISTILBERT EVALUATION: {domain.upper()} (Test Set: {tm['sample_count']} samples)")
        print("=" * 68)
        print(f"Training Time:      {res['training_time_seconds']:.2f}s ({res['training_time_seconds'] / 60:.2f} min)")
        print(f"Overall Accuracy:   {tm['accuracy']:.4f}")
        print(f"Weighted F1-Score:  {tm['weighted_f1']:.4f}  |  Weighted Precision: {tm['weighted_precision']:.4f}  |  Weighted Recall: {tm['weighted_recall']:.4f}")
        print(f"Macro F1-Score:     {tm['macro_f1']:.4f}  |  Macro Precision:    {tm['macro_precision']:.4f}  |  Macro Recall:    {tm['macro_recall']:.4f}")

        print("\nClassification Report (Test Set):")
        print(tm["classification_report_text"])

        print("Confusion Matrix (Labels: ['negative' (0), 'neutral' (1), 'positive' (2)]):")
        cm = np.array(tm["confusion_matrix"])
        print(f"  Pred Negative  Pred Neutral  Pred Positive")
        for i, row in enumerate(cm):
            lbl = CLASS_NAMES[i].capitalize()
            print(f"Actual {lbl:<8}: {row[0]:<14} {row[1]:<13} {row[2]:<13}")

        print("\nNeutral Class (3-Star) Deep Dive:")
        print(f"- Support (True Neutrals): {tm['neutral_support']}")
        print(f"- Precision:               {tm['neutral_precision']:.4f}")
        print(f"- Recall:                  {tm['neutral_recall']:.4f}")
        print(f"- F1-Score:                {tm['neutral_f1']:.4f}")

    # Baseline Comparison Table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE: BASELINE (TF-IDF + LOGISTIC) vs DISTILBERT")
    print("=" * 80)
    print(f"| {'Domain':<12} | {'Model':<28} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'Macro F1':>8} | {'Neut F1':>8} |")
    print(f"| {':-----------':<12} | {':---------------------------':<28} | {'-------:':>8} | {'---------:':>9} | {'-------:':>8} | {'---:':>8} | {'--------:':>8} | {'-------:':>8} |")

    for domain, res in results.items():
        tm = res["test_metrics"]
        base = baseline_metrics.get(domain, {})
        base_acc = base.get("test_accuracy", 0.0)
        base_prec = base.get("test_weighted_precision", 0.0)
        base_rec = base.get("test_weighted_recall", 0.0)
        base_f1 = base.get("test_weighted_f1", 0.0)
        base_macro = base.get("test_macro_f1", 0.0)
        base_neut = base.get("test_neutral_f1", 0.0)

        print(f"| {domain:<12} | {'TF-IDF + Logistic Reg':<28} | {base_acc:>8.4f} | {base_prec:>9.4f} | {base_rec:>8.4f} | {base_f1:>8.4f} | {base_macro:>8.4f} | {base_neut:>8.4f} |")
        print(f"| {domain:<12} | {'DistilBERT (Fine-Tuned)':<28} | {tm['accuracy']:>8.4f} | {tm['weighted_precision']:>9.4f} | {tm['weighted_recall']:>8.4f} | {tm['weighted_f1']:>8.4f} | {tm['macro_f1']:>8.4f} | {tm['neutral_f1']:>8.4f} |")
        print(f"| {'':<12} | {'---------------------------':<28} | {'--------':>8} | {'---------':>9} | {'--------':>8} | {'----':>8} | {'--------':>8} | {'-------':>8} |")


if __name__ == "__main__":
    run_transformer_sentiment_pipeline()
