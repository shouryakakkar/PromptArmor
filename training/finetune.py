"""
training/finetune.py

Fine-tunes a DistilBERT model for binary prompt injection classification.

Architecture:
  - Base: distilbert-base-uncased (66M params)
  - Frozen: Transformer layers 0–3 (first half)
  - Trainable: Transformer layers 4–5 + classifier head
  - Output: 2-class softmax (0=clean, 1=injection)

Training config:
  - 3 epochs, batch_size=16, lr=2e-5, weight_decay=0.01
  - Evaluate on test split every epoch
  - Save best model by eval F1

Usage:
  python -m training.finetune
  python -m training.finetune --seed-path data/seed_injections.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Block TensorFlow imports before transformers loads.
# transformers optionally imports tf_keras which crashes against TF 2.x on
# some environments. Setting these vars forces the PyTorch-only path.
# Must be done HERE at module level before any transformers import occurs.
# ---------------------------------------------------------------------------
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = "./models/classifier"
DATA_DIR = "./training/data"
BASE_MODEL = "distilbert-base-uncased"

NUM_LABELS = 2
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
FREEZE_LAYERS = [0, 1, 2, 3]  # Freeze first 4 transformer layers


def prepare_data(seed_path: str) -> None:
    """Generate and save the augmented dataset if not already done."""
    train_csv = os.path.join(DATA_DIR, "train.csv")
    test_csv = os.path.join(DATA_DIR, "test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        logger.info("Dataset already exists at %s — skipping generation.", DATA_DIR)
        return

    logger.info("Generating augmented dataset from %s …", seed_path)
    # Import here to avoid circular issues when running as module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.dataset import generate_augmented_dataset, save_dataset

    data = generate_augmented_dataset(seed_path=seed_path)
    save_dataset(data, output_dir=DATA_DIR)


def load_csv_dataset(data_dir: str):
    """Load train and test CSVs as HuggingFace DatasetDict."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(data_dir, "train.csv"),
            "test": os.path.join(data_dir, "test.csv"),
        },
    )
    return ds


def tokenize_dataset(dataset, tokenizer):
    """Tokenize text column and add labels."""
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def freeze_lower_layers(model) -> None:
    """Freeze transformer layers 0–3, keep 4–5 + head trainable."""
    try:
        for layer_idx in FREEZE_LAYERS:
            for param in model.distilbert.transformer.layer[layer_idx].parameters():
                param.requires_grad = False
        logger.info("Froze transformer layers %s", FREEZE_LAYERS)
    except AttributeError:
        logger.warning("Could not freeze layers — model architecture may differ. Training all layers.")


def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score  # type: ignore

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1}


def run_finetuning(seed_path: str = "data/seed_injections.json") -> None:
    """Full fine-tuning pipeline."""
    import numpy as np
    import torch
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

    # ── Prepare data ──────────────────────────────────────────────────────────
    prepare_data(seed_path)
    raw_ds = load_csv_dataset(DATA_DIR)

    # ── Load tokenizer and model ──────────────────────────────────────────────
    logger.info("Loading base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
    )

    # ── Freeze early layers ───────────────────────────────────────────────────
    freeze_lower_layers(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.1f%%)", trainable_params, total_params,
                100 * trainable_params / total_params)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer)

    # ── Training arguments ────────────────────────────────────────────────────
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    os.makedirs("./models/checkpoints", exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./models/logs",
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning…")
    trainer.train()

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info("Running final evaluation on test set…")
    predictions_output = trainer.predict(tokenized_ds["test"])
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=-1)

    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(classification_report(labels, preds, target_names=["clean", "injection"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    # Save metrics
    metrics = {
        "final_metrics": predictions_output.metrics,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    with open(os.path.join(MODEL_OUTPUT_PATH, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Save model ────────────────────────────────────────────────────────────
    trainer.save_model(MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    logger.info("Model saved to %s", MODEL_OUTPUT_PATH)
    print(f"\n✓ Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for prompt injection detection")
    parser.add_argument(
        "--seed-path",
        default="data/seed_injections.json",
        help="Path to seed_injections.json",
    )
    args = parser.parse_args()

    run_finetuning(seed_path=args.seed_path)
