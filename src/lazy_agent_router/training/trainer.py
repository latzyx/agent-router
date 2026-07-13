"""Fine-tuning entry point for local or Hugging Face sequence-classification models."""

from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .dataset import labels_for, load_jsonl
from ..utils.device import resolve_device


def train_macbert(
    train_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    device: str = "auto",
) -> Path:
    """Fine tune a local path or Hugging Face model ID and save it for inference."""
    rows = load_jsonl(train_file)
    if not rows:
        raise ValueError("training dataset is empty")
    labels = labels_for(rows)
    label_to_id = {label: index for index, label in enumerate(labels)}
    dataset = Dataset.from_list([{"text": row["text"], "labels": label_to_id[row["intent"]]} for row in rows])
    source = str(model_path)
    # Existing directories must never unexpectedly contact the Hub. Model IDs
    # intentionally use the normal Hugging Face cache/download behaviour.
    local_only = Path(source).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=local_only,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
        # A seed checkpoint can have a different intent label set. Keep its
        # encoder weights while creating a fresh classifier head for this corpus.
        ignore_mismatched_sizes=True,
    )
    output = Path(output_dir)
    selected_device = resolve_device(device)
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
        report_to=[],
        use_cpu=selected_device == "cpu",
    )
    Trainer(model=model, args=arguments, train_dataset=tokenized, processing_class=tokenizer).train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return output
