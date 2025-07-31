"""Dataset loading utilities."""

from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset


def _tokenize_ground_truth(examples, tokenizer):
    """Convert raw `ground_truth` strings to LayoutLMv3 inputs."""
    texts = ["" if gt is None else str(gt) for gt in examples["ground_truth"]]
    enc = tokenizer(texts, padding="max_length", truncation=True)
    # Provide a dummy bounding box for every token produced by the tokenizer
    enc["bbox"] = [[[0, 0, 0, 0]] * len(ids) for ids in enc["input_ids"]]
    # Supply minimal image tensor and ignore labels for this micro dataset
    enc["pixel_values"] = [np.zeros((3, 224, 224), dtype="float32") for _ in texts]
    enc["labels"] = [[-100] * len(ids) for ids in enc["input_ids"]]
    return enc


def load_pl_micro(data_dir: Path = Path("data/pl_micro"), tokenizer=None) -> DatasetDict:
    """Return a DatasetDict with train, dev, and test splits.

    When a tokenizer is provided, the dataset is tokenized and augmented with
    dummy LayoutLMv3 features so the training scripts can run on the micro
    dataset without preprocessing pipelines.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "dev": str(data_dir / "dev.jsonl"),
            "test": str(data_dir / "test.jsonl"),
        },
    )
    if tokenizer is not None:
        dataset = dataset.map(
            lambda examples: _tokenize_ground_truth(examples, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
    return dataset