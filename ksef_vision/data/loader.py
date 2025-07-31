"""Dataset loading utilities."""

from pathlib import Path

from datasets import DatasetDict, load_dataset


def load_pl_micro(data_dir: Path = Path("data/pl_micro")) -> DatasetDict:
    """Return a DatasetDict with train, dev, and test splits."""
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
    return dataset