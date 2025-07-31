"""Convert datasets to FUNSD-style JSONL."""

import json
from pathlib import Path

from datasets import load_dataset

from ksef_vision.data.utils import to_funsd_format

DEFAULT_DATA_DIR = Path("data/pl_micro")


def save_split(dataset, split: str):
    out_path = DEFAULT_DATA_DIR / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in dataset:
            json.dump(to_funsd_format(rec), f)
            f.write("\n")


def main():
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    katana = load_dataset("katanaml-org/invoices-donut-data-v1", split="train[:200]")
    train_test = katana.train_test_split(test_size=0.2, seed=42)
    dev_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
    save_split(train_test["train"], "train")
    save_split(dev_test["train"], "dev")
    save_split(dev_test["test"], "test")


if __name__ == "__main__":
    main()