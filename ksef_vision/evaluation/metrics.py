"""Evaluation metrics."""

from typing import Dict, List


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute simple accuracy."""
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / max(len(references), 1)