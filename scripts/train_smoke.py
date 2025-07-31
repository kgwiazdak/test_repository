"""Simple smoke test for model forward pass.

This test is skipped automatically if required libraries are missing.
"""

import sys
from pathlib import Path

import pytest

# Skip test if transformers or torch are not installed.
pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ksef_vision.models.layoutlmv3_ft import load_model


def test_forward_pass() -> None:
    """Run a forward pass on a simple input."""
    model, tokenizer = load_model(num_labels=2)
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.logits.shape[0] == 1