"""Fine-tuned LayoutLMv3 model."""

from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "microsoft/layoutlmv3-base"


def load_model(num_labels: int):
    """Load LayoutLMv3 model for token classification."""
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer