"""LoRA adapter for LayoutLMv3."""

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForTokenClassification

MODEL_NAME = "microsoft/layoutlmv3-base"


def load_lora_model(num_labels: int, r: int = 8):
    """Load LayoutLMv3 with LoRA adapters."""
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    lora_config = LoraConfig(r=r, lora_alpha=16, target_modules=["query", "value"])
    model = get_peft_model(model, lora_config)
    return model