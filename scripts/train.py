"""Fine-tune LayoutLMv3 on CPU."""

import hydra
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments
import torch

from ksef_vision.data.loader import load_pl_micro
from ksef_vision.models.layoutlmv3_ft import load_model

def check_cuda() -> None:
    """Report CUDA availability."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device}")
    else:
        print("CUDA is not available. Using CPU.")


@hydra.main(config_path="../configs", config_name="train_cpu", version_base=None)
def main(cfg: DictConfig) -> None:
    check_cuda()
    dataset = load_pl_micro()
    model, tokenizer = load_model(num_labels=cfg.num_labels)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        fp16=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])
    trainer.train()
    model.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()