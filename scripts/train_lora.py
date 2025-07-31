"""Train LoRA adapters for LayoutLMv3."""

import hydra
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from ksef_vision.data.loader import load_pl_micro
from ksef_vision.models.layoutlmv3_lora import load_lora_model


@hydra.main(config_path="../configs", config_name="train_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset = load_pl_micro()
    model = load_lora_model(num_labels=cfg.num_labels, r=cfg.lora_r)
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