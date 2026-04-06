"""
Stage 1 — Supervised Fine-Tuning (SFT)

Train a base model on instruction-following data using TRL's SFTTrainer
with LoRA/QLoRA for parameter-efficient training.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from concordlm.config import PipelineConfig, load_config, config_to_dict

logger = logging.getLogger(__name__)


def run_sft(config: PipelineConfig) -> str:
    """
    Execute the SFT training pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline config (only SFT-relevant fields are used).

    Returns
    -------
    str  Path to the saved model checkpoint.
    """
    from trl import SFTTrainer, SFTConfig

    from concordlm.data.sft_dataset import load_sft_dataset
    from concordlm.models.loader import load_model_for_training

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 1: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)

    # --- Load model & tokenizer ---
    model, tokenizer, peft_config = load_model_for_training(
        config.model, config.lora
    )

    # --- Load dataset ---
    dataset = load_sft_dataset(config.data, tokenizer)

    # --- Configure trainer ---
    sft_config = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        learning_rate=config.training.learning_rate,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
        report_to=config.training.report_to,
        dataloader_num_workers=config.training.dataloader_num_workers,
        max_seq_length=config.data.max_seq_length,
        packing=config.data.packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --- Train ---
    logger.info("Starting SFT training...")
    train_result = trainer.train()

    # --- Save ---
    final_path = os.path.join(config.training.output_dir, "checkpoint-final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"SFT training complete. Model saved to: {final_path}")
    logger.info(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")

    return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for SFT training."""
    parser = argparse.ArgumentParser(
        description="ConcordLM — Stage 1: Supervised Fine-Tuning"
    )
    parser.add_argument(
        "--config", type=str, default="configs/sft.yaml",
        help="Path to the SFT config YAML file",
    )
    parser.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Config overrides in dot-notation (e.g. model.name=... training.max_steps=5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config, overrides=args.override)
    run_sft(config)


if __name__ == "__main__":
    main()
