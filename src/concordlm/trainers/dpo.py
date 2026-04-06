"""
Stage 2 — Direct Preference Optimization (DPO)

Align a model to human preferences by training on chosen/rejected pairs.
Uses TRL's DPOTrainer with the model from SFT (Stage 1) as the starting point.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from concordlm.config import PipelineConfig, load_config

logger = logging.getLogger(__name__)


def run_dpo(config: PipelineConfig) -> str:
    """
    Execute the DPO training pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline config (DPO-relevant fields are used).

    Returns
    -------
    str  Path to the saved aligned model checkpoint.
    """
    from trl import DPOTrainer, DPOConfig

    from concordlm.data.preference_dataset import load_preference_dataset
    from concordlm.models.loader import load_model_for_training, load_tokenizer

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 2: Direct Preference Optimization (DPO)")
    logger.info("=" * 60)

    # --- Determine the starting model ---
    # If an SFT checkpoint is provided, use it; otherwise use the base model
    model_name = config.sft_model_path or config.model.name
    logger.info(f"Starting model: {model_name}")
    logger.info(f"DPO β = {config.dpo.beta}, loss_type = {config.dpo.loss_type}")

    # --- Load tokenizer (for dataset prep) ---
    tokenizer = load_tokenizer(
        model_name if not config.sft_model_path else config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )

    # --- Load preference dataset ---
    dataset = load_preference_dataset(config.data, tokenizer)

    # --- Build PEFT config ---
    from concordlm.models.loader import _build_lora_config
    peft_config = _build_lora_config(config.lora)

    # --- Configure DPO trainer ---
    dpo_config = DPOConfig(
        output_dir=config.training.output_dir,
        beta=config.dpo.beta,
        loss_type=config.dpo.loss_type,
        label_smoothing=config.dpo.label_smoothing,
        max_length=config.dpo.max_length,
        max_prompt_length=config.dpo.max_prompt_length,
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
    )

    # If we have an SFT checkpoint with a PEFT adapter, load it
    if config.sft_model_path:
        from peft import AutoPeftModelForCausalLM
        logger.info(f"Loading SFT PEFT model from: {config.sft_model_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            config.sft_model_path,
            is_trainable=True,
        )
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            processing_class=tokenizer,
        )
    else:
        # Pass model_init_kwargs for quantization via DPOConfig
        if config.model.quantization == "4bit":
            import torch
            dpo_config.model_init_kwargs = {
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_use_double_quant": True,
                },
                "trust_remote_code": config.model.trust_remote_code,
            }

        trainer = DPOTrainer(
            model=model_name,
            args=dpo_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    # --- Train ---
    logger.info("Starting DPO training...")
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

    # Log DPO-specific metrics
    logger.info(f"DPO training complete. Model saved to: {final_path}")
    logger.info(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    if "train_rewards/margins" in metrics:
        logger.info(f"Reward margin: {metrics['train_rewards/margins']:.4f}")
    if "train_rewards/accuracies" in metrics:
        logger.info(f"Reward accuracy: {metrics['train_rewards/accuracies']:.4f}")

    return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for DPO training."""
    parser = argparse.ArgumentParser(
        description="ConcordLM — Stage 2: Direct Preference Optimization"
    )
    parser.add_argument(
        "--config", type=str, default="configs/dpo.yaml",
        help="Path to the DPO config YAML file",
    )
    parser.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Config overrides in dot-notation",
    )
    parser.add_argument(
        "--sft-model", type=str, default=None,
        help="Path to SFT checkpoint (overrides config.sft_model_path)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    overrides = list(args.override)
    if args.sft_model:
        overrides.append(f"sft_model_path={args.sft_model}")

    config = load_config(args.config, overrides=overrides)
    run_dpo(config)


if __name__ == "__main__":
    main()
