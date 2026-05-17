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

# Suppress the incredibly noisy tokenization warnings from TRL's DPOTrainer
logging.getLogger("trl.trainer.dpo_trainer").setLevel(logging.ERROR)


def _resolve_starting_model(config: PipelineConfig) -> str:
    """Determine the starting model path/name for DPO.

    Priority: sft_model_path (if exists) → base model name.
    """
    if config.sft_model_path:
        sft_path = Path(config.sft_model_path)
        if sft_path.exists():
            logger.info(f"Using SFT checkpoint: {config.sft_model_path}")
            return config.sft_model_path
        else:
            logger.warning(
                f"SFT model path not found: {config.sft_model_path}. "
                "Falling back to base model."
            )
    return config.model.name


def _load_sft_model(model_path: str, config: PipelineConfig):
    """Load an SFT model checkpoint, trying PEFT first, then standard.

    Returns (model, is_peft) tuple.
    """
    import torch

    from concordlm.models.loader import _build_bnb_config, _get_torch_dtype

    # Determine model kwargs
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
    }

    bnb_config = _build_bnb_config(config.model)
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = _get_torch_dtype(config.model.dtype)

    # Set attention implementation
    if config.model.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "eager"

    # Try PEFT model first
    try:
        from peft import AutoPeftModelForCausalLM

        logger.info(f"Attempting to load as PEFT model: {model_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            is_trainable=True,
            **model_kwargs,
        )
        logger.info("Loaded SFT model as PEFT adapter.")
        return model, True
    except Exception as e:
        logger.info(f"Not a PEFT model ({e}). Loading as standard model.")

    # Fall back to standard AutoModelForCausalLM
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    logger.info("Loaded SFT model as standard CausalLM.")
    return model, False


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
    from concordlm.models.loader import load_tokenizer, _build_lora_config

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 2: Direct Preference Optimization (DPO)")
    logger.info("=" * 60)

    # --- Determine starting model ---
    model_name = _resolve_starting_model(config)
    logger.info(f"Starting model: {model_name}")
    logger.info(f"DPO β = {config.dpo.beta}, loss_type = {config.dpo.loss_type}")

    # --- Load tokenizer ---
    tokenizer = load_tokenizer(
        config.model.name,  # Always use base model's tokenizer
        trust_remote_code=config.model.trust_remote_code,
    )

    # --- Load preference dataset ---
    dataset = load_preference_dataset(config.data, tokenizer)

    # --- Build PEFT config ---
    peft_config = _build_lora_config(config.lora)

    # --- Determine attention implementation ---
    attn_impl = "flash_attention_2" if config.model.use_flash_attention else "eager"

    # --- Configure DPO trainer ---
    dpo_config = DPOConfig(
        output_dir=config.training.output_dir,
        beta=config.dpo.beta,
        loss_type=config.dpo.loss_type,
        label_smoothing=config.dpo.label_smoothing,
        max_length=config.dpo.max_length,
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

    # --- Load model and create trainer ---
    if config.sft_model_path and Path(config.sft_model_path).exists():
        # Load from SFT checkpoint
        model, is_peft = _load_sft_model(config.sft_model_path, config)

        if is_peft:
            # PEFT model is already loaded with adapters — don't pass peft_config again
            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                processing_class=tokenizer,
            )
        else:
            # Standard model — pass peft_config so DPOTrainer wraps it
            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                processing_class=tokenizer,
                peft_config=peft_config,
            )
    else:
        # Load from model name (Hub) — pass quantization via model_init_kwargs
        model_init_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "attn_implementation": attn_impl,
        }
        if config.model.quantization == "4bit":
            model_init_kwargs["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        elif config.model.quantization == "8bit":
            model_init_kwargs["quantization_config"] = {"load_in_8bit": True}

        dpo_config.model_init_kwargs = model_init_kwargs

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

    # DPO reward metrics
    for key in ["train_rewards/margins", "train_rewards/accuracies",
                 "train_rewards/chosen", "train_rewards/rejected",
                 "train_logps/chosen", "train_logps/rejected"]:
        if key in metrics:
            logger.info(f"  {key}: {metrics[key]:.4f}")

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
        "--override", type=str, action="append", default=[],
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
