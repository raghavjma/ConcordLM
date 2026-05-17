"""
Stage 3 (Optional) — RLHF: Reward Model + Policy Optimization

Two-phase pipeline:
  Phase A: Train a reward model on preference data (RewardTrainer).
  Phase B: Fine-tune the policy model using Group Relative Policy Optimization
           (GRPO) with the trained reward model.

TRL v1.0 Migration Notes:
  - AutoModelForCausalLMWithValueHead has been removed
  - PPOTrainer has been removed from the core API
  - GRPOTrainer is the recommended replacement (no value head needed,
    more memory-efficient, state-of-the-art results)
  - RLOOTrainer is available as an alternative
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from concordlm.config import PipelineConfig, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase A — Reward Model Training
# ---------------------------------------------------------------------------


def train_reward_model(config: PipelineConfig) -> str:
    """
    Train a reward model on preference data.

    The reward model learns to assign a scalar score to (prompt, response) pairs,
    preferring chosen over rejected responses.

    Returns path to the saved reward model checkpoint.
    """
    from trl import RewardConfig, RewardTrainer

    from concordlm.data.preference_dataset import load_preference_dataset
    from concordlm.models.loader import load_reward_model, load_tokenizer

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 3A: Reward Model Training")
    logger.info("=" * 60)

    model_name = config.model.name
    logger.info(f"Base model for reward model: {model_name}")

    # --- Load reward model ---
    model, tokenizer, peft_config = load_reward_model(
        config.model, config.lora, num_labels=1
    )

    # --- Load preference dataset ---
    dataset = load_preference_dataset(config.data, tokenizer)

    # --- Configure trainer ---
    reward_config = RewardConfig(
        output_dir=config.reward_model.output_dir,
        num_train_epochs=config.reward_model.num_train_epochs,
        per_device_train_batch_size=config.reward_model.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        learning_rate=config.reward_model.learning_rate,
        bf16=config.training.bf16,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        max_length=config.reward_model.max_length,
        report_to=config.training.report_to,
        seed=config.training.seed,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --- Train ---
    logger.info("Starting reward model training...")
    train_result = trainer.train()

    # --- Save ---
    final_path = os.path.join(config.reward_model.output_dir, "checkpoint-final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Reward model training complete. Saved to: {final_path}")
    if "eval_accuracy" in metrics:
        logger.info(f"Eval accuracy: {metrics['eval_accuracy']:.4f}")

    return final_path


# ---------------------------------------------------------------------------
# Phase B — GRPO Training (TRL v1.0)
# ---------------------------------------------------------------------------


def _build_prompt_dataset(config: PipelineConfig, tokenizer):
    """
    Build a prompt-only dataset for GRPO training.

    GRPO generates completions on-the-fly, so it only needs prompts.
    We extract prompts from the preference dataset.
    """
    from datasets import Dataset, DatasetDict

    from concordlm.data.preference_dataset import load_preference_dataset

    # Load the full preference dataset to extract prompts
    pref_dataset = load_preference_dataset(config.data, tokenizer)

    def extract_prompt(example):
        """Extract just the prompt, formatted for the model."""
        prompt = example.get("prompt", "")

        # Handle conversational format (list of message dicts)
        if isinstance(prompt, list):
            # Apply chat template to format the prompt
            try:
                formatted = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                return {"prompt": formatted}
            except Exception:
                # Fallback: concatenate content
                text = " ".join(
                    m.get("content", "") for m in prompt if m.get("role") == "user"
                )
                return {"prompt": text}

        return {"prompt": str(prompt)}

    train_prompts = pref_dataset["train"].map(
        extract_prompt,
        remove_columns=[c for c in pref_dataset["train"].column_names if c != "prompt"],
        desc="Extracting prompts for GRPO",
    )
    eval_prompts = pref_dataset["eval"].map(
        extract_prompt,
        remove_columns=[c for c in pref_dataset["eval"].column_names if c != "prompt"],
        desc="Extracting eval prompts for GRPO",
    )

    logger.info(
        f"GRPO prompt dataset: train={len(train_prompts)}, eval={len(eval_prompts)}"
    )

    return DatasetDict({"train": train_prompts, "eval": eval_prompts})


def _create_reward_function(reward_model_path: str, config: PipelineConfig):
    """
    Create a reward function from a trained reward model for GRPO.

    GRPOTrainer expects `reward_funcs` — either a model or a callable
    that takes (completions, prompts) and returns rewards.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )

    logger.info(f"Loading reward model from: {reward_model_path}")

    # Try loading as PEFT model first
    try:
        from peft import AutoPeftModelForSequenceClassification

        reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=1,
            device_map="auto",
        )
        reward_model = reward_model.merge_and_unload()
        logger.info("Loaded reward model as PEFT adapter (merged).")
    except Exception:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=1,
            device_map="auto",
        )
        logger.info("Loaded reward model as standard classifier.")

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    reward_model.eval()

    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        """Score completions using the reward model."""
        # Build full texts (prompt + completion)
        if prompts:
            texts = [p + c for p, c in zip(prompts, completions)]
        else:
            texts = completions

        with torch.no_grad():
            inputs = reward_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.reward_model.max_length,
                return_tensors="pt",
            ).to(reward_model.device)
            outputs = reward_model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        return scores

    return reward_fn


def run_grpo(
    config: PipelineConfig,
    reward_model_path: str,
) -> str:
    """
    Fine-tune the policy model using GRPO with a trained reward model.

    GRPO (Group Relative Policy Optimization) generates multiple completions
    per prompt, scores them with the reward model, and optimizes the policy
    to increase the likelihood of higher-reward completions relative to the
    group baseline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline config.
    reward_model_path : str
        Path to the trained reward model checkpoint.

    Returns
    -------
    str  Path to the saved GRPO-trained model.
    """
    from trl import GRPOConfig, GRPOTrainer

    from concordlm.models.loader import (
        _build_lora_config,
        load_tokenizer,
    )

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 3B: GRPO Policy Optimization")
    logger.info("=" * 60)

    # --- Determine starting model ---
    model_name = config.dpo_model_path or config.sft_model_path or config.model.name
    logger.info(f"Policy model: {model_name}")
    logger.info(f"Reward model: {reward_model_path}")
    logger.info(f"GRPO β={config.grpo.beta}, loss_type={config.grpo.loss_type}")

    # --- Load tokenizer ---
    tokenizer = load_tokenizer(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )

    # --- Build prompt dataset ---
    prompt_dataset = _build_prompt_dataset(config, tokenizer)

    # --- Create reward function ---
    reward_fn = _create_reward_function(reward_model_path, config)

    # --- Build PEFT config ---
    peft_config = _build_lora_config(config.lora)

    # --- Determine attention implementation ---
    attn_impl = "flash_attention_2" if config.model.use_flash_attention else "eager"

    # --- GRPO Config ---
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

    grpo_config = GRPOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        learning_rate=config.training.learning_rate,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
        report_to=config.training.report_to,
        # GRPO-specific
        num_generations=config.grpo.num_generations,
        max_completion_length=config.grpo.max_completion_length,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        beta=config.grpo.beta,
        num_iterations=config.grpo.num_iterations,
        loss_type=config.grpo.loss_type,
        scale_rewards=config.grpo.scale_rewards,
        # Model loading
        model_init_kwargs=model_init_kwargs,
        # Log completions for monitoring
        log_completions=True,
    )

    # --- Create trainer ---
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=prompt_dataset["train"],
        eval_dataset=prompt_dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --- Train ---
    logger.info("Starting GRPO training...")
    train_result = trainer.train()

    # --- Save ---
    final_path = os.path.join(config.training.output_dir, "grpo-final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"GRPO training complete. Model saved to: {final_path}")
    logger.info(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")

    # Log GRPO-specific metrics
    for key in ["reward", "reward_std", "kl", "completion_length"]:
        full_key = f"train_{key}"
        if full_key in metrics:
            logger.info(f"  {key}: {metrics[full_key]:.4f}")

    return final_path


def run_rloo(
    config: PipelineConfig,
    reward_model_path: str,
) -> str:
    """
    Fine-tune using RLOO (Reinforcement Learning with Leave-One-Out baseline).

    Alternative to GRPO — uses REINFORCE with a leave-one-out baseline.
    """
    from trl import RLOOConfig, RLOOTrainer

    from concordlm.models.loader import _build_lora_config, load_tokenizer

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 3B: RLOO Policy Optimization")
    logger.info("=" * 60)

    model_name = config.dpo_model_path or config.sft_model_path or config.model.name
    logger.info(f"Policy model: {model_name}")
    logger.info(f"Reward model: {reward_model_path}")

    tokenizer = load_tokenizer(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )

    prompt_dataset = _build_prompt_dataset(config, tokenizer)
    reward_fn = _create_reward_function(reward_model_path, config)
    peft_config = _build_lora_config(config.lora)

    attn_impl = "flash_attention_2" if config.model.use_flash_attention else "eager"

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

    rloo_config = RLOOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        learning_rate=config.training.learning_rate,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        seed=config.training.seed,
        report_to=config.training.report_to,
        model_init_kwargs=model_init_kwargs,
    )

    trainer = RLOOTrainer(
        model=model_name,
        reward_funcs=reward_fn,
        args=rloo_config,
        train_dataset=prompt_dataset["train"],
        eval_dataset=prompt_dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting RLOO training...")
    train_result = trainer.train()

    final_path = os.path.join(config.training.output_dir, "rloo-final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"RLOO training complete. Model saved to: {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Full RLHF pipeline
# ---------------------------------------------------------------------------


def run_rlhf(config: PipelineConfig) -> str:
    """
    Execute the complete RLHF pipeline: reward model training → policy optimization.

    Supports two methods:
      - "grpo" (default): Group Relative Policy Optimization
      - "rloo": REINFORCE Leave-One-Out

    Returns path to the final policy-optimized model.
    """
    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 3: Full RLHF Pipeline")
    logger.info(f"  Method: {config.method}")
    logger.info("=" * 60)

    # Phase A: Train reward model (or use pre-trained one)
    if config.reward_model_name_or_path:
        reward_model_path = config.reward_model_name_or_path
        logger.info(f"Using pre-trained reward model: {reward_model_path}")
    else:
        reward_model_path = train_reward_model(config)

    # Phase B: Policy optimization
    method = config.method.lower()
    if method == "grpo":
        final_path = run_grpo(config, reward_model_path)
    elif method == "rloo":
        final_path = run_rloo(config, reward_model_path)
    else:
        raise ValueError(
            f"Unknown RLHF method: {method}. Choose from: 'grpo', 'rloo'"
        )

    return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for RLHF training."""
    parser = argparse.ArgumentParser(
        description="ConcordLM — Stage 3: RLHF (Reward Model + Policy Optimization)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/rlhf.yaml",
        help="Path to the RLHF config YAML file",
    )
    parser.add_argument(
        "--override", type=str, action="append", default=[],
        help="Config overrides in dot-notation",
    )
    parser.add_argument(
        "--method", type=str, default=None, choices=["grpo", "rloo"],
        help="Policy optimization method (overrides config)",
    )
    parser.add_argument(
        "--reward-model-only", action="store_true",
        help="Only train the reward model (skip policy optimization)",
    )
    parser.add_argument(
        "--policy-only", type=str, default=None,
        help="Only run policy optimization with given reward model path",
    )
    parser.add_argument(
        "--reward-model", type=str, default=None,
        help="Use a pre-trained reward model (skip reward model training)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    overrides = list(args.override)
    if args.method:
        overrides.append(f"method={args.method}")
    if args.reward_model:
        overrides.append(f"reward_model_name_or_path={args.reward_model}")

    config = load_config(args.config, overrides=overrides)

    if args.reward_model_only:
        train_reward_model(config)
    elif args.policy_only:
        method = config.method.lower()
        if method == "grpo":
            run_grpo(config, args.policy_only)
        elif method == "rloo":
            run_rloo(config, args.policy_only)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        run_rlhf(config)


if __name__ == "__main__":
    main()
