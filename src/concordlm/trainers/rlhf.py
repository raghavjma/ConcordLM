"""
Stage 3 (Optional) — RLHF with Reward Model + PPO

Two-phase pipeline:
  Phase A: Train a reward model on preference data (RewardTrainer).
  Phase B: Fine-tune the policy model with PPO using the reward model.
"""

from __future__ import annotations

import argparse
import logging
import os

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
# Phase B — PPO Training
# ---------------------------------------------------------------------------


def run_ppo(
    config: PipelineConfig,
    reward_model_path: str,
) -> str:
    """
    Fine-tune the policy model using PPO with a trained reward model.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline config.
    reward_model_path : str
        Path to the trained reward model checkpoint.

    Returns
    -------
    str  Path to the saved PPO-trained model.
    """
    import torch
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from transformers import pipeline as hf_pipeline

    from concordlm.data.preference_dataset import load_preference_dataset
    from concordlm.models.loader import load_tokenizer, _build_bnb_config, _get_torch_dtype

    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 3B: PPO Training")
    logger.info("=" * 60)

    # --- Determine starting model ---
    model_name = config.dpo_model_path or config.sft_model_path or config.model.name
    logger.info(f"Policy model: {model_name}")
    logger.info(f"Reward model: {reward_model_path}")

    tokenizer = load_tokenizer(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )

    # --- Load policy model with value head ---
    bnb_config = _build_bnb_config(config.model)
    model_kwargs = {}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = _get_torch_dtype(config.model.dtype)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        **model_kwargs,
    )

    # --- Load reward pipeline ---
    reward_pipe = hf_pipeline(
        "text-classification",
        model=reward_model_path,
        tokenizer=tokenizer,
        device_map="auto",
        function_to_apply="none",  # Raw logits
    )

    # --- PPO Config ---
    ppo_config = PPOConfig(
        learning_rate=config.ppo.learning_rate,
        batch_size=config.ppo.batch_size,
        mini_batch_size=config.ppo.mini_batch_size,
        ppo_epochs=config.ppo.ppo_epochs,
        kl_penalty=config.ppo.kl_penalty,
        init_kl_coef=config.ppo.init_kl_coef,
        adap_kl_ctrl=config.ppo.adap_kl_ctrl,
        target_kl=config.ppo.target_kl,
        seed=config.training.seed,
    )

    # --- Load prompts from preference dataset ---
    dataset = load_preference_dataset(config.data, tokenizer)
    prompts = []
    for example in dataset["train"]:
        prompt = example["prompt"]
        if isinstance(prompt, list):
            # Conversational format — apply chat template
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = str(prompt)
        prompts.append(text)

    # --- PPO Training Loop ---
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "max_new_tokens": config.ppo.max_new_tokens,
        "temperature": config.ppo.temperature,
        "top_p": config.ppo.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    logger.info("Starting PPO training loop...")
    batch_size = config.ppo.batch_size

    for step_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[step_idx : step_idx + batch_size]
        if len(batch_prompts) < batch_size:
            break  # Skip incomplete batches

        # Tokenize prompts
        query_tensors = [
            tokenizer.encode(p, return_tensors="pt").squeeze(0)
            for p in batch_prompts
        ]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors, **generation_kwargs
        )

        # Decode responses
        responses = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]

        # Get reward scores
        full_texts = [p + r for p, r in zip(batch_prompts, responses)]
        reward_outputs = reward_pipe(full_texts, batch_size=batch_size)
        rewards = [torch.tensor(o["score"]) for o in reward_outputs]

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, {"query": batch_prompts, "response": responses}, rewards)

        if (step_idx // batch_size) % 10 == 0:
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            logger.info(
                f"Step {step_idx // batch_size}: "
                f"mean_reward={mean_reward:.4f}, "
                f"kl={stats.get('objective/kl', 0):.4f}"
            )

    # --- Save ---
    final_path = os.path.join(config.training.output_dir, "ppo-final")
    ppo_trainer.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"PPO training complete. Model saved to: {final_path}")

    return final_path


# ---------------------------------------------------------------------------
# Full RLHF pipeline
# ---------------------------------------------------------------------------


def run_rlhf(config: PipelineConfig) -> str:
    """
    Execute the complete RLHF pipeline: reward model training → PPO.

    Returns path to the final PPO-trained model.
    """
    logger.info("=" * 60)
    logger.info("  ConcordLM — Stage 3: Full RLHF Pipeline")
    logger.info("=" * 60)

    # Phase A: Train reward model
    reward_model_path = train_reward_model(config)

    # Phase B: PPO training
    final_path = run_ppo(config, reward_model_path)

    return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for RLHF training."""
    parser = argparse.ArgumentParser(
        description="ConcordLM — Stage 3: RLHF (Reward Model + PPO)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/rlhf.yaml",
        help="Path to the RLHF config YAML file",
    )
    parser.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Config overrides in dot-notation",
    )
    parser.add_argument(
        "--reward-model-only", action="store_true",
        help="Only train the reward model (skip PPO)",
    )
    parser.add_argument(
        "--ppo-only", type=str, default=None,
        help="Only run PPO with given reward model path (skip reward model training)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config, overrides=args.override)

    if args.reward_model_only:
        train_reward_model(config)
    elif args.ppo_only:
        run_ppo(config, args.ppo_only)
    else:
        run_rlhf(config)


if __name__ == "__main__":
    main()
