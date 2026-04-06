"""
Data utilities — tokenization, chat-template helpers, statistics.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat-template helpers
# ---------------------------------------------------------------------------


def apply_chat_template(
    example: dict[str, Any],
    tokenizer,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Convert a raw instruction/response dict into the model's chat format.

    Supports both ``{"instruction", "output"}`` and ``{"messages"}`` formats.
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Already in messages format
    if "messages" in example:
        messages.extend(example["messages"])
    # instruction / input / output format (e.g. Alpaca)
    elif "instruction" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += f"\n\n{example['input']}"
        messages.append({"role": "user", "content": user_content})
        if example.get("output"):
            messages.append({"role": "assistant", "content": example["output"]})
    # prompt / response format
    elif "prompt" in example:
        messages.append({"role": "user", "content": example["prompt"]})
        if example.get("response"):
            messages.append({"role": "assistant", "content": example["response"]})
    else:
        raise ValueError(
            f"Unrecognized data format. Keys: {list(example.keys())}. "
            "Expected 'messages', 'instruction', or 'prompt'."
        )

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": formatted}
    except Exception as e:
        # Some datasets have back-to-back user messages or malformed roles
        # that break rigid templates (like Mistral's). Return empty to filter out.
        return {"text": ""}


def format_preference_example(
    example: dict[str, Any],
    tokenizer=None,
) -> dict[str, Any]:
    """
    Normalise a preference example into DPO-ready conversational format.

    Expected output columns: ``prompt``, ``chosen``, ``rejected``.
    Supports both plain-text and conversational inputs.
    """
    result: dict[str, Any] = {}

    # --- Prompt ---
    if "prompt" in example:
        prompt = example["prompt"]
        # Already in conversational format
        if isinstance(prompt, list):
            result["prompt"] = prompt
        else:
            result["prompt"] = [{"role": "user", "content": str(prompt)}]
    elif "instruction" in example:
        content = example["instruction"]
        if example.get("input"):
            content += f"\n\n{example['input']}"
        result["prompt"] = [{"role": "user", "content": content}]
    else:
        raise ValueError(
            f"Cannot extract prompt from example with keys: {list(example.keys())}"
        )

    # --- Chosen / Rejected ---
    for key in ("chosen", "rejected"):
        value = example.get(key)
        if value is None:
            raise ValueError(f"Missing required field: '{key}'")
        if isinstance(value, list):
            result[key] = value  # Already conversational
        else:
            result[key] = [{"role": "assistant", "content": str(value)}]

    return result


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def compute_dataset_stats(dataset: Dataset, tokenizer=None) -> dict[str, Any]:
    """
    Compute basic statistics for a dataset.

    Returns dict with counts, length distributions, etc.
    """
    stats: dict[str, Any] = {
        "num_examples": len(dataset),
        "columns": list(dataset.column_names),
    }

    # Text-length statistics
    text_cols = []
    for col in dataset.column_names:
        sample = dataset[0][col]
        if isinstance(sample, str):
            text_cols.append(col)

    for col in text_cols:
        lengths = [len(str(ex)) for ex in dataset[col]]
        stats[f"{col}_char_length"] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }

    # Token-length statistics (if tokenizer provided)
    if tokenizer and text_cols:
        col = text_cols[0]
        token_lengths = [
            len(tokenizer.encode(str(ex), add_special_tokens=False))
            for ex in dataset[col][:1000]  # Sample first 1000 for speed
        ]
        stats[f"{col}_token_length_sample"] = {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
        }

    logger.info(f"Dataset stats: {stats['num_examples']} examples, columns: {stats['columns']}")
    return stats


def log_dataset_sample(dataset: Dataset, n: int = 3) -> None:
    """Log the first *n* samples from a dataset for sanity checking."""
    for i in range(min(n, len(dataset))):
        example = dataset[i]
        logger.info(f"--- Sample {i} ---")
        for key, value in example.items():
            display = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            logger.info(f"  {key}: {display}")
