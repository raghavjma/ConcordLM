"""
SFT Dataset — load and prepare instruction-tuning datasets.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from concordlm.config import DataConfig
from concordlm.data.utils import apply_chat_template, compute_dataset_stats, log_dataset_sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported dataset adapters
# ---------------------------------------------------------------------------

_KNOWN_DATASETS = {
    "tatsu-lab/alpaca": {
        "format": "alpaca",  # instruction / input / output
    },
    "OpenAssistant/oasst2": {
        "format": "messages",
    },
    "yahma/alpaca-cleaned": {
        "format": "alpaca",
    },
}


def load_sft_dataset(
    data_config: DataConfig,
    tokenizer,
    *,
    system_prompt: str | None = None,
) -> DatasetDict:
    """
    Load an instruction-tuning dataset and format for SFTTrainer.

    Returns a DatasetDict with ``train`` and ``eval`` splits.
    Each example has a ``text`` column containing the chat-template-formatted text.
    """
    dataset_name = data_config.dataset_name
    logger.info(f"Loading SFT dataset: {dataset_name}")

    # --- Load ---
    ds = load_dataset(dataset_name, split=data_config.dataset_split)
    assert isinstance(ds, Dataset)

    # --- Subsample ---
    if data_config.max_samples and data_config.max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(data_config.max_samples))
        logger.info(f"Subsampled to {data_config.max_samples} examples")

    # --- Format ---
    ds = ds.map(
        lambda ex: apply_chat_template(ex, tokenizer, system_prompt=system_prompt),
        num_proc=data_config.preprocessing_num_workers,
        desc="Applying chat template",
    ).filter(lambda x: len(x.get("text", "")) > 0, desc="Filtering invalid templates")

    # Keep only the formatted text column
    keep_cols = {"text"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)

    # --- Train / eval split ---
    split = ds.train_test_split(test_size=data_config.eval_split_ratio, seed=42)
    result = DatasetDict({"train": split["train"], "eval": split["test"]})

    # --- Stats ---
    stats = compute_dataset_stats(result["train"], tokenizer)
    logger.info(f"SFT dataset ready — train: {len(result['train'])}, eval: {len(result['eval'])}")
    log_dataset_sample(result["train"])

    return result


def load_sft_dataset_from_jsonl(
    path: str,
    tokenizer,
    *,
    eval_split_ratio: float = 0.05,
    system_prompt: str | None = None,
) -> DatasetDict:
    """Load a custom SFT dataset from a local JSONL file."""
    logger.info(f"Loading custom SFT dataset from: {path}")
    ds = load_dataset("json", data_files=path, split="train")
    assert isinstance(ds, Dataset)

    ds = ds.map(
        lambda ex: apply_chat_template(ex, tokenizer, system_prompt=system_prompt),
        desc="Applying chat template",
    )
    keep_cols = {"text"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)

    split = ds.train_test_split(test_size=eval_split_ratio, seed=42)
    return DatasetDict({"train": split["train"], "eval": split["test"]})
