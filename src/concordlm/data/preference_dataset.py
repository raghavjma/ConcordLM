"""
Preference Dataset — load and prepare chosen/rejected datasets for DPO & RLHF.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from concordlm.config import DataConfig
from concordlm.data.utils import (
    compute_dataset_stats,
    format_preference_example,
    log_dataset_sample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hub dataset adapters
# ---------------------------------------------------------------------------


def _adapt_ultrafeedback(example: dict[str, Any]) -> dict[str, Any]:
    """Adapt trl-lib/ultrafeedback_binarized to standard DPO format.

    TRL v1.0 native format: ``chosen`` and ``rejected`` are full conversations
    (list of message dicts) where the prompt is embedded as the first user message.
    DPOTrainer v1.0 handles this natively — no separate ``prompt`` column needed.

    We extract the prompt for downstream use (e.g., GRPO prompt-only datasets).
    """
    chosen = example.get("chosen", [])
    rejected = example.get("rejected", [])

    # Extract prompt from the first user message in chosen
    prompt = []
    response_chosen = []
    response_rejected = []

    if isinstance(chosen, list) and len(chosen) > 0:
        # Split into prompt turns and response turns
        for msg in chosen:
            if msg.get("role") == "user":
                prompt.append(msg)
            else:
                response_chosen.append(msg)

        for msg in (rejected if isinstance(rejected, list) else []):
            if msg.get("role") != "user":
                response_rejected.append(msg)

    result = {
        "chosen": chosen,
        "rejected": rejected,
    }

    # Add prompt if extracted (for compatibility with prompt-based workflows)
    if prompt:
        result["prompt"] = prompt

    return result


def _adapt_anthropic_hh(example: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt Anthropic/hh-rlhf to DPO format.

    The raw dataset has ``chosen`` and ``rejected`` fields containing full
    conversation transcripts like:
        "\\n\\nHuman: ... \\n\\nAssistant: ..."

    We split the last assistant turn as the response and everything before as prompt.
    """
    def _split_transcript(transcript: str) -> tuple[str, str]:
        """Split a transcript into (prompt_text, last_assistant_response)."""
        parts = transcript.strip().split("\n\nAssistant: ")
        if len(parts) < 2:
            return transcript, ""
        last_response = parts[-1]
        prompt_text = "\n\nAssistant: ".join(parts[:-1])
        return prompt_text, last_response

    chosen_prompt, chosen_response = _split_transcript(example["chosen"])
    rejected_prompt, rejected_response = _split_transcript(example["rejected"])

    # The prompt should be the same for chosen and rejected — use chosen's
    # Extract just the human turns for the prompt
    human_text = chosen_prompt.replace("\n\nHuman: ", "").strip()

    return {
        "prompt": [{"role": "user", "content": human_text}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


_ADAPTERS = {
    "trl-lib/ultrafeedback_binarized": _adapt_ultrafeedback,
    "Anthropic/hh-rlhf": _adapt_anthropic_hh,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_preference_dataset(
    data_config: DataConfig,
    tokenizer=None,
) -> DatasetDict:
    """
    Load a preference dataset for DPO / RLHF training.

    Supports:
    - Hub datasets with known adapters (UltraFeedback, Anthropic HH-RLHF)
    - Any Hub dataset in TRL v1.0 DPO format (chosen, rejected as conversations)
    - Any Hub dataset with explicit (prompt, chosen, rejected) columns
    - Local JSONL files via ``load_preference_dataset_from_jsonl``

    Returns
    -------
    DatasetDict with ``train`` and ``eval`` splits.
    Each example has at minimum ``chosen``, ``rejected`` columns.
    May also include ``prompt`` if the format provides it.
    """
    dataset_name = data_config.dataset_name
    logger.info(f"Loading preference dataset: {dataset_name}")

    # Check if it's a local file path
    if Path(dataset_name).exists():
        return load_preference_dataset_from_jsonl(
            dataset_name,
            eval_split_ratio=data_config.eval_split_ratio,
        )

    # --- Load from Hub ---
    ds = load_dataset(dataset_name, split=data_config.dataset_split)
    assert isinstance(ds, Dataset)

    # --- Subsample ---
    if data_config.max_samples and data_config.max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(data_config.max_samples))
        logger.info(f"Subsampled to {data_config.max_samples} examples")

    # --- Adapt to standard format ---
    adapter = _ADAPTERS.get(dataset_name)
    if adapter:
        logger.info(f"Applying dataset adapter for: {dataset_name}")
        ds = ds.map(
            adapter,
            num_proc=data_config.preprocessing_num_workers,
            desc="Adapting dataset format",
        )

    # --- Validate required columns ---
    # TRL v1.0 DPO format requires at minimum chosen + rejected.
    # prompt is optional (DPOTrainer extracts it from conversations).
    required = {"chosen", "rejected"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {ds.column_names}. "
            "Consider writing a custom adapter."
        )

    # Remove score columns if present (not needed for training)
    remove_cols = [c for c in ds.column_names
                   if c.startswith("score_") or c in ("source", "category")]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
        logger.info(f"Removed unused columns: {remove_cols}")

    # --- Train / eval split ---
    split = ds.train_test_split(test_size=data_config.eval_split_ratio, seed=42)
    result = DatasetDict({"train": split["train"], "eval": split["test"]})

    # --- Stats ---
    logger.info(
        f"Preference dataset ready — "
        f"train: {len(result['train'])}, eval: {len(result['eval'])} "
        f"columns: {result['train'].column_names}"
    )
    log_dataset_sample(result["train"])

    return result


def load_preference_dataset_from_jsonl(
    path: str,
    *,
    eval_split_ratio: float = 0.05,
) -> DatasetDict:
    """
    Load a custom preference dataset from a local JSONL file.

    Each line should be a JSON object with:
        {"prompt": "...", "chosen": "...", "rejected": "..."}

    Responses can be plain strings or conversational format.
    """
    path = Path(path)
    logger.info(f"Loading custom preference dataset from: {path}")

    ds = load_dataset("json", data_files=str(path), split="train")
    assert isinstance(ds, Dataset)

    # Normalise to conversational format
    ds = ds.map(
        format_preference_example,
        desc="Normalizing to conversational format",
    )

    # Validate
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"Custom dataset missing columns: {missing}")

    split = ds.train_test_split(test_size=eval_split_ratio, seed=42)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def build_preference_dataset_from_pairs(
    pairs: list[dict[str, str]],
    output_path: str | Path,
) -> Path:
    """
    Build a preference JSONL file from a list of dicts.

    Each dict must have: ``prompt``, ``chosen``, ``rejected``.

    Parameters
    ----------
    pairs : list of dict
        Preference pairs.
    output_path : str or Path
        Where to save the JSONL file.

    Returns
    -------
    Path to the created JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs:
            for key in ("prompt", "chosen", "rejected"):
                if key not in pair:
                    raise ValueError(f"Pair missing required key '{key}': {pair}")
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(pairs)} preference pairs to {output_path}")
    return output_path
