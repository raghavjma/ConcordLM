"""
Reward Model utilities for the RLHF pipeline.

The reward model is a sequence classification model with a scalar output head.
It is trained on preference pairs using TRL's RewardTrainer.
"""

from __future__ import annotations

import logging

from concordlm.config import ModelConfig, LoRAConfig
from concordlm.models.loader import load_reward_model

logger = logging.getLogger(__name__)


def get_reward_model(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
):
    """
    Convenience wrapper to load a reward model ready for RewardTrainer.

    Returns (model, tokenizer, peft_config).
    """
    return load_reward_model(model_config, lora_config, num_labels=1)
