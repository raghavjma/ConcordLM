"""
ConcordLM Configuration System

Loads YAML configs with inheritance support, validates parameters,
and provides typed dataclass access.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses — typed representation of every config section
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model loading parameters."""

    name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization: str = "4bit"  # "none" | "4bit" | "8bit"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    max_seq_length: int = 2048


@dataclass
class LoRAConfig:
    """LoRA / QLoRA adapter parameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    """Training hyper-parameters (shared across stages)."""

    output_dir: str = "./outputs"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 100
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    report_to: str = "tensorboard"
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Dataset parameters."""

    dataset_name: str = "trl-lib/ultrafeedback_binarized"
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    eval_split_ratio: float = 0.05
    preprocessing_num_workers: int = 4
    max_seq_length: int = 2048
    text_field: Optional[str] = None
    packing: bool = False


@dataclass
class DPOSpecificConfig:
    """DPO-specific hyper-parameters."""

    beta: float = 0.1
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    max_length: int = 1024
    max_prompt_length: int = 512


@dataclass
class RewardModelConfig:
    """Reward model training parameters (RLHF)."""

    output_dir: str = "./outputs/reward_model"
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    max_length: int = 512


@dataclass
class PPOSpecificConfig:
    """PPO-specific hyper-parameters (RLHF)."""

    learning_rate: float = 1e-6
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    kl_penalty: str = "kl"
    init_kl_coef: float = 0.2
    adap_kl_ctrl: bool = True
    target_kl: float = 6.0
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class PipelineConfig:
    """Top-level config container for the full ConcordLM pipeline."""

    stage: str = "sft"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dpo: DPOSpecificConfig = field(default_factory=DPOSpecificConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    ppo: PPOSpecificConfig = field(default_factory=PPOSpecificConfig)
    sft_model_path: Optional[str] = None
    dpo_model_path: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML loading with inheritance
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_with_inheritance(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving ``inherit:`` directives."""
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    parent_name = raw.pop("inherit", None)
    if parent_name:
        parent_path = path.parent / parent_name
        if not parent_path.exists():
            raise FileNotFoundError(
                f"Inherited config not found: {parent_path} (from {path})"
            )
        parent = _load_yaml_with_inheritance(parent_path)
        raw = _deep_merge(parent, raw)

    return raw


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation CLI overrides, e.g. ``model.name=Qwen/Qwen2.5-0.5B``."""
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (missing '='): {override}")
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the parent dict
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        # Coerce the value to the right Python type
        d[keys[-1]] = _coerce_value(value_str)

    return cfg


def _coerce_value(v: str) -> Any:
    """Best-effort type coercion for CLI override values."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "null" or v.lower() == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _dict_to_dataclass(cls, data: dict) -> Any:
    """Recursively convert a dict into nested dataclasses."""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue  # Skip unknown keys gracefully
        ftype = field_types[key]
        # Resolve the actual type for nested dataclasses
        actual_type = _resolve_type(ftype)
        if actual_type and hasattr(actual_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(actual_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_type(type_hint) -> type | None:
    """Resolve a type hint string or type object to its actual class."""
    type_map = {
        "ModelConfig": ModelConfig,
        "LoRAConfig": LoRAConfig,
        "TrainingConfig": TrainingConfig,
        "DataConfig": DataConfig,
        "DPOSpecificConfig": DPOSpecificConfig,
        "RewardModelConfig": RewardModelConfig,
        "PPOSpecificConfig": PPOSpecificConfig,
    }
    if isinstance(type_hint, str):
        return type_map.get(type_hint)
    if isinstance(type_hint, type):
        return type_hint if hasattr(type_hint, "__dataclass_fields__") else None
    # Handle typing generics (Optional, etc.)
    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        args = getattr(type_hint, "__args__", ())
        for arg in args:
            resolved = _resolve_type(arg)
            if resolved:
                return resolved
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> PipelineConfig:
    """
    Load a pipeline config from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the stage-specific YAML config file.
    overrides : list of str, optional
        Dot-notation overrides, e.g. ``["model.name=...", "training.max_steps=5"]``.

    Returns
    -------
    PipelineConfig
        Fully resolved, typed configuration.
    """
    raw = _load_yaml_with_inheritance(config_path)
    if overrides:
        raw = _apply_overrides(raw, overrides)

    config = _dict_to_dataclass(PipelineConfig, raw)

    # Ensure output directory exists
    os.makedirs(config.training.output_dir, exist_ok=True)

    return config


def config_to_dict(config) -> dict[str, Any]:
    """Serialize a dataclass config back to a plain dict (for logging)."""
    from dataclasses import asdict

    return asdict(config)
