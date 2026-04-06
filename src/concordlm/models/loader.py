"""
Model Loader — load base models with LoRA / QLoRA configuration.

Supports:
- Full precision loading (for evaluation / reference)
- LoRA adapter training
- QLoRA 4-bit adapter training (via bitsandbytes)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from concordlm.config import LoRAConfig as LoRAConfigDC
from concordlm.config import ModelConfig

logger = logging.getLogger(__name__)

# Map string dtype names to torch dtypes
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _get_torch_dtype(name: str) -> torch.dtype:
    """Resolve a string dtype name to a torch.dtype."""
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {name}. Choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _build_bnb_config(model_config: ModelConfig) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytesConfig for quantized loading, or None."""
    if model_config.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_get_torch_dtype(model_config.dtype),
            bnb_4bit_use_double_quant=True,
        )
    elif model_config.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _build_lora_config(lora_dc: LoRAConfigDC) -> LoraConfig:
    """Convert our dataclass LoRA config to a PEFT LoraConfig."""
    return LoraConfig(
        r=lora_dc.r,
        lora_alpha=lora_dc.alpha,
        lora_dropout=lora_dc.dropout,
        bias=lora_dc.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_dc.target_modules,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    padding_side: str = "left",
) -> PreTrainedTokenizerBase:
    """
    Load a tokenizer and configure it for training.

    Sets padding side to ``left`` (required by TRL trainers) and ensures
    a pad_token is defined.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    return tokenizer


def load_model_for_training(
    model_config: ModelConfig,
    lora_config: LoRAConfigDC,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, LoraConfig]:
    """
    Load a causal LM with optional quantization and LoRA config.

    Returns
    -------
    (model, tokenizer, peft_config)
        - model: the base model (not yet wrapped with PEFT — trainers do this)
        - tokenizer: configured tokenizer
        - peft_config: PEFT LoraConfig to pass to the trainer
    """
    logger.info(f"Loading model: {model_config.name}")
    logger.info(f"  Quantization: {model_config.quantization}")
    logger.info(f"  dtype: {model_config.dtype}")
    logger.info(f"  LoRA r={lora_config.r}, alpha={lora_config.alpha}")

    tokenizer = load_tokenizer(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Build quantization config
    bnb_config = _build_bnb_config(model_config)

    # Model kwargs
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = _get_torch_dtype(model_config.dtype)

    if model_config.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        **model_kwargs,
    )

    # Prepare for k-bit training (gradient checkpointing compat)
    if model_config.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # Build PEFT config
    peft_config = _build_lora_config(lora_config)

    # Log trainable params
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    return model, tokenizer, peft_config


def load_model_for_inference(
    model_path: str,
    *,
    merge_adapter: bool = True,
    dtype: str = "bfloat16",
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a trained model (with LoRA adapter) for inference.

    Parameters
    ----------
    model_path : str
        Path to the saved model / adapter checkpoint.
    merge_adapter : bool
        If True, merge LoRA weights into the base model for faster inference.
    dtype : str
        Compute dtype.
    device : str
        Device map strategy.
    """
    from peft import AutoPeftModelForCausalLM

    logger.info(f"Loading model for inference from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_get_torch_dtype(dtype),
        device_map=device,
    )

    if merge_adapter:
        logger.info("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def load_reward_model(
    model_config: ModelConfig,
    lora_config: LoRAConfigDC,
    num_labels: int = 1,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, LoraConfig]:
    """
    Load a sequence classification model for reward model training.

    Uses AutoModelForSequenceClassification with a scalar output head.
    """
    logger.info(f"Loading reward model: {model_config.name}")

    tokenizer = load_tokenizer(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
    )

    bnb_config = _build_bnb_config(model_config)

    model_kwargs: dict[str, Any] = {
        "num_labels": num_labels,
        "trust_remote_code": model_config.trust_remote_code,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = _get_torch_dtype(model_config.dtype)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.name,
        **model_kwargs,
    )

    # Configure pad token for classification
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = _build_lora_config(lora_config)

    return model, tokenizer, peft_config
