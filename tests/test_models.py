"""Tests for model loading utilities (mocked, no GPU needed)."""

import pytest

from concordlm.config import ModelConfig, LoRAConfig


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.quantization == "4bit"
        assert config.dtype == "bfloat16"

    def test_custom_values(self):
        config = ModelConfig(name="test-model", quantization="none")
        assert config.name == "test-model"
        assert config.quantization == "none"


class TestLoRAConfig:
    def test_defaults(self):
        config = LoRAConfig()
        assert config.r == 16
        assert config.alpha == 32
        assert "q_proj" in config.target_modules

    def test_custom_modules(self):
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert len(config.target_modules) == 2


class TestBnBConfig:
    def test_4bit_config(self):
        from concordlm.models.loader import _build_bnb_config

        model_config = ModelConfig(quantization="4bit", dtype="bfloat16")
        bnb = _build_bnb_config(model_config)
        assert bnb is not None
        assert bnb.load_in_4bit is True
        assert bnb.bnb_4bit_quant_type == "nf4"
        assert bnb.bnb_4bit_use_double_quant is True

    def test_8bit_config(self):
        from concordlm.models.loader import _build_bnb_config

        model_config = ModelConfig(quantization="8bit")
        bnb = _build_bnb_config(model_config)
        assert bnb is not None
        assert bnb.load_in_8bit is True

    def test_no_quantization(self):
        from concordlm.models.loader import _build_bnb_config

        model_config = ModelConfig(quantization="none")
        bnb = _build_bnb_config(model_config)
        assert bnb is None


class TestLoRAConfigBuilding:
    def test_builds_peft_config(self):
        from concordlm.models.loader import _build_lora_config

        lora_dc = LoRAConfig(r=8, alpha=16, dropout=0.1)
        peft_config = _build_lora_config(lora_dc)

        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16
        assert peft_config.lora_dropout == 0.1
        assert peft_config.bias == "none"


class TestDtypeMapping:
    def test_valid_dtypes(self):
        from concordlm.models.loader import _get_torch_dtype
        import torch

        assert _get_torch_dtype("float32") == torch.float32
        assert _get_torch_dtype("float16") == torch.float16
        assert _get_torch_dtype("bfloat16") == torch.bfloat16

    def test_invalid_dtype_raises(self):
        from concordlm.models.loader import _get_torch_dtype

        with pytest.raises(ValueError, match="Unknown dtype"):
            _get_torch_dtype("int8")
