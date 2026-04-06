"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from concordlm.config import (
    PipelineConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    load_config,
    config_to_dict,
    _deep_merge,
    _coerce_value,
    _apply_overrides,
)


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"model": {"name": "gpt2", "dtype": "float32"}}
        override = {"model": {"name": "mistral", "quantization": "4bit"}}
        result = _deep_merge(base, override)
        assert result["model"]["name"] == "mistral"
        assert result["model"]["dtype"] == "float32"
        assert result["model"]["quantization"] == "4bit"

    def test_does_not_mutate_original(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1


class TestCoerceValue:
    def test_bool_true(self):
        assert _coerce_value("true") is True
        assert _coerce_value("True") is True

    def test_bool_false(self):
        assert _coerce_value("false") is False

    def test_null(self):
        assert _coerce_value("null") is None
        assert _coerce_value("none") is None

    def test_int(self):
        assert _coerce_value("42") == 42

    def test_float(self):
        assert _coerce_value("5e-7") == 5e-7
        assert _coerce_value("0.1") == 0.1

    def test_string(self):
        assert _coerce_value("hello") == "hello"


class TestApplyOverrides:
    def test_simple_override(self):
        cfg = {"model": {"name": "gpt2"}}
        result = _apply_overrides(cfg, ["model.name=mistral"])
        assert result["model"]["name"] == "mistral"

    def test_nested_override(self):
        cfg = {"training": {"learning_rate": 0.001}}
        result = _apply_overrides(cfg, ["training.learning_rate=5e-7"])
        assert result["training"]["learning_rate"] == 5e-7

    def test_creates_missing_keys(self):
        cfg = {}
        result = _apply_overrides(cfg, ["new.key=value"])
        assert result["new"]["key"] == "value"


class TestLoadConfig:
    def test_load_simple_yaml(self, tmp_path):
        config_data = {
            "stage": "sft",
            "model": {"name": "test-model", "quantization": "none"},
            "training": {"output_dir": str(tmp_path / "output"), "learning_rate": 1e-4},
        }
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert isinstance(config, PipelineConfig)
        assert config.stage == "sft"
        assert config.model.name == "test-model"
        assert config.training.learning_rate == 1e-4

    def test_load_with_inheritance(self, tmp_path):
        # Base config
        base_data = {"model": {"name": "base-model", "dtype": "float32"}}
        base_file = tmp_path / "base.yaml"
        with open(base_file, "w") as f:
            yaml.dump(base_data, f)

        # Child config
        child_data = {
            "inherit": "base.yaml",
            "stage": "dpo",
            "model": {"name": "child-model"},
            "training": {"output_dir": str(tmp_path / "output")},
        }
        child_file = tmp_path / "child.yaml"
        with open(child_file, "w") as f:
            yaml.dump(child_data, f)

        config = load_config(child_file)
        assert config.model.name == "child-model"
        assert config.model.dtype == "float32"  # Inherited from base
        assert config.stage == "dpo"

    def test_load_with_overrides(self, tmp_path):
        config_data = {
            "model": {"name": "original"},
            "training": {"output_dir": str(tmp_path / "output"), "max_steps": 100},
        }
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file, overrides=["model.name=overridden", "training.max_steps=5"])
        assert config.model.name == "overridden"
        assert config.training.max_steps == 5

    def test_creates_output_dir(self, tmp_path):
        output_dir = tmp_path / "new_output"
        config_data = {"training": {"output_dir": str(output_dir)}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        load_config(config_file)
        assert output_dir.exists()


class TestConfigToDict:
    def test_roundtrip(self):
        config = PipelineConfig()
        d = config_to_dict(config)
        assert isinstance(d, dict)
        assert d["stage"] == "sft"
        assert d["model"]["name"] == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_contains_all_sections(self):
        config = PipelineConfig()
        d = config_to_dict(config)
        assert "model" in d
        assert "lora" in d
        assert "training" in d
        assert "data" in d
        assert "dpo" in d
