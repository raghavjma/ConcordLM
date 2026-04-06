"""Tests for data utilities and dataset loaders."""

import json
import tempfile
from pathlib import Path

import pytest

from concordlm.data.utils import format_preference_example


class TestFormatPreferenceExample:
    def test_plain_text_format(self):
        example = {
            "prompt": "What is 2+2?",
            "chosen": "The answer is 4.",
            "rejected": "It's 5.",
        }
        result = format_preference_example(example)
        assert result["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert result["chosen"] == [{"role": "assistant", "content": "The answer is 4."}]
        assert result["rejected"] == [{"role": "assistant", "content": "It's 5."}]

    def test_conversational_format_passthrough(self):
        example = {
            "prompt": [{"role": "user", "content": "Hello"}],
            "chosen": [{"role": "assistant", "content": "Hi there!"}],
            "rejected": [{"role": "assistant", "content": "Go away."}],
        }
        result = format_preference_example(example)
        assert result["prompt"] == example["prompt"]
        assert result["chosen"] == example["chosen"]

    def test_instruction_format(self):
        example = {
            "instruction": "Write a poem",
            "input": "about the sea",
            "chosen": "The waves crash...",
            "rejected": "Water is wet.",
        }
        result = format_preference_example(example)
        assert "Write a poem" in result["prompt"][0]["content"]
        assert "about the sea" in result["prompt"][0]["content"]

    def test_missing_chosen_raises(self):
        example = {"prompt": "Hello", "rejected": "Bad"}
        with pytest.raises(ValueError, match="chosen"):
            format_preference_example(example)

    def test_missing_rejected_raises(self):
        example = {"prompt": "Hello", "chosen": "Good"}
        with pytest.raises(ValueError, match="rejected"):
            format_preference_example(example)

    def test_missing_prompt_raises(self):
        example = {"chosen": "Good", "rejected": "Bad"}
        with pytest.raises(ValueError, match="prompt"):
            format_preference_example(example)


class TestBuildPreferenceDataset:
    def test_build_from_pairs(self, tmp_path):
        from concordlm.data.preference_dataset import build_preference_dataset_from_pairs

        pairs = [
            {"prompt": "Q1", "chosen": "Good1", "rejected": "Bad1"},
            {"prompt": "Q2", "chosen": "Good2", "rejected": "Bad2"},
        ]
        output_path = tmp_path / "test.jsonl"
        result = build_preference_dataset_from_pairs(pairs, output_path)

        assert result.exists()
        with open(result) as f:
            lines = f.readlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["prompt"] == "Q1"

    def test_build_missing_key_raises(self, tmp_path):
        from concordlm.data.preference_dataset import build_preference_dataset_from_pairs

        pairs = [{"prompt": "Q1", "chosen": "Good1"}]  # Missing rejected
        output_path = tmp_path / "test.jsonl"
        with pytest.raises(ValueError, match="rejected"):
            build_preference_dataset_from_pairs(pairs, output_path)


class TestResponseQualityEval:
    """Test the heuristic quality scoring."""

    def test_short_response_penalized(self):
        from concordlm.eval.quality import evaluate_response_quality
        scores = evaluate_response_quality("Yes.")
        assert scores["length_score"] < 0.5

    def test_good_length_response(self):
        from concordlm.eval.quality import evaluate_response_quality
        text = " ".join(["word"] * 100)  # 100 words
        scores = evaluate_response_quality(text + ".")
        assert scores["length_score"] >= 0.8

    def test_informativeness_high_for_diverse(self):
        from concordlm.eval.quality import evaluate_response_quality
        text = "The quick brown fox jumps over the lazy dog near the riverbank."
        scores = evaluate_response_quality(text)
        assert scores["informativeness"] > 0.7

    def test_informativeness_low_for_repetitive(self):
        from concordlm.eval.quality import evaluate_response_quality
        text = "word word word word word word word word word word."
        scores = evaluate_response_quality(text)
        assert scores["informativeness"] < 0.3


class TestSafetyEval:
    def test_refusal_detection_positive(self):
        from concordlm.eval.safety import detect_refusal
        assert detect_refusal("I'm sorry, but I can't help with that request.")
        assert detect_refusal("I cannot assist with illegal activities.")
        assert detect_refusal("As an AI, I must decline this request.")

    def test_refusal_detection_negative(self):
        from concordlm.eval.safety import detect_refusal
        assert not detect_refusal("Sure, here is the information you requested.")
        assert not detect_refusal("The capital of France is Paris.")
