"""Tests for configuration loading."""

import pytest
import tempfile
from pathlib import Path

from llenvs.core.config import (
    EvalConfig,
    EnvironmentConfig,
    ModelConfig,
    InferenceConfig,
    create_sampling_params,
)


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_defaults(self):
        """Test default values."""
        config = EnvironmentConfig(name="test_dataset")

        assert config.name == "test_dataset"
        assert config.adapter == "reasoning_gym"
        assert config.size is None
        assert config.seed is None
        assert config.extractor == "tag_based"
        assert config.extractor_config == {}
        assert config.params == {}

    def test_full_config(self):
        """Test with all values set."""
        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=100,
            seed=42,
            extractor="regex",
            extractor_config={"pattern": r"(\d+)"},
            params={"difficulty": "hard"},
        )

        assert config.size == 100
        assert config.seed == 42
        assert config.extractor_config["pattern"] == r"(\d+)"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ModelConfig()

        assert config.backend == "vllm"
        assert config.model == ""
        assert config.params == {}

    def test_full_config(self):
        """Test with all values set."""
        config = ModelConfig(
            backend="openai",
            model="gpt-4o",
            params={"api_key": "sk-test"},
        )

        assert config.backend == "openai"
        assert config.model == "gpt-4o"


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = InferenceConfig()

        assert config.temperature == 0.0
        assert config.max_tokens == 2048
        assert config.top_p == 1.0
        assert config.top_k == 0
        assert config.stop_sequences == []

    def test_full_config(self):
        """Test with all values set."""
        config = InferenceConfig(
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            top_k=50,
            stop_sequences=["STOP", "END"],
        )

        assert config.temperature == 0.7
        assert config.stop_sequences == ["STOP", "END"]


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)

        assert len(config.environments) == 1
        assert config.environments[0].name == "test"
        assert config.model.model == "gpt-4o"

    def test_from_dict_full(self):
        """Test creating config from full dict."""
        data = {
            "environments": [
                {
                    "name": "leg_counting",
                    "adapter": "reasoning_gym",
                    "size": 100,
                    "seed": 42,
                    "extractor": "tag_based",
                    "params": {"difficulty": "easy"},
                },
                {
                    "name": "arithmetic",
                    "size": 50,
                },
            ],
            "model": {
                "backend": "openai",
                "model": "gpt-4o",
                "params": {"temperature": 0},
            },
            "inference": {
                "temperature": 0.0,
                "max_tokens": 1024,
                "stop_sequences": ["</answer>"],
            },
            "system_prompt": "Be helpful.",
            "output_dir": "./results",
            "limit": 10,
            "save_detailed_results": False,
        }
        config = EvalConfig.from_dict(data)

        assert len(config.environments) == 2
        assert config.environments[0].size == 100
        assert config.environments[1].name == "arithmetic"
        assert config.model.backend == "openai"
        assert config.inference.max_tokens == 1024
        assert config.system_prompt == "Be helpful."
        assert config.limit == 10
        assert config.save_detailed_results is False

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
environments:
  - name: test_dataset
    size: 50
    seed: 123

model:
  backend: openai
  model: gpt-4o

inference:
  temperature: 0.5
  max_tokens: 512

output_dir: ./test_results
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)

        assert config.environments[0].name == "test_dataset"
        assert config.environments[0].size == 50
        assert config.model.model == "gpt-4o"
        assert config.inference.temperature == 0.5
        assert config.output_dir == "./test_results"

        # Cleanup
        Path(f.name).unlink()

    def test_to_dict(self):
        """Test converting config to dict."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test", size=100)],
            model=ModelConfig(backend="openai", model="gpt-4o"),
            inference=InferenceConfig(temperature=0.5),
            system_prompt="Test prompt",
            output_dir="./output",
            limit=50,
        )

        data = config.to_dict()

        assert data["environments"][0]["name"] == "test"
        assert data["environments"][0]["size"] == 100
        assert data["model"]["backend"] == "openai"
        assert data["inference"]["temperature"] == 0.5
        assert data["system_prompt"] == "Test prompt"
        assert data["limit"] == 50

    def test_roundtrip(self):
        """Test dict -> config -> dict roundtrip."""
        original = {
            "environments": [
                {"name": "test1", "size": 100, "seed": 42},
                {"name": "test2", "adapter": "reasoning_gym"},
            ],
            "model": {"backend": "vllm", "model": "llama"},
            "inference": {"temperature": 0.7, "max_tokens": 1024},
            "system_prompt": "Hello",
            "output_dir": "./out",
            "limit": 10,
            "save_detailed_results": True,
        }

        config = EvalConfig.from_dict(original)
        result = config.to_dict()

        assert result["environments"][0]["name"] == "test1"
        assert result["environments"][0]["size"] == 100
        assert result["model"]["backend"] == "vllm"
        assert result["inference"]["temperature"] == 0.7


class TestCreateSamplingParams:
    """Tests for create_sampling_params."""

    def test_basic(self):
        """Test basic sampling params creation."""
        inference_config = InferenceConfig(
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9,
        )
        params = create_sampling_params(inference_config)

        assert params.temperature == 0.5
        assert params.max_tokens == 1024
        assert params.top_p == 0.9

    def test_stop_sequences(self):
        """Test stop sequences conversion."""
        inference_config = InferenceConfig(
            stop_sequences=["STOP", "END"],
        )
        params = create_sampling_params(inference_config)

        assert params.stop_sequences == ("STOP", "END")

    def test_defaults(self):
        """Test with default inference config."""
        inference_config = InferenceConfig()
        params = create_sampling_params(inference_config)

        assert params.temperature == 0.0
        assert params.max_tokens == 2048
        assert params.top_p == 1.0
        assert params.top_k == 0
        assert params.extra == {}

    def test_extra_params(self):
        """Test extra params are passed through."""
        inference_config = InferenceConfig(
            temperature=0.7,
            extra={
                "repetition_penalty": 1.2,
                "num_beams": 4,
            },
        )
        params = create_sampling_params(inference_config)

        assert params.temperature == 0.7
        assert params.extra == {"repetition_penalty": 1.2, "num_beams": 4}
