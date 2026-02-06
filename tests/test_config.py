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


class TestExtractorsChainConfig:
    """Tests for the extractors chain in EnvironmentConfig."""

    def test_extractors_field_default_none(self):
        """Test that extractors defaults to None."""
        config = EnvironmentConfig(name="test")
        assert config.extractors is None

    def test_extractors_field_with_list(self):
        """Test extractors field with a list."""
        config = EnvironmentConfig(
            name="test",
            extractors=[
                {"type": "tag_based", "config": {"tag_name": "answer"}},
                {"type": "numeric"},
            ],
        )
        assert config.extractors is not None
        assert len(config.extractors) == 2
        assert config.extractors[0]["type"] == "tag_based"

    def test_single_extractor_shorthand(self):
        """Test single extractor shorthand (extractor field)."""
        config = EnvironmentConfig(
            name="test",
            extractor="gsm8k",
            extractor_config={"strip_whitespace": True},
        )
        assert config.extractor == "gsm8k"
        assert config.extractors is None

    def test_from_dict_with_extractors(self):
        """Test EvalConfig.from_dict() parses extractors."""
        data = {
            "environments": [
                {
                    "name": "polynomial_equations",
                    "adapter": "reasoning_gym",
                    "extractors": [
                        {"type": "tag_based", "config": {"tag_name": "answer"}},
                        {"type": "pattern_answer"},
                        {"type": "numeric"},
                    ],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)

        env = config.environments[0]
        assert env.extractors is not None
        assert len(env.extractors) == 3
        assert env.extractors[0]["type"] == "tag_based"
        assert env.extractors[1]["type"] == "pattern_answer"
        assert env.extractors[2]["type"] == "numeric"

    def test_from_dict_without_extractors(self):
        """Test EvalConfig.from_dict() with single extractor shorthand."""
        data = {
            "environments": [
                {
                    "name": "test",
                    "extractor": "gsm8k",
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)

        env = config.environments[0]
        assert env.extractors is None
        assert env.extractor == "gsm8k"

    def test_to_dict_with_extractors(self):
        """Test to_dict() serializes extractors."""
        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    extractors=[
                        {"type": "tag_based"},
                        {"type": "numeric"},
                    ],
                )
            ],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()

        env_data = data["environments"][0]
        assert "extractors" in env_data
        assert len(env_data["extractors"]) == 2
        assert env_data["extractors"][0]["type"] == "tag_based"

    def test_to_dict_without_extractors(self):
        """Test to_dict() omits extractors when None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()

        env_data = data["environments"][0]
        assert "extractors" not in env_data

    def test_cleaners_default_none(self):
        """Test that pre_cleaners/post_cleaners default to None."""
        config = EnvironmentConfig(name="test")
        assert config.pre_cleaners is None
        assert config.post_cleaners is None

    def test_cleaners_explicit_empty(self):
        """Test explicit empty list disables cleaning."""
        config = EnvironmentConfig(name="test", pre_cleaners=[], post_cleaners=[])
        assert config.pre_cleaners == []
        assert config.post_cleaners == []

    def test_cleaners_specific_names(self):
        """Test specific cleaner names."""
        config = EnvironmentConfig(
            name="test",
            pre_cleaners=["strip_special_tokens"],
            post_cleaners=["strip_trailing_punctuation", "strip_surrounding_quotes"],
        )
        assert config.pre_cleaners == ["strip_special_tokens"]
        assert len(config.post_cleaners) == 2

    def test_from_dict_with_cleaners(self):
        """Test from_dict parses cleaners."""
        data = {
            "environments": [
                {
                    "name": "test",
                    "pre_cleaners": ["strip_special_tokens"],
                    "post_cleaners": [],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.pre_cleaners == ["strip_special_tokens"]
        assert env.post_cleaners == []

    def test_from_dict_without_cleaners(self):
        """Test from_dict defaults cleaners to None."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.pre_cleaners is None
        assert env.post_cleaners is None

    def test_to_dict_with_cleaners(self):
        """Test to_dict serializes cleaners."""
        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    pre_cleaners=["strip_special_tokens"],
                    post_cleaners=["strip_trailing_punctuation"],
                )
            ],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        env_data = data["environments"][0]
        assert env_data["pre_cleaners"] == ["strip_special_tokens"]
        assert env_data["post_cleaners"] == ["strip_trailing_punctuation"]

    def test_to_dict_without_cleaners(self):
        """Test to_dict omits cleaners when None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        env_data = data["environments"][0]
        assert "pre_cleaners" not in env_data
        assert "post_cleaners" not in env_data

    def test_roundtrip_with_cleaners(self):
        """Test roundtrip with cleaners."""
        original = {
            "environments": [
                {
                    "name": "test",
                    "pre_cleaners": ["strip_special_tokens"],
                    "post_cleaners": ["strip_trailing_punctuation", "strip_surrounding_quotes"],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()
        assert result["environments"][0]["pre_cleaners"] == ["strip_special_tokens"]
        assert result["environments"][0]["post_cleaners"] == [
            "strip_trailing_punctuation",
            "strip_surrounding_quotes",
        ]

    def test_roundtrip_with_extractors(self):
        """Test roundtrip with extractors chain."""
        original = {
            "environments": [
                {
                    "name": "test",
                    "extractors": [
                        {"type": "boxed"},
                        {"type": "numeric"},
                        {"type": "fallback"},
                    ],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()

        assert result["environments"][0]["extractors"] == original["environments"][0]["extractors"]


class TestPromptsConfig:
    """Tests for the prompts field in EnvironmentConfig."""

    def test_prompts_default_none(self):
        """Test that prompts defaults to None."""
        config = EnvironmentConfig(name="test")
        assert config.prompts is None

    def test_prompts_with_dict(self):
        """Test prompts field with a dict."""
        config = EnvironmentConfig(
            name="webshop",
            prompts={"action_hint": "Custom hint.", "step_format": "Turn {step}:"},
        )
        assert config.prompts is not None
        assert config.prompts["action_hint"] == "Custom hint."
        assert config.prompts["step_format"] == "Turn {step}:"

    def test_from_dict_with_prompts(self):
        """Test EvalConfig.from_dict() parses prompts."""
        data = {
            "environments": [
                {
                    "name": "webshop",
                    "adapter": "webshop",
                    "prompts": {
                        "action_hint": "Navigate using search[q] or click[e].",
                        "instruction_prefix": "Your goal: {instruction}",
                    },
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.prompts is not None
        assert env.prompts["action_hint"] == "Navigate using search[q] or click[e]."

    def test_from_dict_without_prompts(self):
        """Test from_dict defaults prompts to None."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.prompts is None

    def test_to_dict_with_prompts(self):
        """Test to_dict serializes prompts."""
        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="webshop",
                    prompts={"action_hint": "Custom."},
                )
            ],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        env_data = data["environments"][0]
        assert "prompts" in env_data
        assert env_data["prompts"]["action_hint"] == "Custom."

    def test_to_dict_without_prompts(self):
        """Test to_dict omits prompts when None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        env_data = data["environments"][0]
        assert "prompts" not in env_data

    def test_roundtrip_with_prompts(self):
        """Test roundtrip with prompts."""
        original = {
            "environments": [
                {
                    "name": "webshop",
                    "prompts": {
                        "action_hint": "Custom hint.",
                        "step_format": "Turn {step}:",
                    },
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()
        assert result["environments"][0]["prompts"] == original["environments"][0]["prompts"]
