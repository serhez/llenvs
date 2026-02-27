"""Tests for configuration loading."""

import tempfile
from pathlib import Path

from llenvs.core.config import (
    EnvironmentConfig,
    EvalConfig,
    InferenceConfig,
    ModelConfig,
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
        assert config.answer_extractor == "tag_based"
        assert config.answer_extractor_config == {}
        assert config.params == {}

    def test_full_config(self):
        """Test with all values set."""
        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=100,
            seed=42,
            answer_extractor="regex",
            answer_extractor_config={"pattern": r"(\d+)"},
            params={"difficulty": "hard"},
        )

        assert config.size == 100
        assert config.seed == 42
        assert config.answer_extractor_config["pattern"] == r"(\d+)"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ModelConfig()

        assert config.backend == "vllm"
        assert config.model == ""
        assert config.params == {}
        assert config.max_concurrency == 64

    def test_full_config(self):
        """Test with all values set."""
        config = ModelConfig(
            backend="openai",
            model="gpt-4o",
            params={"api_key": "sk-test"},
        )

        assert config.backend == "openai"
        assert config.model == "gpt-4o"

    def test_max_concurrency_custom(self):
        """Test custom max_concurrency."""
        config = ModelConfig(backend="openai", model="gpt-4o", max_concurrency=16)
        assert config.max_concurrency == 16


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

    def test_batch_size_default(self):
        """Test batch_size defaults to None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(),
        )
        assert config.batch_size is None

    def test_batch_size_custom(self):
        """Test custom batch_size."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(),
            batch_size=32,
        )
        assert config.batch_size == 32

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
                    "answer_extractor": "tag_based",
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

    def test_from_dict_max_concurrency(self):
        """Test from_dict parses max_concurrency from model section."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o", "max_concurrency": 16},
        }
        config = EvalConfig.from_dict(data)
        assert config.model.max_concurrency == 16

    def test_from_dict_max_concurrency_default(self):
        """Test from_dict defaults max_concurrency to 64."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)
        assert config.model.max_concurrency == 64

    def test_to_dict_max_concurrency(self):
        """Test to_dict serializes max_concurrency in model section."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(backend="openai", model="gpt-4o", max_concurrency=16),
        )
        data = config.to_dict()
        assert data["model"]["max_concurrency"] == 16

    def test_from_dict_batch_size(self):
        """Test from_dict parses batch_size."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o"},
            "batch_size": 32,
        }
        config = EvalConfig.from_dict(data)
        assert config.batch_size == 32

    def test_from_dict_batch_size_default(self):
        """Test from_dict defaults batch_size to None."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)
        assert config.batch_size is None

    def test_to_dict_batch_size(self):
        """Test to_dict serializes batch_size when set."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(),
            batch_size=32,
        )
        data = config.to_dict()
        assert data["batch_size"] == 32

    def test_to_dict_batch_size_none(self):
        """Test to_dict omits batch_size when None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(),
        )
        data = config.to_dict()
        assert "batch_size" not in data

    def test_roundtrip_with_batch_size_and_max_concurrency(self):
        """Test roundtrip preserves batch_size and max_concurrency."""
        original = {
            "environments": [{"name": "test"}],
            "model": {"backend": "openai", "model": "gpt-4o", "max_concurrency": 16},
            "batch_size": 32,
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()
        assert result["model"]["max_concurrency"] == 16
        assert result["batch_size"] == 32

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

    def test_answer_extractors_field_default_none(self):
        """Test that answer_extractors defaults to None."""
        config = EnvironmentConfig(name="test")
        assert config.answer_extractors is None

    def test_answer_extractors_field_with_list(self):
        """Test answer_extractors field with a list."""
        config = EnvironmentConfig(
            name="test",
            answer_extractors=[
                {"type": "tag_based", "config": {"tag_name": "answer"}},
                {"type": "numeric"},
            ],
        )
        assert config.answer_extractors is not None
        assert len(config.answer_extractors) == 2
        assert config.answer_extractors[0]["type"] == "tag_based"

    def test_single_answer_extractor_shorthand(self):
        """Test single answer_extractor shorthand."""
        config = EnvironmentConfig(
            name="test",
            answer_extractor="gsm8k",
            answer_extractor_config={"strip_whitespace": True},
        )
        assert config.answer_extractor == "gsm8k"
        assert config.answer_extractors is None

    def test_from_dict_with_answer_extractors(self):
        """Test EvalConfig.from_dict() parses answer_extractors."""
        data = {
            "environments": [
                {
                    "name": "polynomial_equations",
                    "adapter": "reasoning_gym",
                    "answer_extractors": [
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
        assert env.answer_extractors is not None
        assert len(env.answer_extractors) == 3
        assert env.answer_extractors[0]["type"] == "tag_based"
        assert env.answer_extractors[1]["type"] == "pattern_answer"
        assert env.answer_extractors[2]["type"] == "numeric"

    def test_from_dict_without_answer_extractors(self):
        """Test EvalConfig.from_dict() with single answer_extractor shorthand."""
        data = {
            "environments": [
                {
                    "name": "test",
                    "answer_extractor": "gsm8k",
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)

        env = config.environments[0]
        assert env.answer_extractors is None
        assert env.answer_extractor == "gsm8k"

    def test_to_dict_with_answer_extractors(self):
        """Test to_dict() serializes answer_extractors."""
        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    answer_extractors=[
                        {"type": "tag_based"},
                        {"type": "numeric"},
                    ],
                )
            ],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()

        env_data = data["environments"][0]
        assert "answer_extractors" in env_data
        assert len(env_data["answer_extractors"]) == 2
        assert env_data["answer_extractors"][0]["type"] == "tag_based"

    def test_to_dict_without_answer_extractors(self):
        """Test to_dict() omits answer_extractors when None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()

        env_data = data["environments"][0]
        assert "answer_extractors" not in env_data

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

    def test_parameterized_cleaners_in_config(self):
        """EnvironmentConfig accepts mixed list[str | dict] for cleaners."""
        config = EnvironmentConfig(
            name="test",
            pre_cleaners=["strip_special_tokens"],
            post_cleaners=[
                "strip_trailing_punctuation",
                {"type": "truncate_tail", "config": {"max_chars": 512}},
            ],
        )
        assert len(config.post_cleaners) == 2
        assert config.post_cleaners[0] == "strip_trailing_punctuation"
        assert config.post_cleaners[1] == {"type": "truncate_tail", "config": {"max_chars": 512}}

    def test_from_dict_parameterized_cleaners(self):
        """from_dict parses parameterized cleaner specs."""
        data = {
            "environments": [
                {
                    "name": "test",
                    "post_cleaners": [
                        "strip_trailing_punctuation",
                        {"type": "truncate_tail", "config": {"max_chars": 100}},
                    ],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert len(env.post_cleaners) == 2
        assert env.post_cleaners[1] == {"type": "truncate_tail", "config": {"max_chars": 100}}

    def test_to_dict_roundtrip_parameterized_cleaners(self):
        """to_dict roundtrips parameterized cleaner specs."""
        original = {
            "environments": [
                {
                    "name": "test",
                    "pre_cleaners": [
                        {"type": "truncate_tail", "config": {"max_chars": 200}},
                    ],
                    "post_cleaners": [
                        "strip_trailing_punctuation",
                        {"type": "truncate_tail", "config": {"max_chars": 512}},
                    ],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()
        assert (
            result["environments"][0]["pre_cleaners"] == original["environments"][0]["pre_cleaners"]
        )
        assert (
            result["environments"][0]["post_cleaners"]
            == original["environments"][0]["post_cleaners"]
        )

    def test_roundtrip_with_answer_extractors(self):
        """Test roundtrip with answer_extractors chain."""
        original = {
            "environments": [
                {
                    "name": "test",
                    "answer_extractors": [
                        {"type": "boxed"},
                        {"type": "numeric"},
                        {"type": "raw"},
                    ],
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()

        assert (
            result["environments"][0]["answer_extractors"]
            == original["environments"][0]["answer_extractors"]
        )


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


class TestInferenceConfigExtra:
    """Tests for InferenceConfig.extra YAML round-trip."""

    def test_from_dict_parses_extra(self):
        """from_dict parses extra from inference section."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o"},
            "inference": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "extra": {"thinking_budget": 512, "repetition_penalty": 1.2},
            },
        }
        config = EvalConfig.from_dict(data)
        assert config.inference.extra == {"thinking_budget": 512, "repetition_penalty": 1.2}

    def test_from_dict_extra_default_empty(self):
        """from_dict defaults extra to empty dict."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "gpt-4o"},
            "inference": {"temperature": 0.5},
        }
        config = EvalConfig.from_dict(data)
        assert config.inference.extra == {}

    def test_to_dict_serializes_extra(self):
        """to_dict includes extra in inference section."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(),
            inference=InferenceConfig(extra={"thinking_budget": 256}),
        )
        data = config.to_dict()
        assert data["inference"]["extra"] == {"thinking_budget": 256}

    def test_roundtrip_with_extra(self):
        """Roundtrip preserves inference.extra."""
        original = {
            "environments": [{"name": "test"}],
            "model": {"model": "test-model"},
            "inference": {
                "temperature": 0.7,
                "extra": {"thinking_budget": 512},
            },
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()
        assert result["inference"]["extra"] == {"thinking_budget": 512}


class TestIterativeConfig:
    """Tests for IterativeConfig in EnvironmentConfig."""

    def test_iterative_default_none(self):
        """Test that iterative defaults to None."""
        config = EnvironmentConfig(name="test")
        assert config.iterative is None

    def test_iterative_config_defaults(self):
        """Test IterativeConfig default values."""
        from llenvs.core.config import IterativeConfig

        ic = IterativeConfig()
        assert ic.max_turns == 3
        assert ic.include_history is True
        assert ic.summarize_history is False
        assert ic.submit_keyword == "SUBMIT"
        assert ic.submission_extractor is None
        assert ic.submission_extractor_config == {}
        assert ic.solved_threshold == 1.0
        assert ic.code_execution is None

    def test_code_execution_config_defaults(self):
        """Test CodeExecutionConfig default values."""
        from llenvs.core.config import CodeExecutionConfig

        ce = CodeExecutionConfig()
        assert ce.timeout == 30.0

    def test_from_dict_with_iterative(self):
        """Test from_dict parses iterative config."""
        data = {
            "environments": [
                {
                    "name": "humaneval",
                    "adapter": "huggingface",
                    "iterative": {
                        "max_turns": 5,
                        "submit_keyword": "DONE",
                        "solved_threshold": 0.9,
                        "submission_extractor": "code_block",
                        "submission_extractor_config": {"language": "python"},
                        "code_execution": {"timeout": 60.0},
                    },
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.iterative is not None
        assert env.iterative.max_turns == 5
        assert env.iterative.submit_keyword == "DONE"
        assert env.iterative.solved_threshold == 0.9
        assert env.iterative.submission_extractor == "code_block"
        assert env.iterative.submission_extractor_config == {"language": "python"}
        assert env.iterative.code_execution is not None
        assert env.iterative.code_execution.timeout == 60.0

    def test_from_dict_iterative_defaults(self):
        """Test from_dict uses IterativeConfig defaults for missing fields."""
        data = {
            "environments": [
                {
                    "name": "test",
                    "iterative": {"max_turns": 5},
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.iterative is not None
        assert env.iterative.max_turns == 5
        assert env.iterative.include_history is True
        assert env.iterative.submit_keyword == "SUBMIT"
        assert env.iterative.code_execution is None

    def test_from_dict_without_iterative(self):
        """Test from_dict defaults iterative to None."""
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(data)
        assert config.environments[0].iterative is None

    def test_to_dict_with_iterative(self):
        """Test to_dict serializes iterative config."""
        from llenvs.core.config import CodeExecutionConfig, IterativeConfig

        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    iterative=IterativeConfig(
                        max_turns=5,
                        submit_keyword="DONE",
                        solved_threshold=0.8,
                        code_execution=CodeExecutionConfig(timeout=60.0),
                    ),
                )
            ],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        env_data = data["environments"][0]
        assert "iterative" in env_data
        assert env_data["iterative"]["max_turns"] == 5
        assert env_data["iterative"]["submit_keyword"] == "DONE"
        assert env_data["iterative"]["solved_threshold"] == 0.8
        assert "code_execution" in env_data["iterative"]

    def test_to_dict_without_iterative(self):
        """Test to_dict omits iterative when None."""
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        assert "iterative" not in data["environments"][0]

    def test_to_dict_iterative_defaults_omitted(self):
        """Test to_dict omits default-valued iterative fields."""
        from llenvs.core.config import IterativeConfig

        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    iterative=IterativeConfig(max_turns=5),
                )
            ],
            model=ModelConfig(model="test-model"),
        )
        data = config.to_dict()
        iter_d = data["environments"][0]["iterative"]
        assert iter_d["max_turns"] == 5
        # Default values should be omitted
        assert "include_history" not in iter_d
        assert "summarize_history" not in iter_d
        assert "submit_keyword" not in iter_d  # SUBMIT is default
        assert "submission_extractor" not in iter_d
        assert "solved_threshold" not in iter_d

    def test_roundtrip_with_iterative(self):
        """Test roundtrip preserves iterative config."""
        original = {
            "environments": [
                {
                    "name": "humaneval",
                    "iterative": {
                        "max_turns": 5,
                        "include_history": False,
                        "submit_keyword": "DONE",
                        "solved_threshold": 0.8,
                        "submission_extractor": "code_block",
                        "submission_extractor_config": {"language": "python"},
                        "code_execution": {"timeout": 60.0},
                    },
                }
            ],
            "model": {"model": "test-model"},
        }
        config = EvalConfig.from_dict(original)
        result = config.to_dict()
        iter_d = result["environments"][0]["iterative"]
        assert iter_d["max_turns"] == 5
        assert iter_d["include_history"] is False
        assert iter_d["submit_keyword"] == "DONE"
        assert iter_d["solved_threshold"] == 0.8
        assert iter_d["submission_extractor"] == "code_block"
        assert iter_d["submission_extractor_config"] == {"language": "python"}
        assert iter_d["code_execution"]["timeout"] == 60.0

    def test_factory_wraps_with_iterative(self):
        """Test EnvironmentFactory.create() wraps with IterativeEnvironment."""
        from llenvs.core.config import EnvironmentFactory, IterativeConfig

        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=5,
            iterative=IterativeConfig(max_turns=4),
        )
        env = EnvironmentFactory.create(config)

        from llenvs.adapters.iterative import IterativeEnvironment

        assert isinstance(env, IterativeEnvironment)
        assert env.spec.is_multi_turn
        assert env.spec.max_steps == 4

    def test_factory_wraps_with_code_execution(self):
        """Test factory creates CodeExecutionReward when code_execution set."""
        from llenvs.core.config import (
            CodeExecutionConfig,
            EnvironmentFactory,
            IterativeConfig,
        )

        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=5,
            iterative=IterativeConfig(
                max_turns=3,
                code_execution=CodeExecutionConfig(timeout=10.0),
            ),
        )
        env = EnvironmentFactory.create(config)

        from llenvs.adapters.iterative import IterativeEnvironment

        assert isinstance(env, IterativeEnvironment)
        # Should have code execution reward in extra rewards
        reward_names = {rf.name for rf in env._extra_rewards}
        assert "code_execution" in reward_names

    def test_factory_wraps_with_submission_extractor(self):
        """Test factory resolves submission extractor from config."""
        from llenvs.core.config import EnvironmentFactory, IterativeConfig

        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=5,
            iterative=IterativeConfig(
                max_turns=3,
                submission_extractor="code_block",
                submission_extractor_config={"language": "python"},
            ),
        )
        env = EnvironmentFactory.create(config)

        from llenvs.adapters.iterative import IterativeEnvironment
        from llenvs.core.extraction import CodeBlockExtractor

        assert isinstance(env, IterativeEnvironment)
        assert isinstance(env._submission_extractor, CodeBlockExtractor)

    def test_factory_no_iterative_wrapping_by_default(self):
        """Test factory does NOT wrap when iterative is None."""
        from llenvs.core.config import EnvironmentFactory

        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=5,
        )
        env = EnvironmentFactory.create(config)

        from llenvs.adapters.iterative import IterativeEnvironment

        assert not isinstance(env, IterativeEnvironment)
