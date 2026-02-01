"""YAML configuration loading and environment factory.

Supports loading evaluation configurations from YAML files
and creating environments/backends from configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    """Configuration for an environment.

    Attributes:
        name: Environment/dataset name.
        adapter: Adapter type (e.g., "reasoning_gym").
        size: Number of samples.
        seed: Random seed.
        extractor: Answer extractor type.
        extractor_config: Extractor configuration.
        params: Additional environment-specific parameters.
    """

    name: str
    adapter: str = "reasoning_gym"
    size: int | None = None
    seed: int | None = None
    extractor: str = "tag_based"
    extractor_config: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a model backend.

    Attributes:
        backend: Backend type (vllm, openai, anthropic, openrouter).
        model: Model path or name.
        params: Backend-specific parameters.
    """

    backend: str = "vllm"
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    """Configuration for inference parameters.

    Attributes:
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling.
        stop_sequences: Stop sequences.
    """

    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Complete evaluation configuration.

    Attributes:
        environments: List of environment configurations.
        model: Model configuration.
        inference: Inference parameters.
        system_prompt: Optional system prompt.
        output_dir: Output directory for results.
        limit: Maximum number of tasks per environment.
        save_detailed_results: Whether to save per-episode results.
    """

    environments: list[EnvironmentConfig]
    model: ModelConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    system_prompt: str | None = None
    output_dir: str = "./results"
    limit: int | None = None
    save_detailed_results: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded EvalConfig.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalConfig":
        """Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            EvalConfig instance.
        """
        # Parse environments
        environments = []
        for env_data in data.get("environments", []):
            environments.append(
                EnvironmentConfig(
                    name=env_data["name"],
                    adapter=env_data.get("adapter", "reasoning_gym"),
                    size=env_data.get("size"),
                    seed=env_data.get("seed"),
                    extractor=env_data.get("extractor", "tag_based"),
                    extractor_config=env_data.get("extractor_config", {}),
                    params=env_data.get("params", {}),
                )
            )

        # Parse model config
        model_data = data.get("model", {})
        model = ModelConfig(
            backend=model_data.get("backend", "vllm"),
            model=model_data.get("model", model_data.get("path", "")),
            params=model_data.get("params", {}),
        )

        # Parse inference config
        inference_data = data.get("inference", {})
        inference = InferenceConfig(
            temperature=inference_data.get("temperature", 0.0),
            max_tokens=inference_data.get("max_tokens", 2048),
            top_p=inference_data.get("top_p", 1.0),
            top_k=inference_data.get("top_k", 0),
            stop_sequences=inference_data.get("stop_sequences", []),
        )

        return cls(
            environments=environments,
            model=model,
            inference=inference,
            system_prompt=data.get("system_prompt"),
            output_dir=data.get("output_dir", "./results"),
            limit=data.get("limit"),
            save_detailed_results=data.get("save_detailed_results", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "environments": [
                {
                    "name": env.name,
                    "adapter": env.adapter,
                    "size": env.size,
                    "seed": env.seed,
                    "extractor": env.extractor,
                    "extractor_config": env.extractor_config,
                    "params": env.params,
                }
                for env in self.environments
            ],
            "model": {
                "backend": self.model.backend,
                "model": self.model.model,
                "params": self.model.params,
            },
            "inference": {
                "temperature": self.inference.temperature,
                "max_tokens": self.inference.max_tokens,
                "top_p": self.inference.top_p,
                "top_k": self.inference.top_k,
                "stop_sequences": self.inference.stop_sequences,
            },
            "system_prompt": self.system_prompt,
            "output_dir": self.output_dir,
            "limit": self.limit,
            "save_detailed_results": self.save_detailed_results,
        }


class EnvironmentFactory:
    """Factory for creating environments from configuration."""

    @staticmethod
    def create(config: EnvironmentConfig) -> Any:
        """Create an environment from configuration.

        Args:
            config: Environment configuration.

        Returns:
            Environment instance.

        Raises:
            KeyError: If adapter is not registered.
            ValueError: If environment name is not recognized by adapter.
        """
        from llenvs.core.registry import environment_registry, extractor_registry

        # Create extractor from config
        extractor = extractor_registry.create(config.extractor, **config.extractor_config)

        # Use the environment registry to get the environment
        return environment_registry.get(
            name=config.name,
            adapter=config.adapter,
            size=config.size,
            seed=config.seed,
            extractor=extractor,
            **config.params,
        )


class BackendFactory:
    """Factory for creating model backends from configuration."""

    @staticmethod
    def create(config: ModelConfig) -> Any:
        """Create a model backend from configuration.

        Args:
            config: Model configuration.

        Returns:
            ModelBackend instance.

        Raises:
            ValueError: If backend type is unknown.
        """
        backend_type = config.backend.lower()

        if backend_type == "vllm":
            from llenvs.inference.backends.vllm import VLLMBackend

            return VLLMBackend(model_path=config.model, **config.params)

        elif backend_type == "openai":
            from llenvs.inference.backends.api import OpenAIBackend

            return OpenAIBackend(model=config.model, **config.params)

        elif backend_type == "anthropic":
            from llenvs.inference.backends.api import AnthropicBackend

            return AnthropicBackend(model=config.model, **config.params)

        elif backend_type == "openrouter":
            from llenvs.inference.backends.api import OpenRouterBackend

            return OpenRouterBackend(model=config.model, **config.params)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


def create_sampling_params(config: InferenceConfig) -> Any:
    """Create SamplingParams from InferenceConfig.

    Args:
        config: Inference configuration.

    Returns:
        SamplingParams instance.
    """
    from llenvs.inference.protocol import SamplingParams

    return SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        top_k=config.top_k,
        stop_sequences=tuple(config.stop_sequences),
    )
