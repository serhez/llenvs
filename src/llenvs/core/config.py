"""YAML configuration loading and environment factory.

Supports loading evaluation configurations from YAML files
and creating environments/backends from configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from llenvs.container.config import ContainerConfig


@dataclass
class EnvironmentConfig:
    """Configuration for an environment.

    Attributes:
        name: Environment/dataset name.
        adapter: Adapter type (e.g., "reasoning_gym").
        size: Number of samples.
        seed: Random seed.
        answer_extractor: Answer extractor type (single extractor shorthand).
        answer_extractor_config: Extractor configuration (single extractor shorthand).
        answer_extractors: Ordered list of extractors to try (chain). When set,
            builds a CompositeExtractor. Overrides answer_extractor/answer_extractor_config.
        prompt_template: Per-env prompt template name or literal template string.
        system_prompt: Per-env system prompt override. A string (name or literal)
            or list of fragment/prompt names.
        prompts: Per-env prompt component overrides for multi-step environments.
        params: Additional environment-specific parameters.
        container: Optional container configuration. When set, the environment
            runs inside a container or subprocess instead of in-process.
        judge: Optional LLM judge configuration. A single JudgeConfig or list
            of JudgeConfigs for multiple judges. Per-env overrides eval-level.
        env_llm: Optional environment LLM configuration. When set, creates a
            ModelBackend for environments that use an LLM in their transition
            function (e.g., DialogueEnvironment).
        branching_strategy: Branching strategy preference for ``BranchManager``.
            One of ``"direct"``, ``"action_replay"``, ``"process_fork"``, or
            ``None`` for auto-resolution.
        difficulties: Filter tasks by difficulty level. Only tasks whose
            difficulty is in this set are included. ``None`` means no filtering.
            Tasks without explicit difficulty metadata are assigned ``"n/a"``.
    """

    name: str
    adapter: str = "reasoning_gym"
    size: int | None = None
    seed: int | None = None
    answer_extractor: str = "tag_based"
    answer_extractor_config: dict[str, Any] = field(default_factory=dict)
    answer_extractors: list[dict[str, Any]] | None = None
    pre_cleaners: list[str | dict[str, Any]] | None = None
    post_cleaners: list[str | dict[str, Any]] | None = None
    prompt_template: str | None = None
    system_prompt: str | list[str] | None = None
    prompts: dict[str, str] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    container: ContainerConfig | None = None
    judge: JudgeConfig | list[JudgeConfig] | None = None
    env_llm: EnvironmentLLMConfig | None = None
    branching_strategy: str | None = None
    difficulties: set[str] | None = None
    iterative: IterativeConfig | None = None


@dataclass
class ModelConfig:
    """Configuration for a model backend.

    Attributes:
        backend: Backend type (vllm, openai, anthropic, openrouter).
        model: Model path or name.
        max_concurrency: Maximum concurrent requests for API backends.
        params: Backend-specific parameters.
    """

    backend: str = "vllm"
    model: str = ""
    max_concurrency: int = 64
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
        extra: Backend-specific parameters passed through to the underlying
            inference library (e.g., repetition_penalty for HuggingFace).
        thinking_budget: Maximum thinking tokens. None = no budget.
        thinking_budget_per_block: Per-block budgets (reset on each ``<think>``).
        thinking_budget_soft_ratio: Begin boosting ``</think>`` at this ratio.
        thinking_budget_suffix: Text forced when budget exhausted. None = bare
            ``</think>``. Pass ``DEFAULT_EARLY_STOPPING_SUFFIX`` to enable.
        second_elicitation_suffix: Suffix for follow-up on truncated outputs.
            None = disabled. When set, enables second elicitation.
        second_elicitation_max_tokens: Token budget for the follow-up call.
    """

    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # Thinking budget
    thinking_budget: int | None = None
    thinking_budget_per_block: bool = False
    thinking_budget_soft_ratio: float | None = None
    thinking_budget_suffix: str | None = None
    # Second elicitation
    second_elicitation_suffix: str | None = None
    second_elicitation_max_tokens: int = 256


@dataclass
class EnvironmentLLMConfig:
    """Configuration for an environment-internal LLM.

    Used by environments (like ``DialogueEnvironment``) that call an LLM
    as part of their transition function to generate observations.

    Attributes:
        model: Backend configuration for the env LLM.
        system_prompt: Base system prompt for the env LLM.
        inference: Sampling parameters. Defaults to temperature=0, max_tokens=512.
    """

    model: ModelConfig
    system_prompt: str = ""
    inference: InferenceConfig | None = None


@dataclass
class CodeExecutionConfig:
    """Configuration for code execution in iterative environments.

    Attributes:
        timeout: Maximum execution time per test run in seconds.
    """

    timeout: float = 30.0


@dataclass
class IterativeConfig:
    """Configuration for iterative refinement wrapping.

    When set on an ``EnvironmentConfig``, ``EnvironmentFactory.create()``
    wraps the base environment in an ``IterativeEnvironment``.

    Attributes:
        max_turns: Maximum refinement turns.
        include_history: Whether to include previous attempts in feedback.
        summarize_history: Whether to use env LLM to summarize history.
        submit_keyword: Keyword for early termination. None to disable.
        submission_extractor: Extractor name for parsing submissions.
        submission_extractor_config: Config for the submission extractor.
        solved_threshold: OUTCOME score threshold to consider task solved.
        code_execution: Optional code execution configuration.
    """

    max_turns: int = 3
    include_history: bool = True
    summarize_history: bool = False
    submit_keyword: str | None = "SUBMIT"
    submission_extractor: str | None = None
    submission_extractor_config: dict[str, Any] = field(default_factory=dict)
    solved_threshold: float = 1.0
    code_execution: CodeExecutionConfig | None = None


@dataclass
class JudgeConfig:
    """Configuration for an LLM-as-a-judge reward function.

    Attributes:
        model: Judge LLM backend configuration.
        template: Built-in template name or literal template string.
        system_prompt: Overrides the template's default system prompt.
        score_range: Min and max of the scoring scale, for normalization.
        name: Reward signal name.
        reward_type: Maps to RewardType enum (outcome, step, format, process).
        weight: Reward signal weight.
        normalize: Whether to normalize scores to [0,1] via score_range.
        inference: Judge sampling parameters. Defaults to temperature=0, max_tokens=512.
    """

    model: ModelConfig
    template: str = "correctness"
    system_prompt: str | None = None
    score_range: tuple[float, float] = (1.0, 10.0)
    name: str = "judge"
    reward_type: str = "outcome"
    weight: float = 1.0
    normalize: bool = True
    inference: InferenceConfig | None = None


@dataclass
class EvalConfig:
    """Complete evaluation configuration.

    Attributes:
        environments: List of environment configurations.
        model: Model configuration.
        inference: Inference parameters.
        system_prompt: Optional system prompt. A string (name or literal)
            or list of fragment/prompt names.
        model_profile: Model profile name or "auto" for detection.
        prompt_template: Global default prompt template name.
        output_dir: Output directory for results.
        limit: Maximum number of tasks per environment.
        batch_size: Maximum trajectories per batch for run_batch(). None
            means all trajectories run in a single lockstep batch.
        save_detailed_results: Whether to save per-episode results.
        judge: Optional eval-level LLM judge configuration. Applied to all
            environments unless overridden at the environment level.
    """

    environments: list[EnvironmentConfig]
    model: ModelConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    system_prompt: str | list[str] | None = None
    model_profile: str | None = None
    prompt_template: str | None = None
    output_dir: str = "./results"
    limit: int | None = None
    batch_size: int | None = None
    save_detailed_results: bool = True
    judge: JudgeConfig | list[JudgeConfig] | None = None

    @staticmethod
    def _parse_judge(data: Any) -> JudgeConfig | list[JudgeConfig] | None:
        """Parse judge config from dict data."""
        if data is None:
            return None
        if isinstance(data, list):
            return [EvalConfig._parse_single_judge(d) for d in data]
        return EvalConfig._parse_single_judge(data)

    @staticmethod
    def _parse_single_judge(data: dict[str, Any]) -> JudgeConfig:
        """Parse a single JudgeConfig from dict."""
        model_data = data.get("model", {})
        model = ModelConfig(
            backend=model_data.get("backend", "vllm"),
            model=model_data.get("model", model_data.get("path", "")),
            max_concurrency=model_data.get("max_concurrency", 64),
            params=model_data.get("params", {}),
        )

        inference = None
        inference_data = data.get("inference")
        if inference_data is not None:
            inference = InferenceConfig(
                temperature=inference_data.get("temperature", 0.0),
                max_tokens=inference_data.get("max_tokens", 512),
                top_p=inference_data.get("top_p", 1.0),
                top_k=inference_data.get("top_k", 0),
                stop_sequences=inference_data.get("stop_sequences", []),
            )

        score_range = data.get("score_range", (1.0, 10.0))
        if isinstance(score_range, list):
            score_range = (float(score_range[0]), float(score_range[1]))

        return JudgeConfig(
            model=model,
            template=data.get("template", "correctness"),
            system_prompt=data.get("system_prompt"),
            score_range=score_range,
            name=data.get("name", "judge"),
            reward_type=data.get("reward_type", "outcome"),
            weight=data.get("weight", 1.0),
            normalize=data.get("normalize", True),
            inference=inference,
        )

    @staticmethod
    def _judge_to_dict(judge: JudgeConfig | list[JudgeConfig] | None) -> Any:
        """Serialize judge config to dict."""
        if judge is None:
            return None
        if isinstance(judge, list):
            return [EvalConfig._single_judge_to_dict(j) for j in judge]
        return EvalConfig._single_judge_to_dict(judge)

    @staticmethod
    def _single_judge_to_dict(j: JudgeConfig) -> dict[str, Any]:
        """Serialize a single JudgeConfig to dict."""
        d: dict[str, Any] = {
            "model": {
                "backend": j.model.backend,
                "model": j.model.model,
            },
            "template": j.template,
        }
        if j.system_prompt is not None:
            d["system_prompt"] = j.system_prompt
        if j.score_range != (1.0, 10.0):
            d["score_range"] = list(j.score_range)
        if j.name != "judge":
            d["name"] = j.name
        if j.reward_type != "outcome":
            d["reward_type"] = j.reward_type
        if j.weight != 1.0:
            d["weight"] = j.weight
        if j.normalize is not True:
            d["normalize"] = j.normalize
        if j.inference is not None:
            d["inference"] = {
                "temperature": j.inference.temperature,
                "max_tokens": j.inference.max_tokens,
            }
        return d

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
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
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        """Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            EvalConfig instance.
        """
        # Parse environments
        environments = []
        for env_data in data.get("environments", []):
            # Parse pre_cleaners/post_cleaners: distinguish missing (None) from empty ([])
            pre_cleaners = env_data.get("pre_cleaners")  # None if not in dict
            post_cleaners = env_data.get("post_cleaners")  # None if not in dict

            # Parse container config
            container_data = env_data.get("container")
            container = None
            if container_data is not None:
                from llenvs.container.config import ContainerConfig

                container = ContainerConfig(
                    runtime=container_data.get("runtime", "docker"),
                    image=container_data.get("image"),
                    port=container_data.get("port"),
                    timeout=container_data.get("timeout", 60.0),
                    env_vars=container_data.get("env_vars", {}),
                    volumes=container_data.get("volumes", {}),
                    docker_command=container_data.get("docker_command", "docker"),
                )

            # Parse judge config
            env_judge = cls._parse_judge(env_data.get("judge"))

            # Parse env_llm config
            env_llm_data = env_data.get("env_llm")
            env_llm = None
            if env_llm_data is not None:
                env_llm_model_data = env_llm_data.get("model", {})
                env_llm_model = ModelConfig(
                    backend=env_llm_model_data.get("backend", "vllm"),
                    model=env_llm_model_data.get("model", env_llm_model_data.get("path", "")),
                    max_concurrency=env_llm_model_data.get("max_concurrency", 64),
                    params=env_llm_model_data.get("params", {}),
                )
                env_llm_inference = None
                env_llm_inference_data = env_llm_data.get("inference")
                if env_llm_inference_data is not None:
                    env_llm_inference = InferenceConfig(
                        temperature=env_llm_inference_data.get("temperature", 0.0),
                        max_tokens=env_llm_inference_data.get("max_tokens", 512),
                        top_p=env_llm_inference_data.get("top_p", 1.0),
                        top_k=env_llm_inference_data.get("top_k", 0),
                        stop_sequences=env_llm_inference_data.get("stop_sequences", []),
                    )
                env_llm = EnvironmentLLMConfig(
                    model=env_llm_model,
                    system_prompt=env_llm_data.get("system_prompt", ""),
                    inference=env_llm_inference,
                )

            # Parse iterative config
            iterative_data = env_data.get("iterative")
            iterative = None
            if iterative_data is not None:
                code_exec_data = iterative_data.get("code_execution")
                code_exec = None
                if code_exec_data is not None:
                    code_exec = CodeExecutionConfig(
                        timeout=code_exec_data.get("timeout", 30.0),
                    )
                iterative = IterativeConfig(
                    max_turns=iterative_data.get("max_turns", 3),
                    include_history=iterative_data.get("include_history", True),
                    summarize_history=iterative_data.get("summarize_history", False),
                    submit_keyword=iterative_data.get("submit_keyword", "SUBMIT"),
                    submission_extractor=iterative_data.get("submission_extractor"),
                    submission_extractor_config=iterative_data.get(
                        "submission_extractor_config", {}
                    ),
                    solved_threshold=iterative_data.get("solved_threshold", 1.0),
                    code_execution=code_exec,
                )

            environments.append(
                EnvironmentConfig(
                    name=env_data["name"],
                    adapter=env_data.get("adapter", "reasoning_gym"),
                    size=env_data.get("size"),
                    seed=env_data.get("seed"),
                    answer_extractor=env_data.get("answer_extractor", "tag_based"),
                    answer_extractor_config=env_data.get("answer_extractor_config", {}),
                    answer_extractors=env_data.get("answer_extractors"),
                    pre_cleaners=pre_cleaners,
                    post_cleaners=post_cleaners,
                    prompt_template=env_data.get("prompt_template"),
                    system_prompt=env_data.get("system_prompt"),
                    prompts=env_data.get("prompts"),
                    params=env_data.get("params", {}),
                    container=container,
                    judge=env_judge,
                    env_llm=env_llm,
                    branching_strategy=env_data.get("branching_strategy"),
                    difficulties=set(env_data["difficulties"])
                    if env_data.get("difficulties") is not None
                    else None,
                    iterative=iterative,
                )
            )

        # Parse model config
        model_data = data.get("model", {})
        model = ModelConfig(
            backend=model_data.get("backend", "vllm"),
            model=model_data.get("model", model_data.get("path", "")),
            max_concurrency=model_data.get("max_concurrency", 64),
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
            extra=inference_data.get("extra", {}),
            thinking_budget=inference_data.get("thinking_budget"),
            thinking_budget_per_block=inference_data.get("thinking_budget_per_block", False),
            thinking_budget_soft_ratio=inference_data.get("thinking_budget_soft_ratio"),
            thinking_budget_suffix=inference_data.get("thinking_budget_suffix"),
            second_elicitation_suffix=inference_data.get("second_elicitation_suffix"),
            second_elicitation_max_tokens=inference_data.get("second_elicitation_max_tokens", 256),
        )

        return cls(
            environments=environments,
            model=model,
            inference=inference,
            system_prompt=data.get("system_prompt"),
            model_profile=data.get("model_profile"),
            prompt_template=data.get("prompt_template"),
            output_dir=data.get("output_dir", "./results"),
            limit=data.get("limit"),
            batch_size=data.get("batch_size"),
            save_detailed_results=data.get("save_detailed_results", True),
            judge=cls._parse_judge(data.get("judge")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        env_dicts = []
        for env in self.environments:
            d: dict[str, Any] = {
                "name": env.name,
                "adapter": env.adapter,
                "size": env.size,
                "seed": env.seed,
                "answer_extractor": env.answer_extractor,
                "answer_extractor_config": env.answer_extractor_config,
                "params": env.params,
            }
            if env.answer_extractors is not None:
                d["answer_extractors"] = env.answer_extractors
            if env.pre_cleaners is not None:
                d["pre_cleaners"] = env.pre_cleaners
            if env.post_cleaners is not None:
                d["post_cleaners"] = env.post_cleaners
            if env.prompt_template is not None:
                d["prompt_template"] = env.prompt_template
            if env.system_prompt is not None:
                d["system_prompt"] = env.system_prompt
            if env.prompts is not None:
                d["prompts"] = env.prompts
            if env.container is not None:
                c: dict[str, Any] = {"runtime": env.container.runtime}
                if env.container.image is not None:
                    c["image"] = env.container.image
                if env.container.port is not None:
                    c["port"] = env.container.port
                if env.container.timeout != 60.0:
                    c["timeout"] = env.container.timeout
                if env.container.env_vars:
                    c["env_vars"] = env.container.env_vars
                if env.container.volumes:
                    c["volumes"] = env.container.volumes
                if env.container.docker_command != "docker":
                    c["docker_command"] = env.container.docker_command
                d["container"] = c
            if env.judge is not None:
                d["judge"] = self._judge_to_dict(env.judge)
            if env.env_llm is not None:
                env_llm_d: dict[str, Any] = {
                    "model": {
                        "backend": env.env_llm.model.backend,
                        "model": env.env_llm.model.model,
                    },
                }
                if env.env_llm.system_prompt:
                    env_llm_d["system_prompt"] = env.env_llm.system_prompt
                if env.env_llm.inference is not None:
                    env_llm_d["inference"] = {
                        "temperature": env.env_llm.inference.temperature,
                        "max_tokens": env.env_llm.inference.max_tokens,
                    }
                d["env_llm"] = env_llm_d
            if env.branching_strategy is not None:
                d["branching_strategy"] = env.branching_strategy
            if env.difficulties is not None:
                d["difficulties"] = sorted(env.difficulties)
            if env.iterative is not None:
                iter_d: dict[str, Any] = {
                    "max_turns": env.iterative.max_turns,
                }
                if not env.iterative.include_history:
                    iter_d["include_history"] = False
                if env.iterative.summarize_history:
                    iter_d["summarize_history"] = True
                if env.iterative.submit_keyword != "SUBMIT":
                    iter_d["submit_keyword"] = env.iterative.submit_keyword
                if env.iterative.submission_extractor is not None:
                    iter_d["submission_extractor"] = env.iterative.submission_extractor
                if env.iterative.submission_extractor_config:
                    iter_d["submission_extractor_config"] = (
                        env.iterative.submission_extractor_config
                    )
                if env.iterative.solved_threshold != 1.0:
                    iter_d["solved_threshold"] = env.iterative.solved_threshold
                if env.iterative.code_execution is not None:
                    ce: dict[str, Any] = {}
                    if env.iterative.code_execution.timeout != 30.0:
                        ce["timeout"] = env.iterative.code_execution.timeout
                    iter_d["code_execution"] = ce
                d["iterative"] = iter_d
            env_dicts.append(d)

        result: dict[str, Any] = {
            "environments": env_dicts,
            "model": {
                "backend": self.model.backend,
                "model": self.model.model,
                "max_concurrency": self.model.max_concurrency,
                "params": self.model.params,
            },
            "inference": {
                "temperature": self.inference.temperature,
                "max_tokens": self.inference.max_tokens,
                "top_p": self.inference.top_p,
                "top_k": self.inference.top_k,
                "stop_sequences": self.inference.stop_sequences,
                "extra": self.inference.extra,
                "thinking_budget": self.inference.thinking_budget,
                "thinking_budget_per_block": self.inference.thinking_budget_per_block,
                "thinking_budget_soft_ratio": self.inference.thinking_budget_soft_ratio,
                "thinking_budget_suffix": self.inference.thinking_budget_suffix,
                "second_elicitation_suffix": self.inference.second_elicitation_suffix,
                "second_elicitation_max_tokens": self.inference.second_elicitation_max_tokens,
            },
            "system_prompt": self.system_prompt,
            "model_profile": self.model_profile,
            "prompt_template": self.prompt_template,
            "output_dir": self.output_dir,
            "limit": self.limit,
            "save_detailed_results": self.save_detailed_results,
        }
        if self.batch_size is not None:
            result["batch_size"] = self.batch_size
        if self.judge is not None:
            result["judge"] = self._judge_to_dict(self.judge)
        return result


class EnvironmentFactory:
    """Factory for creating environments from configuration."""

    @staticmethod
    def create(config: EnvironmentConfig) -> Any:
        """Create an environment from configuration.

        If ``config.container`` is set, the environment is created inside a
        container or subprocess and a proxy is returned.

        Args:
            config: Environment configuration.

        Returns:
            Environment instance (or ``ContainerEnvironment`` proxy).

        Raises:
            KeyError: If adapter is not registered.
            ValueError: If environment name is not recognized by adapter,
                or if "native" extraction is requested but not available.
        """
        if config.container is not None:
            from llenvs.container import create_container_environment

            return create_container_environment(config)

        from llenvs.core.cleaning import resolve_cleaners
        from llenvs.core.extraction import CleanedExtractor, CompositeExtractor
        from llenvs.core.registry import answer_extractor_registry, environment_registry

        if config.answer_extractors is not None:
            # Build a CompositeExtractor from the chain
            chain: list[Any] = []
            for entry in config.answer_extractors:
                ext_type = entry["type"]
                ext_config = entry.get("config", {})

                if ext_type == "native":
                    adapter_instance = environment_registry.get_adapter(config.adapter)
                    native_ext = adapter_instance.get_native_answer_extractor(config.name)
                    if native_ext is None:
                        raise ValueError(
                            f"Adapter '{config.adapter}' does not provide "
                            f"native extraction for task '{config.name}'"
                        )
                    chain.append(native_ext)
                else:
                    chain.append(answer_extractor_registry.create(ext_type, **ext_config))

            answer_extractor = CompositeExtractor(extractors=chain)
        else:
            # Single extractor shorthand
            answer_extractor = answer_extractor_registry.create(
                config.answer_extractor, **config.answer_extractor_config
            )

        # Wrap with cleaning layer
        pre_fns = resolve_cleaners(config.pre_cleaners, "pre")
        post_fns = resolve_cleaners(config.post_cleaners, "post")
        if pre_fns or post_fns:
            answer_extractor = CleanedExtractor(
                inner=answer_extractor,
                pre_cleaners=pre_fns,
                post_cleaners=post_fns,
            )

        # Build kwargs, including prompts if set
        env_kwargs: dict[str, Any] = {**config.params}
        if config.prompts is not None:
            env_kwargs["prompts"] = config.prompts

        # Wire up environment LLM if configured
        if config.env_llm is not None:
            env_kwargs["env_llm"] = BackendFactory.create(config.env_llm.model)
            env_llm_inference = config.env_llm.inference or InferenceConfig(
                temperature=0.0, max_tokens=512
            )
            env_kwargs["sampling_params"] = create_sampling_params(env_llm_inference)
            if config.env_llm.system_prompt:
                env_kwargs["system_prompt"] = config.env_llm.system_prompt

        # Pass difficulty filter only when set (avoid polluting adapters that
        # forward **kwargs to library constructors).
        if config.difficulties is not None:
            env_kwargs["difficulties"] = config.difficulties

        # Use the environment registry to get the environment
        env = environment_registry.get(
            name=config.name,
            adapter=config.adapter,
            size=config.size,
            seed=config.seed,
            answer_extractor=answer_extractor,
            **env_kwargs,
        )

        # Wrap with iterative refinement if configured
        if config.iterative is not None:
            env = EnvironmentFactory._wrap_iterative(config, env)

        return env

    @staticmethod
    def _wrap_iterative(config: EnvironmentConfig, env: Any) -> Any:
        """Wrap an environment with iterative refinement.

        Args:
            config: Environment configuration with iterative settings.
            env: Base environment to wrap.

        Returns:
            IterativeEnvironment wrapping the base environment.
        """
        from llenvs.adapters.iterative import IterativeEnvironment
        from llenvs.core.reward import RewardFunction

        assert config.iterative is not None
        ic = config.iterative

        # Build submission extractor
        submission_extractor = None
        if ic.submission_extractor is not None:
            from llenvs.core.registry import answer_extractor_registry

            submission_extractor = answer_extractor_registry.create(
                ic.submission_extractor, **ic.submission_extractor_config
            )

        # Build extra rewards
        extra_rewards: list[RewardFunction] = []

        # Code execution reward
        if ic.code_execution is not None:
            from llenvs.core.code_execution import (
                CodeExecutionReward,
                SubprocessCodeExecutor,
            )

            executor = SubprocessCodeExecutor()
            extra_rewards.append(
                CodeExecutionReward(
                    executor=executor,
                )
            )

        return IterativeEnvironment(
            inner=env,
            max_turns=ic.max_turns,
            include_history=ic.include_history,
            submit_keyword=ic.submit_keyword,
            solved_threshold=ic.solved_threshold,
            submission_extractor=submission_extractor,
            extra_rewards=tuple(extra_rewards),
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

        elif backend_type == "vllm_singularity":
            from llenvs.inference.backends.vllm_singularity import (
                SingularityVLLMBackend,
            )

            return SingularityVLLMBackend(
                model_path=config.model, **config.params
            )

        elif backend_type == "openai":
            from llenvs.inference.backends.api import OpenAIBackend

            return OpenAIBackend(
                model=config.model,
                max_concurrency=config.max_concurrency,
                **config.params,
            )

        elif backend_type == "anthropic":
            from llenvs.inference.backends.api import AnthropicBackend

            return AnthropicBackend(
                model=config.model,
                max_concurrency=config.max_concurrency,
                **config.params,
            )

        elif backend_type == "openrouter":
            from llenvs.inference.backends.api import OpenRouterBackend

            return OpenRouterBackend(
                model=config.model,
                max_concurrency=config.max_concurrency,
                **config.params,
            )

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
        thinking_budget=config.thinking_budget,
        thinking_budget_per_block=config.thinking_budget_per_block,
        thinking_budget_soft_ratio=config.thinking_budget_soft_ratio,
        thinking_budget_suffix=config.thinking_budget_suffix,
        second_elicitation_suffix=config.second_elicitation_suffix,
        second_elicitation_max_tokens=config.second_elicitation_max_tokens,
        extra=dict(config.extra),
    )
