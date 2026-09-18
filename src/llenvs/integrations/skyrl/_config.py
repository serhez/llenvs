"""Connector settings usable without the native training stack.

Concrete annotations and typing.Optional are intentional: the pinned SkyRL
nested builder inspects dataclass fields without resolving postponed types.
"""

import copy
import importlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from llenvs.core.config import EnvironmentConfig, EvalConfig, JudgeConfig
from llenvs.integrations.skyrl._checks import finite_number, identifier, integer
from llenvs.integrations.skyrl.data import _canonical_json, _export_config


def _factory_path(value: str) -> tuple[str, str]:
    identifier(value, "token_scorer.factory")
    parts = value.split(":")
    if (
        len(parts) != 2
        or not all(part.isidentifier() for part in parts[0].split("."))
        or not parts[1].isidentifier()
    ):
        raise ValueError("token_scorer.factory must be module:attribute")
    return parts[0], parts[1]


@dataclass
class TokenScorerConfig:
    factory: str
    revision: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        _factory_path(self.factory)
        identifier(self.revision, "token_scorer.revision")
        finite_number(self.weight, "token_scorer.weight")
        if not isinstance(self.kwargs, dict):
            raise ValueError("token_scorer.kwargs must be a JSON object")
        _canonical_json(self.kwargs)
        self.kwargs = copy.deepcopy(self.kwargs)


@dataclass
class LlenvsConfig:
    config: str = ""
    env_name: Optional[str] = None  # noqa: UP045 - native builder contract
    sampling_contract: str = "native"
    turn_weighting: str = "uniform"
    judge_timing: str = "decision"
    judge_use_images: bool = False
    token_scorer: Optional[TokenScorerConfig] = None  # noqa: UP045 - native builder contract
    max_active_episodes: int = 32
    forward_env: list[str] = field(default_factory=list)
    check_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.config, str):
            raise ValueError("llenvs.config must be a YAML path string")
        if self.env_name is not None:
            identifier(self.env_name, "llenvs.env_name")
        for name, values in (
            ("sampling_contract", ("native", "unmodified")),
            ("turn_weighting", ("uniform", "span_normalized")),
            ("judge_timing", ("decision", "episode")),
        ):
            if getattr(self, name) not in values:
                raise ValueError(f"llenvs.{name} must be one of {values}")
        for name in ("judge_use_images", "check_only"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"llenvs.{name} must be a boolean")
        integer(self.max_active_episodes, "llenvs.max_active_episodes", minimum=1)
        if self.token_scorer is not None and not isinstance(self.token_scorer, TokenScorerConfig):
            raise ValueError("llenvs.token_scorer must be a constructed TokenScorerConfig")
        if not isinstance(self.forward_env, list):
            raise ValueError("llenvs.forward_env must be a list of variable names")
        for name in self.forward_env:
            if not isinstance(name, str) or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError("llenvs.forward_env contains an invalid variable name")
            if name.startswith(
                (
                    "RAY_",
                    "SKYRL_",
                    "VLLM_",
                    "CUDA_",
                    "NCCL_",
                    "PYTHON",
                    "UV_",
                    "LD_",
                    "DYLD_",
                    "PYTORCH_",
                    "TORCH_",
                    "NVTE_",
                    "CUBLAS_",
                )
            ) or name in {"PATH", "HOME", "VIRTUAL_ENV"}:
                raise ValueError(f"llenvs.forward_env cannot override runtime control {name}")
        if len(set(self.forward_env)) != len(self.forward_env):
            raise ValueError("llenvs.forward_env contains duplicate names")
        self.forward_env = list(self.forward_env)


@dataclass(frozen=True)
class SelectedEnvironment:
    path: Path
    environment_config: EnvironmentConfig
    system_prompt: Optional[str]  # noqa: UP045 - keep native boundary annotations concrete
    fingerprint: str
    judges: tuple[JudgeConfig, ...]


def load_selected_environment(config: LlenvsConfig) -> SelectedEnvironment:
    """Resolve YAML identity/judge precedence without constructing any resources."""
    identifier(config.config, "llenvs.config")
    path = Path(config.config).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("llenvs.config must identify a YAML file")
    evaluation = EvalConfig.from_yaml(path)
    environment, system_prompt, fingerprint = _export_config(evaluation, config.env_name)
    judges = environment.judge if environment.judge is not None else evaluation.judge
    selected_judges: tuple[JudgeConfig, ...]
    if judges is None:
        selected_judges = ()
    elif isinstance(judges, JudgeConfig):
        selected_judges = (judges,)
    else:
        selected_judges = tuple(judges)
    return SelectedEnvironment(path, environment, system_prompt, fingerprint, selected_judges)


def forward_environment(
    config: LlenvsConfig, *, environ: Mapping[str, str] = os.environ
) -> dict[str, str]:
    """Return requested values for Ray runtime_env, never for config or logging."""
    missing = [name for name in config.forward_env if name not in environ]
    if missing:
        raise ValueError(f"required forward_env variables are missing: {', '.join(missing)}")
    return {name: environ[name] for name in config.forward_env}


def resolve_factory(path: str) -> Callable[..., Any]:
    """Import and validate a factory without invoking it or starting its scorer."""
    module, name = _factory_path(path)
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("token_scorer.factory must resolve to a callable")
    return factory
