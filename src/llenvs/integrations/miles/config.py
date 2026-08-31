"""Environment-config discovery, caches, and guards for the miles integration.

miles loads our entry points by dotted path inside its own processes, so they
cannot receive an ``EvalConfig`` object directly. Instead, the config YAML is
discovered through the ``LLENVS_MILES_CONFIG`` environment variable, with
optional per-row overrides in sample metadata:

- ``metadata["llenvs_config"]``: path to the EvalConfig YAML (overrides the
  environment variable).
- ``metadata["llenvs_env_name"]`` (or ``LLENVS_MILES_ENV``): which entry of
  ``environments:`` to use when the config defines several. With a single
  entry, selection is implicit.

Loaded configs, Scorers, and DatasetProviders are cached per process;
``create_environment`` always returns a fresh instance because llenvs
environments enforce state continuity and must not be shared across
concurrent episodes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from llenvs.core.config import (
    EnvironmentConfig,
    EnvironmentFactory,
    EvalConfig,
    JudgeConfig,
    ModelConfig,
)
from llenvs.inference.prompts import resolve_system_prompt

if TYPE_CHECKING:
    from llenvs.integrations.dataset_provider import DatasetProvider
    from llenvs.integrations.scoring import Scorer

CONFIG_ENV_VAR = "LLENVS_MILES_CONFIG"
ENV_NAME_ENV_VAR = "LLENVS_MILES_ENV"

_eval_config_cache: dict[str, EvalConfig] = {}
_scorer_cache: dict[tuple[str, str], Scorer] = {}
_provider_cache: dict[tuple[str, str], DatasetProvider] = {}


def clear_caches() -> None:
    """Drop all cached configs, Scorers, and DatasetProviders."""
    _eval_config_cache.clear()
    _scorer_cache.clear()
    _provider_cache.clear()


def _config_path(metadata: dict[str, Any] | None) -> str:
    md = metadata or {}
    path = md.get("llenvs_config") or os.environ.get(CONFIG_ENV_VAR)
    if not path:
        raise ValueError(
            "No llenvs config found for the miles integration. Set the "
            f"{CONFIG_ENV_VAR} environment variable to an EvalConfig YAML path "
            "(export it in the miles launch script so rollout workers inherit it), "
            "or put the path in sample metadata under 'llenvs_config'."
        )
    return str(Path(path).expanduser().resolve())


def _env_name(metadata: dict[str, Any] | None) -> str | None:
    md = metadata or {}
    return md.get("llenvs_env_name") or os.environ.get(ENV_NAME_ENV_VAR)


def load_eval_config(metadata: dict[str, Any] | None = None) -> EvalConfig:
    """Load (and cache) the EvalConfig discovered from metadata / environment."""
    path = _config_path(metadata)
    cfg = _eval_config_cache.get(path)
    if cfg is None:
        cfg = EvalConfig.from_yaml(path)
        _eval_config_cache[path] = cfg
    return cfg


def select_environment(cfg: EvalConfig, name: str | None) -> EnvironmentConfig:
    """Select one entry of ``cfg.environments`` by name.

    With ``name=None`` a sole entry is returned; multiple entries require an
    explicit name.
    """
    envs = cfg.environments
    if not envs:
        raise ValueError("The llenvs config defines no environments.")
    if name is None:
        if len(envs) == 1:
            return envs[0]
        names = ", ".join(e.name for e in envs)
        raise ValueError(
            f"The llenvs config defines multiple environments ({names}); select one "
            f"via metadata['llenvs_env_name'] or the {ENV_NAME_ENV_VAR} environment "
            "variable."
        )
    for env in envs:
        if env.name == name:
            return env
    names = ", ".join(e.name for e in envs)
    raise ValueError(f"Unknown environment '{name}'; available environments: {names}.")


def load_environment_config(metadata: dict[str, Any] | None = None) -> EnvironmentConfig:
    """Load the selected EnvironmentConfig for this sample/process."""
    return select_environment(load_eval_config(metadata), _env_name(metadata))


def create_environment(metadata: dict[str, Any] | None = None) -> Any:
    """Create a FRESH environment instance for the selected config.

    Never cached: llenvs environments enforce state continuity, so each
    concurrent episode needs its own instance.
    """
    return EnvironmentFactory.create(load_environment_config(metadata))


def _cache_key(metadata: dict[str, Any] | None) -> tuple[str, str]:
    return (_config_path(metadata), load_environment_config(metadata).name)


def get_scorer(metadata: dict[str, Any] | None = None) -> Scorer:
    """Get a process-cached Scorer for the selected environment."""
    from llenvs.integrations.scoring import Scorer

    key = _cache_key(metadata)
    scorer = _scorer_cache.get(key)
    if scorer is None:
        scorer = Scorer(create_environment(metadata))
        _scorer_cache[key] = scorer
    return scorer


def get_dataset_provider(metadata: dict[str, Any] | None = None) -> DatasetProvider:
    """Get a process-cached DatasetProvider for the selected environment."""
    from llenvs.integrations.dataset_provider import DatasetProvider

    key = _cache_key(metadata)
    provider = _provider_cache.get(key)
    if provider is None:
        provider = DatasetProvider(create_environment(metadata))
        _provider_cache[key] = provider
    return provider


def resolve_config_system_prompt(cfg: EvalConfig, env_cfg: EnvironmentConfig) -> str | None:
    """Resolve the effective system prompt: env-level overrides eval-level."""
    value = env_cfg.system_prompt if env_cfg.system_prompt is not None else cfg.system_prompt
    if value is None:
        return None
    return resolve_system_prompt(value)


def resolve_system_prompt_for(metadata: dict[str, Any] | None = None) -> str | None:
    """Resolve the effective system prompt for the selected environment."""
    return resolve_config_system_prompt(
        load_eval_config(metadata), load_environment_config(metadata)
    )


# ---------------------------------------------------------------------------
# Session isolation guard
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_SCHEME_PORTS = {"http": 80, "https": 443}


def _endpoint(url: str) -> tuple[str, int | None]:
    """Normalize a URL to a comparable (host, port) endpoint."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        host = "loopback"
    port = parsed.port if parsed.port is not None else _SCHEME_PORTS.get(parsed.scheme)
    return (host, port)


def _judge_list(judge: JudgeConfig | list[JudgeConfig] | None) -> list[JudgeConfig]:
    if judge is None:
        return []
    if isinstance(judge, list):
        return cast("list[JudgeConfig]", judge)
    return [judge]


def _iter_auxiliary_models(
    cfg: EvalConfig, env_cfg: EnvironmentConfig
) -> Iterator[tuple[str, ModelConfig]]:
    """Yield (role, ModelConfig) for every judge and env-LLM in scope."""
    for judge in (*_judge_list(env_cfg.judge), *_judge_list(cfg.judge)):
        yield "judge", judge.model
    if env_cfg.env_llm is not None:
        yield "env_llm", env_cfg.env_llm.model


def ensure_isolated_from_session(
    session_base_url: str, metadata: dict[str, Any] | None = None
) -> None:
    """Refuse configs whose judge/env-LLM backends target the session server.

    The TITO session server records every request against it as trainable
    tokens of the current episode; judge or env-internal LLM traffic pointed
    there would corrupt training data, so it must use a separate endpoint.
    """
    cfg = load_eval_config(metadata)
    env_cfg = load_environment_config(metadata)
    session_endpoint = _endpoint(session_base_url)
    for role, model in _iter_auxiliary_models(cfg, env_cfg):
        base_url = model.params.get("base_url")
        if base_url and _endpoint(str(base_url)) == session_endpoint:
            raise ValueError(
                f"The {role} backend base_url ({base_url}) points at the miles "
                f"session endpoint ({session_base_url}). Judges and env-internal "
                "LLMs must never target the training session server — give them "
                "a separate serving endpoint."
            )
