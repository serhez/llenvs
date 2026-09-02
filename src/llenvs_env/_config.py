"""EvalConfig loading, environment selection, and per-process caches.

Deliberate near-duplicate of ``llenvs.integrations.miles.config``: that
module's env-var discovery and error surface are miles-specific, and the
shipped connector stays untouched. Here the config path and environment name
arrive through the typed taskset config instead.

Configs, Scorers, and DatasetProviders are cached per process (verifiers
scores tasks from worker threads, so the caches are guarded by a lock);
``create_environment`` always returns a fresh instance because llenvs
environments enforce state continuity and must not be shared across
concurrent episodes.

No verifiers imports: this module is tested in the base venv.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenvs.core.config import EnvironmentConfig, EnvironmentFactory, EvalConfig
from llenvs.inference.prompts import resolve_system_prompt

if TYPE_CHECKING:
    from llenvs.integrations.dataset_provider import DatasetProvider
    from llenvs.integrations.scoring import Scorer

_lock = threading.RLock()
_eval_config_cache: dict[str, EvalConfig] = {}
_scorer_cache: dict[tuple[str, str], Scorer] = {}
_provider_cache: dict[tuple[str, str], DatasetProvider] = {}


def clear_caches() -> None:
    """Drop all cached configs, Scorers, and DatasetProviders."""
    with _lock:
        _eval_config_cache.clear()
        _scorer_cache.clear()
        _provider_cache.clear()


def config_key(path: str | Path) -> str:
    """Normalize a config path to the cache key: its resolved absolute path."""
    return str(Path(path).expanduser().resolve())


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load (and cache) the EvalConfig at ``path``."""
    key = config_key(path)
    with _lock:
        cfg = _eval_config_cache.get(key)
        if cfg is None:
            if not Path(key).is_file():
                raise FileNotFoundError(f"llenvs config not found: {key}")
            cfg = EvalConfig.from_yaml(key)
            _eval_config_cache[key] = cfg
    return cfg


def select_environment(cfg: EvalConfig, env_name: str | None) -> EnvironmentConfig:
    """Select one entry of ``cfg.environments`` by name.

    With ``env_name=None`` a sole entry is returned; multiple entries require
    an explicit name.
    """
    envs = cfg.environments
    if not envs:
        raise ValueError("The llenvs config defines no environments.")
    if env_name is None:
        if len(envs) == 1:
            return envs[0]
        names = ", ".join(e.name for e in envs)
        raise ValueError(
            f"The llenvs config defines multiple environments ({names}); select one "
            "with the taskset config field `env_name`."
        )
    for env in envs:
        if env.name == env_name:
            return env
    names = ", ".join(e.name for e in envs)
    raise ValueError(f"Unknown environment '{env_name}'; available environments: {names}.")


def load_environment_config(path: str | Path, env_name: str | None) -> EnvironmentConfig:
    """Load the config at ``path`` and select the named environment."""
    return select_environment(load_eval_config(path), env_name)


def create_environment(path: str | Path, env_name: str | None) -> Any:
    """Create a FRESH environment instance for the selected config.

    Never cached: llenvs environments enforce state continuity, so each
    concurrent episode needs its own instance.
    """
    return EnvironmentFactory.create(load_environment_config(path, env_name))


def _cache_key(path: str | Path, env_name: str | None) -> tuple[str, str]:
    return (config_key(path), load_environment_config(path, env_name).name)


def get_scorer(path: str | Path, env_name: str | None) -> Scorer:
    """Get a process-cached Scorer for the selected environment."""
    from llenvs.integrations.scoring import Scorer

    key = _cache_key(path, env_name)
    with _lock:
        scorer = _scorer_cache.get(key)
        if scorer is None:
            scorer = Scorer(create_environment(path, env_name))
            _scorer_cache[key] = scorer
    return scorer


def get_dataset_provider(path: str | Path, env_name: str | None) -> DatasetProvider:
    """Get a process-cached DatasetProvider for the selected environment."""
    from llenvs.integrations.dataset_provider import DatasetProvider

    key = _cache_key(path, env_name)
    with _lock:
        provider = _provider_cache.get(key)
        if provider is None:
            provider = DatasetProvider(create_environment(path, env_name))
            _provider_cache[key] = provider
    return provider


def resolve_config_system_prompt(cfg: EvalConfig, env_cfg: EnvironmentConfig) -> str | None:
    """Resolve the effective system prompt: env-level overrides eval-level."""
    value = env_cfg.system_prompt if env_cfg.system_prompt is not None else cfg.system_prompt
    if value is None:
        return None
    return resolve_system_prompt(value)
