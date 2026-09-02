"""Tests for ``llenvs_env._config`` — EvalConfig loading, selection, and caches.

This module imports no verifiers symbols, so it runs in the base venv.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from llenvs.core.config import EnvironmentConfig, EvalConfig
from llenvs.integrations.dataset_provider import DatasetProvider
from llenvs.integrations.scoring import Scorer
from llenvs_env import _config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SINGLE_ENV_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 5
    seed: 42
model:
  backend: openai
  model: test-model
"""

MULTI_ENV_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
  - name: chain_sum
    adapter: reasoning_gym
    size: 3
model:
  backend: openai
  model: test-model
"""

NO_ENV_YAML = """
environments: []
model:
  backend: openai
  model: test-model
"""

SYSTEM_PROMPT_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
model:
  backend: openai
  model: test-model
system_prompt: "Be brief."
"""

ENV_SYSTEM_PROMPT_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
    system_prompt: "Env-level prompt."
model:
  backend: openai
  model: test-model
system_prompt: "Eval-level prompt."
"""

REGISTRY_SYSTEM_PROMPT_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
model:
  backend: openai
  model: test-model
system_prompt: general_reasoning
"""

LIST_SYSTEM_PROMPT_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
model:
  backend: openai
  model: test-model
system_prompt:
  - "First part."
  - "Second part."
"""


def _write_config(tmp_path: Path, yaml_text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml_text)
    return path


@pytest.fixture(autouse=True)
def _clean_caches():
    _config.clear_caches()
    yield
    _config.clear_caches()


@pytest.fixture()
def single_env_config(tmp_path: Path) -> Path:
    return _write_config(tmp_path, SINGLE_ENV_YAML)


@pytest.fixture()
def multi_env_config(tmp_path: Path) -> Path:
    return _write_config(tmp_path, MULTI_ENV_YAML, name="multi.yaml")


# ---------------------------------------------------------------------------
# load_eval_config
# ---------------------------------------------------------------------------


class TestLoadEvalConfig:
    def test_loads_yaml(self, single_env_config):
        cfg = _config.load_eval_config(single_env_config)
        assert isinstance(cfg, EvalConfig)
        assert cfg.environments[0].name == "leg_counting"

    def test_accepts_str_and_path(self, single_env_config):
        from_path = _config.load_eval_config(single_env_config)
        from_str = _config.load_eval_config(str(single_env_config))
        assert from_path is from_str

    def test_cached_by_resolved_path(self, single_env_config):
        # Different spellings of the same file share one cache entry.
        spelled = single_env_config.parent / "sub" / ".." / single_env_config.name
        assert _config.load_eval_config(single_env_config) is _config.load_eval_config(spelled)

    def test_distinct_files_are_distinct_entries(self, single_env_config, multi_env_config):
        assert _config.load_eval_config(single_env_config) is not _config.load_eval_config(
            multi_env_config
        )

    def test_clear_caches_reloads(self, single_env_config):
        first = _config.load_eval_config(single_env_config)
        _config.clear_caches()
        assert _config.load_eval_config(single_env_config) is not first

    def test_missing_file_raises_file_not_found_naming_path(self, tmp_path):
        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            _config.load_eval_config(missing)

    def test_config_key_is_resolved_absolute_path(self, single_env_config):
        key = _config.config_key(single_env_config.parent / "sub" / ".." / "config.yaml")
        assert key == str(single_env_config.resolve())
        assert Path(key).is_absolute()


# ---------------------------------------------------------------------------
# select_environment / load_environment_config
# ---------------------------------------------------------------------------


class TestSelectEnvironment:
    def test_sole_entry_is_default(self, single_env_config):
        cfg = _config.load_eval_config(single_env_config)
        env_cfg = _config.select_environment(cfg, None)
        assert isinstance(env_cfg, EnvironmentConfig)
        assert env_cfg.name == "leg_counting"

    def test_explicit_name(self, multi_env_config):
        cfg = _config.load_eval_config(multi_env_config)
        assert _config.select_environment(cfg, "chain_sum").name == "chain_sum"

    def test_ambiguous_without_name_lists_names_and_field(self, multi_env_config):
        cfg = _config.load_eval_config(multi_env_config)
        with pytest.raises(ValueError, match="leg_counting") as excinfo:
            _config.select_environment(cfg, None)
        message = str(excinfo.value)
        assert "chain_sum" in message
        assert "env_name" in message  # points at the taskset config field

    def test_unknown_name_lists_available(self, multi_env_config):
        cfg = _config.load_eval_config(multi_env_config)
        with pytest.raises(ValueError, match="nope") as excinfo:
            _config.select_environment(cfg, "nope")
        assert "leg_counting" in str(excinfo.value)
        assert "chain_sum" in str(excinfo.value)

    def test_no_environments_raises(self, tmp_path):
        path = _write_config(tmp_path, NO_ENV_YAML)
        cfg = _config.load_eval_config(path)
        with pytest.raises(ValueError, match="no environments"):
            _config.select_environment(cfg, None)

    def test_load_environment_config_combines_both(self, multi_env_config):
        env_cfg = _config.load_environment_config(multi_env_config, "leg_counting")
        assert env_cfg.name == "leg_counting"

    def test_load_environment_config_sole_entry(self, single_env_config):
        assert _config.load_environment_config(single_env_config, None).name == "leg_counting"


# ---------------------------------------------------------------------------
# Environments, Scorers, DatasetProviders
# ---------------------------------------------------------------------------


class TestEnvironmentObjects:
    def test_create_environment_returns_fresh_instances(self, single_env_config):
        env1 = _config.create_environment(single_env_config, None)
        env2 = _config.create_environment(single_env_config, None)
        assert env1 is not env2
        assert len(env1) == 5

    def test_get_scorer_is_cached(self, single_env_config):
        s1 = _config.get_scorer(single_env_config, None)
        s2 = _config.get_scorer(single_env_config, None)
        assert isinstance(s1, Scorer)
        assert s1 is s2

    def test_get_scorer_cache_keyed_by_env_name(self, multi_env_config):
        a = _config.get_scorer(multi_env_config, "leg_counting")
        b = _config.get_scorer(multi_env_config, "chain_sum")
        assert a is not b
        assert _config.get_scorer(multi_env_config, "leg_counting") is a

    def test_get_scorer_cache_keyed_by_config_path(self, tmp_path):
        first = _write_config(tmp_path, SINGLE_ENV_YAML, name="first.yaml")
        second = _write_config(tmp_path, SINGLE_ENV_YAML, name="second.yaml")
        assert _config.get_scorer(first, None) is not _config.get_scorer(second, None)

    def test_clear_caches_resets_scorer(self, single_env_config):
        s1 = _config.get_scorer(single_env_config, None)
        _config.clear_caches()
        assert _config.get_scorer(single_env_config, None) is not s1

    def test_get_dataset_provider_is_cached(self, single_env_config):
        p1 = _config.get_dataset_provider(single_env_config, None)
        p2 = _config.get_dataset_provider(single_env_config, None)
        assert isinstance(p1, DatasetProvider)
        assert p1 is p2
        assert len(p1) == 5

    def test_get_scorer_concurrent_callers_share_one_instance(self, single_env_config):
        # Scorers are created inside verifiers worker threads; the cache must
        # hand every concurrent caller the same object.
        results: list[Scorer] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            results.append(_config.get_scorer(single_env_config, None))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 8
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# System prompt resolution
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def _resolve(self, path: Path) -> str | None:
        cfg = _config.load_eval_config(path)
        return _config.resolve_config_system_prompt(cfg, _config.select_environment(cfg, None))

    def test_none_when_unset(self, single_env_config):
        assert self._resolve(single_env_config) is None

    def test_eval_level(self, tmp_path):
        assert self._resolve(_write_config(tmp_path, SYSTEM_PROMPT_YAML)) == "Be brief."

    def test_env_level_overrides_eval_level(self, tmp_path):
        assert self._resolve(_write_config(tmp_path, ENV_SYSTEM_PROMPT_YAML)) == "Env-level prompt."

    def test_registry_name_resolves_to_text(self, tmp_path):
        resolved = self._resolve(_write_config(tmp_path, REGISTRY_SYSTEM_PROMPT_YAML))
        assert resolved is not None
        assert resolved != "general_reasoning"
        assert len(resolved) > len("general_reasoning")

    def test_list_value_joins_parts(self, tmp_path):
        resolved = self._resolve(_write_config(tmp_path, LIST_SYSTEM_PROMPT_YAML))
        assert resolved == "First part.\n\nSecond part."
