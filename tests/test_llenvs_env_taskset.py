"""Tests for ``llenvs_env.taskset`` — the ``llenvs-env`` verifiers v1 taskset.

Requires verifiers v1 (skipped otherwise). Single-turn cases run on the real
reasoning_gym adapter; relay cases use the mock env from ``llenvs_env_stubs``.
"""

# ruff: noqa: E402, I001  (imports follow the importorskip guard)
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")

from pydantic import ValidationError

from llenvs_env import _config
from llenvs_env.taskset import (
    LLEnvsData,
    LLEnvsTask,
    LLEnvsTaskset,
    LLEnvsTasksetConfig,
)
from tests.llenvs_env_stubs import (
    commit_reply,
    ensure_registered,
    mint_trace,
    opening_messages,
)

ensure_registered()

SINGLE_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 5
    seed: 42
model:
  backend: openai
  model: test-model
"""

SHUFFLE_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 20
    seed: 42
model:
  backend: openai
  model: test-model
"""

MULTI_YAML = """
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


def _relay_yaml(**params: object) -> str:
    lines = "\n".join(f"      {key}: {str(value).lower()}" for key, value in params.items())
    return f"""
environments:
  - name: relay
    adapter: llenvs_env_test
    params:
{lines}
model:
  backend: openai
  model: test-model
"""


RELAY_YAML = _relay_yaml(num_tasks=3, total_steps=2)
TOOLS_YAML = _relay_yaml(num_tasks=2, tools=True)
TOOLS_WITH_PROMPT_YAML = TOOLS_YAML + 'system_prompt: "Be brief."\n'
IMAGES_YAML = _relay_yaml(num_tasks=2, images=True)
HISTORY_YAML = _relay_yaml(num_tasks=2, history=True)


def _write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _taskset(path: Path, **kwargs: object) -> LLEnvsTaskset:
    return LLEnvsTaskset(LLEnvsTasksetConfig(id="llenvs-env", config=path, **kwargs))


def _tasks(path: Path, **kwargs: object) -> list[LLEnvsTask]:
    return list(_taskset(path, **kwargs))


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_caches():
    _config.clear_caches()
    yield
    _config.clear_caches()


@pytest.fixture()
def single(tmp_path: Path) -> Path:
    return _write(tmp_path, SINGLE_YAML)


@pytest.fixture()
def relay(tmp_path: Path) -> Path:
    return _write(tmp_path, RELAY_YAML, name="relay.yaml")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestTasksetConfig:
    def test_constructs_from_id_alone(self):
        # The verifiers CLI narrows the run config by instantiating the taskset
        # config from its id before any flag is parsed; a required field breaks it.
        cfg = LLEnvsTasksetConfig(id="llenvs-env")
        assert cfg.config is None

    def test_config_path_is_required_to_load(self):
        with pytest.raises(ValueError, match="env.taskset.config"):
            list(LLEnvsTaskset(LLEnvsTasksetConfig(id="llenvs-env")))

    def test_defaults(self, single):
        cfg = LLEnvsTasksetConfig(id="llenvs-env", config=single)
        assert isinstance(cfg, vf.TasksetConfig)
        assert cfg.config == single
        assert cfg.env_name is None
        assert cfg.num_tasks is None
        assert cfg.shuffle_seed is None

    def test_num_tasks_must_be_positive(self, single):
        with pytest.raises(ValidationError):
            LLEnvsTasksetConfig(id="llenvs-env", config=single, num_tasks=0)


# ---------------------------------------------------------------------------
# Class-level surface
# ---------------------------------------------------------------------------


class TestTasksetClass:
    def test_task_type(self):
        assert LLEnvsTaskset.task_type() is LLEnvsTask

    def test_data_type(self):
        assert LLEnvsTask.data_type() is LLEnvsData

    def test_finite(self):
        assert LLEnvsTaskset.INFINITE is False

    def test_no_toolsets(self, single):
        cfg = LLEnvsTasksetConfig(id="llenvs-env", config=single)
        assert LLEnvsTaskset.toolsets(cfg) == []

    def test_not_containerized(self):
        assert LLEnvsTask.NEEDS_CONTAINER is False


# ---------------------------------------------------------------------------
# Loading single-turn tasks
# ---------------------------------------------------------------------------


class TestLoadSingleTurn:
    def test_one_task_per_llenvs_task(self, single):
        tasks = _tasks(single)
        assert len(tasks) == 5
        assert all(isinstance(t, LLEnvsTask) for t in tasks)
        assert [t.data.idx for t in tasks] == [0, 1, 2, 3, 4]
        assert [t.data.task_index for t in tasks] == [0, 1, 2, 3, 4]

    def test_task_data_mapping(self, single):
        task = _tasks(single)[0]
        data = task.data
        assert isinstance(data, LLEnvsData)
        assert data.name == "leg_counting#0"
        assert data.env_name == "leg_counting"
        assert data.config_path == _config.config_key(single)
        assert isinstance(data.prompt, str) and data.prompt
        assert isinstance(data.answer, str) and data.answer
        assert data.multi_turn is False
        assert data.system_prompt is None
        assert "episode_id" not in data.info

    def test_prompt_and_answer_match_dataset_provider(self, single):
        provider = _config.get_dataset_provider(single, None)
        task = _tasks(single)[2]
        assert task.data.prompt == provider[2].prompt
        assert task.data.answer == provider[2].ground_truth

    def test_key_is_durable_identity(self, single):
        tasks = _tasks(single)
        assert tasks[3].key == "llenvs:leg_counting:3"
        assert len({t.key for t in tasks}) == 5

    def test_hash_is_stable_across_iterations(self, single):
        assert [t.hash for t in _tasks(single)] == [t.hash for t in _tasks(single)]

    def test_num_tasks_limits(self, single):
        assert [t.data.task_index for t in _tasks(single, num_tasks=2)] == [0, 1]

    def test_shuffle_seed_is_a_deterministic_permutation(self, tmp_path):
        path = _write(tmp_path, SHUFFLE_YAML)
        first = [t.data.task_index for t in _tasks(path, shuffle_seed=7)]
        second = [t.data.task_index for t in _tasks(path, shuffle_seed=7)]
        assert first == second
        assert sorted(first) == list(range(20))
        assert first != list(range(20))

    def test_shuffle_then_head_is_a_random_subset(self, tmp_path):
        path = _write(tmp_path, SHUFFLE_YAML)
        full = [t.data.task_index for t in _tasks(path, shuffle_seed=7)]
        subset = [t.data.task_index for t in _tasks(path, shuffle_seed=7, num_tasks=3)]
        assert subset == full[:3]

    def test_idx_follows_yield_order(self, tmp_path):
        path = _write(tmp_path, SHUFFLE_YAML)
        tasks = _tasks(path, shuffle_seed=7, num_tasks=4)
        assert [t.data.idx for t in tasks] == [0, 1, 2, 3]

    def test_different_seeds_differ(self, tmp_path):
        path = _write(tmp_path, SHUFFLE_YAML)
        a = [t.data.task_index for t in _tasks(path, shuffle_seed=1)]
        b = [t.data.task_index for t in _tasks(path, shuffle_seed=2)]
        assert a != b

    def test_head_view(self, single):
        assert len(list(_taskset(single).head(2))) == 2

    def test_multi_env_requires_env_name(self, tmp_path):
        path = _write(tmp_path, MULTI_YAML)
        with pytest.raises(ValueError, match="env_name"):
            _tasks(path)

    def test_env_name_selects(self, tmp_path):
        path = _write(tmp_path, MULTI_YAML)
        task = _tasks(path, env_name="chain_sum")[0]
        assert task.data.env_name == "chain_sum"
        assert task.data.name == "chain_sum#0"
        assert task.key == "llenvs:chain_sum:0"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _tasks(tmp_path / "missing.yaml")

    def test_system_prompt_from_config(self, tmp_path):
        path = _write(tmp_path, SYSTEM_PROMPT_YAML)
        assert _tasks(path)[0].data.system_prompt == "Be brief."

    def test_vf_system_prompt_override_wins(self, tmp_path):
        path = _write(tmp_path, SYSTEM_PROMPT_YAML)
        override = tmp_path / "best_system_prompt.txt"
        override.write_text("Override.")
        tasks = _tasks(path, system_prompt=override)
        assert tasks[0].data.system_prompt == "Override."


# ---------------------------------------------------------------------------
# Loading relay (multi-turn) tasks
# ---------------------------------------------------------------------------


class TestLoadRelay:
    def test_multi_turn_flag(self, relay):
        tasks = _tasks(relay)
        assert len(tasks) == 3
        assert all(t.data.multi_turn for t in tasks)
        assert tasks[0].data.answer is None

    def test_tools_preamble_in_system_prompt(self, tmp_path):
        path = _write(tmp_path, TOOLS_YAML)
        system_prompt = _tasks(path)[0].data.system_prompt
        assert system_prompt is not None
        assert "<tools>" in system_prompt
        assert "lookup" in system_prompt

    def test_tools_preamble_appended_after_config_prompt(self, tmp_path):
        path = _write(tmp_path, TOOLS_WITH_PROMPT_YAML)
        system_prompt = _tasks(path)[0].data.system_prompt
        assert system_prompt is not None
        assert system_prompt.startswith("Be brief.")
        assert "<tools>" in system_prompt

    def test_no_tools_leaves_system_prompt_unset(self, relay):
        assert _tasks(relay)[0].data.system_prompt is None

    def test_images_refused(self, tmp_path):
        path = _write(tmp_path, IMAGES_YAML)
        with pytest.raises(NotImplementedError, match="text-only"):
            _tasks(path)

    def test_history_becomes_typed_messages(self, tmp_path):
        path = _write(tmp_path, HISTORY_YAML)
        prompt = _tasks(path)[0].data.prompt
        assert isinstance(prompt, list)
        assert [m.role for m in prompt] == ["user", "assistant", "user"]
        assert prompt[0].content.startswith("Task 0")
        assert prompt[1].content == "Hello."
        assert prompt[2].content == "Go."


# ---------------------------------------------------------------------------
# Task hooks
# ---------------------------------------------------------------------------


class TestTaskHooks:
    def test_stop_hook_registered(self, single):
        task = _tasks(single)[0]
        assert [fn.__name__ for fn in task.hooks("stop")] == ["single_turn"]

    def test_reward_hook_registered_with_unit_weight(self, single):
        task = _tasks(single)[0]
        rewards = task.hooks("reward")
        assert [fn.__name__ for fn in rewards] == ["llenvs_total"]
        assert rewards[0]._vf_weight == 1.0

    def test_no_metric_hooks(self, single):
        assert _tasks(single)[0].hooks("metric") == []

    def test_single_turn_stop_bites_after_first_reply(self, single):
        task = _tasks(single)[0]
        trace = mint_trace(task)
        assert _run(task.single_turn(trace)) is False
        commit_reply(trace, opening_messages(task.data), "42")
        assert _run(task.single_turn(trace)) is True

    def test_single_turn_stop_never_bites_for_multi_turn(self, relay):
        task = _tasks(relay)[0]
        trace = mint_trace(task)
        commit_reply(trace, opening_messages(task.data), "step")
        assert _run(task.single_turn(trace)) is False

    def test_offline_score_correct_answer(self, single):
        task = _tasks(single)[0]
        trace = mint_trace(task)
        commit_reply(trace, opening_messages(task.data), f"<answer>{task.data.answer}</answer>")
        _run(task.score(trace, None))
        assert trace.rewards["llenvs_total"].score == 1.0
        assert trace.rewards["llenvs_total"].weight == 1.0
        assert trace.reward == 1.0
        assert any(name.startswith("llenvs/") for name in trace.metrics)

    def test_offline_score_wrong_answer(self, single):
        task = _tasks(single)[0]
        trace = mint_trace(task)
        commit_reply(trace, opening_messages(task.data), "<answer>nope</answer>")
        _run(task.score(trace, None))
        assert trace.reward == 0.0

    def test_offline_score_reuses_cached_scorer(self, single):
        tasks = _tasks(single)
        for task in tasks[:2]:
            trace = mint_trace(task)
            commit_reply(trace, opening_messages(task.data), "<answer>1</answer>")
            _run(task.score(trace, None))
        assert len(_config._scorer_cache) == 1

    def test_relay_guard_contributes_zero(self, relay):
        task = _tasks(relay)[0]
        trace = mint_trace(task)
        commit_reply(trace, opening_messages(task.data), "step")
        trace.info["llenvs"] = {"turn_rewards": [0.5]}
        trace.record_reward("llenvs/progress", 0.5)
        _run(task.score(trace, None))
        assert trace.rewards["llenvs_total"].score == 0.0
        assert trace.reward == pytest.approx(0.5)

    def test_multi_turn_task_refuses_offline_scoring(self, relay):
        task = _tasks(relay)[0]
        trace = mint_trace(task)
        commit_reply(trace, opening_messages(task.data), "step")
        with pytest.raises(Exception, match="llenvs-env"):
            _run(task.score(trace, None))
