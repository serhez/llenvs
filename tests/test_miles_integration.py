"""Tests for the miles trainer integration (config discovery, reward model, data export).

Covers the non-agent modules of ``llenvs.integrations.miles``:
- ``config``: env-config discovery via LLENVS_MILES_CONFIG, caches, session-isolation guard
- ``reward``: polymorphic ``reward_func`` for ``--custom-rm-path``
- ``data``: ``build_task_row`` / ``export_prompt_data`` JSONL exporter + CLI

The agent function is tested separately in ``tests/test_miles_agent.py``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from llenvs.core.config import EvalConfig
from llenvs.core.state import ImageContent, ObservationImages
from llenvs.integrations.dataset_provider import TaskItem
from llenvs.integrations.miles import config as miles_config
from llenvs.integrations.miles import data as miles_data
from llenvs.integrations.miles.reward import reward_func

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
  - name: env_a
    adapter: reasoning_gym
    size: 3
  - name: env_b
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
    seed: 42
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

JUDGE_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
    judge:
      model:
        backend: openai
        model: judge-model
        params:
          base_url: http://127.0.0.1:8000/v1
model:
  backend: openai
  model: test-model
"""

EVAL_JUDGE_LIST_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
model:
  backend: openai
  model: test-model
judge:
  - model:
      backend: openai
      model: judge-model
      params:
        base_url: http://127.0.0.1:8000/v1
"""

ENV_LLM_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
    env_llm:
      model:
        backend: openai
        model: env-model
        params:
          base_url: http://localhost:8000/v1
model:
  backend: openai
  model: test-model
"""

MULTI_TURN_YAML = """
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 3
    seed: 42
    iterative:
      max_turns: 2
model:
  backend: openai
  model: test-model
"""


def _write_config(tmp_path, yaml_text: str, name: str = "config.yaml"):
    path = tmp_path / name
    path.write_text(yaml_text)
    return path


@pytest.fixture(autouse=True)
def _clean_miles_state(monkeypatch):
    """Isolate every test from ambient env vars and module caches."""
    monkeypatch.delenv("LLENVS_MILES_CONFIG", raising=False)
    monkeypatch.delenv("LLENVS_MILES_ENV", raising=False)
    miles_config.clear_caches()
    yield
    miles_config.clear_caches()


@pytest.fixture()
def single_env_config(tmp_path, monkeypatch):
    path = _write_config(tmp_path, SINGLE_ENV_YAML)
    monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
    return path


@dataclass
class FakeSample:
    """Duck-typed stand-in for miles' Sample (only .response/.metadata are used)."""

    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _task_item(**overrides: Any) -> TaskItem:
    defaults: dict[str, Any] = {
        "task_index": 0,
        "prompt": "What is 2+2?",
        "messages": (),
        "ground_truth": "4",
        "metadata": {},
        "images": ObservationImages(),
    }
    defaults.update(overrides)
    return TaskItem(**defaults)


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------


class TestConfigDiscovery:
    def test_missing_config_raises_actionable_error(self):
        with pytest.raises(ValueError, match="LLENVS_MILES_CONFIG"):
            miles_config.load_eval_config(None)

    def test_nonexistent_config_path_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(tmp_path / "missing.yaml"))
        with pytest.raises(FileNotFoundError):
            miles_config.load_eval_config(None)

    def test_load_eval_config_from_env_var(self, single_env_config):
        cfg = miles_config.load_eval_config(None)
        assert isinstance(cfg, EvalConfig)
        assert cfg.environments[0].name == "leg_counting"

    def test_metadata_config_path_overrides_env_var(self, single_env_config, tmp_path):
        other = _write_config(tmp_path, MULTI_ENV_YAML, name="other.yaml")
        cfg = miles_config.load_eval_config({"llenvs_config": str(other)})
        assert [e.name for e in cfg.environments] == ["env_a", "env_b"]

    def test_single_env_is_default(self, single_env_config):
        env_cfg = miles_config.load_environment_config(None)
        assert env_cfg.name == "leg_counting"

    def test_env_selected_by_metadata(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, MULTI_ENV_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        env_cfg = miles_config.load_environment_config({"llenvs_env_name": "env_b"})
        assert env_cfg.name == "env_b"

    def test_env_selected_by_env_var(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, MULTI_ENV_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        monkeypatch.setenv("LLENVS_MILES_ENV", "env_a")
        env_cfg = miles_config.load_environment_config(None)
        assert env_cfg.name == "env_a"

    def test_ambiguous_env_raises_listing_names(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, MULTI_ENV_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        with pytest.raises(ValueError, match="env_a") as excinfo:
            miles_config.load_environment_config(None)
        assert "env_b" in str(excinfo.value)

    def test_unknown_env_raises_listing_names(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, MULTI_ENV_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        with pytest.raises(ValueError, match="nope") as excinfo:
            miles_config.load_environment_config({"llenvs_env_name": "nope"})
        assert "env_a" in str(excinfo.value)

    def test_create_environment_returns_fresh_instances(self, single_env_config):
        env1 = miles_config.create_environment(None)
        env2 = miles_config.create_environment(None)
        assert env1 is not env2

    def test_get_scorer_is_cached(self, single_env_config):
        s1 = miles_config.get_scorer(None)
        s2 = miles_config.get_scorer(None)
        assert s1 is s2

    def test_clear_caches_resets_scorer(self, single_env_config):
        s1 = miles_config.get_scorer(None)
        miles_config.clear_caches()
        s2 = miles_config.get_scorer(None)
        assert s1 is not s2

    def test_get_dataset_provider_is_cached(self, single_env_config):
        p1 = miles_config.get_dataset_provider(None)
        p2 = miles_config.get_dataset_provider(None)
        assert p1 is p2

    def test_system_prompt_eval_level(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, SYSTEM_PROMPT_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        assert miles_config.resolve_system_prompt_for(None) == "Be brief."

    def test_system_prompt_env_level_wins(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, ENV_SYSTEM_PROMPT_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        assert miles_config.resolve_system_prompt_for(None) == "Env-level prompt."

    def test_system_prompt_none_when_unset(self, single_env_config):
        assert miles_config.resolve_system_prompt_for(None) is None


# ---------------------------------------------------------------------------
# Session isolation guard
# ---------------------------------------------------------------------------


class TestSessionIsolationGuard:
    SESSION_URL = "http://localhost:8000/sessions/abc123"

    def test_judge_on_session_endpoint_refused(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, JUDGE_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        # judge points at 127.0.0.1:8000, session at localhost:8000 — same endpoint
        with pytest.raises(ValueError, match="session"):
            miles_config.ensure_isolated_from_session(self.SESSION_URL, None)

    def test_env_llm_on_session_endpoint_refused(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, ENV_LLM_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        with pytest.raises(ValueError, match="session"):
            miles_config.ensure_isolated_from_session(self.SESSION_URL, None)

    def test_eval_level_judge_list_refused(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, EVAL_JUDGE_LIST_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        with pytest.raises(ValueError, match="session"):
            miles_config.ensure_isolated_from_session(self.SESSION_URL, None)

    def test_distinct_port_passes(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, JUDGE_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        miles_config.ensure_isolated_from_session("http://localhost:9000/sessions/x", None)

    def test_no_judge_or_env_llm_passes(self, single_env_config):
        miles_config.ensure_isolated_from_session(self.SESSION_URL, None)


# ---------------------------------------------------------------------------
# build_task_row / export_prompt_data
# ---------------------------------------------------------------------------


class TestBuildTaskRow:
    def test_row_shape(self):
        row = miles_data.build_task_row(_task_item())
        assert set(row) == {"prompt", "label", "metadata"}
        assert row["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert row["label"] == "4"
        assert row["metadata"]["task_index"] == 0

    def test_messages_preferred_over_prompt(self):
        messages = ({"role": "user", "content": "turn 1"},)
        row = miles_data.build_task_row(_task_item(messages=messages))
        assert row["prompt"] == [{"role": "user", "content": "turn 1"}]

    def test_system_prompt_prepended(self):
        row = miles_data.build_task_row(_task_item(), system_prompt="Be brief.")
        assert row["prompt"][0] == {"role": "system", "content": "Be brief."}
        assert row["prompt"][1]["role"] == "user"

    def test_label_empty_when_no_ground_truth(self):
        row = miles_data.build_task_row(_task_item(ground_truth=None))
        assert row["label"] == ""

    def test_episode_id_dropped_for_determinism(self):
        item = _task_item(metadata={"episode_id": "some-uuid", "keep": 1})
        row = miles_data.build_task_row(item)
        assert "episode_id" not in row["metadata"]
        assert row["metadata"]["keep"] == 1

    def test_non_json_metadata_dropped_with_warning(self, caplog):
        item = _task_item(metadata={"keep": "yes", "drop": object()})
        with caplog.at_level("WARNING"):
            row = miles_data.build_task_row(item)
        assert row["metadata"]["keep"] == "yes"
        assert "drop" not in row["metadata"]
        assert "drop" in caplog.text

    def test_images_refused(self):
        img = ImageContent(data="aGk=", media_type="image/png")
        item = _task_item(images=ObservationImages(task=(img,)))
        with pytest.raises(ValueError, match="image"):
            miles_data.build_task_row(item)


class TestExportPromptData:
    def test_jsonl_round_trip(self, single_env_config, tmp_path):
        out = tmp_path / "tasks.jsonl"
        n = miles_data.export_prompt_data(single_env_config, out)
        assert n == 5
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert set(row) == {"prompt", "label", "metadata"}
            assert row["metadata"]["task_index"] == i
            assert row["prompt"][-1]["role"] == "user"

    def test_num_tasks_limits(self, single_env_config, tmp_path):
        out = tmp_path / "tasks.jsonl"
        n = miles_data.export_prompt_data(single_env_config, out, num_tasks=2)
        assert n == 2
        assert len(out.read_text().splitlines()) == 2

    def test_indices_select_tasks(self, single_env_config, tmp_path):
        out = tmp_path / "tasks.jsonl"
        miles_data.export_prompt_data(single_env_config, out, indices=[1, 3])
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["metadata"]["task_index"] for r in rows] == [1, 3]

    def test_accepts_eval_config_instance(self, single_env_config, tmp_path):
        cfg = EvalConfig.from_yaml(single_env_config)
        out = tmp_path / "tasks.jsonl"
        n = miles_data.export_prompt_data(cfg, out, num_tasks=1)
        assert n == 1

    def test_system_prompt_from_config_included(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, SYSTEM_PROMPT_YAML)
        out = tmp_path / "tasks.jsonl"
        miles_data.export_prompt_data(path, out, num_tasks=1)
        row = json.loads(out.read_text().splitlines()[0])
        assert row["prompt"][0] == {"role": "system", "content": "Be brief."}

    def test_cli_main(self, single_env_config, tmp_path):
        out = tmp_path / "cli_tasks.jsonl"
        rc = miles_data.main(
            ["--config", str(single_env_config), "--output", str(out), "--num-tasks", "2"]
        )
        assert rc == 0
        assert len(out.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# reward_func (--custom-rm-path)
# ---------------------------------------------------------------------------


class TestRewardFunc:
    def test_metadata_reward_short_circuit_single(self, single_env_config):
        sample = FakeSample(response="anything", metadata={"reward": 3.5, "task_index": 0})
        result = asyncio.run(reward_func(None, sample))
        assert isinstance(result, float)
        assert result == 3.5

    def test_batched_returns_list_in_order(self, single_env_config):
        samples = [
            FakeSample(metadata={"reward": 1.0, "task_index": 0}),
            FakeSample(metadata={"reward": 0.0, "task_index": 1}),
            FakeSample(metadata={"reward": 2.0, "task_index": 2}),
        ]
        result = asyncio.run(reward_func(None, samples))
        assert result == [1.0, 0.0, 2.0]

    def test_scorer_fallback_correct_answer(self, single_env_config):
        gt = miles_config.get_dataset_provider(None)[0].ground_truth
        assert gt is not None
        sample = FakeSample(response=f"<answer>{gt}</answer>", metadata={"task_index": 0})
        result = asyncio.run(reward_func(None, sample))
        assert result == 1.0

    def test_scorer_fallback_wrong_answer(self, single_env_config):
        sample = FakeSample(
            response="<answer>definitely wrong</answer>", metadata={"task_index": 0}
        )
        result = asyncio.run(reward_func(None, sample))
        assert result == 0.0

    def test_multi_turn_env_scores_zero(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, MULTI_TURN_YAML)
        monkeypatch.setenv("LLENVS_MILES_CONFIG", str(path))
        sample = FakeSample(response="<answer>1</answer>", metadata={"task_index": 0})
        result = asyncio.run(reward_func(None, sample))
        assert result == 0.0

    def test_missing_task_index_raises(self, single_env_config):
        sample = FakeSample(response="<answer>1</answer>", metadata={})
        with pytest.raises(ValueError, match="task_index"):
            asyncio.run(reward_func(None, sample))


# ---------------------------------------------------------------------------
# LLEnvsDataSource (--data-source-path)
# ---------------------------------------------------------------------------


@dataclass
class StubSample:
    """Duck-typed stand-in for miles' Sample as the DataSource builds it."""

    group_index: int | None = None
    index: int | None = None
    prompt: Any = ""
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _source_args(tmp_path, **overrides: Any):
    from types import SimpleNamespace

    defaults = {
        "n_samples_per_prompt": 2,
        "rollout_shuffle": False,
        "rollout_seed": 42,
        "save": str(tmp_path / "ckpt"),
        "load": str(tmp_path / "ckpt"),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture()
def data_source_cls(monkeypatch):
    from llenvs.integrations.miles import source as miles_source

    monkeypatch.setattr(miles_source, "_sample_cls", lambda: StubSample)
    return miles_source.LLEnvsDataSource


class TestLLEnvsDataSource:
    def _task_indices(self, groups: list[list[Any]]) -> list[int]:
        return [g[0].metadata["task_index"] for g in groups]

    def test_groups_of_n_samples_per_prompt(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))
        groups = src.get_samples(2)
        assert len(groups) == 2
        for group in groups:
            assert len(group) == 2
            assert group[0] is not group[1]  # deepcopies
            assert group[0].metadata["task_index"] == group[1].metadata["task_index"]

    def test_group_and_sample_indices_monotonic(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))
        groups = src.get_samples(2)
        assert [g[0].group_index for g in groups] == [0, 1]
        assert [s.index for g in groups for s in g] == [0, 1, 2, 3]

    def test_epoch_wraparound(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))  # 5 tasks in config
        assert self._task_indices(src.get_samples(3)) == [0, 1, 2]
        assert self._task_indices(src.get_samples(3)) == [3, 4, 0]

    def test_shuffle_is_deterministic(self, single_env_config, data_source_cls, tmp_path):
        args = _source_args(tmp_path, rollout_shuffle=True)
        order_a = self._task_indices(data_source_cls(args).get_samples(5))
        order_b = self._task_indices(data_source_cls(args).get_samples(5))
        assert order_a == order_b
        assert sorted(order_a) == [0, 1, 2, 3, 4]

    def test_buffer_drained_first(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))
        buffered = [
            StubSample(metadata={"task_index": 99}),
            StubSample(metadata={"task_index": 99}),
        ]
        src.add_samples([buffered])
        groups = src.get_samples(2)
        assert groups[0] is buffered
        assert groups[1][0].metadata["task_index"] == 0

    def test_add_samples_asserts_group_size(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))
        with pytest.raises(AssertionError):
            src.add_samples([[StubSample()]])  # group of 1, n_samples_per_prompt is 2

    def test_get_buffer_length(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))
        assert src.get_buffer_length() == 0
        src.add_samples([[StubSample(), StubSample()]])
        assert src.get_buffer_length() == 1

    def test_save_load_round_trip(self, single_env_config, data_source_cls, tmp_path):
        args = _source_args(tmp_path)
        src_a = data_source_cls(args)
        src_a.get_samples(3)
        src_a.save(7)
        expected = src_a.get_samples(2)

        src_b = data_source_cls(args)
        src_b.load(7)
        resumed = src_b.get_samples(2)
        assert self._task_indices(resumed) == self._task_indices(expected)
        assert [g[0].group_index for g in resumed] == [g[0].group_index for g in expected]
        assert [s.index for g in resumed for s in g] == [s.index for g in expected for s in g]

    def test_load_missing_checkpoint_is_noop(self, single_env_config, data_source_cls, tmp_path):
        src = data_source_cls(_source_args(tmp_path))
        src.load(123)  # nothing saved yet
        assert self._task_indices(src.get_samples(1)) == [0]

    def test_rows_match_exporter(self, single_env_config, data_source_cls, tmp_path):
        provider = miles_config.get_dataset_provider(None)
        expected_row = miles_data.build_task_row(provider[0])
        src = data_source_cls(_source_args(tmp_path))
        sample = src.get_samples(1)[0][0]
        assert sample.prompt == expected_row["prompt"]
        assert sample.label == expected_row["label"]
        assert sample.metadata == expected_row["metadata"]


# ---------------------------------------------------------------------------
# v2 session postprocessor
# ---------------------------------------------------------------------------


@dataclass
class FakeLeafSample:
    """Duck-typed stand-in for a miles v2 leaf Sample."""

    tokens: list[int]
    response_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _event(response_id: str, value: float) -> dict[str, Any]:
    return {"response_id": response_id, "value": value, "signals": {"step_reward": value}}


def _linear_session(
    *,
    spans: list[tuple[int, int]],
    values: list[float],
    prompt_len: int = 10,
    response_length: int | None = None,
) -> tuple[FakeLeafSample, dict[str, Any]]:
    """One leaf whose path has one node per (span, value) pair."""
    nodes = [
        {
            "id": i + 1,
            "parent": i or None,
            "completion_span": list(span),
            "response_id": f"cmpl-{i + 1}",
        }
        for i, span in enumerate(spans)
    ]
    leaf = {"node_id": len(nodes), "path_node_ids": [n["id"] for n in nodes]}
    total = max(end for _, end in spans)
    if response_length is None:
        response_length = total - prompt_len
    sample = FakeLeafSample(
        tokens=list(range(prompt_len + response_length)),
        response_length=response_length,
        metadata={"leaf": leaf},
    )
    session_metadata = {
        "tree": {"nodes": nodes, "leaves": [leaf]},
        "agent": {"reward_events": [_event(f"cmpl-{i + 1}", v) for i, v in enumerate(values)]},
    }
    return sample, session_metadata


class TestAttachDecisionData:
    def test_linear_spans_rebased_to_response(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        # prompt 10 tokens; turn 1 completion [10,14), turn 2 completion [16,18)
        sample, session_metadata = _linear_session(spans=[(10, 14), (16, 18)], values=[1.0, 2.0])
        attach_decision_data([sample], session_metadata)
        assert sample.metadata["decision_spans"] == [[0, 4], [6, 8]]
        assert sample.metadata["decision_rewards"] == [1.0, 2.0]

    def test_truncated_final_span_clamped_out(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        # response truncated to 6 tokens: the [16,18) span falls off the end
        sample, session_metadata = _linear_session(
            spans=[(10, 14), (16, 18)], values=[1.0, 2.0], response_length=6
        )
        attach_decision_data([sample], session_metadata)
        assert sample.metadata["decision_spans"] == [[0, 4]]
        assert sample.metadata["decision_rewards"] == [1.0]

    def test_node_without_event_skipped(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        sample, session_metadata = _linear_session(spans=[(10, 14), (16, 18)], values=[1.0, 2.0])
        # drop the second turn's event (e.g. the env step after it failed)
        session_metadata["agent"]["reward_events"] = session_metadata["agent"]["reward_events"][:1]
        attach_decision_data([sample], session_metadata)
        assert sample.metadata["decision_spans"] == [[0, 4]]
        assert sample.metadata["decision_rewards"] == [1.0]

    def test_branched_tree_per_leaf_paths(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        nodes = [
            {"id": 1, "parent": None, "completion_span": [10, 14], "response_id": "cmpl-1"},
            {"id": 2, "parent": 1, "completion_span": [16, 18], "response_id": "cmpl-2"},
            {"id": 3, "parent": 1, "completion_span": [16, 20], "response_id": "cmpl-3"},
        ]
        leaf_a = {"node_id": 2, "path_node_ids": [1, 2]}
        leaf_b = {"node_id": 3, "path_node_ids": [1, 3]}
        sample_a = FakeLeafSample(
            tokens=list(range(18)), response_length=8, metadata={"leaf": leaf_a}
        )
        sample_b = FakeLeafSample(
            tokens=list(range(20)), response_length=10, metadata={"leaf": leaf_b}
        )
        session_metadata = {
            "tree": {"nodes": nodes, "leaves": [leaf_a, leaf_b]},
            "agent": {
                "reward_events": [
                    _event("cmpl-1", 1.0),
                    _event("cmpl-2", 2.0),
                    _event("cmpl-3", 3.0),
                ]
            },
        }
        attach_decision_data([sample_a, sample_b], session_metadata)
        assert sample_a.metadata["decision_spans"] == [[0, 4], [6, 8]]
        assert sample_a.metadata["decision_rewards"] == [1.0, 2.0]
        assert sample_b.metadata["decision_spans"] == [[0, 4], [6, 10]]
        assert sample_b.metadata["decision_rewards"] == [1.0, 3.0]

    def test_no_agent_metadata_is_noop(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        sample, session_metadata = _linear_session(spans=[(10, 14)], values=[1.0])
        del session_metadata["agent"]
        attach_decision_data([sample], session_metadata)
        assert "decision_spans" not in sample.metadata

    def test_no_reward_events_is_noop(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        sample, session_metadata = _linear_session(spans=[(10, 14)], values=[1.0])
        session_metadata["agent"] = {"reward": 1.0}  # Tier-0 agent without events
        attach_decision_data([sample], session_metadata)
        assert "decision_spans" not in sample.metadata

    def test_sample_without_leaf_skipped(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        sample, session_metadata = _linear_session(spans=[(10, 14)], values=[1.0])
        sample.metadata = {}
        attach_decision_data([sample], session_metadata)
        assert "decision_spans" not in sample.metadata

    def test_output_json_serializable(self):
        from llenvs.integrations.miles.postprocess import attach_decision_data

        sample, session_metadata = _linear_session(spans=[(10, 14), (16, 18)], values=[1.0, 2.0])
        attach_decision_data([sample], session_metadata)
        json.dumps(sample.metadata)


class TestPostprocess:
    def test_delegates_then_attaches(self, monkeypatch):
        from llenvs.integrations.miles import postprocess as pp

        sample, session_metadata = _linear_session(spans=[(10, 14)], values=[1.0])
        calls: list[Any] = []

        def fake_default(leaf_samples, metadata):
            calls.append((leaf_samples, metadata))
            return list(leaf_samples)

        monkeypatch.setattr(pp, "_default_postprocess", lambda: fake_default)
        result = pp.postprocess([sample], session_metadata)
        assert calls == [([sample], session_metadata)]
        assert result[0].metadata["decision_spans"] == [[0, 4]]

    def test_postprocess_is_sync(self):
        from llenvs.integrations.miles import postprocess as pp

        assert not asyncio.iscoroutinefunction(pp.postprocess)

    def test_importing_postprocess_does_not_require_miles(self):
        import llenvs.integrations.miles.postprocess  # noqa: F401

    def test_real_default_postprocess_delegation(self):
        pytest.importorskip("miles")
        from llenvs.integrations.miles import postprocess as pp

        assert callable(pp._default_postprocess())
