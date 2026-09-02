"""Tests for ``llenvs_env.env`` — the relay loop and reward recording.

Requires verifiers v1 (skipped otherwise). ``LLEnvsEnv.run(task, agents)`` is
driven with the stub Agents/Interaction from ``llenvs_env_stubs`` against the
scripted ``MockRelayEnv``, so no model or server is involved.
"""

# ruff: noqa: E402, I001  (imports follow the importorskip guard)
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")

import llenvs_env
from llenvs.core.state import Action
from llenvs_env import _config
from llenvs_env.env import LLEnvsEnv, LLEnvsEnvConfig
from llenvs_env.taskset import LLEnvsTask, LLEnvsTaskset, LLEnvsTasksetConfig
from tests.llenvs_env_stubs import MockRelayEnv, StubAgents, ensure_registered

ensure_registered()

TOOL_CALL_REPLY = (
    "Let me look.\n<tool_call>\n"
    + json.dumps({"name": "lookup", "arguments": {"q": "x"}})
    + "\n</tool_call>"
)


def _relay_config(tmp_path: Path, **params: object) -> Path:
    lines = "\n".join(f"      {key}: {json.dumps(value)}" for key, value in params.items())
    text = f"""
environments:
  - name: relay
    adapter: llenvs_env_test
    params:
{lines}
model:
  backend: openai
  model: test-model
"""
    path = tmp_path / "relay.yaml"
    path.write_text(text)
    return path


def _make_env(path: Path, **env_kwargs: object) -> LLEnvsEnv:
    config = LLEnvsEnvConfig(taskset={"id": "llenvs-env", "config": str(path)}, **env_kwargs)
    return LLEnvsEnv(config)


def _first_task(env: LLEnvsEnv) -> LLEnvsTask:
    task = next(iter(env.taskset))
    assert isinstance(task, LLEnvsTask)
    return task


def _run(env: LLEnvsEnv, task: vf.Task, replies: list[str]):
    agents = StubAgents(replies)
    asyncio.run(env.run(task, agents))
    return agents.agent.completed[-1], MockRelayEnv.instances[-1], agents


def _episode(tmp_path: Path, replies: list[str], *, env_kwargs: dict | None = None, **params):
    env = _make_env(_relay_config(tmp_path, **params), **(env_kwargs or {}))
    return _run(env, _first_task(env), replies)


@pytest.fixture(autouse=True)
def _clean():
    _config.clear_caches()
    MockRelayEnv.instances.clear()
    yield
    _config.clear_caches()
    MockRelayEnv.instances.clear()


# ---------------------------------------------------------------------------
# Plugin surface
# ---------------------------------------------------------------------------


class TestPlugin:
    def test_exports(self):
        from llenvs_env.harness import LLEnvsHarness

        assert llenvs_env.__all__ == ["LLEnvsTaskset", "LLEnvsEnv", "LLEnvsHarness"]
        assert llenvs_env.LLEnvsTaskset is LLEnvsTaskset
        assert llenvs_env.LLEnvsEnv is LLEnvsEnv
        assert llenvs_env.LLEnvsHarness is LLEnvsHarness

    def test_loader_resolves_plugin_by_id(self):
        from verifiers.v1.utils.loaders import environment_class, taskset_config_type

        assert taskset_config_type("llenvs-env") is LLEnvsTasksetConfig
        assert environment_class("llenvs-env") is LLEnvsEnv

    def test_env_config_declares_one_agent_seat(self):
        assert issubclass(LLEnvsEnvConfig, vf.EnvConfig)
        seats = [
            name
            for name, field in LLEnvsEnvConfig.model_fields.items()
            if isinstance(field.default, vf.AgentConfig)
        ]
        assert seats == ["agent"]
        assert LLEnvsEnvConfig.model_fields["max_steps"].default is None

    def test_env_loads_taskset(self, tmp_path):
        env = _make_env(_relay_config(tmp_path, num_tasks=2))
        assert isinstance(env.taskset, LLEnvsTaskset)
        assert isinstance(env.config, LLEnvsEnvConfig)


# ---------------------------------------------------------------------------
# Relay loop
# ---------------------------------------------------------------------------


class TestRelayLoop:
    def test_full_episode_terminated_by_env(self, tmp_path):
        trace, mock, agents = _episode(tmp_path, ["one", "two"], total_steps=2)
        assert trace.stop_condition == "env_terminated"
        assert trace.is_completed
        assert [a.text for a in mock.actions] == ["one", "two"]
        assert mock.closed
        assert agents.agent.interactions[0].turns == [None, "Step 1 done."]

    def test_opens_with_bare_turn_then_feedback(self, tmp_path):
        _, _, agents = _episode(tmp_path, ["a", "b", "c"], total_steps=3)
        assert agents.agent.interactions[0].turns == [None, "Step 1 done.", "Step 2 done."]

    def test_env_truncated(self, tmp_path):
        trace, mock, _ = _episode(tmp_path, ["a", "b", "c"], total_steps=3, truncate_at=1)
        assert trace.stop_condition == "env_truncated"
        assert len(mock.actions) == 1

    def test_max_steps_from_env_config(self, tmp_path):
        trace, mock, _ = _episode(
            tmp_path, ["a", "b", "c"], env_kwargs={"max_steps": 1}, total_steps=3
        )
        assert trace.stop_condition == "max_steps"
        assert len(mock.actions) == 1

    def test_max_steps_defaults_to_spec(self, tmp_path):
        trace, mock, _ = _episode(tmp_path, ["a"] * 5, total_steps=5, max_steps=2)
        assert trace.stop_condition == "max_steps"
        assert len(mock.actions) == 2

    def test_empty_reply_stops_without_stepping(self, tmp_path):
        trace, mock, _ = _episode(tmp_path, [""], total_steps=2)
        assert trace.stop_condition == "empty_action"
        assert mock.actions == []
        assert trace.info["llenvs"]["turn_rewards"] == []

    def test_terminated_segment_leaves_stop_to_framework(self, tmp_path):
        # One scripted reply, then the stub ends the run (as max_turns would).
        trace, mock, _ = _episode(tmp_path, ["a"], total_steps=3)
        assert trace.stop_condition == "user_closed"
        assert trace.info["llenvs"]["stop"] is None
        assert len(mock.actions) == 1
        assert trace.info["llenvs"]["turn_rewards"] == [0.5]

    def test_env_error_propagates_and_closes(self, tmp_path):
        env = _make_env(_relay_config(tmp_path, total_steps=3, fail_on_step=1))
        with pytest.raises(RuntimeError, match="exploded"):
            _run(env, _first_task(env), ["a", "b"])
        assert MockRelayEnv.instances[-1].closed

    def test_reset_and_step_run_off_the_event_loop_thread(self, tmp_path):
        _, mock, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        assert mock.threads  # reset + 2 steps
        assert threading.get_ident() not in mock.threads

    def test_fresh_env_per_episode(self, tmp_path):
        env = _make_env(_relay_config(tmp_path, total_steps=1))
        task = _first_task(env)
        _run(env, task, ["a"])
        _run(env, task, ["a"])
        assert len(MockRelayEnv.instances) >= 2
        assert MockRelayEnv.instances[-1] is not MockRelayEnv.instances[-2]

    def test_resets_to_task_index(self, tmp_path):
        env = _make_env(_relay_config(tmp_path, num_tasks=3, total_steps=1))
        task = list(env.taskset)[2]
        trace, mock, _ = _run(env, task, ["a"])
        assert trace.info["llenvs"]["task_index"] == 2
        assert mock.actions and trace.task.data.task_index == 2

    def test_rejects_foreign_task(self, tmp_path):
        env = _make_env(_relay_config(tmp_path, total_steps=1))
        with pytest.raises(TypeError, match="LLEnvsTask"):
            asyncio.run(env.run(vf.Task(vf.TaskData(prompt="x")), StubAgents(["a"])))

    def test_images_refused_at_reset(self, tmp_path):
        env = _make_env(_relay_config(tmp_path, total_steps=1))
        task = _first_task(env)
        # Swap the YAML to an image-emitting env for the episode itself.
        _relay_config(tmp_path, total_steps=1, images=True)
        _config.clear_caches()
        with pytest.raises(NotImplementedError, match="text-only"):
            _run(env, task, ["a"])
        assert MockRelayEnv.instances[-1].closed


# ---------------------------------------------------------------------------
# Reward recording
# ---------------------------------------------------------------------------


class TestRewardRecording:
    def test_weighted_signal_recorded_with_native_weight(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        progress = trace.rewards["llenvs/progress"]
        assert progress.score == pytest.approx(1.0)
        assert progress.weight == 1.0

    def test_weight_zero_signal_becomes_metric(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        assert "llenvs/format" not in trace.rewards
        assert trace.metrics["llenvs/format"] == pytest.approx(2.0)

    def test_reward_less_signal_is_skipped(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2, none_signal=True)
        assert "llenvs/hint" not in trace.rewards
        assert "llenvs/hint" not in trace.metrics

    def test_env_steps_metric(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        assert trace.metrics["llenvs/env_steps"] == 2

    def test_llenvs_total_hook_contributes_zero_on_relay(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        assert trace.rewards["llenvs_total"].score == 0.0

    @pytest.mark.parametrize("varying_weights", [False, True])
    def test_trace_reward_equals_sum_of_turn_rewards(self, tmp_path, varying_weights):
        trace, _, _ = _episode(
            tmp_path, ["a", "b", "c"], total_steps=3, varying_weights=varying_weights
        )
        turn_rewards = trace.info["llenvs"]["turn_rewards"]
        assert len(turn_rewards) == 3
        assert trace.reward == pytest.approx(sum(turn_rewards))

    def test_turn_rewards_are_bundle_totals(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2, varying_weights=True)
        # step 1: 0.5 * 1.0; step 2: 0.5 * 2.0 (format has weight 0)
        assert trace.info["llenvs"]["turn_rewards"] == pytest.approx([0.5, 1.0])

    def test_varying_weights_fall_back_to_unit_weight(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2, varying_weights=True)
        progress = trace.rewards["llenvs/progress"]
        assert progress.weight == 1.0
        assert progress.score == pytest.approx(1.5)
        assert "progress" in caplog.text

    def test_info_payload(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        payload = trace.info["llenvs"]
        assert set(payload) == {
            "turn_rewards",
            "turn_signals",
            "signal_weights",
            "env_steps",
            "stop",
            "task_index",
            "env_name",
        }
        assert payload["turn_signals"] == [
            {"progress": 0.5, "format": 1.0},
            {"progress": 0.5, "format": 1.0},
        ]
        assert payload["signal_weights"] == {"progress": 1.0, "format": 0.0}
        assert payload["env_steps"] == 2
        assert payload["stop"] == "env_terminated"
        assert payload["env_name"] == "relay"

    def test_info_survives_wire_round_trip(self, tmp_path):
        trace, _, _ = _episode(tmp_path, ["a", "b"], total_steps=2)
        record = trace.to_record()
        assert record["info"]["llenvs"]["turn_rewards"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Tools through the loop
# ---------------------------------------------------------------------------


class TestTools:
    def test_tool_call_reaches_env_and_result_feeds_back(self, tmp_path):
        trace, mock, agents = _episode(
            tmp_path, [TOOL_CALL_REPLY, "done"], total_steps=2, tools=True
        )
        first = mock.actions[0]
        assert first.text == "Let me look."
        assert [tc.name for tc in first.tool_calls] == ["lookup"]
        assert first.tool_calls[0].arguments == {"q": "x"}
        feedback = agents.agent.interactions[0].turns[1]
        assert isinstance(feedback, str)
        assert "<tool_response>" in feedback
        assert '"lookup"' in feedback
        assert "42" in feedback
        assert trace.stop_condition == "env_terminated"

    def test_plain_text_with_tools_is_a_text_action(self, tmp_path):
        _, mock, _ = _episode(tmp_path, ["just text", "x"], total_steps=2, tools=True)
        assert mock.actions[0] == Action.from_text("just text")

    def test_tool_markup_without_tools_stays_text(self, tmp_path):
        _, mock, _ = _episode(tmp_path, [TOOL_CALL_REPLY], total_steps=1, tools=False)
        assert mock.actions[0] == Action.from_text(TOOL_CALL_REPLY)
