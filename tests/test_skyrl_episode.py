"""Owned environment sessions; no policy engine or SkyRL runtime involved."""

import copy
import importlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from llenvs.core.config import EnvironmentFactory, EvalConfig
from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.core.tools import ToolDefinition, ToolResult
from llenvs.integrations.dataset_provider import TaskItem
from llenvs.integrations.skyrl._config import SelectedEnvironment
from llenvs.integrations.skyrl.data import (
    _export_config,
    _fingerprint,
    _initial_messages,
    _task_id,
)


@pytest.fixture
def episode_module():
    return importlib.import_module("llenvs.integrations.skyrl._episode")


@pytest.fixture
def setup(monkeypatch, tmp_path):
    config = EvalConfig.from_dict(
        {"environments": [{"name": "fixture", "adapter": "test", "seed": 41}]}
    )
    env_config, system_prompt, fingerprint = _export_config(config, None)
    selected = SelectedEnvironment(
        tmp_path / "env.yaml", env_config, system_prompt, fingerprint, ()
    )
    state = State(
        observation=Observation(prompt="Task"),
        hidden=SimpleNamespace(expected_answer="hidden"),
        metadata=StateMetadata(step=0, episode_id="external-session"),
    )
    created = []

    def create(config):
        assert config.seed == 41
        env = Mock(spec=["spec", "__len__", "reset", "step", "close"])
        env.spec = EnvironmentSpec("fixture", "test")
        env.__len__ = Mock(return_value=3)
        env.reset.return_value = (state, {"secret": "not-forwarded"})
        env.step.return_value = StepResult(state, SignalBundle.single(0.25))
        created.append(env)
        return env

    factory = Mock(side_effect=create)
    monkeypatch.setattr(EnvironmentFactory, "create", factory)

    def row():
        item = TaskItem(
            1,
            state.observation.prompt,
            state.observation.messages,
            None,
            {},
            images=state.observation.get_images(),
            available_tools=state.observation.available_tools,
        )
        messages = _initial_messages(item, system_prompt)
        return {
            "prompt": messages,
            "env_class": "llenvs",
            "data_source": "test/fixture",
            "llenvs": {
                "schema_version": 1,
                "env_fingerprint": fingerprint,
                "task_index": 1,
                "task_id": _task_id(fingerprint, 1),
                "initial_fingerprint": _fingerprint(messages),
            },
        }

    return SimpleNamespace(
        selected=selected, state=state, created=created, factory=factory, row=row
    )


def test_each_repetition_owns_one_reset_and_close(episode_module, setup):
    prepared = setup.row()
    original = copy.deepcopy(prepared)
    episodes = [episode_module.Episode(setup.selected, prepared) for _ in range(2)]
    assert not setup.created
    for episode in episodes:
        assert episode.open() == prepared["prompt"]
        episode.step("raw answer")
        episode.close()
        episode.close()
    assert len(setup.created) == 2
    for env in setup.created:
        env.reset.assert_called_once_with(options={"task_index": 1})
        env.step.assert_called_once_with(setup.state, Action.from_text("raw answer"))
        env.close.assert_called_once_with()
    assert prepared == original


def test_reset_identity_failure_is_not_repaired(episode_module, setup):
    row = setup.row()
    row["prompt"][0]["content"] = "different"
    row["llenvs"]["initial_fingerprint"] = _fingerprint(row["prompt"])
    episode = episode_module.Episode(setup.selected, row)
    with pytest.raises(ValueError, match="initial"):
        episode.open()
    episode.close()
    setup.created[0].close.assert_called_once()


def test_reset_failure_retains_owned_resource_for_cleanup(episode_module, setup):
    original_factory = setup.factory.side_effect

    def create(config):
        env = original_factory(config)
        env.reset.side_effect = RuntimeError("reset failed")
        return env

    setup.factory.side_effect = create
    episode = episode_module.Episode(setup.selected, setup.row())
    with pytest.raises(RuntimeError, match="reset failed"):
        episode.open()
    episode.close()
    setup.created[0].close.assert_called_once()


def test_tools_execute_one_decision_and_raw_reply_is_not_replaced(episode_module, setup):
    tool = ToolDefinition(name="lookup", description="Lookup", parameters={})
    setup.state = replace(
        setup.state, observation=replace(setup.state.observation, available_tools=(tool,))
    )
    # The factory intentionally closes over its initial state; provide this task's fresh reset.
    original_factory = setup.factory.side_effect

    def create(config):
        env = original_factory(config)
        env.reset.return_value = (setup.state, {})
        env.step.return_value = StepResult(setup.state, SignalBundle.single(1), terminated=True)
        return env

    setup.factory.side_effect = create
    item = TaskItem(1, "Task", (), None, {}, available_tools=(tool,))
    row = setup.row()
    row["prompt"] = _initial_messages(item, None)
    row["llenvs"]["initial_fingerprint"] = _fingerprint(row["prompt"])
    episode = episode_module.Episode(setup.selected, row)
    episode.open()
    raw = 'before<tool_call>{"name":"lookup","arguments":{}}</tool_call><tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    result = episode.step(raw)
    action = setup.created[0].step.call_args.args[1]
    assert len(action.tool_calls) == 2 and action.text == "before"
    assert result.rewards.total == 1 and result.done
    assert raw.startswith("before<tool_call>")
    with pytest.raises(RuntimeError, match="ended"):
        episode.step("another")
    episode.close()


def test_dynamic_tool_schema_fails_before_next_policy_call(episode_module, setup):
    episode = episode_module.Episode(setup.selected, setup.row())
    episode.open()
    changed = replace(
        setup.state,
        observation=replace(
            setup.state.observation, available_tools=(ToolDefinition("new", "", {}),)
        ),
    )
    setup.created[0].step.return_value = StepResult(changed, SignalBundle.empty())
    with pytest.raises(ValueError, match="tool schema"):
        episode.step("answer")
    episode.close()


def test_feedback_uses_tool_results_then_dynamic_state(episode_module, setup):
    state = replace(
        setup.state,
        observation=Observation(
            prompt="task",
            state=ObservationContent(text="dynamic"),
            tool_results=(
                ToolResult.success("a", "lookup", {"x": 1}),
                ToolResult.from_error("b", "lookup", "failed"),
            ),
        ),
    )
    message = episode_module.feedback_message(state)
    assert message["role"] == "user"
    assert message["content"].count("<tool_response>") == 2
    assert '"x": 1' in message["content"] and "failed" in message["content"]
    state = replace(state, observation=replace(state.observation, tool_results=()))
    assert episode_module.feedback_message(state) == {"role": "user", "content": "dynamic"}


def test_reward_validation_preserves_weights_and_native_error_signals(episode_module):
    bundle = SignalBundle(
        (
            Signal("native", RewardType.STEP, 2, weight=0.25, metadata={"error": "invalid action"}),
            Signal("feedback", RewardType.PROCESS, feedback="try again"),
            Signal("native", RewardType.OUTCOME, -1, weight=2),
        )
    )
    assert episode_module.reward_total(bundle) == bundle.total == -1.5


@pytest.mark.parametrize(
    "reward,weight", [(float("nan"), 0), (1, float("inf")), (1e308, 2), (True, 1)]
)
def test_nonfinite_or_nonnumeric_components_fail_even_if_weight_zero(
    episode_module, reward, weight
):
    with pytest.raises(ValueError):
        episode_module.reward_total(
            SignalBundle((Signal("x", RewardType.STEP, reward, weight=weight),))
        )


def test_cleanup_failure_is_visible_and_not_retried(episode_module, setup):
    episode = episode_module.Episode(setup.selected, setup.row())
    episode.open()
    setup.created[0].close.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        episode.close()
    episode.close()
    assert setup.created[0].close.call_args_list == [call()]
