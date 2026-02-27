"""Tests for container serialization layer."""

from dataclasses import dataclass

import pytest

from llenvs.container.serialization import (
    OpaqueHidden,
    deserialize_action,
    deserialize_env_spec,
    deserialize_observation,
    deserialize_reward_bundle,
    deserialize_reward_signal,
    deserialize_state,
    deserialize_state_metadata,
    deserialize_state_typed,
    deserialize_step_result,
    deserialize_tool_call,
    deserialize_tool_definition,
    deserialize_tool_parameter,
    deserialize_tool_result,
    reconstruct_hidden,
    serialize_action,
    serialize_env_spec,
    serialize_observation,
    serialize_reward_bundle,
    serialize_reward_signal,
    serialize_state,
    serialize_state_metadata,
    serialize_step_result,
    serialize_tool_call,
    serialize_tool_definition,
    serialize_tool_parameter,
    serialize_tool_result,
)
from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
)

# ---------------------------------------------------------------------------
# OpaqueHidden
# ---------------------------------------------------------------------------


class TestOpaqueHidden:
    def test_attribute_access(self):
        h = OpaqueHidden({"expected_answer": "42", "category": "math"})
        assert h.expected_answer == "42"
        assert h.category == "math"

    def test_immutability(self):
        h = OpaqueHidden({"x": 1})
        with pytest.raises(AttributeError, match="immutable"):
            h.x = 2

    def test_immutability_new_attr(self):
        h = OpaqueHidden({"x": 1})
        with pytest.raises(AttributeError, match="immutable"):
            h.new_field = 99

    def test_to_dict(self):
        data = {"a": 1, "b": [2, 3]}
        h = OpaqueHidden(data)
        assert h.to_dict() == data

    def test_to_dict_is_copy(self):
        data = {"a": 1}
        h = OpaqueHidden(data)
        d = h.to_dict()
        d["a"] = 999
        assert h.a == 1

    def test_equality(self):
        h1 = OpaqueHidden({"x": 1})
        h2 = OpaqueHidden({"x": 1})
        assert h1 == h2

    def test_inequality(self):
        h1 = OpaqueHidden({"x": 1})
        h2 = OpaqueHidden({"x": 2})
        assert h1 != h2

    def test_equality_non_opaque(self):
        h = OpaqueHidden({"x": 1})
        assert h != {"x": 1}

    def test_repr(self):
        h = OpaqueHidden({"x": 1})
        r = repr(h)
        assert "OpaqueHidden" in r
        assert "x=1" in r

    def test_empty(self):
        h = OpaqueHidden({})
        assert h.to_dict() == {}

    def test_nested_data(self):
        h = OpaqueHidden({"nested": {"a": 1}, "items": [1, 2, 3]})
        assert h.nested == {"a": 1}
        assert h.items == [1, 2, 3]


# ---------------------------------------------------------------------------
# ToolCall round-trip
# ---------------------------------------------------------------------------


class TestToolCallSerialization:
    def test_round_trip(self):
        tc = ToolCall(id="tc1", name="search", arguments={"query": "hello"})
        data = serialize_tool_call(tc)
        result = deserialize_tool_call(data)
        assert result == tc

    def test_empty_arguments(self):
        tc = ToolCall(id="tc2", name="noop")
        data = serialize_tool_call(tc)
        result = deserialize_tool_call(data)
        assert result.arguments == {}

    def test_json_compatible(self):
        import json

        tc = ToolCall(id="tc3", name="run", arguments={"x": [1, 2]})
        data = serialize_tool_call(tc)
        json_str = json.dumps(data)
        restored = deserialize_tool_call(json.loads(json_str))
        assert restored.id == tc.id
        assert restored.name == tc.name


# ---------------------------------------------------------------------------
# ToolResult round-trip
# ---------------------------------------------------------------------------


class TestToolResultSerialization:
    def test_success(self):
        tr = ToolResult.success("c1", "search", "result text")
        data = serialize_tool_result(tr)
        result = deserialize_tool_result(data)
        assert result == tr

    def test_error(self):
        tr = ToolResult.from_error("c2", "bad", "not found", ToolResultStatus.INVALID_TOOL)
        data = serialize_tool_result(tr)
        result = deserialize_tool_result(data)
        assert result == tr

    def test_all_statuses(self):
        for status in ToolResultStatus:
            tr = ToolResult(call_id="c", tool_name="t", status=status, output="out")
            data = serialize_tool_result(tr)
            result = deserialize_tool_result(data)
            assert result.status == status

    def test_dict_output(self):
        tr = ToolResult.success("c1", "search", {"key": "value"})
        data = serialize_tool_result(tr)
        result = deserialize_tool_result(data)
        assert result.output == {"key": "value"}


# ---------------------------------------------------------------------------
# ToolParameter round-trip
# ---------------------------------------------------------------------------


class TestToolParameterSerialization:
    def test_round_trip(self):
        tp = ToolParameter(
            name="query",
            type=ToolParameterType.STRING,
            description="Search query",
            required=True,
        )
        data = serialize_tool_parameter(tp)
        result = deserialize_tool_parameter(data)
        assert result == tp

    def test_with_enum(self):
        tp = ToolParameter(
            name="color",
            type=ToolParameterType.STRING,
            description="Color",
            enum=("red", "blue", "green"),
        )
        data = serialize_tool_parameter(tp)
        result = deserialize_tool_parameter(data)
        assert result == tp

    def test_all_types(self):
        for param_type in ToolParameterType:
            tp = ToolParameter(name="x", type=param_type, description="d")
            data = serialize_tool_parameter(tp)
            result = deserialize_tool_parameter(data)
            assert result.type == param_type


# ---------------------------------------------------------------------------
# ToolDefinition round-trip
# ---------------------------------------------------------------------------


class TestToolDefinitionSerialization:
    def test_round_trip(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=(
                ToolParameter(name="q", type=ToolParameterType.STRING, description="query"),
            ),
            is_terminal=False,
        )
        data = serialize_tool_definition(td)
        result = deserialize_tool_definition(data)
        assert result == td

    def test_terminal(self):
        td = ToolDefinition(name="submit", description="Submit answer", is_terminal=True)
        data = serialize_tool_definition(td)
        result = deserialize_tool_definition(data)
        assert result.is_terminal is True

    def test_no_parameters(self):
        td = ToolDefinition(name="noop", description="No-op")
        data = serialize_tool_definition(td)
        result = deserialize_tool_definition(data)
        assert result.parameters == ()


# ---------------------------------------------------------------------------
# Signal round-trip
# ---------------------------------------------------------------------------


class TestSignalSerialization:
    def test_round_trip(self):
        sig = Signal(name="correct", reward_type=RewardType.OUTCOME, reward=1.0)
        data = serialize_reward_signal(sig)
        result = deserialize_reward_signal(data)
        assert result == sig

    def test_with_metadata(self):
        sig = Signal(
            name="format",
            reward_type=RewardType.FORMAT,
            reward=0.5,
            metadata={"extraction": {"found": True}},
        )
        data = serialize_reward_signal(sig)
        result = deserialize_reward_signal(data)
        assert result == sig

    def test_with_weight(self):
        sig = Signal(name="rubric", reward_type=RewardType.PROCESS, reward=0.8, weight=2.0)
        data = serialize_reward_signal(sig)
        result = deserialize_reward_signal(data)
        assert result.weight == 2.0

    def test_all_types(self):
        for rtype in RewardType:
            sig = Signal(name="x", reward_type=rtype, reward=0.0)
            data = serialize_reward_signal(sig)
            result = deserialize_reward_signal(data)
            assert result.reward_type == rtype


# ---------------------------------------------------------------------------
# SignalBundle round-trip
# ---------------------------------------------------------------------------


class TestSignalBundleSerialization:
    def test_round_trip(self):
        bundle = SignalBundle(
            signals=(
                Signal(name="correct", reward_type=RewardType.OUTCOME, reward=1.0),
                Signal(name="format", reward_type=RewardType.FORMAT, reward=0.5),
            )
        )
        data = serialize_reward_bundle(bundle)
        result = deserialize_reward_bundle(data)
        assert result == bundle

    def test_empty(self):
        bundle = SignalBundle.empty()
        data = serialize_reward_bundle(bundle)
        result = deserialize_reward_bundle(data)
        assert result.signals == ()

    def test_total_preserved(self):
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0, weight=2.0),
                Signal(name="b", reward_type=RewardType.FORMAT, reward=0.5),
            )
        )
        data = serialize_reward_bundle(bundle)
        result = deserialize_reward_bundle(data)
        assert result.total == bundle.total


# ---------------------------------------------------------------------------
# Observation round-trip
# ---------------------------------------------------------------------------


class TestObservationSerialization:
    def test_simple(self):
        obs = Observation(prompt="What is 2+2?")
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result == obs

    def test_with_messages(self):
        obs = Observation(
            prompt="Hello",
            messages=({"role": "user", "content": "hi"},),
        )
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.messages == ({"role": "user", "content": "hi"},)

    def test_with_tool_results(self):
        tr = ToolResult.success("c1", "search", "found it")
        obs = Observation(prompt="p", tool_results=(tr,))
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.tool_results == (tr,)

    def test_with_available_tools(self):
        td = ToolDefinition(name="search", description="Search")
        obs = Observation(prompt="p", available_tools=(td,))
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.available_tools == (td,)

    def test_with_task_and_state(self):
        obs = Observation(
            prompt="Solve it",
            task=ObservationContent(text="Task description"),
            state=ObservationContent(text="Current state"),
        )
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.task is not None
        assert result.task.text == "Task description"
        assert result.state is not None
        assert result.state.text == "Current state"

    def test_task_none_state_none(self):
        obs = Observation(prompt="p")
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.task is None
        assert result.state is None

    def test_task_with_images(self):
        img = ImageContent(data="abc123", media_type="image/jpeg")
        obs = Observation(
            prompt="p",
            task=ObservationContent(text="task", images=(img,)),
        )
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.task is not None
        assert len(result.task.images) == 1
        assert result.task.images[0].data == "abc123"
        assert result.task.images[0].media_type == "image/jpeg"

    def test_state_with_data(self):
        obs = Observation(
            prompt="p",
            state=ObservationContent(text="state", data={"score": 42, "items": ["a", "b"]}),
        )
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.state is not None
        assert result.state.data == {"score": 42, "items": ["a", "b"]}

    def test_task_and_state_full_round_trip(self):
        import json

        img = ImageContent(data="base64data", media_type="image/png")
        obs = Observation(
            prompt="prompt",
            task=ObservationContent(text="task text", images=(img,), data={"key": "val"}),
            state=ObservationContent(text="state text", data={"step": 3}),
        )
        data = serialize_observation(obs)
        json_str = json.dumps(data)
        restored = deserialize_observation(json.loads(json_str))
        assert restored.task.text == "task text"
        assert restored.task.images[0].data == "base64data"
        assert restored.task.data == {"key": "val"}
        assert restored.state.text == "state text"
        assert restored.state.data == {"step": 3}

    def test_observation_content_empty_text(self):
        obs = Observation(
            prompt="p",
            state=ObservationContent(data={"structured": True}),
        )
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.state.text == ""
        assert result.state.data == {"structured": True}


# ---------------------------------------------------------------------------
# StateMetadata round-trip
# ---------------------------------------------------------------------------


class TestStateMetadataSerialization:
    def test_round_trip(self):
        meta = StateMetadata(step=3, episode_id="ep-1", is_terminal=True, info={"x": 1})
        data = serialize_state_metadata(meta)
        result = deserialize_state_metadata(data)
        assert result == meta

    def test_defaults(self):
        meta = StateMetadata(step=0, episode_id="ep-0")
        data = serialize_state_metadata(meta)
        result = deserialize_state_metadata(data)
        assert result.is_terminal is False
        assert result.info == {}


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockHidden:
    expected_answer: str
    category: str = "general"


class TestStateSerialization:
    def test_client_side_round_trip(self):
        state = State(
            observation=Observation(prompt="Q?"),
            hidden=MockHidden(expected_answer="42", category="math"),
            metadata=StateMetadata(step=0, episode_id="ep-1"),
        )
        data = serialize_state(state)
        result = deserialize_state(data)
        assert result.observation == state.observation
        assert result.metadata == state.metadata
        assert isinstance(result.hidden, OpaqueHidden)
        assert result.hidden.expected_answer == "42"
        assert result.hidden.category == "math"

    def test_server_side_typed(self):
        state = State(
            observation=Observation(prompt="Q?"),
            hidden=MockHidden(expected_answer="7"),
            metadata=StateMetadata(step=1, episode_id="ep-2"),
        )
        data = serialize_state(state)
        result = deserialize_state_typed(data, MockHidden)
        assert isinstance(result.hidden, MockHidden)
        assert result.hidden.expected_answer == "7"

    def test_opaque_hidden_round_trip(self):
        """OpaqueHidden serializes and deserializes correctly."""
        hidden = OpaqueHidden({"x": 1, "y": "hello"})
        state = State(
            observation=Observation(prompt="p"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e"),
        )
        data = serialize_state(state)
        result = deserialize_state(data)
        assert result.hidden.x == 1
        assert result.hidden.y == "hello"

    def test_dict_hidden(self):
        state = State(
            observation=Observation(prompt="p"),
            hidden={"key": "value"},
            metadata=StateMetadata(step=0, episode_id="e"),
        )
        data = serialize_state(state)
        result = deserialize_state(data)
        assert result.hidden.key == "value"


# ---------------------------------------------------------------------------
# reconstruct_hidden
# ---------------------------------------------------------------------------


class TestReconstructHidden:
    def test_dataclass(self):
        result = reconstruct_hidden({"expected_answer": "42", "category": "math"}, MockHidden)
        assert isinstance(result, MockHidden)
        assert result.expected_answer == "42"

    def test_extra_keys_ignored(self):
        result = reconstruct_hidden(
            {"expected_answer": "42", "category": "math", "extra": "ignored"},
            MockHidden,
        )
        assert isinstance(result, MockHidden)
        assert result.expected_answer == "42"

    def test_non_dataclass_returns_dict(self):
        result = reconstruct_hidden({"a": 1}, dict)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# Action round-trip
# ---------------------------------------------------------------------------


class TestActionSerialization:
    def test_text_only(self):
        action = Action.from_text("hello")
        data = serialize_action(action)
        result = deserialize_action(data)
        assert result == action

    def test_tool_calls(self):
        tc = ToolCall(id="tc1", name="search", arguments={"q": "test"})
        action = Action.from_tool_call(tc)
        data = serialize_action(action)
        result = deserialize_action(data)
        assert result == action

    def test_both(self):
        tc = ToolCall(id="tc1", name="run", arguments={})
        action = Action(text="thinking...", tool_calls=(tc,))
        data = serialize_action(action)
        result = deserialize_action(data)
        assert result == action

    def test_none_text(self):
        action = Action(text=None, tool_calls=())
        data = serialize_action(action)
        result = deserialize_action(data)
        assert result.text is None


# ---------------------------------------------------------------------------
# StepResult round-trip
# ---------------------------------------------------------------------------


class TestStepResultSerialization:
    def test_round_trip(self):
        step_result = StepResult(
            next_state=State(
                observation=Observation(prompt="done"),
                hidden=MockHidden(expected_answer="42"),
                metadata=StateMetadata(step=1, episode_id="ep-1", is_terminal=True),
            ),
            rewards=SignalBundle.single(1.0, "correct", RewardType.OUTCOME),
            terminated=True,
            truncated=False,
            info={"turns": 1},
        )
        data = serialize_step_result(step_result)
        result = deserialize_step_result(data)
        assert result.terminated is True
        assert result.truncated is False
        assert result.info == {"turns": 1}
        assert isinstance(result.next_state.hidden, OpaqueHidden)
        assert result.next_state.hidden.expected_answer == "42"
        assert result.rewards.total == 1.0

    def test_empty_rewards(self):
        step_result = StepResult(
            next_state=State(
                observation=Observation(prompt="p"),
                hidden=MockHidden(expected_answer="?"),
                metadata=StateMetadata(step=0, episode_id="e"),
            ),
            rewards=SignalBundle.empty(),
        )
        data = serialize_step_result(step_result)
        result = deserialize_step_result(data)
        assert result.rewards.signals == ()


# ---------------------------------------------------------------------------
# EnvironmentSpec round-trip
# ---------------------------------------------------------------------------


class TestEnvironmentSpecSerialization:
    def test_round_trip(self):
        spec = EnvironmentSpec(
            name="sudoku",
            adapter="reasoning_gym",
            max_steps=10,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            metadata={"difficulty": "hard"},
        )
        data = serialize_env_spec(spec)
        result = deserialize_env_spec(data)
        assert result.name == spec.name
        assert result.adapter == spec.adapter
        assert result.max_steps == spec.max_steps
        assert result.is_multi_turn == spec.is_multi_turn
        assert result.supports_seed is False
        assert result.metadata == {"difficulty": "hard"}

    def test_defaults(self):
        spec = EnvironmentSpec(name="test")
        data = serialize_env_spec(spec)
        result = deserialize_env_spec(data)
        assert result.adapter == ""
        assert result.max_steps is None
        assert result.is_multi_turn is False
        assert result.supports_task_index is True

    def test_type_fields_not_serialized(self):
        """observation_type and action_type are type objects, not serialized."""
        spec = EnvironmentSpec(name="test", observation_type=str, action_type=int)
        data = serialize_env_spec(spec)
        assert "observation_type" not in data
        assert "action_type" not in data


# ---------------------------------------------------------------------------
# JSON round-trip (full pipeline through json.dumps/loads)
# ---------------------------------------------------------------------------


class TestFullJsonRoundTrip:
    def test_state_through_json(self):
        import json

        state = State(
            observation=Observation(
                prompt="Solve: 2+2",
                messages=({"role": "user", "content": "hi"},),
                tool_results=(ToolResult.success("c1", "calc", "4"),),
                available_tools=(
                    ToolDefinition(
                        name="calc",
                        description="Calculator",
                        parameters=(
                            ToolParameter(
                                name="expr",
                                type=ToolParameterType.STRING,
                                description="Expression",
                            ),
                        ),
                    ),
                ),
            ),
            hidden=MockHidden(expected_answer="4", category="math"),
            metadata=StateMetadata(step=0, episode_id="ep-1", info={"source": "test"}),
        )
        data = serialize_state(state)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        result = deserialize_state(restored_data)
        assert result.observation.prompt == "Solve: 2+2"
        assert result.hidden.expected_answer == "4"
        assert len(result.observation.tool_results) == 1
        assert len(result.observation.available_tools) == 1

    def test_step_result_through_json(self):
        import json

        sr = StepResult(
            next_state=State(
                observation=Observation(prompt="done"),
                hidden=MockHidden(expected_answer="42"),
                metadata=StateMetadata(step=1, episode_id="ep-1"),
            ),
            rewards=SignalBundle(
                signals=(
                    Signal(name="correct", reward_type=RewardType.OUTCOME, reward=1.0),
                    Signal(
                        name="format",
                        reward_type=RewardType.FORMAT,
                        reward=0.5,
                        weight=0.5,
                    ),
                )
            ),
            terminated=True,
            info={"detail": [1, 2, 3]},
        )
        data = serialize_step_result(sr)
        json_str = json.dumps(data)
        restored = deserialize_step_result(json.loads(json_str))
        assert restored.terminated is True
        assert len(restored.rewards.signals) == 2
        assert restored.rewards.signals[1].weight == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_metadata_in_reward(self):
        sig = Signal(name="x", reward_type=RewardType.STEP, reward=0.0, metadata=None)
        data = serialize_reward_signal(sig)
        result = deserialize_reward_signal(data)
        assert result.metadata is None

    def test_none_error_in_tool_result(self):
        tr = ToolResult(call_id="c", tool_name="t", status=ToolResultStatus.SUCCESS, output="ok")
        data = serialize_tool_result(tr)
        result = deserialize_tool_result(data)
        assert result.error is None

    def test_empty_tool_calls_in_action(self):
        action = Action(text="hello", tool_calls=())
        data = serialize_action(action)
        result = deserialize_action(data)
        assert result.tool_calls == ()

    def test_empty_observation_fields(self):
        obs = Observation(prompt="p")
        data = serialize_observation(obs)
        result = deserialize_observation(data)
        assert result.messages == ()
        assert result.tool_results == ()
        assert result.available_tools == ()
