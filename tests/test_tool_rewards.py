"""Tests for tool-specific reward functions."""

import pytest
from typing import Any

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.tools import ToolCall, ToolResult, ToolResultStatus
from llenvs.core.tool_rewards import ToolValidityReward, ToolEfficiencyReward
from llenvs.core.reward import RewardType


@pytest.fixture
def sample_state() -> State[Any]:
    """Create a sample state for tests."""
    obs = Observation(
        prompt="Test prompt",
        messages=(),
        tool_results=(),
        available_tools=(),
    )
    metadata = StateMetadata(step=0, episode_id="test", is_terminal=False)
    return State(observation=obs, hidden=None, metadata=metadata)


def make_next_state(
    state: State, tool_results: tuple[ToolResult, ...]
) -> State[Any]:
    """Create a next state with tool results."""
    next_obs = Observation(
        prompt=state.observation.prompt,
        messages=(),
        tool_results=tool_results,
        available_tools=(),
    )
    next_metadata = StateMetadata(
        step=state.metadata.step + 1,
        episode_id=state.metadata.episode_id,
        is_terminal=False,
    )
    return State(observation=next_obs, hidden=None, metadata=next_metadata)


class TestToolValidityReward:
    """Tests for ToolValidityReward."""

    def test_properties(self):
        """Test reward function properties."""
        reward = ToolValidityReward()
        assert reward.name == "tool_validity"
        assert reward.reward_type == RewardType.STEP

    def test_no_tool_calls(self, sample_state):
        """Test reward when no tool calls are made."""
        reward = ToolValidityReward()
        action = Action(text="Just some text")
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == 1.0
        assert signal.metadata["num_calls"] == 0

    def test_all_valid_calls(self, sample_state):
        """Test reward when all tool calls are valid."""
        reward = ToolValidityReward()
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="multiply", arguments={"a": 3, "b": 4}),
            )
        )
        tool_results = (
            ToolResult.success("1", "add", "3"),
            ToolResult.success("2", "multiply", "12"),
        )
        next_state = make_next_state(sample_state, tool_results)

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == 1.0
        assert signal.metadata["num_calls"] == 2
        assert signal.metadata["num_valid"] == 2

    def test_partial_valid_calls(self, sample_state):
        """Test partial credit when some calls fail."""
        reward = ToolValidityReward()
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="unknown", arguments={}),
                ToolCall(id="3", name="multiply", arguments={"a": 3, "b": 4}),
            )
        )
        tool_results = (
            ToolResult.success("1", "add", "3"),
            ToolResult.from_error("2", "unknown", "Unknown tool", ToolResultStatus.INVALID_TOOL),
            ToolResult.success("3", "multiply", "12"),
        )
        next_state = make_next_state(sample_state, tool_results)

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == pytest.approx(2 / 3)
        assert signal.metadata["num_calls"] == 3
        assert signal.metadata["num_valid"] == 2

    def test_all_invalid_calls(self, sample_state):
        """Test zero reward when all calls fail."""
        reward = ToolValidityReward()
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="unknown1", arguments={}),
                ToolCall(id="2", name="unknown2", arguments={}),
            )
        )
        tool_results = (
            ToolResult.from_error("1", "unknown1", "Unknown", ToolResultStatus.INVALID_TOOL),
            ToolResult.from_error("2", "unknown2", "Unknown", ToolResultStatus.INVALID_TOOL),
        )
        next_state = make_next_state(sample_state, tool_results)

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == 0.0

    def test_custom_name(self):
        """Test custom reward name."""
        reward = ToolValidityReward(_name="my_validity")
        assert reward.name == "my_validity"


class TestToolEfficiencyReward:
    """Tests for ToolEfficiencyReward."""

    def test_properties(self):
        """Test reward function properties."""
        reward = ToolEfficiencyReward()
        assert reward.name == "tool_efficiency"
        assert reward.reward_type == RewardType.STEP

    def test_no_tool_calls(self, sample_state):
        """Test perfect efficiency when no tool calls."""
        reward = ToolEfficiencyReward()
        action = Action(text="No tools needed")
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == 1.0
        assert signal.metadata["num_calls"] == 0

    def test_under_max_calls(self, sample_state):
        """Test full reward when under max calls."""
        reward = ToolEfficiencyReward(max_calls_per_step=5)
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="multiply", arguments={"a": 3, "b": 4}),
            )
        )
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == 1.0
        assert signal.metadata["excess"] == 0

    def test_excess_calls_penalty(self, sample_state):
        """Test penalty for excess calls."""
        reward = ToolEfficiencyReward(max_calls_per_step=2, penalty_per_excess=0.2)
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="a", arguments={}),
                ToolCall(id="2", name="b", arguments={}),
                ToolCall(id="3", name="c", arguments={}),
                ToolCall(id="4", name="d", arguments={}),
            )
        )
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        # 4 calls - 2 max = 2 excess, penalty = 2 * 0.2 = 0.4
        assert signal.value == pytest.approx(0.6)
        assert signal.metadata["excess"] == 2

    def test_duplicate_calls_penalty(self, sample_state):
        """Test penalty for duplicate calls."""
        reward = ToolEfficiencyReward(duplicate_penalty=0.2)
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="add", arguments={"a": 1, "b": 2}),  # Duplicate
                ToolCall(id="3", name="add", arguments={"a": 1, "b": 2}),  # Duplicate
            )
        )
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        # 2 duplicates, penalty = 2 * 0.2 = 0.4
        assert signal.value == pytest.approx(0.6)
        assert signal.metadata["duplicates"] == 2

    def test_different_args_not_duplicates(self, sample_state):
        """Test that same tool with different args is not duplicate."""
        reward = ToolEfficiencyReward(duplicate_penalty=0.5)
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="add", arguments={"a": 3, "b": 4}),  # Different args
            )
        )
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        assert signal.value == 1.0
        assert signal.metadata["duplicates"] == 0

    def test_combined_penalties(self, sample_state):
        """Test combined excess and duplicate penalties."""
        reward = ToolEfficiencyReward(
            max_calls_per_step=2,
            penalty_per_excess=0.1,
            duplicate_penalty=0.2,
        )
        action = Action(
            tool_calls=(
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="add", arguments={"a": 1, "b": 2}),  # Duplicate
                ToolCall(id="3", name="multiply", arguments={"a": 3, "b": 4}),
                ToolCall(id="4", name="subtract", arguments={"a": 5, "b": 1}),  # Excess
            )
        )
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        # 2 excess (4 - 2) * 0.1 = 0.2 penalty
        # 1 duplicate * 0.2 = 0.2 penalty
        # Total: 1.0 - 0.2 - 0.2 = 0.6
        assert signal.value == pytest.approx(0.6)

    def test_reward_clamped_to_zero(self, sample_state):
        """Test that reward doesn't go below zero."""
        reward = ToolEfficiencyReward(
            max_calls_per_step=1,
            penalty_per_excess=0.5,
        )
        action = Action(
            tool_calls=tuple(
                ToolCall(id=str(i), name="tool", arguments={"i": i})
                for i in range(10)
            )
        )
        next_state = make_next_state(sample_state, ())

        signal = reward.compute(sample_state, action, next_state)

        # 9 excess * 0.5 = 4.5 penalty, clamped to 0
        assert signal.value == 0.0

    def test_custom_parameters(self):
        """Test with custom parameters."""
        reward = ToolEfficiencyReward(
            _name="my_efficiency",
            max_calls_per_step=3,
            penalty_per_excess=0.25,
            duplicate_penalty=0.15,
        )
        assert reward.name == "my_efficiency"
        assert reward.max_calls_per_step == 3
        assert reward.penalty_per_excess == 0.25
        assert reward.duplicate_penalty == 0.15
