"""Tests for tool environment protocol and base implementation."""

import pytest
from typing import Any
from dataclasses import dataclass, field

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.environment import StepResult, EnvironmentSpec
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType, RewardFunction
from llenvs.core.tools import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    SimpleToolExecutor,
)
from llenvs.core.tool_environment import BaseToolEnvironment


@dataclass
class MockHidden:
    """Mock hidden state for tests."""

    expected_answer: str = "42"
    calls_made: list[ToolCall] = field(default_factory=list)


class CorrectnessReward:
    """Simple correctness reward for testing."""

    @property
    def name(self) -> str:
        return "correctness"

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[Any],
        action: Any,
        next_state: State[Any],
    ) -> RewardSignal:
        return RewardSignal(value=1.0, name=self.name, reward_type=self.reward_type)


@dataclass
class CalculatorEnvironment(BaseToolEnvironment[MockHidden]):
    """Mock calculator environment for testing."""

    def __post_init__(self):
        """Set up tools and executor."""
        self._tools = (
            ToolDefinition(
                name="add",
                description="Add two numbers",
                parameters=(
                    ToolParameter("a", ToolParameterType.NUMBER, "First number"),
                    ToolParameter("b", ToolParameterType.NUMBER, "Second number"),
                ),
            ),
            ToolDefinition(
                name="multiply",
                description="Multiply two numbers",
                parameters=(
                    ToolParameter("a", ToolParameterType.NUMBER, "First number"),
                    ToolParameter("b", ToolParameterType.NUMBER, "Second number"),
                ),
            ),
            ToolDefinition(
                name="submit_answer",
                description="Submit the final answer",
                parameters=(
                    ToolParameter("answer", ToolParameterType.STRING, "The answer"),
                ),
                is_terminal=True,
            ),
        )

        def add(a: float, b: float) -> str:
            return str(a + b)

        def multiply(a: float, b: float) -> str:
            return str(a * b)

        def submit_answer(answer: str) -> str:
            return f"Submitted: {answer}"

        self._executor = SimpleToolExecutor({
            "add": add,
            "multiply": multiply,
            "submit_answer": submit_answer,
        })

        self._reward_functions = (CorrectnessReward(),)

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="calculator",
            max_steps=10,
            is_multi_turn=True,
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction[MockHidden], ...]:
        return self._reward_functions

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[MockHidden], dict[str, Any]]:
        obs = Observation(
            prompt="Calculate (5 + 3) * 7",
            messages=(),
            tool_results=(),
            available_tools=self._tools,
        )
        hidden = MockHidden(expected_answer="56")
        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", "test") if options else "test",
            is_terminal=False,
        )
        return State(observation=obs, hidden=hidden, metadata=metadata), {}

    def step(
        self,
        state: State[MockHidden],
        action: Action,
    ) -> StepResult[MockHidden]:
        # Execute any tool calls
        tool_results = ()
        if action.has_tool_calls:
            tool_results = self.execute_tools(action.tool_calls)

        # Check for terminal tools
        terminated = self._check_terminal_tools(action.tool_calls)

        # Build next observation
        next_obs = self._build_next_observation(
            state.observation, action, tool_results
        )

        # Update hidden state
        new_hidden = MockHidden(
            expected_answer=state.hidden.expected_answer,
            calls_made=state.hidden.calls_made + list(action.tool_calls),
        )

        next_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated,
        )

        next_state = State(
            observation=next_obs,
            hidden=new_hidden,
            metadata=next_metadata,
        )

        rewards = RewardBundle.single(1.0 if terminated else 0.0, "correctness")

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=False,
        )


class TestBaseToolEnvironment:
    """Tests for BaseToolEnvironment."""

    @pytest.fixture
    def env(self) -> CalculatorEnvironment:
        return CalculatorEnvironment()

    def test_available_tools(self, env: CalculatorEnvironment):
        """Test that available tools are correctly returned."""
        tools = env.available_tools
        assert len(tools) == 3
        assert tools[0].name == "add"
        assert tools[1].name == "multiply"
        assert tools[2].name == "submit_answer"

    def test_reset(self, env: CalculatorEnvironment):
        """Test environment reset."""
        state, info = env.reset()

        assert state.observation.prompt == "Calculate (5 + 3) * 7"
        assert state.observation.available_tools == env.available_tools
        assert len(state.observation.tool_results) == 0
        assert state.hidden.expected_answer == "56"
        assert state.metadata.step == 0
        assert not state.metadata.is_terminal

    def test_step_with_tool_call(self, env: CalculatorEnvironment):
        """Test step with a tool call."""
        state, _ = env.reset()

        action = Action(
            text="Let me add 5 and 3",
            tool_calls=(
                ToolCall(id="call_1", name="add", arguments={"a": 5, "b": 3}),
            ),
        )

        result = env.step(state, action)

        assert len(result.next_state.observation.tool_results) == 1
        assert result.next_state.observation.tool_results[0].is_success
        assert result.next_state.observation.tool_results[0].output == "8"
        assert not result.terminated

    def test_step_with_multiple_tool_calls(self, env: CalculatorEnvironment):
        """Test step with multiple tool calls."""
        state, _ = env.reset()

        action = Action(
            tool_calls=(
                ToolCall(id="call_1", name="add", arguments={"a": 5, "b": 3}),
                ToolCall(id="call_2", name="multiply", arguments={"a": 8, "b": 7}),
            ),
        )

        result = env.step(state, action)

        assert len(result.next_state.observation.tool_results) == 2
        assert result.next_state.observation.tool_results[0].output == "8"
        assert result.next_state.observation.tool_results[1].output == "56"

    def test_step_with_terminal_tool(self, env: CalculatorEnvironment):
        """Test step with terminal tool ends episode."""
        state, _ = env.reset()

        action = Action(
            tool_calls=(
                ToolCall(id="call_1", name="submit_answer", arguments={"answer": "56"}),
            ),
        )

        result = env.step(state, action)

        assert result.terminated
        assert result.done
        assert result.next_state.metadata.is_terminal

    def test_step_with_invalid_tool(self, env: CalculatorEnvironment):
        """Test step with invalid tool name."""
        state, _ = env.reset()

        action = Action(
            tool_calls=(
                ToolCall(id="call_1", name="divide", arguments={"a": 10, "b": 2}),
            ),
        )

        result = env.step(state, action)

        assert len(result.next_state.observation.tool_results) == 1
        tool_result = result.next_state.observation.tool_results[0]
        assert tool_result.status == ToolResultStatus.INVALID_TOOL
        assert "Unknown tool" in tool_result.error

    def test_step_with_text_only(self, env: CalculatorEnvironment):
        """Test step with text-only action."""
        state, _ = env.reset()

        action = Action(text="I'm thinking about the problem...")

        result = env.step(state, action)

        assert len(result.next_state.observation.tool_results) == 0
        assert not result.terminated

    def test_message_history_builds_correctly(self, env: CalculatorEnvironment):
        """Test that message history accumulates correctly."""
        state, _ = env.reset()

        # First action
        action1 = Action(
            text="Let me add 5 and 3",
            tool_calls=(
                ToolCall(id="call_1", name="add", arguments={"a": 5, "b": 3}),
            ),
        )
        result1 = env.step(state, action1)

        # Check messages contain assistant and tool messages
        messages = result1.next_state.observation.messages
        assert len(messages) == 2  # assistant message + tool result
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Let me add 5 and 3"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_1"

        # Second action
        action2 = Action(
            text="Now multiply by 7",
            tool_calls=(
                ToolCall(id="call_2", name="multiply", arguments={"a": 8, "b": 7}),
            ),
        )
        result2 = env.step(result1.next_state, action2)

        # Check messages accumulated
        messages = result2.next_state.observation.messages
        assert len(messages) == 4

    def test_tool_results_only_from_last_step(self, env: CalculatorEnvironment):
        """Test that tool_results only contains results from the last step."""
        state, _ = env.reset()

        # First action
        action1 = Action(
            tool_calls=(
                ToolCall(id="call_1", name="add", arguments={"a": 5, "b": 3}),
            ),
        )
        result1 = env.step(state, action1)
        assert len(result1.next_state.observation.tool_results) == 1

        # Second action
        action2 = Action(
            tool_calls=(
                ToolCall(id="call_2", name="multiply", arguments={"a": 8, "b": 7}),
            ),
        )
        result2 = env.step(result1.next_state, action2)

        # tool_results should only have results from the second step
        assert len(result2.next_state.observation.tool_results) == 1
        assert result2.next_state.observation.tool_results[0].call_id == "call_2"

    def test_hidden_state_tracks_calls(self, env: CalculatorEnvironment):
        """Test that hidden state tracks all tool calls made."""
        state, _ = env.reset()

        action1 = Action(
            tool_calls=(
                ToolCall(id="call_1", name="add", arguments={"a": 5, "b": 3}),
            ),
        )
        result1 = env.step(state, action1)

        action2 = Action(
            tool_calls=(
                ToolCall(id="call_2", name="multiply", arguments={"a": 8, "b": 7}),
            ),
        )
        result2 = env.step(result1.next_state, action2)

        # Hidden state should have all calls
        assert len(result2.next_state.hidden.calls_made) == 2
