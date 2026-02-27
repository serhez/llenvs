"""Tests for the tau2 adapter."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from llenvs.core.extraction import TagBasedExtractor
from llenvs.core.reward import FormatReward, RewardType
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.core.tools import ToolCall, ToolParameterType

# ── Mock tau2 objects ────────────────────────────────────────────


@dataclass
class MockTau2Tool:
    """Mock tau2 Tool with openai_schema property."""

    name: str
    description: str = "A mock tool"
    _schema_params: dict = field(default_factory=dict)

    @property
    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._schema_params
                or {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "First argument"},
                    },
                    "required": ["arg1"],
                },
            },
        }


def _make_nested_tool() -> MockTau2Tool:
    """Create a tool with nested Pydantic-style schema."""
    return MockTau2Tool(
        name="update_booking",
        description="Update a flight booking",
        _schema_params={
            "type": "object",
            "properties": {
                "booking_id": {"type": "string", "description": "Booking ID"},
                "passengers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "dob": {"type": "string", "format": "date"},
                        },
                        "required": ["name"],
                    },
                    "description": "Updated passenger list",
                },
            },
            "required": ["booking_id", "passengers"],
        },
    )


class MockTau2Environment:
    """Mock tau2 Environment."""

    def __init__(
        self,
        tools: list[MockTau2Tool] | None = None,
        policy: str = "Be helpful.",
        solo_mode: bool = False,
    ):
        self.tools = tools or [
            MockTau2Tool("get_user_details", "Get user info"),
            MockTau2Tool("update_order", "Update an order"),
            MockTau2Tool(
                "transfer_to_human",
                "Transfer to human",
            ),
        ]
        self.policy = policy
        self.solo_mode = solo_mode
        self._state_set = False
        self._tool_calls: list[dict] = []

    def get_tools(self) -> list[MockTau2Tool]:
        return self.tools

    def get_user_tools(self) -> list:
        return []

    def get_policy(self) -> str:
        return self.policy

    def set_state(
        self, initialization_data=None, initialization_actions=None, message_history=None
    ):
        self._state_set = True

    def make_tool_call(self, tool_name: str, requestor: str = "assistant", **kwargs) -> Any:
        self._tool_calls.append({"name": tool_name, "arguments": kwargs})
        for t in self.tools:
            if t.name == tool_name:
                return {"status": "success", "result": f"Called {tool_name}"}
        raise ValueError(f"Tool {tool_name} not found")

    def sync_tools(self):
        pass


class MockTau2Task:
    """Mock tau2 Task."""

    def __init__(
        self,
        task_id: str = "task_001",
        user_instructions: str = "I want to change my booking.",
        ticket: str | None = None,
        initial_state: dict | None = None,
        evaluation_criteria: dict | None = None,
        description: dict | None = None,
    ):
        self.id = task_id
        self.ticket = ticket

        # UserScenario mock
        self.user_scenario = MagicMock()
        self.user_scenario.instructions = user_instructions
        self.user_scenario.persona = "Frustrated customer"

        # InitialState mock
        self.initial_state = MagicMock()
        if initial_state:
            self.initial_state.initialization_data = initial_state.get("initialization_data")
            self.initial_state.initialization_actions = initial_state.get("initialization_actions")
            self.initial_state.message_history = initial_state.get("message_history")
        else:
            self.initial_state.initialization_data = None
            self.initial_state.initialization_actions = None
            self.initial_state.message_history = None

        # EvaluationCriteria mock
        self.evaluation_criteria = MagicMock()
        if evaluation_criteria:
            self.evaluation_criteria.reward_basis = evaluation_criteria.get("reward_basis", [])
        else:
            self.evaluation_criteria.reward_basis = []

        # Description mock
        self.description = MagicMock()
        if description:
            self.description.purpose = description.get("purpose", "")
        else:
            self.description.purpose = "Test task"


class MockUserSimulator:
    """Mock tau2 UserSimulator."""

    def __init__(self, responses: list[str] | None = None, stop_after: int | None = None):
        self._responses = list(responses or ["Sure, let me provide my details."])
        self._call_count = 0
        self._stop_after = stop_after

    def get_init_state(self, message_history=None):
        return {"messages": list(message_history or [])}

    def generate_next_message(self, message, state):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1

        if self._stop_after is not None and self._call_count >= self._stop_after:
            content = "###STOP###"
        else:
            content = self._responses[idx]

        user_msg = MagicMock()
        user_msg.content = content
        user_msg.role = "user"
        user_msg.tool_calls = None

        state["messages"] = state.get("messages", []) + [message, user_msg]
        return user_msg, state

    @staticmethod
    def is_stop(message) -> bool:
        content = getattr(message, "content", str(message))
        return (
            "###STOP###" in content
            or "###TRANSFER###" in content
            or "###OUT-OF-SCOPE###" in content
        )


class MockRewardInfo:
    """Mock tau2 RewardInfo."""

    def __init__(
        self,
        reward: float = 0.5,
        db_check: dict | None = None,
        action_checks: list | None = None,
        communicate_checks: list | None = None,
        nl_assertions: list | None = None,
    ):
        self.reward = reward
        self.db_check = db_check
        self.action_checks = action_checks
        self.communicate_checks = communicate_checks
        self.nl_assertions = nl_assertions
        self.reward_breakdown = {}
        self.reward_basis = []
        self.info = {}
        self.env_assertions = None


def _make_tasks(n: int = 5) -> list[MockTau2Task]:
    return [MockTau2Task(task_id=f"task_{i:03d}") for i in range(n)]


# ── TestTau2ToolConversion ───────────────────────────────────────


class TestTau2ToolConversion:
    def test_basic_conversion(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        tools = [MockTau2Tool("get_info", "Get information")]
        defs = _tau2_tools_to_definitions(tools)

        assert len(defs) == 1
        assert defs[0].name == "get_info"
        assert defs[0].description == "Get information"

    def test_empty_tools(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        assert _tau2_tools_to_definitions([]) == ()

    def test_multiple_tools(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        tools = [
            MockTau2Tool("tool_a", "Tool A"),
            MockTau2Tool("tool_b", "Tool B"),
            MockTau2Tool("tool_c", "Tool C"),
        ]
        defs = _tau2_tools_to_definitions(tools)
        assert len(defs) == 3
        names = {d.name for d in defs}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_raw_schema_preserved(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        tool = _make_nested_tool()
        defs = _tau2_tools_to_definitions([tool])

        assert defs[0].raw_schema is not None
        # Round-trip via to_openai_schema should match original
        assert defs[0].to_openai_schema() == tool.openai_schema

    def test_nested_schema_roundtrip(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        tool = _make_nested_tool()
        defs = _tau2_tools_to_definitions([tool])
        oai = defs[0].to_openai_schema()

        # Nested structure preserved
        passengers = oai["function"]["parameters"]["properties"]["passengers"]
        assert passengers["type"] == "array"
        assert passengers["items"]["type"] == "object"
        assert "name" in passengers["items"]["properties"]

    def test_flat_params_best_effort(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        tool = MockTau2Tool("simple", "Simple tool")
        defs = _tau2_tools_to_definitions([tool])
        assert len(defs[0].parameters) == 1
        assert defs[0].parameters[0].name == "arg1"
        assert defs[0].parameters[0].type == ToolParameterType.STRING

    def test_anthropic_schema_from_raw(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        tool = _make_nested_tool()
        defs = _tau2_tools_to_definitions([tool])
        anthropic = defs[0].to_anthropic_schema()

        assert anthropic["name"] == "update_booking"
        assert anthropic["description"] == "Update a flight booking"
        assert "passengers" in anthropic["input_schema"]["properties"]

    def test_tool_without_openai_schema_skipped(self):
        from llenvs.adapters.tau2 import _tau2_tools_to_definitions

        bad_tool = MagicMock(spec=[])  # No openai_schema attribute
        good_tool = MockTau2Tool("good", "Good tool")
        defs = _tau2_tools_to_definitions([bad_tool, good_tool])
        assert len(defs) == 1
        assert defs[0].name == "good"


# ── TestTau2Hidden ──────────────────────────────────────────────


class TestTau2Hidden:
    def test_creation(self):
        from llenvs.adapters.tau2 import Tau2Hidden

        hidden = Tau2Hidden(
            task_index=0,
            task_id="task_001",
            domain="airline",
            episode_step=0,
        )
        assert hidden.task_index == 0
        assert hidden.task_id == "task_001"
        assert hidden.domain == "airline"
        assert hidden.episode_step == 0

    def test_frozen(self):
        from llenvs.adapters.tau2 import Tau2Hidden

        hidden = Tau2Hidden(task_index=0, task_id="t1", domain="retail")
        with pytest.raises(AttributeError):
            hidden.task_index = 1  # type: ignore

    def test_defaults(self):
        from llenvs.adapters.tau2 import Tau2Hidden

        hidden = Tau2Hidden(task_index=0, task_id="t1", domain="retail")
        assert hidden.episode_step == 0
        assert hidden.last_action is None
        assert hidden.messages == ()
        assert hidden.termination_reason is None
        assert hidden.reward_info is None

    def test_full_creation(self):
        from llenvs.adapters.tau2 import Tau2Hidden

        hidden = Tau2Hidden(
            task_index=2,
            task_id="task_005",
            domain="telecom",
            episode_step=3,
            last_action="update_order",
            messages=({"role": "user", "content": "hello"},),
            termination_reason="agent_stop",
        )
        assert hidden.task_index == 2
        assert hidden.domain == "telecom"
        assert hidden.episode_step == 3
        assert len(hidden.messages) == 1
        assert hidden.termination_reason == "agent_stop"


# ── TestTau2Reward ──────────────────────────────────────────────


class TestTau2Reward:
    def _make_states(self, is_terminal=True, reward_info=None):
        from llenvs.adapters.tau2 import Tau2Hidden

        hidden = Tau2Hidden(
            task_index=0,
            task_id="t1",
            domain="airline",
            reward_info=reward_info,
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )
        next_state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="e1", is_terminal=is_terminal),
        )
        action = Action(text="done")
        return state, action, next_state

    def test_name(self):
        from llenvs.adapters.tau2 import Tau2Reward

        reward = Tau2Reward()
        assert reward.name == "tau2"

    def test_reward_type(self):
        from llenvs.adapters.tau2 import Tau2Reward

        reward = Tau2Reward()
        assert reward.reward_type == RewardType.OUTCOME

    def test_intermediate_none(self):
        from llenvs.adapters.tau2 import Tau2Reward

        reward = Tau2Reward()
        state, action, next_state = self._make_states(is_terminal=False)
        signal = reward.compute(state, action, next_state)
        assert signal.reward is None
        assert signal.reward_type == RewardType.STEP

    def test_terminal_no_reward_info(self):
        from llenvs.adapters.tau2 import Tau2Reward

        reward = Tau2Reward()
        state, action, next_state = self._make_states(is_terminal=True)
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.0
        assert signal.reward_type == RewardType.OUTCOME

    def test_terminal_with_reward_info(self):
        from llenvs.adapters.tau2 import Tau2Reward

        reward_info = MockRewardInfo(reward=0.8)
        reward = Tau2Reward()
        state, action, next_state = self._make_states(is_terminal=True, reward_info=reward_info)
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.8
        assert signal.reward_type == RewardType.OUTCOME


class TestTau2DetailedRewards:
    def _make_states(self, is_terminal=True, reward_info=None):
        from llenvs.adapters.tau2 import Tau2Hidden

        hidden = Tau2Hidden(
            task_index=0,
            task_id="t1",
            domain="airline",
            reward_info=reward_info,
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )
        next_state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="e1", is_terminal=is_terminal),
        )
        action = Action(text="done")
        return state, action, next_state

    def test_name(self):
        from llenvs.adapters.tau2 import Tau2DetailedRewards

        reward = Tau2DetailedRewards()
        assert reward.name == "tau2_detailed"

    def test_intermediate_none(self):
        from llenvs.adapters.tau2 import Tau2DetailedRewards

        reward = Tau2DetailedRewards()
        state, action, next_state = self._make_states(is_terminal=False)
        signal = reward.compute(state, action, next_state)
        assert signal.reward is None
        assert signal.reward_type == RewardType.STEP

    def test_terminal_with_db_check(self):
        from llenvs.adapters.tau2 import Tau2DetailedRewards

        db_check = MagicMock()
        db_check.db_reward = 1.0
        db_check.db_match = True

        reward_info = MockRewardInfo(reward=0.8, db_check=db_check)
        reward = Tau2DetailedRewards()
        state, action, next_state = self._make_states(is_terminal=True, reward_info=reward_info)
        signal = reward.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        # Metadata should have breakdown
        assert signal.metadata is not None
        assert "db_reward" in signal.metadata

    def test_terminal_no_reward_info(self):
        from llenvs.adapters.tau2 import Tau2DetailedRewards

        reward = Tau2DetailedRewards()
        state, action, next_state = self._make_states(is_terminal=True)
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.0
        assert signal.reward_type == RewardType.OUTCOME


# ── TestTau2Environment ─────────────────────────────────────────


class TestTau2Environment:
    def _make_env(self, tasks=None, domain="airline", **kwargs):
        from llenvs.adapters.tau2 import Tau2Environment

        t = tasks or _make_tasks()
        mock_tau2_env = MockTau2Environment()
        return Tau2Environment(
            domain=domain,
            tasks=t,
            tau2_env=mock_tau2_env,
            **kwargs,
        )

    def test_spec(self):
        env = self._make_env()
        spec = env.spec
        assert spec.name == "tau2:airline"
        assert spec.adapter == "tau2"
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True

    def test_spec_max_steps(self):
        env = self._make_env(max_steps=50)
        assert env.spec.max_steps == 50

    def test_len(self):
        env = self._make_env()
        assert len(env) == 5

    def test_len_custom(self):
        tasks = _make_tasks(10)
        env = self._make_env(tasks=tasks)
        assert len(env) == 10

    def test_prompts_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_reward_functions(self):
        env = self._make_env()
        assert len(env.reward_functions) >= 1
        assert env.reward_functions[0].name == "tau2"

    def test_extra_rewards(self):
        extractor = TagBasedExtractor()
        format_reward = FormatReward(extractor)
        env = self._make_env(extra_rewards=(format_reward,))
        names = [r.name for r in env.reward_functions]
        assert "tau2" in names
        assert "format" in names

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.task_id == "task_000"
        assert state.hidden.domain == "airline"
        assert state.hidden.episode_step == 0
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0

        # Structured observation: task set on reset (tool env, no state)
        obs = state.observation
        assert isinstance(obs.task, ObservationContent)
        assert obs.task.text == obs.prompt
        assert obs.state is None  # tool adapters don't set state

    def test_reset_tools_available(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert len(state.observation.available_tools) == 3
        assert info["num_tools"] == 3
        tool_names = {t.name for t in state.observation.available_tools}
        assert "get_user_details" in tool_names
        assert "update_order" in tool_names

    def test_reset_requires_task_index(self):
        env = self._make_env()
        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_reset_validates_bounds(self):
        env = self._make_env()
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 100})
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": -1})

    def test_reset_custom_episode_id(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0, "episode_id": "custom-ep"})
        assert state.metadata.episode_id == "custom-ep"

    def test_step_tool_call(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1",
                    name="get_user_details",
                    arguments={"arg1": "user123"},
                ),
            )
        )
        result = env.step(state, action)

        assert result.terminated is False
        assert result.next_state.hidden.episode_step == 1
        assert len(result.next_state.observation.messages) > 0

        # Structured observation: task carried forward, state has tool results
        next_obs = result.next_state.observation
        assert next_obs.task is not None
        assert next_obs.task.text == state.observation.prompt  # task stays as initial prompt
        assert next_obs.state is not None  # state reflects tool results

    def test_step_text_to_user(self):
        """Text-only action goes to user simulator."""
        user_sim = MockUserSimulator(responses=["Thanks, I'll check."])
        env = self._make_env(user_simulator=user_sim)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="Your booking has been updated.")
        result = env.step(state, action)

        assert result.terminated is False
        # Should have new messages
        assert len(result.next_state.observation.messages) > 0

    def test_step_user_stop(self):
        """User sending ###STOP### terminates episode."""
        user_sim = MockUserSimulator(stop_after=1)
        env = self._make_env(user_simulator=user_sim)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="Anything else?")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True
        assert result.next_state.hidden.termination_reason == "user_stop"

    def test_step_max_steps_truncation(self):
        env = self._make_env(max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "x"}),)
        )
        result = env.step(state, action)

        assert result.truncated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_message_history_grows(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action1 = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "x"}),)
        )
        result1 = env.step(state, action1)
        msg_count_1 = len(result1.next_state.observation.messages)
        assert msg_count_1 > 0

        action2 = Action(
            tool_calls=(ToolCall(id="tc-2", name="update_order", arguments={"arg1": "y"}),)
        )
        result2 = env.step(result1.next_state, action2)
        msg_count_2 = len(result2.next_state.observation.messages)
        assert msg_count_2 > msg_count_1

    def test_step_stale_state_raises(self):
        env = self._make_env(max_steps=10)
        state_0, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "x"}),)
        )
        env.step(state_0, action)

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, action)

    def test_step_rewards_computed(self):
        user_sim = MockUserSimulator(stop_after=1)
        env = self._make_env(user_simulator=user_sim)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="Done")
        result = env.step(state, action)

        sig = result.rewards.by_name("tau2")
        assert sig is not None

    def test_step_tool_results_in_info(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "x"}),)
        )
        result = env.step(state, action)
        assert "tool_results" in result.info

    def test_step_unknown_tool(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(tool_calls=(ToolCall(id="tc-1", name="nonexistent_tool", arguments={}),))
        result = env.step(state, action)
        tool_results = result.info["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error

    def test_step_tool_execution_error(self):
        """Tool execution that raises should produce error result."""
        mock_env = MockTau2Environment()
        # Override make_tool_call to raise
        mock_env.make_tool_call = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("DB error"))

        from llenvs.adapters.tau2 import Tau2Environment

        env = Tau2Environment(
            domain="airline",
            tasks=_make_tasks(),
            tau2_env=mock_env,
            max_steps=10,
        )
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "x"}),)
        )
        result = env.step(state, action)
        tool_results = result.info["tool_results"]
        assert tool_results[0].is_error

    def test_compute_rewards_directly(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        next_state = State(
            observation=state.observation,
            hidden=state.hidden,
            metadata=StateMetadata(step=1, episode_id=state.metadata.episode_id, is_terminal=True),
        )
        rewards = env.compute_rewards(state, Action(text="x"), next_state)
        assert len(rewards.signals) >= 1


# ── TestTau2SoloMode ────────────────────────────────────────────


class TestTau2SoloMode:
    def _make_env(self, tasks=None, **kwargs):
        from llenvs.adapters.tau2 import Tau2Environment

        t = tasks or [MockTau2Task(ticket="Fix order #12345")]
        mock_tau2_env = MockTau2Environment(solo_mode=True)
        return Tau2Environment(
            domain="airline",
            tasks=t,
            tau2_env=mock_tau2_env,
            solo_mode=True,
            **kwargs,
        )

    def test_solo_mode_spec(self):
        env = self._make_env()
        assert env._solo_mode is True

    def test_solo_reset_with_ticket(self):
        tasks = [MockTau2Task(ticket="Fix order #12345")]
        env = self._make_env(tasks=tasks)
        state, _ = env.reset(options={"task_index": 0})
        # Ticket should be in the prompt
        assert "Fix order #12345" in state.observation.prompt

    def test_solo_tool_only(self):
        """Solo mode agent sends tool calls."""
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "x"}),)
        )
        result = env.step(state, action)
        assert result.terminated is False

    def test_solo_text_with_stop_terminates(self):
        """In solo mode, text containing ###STOP### terminates."""
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="###STOP###")
        result = env.step(state, action)
        assert result.terminated is True
        assert result.next_state.hidden.termination_reason == "agent_stop"

    def test_solo_text_without_stop_ignored(self):
        """In solo mode, regular text doesn't go to user simulator."""
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="Thinking about what to do next...")
        result = env.step(state, action)
        # Should not terminate (no user to send stop)
        assert result.terminated is False


# ── TestTau2Adapter ─────────────────────────────────────────────


class TestTau2Adapter:
    def test_name(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        assert adapter.name == "tau2"

    def test_list_environments(self):
        from llenvs.adapters.tau2 import TAU2_DOMAINS, Tau2Adapter

        adapter = Tau2Adapter()
        envs = adapter.list_environments()
        for domain in TAU2_DOMAINS:
            assert f"tau2:{domain}" in envs

    def test_import_error(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        with pytest.raises(ImportError, match="tau2"):
            adapter._get_tau2()

    def test_get_environment_with_tasks(self, monkeypatch):
        from llenvs.adapters.tau2 import Tau2Adapter, Tau2Environment

        adapter = Tau2Adapter()
        mock_tau2 = MagicMock()
        monkeypatch.setattr(adapter, "_get_tau2", lambda: mock_tau2)

        tasks = _make_tasks()
        mock_tau2_env = MockTau2Environment()
        env = adapter.get_environment("tau2:airline", tasks=tasks, tau2_env=mock_tau2_env)
        assert isinstance(env, Tau2Environment)

    def test_get_environment_requires_tasks_or_loader(self, monkeypatch):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        mock_tau2 = MagicMock()
        # Make registry.get_tasks_loader raise so we get the ValueError
        mock_tau2.registry.get_tasks_loader.side_effect = RuntimeError("no tasks")
        monkeypatch.setattr(adapter, "_get_tau2", lambda: mock_tau2)

        with pytest.raises(ValueError, match="tasks"):
            adapter.get_environment("tau2:airline")

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        assert adapter.get_native_answer_extractor("tau2:airline") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        assert adapter.get_prompt_template("tau2:airline") is None

    def test_get_environment_info(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        info = adapter.get_environment_info("tau2:airline")
        assert info["name"] == "tau2:airline"
        assert info["adapter"] == "tau2"

    def test_max_steps_passed_through(self, monkeypatch):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        mock_tau2 = MagicMock()
        monkeypatch.setattr(adapter, "_get_tau2", lambda: mock_tau2)

        tasks = _make_tasks()
        mock_tau2_env = MockTau2Environment()
        env = adapter.get_environment(
            "tau2:airline", tasks=tasks, tau2_env=mock_tau2_env, max_steps=50
        )
        assert env._max_steps == 50

    def test_get_default_system_prompt(self, monkeypatch):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        mock_tau2 = MagicMock()
        monkeypatch.setattr(adapter, "_get_tau2", lambda: mock_tau2)

        mock_tau2_env = MockTau2Environment(policy="Always help the customer.")
        prompt = adapter.get_default_system_prompt("tau2:airline", tau2_env=mock_tau2_env)
        assert "Always help the customer" in prompt

    def test_domain_parsing(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        assert adapter._parse_domain("tau2:airline") == "airline"
        assert adapter._parse_domain("tau2:retail") == "retail"
        assert adapter._parse_domain("tau2:telecom") == "telecom"
        assert adapter._parse_domain("tau2:airline:base") == "airline"

    def test_split_parsing(self):
        from llenvs.adapters.tau2 import Tau2Adapter

        adapter = Tau2Adapter()
        assert adapter._parse_split("tau2:airline") is None
        assert adapter._parse_split("tau2:airline:base") == "base"
        assert adapter._parse_split("tau2:airline:test") == "test"


# ── TestTau2Integration ─────────────────────────────────────────


class TestTau2Integration:
    """Integration-style tests using the full adapter flow."""

    def test_tool_call_then_text_then_stop(self):
        """Full episode: tool call -> text to user -> user stops."""
        from llenvs.adapters.tau2 import Tau2Environment

        user_sim = MockUserSimulator(
            responses=["I'll check.", "###STOP###"],
        )
        env = Tau2Environment(
            domain="airline",
            tasks=_make_tasks(),
            tau2_env=MockTau2Environment(),
            user_simulator=user_sim,
            max_steps=10,
        )

        state, _ = env.reset(options={"task_index": 0})

        # Step 1: tool call
        action1 = Action(
            tool_calls=(ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "u1"}),)
        )
        result1 = env.step(state, action1)
        assert not result1.done

        # Step 2: text to user
        action2 = Action(text="Your details are: ...")
        result2 = env.step(result1.next_state, action2)
        assert not result2.done

        # Step 3: text to user -> user responds with STOP
        action3 = Action(text="Anything else?")
        result3 = env.step(result2.next_state, action3)
        assert result3.terminated is True

    def test_multiple_tool_calls_single_step(self):
        """Multiple tool calls in a single action."""
        from llenvs.adapters.tau2 import Tau2Environment

        env = Tau2Environment(
            domain="airline",
            tasks=_make_tasks(),
            tau2_env=MockTau2Environment(),
            max_steps=10,
        )
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "u1"}),
                ToolCall(id="tc-2", name="update_order", arguments={"arg1": "o1"}),
            )
        )
        result = env.step(state, action)

        tool_results = result.info["tool_results"]
        assert len(tool_results) == 2
        assert all(r.is_success for r in tool_results)

    def test_mixed_valid_invalid_tools(self):
        """Mix of valid and invalid tool calls."""
        from llenvs.adapters.tau2 import Tau2Environment

        env = Tau2Environment(
            domain="airline",
            tasks=_make_tasks(),
            tau2_env=MockTau2Environment(),
            max_steps=10,
        )
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(id="tc-1", name="get_user_details", arguments={"arg1": "u1"}),
                ToolCall(id="tc-2", name="nonexistent", arguments={}),
            )
        )
        result = env.step(state, action)

        tool_results = result.info["tool_results"]
        assert tool_results[0].is_success
        assert tool_results[1].is_error
