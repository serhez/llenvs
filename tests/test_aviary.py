"""Tests for the Aviary adapter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llenvs.core.extraction import TagBasedExtractor
from llenvs.core.reward import FormatReward, RewardType
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.tools import ToolCall, ToolDefinition, ToolParameter, ToolParameterType

# ── Mock Aviary objects ────────────────────────────────────────────


def _make_mock_tool(name="search", description="Search for info", parameters=None):
    """Create a mock Aviary Tool object that model_dump returns OAI schema."""
    tool = MagicMock()
    params = parameters or {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }
    tool.model_dump.return_value = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }
    # Fallback info
    tool.info = MagicMock()
    tool.info.name = name
    tool.info.description = description
    tool.info.parameters = params
    return tool


def _make_mock_message(role="user", content="Hello"):
    """Create a mock Aviary Message."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    msg.tool_call_id = None
    msg.name = None
    return msg


def _make_mock_tool_response(tool_call_id="tc-1", name="search", content="Results here"):
    """Create a mock Aviary ToolResponseMessage."""
    msg = MagicMock()
    msg.role = "tool"
    msg.content = content
    msg.tool_call_id = tool_call_id
    msg.name = name
    return msg


def _make_mock_dataset(num_tasks=5):
    """Create a mock Aviary TaskDataset."""
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=num_tasks)

    # get_new_env_by_idx returns a mock env
    async def get_new_env(idx):
        env = MagicMock()

        async def reset():
            tools = [_make_mock_tool("search"), _make_mock_tool("calculate", "Calculate math")]
            messages = [_make_mock_message("system", f"Task {idx}: Solve the problem.")]
            return messages, tools

        async def step(msg):
            return (
                [_make_mock_tool_response("tc-1", "search", "Result for query")],
                1.0,
                True,
                False,
            )

        env.reset = reset
        env.step = step
        env.close = AsyncMock()
        return env

    ds.get_new_env_by_idx = get_new_env
    return ds


def _make_mock_dataset_intermediate(num_tasks=3, max_turns=3):
    """Create a mock dataset where step returns intermediate (not done) results."""
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=num_tasks)

    step_count = [0]

    async def get_new_env(idx):
        env = MagicMock()
        step_count[0] = 0

        async def reset():
            step_count[0] = 0
            tools = [_make_mock_tool("act")]
            messages = [_make_mock_message("system", f"Task {idx}")]
            return messages, tools

        async def step(msg):
            step_count[0] += 1
            done = step_count[0] >= max_turns
            return (
                [_make_mock_message("assistant", f"Step {step_count[0]} result")],
                0.5 if not done else 1.0,
                done,
                False,
            )

        env.reset = reset
        env.step = step
        env.close = AsyncMock()
        return env

    ds.get_new_env_by_idx = get_new_env
    return ds


# ── TestAviaryHidden ───────────────────────────────────────────────


class TestAviaryHidden:
    def test_creation(self):
        from llenvs.adapters.aviary import AviaryHidden

        hidden = AviaryHidden(
            task_index=0,
            env_name="gsm8k",
            episode_step=0,
        )
        assert hidden.task_index == 0
        assert hidden.env_name == "gsm8k"
        assert hidden.episode_step == 0

    def test_frozen(self):
        from llenvs.adapters.aviary import AviaryHidden

        hidden = AviaryHidden(task_index=0, env_name="test")
        with pytest.raises(AttributeError):
            hidden.task_index = 1  # type: ignore

    def test_defaults(self):
        from llenvs.adapters.aviary import AviaryHidden

        hidden = AviaryHidden(task_index=0, env_name="test")
        assert hidden.episode_step == 0
        assert hidden.last_action is None
        assert hidden.cumulative_reward == 0.0
        assert hidden.aviary_reward == 0.0

    def test_full_creation(self):
        from llenvs.adapters.aviary import AviaryHidden

        hidden = AviaryHidden(
            task_index=2,
            env_name="hotpotqa",
            episode_step=3,
            last_action="search query",
            cumulative_reward=2.5,
            aviary_reward=0.5,
        )
        assert hidden.task_index == 2
        assert hidden.env_name == "hotpotqa"
        assert hidden.episode_step == 3
        assert hidden.last_action == "search query"
        assert hidden.cumulative_reward == 2.5
        assert hidden.aviary_reward == 0.5


# ── TestAviaryReward ──────────────────────────────────────────────


class TestAviaryReward:
    def test_name(self):
        from llenvs.adapters.aviary import AviaryReward

        reward = AviaryReward()
        assert reward.name == "aviary"

    def test_reward_type(self):
        from llenvs.adapters.aviary import AviaryReward

        reward = AviaryReward()
        assert reward.reward_type == RewardType.OUTCOME

    def test_compute_intermediate(self):
        from llenvs.adapters.aviary import AviaryHidden, AviaryReward

        reward = AviaryReward()
        hidden = AviaryHidden(
            task_index=0,
            env_name="test",
            episode_step=1,
            aviary_reward=0.5,
            cumulative_reward=0.5,
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=AviaryHidden(task_index=0, env_name="test"),
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )
        next_state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="e1", is_terminal=False),
        )

        signal = reward.compute(state, Action(text="act"), next_state)
        assert signal.reward == 0.5
        assert signal.reward_type == RewardType.STEP
        assert signal.metadata["is_terminal"] is False

    def test_compute_terminal(self):
        from llenvs.adapters.aviary import AviaryHidden, AviaryReward

        reward = AviaryReward()
        hidden = AviaryHidden(
            task_index=0,
            env_name="test",
            episode_step=3,
            aviary_reward=1.0,
            cumulative_reward=2.5,
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=AviaryHidden(task_index=0, env_name="test"),
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )
        next_state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=3, episode_id="e1", is_terminal=True),
        )

        signal = reward.compute(state, Action(text="done"), next_state)
        assert signal.reward == 1.0
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.metadata["cumulative_reward"] == 2.5

    def test_compute_zero_reward(self):
        from llenvs.adapters.aviary import AviaryHidden, AviaryReward

        reward = AviaryReward()
        hidden = AviaryHidden(task_index=0, env_name="test", aviary_reward=0.0)
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )

        signal = reward.compute(state, Action(text="x"), state)
        assert signal.reward == 0.0

    def test_compute_negative_reward(self):
        from llenvs.adapters.aviary import AviaryHidden, AviaryReward

        reward = AviaryReward()
        hidden = AviaryHidden(task_index=0, env_name="test", aviary_reward=-0.5)
        next_state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="e1", is_terminal=False),
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=AviaryHidden(task_index=0, env_name="test"),
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )

        signal = reward.compute(state, Action(text="bad"), next_state)
        assert signal.reward == -0.5

    def test_custom_name(self):
        from llenvs.adapters.aviary import AviaryReward

        reward = AviaryReward(_name="custom_aviary")
        assert reward.name == "custom_aviary"


# ── TestAviaryToolConversion ──────────────────────────────────────


class TestAviaryToolConversion:
    def test_basic_conversion(self):
        from llenvs.adapters.aviary import _aviary_tools_to_definitions

        tools = [_make_mock_tool("search", "Search for info")]
        defs = _aviary_tools_to_definitions(tools)

        assert len(defs) == 1
        assert defs[0].name == "search"
        assert defs[0].description == "Search for info"

    def test_empty_tools(self):
        from llenvs.adapters.aviary import _aviary_tools_to_definitions

        assert _aviary_tools_to_definitions([]) == ()

    def test_fallback_to_info(self):
        from llenvs.adapters.aviary import _aviary_tools_to_definitions

        tool = MagicMock()
        tool.model_dump.side_effect = AttributeError("no model_dump")
        tool.info = MagicMock()
        tool.info.name = "fallback_tool"
        tool.info.description = "A fallback tool"
        tool.info.parameters = {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "A number"},
            },
            "required": ["x"],
        }

        defs = _aviary_tools_to_definitions([tool])
        assert len(defs) == 1
        assert defs[0].name == "fallback_tool"
        assert len(defs[0].parameters) == 1

    def test_parameters_preserved(self):
        from llenvs.adapters.aviary import _aviary_tools_to_definitions

        tools = [
            _make_mock_tool(
                "calc",
                "Calculate",
                {
                    "type": "object",
                    "properties": {
                        "expr": {"type": "string", "description": "Expression"},
                        "precision": {"type": "integer", "description": "Decimal places"},
                    },
                    "required": ["expr"],
                },
            )
        ]
        defs = _aviary_tools_to_definitions(tools)
        params = {p.name: p for p in defs[0].parameters}
        assert params["expr"].required is True
        assert params["precision"].required is False

    def test_multiple_tools(self):
        from llenvs.adapters.aviary import _aviary_tools_to_definitions

        tools = [
            _make_mock_tool("search", "Search"),
            _make_mock_tool("calculate", "Calculate"),
            _make_mock_tool("submit", "Submit answer"),
        ]
        defs = _aviary_tools_to_definitions(tools)
        assert len(defs) == 3
        names = {d.name for d in defs}
        assert names == {"search", "calculate", "submit"}


# ── TestAviaryMessageConversion ───────────────────────────────────


class TestAviaryMessageConversion:
    def test_tool_responses(self):
        from llenvs.adapters.aviary import _aviary_messages_to_observation

        messages = [_make_mock_tool_response("tc-1", "search", "Found it")]
        tools = (_make_tool_def("search"),)
        action = Action(
            text="",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"query": "test"}),),
        )

        obs, results = _aviary_messages_to_observation(
            messages=messages,
            prompt="Original prompt",
            prior_messages=(),
            action=action,
            available_tools=tools,
        )

        assert len(results) == 1
        assert results[0].is_success
        assert results[0].tool_name == "search"
        assert "Found it" in str(results[0].output)

    def test_user_messages(self):
        from llenvs.adapters.aviary import _aviary_messages_to_observation

        messages = [_make_mock_message("assistant", "Here is my response")]
        action = Action(text="Hello")

        obs, results = _aviary_messages_to_observation(
            messages=messages,
            prompt="Q",
            prior_messages=(),
            action=action,
            available_tools=(),
        )

        assert len(results) == 0
        # Should have assistant action + the response message
        assert len(obs.messages) == 2
        assert obs.messages[0]["role"] == "assistant"
        assert obs.messages[1]["role"] == "assistant"

    def test_mixed_messages(self):
        from llenvs.adapters.aviary import _aviary_messages_to_observation

        messages = [
            _make_mock_tool_response("tc-1", "search", "Result"),
            _make_mock_message("assistant", "Based on the search..."),
        ]
        action = Action(
            text="Let me search",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"q": "x"}),),
        )

        obs, results = _aviary_messages_to_observation(
            messages=messages,
            prompt="Q",
            prior_messages=(),
            action=action,
            available_tools=(),
        )

        assert len(results) == 1
        # assistant action + tool response + assistant message = 3
        assert len(obs.messages) == 3

    def test_action_history_preserved(self):
        from llenvs.adapters.aviary import _aviary_messages_to_observation

        prior = (
            {"role": "assistant", "content": "Previous action"},
            {"role": "tool", "content": "Previous result"},
        )
        messages = [_make_mock_message("assistant", "New result")]
        action = Action(text="New action")

        obs, _ = _aviary_messages_to_observation(
            messages=messages,
            prompt="Q",
            prior_messages=prior,
            action=action,
            available_tools=(),
        )

        # prior(2) + action(1) + new_message(1) = 4
        assert len(obs.messages) == 4
        assert obs.messages[0]["role"] == "assistant"
        assert obs.messages[0]["content"] == "Previous action"

    def test_prompt_preserved(self):
        from llenvs.adapters.aviary import _aviary_messages_to_observation

        obs, _ = _aviary_messages_to_observation(
            messages=[],
            prompt="My original prompt",
            prior_messages=(),
            action=Action(text="act"),
            available_tools=(),
        )

        assert obs.prompt == "My original prompt"

    def test_tool_calls_in_assistant_message(self):
        from llenvs.adapters.aviary import _aviary_messages_to_observation

        action = Action(
            text="Searching...",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"q": "test"}),),
        )
        messages = [_make_mock_tool_response("tc-1", "search", "Found")]

        obs, _ = _aviary_messages_to_observation(
            messages=messages,
            prompt="Q",
            prior_messages=(),
            action=action,
            available_tools=(),
        )

        # Check assistant message has tool_calls
        assert "tool_calls" in obs.messages[0]
        assert obs.messages[0]["tool_calls"][0]["name"] == "search"


# ── TestActionToToolRequest ───────────────────────────────────────


class TestActionToToolRequest:
    @pytest.fixture(autouse=True)
    def _mock_aviary_imports(self, monkeypatch):
        """Mock aviary.core imports for _action_to_tool_request."""
        self.mock_aviary_core = MagicMock()

        # Create proper mock classes
        class MockToolCall:
            def __init__(self, id, function):
                self.id = id
                self.function = function

        class MockToolCallFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class MockToolRequestMessage:
            def __init__(self, content="", tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        self.mock_aviary_core.ToolCall = MockToolCall
        self.mock_aviary_core.ToolCallFunction = MockToolCallFunction
        self.mock_aviary_core.ToolRequestMessage = MockToolRequestMessage
        self.MockToolRequestMessage = MockToolRequestMessage

        import sys

        monkeypatch.setitem(sys.modules, "aviary", MagicMock())
        monkeypatch.setitem(sys.modules, "aviary.core", self.mock_aviary_core)

    def test_basic_tool_call(self):
        from llenvs.adapters.aviary import _action_to_tool_request

        action = Action(
            text="",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"query": "hello"}),),
        )
        result = _action_to_tool_request(action)

        assert isinstance(result, self.MockToolRequestMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].function.name == "search"

    def test_multiple_tool_calls(self):
        from llenvs.adapters.aviary import _action_to_tool_request

        action = Action(
            text="",
            tool_calls=(
                ToolCall(id="tc-1", name="search", arguments={"q": "a"}),
                ToolCall(id="tc-2", name="calc", arguments={"expr": "1+1"}),
            ),
        )
        result = _action_to_tool_request(action)
        assert len(result.tool_calls) == 2

    def test_text_only_action(self):
        from llenvs.adapters.aviary import _action_to_tool_request

        action = Action(text="Just text, no tools")
        result = _action_to_tool_request(action)

        assert result.content == "Just text, no tools"
        assert result.tool_calls is None

    def test_preserves_arguments(self):
        from llenvs.adapters.aviary import _action_to_tool_request

        args = {"query": "hello world", "limit": 10, "nested": {"key": "val"}}
        action = Action(
            text="",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments=args),),
        )
        result = _action_to_tool_request(action)

        parsed = json.loads(result.tool_calls[0].function.arguments)
        assert parsed == args

    def test_text_with_tool_calls(self):
        from llenvs.adapters.aviary import _action_to_tool_request

        action = Action(
            text="Let me search for that",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"q": "test"}),),
        )
        result = _action_to_tool_request(action)

        assert result.content == "Let me search for that"
        assert len(result.tool_calls) == 1


# ── TestAviaryEnvironment ─────────────────────────────────────────


def _make_tool_def(name: str) -> ToolDefinition:
    """Helper to create a simple ToolDefinition."""
    return ToolDefinition(
        name=name,
        description=f"Tool: {name}",
        parameters=(
            ToolParameter(
                name="query",
                type=ToolParameterType.STRING,
                description="Input",
                required=True,
            ),
        ),
    )


class TestAviaryEnvironment:
    @pytest.fixture(autouse=True)
    def _mock_aviary(self, monkeypatch):
        """Mock aviary imports for environment tests."""
        import sys

        # Mock aviary.core for _action_to_tool_request
        mock_core = MagicMock()

        class MockToolCall:
            def __init__(self, id, function):
                self.id = id
                self.function = function

        class MockToolCallFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class MockToolRequestMessage:
            def __init__(self, content="", tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        mock_core.ToolCall = MockToolCall
        mock_core.ToolCallFunction = MockToolCallFunction
        mock_core.ToolRequestMessage = MockToolRequestMessage

        monkeypatch.setitem(sys.modules, "aviary", MagicMock())
        monkeypatch.setitem(sys.modules, "aviary.core", mock_core)

    def _make_env(self, dataset=None, **kwargs):
        from llenvs.adapters.aviary import AviaryEnvironment

        ds = dataset or _make_mock_dataset()
        return AviaryEnvironment(dataset=ds, env_name="test", **kwargs)

    def test_spec(self):
        env = self._make_env()
        spec = env.spec
        assert spec.name == "test"
        assert spec.adapter == "aviary"
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is False

    def test_spec_max_steps(self):
        env = self._make_env(max_steps=10)
        assert env.spec.max_steps == 10

    def test_spec_no_max_steps(self):
        env = self._make_env()
        assert env.spec.max_steps is None

    def test_len(self):
        env = self._make_env()
        assert len(env) == 5

    def test_prompts_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_reward_functions(self):
        env = self._make_env()
        # AviaryReward + 2 monitoring rewards
        assert len(env.reward_functions) >= 1
        assert env.reward_functions[0].name == "aviary"

    def test_extra_rewards(self):
        extractor = TagBasedExtractor()
        format_reward = FormatReward(extractor)
        env = self._make_env(extra_rewards=(format_reward,))
        names = [r.name for r in env.reward_functions]
        assert "aviary" in names
        assert "format" in names

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.env_name == "test"
        assert state.hidden.episode_step == 0
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0

    def test_reset_prompt_from_messages(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 2})
        assert "Task 2" in state.observation.prompt

    def test_reset_tools_available(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert len(state.observation.available_tools) == 2
        assert info["num_tools"] == 2
        tool_names = {t.name for t in state.observation.available_tools}
        assert "search" in tool_names
        assert "calculate" in tool_names

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
        state, _ = env.reset(options={"task_index": 0, "episode_id": "custom"})
        assert state.metadata.episode_id == "custom"

    def test_step(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            text="",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"query": "test"}),),
        )
        result = env.step(state, action)

        assert result.terminated is True
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.aviary_reward == 1.0
        assert result.next_state.hidden.cumulative_reward == 1.0

    def test_step_rewards_computed(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            text="",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"query": "x"}),),
        )
        result = env.step(state, action)

        aviary_signal = result.rewards.by_name("aviary")
        assert aviary_signal is not None
        assert aviary_signal.reward == 1.0

    def test_step_tool_results_in_info(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            text="",
            tool_calls=(ToolCall(id="tc-1", name="search", arguments={"query": "x"}),),
        )
        result = env.step(state, action)

        assert "tool_results" in result.info
        assert len(result.info["tool_results"]) >= 1

    def test_step_raises_on_stale_state(self):
        env = self._make_env()
        state_0, _ = env.reset(options={"task_index": 0})

        action = Action(text="act")
        # Use intermediate dataset so first step isn't terminal
        env2 = self._make_env(dataset=_make_mock_dataset_intermediate())
        state_0, _ = env2.reset(options={"task_index": 0})
        env2.step(state_0, action)

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env2.step(state_0, action)

    def test_step_no_active_env(self):
        env = self._make_env()
        hidden = MagicMock()
        hidden.episode_step = 0
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )

        with pytest.raises(RuntimeError, match="No active Aviary environment"):
            env.step(state, Action(text="x"))

    def test_step_max_steps_truncation(self):
        ds = _make_mock_dataset_intermediate(max_turns=10)
        env = self._make_env(dataset=ds, max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="act"))
        assert result.truncated is True

    def test_step_message_history_grows(self):
        ds = _make_mock_dataset_intermediate(max_turns=5)
        env = self._make_env(dataset=ds)
        state, _ = env.reset(options={"task_index": 0})

        result1 = env.step(state, Action(text="step 1"))
        assert len(result1.next_state.observation.messages) > 0

        result2 = env.step(result1.next_state, Action(text="step 2"))
        assert len(result2.next_state.observation.messages) > len(
            result1.next_state.observation.messages
        )

    def test_step_cumulative_reward(self):
        ds = _make_mock_dataset_intermediate(max_turns=3)
        env = self._make_env(dataset=ds)
        state, _ = env.reset(options={"task_index": 0})

        result1 = env.step(state, Action(text="step 1"))
        assert result1.next_state.hidden.cumulative_reward == 0.5

        result2 = env.step(result1.next_state, Action(text="step 2"))
        assert result2.next_state.hidden.cumulative_reward == 1.0

    def test_close(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        env.close()
        assert env._active_env is None

    def test_close_without_reset(self):
        env = self._make_env()
        # Should not raise
        env.close()

    def test_compute_rewards_directly(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        hidden = state.hidden
        next_state = State(
            observation=state.observation,
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id=state.metadata.episode_id, is_terminal=True),
        )
        rewards = env.compute_rewards(state, Action(text="x"), next_state)
        assert len(rewards.signals) >= 1


# ── TestAviaryAdapter ─────────────────────────────────────────────


class TestAviaryAdapter:
    def test_name(self):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        assert adapter.name == "aviary"

    def test_list_environments(self):
        from llenvs.adapters.aviary import AVIARY_PRESETS, AviaryAdapter

        adapter = AviaryAdapter()
        envs = adapter.list_environments()
        assert set(envs) == set(AVIARY_PRESETS.keys())
        assert "gsm8k" in envs
        assert "hotpotqa" in envs

    def test_get_aviary_import_error(self):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        with pytest.raises(ImportError, match="fhaviary"):
            adapter._get_aviary()

    def test_get_environment_with_dataset(self, monkeypatch):
        from llenvs.adapters.aviary import AviaryAdapter, AviaryEnvironment

        adapter = AviaryAdapter()
        monkeypatch.setattr(adapter, "_get_aviary", lambda: MagicMock())

        ds = _make_mock_dataset()
        env = adapter.get_environment("custom", dataset=ds)
        assert isinstance(env, AviaryEnvironment)
        assert env._env_name == "custom"

    def test_get_environment_unknown_preset(self, monkeypatch):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        monkeypatch.setattr(adapter, "_get_aviary", lambda: MagicMock())

        with pytest.raises(ValueError, match="Unknown Aviary environment"):
            adapter.get_environment("nonexistent")

    def test_get_environment_from_preset(self, monkeypatch):
        from llenvs.adapters.aviary import AviaryAdapter, AviaryEnvironment

        adapter = AviaryAdapter()
        monkeypatch.setattr(adapter, "_get_aviary", lambda: MagicMock())

        # Mock the importlib.import_module call
        mock_module = MagicMock()
        mock_dataset_cls = MagicMock(return_value=_make_mock_dataset())
        mock_module.GSM8kDataset = mock_dataset_cls

        import importlib

        monkeypatch.setattr(importlib, "import_module", lambda m: mock_module)

        env = adapter.get_environment("gsm8k")
        assert isinstance(env, AviaryEnvironment)
        mock_dataset_cls.assert_called_once()

    def test_get_environment_passes_kwargs(self, monkeypatch):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        monkeypatch.setattr(adapter, "_get_aviary", lambda: MagicMock())

        mock_module = MagicMock()
        mock_dataset_cls = MagicMock(return_value=_make_mock_dataset())
        mock_module.GSM8kDataset = mock_dataset_cls

        import importlib

        monkeypatch.setattr(importlib, "import_module", lambda m: mock_module)

        adapter.get_environment("gsm8k", split="test")
        mock_dataset_cls.assert_called_once_with(split="test")

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        assert adapter.get_native_answer_extractor("gsm8k") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        assert adapter.get_prompt_template("gsm8k") is None

    def test_get_environment_info(self):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        info = adapter.get_environment_info("gsm8k")
        assert info["name"] == "gsm8k"
        assert info["adapter"] == "aviary"
        assert "preset" in info

    def test_get_environment_info_unknown(self):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        info = adapter.get_environment_info("custom")
        assert info["name"] == "custom"
        assert "preset" not in info

    def test_max_steps_passed_through(self, monkeypatch):
        from llenvs.adapters.aviary import AviaryAdapter

        adapter = AviaryAdapter()
        monkeypatch.setattr(adapter, "_get_aviary", lambda: MagicMock())

        ds = _make_mock_dataset()
        env = adapter.get_environment("custom", dataset=ds, max_steps=20)
        assert env._max_steps == 20
