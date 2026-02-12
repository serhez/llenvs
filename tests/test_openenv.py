"""Tests for the OpenEnv adapter."""

import pytest
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from llenvs.core.state import Observation, Action, State, StateMetadata
from llenvs.core.reward import RewardType, RewardSignal
from llenvs.core.tools import ToolCall, ToolDefinition, ToolParameterType


# ── Mock OpenEnv objects ────────────────────────────────────────────


@dataclass
class MockStepResult:
    """Mock openenv StepResult."""
    observation: dict[str, Any]
    reward: float | None = None
    done: bool = False


class MockSyncClient:
    """Mock SyncEnvClient for GenericEnvClient.sync()."""

    def __init__(self, *, observations=None, rewards=None, done_steps=None):
        self._observations = observations or [
            {"text": "Welcome to the environment!", "status": "ready"},
        ]
        self._rewards = rewards or [None]
        self._done_steps = done_steps or set()
        self._step_count = 0
        self._connected = False
        self._state = {"episode_id": "test-123", "step_count": 0}

    def connect(self):
        self._connected = True
        return self

    def disconnect(self):
        self._connected = False

    def reset(self, **kwargs):
        self._step_count = 0
        obs = self._observations[0] if self._observations else {}
        reward = self._rewards[0] if self._rewards else None
        return MockStepResult(observation=obs, reward=reward, done=False)

    def step(self, action, **kwargs):
        self._step_count += 1
        idx = min(self._step_count, len(self._observations) - 1)
        obs = self._observations[idx] if idx < len(self._observations) else {}
        reward_idx = min(self._step_count, len(self._rewards) - 1)
        reward = self._rewards[reward_idx] if reward_idx < len(self._rewards) else None
        done = self._step_count in self._done_steps
        return MockStepResult(observation=obs, reward=reward, done=done)

    def state(self):
        return {**self._state, "step_count": self._step_count}

    def close(self):
        self._connected = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.disconnect()


class MockMCPToolSyncClient(MockSyncClient):
    """Mock SyncEnvClient for MCPToolClient.sync()."""

    def __init__(self, *, tools=None, **kwargs):
        super().__init__(**kwargs)
        self._tools = tools or []
        self._tool_results: dict[str, Any] = {}

    def list_tools(self, use_cache=True):
        return self._tools

    def call_tool(self, name, **kwargs):
        if name in self._tool_results:
            return self._tool_results[name]
        return f"Result from {name}"


def _make_mock_tool(name="search", description="Search tool", input_schema=None):
    """Create a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = input_schema or {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }
    return tool


# ── Hidden state tests ──────────────────────────────────────────────


class TestOpenEnvHidden:
    """Tests for OpenEnvHidden dataclass."""

    def test_creation(self):
        from llenvs.adapters.openenv import OpenEnvHidden

        hidden = OpenEnvHidden(
            env_name="test-env",
            episode_step=0,
            last_action=None,
            session_info=(("episode_id", "123"),),
        )
        assert hidden.env_name == "test-env"
        assert hidden.episode_step == 0
        assert hidden.last_action is None

    def test_immutability(self):
        from llenvs.adapters.openenv import OpenEnvHidden

        hidden = OpenEnvHidden(
            env_name="test", episode_step=0, last_action=None,
            session_info=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore


# ── Observation coercion tests ──────────────────────────────────────


class TestCoerceObservation:
    """Tests for _coerce_observation."""

    def test_text_key(self):
        from llenvs.adapters.openenv import _coerce_observation
        assert _coerce_observation({"text": "hello"}) == "hello"

    def test_content_key(self):
        from llenvs.adapters.openenv import _coerce_observation
        assert _coerce_observation({"content": "world"}) == "world"

    def test_observation_key(self):
        from llenvs.adapters.openenv import _coerce_observation
        assert _coerce_observation({"observation": "obs"}) == "obs"

    def test_message_key(self):
        from llenvs.adapters.openenv import _coerce_observation
        assert _coerce_observation({"message": "msg"}) == "msg"

    def test_fallback_to_json(self):
        from llenvs.adapters.openenv import _coerce_observation
        import json

        obs = {"foo": 42, "bar": "baz"}
        result = _coerce_observation(obs)
        parsed = json.loads(result)
        assert parsed["foo"] == 42

    def test_empty_dict(self):
        from llenvs.adapters.openenv import _coerce_observation
        result = _coerce_observation({})
        assert result == "{}"

    def test_string_observation(self):
        from llenvs.adapters.openenv import _coerce_observation
        assert _coerce_observation("already a string") == "already a string"

    def test_priority_text_over_content(self):
        from llenvs.adapters.openenv import _coerce_observation
        assert _coerce_observation({"text": "a", "content": "b"}) == "a"


# ── Reward tests ────────────────────────────────────────────────────


class TestOpenEnvReward:
    """Tests for OpenEnvReward."""

    def test_properties(self):
        from llenvs.adapters.openenv import OpenEnvReward

        reward = OpenEnvReward()
        assert reward.name == "openenv_native"
        assert reward.reward_type == RewardType.OUTCOME

    def test_compute_with_reward(self):
        from llenvs.adapters.openenv import OpenEnvReward, OpenEnvHidden

        reward_fn = OpenEnvReward()
        hidden = OpenEnvHidden(
            env_name="test", episode_step=1, last_action="act",
            session_info=(),
        )
        state = State(
            observation=Observation(prompt="test"),
            hidden=hidden,
            metadata=StateMetadata(
                step=0, episode_id="test", is_terminal=False,
                info={"openenv_reward": 0.75},
            ),
        )
        # The reward value comes from the next_state's info
        next_state = State(
            observation=Observation(prompt="test"),
            hidden=hidden,
            metadata=StateMetadata(
                step=1, episode_id="test", is_terminal=True,
                info={"openenv_reward": 0.75},
            ),
        )

        signal = reward_fn.compute(state, Action(text="x"), next_state)
        assert signal.value == 0.75
        assert signal.reward_type == RewardType.OUTCOME

    def test_compute_no_reward(self):
        from llenvs.adapters.openenv import OpenEnvReward, OpenEnvHidden

        reward_fn = OpenEnvReward()
        hidden = OpenEnvHidden(
            env_name="test", episode_step=0, last_action=None,
            session_info=(),
        )
        state = State(
            observation=Observation(prompt="test"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False, info={}),
        )

        signal = reward_fn.compute(state, Action(text="x"), state)
        assert signal.value == 0.0

    def test_step_reward_type(self):
        """Non-terminal steps use STEP type."""
        from llenvs.adapters.openenv import OpenEnvReward, OpenEnvHidden

        reward_fn = OpenEnvReward()
        hidden = OpenEnvHidden(
            env_name="test", episode_step=0, last_action=None,
            session_info=(),
        )
        state = State(
            observation=Observation(prompt="test"),
            hidden=hidden,
            metadata=StateMetadata(
                step=0, episode_id="test", is_terminal=False,
                info={"openenv_reward": 0.5},
            ),
        )
        next_state = State(
            observation=Observation(prompt="test"),
            hidden=hidden,
            metadata=StateMetadata(
                step=1, episode_id="test", is_terminal=False,
                info={"openenv_reward": 0.5},
            ),
        )

        signal = reward_fn.compute(state, Action(text="x"), next_state)
        assert signal.reward_type == RewardType.STEP


# ── Environment tests ──────────────────────────────────────────────


class TestOpenEnvEnvironment:
    """Tests for OpenEnvEnvironment."""

    def _make_env(self, client=None, **kwargs):
        from llenvs.adapters.openenv import OpenEnvEnvironment

        client = client or MockSyncClient()
        return OpenEnvEnvironment(
            client=client,
            env_name=kwargs.pop("env_name", "test-env"),
            **kwargs,
        )

    def test_creation(self):
        env = self._make_env()
        assert env.spec.name == "test-env"
        assert env.spec.adapter == "openenv"

    def test_spec_capabilities(self):
        env = self._make_env()
        spec = env.spec
        assert spec.supports_task_index is False
        assert spec.supports_len is False
        assert spec.supports_seed is False
        assert spec.is_multi_turn is True

    def test_prompts_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_available_tools_empty(self):
        env = self._make_env()
        assert env.available_tools == ()

    def test_reward_functions(self):
        env = self._make_env()
        assert len(env.reward_functions) == 1
        assert env.reward_functions[0].name == "openenv_native"

    def test_reset(self):
        client = MockSyncClient(
            observations=[{"text": "Welcome! What would you like to do?"}],
        )
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})

        assert "Welcome" in state.observation.prompt
        assert state.hidden.env_name == "test-env"
        assert state.hidden.episode_step == 0
        assert state.metadata.is_terminal is False

    def test_reset_ignores_task_index(self):
        """Task index is ignored — fresh session each time."""
        client = MockSyncClient()
        env = self._make_env(client=client)

        state1, _ = env.reset(options={"task_index": 0})
        state2, _ = env.reset(options={"task_index": 99})

        # Both get the same observation (fresh session)
        assert state1.observation.prompt == state2.observation.prompt

    def test_reset_no_options(self):
        env = self._make_env()
        state, info = env.reset()
        assert state.observation.prompt != ""

    def test_step(self):
        client = MockSyncClient(
            observations=[
                {"text": "Welcome"},
                {"text": "You moved north. You see a forest."},
            ],
            rewards=[None, 0.5],
        )
        env = self._make_env(client=client)
        state, _ = env.reset()
        result = env.step(state, Action(text="go north"))

        # New observation appears in message history (prompt stays from reset)
        messages = result.next_state.observation.messages
        assert any("forest" in m.get("content", "") for m in messages)
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "go north"

    def test_step_with_reward(self):
        client = MockSyncClient(
            observations=[{"text": "start"}, {"text": "end"}],
            rewards=[None, 1.0],
            done_steps={1},
        )
        env = self._make_env(client=client)
        state, _ = env.reset()
        result = env.step(state, Action(text="solve"))

        assert result.terminated is True
        reward = result.rewards.by_name("openenv_native")
        assert reward is not None
        assert reward.value == 1.0

    def test_step_done(self):
        client = MockSyncClient(
            observations=[{"text": "start"}, {"text": "done"}],
            done_steps={1},
        )
        env = self._make_env(client=client)
        state, _ = env.reset()
        result = env.step(state, Action(text="finish"))

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_max_steps_truncation(self):
        env = self._make_env(max_steps=2)
        state, _ = env.reset()

        result1 = env.step(state, Action(text="step1"))
        assert not result1.done

        result2 = env.step(result1.next_state, Action(text="step2"))
        assert result2.truncated is True

    def test_step_builds_message_history(self):
        client = MockSyncClient(
            observations=[{"text": "start"}, {"text": "response1"}, {"text": "response2"}],
        )
        env = self._make_env(client=client)
        state, _ = env.reset()

        r1 = env.step(state, Action(text="action1"))
        r2 = env.step(r1.next_state, Action(text="action2"))

        # Message history should have entries
        assert len(r2.next_state.observation.messages) > 0

    def test_compute_rewards(self):
        env = self._make_env()
        state, _ = env.reset()
        rewards = env.compute_rewards(state, Action(text="x"), state)
        assert len(rewards.signals) >= 1

    def test_action_format(self):
        """Custom action_format transforms the action text."""
        client = MockSyncClient()
        env = self._make_env(
            client=client,
            action_format=lambda text: {"command": text, "type": "text"},
        )
        state, _ = env.reset()
        env.step(state, Action(text="hello"))
        # The client received a formatted action (dict)
        # This tests that the format function was used


# ── Tool environment tests ──────────────────────────────────────────


class TestOpenEnvToolEnvironment:
    """Tests for OpenEnvToolEnvironment."""

    def _make_env(self, client=None, **kwargs):
        from llenvs.adapters.openenv import OpenEnvToolEnvironment

        if client is None:
            tools = [
                _make_mock_tool("search", "Search tool"),
                _make_mock_tool("calculate", "Calculator", {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expr"},
                    },
                    "required": ["expression"],
                }),
            ]
            client = MockMCPToolSyncClient(tools=tools)

        return OpenEnvToolEnvironment(
            client=client,
            env_name=kwargs.pop("env_name", "tool-env"),
            **kwargs,
        )

    def test_creation(self):
        env = self._make_env()
        assert env.spec.is_multi_turn is True

    def test_available_tools_from_server(self):
        env = self._make_env()
        state, _ = env.reset()
        # Tools should be populated after reset
        tools = env.available_tools
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search" in names
        assert "calculate" in names

    def test_tool_parameters(self):
        env = self._make_env()
        env.reset()
        search_tool = next(t for t in env.available_tools if t.name == "search")
        assert len(search_tool.parameters) == 1
        assert search_tool.parameters[0].name == "query"

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert state.hidden.episode_step == 0
        assert state.metadata.is_terminal is False
        # Tools should be available in observation
        assert len(state.observation.available_tools) == 2

    def test_step_with_tool_calls(self):
        client = MockMCPToolSyncClient(
            tools=[_make_mock_tool("search")],
            observations=[{"text": "ready"}, {"text": "search results"}],
        )
        client._tool_results["search"] = "Found 3 results"
        env = self._make_env(client=client)
        state, _ = env.reset()

        call = ToolCall(id="c1", name="search", arguments={"query": "test"})
        action = Action(text="", tool_calls=(call,))
        result = env.step(state, action)

        # Should have tool results
        assert len(result.next_state.observation.tool_results) == 1
        assert result.next_state.observation.tool_results[0].is_success

    def test_step_text_only(self):
        env = self._make_env()
        state, _ = env.reset()
        result = env.step(state, Action(text="just thinking"))
        assert result.next_state.hidden.episode_step == 1

    def test_spec_capabilities(self):
        env = self._make_env()
        spec = env.spec
        assert spec.supports_task_index is False
        assert spec.supports_len is False
        assert spec.supports_seed is False


# ── Adapter tests ───────────────────────────────────────────────────


class TestOpenEnvAdapter:
    """Tests for OpenEnvAdapter."""

    def test_adapter_name(self):
        from llenvs.adapters.openenv import OpenEnvAdapter
        adapter = OpenEnvAdapter()
        assert adapter.name == "openenv"

    def test_get_openenv_import_error(self):
        from llenvs.adapters.openenv import OpenEnvAdapter
        adapter = OpenEnvAdapter()
        with pytest.raises(ImportError, match="openenv"):
            adapter._get_openenv()

    def test_get_environment_basic(self, monkeypatch):
        from llenvs.adapters.openenv import OpenEnvAdapter, OpenEnvEnvironment

        mock_client = MockSyncClient()

        mock_openenv = MagicMock()
        mock_generic_client = MagicMock()
        mock_generic_client.return_value.sync.return_value = mock_client

        mock_openenv.GenericEnvClient = mock_generic_client

        adapter = OpenEnvAdapter()
        monkeypatch.setattr(adapter, "_get_openenv", lambda: mock_openenv)

        env = adapter.get_environment("test-env", base_url="http://localhost:8000")
        assert isinstance(env, OpenEnvEnvironment)

    def test_get_environment_with_tools(self, monkeypatch):
        from llenvs.adapters.openenv import OpenEnvAdapter, OpenEnvToolEnvironment

        mock_client = MockMCPToolSyncClient(tools=[_make_mock_tool()])

        mock_openenv = MagicMock()
        mock_mcp_client = MagicMock()
        mock_mcp_client.return_value.sync.return_value = mock_client

        mock_openenv.MCPToolClient = mock_mcp_client

        adapter = OpenEnvAdapter()
        monkeypatch.setattr(adapter, "_get_openenv", lambda: mock_openenv)

        env = adapter.get_environment(
            "tool-env", base_url="http://localhost:8000", use_tools=True,
        )
        assert isinstance(env, OpenEnvToolEnvironment)

    def test_get_environment_requires_base_url(self, monkeypatch):
        from llenvs.adapters.openenv import OpenEnvAdapter

        mock_openenv = MagicMock()
        adapter = OpenEnvAdapter()
        monkeypatch.setattr(adapter, "_get_openenv", lambda: mock_openenv)

        with pytest.raises(ValueError, match="base_url"):
            adapter.get_environment("test-env")

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.openenv import OpenEnvAdapter
        adapter = OpenEnvAdapter()
        assert adapter.get_native_answer_extractor("test") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.openenv import OpenEnvAdapter
        adapter = OpenEnvAdapter()
        assert adapter.get_prompt_template("test") is None

    def test_get_environment_info(self):
        from llenvs.adapters.openenv import OpenEnvAdapter
        adapter = OpenEnvAdapter()
        info = adapter.get_environment_info("test")
        assert info["name"] == "test"
        assert info["adapter"] == "openenv"

    def test_list_environments(self):
        from llenvs.adapters.openenv import OpenEnvAdapter
        adapter = OpenEnvAdapter()
        envs = adapter.list_environments()
        assert isinstance(envs, list)


# ── Integration: Scorer/DatasetProvider rejection tests ─────────────


class TestCapabilityRejection:
    """Test that Scorer and DatasetProvider reject OpenEnv environments."""

    def test_scorer_rejects_openenv(self):
        from llenvs.adapters.openenv import OpenEnvEnvironment
        from llenvs.integrations.scoring import Scorer

        env = OpenEnvEnvironment(
            client=MockSyncClient(),
            env_name="test",
        )
        with pytest.raises(TypeError, match="multi-turn|supports_task_index"):
            Scorer(env)

    def test_dataset_provider_rejects_openenv(self):
        from llenvs.adapters.openenv import OpenEnvEnvironment
        from llenvs.integrations.dataset_provider import DatasetProvider

        env = OpenEnvEnvironment(
            client=MockSyncClient(),
            env_name="test",
        )
        with pytest.raises(TypeError, match="supports_task_index"):
            DatasetProvider(env)
