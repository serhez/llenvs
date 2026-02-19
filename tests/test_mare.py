"""Tests for the MARE (Meta Agents Research Environments) adapter."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from llenvs.core.extraction import TagBasedExtractor
from llenvs.core.reward import FormatReward, RewardType
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.tools import ToolCall, ToolParameterType

# ── Mock ARE objects ─────────────────────────────────────────────


@dataclass
class MockARETool:
    """Mock ARE Tool object."""

    name: str
    description: str
    inputs: dict
    output_type: str = "string"

    def forward(self, **kwargs):
        return f"Result for {self.name}({kwargs})"


@dataclass
class MockAREApp:
    """Mock ARE App with tools."""

    name: str
    tools: list = field(default_factory=list)

    def get_tools(self):
        return self.tools


@dataclass
class MockARENotification:
    """Mock ARE notification."""

    source: str
    content: str
    timestamp: float = 0.0

    def __str__(self):
        return f"[{self.source}] {self.content}"


class MockAREEnvironment:
    """Mock ARE Environment with start/stop/tick."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.tick_count = 0
        self._notifications: list[MockARENotification] = []

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def tick(self):
        self.tick_count += 1

    def get_pending_notifications(self):
        notifs = list(self._notifications)
        self._notifications.clear()
        return notifs

    def add_notification(self, notif: MockARENotification):
        self._notifications.append(notif)


@dataclass
class MockScenario:
    """Mock ARE Scenario."""

    id: str
    prompt: str
    task_description: str
    seed: int = 42
    apps: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    _initialized: bool = False

    async def initialize(self):
        self._initialized = True

    def get_user_tools(self):
        return self.tools

    def get_apps(self):
        return self.apps

    async def validate(self, write_actions):
        """Validate write actions against oracle annotations."""
        if not self.validation:
            return {"score": 0.0, "details": {}}
        expected = self.validation.get("expected_actions", [])
        if not expected:
            return {"score": 0.0, "details": {}}
        matched = sum(1 for a in write_actions if a in expected)
        score = matched / len(expected) if expected else 0.0
        return {"score": score, "details": {"matched": matched, "total": len(expected)}}


def _make_mock_tool(name="send_email", description="Send an email", inputs=None):
    """Create a mock ARE Tool."""
    if inputs is None:
        inputs = {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
        }
    return MockARETool(name=name, description=description, inputs=inputs)


def _make_scenario(
    scenario_id="scenario_001",
    prompt="You are an assistant helping manage tasks.",
    task_description="Send an email to alice@example.com about the meeting.",
    num_tools=3,
    seed=42,
    validation=None,
):
    """Create a mock Scenario."""
    tools = [
        _make_mock_tool("send_email", "Send an email"),
        _make_mock_tool(
            "search_contacts",
            "Search contacts",
            {"query": {"type": "string", "description": "Search query"}},
        ),
        _make_mock_tool(
            "check_calendar",
            "Check calendar events",
            {"date": {"type": "string", "description": "Date to check"}},
        ),
    ][:num_tools]
    return MockScenario(
        id=scenario_id,
        prompt=prompt,
        task_description=task_description,
        seed=seed,
        tools=tools,
        validation=validation or {},
    )


def _make_scenario_loader(scenarios=None, num_scenarios=5):
    """Create a list of mock scenarios."""
    if scenarios is not None:
        return scenarios
    return [
        _make_scenario(
            scenario_id=f"scenario_{i:03d}",
            task_description=f"Task {i}: Complete the objective.",
            seed=42 + i,
        )
        for i in range(num_scenarios)
    ]


# ── TestMAREToolConversion ───────────────────────────────────────


class TestMAREToolConversion:
    def test_basic_conversion(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        tools = [_make_mock_tool("send_email", "Send an email")]
        defs = _mare_tools_to_definitions(tools)

        assert len(defs) == 1
        assert defs[0].name == "send_email"
        assert defs[0].description == "Send an email"

    def test_empty_tools(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        assert _mare_tools_to_definitions([]) == ()

    def test_parameters_converted(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        tools = [_make_mock_tool("send_email", "Send")]
        defs = _mare_tools_to_definitions(tools)

        params = {p.name: p for p in defs[0].parameters}
        assert "to" in params
        assert "subject" in params
        assert "body" in params
        assert params["to"].type == ToolParameterType.STRING

    def test_multiple_tools(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        tools = [
            _make_mock_tool("send_email", "Send"),
            _make_mock_tool(
                "search",
                "Search",
                {"query": {"type": "string", "description": "Q"}},
            ),
        ]
        defs = _mare_tools_to_definitions(tools)
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"send_email", "search"}

    def test_type_mapping(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        tools = [
            _make_mock_tool(
                "test_types",
                "Test type mapping",
                {
                    "s": {"type": "string", "description": "str"},
                    "n": {"type": "number", "description": "num"},
                    "i": {"type": "integer", "description": "int"},
                    "b": {"type": "boolean", "description": "bool"},
                    "a": {"type": "array", "description": "arr"},
                    "o": {"type": "object", "description": "obj"},
                },
            )
        ]
        defs = _mare_tools_to_definitions(tools)
        params = {p.name: p for p in defs[0].parameters}
        assert params["s"].type == ToolParameterType.STRING
        assert params["n"].type == ToolParameterType.NUMBER
        assert params["i"].type == ToolParameterType.INTEGER
        assert params["b"].type == ToolParameterType.BOOLEAN
        assert params["a"].type == ToolParameterType.ARRAY
        assert params["o"].type == ToolParameterType.OBJECT

    def test_unknown_type_defaults_to_string(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        tools = [
            _make_mock_tool(
                "test",
                "Test",
                {"x": {"type": "unknown_type", "description": "mystery"}},
            )
        ]
        defs = _mare_tools_to_definitions(tools)
        params = {p.name: p for p in defs[0].parameters}
        assert params["x"].type == ToolParameterType.STRING

    def test_missing_description_defaults(self):
        from llenvs.adapters.mare import _mare_tools_to_definitions

        tools = [
            _make_mock_tool(
                "test",
                "Test",
                {"x": {"type": "string"}},  # No description
            )
        ]
        defs = _mare_tools_to_definitions(tools)
        params = {p.name: p for p in defs[0].parameters}
        assert params["x"].description == ""


# ── TestMAREHidden ───────────────────────────────────────────────


class TestMAREHidden:
    def test_creation(self):
        from llenvs.adapters.mare import MAREHidden

        hidden = MAREHidden(
            task_index=0,
            scenario_id="scenario_001",
            episode_step=0,
        )
        assert hidden.task_index == 0
        assert hidden.scenario_id == "scenario_001"
        assert hidden.episode_step == 0

    def test_frozen(self):
        from llenvs.adapters.mare import MAREHidden

        hidden = MAREHidden(task_index=0, scenario_id="s1")
        with pytest.raises(AttributeError):
            hidden.task_index = 1  # type: ignore

    def test_defaults(self):
        from llenvs.adapters.mare import MAREHidden

        hidden = MAREHidden(task_index=0, scenario_id="s1")
        assert hidden.episode_step == 0
        assert hidden.last_action is None
        assert hidden.notifications == ()
        assert hidden.write_actions == ()

    def test_full_creation(self):
        from llenvs.adapters.mare import MAREHidden

        hidden = MAREHidden(
            task_index=2,
            scenario_id="scenario_005",
            episode_step=3,
            last_action="send_email",
            notifications=("New email received",),
            write_actions=({"action": "send_email", "to": "alice"},),
        )
        assert hidden.task_index == 2
        assert hidden.scenario_id == "scenario_005"
        assert hidden.episode_step == 3
        assert hidden.last_action == "send_email"
        assert len(hidden.notifications) == 1
        assert len(hidden.write_actions) == 1


# ── TestMAREReward ───────────────────────────────────────────────


class TestMAREReward:
    def _make_states(self, scenario_id="s1", is_terminal=True, write_actions=()):
        from llenvs.adapters.mare import MAREHidden

        hidden = MAREHidden(
            task_index=0,
            scenario_id=scenario_id,
            write_actions=write_actions,
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
        from llenvs.adapters.mare import MAREReward

        reward = MAREReward()
        assert reward.name == "mare"

    def test_reward_type(self):
        from llenvs.adapters.mare import MAREReward

        reward = MAREReward()
        assert reward.reward_type == RewardType.OUTCOME

    def test_intermediate_none(self):
        from llenvs.adapters.mare import MAREReward

        reward = MAREReward()
        state, action, next_state = self._make_states(is_terminal=False)
        signal = reward.compute(state, action, next_state)
        assert signal.reward is None
        assert signal.reward_type == RewardType.STEP

    def test_terminal_no_validator(self):
        from llenvs.adapters.mare import MAREReward

        reward = MAREReward()
        state, action, next_state = self._make_states(is_terminal=True)
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.0
        assert signal.reward_type == RewardType.OUTCOME

    def test_terminal_with_validator(self):
        from llenvs.adapters.mare import MAREReward

        async def validator(write_actions):
            return {"score": 0.8, "details": {"matched": 4, "total": 5}}

        reward = MAREReward(validator=validator)
        state, action, next_state = self._make_states(
            is_terminal=True,
            write_actions=({"action": "send_email"},),
        )
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.8
        assert signal.reward_type == RewardType.OUTCOME

    def test_terminal_validator_error_fallback(self):
        from llenvs.adapters.mare import MAREReward

        async def failing_validator(write_actions):
            raise RuntimeError("Validation failed")

        reward = MAREReward(validator=failing_validator)
        state, action, next_state = self._make_states(is_terminal=True)
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.0
        assert signal.reward_type == RewardType.OUTCOME


# ── TestMAREEnvironment ──────────────────────────────────────────


class TestMAREEnvironment:
    def _make_env(self, scenarios=None, **kwargs):
        from llenvs.adapters.mare import MAREEnvironment

        sc = scenarios or _make_scenario_loader()
        return MAREEnvironment(scenarios=sc, **kwargs)

    def test_spec(self):
        env = self._make_env()
        spec = env.spec
        assert spec.name == "mare"
        assert spec.adapter == "mare"
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is True

    def test_spec_max_steps(self):
        env = self._make_env(max_steps=10)
        assert env.spec.max_steps == 10

    def test_spec_default_max_steps(self):
        env = self._make_env()
        assert env.spec.max_steps is None

    def test_len(self):
        env = self._make_env()
        assert len(env) == 5

    def test_len_custom(self):
        scenarios = _make_scenario_loader(num_scenarios=10)
        env = self._make_env(scenarios=scenarios)
        assert len(env) == 10

    def test_prompts_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_reward_functions(self):
        env = self._make_env()
        assert len(env.reward_functions) >= 1
        assert env.reward_functions[0].name == "mare"

    def test_extra_rewards(self):
        extractor = TagBasedExtractor()
        format_reward = FormatReward(extractor)
        env = self._make_env(extra_rewards=(format_reward,))
        names = [r.name for r in env.reward_functions]
        assert "mare" in names
        assert "format" in names

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.scenario_id == "scenario_000"
        assert state.hidden.episode_step == 0
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0

    def test_reset_prompt_from_scenario(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 2})
        assert "Task 2" in state.observation.prompt

    def test_reset_tools_available(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert len(state.observation.available_tools) == 3
        assert info["num_tools"] == 3
        tool_names = {t.name for t in state.observation.available_tools}
        assert "send_email" in tool_names

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
                    name="send_email",
                    arguments={"to": "alice@example.com", "subject": "hi", "body": "hello"},
                ),
            )
        )
        result = env.step(state, action)

        assert result.terminated is False
        assert result.next_state.hidden.episode_step == 1
        assert len(result.next_state.observation.messages) > 0

    def test_step_text_only(self):
        """Text-only action signals agent is done."""
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="I have completed all tasks.")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_max_steps_truncation(self):
        env = self._make_env(max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1", name="send_email", arguments={"to": "a", "subject": "b", "body": "c"}
                ),
            )
        )
        result = env.step(state, action)

        assert result.truncated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_message_history_grows(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action1 = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1", name="send_email", arguments={"to": "a", "subject": "b", "body": "c"}
                ),
            )
        )
        result1 = env.step(state, action1)
        msg_count_1 = len(result1.next_state.observation.messages)
        assert msg_count_1 > 0

        action2 = Action(
            tool_calls=(ToolCall(id="tc-2", name="search_contacts", arguments={"query": "alice"}),)
        )
        result2 = env.step(result1.next_state, action2)
        msg_count_2 = len(result2.next_state.observation.messages)
        assert msg_count_2 > msg_count_1

    def test_step_stale_state_raises(self):
        env = self._make_env(max_steps=10)
        state_0, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1", name="send_email", arguments={"to": "a", "subject": "b", "body": "c"}
                ),
            )
        )
        env.step(state_0, action)

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, action)

    def test_step_no_active_env(self):
        env = self._make_env()
        hidden = MagicMock()
        hidden.episode_step = 0
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )

        with pytest.raises(RuntimeError, match="No active"):
            env.step(state, Action(text="x"))

    def test_step_rewards_computed(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="Done")
        result = env.step(state, action)

        sig = result.rewards.by_name("mare")
        assert sig is not None
        assert sig.reward_type == RewardType.OUTCOME

    def test_step_tool_results_in_info(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1", name="send_email", arguments={"to": "a", "subject": "b", "body": "c"}
                ),
            )
        )
        result = env.step(state, action)
        assert "tool_results" in result.info

    def test_step_notifications_in_observation(self):
        """Notifications from the environment should appear in the observation."""
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        # Inject a notification into the active environment
        env._active_are_env.add_notification(
            MockARENotification(source="email", content="New email from Bob")
        )

        action = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1", name="send_email", arguments={"to": "a", "subject": "b", "body": "c"}
                ),
            )
        )
        result = env.step(state, action)

        # Notifications should be in the hidden state
        assert len(result.next_state.hidden.notifications) > 0

    def test_step_write_action_tracking(self):
        """Write actions should be tracked in hidden state."""
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        # send_email is a write action
        action = Action(
            tool_calls=(
                ToolCall(
                    id="tc-1", name="send_email", arguments={"to": "a", "subject": "b", "body": "c"}
                ),
            )
        )
        result = env.step(state, action)

        assert len(result.next_state.hidden.write_actions) >= 1

    def test_close(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        env.close()
        assert env._active_are_env is None

    def test_close_without_reset(self):
        env = self._make_env()
        env.close()  # Should not raise

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


# ── TestMAREAdapter ──────────────────────────────────────────────


class TestMAREAdapter:
    def test_name(self):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        assert adapter.name == "mare"

    def test_list_environments(self):
        from llenvs.adapters.mare import MARE_CAPABILITIES, MAREAdapter

        adapter = MAREAdapter()
        envs = adapter.list_environments()
        assert "mare" in envs
        for cap in MARE_CAPABILITIES:
            assert f"mare:{cap}" in envs

    def test_import_error(self):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        with pytest.raises(ImportError, match="meta-agents-research-environments"):
            adapter._get_mare()

    def test_get_environment_with_scenarios(self, monkeypatch):
        from llenvs.adapters.mare import MAREAdapter, MAREEnvironment

        adapter = MAREAdapter()
        monkeypatch.setattr(adapter, "_get_mare", lambda: MagicMock())

        scenarios = _make_scenario_loader()
        env = adapter.get_environment("mare", scenarios=scenarios)
        assert isinstance(env, MAREEnvironment)

    def test_get_environment_requires_scenarios(self, monkeypatch):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        monkeypatch.setattr(adapter, "_get_mare", lambda: MagicMock())

        with pytest.raises(ValueError, match="scenarios"):
            adapter.get_environment("mare")

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        assert adapter.get_native_answer_extractor("mare") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        assert adapter.get_prompt_template("mare") is None

    def test_get_environment_info(self):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        info = adapter.get_environment_info("mare")
        assert info["name"] == "mare"
        assert info["adapter"] == "mare"

    def test_max_steps_passed_through(self, monkeypatch):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        monkeypatch.setattr(adapter, "_get_mare", lambda: MagicMock())

        scenarios = _make_scenario_loader()
        env = adapter.get_environment("mare", scenarios=scenarios, max_steps=20)
        assert env._max_steps == 20

    def test_capability_filter(self, monkeypatch):
        from llenvs.adapters.mare import MAREAdapter

        adapter = MAREAdapter()
        monkeypatch.setattr(adapter, "_get_mare", lambda: MagicMock())

        scenarios = _make_scenario_loader()
        env = adapter.get_environment("mare:execution", scenarios=scenarios)
        assert isinstance(
            env, adapter.get_environment.__func__(adapter, "mare", scenarios=scenarios).__class__
        )


# ── TestRunAsync ─────────────────────────────────────────────────


class TestRunAsync:
    def test_run_async_basic(self):
        from llenvs.core.async_utils import run_async

        async def coro():
            return 42

        assert run_async(coro()) == 42

    def test_run_async_with_args(self):
        from llenvs.core.async_utils import run_async

        async def add(a, b):
            return a + b

        assert run_async(add(3, 4)) == 7
