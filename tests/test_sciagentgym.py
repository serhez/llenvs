"""Tests for the SciAgentGYM adapter."""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from llenvs.core.extraction import TagBasedExtractor
from llenvs.core.reward import FormatReward, RewardType
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.tools import ToolCall

# ── Mock SciAgentGYM objects ─────────────────────────────────────


def _make_query(
    task_id=1,
    question="What is the threshold energy?",
    answer="2.6e5 GeV",
    subject="Physics",
    topic="Particle Physics",
    tools=None,
):
    """Create a mock SciAgentGYM test case dict."""
    if tools is None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_threshold",
                    "description": "Analyze threshold conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "energy": {"type": "number", "description": "Energy in eV"},
                            "particle": {"type": "string", "description": "Particle type"},
                        },
                        "required": ["energy"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "visualize_spectrum",
                    "description": "Plot energy spectrum",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Energy data points"},
                        },
                        "required": ["data"],
                    },
                },
            },
        ]
    return {
        "id": task_id,
        "question": question,
        "answer": answer,
        "metadata": {
            "subject": subject,
            "topic": topic,
            "golden_answer": {},
            "tool_expected": [t["function"]["name"] for t in tools],
        },
        "usage_tool_protocol": tools,
    }


def _make_dataset(num_tasks=5, subject="Physics"):
    """Create a list of mock test case dicts."""
    return [
        _make_query(
            task_id=i,
            question=f"Question {i}",
            answer=f"Answer {i}",
            subject=subject,
        )
        for i in range(num_tasks)
    ]


@dataclass
class MockObservation:
    """Mock SciAgentGYM Observation."""

    source: str
    observation: str

    def __str__(self):
        return self.observation


@dataclass
class MockStepOutput:
    """Mock SciAgentGYM StepOutput."""

    observation: MockObservation | str
    done: bool = False
    info: dict | None = None


@dataclass
class MockSciToolCall:
    """Mock SciAgentGYM ToolCall."""

    id: str
    name: str
    arguments: dict


def _make_mock_env(tools=None):
    """Create a mock MinimalSciEnv."""
    env = MagicMock()
    env._tools = {}
    env._step_count = 0

    if tools:
        for t in tools:
            env._tools[t] = MagicMock()

    def reset():
        env._step_count = 0
        return {"observation": "MinimalSciEnv reset.", "available_tools": list(env._tools.keys())}

    def step(action):
        env._step_count += 1
        obs = MockObservation(source=action.name, observation=json.dumps({"result": "computed"}))
        return MockStepOutput(
            observation=obs,
            done=False,
            info={"step": env._step_count, "tool_name": action.name, "tool_call_id": action.id},
        )

    env.reset = reset
    env.step = step
    return env


def _mock_prepare_env_from_query(query_data):
    """Mock prepare_env_from_query returning (env, tools, schema, registry)."""
    tool_protocols = query_data.get("usage_tool_protocol", [])
    tool_names = [t["function"]["name"] for t in tool_protocols]
    mock_env = _make_mock_env(tools=tool_names)
    mock_env.reset()
    tool_instances = [MagicMock(name=n) for n in tool_names]
    tool_registry = {n: MagicMock() for n in tool_names}
    return mock_env, tool_instances, tool_protocols, tool_registry


# ── TestSciAgentGymHidden ────────────────────────────────────────


class TestSciAgentGymHidden:
    def test_creation(self):
        from llenvs.adapters.sciagentgym import SciAgentGymHidden

        hidden = SciAgentGymHidden(
            task_index=0,
            task_id=1,
            question="What is X?",
            gold_answer="42",
            subject="Physics",
        )
        assert hidden.task_index == 0
        assert hidden.task_id == 1
        assert hidden.question == "What is X?"
        assert hidden.gold_answer == "42"
        assert hidden.subject == "Physics"

    def test_frozen(self):
        from llenvs.adapters.sciagentgym import SciAgentGymHidden

        hidden = SciAgentGymHidden(
            task_index=0, task_id=1, question="Q", gold_answer="A", subject="Physics"
        )
        with pytest.raises(AttributeError):
            hidden.task_index = 1  # type: ignore

    def test_defaults(self):
        from llenvs.adapters.sciagentgym import SciAgentGymHidden

        hidden = SciAgentGymHidden(
            task_index=0, task_id=1, question="Q", gold_answer="A", subject="Physics"
        )
        assert hidden.episode_step == 0
        assert hidden.last_action is None
        assert hidden.tool_names_used == ()

    def test_full_creation(self):
        from llenvs.adapters.sciagentgym import SciAgentGymHidden

        hidden = SciAgentGymHidden(
            task_index=2,
            task_id=5,
            question="Complex question",
            gold_answer="complex answer",
            subject="Chemistry",
            episode_step=3,
            last_action="used tool",
            tool_names_used=("tool_a", "tool_b"),
        )
        assert hidden.task_index == 2
        assert hidden.task_id == 5
        assert hidden.subject == "Chemistry"
        assert hidden.episode_step == 3
        assert hidden.last_action == "used tool"
        assert hidden.tool_names_used == ("tool_a", "tool_b")

    def test_gold_answer_storage(self):
        from llenvs.adapters.sciagentgym import SciAgentGymHidden

        hidden = SciAgentGymHidden(
            task_index=0,
            task_id=1,
            question="Q",
            gold_answer="2.6*1e5 GeV",
            subject="Physics",
        )
        assert hidden.gold_answer == "2.6*1e5 GeV"


# ── TestSciAgentGymReward ────────────────────────────────────────


class TestSciAgentGymReward:
    def _make_states(self, gold_answer="42", response_text="\\boxed{42}", is_terminal=True):
        from llenvs.adapters.sciagentgym import SciAgentGymHidden

        hidden = SciAgentGymHidden(
            task_index=0, task_id=1, question="Q", gold_answer=gold_answer, subject="Physics"
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
        action = Action(text=response_text)
        return state, action, next_state

    def test_name(self):
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        assert reward.name == "sciagentgym"

    def test_reward_type(self):
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        assert reward.reward_type == RewardType.OUTCOME

    def test_intermediate_none(self):
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        state, action, next_state = self._make_states(is_terminal=False)
        signal = reward.compute(state, action, next_state)
        assert signal.reward is None
        assert signal.reward_type == RewardType.STEP

    def test_terminal_correct_fallback(self):
        """Test terminal scoring with fallback (no SciAgentGYM evaluator)."""
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        state, action, next_state = self._make_states(gold_answer="42", response_text="\\boxed{42}")
        signal = reward.compute(state, action, next_state)
        assert signal.reward is not None
        assert signal.reward_type == RewardType.OUTCOME

    def test_terminal_no_boxed_content(self):
        """When there's no boxed answer, score should be 0."""
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        state, action, next_state = self._make_states(
            gold_answer="42", response_text="I think the answer is 42"
        )
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.0
        assert signal.reward_type == RewardType.OUTCOME

    def test_terminal_incorrect(self):
        """When boxed answer doesn't match gold, score should be 0."""
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        state, action, next_state = self._make_states(gold_answer="42", response_text="\\boxed{99}")
        signal = reward.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0

    def test_terminal_with_native_scoring(self):
        """Test scoring when SciAgentGYM evaluator is available."""
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        mock_evaluator = MagicMock()
        mock_evaluator.extract_boxed_answer.return_value = "42"
        mock_evaluator.calculate_answer_score.return_value = (1.0, "exact match", {})

        reward = SciAgentGymReward()
        state, action, next_state = self._make_states(gold_answer="42", response_text="\\boxed{42}")

        with patch.dict(
            "sys.modules",
            {"gym": MagicMock(), "gym.core": MagicMock(), "gym.core.evaluator": mock_evaluator},
        ):
            # Need to patch the import inside the compute method
            with patch("llenvs.adapters.sciagentgym._try_native_scoring") as mock_native:
                mock_native.return_value = (1.0, {"summary": "exact match", "details": {}})
                signal = reward.compute(state, action, next_state)
                assert signal.reward == 1.0

    def test_terminal_numeric_tolerance(self):
        """Test that numeric comparison handles close values."""
        from llenvs.adapters.sciagentgym import SciAgentGymReward

        reward = SciAgentGymReward()
        # Exact string match should work even in fallback
        state, action, next_state = self._make_states(
            gold_answer="3.14", response_text="\\boxed{3.14}"
        )
        signal = reward.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME

    def test_no_text_action(self):
        """When action has no text (tool calls only), and is terminal, score should be 0."""
        from llenvs.adapters.sciagentgym import SciAgentGymHidden, SciAgentGymReward

        reward = SciAgentGymReward()
        hidden = SciAgentGymHidden(
            task_index=0, task_id=1, question="Q", gold_answer="42", subject="Physics"
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="e1", is_terminal=False),
        )
        next_state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="e1", is_terminal=True),
        )
        action = Action(tool_calls=(ToolCall(id="tc-1", name="analyze", arguments={"x": 1}),))
        signal = reward.compute(state, action, next_state)
        assert signal.reward == 0.0


# ── TestSciAgentGymEnvironment ───────────────────────────────────


class TestSciAgentGymEnvironment:
    @pytest.fixture(autouse=True)
    def _mock_sciagentgym(self, monkeypatch):
        """Mock SciAgentGYM imports."""
        import sys

        mock_gym = MagicMock()
        mock_gym_core = MagicMock()
        mock_gym_core_tool_loader = MagicMock()
        mock_gym_core_tool_loader.prepare_env_from_query = _mock_prepare_env_from_query
        mock_gym_tool = MagicMock()
        mock_gym_tool.ToolCall = MockSciToolCall
        mock_gym_env = MagicMock()

        monkeypatch.setitem(sys.modules, "gym", mock_gym)
        monkeypatch.setitem(sys.modules, "gym.core", mock_gym_core)
        monkeypatch.setitem(sys.modules, "gym.core.tool_loader", mock_gym_core_tool_loader)
        monkeypatch.setitem(sys.modules, "gym.tool", mock_gym_tool)
        monkeypatch.setitem(sys.modules, "gym.env", mock_gym_env)

    def _make_env(self, dataset=None, **kwargs):
        from llenvs.adapters.sciagentgym import SciAgentGymEnvironment

        ds = dataset or _make_dataset()
        return SciAgentGymEnvironment(dataset=ds, **kwargs)

    def test_spec(self):
        env = self._make_env()
        spec = env.spec
        assert spec.name == "sciagentgym"
        assert spec.adapter == "sciagentgym"
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is False

    def test_spec_max_steps(self):
        env = self._make_env(max_steps=15)
        assert env.spec.max_steps == 15

    def test_spec_default_max_steps(self):
        env = self._make_env()
        assert env.spec.max_steps == 30

    def test_len(self):
        env = self._make_env()
        assert len(env) == 5

    def test_len_custom(self):
        ds = _make_dataset(num_tasks=10)
        env = self._make_env(dataset=ds)
        assert len(env) == 10

    def test_prompts_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_reward_functions(self):
        env = self._make_env()
        assert len(env.reward_functions) >= 1
        assert env.reward_functions[0].name == "sciagentgym"

    def test_extra_rewards(self):
        extractor = TagBasedExtractor()
        format_reward = FormatReward(extractor)
        env = self._make_env(extra_rewards=(format_reward,))
        names = [r.name for r in env.reward_functions]
        assert "sciagentgym" in names
        assert "format" in names

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.task_id == 0
        assert state.hidden.subject == "Physics"
        assert state.hidden.episode_step == 0
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0

    def test_reset_prompt_from_question(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 2})
        assert "Question 2" in state.observation.prompt

    def test_reset_tools_available(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert len(state.observation.available_tools) == 2
        assert info["num_tools"] == 2
        tool_names = {t.name for t in state.observation.available_tools}
        assert "analyze_threshold" in tool_names
        assert "visualize_spectrum" in tool_names

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

    def test_reset_gold_answer_in_hidden(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.gold_answer == "Answer 0"

    def test_step_tool_call(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="analyze_threshold", arguments={"energy": 1.0}),)
        )
        result = env.step(state, action)

        assert result.terminated is False
        assert result.next_state.hidden.episode_step == 1
        assert "analyze_threshold" in result.next_state.hidden.tool_names_used
        assert len(result.next_state.observation.messages) > 0

    def test_step_text_only_terminates(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="The answer is \\boxed{42}")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_max_steps_truncation(self):
        env = self._make_env(max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="analyze_threshold", arguments={"energy": 1.0}),)
        )
        result = env.step(state, action)

        assert result.truncated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_message_history_grows(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action1 = Action(
            tool_calls=(ToolCall(id="tc-1", name="analyze_threshold", arguments={"energy": 1.0}),)
        )
        result1 = env.step(state, action1)
        msg_count_1 = len(result1.next_state.observation.messages)
        assert msg_count_1 > 0

        action2 = Action(
            tool_calls=(ToolCall(id="tc-2", name="visualize_spectrum", arguments={"data": [1, 2]}),)
        )
        result2 = env.step(result1.next_state, action2)
        msg_count_2 = len(result2.next_state.observation.messages)
        assert msg_count_2 > msg_count_1

    def test_step_stale_state_raises(self):
        env = self._make_env(max_steps=10)
        state_0, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="analyze_threshold", arguments={"energy": 1.0}),)
        )
        env.step(state_0, action)

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, action)

    def test_step_no_active_env(self):
        from llenvs.adapters.sciagentgym import SciAgentGymEnvironment

        env = SciAgentGymEnvironment(dataset=_make_dataset())
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

        action = Action(text="The answer is \\boxed{Answer 0}")
        result = env.step(state, action)

        sig = result.rewards.by_name("sciagentgym")
        assert sig is not None
        assert sig.reward_type == RewardType.OUTCOME

    def test_step_multiple_tool_calls(self):
        env = self._make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(
                ToolCall(id="tc-1", name="analyze_threshold", arguments={"energy": 1.0}),
                ToolCall(id="tc-2", name="visualize_spectrum", arguments={"data": [1]}),
            )
        )
        result = env.step(state, action)

        assert result.next_state.hidden.episode_step == 1
        assert "analyze_threshold" in result.next_state.hidden.tool_names_used
        assert "visualize_spectrum" in result.next_state.hidden.tool_names_used

    def test_step_tool_results_in_info(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(
            tool_calls=(ToolCall(id="tc-1", name="analyze_threshold", arguments={"energy": 1.0}),)
        )
        result = env.step(state, action)
        assert "tool_results" in result.info

    def test_close(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        env.close()
        assert env._active_env is None

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
        rewards = env.compute_rewards(state, Action(text="\\boxed{x}"), next_state)
        assert len(rewards.signals) >= 1


# ── TestSciAgentGymAdapter ───────────────────────────────────────


class TestSciAgentGymAdapter:
    def test_name(self):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        assert adapter.name == "sciagentgym"

    def test_list_environments(self):
        from llenvs.adapters.sciagentgym import SCIAGENTGYM_SUBJECTS, SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        envs = adapter.list_environments()
        assert "sciagentgym" in envs
        for subject in SCIAGENTGYM_SUBJECTS:
            assert f"sciagentgym:{subject}" in envs

    def test_import_error(self):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        with pytest.raises(ImportError, match="SciAgentGYM"):
            adapter._get_sciagentgym()

    def test_get_environment_with_dataset(self, monkeypatch):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter, SciAgentGymEnvironment

        adapter = SciAgentGymAdapter()
        monkeypatch.setattr(adapter, "_get_sciagentgym", lambda: MagicMock())

        ds = _make_dataset()
        env = adapter.get_environment("sciagentgym", dataset=ds)
        assert isinstance(env, SciAgentGymEnvironment)

    def test_get_environment_with_data_path(self, monkeypatch, tmp_path):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter, SciAgentGymEnvironment

        adapter = SciAgentGymAdapter()
        monkeypatch.setattr(adapter, "_get_sciagentgym", lambda: MagicMock())

        # Write a test JSON file
        data = _make_dataset(num_tasks=3)
        data_file = tmp_path / "test_cases.json"
        data_file.write_text(json.dumps(data))

        env = adapter.get_environment("sciagentgym", data_path=str(data_file))
        assert isinstance(env, SciAgentGymEnvironment)
        assert len(env) == 3

    def test_get_environment_subject_filter(self, monkeypatch):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        monkeypatch.setattr(adapter, "_get_sciagentgym", lambda: MagicMock())

        ds = [
            _make_query(task_id=0, subject="Physics"),
            _make_query(task_id=1, subject="Chemistry"),
            _make_query(task_id=2, subject="Physics"),
            _make_query(task_id=3, subject="Astronomy"),
        ]
        env = adapter.get_environment("sciagentgym:physics", dataset=ds)
        assert len(env) == 2

    def test_get_environment_subject_filter_case_insensitive(self, monkeypatch):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        monkeypatch.setattr(adapter, "_get_sciagentgym", lambda: MagicMock())

        ds = [
            _make_query(task_id=0, subject="Physics"),
            _make_query(task_id=1, subject="Chemistry"),
        ]
        env = adapter.get_environment("sciagentgym:PHYSICS", dataset=ds)
        assert len(env) == 1

    def test_get_environment_requires_data(self, monkeypatch):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        monkeypatch.setattr(adapter, "_get_sciagentgym", lambda: MagicMock())

        with pytest.raises(ValueError, match="dataset.*data_path"):
            adapter.get_environment("sciagentgym")

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        assert adapter.get_native_answer_extractor("sciagentgym") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        assert adapter.get_prompt_template("sciagentgym") is None

    def test_get_environment_info(self):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        info = adapter.get_environment_info("sciagentgym")
        assert info["name"] == "sciagentgym"
        assert info["adapter"] == "sciagentgym"
        assert "subjects" in info

    def test_max_steps_passed_through(self, monkeypatch):
        from llenvs.adapters.sciagentgym import SciAgentGymAdapter

        adapter = SciAgentGymAdapter()
        monkeypatch.setattr(adapter, "_get_sciagentgym", lambda: MagicMock())

        ds = _make_dataset()
        env = adapter.get_environment("sciagentgym", dataset=ds, max_steps=20)
        assert env._max_steps == 20


# ── TestActionConversion ─────────────────────────────────────────


class TestActionConversion:
    @pytest.fixture(autouse=True)
    def _mock_sciagentgym(self, monkeypatch):
        """Mock SciAgentGYM imports for action conversion."""
        import sys

        mock_gym_tool = MagicMock()
        mock_gym_tool.ToolCall = MockSciToolCall

        monkeypatch.setitem(sys.modules, "gym", MagicMock())
        monkeypatch.setitem(sys.modules, "gym.tool", mock_gym_tool)

    def test_single_tool_call(self):
        from llenvs.adapters.sciagentgym import _to_sci_tool_call

        tc = ToolCall(id="tc-1", name="analyze", arguments={"energy": 1.0})
        result = _to_sci_tool_call(tc)

        assert result.id == "tc-1"
        assert result.name == "analyze"
        assert result.arguments == {"energy": 1.0}

    def test_preserves_arguments(self):
        from llenvs.adapters.sciagentgym import _to_sci_tool_call

        args = {"query": "hello", "limit": 10, "nested": {"key": "val"}}
        tc = ToolCall(id="tc-1", name="search", arguments=args)
        result = _to_sci_tool_call(tc)

        assert result.arguments == args

    def test_empty_arguments(self):
        from llenvs.adapters.sciagentgym import _to_sci_tool_call

        tc = ToolCall(id="tc-1", name="reset_env", arguments={})
        result = _to_sci_tool_call(tc)

        assert result.arguments == {}

    def test_complex_nested_arguments(self):
        from llenvs.adapters.sciagentgym import _to_sci_tool_call

        args = {
            "data": [1, 2, 3],
            "config": {"nested": {"deep": True}},
            "name": "test",
        }
        tc = ToolCall(id="tc-1", name="process", arguments=args)
        result = _to_sci_tool_call(tc)

        assert result.arguments["data"] == [1, 2, 3]
        assert result.arguments["config"]["nested"]["deep"] is True


# ── TestLoadDataset ──────────────────────────────────────────────


class TestLoadDataset:
    def test_load_json_file(self, tmp_path):
        from llenvs.adapters.sciagentgym import _load_dataset

        data = _make_dataset(num_tasks=3)
        data_file = tmp_path / "test.json"
        data_file.write_text(json.dumps(data))

        result = _load_dataset(str(data_file))
        assert len(result) == 3

    def test_load_nonexistent_file(self):
        from llenvs.adapters.sciagentgym import _load_dataset

        with pytest.raises(FileNotFoundError):
            _load_dataset("/nonexistent/path/data.json")

    def test_load_directory(self, tmp_path):
        from llenvs.adapters.sciagentgym import _load_dataset

        data1 = _make_dataset(num_tasks=2)
        data2 = _make_dataset(num_tasks=3)
        (tmp_path / "file1.json").write_text(json.dumps(data1))
        (tmp_path / "file2.json").write_text(json.dumps(data2))

        result = _load_dataset(str(tmp_path))
        assert len(result) == 5

    def test_load_empty_directory(self, tmp_path):
        from llenvs.adapters.sciagentgym import _load_dataset

        result = _load_dataset(str(tmp_path))
        assert len(result) == 0


# ── TestSubjectConstants ─────────────────────────────────────────


class TestSubjectConstants:
    def test_subjects_defined(self):
        from llenvs.adapters.sciagentgym import SCIAGENTGYM_SUBJECTS

        assert isinstance(SCIAGENTGYM_SUBJECTS, tuple)
        assert len(SCIAGENTGYM_SUBJECTS) >= 6
        assert "physics" in SCIAGENTGYM_SUBJECTS
        assert "chemistry" in SCIAGENTGYM_SUBJECTS
        assert "astronomy" in SCIAGENTGYM_SUBJECTS
