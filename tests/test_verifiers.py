"""Tests for the verifiers adapter."""

import asyncio
import pytest
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

from llenvs.core.state import Observation, Action, State, StateMetadata
from llenvs.core.reward import RewardType, Signal, FormatReward
from llenvs.core.extraction import TagBasedExtractor
from llenvs.core.tools import ToolCall, ToolDefinition, ToolParameter, ToolParameterType


# ── Mock verifiers objects ──────────────────────────────────────────


def _make_mock_dataset(rows=None):
    """Create a mock HuggingFace-style dataset."""
    rows = rows or [
        {
            "prompt": [
                {"role": "system", "content": "Solve the math problem."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "answer": "4",
            "example_id": 0,
        },
        {
            "prompt": [
                {"role": "system", "content": "Solve the math problem."},
                {"role": "user", "content": "What is 3*3?"},
            ],
            "answer": "9",
            "example_id": 1,
        },
        {
            "prompt": [
                {"role": "user", "content": "What is 10/2?"},
            ],
            "answer": "5",
            "example_id": 2,
        },
    ]
    ds = MagicMock()
    ds.__len__ = lambda self: len(rows)
    ds.__getitem__ = lambda self, idx: rows[idx]
    ds.column_names = list(rows[0].keys())
    return ds


def _make_reward_func(name="correct_answer", weight=1.0, return_value=1.0):
    """Create a mock async reward function."""
    func = AsyncMock(return_value=return_value)
    func.__name__ = name
    func.__qualname__ = name
    # Simulate inspect.signature: the func takes (completion, answer)
    import inspect

    params = [
        inspect.Parameter("completion", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("answer", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    func.__signature__ = inspect.Signature(params)
    return func


def _make_rubric(funcs=None, weights=None, parser=None):
    """Create a mock Rubric."""
    if funcs is None:
        funcs = [_make_reward_func("correct_answer", return_value=1.0)]
    if weights is None:
        weights = [1.0] * len(funcs)

    rubric = MagicMock()
    rubric.funcs = funcs
    rubric.weights = weights
    rubric.parser = parser
    return rubric


def _make_verifiers_env(env_type="SingleTurnEnv", dataset=None, rubric=None,
                        system_prompt=None, tools=None, oai_tools=None, tool_map=None):
    """Create a mock verifiers environment."""
    env = MagicMock()
    env.__class__.__name__ = env_type
    env.dataset = dataset or _make_mock_dataset()
    env.rubric = rubric or _make_rubric()
    env.system_prompt = system_prompt or "Solve the math problem."
    env.env_id = "test-env"
    env.env_args = {}

    # Type checks
    env._is_single_turn = env_type == "SingleTurnEnv"
    env._is_tool_env = env_type in ("ToolEnv", "StatefulToolEnv")

    if tools is not None:
        env.tools = tools
    if oai_tools is not None:
        env.oai_tools = oai_tools
    else:
        env.oai_tools = []
    if tool_map is not None:
        env.tool_map = tool_map
    else:
        env.tool_map = {}
    env.max_turns = 1 if env_type == "SingleTurnEnv" else 10

    return env


# ── Phase 1: Single-Turn Environment Tests ──────────────────────────


class TestVerifiersHidden:
    """Tests for VerifiersHidden dataclass."""

    def test_creation(self):
        from llenvs.adapters.verifiers import VerifiersHidden

        hidden = VerifiersHidden(
            env_id="gsm8k",
            task_index=0,
            expected_answer="4",
            dataset_item=(("prompt", []), ("answer", "4")),
        )
        assert hidden.env_id == "gsm8k"
        assert hidden.task_index == 0
        assert hidden.expected_answer == "4"

    def test_immutability(self):
        from llenvs.adapters.verifiers import VerifiersHidden

        hidden = VerifiersHidden(
            env_id="gsm8k",
            task_index=0,
            expected_answer="4",
            dataset_item=(("answer", "4"),),
        )
        with pytest.raises(AttributeError):
            hidden.expected_answer = "5"  # type: ignore

    def test_none_expected_answer(self):
        from llenvs.adapters.verifiers import VerifiersHidden

        hidden = VerifiersHidden(
            env_id="test",
            task_index=0,
            expected_answer=None,
            dataset_item=(),
        )
        assert hidden.expected_answer is None


class TestVerifiersRubricReward:
    """Tests for VerifiersRubricReward."""

    def test_properties(self):
        from llenvs.adapters.verifiers import VerifiersRubricReward

        rubric = _make_rubric()
        reward = VerifiersRubricReward(rubric=rubric, env_id="test")

        assert reward.name == "verifiers_rubric"
        assert reward.reward_type == RewardType.OUTCOME

    def test_compute_correct_answer(self):
        from llenvs.adapters.verifiers import VerifiersRubricReward, VerifiersHidden

        func = _make_reward_func("correct_answer", return_value=1.0)
        rubric = _make_rubric(funcs=[func], weights=[1.0])
        reward = VerifiersRubricReward(rubric=rubric, env_id="test")

        hidden = VerifiersHidden(
            env_id="test",
            task_index=0,
            expected_answer="4",
            dataset_item=(("answer", "4"), ("prompt", [])),
        )
        state = State(
            observation=Observation(prompt="What is 2+2?"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )
        action = Action(text="The answer is 4")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0
        assert signal.name == "verifiers_rubric"
        assert signal.reward_type == RewardType.OUTCOME

    def test_compute_with_weights(self):
        from llenvs.adapters.verifiers import VerifiersRubricReward, VerifiersHidden

        func1 = _make_reward_func("correctness", return_value=1.0)
        func2 = _make_reward_func("format", return_value=0.5)
        rubric = _make_rubric(funcs=[func1, func2], weights=[2.0, 0.5])
        reward = VerifiersRubricReward(rubric=rubric, env_id="test")

        hidden = VerifiersHidden(
            env_id="test", task_index=0, expected_answer="4",
            dataset_item=(("answer", "4"), ("prompt", [])),
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )

        signal = reward.compute(state, action=Action(text="4"), next_state=state)
        # Weighted sum: 1.0*2.0 + 0.5*0.5 = 2.25, but we return the bundle's
        # total. Individual signals use weight field.
        assert signal.reward == pytest.approx(2.25)

    def test_compute_incorrect_answer(self):
        from llenvs.adapters.verifiers import VerifiersRubricReward, VerifiersHidden

        func = _make_reward_func("correct_answer", return_value=0.0)
        rubric = _make_rubric(funcs=[func], weights=[1.0])
        reward = VerifiersRubricReward(rubric=rubric, env_id="test")

        hidden = VerifiersHidden(
            env_id="test", task_index=0, expected_answer="4",
            dataset_item=(("answer", "4"), ("prompt", [])),
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )

        signal = reward.compute(state, Action(text="wrong"), state)
        assert signal.reward == 0.0

    def test_compute_exception_returns_zero(self):
        from llenvs.adapters.verifiers import VerifiersRubricReward, VerifiersHidden

        func = _make_reward_func("bad_func")
        func.side_effect = RuntimeError("boom")
        rubric = _make_rubric(funcs=[func], weights=[1.0])
        reward = VerifiersRubricReward(rubric=rubric, env_id="test")

        hidden = VerifiersHidden(
            env_id="test", task_index=0, expected_answer="4",
            dataset_item=(("answer", "4"), ("prompt", [])),
        )
        state = State(
            observation=Observation(prompt="Q"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )

        signal = reward.compute(state, Action(text="x"), state)
        assert signal.reward == 0.0
        assert signal.metadata is not None
        assert "error" in signal.metadata


class TestVerifiersSingleTurnEnvironment:
    """Tests for VerifiersSingleTurnEnvironment."""

    def _make_env(self, **kwargs):
        from llenvs.adapters.verifiers import VerifiersSingleTurnEnvironment

        vf_env = kwargs.pop("vf_env", _make_verifiers_env())
        return VerifiersSingleTurnEnvironment(vf_env=vf_env, **kwargs)

    def test_creation(self):
        env = self._make_env()
        assert env.spec.name == "test-env"
        assert env.spec.adapter == "verifiers"
        assert env.spec.max_steps == 1
        assert env.spec.is_multi_turn is False

    def test_len(self):
        env = self._make_env()
        assert len(env) == 3

    def test_spec_properties(self):
        env = self._make_env()
        spec = env.spec
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is False
        assert spec.pure_step is True

    def test_prompts_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_available_tools_empty(self):
        env = self._make_env()
        assert env.available_tools == ()

    def test_reward_functions_default(self):
        env = self._make_env()
        assert len(env.reward_functions) == 1
        assert env.reward_functions[0].name == "verifiers_rubric"

    def test_extra_rewards(self):
        extractor = TagBasedExtractor()
        format_reward = FormatReward(extractor)
        env = self._make_env(extra_rewards=(format_reward,))
        assert len(env.reward_functions) == 2
        assert env.reward_functions[0].name == "verifiers_rubric"
        assert env.reward_functions[1].name == "format"

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert "What is 2+2?" in state.observation.prompt
        assert state.hidden.expected_answer == "4"
        assert state.hidden.task_index == 0
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0

    def test_reset_extracts_system_prompt_from_dataset(self):
        """System prompt comes from dataset row's prompt messages."""
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})
        # Row 0 has system message in prompt
        assert info.get("system_prompt") == "Solve the math problem."

    def test_reset_no_system_prompt_in_row(self):
        """Row without system message uses env-level system_prompt."""
        env = self._make_env()
        state, info = env.reset(options={"task_index": 2})
        # Row 2 has no system message — env's system_prompt used
        assert info.get("system_prompt") == "Solve the math problem."

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

    def test_step_terminates(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="4"))

        assert result.terminated is True
        assert result.truncated is False
        assert result.next_state.metadata.is_terminal is True

    def test_step_computes_rewards(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="The answer is 4"))

        rubric_signal = result.rewards.by_name("verifiers_rubric")
        assert rubric_signal is not None
        assert rubric_signal.reward_type == RewardType.OUTCOME

    def test_step_extracted_answer_in_info(self):
        """When answer_extractor is provided, extracted answer shows in info."""
        extractor = TagBasedExtractor()
        env = self._make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="<answer>4</answer>"))

        assert result.info["extracted_answer"] == "4"

    def test_step_no_extractor_no_extracted_answer(self):
        """Without an extractor, extracted_answer is None."""
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="4"))

        assert result.info.get("extracted_answer") is None

    def test_step_state_immutable(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        original_step = state.metadata.step

        env.step(state, Action(text="4"))
        assert state.metadata.step == original_step

    def test_step_next_state_updated(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="4"))

        assert result.next_state.metadata.step == 1
        assert result.next_state.metadata.is_terminal is True

    def test_compute_rewards_directly(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        rewards = env.compute_rewards(state, Action(text="4"), state)

        assert len(rewards.signals) >= 1

    def test_system_prompt_property(self):
        env = self._make_env()
        assert env.system_prompt == "Solve the math problem."

    def test_system_prompt_none(self):
        vf_env = _make_verifiers_env()
        vf_env.system_prompt = None
        env = self._make_env(vf_env=vf_env)
        assert env.system_prompt is None


# ── Phase 2: Tool Environment Tests ────────────────────────────────


def _make_oai_tools():
    """Create mock OpenAI-format tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"},
                    },
                    "required": ["expression"],
                },
            },
        },
    ]


def _make_tool_funcs():
    """Create mock tool callables."""
    async def search(query: str) -> str:
        return f"Results for: {query}"

    async def calculate(expression: str) -> str:
        return "42"

    return {"search": search, "calculate": calculate}


class TestVerifiersToolHidden:
    """Tests for VerifiersToolHidden dataclass."""

    def test_creation(self):
        from llenvs.adapters.verifiers import VerifiersToolHidden

        hidden = VerifiersToolHidden(
            env_id="tool-test",
            task_index=0,
            expected_answer="42",
            dataset_item=(("answer", "42"),),
            episode_step=0,
            last_action=None,
        )
        assert hidden.env_id == "tool-test"
        assert hidden.episode_step == 0

    def test_immutability(self):
        from llenvs.adapters.verifiers import VerifiersToolHidden

        hidden = VerifiersToolHidden(
            env_id="test", task_index=0, expected_answer=None,
            dataset_item=(), episode_step=0, last_action=None,
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore


class TestVerifiersToolExecutor:
    """Tests for VerifiersToolExecutor."""

    def test_execute_success(self):
        from llenvs.adapters.verifiers import VerifiersToolExecutor

        tool_map = _make_tool_funcs()
        executor = VerifiersToolExecutor(tool_map=tool_map)

        call = ToolCall(id="call-1", name="search", arguments={"query": "hello"})
        result = executor.execute(call)

        assert result.is_success
        assert "Results for: hello" in str(result.output)

    def test_execute_unknown_tool(self):
        from llenvs.adapters.verifiers import VerifiersToolExecutor

        executor = VerifiersToolExecutor(tool_map={})
        call = ToolCall(id="call-1", name="unknown", arguments={})
        result = executor.execute(call)

        assert not result.is_success
        assert "unknown" in result.error.lower()

    def test_execute_error_handling(self):
        from llenvs.adapters.verifiers import VerifiersToolExecutor

        async def bad_tool(**kwargs):
            raise ValueError("tool error")

        executor = VerifiersToolExecutor(tool_map={"bad": bad_tool})
        call = ToolCall(id="call-1", name="bad", arguments={})
        result = executor.execute(call)

        assert not result.is_success
        assert "tool error" in result.error


class TestVerifiersToolEnvironment:
    """Tests for VerifiersToolEnvironment."""

    def _make_env(self, **kwargs):
        from llenvs.adapters.verifiers import VerifiersToolEnvironment

        tool_map = kwargs.pop("tool_map", _make_tool_funcs())
        oai_tools = kwargs.pop("oai_tools", _make_oai_tools())
        vf_env = kwargs.pop("vf_env", _make_verifiers_env(
            env_type="ToolEnv",
            oai_tools=oai_tools,
            tool_map=tool_map,
        ))
        return VerifiersToolEnvironment(vf_env=vf_env, **kwargs)

    def test_creation(self):
        env = self._make_env()
        assert env.spec.is_multi_turn is True
        assert env.spec.adapter == "verifiers"

    def test_spec_max_steps(self):
        env = self._make_env()
        assert env.spec.max_steps == 10

    def test_spec_pure_step_false(self):
        env = self._make_env()
        assert env.spec.pure_step is False

    def test_step_raises_on_stale_state(self):
        """Replaying the initial state after a step raises NotImplementedError."""
        env = self._make_env()
        state_0, _ = env.reset(options={"task_index": 0})

        env.step(state_0, Action(text="Let me think..."))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, Action(text="Let me think..."))

    def test_available_tools(self):
        env = self._make_env()
        tools = env.available_tools
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search" in names
        assert "calculate" in names

    def test_tool_definitions_have_parameters(self):
        env = self._make_env()
        search_tool = next(t for t in env.available_tools if t.name == "search")
        assert len(search_tool.parameters) == 1
        assert search_tool.parameters[0].name == "query"
        assert search_tool.parameters[0].type == ToolParameterType.STRING

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert "What is 2+2?" in state.observation.prompt
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None
        assert state.metadata.is_terminal is False

    def test_step_text_only(self):
        """Text-only action in a tool env (no tool calls)."""
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        # Text-only action — intermediate step, not terminal
        action = Action(text="Let me think about this...")
        result = env.step(state, action)

        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "Let me think about this..."

    def test_step_with_tool_calls(self):
        """Action with tool calls executes tools and returns results."""
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        call = ToolCall(id="call-1", name="search", arguments={"query": "2+2"})
        action = Action(text="", tool_calls=(call,))
        result = env.step(state, action)

        # Should have tool results in the observation
        assert len(result.next_state.observation.tool_results) == 1
        assert result.next_state.observation.tool_results[0].is_success

    def test_step_max_steps_truncation(self):
        """Environment truncates at max_steps."""
        vf_env = _make_verifiers_env(env_type="ToolEnv")
        vf_env.max_turns = 2
        from llenvs.adapters.verifiers import VerifiersToolEnvironment
        env = VerifiersToolEnvironment(vf_env=vf_env)

        state, _ = env.reset(options={"task_index": 0})

        # Step 1
        result = env.step(state, Action(text="step 1"))
        assert not result.done

        # Step 2 — should truncate
        result = env.step(result.next_state, Action(text="step 2"))
        assert result.truncated is True
        assert result.terminated is False

    def test_reward_functions(self):
        env = self._make_env()
        assert len(env.reward_functions) >= 1
        assert env.reward_functions[0].name == "verifiers_rubric"

    def test_len(self):
        env = self._make_env()
        assert len(env) == 3

    def test_system_prompt(self):
        env = self._make_env()
        assert env.system_prompt == "Solve the math problem."


# ── Adapter Tests ───────────────────────────────────────────────────


class TestVerifiersAdapter:
    """Tests for VerifiersAdapter."""

    def test_adapter_name(self):
        from llenvs.adapters.verifiers import VerifiersAdapter
        adapter = VerifiersAdapter()
        assert adapter.name == "verifiers"

    def test_get_verifiers_import_error(self):
        from llenvs.adapters.verifiers import VerifiersAdapter
        adapter = VerifiersAdapter()
        with pytest.raises(ImportError, match="verifiers"):
            adapter._get_verifiers()

    def test_get_environment_single_turn(self, monkeypatch):
        from llenvs.adapters.verifiers import (
            VerifiersAdapter,
            VerifiersSingleTurnEnvironment,
        )

        mock_vf_env = _make_verifiers_env(env_type="SingleTurnEnv")

        mock_vf = MagicMock()
        mock_vf.load_environment.return_value = mock_vf_env
        mock_vf.SingleTurnEnv = type("SingleTurnEnv", (), {})
        mock_vf.ToolEnv = type("ToolEnv", (), {})
        mock_vf.MultiTurnEnv = type("MultiTurnEnv", (), {})
        # Make isinstance work
        mock_vf_env.__class__ = mock_vf.SingleTurnEnv

        adapter = VerifiersAdapter()
        monkeypatch.setattr(adapter, "_get_verifiers", lambda: mock_vf)

        env = adapter.get_environment("gsm8k")
        assert isinstance(env, VerifiersSingleTurnEnvironment)

    def test_get_environment_tool_env(self, monkeypatch):
        from llenvs.adapters.verifiers import (
            VerifiersAdapter,
            VerifiersToolEnvironment,
        )

        mock_vf_env = _make_verifiers_env(env_type="ToolEnv")

        mock_vf = MagicMock()
        mock_vf.load_environment.return_value = mock_vf_env
        mock_vf.SingleTurnEnv = type("SingleTurnEnv", (), {})
        mock_vf.ToolEnv = type("ToolEnv", (), {})
        mock_vf.MultiTurnEnv = type("MultiTurnEnv", (), {})
        mock_vf_env.__class__ = mock_vf.ToolEnv

        adapter = VerifiersAdapter()
        monkeypatch.setattr(adapter, "_get_verifiers", lambda: mock_vf)

        env = adapter.get_environment("tool-test")
        assert isinstance(env, VerifiersToolEnvironment)

    def test_get_environment_sandbox_raises(self, monkeypatch):
        from llenvs.adapters.verifiers import VerifiersAdapter

        mock_vf_env = MagicMock()

        mock_vf = MagicMock()
        mock_vf.load_environment.return_value = mock_vf_env
        mock_vf.SingleTurnEnv = type("SingleTurnEnv", (), {})
        mock_vf.ToolEnv = type("ToolEnv", (), {})
        mock_vf.MultiTurnEnv = type("MultiTurnEnv", (), {})
        # Not any recognized type
        mock_vf_env.__class__ = type("SandboxEnv", (), {})

        adapter = VerifiersAdapter()
        monkeypatch.setattr(adapter, "_get_verifiers", lambda: mock_vf)

        with pytest.raises(NotImplementedError, match="[Ss]andbox|not supported"):
            adapter.get_environment("sandbox-test")

    def test_get_environment_passes_kwargs(self, monkeypatch):
        from llenvs.adapters.verifiers import VerifiersAdapter

        mock_vf_env = _make_verifiers_env(env_type="SingleTurnEnv")

        mock_vf = MagicMock()
        mock_vf.load_environment.return_value = mock_vf_env
        mock_vf.SingleTurnEnv = type("SingleTurnEnv", (), {})
        mock_vf.ToolEnv = type("ToolEnv", (), {})
        mock_vf.MultiTurnEnv = type("MultiTurnEnv", (), {})
        mock_vf_env.__class__ = mock_vf.SingleTurnEnv

        adapter = VerifiersAdapter()
        monkeypatch.setattr(adapter, "_get_verifiers", lambda: mock_vf)

        adapter.get_environment("gsm8k", system_prompt="Custom prompt")
        mock_vf.load_environment.assert_called_once_with(
            "gsm8k", system_prompt="Custom prompt"
        )

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.verifiers import VerifiersAdapter
        adapter = VerifiersAdapter()
        # No native answer extractor
        assert adapter.get_native_answer_extractor("gsm8k") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.verifiers import VerifiersAdapter
        adapter = VerifiersAdapter()
        assert adapter.get_prompt_template("gsm8k") is None

    def test_get_environment_info(self):
        from llenvs.adapters.verifiers import VerifiersAdapter
        adapter = VerifiersAdapter()
        info = adapter.get_environment_info("gsm8k")
        assert info["name"] == "gsm8k"
        assert info["adapter"] == "verifiers"


# ── OAI schema conversion tests ────────────────────────────────────


class TestOaiSchemaConversion:
    """Tests for converting OpenAI tool schemas to ToolDefinition."""

    def test_basic_conversion(self):
        from llenvs.adapters.verifiers import _oai_tools_to_definitions

        oai_tools = _make_oai_tools()
        defs = _oai_tools_to_definitions(oai_tools)

        assert len(defs) == 2
        assert defs[0].name == "search"
        assert defs[0].description == "Search for information"
        assert len(defs[0].parameters) == 1
        assert defs[0].parameters[0].name == "query"
        assert defs[0].parameters[0].required is True

    def test_optional_parameters(self):
        from llenvs.adapters.verifiers import _oai_tools_to_definitions

        oai_tools = [{
            "type": "function",
            "function": {
                "name": "test",
                "description": "Test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "required_param": {"type": "string"},
                        "optional_param": {"type": "integer"},
                    },
                    "required": ["required_param"],
                },
            },
        }]
        defs = _oai_tools_to_definitions(oai_tools)
        params = {p.name: p for p in defs[0].parameters}
        assert params["required_param"].required is True
        assert params["optional_param"].required is False

    def test_type_mapping(self):
        from llenvs.adapters.verifiers import _oai_tools_to_definitions

        oai_tools = [{
            "type": "function",
            "function": {
                "name": "test",
                "description": "Test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "s": {"type": "string"},
                        "i": {"type": "integer"},
                        "n": {"type": "number"},
                        "b": {"type": "boolean"},
                        "a": {"type": "array"},
                        "o": {"type": "object"},
                    },
                    "required": [],
                },
            },
        }]
        defs = _oai_tools_to_definitions(oai_tools)
        params = {p.name: p for p in defs[0].parameters}
        assert params["s"].type == ToolParameterType.STRING
        assert params["i"].type == ToolParameterType.INTEGER
        assert params["n"].type == ToolParameterType.NUMBER
        assert params["b"].type == ToolParameterType.BOOLEAN
        assert params["a"].type == ToolParameterType.ARRAY
        assert params["o"].type == ToolParameterType.OBJECT

    def test_empty_tools(self):
        from llenvs.adapters.verifiers import _oai_tools_to_definitions
        assert _oai_tools_to_definitions([]) == ()

    def test_no_parameters(self):
        from llenvs.adapters.verifiers import _oai_tools_to_definitions

        oai_tools = [{
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Ping",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        defs = _oai_tools_to_definitions(oai_tools)
        assert len(defs) == 1
        assert len(defs[0].parameters) == 0
