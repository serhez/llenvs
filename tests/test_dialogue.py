"""Tests for DialogueEnvironment and related components."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType, RewardFunction
from llenvs.core.environment import StepResult, EnvironmentSpec
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response_text: str = "No.") -> MagicMock:
    """Create a mock ModelBackend that returns a fixed response."""
    backend = MagicMock()
    backend.generate_chat.return_value = GenerationResult(
        text=response_text,
        finish_reason=StopReason.END_OF_TEXT,
    )
    return backend


def _mock_backend_sequence(responses: list[str]) -> MagicMock:
    """Create a mock ModelBackend that returns responses in sequence."""
    backend = MagicMock()
    results = [
        GenerationResult(text=r, finish_reason=StopReason.END_OF_TEXT)
        for r in responses
    ]
    backend.generate_chat.side_effect = results
    return backend


class DummyReward:
    """A simple reward function for testing extra_rewards."""

    def __init__(self, value: float = 0.5, name: str = "dummy"):
        self._name = name
        self._value = value
        self._reward_type = RewardType.STEP

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(self, state: Any, action: Any, next_state: Any) -> RewardSignal:
        return RewardSignal(
            value=self._value, name=self._name, reward_type=self._reward_type
        )


# ---------------------------------------------------------------------------
# DialogueTask tests
# ---------------------------------------------------------------------------


class TestDialogueTask:
    def test_creation(self):
        from llenvs.adapters.dialogue import DialogueTask

        task = DialogueTask(prompt="Hello")
        assert task.prompt == "Hello"
        assert task.context == ""
        assert task.ground_truth == ""
        assert task.metadata == {}

    def test_creation_full(self):
        from llenvs.adapters.dialogue import DialogueTask

        task = DialogueTask(
            prompt="Ask questions",
            context="Secret word: cat",
            ground_truth="cat",
            metadata={"difficulty": "easy"},
        )
        assert task.prompt == "Ask questions"
        assert task.context == "Secret word: cat"
        assert task.ground_truth == "cat"
        assert task.metadata == {"difficulty": "easy"}

    def test_frozen(self):
        from llenvs.adapters.dialogue import DialogueTask

        task = DialogueTask(prompt="Hello")
        with pytest.raises(AttributeError):
            task.prompt = "Goodbye"  # type: ignore[misc]

    def test_defaults(self):
        from llenvs.adapters.dialogue import DialogueTask

        task = DialogueTask(prompt="Q")
        assert task.context == ""
        assert task.ground_truth == ""
        assert task.metadata == {}


# ---------------------------------------------------------------------------
# DialogueHidden tests
# ---------------------------------------------------------------------------


class TestDialogueHidden:
    def test_creation(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueHidden

        task = DialogueTask(prompt="Q", ground_truth="A")
        hidden = DialogueHidden(task_index=0, task=task, step_count=0)
        assert hidden.task_index == 0
        assert hidden.task is task
        assert hidden.step_count == 0

    def test_frozen(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueHidden

        task = DialogueTask(prompt="Q")
        hidden = DialogueHidden(task_index=0, task=task, step_count=0)
        with pytest.raises(AttributeError):
            hidden.step_count = 5  # type: ignore[misc]

    def test_ground_truth_property(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueHidden

        task = DialogueTask(prompt="Q", ground_truth="cat")
        hidden = DialogueHidden(task_index=0, task=task, step_count=0)
        assert hidden.ground_truth == "cat"

    def test_ground_truth_empty(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueHidden

        task = DialogueTask(prompt="Q")
        hidden = DialogueHidden(task_index=0, task=task, step_count=0)
        assert hidden.ground_truth == ""


# ---------------------------------------------------------------------------
# DialogueEnvironment.reset() tests
# ---------------------------------------------------------------------------


class TestDialogueEnvironmentReset:
    def test_initial_state(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        tasks = (DialogueTask(prompt="Hello agent"),)
        env = DialogueEnvironment(backend=backend, tasks=tasks)

        state, info = env.reset(options={"task_index": 0})
        assert state.observation.prompt == "Hello agent"
        assert state.observation.messages == ()
        assert state.hidden.task_index == 0
        assert state.hidden.step_count == 0
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

    def test_task_prompt_in_observation(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        tasks = (
            DialogueTask(prompt="First"),
            DialogueTask(prompt="Second"),
        )
        env = DialogueEnvironment(backend=backend, tasks=tasks)

        state, _ = env.reset(options={"task_index": 1})
        assert state.observation.prompt == "Second"
        assert state.hidden.task_index == 1

    def test_hidden_state_populated(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        task = DialogueTask(prompt="Q", context="ctx", ground_truth="GT")
        env = DialogueEnvironment(backend=backend, tasks=(task,))

        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.task.context == "ctx"
        assert state.hidden.task.ground_truth == "GT"

    def test_task_index_bounds_check(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )

        with pytest.raises(ValueError):
            env.reset(options={"task_index": 1})

        with pytest.raises(ValueError):
            env.reset(options={"task_index": -1})

    def test_default_task_index(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )
        state, _ = env.reset()
        assert state.hidden.task_index == 0

    def test_len(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        tasks = tuple(DialogueTask(prompt=f"Q{i}") for i in range(5))
        env = DialogueEnvironment(backend=backend, tasks=tasks)
        assert len(env) == 5


# ---------------------------------------------------------------------------
# DialogueEnvironment.step() tests
# ---------------------------------------------------------------------------


class TestDialogueEnvironmentStep:
    def test_env_llm_called_with_correct_messages(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("No.")
        task = DialogueTask(
            prompt="Guess the word",
            context="The word is cat.",
        )
        env = DialogueEnvironment(
            backend=backend,
            tasks=(task,),
            system_prompt="You are an oracle. {context}",
        )

        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("Is it a dog?"))

        # Check the backend was called
        backend.generate_chat.assert_called_once()
        messages = backend.generate_chat.call_args[0][0]

        # System prompt with context interpolated
        assert messages[0].role == "system"
        assert "The word is cat." in messages[0].content
        # Agent's action as "user" message
        assert messages[1].role == "user"
        assert messages[1].content == "Is it a dog?"

    def test_response_becomes_observation(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("No, try again.")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("My guess"))

        # New messages should include agent action + env response
        msgs = result.next_state.observation.messages
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "My guess"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "No, try again."

    def test_message_history_grows(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend_sequence(["No.", "No.", "Correct!"])
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Guess"),),
            max_steps=10,
        )

        state, _ = env.reset(options={"task_index": 0})

        # Step 1
        result1 = env.step(state, Action.from_text("Q1"))
        assert len(result1.next_state.observation.messages) == 2

        # Step 2
        result2 = env.step(result1.next_state, Action.from_text("Q2"))
        assert len(result2.next_state.observation.messages) == 4

        # Step 3
        result3 = env.step(result2.next_state, Action.from_text("Q3"))
        assert len(result3.next_state.observation.messages) == 6

    def test_step_count_increments(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("No.")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            max_steps=10,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("A"))
        assert result.next_state.hidden.step_count == 1
        assert result.next_state.metadata.step == 1

    def test_action_text_in_messages(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Response")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
        )

        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("My specific text"))

        messages = backend.generate_chat.call_args[0][0]
        assert any(m.content == "My specific text" for m in messages)

    def test_none_action_text_becomes_empty(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Response")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
        )

        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action(text=None))

        messages = backend.generate_chat.call_args[0][0]
        # The user message should be empty string, not None
        user_msgs = [m for m in messages if m.role == "user"]
        assert user_msgs[-1].content == ""

    def test_system_prompt_without_context(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("OK")
        task = DialogueTask(prompt="Q", context="")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(task,),
            system_prompt="You are a helpful bot.",
        )

        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("Hi"))

        messages = backend.generate_chat.call_args[0][0]
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful bot."

    def test_conversation_history_in_backend_call(self):
        """After multiple steps, the full history is sent to the backend."""
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend_sequence(["R1", "R2"])
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            system_prompt="System",
            max_steps=10,
        )

        state, _ = env.reset(options={"task_index": 0})
        r1 = env.step(state, Action.from_text("A1"))
        env.step(r1.next_state, Action.from_text("A2"))

        # Second call should include full history
        messages = backend.generate_chat.call_args_list[1][0][0]
        roles = [m.role for m in messages]
        assert roles == ["system", "assistant", "user", "user"]
        # assistant=A1, user=R1, user=A2


# ---------------------------------------------------------------------------
# Termination tests
# ---------------------------------------------------------------------------


class TestDialogueTermination:
    def test_is_terminal_callback(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Correct!")

        def check_terminal(env_resp: str, agent_action: str, step: int) -> bool:
            return "correct" in env_resp.lower()

        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            is_terminal=check_terminal,
            max_steps=20,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("cat"))
        assert result.terminated is True
        assert result.truncated is False

    def test_max_steps_truncation(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("No.")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            max_steps=1,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("Guess"))
        assert result.terminated is False
        assert result.truncated is True

    def test_not_done_continues(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("No.")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            max_steps=5,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("Guess"))
        assert result.terminated is False
        assert result.truncated is False
        assert result.done is False

    def test_terminal_and_truncated_both_set(self):
        """If terminal triggers on the last step, both flags can be true."""
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Correct!")

        def always_terminal(resp: str, action: str, step: int) -> bool:
            return True

        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            is_terminal=always_terminal,
            max_steps=1,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("A"))
        assert result.terminated is True
        assert result.truncated is True


# ---------------------------------------------------------------------------
# Rewards tests
# ---------------------------------------------------------------------------


class TestDialogueRewards:
    def test_no_native_rewards(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )
        assert env.reward_functions == ()

    def test_extra_rewards_included(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        dummy = DummyReward(0.7, "test_reward")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            extra_rewards=(dummy,),
        )
        assert len(env.reward_functions) == 1
        assert env.reward_functions[0].name == "test_reward"

    def test_rewards_computed_on_step(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Response")
        dummy = DummyReward(0.7, "test_reward")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            extra_rewards=(dummy,),
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("A"))
        assert len(result.rewards.signals) == 1
        assert result.rewards.signals[0].value == 0.7

    def test_judge_reward_compatibility(self):
        """DialogueHidden.ground_truth is accessible for JudgeReward."""
        from llenvs.adapters.dialogue import DialogueTask, DialogueHidden
        from llenvs.core.judge import _gather_judge_context

        task = DialogueTask(prompt="Question", ground_truth="expected_answer")
        hidden = DialogueHidden(task_index=0, task=task, step_count=0)
        state = State(
            observation=Observation(prompt="Question"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test"),
        )
        action = Action.from_text("my response")

        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "expected_answer"
        assert ctx["question"] == "Question"
        assert ctx["response"] == "my response"


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------


class TestDialogueSpec:
    def test_spec_properties(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            max_steps=10,
        )
        spec = env.spec
        assert spec.name == "dialogue"
        assert spec.adapter == "dialogue"
        assert spec.max_steps == 10
        assert spec.is_multi_turn is True
        assert spec.pure_step is True

    def test_prompts_empty(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )
        assert env.prompts == {}

    def test_available_tools_empty(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend()
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )
        assert env.available_tools == ()


# ---------------------------------------------------------------------------
# DIALOGUE_PRESETS tests
# ---------------------------------------------------------------------------


class TestDialoguePresets:
    def test_twenty_questions_preset(self):
        from llenvs.adapters.dialogue import DIALOGUE_PRESETS

        preset = DIALOGUE_PRESETS["twenty_questions"]
        assert "system_prompt_template" in preset
        assert "default_max_steps" in preset
        assert preset["default_max_steps"] == 20
        assert "is_terminal" in preset

    def test_teacher_preset(self):
        from llenvs.adapters.dialogue import DIALOGUE_PRESETS

        preset = DIALOGUE_PRESETS["teacher"]
        assert "system_prompt_template" in preset
        assert "default_max_steps" in preset
        assert preset["is_terminal"] is None

    def test_twenty_questions_terminal_correct(self):
        from llenvs.adapters.dialogue import DIALOGUE_PRESETS

        is_terminal = DIALOGUE_PRESETS["twenty_questions"]["is_terminal"]
        assert is_terminal("Correct! The word was cat.", "", 5) is True
        assert is_terminal("No, that's not right.", "", 5) is False


# ---------------------------------------------------------------------------
# Task creation helpers
# ---------------------------------------------------------------------------


class TestTaskCreation:
    def test_from_words(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend()
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "twenty_questions", env_llm=backend, words=["cat", "dog"]
        )
        assert len(env) == 2

    def test_from_questions(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend()
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "teacher",
            env_llm=backend,
            questions=[
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"},
            ],
        )
        assert len(env) == 2

    def test_from_direct_tasks(self):
        from llenvs.adapters.dialogue import DialogueAdapter, DialogueTask

        backend = _mock_backend()
        adapter = DialogueAdapter()
        tasks = [
            {"prompt": "P1", "context": "C1", "ground_truth": "GT1"},
            {"prompt": "P2"},
        ]
        env = adapter.get_environment("twenty_questions", env_llm=backend, tasks=tasks)
        assert len(env) == 2

    def test_words_creates_correct_tasks(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend()
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "twenty_questions", env_llm=backend, words=["cat"]
        )
        state, _ = env.reset(options={"task_index": 0})
        # Context should mention the secret word
        assert "cat" in state.hidden.task.context
        assert state.hidden.task.ground_truth == "cat"


# ---------------------------------------------------------------------------
# DialogueAdapter tests
# ---------------------------------------------------------------------------


class TestDialogueAdapter:
    def test_name(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        adapter = DialogueAdapter()
        assert adapter.name == "dialogue"

    def test_list_environments(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        adapter = DialogueAdapter()
        envs = adapter.list_environments()
        assert "twenty_questions" in envs
        assert "teacher" in envs

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        adapter = DialogueAdapter()
        assert adapter.get_native_answer_extractor("any") is None

    def test_get_default_system_prompt(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        adapter = DialogueAdapter()
        assert adapter.get_default_system_prompt("any") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        adapter = DialogueAdapter()
        assert adapter.get_prompt_template("any") is None

    def test_get_environment_info(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        adapter = DialogueAdapter()
        info = adapter.get_environment_info("twenty_questions")
        assert info["adapter"] == "dialogue"

    def test_custom_max_steps(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend()
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "twenty_questions",
            env_llm=backend,
            words=["cat"],
            max_steps=5,
        )
        assert env.spec.max_steps == 5

    def test_custom_system_prompt(self):
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend("No.")
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "twenty_questions",
            env_llm=backend,
            words=["cat"],
            system_prompt="Custom prompt. {context}",
        )
        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("Q?"))

        messages = backend.generate_chat.call_args[0][0]
        assert "Custom prompt." in messages[0].content


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestEnvironmentLLMConfig:
    def test_creation(self):
        from llenvs.core.config import EnvironmentLLMConfig, ModelConfig

        config = EnvironmentLLMConfig(
            model=ModelConfig(backend="openai", model="gpt-4o-mini")
        )
        assert config.model.backend == "openai"
        assert config.system_prompt == ""
        assert config.inference is None

    def test_with_inference(self):
        from llenvs.core.config import (
            EnvironmentLLMConfig,
            ModelConfig,
            InferenceConfig,
        )

        config = EnvironmentLLMConfig(
            model=ModelConfig(backend="anthropic", model="claude-sonnet-4-20250514"),
            system_prompt="You are an oracle.",
            inference=InferenceConfig(temperature=0.5, max_tokens=256),
        )
        assert config.system_prompt == "You are an oracle."
        assert config.inference.temperature == 0.5
        assert config.inference.max_tokens == 256

    def test_from_dict_env_llm(self):
        from llenvs.core.config import EvalConfig

        data = {
            "environments": [
                {
                    "name": "twenty_questions",
                    "adapter": "dialogue",
                    "params": {"words": ["cat"]},
                    "env_llm": {
                        "model": {"backend": "openai", "model": "gpt-4o-mini"},
                        "system_prompt": "Oracle prompt",
                        "inference": {
                            "temperature": 0.0,
                            "max_tokens": 256,
                        },
                    },
                }
            ],
            "model": {"backend": "openai", "model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)
        env_config = config.environments[0]
        assert env_config.env_llm is not None
        assert env_config.env_llm.model.backend == "openai"
        assert env_config.env_llm.model.model == "gpt-4o-mini"
        assert env_config.env_llm.system_prompt == "Oracle prompt"
        assert env_config.env_llm.inference.max_tokens == 256

    def test_from_dict_no_env_llm(self):
        from llenvs.core.config import EvalConfig

        data = {
            "environments": [
                {"name": "sudoku", "adapter": "reasoning_gym"}
            ],
            "model": {"backend": "openai", "model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)
        assert config.environments[0].env_llm is None

    def test_to_dict_with_env_llm(self):
        from llenvs.core.config import (
            EvalConfig,
            EnvironmentConfig,
            EnvironmentLLMConfig,
            ModelConfig,
            InferenceConfig,
        )

        env_config = EnvironmentConfig(
            name="twenty_questions",
            adapter="dialogue",
            env_llm=EnvironmentLLMConfig(
                model=ModelConfig(backend="openai", model="gpt-4o-mini"),
                system_prompt="Oracle",
                inference=InferenceConfig(temperature=0.0, max_tokens=256),
            ),
        )
        config = EvalConfig(
            environments=[env_config],
            model=ModelConfig(backend="openai", model="gpt-4o"),
        )
        d = config.to_dict()
        env_d = d["environments"][0]
        assert "env_llm" in env_d
        assert env_d["env_llm"]["model"]["backend"] == "openai"
        assert env_d["env_llm"]["system_prompt"] == "Oracle"
        assert env_d["env_llm"]["inference"]["max_tokens"] == 256

    def test_to_dict_without_env_llm(self):
        from llenvs.core.config import EvalConfig, EnvironmentConfig, ModelConfig

        config = EvalConfig(
            environments=[EnvironmentConfig(name="sudoku")],
            model=ModelConfig(backend="openai", model="gpt-4o"),
        )
        d = config.to_dict()
        assert "env_llm" not in d["environments"][0]

    def test_round_trip(self):
        from llenvs.core.config import EvalConfig

        data = {
            "environments": [
                {
                    "name": "twenty_questions",
                    "adapter": "dialogue",
                    "params": {"words": ["cat"]},
                    "env_llm": {
                        "model": {"backend": "openai", "model": "gpt-4o-mini"},
                        "system_prompt": "Oracle",
                        "inference": {
                            "temperature": 0.0,
                            "max_tokens": 256,
                        },
                    },
                }
            ],
            "model": {"backend": "openai", "model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)
        d = config.to_dict()
        config2 = EvalConfig.from_dict(d)
        assert config2.environments[0].env_llm is not None
        assert config2.environments[0].env_llm.model.model == "gpt-4o-mini"
        assert config2.environments[0].env_llm.system_prompt == "Oracle"

    def test_from_dict_env_llm_minimal(self):
        """env_llm with just model, no system_prompt/inference."""
        from llenvs.core.config import EvalConfig

        data = {
            "environments": [
                {
                    "name": "twenty_questions",
                    "adapter": "dialogue",
                    "env_llm": {
                        "model": {"backend": "openai", "model": "gpt-4o-mini"},
                    },
                }
            ],
            "model": {"backend": "openai", "model": "gpt-4o"},
        }
        config = EvalConfig.from_dict(data)
        env_config = config.environments[0]
        assert env_config.env_llm is not None
        assert env_config.env_llm.system_prompt == ""
        assert env_config.env_llm.inference is None


# ---------------------------------------------------------------------------
# Full episode test
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_twenty_questions_game(self):
        """Simulate a 3-turn 20-questions game ending with correct guess."""
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend_sequence(["No.", "No.", "Correct! The word was cat."])
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "twenty_questions", env_llm=backend, words=["cat"]
        )

        state, info = env.reset(options={"task_index": 0})
        assert state.hidden.task.ground_truth == "cat"

        # Turn 1
        result = env.step(state, Action.from_text("Is it an animal?"))
        assert result.done is False
        assert result.next_state.hidden.step_count == 1

        # Turn 2
        result = env.step(result.next_state, Action.from_text("Is it a dog?"))
        assert result.done is False
        assert result.next_state.hidden.step_count == 2

        # Turn 3 - correct guess
        result = env.step(result.next_state, Action.from_text("Is it a cat?"))
        assert result.terminated is True
        assert result.next_state.hidden.step_count == 3
        assert len(result.next_state.observation.messages) == 6  # 3 pairs

    def test_teacher_session(self):
        """Simulate a 2-turn teacher session."""
        from llenvs.adapters.dialogue import DialogueAdapter

        backend = _mock_backend_sequence([
            "Good attempt, but the answer should be x^3/3 + C. Try again.",
            "That's correct! x^3/3 + C is right.",
        ])
        adapter = DialogueAdapter()
        env = adapter.get_environment(
            "teacher",
            env_llm=backend,
            questions=[{"question": "Integrate x^2", "answer": "x^3/3 + C"}],
        )

        state, _ = env.reset(options={"task_index": 0})
        assert "Integrate" in state.observation.prompt

        # Student's first attempt
        r1 = env.step(state, Action.from_text("x^2/2"))
        assert r1.done is False

        # Student's corrected answer
        r2 = env.step(r1.next_state, Action.from_text("x^3/3 + C"))
        assert r2.done is False  # teacher preset has no is_terminal, only truncation


# ---------------------------------------------------------------------------
# compute_rewards test
# ---------------------------------------------------------------------------


class TestComputeRewards:
    def test_compute_rewards_directly(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Response")
        dummy = DummyReward(0.9, "direct_test")
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            extra_rewards=(dummy,),
        )

        state, _ = env.reset(options={"task_index": 0})
        bundle = env.compute_rewards(state, Action.from_text("A"), state)
        assert len(bundle.signals) == 1
        assert bundle.signals[0].value == 0.9

    def test_empty_rewards(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("Response")
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )

        state, _ = env.reset(options={"task_index": 0})
        bundle = env.compute_rewards(state, Action.from_text("A"), state)
        assert len(bundle.signals) == 0
        assert bundle.total == 0.0


# ---------------------------------------------------------------------------
# Sampling params test
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_default_sampling_params(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("OK")
        env = DialogueEnvironment(
            backend=backend, tasks=(DialogueTask(prompt="Q"),)
        )

        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("A"))

        # Check that sampling params were passed
        params = backend.generate_chat.call_args[0][1]
        assert isinstance(params, SamplingParams)
        assert params.temperature == 0.0
        assert params.max_tokens == 512

    def test_custom_sampling_params(self):
        from llenvs.adapters.dialogue import DialogueTask, DialogueEnvironment

        backend = _mock_backend("OK")
        custom_params = SamplingParams(temperature=0.7, max_tokens=256)
        env = DialogueEnvironment(
            backend=backend,
            tasks=(DialogueTask(prompt="Q"),),
            sampling_params=custom_params,
        )

        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("A"))

        params = backend.generate_chat.call_args[0][1]
        assert params.temperature == 0.7
        assert params.max_tokens == 256
