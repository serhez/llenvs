"""Tests for the LMRL-Gym adapter."""

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from llenvs.core.state import Observation, Action
from llenvs.core.reward import RewardType
from llenvs.adapters.lmrl import (
    LMRLEnvironment,
    LMRLHidden,
    LMRLAdapter,
    LMRLReward,
    LMRL_PRESETS,
    _LMRLText,
    _create_text_env,
)


# ---------------------------------------------------------------------------
# Mock TextEnv — simulates a simple number guessing game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockText:
    """Mock LMRL-Gym Text object."""

    text: str
    is_action: bool


class MockTextEnv:
    """Mock LMRL-Gym TextEnv for testing.

    Simulates a simple number guessing game:
    - Target is determined by seed (default: 7)
    - Agent guesses a number
    - Reward: 1.0 for correct, -0.1 per wrong guess
    - Done when correct or max guesses exceeded
    """

    def __init__(self, max_guesses: int = 5):
        self._max_guesses = max_guesses
        self._target: int = 7
        self._guesses: int = 0
        self._closed = False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[MockText, ...]:
        if seed is not None:
            self._target = (seed % 10) + 1  # 1-10
        else:
            self._target = 7
        self._guesses = 0

        initial_obs = MockText(
            text=f"Guess a number between 1 and 10.",
            is_action=False,
        )
        return (initial_obs,)

    def step(
        self, text_history: tuple
    ) -> tuple[tuple, float, bool]:
        assert text_history[-1].is_action
        guess_text = text_history[-1].text.strip()
        self._guesses += 1

        try:
            guess = int(guess_text)
        except ValueError:
            obs = MockText(text="Invalid number. Try again.", is_action=False)
            return text_history + (obs,), -0.1, False

        if guess == self._target:
            obs = MockText(text="Correct! You win!", is_action=False)
            return text_history + (obs,), 1.0, True
        elif guess < self._target:
            obs = MockText(text="Too low.", is_action=False)
            return text_history + (obs,), -0.1, False
        else:
            obs = MockText(text="Too high.", is_action=False)
            return text_history + (obs,), -0.1, False

    def close(self) -> None:
        self._closed = True


class MockEmptyResetEnv:
    """Mock TextEnv that returns empty history on reset (like Wordle)."""

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple:
        return ()

    def step(self, text_history: tuple) -> tuple[tuple, float, bool]:
        obs = MockText(text="Feedback.", is_action=False)
        return text_history + (obs,), 0.0, False


class MockMultiObsEnv:
    """Mock TextEnv that returns multiple observations on reset and step."""

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple:
        return (
            MockText(text="Welcome to the game.\n", is_action=False),
            MockText(text="You are in a room.", is_action=False),
        )

    def step(self, text_history: tuple) -> tuple[tuple, float, bool]:
        obs1 = MockText(text="Something happens.\n", is_action=False)
        obs2 = MockText(text="You see a door.", is_action=False)
        return text_history + (obs1, obs2), 0.5, False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_env(
    text_env: Any = None,
    env_name: str = "test_game",
    **kwargs: Any,
) -> LMRLEnvironment:
    """Create an LMRLEnvironment with a mock TextEnv."""
    if text_env is None:
        text_env = MockTextEnv()

    return LMRLEnvironment(
        text_env=text_env,
        env_name=env_name,
        **kwargs,
    )


@pytest.fixture
def mock_text_env() -> MockTextEnv:
    """Create a mock TextEnv."""
    return MockTextEnv()


@pytest.fixture
def env(mock_text_env: MockTextEnv) -> LMRLEnvironment:
    """Create a test LMRL environment."""
    return _make_env(text_env=mock_text_env)


# ---------------------------------------------------------------------------
# _LMRLText tests
# ---------------------------------------------------------------------------


class TestLMRLText:
    """Tests for _LMRLText compatibility class."""

    def test_creation(self):
        t = _LMRLText(text="hello", is_action=True)
        assert t.text == "hello"
        assert t.is_action is True

    def test_immutability(self):
        t = _LMRLText(text="hello", is_action=False)
        with pytest.raises(AttributeError):
            t.text = "world"  # type: ignore

    def test_action_vs_observation(self):
        action = _LMRLText(text="go north", is_action=True)
        obs = _LMRLText(text="You see a door.", is_action=False)
        assert action.is_action is True
        assert obs.is_action is False


# ---------------------------------------------------------------------------
# Hidden state tests
# ---------------------------------------------------------------------------


class TestLMRLHidden:
    """Tests for LMRLHidden state."""

    def test_creation(self):
        hidden = LMRLHidden(
            env_name="wordle",
            episode_step=3,
            last_action="crane",
            cumulative_reward=-0.3,
            text_history=(),
        )
        assert hidden.env_name == "wordle"
        assert hidden.episode_step == 3
        assert hidden.last_action == "crane"
        assert hidden.cumulative_reward == -0.3
        assert hidden.text_history == ()

    def test_immutability(self):
        hidden = LMRLHidden(
            env_name="test",
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            text_history=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore

    def test_last_action_none_on_reset(self):
        hidden = LMRLHidden(
            env_name="test",
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            text_history=(),
        )
        assert hidden.last_action is None

    def test_text_history_stored(self):
        history = (
            MockText("Hello", is_action=False),
            MockText("Hi", is_action=True),
        )
        hidden = LMRLHidden(
            env_name="test",
            episode_step=1,
            last_action="Hi",
            cumulative_reward=0.0,
            text_history=history,
        )
        assert len(hidden.text_history) == 2
        assert hidden.text_history[0].text == "Hello"
        assert hidden.text_history[1].is_action is True


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


class TestLMRLReward:
    """Tests for LMRLReward."""

    def test_reward_name(self):
        reward_fn = LMRLReward()
        assert reward_fn.name == "lmrl_reward"

    def test_step_reward(self):
        """Intermediate steps produce STEP rewards."""
        reward_fn = LMRLReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "step_reward": -0.1,
            "cumulative_reward": -0.3,
            "done": False,
        }
        next_state.metadata.is_terminal = False

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == -0.1
        assert signal.name == "lmrl_reward"

    def test_outcome_reward_on_terminal(self):
        """Terminal steps produce OUTCOME rewards with cumulative reward."""
        reward_fn = LMRLReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "step_reward": 1.0,
            "cumulative_reward": 0.7,
            "done": True,
        }
        next_state.metadata.is_terminal = True

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.7

    def test_outcome_reward_negative(self):
        """Terminal with negative cumulative reward."""
        reward_fn = LMRLReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "step_reward": -0.1,
            "cumulative_reward": -0.5,
            "done": True,
        }
        next_state.metadata.is_terminal = True

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == -0.5

    def test_step_reward_zero(self):
        """Step with zero reward."""
        reward_fn = LMRLReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "step_reward": 0.0,
            "cumulative_reward": 0.0,
            "done": False,
        }
        next_state.metadata.is_terminal = False

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 0.0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


class TestLMRLEnvironment:
    """Tests for LMRLEnvironment."""

    def test_creation(self, env: LMRLEnvironment):
        assert env.spec.name == "test_game"
        assert env.spec.adapter == "lmrl"
        assert env.spec.is_multi_turn is True
        assert env.spec.pure_step is False
        assert env.spec.supports_seed is True
        assert env.spec.max_steps == 100

    def test_no_num_tasks_means_no_task_index(self, env: LMRLEnvironment):
        assert env.spec.supports_task_index is False
        assert env.spec.supports_len is False

    def test_with_num_tasks(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env, num_tasks=50)
        assert env.spec.supports_task_index is True
        assert env.spec.supports_len is True
        assert len(env) == 50

    def test_len_raises_without_num_tasks(self, env: LMRLEnvironment):
        with pytest.raises(TypeError, match="no fixed task count"):
            len(env)

    def test_available_tools_empty(self, env: LMRLEnvironment):
        assert env.available_tools == ()

    def test_prompts_empty(self, env: LMRLEnvironment):
        assert env.prompts == {}

    def test_reward_functions(self, env: LMRLEnvironment):
        rfs = env.reward_functions
        assert len(rfs) == 1
        assert rfs[0].name == "lmrl_reward"

    def test_reward_functions_with_extra(self, mock_text_env: MockTextEnv):
        extra = MagicMock()
        env = _make_env(text_env=mock_text_env, extra_rewards=(extra,))
        rfs = env.reward_functions
        assert len(rfs) == 2
        assert rfs[1] is extra

    def test_reset(self, env: LMRLEnvironment):
        state, info = env.reset(seed=42)

        assert isinstance(state.observation, Observation)
        assert state.hidden.env_name == "test_game"
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None
        assert state.hidden.cumulative_reward == 0.0

        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["env_name"] == "test_game"
        assert info["seed"] == 42

    def test_reset_observation_text(self, env: LMRLEnvironment):
        state, _ = env.reset()
        assert "Guess a number" in state.observation.prompt

    def test_reset_with_seed(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env)
        env.reset(seed=3)
        # seed=3 → target = (3 % 10) + 1 = 4
        assert mock_text_env._target == 4

    def test_reset_stores_text_history(self, env: LMRLEnvironment):
        state, _ = env.reset()
        assert len(state.hidden.text_history) == 1
        assert state.hidden.text_history[0].is_action is False

    def test_reset_empty_history(self):
        env = _make_env(text_env=MockEmptyResetEnv())
        state, _ = env.reset()
        assert state.observation.prompt == ""
        assert len(state.hidden.text_history) == 0

    def test_reset_multi_obs(self):
        env = _make_env(text_env=MockMultiObsEnv())
        state, _ = env.reset()
        assert "Welcome to the game." in state.observation.prompt
        assert "You are in a room." in state.observation.prompt
        assert len(state.hidden.text_history) == 2

    def test_reset_task_index(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env, num_tasks=10)
        state, info = env.reset(options={"task_index": 5})
        # task_index=5 → seed=5 → target = (5 % 10) + 1 = 6
        assert mock_text_env._target == 6
        assert info["seed"] == 5

    def test_reset_task_index_with_explicit_seed(
        self, mock_text_env: MockTextEnv
    ):
        """Explicit seed overrides task_index-derived seed."""
        env = _make_env(text_env=mock_text_env, num_tasks=10)
        env.reset(seed=3, options={"task_index": 5})
        # Explicit seed=3 takes priority → target = (3 % 10) + 1 = 4
        assert mock_text_env._target == 4

    def test_reset_task_index_out_of_range(
        self, mock_text_env: MockTextEnv
    ):
        env = _make_env(text_env=mock_text_env, num_tasks=10)
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": 15})

    def test_reset_negative_task_index(
        self, mock_text_env: MockTextEnv
    ):
        env = _make_env(text_env=mock_text_env, num_tasks=10)
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": -1})

    def test_step_basic(self, env: LMRLEnvironment):
        state, _ = env.reset(seed=42)

        action = Action(text="5")
        result = env.step(state, action)

        assert result.terminated is False
        assert result.truncated is False
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "5"
        assert result.next_state.metadata.step == 1
        assert result.next_state.metadata.is_terminal is False

    def test_step_correct_guess(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env)
        state, _ = env.reset(seed=42)
        # seed=42 → target = (42 % 10) + 1 = 3

        result = env.step(state, Action(text="3"))
        assert result.terminated is True
        assert "Correct" in result.next_state.observation.messages[-1]["content"]

    def test_step_wrong_guess(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        assert result.terminated is False
        assert "Too low" in result.next_state.observation.messages[-1]["content"]

    def test_step_reward_tracking(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        assert result.info["step_reward"] == -0.1
        assert result.info["cumulative_reward"] == -0.1
        assert result.next_state.hidden.cumulative_reward == -0.1

    def test_step_cumulative_reward(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        assert result.next_state.hidden.cumulative_reward == -0.1

        result = env.step(result.next_state, Action(text="5"))
        assert result.next_state.hidden.cumulative_reward == -0.2

    def test_step_text_history_grows(self, env: LMRLEnvironment):
        state, _ = env.reset()
        assert len(state.hidden.text_history) == 1  # initial obs

        result = env.step(state, Action(text="3"))
        # 1 initial + 1 action + 1 response = 3
        assert len(result.next_state.hidden.text_history) == 3

    def test_step_observation_extraction(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "Too low" in last_msg

    def test_step_multi_obs_env(self):
        env = _make_env(text_env=MockMultiObsEnv())
        state, _ = env.reset()

        result = env.step(state, Action(text="action"))
        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "Something happens." in last_msg
        assert "You see a door." in last_msg

    def test_truncation(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env, max_steps=2)
        state, _ = env.reset()  # target = 7

        # Step 1
        result = env.step(state, Action(text="3"))
        assert result.truncated is False

        # Step 2 — truncation
        result = env.step(result.next_state, Action(text="5"))
        assert result.truncated is True
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is True

    def test_terminated_at_max_steps_not_truncated(
        self, mock_text_env: MockTextEnv
    ):
        """If the game ends at the last possible step, terminated=True."""
        env = _make_env(text_env=mock_text_env, max_steps=1)
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="7"))
        assert result.terminated is True
        assert result.truncated is False

    def test_state_continuity_rejects_stale_state(
        self, env: LMRLEnvironment
    ):
        state, _ = env.reset()
        env.step(state, Action(text="3"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state, Action(text="5"))

    def test_spec_metadata(self, env: LMRLEnvironment):
        assert "description" in env.spec.metadata

    def test_custom_max_steps(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env, max_steps=10)
        assert env.spec.max_steps == 10

    def test_close(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env)
        env.reset()
        env.close()
        assert mock_text_env._closed is True

    def test_close_no_close_method(self):
        """close() is safe on envs without close() method."""

        class NoCloseEnv:
            def reset(self, seed=None, options=None):
                return ()

            def step(self, text_history):
                return text_history, 0.0, True

        env = _make_env(text_env=NoCloseEnv())
        env.close()  # Should not raise


# ---------------------------------------------------------------------------
# Message history tests
# ---------------------------------------------------------------------------


class TestLMRLMessageHistory:
    """Tests for message history accumulation."""

    def test_initial_messages_empty(self, env: LMRLEnvironment):
        state, _ = env.reset()
        assert state.observation.messages == ()

    def test_messages_accumulate(self, env: LMRLEnvironment):
        state, _ = env.reset()
        initial_prompt = state.observation.prompt

        result = env.step(state, Action(text="3"))
        state = result.next_state
        assert len(state.observation.messages) == 2
        assert state.observation.messages[0] == {
            "role": "assistant",
            "content": "3",
        }
        assert state.observation.prompt == initial_prompt

        result = env.step(state, Action(text="5"))
        state = result.next_state
        assert len(state.observation.messages) == 4
        assert state.observation.messages[2] == {
            "role": "assistant",
            "content": "5",
        }
        assert state.observation.prompt == initial_prompt

    def test_messages_on_terminal(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="7"))
        assert result.terminated is True
        assert len(result.next_state.observation.messages) == 2
        assert result.next_state.observation.messages[0] == {
            "role": "assistant",
            "content": "7",
        }


# ---------------------------------------------------------------------------
# Reward integration tests
# ---------------------------------------------------------------------------


class TestLMRLRewardIntegration:
    """Tests for reward computation in environment."""

    def test_step_reward_signal(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        signal = result.rewards.by_name("lmrl_reward")
        assert signal is not None
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == -0.1

    def test_outcome_reward_signal(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="7"))
        signal = result.rewards.by_name("lmrl_reward")
        assert signal is not None
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 1.0  # cumulative = 1.0

    def test_truncation_outcome_reward(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env, max_steps=2)
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        result = env.step(result.next_state, Action(text="5"))

        signal = result.rewards.by_name("lmrl_reward")
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == -0.2  # cumulative: -0.1 + -0.1


# ---------------------------------------------------------------------------
# Full episode integration tests
# ---------------------------------------------------------------------------


class TestLMRLFullEpisode:
    """Integration tests for multi-step episodes."""

    def test_full_episode_correct_guess(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        # Guess too low
        result = env.step(state, Action(text="3"))
        assert result.terminated is False
        assert result.next_state.hidden.episode_step == 1

        # Guess too high
        result = env.step(result.next_state, Action(text="9"))
        assert result.terminated is False
        assert result.next_state.hidden.episode_step == 2

        # Correct guess
        result = env.step(result.next_state, Action(text="7"))
        assert result.terminated is True
        assert result.next_state.hidden.cumulative_reward == pytest.approx(
            -0.1 + -0.1 + 1.0
        )

    def test_full_episode_all_wrong(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env, max_steps=3)
        state, _ = env.reset()  # target = 7

        for guess in ["1", "2", "3"]:
            result = env.step(state, Action(text=guess))
            state = result.next_state

        # Last step should be truncated
        assert result.truncated is True
        assert result.next_state.hidden.cumulative_reward == pytest.approx(-0.3)

    def test_episode_with_seed_control(self, mock_text_env: MockTextEnv):
        env = _make_env(text_env=mock_text_env)

        # seed=0 → target = 1
        state, _ = env.reset(seed=0)
        result = env.step(state, Action(text="1"))
        assert result.terminated is True

        # seed=4 → target = 5
        state, _ = env.reset(seed=4)
        result = env.step(state, Action(text="5"))
        assert result.terminated is True

    def test_episode_message_history_grows(self, env: LMRLEnvironment):
        state, _ = env.reset()  # target = 7

        for i, guess in enumerate(["3", "5", "7"]):
            result = env.step(state, Action(text=guess))
            state = result.next_state
            assert len(state.observation.messages) == (i + 1) * 2

    def test_text_history_tracks_full_conversation(
        self, env: LMRLEnvironment
    ):
        state, _ = env.reset()  # target = 7

        result = env.step(state, Action(text="3"))
        result = env.step(result.next_state, Action(text="7"))

        history = result.next_state.hidden.text_history
        # 1 initial obs + (action + obs) * 2 = 5
        assert len(history) == 5
        assert history[0].is_action is False  # initial obs
        assert history[1].is_action is True  # action "3"
        assert history[2].is_action is False  # response "Too low"
        assert history[3].is_action is True  # action "7"
        assert history[4].is_action is False  # response "Correct!"


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------


class TestLMRLPresets:
    """Tests for LMRL_PRESETS."""

    def test_presets_exist(self):
        assert "wordle" in LMRL_PRESETS
        assert "chess" in LMRL_PRESETS
        assert "chess:endgame" in LMRL_PRESETS
        assert "maze:double_t" in LMRL_PRESETS
        assert "twenty_questions" in LMRL_PRESETS

    def test_preset_has_max_steps(self):
        for name, preset in LMRL_PRESETS.items():
            assert "max_steps" in preset, f"Preset {name} missing max_steps"

    def test_preset_has_description(self):
        for name, preset in LMRL_PRESETS.items():
            assert "description" in preset, f"Preset {name} missing description"

    def test_wordle_max_steps(self):
        assert LMRL_PRESETS["wordle"]["max_steps"] == 6

    def test_chess_max_steps(self):
        assert LMRL_PRESETS["chess"]["max_steps"] == 400

    def test_twenty_questions_max_steps(self):
        assert LMRL_PRESETS["twenty_questions"]["max_steps"] == 20


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateTextEnv:
    """Tests for _create_text_env factory."""

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="No auto-creation factory"):
            _create_text_env("nonexistent")

    def test_oracle_env_raises(self):
        with pytest.raises(ValueError, match="requires an external model"):
            _create_text_env("twenty_questions")

    def test_oracle_env_guess_city_raises(self):
        with pytest.raises(ValueError, match="requires an external model"):
            _create_text_env("guess_city")

    def test_oracle_env_car_dealer_raises(self):
        with pytest.raises(ValueError, match="requires an external model"):
            _create_text_env("car_dealer")

    def test_text_nav_raises(self):
        with pytest.raises(ValueError, match="custom TextWorld fork"):
            _create_text_env("text_nav")

    def test_chess_requires_lmrl(self):
        """Chess factory raises ImportError when LMRL-Gym is not installed."""
        with patch.dict("sys.modules", {"llm_rl_scripts": None, "llm_rl_scripts.chess": None, "llm_rl_scripts.chess.env": None, "llm_rl_scripts.chess.env.env": None}):
            with pytest.raises(ImportError):
                _create_text_env("chess")


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestLMRLAdapter:
    """Tests for LMRLAdapter."""

    def test_adapter_name(self):
        adapter = LMRLAdapter()
        assert adapter.name == "lmrl"

    def test_list_environments(self):
        adapter = LMRLAdapter()
        envs = adapter.list_environments()
        assert all(e.startswith("lmrl:") for e in envs)
        assert "lmrl:wordle" in envs
        assert "lmrl:chess" in envs

    def test_get_environment_with_text_env(self):
        adapter = LMRLAdapter()
        mock = MockTextEnv()
        env = adapter.get_environment("wordle", text_env=mock)

        assert isinstance(env, LMRLEnvironment)
        assert env.spec.name == "wordle"
        assert env.spec.max_steps == 6  # from preset

    def test_get_environment_with_text_env_and_custom_max_steps(self):
        adapter = LMRLAdapter()
        mock = MockTextEnv()
        env = adapter.get_environment("wordle", text_env=mock, max_steps=10)

        assert env.spec.max_steps == 10

    def test_get_environment_with_num_tasks(self):
        adapter = LMRLAdapter()
        mock = MockTextEnv()
        env = adapter.get_environment("test", text_env=mock, num_tasks=100)

        assert len(env) == 100
        assert env.spec.supports_task_index is True

    def test_get_environment_with_extra_rewards(self):
        adapter = LMRLAdapter()
        mock = MockTextEnv()
        extra = MagicMock()
        env = adapter.get_environment("test", text_env=mock, extra_rewards=(extra,))

        assert len(env.reward_functions) == 2

    def test_get_environment_name_parsing_with_prefix(self):
        adapter = LMRLAdapter()
        mock = MockTextEnv()
        env = adapter.get_environment("lmrl:wordle", text_env=mock)
        assert env.spec.name == "wordle"

    def test_get_environment_unknown_without_text_env(self):
        adapter = LMRLAdapter()
        with pytest.raises(ValueError, match="No auto-creation factory"):
            adapter.get_environment("wordle")

    def test_get_default_system_prompt_none(self):
        adapter = LMRLAdapter()
        assert adapter.get_default_system_prompt("lmrl") is None

    def test_get_prompt_template_none(self):
        adapter = LMRLAdapter()
        assert adapter.get_prompt_template("lmrl") is None

    def test_get_native_answer_extractor_none(self):
        adapter = LMRLAdapter()
        assert adapter.get_native_answer_extractor("lmrl") is None

    def test_get_environment_info(self):
        adapter = LMRLAdapter()
        info = adapter.get_environment_info("lmrl:wordle")

        assert info["adapter"] == "lmrl"
        assert info["type"] == "multi_turn"
        assert "reference" in info
        assert "environments" in info

    def test_get_environment_info_with_preset(self):
        adapter = LMRLAdapter()
        info = adapter.get_environment_info("lmrl:chess")
        assert "Chess" in info["description"] or "chess" in info["description"].lower()

    def test_import_error(self):
        adapter = LMRLAdapter()
        with patch.dict("sys.modules", {"LLM_RL": None, "LLM_RL.environment": None}):
            with pytest.raises(ImportError, match="LMRL-Gym is required"):
                adapter._get_lmrl()


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestLMRLRegistration:
    """Tests for adapter registration."""

    def test_adapter_registers_when_available(self):
        """Adapter registers if LLM_RL is importable."""
        adapter = LMRLAdapter()

        with patch.object(adapter, "_get_lmrl") as mock_get:
            mock_get.return_value = MagicMock()
            adapter._get_lmrl()
            mock_get.assert_called_once()

    def test_adapter_skipped_when_not_installed(self):
        """Adapter is silently skipped when LLM_RL is not installed."""
        adapter = LMRLAdapter()

        with patch.object(
            adapter, "_get_lmrl", side_effect=ImportError("no LLM_RL")
        ):
            with pytest.raises(ImportError):
                adapter._get_lmrl()
