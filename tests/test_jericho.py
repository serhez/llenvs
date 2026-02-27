"""Tests for the Jericho adapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenvs.adapters.jericho import (
    DEFAULT_JERICHO_PROMPTS,
    JerichoAdapter,
    JerichoEnvironment,
    JerichoHidden,
    JerichoReward,
    _game_name_from_path,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, Observation, ObservationContent

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockFrotzEnv:
    """Mock Jericho FrotzEnv for testing.

    Simulates a simple text adventure with rooms, items, and scoring.
    """

    def __init__(self, game_file: str = "zork1.z5"):
        self._game_file = game_file
        self._step_count = 0
        self._score = 0
        self._max_score = 350
        self._done = False
        self._closed = False
        self._seeded = False
        self._seed_value: int | None = None

    def reset(self) -> str:
        self._step_count = 0
        self._score = 0
        self._done = False
        return (
            "ZORK I: The Great Underground Empire\n"
            "West of House\n"
            "You are standing in an open field west of a white house, "
            "with a boarded front door.\n"
            "There is a small mailbox here."
        )

    def step(self, action: str) -> tuple[str, int, bool, dict[str, Any]]:
        self._step_count += 1

        if action == "open mailbox":
            obs = "Opening the small mailbox reveals a leaflet."
            self._score = 0
            return obs, 0, False, {}

        elif action == "take leaflet":
            obs = "Taken."
            self._score = 5
            return obs, 5, False, {}

        elif action == "go north":
            obs = "North of House\nYou are facing the north side of a white house."
            self._score = self._score  # no score change
            return obs, 0, False, {}

        elif action == "win game":
            obs = "Congratulations! You have won!"
            self._score = self._max_score
            self._done = True
            return obs, self._max_score - (self._score - self._max_score + self._score), True, {}

        elif action == "die":
            obs = "You have died."
            self._done = True
            return obs, 0, True, {}

        else:
            obs = "I don't understand that."
            return obs, 0, False, {}

    def get_valid_actions(self) -> list[str]:
        return ["open mailbox", "go north", "go south", "look"]

    def get_score(self) -> int:
        return self._score

    def get_max_score(self) -> int:
        return self._max_score

    def get_moves(self) -> int:
        return self._step_count

    def victory(self) -> bool:
        return self._done and self._score == self._max_score

    def game_over(self) -> bool:
        return self._done

    def seed(self, seed: int) -> None:
        self._seeded = True
        self._seed_value = seed

    def close(self) -> None:
        self._closed = True


MOCK_GAME_FILES = (
    "/games/zork1.z5",
    "/games/detective.z5",
    "/games/hhgg.z3",
    "/games/planetfall.z5",
    "/games/trinity.z5",
)

MOCK_GAME_NAMES = ("zork1", "detective", "hhgg", "planetfall", "trinity")


def _make_env(
    game_files: tuple[str, ...] = MOCK_GAME_FILES,
    game_names: tuple[str, ...] = MOCK_GAME_NAMES,
    mock_frotz: MockFrotzEnv | None = None,
    **kwargs: Any,
) -> JerichoEnvironment:
    """Create a JerichoEnvironment with mocked _init_game."""
    env = JerichoEnvironment(
        game_files=game_files,
        game_names=game_names,
        **kwargs,
    )

    if mock_frotz is None:
        mock_frotz = MockFrotzEnv()

    def mock_init_game(game_file: str) -> tuple[str, dict[str, Any]]:
        env._frotz_env = mock_frotz
        obs = mock_frotz.reset()
        info = {
            "score": mock_frotz.get_score(),
            "max_score": mock_frotz.get_max_score(),
            "moves": mock_frotz.get_moves(),
            "done": False,
        }
        return obs, info

    env._init_game = mock_init_game  # type: ignore[assignment]
    return env


@pytest.fixture
def mock_frotz() -> MockFrotzEnv:
    """Create a mock Jericho FrotzEnv."""
    return MockFrotzEnv()


@pytest.fixture
def env(mock_frotz: MockFrotzEnv) -> JerichoEnvironment:
    """Create a test Jericho environment."""
    return _make_env(mock_frotz=mock_frotz)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestGameNameFromPath:
    """Tests for _game_name_from_path."""

    def test_z5_extension(self):
        assert _game_name_from_path("/games/zork1.z5") == "zork1"

    def test_z3_extension(self):
        assert _game_name_from_path("/games/hhgg.z3") == "hhgg"

    def test_z8_extension(self):
        assert _game_name_from_path("/games/curses.z8") == "curses"

    def test_no_extension(self):
        assert _game_name_from_path("/games/zork1") == "zork1"

    def test_nested_path(self):
        assert _game_name_from_path("/usr/share/jericho/games/detective.z5") == "detective"


# ---------------------------------------------------------------------------
# Hidden state tests
# ---------------------------------------------------------------------------


class TestJerichoHidden:
    """Tests for JerichoHidden state."""

    def test_creation(self):
        hidden = JerichoHidden(
            task_index=0,
            game_name="zork1",
            game_file="/games/zork1.z5",
            episode_step=2,
            last_action="go north",
            score=10,
            max_score=350,
            moves=2,
            valid_actions=("open mailbox", "go north"),
        )

        assert hidden.task_index == 0
        assert hidden.game_name == "zork1"
        assert hidden.game_file == "/games/zork1.z5"
        assert hidden.episode_step == 2
        assert hidden.last_action == "go north"
        assert hidden.score == 10
        assert hidden.max_score == 350
        assert hidden.moves == 2
        assert "open mailbox" in hidden.valid_actions

    def test_immutability(self):
        hidden = JerichoHidden(
            task_index=0,
            game_name="zork1",
            game_file="/games/zork1.z5",
            episode_step=0,
            last_action=None,
            score=0,
            max_score=350,
            moves=0,
            valid_actions=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore

    def test_last_action_none_on_reset(self):
        hidden = JerichoHidden(
            task_index=0,
            game_name="zork1",
            game_file="/games/zork1.z5",
            episode_step=0,
            last_action=None,
            score=0,
            max_score=350,
            moves=0,
            valid_actions=(),
        )
        assert hidden.last_action is None


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


class TestJerichoReward:
    """Tests for JerichoReward."""

    def test_reward_name(self):
        reward_fn = JerichoReward()
        assert reward_fn.name == "game_score"

    def test_step_reward_type(self):
        """Intermediate steps produce STEP rewards with score delta."""
        reward_fn = JerichoReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "score": 10,
            "max_score": 350,
            "score_delta": 5,
            "done": False,
        }
        next_state.metadata.is_terminal = False

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == 5
        assert signal.name == "game_score"

    def test_outcome_reward_on_terminal(self):
        """Terminal steps produce OUTCOME rewards with normalized score."""
        reward_fn = JerichoReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "score": 350,
            "max_score": 350,
            "score_delta": 10,
            "done": True,
        }
        next_state.metadata.is_terminal = True

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 1.0  # 350/350

    def test_outcome_reward_partial_score(self):
        """Terminal with partial score gives normalized value."""
        reward_fn = JerichoReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "score": 175,
            "max_score": 350,
            "score_delta": 0,
            "done": True,
        }
        next_state.metadata.is_terminal = True

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.5

    def test_step_reward_no_score_change(self):
        """Step with no score change gives reward=0."""
        reward_fn = JerichoReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "score": 10,
            "max_score": 350,
            "score_delta": 0,
            "done": False,
        }
        next_state.metadata.is_terminal = False

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 0

    def test_outcome_zero_max_score(self):
        """Terminal with max_score=0 gives reward=0.0 (avoid division by zero)."""
        reward_fn = JerichoReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {
            "score": 0,
            "max_score": 0,
            "score_delta": 0,
            "done": True,
        }
        next_state.metadata.is_terminal = True

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


class TestJerichoEnvironment:
    """Tests for JerichoEnvironment."""

    def test_creation(self, env: JerichoEnvironment):
        assert env.spec.name == "jericho"
        assert env.spec.adapter == "jericho"
        assert env.spec.is_multi_turn is True
        assert env.spec.pure_step is False
        assert env.spec.supports_task_index is True
        assert env.spec.supports_len is True
        assert env.spec.supports_seed is True
        assert env.spec.max_steps == 100

    def test_len(self, env: JerichoEnvironment):
        assert len(env) == len(MOCK_GAME_FILES)

    def test_available_tools_empty(self, env: JerichoEnvironment):
        assert env.available_tools == ()

    def test_reward_functions(self, env: JerichoEnvironment):
        rfs = env.reward_functions
        assert len(rfs) == 1
        assert rfs[0].name == "game_score"

    def test_reward_functions_with_extra(self, mock_frotz: MockFrotzEnv):
        extra = MagicMock()
        env = _make_env(mock_frotz=mock_frotz, extra_rewards=(extra,))
        rfs = env.reward_functions
        assert len(rfs) == 2
        assert rfs[1] is extra

    def test_reset(self, env: JerichoEnvironment):
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.game_name == "zork1"
        assert state.hidden.game_file == MOCK_GAME_FILES[0]
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None
        assert state.hidden.score == 0
        assert state.hidden.max_score == 350

        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0
        assert info["game_name"] == "zork1"

    def test_reset_default_task_index(self, env: JerichoEnvironment):
        state, info = env.reset()
        assert state.hidden.task_index == 0

    def test_reset_out_of_range(self, env: JerichoEnvironment):
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": 999})

    def test_reset_negative_index(self, env: JerichoEnvironment):
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": -1})

    def test_reset_with_seed(self, env: JerichoEnvironment, mock_frotz: MockFrotzEnv):
        env.reset(seed=42, options={"task_index": 0})
        assert mock_frotz._seeded is True
        assert mock_frotz._seed_value == 42

    def test_step_basic(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="open mailbox")
        result = env.step(state, action)

        assert result.terminated is False
        assert result.truncated is False
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "open mailbox"
        assert result.next_state.metadata.step == 1
        assert result.next_state.metadata.is_terminal is False

        # Jericho populates data with game state
        obs_state = result.next_state.observation.state
        assert obs_state is not None
        assert obs_state.data is not None
        assert "valid_actions" in obs_state.data
        assert "score" in obs_state.data
        assert "max_score" in obs_state.data
        assert "moves" in obs_state.data

    def test_step_score_update(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="take leaflet"))
        assert result.next_state.hidden.score == 5

    def test_step_updates_valid_actions(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        initial_actions = state.hidden.valid_actions
        assert len(initial_actions) > 0

        result = env.step(state, Action(text="open mailbox"))
        # Valid actions are re-fetched each step
        assert result.next_state.hidden.valid_actions is not None

    def test_step_game_over(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="die"))

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_truncation(self, mock_frotz: MockFrotzEnv):
        env = _make_env(mock_frotz=mock_frotz, max_steps=2)
        state, _ = env.reset(options={"task_index": 0})

        # Step 1
        result = env.step(state, Action(text="look"))
        assert result.truncated is False

        # Step 2 — truncation
        result = env.step(result.next_state, Action(text="look"))
        assert result.truncated is True
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is True

    def test_terminated_at_max_steps_not_truncated(self, mock_frotz: MockFrotzEnv):
        """If the game ends on the last possible step, terminated=True, truncated=False."""
        env = _make_env(mock_frotz=mock_frotz, max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="die"))
        assert result.terminated is True
        assert result.truncated is False

    def test_state_continuity_rejects_stale_state(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action(text="open mailbox"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state, Action(text="go north"))

    def test_valid_actions_not_in_observation_by_default(self, env: JerichoEnvironment):
        """By default, valid actions are NOT shown in observation (wrapper fidelity)."""
        state, _ = env.reset(options={"task_index": 0})
        assert "Valid actions:" not in state.observation.prompt

    def test_valid_actions_in_observation_when_enabled(self, mock_frotz: MockFrotzEnv):
        env = _make_env(mock_frotz=mock_frotz, include_valid_actions=True)
        state, _ = env.reset(options={"task_index": 0})
        assert "Valid actions:" in state.observation.prompt
        assert "open mailbox" in state.observation.prompt

    def test_valid_actions_always_in_hidden(self, env: JerichoEnvironment):
        """Valid actions are always stored in hidden state regardless of obs setting."""
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.hidden.valid_actions) > 0

    def test_game_file_in_hidden(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.game_file == MOCK_GAME_FILES[0]

    def test_spec_metadata(self, env: JerichoEnvironment):
        assert env.spec.metadata["num_games"] == len(MOCK_GAME_FILES)

    def test_close(self, env: JerichoEnvironment, mock_frotz: MockFrotzEnv):
        env.reset(options={"task_index": 0})
        env.close()
        assert mock_frotz._closed is True
        assert env._frotz_env is None

    def test_close_when_no_env(self, env: JerichoEnvironment):
        """close() is safe to call even before reset."""
        env.close()  # Should not raise

    def test_different_task_indices(self, mock_frotz: MockFrotzEnv):
        """Resetting with different task indices selects different game files."""
        env = _make_env(mock_frotz=mock_frotz)
        state0, _ = env.reset(options={"task_index": 0})
        assert state0.hidden.game_file == MOCK_GAME_FILES[0]
        assert state0.hidden.game_name == "zork1"

        state1, _ = env.reset(options={"task_index": 1})
        assert state1.hidden.game_file == MOCK_GAME_FILES[1]
        assert state1.hidden.game_name == "detective"

    def test_step_obs_in_messages(self, env: JerichoEnvironment):
        """Step observations go into messages, not prompt."""
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="open mailbox"))

        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "leaflet" in last_msg

    def test_valid_actions_in_step_obs_when_enabled(self, mock_frotz: MockFrotzEnv):
        """Valid actions appear in step observations when enabled."""
        env = _make_env(mock_frotz=mock_frotz, include_valid_actions=True)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="open mailbox"))

        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "Valid actions:" in last_msg

    def test_step_info_contains_score(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="take leaflet"))
        assert "score" in result.info
        assert "score_delta" in result.info

    def test_custom_max_steps(self, mock_frotz: MockFrotzEnv):
        env = _make_env(mock_frotz=mock_frotz, max_steps=10)
        assert env.spec.max_steps == 10

    def test_score_tracking_across_steps(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.score == 0

        result = env.step(state, Action(text="take leaflet"))
        assert result.next_state.hidden.score == 5

        result2 = env.step(result.next_state, Action(text="go north"))
        # Score unchanged on a move
        assert result2.next_state.hidden.score == 5


# ---------------------------------------------------------------------------
# Message history tests
# ---------------------------------------------------------------------------


class TestJerichoMessageHistory:
    """Tests for message history accumulation."""

    def test_initial_messages_empty(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.messages == ()

    def test_messages_accumulate(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        initial_prompt = state.observation.prompt

        result = env.step(state, Action(text="open mailbox"))
        state = result.next_state
        assert len(state.observation.messages) == 2
        assert state.observation.messages[0] == {"role": "assistant", "content": "open mailbox"}
        assert state.observation.prompt == initial_prompt

        result = env.step(state, Action(text="go north"))
        state = result.next_state
        assert len(state.observation.messages) == 4
        assert state.observation.messages[2] == {"role": "assistant", "content": "go north"}
        assert state.observation.prompt == initial_prompt

    def test_messages_on_terminal(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="die"))

        assert result.terminated is True
        assert len(result.next_state.observation.messages) == 2
        assert result.next_state.observation.messages[0] == {"role": "assistant", "content": "die"}


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestJerichoPrompts:
    """Tests for configurable prompt components."""

    def test_default_prompts(self, env: JerichoEnvironment):
        prompts = env.prompts
        assert "valid_actions_prefix" in prompts

    def test_default_prompts_match_constants(self, env: JerichoEnvironment):
        assert env.prompts == DEFAULT_JERICHO_PROMPTS

    def test_prompts_returns_copy(self, env: JerichoEnvironment):
        p1 = env.prompts
        p2 = env.prompts
        assert p1 == p2
        assert p1 is not p2

    def test_custom_valid_actions_prefix(self, mock_frotz: MockFrotzEnv):
        custom = {"valid_actions_prefix": "Available commands:"}
        env = _make_env(mock_frotz=mock_frotz, include_valid_actions=True, prompts=custom)
        state, _ = env.reset(options={"task_index": 0})

        assert "Available commands:" in state.observation.prompt
        assert "Valid actions:" not in state.observation.prompt

    def test_custom_prompts_merge(self, mock_frotz: MockFrotzEnv):
        """Custom prompts only override specified keys."""
        custom = {"valid_actions_prefix": "Commands:"}
        env = _make_env(mock_frotz=mock_frotz, prompts=custom)
        prompts = env.prompts

        assert prompts["valid_actions_prefix"] == "Commands:"


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestJerichoAdapter:
    """Tests for JerichoAdapter."""

    def test_adapter_name(self):
        adapter = JerichoAdapter()
        assert adapter.name == "jericho"

    def test_list_environments(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            with patch(
                "llenvs.adapters.jericho._list_bundled_games",
                return_value={"zork1": "/games/zork1.z5", "detective": "/games/detective.z5"},
            ):
                envs = adapter.list_environments()

            assert "jericho:zork1" in envs
            assert "jericho:detective" in envs

    def test_get_environment_specific_game(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            with patch(
                "llenvs.adapters.jericho._list_bundled_games",
                return_value={"zork1": "/games/zork1.z5"},
            ):
                env = adapter.get_environment(name="jericho:zork1", max_steps=50)

            assert isinstance(env, JerichoEnvironment)
            assert len(env) == 1
            assert env.spec.max_steps == 50

    def test_get_environment_all_games(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            with patch(
                "llenvs.adapters.jericho._list_bundled_games",
                return_value={
                    "zork1": "/games/zork1.z5",
                    "detective": "/games/detective.z5",
                },
            ):
                env = adapter.get_environment(name="jericho")

            assert isinstance(env, JerichoEnvironment)
            assert len(env) == 2

    def test_get_environment_with_games_list(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            with patch(
                "llenvs.adapters.jericho._list_bundled_games",
                return_value={
                    "zork1": "/games/zork1.z5",
                    "detective": "/games/detective.z5",
                    "hhgg": "/games/hhgg.z3",
                },
            ):
                env = adapter.get_environment(games=["zork1", "detective"])

            assert isinstance(env, JerichoEnvironment)
            assert len(env) == 2

    def test_get_environment_with_game_files(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            env = adapter.get_environment(game_files=["/custom/path/mygame.z5"])

            assert isinstance(env, JerichoEnvironment)
            assert len(env) == 1

    def test_get_environment_unknown_game_name(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            with patch(
                "llenvs.adapters.jericho._list_bundled_games",
                return_value={"zork1": "/games/zork1.z5"},
            ):
                with pytest.raises(ValueError, match="Unknown game"):
                    adapter.get_environment(name="jericho:nonexistent")

    def test_get_environment_unknown_game_in_games_list(self):
        adapter = JerichoAdapter()
        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_jericho = MagicMock()
            mock_get.return_value = mock_jericho

            with patch(
                "llenvs.adapters.jericho._list_bundled_games",
                return_value={"zork1": "/games/zork1.z5"},
            ):
                with pytest.raises(ValueError, match="Unknown game"):
                    adapter.get_environment(games=["nonexistent"])

    def test_get_default_system_prompt_none(self):
        adapter = JerichoAdapter()
        assert adapter.get_default_system_prompt("jericho") is None

    def test_get_prompt_template_none(self):
        adapter = JerichoAdapter()
        assert adapter.get_prompt_template("jericho") is None

    def test_get_native_answer_extractor_none(self):
        adapter = JerichoAdapter()
        assert adapter.get_native_answer_extractor("jericho") is None

    def test_get_environment_info(self):
        adapter = JerichoAdapter()
        info = adapter.get_environment_info()

        assert info["name"] == "jericho"
        assert info["adapter"] == "jericho"
        assert info["type"] == "multi_turn"
        assert "reference" in info

    def test_import_error(self):
        adapter = JerichoAdapter()
        with patch.dict("sys.modules", {"jericho": None}):
            with pytest.raises(ImportError, match="Jericho is required"):
                adapter._get_jericho()


# ---------------------------------------------------------------------------
# Full episode integration tests
# ---------------------------------------------------------------------------


class TestJerichoFullEpisode:
    """Integration tests for multi-step episodes."""

    def test_full_episode_with_scoring(self, env: JerichoEnvironment):
        state, info = env.reset(options={"task_index": 0})

        assert state.metadata.step == 0
        assert state.hidden.score == 0

        # task = synthetic description, state = game text
        assert isinstance(state.observation.task, ObservationContent)
        assert "zork1" in state.observation.task.text.lower()
        assert "interactive fiction" in state.observation.task.text.lower()
        assert isinstance(state.observation.state, ObservationContent)
        assert state.observation.state.text == state.observation.prompt
        # task and state are distinct
        assert state.observation.task.text != state.observation.state.text
        reset_task = state.observation.task

        # Explore and earn points
        actions = ["open mailbox", "take leaflet", "go north"]

        for i, act_text in enumerate(actions):
            result = env.step(state, Action(text=act_text))
            state = result.next_state
            assert state.metadata.step == i + 1
            assert state.hidden.episode_step == i + 1
            assert state.hidden.last_action == act_text

            # task carried forward, state updated to step obs
            assert state.observation.task is reset_task
            assert isinstance(state.observation.state, ObservationContent)
            step_obs = state.observation.messages[-1]["content"]
            assert state.observation.state.text == step_obs

        assert state.hidden.score == 5  # only "take leaflet" gives points

    def test_game_over_episode(self, env: JerichoEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="die"))
        assert result.terminated is True

        # Check OUTCOME reward
        signal = result.rewards.by_name("game_score")
        assert signal is not None
        assert signal.reward_type == RewardType.OUTCOME

    def test_truncation_episode(self, mock_frotz: MockFrotzEnv):
        env = _make_env(mock_frotz=mock_frotz, max_steps=3)
        state, _ = env.reset(options={"task_index": 0})

        for act in ["open mailbox", "go north", "look"]:
            result = env.step(state, Action(text=act))
            state = result.next_state

        assert result.truncated is True
        assert result.terminated is False

        signal = result.rewards.by_name("game_score")
        assert signal.reward_type == RewardType.OUTCOME

    def test_step_reward_score_delta(self, env: JerichoEnvironment):
        """Intermediate steps give STEP rewards with score delta."""
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="take leaflet"))
        signal = result.rewards.by_name("game_score")
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == 5  # score went from 0 to 5

    def test_step_reward_zero_delta(self, env: JerichoEnvironment):
        """Step with no score change gives 0 reward."""
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="open mailbox"))
        signal = result.rewards.by_name("game_score")
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == 0


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestJerichoRegistration:
    """Tests for adapter registration."""

    def test_adapter_registers_when_available(self):
        """Adapter registers if jericho is importable."""
        adapter = JerichoAdapter()

        with patch.object(adapter, "_get_jericho") as mock_get:
            mock_get.return_value = MagicMock()
            adapter._get_jericho()
            mock_get.assert_called_once()

    def test_adapter_skipped_when_not_installed(self):
        """Adapter is silently skipped when jericho is not installed."""
        adapter = JerichoAdapter()

        with patch.object(adapter, "_get_jericho", side_effect=ImportError("no jericho")):
            with pytest.raises(ImportError):
                adapter._get_jericho()
