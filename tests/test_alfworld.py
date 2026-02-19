"""Tests for the AlfWorld adapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llenvs.adapters.alfworld import (
    ALFWORLD_TASK_TYPES,
    DEFAULT_ALFWORLD_PROMPTS,
    AlfWorldAdapter,
    AlfWorldEnvironment,
    AlfWorldHidden,
    AlfWorldReward,
    _extract_objective,
    _extract_task_type,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, ImageContent, Observation

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockAlfWorldGymEnv:
    """Mock TextWorld gym environment for testing.

    Simulates the batched interface: reset/step return lists.
    """

    def __init__(self, game_file: str = "pick_and_place_simple-Mug-001/game.tw"):
        self._game_file = game_file
        self._step_count = 0
        self._won = False
        self._closed = False

    def reset(self) -> tuple[list[str], dict[str, list]]:
        self._step_count = 0
        self._won = False
        obs = (
            "-= Welcome to TextWorld, ALFRED! =-\n\n"
            "You are in the middle of a room. Looking quickly around you, "
            "you see a desk 1, a shelf 1, and a drawer 1.\n\n"
            "Your task is to: put a clean mug on desk 1."
        )
        infos = {
            "won": [False],
            "admissible_commands": [
                ["go to desk 1", "go to shelf 1", "go to drawer 1", "inventory", "look"]
            ],
        }
        return [obs], infos

    def step(
        self, actions: list[str]
    ) -> tuple[list[str], list[float], list[bool], dict[str, list]]:
        self._step_count += 1
        action = actions[0]

        if action == "go to desk 1":
            obs = "You arrive at desk 1. On the desk 1, you see a pencil 1."
            admissible = ["go to shelf 1", "go to drawer 1", "examine desk 1", "inventory", "look"]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

        elif action == "go to shelf 1":
            obs = "You arrive at shelf 1. On the shelf 1, you see a mug 1."
            admissible = [
                "take mug 1 from shelf 1",
                "examine shelf 1",
                "go to desk 1",
                "inventory",
                "look",
            ]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

        elif action == "take mug 1 from shelf 1":
            obs = "You pick up the mug 1 from the shelf 1."
            admissible = ["go to desk 1", "go to drawer 1", "examine mug 1", "inventory", "look"]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

        elif action == "go to sinkbasin 1":
            obs = "You arrive at sinkbasin 1."
            admissible = ["clean mug 1 with sinkbasin 1", "inventory", "look"]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

        elif action == "clean mug 1 with sinkbasin 1":
            obs = "You clean the mug 1 using the sinkbasin 1."
            admissible = ["go to desk 1", "inventory", "look"]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

        elif action == "put mug 1 in/on desk 1":
            obs = "You put the mug 1 in/on the desk 1."
            self._won = True
            return [obs], [1.0], [True], {"won": [True], "admissible_commands": [[]]}

        else:
            obs = "Nothing happens."
            admissible = ["go to desk 1", "go to shelf 1", "inventory", "look"]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

    def close(self) -> None:
        self._closed = True


MOCK_GAME_FILES = (
    "/data/alfworld/pick_and_place_simple-Mug-001/game.tw",
    "/data/alfworld/look_at_obj_in_light-Candle-002/game.tw",
    "/data/alfworld/pick_clean_then_place_in_recep-Cup-003/game.tw",
    "/data/alfworld/pick_heat_then_place_in_recep-Egg-004/game.tw",
    "/data/alfworld/pick_cool_then_place_in_recep-Apple-005/game.tw",
    "/data/alfworld/pick_two_obj_and_place-Pen-006/game.tw",
)


def _make_env(
    game_files: tuple[str, ...] = MOCK_GAME_FILES,
    mock_gym: MockAlfWorldGymEnv | None = None,
    **kwargs: Any,
) -> AlfWorldEnvironment:
    """Create an AlfWorldEnvironment with mocked _init_game."""
    env = AlfWorldEnvironment(game_files=game_files, config={}, **kwargs)

    if mock_gym is None:
        mock_gym = MockAlfWorldGymEnv()

    # Patch _init_game to use our mock instead of real textworld
    def mock_init_game(game_file: str) -> tuple[str, dict[str, Any], tuple]:
        env._gym_env = mock_gym
        obs_list, infos = mock_gym.reset()
        raw_obs = obs_list[0]
        info = {k: v[0] for k, v in infos.items()}
        return raw_obs, info, ()

    env._init_game = mock_init_game  # type: ignore[assignment]
    return env


@pytest.fixture
def mock_gym() -> MockAlfWorldGymEnv:
    """Create a mock AlfWorld gym environment."""
    return MockAlfWorldGymEnv()


@pytest.fixture
def env(mock_gym: MockAlfWorldGymEnv) -> AlfWorldEnvironment:
    """Create a test AlfWorld environment."""
    return _make_env(mock_gym=mock_gym)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExtractObjective:
    """Tests for _extract_objective."""

    def test_standard_format(self):
        obs = "Some text.\n\nYour task is to: put a clean mug on desk 1."
        assert _extract_objective(obs) == "put a clean mug on desk 1."

    def test_at_end_of_string(self):
        obs = "Your task is to: look at pen under desklamp"
        assert _extract_objective(obs) == "look at pen under desklamp"

    def test_no_match(self):
        obs = "You are in a room."
        assert _extract_objective(obs) == ""

    def test_multiline(self):
        obs = "Your task is to: heat an egg\nYou are in the kitchen."
        assert _extract_objective(obs) == "heat an egg"


class TestExtractTaskType:
    """Tests for _extract_task_type."""

    def test_pick_and_place(self):
        assert (
            _extract_task_type("/data/pick_and_place_simple-Mug-001/game.tw")
            == "pick_and_place_simple"
        )

    def test_look_at_obj(self):
        assert (
            _extract_task_type("/data/look_at_obj_in_light-Candle/game.tw")
            == "look_at_obj_in_light"
        )

    def test_pick_clean(self):
        assert (
            _extract_task_type("/data/pick_clean_then_place_in_recep-Cup/game.tw")
            == "pick_clean_then_place_in_recep"
        )

    def test_pick_heat(self):
        assert (
            _extract_task_type("/data/pick_heat_then_place_in_recep-Egg/game.tw")
            == "pick_heat_then_place_in_recep"
        )

    def test_pick_cool(self):
        assert (
            _extract_task_type("/data/pick_cool_then_place_in_recep-Apple/game.tw")
            == "pick_cool_then_place_in_recep"
        )

    def test_pick_two(self):
        assert (
            _extract_task_type("/data/pick_two_obj_and_place-Pen/game.tw")
            == "pick_two_obj_and_place"
        )

    def test_unknown(self):
        assert _extract_task_type("/data/some_other_task/game.tw") == "unknown"


# ---------------------------------------------------------------------------
# Hidden state tests
# ---------------------------------------------------------------------------


class TestAlfWorldHidden:
    """Tests for AlfWorldHidden state."""

    def test_creation(self):
        hidden = AlfWorldHidden(
            task_index=0,
            task_type="pick_and_place_simple",
            objective="put a mug on desk",
            game_file="/data/game.tw",
            episode_step=2,
            last_action="go to desk 1",
            admissible_commands=("take mug 1", "look"),
        )

        assert hidden.task_index == 0
        assert hidden.task_type == "pick_and_place_simple"
        assert hidden.objective == "put a mug on desk"
        assert hidden.game_file == "/data/game.tw"
        assert hidden.episode_step == 2
        assert hidden.last_action == "go to desk 1"
        assert "take mug 1" in hidden.admissible_commands

    def test_immutability(self):
        hidden = AlfWorldHidden(
            task_index=0,
            task_type="pick_and_place_simple",
            objective="test",
            game_file="/data/game.tw",
            episode_step=0,
            last_action=None,
            admissible_commands=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore

    def test_last_action_none_on_reset(self):
        hidden = AlfWorldHidden(
            task_index=0,
            task_type="pick_and_place_simple",
            objective="test",
            game_file="/data/game.tw",
            episode_step=0,
            last_action=None,
            admissible_commands=(),
        )
        assert hidden.last_action is None


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


class TestAlfWorldReward:
    """Tests for AlfWorldReward."""

    def test_reward_name(self):
        reward_fn = AlfWorldReward()
        assert reward_fn.name == "task_completion"

    def test_reward_type(self):
        reward_fn = AlfWorldReward()
        assert reward_fn.reward_type == RewardType.OUTCOME

    def test_compute_won(self):
        reward_fn = AlfWorldReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {"won": True}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 1.0
        assert signal.name == "task_completion"
        assert signal.reward_type == RewardType.OUTCOME

    def test_compute_not_won(self):
        reward_fn = AlfWorldReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {"won": False}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 0.0

    def test_compute_won_missing_defaults_false(self):
        reward_fn = AlfWorldReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.info = {}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 0.0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


class TestAlfWorldEnvironment:
    """Tests for AlfWorldEnvironment."""

    def test_creation(self, env: AlfWorldEnvironment):
        assert env.spec.name == "alfworld"
        assert env.spec.adapter == "alfworld"
        assert env.spec.is_multi_turn is True
        assert env.spec.pure_step is False
        assert env.spec.supports_task_index is True
        assert env.spec.supports_len is True
        assert env.spec.supports_seed is False
        assert env.spec.max_steps == 50

    def test_len(self, env: AlfWorldEnvironment):
        assert len(env) == len(MOCK_GAME_FILES)

    def test_available_tools_empty(self, env: AlfWorldEnvironment):
        assert env.available_tools == ()

    def test_reward_functions(self, env: AlfWorldEnvironment):
        rfs = env.reward_functions
        assert len(rfs) == 1
        assert rfs[0].name == "task_completion"

    def test_reward_functions_with_extra(self, mock_gym: MockAlfWorldGymEnv):
        extra = MagicMock()
        env = _make_env(mock_gym=mock_gym, extra_rewards=(extra,))
        rfs = env.reward_functions
        assert len(rfs) == 2
        assert rfs[1] is extra

    def test_reset(self, env: AlfWorldEnvironment):
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.task_type == "pick_and_place_simple"
        assert state.hidden.objective == "put a clean mug on desk 1."
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None
        assert len(state.hidden.admissible_commands) > 0

        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0
        assert info["task_type"] == "pick_and_place_simple"

    def test_reset_default_task_index(self, env: AlfWorldEnvironment):
        state, info = env.reset()
        assert state.hidden.task_index == 0

    def test_reset_out_of_range(self, env: AlfWorldEnvironment):
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": 999})

    def test_reset_negative_index(self, env: AlfWorldEnvironment):
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": -1})

    def test_step_basic(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="go to desk 1")
        result = env.step(state, action)

        assert result.terminated is False
        assert result.truncated is False
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "go to desk 1"
        assert result.next_state.metadata.step == 1
        assert result.next_state.metadata.is_terminal is False

    def test_step_updates_admissible_commands(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        initial_cmds = state.hidden.admissible_commands

        action = Action(text="go to shelf 1")
        result = env.step(state, action)

        new_cmds = result.next_state.hidden.admissible_commands
        assert new_cmds != initial_cmds
        assert "take mug 1 from shelf 1" in new_cmds

    def test_step_won(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        # Navigate: shelf → take mug → sinkbasin → clean → desk → put
        actions = [
            "go to shelf 1",
            "take mug 1 from shelf 1",
            "go to sinkbasin 1",
            "clean mug 1 with sinkbasin 1",
            "go to desk 1",
            "put mug 1 in/on desk 1",
        ]

        for act_text in actions[:-1]:
            result = env.step(state, Action(text=act_text))
            state = result.next_state
            assert result.terminated is False

        # Final action wins
        result = env.step(state, Action(text=actions[-1]))
        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True
        assert result.info["won"] is True

        # Check reward
        signal = result.rewards.by_name("task_completion")
        assert signal is not None
        assert signal.reward == 1.0

    def test_truncation(self, mock_gym: MockAlfWorldGymEnv):
        env = _make_env(mock_gym=mock_gym, max_steps=2)
        state, _ = env.reset(options={"task_index": 0})

        # Step 1
        result = env.step(state, Action(text="look"))
        assert result.truncated is False

        # Step 2 — truncation
        result = env.step(result.next_state, Action(text="look"))
        assert result.truncated is True
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is True

    def test_won_at_max_steps_not_truncated(self, mock_gym: MockAlfWorldGymEnv):
        """If the agent wins on the last possible step, terminated=True, truncated=False."""
        env = _make_env(mock_gym=mock_gym, max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="put mug 1 in/on desk 1"))
        assert result.terminated is True
        assert result.truncated is False

    def test_state_continuity_rejects_stale_state(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action(text="go to desk 1"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state, Action(text="go to shelf 1"))

    def test_objective_in_observation(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert "Objective:" in state.observation.prompt
        assert "put a clean mug on desk 1" in state.observation.prompt

    def test_objective_not_in_observation(self, mock_gym: MockAlfWorldGymEnv):
        env = _make_env(mock_gym=mock_gym, include_objective_in_obs=False)
        state, _ = env.reset(options={"task_index": 0})
        assert "Objective:" not in state.observation.prompt

    def test_admissible_commands_in_observation(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert "Admissible commands:" in state.observation.prompt
        assert "go to desk 1" in state.observation.prompt

    def test_admissible_commands_not_in_observation(self, mock_gym: MockAlfWorldGymEnv):
        env = _make_env(mock_gym=mock_gym, include_admissible_commands=False)
        state, _ = env.reset(options={"task_index": 0})
        assert "Admissible commands:" not in state.observation.prompt

    def test_admissible_commands_always_in_hidden(self, mock_gym: MockAlfWorldGymEnv):
        """Admissible commands are stored in hidden state regardless of obs setting."""
        env = _make_env(mock_gym=mock_gym, include_admissible_commands=False)
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.hidden.admissible_commands) > 0

    def test_task_type_different_files(self):
        """Each game file produces the correct task type."""
        for game_file, expected_type in [
            ("/data/pick_and_place_simple-Mug/game.tw", "pick_and_place_simple"),
            ("/data/look_at_obj_in_light-Lamp/game.tw", "look_at_obj_in_light"),
            ("/data/pick_two_obj_and_place-Pen/game.tw", "pick_two_obj_and_place"),
        ]:
            env = _make_env(game_files=(game_file,))
            state, info = env.reset(options={"task_index": 0})
            assert state.hidden.task_type == expected_type
            assert info["task_type"] == expected_type

    def test_game_file_in_hidden(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.game_file == MOCK_GAME_FILES[0]

    def test_spec_metadata(self, env: AlfWorldEnvironment):
        assert env.spec.metadata["num_games"] == len(MOCK_GAME_FILES)

    def test_close(self, env: AlfWorldEnvironment, mock_gym: MockAlfWorldGymEnv):
        env.reset(options={"task_index": 0})
        env.close()
        assert mock_gym._closed is True
        assert env._gym_env is None

    def test_close_when_no_env(self, env: AlfWorldEnvironment):
        """close() is safe to call even before reset."""
        env.close()  # Should not raise

    def test_different_task_indices(self, mock_gym: MockAlfWorldGymEnv):
        """Resetting with different task indices selects different game files."""
        env = _make_env(mock_gym=mock_gym)
        state0, _ = env.reset(options={"task_index": 0})
        assert state0.hidden.game_file == MOCK_GAME_FILES[0]

        state1, _ = env.reset(options={"task_index": 1})
        assert state1.hidden.game_file == MOCK_GAME_FILES[1]

    def test_step_obs_in_messages(self, env: AlfWorldEnvironment):
        """Step observations go into messages, not prompt."""
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="go to desk 1"))

        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "desk 1" in last_msg

    def test_admissible_commands_in_step_obs(self, env: AlfWorldEnvironment):
        """Admissible commands appear in step observations too."""
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="go to shelf 1"))

        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "Admissible commands:" in last_msg
        assert "take mug 1 from shelf 1" in last_msg

    def test_step_info_contains_admissible_commands(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="go to shelf 1"))
        assert "admissible_commands" in result.info
        assert "take mug 1 from shelf 1" in result.info["admissible_commands"]


# ---------------------------------------------------------------------------
# Message history tests
# ---------------------------------------------------------------------------


class TestAlfWorldMessageHistory:
    """Tests for message history accumulation."""

    def test_initial_messages_empty(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.messages == ()

    def test_messages_accumulate(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        initial_prompt = state.observation.prompt

        result = env.step(state, Action(text="go to desk 1"))
        state = result.next_state
        assert len(state.observation.messages) == 2
        assert state.observation.messages[0] == {"role": "assistant", "content": "go to desk 1"}
        assert state.observation.prompt == initial_prompt

        result = env.step(state, Action(text="go to shelf 1"))
        state = result.next_state
        assert len(state.observation.messages) == 4
        assert state.observation.messages[2] == {"role": "assistant", "content": "go to shelf 1"}
        assert state.observation.prompt == initial_prompt

    def test_messages_on_terminal(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="put mug 1 in/on desk 1"))

        assert result.terminated is True
        assert len(result.next_state.observation.messages) == 2
        assert result.next_state.observation.messages[0] == {
            "role": "assistant",
            "content": "put mug 1 in/on desk 1",
        }


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestAlfWorldPrompts:
    """Tests for configurable prompt components."""

    def test_default_prompts(self, env: AlfWorldEnvironment):
        prompts = env.prompts
        assert "objective_prefix" in prompts
        assert "admissible_commands_prefix" in prompts

    def test_default_prompts_match_constants(self, env: AlfWorldEnvironment):
        assert env.prompts == DEFAULT_ALFWORLD_PROMPTS

    def test_prompts_returns_copy(self, env: AlfWorldEnvironment):
        p1 = env.prompts
        p2 = env.prompts
        assert p1 == p2
        assert p1 is not p2

    def test_custom_objective_prefix(self, mock_gym: MockAlfWorldGymEnv):
        custom = {"objective_prefix": "Goal: {objective}"}
        env = _make_env(mock_gym=mock_gym, prompts=custom)
        state, _ = env.reset(options={"task_index": 0})

        assert "Goal:" in state.observation.prompt
        assert "Objective:" not in state.observation.prompt

    def test_custom_admissible_commands_prefix(self, mock_gym: MockAlfWorldGymEnv):
        custom = {"admissible_commands_prefix": "Available actions:"}
        env = _make_env(mock_gym=mock_gym, prompts=custom)
        state, _ = env.reset(options={"task_index": 0})

        assert "Available actions:" in state.observation.prompt
        assert "Admissible commands:" not in state.observation.prompt

    def test_custom_prompts_merge(self, mock_gym: MockAlfWorldGymEnv):
        """Custom prompts only override specified keys."""
        custom = {"objective_prefix": "Task: {objective}"}
        env = _make_env(mock_gym=mock_gym, prompts=custom)
        prompts = env.prompts

        assert prompts["objective_prefix"] == "Task: {objective}"
        assert (
            prompts["admissible_commands_prefix"]
            == DEFAULT_ALFWORLD_PROMPTS["admissible_commands_prefix"]
        )


# ---------------------------------------------------------------------------
# ALFWORLD_TASK_TYPES constant tests
# ---------------------------------------------------------------------------


class TestTaskTypes:
    """Tests for the ALFWORLD_TASK_TYPES constant."""

    def test_has_six_types(self):
        assert len(ALFWORLD_TASK_TYPES) == 6

    def test_ids_are_1_to_6(self):
        assert set(ALFWORLD_TASK_TYPES.keys()) == {1, 2, 3, 4, 5, 6}

    def test_values_are_strings(self):
        for v in ALFWORLD_TASK_TYPES.values():
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestAlfWorldAdapter:
    """Tests for AlfWorldAdapter."""

    def test_adapter_name(self):
        adapter = AlfWorldAdapter()
        assert adapter.name == "alfworld"

    def test_list_environments(self):
        adapter = AlfWorldAdapter()
        envs = adapter.list_environments()
        assert "alfworld:train" in envs
        assert "alfworld:eval_in_distribution" in envs
        assert "alfworld:eval_out_of_distribution" in envs

    def test_get_environment_info(self):
        adapter = AlfWorldAdapter()
        info = adapter.get_environment_info()

        assert info["name"] == "alfworld"
        assert info["adapter"] == "alfworld"
        assert info["type"] == "multi_turn"
        assert "task_types" in info
        assert "actions" in info
        assert "go to {recep}" in info["actions"]

    def test_get_default_system_prompt_none(self):
        adapter = AlfWorldAdapter()
        assert adapter.get_default_system_prompt("alfworld") is None

    def test_get_prompt_template_none(self):
        adapter = AlfWorldAdapter()
        assert adapter.get_prompt_template("alfworld") is None

    def test_get_native_answer_extractor_none(self):
        adapter = AlfWorldAdapter()
        assert adapter.get_native_answer_extractor("alfworld") is None

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_invalid_split(self, mock_get):
        mock_alfworld = MagicMock()
        mock_env_mod = MagicMock()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        with pytest.raises(ValueError, match="Invalid split"):
            adapter.get_environment(split="invalid_split")

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_parses_split_from_name(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(name="alfworld:train", max_steps=30)

        assert isinstance(env, AlfWorldEnvironment)
        assert env.spec.max_steps == 30

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_default_name(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment()

        assert isinstance(env, AlfWorldEnvironment)
        assert len(env) == len(MOCK_GAME_FILES)

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_with_config_dict(self, mock_get):
        mock_alfworld = MagicMock()
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        custom_config = {"env": {"some_key": "some_val"}}
        env = adapter.get_environment(config=custom_config)

        assert isinstance(env, AlfWorldEnvironment)
        # The config should have been used (not getconfig)
        mock_alfworld.getconfig.assert_not_called()

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_filters_task_types(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        # Filter to only pick_and_place_simple (type 1)
        env = adapter.get_environment(task_types=[1])

        assert len(env) == 1
        assert "pick_and_place_simple" in env._game_files[0]

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_invalid_task_type(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        with pytest.raises(ValueError, match="Invalid task type IDs"):
            adapter.get_environment(task_types=[99])

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_multiple_task_types(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(task_types=[1, 2])

        assert len(env) == 2

    def test_import_error(self):
        adapter = AlfWorldAdapter()
        with patch.dict("sys.modules", {"alfworld": None, "alfworld.agents.environment": None}):
            with pytest.raises(ImportError, match="AlfWorld is required"):
                adapter._get_alfworld()


# ---------------------------------------------------------------------------
# Full episode integration tests
# ---------------------------------------------------------------------------


class TestAlfWorldFullEpisode:
    """Integration tests for multi-step episodes."""

    def test_full_task_completion(self, env: AlfWorldEnvironment):
        state, info = env.reset(options={"task_index": 0})

        assert "put a clean mug on desk 1" in info["objective"]
        assert state.metadata.step == 0

        # Navigate and complete task
        actions = [
            "go to shelf 1",
            "take mug 1 from shelf 1",
            "go to sinkbasin 1",
            "clean mug 1 with sinkbasin 1",
            "go to desk 1",
            "put mug 1 in/on desk 1",
        ]

        for i, act_text in enumerate(actions):
            result = env.step(state, Action(text=act_text))
            state = result.next_state
            assert state.metadata.step == i + 1
            assert state.hidden.episode_step == i + 1
            assert state.hidden.last_action == act_text

        assert result.terminated is True
        assert result.rewards.by_name("task_completion").reward == 1.0

    def test_failed_task_truncation(self, mock_gym: MockAlfWorldGymEnv):
        env = _make_env(mock_gym=mock_gym, max_steps=3)
        state, _ = env.reset(options={"task_index": 0})

        # Wander around without completing task
        for act in ["go to desk 1", "go to shelf 1", "look"]:
            result = env.step(state, Action(text=act))
            state = result.next_state

        assert result.truncated is True
        assert result.terminated is False
        assert result.rewards.by_name("task_completion").reward == 0.0

    def test_reward_zero_before_completion(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="go to desk 1"))
        signal = result.rewards.by_name("task_completion")
        assert signal.reward == 0.0


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestAlfWorldRegistration:
    """Tests for adapter registration."""

    def test_adapter_registers_when_available(self):
        """Adapter registers if alfworld is importable."""
        adapter = AlfWorldAdapter()

        with patch.object(adapter, "_get_alfworld") as mock_get:
            mock_get.return_value = (MagicMock(), MagicMock())
            # Should not raise
            adapter._get_alfworld()
            mock_get.assert_called_once()

    def test_adapter_skipped_when_not_installed(self):
        """Adapter is silently skipped when alfworld is not installed."""
        adapter = AlfWorldAdapter()

        with patch.object(adapter, "_get_alfworld", side_effect=ImportError("no alfworld")):
            with pytest.raises(ImportError):
                adapter._get_alfworld()


# ---------------------------------------------------------------------------
# Visual mode mock
# ---------------------------------------------------------------------------


class MockThorGymEnv:
    """Mock THOR gym environment returning dict observations with RGB frames.

    Simulates the batched interface: reset/step return lists.
    THOR mode returns ``{"text": str, "rgb": np.ndarray(H, W, 3)}``.
    """

    def __init__(self) -> None:
        self._step_count = 0
        self._won = False
        self._closed = False

    def reset(self) -> tuple[list[dict[str, Any]], dict[str, list]]:
        self._step_count = 0
        self._won = False
        obs = {
            "text": (
                "-= Welcome to TextWorld, ALFRED! =-\n\n"
                "You are in the middle of a room. Looking quickly around you, "
                "you see a desk 1, a shelf 1, and a drawer 1.\n\n"
                "Your task is to: put a clean mug on desk 1."
            ),
            "rgb": np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
        }
        infos = {
            "won": [False],
            "admissible_commands": [
                ["go to desk 1", "go to shelf 1", "go to drawer 1", "inventory", "look"]
            ],
        }
        return [obs], infos

    def step(
        self, actions: list[str]
    ) -> tuple[list[dict[str, Any]], list[float], list[bool], dict[str, list]]:
        self._step_count += 1
        action = actions[0]

        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        if action == "put mug 1 in/on desk 1":
            obs = {"text": "You put the mug 1 in/on the desk 1.", "rgb": frame}
            self._won = True
            return [obs], [1.0], [True], {"won": [True], "admissible_commands": [[]]}
        else:
            obs = {"text": "You look around.", "rgb": frame}
            admissible = ["go to desk 1", "go to shelf 1", "inventory", "look"]
            return [obs], [0.0], [False], {"won": [False], "admissible_commands": [admissible]}

    def close(self) -> None:
        self._closed = True


def _make_visual_env(
    game_files: tuple[str, ...] = MOCK_GAME_FILES,
    mock_gym: MockThorGymEnv | None = None,
    **kwargs: Any,
) -> AlfWorldEnvironment:
    """Create an AlfWorldEnvironment with visual=True and mocked THOR env."""
    env = AlfWorldEnvironment(game_files=game_files, config={}, visual=True, **kwargs)

    if mock_gym is None:
        mock_gym = MockThorGymEnv()

    def mock_init_game(game_file: str) -> tuple[str, dict[str, Any], tuple]:
        env._gym_env = mock_gym
        obs_list, infos = mock_gym.reset()
        obs_dict = obs_list[0]
        info = {k: v[0] for k, v in infos.items()}

        raw_text = obs_dict.get("text", str(obs_dict))
        frame = obs_dict.get("rgb")
        images: tuple[ImageContent, ...] = ()
        if frame is not None:
            images = (env._frame_to_image(frame),)
        return raw_text, info, images

    env._init_game = mock_init_game  # type: ignore[assignment]
    return env


# ---------------------------------------------------------------------------
# Visual mode tests
# ---------------------------------------------------------------------------


class TestAlfWorldVisualMode:
    """Tests for AlfWorld visual (AI2-THOR) mode."""

    def test_visual_flag_stored(self):
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={}, visual=True)
        assert env._visual is True

    def test_default_not_visual(self):
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})
        assert env._visual is False

    def test_text_mode_unchanged(self, env: AlfWorldEnvironment):
        """Default text mode should have no images."""
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.images == ()

    def test_visual_reset_has_images(self):
        """Visual mode reset should include ImageContent."""
        env = _make_visual_env()
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.observation.images) == 1
        assert isinstance(state.observation.images[0], ImageContent)
        assert state.observation.images[0].media_type == "image/png"

    def test_visual_reset_still_has_text(self):
        """Visual mode should still have text in prompt."""
        env = _make_visual_env()
        state, _ = env.reset(options={"task_index": 0})
        assert "desk 1" in state.observation.prompt or "room" in state.observation.prompt
        assert len(state.observation.prompt) > 0

    def test_visual_step_has_images(self):
        """Visual mode step should include images in observation."""
        env = _make_visual_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="look"))
        assert len(result.next_state.observation.images) == 1
        assert isinstance(result.next_state.observation.images[0], ImageContent)

    def test_visual_step_images_in_history(self):
        """Visual mode should include image data in message history."""
        env = _make_visual_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="look"))
        user_msgs = [m for m in result.next_state.observation.messages if m.get("role") == "user"]
        assert len(user_msgs) > 0
        assert "images" in user_msgs[-1]

    def test_visual_won_still_works(self):
        """Task completion should still work in visual mode."""
        env = _make_visual_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="put mug 1 in/on desk 1"))
        assert result.terminated is True
        signal = result.rewards.by_name("task_completion")
        assert signal is not None
        assert signal.reward == 1.0

    def test_frame_to_image(self):
        """_frame_to_image should convert numpy array to ImageContent."""
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={}, visual=True)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = env._frame_to_image(frame)
        assert isinstance(img, ImageContent)
        assert img.media_type == "image/png"

        import base64

        decoded = base64.b64decode(img.data)
        assert decoded[:4] == b"\x89PNG"

    def test_visual_spec_metadata(self):
        """Visual mode should be reflected in spec metadata."""
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={}, visual=True)
        assert env.spec.metadata.get("visual") is True

    def test_text_mode_spec_no_visual(self):
        """Text mode spec should not have visual=True."""
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})
        assert env.spec.metadata.get("visual") is not True

    def test_visual_frame_missing_graceful(self):
        """If THOR returns no rgb frame, images should be empty."""
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={}, visual=True)

        # Mock that returns dict without "rgb" key
        mock_gym = MagicMock()
        mock_gym.reset.return_value = (
            [{"text": "You are in a room.\n\nYour task is to: test"}],
            {"won": [False], "admissible_commands": [["look"]]},
        )

        def mock_init_game(game_file):
            env._gym_env = mock_gym
            obs_list, infos = mock_gym.reset()
            obs_dict = obs_list[0]
            info = {k: v[0] for k, v in infos.items()}
            raw_text = obs_dict.get("text", str(obs_dict))
            frame = obs_dict.get("rgb")
            images: tuple[ImageContent, ...] = ()
            if frame is not None:
                images = (env._frame_to_image(frame),)
            return raw_text, info, images

        env._init_game = mock_init_game  # type: ignore[assignment]
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.images == ()


class TestAlfWorldAdapterVisual:
    """Tests for visual parameter on AlfWorldAdapter."""

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_adapter_passes_visual_flag(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(visual=True)
        assert isinstance(env, AlfWorldEnvironment)
        assert env._visual is True

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_adapter_default_not_visual(self, mock_get):
        mock_alfworld = MagicMock()
        mock_alfworld.getconfig.return_value = {}
        mock_env_mod = MagicMock()
        mock_tw_env = MagicMock()
        mock_tw_env.game_files = list(MOCK_GAME_FILES)
        mock_env_mod.AlfredTWEnv.return_value = mock_tw_env
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment()
        assert env._visual is False
