"""Tests for the AlfWorld adapter."""

import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace
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
    _unbatch_admissible_commands,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, ImageContent, Observation, ObservationContent

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
        self, action: str
    ) -> tuple[list[str], list[float], list[bool], dict[str, list]]:
        self._step_count += 1

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


def _install_fake_textworld_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    env_factory: Any = MockAlfWorldGymEnv,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Install minimal fake ``textworld`` modules for ``_init_game`` tests."""
    register_calls: list[dict[str, Any]] = []
    make_calls: list[str] = []
    envs_by_id: dict[str, Any] = {}

    fake_textworld = ModuleType("textworld")
    fake_gym = ModuleType("textworld.gym")
    fake_alfworld = ModuleType("alfworld")
    fake_alfworld_agents = ModuleType("alfworld.agents")
    fake_alfworld_env = ModuleType("alfworld.agents.environment")
    fake_alfred_tw_env = ModuleType("alfworld.agents.environment.alfred_tw_env")

    class _Wrapper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    fake_alfred_tw_env.AlfredDemangler = _Wrapper
    fake_alfred_tw_env.AlfredInfos = _Wrapper
    fake_alfworld_env.alfred_tw_env = fake_alfred_tw_env
    fake_alfworld_agents.environment = fake_alfworld_env
    fake_alfworld.agents = fake_alfworld_agents

    def fake_register_games(
        gamefiles: list[str],
        request_infos: Any = None,
        max_episode_steps: int | None = None,
        wrappers: list[Any] | None = None,
        **_kwargs: Any,
    ) -> str:
        env_id = f"fake-env-{len(register_calls)}"
        register_calls.append({
            "gamefiles": tuple(gamefiles),
            "request_infos": request_infos,
            "max_episode_steps": max_episode_steps,
            "wrappers": tuple(wrappers or ()),
            "env_id": env_id,
        })
        return env_id

    def fake_make(env_id: str) -> Any:
        make_calls.append(env_id)
        env = envs_by_id.get(env_id)
        if env is None:
            game_file = register_calls[int(env_id.rsplit("-", 1)[-1])]["gamefiles"][0]
            env = env_factory(game_file)
            envs_by_id[env_id] = env
        return env

    fake_gym.register_games = fake_register_games
    fake_gym.make = fake_make
    fake_textworld.EnvInfos = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_textworld.gym = fake_gym

    monkeypatch.setitem(sys.modules, "textworld", fake_textworld)
    monkeypatch.setitem(sys.modules, "textworld.gym", fake_gym)
    monkeypatch.setitem(sys.modules, "alfworld", fake_alfworld)
    monkeypatch.setitem(sys.modules, "alfworld.agents", fake_alfworld_agents)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_alfworld_env)
    monkeypatch.setitem(
        sys.modules,
        "alfworld.agents.environment.alfred_tw_env",
        fake_alfred_tw_env,
    )
    return register_calls, make_calls, envs_by_id


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
    **kwargs: Any,
) -> AlfWorldEnvironment:
    """Create an AlfWorldEnvironment with mocked _init_game.

    Each call to _init_game creates a fresh MockAlfWorldGymEnv,
    supporting pure-step replay.
    """
    env = AlfWorldEnvironment(game_files=game_files, config={}, **kwargs)

    # Patch _init_game to use our mock instead of real textworld
    def mock_init_game(game_file: str) -> tuple[Any, str, dict[str, Any], tuple]:
        fresh = MockAlfWorldGymEnv(game_file)
        obs_list, infos = fresh.reset()
        raw_obs = obs_list[0]
        info = {k: v[0] for k, v in infos.items()}
        return fresh, raw_obs, info, ()

    env._init_game = mock_init_game  # type: ignore[assignment]
    return env


def _make_mock_alfworld() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create mocked AlfWorld modules for adapter tests."""
    mock_alfworld = MagicMock()
    mock_alfworld.ALFWORLD_DATA = "/mock/alfworld-data"
    mock_alfworld.ALFRED_PDDL_PATH = "/mock/pkg/alfred.pddl"
    mock_alfworld.ALFRED_TWL2_PATH = "/mock/pkg/alfred.twl2"

    mock_env_mod = MagicMock()
    mock_tw_env_cls = MagicMock()
    mock_tw_env = MagicMock()
    mock_tw_env.game_files = list(MOCK_GAME_FILES)
    mock_tw_env_cls.return_value = mock_tw_env
    mock_env_mod.get_environment.return_value = mock_tw_env_cls

    return mock_alfworld, mock_env_mod, mock_tw_env


@pytest.fixture
def env() -> AlfWorldEnvironment:
    """Create a test AlfWorld environment."""
    return _make_env()


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
        assert env.spec.pure_step is True
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

    def test_reward_functions_with_extra(self):
        extra = MagicMock()
        env = _make_env(extra_rewards=(extra,))
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

        # Structured observation: task = objective, state = room + commands
        obs = state.observation
        assert isinstance(obs.task, ObservationContent)
        assert "put a clean mug on desk 1" in obs.task.text  # task is objective
        assert obs.task.text != obs.state.text  # task ≠ state
        assert isinstance(obs.state, ObservationContent)
        assert "room" in obs.state.text or "desk" in obs.state.text  # state is room desc
        assert "Admissible commands:" in obs.state.text  # state includes commands
        assert obs.state.images == ()  # text mode: no images in state

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

        # Structured observation: task carried forward, state updated
        next_obs = result.next_state.observation
        assert next_obs.task is not None
        assert next_obs.task == state.observation.task  # task unchanged across steps
        assert "put a clean mug on desk 1" in next_obs.task.text
        assert next_obs.state is not None
        assert "desk 1" in next_obs.state.text  # state reflects step observation
        assert next_obs.state.images == ()  # text mode

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

    def test_truncation(self):
        env = _make_env(max_steps=2)
        state, _ = env.reset(options={"task_index": 0})

        # Step 1
        result = env.step(state, Action(text="look"))
        assert result.truncated is False

        # Step 2 — truncation
        result = env.step(result.next_state, Action(text="look"))
        assert result.truncated is True
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is True

    def test_won_at_max_steps_not_truncated(self):
        """If the agent wins on the last possible step, terminated=True, truncated=False."""
        env = _make_env(max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="put mug 1 in/on desk 1"))
        assert result.terminated is True
        assert result.truncated is False

    def test_pure_step_allows_reuse_of_state(self, env: AlfWorldEnvironment):
        """Same state can be stepped multiple times with different actions."""
        state, _ = env.reset(options={"task_index": 0})
        result_a = env.step(state, Action(text="go to desk 1"))
        result_b = env.step(state, Action(text="go to shelf 1"))

        assert result_a.next_state.hidden.last_action == "go to desk 1"
        assert result_b.next_state.hidden.last_action == "go to shelf 1"
        assert "desk 1" in result_a.next_state.observation.messages[-1]["content"]
        assert "shelf 1" in result_b.next_state.observation.messages[-1]["content"]

    def test_objective_in_observation(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert "Objective:" in state.observation.prompt
        assert "put a clean mug on desk 1" in state.observation.prompt

    def test_objective_not_in_observation(self):
        env = _make_env(include_objective_in_obs=False)
        state, _ = env.reset(options={"task_index": 0})
        assert "Objective:" not in state.observation.prompt

    def test_admissible_commands_in_observation(self, env: AlfWorldEnvironment):
        state, _ = env.reset(options={"task_index": 0})
        assert "Admissible commands:" in state.observation.prompt
        assert "go to desk 1" in state.observation.prompt

    def test_admissible_commands_not_in_observation(self):
        env = _make_env(include_admissible_commands=False)
        state, _ = env.reset(options={"task_index": 0})
        assert "Admissible commands:" not in state.observation.prompt

    def test_admissible_commands_always_in_hidden(self):
        """Admissible commands are stored in hidden state regardless of obs setting."""
        env = _make_env(include_admissible_commands=False)
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

    def test_close(self, env: AlfWorldEnvironment):
        env.reset(options={"task_index": 0})
        env.close()
        assert env._env_id_cache == {}

    def test_close_when_no_env(self, env: AlfWorldEnvironment):
        """close() is safe to call even before reset."""
        env.close()  # Should not raise

    def test_init_game_reuses_cached_gym_env_per_game_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        register_calls, make_calls, envs_by_id = _install_fake_textworld_modules(monkeypatch)
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})

        gym_env_1, raw_obs_1, init_info_1, images_1 = env._init_game(MOCK_GAME_FILES[0])
        gym_env_2, raw_obs_2, init_info_2, images_2 = env._init_game(MOCK_GAME_FILES[0])

        assert gym_env_1 is gym_env_2
        assert make_calls == ["fake-env-0"]
        assert len(register_calls) == 1
        assert env._env_id_cache[MOCK_GAME_FILES[0]] == "fake-env-0"
        assert env._gym_env_cache[MOCK_GAME_FILES[0]] is gym_env_1
        assert gym_env_1._step_count == 0
        assert raw_obs_1 == raw_obs_2
        assert init_info_1 == init_info_2
        assert images_1 == images_2 == ()
        assert not gym_env_1._closed

        second_env, _, _, _ = env._init_game(MOCK_GAME_FILES[1])
        assert second_env is envs_by_id["fake-env-1"]
        assert second_env is not gym_env_1
        assert make_calls == ["fake-env-0", "fake-env-1"]
        assert len(register_calls) == 2

    def test_close_closes_all_cached_gym_envs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_textworld_modules(monkeypatch)
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})

        first_env, _, _, _ = env._init_game(MOCK_GAME_FILES[0])
        second_env, _, _, _ = env._init_game(MOCK_GAME_FILES[1])
        assert not first_env._closed
        assert not second_env._closed

        env.close()

        assert first_env._closed
        assert second_env._closed
        assert env._gym_env_cache == {}
        assert env._env_id_cache == {}

    def test_reused_cached_gym_env_preserves_pure_step_semantics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_textworld_modules(monkeypatch)
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})
        state, _ = env.reset(options={"task_index": 0})

        first = env.step(state, Action(text="go to shelf 1"))
        second = env.step(state, Action(text="go to shelf 1"))

        assert first.next_state.observation.state.text == second.next_state.observation.state.text
        assert first.next_state.hidden.admissible_commands == second.next_state.hidden.admissible_commands
        assert first.rewards.total == second.rewards.total
        assert first.done == second.done
        assert first.next_state.hidden.trajectory == ("go to shelf 1",)
        assert second.next_state.hidden.trajectory == ("go to shelf 1",)

    def test_different_task_indices(self):
        """Resetting with different task indices selects different game files."""
        env = _make_env()
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

    def test_custom_objective_prefix(self):
        custom = {"objective_prefix": "Goal: {objective}"}
        env = _make_env(prompts=custom)
        state, _ = env.reset(options={"task_index": 0})

        assert "Goal:" in state.observation.prompt
        assert "Objective:" not in state.observation.prompt

    def test_custom_admissible_commands_prefix(self):
        custom = {"admissible_commands_prefix": "Available actions:"}
        env = _make_env(prompts=custom)
        state, _ = env.reset(options={"task_index": 0})

        assert "Available actions:" in state.observation.prompt
        assert "Admissible commands:" not in state.observation.prompt

    def test_custom_prompts_merge(self):
        """Custom prompts only override specified keys."""
        custom = {"objective_prefix": "Task: {objective}"}
        env = _make_env(prompts=custom)
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

    @pytest.fixture(autouse=True)
    def _mock_data_dir_exists(self):
        with patch("llenvs.adapters.alfworld.os.path.isdir", return_value=True):
            yield

    @staticmethod
    def _resolved_config(mock_env_mod: MagicMock) -> dict[str, Any]:
        return mock_env_mod.get_environment.return_value.call_args.args[0]

    @staticmethod
    def _set_game_files(mock_env_mod: MagicMock, *game_files: str) -> None:
        mock_env_mod.get_environment.return_value.return_value.game_files = list(game_files)

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
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(name="alfworld:train", max_steps=30)
        resolved_config = self._resolved_config(mock_env_mod)

        assert isinstance(env, AlfWorldEnvironment)
        assert env.spec.max_steps == 30
        assert resolved_config["env"]["train_eval"] == "train"
        mock_env_mod.get_environment.assert_called_once_with("AlfredTWEnv")
        assert mock_env_mod.get_environment.return_value.call_args.kwargs["train_eval"] == "train"
        assert resolved_config["dataset"]["data_path"] == "/mock/alfworld-data/json_2.1.1/train"

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_default_name(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment()
        resolved_config = self._resolved_config(mock_env_mod)

        assert isinstance(env, AlfWorldEnvironment)
        assert len(env) == len(MOCK_GAME_FILES)
        assert resolved_config["env"]["train_eval"] == "eval_out_of_distribution"
        assert resolved_config["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
        assert resolved_config["logic"]["domain"] == "/mock/pkg/alfred.pddl"
        assert resolved_config["logic"]["grammar"] == "/mock/pkg/alfred.twl2"
        assert (
            resolved_config["dataset"]["eval_ood_data_path"]
            == "/mock/alfworld-data/json_2.1.1/valid_unseen"
        )
        mock_env_mod.get_environment.assert_called_once_with("AlfredTWEnv")
        assert (
            mock_env_mod.get_environment.return_value.call_args.kwargs["train_eval"]
            == "eval_out_of_distribution"
        )

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_with_config_dict(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        custom_config = {
            "dataset": {"num_eval_games": 7},
            "env": {"some_key": "some_val"},
        }
        original_config = deepcopy(custom_config)
        env = adapter.get_environment(config=custom_config)
        resolved_config = self._resolved_config(mock_env_mod)

        assert isinstance(env, AlfWorldEnvironment)
        assert resolved_config["dataset"]["num_eval_games"] == 7
        assert resolved_config["dataset"]["data_path"] == "/mock/alfworld-data/json_2.1.1/train"
        assert resolved_config["env"]["some_key"] == "some_val"
        assert resolved_config["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
        assert resolved_config["env"]["train_eval"] == "eval_out_of_distribution"
        assert custom_config == original_config

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_config_path_overrides_config(self, mock_get, tmp_path):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        config_path = tmp_path / "alfworld.yaml"
        config_path.write_text(
            "dataset:\n  num_eval_games: 11\nenv:\n  goal_desc_human_anns_prob: 0.25\n",
            encoding="ascii",
        )

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(
            config={"dataset": {"num_eval_games": 7}, "env": {"task_types": [1]}},
            config_path=str(config_path),
        )
        resolved_config = self._resolved_config(mock_env_mod)

        assert isinstance(env, AlfWorldEnvironment)
        assert resolved_config["dataset"]["num_eval_games"] == 11
        assert resolved_config["env"]["task_types"] == [1]
        assert resolved_config["env"]["goal_desc_human_anns_prob"] == 0.25
        assert resolved_config["env"]["train_eval"] == "eval_out_of_distribution"

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_uses_upstream_text_loader_not_direct_attribute(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        del mock_env_mod.AlfredTWEnv
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment()

        assert isinstance(env, AlfWorldEnvironment)
        mock_env_mod.get_environment.assert_called_once_with("AlfredTWEnv")

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_filters_task_types(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        self._set_game_files(mock_env_mod, *MOCK_GAME_FILES)
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        # Filter to only pick_and_place_simple (type 1)
        env = adapter.get_environment(task_types=[1])

        assert len(env) == 1
        assert "pick_and_place_simple" in env._game_files[0]

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_invalid_task_type(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        with pytest.raises(ValueError, match="Invalid task type IDs"):
            adapter.get_environment(task_types=[99])

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_multiple_task_types(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        self._set_game_files(mock_env_mod, *MOCK_GAME_FILES)
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(task_types=[1, 2])

        assert len(env) == 2

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_get_environment_empty_filtered_task_types_raises(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        self._set_game_files(mock_env_mod, "/data/unexpected_layout/game.tw")
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        with pytest.raises(ValueError, match="No ALFWorld games found"):
            adapter.get_environment(task_types=[1])

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_missing_data_directory_raises_with_path(self, mock_get):
        """Adapter raises early when the data directory does not exist."""
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        with patch("llenvs.adapters.alfworld.os.path.isdir", return_value=False):
            with pytest.raises(ValueError, match="does not exist") as exc_info:
                adapter.get_environment()

        msg = str(exc_info.value)
        assert "/mock/alfworld-data" in msg
        assert "alfworld-download" in msg

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_empty_game_files_error_includes_data_path(self, mock_get):
        """'No games found' error includes the scanned data path."""
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        self._set_game_files(mock_env_mod)  # empty game files
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        with pytest.raises(ValueError, match="No ALFWorld games found") as exc_info:
            adapter.get_environment()

        msg = str(exc_info.value)
        assert "Scanned data path" in msg
        assert "/mock/alfworld-data" in msg

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

    def test_failed_task_truncation(self):
        env = _make_env(max_steps=3)
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
# Pure-step tests
# ---------------------------------------------------------------------------


class TestAlfWorldPureStep:
    """Tests for pure_step=True replay-based stepping."""

    def test_branching_from_same_state(self):
        """Stepping the same state with different actions produces different results."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        # First branch: go to shelf
        result_a = env.step(state, Action(text="go to shelf 1"))
        # Second branch: go to desk
        result_b = env.step(state, Action(text="go to desk 1"))

        assert result_a.next_state.hidden.last_action == "go to shelf 1"
        assert result_b.next_state.hidden.last_action == "go to desk 1"
        # Different observations
        msg_a = result_a.next_state.observation.messages[-1]["content"]
        msg_b = result_b.next_state.observation.messages[-1]["content"]
        assert "shelf 1" in msg_a
        assert "desk 1" in msg_b
        assert msg_a != msg_b

    def test_replay_matches_sequential(self):
        """Building state via replay produces the same result as sequential stepping."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        # Sequential: step through actions one at a time
        actions = ["go to shelf 1", "take mug 1 from shelf 1", "go to desk 1"]
        sequential_state = state
        for act_text in actions:
            result = env.step(sequential_state, Action(text=act_text))
            sequential_state = result.next_state

        # Replay: step from initial state, building up trajectory
        replay_state = state
        for act_text in actions:
            result = env.step(replay_state, Action(text=act_text))
            replay_state = result.next_state

        # Should produce identical hidden state
        assert sequential_state.hidden.episode_step == replay_state.hidden.episode_step
        assert sequential_state.hidden.last_action == replay_state.hidden.last_action
        assert sequential_state.hidden.trajectory == replay_state.hidden.trajectory
        assert (
            sequential_state.hidden.admissible_commands == replay_state.hidden.admissible_commands
        )

    def test_trajectory_accumulation(self):
        """Hidden trajectory grows correctly across steps."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        assert state.hidden.trajectory == ()

        result = env.step(state, Action(text="go to shelf 1"))
        state = result.next_state
        assert state.hidden.trajectory == ("go to shelf 1",)

        result = env.step(state, Action(text="take mug 1 from shelf 1"))
        state = result.next_state
        assert state.hidden.trajectory == ("go to shelf 1", "take mug 1 from shelf 1")

        result = env.step(state, Action(text="go to desk 1"))
        state = result.next_state
        assert state.hidden.trajectory == (
            "go to shelf 1",
            "take mug 1 from shelf 1",
            "go to desk 1",
        )

    def test_truncation_survives_replay(self):
        """Truncation applies correctly when replaying near max_steps."""
        env = _make_env(max_steps=3)
        state, _ = env.reset(options={"task_index": 0})

        # Step to penultimate
        result = env.step(state, Action(text="go to shelf 1"))
        state = result.next_state
        assert result.truncated is False

        result = env.step(state, Action(text="go to desk 1"))
        state = result.next_state
        assert result.truncated is False

        # Third step should truncate
        result = env.step(state, Action(text="look"))
        assert result.truncated is True
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is True

    def test_terminal_survives_replay(self):
        """Winning via replay produces terminated=True."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        # Full winning sequence
        actions = [
            "go to shelf 1",
            "take mug 1 from shelf 1",
            "go to sinkbasin 1",
            "clean mug 1 with sinkbasin 1",
            "go to desk 1",
            "put mug 1 in/on desk 1",
        ]

        for act_text in actions:
            result = env.step(state, Action(text=act_text))
            state = result.next_state

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True
        assert result.info["won"] is True
        assert result.rewards.by_name("task_completion").reward == 1.0

    def test_branch_after_multiple_steps(self):
        """Branching works from a state reached after several steps."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        # Build up some trajectory
        result = env.step(state, Action(text="go to shelf 1"))
        state = result.next_state
        result = env.step(state, Action(text="take mug 1 from shelf 1"))
        state = result.next_state

        # Branch from here
        result_a = env.step(state, Action(text="go to desk 1"))
        result_b = env.step(state, Action(text="go to sinkbasin 1"))

        assert result_a.next_state.hidden.last_action == "go to desk 1"
        assert result_b.next_state.hidden.last_action == "go to sinkbasin 1"
        assert result_a.next_state.hidden.trajectory != result_b.next_state.hidden.trajectory

    def test_initial_state_trajectory_empty(self):
        """Reset produces state with empty trajectory."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.trajectory == ()


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
        self, action: str
    ) -> tuple[list[dict[str, Any]], list[float], list[bool], dict[str, list]]:
        self._step_count += 1

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
    **kwargs: Any,
) -> AlfWorldEnvironment:
    """Create an AlfWorldEnvironment with visual=True and mocked THOR env."""
    env = AlfWorldEnvironment(game_files=game_files, config={}, visual=True, **kwargs)

    def mock_init_game(game_file: str) -> tuple[Any, str, dict[str, Any], tuple]:
        fresh = MockThorGymEnv()
        obs_list, infos = fresh.reset()
        obs_dict = obs_list[0]
        info = {k: v[0] for k, v in infos.items()}

        raw_text = obs_dict.get("text", str(obs_dict))
        frame = obs_dict.get("rgb")
        images: tuple[ImageContent, ...] = ()
        if frame is not None:
            images = (env._frame_to_image(frame),)
        return fresh, raw_text, info, images

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
        assert state.observation.get_images().all == ()

    def test_visual_reset_has_images(self):
        """Visual mode reset should include ImageContent."""
        env = _make_visual_env()
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.observation.state.images) == 1
        assert isinstance(state.observation.state.images[0], ImageContent)
        assert state.observation.state.images[0].media_type == "image/png"

        # Structured observation: state includes images in visual mode
        obs = state.observation
        assert isinstance(obs.task, ObservationContent)
        assert isinstance(obs.state, ObservationContent)
        assert len(obs.state.images) == 1
        assert obs.state.images[0].media_type == "image/png"

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
        assert len(result.next_state.observation.state.images) == 1
        assert isinstance(result.next_state.observation.state.images[0], ImageContent)

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
        def mock_init_game(game_file):
            fresh = MagicMock()
            fresh.reset.return_value = (
                [{"text": "You are in a room.\n\nYour task is to: test"}],
                {"won": [False], "admissible_commands": [["look"]]},
            )
            obs_list, infos = fresh.reset()
            obs_dict = obs_list[0]
            info = {k: v[0] for k, v in infos.items()}
            raw_text = obs_dict.get("text", str(obs_dict))
            frame = obs_dict.get("rgb")
            images: tuple[ImageContent, ...] = ()
            if frame is not None:
                images = (env._frame_to_image(frame),)
            return fresh, raw_text, info, images

        env._init_game = mock_init_game  # type: ignore[assignment]
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.get_images().all == ()


class TestAlfWorldAdapterVisual:
    """Tests for visual parameter on AlfWorldAdapter."""

    @pytest.fixture(autouse=True)
    def _mock_data_dir_exists(self):
        with patch("llenvs.adapters.alfworld.os.path.isdir", return_value=True):
            yield

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_adapter_passes_visual_flag(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment(visual=True)
        assert isinstance(env, AlfWorldEnvironment)
        assert env._visual is True

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_adapter_default_not_visual(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        adapter = AlfWorldAdapter()
        env = adapter.get_environment()
        assert env._visual is False


# ---------------------------------------------------------------------------
# Admissible commands unbatching tests
# ---------------------------------------------------------------------------


class TestUnbatchAdmissibleCommands:
    """Tests for _unbatch_admissible_commands."""

    def test_batched_format(self):
        """TextWorld batched format: [['cmd1', 'cmd2']]."""
        raw = [["go to shelf 1", "go to desk 1", "inventory"]]
        result = _unbatch_admissible_commands(raw)
        assert result == ("go to shelf 1", "go to desk 1", "inventory")

    def test_flat_format(self):
        """TextworldGymEnv single-game format: ['cmd1', 'cmd2']."""
        raw = ["go to shelf 1", "go to desk 1", "inventory"]
        result = _unbatch_admissible_commands(raw)
        assert result == ("go to shelf 1", "go to desk 1", "inventory")

    def test_empty_list(self):
        raw: list = []
        assert _unbatch_admissible_commands(raw) == ()

    def test_empty_inner_list(self):
        """Batched with empty inner list."""
        raw: list = [[]]
        assert _unbatch_admissible_commands(raw) == ()

    def test_none_input(self):
        assert _unbatch_admissible_commands(None) == ()  # type: ignore[arg-type]

    def test_single_command_flat(self):
        raw = ["look"]
        result = _unbatch_admissible_commands(raw)
        assert result == ("look",)

    def test_single_command_batched(self):
        raw = [["look"]]
        result = _unbatch_admissible_commands(raw)
        assert result == ("look",)

    def test_returns_tuple(self):
        raw = ["cmd1", "cmd2"]
        result = _unbatch_admissible_commands(raw)
        assert isinstance(result, tuple)


class TestFlatFormatAdmissibleCommandsInStep:
    """Verify step() handles flat admissible commands (TextworldGymEnv single-game)."""

    def test_step_flat_format_admissible_commands(self):
        """step() using _unbatch_admissible_commands produces correct tuple from flat list."""
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})

        flat_commands = ["go to shelf 1", "look", "inventory"]

        def mock_init_game(game_file):
            fresh = MagicMock()
            # simulate TextworldGymEnv single-game: already flat
            fresh.reset.return_value = (
                ["-= Welcome =-\n\nYour task is to: test task."],
                {"won": [False], "admissible_commands": flat_commands},
            )
            obs_list, infos = fresh.reset()
            raw_obs = obs_list[0]
            # Use _unbatch_admissible_commands as the real _init_game now does
            info = {}
            for k, v in infos.items():
                if k == "admissible_commands":
                    info[k] = _unbatch_admissible_commands(v)
                elif isinstance(v, (list, tuple)):
                    info[k] = v[0]
                else:
                    info[k] = v
            return fresh, raw_obs, info, ()

        env._init_game = mock_init_game  # type: ignore[assignment]
        state, _ = env.reset(options={"task_index": 0})
        # admissible_commands should be a proper tuple of strings, not characters
        assert state.hidden.admissible_commands == tuple(flat_commands)
        for cmd in state.hidden.admissible_commands:
            assert isinstance(cmd, str)
            assert len(cmd) > 1  # not individual characters

    def test_step_flat_format_in_step(self):
        """step() correctly unbatches flat admissible commands from step response."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        # Override _init_game to return flat format for step's replay
        flat_cmds_after_step = ["take mug 1 from shelf 1", "examine shelf 1", "inventory"]

        real_init_game = env._init_game

        def mock_init_for_step(game_file):
            fresh_mock = MagicMock()
            _, raw_obs, info, imgs = real_init_game(game_file)
            # Override step to return flat format
            fresh_mock.step.return_value = (
                [raw_obs],
                [0.0],
                [False],
                {"won": False, "admissible_commands": flat_cmds_after_step},
            )
            return fresh_mock, raw_obs, info, imgs

        env._init_game = mock_init_for_step  # type: ignore[assignment]
        result = env.step(state, Action(text="go to shelf 1"))
        assert result.next_state.hidden.admissible_commands == tuple(flat_cmds_after_step)
        for cmd in result.next_state.hidden.admissible_commands:
            assert isinstance(cmd, str)
            assert len(cmd) > 1


# ---------------------------------------------------------------------------
# Answer extractor tests
# ---------------------------------------------------------------------------


class SimpleTagExtractor:
    """Minimal extractor that pulls text from <answer>...</answer>."""

    def extract(self, text: str | None) -> tuple[str | None, dict]:
        if not text:
            return None, {}
        import re
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return m.group(1).strip(), {}
        return None, {}


def _make_cleaned_tag_extractor() -> Any:
    """Create a CleanedExtractor(TagBased) with a think-stripping pre-cleaner."""
    import re
    from llenvs.core.extraction import CleanedExtractor

    def strip_think(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return CleanedExtractor(inner=SimpleTagExtractor(), pre_cleaners=[strip_think])


class TestAlfWorldAnswerExtractor:
    """Tests for answer_extractor parameter on AlfWorldEnvironment."""

    def test_extractor_applied_to_env_step(self):
        """Extracted command is forwarded to gym_env.step()."""
        extractor = SimpleTagExtractor()
        env = _make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})

        # Raw text includes thinking + answer tag; only extracted part should reach TextWorld
        raw_text = "Let me think... I should go to the shelf. <answer>go to shelf 1</answer>"
        result = env.step(state, Action(text=raw_text))

        # TextWorld received "go to shelf 1" → shelf observation
        obs_text = result.next_state.observation.state.text
        assert "shelf 1" in obs_text

    def test_extracted_action_set_in_step_result(self):
        """StepResult.extracted_action and resolved_action reflect extracted command."""
        extractor = SimpleTagExtractor()
        env = _make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "Some reasoning <answer>go to desk 1</answer> done."
        result = env.step(state, Action(text=raw_text))

        assert result.extracted_action == "go to desk 1"
        assert result.resolved_action == "go to desk 1"

    def test_history_contains_extracted_command(self):
        """Conversation history stores the clean extracted command, not raw text."""
        extractor = SimpleTagExtractor()
        env = _make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "<think>thinking...</think> <answer>go to shelf 1</answer>"
        result = env.step(state, Action(text=raw_text))

        assistant_msgs = [
            m for m in result.next_state.observation.messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "go to shelf 1"
        assert "<think>" not in assistant_msgs[0]["content"]

    def test_trajectory_contains_extracted_command(self):
        """Hidden trajectory stores extracted commands for replay."""
        extractor = SimpleTagExtractor()
        env = _make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "<answer>go to shelf 1</answer>"
        result = env.step(state, Action(text=raw_text))

        assert result.next_state.hidden.trajectory == ("go to shelf 1",)
        assert result.next_state.hidden.last_action == "go to shelf 1"

    def test_invalid_extraction_uses_sentinel_command(self):
        """Extraction failure wastes a real turn with the configured sentinel."""
        extractor = SimpleTagExtractor()
        env = _make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "go to desk 1"
        result = env.step(state, Action(text=raw_text))

        assert result.extracted_action is None
        assert result.resolved_action == "[invalid action]"
        assert result.info["invalid_action_format"] is True
        assert result.info["action"] == "__invalid_action_noop__"
        assert result.next_state.hidden.last_action == "__invalid_action_noop__"
        assert result.next_state.hidden.trajectory == ("__invalid_action_noop__",)
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.episode_step == 1
        obs_text = result.next_state.observation.state.text
        assert "invalid" in obs_text.lower()
        assert "Nothing happens." in obs_text

    def test_pre_cleaners_applied_to_history_on_extraction_failure(self):
        """On extraction failure, placeholder is stored instead of cleaned raw text."""
        extractor = _make_cleaned_tag_extractor()
        env = _make_env(answer_extractor=extractor)
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "<think>I need to go to the desk.</think> go to desk 1"
        result = env.step(state, Action(text=raw_text))

        assert result.extracted_action is None
        assert result.resolved_action == "[invalid action]"
        assistant_msgs = [
            m for m in result.next_state.observation.messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "[invalid action]"

    def test_custom_invalid_action_text_used_in_history(self):
        """Custom invalid_action_text is used for malformed turns."""
        env = _make_env(
            answer_extractor=SimpleTagExtractor(),
            invalid_action_text="[bad action]",
        )
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="<think>thinking</think> go to desk 1"))

        assert result.extracted_action is None
        assert result.resolved_action == "[bad action]"
        assistant_msgs = [
            m for m in result.next_state.observation.messages if m.get("role") == "assistant"
        ]
        assert assistant_msgs[0]["content"] == "[bad action]"

    def test_none_invalid_action_text_preserves_cleaned_raw_history(self):
        """When invalid_action_text is None, cleaned raw text stays in history."""
        extractor = _make_cleaned_tag_extractor()
        env = _make_env(answer_extractor=extractor, invalid_action_text=None)
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "<think>thinking</think> go to desk 1"
        result = env.step(state, Action(text=raw_text))

        assert result.extracted_action is None
        assert result.resolved_action is None
        assistant_msgs = [
            m for m in result.next_state.observation.messages if m.get("role") == "assistant"
        ]
        assert assistant_msgs[0]["content"] == "go to desk 1"

    def test_pre_cleaners_not_applied_when_extractor_is_plain(self):
        """With a plain extractor and no placeholder, raw text stays in history."""
        env = _make_env(answer_extractor=SimpleTagExtractor())
        state, _ = env.reset(options={"task_index": 0})

        raw_text = "<think>thinking</think> go to desk 1"
        result = env.step(state, Action(text=raw_text))

        assert result.extracted_action is None
        assert result.resolved_action == "[invalid action]"
        assistant_msgs = [
            m for m in result.next_state.observation.messages if m.get("role") == "assistant"
        ]
        assert assistant_msgs[0]["content"] == "[invalid action]"

    def test_no_extractor_uses_raw_text(self):
        """Without extractor, extracted_action/resolved_action are None."""
        env = _make_env()  # no extractor
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="go to shelf 1"))
        assert result.extracted_action is None
        assert result.resolved_action is None

    def test_extractor_stored_on_env(self):
        extractor = SimpleTagExtractor()
        env = AlfWorldEnvironment(
            game_files=MOCK_GAME_FILES, config={}, answer_extractor=extractor
        )
        assert env._answer_extractor is extractor

    def test_no_extractor_stored_as_none(self):
        env = AlfWorldEnvironment(game_files=MOCK_GAME_FILES, config={})
        assert env._answer_extractor is None

    @patch("llenvs.adapters.alfworld.AlfWorldAdapter._get_alfworld")
    def test_adapter_passes_answer_extractor(self, mock_get):
        mock_alfworld, mock_env_mod, _ = _make_mock_alfworld()
        mock_get.return_value = (mock_alfworld, mock_env_mod)

        with patch("llenvs.adapters.alfworld.os.path.isdir", return_value=True):
            extractor = SimpleTagExtractor()
            env = AlfWorldAdapter().get_environment(answer_extractor=extractor)

        assert isinstance(env, AlfWorldEnvironment)
        assert env._answer_extractor is extractor
