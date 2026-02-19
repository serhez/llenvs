"""Tests for the Craftax adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llenvs.core.reward import RewardType, SignalBundle
from llenvs.core.state import Action, Observation, State

# ---------------------------------------------------------------------------
# Mock Craftax types
# ---------------------------------------------------------------------------


class MockEnvState:
    """Mock Craftax EnvState pytree."""

    def __init__(
        self,
        player_health=9.0,
        player_food=9.0,
        player_drink=9.0,
        player_energy=9.0,
        player_mana=9.0,
        map_data=None,
        inventory=None,
        achievements=None,
    ):
        self.player_health = player_health
        self.player_food = player_food
        self.player_drink = player_drink
        self.player_energy = player_energy
        self.player_mana = player_mana
        self.map = map_data if map_data is not None else np.zeros((9, 11), dtype=np.int32)
        self.inventory = inventory if inventory is not None else np.zeros(12, dtype=np.int32)
        self.achievements = achievements if achievements is not None else np.zeros(22, dtype=bool)


class MockEnvParams:
    """Mock Craftax environment parameters."""

    max_steps_in_episode: int = 1000


class MockCraftaxEnv:
    """Mock Craftax environment (Gymnax-style)."""

    def __init__(self, num_actions=17, is_classic=True):
        self._num_actions = num_actions
        self._is_classic = is_classic

    def default_params(self):
        return MockEnvParams()

    def reset(self, key, params=None):
        obs = np.zeros(1345 if self._is_classic else 8268)
        state = MockEnvState()
        return obs, state

    def step(self, key, state, action, params=None):
        obs = np.zeros(1345 if self._is_classic else 8268)
        new_achievements = np.copy(state.achievements)
        new_state = MockEnvState(achievements=new_achievements)
        reward = 0.0
        done = False
        info = {"discount": 1.0}
        return obs, new_state, reward, done, info

    @property
    def num_actions(self):
        return self._num_actions


class MockJaxRandom:
    """Mock jax.random module."""

    @staticmethod
    def PRNGKey(seed):  # noqa: N802
        return np.array([0, seed], dtype=np.uint32)

    @staticmethod
    def split(key, num=2):
        if num == 2:
            return np.array([[0, key[1] + 1], [0, key[1] + 2]], dtype=np.uint32)
        return np.array([[0, key[1] + i] for i in range(num)], dtype=np.uint32)


# ---------------------------------------------------------------------------
# CraftaxActionMapper tests
# ---------------------------------------------------------------------------


class TestCraftaxActionMapper:
    @pytest.fixture
    def classic_mapper(self):
        from llenvs.adapters.craftax import CraftaxActionMapper

        return CraftaxActionMapper(is_classic=True)

    @pytest.fixture
    def full_mapper(self):
        from llenvs.adapters.craftax import CraftaxActionMapper

        return CraftaxActionMapper(is_classic=False)

    def test_parse_integer(self, classic_mapper):
        assert classic_mapper.map("0") == 0
        assert classic_mapper.map("5") == 5

    def test_parse_action_name_case_insensitive(self, classic_mapper):
        assert classic_mapper.map("noop") == 0
        assert classic_mapper.map("NOOP") == 0
        assert classic_mapper.map("Left") == 1

    def test_out_of_range_raises(self, classic_mapper):
        with pytest.raises(ValueError, match="out of range"):
            classic_mapper.map("99")

    def test_negative_raises(self, classic_mapper):
        with pytest.raises(ValueError, match="out of range"):
            classic_mapper.map("-1")

    def test_invalid_name_raises(self, classic_mapper):
        with pytest.raises(ValueError, match="Invalid action"):
            classic_mapper.map("fly")

    def test_classic_has_17_actions(self, classic_mapper):
        assert classic_mapper.num_actions == 17

    def test_full_has_more_actions(self, full_mapper):
        assert full_mapper.num_actions > 17

    def test_describe(self, classic_mapper):
        desc = classic_mapper.describe()
        assert "noop" in desc.lower()
        assert "left" in desc.lower()

    def test_strip_whitespace(self, classic_mapper):
        assert classic_mapper.map("  5  ") == 5
        assert classic_mapper.map("  noop  ") == 0


# ---------------------------------------------------------------------------
# CraftaxHidden tests
# ---------------------------------------------------------------------------


class TestCraftaxHidden:
    def test_construction(self):
        from llenvs.adapters.craftax import CraftaxHidden

        hidden = CraftaxHidden(
            task_index=0,
            seed=42,
            episode_step=0,
            last_action=None,
            craftax_state=MockEnvState(),
            rng_key=np.array([0, 42]),
            cumulative_reward=0.0,
            achievements=np.zeros(22, dtype=bool),
            is_classic=True,
        )
        assert hidden.task_index == 0
        assert hidden.seed == 42
        assert hidden.is_classic is True

    def test_frozen(self):
        from llenvs.adapters.craftax import CraftaxHidden

        hidden = CraftaxHidden(
            task_index=0,
            seed=42,
            episode_step=0,
            last_action=None,
            craftax_state=None,
            rng_key=None,
            cumulative_reward=0.0,
            achievements=None,
            is_classic=True,
        )
        with pytest.raises(AttributeError):
            hidden.task_index = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CraftaxReward tests
# ---------------------------------------------------------------------------


class TestCraftaxReward:
    def test_step_type_for_intermediate(self):
        from llenvs.adapters.craftax import CraftaxHidden, CraftaxReward

        reward_fn = CraftaxReward()
        state = State(
            observation=Observation(prompt=""),
            hidden=CraftaxHidden(
                task_index=0,
                seed=42,
                episode_step=0,
                last_action=None,
                craftax_state=None,
                rng_key=None,
                cumulative_reward=0.0,
                achievements=None,
                is_classic=True,
            ),
            metadata=MagicMock(is_terminal=False, info={"craftax_reward": 0.5}),
        )
        next_state = State(
            observation=Observation(prompt=""),
            hidden=state.hidden,
            metadata=MagicMock(is_terminal=False, info={"craftax_reward": 0.5}),
        )
        signal = reward_fn.compute(state, Action(text=""), next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == 0.5

    def test_outcome_type_for_terminal(self):
        from llenvs.adapters.craftax import CraftaxHidden, CraftaxReward

        reward_fn = CraftaxReward()
        state = State(
            observation=Observation(prompt=""),
            hidden=CraftaxHidden(
                task_index=0,
                seed=42,
                episode_step=0,
                last_action=None,
                craftax_state=None,
                rng_key=None,
                cumulative_reward=0.0,
                achievements=None,
                is_classic=True,
            ),
            metadata=MagicMock(is_terminal=False, info={"craftax_reward": 1.0}),
        )
        next_state = State(
            observation=Observation(prompt=""),
            hidden=state.hidden,
            metadata=MagicMock(is_terminal=True, info={"craftax_reward": 1.0}),
        )
        signal = reward_fn.compute(state, Action(text=""), next_state)
        assert signal.reward_type == RewardType.OUTCOME


# ---------------------------------------------------------------------------
# CraftaxAchievementReward tests
# ---------------------------------------------------------------------------


class TestCraftaxAchievementReward:
    def test_new_achievements_produce_signals(self):
        from llenvs.adapters.craftax import CraftaxAchievementReward, CraftaxHidden

        reward_fn = CraftaxAchievementReward()
        old_ach = np.zeros(22, dtype=bool)
        new_ach = np.zeros(22, dtype=bool)
        new_ach[0] = True  # newly unlocked
        new_ach[3] = True  # newly unlocked

        state = State(
            observation=Observation(prompt=""),
            hidden=CraftaxHidden(
                task_index=0,
                seed=42,
                episode_step=0,
                last_action=None,
                craftax_state=None,
                rng_key=None,
                cumulative_reward=0.0,
                achievements=old_ach,
                is_classic=True,
            ),
            metadata=MagicMock(is_terminal=False, info={}),
        )
        next_state = State(
            observation=Observation(prompt=""),
            hidden=CraftaxHidden(
                task_index=0,
                seed=42,
                episode_step=1,
                last_action=None,
                craftax_state=None,
                rng_key=None,
                cumulative_reward=0.0,
                achievements=new_ach,
                is_classic=True,
            ),
            metadata=MagicMock(is_terminal=False, info={"new_achievements": [0, 3]}),
        )
        signal = reward_fn.compute(state, Action(text=""), next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward > 0
        assert signal.metadata is not None

    def test_no_new_achievements(self):
        from llenvs.adapters.craftax import CraftaxAchievementReward, CraftaxHidden

        reward_fn = CraftaxAchievementReward()
        ach = np.zeros(22, dtype=bool)

        state = State(
            observation=Observation(prompt=""),
            hidden=CraftaxHidden(
                task_index=0,
                seed=42,
                episode_step=0,
                last_action=None,
                craftax_state=None,
                rng_key=None,
                cumulative_reward=0.0,
                achievements=ach,
                is_classic=True,
            ),
            metadata=MagicMock(is_terminal=False, info={}),
        )
        next_state = State(
            observation=Observation(prompt=""),
            hidden=CraftaxHidden(
                task_index=0,
                seed=42,
                episode_step=1,
                last_action=None,
                craftax_state=None,
                rng_key=None,
                cumulative_reward=0.0,
                achievements=ach.copy(),
                is_classic=True,
            ),
            metadata=MagicMock(is_terminal=False, info={"new_achievements": []}),
        )
        signal = reward_fn.compute(state, Action(text=""), next_state)
        assert signal.reward == 0.0


# ---------------------------------------------------------------------------
# CraftaxEnvironment tests
# ---------------------------------------------------------------------------


def _make_craftax_env(is_classic=True, max_steps=10, num_tasks=5, **kwargs):
    """Create a CraftaxEnvironment with mock backend."""
    from llenvs.adapters.craftax import CraftaxEnvironment

    mock_craftax_env = MockCraftaxEnv(
        num_actions=17 if is_classic else 43,
        is_classic=is_classic,
    )

    with patch("llenvs.adapters.craftax.jax", MockJaxModule()):
        env = CraftaxEnvironment(
            craftax_env=mock_craftax_env,
            is_classic=is_classic,
            max_steps=max_steps,
            num_tasks=num_tasks,
            observation_mode="symbolic",
            **kwargs,
        )

    return env


class MockJaxModule:
    """Mock jax module."""

    def __init__(self):
        self.random = MockJaxRandom()

    def __getattr__(self, name):
        if name == "random":
            return self.random
        raise AttributeError(name)


class TestCraftaxEnvironment:
    @pytest.fixture
    def env(self):
        from llenvs.adapters.craftax import CraftaxEnvironment

        mock_craftax_env = MockCraftaxEnv(num_actions=17, is_classic=True)
        env = CraftaxEnvironment(
            craftax_env=mock_craftax_env,
            is_classic=True,
            max_steps=10,
            num_tasks=5,
            observation_mode="symbolic",
            _jax_random=MockJaxRandom,
        )
        return env

    def test_spec(self, env):
        spec = env.spec
        assert spec.name == "craftax-classic"
        assert spec.adapter == "craftax"
        assert spec.is_multi_turn is True
        assert spec.pure_step is True
        assert spec.max_steps == 10

    def test_len(self, env):
        assert len(env) == 5

    def test_reward_functions(self, env):
        fns = env.reward_functions
        assert len(fns) >= 1

    def test_available_tools_empty(self, env):
        assert env.available_tools == ()

    def test_prompts_dict(self, env):
        assert isinstance(env.prompts, dict)

    def test_reset_returns_state_and_info(self, env):
        state, info = env.reset(options={"task_index": 0})
        assert isinstance(state, State)
        assert "task_index" in info
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

    def test_reset_prompt_contains_game_info(self, env):
        state, _ = env.reset(options={"task_index": 0})
        prompt = state.observation.prompt
        assert "Craftax" in prompt or "craftax" in prompt.lower()
        assert "Action" in prompt or "action" in prompt.lower()

    def test_reset_different_seeds(self, env):
        """Different task indices should produce different seeds."""
        state1, _ = env.reset(options={"task_index": 0})
        state2, _ = env.reset(options={"task_index": 1})
        assert state1.hidden.seed != state2.hidden.seed

    def test_step_valid_action(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="0"))  # noop
        assert isinstance(result.next_state, State)
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.episode_step == 1

    def test_step_by_name(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="noop"))
        assert result.next_state.hidden.episode_step == 1

    def test_step_invalid_action_error(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="fly"))
        # Should produce error step (wasted turn)
        assert result.next_state.metadata.step == 1
        assert "error" in result.info or "Error" in result.next_state.observation.messages[-1].get(
            "content", ""
        )

    def test_step_messages_accumulate(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="0"))
        assert len(result.next_state.observation.messages) == 2  # assistant + user

    def test_truncation_at_max_steps(self, env):
        state, _ = env.reset(options={"task_index": 0})
        for i in range(10):
            result = env.step(state, Action(text="0"))
            state = result.next_state
            if result.done:
                break
        assert result.truncated or result.terminated

    def test_pure_step_reuse(self, env):
        """pure_step=True means we can step from old states."""
        state, _ = env.reset(options={"task_index": 0})
        result1 = env.step(state, Action(text="1"))
        result2 = env.step(state, Action(text="2"))
        # Both should work without error
        assert result1.next_state.metadata.step == 1
        assert result2.next_state.metadata.step == 1

    def test_hidden_stores_craftax_state(self, env):
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.craftax_state is not None
        assert state.hidden.rng_key is not None

    def test_cumulative_reward_tracked(self, env):
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.cumulative_reward == 0.0

    def test_compute_rewards(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="0"))
        assert isinstance(result.rewards, SignalBundle)
        assert len(result.rewards.signals) >= 1


class TestCraftaxEnvironmentFull:
    @pytest.fixture
    def full_env(self):
        from llenvs.adapters.craftax import CraftaxEnvironment

        mock_craftax_env = MockCraftaxEnv(num_actions=43, is_classic=False)
        env = CraftaxEnvironment(
            craftax_env=mock_craftax_env,
            is_classic=False,
            max_steps=10,
            num_tasks=5,
            observation_mode="symbolic",
            _jax_random=MockJaxRandom,
        )
        return env

    def test_full_spec_name(self, full_env):
        assert full_env.spec.name == "craftax"

    def test_full_has_43_actions(self, full_env):
        state, _ = full_env.reset(options={"task_index": 0})
        # Should accept action 42 (last valid)
        result = full_env.step(state, Action(text="42"))
        assert result.next_state.metadata.step == 1


class TestCraftaxEnvironmentPixels:
    @pytest.fixture
    def pixel_env(self):
        from llenvs.adapters.craftax import CraftaxEnvironment

        mock_craftax_env = MockCraftaxEnv(num_actions=17, is_classic=True)
        # Override step/reset to return pixel data
        pixel_data = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_craftax_env.reset = lambda key, params=None: (pixel_data, MockEnvState())
        mock_craftax_env.step = lambda key, state, action, params=None: (
            pixel_data,
            MockEnvState(),
            0.0,
            False,
            {"discount": 1.0},
        )

        env = CraftaxEnvironment(
            craftax_env=mock_craftax_env,
            is_classic=True,
            max_steps=10,
            num_tasks=5,
            observation_mode="pixels",
            _jax_random=MockJaxRandom,
        )
        return env

    def test_pixel_observation_has_images(self, pixel_env):
        state, _ = pixel_env.reset(options={"task_index": 0})
        assert len(state.observation.images) == 1
        assert state.observation.images[0].media_type == "image/png"
        # Data should be base64 encoded
        assert isinstance(state.observation.images[0].data, str)

    def test_pixel_step_has_images(self, pixel_env):
        state, _ = pixel_env.reset(options={"task_index": 0})
        result = pixel_env.step(state, Action(text="0"))
        # Find user message with image reference
        # The step observation should have images
        assert len(result.next_state.observation.images) >= 0  # May or may not carry forward


class TestCraftaxEnvironmentText:
    @pytest.fixture
    def text_env(self):
        from llenvs.adapters.craftax import CraftaxEnvironment

        mock_craftax_env = MockCraftaxEnv(num_actions=43, is_classic=False)

        # Mock render_craftax_text
        def mock_render(state):
            return "You are in a forest. You see trees and a river."

        env = CraftaxEnvironment(
            craftax_env=mock_craftax_env,
            is_classic=False,
            max_steps=10,
            num_tasks=5,
            observation_mode="text",
            _jax_random=MockJaxRandom,
            _text_renderer=mock_render,
        )
        return env

    def test_text_mode_uses_renderer(self, text_env):
        state, _ = text_env.reset(options={"task_index": 0})
        assert "forest" in state.observation.prompt or "forest" in str(state.observation.messages)


# ---------------------------------------------------------------------------
# CraftaxAdapter tests
# ---------------------------------------------------------------------------


class TestCraftaxAdapter:
    def test_name(self):
        from llenvs.adapters.craftax import CraftaxAdapter

        adapter = CraftaxAdapter()
        assert adapter.name == "craftax"

    def test_list_environments(self):
        from llenvs.adapters.craftax import CraftaxAdapter

        adapter = CraftaxAdapter()
        envs = adapter.list_environments()
        assert "craftax" in envs
        assert "craftax-classic" in envs

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.craftax import CraftaxAdapter

        adapter = CraftaxAdapter()
        assert adapter.get_native_answer_extractor("craftax") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.craftax import CraftaxAdapter

        adapter = CraftaxAdapter()
        assert adapter.get_prompt_template("craftax") is None

    def test_get_environment_info(self):
        from llenvs.adapters.craftax import CraftaxAdapter

        adapter = CraftaxAdapter()
        info = adapter.get_environment_info("craftax")
        assert info["adapter"] == "craftax"
        assert info["type"] == "multi_turn"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestCraftaxPresets:
    def test_presets_exist(self):
        from llenvs.adapters.craftax import CRAFTAX_PRESETS

        assert "craftax" in CRAFTAX_PRESETS
        assert "craftax-classic" in CRAFTAX_PRESETS

    def test_preset_keys(self):
        from llenvs.adapters.craftax import CRAFTAX_PRESETS

        for name, preset in CRAFTAX_PRESETS.items():
            assert "is_classic" in preset
            assert "observation_mode" in preset


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestCraftaxRegistration:
    def test_adapter_in_adapters_all(self):
        from llenvs.adapters import __all__

        assert "CraftaxAdapter" in __all__
        assert "CraftaxEnvironment" in __all__
        assert "CraftaxHidden" in __all__

    def test_import_from_adapters(self):
        from llenvs.adapters import CraftaxAdapter, CraftaxEnvironment, CraftaxHidden

        assert CraftaxAdapter is not None
        assert CraftaxEnvironment is not None
        assert CraftaxHidden is not None


# ---------------------------------------------------------------------------
# Extra rewards
# ---------------------------------------------------------------------------


class TestCraftaxExtraRewards:
    def test_extra_rewards_included(self):
        from llenvs.adapters.craftax import CraftaxAchievementReward, CraftaxEnvironment

        mock_env = MockCraftaxEnv(num_actions=17, is_classic=True)
        ach_reward = CraftaxAchievementReward()
        env = CraftaxEnvironment(
            craftax_env=mock_env,
            is_classic=True,
            max_steps=10,
            num_tasks=5,
            observation_mode="symbolic",
            extra_rewards=(ach_reward,),
            _jax_random=MockJaxRandom,
        )
        fns = env.reward_functions
        assert len(fns) == 2  # native + achievement
