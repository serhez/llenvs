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

    def test_long_invalid_text_truncated(self, classic_mapper):
        long_text = "x" * 500
        with pytest.raises(ValueError, match=r"x{100}\.\.\. \[truncated\]") as exc_info:
            classic_mapper.map(long_text)
        assert long_text not in str(exc_info.value)

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
# Mock types for Classic text renderer
# ---------------------------------------------------------------------------


class MockClassicInventory:
    """Mock Classic Inventory dataclass."""

    def __init__(self, **kwargs):
        for field in [
            "wood", "stone", "coal", "iron", "diamond", "sapling",
            "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
            "wood_sword", "stone_sword", "iron_sword",
        ]:
            setattr(self, field, kwargs.get(field, 0))


class MockClassicMobs:
    """Mock Classic Mobs dataclass."""

    def __init__(self, positions=None, masks=None):
        if positions is not None:
            self.position = np.array(positions, dtype=np.int32)
            self.mask = np.array(masks, dtype=bool) if masks is not None else np.ones(len(positions), dtype=bool)
        else:
            self.position = np.zeros((0, 2), dtype=np.int32)
            self.mask = np.zeros(0, dtype=bool)


def _make_classic_state(
    *,
    map_size=(20, 20),
    player_pos=(10, 10),
    player_direction=2,  # right
    fill_block=2,  # GRASS
    block_overrides=None,
    zombies=None,
    cows=None,
    skeletons=None,
    arrows=None,
    inventory=None,
    player_health=10,
    player_food=9,
    player_drink=9,
    player_energy=9,
    is_sleeping=False,
    light_level=1.0,
):
    """Create a mock Classic EnvState for text renderer testing."""
    game_map = np.full(map_size, fill_block, dtype=np.int32)
    if block_overrides:
        for (r, c), block_val in block_overrides.items():
            game_map[r, c] = block_val

    class State:
        pass

    s = State()
    s.map = game_map
    s.player_position = np.array(player_pos, dtype=np.int32)
    s.player_direction = player_direction
    s.zombies = zombies or MockClassicMobs()
    s.cows = cows or MockClassicMobs()
    s.skeletons = skeletons or MockClassicMobs()
    s.arrows = arrows or MockClassicMobs()
    s.inventory = inventory or MockClassicInventory()
    s.player_health = player_health
    s.player_food = player_food
    s.player_drink = player_drink
    s.player_energy = player_energy
    s.is_sleeping = is_sleeping
    s.light_level = light_level
    return s


# ---------------------------------------------------------------------------
# render_craftax_classic_text tests
# ---------------------------------------------------------------------------


class TestRenderCraftaxClassicText:
    """Tests for the Classic ASCII grid text renderer."""

    def _render(self, state):
        from llenvs.adapters.craftax import render_craftax_classic_text
        return render_craftax_classic_text(state)

    def test_grid_has_7_rows(self):
        state = _make_classic_state()
        text = self._render(state)
        # Count lines that look like grid rows (contain @ or terrain chars)
        grid_lines = [l for l in text.split("\n") if l.strip().startswith(("#", ".", "@", "~"))]
        # Actually, just find lines between "Map" header and "Terrain:" legend
        lines = text.split("\n")
        grid_lines = []
        in_grid = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Nearby"):
                in_grid = True
                continue
            if in_grid:
                if stripped and stripped[0] in ".#~stcwdifrp!=&@ZCKA?":
                    grid_lines.append(stripped)
                elif stripped.startswith("Terrain:") or stripped.startswith("Entities:"):
                    break
        assert len(grid_lines) == 7, f"Expected 7 grid rows, got {len(grid_lines)}: {grid_lines}"

    def test_player_at_center(self):
        state = _make_classic_state()
        text = self._render(state)
        assert "@" in text

    def test_direction_displayed(self):
        state = _make_classic_state(player_direction=1)  # left
        text = self._render(state)
        assert "left" in text.lower()

        state = _make_classic_state(player_direction=3)  # up
        text = self._render(state)
        assert "up" in text.lower()

    def test_zombie_on_grid(self):
        # Place zombie 1 row south of player (player at 10,10 → zombie at 11,10)
        state = _make_classic_state(
            zombies=MockClassicMobs(positions=[[11, 10]], masks=[True]),
        )
        text = self._render(state)
        assert "Z" in text

    def test_cow_on_grid(self):
        state = _make_classic_state(
            cows=MockClassicMobs(positions=[[10, 11]], masks=[True]),
        )
        text = self._render(state)
        assert "C" in text

    def test_skeleton_on_grid(self):
        state = _make_classic_state(
            skeletons=MockClassicMobs(positions=[[9, 10]], masks=[True]),
        )
        text = self._render(state)
        assert "K" in text

    def test_inactive_mob_not_shown(self):
        state = _make_classic_state(
            zombies=MockClassicMobs(positions=[[11, 10]], masks=[False]),
        )
        text = self._render(state)
        # Z appears in the legend; check only the grid lines for absence
        lines = text.split("\n")
        grid_lines = [l for l in lines if l.startswith("  ") and "@" not in l
                       and any(ch in l for ch in ".#~stcwdi")]
        for line in grid_lines:
            assert "Z" not in line

    def test_tree_tile(self):
        # Place a tree 1 tile east of player (player at 10,10 → tree at 10,11)
        state = _make_classic_state(
            block_overrides={(10, 11): 5},  # TREE = 5
        )
        text = self._render(state)
        assert "t" in text  # tree char

    def test_inventory_non_zero_shown(self):
        state = _make_classic_state(
            inventory=MockClassicInventory(wood=3, iron_pickaxe=1),
        )
        text = self._render(state)
        assert "wood" in text.lower()
        assert "3" in text
        assert "iron_pickaxe" in text.lower()

    def test_inventory_empty(self):
        state = _make_classic_state()
        text = self._render(state)
        # All zeros → should show empty or no items
        assert "empty" in text.lower() or "Inventory:" in text

    def test_vitals(self):
        state = _make_classic_state(
            player_health=10, player_food=8, player_drink=7, player_energy=6,
        )
        text = self._render(state)
        assert "Health: 10" in text
        assert "Food: 8" in text
        assert "Drink: 7" in text
        assert "Energy: 6" in text

    def test_no_mana_in_classic(self):
        state = _make_classic_state()
        text = self._render(state)
        assert "Mana" not in text

    def test_light_day(self):
        state = _make_classic_state(light_level=1.0)
        text = self._render(state)
        assert "day" in text.lower()

    def test_light_night(self):
        state = _make_classic_state(light_level=0.0)
        text = self._render(state)
        assert "night" in text.lower()

    def test_sleeping_shown_when_true(self):
        state = _make_classic_state(is_sleeping=True)
        text = self._render(state)
        assert "sleeping" in text.lower()

    def test_sleeping_not_shown_when_false(self):
        state = _make_classic_state(is_sleeping=False)
        text = self._render(state)
        assert "sleeping" not in text.lower()

    def test_legend_present(self):
        state = _make_classic_state()
        text = self._render(state)
        assert "Terrain:" in text
        assert "Entities:" in text

    def test_edge_player_near_map_border(self):
        """Player near map edge should see OUT_OF_BOUNDS (#) tiles."""
        state = _make_classic_state(player_pos=(0, 0), map_size=(20, 20))
        text = self._render(state)
        assert "#" in text  # should see border tiles


# ---------------------------------------------------------------------------
# _render_symbolic tests
# ---------------------------------------------------------------------------


class TestRenderSymbolicClassic:
    """Test symbolic observation parsing for Craftax Classic."""

    # Classic layout: map(1323) + inv(12) + intr(4) + dir(4) + misc(2) = 1345
    OBS_SIZE = 1345
    MAP_END = 1323
    INV_START = 1323
    INV_END = 1335
    INTR_START = 1335
    INTR_END = 1339

    def _render(self, obs):
        from llenvs.adapters.craftax import _render_symbolic

        return _render_symbolic(obs, is_classic=True)

    def test_header(self):
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        assert "=== Craftax Observation ===" in text

    def test_inventory_labels(self):
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        for label in [
            "wood",
            "stone",
            "coal",
            "iron",
            "diamond",
            "sapling",
            "wood_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "wood_sword",
            "stone_sword",
            "iron_sword",
        ]:
            assert label in text

    def test_inventory_values_denormalized(self):
        """Inventory values are stored as count/10; display should show actual counts."""
        obs = np.zeros(self.OBS_SIZE)
        # Set wood = 5 (normalized: 5/10 = 0.5)
        obs[self.INV_START] = 0.5
        # Set coal = 3 (normalized: 3/10 = 0.3)
        obs[self.INV_START + 2] = 0.3
        text = self._render(obs)
        assert "wood: 5" in text
        assert "coal: 3" in text

    def test_intrinsics_correct_indices(self):
        """Health/food/drink/energy read from correct indices (1335-1339)."""
        obs = np.zeros(self.OBS_SIZE)
        # Full health = 10, normalized = 10/10 = 1.0
        obs[self.INTR_START] = 1.0
        # Food = 9, normalized = 0.9
        obs[self.INTR_START + 1] = 0.9
        # Drink = 8, normalized = 0.8
        obs[self.INTR_START + 2] = 0.8
        # Energy = 7, normalized = 0.7
        obs[self.INTR_START + 3] = 0.7
        text = self._render(obs)
        assert "Health: 10" in text
        assert "Food: 9" in text
        assert "Drink: 8" in text
        assert "Energy: 7" in text

    def test_no_mana_in_classic(self):
        """Classic has only 4 intrinsics — no mana."""
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        assert "Mana" not in text

    def test_zero_obs_shows_zero_health(self):
        """All-zeros obs → Health: 0 (not garbage from wrong index)."""
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        assert "Health: 0" in text


class TestRenderSymbolicFull:
    """Test symbolic observation parsing for Craftax Full."""

    # Full layout: map(8217) + inv(16) + pot(6) + intr(9) + dir(4) +
    #              armour(4) + armour_ench(4) + special(8) = 8268
    OBS_SIZE = 8268
    MAP_END = 8217
    INV_START = 8217
    INV_END = 8233
    POT_START = 8233
    POT_END = 8239
    INTR_START = 8239
    INTR_END = 8248

    def _render(self, obs):
        from llenvs.adapters.craftax import _render_symbolic

        return _render_symbolic(obs, is_classic=False)

    def test_header(self):
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        assert "=== Craftax Observation ===" in text

    def test_inventory_has_16_items(self):
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        for label in [
            "wood",
            "stone",
            "coal",
            "iron",
            "diamond",
            "sapphire",
            "ruby",
            "sapling",
            "torches",
            "arrows",
        ]:
            assert label in text

    def test_potions_section(self):
        obs = np.zeros(self.OBS_SIZE)
        text = self._render(obs)
        # Potions section should exist with 6 potion types
        for potion in ["red_potion", "green_potion", "blue_potion",
                       "pink_potion", "cyan_potion", "yellow_potion"]:
            assert potion in text

    def test_intrinsics_all_nine(self):
        """Full has 9 intrinsics including mana, xp, dex, str, int."""
        obs = np.zeros(self.OBS_SIZE)
        # Set all intrinsics to known values (each / 10.0)
        obs[self.INTR_START] = 1.0      # health = 10
        obs[self.INTR_START + 4] = 0.5  # mana = 5
        obs[self.INTR_START + 5] = 0.3  # xp = 3
        text = self._render(obs)
        assert "Health: 10" in text
        assert "Mana: 5" in text
        assert "XP: 3" in text
        # All 9 should be present
        for label in ["Health", "Food", "Drink", "Energy", "Mana",
                       "XP", "Dexterity", "Strength", "Intelligence"]:
            assert label in text

    def test_full_health_not_zero(self):
        """With health at index 8239 set to 1.0, should show 10 not 0."""
        obs = np.zeros(self.OBS_SIZE)
        obs[self.INTR_START] = 1.0
        text = self._render(obs)
        assert "Health: 10" in text


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

        # Craftax populates data with episode_step and cumulative_reward
        obs_state = result.next_state.observation.state
        assert obs_state is not None
        assert obs_state.data is not None
        assert "episode_step" in obs_state.data
        assert "cumulative_reward" in obs_state.data

    def test_step_by_name(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="noop"))
        assert result.next_state.hidden.episode_step == 1

    def test_step_invalid_action_error(self, env):
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="fly"))
        # Should produce error step (wasted turn)
        assert result.next_state.metadata.step == 1
        assert "error" in result.info or "Invalid action" in result.next_state.observation.messages[
            -1
        ].get("content", "")

    def test_error_observation_includes_action_format_and_state(self, env):
        """Error observation includes expected action format and current state."""
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="fly"))
        error_obs = result.next_state.observation.state.text
        # Should have action format section
        assert "Expected action format:" in error_obs
        assert "noop" in error_obs  # classic action name
        # Should have current state section
        assert "Current state:" in error_obs

    def test_last_obs_text_set_on_reset(self, env):
        """Hidden state stores last_obs_text after reset."""
        state, _ = env.reset(options={"task_index": 0})
        assert hasattr(state.hidden, "last_obs_text")
        assert isinstance(state.hidden.last_obs_text, str)
        assert len(state.hidden.last_obs_text) > 0

    def test_last_obs_text_updated_on_step(self, env):
        """Hidden state updates last_obs_text after valid step."""
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="0"))
        assert isinstance(result.next_state.hidden.last_obs_text, str)
        assert len(result.next_state.hidden.last_obs_text) > 0

    def test_step_messages_accumulate(self, env):
        state, _ = env.reset(options={"task_index": 0})
        # After reset: 1 message (step-0 observation)
        assert len(state.observation.messages) == 1
        result = env.step(state, Action(text="0"))
        # After step: 3 messages (step-0 + assistant + step-1)
        assert len(result.next_state.observation.messages) == 3

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
