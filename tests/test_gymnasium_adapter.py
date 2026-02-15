"""Tests for the Gymnasium adapter."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Any

from llenvs.core.state import Action, Observation, State
from llenvs.core.reward import RewardType, SignalBundle, FormatReward
from llenvs.core.extraction import RawGenerationExtractor, TagBasedExtractor


# ---------------------------------------------------------------------------
# Mock gymnasium spaces
# ---------------------------------------------------------------------------

class MockDiscrete:
    """Mock gymnasium.spaces.Discrete."""
    def __init__(self, n: int, start: int = 0):
        self.n = n
        self.start = start
        self.shape = ()
        self.dtype = np.int64

    def __contains__(self, x):
        return self.start <= x < self.start + self.n


class MockBox:
    """Mock gymnasium.spaces.Box."""
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.array(low) if not isinstance(low, np.ndarray) else low
        self.high = np.array(high) if not isinstance(high, np.ndarray) else high
        if shape is not None:
            self.shape = shape
            self.low = np.full(shape, low) if np.isscalar(low) else self.low
            self.high = np.full(shape, high) if np.isscalar(high) else self.high
        else:
            self.shape = self.low.shape
        self.dtype = dtype

    def __contains__(self, x):
        return True


class MockMultiDiscrete:
    """Mock gymnasium.spaces.MultiDiscrete."""
    def __init__(self, nvec):
        self.nvec = np.array(nvec)
        self.shape = self.nvec.shape
        self.dtype = np.int64


class MockMultiBinary:
    """Mock gymnasium.spaces.MultiBinary."""
    def __init__(self, n):
        self.n = n
        self.shape = (n,)
        self.dtype = np.int8


class MockText:
    """Mock gymnasium.spaces.Text."""
    def __init__(self, max_length=100):
        self.max_length = max_length


class MockDict:
    """Mock gymnasium.spaces.Dict."""
    def __init__(self, spaces: dict):
        self.spaces = spaces

    def items(self):
        return self.spaces.items()


class MockTuple:
    """Mock gymnasium.spaces.Tuple."""
    def __init__(self, spaces: tuple):
        self.spaces = spaces


# ---------------------------------------------------------------------------
# Mock gymnasium environments
# ---------------------------------------------------------------------------

@dataclass
class MockSpec:
    """Mock gymnasium env spec."""
    id: str = "MockEnv-v0"
    max_episode_steps: int | None = None


class MockGymEnv:
    """Mock gymnasium environment with Discrete obs/action."""

    def __init__(
        self,
        obs_space=None,
        action_space=None,
        max_episode_steps: int | None = None,
        terminal_step: int = 3,
        reward_per_step: float = 0.0,
        terminal_reward: float = 1.0,
    ):
        self.observation_space = obs_space or MockDiscrete(4)
        self.action_space = action_space or MockDiscrete(3)
        self.spec = MockSpec(max_episode_steps=max_episode_steps)
        self._terminal_step = terminal_step
        self._reward_per_step = reward_per_step
        self._terminal_reward = terminal_reward
        self._step_count = 0
        self._current_obs = 0
        self.render_mode = None
        self._render_output = None

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._current_obs = 0
        return self._current_obs, {"seed": seed}

    def step(self, action):
        self._step_count += 1
        self._current_obs = min(action, self.observation_space.n - 1) if isinstance(self.observation_space, MockDiscrete) else action
        terminated = self._step_count >= self._terminal_step
        reward = self._terminal_reward if terminated else self._reward_per_step
        return self._current_obs, reward, terminated, False, {"step": self._step_count}

    def render(self):
        return self._render_output


class MockBoxGymEnv:
    """Mock gymnasium environment with Box obs/action."""

    def __init__(self, obs_dim=4, action_dim=1):
        self.observation_space = MockBox(-1.0, 1.0, shape=(obs_dim,))
        self.action_space = MockBox(-1.0, 1.0, shape=(action_dim,))
        self.spec = MockSpec()
        self._step_count = 0
        self.render_mode = None

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return np.zeros(self.observation_space.shape[0]), {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.default_rng(self._step_count).uniform(
            -1, 1, size=self.observation_space.shape[0]
        )
        terminated = self._step_count >= 5
        return obs, 0.5 if terminated else 0.0, terminated, False, {}

    def render(self):
        return None


class MockGridEnv:
    """Mock grid-world environment with 2D matrix observation."""

    def __init__(self, grid_size=5):
        self.observation_space = MockBox(0.0, 1.0, shape=(grid_size, grid_size))
        self.action_space = MockDiscrete(4)
        self.spec = MockSpec()
        self._grid_size = grid_size
        self._step_count = 0
        self.render_mode = None

    def reset(self, seed=None, options=None):
        self._step_count = 0
        grid = np.zeros((self._grid_size, self._grid_size))
        grid[0, 0] = 0.6  # Robot
        grid[2, 3] = 1.0  # Wall
        return grid, {}

    def step(self, action):
        self._step_count += 1
        grid = np.zeros((self._grid_size, self._grid_size))
        grid[0, 1] = 0.6  # Robot moved
        grid[2, 3] = 1.0  # Wall
        terminated = self._step_count >= 10
        return grid, 0.1, terminated, False, {}

    def render(self):
        return None


# ---------------------------------------------------------------------------
# We need to patch gymnasium.spaces for isinstance checks in the adapter
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_gymnasium_spaces(monkeypatch):
    """Patch gymnasium.spaces so isinstance checks work with our mocks."""
    import sys

    # Create mock gymnasium module hierarchy
    mock_gymnasium = MagicMock()
    mock_spaces = MagicMock()

    mock_spaces.Discrete = MockDiscrete
    mock_spaces.Box = MockBox
    mock_spaces.MultiDiscrete = MockMultiDiscrete
    mock_spaces.MultiBinary = MockMultiBinary
    mock_spaces.Text = MockText
    mock_spaces.Dict = MockDict
    mock_spaces.Tuple = MockTuple

    mock_gymnasium.spaces = mock_spaces

    # Store original modules if they exist
    originals = {}
    for mod_name in ["gymnasium", "gymnasium.spaces"]:
        if mod_name in sys.modules:
            originals[mod_name] = sys.modules[mod_name]

    sys.modules["gymnasium"] = mock_gymnasium
    sys.modules["gymnasium.spaces"] = mock_spaces

    yield mock_spaces

    # Restore
    for mod_name in ["gymnasium", "gymnasium.spaces"]:
        if mod_name in originals:
            sys.modules[mod_name] = originals[mod_name]
        else:
            sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Now import the adapter (after mock is set up)
# ---------------------------------------------------------------------------

@pytest.fixture
def import_adapter(mock_gymnasium_spaces):
    """Import adapter module after gymnasium is mocked."""
    import importlib
    import llenvs.adapters.gymnasium as mod
    importlib.reload(mod)
    return mod


# ===========================================================================
# AutoObservationMapper tests
# ===========================================================================

class TestAutoObservationMapper:

    def test_discrete_without_labels(self, import_adapter):
        space = MockDiscrete(5)
        mapper = import_adapter.AutoObservationMapper(space)
        result = mapper.map(3, {})
        assert "3" in result

    def test_discrete_with_labels(self, import_adapter):
        space = MockDiscrete(3)
        mapper = import_adapter.AutoObservationMapper(
            space, labels={0: "left", 1: "right", 2: "up"}
        )
        result = mapper.map(1, {})
        assert "right" in result

    def test_box_1d_small(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(3,))
        mapper = import_adapter.AutoObservationMapper(space)
        obs = np.array([0.123, -0.456, 0.789])
        result = mapper.map(obs, {})
        assert "0.12" in result
        assert "-0.46" in result
        assert "0.79" in result

    def test_box_1d_with_labels(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(2,))
        mapper = import_adapter.AutoObservationMapper(
            space, labels={0: "position", 1: "velocity"}
        )
        obs = np.array([1.5, -0.3])
        result = mapper.map(obs, {})
        assert "position" in result.lower() or "Position" in result
        assert "velocity" in result.lower() or "Velocity" in result

    def test_box_1d_large_summary(self, import_adapter):
        """Box with >20 dims should produce a summary."""
        space = MockBox(-1.0, 1.0, shape=(50,))
        mapper = import_adapter.AutoObservationMapper(space)
        obs = np.random.default_rng(42).uniform(-1, 1, size=50)
        result = mapper.map(obs, {})
        assert "50" in result  # dimension count
        assert "min" in result.lower() or "max" in result.lower()

    def test_box_2d_raises(self, import_adapter):
        """Box with 2+ dims raises ValueError at construction."""
        space = MockBox(0.0, 1.0, shape=(5, 5))
        with pytest.raises(ValueError, match="[Ii]mage|2D|ndim|custom mapper"):
            import_adapter.AutoObservationMapper(space)

    def test_box_precision(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(2,))
        mapper = import_adapter.AutoObservationMapper(space, precision=4)
        obs = np.array([0.123456, -0.654321])
        result = mapper.map(obs, {})
        assert "0.1235" in result
        assert "-0.6543" in result

    def test_multi_discrete(self, import_adapter):
        space = MockMultiDiscrete([3, 5, 2])
        mapper = import_adapter.AutoObservationMapper(space)
        obs = np.array([1, 3, 0])
        result = mapper.map(obs, {})
        assert "1" in result
        assert "3" in result

    def test_multi_binary(self, import_adapter):
        space = MockMultiBinary(4)
        mapper = import_adapter.AutoObservationMapper(space)
        obs = np.array([1, 0, 1, 0])
        result = mapper.map(obs, {})
        assert "1" in result
        assert "0" in result

    def test_text_passthrough(self, import_adapter):
        space = MockText()
        mapper = import_adapter.AutoObservationMapper(space)
        result = mapper.map("hello world", {})
        assert result == "hello world"

    def test_dict_space(self, import_adapter):
        space = MockDict({
            "pos": MockDiscrete(10),
            "vel": MockBox(-1.0, 1.0, shape=(2,)),
        })
        mapper = import_adapter.AutoObservationMapper(space)
        obs = {"pos": 3, "vel": np.array([0.5, -0.5])}
        result = mapper.map(obs, {})
        assert "pos" in result.lower() or "3" in result

    def test_tuple_space(self, import_adapter):
        space = MockTuple((MockDiscrete(5), MockDiscrete(3)))
        mapper = import_adapter.AutoObservationMapper(space)
        obs = (2, 1)
        result = mapper.map(obs, {})
        assert "2" in result
        assert "1" in result

    def test_describe_returns_string(self, import_adapter):
        space = MockDiscrete(5)
        mapper = import_adapter.AutoObservationMapper(space)
        desc = mapper.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0


# ===========================================================================
# AutoActionMapper tests
# ===========================================================================

class TestAutoActionMapper:

    def test_discrete_by_number(self, import_adapter):
        space = MockDiscrete(3)
        mapper = import_adapter.AutoActionMapper(space)
        assert mapper.map("1") == 1

    def test_discrete_by_name(self, import_adapter):
        space = MockDiscrete(3)
        mapper = import_adapter.AutoActionMapper(
            space, action_names={0: "left", 1: "right", 2: "up"}
        )
        assert mapper.map("right") == 1
        assert mapper.map("RIGHT") == 1  # case-insensitive

    def test_discrete_out_of_range(self, import_adapter):
        space = MockDiscrete(3)
        mapper = import_adapter.AutoActionMapper(space)
        with pytest.raises(ValueError, match="[Rr]ange|[Ii]nvalid"):
            mapper.map("5")

    def test_discrete_invalid_text(self, import_adapter):
        space = MockDiscrete(3)
        mapper = import_adapter.AutoActionMapper(space)
        with pytest.raises(ValueError):
            mapper.map("not_a_number")

    def test_box_single(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(1,))
        mapper = import_adapter.AutoActionMapper(space)
        result = mapper.map("0.5")
        assert np.isclose(result, 0.5).all()

    def test_box_multi(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(3,))
        mapper = import_adapter.AutoActionMapper(space)
        result = mapper.map("0.1, 0.2, 0.3")
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3], atol=1e-5)

    def test_box_clips_to_bounds(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(2,))
        mapper = import_adapter.AutoActionMapper(space)
        result = mapper.map("5.0, -5.0")
        np.testing.assert_allclose(result, [1.0, -1.0])

    def test_box_wrong_dim(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(3,))
        mapper = import_adapter.AutoActionMapper(space)
        with pytest.raises(ValueError, match="[Ee]xpect|dimension"):
            mapper.map("0.1, 0.2")

    def test_box_whitespace_separator(self, import_adapter):
        space = MockBox(-1.0, 1.0, shape=(2,))
        mapper = import_adapter.AutoActionMapper(space)
        result = mapper.map("0.5  -0.3")
        np.testing.assert_allclose(result, [0.5, -0.3], atol=1e-5)

    def test_multi_discrete(self, import_adapter):
        space = MockMultiDiscrete([3, 5])
        mapper = import_adapter.AutoActionMapper(space)
        result = mapper.map("1, 3")
        np.testing.assert_array_equal(result, [1, 3])

    def test_multi_discrete_out_of_range(self, import_adapter):
        space = MockMultiDiscrete([3, 5])
        mapper = import_adapter.AutoActionMapper(space)
        with pytest.raises(ValueError, match="[Rr]ange|[Ii]nvalid"):
            mapper.map("5, 3")

    def test_multi_binary(self, import_adapter):
        space = MockMultiBinary(3)
        mapper = import_adapter.AutoActionMapper(space)
        result = mapper.map("1, 0, 1")
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_multi_binary_invalid(self, import_adapter):
        space = MockMultiBinary(3)
        mapper = import_adapter.AutoActionMapper(space)
        with pytest.raises(ValueError):
            mapper.map("1, 2, 0")

    def test_text_passthrough(self, import_adapter):
        space = MockText()
        mapper = import_adapter.AutoActionMapper(space)
        assert mapper.map("hello") == "hello"

    def test_describe_returns_string(self, import_adapter):
        space = MockDiscrete(3)
        mapper = import_adapter.AutoActionMapper(space)
        desc = mapper.describe()
        assert isinstance(desc, str)


# ===========================================================================
# GridObservationMapper tests
# ===========================================================================

class TestGridObservationMapper:

    def test_basic_grid(self, import_adapter):
        value_map = {0.0: ".", 0.6: "R", 1.0: "#"}
        mapper = import_adapter.GridObservationMapper(value_map)

        grid = np.array([
            [0.6, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        result = mapper.map(grid, {})
        assert "R" in result
        assert "#" in result
        assert "." in result

    def test_3d_grid(self, import_adapter):
        """3D grids (e.g., channels) should work."""
        value_map = {0.0: ".", 1.0: "#"}
        mapper = import_adapter.GridObservationMapper(value_map)

        grid = np.zeros((1, 3, 3))
        grid[0, 1, 1] = 1.0
        result = mapper.map(grid, {})
        assert "#" in result

    def test_nearest_value_match(self, import_adapter):
        """Values not exactly in map should match nearest key."""
        value_map = {0.0: ".", 0.3: " ", 0.6: "R", 1.0: "#"}
        mapper = import_adapter.GridObservationMapper(value_map)

        grid = np.array([[0.29, 0.61]])
        result = mapper.map(grid, {})
        assert " " in result  # 0.29 -> nearest 0.3 -> " "
        assert "R" in result  # 0.61 -> nearest 0.6 -> "R"

    def test_describe(self, import_adapter):
        value_map = {0.0: ".", 1.0: "#"}
        mapper = import_adapter.GridObservationMapper(value_map)
        desc = mapper.describe()
        assert isinstance(desc, str)
        assert "." in desc or "#" in desc


# ===========================================================================
# GymnasiumReward tests
# ===========================================================================

class TestGymnasiumReward:

    def test_intermediate_step_reward(self, import_adapter):
        reward_fn = import_adapter.GymnasiumReward()
        from llenvs.core.state import StateMetadata

        state = State(
            observation=Observation(prompt="test"),
            hidden=import_adapter.GymnasiumHidden(
                task_index=0, seed=None, episode_step=0,
                last_action=None, raw_observation=0, gym_reward=0.0,
            ),
            metadata=StateMetadata(step=0, episode_id="ep1"),
        )
        next_state = State(
            observation=Observation(prompt="test"),
            hidden=import_adapter.GymnasiumHidden(
                task_index=0, seed=None, episode_step=1,
                last_action="1", raw_observation=1, gym_reward=0.5,
            ),
            metadata=StateMetadata(step=1, episode_id="ep1", is_terminal=False,
                                   info={"gym_reward": 0.5}),
        )
        signal = reward_fn.compute(state, Action(text="1"), next_state)
        assert signal.reward == 0.5
        assert signal.reward_type == RewardType.STEP

    def test_terminal_outcome_reward(self, import_adapter):
        reward_fn = import_adapter.GymnasiumReward()
        from llenvs.core.state import StateMetadata

        state = State(
            observation=Observation(prompt="test"),
            hidden=import_adapter.GymnasiumHidden(
                task_index=0, seed=None, episode_step=2,
                last_action="1", raw_observation=1, gym_reward=0.5,
            ),
            metadata=StateMetadata(step=2, episode_id="ep1"),
        )
        next_state = State(
            observation=Observation(prompt="test"),
            hidden=import_adapter.GymnasiumHidden(
                task_index=0, seed=None, episode_step=3,
                last_action="2", raw_observation=2, gym_reward=1.5,
            ),
            metadata=StateMetadata(step=3, episode_id="ep1", is_terminal=True,
                                   info={"gym_reward": 1.0}),
        )
        signal = reward_fn.compute(state, Action(text="2"), next_state)
        assert signal.reward == 1.0
        assert signal.reward_type == RewardType.OUTCOME

    def test_reward_name_and_type(self, import_adapter):
        reward_fn = import_adapter.GymnasiumReward()
        assert reward_fn.name == "gym_reward"
        assert reward_fn.reward_type == RewardType.OUTCOME


# ===========================================================================
# GymnasiumEnvironment tests
# ===========================================================================

class TestGymnasiumEnvironment:

    @pytest.fixture
    def discrete_env(self, import_adapter):
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
            terminal_step=3,
        )
        return import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            action_names={0: "left", 1: "right", 2: "up"},
            num_tasks=10,
        )

    @pytest.fixture
    def box_env(self, import_adapter):
        gym_env = MockBoxGymEnv(obs_dim=4, action_dim=2)
        return import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            observation_labels={0: "x", 1: "y", 2: "vx", 3: "vy"},
            num_tasks=5,
        )

    # --- Reset ---

    def test_reset_returns_state_and_info(self, discrete_env):
        state, info = discrete_env.reset()
        assert isinstance(state, State)
        assert isinstance(info, dict)

    def test_reset_prompt_contains_env_info(self, discrete_env):
        state, _ = discrete_env.reset()
        prompt = state.observation.prompt
        # Should have observation space and action space descriptions
        assert len(prompt) > 0

    def test_reset_prompt_contains_action_names(self, discrete_env):
        state, _ = discrete_env.reset()
        prompt = state.observation.prompt
        assert "left" in prompt.lower() or "right" in prompt.lower()

    def test_reset_initial_observation_in_prompt(self, discrete_env):
        state, _ = discrete_env.reset()
        # The initial observation should be rendered in the prompt
        assert "Step 0" in state.observation.prompt or "step 0" in state.observation.prompt.lower()

    def test_reset_hidden_state(self, discrete_env, import_adapter):
        state, _ = discrete_env.reset(seed=42)
        hidden = state.hidden
        assert isinstance(hidden, import_adapter.GymnasiumHidden)
        assert hidden.task_index == 0
        assert hidden.episode_step == 0
        assert hidden.seed == 42

    def test_reset_with_task_index(self, discrete_env):
        state, info = discrete_env.reset(options={"task_index": 3})
        assert state.hidden.task_index == 3

    def test_reset_metadata(self, discrete_env):
        state, _ = discrete_env.reset()
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

    def test_reset_messages_empty(self, discrete_env):
        state, _ = discrete_env.reset()
        assert state.observation.messages == ()

    # --- Step ---

    def test_step_valid_action_by_name(self, discrete_env):
        state, _ = discrete_env.reset()
        result = discrete_env.step(state, Action(text="right"))
        assert isinstance(result.next_state, State)
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.episode_step == 1

    def test_step_valid_action_by_number(self, discrete_env):
        state, _ = discrete_env.reset()
        result = discrete_env.step(state, Action(text="1"))
        assert result.next_state.metadata.step == 1

    def test_step_invalid_action_wasted_turn(self, discrete_env):
        state, _ = discrete_env.reset()
        result = discrete_env.step(state, Action(text="invalid_action"))
        # Step counter advances
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.episode_step == 1
        # Not terminated (just wasted a turn)
        assert result.terminated is False
        # Observation should mention error
        last_msg = result.next_state.observation.messages[-1]
        assert last_msg["role"] == "user"

    def test_step_extraction_failure_wasted_turn(self, import_adapter):
        """With a non-raw extractor, extraction failure wastes a turn."""
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            answer_extractor=TagBasedExtractor(tag_name="action"),
            num_tasks=1,
        )
        state, _ = env.reset()
        # No <action> tag -> extraction fails -> wasted turn
        result = env.step(state, Action(text="just some text"))
        assert result.next_state.metadata.step == 1
        assert result.terminated is False

    def test_step_with_tag_extractor(self, import_adapter):
        """Tag-based extractor should extract action from tags."""
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            answer_extractor=TagBasedExtractor(tag_name="action"),
            num_tasks=1,
        )
        state, _ = env.reset()
        result = env.step(state, Action(text="I think <action>1</action>"))
        assert result.next_state.metadata.step == 1
        # This should actually step the gym env since "1" was extracted

    def test_step_messages_accumulate(self, discrete_env):
        state, _ = discrete_env.reset()
        # Step 1
        result = discrete_env.step(state, Action(text="1"))
        assert len(result.next_state.observation.messages) == 2
        assert result.next_state.observation.messages[0]["role"] == "assistant"
        assert result.next_state.observation.messages[1]["role"] == "user"

        # Step 2
        result2 = discrete_env.step(result.next_state, Action(text="0"))
        assert len(result2.next_state.observation.messages) == 4

    def test_step_prompt_stays_constant(self, discrete_env):
        state, _ = discrete_env.reset()
        initial_prompt = state.observation.prompt

        result = discrete_env.step(state, Action(text="1"))
        assert result.next_state.observation.prompt == initial_prompt

    def test_step_terminated(self, discrete_env):
        """After terminal_step steps, environment should terminate."""
        state, _ = discrete_env.reset()
        for i in range(3):
            result = discrete_env.step(state, Action(text="1"))
            state = result.next_state
        assert result.terminated is True

    def test_step_truncated_by_max_steps(self, import_adapter):
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
            terminal_step=100,  # won't terminate naturally
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            max_steps=2,
            num_tasks=1,
        )
        state, _ = env.reset()
        result = env.step(state, Action(text="1"))
        assert result.truncated is False
        result = env.step(result.next_state, Action(text="1"))
        assert result.truncated is True

    def test_step_rewards(self, discrete_env):
        state, _ = discrete_env.reset()
        result = discrete_env.step(state, Action(text="1"))
        assert isinstance(result.rewards, SignalBundle)
        reward = result.rewards.by_name("gym_reward")
        assert reward is not None

    def test_step_cumulative_reward_in_hidden(self, discrete_env, import_adapter):
        """Hidden state tracks cumulative reward."""
        state, _ = discrete_env.reset()
        assert state.hidden.gym_reward == 0.0

        result = discrete_env.step(state, Action(text="1"))
        # After first step, gym_reward should be accumulated
        assert isinstance(result.next_state.hidden.gym_reward, float)

    # --- Box environment ---

    def test_box_env_step(self, box_env):
        state, _ = box_env.reset()
        result = box_env.step(state, Action(text="0.5, -0.3"))
        assert result.next_state.metadata.step == 1

    def test_box_obs_labels(self, box_env):
        state, _ = box_env.reset()
        prompt = state.observation.prompt
        # Labels should appear in observation description
        assert any(label in prompt for label in ["x", "y", "vx", "vy"])

    # --- Seeds and __len__ ---

    def test_len_with_num_tasks(self, discrete_env):
        assert len(discrete_env) == 10

    def test_len_with_seeds(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env, seeds=[1, 2, 3, 4, 5]
        )
        assert len(env) == 5

    def test_len_raises_without_seeds_or_num_tasks(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env)
        with pytest.raises(TypeError):
            len(env)

    def test_seed_from_seeds_list(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env, seeds=[10, 20, 30]
        )
        state, _ = env.reset(options={"task_index": 1})
        assert state.hidden.seed == 20

    def test_seed_explicit_overrides_seeds_list(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env, seeds=[10, 20, 30]
        )
        state, _ = env.reset(seed=99, options={"task_index": 1})
        assert state.hidden.seed == 99

    # --- Spec ---

    def test_spec_multi_turn(self, discrete_env):
        spec = discrete_env.spec
        assert spec.is_multi_turn is True
        assert spec.adapter == "gymnasium"

    def test_spec_pure_step_false(self, discrete_env):
        """Gymnasium envs are stateful (non-pure)."""
        assert discrete_env.spec.pure_step is False

    def test_spec_max_steps(self, import_adapter):
        gym_env = MockGymEnv(max_episode_steps=200)
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env, num_tasks=1
        )
        assert env.spec.max_steps == 200

    def test_spec_explicit_max_steps_overrides(self, import_adapter):
        gym_env = MockGymEnv(max_episode_steps=200)
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env, max_steps=50, num_tasks=1
        )
        assert env.spec.max_steps == 50

    # --- Reward functions ---

    def test_reward_functions_native(self, discrete_env):
        rfs = discrete_env.reward_functions
        assert len(rfs) >= 1
        assert rfs[0].name == "gym_reward"

    def test_extra_rewards(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            extra_rewards=(FormatReward(RawGenerationExtractor()),),
            num_tasks=1,
        )
        rfs = env.reward_functions
        assert len(rfs) == 2
        names = {r.name for r in rfs}
        assert "gym_reward" in names
        assert "format" in names

    # --- State tracker ---

    def test_state_continuity_enforced(self, discrete_env):
        """Stepping from a stale state should raise."""
        state, _ = discrete_env.reset()
        result = discrete_env.step(state, Action(text="1"))
        # Try stepping from original state again — should fail
        with pytest.raises(NotImplementedError, match="stale"):
            discrete_env.step(state, Action(text="0"))

    # --- Prompts ---

    def test_custom_prompts(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            prompts={"description": "A custom game description."},
            num_tasks=1,
        )
        assert "description" in env.prompts

    def test_prompts_property(self, discrete_env):
        assert isinstance(discrete_env.prompts, dict)

    # --- ANSI render ---

    def test_ansi_render_used_when_available(self, import_adapter):
        gym_env = MockGymEnv()
        gym_env._render_output = "ANSI output here"
        gym_env.render_mode = "ansi"
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            use_ansi_render=True,
            num_tasks=1,
        )
        state, _ = env.reset()
        # The prompt or observation should contain the ANSI output
        assert "ANSI output here" in state.observation.prompt

    def test_ansi_render_fallback_to_mapper(self, import_adapter):
        gym_env = MockGymEnv()
        gym_env._render_output = None  # render returns None
        gym_env.render_mode = "ansi"
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            use_ansi_render=True,
            num_tasks=1,
        )
        state, _ = env.reset()
        # Should still have an observation (from mapper)
        assert len(state.observation.prompt) > 0

    # --- compute_rewards ---

    def test_compute_rewards(self, discrete_env):
        state, _ = discrete_env.reset()
        result = discrete_env.step(state, Action(text="1"))
        rewards = discrete_env.compute_rewards(
            state, Action(text="1"), result.next_state
        )
        assert isinstance(rewards, SignalBundle)


# ===========================================================================
# GymnasiumAdapter tests
# ===========================================================================

class TestGymnasiumAdapter:

    @pytest.fixture
    def adapter(self, import_adapter):
        return import_adapter.GymnasiumAdapter()

    def test_name(self, adapter):
        assert adapter.name == "gymnasium"

    def test_list_environments(self, adapter):
        envs = adapter.list_environments()
        assert isinstance(envs, list)
        # Should include presets
        assert len(envs) > 0

    def test_get_environment_with_gym_env(self, adapter, import_adapter):
        gym_env = MockGymEnv()
        env = adapter.get_environment(
            "test-env", gym_env=gym_env, num_tasks=5
        )
        assert isinstance(env, import_adapter.GymnasiumEnvironment)

    def test_get_environment_preset_merging(self, adapter, import_adapter):
        """Explicit params should override preset defaults."""
        # Use a preset that exists
        presets = import_adapter.GYMNASIUM_PRESETS
        if presets:
            preset_name = next(iter(presets))
            # Should not raise when passing gym_env for a preset
            gym_env = MockGymEnv()
            env = adapter.get_environment(
                preset_name,
                gym_env=gym_env,
                max_steps=99,
                num_tasks=1,
            )
            assert env.spec.max_steps == 99

    def test_get_environment_unknown_without_gym_env(self, adapter):
        """Unknown env without gym_env should try gymnasium.make."""
        # With mocked gymnasium, make() will be MagicMock
        # which is fine — it returns a MagicMock that has attributes
        import sys
        mock_gymnasium = sys.modules["gymnasium"]
        mock_env = MockGymEnv()
        mock_gymnasium.make = MagicMock(return_value=mock_env)

        env = adapter.get_environment("CartPole-v1", num_tasks=5)
        mock_gymnasium.make.assert_called_once()

    def test_get_default_system_prompt(self, adapter):
        assert adapter.get_default_system_prompt("test") is None

    def test_get_prompt_template(self, adapter):
        assert adapter.get_prompt_template("test") is None

    def test_get_native_answer_extractor(self, adapter):
        assert adapter.get_native_answer_extractor("test") is None

    def test_get_environment_info(self, adapter):
        info = adapter.get_environment_info("test-env")
        assert info["adapter"] == "gymnasium"

    def test_render_mode_auto_ansi(self, adapter):
        """use_ansi_render=True should set render_mode='ansi'."""
        import sys
        mock_gymnasium = sys.modules["gymnasium"]
        mock_env = MockGymEnv()
        mock_gymnasium.make = MagicMock(return_value=mock_env)

        env = adapter.get_environment(
            "CartPole-v1",
            use_ansi_render=True,
            num_tasks=1,
        )
        call_kwargs = mock_gymnasium.make.call_args
        assert call_kwargs is not None
        # render_mode should be "ansi"
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("render_mode") == "ansi"
        else:
            # positional + keyword
            assert "ansi" in str(call_kwargs)


# ===========================================================================
# Preset tests
# ===========================================================================

class TestPresets:

    def test_presets_exist(self, import_adapter):
        assert isinstance(import_adapter.GYMNASIUM_PRESETS, dict)
        assert len(import_adapter.GYMNASIUM_PRESETS) > 0

    def test_mars_explorer_preset(self, import_adapter):
        presets = import_adapter.GYMNASIUM_PRESETS
        if "mars_explorer" in presets:
            preset = presets["mars_explorer"]
            assert "action_names" in preset or "action_mapper" in preset

    def test_preset_keys_are_strings(self, import_adapter):
        for key in import_adapter.GYMNASIUM_PRESETS:
            assert isinstance(key, str)


# ===========================================================================
# GymnasiumHidden tests
# ===========================================================================

class TestGymnasiumHidden:

    def test_frozen(self, import_adapter):
        hidden = import_adapter.GymnasiumHidden(
            task_index=0, seed=42, episode_step=0,
            last_action=None, raw_observation=0, gym_reward=0.0,
        )
        with pytest.raises(AttributeError):
            hidden.task_index = 1

    def test_fields(self, import_adapter):
        hidden = import_adapter.GymnasiumHidden(
            task_index=5, seed=123, episode_step=3,
            last_action="right", raw_observation=np.array([1.0, 2.0]),
            gym_reward=1.5,
        )
        assert hidden.task_index == 5
        assert hidden.seed == 123
        assert hidden.episode_step == 3
        assert hidden.last_action == "right"
        assert hidden.gym_reward == 1.5


# ===========================================================================
# Integration: full episode
# ===========================================================================

class TestFullEpisode:

    def test_complete_discrete_episode(self, import_adapter):
        """Play through a full episode with discrete env."""
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
            terminal_step=3,
            terminal_reward=1.0,
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            action_names={0: "left", 1: "right", 2: "up"},
            num_tasks=1,
        )

        state, _ = env.reset(seed=42)
        assert not state.metadata.is_terminal

        # 3 steps to terminal
        for i in range(3):
            result = env.step(state, Action(text="right"))
            state = result.next_state

        assert result.terminated is True
        reward = result.rewards.by_name("gym_reward")
        assert reward is not None
        assert reward.reward == 1.0

    def test_complete_box_episode(self, import_adapter):
        """Play through a full episode with continuous env."""
        gym_env = MockBoxGymEnv(obs_dim=4, action_dim=2)
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            observation_labels={0: "x", 1: "y", 2: "vx", 3: "vy"},
            num_tasks=1,
        )

        state, _ = env.reset()
        while not state.metadata.is_terminal:
            result = env.step(state, Action(text="0.5, -0.3"))
            state = result.next_state

        assert result.terminated is True

    def test_invalid_action_recovery(self, import_adapter):
        """Can continue after invalid action."""
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
            terminal_step=5,
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            num_tasks=1,
        )

        state, _ = env.reset()

        # Invalid action
        result = env.step(state, Action(text="bad"))
        state = result.next_state
        assert result.terminated is False

        # Valid action
        result = env.step(state, Action(text="1"))
        assert result.next_state.metadata.step == 2

    def test_episode_with_grid_mapper(self, import_adapter):
        """Play with GridObservationMapper."""
        gym_env = MockGridEnv(grid_size=5)
        value_map = {0.0: ".", 0.6: "R", 1.0: "#"}
        obs_mapper = import_adapter.GridObservationMapper(value_map)
        action_mapper = import_adapter.AutoActionMapper(
            MockDiscrete(4),
            action_names={0: "right", 1: "left", 2: "up", 3: "down"},
        )

        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            observation_mapper=obs_mapper,
            action_mapper=action_mapper,
            num_tasks=1,
        )

        state, _ = env.reset()
        assert "R" in state.observation.prompt
        assert "#" in state.observation.prompt

        result = env.step(state, Action(text="right"))
        assert result.next_state.metadata.step == 1
