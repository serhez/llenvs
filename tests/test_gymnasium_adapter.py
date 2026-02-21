"""Tests for the Gymnasium adapter."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from llenvs.core.extraction import RawGenerationExtractor, TagBasedExtractor
from llenvs.core.reward import FormatReward, RewardType, SignalBundle
from llenvs.core.state import Action, Observation, State

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
        self._current_obs = (
            min(action, self.observation_space.n - 1)
            if isinstance(self.observation_space, MockDiscrete)
            else action
        )
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
        mapper = import_adapter.AutoObservationMapper(space, labels={0: "position", 1: "velocity"})
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
        space = MockDict(
            {
                "pos": MockDiscrete(10),
                "vel": MockBox(-1.0, 1.0, shape=(2,)),
            }
        )
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

        grid = np.array(
            [
                [0.6, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
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
                task_index=0,
                seed=None,
                episode_step=0,
                last_action=None,
                raw_observation=0,
                gym_reward=0.0,
            ),
            metadata=StateMetadata(step=0, episode_id="ep1"),
        )
        next_state = State(
            observation=Observation(prompt="test"),
            hidden=import_adapter.GymnasiumHidden(
                task_index=0,
                seed=None,
                episode_step=1,
                last_action="1",
                raw_observation=1,
                gym_reward=0.5,
            ),
            metadata=StateMetadata(
                step=1, episode_id="ep1", is_terminal=False, info={"gym_reward": 0.5}
            ),
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
                task_index=0,
                seed=None,
                episode_step=2,
                last_action="1",
                raw_observation=1,
                gym_reward=0.5,
            ),
            metadata=StateMetadata(step=2, episode_id="ep1"),
        )
        next_state = State(
            observation=Observation(prompt="test"),
            hidden=import_adapter.GymnasiumHidden(
                task_index=0,
                seed=None,
                episode_step=3,
                last_action="2",
                raw_observation=2,
                gym_reward=1.5,
            ),
            metadata=StateMetadata(
                step=3, episode_id="ep1", is_terminal=True, info={"gym_reward": 1.0}
            ),
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

    def test_reset_initial_observation_in_messages(self, discrete_env):
        state, _ = discrete_env.reset()
        # Step 0 observation should be in messages, not in prompt
        assert "Step 0" not in state.observation.prompt
        assert len(state.observation.messages) >= 1
        step_0_msg = state.observation.messages[0]
        assert "Step 0" in step_0_msg["content"]

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

    def test_reset_messages_has_step_0(self, discrete_env):
        state, _ = discrete_env.reset()
        # Messages should contain the step-0 observation
        assert len(state.observation.messages) == 1
        assert state.observation.messages[0]["role"] == "user"
        assert "Step 0" in state.observation.messages[0]["content"]

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
        # After reset: 1 message (step-0 observation)
        assert len(state.observation.messages) == 1

        # Step 1: +2 (assistant + user) = 3 total
        result = discrete_env.step(state, Action(text="1"))
        assert len(result.next_state.observation.messages) == 3
        assert result.next_state.observation.messages[1]["role"] == "assistant"
        assert result.next_state.observation.messages[2]["role"] == "user"

        # Step 2: +2 = 5 total
        result2 = discrete_env.step(result.next_state, Action(text="0"))
        assert len(result2.next_state.observation.messages) == 5

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
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env, seeds=[1, 2, 3, 4, 5])
        assert len(env) == 5

    def test_len_raises_without_seeds_or_num_tasks(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env)
        with pytest.raises(TypeError):
            len(env)

    def test_seed_from_seeds_list(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env, seeds=[10, 20, 30])
        state, _ = env.reset(options={"task_index": 1})
        assert state.hidden.seed == 20

    def test_seed_explicit_overrides_seeds_list(self, import_adapter):
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env, seeds=[10, 20, 30])
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
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env, num_tasks=1)
        assert env.spec.max_steps == 200

    def test_spec_explicit_max_steps_overrides(self, import_adapter):
        gym_env = MockGymEnv(max_episode_steps=200)
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env, max_steps=50, num_tasks=1)
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
        # The ANSI output should be in the state content and messages
        assert "ANSI output here" in state.observation.state.text
        assert "ANSI output here" in state.observation.messages[0]["content"]

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
        rewards = discrete_env.compute_rewards(state, Action(text="1"), result.next_state)
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
        env = adapter.get_environment("test-env", gym_env=gym_env, num_tasks=5)
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
# Atari preset tests
# ===========================================================================


ATARI_PRESETS = [
    "atari/breakout",
    "atari/pong",
    "atari/space_invaders",
    "atari/ms_pacman",
    "atari/freeway",
]


class TestAtariPresets:
    def test_atari_presets_exist(self, import_adapter):
        """All Atari presets should be in GYMNASIUM_PRESETS."""
        for name in ATARI_PRESETS:
            assert name in import_adapter.GYMNASIUM_PRESETS, f"Missing preset: {name}"

    def test_atari_preset_has_image_mapper(self, import_adapter):
        """Each Atari preset should use ImageObservationMapper."""
        for name in ATARI_PRESETS:
            preset = import_adapter.GYMNASIUM_PRESETS[name]
            mapper = preset.get("observation_mapper")
            assert isinstance(mapper, import_adapter.ImageObservationMapper), (
                f"{name} should have ImageObservationMapper, got {type(mapper)}"
            )

    def test_atari_preset_has_action_names(self, import_adapter):
        """Each Atari preset should define action_names."""
        for name in ATARI_PRESETS:
            preset = import_adapter.GYMNASIUM_PRESETS[name]
            assert "action_names" in preset, f"{name} missing action_names"
            assert isinstance(preset["action_names"], dict)
            assert len(preset["action_names"]) > 0

    def test_atari_preset_has_max_steps(self, import_adapter):
        """Each Atari preset should define max_steps."""
        for name in ATARI_PRESETS:
            preset = import_adapter.GYMNASIUM_PRESETS[name]
            assert "max_steps" in preset, f"{name} missing max_steps"
            assert preset["max_steps"] > 0

    def test_atari_preset_has_env_id(self, import_adapter):
        """Each Atari preset should have an ALE/ env_id."""
        for name in ATARI_PRESETS:
            preset = import_adapter.GYMNASIUM_PRESETS[name]
            assert "env_id" in preset
            assert preset["env_id"].startswith("ALE/")

    def test_atari_preset_has_description(self, import_adapter):
        """Each Atari preset should have a description."""
        for name in ATARI_PRESETS:
            preset = import_adapter.GYMNASIUM_PRESETS[name]
            assert "description" in preset
            assert "VLM" in preset["description"]

    def test_atari_preset_integration(self, import_adapter):
        """Integration: mock pixel env with Atari preset, verify ImageContent in observation."""
        import sys

        mock_gymnasium = sys.modules["gymnasium"]

        # Create a mock env that returns pixel observations
        pixel_obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        mock_gym_env = MagicMock()
        mock_gym_env.observation_space = MockBox(0, 255, shape=(210, 160, 3), dtype=np.uint8)
        mock_gym_env.action_space = MockDiscrete(4)
        mock_gym_env.spec = MockSpec(id="ALE/Breakout-v5")
        mock_gym_env.reset.return_value = (pixel_obs, {})
        mock_gym_env.step.return_value = (pixel_obs, 1.0, False, False, {})

        mock_gymnasium.make = MagicMock(return_value=mock_gym_env)

        adapter = import_adapter.GymnasiumAdapter()
        env = adapter.get_environment("atari/breakout", num_tasks=10)

        # Verify the mapper is ImageObservationMapper
        assert isinstance(env._observation_mapper, import_adapter.ImageObservationMapper)

        # Reset and check observation has images
        state, _ = env.reset()
        assert len(state.observation.images) == 1

        from llenvs.core.state import ImageContent

        assert isinstance(state.observation.images[0], ImageContent)

        # Step and check images
        result = env.step(state, Action(text="fire"))
        assert len(result.next_state.observation.images) == 1

    def test_atari_preset_max_steps_merges(self, import_adapter):
        """Preset max_steps should be used when not explicitly overridden."""
        import sys

        mock_gymnasium = sys.modules["gymnasium"]
        mock_gym_env = MagicMock()
        mock_gym_env.observation_space = MockBox(0, 255, shape=(210, 160, 3), dtype=np.uint8)
        mock_gym_env.action_space = MockDiscrete(4)
        mock_gym_env.spec = MockSpec(id="ALE/Breakout-v5")
        mock_gymnasium.make = MagicMock(return_value=mock_gym_env)

        adapter = import_adapter.GymnasiumAdapter()

        # Without explicit max_steps, should use preset (2000)
        env = adapter.get_environment("atari/breakout", num_tasks=1)
        assert env.spec.max_steps == 2000

        # With explicit max_steps, should override
        env2 = adapter.get_environment("atari/breakout", num_tasks=1, max_steps=500)
        assert env2.spec.max_steps == 500


# ===========================================================================
# GymnasiumHidden tests
# ===========================================================================


class TestGymnasiumHidden:
    def test_frozen(self, import_adapter):
        hidden = import_adapter.GymnasiumHidden(
            task_index=0,
            seed=42,
            episode_step=0,
            last_action=None,
            raw_observation=0,
            gym_reward=0.0,
        )
        with pytest.raises(AttributeError):
            hidden.task_index = 1

    def test_fields(self, import_adapter):
        hidden = import_adapter.GymnasiumHidden(
            task_index=5,
            seed=123,
            episode_step=3,
            last_action="right",
            raw_observation=np.array([1.0, 2.0]),
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


# =============================================================================
# FrozenLake Observation Mapper
# =============================================================================


class TestFrozenLakeObservationMapper:
    """Tests for FrozenLakeObservationMapper."""

    @pytest.fixture
    def import_adapter(self, mock_gymnasium_spaces):
        import importlib

        import llenvs.adapters.gymnasium as mod

        importlib.reload(mod)
        return mod

    def test_4x4_grid_rendering(self, import_adapter):
        desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
        mapper = import_adapter.FrozenLakeObservationMapper(desc=desc)
        # Agent at position 0 (top-left = S)
        result = mapper.map(0, {})
        assert "@ F F F" in result
        # Position 5 (row 1, col 1 = H)
        result = mapper.map(5, {})
        lines = result.split("\n")
        assert lines[1] == "F @ F H"

    def test_8x8_grid_rendering(self, import_adapter):
        desc = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
        mapper = import_adapter.FrozenLakeObservationMapper(desc=desc)
        # Position 0 (top-left)
        result = mapper.map(0, {})
        assert result.startswith("@ F F F F F F F")
        # Position 63 (bottom-right = G)
        result = mapper.map(63, {})
        last_line = result.split("\n")[-1]
        assert last_line.endswith("@")

    def test_numpy_byte_desc(self, import_adapter):
        desc = np.array([[b"S", b"F"], [b"H", b"G"]])
        mapper = import_adapter.FrozenLakeObservationMapper(desc=desc)
        result = mapper.map(0, {})
        assert "@ F" in result

    def test_describe(self, import_adapter):
        desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
        mapper = import_adapter.FrozenLakeObservationMapper(desc=desc)
        description = mapper.describe()
        assert "start" in description.lower() or "S" in description
        assert "goal" in description.lower() or "G" in description
        assert "hole" in description.lower() or "H" in description
        assert "4x4" in description

    def test_agent_char_custom(self, import_adapter):
        desc = ["SF", "HG"]
        mapper = import_adapter.FrozenLakeObservationMapper(desc=desc, agent_char="A")
        result = mapper.map(0, {})
        assert "A" in result
        assert "@" not in result


# =============================================================================
# FrozenLake Presets
# =============================================================================


class TestFrozenLakePreset:
    """Tests for FrozenLake presets in GYMNASIUM_PRESETS."""

    def test_preset_exists(self):
        from llenvs.adapters.gymnasium import GYMNASIUM_PRESETS

        assert "frozen_lake" in GYMNASIUM_PRESETS
        assert "frozen_lake/8x8" in GYMNASIUM_PRESETS

    def test_preset_fields(self):
        from llenvs.adapters.gymnasium import GYMNASIUM_PRESETS

        preset = GYMNASIUM_PRESETS["frozen_lake"]
        assert preset["env_id"] == "FrozenLake-v1"
        assert preset["max_steps"] == 100
        assert preset["action_names"] == {0: "left", 1: "down", 2: "right", 3: "up"}

    def test_preset_8x8_has_make_kwargs(self):
        from llenvs.adapters.gymnasium import GYMNASIUM_PRESETS

        preset = GYMNASIUM_PRESETS["frozen_lake/8x8"]
        assert preset["make_kwargs"] == {"map_name": "8x8"}
        assert preset["max_steps"] == 200


# =============================================================================
# Prompt Separation (task vs state)
# =============================================================================


class TestPromptSeparation:
    """Tests verifying task/state separation in observations."""

    @pytest.fixture
    def env(self, import_adapter):
        gym_env = MockGymEnv()
        return import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            action_names={0: "left", 1: "right", 2: "up"},
            num_tasks=10,
        )

    def test_prompt_has_no_step_0(self, env):
        state, _ = env.reset()
        assert "Step 0" not in state.observation.prompt
        assert "State:" not in state.observation.prompt

    def test_step_0_in_messages(self, env):
        state, _ = env.reset()
        assert len(state.observation.messages) == 1
        assert "Step 0" in state.observation.messages[0]["content"]

    def test_prompt_stable_across_steps(self, env):
        state, _ = env.reset()
        initial_prompt = state.observation.prompt

        result = env.step(state, Action(text="1"))
        assert result.next_state.observation.prompt == initial_prompt

        result2 = env.step(result.next_state, Action(text="0"))
        assert result2.next_state.observation.prompt == initial_prompt


# =============================================================================
# Structured Observation (task/state fields)
# =============================================================================


class TestStructuredObservation:
    """Tests for ObservationContent task/state fields on Observation."""

    @pytest.fixture
    def env(self, import_adapter):
        gym_env = MockGymEnv()
        return import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            action_names={0: "left", 1: "right", 2: "up"},
            num_tasks=10,
        )

    def test_task_set_on_reset(self, env):
        state, _ = env.reset()
        obs = state.observation
        assert obs.task is not None
        assert "gymnasium environment" in obs.task.text.lower()
        assert "Action space" in obs.task.text

    def test_state_set_on_reset(self, env):
        state, _ = env.reset()
        obs = state.observation
        assert obs.state is not None
        assert "Step 0" in obs.state.text

    def test_state_updates_on_step(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="1"))
        next_obs = result.next_state.observation
        assert next_obs.state is not None
        assert "Step 1" in next_obs.state.text
        assert "Step 0" not in next_obs.state.text

    def test_task_stable_across_steps(self, env):
        state, _ = env.reset()
        task_text = state.observation.task.text

        result = env.step(state, Action(text="1"))
        assert result.next_state.observation.task.text == task_text

        result2 = env.step(result.next_state, Action(text="0"))
        assert result2.next_state.observation.task.text == task_text

    def test_error_step_has_task_and_state(self, env):
        state, _ = env.reset()
        # Trigger an error by sending invalid action
        result = env.step(state, Action(text="invalid_action_xyz"))
        next_obs = result.next_state.observation
        assert next_obs.task is not None
        assert next_obs.state is not None
        assert "Error" in next_obs.state.text


# =============================================================================
# Pure Step (pickle-based state snapshots)
# =============================================================================


class TestPureStep:
    """Tests for pure_step=True mode with pickle-based gym state snapshots."""

    @pytest.fixture
    def pure_env(self, import_adapter):
        """Create a pure_step=True gymnasium environment."""
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
            terminal_step=10,
            reward_per_step=0.1,
            terminal_reward=1.0,
        )
        return import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            action_names={0: "left", 1: "right", 2: "up"},
            num_tasks=10,
            pure_step=True,
        )

    def test_pure_step_basic(self, pure_env):
        """pure_step=True works for basic reset and step."""
        state, _ = pure_env.reset(seed=42)
        assert state.hidden.gym_snapshot is not None
        assert isinstance(state.hidden.gym_snapshot, bytes)

        result = pure_env.step(state, Action(text="right"))
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.gym_snapshot is not None

    def test_spec_reflects_pure_step(self, pure_env, import_adapter):
        """env.spec.pure_step should match constructor arg."""
        assert pure_env.spec.pure_step is True

        # Default (False)
        gym_env = MockGymEnv()
        env = import_adapter.GymnasiumEnvironment(gym_env=gym_env, num_tasks=1)
        assert env.spec.pure_step is False

    def test_stepping_from_stale_state_works(self, pure_env):
        """With pure_step=True, stepping from a previous state should work."""
        state, _ = pure_env.reset(seed=42)
        # Step to get state_1
        result1 = pure_env.step(state, Action(text="right"))
        state_1 = result1.next_state

        # Step from state_1 to get state_2
        result2 = pure_env.step(state_1, Action(text="left"))

        # Now step from the *original* state again — this would fail without pure_step
        result_from_stale = pure_env.step(state, Action(text="up"))
        assert result_from_stale.next_state.metadata.step == 1
        assert result_from_stale.next_state.hidden.episode_step == 1

    def test_branching_produces_independent_trajectories(self, pure_env):
        """Checkpoint, branch, and verify divergent trajectories from same state."""
        state, _ = pure_env.reset(seed=42)

        # Take 2 steps
        result = pure_env.step(state, Action(text="right"))
        state_1 = result.next_state
        result = pure_env.step(state_1, Action(text="left"))
        state_2 = result.next_state

        # Branch A: take "right" from state_2
        result_a = pure_env.step(state_2, Action(text="right"))
        # Branch B: take "up" from the same state_2
        result_b = pure_env.step(state_2, Action(text="up"))

        # Both should advance from the same checkpoint
        assert result_a.next_state.hidden.episode_step == 3
        assert result_b.next_state.hidden.episode_step == 3
        # But with different actions
        assert result_a.next_state.hidden.last_action == "right"
        assert result_b.next_state.hidden.last_action == "up"

    def test_error_step_preserves_snapshot(self, pure_env):
        """Invalid action shouldn't corrupt gym state; next valid step works."""
        state, _ = pure_env.reset(seed=42)

        # Invalid action
        result = pure_env.step(state, Action(text="invalid_garbage"))
        error_state = result.next_state
        assert error_state.hidden.gym_snapshot is not None
        # Snapshot should be the same as the original state's snapshot
        assert error_state.hidden.gym_snapshot == state.hidden.gym_snapshot

        # Next valid step should work from the error state
        result2 = pure_env.step(error_state, Action(text="right"))
        assert result2.next_state.metadata.step == 2
        assert result2.next_state.hidden.gym_snapshot is not None

    def test_unpicklable_env_raises_type_error(self, import_adapter):
        """pure_step=True with unpicklable env should raise TypeError eagerly."""

        class UnpicklableEnv:
            observation_space = MockDiscrete(4)
            action_space = MockDiscrete(3)
            spec = MockSpec()
            render_mode = None

            def __reduce__(self):
                raise TypeError("cannot pickle")

            def render(self):
                return None

        with pytest.raises(TypeError, match="pickle|picklable"):
            import_adapter.GymnasiumEnvironment(
                gym_env=UnpicklableEnv(),
                num_tasks=1,
                pure_step=True,
            )

    def test_default_pure_step_false_unchanged(self, import_adapter):
        """Default (pure_step=False) should still enforce StateContinuityTracker."""
        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
            terminal_step=10,
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            num_tasks=10,
        )
        assert env.spec.pure_step is False

        state, _ = env.reset()
        # Hidden should NOT have a snapshot
        assert state.hidden.gym_snapshot is None

        result = env.step(state, Action(text="1"))
        assert result.next_state.hidden.gym_snapshot is None

        # Stale state should raise (StateContinuityTracker enforced)
        with pytest.raises(NotImplementedError, match="stale"):
            env.step(state, Action(text="0"))

    def test_adapter_pure_step_passthrough(self, import_adapter):
        """GymnasiumAdapter.get_environment should pass pure_step through."""
        gym_env = MockGymEnv()
        adapter = import_adapter.GymnasiumAdapter()
        env = adapter.get_environment("test", gym_env=gym_env, num_tasks=1, pure_step=True)
        assert env.spec.pure_step is True
