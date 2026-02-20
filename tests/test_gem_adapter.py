"""Tests for the GEM adapter."""

from typing import Any

import pytest

from llenvs.adapters.gem import (
    MULTI_TURN_ENVS,
    GemAdapter,
    GemCorrectnessReward,
    GemEnvironment,
    GemHidden,
    _freeze_dict,
    _is_multi_turn_env,
    _unfreeze_dict,
)
from llenvs.core.extraction import RegexExtractor, TagBasedExtractor
from llenvs.core.registry import EnvironmentRegistry
from llenvs.core.reward import FormatReward, RewardType
from llenvs.core.state import Action, Observation, ObservationContent


class MockGemEnv:
    """Mock GEM environment for testing."""

    def __init__(
        self,
        env_id: str = "game:GuessTheNumber-v0",
        target: int = 50,
        max_guesses: int = 10,
    ):
        self.env_id = env_id
        self.target = target
        self.max_guesses = max_guesses
        self._guesses = 0
        self._terminated = False
        self._last_obs = ""

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Reset the mock environment."""
        self._guesses = 0
        self._terminated = False
        self._last_obs = (
            f"I'm thinking of a number between 1 and 100. You have {self.max_guesses} guesses."
        )
        return self._last_obs, {"target": self.target}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Take a step in the mock environment."""
        self._guesses += 1

        try:
            guess = int(action.strip())
        except ValueError:
            obs = "Please enter a valid number."
            return obs, 0.0, False, False, {"guess": None}

        if guess == self.target:
            obs = f"Correct! The number was {self.target}."
            reward = 1.0
            self._terminated = True
        elif guess < self.target:
            obs = f"{guess} is too low."
            reward = 0.0
        else:
            obs = f"{guess} is too high."
            reward = 0.0

        truncated = self._guesses >= self.max_guesses and not self._terminated

        return obs, reward, self._terminated, truncated, {"guess": guess}

    def get_state(self) -> dict[str, Any]:
        """Get serialized state."""
        return {
            "guesses": self._guesses,
            "terminated": self._terminated,
            "last_obs": self._last_obs,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore from serialized state."""
        self._guesses = state["guesses"]
        self._terminated = state["terminated"]
        self._last_obs = state["last_obs"]


class MockSingleTurnEnv:
    """Mock single-turn GEM environment (like math problems)."""

    def __init__(self, env_id: str = "math:GSM8K"):
        self.env_id = env_id
        self.problem = "What is 2 + 2?"
        self.answer = "4"
        self._state = {}

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Reset with a math problem."""
        self._state = {"problem": self.problem, "answer": self.answer}
        return self.problem, {"answer": self.answer}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Check answer (always terminates)."""
        # Extract answer from action
        if "4" in action:
            reward = 1.0
            obs = "Correct!"
        else:
            reward = 0.0
            obs = f"Incorrect. The answer was {self.answer}."

        return obs, reward, True, False, {"submitted_answer": action}

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)


@pytest.fixture
def mock_gem_env() -> MockGemEnv:
    """Create mock GEM environment."""
    return MockGemEnv()


@pytest.fixture
def mock_single_turn_env() -> MockSingleTurnEnv:
    """Create mock single-turn environment."""
    return MockSingleTurnEnv()


class TestMultiTurnDetection:
    """Tests for multi-turn environment detection."""

    def test_known_multi_turn_envs(self):
        """Test that known multi-turn envs are detected."""
        assert _is_multi_turn_env("game:GuessTheNumber-v0") is True
        assert _is_multi_turn_env("game:Sudoku-v0") is True
        assert _is_multi_turn_env("game:Wordle-v0") is True

    def test_multi_turn_prefix_matching(self):
        """Test prefix-based multi-turn detection."""
        assert _is_multi_turn_env("game:NewGame-v0") is True
        assert _is_multi_turn_env("game:CustomGame-v1") is True

    def test_single_turn_envs(self):
        """Test that single-turn envs are not detected as multi-turn."""
        assert _is_multi_turn_env("math:GSM8K") is False
        assert _is_multi_turn_env("code:CodeContest") is False
        assert _is_multi_turn_env("qa:HotpotQA") is False

    def test_multi_turn_envs_set(self):
        """Test MULTI_TURN_ENVS contains expected envs."""
        assert "game:GuessTheNumber-v0" in MULTI_TURN_ENVS
        assert "game:Sudoku-v0" in MULTI_TURN_ENVS
        assert "game:Wordle-v0" in MULTI_TURN_ENVS
        assert "game:Mastermind-v0" in MULTI_TURN_ENVS


class TestFreezeUnfreeze:
    """Tests for dict freezing utilities."""

    def test_freeze_dict(self):
        """Test freezing dict to tuple."""
        d = {"a": 1, "b": 2}
        frozen = _freeze_dict(d)
        assert frozen == (("a", 1), ("b", 2))

    def test_unfreeze_dict(self):
        """Test unfreezing tuple back to dict."""
        frozen = (("a", 1), ("b", 2))
        d = _unfreeze_dict(frozen)
        assert d == {"a": 1, "b": 2}

    def test_roundtrip(self):
        """Test freeze/unfreeze roundtrip."""
        original = {"x": 10, "y": 20, "z": 30}
        frozen = _freeze_dict(original)
        restored = _unfreeze_dict(frozen)
        assert restored == original


class TestGemHidden:
    """Tests for GemHidden."""

    def test_creation(self):
        """Test hidden state creation."""
        gem_state = (("key", "value"),)
        hidden = GemHidden(
            env_id="game:Test-v0",
            gem_state=gem_state,
            task_index=0,
            is_multi_turn=True,
            episode_step=5,
        )

        assert hidden.env_id == "game:Test-v0"
        assert hidden.gem_state == gem_state
        assert hidden.task_index == 0
        assert hidden.is_multi_turn is True
        assert hidden.episode_step == 5

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = GemHidden(
            env_id="test",
            gem_state=(),
            task_index=0,
            is_multi_turn=False,
            episode_step=0,
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore


class TestGemCorrectnessReward:
    """Tests for GemCorrectnessReward."""

    def test_reward_name(self):
        """Test reward function name."""
        reward_fn = GemCorrectnessReward()
        assert reward_fn.name == "correctness"

    def test_reward_type(self):
        """Test reward function type."""
        reward_fn = GemCorrectnessReward()
        assert reward_fn.reward_type == RewardType.OUTCOME


class TestFormatRewardWithGem:
    """Tests for FormatReward used with GEM environments."""

    def test_reward_name(self):
        """Test reward function name."""
        extractor = TagBasedExtractor()
        reward_fn = FormatReward(extractor)
        assert reward_fn.name == "format"

    def test_reward_type(self):
        """Test reward function type."""
        extractor = TagBasedExtractor()
        reward_fn = FormatReward(extractor)
        assert reward_fn.reward_type == RewardType.FORMAT


class TestGemEnvironment:
    """Tests for GemEnvironment."""

    def test_creation_multi_turn(self, mock_gem_env):
        """Test multi-turn environment creation."""
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )

        assert env.spec.name == "game:GuessTheNumber-v0"
        assert env.spec.adapter == "gem"
        assert env.spec.is_multi_turn is True

    def test_creation_single_turn(self, mock_single_turn_env):
        """Test single-turn environment creation."""
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
        )

        assert env.spec.name == "math:GSM8K"
        assert env.spec.max_steps == 1
        assert env.spec.is_multi_turn is False

    def test_reset(self, mock_gem_env):
        """Test environment reset."""
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, info = env.reset(seed=42)

        # Check observation
        assert isinstance(state.observation, Observation)
        assert "number" in state.observation.prompt.lower()

        # Check hidden state
        assert isinstance(state.hidden, GemHidden)
        assert state.hidden.env_id == "game:GuessTheNumber-v0"
        assert state.hidden.is_multi_turn is True
        assert state.hidden.episode_step == 0

        # Check metadata
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

        # Structured observation: task and state set on reset
        obs = state.observation
        assert isinstance(obs.task, ObservationContent)
        assert obs.task.text == obs.prompt
        assert isinstance(obs.state, ObservationContent)
        assert obs.state.text == obs.prompt

    def test_step_correct_guess(self, mock_gem_env):
        """Test step with correct answer."""
        mock_gem_env.target = 50
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        action = Action(text="50")
        result = env.step(state, action)

        # Check termination
        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

        # Check reward
        correctness = result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.reward == 1.0
        assert correctness.reward_type == RewardType.OUTCOME

    def test_step_incorrect_guess(self, mock_gem_env):
        """Test step with incorrect answer."""
        mock_gem_env.target = 50
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        action = Action(text="25")
        result = env.step(state, action)

        # Check not terminated
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is False

        # Check reward is step type for intermediate
        correctness = result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.reward == 0.0
        assert correctness.reward_type == RewardType.STEP

        # Check observation indicates too low (in messages, not prompt)
        assert "too low" in result.next_state.observation.messages[-1]["content"].lower()

        # Structured observation: task carried forward, state updated
        next_obs = result.next_state.observation
        assert next_obs.task is not None
        assert next_obs.task.text == state.observation.prompt  # task stays as initial prompt
        assert next_obs.state is not None
        assert "too low" in next_obs.state.text.lower()  # state reflects step feedback

    def test_multi_step_game(self, mock_gem_env):
        """Test playing through multiple steps."""
        mock_gem_env.target = 50
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        # Binary search
        guesses = ["50"]  # Direct hit
        for guess in guesses:
            action = Action(text=guess)
            result = env.step(state, action)
            state = result.next_state

            if result.terminated:
                break

        # Should win
        assert result.terminated is True
        assert result.rewards.by_name("correctness").reward == 1.0

    def test_single_turn_terminates(self, mock_single_turn_env):
        """Test that single-turn env terminates after one step."""
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
        )
        state, _ = env.reset()

        action = Action(text="<answer>4</answer>")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.rewards.by_name("correctness").reward == 1.0

    def test_format_reward_via_extra_rewards(self, mock_single_turn_env):
        """Test format reward when added via extra_rewards."""
        extractor = TagBasedExtractor()
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
            extra_rewards=(FormatReward(extractor),),
        )
        state, _ = env.reset()

        # With tags
        action = Action(text="The answer is <answer>4</answer>")
        result = env.step(state, action)

        format_reward = result.rewards.by_name("format")
        assert format_reward is not None
        assert format_reward.reward == 1.0

    def test_no_format_reward_by_default(self, mock_single_turn_env):
        """Test no format reward by default (native-only)."""
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
        )
        state, _ = env.reset()

        action = Action(text="The answer is 4")
        result = env.step(state, action)

        # No format reward by default
        assert result.rewards.by_name("format") is None

    def test_custom_extractor(self, mock_single_turn_env):
        """Test with custom extractor."""
        extractor = RegexExtractor(pattern=r"answer is (\d+)")
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
            answer_extractor=extractor,
        )
        state, _ = env.reset()

        action = Action(text="The answer is 4")
        result = env.step(state, action)

        assert result.info["extracted_answer"] == "4"

    def test_default_native_only_rewards(self, mock_single_turn_env):
        """Test default rewards are native-only."""
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
        )

        assert len(env.reward_functions) == 1
        assert env.reward_functions[0].name == "correctness"

    def test_state_restoration(self, mock_gem_env):
        """Test that state is properly restored for replay."""
        mock_gem_env.target = 50
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        # Take a step
        action1 = Action(text="25")
        result1 = env.step(state, action1)

        # Take same step from same state (should give same result)
        result1_replay = env.step(state, action1)

        # Results should be identical
        assert result1.next_state.observation.prompt == result1_replay.next_state.observation.prompt
        assert result1.rewards.total == result1_replay.rewards.total


class TestGemAdapter:
    """Tests for GemAdapter."""

    def test_adapter_name(self):
        """Test adapter name property."""
        adapter = GemAdapter()
        assert adapter.name == "gem"

    def test_get_environment_info_multi_turn(self):
        """Test get_environment_info for multi-turn env."""
        adapter = GemAdapter()
        info = adapter.get_environment_info("game:Sudoku-v0")

        assert info["name"] == "game:Sudoku-v0"
        assert info["adapter"] == "gem"
        assert info["type"] == "multi_turn"

    def test_get_environment_info_single_turn(self):
        """Test get_environment_info for single-turn env."""
        adapter = GemAdapter()
        info = adapter.get_environment_info("math:GSM8K")

        assert info["name"] == "math:GSM8K"
        assert info["adapter"] == "gem"
        assert info["type"] == "single_turn"


class TestEnvironmentRegistryIntegration:
    """Tests for EnvironmentRegistry integration."""

    def test_register_gem_adapter(self, monkeypatch):
        """Test GEM adapter registration."""
        registry = EnvironmentRegistry()

        # Mock the gem library check
        def mock_get_gem():
            class MockGem:
                ENV_REGISTRY = {"game:Test-v0": {}}

            return MockGem()

        adapter = GemAdapter()
        monkeypatch.setattr(adapter, "_get_gem", mock_get_gem)

        registry.register_adapter(adapter)
        assert "gem" in registry.list_adapters()

    def test_get_adapter(self, monkeypatch):
        """Test getting GEM adapter from registry."""
        registry = EnvironmentRegistry()

        def mock_get_gem():
            class MockGem:
                ENV_REGISTRY = {}

            return MockGem()

        adapter = GemAdapter()
        monkeypatch.setattr(adapter, "_get_gem", mock_get_gem)

        registry.register_adapter(adapter)
        retrieved = registry.get_adapter("gem")
        assert retrieved is adapter


class TestGemEnvironmentSpec:
    """Tests for GemEnvironment spec property."""

    def test_spec_multi_turn(self, mock_gem_env):
        """Test spec for multi-turn environment."""
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
            max_steps=20,
        )
        spec = env.spec

        assert spec.name == "game:GuessTheNumber-v0"
        assert spec.adapter == "gem"
        assert spec.max_steps == 20
        assert spec.is_multi_turn is True
        assert spec.observation_type == Observation
        assert spec.action_type == Action
        assert spec.pure_step is True

    def test_spec_single_turn(self, mock_single_turn_env):
        """Test spec for single-turn environment."""
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
        )
        spec = env.spec

        assert spec.name == "math:GSM8K"
        assert spec.max_steps == 1
        assert spec.is_multi_turn is False
        assert spec.pure_step is True


class TestSignalBundle:
    """Tests for reward bundle from GEM environments."""

    def test_total_reward_native_only(self, mock_single_turn_env):
        """Test total reward with native-only default."""
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
        )
        state, _ = env.reset()

        action = Action(text="<answer>4</answer>")
        result = env.step(state, action)

        # Only correctness (native) reward
        assert result.rewards.total == 1.0

    def test_total_reward_with_extra(self, mock_single_turn_env):
        """Test total reward with extra format reward."""
        extractor = TagBasedExtractor()
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
            extra_rewards=(FormatReward(extractor),),
        )
        state, _ = env.reset()

        action = Action(text="<answer>4</answer>")
        result = env.step(state, action)

        # correctness + format = 1.0 + 1.0 = 2.0
        assert result.rewards.total == 2.0

    def test_reward_by_type(self, mock_single_turn_env):
        """Test filtering rewards by type."""
        extractor = TagBasedExtractor()
        env = GemEnvironment(
            gem_env=mock_single_turn_env,
            env_id="math:GSM8K",
            is_multi_turn=False,
            extra_rewards=(FormatReward(extractor),),
        )
        state, _ = env.reset()

        action = Action(text="<answer>4</answer>")
        result = env.step(state, action)

        outcome_rewards = result.rewards.by_type(RewardType.OUTCOME)
        format_rewards = result.rewards.by_type(RewardType.FORMAT)

        assert len(outcome_rewards) == 1
        assert len(format_rewards) == 1


class MockStatelessGemEnv:
    """Mock GEM environment without get_state/set_state (like some game envs)."""

    def __init__(self, target: int = 50, max_guesses: int = 10):
        self.target = target
        self.max_guesses = max_guesses
        self._guesses = 0
        self._terminated = False

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        self._guesses = 0
        self._terminated = False
        return f"Guess a number between 1 and 100. You have {self.max_guesses} guesses.", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self._guesses += 1
        try:
            guess = int(action.strip())
        except ValueError:
            return "Invalid number.", 0.0, False, False, {}

        if guess == self.target:
            self._terminated = True
            return f"Correct! It was {self.target}.", 1.0, True, False, {}
        elif guess < self.target:
            return f"{guess} is too low.", 0.0, False, False, {}
        else:
            return f"{guess} is too high.", 0.0, False, False, {}


@pytest.fixture
def mock_stateless_env() -> MockStatelessGemEnv:
    return MockStatelessGemEnv()


class TestStatelessGemEnvironment:
    """Tests for GEM environments without get_state/set_state support."""

    def test_reset_works_without_state_methods(self, mock_stateless_env):
        """Reset should work even without get_state/set_state."""
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, info = env.reset(seed=42)

        assert isinstance(state.observation, Observation)
        assert isinstance(state.hidden.gem_state, dict)
        assert state.metadata.is_terminal is False

    def test_step_works_without_state_methods(self, mock_stateless_env):
        """Step should work even without get_state/set_state."""
        mock_stateless_env.target = 50
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        result = env.step(state, Action(text="50"))
        assert result.terminated is True
        assert result.rewards.by_name("correctness").reward == 1.0

    def test_multi_step_without_state_methods(self, mock_stateless_env):
        """Multi-step gameplay should work without state methods."""
        mock_stateless_env.target = 50
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        # Step 1: too low — feedback is in messages, not prompt
        result = env.step(state, Action(text="25"))
        assert "too low" in result.next_state.observation.messages[-1]["content"].lower()
        state = result.next_state

        # Step 2: correct
        result = env.step(state, Action(text="50"))
        assert result.terminated is True

    def test_no_direct_branching_without_state_methods(self, mock_stateless_env):
        """Stateless envs now use deepcopy fallback, so pure_step=True."""
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        assert env.spec.pure_step is True

    def test_direct_branching_with_state_methods(self, mock_gem_env):
        """With state methods, pure_step=True (DirectStrategy)."""
        env = GemEnvironment(
            gem_env=mock_gem_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        assert env.spec.pure_step is True

    def test_gem_state_deepcopy_without_state_methods(self, mock_stateless_env):
        """Hidden gem_state should be a dict (deepcopy) when no state support."""
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()
        assert isinstance(state.hidden.gem_state, dict)

        result = env.step(state, Action(text="25"))
        assert isinstance(result.next_state.hidden.gem_state, dict)

    def test_state_restoration_deepcopy(self, mock_stateless_env):
        """Step from same state twice on stateless env gives identical results."""
        mock_stateless_env.target = 50
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        result1 = env.step(state, Action(text="25"))
        result2 = env.step(state, Action(text="25"))

        assert result1.next_state.observation.messages == result2.next_state.observation.messages
        assert result1.rewards.total == result2.rewards.total

    def test_interleaved_steps_no_corruption(self, mock_stateless_env):
        """Step from same initial state with different actions — results are independent."""
        mock_stateless_env.target = 50
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        result_low = env.step(state, Action(text="25"))
        result_high = env.step(state, Action(text="75"))

        # Each result should reflect only its own action
        assert "too low" in result_low.next_state.observation.messages[-1]["content"].lower()
        assert "too high" in result_high.next_state.observation.messages[-1]["content"].lower()

        # Both should be at episode_step 1
        assert result_low.next_state.hidden.episode_step == 1
        assert result_high.next_state.hidden.episode_step == 1

    def test_message_history_accumulates(self, mock_stateless_env):
        """Message history grows by 2 each turn; prompt stays as initial instructions."""
        mock_stateless_env.target = 50
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()
        initial_prompt = state.observation.prompt

        # Turn 1
        result = env.step(state, Action(text="25"))
        state = result.next_state
        assert len(state.observation.messages) == 2
        assert state.observation.messages[0] == {"role": "assistant", "content": "25"}
        assert "too low" in state.observation.messages[1]["content"].lower()
        assert state.observation.prompt == initial_prompt

        # Turn 2
        result = env.step(state, Action(text="75"))
        state = result.next_state
        assert len(state.observation.messages) == 4
        assert state.observation.messages[2] == {"role": "assistant", "content": "75"}
        assert "too high" in state.observation.messages[3]["content"].lower()
        assert state.observation.prompt == initial_prompt

        # Turn 3: correct
        result = env.step(state, Action(text="50"))
        state = result.next_state
        assert len(state.observation.messages) == 6
        assert state.observation.prompt == initial_prompt

    def test_message_history_on_correct_guess(self, mock_stateless_env):
        """Messages accumulate even on terminal step."""
        mock_stateless_env.target = 50
        env = GemEnvironment(
            gem_env=mock_stateless_env,
            env_id="game:GuessTheNumber-v0",
            is_multi_turn=True,
        )
        state, _ = env.reset()

        result = env.step(state, Action(text="50"))
        assert result.terminated is True
        assert len(result.next_state.observation.messages) == 2
        assert result.next_state.observation.messages[0] == {"role": "assistant", "content": "50"}
        assert "correct" in result.next_state.observation.messages[1]["content"].lower()

    def test_action_replay_branching_without_state_methods(self):
        """ActionReplayStrategy works for stateless GEM envs."""
        from llenvs.core.branching import BranchManager

        def env_factory():
            return GemEnvironment(
                gem_env=MockStatelessGemEnv(target=50),
                env_id="game:GuessTheNumber-v0",
                is_multi_turn=True,
            )

        env = env_factory()
        with BranchManager.create(env, strategy="action_replay", env_factory=env_factory) as mgr:
            state, _ = env.reset(seed=42)
            actions = []

            # Step 1: guess too low
            a1 = Action(text="25")
            result1 = env.step(state, a1)
            state = result1.next_state
            actions.append(a1)

            # Checkpoint after first guess
            mgr.checkpoint("after_guess1", state, tuple(actions), {"seed": 42})

            # Branch and try a different second guess
            branch_env, branch_state = mgr.branch("after_guess1")
            result_branch = branch_env.step(branch_state, Action(text="50"))
            assert result_branch.terminated is True
            assert result_branch.rewards.by_name("correctness").reward == 1.0
