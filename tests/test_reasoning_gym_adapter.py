"""Tests for the ReasoningGym adapter."""

import pytest

from llenvs.adapters.reasoning_gym import (
    CorrectnessRewardFunction,
    ReasoningGymAdapter,
    ReasoningGymEnvironment,
    ReasoningGymHidden,
)
from llenvs.core.extraction import RegexExtractor, TagBasedExtractor
from llenvs.core.registry import EnvironmentRegistry
from llenvs.core.reward import FormatReward, RewardType
from llenvs.core.state import Action, Observation, ObservationContent


class TestReasoningGymEnvironment:
    """Tests for ReasoningGymEnvironment."""

    def test_creation(self, mock_dataset):
        """Test environment creation."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)

        assert env.spec.name == "mock_dataset"
        assert env.spec.max_steps == 1
        assert env.spec.is_multi_turn is False
        assert len(env) == 3

    def test_custom_extractor(self, mock_dataset):
        """Test with custom extractor."""
        extractor = RegexExtractor(pattern=r"(\d+)")
        env = ReasoningGymEnvironment(dataset=mock_dataset, answer_extractor=extractor)

        # Use the environment
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="The answer is 4"))

        assert result.info["extracted_answer"] == "4"

    def test_spec(self, mock_dataset):
        """Test environment specification."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        spec = env.spec

        assert spec.observation_type == Observation
        assert spec.action_type == Action
        assert spec.metadata["dataset_size"] == 3
        assert spec.pure_step is True

    def test_reward_functions_default_native_only(self, mock_dataset):
        """Test default reward functions are native-only."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        assert len(env.reward_functions) == 1
        assert env.reward_functions[0].name == "correctness"

    def test_extra_rewards(self, mock_dataset):
        """Test extra_rewards are appended to native rewards."""
        extractor = TagBasedExtractor()
        format_reward = FormatReward(extractor)
        env = ReasoningGymEnvironment(
            dataset=mock_dataset,
            extra_rewards=(format_reward,),
        )
        assert len(env.reward_functions) == 2
        assert env.reward_functions[0].name == "correctness"
        assert env.reward_functions[1].name == "format"

    def test_reset(self, mock_dataset):
        """Test environment reset."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, info = env.reset(options={"task_index": 1})

        # Check observation
        assert isinstance(state.observation, Observation)
        assert state.observation.prompt == "What is 3 * 3?"

        # Check task/state structured fields
        assert state.observation.task is not None
        assert isinstance(state.observation.task, ObservationContent)
        assert state.observation.task.text == state.observation.prompt
        assert state.observation.state is None

        # Check hidden state
        assert isinstance(state.hidden, ReasoningGymHidden)
        assert state.hidden.expected_answer == "9"
        assert state.hidden.task_index == 1

        # Check metadata
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

        # Check info
        assert info["task_index"] == 1
        assert "question" in info

    def test_reset_requires_task_index(self, mock_dataset):
        """Test that reset requires task_index."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)

        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_reset_validates_task_index(self, mock_dataset):
        """Test task_index bounds checking."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)

        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 100})

        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": -1})

    def test_reset_custom_episode_id(self, mock_dataset):
        """Test reset with custom episode ID."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0, "episode_id": "custom-id"})

        assert state.metadata.episode_id == "custom-id"

    def test_step_correct_answer(self, mock_dataset):
        """Test step with correct answer."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="The answer is <answer>4</answer>")
        result = env.step(state, action)

        # Check termination
        assert result.terminated is True
        assert result.truncated is False
        assert result.next_state.metadata.is_terminal is True

        # Check rewards (native-only by default)
        correctness = result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.reward == 1.0

        # No format reward by default
        assert result.rewards.by_name("format") is None

        # Check info
        assert result.info["extracted_answer"] == "4"
        assert result.info["expected_answer"] == "4"

    def test_step_incorrect_answer(self, mock_dataset):
        """Test step with incorrect answer."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="The answer is <answer>5</answer>")
        result = env.step(state, action)

        correctness = result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.reward == 0.0

    def test_step_no_answer_extracted(self, mock_dataset):
        """Test step when no answer can be extracted."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="I don't know the answer")
        result = env.step(state, action)

        correctness = result.rewards.by_name("correctness")
        assert correctness.reward == 0.0

        assert result.info["extracted_answer"] is None

    def test_step_state_unchanged(self, mock_dataset):
        """Test that step doesn't mutate input state."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        original_step = state.metadata.step
        original_terminal = state.metadata.is_terminal

        env.step(state, Action(text="<answer>4</answer>"))

        # Original state unchanged
        assert state.metadata.step == original_step
        assert state.metadata.is_terminal == original_terminal

    def test_step_next_state_updated(self, mock_dataset):
        """Test that next_state has updated metadata."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="<answer>4</answer>"))

        assert result.next_state.metadata.step == 1
        assert result.next_state.metadata.is_terminal is True
        assert result.next_state.metadata.episode_id == state.metadata.episode_id

    def test_compute_rewards_directly(self, mock_dataset):
        """Test compute_rewards can be called directly."""
        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="<answer>4</answer>")
        rewards = env.compute_rewards(state, action, state)

        assert rewards.total == 1.0  # correctness only (native-only default)


class TestCorrectnessRewardFunction:
    """Tests for CorrectnessRewardFunction."""

    def test_correct_answer(self, mock_dataset):
        """Test reward for correct answer."""
        extractor = TagBasedExtractor()
        reward_fn = CorrectnessRewardFunction(mock_dataset, extractor)

        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="<answer>4</answer>")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 1.0
        assert signal.name == "correctness"
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.metadata["extracted"] == "4"

    def test_incorrect_answer(self, mock_dataset):
        """Test reward for incorrect answer."""
        extractor = TagBasedExtractor()
        reward_fn = CorrectnessRewardFunction(mock_dataset, extractor)

        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="<answer>wrong</answer>")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 0.0

    def test_no_extraction(self, mock_dataset):
        """Test reward when extraction fails."""
        extractor = TagBasedExtractor()
        reward_fn = CorrectnessRewardFunction(mock_dataset, extractor)

        env = ReasoningGymEnvironment(dataset=mock_dataset)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="No tags here")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 0.0
        assert signal.metadata["extracted"] is None


class TestFormatReward:
    """Tests for the unified FormatReward from core."""

    def test_format_followed(self):
        """Test reward when format is followed."""
        extractor = TagBasedExtractor()
        reward_fn = FormatReward(extractor)

        from llenvs.core.state import State, StateMetadata

        state = State(
            observation=Observation(prompt="test"),
            hidden=ReasoningGymHidden(
                entry={}, expected_answer="", task_index=0, dataset_name="test"
            ),
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )

        action = Action(text="<answer>anything</answer>")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 1.0
        assert signal.name == "format"
        assert signal.reward_type == RewardType.FORMAT

    def test_format_not_followed(self):
        """Test reward when format is not followed."""
        extractor = TagBasedExtractor()
        reward_fn = FormatReward(extractor)

        from llenvs.core.state import State, StateMetadata

        state = State(
            observation=Observation(prompt="test"),
            hidden=ReasoningGymHidden(
                entry={}, expected_answer="", task_index=0, dataset_name="test"
            ),
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )

        action = Action(text="No proper tags")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 0.0


class TestReasoningGymHidden:
    """Tests for ReasoningGymHidden."""

    def test_creation(self):
        """Test hidden state creation."""
        entry = {"question": "Q", "answer": "A"}
        hidden = ReasoningGymHidden(
            entry=entry,
            expected_answer="A",
            task_index=5,
            dataset_name="test_dataset",
        )

        assert hidden.entry == entry
        assert hidden.expected_answer == "A"
        assert hidden.task_index == 5
        assert hidden.dataset_name == "test_dataset"

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = ReasoningGymHidden(
            entry={},
            expected_answer="A",
            task_index=0,
            dataset_name="test",
        )
        with pytest.raises(AttributeError):
            hidden.expected_answer = "B"  # type: ignore


class TestReasoningGymAdapter:
    """Tests for ReasoningGymAdapter."""

    def test_adapter_name(self):
        """Test adapter name property."""
        adapter = ReasoningGymAdapter()
        assert adapter.name == "reasoning_gym"

    def test_list_environments_returns_list(self, monkeypatch):
        """Test list_environments returns a list."""

        # Mock reasoning_gym to avoid import errors in tests
        class MockReasoningGym:
            @staticmethod
            def list_datasets():
                return ["sudoku", "leg_counting", "arithmetic"]

        adapter = ReasoningGymAdapter()
        monkeypatch.setattr(adapter, "_get_reasoning_gym", lambda: MockReasoningGym())

        envs = adapter.list_environments()
        assert isinstance(envs, list)
        assert "sudoku" in envs

    def test_list_environments_fallback(self, monkeypatch):
        """Test fallback when reasoning_gym doesn't have list_datasets."""

        class MockReasoningGym:
            pass  # No list_datasets method

        adapter = ReasoningGymAdapter()
        monkeypatch.setattr(adapter, "_get_reasoning_gym", lambda: MockReasoningGym())

        envs = adapter.list_environments()
        assert isinstance(envs, list)
        assert len(envs) > 0  # Should have fallback datasets

    def test_get_environment_with_mock(self, mock_dataset, monkeypatch):
        """Test get_environment creates an environment."""
        import reasoning_gym as rg_module

        monkeypatch.setattr(rg_module, "create_dataset", lambda name, **kwargs: mock_dataset)

        adapter = ReasoningGymAdapter()
        env = adapter.get_environment("test_dataset")

        assert isinstance(env, ReasoningGymEnvironment)
        assert env.spec.adapter == "reasoning_gym"

    def test_get_environment_info(self):
        """Test get_environment_info returns metadata."""
        adapter = ReasoningGymAdapter()
        info = adapter.get_environment_info("sudoku")

        assert info["name"] == "sudoku"
        assert info["adapter"] == "reasoning_gym"
        assert info["type"] == "single_turn"


class TestEnvironmentRegistry:
    """Tests for EnvironmentRegistry with adapter support."""

    def test_register_adapter(self, mock_dataset, monkeypatch):
        """Test adapter registration."""
        registry = EnvironmentRegistry()

        # Patch to avoid import errors
        def mock_get_rg():
            class MockRG:
                @staticmethod
                def list_datasets():
                    return ["test"]

            return MockRG()

        adapter = ReasoningGymAdapter()
        monkeypatch.setattr(adapter, "_get_reasoning_gym", mock_get_rg)

        registry.register_adapter(adapter)
        assert "reasoning_gym" in registry.list_adapters()

    def test_register_duplicate_adapter_raises(self, monkeypatch):
        """Test that registering duplicate adapter raises ValueError."""
        registry = EnvironmentRegistry()

        def mock_get_rg():
            class MockRG:
                @staticmethod
                def list_datasets():
                    return ["test"]

            return MockRG()

        adapter1 = ReasoningGymAdapter()
        adapter2 = ReasoningGymAdapter()
        monkeypatch.setattr(adapter1, "_get_reasoning_gym", mock_get_rg)
        monkeypatch.setattr(adapter2, "_get_reasoning_gym", mock_get_rg)

        registry.register_adapter(adapter1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_adapter(adapter2)

    def test_get_adapter(self, monkeypatch):
        """Test getting a registered adapter."""
        registry = EnvironmentRegistry()

        def mock_get_rg():
            class MockRG:
                @staticmethod
                def list_datasets():
                    return ["test"]

            return MockRG()

        adapter = ReasoningGymAdapter()
        monkeypatch.setattr(adapter, "_get_reasoning_gym", mock_get_rg)

        registry.register_adapter(adapter)
        retrieved = registry.get_adapter("reasoning_gym")
        assert retrieved is adapter

    def test_get_adapter_not_found(self):
        """Test getting non-existent adapter raises KeyError."""
        registry = EnvironmentRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_adapter("nonexistent")

    def test_unregister_adapter(self, monkeypatch):
        """Test adapter unregistration."""
        registry = EnvironmentRegistry()

        def mock_get_rg():
            class MockRG:
                @staticmethod
                def list_datasets():
                    return ["test"]

            return MockRG()

        adapter = ReasoningGymAdapter()
        monkeypatch.setattr(adapter, "_get_reasoning_gym", mock_get_rg)

        registry.register_adapter(adapter)
        assert "reasoning_gym" in registry.list_adapters()

        registry.unregister_adapter("reasoning_gym")
        assert "reasoning_gym" not in registry.list_adapters()

    def test_get_environment_via_registry(self, mock_dataset, monkeypatch):
        """Test getting environment via registry.get()."""
        registry = EnvironmentRegistry()

        # Create adapter with mocked get_environment
        class MockAdapter:
            @property
            def name(self):
                return "test_adapter"

            def list_environments(self):
                return ["test_env"]

            def get_environment(self, name, **kwargs):
                return ReasoningGymEnvironment(dataset=mock_dataset)

        registry.register_adapter(MockAdapter())
        env = registry.get(name="test_env", adapter="test_adapter")

        assert isinstance(env, ReasoningGymEnvironment)

    def test_list_environments_all_adapters(self, monkeypatch):
        """Test listing environments from all adapters."""
        registry = EnvironmentRegistry()

        class MockAdapter1:
            @property
            def name(self):
                return "adapter1"

            def list_environments(self):
                return ["env_a", "env_b"]

        class MockAdapter2:
            @property
            def name(self):
                return "adapter2"

            def list_environments(self):
                return ["env_c"]

        registry.register_adapter(MockAdapter1())
        registry.register_adapter(MockAdapter2())

        envs = registry.list_environments()
        assert len(envs) == 3
        assert ("adapter1", "env_a") in envs
        assert ("adapter1", "env_b") in envs
        assert ("adapter2", "env_c") in envs

    def test_list_environments_specific_adapter(self, monkeypatch):
        """Test listing environments from specific adapter."""
        registry = EnvironmentRegistry()

        class MockAdapter1:
            @property
            def name(self):
                return "adapter1"

            def list_environments(self):
                return ["env_a", "env_b"]

        class MockAdapter2:
            @property
            def name(self):
                return "adapter2"

            def list_environments(self):
                return ["env_c"]

        registry.register_adapter(MockAdapter1())
        registry.register_adapter(MockAdapter2())

        envs = registry.list_environments(adapter="adapter1")
        assert len(envs) == 2
        assert all(a == "adapter1" for a, _ in envs)

    def test_contains(self, monkeypatch):
        """Test __contains__ for checking availability."""
        registry = EnvironmentRegistry()

        class MockAdapter:
            @property
            def name(self):
                return "test_adapter"

            def list_environments(self):
                return ["env_a", "env_b"]

        registry.register_adapter(MockAdapter())

        assert ("test_adapter", "env_a") in registry
        assert ("test_adapter", "env_b") in registry
        assert ("test_adapter", "env_c") not in registry
        assert ("other_adapter", "env_a") not in registry
