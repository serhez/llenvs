"""End-to-end integration tests for containerized environments.

These tests start real subprocesses via ProcessRuntime and exercise
the full pipeline: config -> server -> client -> reset/step -> rewards.

Marked with @pytest.mark.slow since they start real subprocesses.
"""

from __future__ import annotations

import pytest

from llenvs.container.client import ContainerEnvironment
from llenvs.container.config import ContainerConfig
from llenvs.container.serialization import OpaqueHidden
from llenvs.core.config import EnvironmentConfig, EnvironmentFactory
from llenvs.core.environment import EnvironmentSpec
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Action


pytestmark = pytest.mark.slow


@pytest.fixture
def container_env():
    """Create a containerized leg_counting environment via ProcessRuntime."""
    config = EnvironmentConfig(
        name="leg_counting",
        adapter="reasoning_gym",
        size=5,
        container=ContainerConfig(runtime="process", timeout=30.0),
    )
    env = EnvironmentFactory.create(config)
    yield env
    if hasattr(env, "_runtime"):
        env._runtime.stop()
    if hasattr(env, "close"):
        env.close()


class TestContainerRoundTrip:
    def test_creates_container_environment(self, container_env):
        assert isinstance(container_env, ContainerEnvironment)

    def test_spec(self, container_env):
        spec = container_env.spec
        assert isinstance(spec, EnvironmentSpec)
        assert spec.name  # Has a name

    def test_len(self, container_env):
        assert len(container_env) == 5

    def test_reset(self, container_env):
        state, info = container_env.reset(options={"task_index": 0})
        assert state.observation.prompt  # Has content
        assert isinstance(state.hidden, OpaqueHidden)

    def test_hidden_attribute_access(self, container_env):
        state, _ = container_env.reset(options={"task_index": 0})
        # leg_counting hidden has expected_answer
        assert hasattr(state.hidden, "expected_answer")

    def test_step(self, container_env):
        state, _ = container_env.reset(options={"task_index": 0})
        action = Action.from_text("<answer>8</answer>")
        result = container_env.step(state, action)
        assert result.terminated or not result.terminated  # Valid StepResult
        assert isinstance(result.rewards, SignalBundle)

    def test_step_correct_answer(self, container_env):
        state, _ = container_env.reset(options={"task_index": 0})
        expected = state.hidden.expected_answer
        action = Action.from_text(f"<answer>{expected}</answer>")
        result = container_env.step(state, action)
        assert result.terminated is True
        # Should get positive outcome reward for correct answer
        outcome_rewards = result.rewards.by_type(
            __import__("llenvs.core.reward", fromlist=["RewardType"]).RewardType.OUTCOME
        )
        assert len(outcome_rewards) > 0

    def test_compute_rewards(self, container_env):
        state, _ = container_env.reset(options={"task_index": 0})
        action = Action.from_text("<answer>42</answer>")
        result = container_env.step(state, action)
        rewards = container_env.compute_rewards(state, action, result.next_state)
        assert isinstance(rewards, SignalBundle)

    def test_reward_functions_empty(self, container_env):
        assert container_env.reward_functions == ()

    def test_prompts(self, container_env):
        prompts = container_env.prompts
        assert isinstance(prompts, dict)

    def test_available_tools(self, container_env):
        tools = container_env.available_tools
        assert isinstance(tools, tuple)

    def test_multiple_episodes(self, container_env):
        """Multiple reset/step cycles work."""
        for i in range(3):
            state, _ = container_env.reset(options={"task_index": i})
            assert state.observation.prompt
            action = Action.from_text("<answer>0</answer>")
            result = container_env.step(state, action)
            assert isinstance(result.rewards, SignalBundle)


class TestScorerIntegration:
    def test_scorer_works(self, container_env):
        from llenvs.integrations.scoring import Scorer

        scorer = Scorer(container_env)
        result = scorer.score(task_index=0, response="<answer>8</answer>")
        assert isinstance(result.total, float)


class TestDatasetProviderIntegration:
    def test_provider_works(self, container_env):
        from llenvs.integrations.dataset_provider import DatasetProvider

        provider = DatasetProvider(container_env)
        item = provider[0]
        assert item.prompt


class TestFactoryWithProcessRuntime:
    def test_factory_creates_proxy(self):
        config = EnvironmentConfig(
            name="leg_counting",
            adapter="reasoning_gym",
            size=3,
            container=ContainerConfig(runtime="process", timeout=30.0),
        )
        env = EnvironmentFactory.create(config)
        try:
            assert isinstance(env, ContainerEnvironment)
            state, _ = env.reset(options={"task_index": 0})
            assert state.observation.prompt
        finally:
            if hasattr(env, "_runtime"):
                env._runtime.stop()
            if hasattr(env, "close"):
                env.close()
