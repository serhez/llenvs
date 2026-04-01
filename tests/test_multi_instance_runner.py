"""Tests for the multi-instance runner path.

Tests the env_factory + restore_fn functionality in TrajectoryRunner
for non-pure environments where each rollout needs its own instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.evaluation.runner import TrajectoryRunner
from llenvs.inference.protocol import GenerationResult, SamplingParams, StopReason

# ── Mock infrastructure ──────────────────────────────────────────


@dataclass
class _MockHidden:
    task_index: int = 0
    trajectory: tuple[str, ...] = ()
    step: int = 0


def _make_state(
    step: int = 0,
    task_index: int = 0,
    trajectory: tuple[str, ...] = (),
    is_terminal: bool = False,
    max_steps: int = 10,
) -> State[_MockHidden]:
    hidden = _MockHidden(task_index=task_index, trajectory=trajectory, step=step)
    return State(
        observation=Observation(
            prompt=f"Task {task_index}",
            task=None,
            state=None,
        ),
        hidden=hidden,
        metadata=StateMetadata(
            step=step,
            episode_id=f"ep_{task_index}",
            is_terminal=is_terminal,
            info={"task_index": task_index},
        ),
    )


class MockNonPureEnvironment:
    """Non-pure environment that tracks step calls per instance."""

    _instance_count = 0

    def __init__(self, max_steps: int = 5, terminate_after: int = 3):
        MockNonPureEnvironment._instance_count += 1
        self.instance_id = MockNonPureEnvironment._instance_count
        self._max_steps = max_steps
        self._terminate_after = terminate_after
        self.step_calls: list[tuple[Any, Any]] = []
        self.closed = False

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="mock-non-pure",
            adapter="mock",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
        )

    @property
    def reward_functions(self) -> tuple:
        return ()

    def reset(self, *, seed=None, options=None):
        return _make_state(), {}

    def step(self, state: State[Any], action: Action) -> StepResult[Any]:
        self.step_calls.append((state, action))
        next_step = state.metadata.step + 1
        is_terminal = next_step >= self._terminate_after
        reward = 1.0 if is_terminal else 0.0

        next_state = _make_state(
            step=next_step,
            task_index=state.hidden.task_index,
            trajectory=state.hidden.trajectory + (action.text or "",),
            is_terminal=is_terminal,
        )

        signal = Signal(
            name="mock",
            reward_type=RewardType.OUTCOME if is_terminal else RewardType.STEP,
            reward=reward if is_terminal else None,
        )

        return StepResult(
            next_state=next_state,
            rewards=SignalBundle(signals=(signal,)),
            terminated=is_terminal,
            truncated=False,
            info={"instance_id": self.instance_id},
        )

    def close(self):
        self.closed = True


class MockPureEnvironment(MockNonPureEnvironment):
    """Pure-step environment (for bypass test)."""

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="mock-pure",
            adapter="mock",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=True,
        )


def _mock_env_factory(
    max_steps: int = 5,
    terminate_after: int = 3,
) -> tuple[list[MockNonPureEnvironment], callable]:
    """Factory that creates and tracks env instances."""
    created: list[MockNonPureEnvironment] = []

    def factory() -> MockNonPureEnvironment:
        env = MockNonPureEnvironment(max_steps=max_steps, terminate_after=terminate_after)
        created.append(env)
        return env

    return created, factory


def _mock_restore_fn(env: MockNonPureEnvironment, state: State[Any]) -> State[Any]:
    """Restore by replaying trajectory commands."""
    current = _make_state(task_index=state.hidden.task_index)
    for cmd in state.hidden.trajectory:
        result = env.step(current, Action(text=cmd))
        current = result.next_state
    return current


def _make_mock_backend(response_text: str = "action_text") -> MagicMock:
    """Create a mock backend that returns fixed generation results."""
    backend = MagicMock()
    backend.capabilities = MagicMock()
    backend.capabilities.supports_function_calling = False

    gen_result = GenerationResult(
        text=response_text,
        finish_reason=StopReason.END_OF_TEXT,
        tool_calls=(),
        prompt_tokens=10,
        completion_tokens=5,
    )

    # generate_chat_batch returns a list of results matching input length
    def batch_generate(messages_batch, params):
        return [gen_result] * len(messages_batch)

    backend.generate_chat_batch = MagicMock(side_effect=batch_generate)
    backend.generate_chat = MagicMock(return_value=gen_result)

    return backend


# ── Tests ────────────────────────────────────────────────────────


class TestMultiInstanceRunner:
    """Tests for the multi-instance runner path."""

    @pytest.fixture(autouse=True)
    def reset_instance_count(self):
        MockNonPureEnvironment._instance_count = 0

    def test_multi_instance_produces_valid_trajectories(self):
        """Multi-instance rollouts produce valid trajectories."""
        created, factory = _mock_env_factory(terminate_after=2)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        states = [_make_state(task_index=i) for i in range(3)]
        trajectories = runner.run_batch_from_states(states)

        assert len(trajectories) == 3
        for traj in trajectories:
            assert len(traj) > 0

    def test_restore_fn_called_per_instance(self):
        """restore_fn is called once per env instance with correct state."""
        created, factory = _mock_env_factory(terminate_after=2)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        restore_calls: list[tuple[Any, Any]] = []
        original_restore = _mock_restore_fn

        def tracking_restore(env, state):
            restore_calls.append((env, state))
            return original_restore(env, state)

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=tracking_restore,
        )

        states = [
            _make_state(task_index=0, trajectory=("cmd1",)),
            _make_state(task_index=1, trajectory=("cmd2", "cmd3")),
        ]
        runner.run_batch_from_states(states)

        assert len(restore_calls) == 2
        # Each call should get a different env instance
        assert restore_calls[0][0] is not restore_calls[1][0]
        # Each call should get the correct state
        assert restore_calls[0][1].hidden.task_index == 0
        assert restore_calls[1][1].hidden.task_index == 1

    def test_forced_actions_work(self):
        """Forced actions (Q-value path) work in multi-instance mode."""
        created, factory = _mock_env_factory(terminate_after=2)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        states = [_make_state(task_index=0)]
        actions = [Action(text="forced_action")]
        trajectories = runner.run_batch_from_state_actions(states, actions)

        assert len(trajectories) == 1
        first_transition = trajectories[0].transitions[0]
        assert first_transition.action.text == "forced_action"

    def test_all_envs_closed_after_completion(self):
        """All env instances are closed after completion."""
        created, factory = _mock_env_factory(terminate_after=2)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        states = [_make_state(task_index=i) for i in range(3)]
        runner.run_batch_from_states(states)

        assert len(created) == 3
        for env in created:
            assert env.closed

    def test_envs_closed_on_error(self):
        """All env instances are closed even when an error occurs."""
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        created: list[MockNonPureEnvironment] = []

        def error_restore(env, state):
            if state.hidden.task_index == 1:
                raise RuntimeError("Restore failed")
            return _mock_restore_fn(env, state)

        def factory():
            env = MockNonPureEnvironment()
            created.append(env)
            return env

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=error_restore,
        )

        states = [_make_state(task_index=i) for i in range(3)]
        with pytest.raises(RuntimeError, match="Restore failed"):
            runner.run_batch_from_states(states)

        # All created envs should be closed
        for env in created:
            assert env.closed

    def test_batch_size_limits_concurrent_instances(self):
        """batch_size limits concurrent env instances."""
        created, factory = _mock_env_factory(terminate_after=1)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        states = [_make_state(task_index=i) for i in range(6)]
        trajectories = runner.run_batch_from_states(states, batch_size=2)

        assert len(trajectories) == 6
        # All envs closed
        for env in created:
            assert env.closed

    def test_pure_step_bypasses_multi_instance(self):
        """Pure-step environments bypass multi-instance path."""
        primary_env = MockPureEnvironment(terminate_after=2)
        backend = _make_mock_backend()

        factory_called = []

        def factory():
            factory_called.append(True)
            return MockPureEnvironment()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        states = [_make_state(task_index=0)]
        trajectories = runner.run_batch_from_states(states)

        # Factory should NOT be called for pure-step envs
        assert len(factory_called) == 0
        assert len(trajectories) == 1

    def test_batched_llm_generation_across_instances(self):
        """Batched LLM generation works across multiple env instances."""
        created, factory = _mock_env_factory(terminate_after=2)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        states = [_make_state(task_index=i) for i in range(4)]
        runner.run_batch_from_states(states)

        # Backend should have been called with batched messages
        assert backend.generate_chat_batch.call_count > 0
        # Each call should batch across all active rollouts
        first_call_args = backend.generate_chat_batch.call_args_list[0]
        messages_batch = first_call_args[0][0]
        assert len(messages_batch) == 4  # All 4 rollouts in first batch

    def test_terminal_states_handled(self):
        """Already-terminal restored states produce empty trajectories."""
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        created: list[MockNonPureEnvironment] = []

        def factory():
            env = MockNonPureEnvironment()
            created.append(env)
            return env

        def terminal_restore(env, state):
            """Restore that returns a terminal state."""
            return _make_state(
                task_index=state.hidden.task_index,
                step=5,
                is_terminal=True,
            )

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=terminal_restore,
        )

        states = [_make_state(task_index=0)]
        trajectories = runner.run_batch_from_states(states)

        assert len(trajectories) == 1
        assert len(trajectories[0]) == 0  # No transitions

    def test_progress_callback(self):
        """Progress callback is called during multi-instance execution."""
        created, factory = _mock_env_factory(terminate_after=1)
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        progress_calls: list[tuple[int, int]] = []

        def callback(done: int, total: int):
            progress_calls.append((done, total))

        states = [_make_state(task_index=i) for i in range(2)]
        runner.run_batch_from_states(states, progress_callback=callback)

        assert len(progress_calls) > 0
        # Last call should show all done
        assert progress_calls[-1] == (2, 2)

    def test_run_batch_debug_logs_round_generation_and_step_waits(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        created, factory = _mock_env_factory(terminate_after=2)
        primary_env = MockNonPureEnvironment(terminate_after=2)
        backend = _make_mock_backend()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        with caplog.at_level(logging.DEBUG, logger="llenvs.evaluation.runner"):
            result = runner.run_batch([0, 1], batch_size=2)

        assert len(result.trajectory_results) == 2
        messages = [record.getMessage() for record in caplog.records]
        assert any("Trajectory batch start: mode=multi-instance" in msg for msg in messages)
        assert any("Trajectory round 1 start:" in msg for msg in messages)
        assert any("Trajectory round 1 generation finished" in msg for msg in messages)
        assert any("Trajectory round 1 waiting for step result" in msg for msg in messages)
        assert any("Trajectory round 1 complete:" in msg for msg in messages)

    def test_empty_states_returns_empty(self):
        """Empty states list returns empty results."""
        primary_env = MockNonPureEnvironment()
        backend = _make_mock_backend()
        _, factory = _mock_env_factory()

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            sampling_params=SamplingParams(),
            env_factory=factory,
            restore_fn=_mock_restore_fn,
        )

        trajectories = runner.run_batch_from_states([])
        assert trajectories == []
