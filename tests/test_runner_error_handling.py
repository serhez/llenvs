"""Tests for TrajectoryRunner generation-error callback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llenvs.core.reward import SignalBundle
from llenvs.core.environment import StepResult
from llenvs.core.state import (
    Action,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.trajectory import Trajectory
from llenvs.evaluation.runner import TrajectoryRunner
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    PromptTooLongError,
    SamplingParams,
    StopReason,
)


def _make_state(text: str = "obs", step: int = 0) -> State:
    return State(
        observation=Observation(
            prompt="",
            messages=(),
            task=ObservationContent(text="task"),
            state=ObservationContent(text=text),
        ),
        hidden=None,
        metadata=StateMetadata(step=step, episode_id="test"),
    )


def _gen_result(text: str = "action") -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason=StopReason.STOP_SEQUENCE,
    )


def _make_step_result(
    next_text: str = "next", step: int = 1, done: bool = True,
) -> StepResult:
    return StepResult(
        next_state=_make_state(next_text, step=step),
        rewards=SignalBundle(()),
        terminated=done,
        truncated=False,
    )


def _make_runner(mock_backend, max_steps=2):
    mock_env = MagicMock()
    mock_env.spec.max_steps = max_steps
    mock_env.spec.pure_step = True
    mock_env.step.return_value = _make_step_result(done=True)

    return TrajectoryRunner(
        environment=mock_env,
        backend=mock_backend,
        sampling_params=SamplingParams(),
    ), mock_env


class TestGenerationErrorCallback:
    """Tests for on_generation_error callback in run_batch_from_states."""

    def test_no_callback_raises_as_before(self) -> None:
        """Without callback, errors propagate (backward compat)."""
        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = PromptTooLongError(
            "too long", model_name="test", max_model_len=100,
            offending_indices=[0],
        )
        runner, _ = _make_runner(mock_backend)

        with pytest.raises(PromptTooLongError):
            runner.run_batch_from_states([_make_state()])

    def test_callback_raise_propagates(self) -> None:
        """Callback returning 'raise' propagates the error."""
        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = PromptTooLongError(
            "too long", model_name="test", max_model_len=100,
            offending_indices=[0],
        )
        runner, _ = _make_runner(mock_backend)

        with pytest.raises(PromptTooLongError):
            runner.run_batch_from_states(
                [_make_state()],
                on_generation_error=lambda exc: "raise",
            )

    def test_skip_with_offending_indices_marks_only_offenders(self) -> None:
        """'skip' with offending_indices fails only those trajectories."""
        call_count = [0]

        def side_effect(messages_batch, params):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: trajectory 1 is too long
                raise PromptTooLongError(
                    "too long", model_name="test", max_model_len=100,
                    offending_indices=[1],
                )
            # Second call: remaining trajectory 0 succeeds
            return [_gen_result() for _ in messages_batch]

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1")],
            on_generation_error=lambda exc: "skip",
        )

        assert len(results) == 2
        assert results[0] is not None  # trajectory 0 succeeded
        assert results[1] is None  # trajectory 1 failed

    def test_abort_marks_all_active_but_preserves_completed(self) -> None:
        """'abort' fails active trajectories but keeps completed ones."""
        call_count = [0]

        def side_effect(messages_batch, params):
            call_count[0] += 1
            if call_count[0] == 1:
                # Step 0: all succeed
                return [_gen_result() for _ in messages_batch]
            # Step 1: error (only trajectory 0 is still active)
            raise RuntimeError("something broke")

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=3)

        # Step 0: both active, both succeed. Trajectory 1 terminates.
        step_results = [
            _make_step_result(done=False, step=1),  # traj 0 continues
            _make_step_result(done=True, step=1),    # traj 1 terminates
        ]
        call_idx = [0]

        def env_step(state, action):
            idx = call_idx[0]
            call_idx[0] += 1
            return step_results[idx % len(step_results)]

        env.step.side_effect = env_step

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1")],
            on_generation_error=lambda exc: "abort",
        )

        assert len(results) == 2
        assert results[0] is None      # active at failure → failed
        assert results[1] is not None   # completed before failure → preserved

    def test_skip_without_offending_indices_aborts_all_active(self) -> None:
        """'skip' without offending_indices falls back to aborting all active."""
        mock_backend = MagicMock()
        # Error without offending_indices attribute
        mock_backend.generate_chat_batch.side_effect = ValueError("bad batch")
        runner, _ = _make_runner(mock_backend, max_steps=1)

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1")],
            on_generation_error=lambda exc: "skip",
        )

        assert len(results) == 2
        assert results[0] is None
        assert results[1] is None

    def test_without_callback_returns_trajectory_list(self) -> None:
        """Without callback, return type is list[Trajectory] (no None)."""
        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.return_value = [_gen_result()]
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states([_make_state()])

        assert len(results) == 1
        assert isinstance(results[0], Trajectory)

    def test_chunked_batch_with_callback(self) -> None:
        """Error in one chunk doesn't affect other chunks."""
        call_count = [0]

        def side_effect(messages_batch, params):
            call_count[0] += 1
            if call_count[0] == 2:
                # Second chunk fails
                raise PromptTooLongError(
                    "too long", model_name="test", max_model_len=100,
                    offending_indices=[0],
                )
            return [_gen_result() for _ in messages_batch]

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1"), _make_state("s2")],
            batch_size=1,
            on_generation_error=lambda exc: "skip",
        )

        assert len(results) == 3
        assert results[0] is not None   # chunk 0 succeeded
        assert results[1] is None       # chunk 1 failed
        assert results[2] is not None   # chunk 2 succeeded

    def test_retry_failure_aborts_all_remaining(self) -> None:
        """When retry after skip also fails, all remaining are aborted."""
        call_count = [0]

        def side_effect(messages_batch, params):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: trajectory 1 is offending
                raise PromptTooLongError(
                    "too long", model_name="test", max_model_len=100,
                    offending_indices=[1],
                )
            # Retry: also fails (different offender or cascading)
            raise PromptTooLongError(
                "still too long", model_name="test", max_model_len=100,
                offending_indices=[0],
            )

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1")],
            on_generation_error=lambda exc: "skip",
        )

        assert len(results) == 2
        # Both should be None: s1 from first error, s0 from retry failure
        assert results[0] is None
        assert results[1] is None

    def test_retry_non_recoverable_error_propagates(self) -> None:
        """When retry hits a non-recoverable error, it re-raises."""
        call_count = [0]

        def side_effect(messages_batch, params):
            call_count[0] += 1
            if call_count[0] == 1:
                # First: recoverable error
                raise PromptTooLongError(
                    "too long", model_name="test", max_model_len=100,
                    offending_indices=[1],
                )
            # Retry: non-recoverable OOM
            raise RuntimeError("CUDA out of memory")

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        def error_callback(exc):
            if isinstance(exc, PromptTooLongError):
                return "skip"
            return "raise"

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            runner.run_batch_from_states(
                [_make_state("s0"), _make_state("s1")],
                on_generation_error=error_callback,
            )
