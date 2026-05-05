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
from llenvs.core.tool_parsing import HermesToolCallParser
from llenvs.core.tools import ToolDefinition
from llenvs.evaluation.runner import TrajectoryRunner
from llenvs.inference import protocol as inference_protocol
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    PartialBatchError,
    PromptTooLongError,
    RecoverableInputError,
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


def _make_tool_state(
    text: str = "obs",
    step: int = 0,
    tools: tuple[ToolDefinition, ...] = (),
) -> State:
    return State(
        observation=Observation(
            prompt="",
            messages=(),
            task=ObservationContent(text="task"),
            state=ObservationContent(text=text),
            available_tools=tools,
        ),
        hidden=None,
        metadata=StateMetadata(step=step, episode_id="test"),
    )


def _gen_result(
    text: str = "action",
    *,
    finish_reason: StopReason = StopReason.STOP_SEQUENCE,
) -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason=finish_reason,
    )


def _make_step_result(
    next_text: str = "next", step: int = 1, done: bool = True,
) -> StepResult:
    return StepResult(
        next_state=_make_state(next_text, step=step),
        rewards=SignalBundle.single(reward=1.0 if done else 0.0, name="correctness"),
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


def _make_batch_runner(
    mock_backend,
    *,
    max_steps: int = 1,
    tool_defs: tuple[ToolDefinition, ...] = (),
    sampling_params: SamplingParams | None = None,
    tool_call_parser: HermesToolCallParser | None = None,
):
    mock_env = MagicMock()
    mock_env.spec.max_steps = max_steps
    mock_env.spec.pure_step = True
    mock_env.reset.side_effect = lambda *, options=None: (
        _make_tool_state(
            text=f"s{(options or {}).get('task_index', 0)}",
            tools=tool_defs,
        ),
        {"task_index": (options or {}).get("task_index", 0)},
    )
    mock_env.step.return_value = _make_step_result(done=True)

    return TrajectoryRunner(
        environment=mock_env,
        backend=mock_backend,
        sampling_params=sampling_params or SamplingParams(),
        tool_call_parser=tool_call_parser,
    ), mock_env


class TestGenerationErrorCallback:
    """Tests for on_generation_error callback in run_batch_from_states."""

    def test_no_callback_recovers_skippable_failures(self) -> None:
        """Without callback, recoverable per-input failures are dropped."""
        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = PromptTooLongError(
            "too long", model_name="test", max_model_len=100,
            offending_indices=[0],
        )
        runner, _ = _make_runner(mock_backend)

        results = runner.run_batch_from_states([_make_state()])

        assert results == []

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
        """Without callback, successful trajectories are still returned."""
        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.return_value = [_gen_result()]
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states([_make_state()])

        assert len(results) == 1
        assert isinstance(results[0], Trajectory)

    def test_transient_partial_failure_without_callback_retries_then_skips(self) -> None:
        """Retry-exhausted transient subset failures are dropped, not raised."""
        call_sizes: list[int] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            transient = RuntimeError("rate limited")
            if len(messages_batch) == 2:
                raise PartialBatchError(
                    [_gen_result("ok0"), transient],
                    {1: transient},
                )
            if len(call_sizes) <= 11:
                raise PartialBatchError([transient], {0: transient})
            raise AssertionError("unexpected extra retry")

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        with patch("llenvs.evaluation.runner.time.sleep", lambda *_: None):
            results = runner.run_batch_from_states(
                [_make_state("s0"), _make_state("s1")],
            )

        assert len(results) == 1
        assert results[0] is not None
        assert call_sizes == [2] + [1] * 10

    def test_retry_exhausted_partial_failure_wraps_error_for_callback(self) -> None:
        exc_cls = getattr(inference_protocol, "RetryExhaustedTransientError", None)
        assert exc_cls is not None

        call_sizes: list[int] = []
        seen_errors: list[Exception] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            transient = RuntimeError("rate limited")
            if len(messages_batch) == 2:
                raise PartialBatchError(
                    [_gen_result("ok0"), transient],
                    {1: transient},
                )
            raise PartialBatchError([transient], {0: transient})

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        with patch("llenvs.evaluation.runner.time.sleep", lambda *_: None):
            results = runner.run_batch_from_states(
                [_make_state("s0"), _make_state("s1")],
                on_generation_error=lambda exc: (
                    seen_errors.append(exc),
                    "skip" if isinstance(exc, exc_cls) else "raise",
                )[1],
            )

        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None
        assert call_sizes == [2] + [1] * 10
        assert len(seen_errors) == 1
        assert isinstance(seen_errors[0], exc_cls)
        assert getattr(seen_errors[0], "retry_count", None) == 10
        assert isinstance(getattr(seen_errors[0], "original_error", None), RuntimeError)

    def test_retry_exhausted_whole_batch_failure_wraps_error_for_callback(self) -> None:
        exc_cls = getattr(inference_protocol, "RetryExhaustedTransientError", None)
        assert exc_cls is not None

        seen_errors: list[Exception] = []
        call_count = 0

        def side_effect(messages_batch, params):
            del messages_batch, params
            nonlocal call_count
            call_count += 1
            raise RuntimeError("rate limited")

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, _ = _make_runner(mock_backend, max_steps=1)

        delays: list[float] = []
        with patch(
            "llenvs.evaluation.runner.time.sleep",
            lambda delay: delays.append(delay),
        ):
            results = runner.run_batch_from_states(
                [_make_state("s0")],
                on_generation_error=lambda exc: (
                    seen_errors.append(exc),
                    "skip" if isinstance(exc, exc_cls) else "raise",
                )[1],
            )

        assert results == [None]
        assert call_count == 11
        assert delays == [2.0, 4.0, 8.0] + [8.0] * 7
        assert len(seen_errors) == 1
        assert isinstance(seen_errors[0], exc_cls)
        assert getattr(seen_errors[0], "retry_count", None) == 10

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

    def test_partial_prompt_too_long_skips_only_failed_subset(self) -> None:
        """PartialBatchError should preserve successes and skip only recoverable failures."""
        too_long = PromptTooLongError(
            "too long",
            model_name="test",
            max_model_len=100,
            offending_indices=[1],
        )
        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = PartialBatchError(
            [_gen_result("ok0"), too_long, _gen_result("ok2")],
            {1: too_long},
        )
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1"), _make_state("s2")],
            on_generation_error=lambda exc: (
                "skip" if isinstance(exc, PromptTooLongError) else "raise"
            ),
        )

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
        assert env.step.call_count == 2
        assert mock_backend.generate_chat_batch.call_count == 1

    def test_partial_transient_failure_retries_only_failed_subset(self) -> None:
        """Transient failures inside PartialBatchError should retry only the failed slots."""
        call_sizes: list[int] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            if len(call_sizes) == 1:
                transient = RuntimeError("rate limited")
                raise PartialBatchError(
                    [_gen_result("ok0"), transient, _gen_result("ok2")],
                    {1: transient},
                )
            return [_gen_result("retry1")]

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_runner(mock_backend, max_steps=1)
        env.step.return_value = _make_step_result(done=True)

        results = runner.run_batch_from_states(
            [_make_state("s0"), _make_state("s1"), _make_state("s2")],
            on_generation_error=lambda exc: (
                "skip" if isinstance(exc, PromptTooLongError) else "raise"
            ),
        )

        assert len(results) == 3
        assert all(result is not None for result in results)
        assert env.step.call_count == 3
        assert call_sizes == [3, 1]

    def test_partial_unrecoverable_failure_raises_without_extra_retry(self) -> None:
        call_sizes: list[int] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            fatal = RuntimeError("GPU OOM")
            raise PartialBatchError(
                [_gen_result("ok0"), fatal],
                {1: fatal},
            )

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = side_effect
        runner, _ = _make_runner(mock_backend, max_steps=1)

        with pytest.raises(RuntimeError, match="GPU OOM"):
            runner.run_batch_from_states(
                [_make_state("s0"), _make_state("s1")],
                on_generation_error=lambda exc: "raise",
            )

        assert call_sizes == [2]

    def test_partial_batch_short_results_raise_contract_error(self) -> None:
        mock_backend = MagicMock()
        transient = RuntimeError("rate limited")
        mock_backend.generate_chat_batch.side_effect = PartialBatchError(
            [],
            {1: transient},
        )
        runner, _ = _make_runner(mock_backend, max_steps=1)

        with pytest.raises(RuntimeError, match="PartialBatchError"):
            runner.run_batch_from_states(
                [_make_state("s0"), _make_state("s1")],
                on_generation_error=lambda exc: "raise",
            )


class TestRunBatchPartialRecovery:
    def test_partial_prompt_too_long_skips_only_failed_trajectory(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = False
        too_long = PromptTooLongError(
            "too long",
            model_name="test",
            max_model_len=100,
            offending_indices=[1],
        )
        backend.generate_chat_batch.side_effect = PartialBatchError(
            [_gen_result("ok0"), too_long, _gen_result("ok2")],
            {1: too_long},
        )
        runner, env = _make_batch_runner(backend)

        result = runner.run_batch([0, 1, 2])

        assert [tr.success for tr in result.trajectory_results] == [True, False, True]
        assert env.step.call_count == 2
        assert "Generation error" in result.trajectory_results[1].metadata["error"]
        assert backend.generate_chat_batch.call_count == 1

    def test_partial_transient_failure_retries_only_failed_subset(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = False
        call_sizes: list[int] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            if len(call_sizes) == 1:
                transient = RuntimeError("rate limited")
                raise PartialBatchError(
                    [_gen_result("ok0"), transient, _gen_result("ok2")],
                    {1: transient},
                )
            return [_gen_result("retry1")]

        backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_batch_runner(backend)

        result = runner.run_batch([0, 1, 2])

        assert [tr.success for tr in result.trajectory_results] == [True, True, True]
        assert env.step.call_count == 3
        assert call_sizes == [3, 1]

    def test_retry_exhausted_transient_failure_marks_failed_trajectory(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = False
        call_sizes: list[int] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            transient = RuntimeError("rate limited")
            if len(messages_batch) == 2:
                raise PartialBatchError(
                    [_gen_result("ok0"), transient],
                    {1: transient},
                )
            raise PartialBatchError([transient], {0: transient})

        backend.generate_chat_batch.side_effect = side_effect
        runner, _ = _make_batch_runner(backend)

        with patch("llenvs.evaluation.runner.time.sleep", lambda *_: None):
            result = runner.run_batch([0, 1])

        assert [tr.success for tr in result.trajectory_results] == [True, False]
        assert "Generation error" in result.trajectory_results[1].metadata["error"]
        assert call_sizes == [2] + [1] * 10

    def test_partial_native_tool_batch_uses_generate_with_tools_batch(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = True
        tools = (ToolDefinition(name="search", description="Search"),)
        too_long = PromptTooLongError(
            "too long",
            model_name="test",
            max_model_len=100,
            offending_indices=[1],
        )
        backend.generate_with_tools_batch.side_effect = PartialBatchError(
            [_gen_result("ok0"), too_long],
            {1: too_long},
        )
        runner, env = _make_batch_runner(backend, tool_defs=tools)

        result = runner.run_batch([0, 1])

        assert [tr.success for tr in result.trajectory_results] == [True, False]
        assert env.step.call_count == 1
        backend.generate_with_tools_batch.assert_called_once()
        backend.generate_chat_batch.assert_not_called()

    def test_partial_text_tool_batch_retries_subset_and_reparses_tools(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = False
        tools = (ToolDefinition(name="search", description="Search"),)
        parser = HermesToolCallParser()
        raw_tool_call = (
            '<tool_call>{"name": "search", "arguments": {}}</tool_call>'
        )
        call_sizes: list[int] = []

        def side_effect(messages_batch, params):
            del params
            call_sizes.append(len(messages_batch))
            if len(call_sizes) == 1:
                transient = RuntimeError("temporary failure")
                raise PartialBatchError(
                    [
                        GenerationResult(text=raw_tool_call, finish_reason=StopReason.END_OF_TEXT),
                        transient,
                    ],
                    {1: transient},
                )
            return [GenerationResult(text=raw_tool_call, finish_reason=StopReason.END_OF_TEXT)]

        backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_batch_runner(
            backend,
            tool_defs=tools,
            tool_call_parser=parser,
        )

        result = runner.run_batch([0, 1])

        assert [tr.success for tr in result.trajectory_results] == [True, True]
        assert call_sizes == [2, 1]
        assert env.step.call_count == 2
        for step_call in env.step.call_args_list:
            action = step_call.args[1]
            assert action.has_tool_calls
        for call in backend.generate_chat_batch.call_args_list:
            system_message = call.args[0][0][0]
            assert system_message.role == "system"
            assert "<tools>" in system_message.content

    def test_partial_second_elicitation_failure_skips_only_failed_subset(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = False
        sampling_params = SamplingParams(
            second_elicitation_suffix="\nAnswer:",
            second_elicitation_max_tokens=16,
        )
        too_long = PromptTooLongError(
            "too long",
            model_name="test",
            max_model_len=100,
            offending_indices=[1],
        )

        def side_effect(messages_batch, params):
            if params.second_elicitation_suffix is not None:
                return [
                    _gen_result("partial0", finish_reason=StopReason.MAX_TOKENS),
                    _gen_result("partial1", finish_reason=StopReason.MAX_TOKENS),
                ]
            raise PartialBatchError(
                [
                    _gen_result("done0", finish_reason=StopReason.END_OF_TEXT),
                    too_long,
                ],
                {1: too_long},
            )

        backend.generate_chat_batch.side_effect = side_effect
        runner, env = _make_batch_runner(
            backend,
            sampling_params=sampling_params,
        )

        result = runner.run_batch([0, 1])

        assert [tr.success for tr in result.trajectory_results] == [True, True]
        assert env.step.call_count == 2
        assert [call.args[1].text for call in env.step.call_args_list] == [
            "partial0\nAnswer:done0",
            "partial1",
        ]


class TestRunBatchFromStatesToolParity:
    def test_native_tools_use_generate_with_tools_batch(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = True
        tools = (ToolDefinition(name="search", description="Search"),)
        backend.generate_with_tools_batch.return_value = [
            _gen_result("ok0"),
            _gen_result("ok1"),
        ]
        runner, env = _make_batch_runner(backend, tool_defs=tools)

        results = runner.run_batch_from_states(
            [
                _make_tool_state("s0", tools=tools),
                _make_tool_state("s1", tools=tools),
            ]
        )

        assert len(results) == 2
        assert all(result is not None for result in results)
        assert env.step.call_count == 2
        backend.generate_with_tools_batch.assert_called_once()
        backend.generate_chat_batch.assert_not_called()
        assert tuple(backend.generate_with_tools_batch.call_args.args[1]) == tools

    def test_text_tool_parser_uses_tool_prompt_and_parsed_action(self) -> None:
        backend = MagicMock()
        backend.capabilities.supports_function_calling = False
        tools = (ToolDefinition(name="search", description="Search"),)
        parser = HermesToolCallParser()
        raw_tool_call = '<tool_call>{"name": "search", "arguments": {}}</tool_call>'
        backend.generate_chat_batch.return_value = [
            GenerationResult(text=raw_tool_call, finish_reason=StopReason.END_OF_TEXT),
        ]
        runner, env = _make_batch_runner(
            backend,
            tool_defs=tools,
            tool_call_parser=parser,
        )

        results = runner.run_batch_from_states(
            [_make_tool_state("s0", tools=tools)]
        )

        assert len(results) == 1
        assert results[0] is not None
        backend.generate_chat_batch.assert_called_once()
        backend.generate_with_tools_batch.assert_not_called()
        system_message = backend.generate_chat_batch.call_args.args[0][0][0]
        assert system_message.role == "system"
        assert "<tools>" in system_message.content
        action = env.step.call_args.args[1]
        assert action.has_tool_calls
