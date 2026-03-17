"""Tests for evaluation logging system."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Observation, State, StateMetadata
from llenvs.evaluation.logging import (
    LogConfig,
    _BatchEndEvent,
    _BatchStartEvent,
    _ConsoleTarget,
    _ErrorEvent,
    _FileTarget,
    _StepEvent,
    _TrajectoryEndEvent,
    _WandbTarget,
)
from llenvs.evaluation.runner import (
    MultiEvalEntry,
    TrajectoryRunner,
    run_evaluation,
    run_multi_evaluation,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    GenerationResult,
    ModelBackend,
    StopReason,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason=StopReason.END_OF_TEXT,
        prompt_tokens=10,
        completion_tokens=5,
    )


class MockSingleTurnEnv:
    """Single-step environment that completes immediately."""

    def __init__(self, num_tasks: int = 5):
        self._num_tasks = num_tasks

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_single", max_steps=10)

    @property
    def reward_functions(self):
        return ()

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return self._num_tasks

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(prompt=f"Question {idx}?"),
            hidden={"answer": str(idx)},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        return StepResult(
            next_state=state.with_metadata(step=1, is_terminal=True),
            rewards=SignalBundle.single(reward=1.0, name="correctness"),
            terminated=True,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.single(reward=1.0, name="correctness")


class MockMultiTurnEnv:
    """Multi-turn environment with variable steps per task."""

    def __init__(self, steps_per_task: dict[int, int]):
        self._steps_per_task = steps_per_task

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_multi", max_steps=100)

    @property
    def reward_functions(self):
        return ()

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return max(self._steps_per_task.keys()) + 1

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(prompt=f"Q{idx}"),
            hidden={"task_index": idx, "target_steps": self._steps_per_task.get(idx, 1)},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        next_step = state.metadata.step + 1
        target = state.hidden["target_steps"]
        done = next_step >= target
        return StepResult(
            next_state=state.with_metadata(step=next_step, is_terminal=done),
            rewards=SignalBundle.single(reward=1.0 if done else 0.0, name="correctness"),
            terminated=done,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.empty()


class MockFailingResetEnv:
    """Environment where specific task indices fail on reset."""

    def __init__(self, failing_indices: set[int], num_tasks: int = 5):
        self._failing = failing_indices
        self._num_tasks = num_tasks

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_failing", max_steps=10)

    @property
    def reward_functions(self):
        return ()

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return self._num_tasks

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        if idx in self._failing:
            raise RuntimeError(f"Reset failed for task {idx}")
        return State(
            observation=Observation(prompt=f"Q{idx}"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        return StepResult(
            next_state=state.with_metadata(step=1, is_terminal=True),
            rewards=SignalBundle.single(reward=1.0, name="correctness"),
            terminated=True,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.single(reward=1.0, name="correctness")


class MockStepFailEnv:
    """Environment that fails during step for specific task indices."""

    def __init__(self, failing_indices: set[int], num_tasks: int = 5):
        self._failing = failing_indices
        self._num_tasks = num_tasks

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_step_fail", max_steps=10)

    @property
    def reward_functions(self):
        return ()

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return self._num_tasks

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(prompt=f"Q{idx}"),
            hidden={"task_index": idx},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        idx = state.hidden["task_index"]
        if idx in self._failing:
            raise RuntimeError(f"Step failed for task {idx}")
        return StepResult(
            next_state=state.with_metadata(step=1, is_terminal=True),
            rewards=SignalBundle.single(reward=1.0, name="correctness"),
            terminated=True,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.single(reward=1.0, name="correctness")


class BatchTrackingBackend(ModelBackend):
    """Backend that tracks batch calls and returns canned responses."""

    def __init__(self) -> None:
        self._call_index = 0
        self.batch_call_sizes: list[int] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True, supports_batching=True)

    @property
    def model_name(self) -> str:
        return "batch-tracking"

    def generate(self, prompts, params):
        return [self._next_result() for _ in prompts]

    def generate_chat(self, messages, params):
        return self._next_result()

    def generate_chat_batch(self, messages_batch, params):
        self.batch_call_sizes.append(len(messages_batch))
        return [self._next_result() for _ in messages_batch]

    def _next_result(self) -> GenerationResult:
        text = f"response_{self._call_index}"
        self._call_index += 1
        return _make_result(text)


# ===========================================================================
# LogConfig validation
# ===========================================================================


class TestLogConfigValidation:
    def test_valid_targets_accepted(self):
        cfg = LogConfig(targets=("console", "file", "wandb"))
        assert cfg.targets == ("console", "file", "wandb")

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown log target"):
            LogConfig(targets=("console", "unknown"))

    def test_empty_targets_valid(self):
        cfg = LogConfig(targets=())
        assert cfg.targets == ()


# ===========================================================================
# _ConsoleTarget
# ===========================================================================


class TestConsoleTarget:
    def test_batch_start_logs_info(self, caplog):
        target = _ConsoleTarget()
        event = _BatchStartEvent(num_tasks=10, environment_name="test_env", max_steps=50)
        with caplog.at_level(logging.INFO, logger="llenvs.evaluation"):
            target.on_batch_start(event)
        assert "test_env" in caplog.text
        assert "10 tasks" in caplog.text

    def test_trajectory_end_logs_info(self, caplog):
        target = _ConsoleTarget()
        event = _TrajectoryEndEvent(
            task_index=3,
            success=True,
            total_reward=0.85,
            num_steps=5,
            completed_count=7,
            total_count=10,
        )
        with caplog.at_level(logging.INFO, logger="llenvs.evaluation"):
            target.on_trajectory_end(event)
        assert "7/10" in caplog.text
        assert "Task 3" in caplog.text
        assert "OK" in caplog.text

    def test_trajectory_end_fail(self, caplog):
        target = _ConsoleTarget()
        event = _TrajectoryEndEvent(
            task_index=1,
            success=False,
            total_reward=0.0,
            num_steps=2,
            completed_count=1,
            total_count=5,
        )
        with caplog.at_level(logging.INFO, logger="llenvs.evaluation"):
            target.on_trajectory_end(event)
        assert "FAIL" in caplog.text

    def test_batch_end_logs_info(self, caplog):
        target = _ConsoleTarget()
        event = _BatchEndEvent(success_rate=0.85, mean_reward=0.75, num_tasks=20)
        with caplog.at_level(logging.INFO, logger="llenvs.evaluation"):
            target.on_batch_end(event)
        assert "85" in caplog.text
        assert "0.750" in caplog.text

    def test_step_logs_debug(self, caplog):
        target = _ConsoleTarget()
        event = _StepEvent(
            task_index=0,
            step_num=1,
            reward_total=0.5,
            prompt_tokens=10,
            completion_tokens=5,
            has_tool_calls=False,
            num_tool_calls=0,
        )
        with caplog.at_level(logging.DEBUG, logger="llenvs.evaluation"):
            target.on_step(event)
        assert "Task 0" in caplog.text
        assert "step 1" in caplog.text

    def test_error_logs_warning(self, caplog):
        target = _ConsoleTarget()
        event = _ErrorEvent(task_index=2, phase="reset", error="boom")
        with caplog.at_level(logging.WARNING, logger="llenvs.evaluation"):
            target.on_error(event)
        assert "Task 2" in caplog.text
        assert "reset" in caplog.text
        assert "boom" in caplog.text


# ===========================================================================
# _FileTarget
# ===========================================================================


class TestFileTarget:
    def test_creates_directory_structure(self, tmp_path):
        target = _FileTarget(str(tmp_path / "logs"), "my_env")
        target.close()
        assert (tmp_path / "logs" / "my_env").is_dir()

    def test_writes_valid_jsonl(self, tmp_path):
        target = _FileTarget(str(tmp_path / "logs"), "env1")
        target.on_batch_start(_BatchStartEvent(num_tasks=5, environment_name="env1", max_steps=10))
        target.on_trajectory_end(
            _TrajectoryEndEvent(
                task_index=0,
                success=True,
                total_reward=1.0,
                num_steps=1,
                completed_count=1,
                total_count=5,
            )
        )
        target.close()

        # Find the JSONL file
        files = list((tmp_path / "logs" / "env1").glob("*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)
            assert "event" in parsed

    def test_event_types_in_json(self, tmp_path):
        target = _FileTarget(str(tmp_path / "logs"), "env1")
        target.on_batch_start(_BatchStartEvent(num_tasks=5, environment_name="env1", max_steps=10))
        target.on_step(
            _StepEvent(
                task_index=0,
                step_num=1,
                reward_total=0.5,
                prompt_tokens=10,
                completion_tokens=5,
                has_tool_calls=False,
                num_tool_calls=0,
            )
        )
        target.on_error(_ErrorEvent(task_index=1, phase="step", error="oops"))
        target.on_batch_end(_BatchEndEvent(success_rate=0.8, mean_reward=0.7, num_tasks=5))
        target.close()

        files = list((tmp_path / "logs" / "env1").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        events = [json.loads(line)["event"] for line in lines]
        assert events == ["batch_start", "step", "error", "batch_end"]

    def test_close_flushes_file(self, tmp_path):
        target = _FileTarget(str(tmp_path / "logs"), "env1")
        target.on_batch_start(_BatchStartEvent(num_tasks=1, environment_name="env1", max_steps=10))
        target.close()

        files = list((tmp_path / "logs" / "env1").glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text()
        assert content.strip()  # Not empty

    def test_multiple_events_multiple_lines(self, tmp_path):
        target = _FileTarget(str(tmp_path / "logs"), "env1")
        for i in range(5):
            target.on_step(
                _StepEvent(
                    task_index=0,
                    step_num=i,
                    reward_total=float(i),
                    prompt_tokens=10,
                    completion_tokens=5,
                    has_tool_calls=False,
                    num_tool_calls=0,
                )
            )
        target.close()

        files = list((tmp_path / "logs" / "env1").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 5

    def test_default_log_dir(self, tmp_path, monkeypatch):
        # Change cwd so .logs goes in tmp
        monkeypatch.chdir(tmp_path)
        target = _FileTarget(".logs", "test_env")
        target.on_batch_start(
            _BatchStartEvent(num_tasks=1, environment_name="test_env", max_steps=10)
        )
        target.close()
        assert (tmp_path / ".logs" / "test_env").is_dir()


# ===========================================================================
# _WandbTarget
# ===========================================================================


class TestWandbTarget:
    def test_auto_creates_run(self):
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            config = LogConfig(targets=("wandb",), wandb_project="test-proj")
            target = _WandbTarget(config)
            mock_wandb.init.assert_called_once()
            assert target._owns_run

    def test_uses_existing_run(self):
        mock_wandb = MagicMock()
        existing_run = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            config = LogConfig(targets=("wandb",), wandb_run=existing_run)
            target = _WandbTarget(config)
            mock_wandb.init.assert_not_called()
            assert not target._owns_run

    def test_close_finishes_only_owned_run(self):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Owned run
            config = LogConfig(targets=("wandb",), wandb_project="test")
            target = _WandbTarget(config)
            target.close()
            mock_run.finish.assert_called_once()

        # Not owned
        mock_wandb2 = MagicMock()
        existing_run = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb2}):
            config = LogConfig(targets=("wandb",), wandb_run=existing_run)
            target = _WandbTarget(config)
            target.close()
            existing_run.finish.assert_not_called()

    def test_import_error_message(self):
        with patch.dict("sys.modules", {"wandb": None}):
            config = LogConfig(targets=("wandb",))
            with pytest.raises(ImportError, match="wandb"):
                _WandbTarget(config)


# ===========================================================================
# Runner integration
# ===========================================================================


class TestRunnerIntegration:
    def test_run_batch_emits_all_events(self, tmp_path):
        """run_batch with log emits batch_start, step, trajectory_end, batch_end."""
        log_dir = str(tmp_path / "logs")
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )

        result = runner.run_batch([0, 1, 2])
        assert len(result.trajectory_results) == 3

        # Check JSONL file
        files = list((tmp_path / "logs" / "mock_single").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        events = [json.loads(line)["event"] for line in lines]

        assert events[0] == "batch_start"
        assert events[-1] == "batch_end"
        assert "step" in events
        assert "trajectory_end" in events

    def test_run_batch_without_log_works(self):
        """run_batch without log parameter works fine."""
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])
        assert len(result.trajectory_results) == 3
        assert result.success_rate == 1.0

    def test_run_trajectory_emits_trajectory_events(self, tmp_path):
        """run_trajectory emits step and trajectory_end but not batch events."""
        log_dir = str(tmp_path / "logs")
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )

        result = runner.run_trajectory(task_index=0)
        assert result.success

        files = list((tmp_path / "logs" / "mock_single").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        events = [json.loads(line)["event"] for line in lines]

        assert "batch_start" not in events
        assert "batch_end" not in events
        assert "step" in events
        assert "trajectory_end" in events

    def test_error_during_reset_emits_error_event(self, tmp_path):
        """Error during reset emits error event."""
        log_dir = str(tmp_path / "logs")
        env = MockFailingResetEnv(failing_indices={1}, num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )

        with pytest.raises(RuntimeError, match="Error resetting task 1"):
            runner.run_batch([0, 1, 2])

        files = list((tmp_path / "logs" / "mock_failing").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["phase"] == "reset"
        assert error_events[0]["task_index"] == 1

    def test_error_during_step_emits_error_event(self, tmp_path):
        """Error during step emits error event."""
        log_dir = str(tmp_path / "logs")
        env = MockStepFailEnv(failing_indices={1}, num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )

        with pytest.raises(RuntimeError, match="Error stepping task 1"):
            runner.run_batch([0, 1, 2])

        files = list((tmp_path / "logs" / "mock_step_fail").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["phase"] == "step"
        assert error_events[0]["task_index"] == 1

    def test_chunked_execution_single_logger(self, tmp_path):
        """Chunked execution: logger created once, events emitted correctly."""
        log_dir = str(tmp_path / "logs")
        env = MockSingleTurnEnv(num_tasks=6)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )

        result = runner.run_batch([0, 1, 2, 3, 4, 5], batch_size=2)
        assert len(result.trajectory_results) == 6

        # Should have exactly one JSONL file (one logger)
        files = list((tmp_path / "logs" / "mock_single").glob("*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        event_types = [e["event"] for e in events]

        # batch_start and batch_end should be present
        assert event_types[0] == "batch_start"
        assert event_types[-1] == "batch_end"
        # All 6 trajectories should have trajectory_end
        traj_ends = [e for e in events if e["event"] == "trajectory_end"]
        assert len(traj_ends) == 6

    def test_run_evaluation_passes_log(self, tmp_path):
        """run_evaluation() convenience passes log to runner."""
        log_dir = str(tmp_path / "logs")
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()

        result = run_evaluation(
            environment=env,
            backend=backend,
            task_indices=[0, 1, 2],
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )
        assert result.success_rate == 1.0

        files = list((tmp_path / "logs" / "mock_single").glob("*.jsonl"))
        assert len(files) == 1

    def test_multi_eval_separate_loggers(self, tmp_path):
        """Multi-eval creates separate loggers per entry."""
        log_dir = str(tmp_path / "logs")
        env1 = MockSingleTurnEnv(num_tasks=2)
        env2 = MockMultiTurnEnv(steps_per_task={0: 1, 1: 2})
        backend = BatchTrackingBackend()

        runner1 = TrajectoryRunner(
            environment=env1,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )
        runner2 = TrajectoryRunner(
            environment=env2,
            backend=backend,
            log=LogConfig(targets=("file",), log_dir=log_dir),
        )

        results = run_multi_evaluation(
            [
                MultiEvalEntry(runner=runner1, task_indices=[0, 1]),
                MultiEvalEntry(runner=runner2, task_indices=[0, 1]),
            ]
        )

        assert len(results) == 2
        # Each env should have its own log directory
        assert (tmp_path / "logs" / "mock_single").is_dir()
        assert (tmp_path / "logs" / "mock_multi").is_dir()
