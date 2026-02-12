"""Evaluation logging system.

Provides structured logging for evaluation runs with three targets:
console (Python logging), file (JSONL), and W&B. Configured via LogConfig.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_VALID_TARGETS = frozenset({"console", "file", "wandb"})

logger = logging.getLogger("llenvs.evaluation")


# ===========================================================================
# LogConfig (public)
# ===========================================================================


@dataclass
class LogConfig:
    """Configuration for evaluation logging.

    Attributes:
        targets: Tuple of target names to activate.
        log_dir: Directory for JSONL file logging.
        wandb_run: Existing wandb.Run to use (skips init).
        wandb_project: W&B project name for auto-created runs.
        wandb_name: W&B run name (auto-generated if None).
        wandb_config: Extra config dict to log to W&B.
    """

    targets: tuple[str, ...] = ("console",)
    log_dir: str = ".logs"
    wandb_run: Any = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    wandb_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.targets) - _VALID_TARGETS
        if unknown:
            raise ValueError(
                f"Unknown log target(s): {unknown}. Valid targets: {sorted(_VALID_TARGETS)}"
            )


# ===========================================================================
# Internal events (private, frozen)
# ===========================================================================


@dataclass(frozen=True)
class _BatchStartEvent:
    num_tasks: int
    environment_name: str
    max_steps: int


@dataclass(frozen=True)
class _BatchEndEvent:
    success_rate: float
    mean_reward: float
    num_tasks: int


@dataclass(frozen=True)
class _TrajectoryEndEvent:
    task_index: int
    success: bool
    total_reward: float
    num_steps: int
    completed_count: int
    total_count: int


@dataclass(frozen=True)
class _StepEvent:
    task_index: int
    step_num: int
    reward_total: float
    prompt_tokens: int
    completion_tokens: int
    has_tool_calls: bool
    num_tool_calls: int


@dataclass(frozen=True)
class _ErrorEvent:
    task_index: int
    phase: str
    error: str


# ===========================================================================
# _LogTarget protocol (private)
# ===========================================================================


@runtime_checkable
class _LogTarget(Protocol):
    def on_batch_start(self, event: _BatchStartEvent) -> None: ...
    def on_batch_end(self, event: _BatchEndEvent) -> None: ...
    def on_trajectory_end(self, event: _TrajectoryEndEvent) -> None: ...
    def on_step(self, event: _StepEvent) -> None: ...
    def on_error(self, event: _ErrorEvent) -> None: ...
    def close(self) -> None: ...


# ===========================================================================
# _ConsoleTarget
# ===========================================================================


class _ConsoleTarget:
    """Logs events to Python logging at INFO/DEBUG/WARNING levels."""

    def on_batch_start(self, event: _BatchStartEvent) -> None:
        logger.info(
            "Starting evaluation: %s, %d tasks, max %d steps",
            event.environment_name, event.num_tasks, event.max_steps,
        )

    def on_batch_end(self, event: _BatchEndEvent) -> None:
        logger.info(
            "Evaluation complete: success_rate=%.2f%%, mean_reward=%.3f",
            event.success_rate * 100, event.mean_reward,
        )

    def on_trajectory_end(self, event: _TrajectoryEndEvent) -> None:
        status = "OK" if event.success else "FAIL"
        logger.info(
            "[%d/%d] Task %d: %s reward=%.3f steps=%d",
            event.completed_count, event.total_count,
            event.task_index, status, event.total_reward, event.num_steps,
        )

    def on_step(self, event: _StepEvent) -> None:
        logger.debug(
            "Task %d step %d: reward=%.3f tokens=%d",
            event.task_index, event.step_num, event.reward_total,
            event.prompt_tokens + event.completion_tokens,
        )

    def on_error(self, event: _ErrorEvent) -> None:
        logger.warning(
            "Task %d error in %s: %s",
            event.task_index, event.phase, event.error,
        )

    def close(self) -> None:
        pass


# ===========================================================================
# _FileTarget
# ===========================================================================


class _FileTarget:
    """Writes JSONL to {log_dir}/{env_name}/{timestamp}.jsonl."""

    def __init__(self, log_dir: str, environment_name: str) -> None:
        dir_path = Path(log_dir) / environment_name
        dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = dir_path / f"{timestamp}.jsonl"
        self._file = open(self._path, "w")

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        record = {"event": event_type, **data}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def on_batch_start(self, event: _BatchStartEvent) -> None:
        self._write("batch_start", asdict(event))

    def on_batch_end(self, event: _BatchEndEvent) -> None:
        self._write("batch_end", asdict(event))

    def on_trajectory_end(self, event: _TrajectoryEndEvent) -> None:
        self._write("trajectory_end", asdict(event))

    def on_step(self, event: _StepEvent) -> None:
        self._write("step", asdict(event))

    def on_error(self, event: _ErrorEvent) -> None:
        self._write("error", asdict(event))

    def close(self) -> None:
        self._file.flush()
        self._file.close()


# ===========================================================================
# _WandbTarget
# ===========================================================================


class _WandbTarget:
    """Logs events to Weights & Biases."""

    def __init__(self, config: LogConfig) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for W&B logging. "
                "Install it with: pip install 'llenvs[wandb]'"
            )

        if config.wandb_run is not None:
            self._run = config.wandb_run
            self._owns_run = False
        else:
            self._run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                config=config.wandb_config or {},
            )
            self._owns_run = True

        self._wandb = wandb

    def on_batch_start(self, event: _BatchStartEvent) -> None:
        self._run.config.update({
            "environment": event.environment_name,
            "num_tasks": event.num_tasks,
            "max_steps": event.max_steps,
        })

    def on_batch_end(self, event: _BatchEndEvent) -> None:
        self._run.log({
            "batch/success_rate": event.success_rate,
            "batch/mean_reward": event.mean_reward,
        })
        self._run.summary.update({
            "success_rate": event.success_rate,
            "mean_reward": event.mean_reward,
            "num_tasks": event.num_tasks,
        })

    def on_trajectory_end(self, event: _TrajectoryEndEvent) -> None:
        self._run.log({
            "trajectory/success": int(event.success),
            "trajectory/reward": event.total_reward,
            "trajectory/completed": event.completed_count,
        })

    def on_step(self, event: _StepEvent) -> None:
        self._run.log({
            "step/reward": event.reward_total,
            "step/tokens": event.prompt_tokens + event.completion_tokens,
            "step/prompt_tokens": event.prompt_tokens,
            "step/completion_tokens": event.completion_tokens,
            "step/num_tool_calls": event.num_tool_calls,
        })

    def on_error(self, event: _ErrorEvent) -> None:
        self._run.log({
            "error/task_index": event.task_index,
            "error/phase": event.phase,
        })

    def close(self) -> None:
        if self._owns_run:
            self._run.finish()


# ===========================================================================
# _EvaluationLogger (internal dispatcher)
# ===========================================================================


class _EvaluationLogger:
    """Internal dispatcher. Created from LogConfig, not user-facing."""

    def __init__(self, config: LogConfig, environment_name: str) -> None:
        self._targets: list[_LogTarget] = []
        if "console" in config.targets:
            self._targets.append(_ConsoleTarget())
        if "file" in config.targets:
            self._targets.append(_FileTarget(config.log_dir, environment_name))
        if "wandb" in config.targets:
            self._targets.append(_WandbTarget(config))

    def on_batch_start(self, event: _BatchStartEvent) -> None:
        for t in self._targets:
            t.on_batch_start(event)

    def on_batch_end(self, event: _BatchEndEvent) -> None:
        for t in self._targets:
            t.on_batch_end(event)

    def on_trajectory_end(self, event: _TrajectoryEndEvent) -> None:
        for t in self._targets:
            t.on_trajectory_end(event)

    def on_step(self, event: _StepEvent) -> None:
        for t in self._targets:
            t.on_step(event)

    def on_error(self, event: _ErrorEvent) -> None:
        for t in self._targets:
            t.on_error(event)

    def close(self) -> None:
        for t in self._targets:
            t.close()
