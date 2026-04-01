"""Harbor adapter — wraps Harbor containerized evaluation environments.

Harbor (Laude Institute) is a generic framework for containerized agent
evaluation. It manages Docker containers, task discovery via a JSON registry,
and verification (test scripts produce binary pass/fail rewards).

By wrapping Harbor (not individual benchmarks), this adapter provides access
to Terminal-Bench, aider-polyglot, swe-bench, and other datasets through a
single interface.

Dual-mode design:
- **Text mode** (``HarborEnvironment``): Agent sends shell commands as text,
  receives stdout/stderr. Submit via keyword in action text.
- **Tool mode** (``HarborToolEnvironment``): Agent uses structured tool calls
  (``execute_command``, ``read_file``, ``write_file``, ``submit``).

Reference: https://github.com/laude-institute/harbor
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from llenvs.core.async_utils import run_async
from llenvs.core.environment import (
    EnvironmentSpec,
    StepResult,
    _StateContinuityTracker,
)
from llenvs.core.extraction import AnswerExtractor
from llenvs.core.reward import (
    RewardFunction,
    RewardType,
    Signal,
    SignalBundle,
)
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)

logger = logging.getLogger(__name__)

_HARBOR_TASK_CACHE: dict[tuple[Any, ...], tuple[Any, ...]] = {}
_HARBOR_TASK_CACHE_LOCK = threading.Lock()

_APPTAINER_VERSION_CACHE: dict[str, str] = {}
_APPTAINER_VERSION_CACHE_LOCK = threading.Lock()
_APPTAINER_RUNTIME_INFO_LOGGED_KEYS: set[tuple[Any, ...]] = set()
_APPTAINER_RUNTIME_INFO_LOGGED_KEYS_LOCK = threading.Lock()
_APPTAINER_PID_FLAG_CACHE: dict[tuple[str, str], str] = {}
_APPTAINER_PID_FLAG_CACHE_LOCK = threading.Lock()
_APPTAINER_PID_FLAG_EVENTS: dict[tuple[str, str], threading.Event] = {}
_APPTAINER_ROOTFS_PROBE_CACHE: dict[tuple[Any, ...], bool] = {}
_APPTAINER_ROOTFS_PROBE_CACHE_LOCK = threading.Lock()
_APPTAINER_ROOTFS_PROBE_EVENTS: dict[tuple[Any, ...], threading.Event] = {}


def _run_with_timeout(coro: Any, timeout: int | None, label: str) -> Any:
    if timeout is None:
        return run_async(coro)
    try:
        return run_async(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout}s") from exc

# ── Tool definitions (for tool mode) ────────────────────────────

HARBOR_EXECUTE_COMMAND_TOOL = ToolDefinition(
    name="execute_command",
    description="Run a shell command in the container.",
    parameters=(
        ToolParameter(
            name="command",
            type=ToolParameterType.STRING,
            description="Shell command to execute",
        ),
        ToolParameter(
            name="cwd",
            type=ToolParameterType.STRING,
            description="Working directory (default: /)",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type=ToolParameterType.INTEGER,
            description="Timeout in seconds (default: 120)",
            required=False,
        ),
    ),
)

HARBOR_READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read file contents from the container.",
    parameters=(
        ToolParameter(
            name="path",
            type=ToolParameterType.STRING,
            description="Absolute file path to read",
        ),
    ),
)

HARBOR_WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write file contents to the container.",
    parameters=(
        ToolParameter(
            name="path",
            type=ToolParameterType.STRING,
            description="Absolute file path to write",
        ),
        ToolParameter(
            name="content",
            type=ToolParameterType.STRING,
            description="Content to write to the file",
        ),
    ),
)

HARBOR_SUBMIT_TOOL = ToolDefinition(
    name="submit",
    description="Signal task completion and trigger verification.",
    parameters=(),
    is_terminal=True,
)

HARBOR_TOOLS: tuple[ToolDefinition, ...] = (
    HARBOR_EXECUTE_COMMAND_TOOL,
    HARBOR_READ_FILE_TOOL,
    HARBOR_WRITE_FILE_TOOL,
    HARBOR_SUBMIT_TOOL,
)


# ── Hidden state ────────────────────────────────────────────────


@dataclass(frozen=True)
class HarborSnapshotOptions:
    """Options required to export and restore an exact Harbor snapshot."""

    file_locks: bool = False
    tcp_established: bool = False
    tcp_close: bool = False
    ignore_volumes: bool = False


@dataclass(frozen=True)
class HarborSnapshotRef:
    """Reference to an exact runtime snapshot artifact on disk."""

    runtime: str
    relative_path: str
    options: HarborSnapshotOptions = field(default_factory=HarborSnapshotOptions)


@dataclass(frozen=True)
class HarborSnapshotEligibility:
    """Static exact-snapshot eligibility for a Harbor task."""

    task_index: int
    task_name: str
    eligible: bool
    reason_code: str | None = None
    reason_detail: str | None = None


@dataclass(frozen=True)
class RuntimeEligibility:
    """Static runtime eligibility for a Harbor task."""

    task_index: int
    task_name: str
    eligible: bool
    reason_code: str | None = None
    reason_detail: str | None = None


@dataclass(frozen=True)
class HarborHidden:
    """Hidden state for Harbor environments.

    Attributes:
        task_index: Index into the task list.
        task_name: The Harbor task identifier.
        instruction: Task instruction text.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        trajectory: Command history (frozen tuple).
        snapshot_ref: Optional exact runtime snapshot artifact for this state.
        fs_restore_risk_now: Whether the current state has filesystem-restore risk.
        fs_restore_risk_reasons: Reasons for the current risk signal.
        fs_restore_risk_ever: Sticky flag — True if any prior state had risk.
    """

    task_index: int
    task_name: str
    instruction: str
    episode_step: int
    last_action: str | None = None
    trajectory: tuple[str, ...] = ()
    snapshot_ref: HarborSnapshotRef | None = None
    fs_restore_risk_now: bool = False
    fs_restore_risk_reasons: tuple[str, ...] = ()
    fs_restore_risk_ever: bool = False


@dataclass(frozen=True)
class RuntimeProbeSnapshot:
    """Snapshot of container runtime state for filesystem-restore risk detection.

    Captured via a single exec into the running container. Used to detect
    state that may not survive a tar-based filesystem checkpoint/restore
    (e.g., background processes, new mounts, open sockets).
    """

    process_commands: frozenset[str]
    mount_fingerprint: str
    listening_ports: frozenset[int]
    staging_has_content: bool
    probe_failed: bool = False
    probe_error: str | None = None


# ── Reward function ─────────────────────────────────────────────


@dataclass
class HarborReward:
    """Native reward function reading Harbor's verifier result.

    Non-terminal steps return STEP signal with None reward.
    Terminal steps return OUTCOME signal with the verifier reward
    (read from ``next_state.metadata.info["reward"]``).
    """

    _name: str = "harbor"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> Signal:
        is_terminal = next_state.metadata.is_terminal

        if not is_terminal:
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=None,
                metadata={"is_terminal": False},
            )

        reward = next_state.metadata.info.get("reward", 0.0)
        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=float(reward),
            metadata={"is_terminal": True},
        )


@dataclass(frozen=True)
class _HarborAPI:
    registry_client_factory: Any
    task_client: Any
    task_class: Any
    task_paths_class: Any
    environment_factory: Any
    environment_type_enum: Any
    trial_paths_class: Any
    verifier_class: Any


@dataclass(frozen=True)
class _CLIResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass(frozen=True)
class _PodmanVolumeMount:
    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class _PodmanHealthcheck:
    test: str | tuple[str, ...] | None = None
    interval_sec: float = 1.0
    timeout_sec: float = 30.0
    retries: int = 30
    start_period_sec: float = 0.0


@dataclass(frozen=True)
class _PodmanServiceSpec:
    name: str
    image: str | None
    build_context: Path | None
    dockerfile: Path | None
    command: str | tuple[str, ...] | None
    entrypoint: str | tuple[str, ...] | None
    environment: tuple[tuple[str, str], ...]
    working_dir: str | None
    volumes: tuple[_PodmanVolumeMount, ...]
    depends_on: tuple[str, ...]
    healthcheck: _PodmanHealthcheck | None = None


# ── Helpers ─────────────────────────────────────────────────────


def _format_exec_result(result: Any) -> str:
    """Format an exec result as observation text.

    Shows stdout always, stderr with [stderr] prefix when non-empty.
    When both empty, shows exit code.
    """
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return_code = getattr(result, "return_code", 0)

    if not stdout and not stderr:
        return f"[exit code: {return_code}]"

    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr] {stderr}")
    return "\n".join(parts)


def _normalize_text_exec_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"independent_exec", "tmux_session"}:
        raise ValueError(
            f"Unknown Harbor text_exec_mode: {mode!r}. "
            "Valid values: ['independent_exec', 'tmux_session']"
        )
    return normalized


def _pick_heredoc_delimiter(text: str) -> str:
    """Choose a heredoc delimiter that does not collide with command lines."""
    existing_lines = set(text.splitlines())
    while True:
        candidate = f"LLENVS_HARBOR_CMD_{uuid.uuid4().hex}"
        if candidate not in existing_lines:
            return candidate


def _tmux_wait_channel(prefix: str = "llenvs_harbor") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class _HarborTmuxTextSession:
    """Persistent tmux-backed shell for Harbor text-mode environments."""

    _SESSION_NAME = "llenvs-harbor"
    _TOKEN_FILE = "/tmp/.llenvs_harbor_tmux_token"
    _COMMAND_FILE = "/tmp/.llenvs_harbor_tmux_command"
    _BUFFER_NAME = "llenvs-harbor-buffer"
    _INLINE_COMMAND_MAX_CHARS = 32768
    _DIAGNOSTIC_TAIL_LINES = 200

    def __init__(
        self,
        harbor_env: Any,
        *,
        exec_timeout: int,
        bootstrap_if_missing: bool,
    ) -> None:
        self._harbor_env = harbor_env
        self._exec_timeout = exec_timeout
        self._bootstrap_if_missing = bootstrap_if_missing
        self._previous_full_buffer = ""
        self.tmux_bootstrapped = False
        self.tmux_start_method = "direct"

    def start(self) -> None:
        if not self._probe_tmux():
            if not self._bootstrap_if_missing:
                raise RuntimeError(
                    "tmux is not available inside the Harbor task container. "
                    "Set tmux_bootstrap_if_missing=True or preinstall tmux in the image."
                )
            self._bootstrap_tmux()
            self.tmux_bootstrapped = True
            if not self._probe_tmux():
                raise RuntimeError(
                    "tmux bootstrap completed but `tmux -V` still failed inside the container"
                )

        self._start_session()
        self._exec(
            f"tmux set-option -t {shlex.quote(self._SESSION_NAME)} history-limit 50000"
        )
        self._install_prompt_hook()
        self._previous_full_buffer = self._capture_full_buffer()

    def resync_after_restore(self) -> None:
        self._exec(f"tmux has-session -t {shlex.quote(self._SESSION_NAME)}")
        self._previous_full_buffer = self._capture_full_buffer()

    def run_command(self, command: str) -> str:
        command_text = command[:-1] if command.endswith("\n") else command
        step_token = _tmux_wait_channel("llenvs_harbor_step")
        self._send_command(command_text, step_token=step_token)

        wait_and_capture = (
            f"tmux wait-for -L {shlex.quote(step_token)}"
            f" && tmux wait-for -U {shlex.quote(step_token)}"
            f" && tmux capture-pane -p -S - -t {shlex.quote(self._SESSION_NAME)}"
        )
        try:
            result = self._exec(wait_and_capture, timeout_sec=self._exec_timeout)
        except Exception as exc:
            if self._is_timeout_error(exc):
                raise self._handle_timeout(command_text, step_token, exc) from exc
            raise

        full_buffer = getattr(result, "stdout", "") or ""
        observation = self._diff_full_buffer(full_buffer)
        self._previous_full_buffer = full_buffer
        return observation

    def _send_command(self, command: str, *, step_token: str) -> None:
        session_q = shlex.quote(self._SESSION_NAME)
        token_q = shlex.quote(step_token)
        token_file_q = shlex.quote(self._TOKEN_FILE)
        command_file_q = shlex.quote(self._COMMAND_FILE)
        buffer_q = shlex.quote(self._BUFFER_NAME)

        control_parts = [
            f"tmux has-session -t {session_q}",
            f"tmux wait-for -L {token_q}",
        ]
        if len(command) <= self._INLINE_COMMAND_MAX_CHARS:
            delimiter = _pick_heredoc_delimiter(command)
            control_parts.append(f"printf '%s' {token_q} > {token_file_q}")
            control_parts.append(
                "cat > "
                f"{command_file_q} << '{delimiter}'\n{command}\n{delimiter}"
            )
            control_parts.append(
                f"tmux load-buffer -b {buffer_q} {command_file_q}"
            )
        else:
            self._upload_command_file(command)
            control_parts.append(f"printf '%s' {token_q} > {token_file_q}")
            control_parts.append(
                f"tmux load-buffer -b {buffer_q} {command_file_q}"
            )
        control_parts.append(
            f"tmux paste-buffer -b {buffer_q} -t {session_q}"
        )
        control_parts.append(
            f"tmux send-keys -t {session_q} Enter"
        )
        self._exec(" && ".join(control_parts), timeout_sec=self._exec_timeout)

    def _upload_command_file(self, command: str) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(command)
            temp_path = Path(handle.name)
        try:
            run_async(self._harbor_env.upload_file(str(temp_path), self._COMMAND_FILE))
        finally:
            temp_path.unlink(missing_ok=True)

    def _capture_full_buffer(self) -> str:
        result = self._exec(
            f"tmux capture-pane -p -S - -t {shlex.quote(self._SESSION_NAME)}",
            timeout_sec=self._exec_timeout,
        )
        return getattr(result, "stdout", "") or ""

    def _capture_visible_screen(self) -> str:
        result = self._exec(
            f"tmux capture-pane -p -t {shlex.quote(self._SESSION_NAME)}",
            timeout_sec=self._exec_timeout,
        )
        return getattr(result, "stdout", "") or ""

    def _diff_full_buffer(self, full_buffer: str) -> str:
        previous = self._previous_full_buffer
        if not previous:
            return full_buffer
        if full_buffer.startswith(previous):
            return full_buffer[len(previous) :]
        start = full_buffer.rfind(previous)
        if start != -1:
            return full_buffer[start + len(previous) :]
        return self._capture_visible_screen()

    def _install_prompt_hook(self) -> None:
        init_token = _tmux_wait_channel("llenvs_harbor_init")
        token_file_q = shlex.quote(self._TOKEN_FILE)
        init_script = "\n".join(
            [
                "__llenvs_harbor_prompt_hook() {",
                f"  local token_file={token_file_q}",
                "  local token=\"\"",
                "  if [ -r \"$token_file\" ]; then",
                "    token=$(cat \"$token_file\" 2>/dev/null || true)",
                "    if [ -n \"$token\" ]; then",
                "      tmux wait-for -U \"$token\" 2>/dev/null || true",
                "      : > \"$token_file\"",
                "    fi",
                "  fi",
                "}",
                "__llenvs_harbor_extend_prompt_command() {",
                "  local hook=\"__llenvs_harbor_prompt_hook\"",
                "  local decl=\"\"",
                "  decl=$(declare -p PROMPT_COMMAND 2>/dev/null || true)",
                "  case \"$decl\" in",
                "    \"declare -a \"*)",
                "      local entry",
                "      for entry in \"${PROMPT_COMMAND[@]}\"; do",
                "        if [ \"$entry\" = \"$hook\" ]; then",
                "          return",
                "        fi",
                "      done",
                "      PROMPT_COMMAND=(\"$hook\" \"${PROMPT_COMMAND[@]}\")",
                "      ;;",
                "    *)",
                "      case \";${PROMPT_COMMAND:-};\" in",
                "        *\";$hook;\"*) ;;",
                "        *)",
                "          if [ -n \"${PROMPT_COMMAND:-}\" ]; then",
                "            PROMPT_COMMAND=\"$hook;$PROMPT_COMMAND\"",
                "          else",
                "            PROMPT_COMMAND=\"$hook\"",
                "          fi",
                "          ;;",
                "      esac",
                "      ;;",
                "  esac",
                "}",
                "__llenvs_harbor_extend_prompt_command",
            ]
        )
        self._run_internal_session_command(init_script, wait_token=init_token)

    def _run_internal_session_command(self, command_text: str, *, wait_token: str) -> None:
        command_file_q = shlex.quote(self._COMMAND_FILE)
        buffer_q = shlex.quote(self._BUFFER_NAME)
        session_q = shlex.quote(self._SESSION_NAME)
        token_q = shlex.quote(wait_token)
        token_file_q = shlex.quote(self._TOKEN_FILE)
        delimiter = _pick_heredoc_delimiter(command_text)
        control_cmd = " && ".join(
            [
                f"tmux has-session -t {session_q}",
                f"tmux wait-for -L {token_q}",
                f"printf '%s' {token_q} > {token_file_q}",
                "cat > "
                f"{command_file_q} << '{delimiter}'\n{command_text}\n{delimiter}",
                f"tmux load-buffer -b {buffer_q} {command_file_q}",
                f"tmux paste-buffer -b {buffer_q} -t {session_q}",
                f"tmux send-keys -t {session_q} Enter",
            ]
        )
        self._exec(control_cmd, timeout_sec=self._exec_timeout)
        self._exec(
            f"tmux wait-for -L {token_q}"
            f" && tmux wait-for -U {token_q}",
            timeout_sec=self._exec_timeout,
        )

    def _handle_timeout(self, command: str, step_token: str, exc: Exception) -> RuntimeError:
        visible = self._safe_capture(self._capture_visible_screen)
        full = self._safe_capture(self._capture_full_buffer)
        self._safe_exec(
            f"tmux send-keys -t {shlex.quote(self._SESSION_NAME)} C-c",
            timeout_sec=10,
        )
        recovered = True
        try:
            token_q = shlex.quote(step_token)
            self._exec(
                f"tmux wait-for -L {token_q}"
                f" && tmux wait-for -U {token_q}",
                timeout_sec=5,
            )
        except Exception:
            recovered = False
        recovered_buffer = self._safe_capture(self._capture_full_buffer)
        if recovered_buffer:
            self._previous_full_buffer = recovered_buffer
        details: list[str] = [
            f"Harbor tmux session command timed out after {self._exec_timeout}s: {command}"
        ]
        if visible:
            details.append("Visible screen:\n" + visible)
        if full:
            tail_lines = "\n".join(full.splitlines()[-self._DIAGNOSTIC_TAIL_LINES :])
            details.append("Full buffer tail:\n" + tail_lines)
        if not recovered:
            details.append("Session unrecoverable after timeout")
        return RuntimeError("\n".join(details))

    def _probe_tmux(self) -> bool:
        try:
            self._exec("tmux -V", timeout_sec=30)
        except Exception:
            return False
        return True

    def _bootstrap_tmux(self) -> None:
        bootstrap_cmd = "\n".join(
            [
                "set -e",
                "export TMPDIR=/tmp TMP=/tmp TEMP=/tmp",
                "if command -v apt-get >/dev/null 2>&1; then",
                "  export DEBIAN_FRONTEND=noninteractive",
                "  apt-get update",
                "  apt-get install -y tmux",
                "  apt-get clean",
                "  rm -rf /var/lib/apt/lists/*",
                "elif command -v dnf >/dev/null 2>&1; then",
                "  dnf install -y tmux",
                "  dnf clean all",
                "elif command -v yum >/dev/null 2>&1; then",
                "  yum install -y tmux",
                "  yum clean all",
                "elif command -v apk >/dev/null 2>&1; then",
                "  apk add --no-cache tmux",
                "else",
                "  echo 'No supported package manager found for tmux bootstrap' >&2",
                "  exit 1",
                "fi",
            ]
        )
        self._exec(bootstrap_cmd, timeout_sec=max(self._exec_timeout, 120))

    def _start_session(self) -> None:
        session_q = shlex.quote(self._SESSION_NAME)
        direct_cmd = (
            f"tmux new-session -d -s {session_q} "
            f"{shlex.quote('bash --login')}"
        )
        try:
            self._exec(direct_cmd, timeout_sec=self._exec_timeout)
            self.tmux_start_method = "direct"
            return
        except Exception as exc:
            if not self._looks_like_pty_error(exc) or not self._script_available():
                raise
        script_wrapper = shlex.quote("script -qc 'bash --login' /dev/null")
        fallback_cmd = (
            f"tmux new-session -d -s {session_q} "
            f"{script_wrapper}"
        )
        self._exec(fallback_cmd, timeout_sec=self._exec_timeout)
        self.tmux_start_method = "script_fallback"

    def _script_available(self) -> bool:
        result = self._exec(
            "command -v script >/dev/null 2>&1 && printf '%s' yes || printf '%s' no",
            timeout_sec=10,
        )
        stdout = (getattr(result, "stdout", "") or "").strip()
        return bool(stdout) and stdout != "no"

    @staticmethod
    def _looks_like_pty_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            needle in text
            for needle in (
                "open terminal failed",
                "not a terminal",
                "tty",
                "pseudoterminal",
                "terminal is required",
            )
        )

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "timed out" in text or "timeout" in text

    def _safe_exec(self, command: str, *, timeout_sec: int | None = None) -> Any | None:
        try:
            return self._exec(command, timeout_sec=timeout_sec)
        except Exception:
            return None

    @staticmethod
    def _safe_capture(capture_fn: Callable[[], str]) -> str:
        try:
            return capture_fn()
        except Exception:
            return ""

    def _exec(self, command: str, *, timeout_sec: int | None = None) -> Any:
        result = run_async(
            self._harbor_env.exec(
                command,
                timeout_sec=self._exec_timeout if timeout_sec is None else timeout_sec,
            )
        )
        return_code = getattr(result, "return_code", 0)
        if return_code != 0:
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            raise RuntimeError(
                "Harbor tmux helper command failed "
                f"(exit {return_code}): {command}\nstdout: {stdout}\nstderr: {stderr}"
            )
        return result


def _run_verifier(
    verifier_factory: Any,
    task: Any,
    harbor_env: Any,
) -> dict[str, float]:
    """Run the verifier and return rewards dict."""
    verifier = verifier_factory(task, harbor_env)
    result = run_async(verifier.verify())
    return result.rewards


def _normalize_container_name(name: str) -> str:
    normalized = name.lower().replace(".", "-")
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in normalized)


def _normalize_snapshot_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"replay", "snapshot_exact"}:
        raise ValueError(
            f"Unknown Harbor state capture mode: {mode!r}. "
            "Valid values: ['replay', 'snapshot_exact']"
        )
    return normalized


def _snapshot_relative_path(
    snapshot_artifact_root: Path,
    state: State[HarborHidden],
) -> Path:
    return (
        Path(snapshot_artifact_root.name)
        / _normalize_container_name(state.hidden.task_name)
        / state.metadata.episode_id
        / f"state_{state.metadata.step:04d}.tar"
    )


def _snapshot_runtime_name(harbor_env: Any) -> str:
    runtime = getattr(harbor_env, "snapshot_runtime", None)
    if isinstance(runtime, str) and runtime:
        return runtime
    raise RuntimeError(
        "Harbor exact snapshots require a runtime that exposes snapshot_runtime"
    )


def _capture_state_snapshot(
    harbor_env: Any,
    state: State[HarborHidden],
    *,
    state_capture_mode: str,
    snapshot_artifact_root: Path | None,
    snapshot_options: HarborSnapshotOptions,
) -> State[HarborHidden]:
    if state_capture_mode == "replay":
        return state
    if snapshot_artifact_root is None:
        raise ValueError(
            "snapshot_artifact_root is required when state_capture_mode='snapshot_exact'"
        )

    export_checkpoint = getattr(harbor_env, "export_checkpoint", None)
    if not callable(export_checkpoint):
        raise RuntimeError(
            "Harbor exact snapshots require a runtime with export_checkpoint() support"
        )

    relative_path = _snapshot_relative_path(snapshot_artifact_root, state)
    artifact_path = snapshot_artifact_root.parent / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    run_async(
        export_checkpoint(
            artifact_path,
            file_locks=snapshot_options.file_locks,
            tcp_established=snapshot_options.tcp_established,
            ignore_volumes=snapshot_options.ignore_volumes,
        )
    )
    return State(
        observation=state.observation,
        hidden=replace(
            state.hidden,
            snapshot_ref=HarborSnapshotRef(
                runtime=_snapshot_runtime_name(harbor_env),
                relative_path=str(relative_path),
                options=snapshot_options,
            ),
        ),
        metadata=state.metadata,
    )


def _restore_state_snapshot(
    harbor_env: Any,
    snapshot_ref: HarborSnapshotRef,
    *,
    artifact_root: Path | str,
) -> None:
    restore_checkpoint = getattr(harbor_env, "restore_checkpoint", None)
    if not callable(restore_checkpoint):
        raise RuntimeError(
            "Harbor exact snapshots require a runtime with restore_checkpoint() support"
        )

    runtime_name = _snapshot_runtime_name(harbor_env)
    if snapshot_ref.runtime != runtime_name:
        raise ValueError(
            f"Snapshot runtime mismatch: dataset requires {snapshot_ref.runtime!r}, "
            f"but environment runtime is {runtime_name!r}"
        )

    import_path = Path(artifact_root) / snapshot_ref.relative_path
    if not import_path.exists():
        raise FileNotFoundError(f"Snapshot artifact not found: {import_path}")

    run_async(
        restore_checkpoint(
            import_path,
            file_locks=snapshot_ref.options.file_locks,
            tcp_established=snapshot_ref.options.tcp_established,
            tcp_close=snapshot_ref.options.tcp_close,
            ignore_volumes=snapshot_ref.options.ignore_volumes,
        )
    )


def _parse_probe_output(
    stdout: str, *, has_pid_namespace: bool
) -> RuntimeProbeSnapshot:
    """Parse the combined probe script output into a RuntimeProbeSnapshot."""
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("===") and line.endswith("==="):
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines)
            current_section = line.strip("=")
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines)

    # Processes
    process_commands: frozenset[str] = frozenset()
    if has_pid_namespace:
        procs_text = sections.get("PROCS", "UNAVAILABLE").strip()
        if procs_text and procs_text != "UNAVAILABLE":
            process_commands = frozenset(
                line.strip() for line in procs_text.splitlines() if line.strip()
            )

    # Mounts
    mounts_text = sections.get("MOUNTS", "UNAVAILABLE").strip()
    if mounts_text and mounts_text != "UNAVAILABLE":
        mount_fingerprint = mounts_text.split()[0] if mounts_text else ""
    else:
        mount_fingerprint = ""

    # Sockets
    listening_ports: frozenset[int] = frozenset()
    sockets_text = sections.get("SOCKETS", "UNAVAILABLE").strip()
    if sockets_text and sockets_text != "UNAVAILABLE":
        ports: set[int] = set()
        for line in sockets_text.splitlines():
            # ss output: proto state recv-q send-q local:port peer:port ...
            parts = line.split()
            for part in parts:
                if ":" in part:
                    port_str = part.rsplit(":", 1)[-1]
                    if port_str.isdigit():
                        ports.add(int(port_str))
                        break  # take first port match per line
        listening_ports = frozenset(ports)

    # Staging
    staging_text = sections.get("STAGING", "UNAVAILABLE").strip()
    staging_has_content = bool(
        staging_text and staging_text != "UNAVAILABLE"
    )

    return RuntimeProbeSnapshot(
        process_commands=process_commands,
        mount_fingerprint=mount_fingerprint,
        listening_ports=listening_ports,
        staging_has_content=staging_has_content,
    )


def _probe_and_annotate_state(
    harbor_env: Any,
    state: State[HarborHidden],
    *,
    runtime_probing: bool,
) -> State[HarborHidden]:
    """Capture runtime probe and annotate state with risk signals."""
    if not runtime_probing:
        return state
    capture_fn = getattr(harbor_env, "capture_runtime_probe", None)
    if not callable(capture_fn):
        return state
    probe = run_async(capture_fn())
    if harbor_env._probe_baseline is None:
        harbor_env._probe_baseline = probe
        if probe.probe_failed:
            return State(
                observation=state.observation,
                hidden=replace(
                    state.hidden,
                    fs_restore_risk_now=True,
                    fs_restore_risk_reasons=("baseline_probe_degraded",),
                ),
                metadata=state.metadata,
            )
        return state
    risk_now, reasons = harbor_env.detect_runtime_risk(probe)
    return State(
        observation=state.observation,
        hidden=replace(
            state.hidden,
            fs_restore_risk_now=risk_now,
            fs_restore_risk_reasons=reasons,
        ),
        metadata=state.metadata,
    )


_COMPOSE_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
_PODMAN_UNSUPPORTED_COMPOSE_SERVICE_KEYS = frozenset(
    {"ports", "networks", "secrets", "configs", "profiles", "devices"}
)


def _parse_compose_duration(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if text.isdigit():
        return float(text)

    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text):
        matched = True
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
        total += float(amount) * scale
    if matched:
        return total

    raise ValueError(f"Unsupported compose duration: {value!r}")


def _topological_service_order(services: dict[str, _PodmanServiceSpec]) -> list[str]:
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Cyclic compose dependency involving {name!r}")
        visiting.add(name)
        spec = services[name]
        for dep in spec.depends_on:
            if dep not in services:
                raise ValueError(f"Compose service {name!r} depends on unknown service {dep!r}")
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for service_name in services:
        visit(service_name)
    return order


def _compose_shell_command(
    value: str | tuple[str, ...] | None,
    *,
    entrypoint_present: bool = False,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return list(value)
    if entrypoint_present:
        return [value]
    return ["sh", "-lc", value]


def _normalize_compose_environment(raw_env: Any) -> tuple[tuple[str, str], ...]:
    if raw_env is None:
        return ()
    if isinstance(raw_env, dict):
        return tuple(
            (str(key), "" if value is None else str(value))
            for key, value in raw_env.items()
        )
    if isinstance(raw_env, list):
        pairs: list[tuple[str, str]] = []
        for item in raw_env:
            if not isinstance(item, str):
                raise NotImplementedError("Compose environment list entries must be strings")
            key, sep, value = item.partition("=")
            pairs.append((key, value if sep else ""))
        return tuple(pairs)
    raise NotImplementedError("Unsupported compose environment format")


def _validate_compose_volume_entry(raw_volume: Any) -> None:
    if isinstance(raw_volume, str):
        parts = raw_volume.split(":")
        if len(parts) < 2:
            raise NotImplementedError("Compose volume entries must include source and target")
        return

    if isinstance(raw_volume, dict):
        volume_type = raw_volume.get("type", "volume")
        if volume_type not in {"bind", "volume"}:
            raise NotImplementedError(
                f"Unsupported compose volume type: {volume_type!r}"
            )
        source = raw_volume.get("source")
        target = raw_volume.get("target")
        if not source or not target:
            raise NotImplementedError("Compose volume mappings require source and target")
        return

    raise NotImplementedError("Unsupported compose volume format")


def _validate_compose_healthcheck(raw_healthcheck: Any) -> None:
    if raw_healthcheck in (None, False):
        return
    if not isinstance(raw_healthcheck, dict):
        raise NotImplementedError("Unsupported compose healthcheck format")
    if raw_healthcheck.get("disable") is True:
        return
    test = raw_healthcheck.get("test")
    if not isinstance(test, (list, str, type(None))):
        raise NotImplementedError("Unsupported compose healthcheck.test format")
    _parse_compose_duration(raw_healthcheck.get("interval"), 1.0)
    _parse_compose_duration(raw_healthcheck.get("timeout"), 30.0)
    _parse_compose_duration(raw_healthcheck.get("start_period"), 0.0)


def _analyze_podman_snapshot_definition(
    *,
    task_index: int,
    task_name: str,
    environment_dir: Path,
    task_env_config: Any,
) -> HarborSnapshotEligibility:
    dockerfile_path = environment_dir / "Dockerfile"
    compose_path = environment_dir / "docker-compose.yaml"

    def ineligible(code: str, detail: str) -> HarborSnapshotEligibility:
        return HarborSnapshotEligibility(
            task_index=task_index,
            task_name=task_name,
            eligible=False,
            reason_code=code,
            reason_detail=detail,
        )

    if not compose_path.exists():
        docker_image = getattr(task_env_config, "docker_image", None)
        if not dockerfile_path.exists() and not docker_image:
            return ineligible(
                "missing_container_source",
                "Task defines neither environment/Dockerfile nor task_env_config.docker_image.",
            )
        return HarborSnapshotEligibility(task_index=task_index, task_name=task_name, eligible=True)

    try:
        data = yaml.safe_load(compose_path.read_text()) or {}
    except Exception as exc:
        return ineligible("invalid_compose_yaml", str(exc))

    if not isinstance(data, dict):
        return ineligible("invalid_compose_yaml", "docker-compose.yaml must define a mapping")

    top_level_networks = data.get("networks")
    if top_level_networks:
        return ineligible(
            "unsupported_compose_networks",
            "Compose networks are not supported by podman-hpc runtime.",
        )

    top_level_volumes = data.get("volumes", {})
    if not isinstance(top_level_volumes, dict):
        return ineligible(
            "invalid_compose_volumes",
            "Top-level compose volumes must be a mapping.",
        )
    for name, cfg in top_level_volumes.items():
        if not cfg:
            continue
        if not isinstance(cfg, dict):
            return ineligible(
                "invalid_compose_volumes",
                "Unsupported top-level compose volume configuration.",
            )
        if cfg.get("external"):
            return ineligible(
                "unsupported_external_volume",
                f"External compose volume {name!r} is not supported.",
            )

    raw_services = data.get("services")
    if not isinstance(raw_services, dict) or not raw_services:
        return ineligible(
            "invalid_compose_services",
            "docker-compose.yaml must define at least one service.",
        )

    if "main" not in raw_services:
        return ineligible(
            "missing_main_service",
            "Compose environments must define a 'main' service.",
        )

    if len(raw_services) != 1:
        return ineligible(
            "multi_service_compose",
            f"Exact Harbor snapshots currently support only one compose service; found {len(raw_services)}.",
        )

    for name, raw_service in raw_services.items():
        if not isinstance(raw_service, dict):
            return ineligible(
                "invalid_compose_service",
                f"Compose service {name!r} must be a mapping.",
            )

        present_unsupported = _PODMAN_UNSUPPORTED_COMPOSE_SERVICE_KEYS.intersection(
            raw_service
        )
        if present_unsupported:
            return ineligible(
                "unsupported_compose_service_fields",
                f"Unsupported compose fields for service {name!r}: "
                + ", ".join(sorted(present_unsupported)),
            )

        build = raw_service.get("build")
        if isinstance(build, dict):
            unsupported_build_keys = set(build).difference({"context", "dockerfile"})
            if unsupported_build_keys:
                return ineligible(
                    "unsupported_compose_build_fields",
                    f"Unsupported compose build fields for service {name!r}: "
                    + ", ".join(sorted(unsupported_build_keys)),
                )
        elif build is not None and not isinstance(build, str):
            return ineligible(
                "invalid_compose_build",
                "Unsupported compose build format.",
            )

        image = raw_service.get("image")
        if build is None and image is None:
            return ineligible(
                "missing_build_or_image",
                f"Compose service {name!r} must define either image or build.",
            )

        depends_on_raw = raw_service.get("depends_on", ())
        if not isinstance(depends_on_raw, (dict, list, tuple, type(None))):
            return ineligible(
                "invalid_compose_depends_on",
                "Unsupported compose depends_on format.",
            )

        try:
            _normalize_compose_environment(raw_service.get("environment"))
            for volume in raw_service.get("volumes", ()):
                _validate_compose_volume_entry(volume)
            _validate_compose_healthcheck(raw_service.get("healthcheck"))
        except (NotImplementedError, ValueError) as exc:
            return ineligible("invalid_compose_runtime_shape", str(exc))

    return HarborSnapshotEligibility(task_index=task_index, task_name=task_name, eligible=True)


class PodmanHPCEnvironment:
    """Local Harbor-compatible runtime using ``podman-hpc``.

    This is a Harbor-facing environment object with the methods the llenvs
    Harbor adapter and Harbor verifiers rely on. It supports Harbor's default
    single-container tasks and a constrained task-local ``docker-compose.yaml``
    subset centered on a ``main`` service plus sidecars.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        logger: logging.Logger | None = None,
        *,
        podman_command: str = "podman-hpc",
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.environment_dir = Path(environment_dir)
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.logger = logger or logging.getLogger(__name__)
        self._podman = podman_command
        self.snapshot_runtime = "podman-hpc"
        self._container_name = _normalize_container_name(session_id)
        self._image_name = f"hb__{_normalize_container_name(environment_name)}"
        self._started = False
        self.is_mounted = False

        self._dockerfile_path = self.environment_dir / "Dockerfile"
        self._compose_path = self.environment_dir / "docker-compose.yaml"
        self._network_name = f"{self._container_name}-net"
        self._volume_root = Path(self.trial_paths.trial_dir) / "compose-volumes"
        self._compose_services: dict[str, _PodmanServiceSpec] = {}
        self._service_order: tuple[str, ...] = ()
        self._service_container_names: dict[str, str] = {}
        self._main_container_name = self._container_name
        self._validate_definition()

    def _validate_definition(self) -> None:
        if self._compose_path.exists():
            self._compose_services = self._parse_compose_definition()
            if "main" not in self._compose_services:
                raise ValueError("Compose environments must define a 'main' service")
            self._service_order = tuple(_topological_service_order(self._compose_services))
            self._service_container_names = {
                name: _normalize_container_name(f"{self.session_id}-{name}")
                for name in self._compose_services
            }
            self._main_container_name = self._service_container_names["main"]
            return

        docker_image = getattr(self.task_env_config, "docker_image", None)
        if not self._dockerfile_path.exists() and not docker_image:
            raise FileNotFoundError(
                f"{self._dockerfile_path} not found and task_env_config.docker_image is unset."
            )

    async def _run_podman_command(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> _CLIResult:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.environment_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        try:
            if timeout_sec is not None:
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            else:
                stdout_b, stderr_b = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                stdout_b, stderr_b = await process.communicate()
            raise RuntimeError(f"podman-hpc command timed out after {timeout_sec}s")

        result = _CLIResult(
            stdout=stdout_b.decode(errors="replace").strip() if stdout_b else "",
            stderr=stderr_b.decode(errors="replace").strip() if stderr_b else "",
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"podman-hpc command failed (exit {result.return_code}): "
                f"{' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    def _docker_image_source(self) -> str:
        image = getattr(self.task_env_config, "docker_image", None)
        if image is None:
            raise ValueError("docker_image is required for migrate-based startup")
        return image if "://" in image else f"docker://{image}"

    def _compose_image_source(self, image: str) -> str:
        return image if "://" in image else f"docker://{image}"

    def _runtime_env_vars(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "MAIN_IMAGE_NAME": self._image_name,
                "CONTEXT_DIR": str(self.environment_dir.resolve()),
                "TEST_DIR": "/tests",
                "HOST_VERIFIER_LOGS_PATH": str(Path(self.trial_paths.verifier_dir).resolve()),
                "HOST_AGENT_LOGS_PATH": str(Path(self.trial_paths.agent_dir).resolve()),
                "ENV_VERIFIER_LOGS_PATH": "/logs/verifier",
                "ENV_AGENT_LOGS_PATH": "/logs/agent",
                "CPUS": str(getattr(self.task_env_config, "cpus", 1)),
                "MEMORY": f"{getattr(self.task_env_config, 'memory_mb', 1024)}M",
            }
        )
        docker_image = getattr(self.task_env_config, "docker_image", None)
        if docker_image is not None:
            env["PREBUILT_IMAGE_NAME"] = str(docker_image)
        return env

    def _interpolate_compose_value(self, value: Any) -> Any:
        if isinstance(value, str):
            def repl(match: re.Match[str]) -> str:
                name = match.group(1)
                default = match.group(2)
                return self._runtime_env_vars().get(name, default or "")

            return _COMPOSE_VAR_PATTERN.sub(repl, value)
        if isinstance(value, list):
            return [self._interpolate_compose_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._interpolate_compose_value(item)
                for key, item in value.items()
            }
        return value

    def _normalize_environment(self, raw_env: Any) -> tuple[tuple[str, str], ...]:
        if raw_env is None:
            return ()
        if isinstance(raw_env, dict):
            return tuple(
                (str(key), "" if value is None else str(value))
                for key, value in raw_env.items()
            )
        if isinstance(raw_env, list):
            pairs: list[tuple[str, str]] = []
            for item in raw_env:
                if not isinstance(item, str):
                    raise NotImplementedError("Compose environment list entries must be strings")
                key, sep, value = item.partition("=")
                pairs.append((key, value if sep else ""))
            return tuple(pairs)
        raise NotImplementedError("Unsupported compose environment format")

    def _resolve_volume_source(self, source: str, *, named: bool) -> str:
        if named:
            host_path = self._volume_root / source
            host_path.mkdir(parents=True, exist_ok=True)
            return str(host_path)
        host_path = Path(source)
        if not host_path.is_absolute():
            host_path = (self.environment_dir / host_path).resolve()
        return str(host_path)

    def _parse_volume_mount(self, raw_volume: Any) -> _PodmanVolumeMount:
        if isinstance(raw_volume, str):
            parts = raw_volume.split(":")
            if len(parts) < 2:
                raise NotImplementedError("Compose volume entries must include source and target")
            source, target = parts[0], parts[1]
            mode = parts[2] if len(parts) > 2 else ""
            read_only = "ro" in mode.split(",")
            named = not source.startswith(("/", ".", "~"))
            return _PodmanVolumeMount(
                source=self._resolve_volume_source(source, named=named),
                target=target,
                read_only=read_only,
            )

        if isinstance(raw_volume, dict):
            volume_type = raw_volume.get("type", "volume")
            if volume_type not in {"bind", "volume"}:
                raise NotImplementedError(
                    f"Unsupported compose volume type: {volume_type!r}"
                )
            source = raw_volume.get("source")
            target = raw_volume.get("target")
            if not source or not target:
                raise NotImplementedError("Compose volume mappings require source and target")
            return _PodmanVolumeMount(
                source=self._resolve_volume_source(
                    str(source),
                    named=volume_type == "volume",
                ),
                target=str(target),
                read_only=bool(raw_volume.get("read_only", False)),
            )

        raise NotImplementedError("Unsupported compose volume format")

    def _parse_healthcheck(self, raw_healthcheck: Any) -> _PodmanHealthcheck | None:
        if raw_healthcheck in (None, False):
            return None
        if not isinstance(raw_healthcheck, dict):
            raise NotImplementedError("Unsupported compose healthcheck format")
        if raw_healthcheck.get("disable") is True:
            return None
        test = raw_healthcheck.get("test")
        if isinstance(test, list):
            normalized_test: str | tuple[str, ...] | None = tuple(str(part) for part in test)
        elif test is None or isinstance(test, str):
            normalized_test = test
        else:
            raise NotImplementedError("Unsupported compose healthcheck.test format")
        return _PodmanHealthcheck(
            test=normalized_test,
            interval_sec=_parse_compose_duration(raw_healthcheck.get("interval"), 1.0),
            timeout_sec=_parse_compose_duration(raw_healthcheck.get("timeout"), 30.0),
            retries=int(raw_healthcheck.get("retries", 30)),
            start_period_sec=_parse_compose_duration(raw_healthcheck.get("start_period"), 0.0),
        )

    def _parse_compose_definition(self) -> dict[str, _PodmanServiceSpec]:
        data = yaml.safe_load(self._compose_path.read_text()) or {}
        data = self._interpolate_compose_value(data)
        if not isinstance(data, dict):
            raise ValueError("docker-compose.yaml must define a mapping")

        top_level_networks = data.get("networks")
        if top_level_networks:
            raise NotImplementedError("Compose networks are not supported by podman-hpc runtime")

        top_level_volumes = data.get("volumes", {})
        if not isinstance(top_level_volumes, dict):
            raise ValueError("Top-level compose volumes must be a mapping")
        for name, cfg in top_level_volumes.items():
            if not cfg:
                continue
            if not isinstance(cfg, dict):
                raise NotImplementedError("Unsupported top-level compose volume configuration")
            if cfg.get("external"):
                raise NotImplementedError(
                    f"External compose volume {name!r} is not supported"
                )

        raw_services = data.get("services")
        if not isinstance(raw_services, dict) or not raw_services:
            raise ValueError("docker-compose.yaml must define at least one service")

        services: dict[str, _PodmanServiceSpec] = {}
        for name, raw_service in raw_services.items():
            if not isinstance(raw_service, dict):
                raise ValueError(f"Compose service {name!r} must be a mapping")
            present_unsupported = _PODMAN_UNSUPPORTED_COMPOSE_SERVICE_KEYS.intersection(
                raw_service
            )
            if present_unsupported:
                raise NotImplementedError(
                    f"Unsupported compose fields for service {name!r}: "
                    + ", ".join(sorted(present_unsupported))
                )

            build_context: Path | None = None
            dockerfile: Path | None = None
            build = raw_service.get("build")
            if isinstance(build, str):
                build_context = (self.environment_dir / build).resolve()
            elif isinstance(build, dict):
                unsupported_build_keys = set(build).difference({"context", "dockerfile"})
                if unsupported_build_keys:
                    raise NotImplementedError(
                        f"Unsupported compose build fields for service {name!r}: "
                        + ", ".join(sorted(unsupported_build_keys))
                    )
                build_context = (self.environment_dir / build.get("context", ".")).resolve()
                dockerfile_name = build.get("dockerfile")
                if dockerfile_name is not None:
                    dockerfile = (build_context / str(dockerfile_name)).resolve()
            elif build is not None:
                raise NotImplementedError("Unsupported compose build format")

            image = raw_service.get("image")
            if build_context is None and image is None:
                raise ValueError(
                    f"Compose service {name!r} must define either image or build"
                )

            command = raw_service.get("command")
            if isinstance(command, list):
                normalized_command: str | tuple[str, ...] | None = tuple(
                    str(part) for part in command
                )
            else:
                normalized_command = None if command is None else str(command)

            entrypoint = raw_service.get("entrypoint")
            if isinstance(entrypoint, list):
                normalized_entrypoint: str | tuple[str, ...] | None = tuple(
                    str(part) for part in entrypoint
                )
            else:
                normalized_entrypoint = None if entrypoint is None else str(entrypoint)

            depends_on_raw = raw_service.get("depends_on", ())
            if isinstance(depends_on_raw, dict):
                depends_on = tuple(str(dep_name) for dep_name in depends_on_raw)
            elif isinstance(depends_on_raw, list):
                depends_on = tuple(str(dep_name) for dep_name in depends_on_raw)
            elif depends_on_raw in (None, ()):
                depends_on = ()
            else:
                raise NotImplementedError("Unsupported compose depends_on format")

            services[name] = _PodmanServiceSpec(
                name=name,
                image=None if image is None else str(image),
                build_context=build_context,
                dockerfile=dockerfile,
                command=normalized_command,
                entrypoint=normalized_entrypoint,
                environment=self._normalize_environment(raw_service.get("environment")),
                working_dir=(
                    None
                    if raw_service.get("working_dir") is None
                    else str(raw_service.get("working_dir"))
                ),
                volumes=tuple(
                    self._parse_volume_mount(volume)
                    for volume in raw_service.get("volumes", ())
                ),
                depends_on=depends_on,
                healthcheck=self._parse_healthcheck(raw_service.get("healthcheck")),
            )
        return services

    def _service_image_name(self, service_name: str) -> str:
        return f"{self._image_name}__{_normalize_container_name(service_name)}"

    async def _prepare_service_image(
        self,
        service: _PodmanServiceSpec,
        *,
        force_build: bool,
    ) -> str:
        if service.build_context is not None:
            build_cmd = [
                self._podman,
                "build",
                "-t",
                self._service_image_name(service.name),
            ]
            if service.dockerfile is not None:
                build_cmd.extend(["-f", str(service.dockerfile)])
            build_cmd.append(str(service.build_context))
            await self._run_podman_command(build_cmd)
            return self._service_image_name(service.name)

        if service.image is None:
            raise ValueError(f"Compose service {service.name!r} has no runnable image")
        if not force_build:
            await self._run_podman_command(
                [self._podman, "migrate", self._compose_image_source(service.image)],
            )
        return service.image

    def _build_service_run_command(
        self,
        service: _PodmanServiceSpec,
        image_ref: str,
    ) -> list[str]:
        cmd = [
            self._podman,
            "run",
            "-d",
            "--name",
            self._service_container_names[service.name],
            "--network",
            self._network_name,
            "--network-alias",
            service.name,
            "--cpus",
            str(getattr(self.task_env_config, "cpus", 1)),
            "--memory",
            f"{getattr(self.task_env_config, 'memory_mb', 1024)}M",
        ]
        if service.working_dir:
            cmd.extend(["-w", service.working_dir])
        for key, value in service.environment:
            cmd.extend(["-e", f"{key}={value}"])
        for volume in service.volumes:
            suffix = ":ro" if volume.read_only else ""
            cmd.extend(["-v", f"{volume.source}:{volume.target}{suffix}"])
        if service.entrypoint is not None:
            entrypoint_value = (
                json.dumps(list(service.entrypoint))
                if isinstance(service.entrypoint, tuple)
                else service.entrypoint
            )
            cmd.extend(["--entrypoint", entrypoint_value])
        cmd.append(image_ref)
        cmd.extend(
            _compose_shell_command(
                service.command,
                entrypoint_present=service.entrypoint is not None,
            )
        )
        return cmd

    def _healthcheck_command(self, service_name: str) -> str | None:
        healthcheck = self._compose_services[service_name].healthcheck
        if healthcheck is None or healthcheck.test is None:
            return None
        if isinstance(healthcheck.test, str):
            return healthcheck.test
        if not healthcheck.test:
            return None
        head, *tail = healthcheck.test
        upper_head = head.upper()
        if upper_head == "NONE":
            return None
        if upper_head == "CMD":
            return shlex.join(tail)
        if upper_head == "CMD-SHELL":
            return " ".join(tail)
        return shlex.join(list(healthcheck.test))

    async def _wait_for_service_health(self, service_name: str) -> None:
        healthcheck = self._compose_services[service_name].healthcheck
        command = self._healthcheck_command(service_name)
        if healthcheck is None or command is None:
            return
        if healthcheck.start_period_sec > 0:
            await asyncio.sleep(healthcheck.start_period_sec)
        attempts = max(1, healthcheck.retries)
        for attempt in range(attempts):
            result = await self.exec_service(
                service_name,
                command,
                timeout_sec=max(1, int(healthcheck.timeout_sec)),
            )
            if result.return_code == 0:
                return
            if attempt < attempts - 1:
                await asyncio.sleep(healthcheck.interval_sec)
        raise RuntimeError(
            f"Compose service {service_name!r} failed healthcheck after {attempts} attempts"
        )

    async def exec_service(
        self,
        service_name: str,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> _CLIResult:
        if service_name not in self._service_container_names:
            raise ValueError(f"Unknown compose service: {service_name}")
        cmd = [self._podman, "exec"]
        if cwd:
            cmd.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._service_container_names[service_name], "bash", "-lc", command])
        return await self._run_podman_command(cmd, check=False, timeout_sec=timeout_sec)

    async def _bootstrap_runtime_dirs(self) -> None:
        if self._compose_services:
            await self.exec_service("main", "mkdir -p /logs/agent /logs/verifier")
            return
        await self._run_podman_command(
            [
                self._podman,
                "exec",
                self._main_container_name,
                "bash",
                "-lc",
                "mkdir -p /logs/agent /logs/verifier",
            ],
            check=False,
        )

    async def _start_compose(self, force_build: bool) -> None:
        network_cmd = [self._podman, "network", "create"]
        if getattr(self.task_env_config, "allow_internet", True) is False:
            network_cmd.append("--internal")
        network_cmd.append(self._network_name)
        await self._run_podman_command(network_cmd)

        image_refs: dict[str, str] = {}
        for service_name in self._service_order:
            image_refs[service_name] = await self._prepare_service_image(
                self._compose_services[service_name],
                force_build=force_build,
            )

        for service_name in self._service_order:
            await self._run_podman_command(
                self._build_service_run_command(
                    self._compose_services[service_name],
                    image_refs[service_name],
                )
            )
            if self._compose_services[service_name].healthcheck is not None:
                await self._wait_for_service_health(service_name)

        await self._bootstrap_runtime_dirs()
        self._started = True

    async def start(self, force_build: bool = False) -> None:
        if self._compose_services:
            await self._start_compose(force_build=force_build)
            return

        docker_image = getattr(self.task_env_config, "docker_image", None)
        if docker_image and not force_build:
            await self._run_podman_command(
                [self._podman, "migrate", self._docker_image_source()],
            )
            image_ref = docker_image
        else:
            await self._run_podman_command(
                [self._podman, "build", "-t", self._image_name, str(self.environment_dir)],
            )
            image_ref = self._image_name

        run_cmd = [
            self._podman,
            "run",
            "-d",
            "--name",
            self._container_name,
        ]
        if getattr(self.task_env_config, "allow_internet", True) is False:
            run_cmd.extend(["--network", "none"])
        run_cmd.extend([image_ref, "bash", "-lc", "while true; do sleep 3600; done"])
        await self._run_podman_command(run_cmd)
        await self._bootstrap_runtime_dirs()
        self._started = True

    async def stop(self, delete: bool = True) -> None:
        if not self._started:
            return
        try:
            if self._compose_services:
                verb = ["rm", "-f"] if delete else ["stop"]
                for service_name in reversed(self._service_order):
                    await self._run_podman_command(
                        [
                            self._podman,
                            *verb,
                            self._service_container_names[service_name],
                        ],
                        check=False,
                    )
                if delete:
                    await self._run_podman_command(
                        [self._podman, "network", "rm", self._network_name],
                        check=False,
                    )
            else:
                cmd = [self._podman, "rm", "-f", self._container_name]
                if not delete:
                    cmd = [self._podman, "stop", self._container_name]
                await self._run_podman_command(cmd, check=False)
        finally:
            self._started = False

    async def export_checkpoint(
        self,
        export_path: Path | str,
        *,
        file_locks: bool = False,
        tcp_established: bool = False,
        ignore_volumes: bool = False,
    ) -> None:
        if self._compose_services:
            raise NotImplementedError(
                "Exact Harbor snapshots are not supported for compose-backed podman-hpc tasks"
            )
        if not self._started:
            raise RuntimeError("podman-hpc environment has not been started")

        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._podman,
            "container",
            "checkpoint",
            "--export",
            str(export_path),
            "--compress",
            "none",
            "--leave-running",
        ]
        if file_locks:
            cmd.append("--file-locks")
        if tcp_established:
            cmd.append("--tcp-established")
        if ignore_volumes:
            cmd.append("--ignore-volumes")
        cmd.append(self._container_name)
        await self._run_podman_command(cmd)

    async def restore_checkpoint(
        self,
        import_path: Path | str,
        *,
        file_locks: bool = False,
        tcp_established: bool = False,
        tcp_close: bool = False,
        ignore_volumes: bool = False,
    ) -> None:
        if self._compose_services:
            raise NotImplementedError(
                "Exact Harbor snapshots are not supported for compose-backed podman-hpc tasks"
            )

        if self._started:
            await self.stop(delete=True)

        cmd = [
            self._podman,
            "container",
            "restore",
            "--import",
            str(import_path),
            "--name",
            self._container_name,
            "--keep",
        ]
        if file_locks:
            cmd.append("--file-locks")
        if tcp_established:
            cmd.append("--tcp-established")
        elif tcp_close:
            cmd.append("--tcp-close")
        if ignore_volumes:
            cmd.append("--ignore-volumes")
        await self._run_podman_command(cmd)
        self._started = True

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> _CLIResult:
        if not self._started:
            raise RuntimeError("podman-hpc environment has not been started")

        cmd = [self._podman, "exec"]
        if user is not None:
            cmd.extend(["--user", str(user)])
        if cwd:
            cmd.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._main_container_name, "bash", "-lc", command])
        return await self._run_podman_command(cmd, check=False, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", str(source_path), f"{self._main_container_name}:{target_path}"]
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", str(source_dir), f"{self._main_container_name}:{target_dir}"]
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", f"{self._main_container_name}:{source_path}", str(target_path)]
        )

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", f"{self._main_container_name}:{source_dir}", str(target_dir)]
        )


# ── Apptainer/Singularity HPC runtime ──────────────────────────


_APPTAINER_RUNTIME_NAME = "apptainer-hpc"
_APPTAINER_ALIASES = frozenset({"apptainer-hpc", "singularity-hpc"})

_SIF_MANIFEST_FILENAME = "manifest.json"


def _sif_cache_key(image_ref: str) -> str:
    """Hash-based SIF filename from a Docker/OCI image reference."""
    return hashlib.sha256(image_ref.encode()).hexdigest()[:16]


def _load_sif_manifest(cache_dir: Path) -> dict[str, str]:
    manifest_path = cache_dir / _SIF_MANIFEST_FILENAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def _save_sif_manifest(cache_dir: Path, manifest: dict[str, str]) -> None:
    manifest_path = cache_dir / _SIF_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


class ApptainerHPCEnvironment:
    """Local Harbor-compatible runtime using Apptainer/Singularity.

    Single-container tasks only. Supports an overlay fast path plus a
    writable-sandbox fallback, and uses ``--cleanenv --contain --no-home``
    for host isolation. File transfer uses a bind-mounted staging
    directory since Apptainer has no ``cp`` subcommand.

    Compose-backed tasks are rejected at construction time.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        logger: logging.Logger | None = None,
        *,
        apptainer_command: str = "apptainer",
        sif_cache_dir: str | None = None,
        fakeroot: bool = False,
        overlay_size_mb: int = 512,
        writable_tmpfs: bool = False,
        rootfs_mode: str = "auto",
        pid_namespace: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.environment_dir = Path(environment_dir)
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.logger = logger or logging.getLogger(__name__)

        self._apptainer = apptainer_command
        self._fakeroot = fakeroot
        self._pid_namespace = pid_namespace
        self._pid_flag: str | None = None  # resolved in start() → _probe_pid_support()
        self._probe_baseline: RuntimeProbeSnapshot | None = None
        self._overlay_size_mb = overlay_size_mb
        self._writable_tmpfs = writable_tmpfs
        normalized_rootfs_mode = rootfs_mode.strip().lower()
        if normalized_rootfs_mode not in {"auto", "overlay", "sandbox"}:
            raise ValueError(
                "rootfs_mode must be one of {'auto', 'overlay', 'sandbox'}, "
                f"got {rootfs_mode!r}"
            )
        self._rootfs_mode = normalized_rootfs_mode
        self.snapshot_runtime = _APPTAINER_RUNTIME_NAME
        self._instance_name = _normalize_container_name(session_id)
        self._started = False
        self.is_mounted = True  # log dirs bind-mounted
        self._active_rootfs_mode: str | None = None

        self._sif_cache_dir = (
            Path(sif_cache_dir).resolve()
            if sif_cache_dir is not None
            else Path.home() / ".cache" / "llenvs" / "sif"
        )
        self._trial_dir = Path(self.trial_paths.trial_dir).resolve()
        self._staging_dir = self._trial_dir / "staging"
        self._binds_dir = self._trial_dir / "binds"
        self._app_bind_dir = self._binds_dir / "app"
        self._tests_bind_dir = self._binds_dir / "tests"
        self._overlay_path = self._trial_dir / "overlay.img"
        self._sandbox_rootfs_dir = self._trial_dir / "rootfs"
        self._host_tmp_dir = self._trial_dir / "tmp"
        self._host_var_tmp_dir = self._trial_dir / "var_tmp"
        self._dockerfile_path = self.environment_dir / "Dockerfile"
        cache_root_base = (
            Path(os.environ["TMPDIR"]).resolve()
            if "TMPDIR" in os.environ
            else self._sif_cache_dir.parent
        )
        self._app_seed_cache_dir = (
            cache_root_base / "llenvs" / "apptainer-app-seeds"
        )
        self._sandbox_seed_cache_dir = (
            cache_root_base / "llenvs" / "apptainer-sandboxes"
        )

        self._validate_definition()
        self._sif_path = self._resolve_sif_path()
        self._default_cwd = self._resolve_default_cwd()

    def _app_seed_cache_key(self) -> str:
        try:
            stat = self._sif_path.stat()
            material = (
                f"{self.environment_name}:{self._sif_path}:"
                f"{stat.st_size}:{stat.st_mtime_ns}"
            )
        except FileNotFoundError:
            material = f"{self.environment_name}:{self._sif_path}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def _prepare_app_seed_dir(self) -> Path:
        self._app_seed_cache_dir.mkdir(parents=True, exist_ok=True)
        seed_dir = self._app_seed_cache_dir / self._app_seed_cache_key()
        lock_path = self._app_seed_cache_dir / f"{seed_dir.name}.lock"
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if seed_dir.exists():
                return seed_dir

            tmp_seed_dir = self._app_seed_cache_dir / f".{seed_dir.name}.{uuid.uuid4().hex}.tmp"
            tmp_seed_dir.mkdir(parents=True, exist_ok=False)
            try:
                self._extract_app_seed(tmp_seed_dir)
                os.replace(tmp_seed_dir, seed_dir)
            finally:
                if tmp_seed_dir.exists():
                    shutil.rmtree(tmp_seed_dir, ignore_errors=True)
        return seed_dir

    def _extract_app_seed(self, target_dir: Path) -> None:
        cmd = [
            self._apptainer,
            "exec",
            "--cleanenv",
            "--bind",
            f"{target_dir}:/seed",
            str(self._sif_path),
            "bash",
            "-lc",
            "if [ -d /app ]; then cp -a /app/. /seed/; fi",
        ]
        self.logger.debug("apptainer seed cmd: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(self.environment_dir),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"apptainer seed extraction failed (exit {result.returncode}): "
                f"{' '.join(cmd)}\nstdout: {result.stdout.rstrip()}\n"
                f"stderr: {result.stderr.rstrip()}"
            )

    @staticmethod
    def _copy_dir_contents(source_dir: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            dest = target_dir / item.name
            if item.is_symlink():
                if dest.exists() or dest.is_symlink():
                    if dest.is_dir() and not dest.is_symlink():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                os.symlink(os.readlink(item), dest)
            elif item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True, symlinks=True)
            else:
                shutil.copy2(item, dest)

    def _prepare_trial_bind_dirs(self) -> None:
        app_seed_dir = self._prepare_app_seed_dir()
        if self._binds_dir.exists():
            shutil.rmtree(self._binds_dir, ignore_errors=True)
        self._binds_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(app_seed_dir, self._app_bind_dir, symlinks=True)
        self._tests_bind_dir.mkdir(parents=True, exist_ok=True)

    def _sandbox_seed_cache_key(self) -> str:
        try:
            stat = self._sif_path.stat()
            material = (
                f"{self.environment_name}:{self._sif_path}:"
                f"{stat.st_size}:{stat.st_mtime_ns}"
            )
        except FileNotFoundError:
            material = f"{self.environment_name}:{self._sif_path}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def _build_sandbox_seed(self, target_dir: Path) -> None:
        cmd = [
            self._apptainer,
            "build",
            "--sandbox",
            str(target_dir),
            str(self._sif_path),
        ]
        self.logger.debug("apptainer sandbox seed cmd: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(self.environment_dir),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"apptainer sandbox build failed (exit {result.returncode}): "
                f"{' '.join(cmd)}\nstdout: {result.stdout.rstrip()}\n"
                f"stderr: {result.stderr.rstrip()}"
            )

    def _prepare_sandbox_seed_dir(self) -> Path:
        self._sandbox_seed_cache_dir.mkdir(parents=True, exist_ok=True)
        seed_dir = self._sandbox_seed_cache_dir / self._sandbox_seed_cache_key()
        lock_path = self._sandbox_seed_cache_dir / f"{seed_dir.name}.lock"
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if seed_dir.exists():
                return seed_dir

            tmp_seed_dir = self._sandbox_seed_cache_dir / f".{seed_dir.name}.{uuid.uuid4().hex}.tmp"
            try:
                self._build_sandbox_seed(tmp_seed_dir)
                os.replace(tmp_seed_dir, seed_dir)
            finally:
                if tmp_seed_dir.exists():
                    shutil.rmtree(tmp_seed_dir, ignore_errors=True)
        return seed_dir

    def _copy_tree_reflink(self, source_dir: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        source = f"{source_dir}/."

        def run_copy(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                cmd,
                cwd=str(self.environment_dir),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                check=False,
            )

        reflink_cmd = ["cp", "-a", "--reflink=auto", source, str(target_dir)]
        reflink_result = run_copy(reflink_cmd)
        if reflink_result.returncode == 0:
            return

        fallback_cmd = ["cp", "-a", source, str(target_dir)]
        fallback_result = run_copy(fallback_cmd)
        if fallback_result.returncode != 0:
            raise RuntimeError(
                f"apptainer sandbox clone failed (exit {fallback_result.returncode}): "
                f"{' '.join(fallback_cmd)}\n"
                f"stdout: {fallback_result.stdout.rstrip()}\n"
                f"stderr: {fallback_result.stderr.rstrip()}"
            )

    def _prepare_trial_rootfs(self) -> Path:
        seed_dir = self._prepare_sandbox_seed_dir()
        if self._sandbox_rootfs_dir.exists():
            shutil.rmtree(self._sandbox_rootfs_dir, ignore_errors=True)
        self._copy_tree_reflink(seed_dir, self._sandbox_rootfs_dir)
        return self._sandbox_rootfs_dir

    def _resolve_bind_target(self, container_path: str) -> Path | None:
        if self._active_rootfs_mode != "overlay":
            return None
        path = PurePosixPath(container_path)
        if not path.is_absolute():
            return None
        parts = path.parts[1:]
        if not parts:
            return None
        if parts[0] == "app":
            root = self._app_bind_dir
        elif parts[0] == "tests":
            root = self._tests_bind_dir
        else:
            return None
        rel_parts = parts[1:]
        if any(part in {"..", "."} for part in rel_parts):
            raise ValueError(f"Unsupported container path: {container_path!r}")
        return root.joinpath(*rel_parts)

    def _validate_definition(self) -> None:
        compose_path = self.environment_dir / "docker-compose.yaml"
        if compose_path.exists():
            raise NotImplementedError(
                "Compose-backed tasks are not supported by the Apptainer runtime. "
                f"Found docker-compose.yaml in {self.environment_dir}"
            )
        docker_image = getattr(self.task_env_config, "docker_image", None)
        if not self._dockerfile_path.exists() and not docker_image:
            raise FileNotFoundError(
                f"Task {self.environment_name!r} defines neither a Dockerfile "
                "nor task_env_config.docker_image."
            )

    def _resolve_default_cwd(self) -> str:
        workdir = "/app"
        if not self._dockerfile_path.exists():
            return workdir

        current = PurePosixPath("/")
        for raw_line in self._dockerfile_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if not upper.startswith("WORKDIR "):
                continue
            value = line.split(None, 1)[1].strip()
            if not value:
                continue
            if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
                value = value[1:-1]
            next_path = PurePosixPath(value)
            if not next_path.is_absolute():
                next_path = current / next_path
            current = next_path
            workdir = current.as_posix()
        return workdir

    def _resolve_sif_path(self) -> Path:
        docker_image = getattr(self.task_env_config, "docker_image", None)
        if docker_image:
            cache_key = _sif_cache_key(docker_image)
            return self._sif_cache_dir / f"{cache_key}.sif"
        # Dockerfile-only: look for a SIF keyed by environment name
        cache_key = _sif_cache_key(f"dockerfile://{self.environment_name}")
        return self._sif_cache_dir / f"{cache_key}.sif"

    def _rootfs_probe_cache_key(self) -> tuple[Any, ...]:
        try:
            stat = self._sif_path.stat()
            size = stat.st_size
            mtime_ns = stat.st_mtime_ns
        except FileNotFoundError:
            size = None
            mtime_ns = None
        return (
            str(self._sif_path),
            size,
            mtime_ns,
            self._fakeroot,
            self._overlay_size_mb,
            self._writable_tmpfs,
        )

    async def _get_runtime_version(self) -> str:
        with _APPTAINER_VERSION_CACHE_LOCK:
            cached = _APPTAINER_VERSION_CACHE.get(self._apptainer)
        if cached is not None:
            return cached

        version_str = "unknown"
        try:
            result = await self._run_apptainer_command(
                [self._apptainer, "--version"], check=False
            )
            if result.return_code == 0 and result.stdout:
                version_str = result.stdout.strip()
        except Exception:
            pass

        with _APPTAINER_VERSION_CACHE_LOCK:
            _APPTAINER_VERSION_CACHE.setdefault(self._apptainer, version_str)
            return _APPTAINER_VERSION_CACHE[self._apptainer]

    def _runtime_info_cache_key(self, version_str: str) -> tuple[Any, ...]:
        return (
            self._apptainer,
            version_str,
            self._fakeroot,
            self._rootfs_mode,
            self._overlay_size_mb,
            self._writable_tmpfs,
            str(self._sif_cache_dir),
        )

    def _pid_support_cache_key(self, version_str: str) -> tuple[str, str]:
        return (self._apptainer, version_str)

    def _finish_pid_probe(self, cache_key: tuple[str, str], value: str | None) -> None:
        with _APPTAINER_PID_FLAG_CACHE_LOCK:
            if value is not None:
                _APPTAINER_PID_FLAG_CACHE[cache_key] = value
            event = _APPTAINER_PID_FLAG_EVENTS.pop(cache_key, None)
        if event is not None:
            event.set()

    def _claim_pid_probe(self, cache_key: tuple[str, str]) -> tuple[str | None, bool]:
        while True:
            with _APPTAINER_PID_FLAG_CACHE_LOCK:
                cached = _APPTAINER_PID_FLAG_CACHE.get(cache_key)
                if cached is not None:
                    return cached, False
                event = _APPTAINER_PID_FLAG_EVENTS.get(cache_key)
                if event is None:
                    _APPTAINER_PID_FLAG_EVENTS[cache_key] = threading.Event()
                    return None, True

            event.wait()

    def _finish_overlay_probe(self, value: bool | None) -> None:
        key = self._rootfs_probe_cache_key()
        with _APPTAINER_ROOTFS_PROBE_CACHE_LOCK:
            if value is not None:
                _APPTAINER_ROOTFS_PROBE_CACHE[key] = value
            event = _APPTAINER_ROOTFS_PROBE_EVENTS.pop(key, None)
        if event is not None:
            event.set()

    def _claim_overlay_probe(self) -> tuple[bool | None, bool]:
        """Return cached probe result or claim responsibility for probing.

        Returns ``(cached_result, is_probe_owner)``. When another thread is
        already probing, this method waits until that probe finishes and then
        retries. If the probing thread fails before producing a result, a
        waiting caller will claim probe ownership and continue.
        """
        key = self._rootfs_probe_cache_key()
        while True:
            with _APPTAINER_ROOTFS_PROBE_CACHE_LOCK:
                cached = _APPTAINER_ROOTFS_PROBE_CACHE.get(key)
                if cached is not None:
                    return cached, False
                event = _APPTAINER_ROOTFS_PROBE_EVENTS.get(key)
                if event is None:
                    _APPTAINER_ROOTFS_PROBE_EVENTS[key] = threading.Event()
                    return None, True

            event.wait()

    async def _run_apptainer_command(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> _CLIResult:
        self.logger.debug("apptainer cmd: %s", " ".join(cmd))
        env = os.environ.copy()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.environment_dir),
            env=env,
        )
        try:
            if timeout_sec is not None:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec
                )
            else:
                stdout, stderr = await proc.communicate()
        except asyncio.TimeoutError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.communicate(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()
            raise RuntimeError(
                f"apptainer command timed out after {timeout_sec}s: {' '.join(cmd)}"
            )

        result = _CLIResult(
            stdout=stdout.decode("utf-8", errors="replace").rstrip(),
            stderr=stderr.decode("utf-8", errors="replace").rstrip(),
            return_code=proc.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"apptainer command failed (exit {result.return_code}): "
                f"{' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    async def _log_runtime_info(self) -> None:
        """Probe and log Apptainer runtime capabilities."""
        version_str = await self._get_runtime_version()
        cache_key = self._runtime_info_cache_key(version_str)
        with _APPTAINER_RUNTIME_INFO_LOGGED_KEYS_LOCK:
            if cache_key in _APPTAINER_RUNTIME_INFO_LOGGED_KEYS:
                return
            _APPTAINER_RUNTIME_INFO_LOGGED_KEYS.add(cache_key)

        # Overlay mode
        if self._writable_tmpfs:
            overlay_mode = "writable-tmpfs (in-memory)"
        else:
            overlay_mode = f"disk-backed overlay ({self._overlay_size_mb} MB)"

        # SIF cache stats
        sif_count = 0
        if self._sif_cache_dir.exists():
            sif_count = len(list(self._sif_cache_dir.glob("*.sif")))

        # Fakeroot
        fakeroot_str = "enabled" if self._fakeroot else "disabled"

        self.logger.info(
            "Harbor runtime: %s (%s %s)\n"
            "  fakeroot: %s\n"
            "  rootfs mode request: %s\n"
            "  overlay mode: %s\n"
            "  SIF cache: %s (%d images)\n"
            "  isolation: --cleanenv --contain --no-home",
            _APPTAINER_RUNTIME_NAME,
            self._apptainer,
            version_str,
            fakeroot_str,
            self._rootfs_mode,
            overlay_mode,
            self._sif_cache_dir,
            sif_count,
        )

    async def _probe_pid_support(self) -> None:
        """Detect whether the runtime supports ``--pid`` for PID namespace.

        Older SingularityCE (< 4.4) lacks ``--pid``; the fallback is
        ``--containall`` which implies PID + IPC + clean-env containment.
        """
        if not self._pid_namespace:
            return
        if self._pid_flag is not None:
            return

        version_str = await self._get_runtime_version()
        cache_key = self._pid_support_cache_key(version_str)
        cached_flag, is_probe_owner = self._claim_pid_probe(cache_key)
        if cached_flag is not None:
            self._pid_flag = cached_flag
            return

        resolved_flag: str | None = None
        try:
            result = await self._run_apptainer_command(
                [self._apptainer, "instance", "start", "--help"],
                check=False,
            )
            # Parse flag names from help output.  Each flag line starts
            # with optional whitespace then the flag token.  We need the
            # exact flag ``--pid``, not ``--pid-file`` or ``--pids-limit``
            # (which share the prefix), nor ``--pid`` mentioned inside
            # another flag's description text (e.g. ``--no-init``).
            found_pid = False
            for line in result.stdout.splitlines():
                tokens = line.strip().split()
                if not tokens:
                    continue
                flag = tokens[0]
                if "," in flag and len(tokens) > 1:
                    flag = tokens[1]
                if flag == "--pid":
                    found_pid = True
                    break
            resolved_flag = "--pid" if found_pid else "--containall"
            self._pid_flag = resolved_flag
            if resolved_flag == "--containall":
                self.logger.info(
                    "Runtime lacks --pid flag; using --containall for PID namespace"
                )
        except Exception:
            resolved_flag = "--containall"
            self._pid_flag = resolved_flag
        finally:
            if is_probe_owner:
                self._finish_pid_probe(cache_key, resolved_flag)

    async def _probe_root_writability(self) -> bool:
        result = await self._run_apptainer_command(
            [
                self._apptainer,
                "exec",
                "--cleanenv",
                f"instance://{self._instance_name}",
                "bash",
                "-lc",
                "touch /.vb_probe && rm /.vb_probe",
            ],
            check=False,
        )
        return result.return_code == 0

    async def _bootstrap_log_dirs(self) -> None:
        await self._run_apptainer_command([
            self._apptainer,
            "exec",
            "--cleanenv",
            f"instance://{self._instance_name}",
            "bash",
            "-lc",
            "mkdir -p /logs/agent /logs/verifier",
        ])
        # Verify /tmp is writable — fail early with a clear message instead
        # of letting downstream commands (e.g. apt-get) produce cryptic errors.
        result = await self._run_apptainer_command(
            [
                self._apptainer,
                "exec",
                "--cleanenv",
                f"instance://{self._instance_name}",
                "bash",
                "-lc",
                "touch /tmp/.llenvs_tmpdir_probe && rm /tmp/.llenvs_tmpdir_probe",
            ],
            check=False,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "/tmp is not writable inside the Apptainer container. "
                "This usually means the container's tempdir was not "
                "correctly bind-mounted. "
                f"stderr: {result.stderr}"
            )

    def _prepare_runtime_dirs(self) -> tuple[Path, Path]:
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        verifier_dir = self._trial_dir / "verifier"
        agent_dir = self._trial_dir / "agent"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        agent_dir.mkdir(parents=True, exist_ok=True)
        # Create host-backed tempdirs with sticky bit (mode 1777) so that
        # /tmp and /var/tmp are always writable inside the container,
        # regardless of --contain / --containall rootfs semantics.
        for d in (self._host_tmp_dir, self._host_var_tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o1777)
        return verifier_dir, agent_dir

    async def _start_overlay_instance(self) -> None:
        verifier_dir, agent_dir = self._prepare_runtime_dirs()
        self._prepare_trial_bind_dirs()

        if not self._writable_tmpfs:
            self._overlay_path.parent.mkdir(parents=True, exist_ok=True)
            await self._run_apptainer_command([
                self._apptainer,
                "overlay",
                "create",
                "--size",
                str(self._overlay_size_mb),
                str(self._overlay_path),
            ])

        cmd = [self._apptainer, "instance", "start"]
        if self._writable_tmpfs:
            cmd.append("--writable-tmpfs")
        else:
            cmd.extend(["--overlay", str(self._overlay_path)])
        cmd.extend([
            "--cleanenv",
            "--contain",
            "--no-home",
        ])
        if self._fakeroot:
            cmd.append("--fakeroot")
        cmd.extend([
            "--bind", f"{self._host_tmp_dir}:/tmp",
            "--bind", f"{self._host_var_tmp_dir}:/var/tmp",
            "--bind", f"{self._staging_dir}:/staging",
            "--bind", f"{self._app_bind_dir}:/app",
            "--bind", f"{self._tests_bind_dir}:/tests",
            "--bind", f"{verifier_dir}:/logs/verifier",
            "--bind", f"{agent_dir}:/logs/agent",
            str(self._sif_path),
            self._instance_name,
        ])
        await self._run_apptainer_command(cmd)
        await self._bootstrap_log_dirs()
        self._started = True
        self._active_rootfs_mode = "overlay"

    async def _start_sandbox_from_rootfs(self, rootfs_dir: Path) -> None:
        """Start sandbox instance from an existing rootfs directory."""
        # Lazy-probe PID support if not yet resolved (e.g., restore_checkpoint
        # path that bypasses start())
        if self._pid_namespace and self._pid_flag is None:
            await self._probe_pid_support()
        verifier_dir, agent_dir = self._prepare_runtime_dirs()
        for dest in ("staging", "logs/verifier", "logs/agent"):
            (rootfs_dir / dest).mkdir(parents=True, exist_ok=True)
        cmd = [self._apptainer, "instance", "start", "--writable",
               "--cleanenv", "--contain", "--no-home"]
        if self._fakeroot:
            cmd.append("--fakeroot")
        if self._pid_namespace and self._pid_flag:
            if self._pid_flag == "--containall":
                # --containall supersedes --contain (already in cmd),
                # so replace it to avoid redundancy
                if "--contain" in cmd:
                    cmd.remove("--contain")
            cmd.append(self._pid_flag)
        cmd.extend([
            "--bind", f"{self._host_tmp_dir}:/tmp",
            "--bind", f"{self._host_var_tmp_dir}:/var/tmp",
            "--bind", f"{self._staging_dir}:/staging",
            "--bind", f"{verifier_dir}:/logs/verifier",
            "--bind", f"{agent_dir}:/logs/agent",
            str(rootfs_dir),
            self._instance_name,
        ])
        await self._run_apptainer_command(cmd)
        await self._bootstrap_log_dirs()
        self._started = True
        self._active_rootfs_mode = "sandbox"

    async def _start_sandbox_instance(self) -> None:
        rootfs_dir = self._prepare_trial_rootfs()
        await self._start_sandbox_from_rootfs(rootfs_dir)

    async def start(self, force_build: bool = False) -> None:
        await self._log_runtime_info()
        await self._probe_pid_support()

        allow_internet = getattr(self.task_env_config, "allow_internet", True)
        if not allow_internet:
            raise RuntimeError(
                "Apptainer runtime cannot enforce network isolation "
                "(allow_internet=False). Use podman-hpc for tasks requiring "
                "network isolation."
            )

        if not self._sif_path.exists():
            docker_image = getattr(self.task_env_config, "docker_image", None)
            if docker_image and not force_build:
                raise FileNotFoundError(
                    f"SIF image not found at {self._sif_path}. "
                    f"Pre-build it on a login node: "
                    f"{self._apptainer} build {self._sif_path} "
                    f"docker://{docker_image}"
                )
            raise FileNotFoundError(
                f"SIF image not found at {self._sif_path}. "
                "Pre-build it on a node with Docker/Podman access."
            )

        if self._rootfs_mode == "sandbox":
            await self._start_sandbox_instance()
            self.logger.info("Apptainer rootfs mode selected: sandbox")
            return

        cached_probe, probe_owner = self._claim_overlay_probe()

        if self._rootfs_mode == "overlay" and cached_probe is False:
            raise RuntimeError(
                "Apptainer overlay mode did not provide writable root semantics "
                "for this image on this host. Set rootfs_mode: auto or "
                "rootfs_mode: sandbox to use writable sandboxes instead."
            )
        if self._rootfs_mode == "auto" and cached_probe is False:
            await self._start_sandbox_instance()
            return

        probe_result: bool | None = cached_probe
        try:
            await self._start_overlay_instance()
            if cached_probe is True:
                return

            overlay_ok = await self._probe_root_writability()
            probe_result = overlay_ok
            if overlay_ok:
                self.logger.info("Apptainer rootfs mode selected: overlay")
                return

            await self.stop(delete=True)
            if self._rootfs_mode == "overlay":
                raise RuntimeError(
                    "Apptainer overlay mode did not provide writable root semantics "
                    "for this image on this host. Set rootfs_mode: auto or "
                    "rootfs_mode: sandbox to use writable sandboxes instead."
                )

            self.logger.info(
                "Apptainer overlay probe failed; falling back to writable sandbox"
            )
            await self._start_sandbox_instance()
            self.logger.info("Apptainer rootfs mode selected: sandbox")
        finally:
            if probe_owner:
                self._finish_overlay_probe(probe_result)

    async def stop(self, delete: bool = True) -> None:
        if not self._started:
            return
        try:
            await self._run_apptainer_command(
                [self._apptainer, "instance", "stop", self._instance_name],
                check=False,
            )
        finally:
            self._started = False
            self._active_rootfs_mode = None
            if delete:
                if self._overlay_path.exists():
                    self._overlay_path.unlink()
                if self._binds_dir.exists():
                    shutil.rmtree(self._binds_dir, ignore_errors=True)
                if self._sandbox_rootfs_dir.exists():
                    shutil.rmtree(self._sandbox_rootfs_dir, ignore_errors=True)
                if self._staging_dir.exists():
                    shutil.rmtree(self._staging_dir, ignore_errors=True)
                for tmp_dir in (self._host_tmp_dir, self._host_var_tmp_dir):
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir, ignore_errors=True)

    async def export_checkpoint(
        self,
        export_path: Path | str,
        *,
        file_locks: bool = False,
        tcp_established: bool = False,
        ignore_volumes: bool = False,
    ) -> None:
        """Export sandbox rootfs as a tar archive (filesystem checkpoint)."""
        if self._active_rootfs_mode != "sandbox":
            raise RuntimeError(
                "Filesystem checkpoint only supported in sandbox mode"
            )
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = f".{export_path.name}.{uuid.uuid4().hex}.tmp"
        temp_path = export_path.parent / temp_name
        command = (
            f"tar -cf {shlex.quote(f'/.vb_checkpoint_out/{temp_name}')} "
            f"-C /.vb_checkpoint_src ."
        )
        try:
            await self._run_apptainer_checkpoint_command(
                binds=(
                    (self._sandbox_rootfs_dir, "/.vb_checkpoint_src"),
                    (export_path.parent, "/.vb_checkpoint_out"),
                ),
                command=command,
                operation="export",
            )
            os.replace(temp_path, export_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    async def restore_checkpoint(
        self,
        import_path: Path | str,
        *,
        file_locks: bool = False,
        tcp_established: bool = False,
        tcp_close: bool = False,
        ignore_volumes: bool = False,
    ) -> None:
        """Restore sandbox rootfs from a tar archive and restart the instance."""
        import_path = Path(import_path)
        if not import_path.exists():
            raise FileNotFoundError(f"Checkpoint artifact not found: {import_path}")
        if self._started:
            await self.stop(delete=False)
        self._sandbox_rootfs_dir.mkdir(parents=True, exist_ok=True)
        command = (
            "find /.vb_checkpoint_dst -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + "
            f"&& tar -xf {shlex.quote(f'/.vb_checkpoint_in/{import_path.name}')} "
            "-C /.vb_checkpoint_dst"
        )
        await self._run_apptainer_checkpoint_command(
            binds=(
                (import_path.parent, "/.vb_checkpoint_in"),
                (self._sandbox_rootfs_dir, "/.vb_checkpoint_dst"),
            ),
            command=command,
            operation="restore",
        )
        await self._start_sandbox_from_rootfs(self._sandbox_rootfs_dir)

    async def _run_apptainer_checkpoint_command(
        self,
        *,
        binds: tuple[tuple[Path | str, str], ...],
        command: str,
        operation: str,
    ) -> _CLIResult:
        cmd = [self._apptainer, "exec", "--cleanenv"]
        if self._fakeroot:
            cmd.append("--fakeroot")
        for host_path, container_path in binds:
            cmd.extend(["--bind", f"{host_path}:{container_path}"])
        cmd.extend([str(self._sif_path), "bash", "-lc", command])
        try:
            return await self._run_apptainer_command(cmd)
        except RuntimeError as exc:
            message = str(exc)
            if "tar: command not found" in message or "tar: not found" in message:
                raise RuntimeError(
                    "apptainer-hpc filesystem checkpointing requires `tar` inside "
                    "the image used for checkpoint operations"
                ) from exc
            raise RuntimeError(
                f"apptainer-hpc filesystem checkpoint {operation} failed: {message}"
            ) from exc

    async def capture_runtime_probe(self) -> RuntimeProbeSnapshot:
        """Capture runtime state snapshot for filesystem-restore risk detection."""
        probe_script = (
            'echo "===PROCS===";'
            'ps -eo comm= --no-headers 2>/dev/null || echo "UNAVAILABLE";'
            'echo "===MOUNTS===";'
            'md5sum /proc/self/mountinfo 2>/dev/null || echo "UNAVAILABLE";'
            'echo "===SOCKETS===";'
            'ss -lntup --no-header 2>/dev/null || netstat -lntup 2>/dev/null || echo "UNAVAILABLE";'
            'echo "===STAGING===";'
            'ls -A /staging 2>/dev/null || echo "UNAVAILABLE"'
        )
        try:
            result = await self._run_apptainer_command(
                [
                    self._apptainer,
                    "exec",
                    "--cleanenv",
                    f"instance://{self._instance_name}",
                    "bash",
                    "-lc",
                    probe_script,
                ],
                check=False,
            )
            return _parse_probe_output(
                result.stdout, has_pid_namespace=self._pid_namespace
            )
        except Exception as exc:
            return RuntimeProbeSnapshot(
                process_commands=frozenset(),
                mount_fingerprint="",
                listening_ports=frozenset(),
                staging_has_content=False,
                probe_failed=True,
                probe_error=str(exc),
            )

    def detect_runtime_risk(
        self, current: RuntimeProbeSnapshot
    ) -> tuple[bool, tuple[str, ...]]:
        """Compare current probe against baseline and return risk signals."""
        if self._probe_baseline is None:
            return False, ()
        reasons: list[str] = []
        if self._pid_namespace:
            probe_commands = {
                "bash", "ps", "ss", "md5sum", "ls", "cat", "wc", "netstat",
            }
            extra = (
                current.process_commands
                - self._probe_baseline.process_commands
                - probe_commands
            )
            if extra:
                reasons.append(
                    f"extra_processes:{','.join(sorted(extra))}"
                )
        if current.mount_fingerprint != self._probe_baseline.mount_fingerprint:
            reasons.append("mount_table_changed")
        new_ports = current.listening_ports - self._probe_baseline.listening_ports
        if new_ports:
            reasons.append(
                f"new_listening_ports:{','.join(str(p) for p in sorted(new_ports))}"
            )
        if current.staging_has_content:
            reasons.append("staging_content_detected")
        if current.probe_failed and not self._probe_baseline.probe_failed:
            reasons.append("probe_degraded")
        return bool(reasons), tuple(reasons)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> _CLIResult:
        cmd = [self._apptainer, "exec", "--cleanenv"]
        effective_cwd = cwd if cwd is not None else self._default_cwd
        if effective_cwd is not None:
            cmd.extend(["--pwd", effective_cwd])
        if env:
            for key, value in env.items():
                cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([
            f"instance://{self._instance_name}",
            "bash", "-lc", command,
        ])
        return await self._run_apptainer_command(cmd, check=False, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        host_target = self._resolve_bind_target(target_path)
        if host_target is not None:
            host_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(host_target))
            return
        upload_id = str(uuid.uuid4())[:8]
        staging = self._staging_dir / "upload" / upload_id
        staging.mkdir(parents=True, exist_ok=True)
        src = Path(source_path)
        staged = staging / src.name
        shutil.copy2(str(src), str(staged))
        await self._run_apptainer_command([
            self._apptainer, "exec", "--cleanenv",
            f"instance://{self._instance_name}",
            "bash", "-lc", f"cp -a /staging/upload/{upload_id}/{src.name} {target_path}",
        ])
        shutil.rmtree(str(staging), ignore_errors=True)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        host_target = self._resolve_bind_target(target_dir)
        if host_target is not None:
            self._copy_dir_contents(Path(source_dir), host_target)
            return
        upload_id = str(uuid.uuid4())[:8]
        staging = self._staging_dir / "upload" / upload_id
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source_dir), str(staging))
        await self._run_apptainer_command([
            self._apptainer, "exec", "--cleanenv",
            f"instance://{self._instance_name}",
            "bash", "-lc", f"mkdir -p {target_dir} && cp -a /staging/upload/{upload_id}/. {target_dir}/",
        ])
        shutil.rmtree(str(staging), ignore_errors=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        download_id = str(uuid.uuid4())[:8]
        staging = self._staging_dir / "download" / download_id
        staging.mkdir(parents=True, exist_ok=True)
        basename = Path(source_path).name
        await self._run_apptainer_command([
            self._apptainer, "exec", "--cleanenv",
            f"instance://{self._instance_name}",
            "bash", "-lc", f"cp -a {source_path} /staging/download/{download_id}/{basename}",
        ])
        shutil.copy2(str(staging / basename), str(target_path))
        shutil.rmtree(str(staging), ignore_errors=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        download_id = str(uuid.uuid4())[:8]
        staging = self._staging_dir / "download" / download_id
        staging.mkdir(parents=True, exist_ok=True)
        await self._run_apptainer_command([
            self._apptainer, "exec", "--cleanenv",
            f"instance://{self._instance_name}",
            "bash", "-lc", f"cp -a {source_dir}/. /staging/download/{download_id}/",
        ])
        if Path(target_dir).exists():
            shutil.rmtree(str(target_dir))
        shutil.copytree(str(staging), str(target_dir))
        shutil.rmtree(str(staging), ignore_errors=True)


# ── Runtime eligibility ────────────────────────────────────────


def _analyze_apptainer_runtime_eligibility(
    *,
    task_index: int,
    task_name: str,
    environment_dir: Path,
    task_env_config: Any,
    sif_cache_dir: Path | None = None,
) -> RuntimeEligibility:
    """Check whether a Harbor task can run on the Apptainer HPC runtime."""

    def ineligible(code: str, detail: str) -> RuntimeEligibility:
        return RuntimeEligibility(
            task_index=task_index,
            task_name=task_name,
            eligible=False,
            reason_code=code,
            reason_detail=detail,
        )

    compose_path = environment_dir / "docker-compose.yaml"
    if compose_path.exists():
        return ineligible(
            "multi_service_compose",
            "Compose-backed tasks are not supported by the Apptainer runtime.",
        )

    dockerfile_path = environment_dir / "Dockerfile"
    docker_image = getattr(task_env_config, "docker_image", None)
    if not dockerfile_path.exists() and not docker_image:
        return ineligible(
            "missing_container_source",
            "Task defines neither a Dockerfile nor task_env_config.docker_image.",
        )

    # Check if SIF exists when sif_cache_dir is provided
    if sif_cache_dir is not None:
        if docker_image:
            sif_key = _sif_cache_key(docker_image)
        else:
            sif_key = _sif_cache_key(f"dockerfile://{task_name}")
        sif_path = Path(sif_cache_dir) / f"{sif_key}.sif"
        if not sif_path.exists():
            return ineligible(
                "missing_sif_image",
                f"Pre-built SIF image not found at {sif_path}.",
            )

    allow_internet = getattr(task_env_config, "allow_internet", True)
    if not allow_internet:
        return ineligible(
            "network_isolation",
            "Apptainer runtime cannot enforce network isolation (allow_internet=False).",
        )

    return RuntimeEligibility(
        task_index=task_index, task_name=task_name, eligible=True
    )


def inspect_harbor_runtime_eligibility(
    tasks: tuple[Any, ...],
    environment_type: str,
    *,
    sif_cache_dir: str | None = None,
) -> tuple[RuntimeEligibility, ...]:
    """Statically inspect which Harbor tasks are eligible for a given runtime.

    For ``podman-hpc``, all tasks are eligible (it supports compose and
    Dockerfiles natively). For ``apptainer-hpc``, tasks are checked for
    compose, SIF availability, and network isolation constraints.
    """
    normalized = environment_type.strip().lower()
    if normalized in _APPTAINER_ALIASES:
        cache_dir = Path(sif_cache_dir) if sif_cache_dir else None
        return tuple(
            _analyze_apptainer_runtime_eligibility(
                task_index=i,
                task_name=task.name,
                environment_dir=Path(task.paths.environment_dir),
                task_env_config=task.config.environment,
                sif_cache_dir=cache_dir,
            )
            for i, task in enumerate(tasks)
        )

    # podman-hpc and other runtimes: all eligible
    return tuple(
        RuntimeEligibility(task_index=i, task_name=task.name, eligible=True)
        for i, task in enumerate(tasks)
    )


# ── Text-mode environment ───────────────────────────────────────


class HarborEnvironment:
    """Text-based MDP wrapper for Harbor containerized environments.

    Agents send shell commands as ``Action(text="ls -la")`` and receive
    stdout/stderr as observation text. Termination occurs via a submit
    keyword in the action text or truncation at ``max_steps``.

    Example:
        >>> env = HarborEnvironment(tasks=tasks, harbor_env_factory=factory, ...)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> result = env.step(state, Action(text="ls"))
        >>> result = env.step(result.next_state, Action(text="SUBMIT"))
    """

    def __init__(
        self,
        tasks: tuple[Any, ...],
        harbor_env_factory: Any,
        verifier_factory: Any | None = None,
        *,
        dataset_name: str = "terminal-bench",
        max_steps: int = 30,
        submit_keyword: str = "SUBMIT",
        verify_on_truncation: bool = True,
        start_timeout: int | None = 120,
        exec_timeout: int = 120,
        extra_rewards: tuple[RewardFunction, ...] = (),
        state_capture_mode: str = "replay",
        snapshot_artifact_root: Path | str | None = None,
        snapshot_options: HarborSnapshotOptions | None = None,
        answer_extractor: AnswerExtractor | None = None,
        runtime_probing: bool = False,
        text_exec_mode: str = "independent_exec",
        tmux_bootstrap_if_missing: bool = False,
    ) -> None:
        self._tasks = tasks
        self._harbor_env_factory = harbor_env_factory
        self._verifier_factory = verifier_factory
        self._dataset_name = dataset_name
        self._max_steps = max_steps
        self._submit_keyword = submit_keyword
        self._verify_on_truncation = verify_on_truncation
        self._start_timeout = start_timeout
        self._exec_timeout = exec_timeout
        self._state_capture_mode = _normalize_snapshot_mode(state_capture_mode)
        self._snapshot_artifact_root = (
            None if snapshot_artifact_root is None else Path(snapshot_artifact_root).resolve()
        )
        self._snapshot_options = snapshot_options or HarborSnapshotOptions()
        self._answer_extractor = answer_extractor
        self._runtime_probing = runtime_probing
        self._text_exec_mode = _normalize_text_exec_mode(text_exec_mode)
        self._tmux_bootstrap_if_missing = tmux_bootstrap_if_missing

        self._native_rewards: tuple[RewardFunction, ...] = (HarborReward(),)
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

        self._harbor_env: Any = None
        self._current_task: Any = None
        self._text_session: _HarborTmuxTextSession | None = None

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=f"harbor:{self._dataset_name}",
            adapter="harbor",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={
                "dataset": self._dataset_name,
                "text_exec_mode": self._text_exec_mode,
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HarborHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        # Stop previous container if running
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass
            self._text_session = None

        task = self._tasks[task_index]
        self._current_task = task

        # Create and start container
        self._harbor_env = self._harbor_env_factory(task)
        _run_with_timeout(
            self._harbor_env.start(force_build=False),
            self._start_timeout,
            "Harbor container start",
        )
        session_info = {
            "text_exec_mode": self._text_exec_mode,
            "tmux_bootstrapped": False,
            "tmux_start_method": None,
        }
        if self._text_exec_mode == "tmux_session":
            self._text_session = _HarborTmuxTextSession(
                self._harbor_env,
                exec_timeout=self._exec_timeout,
                bootstrap_if_missing=self._tmux_bootstrap_if_missing,
            )
            self._text_session.start()
            session_info["tmux_bootstrapped"] = self._text_session.tmux_bootstrapped
            session_info["tmux_start_method"] = self._text_session.tmux_start_method
        else:
            self._text_session = None

        instruction = getattr(task, "instruction", str(task))

        hidden = HarborHidden(
            task_index=task_index,
            task_name=getattr(task, "name", str(task_index)),
            instruction=instruction,
            episode_step=0,
        )

        observation = Observation(
            prompt=instruction,
            task=ObservationContent(text=instruction),
            state=ObservationContent(text=instruction),
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index, **session_info},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        state = _capture_state_snapshot(
            self._harbor_env,
            state,
            state_capture_mode=self._state_capture_mode,
            snapshot_artifact_root=self._snapshot_artifact_root,
            snapshot_options=self._snapshot_options,
        )
        state = _probe_and_annotate_state(
            self._harbor_env, state, runtime_probing=self._runtime_probing,
        )
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "task_name": hidden.task_name,
            **session_info,
        }

    def _text_for_history(self, raw_text: str, extracted_cmd: str | None) -> str:
        """Return text for the assistant turn in conversation history.

        Uses extracted command when available. On extraction failure, applies
        the extractor's pre-cleaners to strip reasoning tokens from history.
        """
        if extracted_cmd is not None:
            return extracted_cmd
        if self._answer_extractor is None:
            return raw_text
        from llenvs.core.extraction import CleanedExtractor

        if isinstance(self._answer_extractor, CleanedExtractor):
            cleaned = raw_text
            for cleaner in self._answer_extractor.pre_cleaners:
                cleaned = cleaner(cleaned)
            return cleaned
        return raw_text

    def step(
        self,
        state: State[HarborHidden],
        action: Action,
    ) -> StepResult[HarborHidden]:
        self._state_tracker.validate(state, "HarborEnvironment")

        next_step = state.hidden.episode_step + 1
        action_text = action.text or ""
        terminated = False
        truncated = False

        # Extract clean command (strips reasoning tokens etc.)
        extracted_cmd: str | None = None
        if self._answer_extractor is not None and action_text:
            extracted_cmd, _ = self._answer_extractor.extract(action_text)
        cmd_for_env = extracted_cmd or action_text

        # Check for submit keyword on extracted command
        if self._submit_keyword in cmd_for_env:
            terminated = True

        # Execute command in container (even for submit, to maintain trajectory)
        if not terminated:
            if self._text_exec_mode == "tmux_session":
                if self._text_session is None:
                    raise RuntimeError("Harbor tmux text session was not initialized")
                obs_text = self._text_session.run_command(cmd_for_env)
            else:
                exec_result = run_async(
                    self._harbor_env.exec(cmd_for_env, timeout_sec=self._exec_timeout)
                )
                obs_text = _format_exec_result(exec_result)
        else:
            obs_text = "Submitting for verification..."

        # Check truncation
        if not terminated and next_step >= self._max_steps:
            truncated = True

        # Run verifier at terminal
        reward_value: float | None = None
        if terminated or (truncated and self._verify_on_truncation):
            if self._verifier_factory is not None:
                try:
                    rewards = _run_verifier(
                        self._verifier_factory, self._current_task, self._harbor_env
                    )
                    reward_value = rewards.get("reward", 0.0)
                except Exception as e:
                    cause = e.__cause__ if e.__cause__ else e
                    logger.warning("Verifier failed: %s (cause: %s)", e, cause)
                    reward_value = 0.0

        # Build next hidden
        next_hidden = HarborHidden(
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            instruction=state.hidden.instruction,
            episode_step=next_step,
            last_action=cmd_for_env,
            trajectory=state.hidden.trajectory + (cmd_for_env,),
            fs_restore_risk_ever=state.hidden.fs_restore_risk_ever or state.hidden.fs_restore_risk_now,
        )

        # Build messages
        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": self._text_for_history(action_text, extracted_cmd)},
            {"role": "user", "content": obs_text},
        )

        next_obs = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=obs_text),
        )

        info: dict[str, Any] = {
            **state.metadata.info,
            "episode_step": next_step,
        }
        if reward_value is not None:
            info["reward"] = reward_value

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info=info,
        )

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=next_metadata,
        )
        next_state = _capture_state_snapshot(
            self._harbor_env,
            next_state,
            state_capture_mode=self._state_capture_mode,
            snapshot_artifact_root=self._snapshot_artifact_root,
            snapshot_options=self._snapshot_options,
        )
        next_state = _probe_and_annotate_state(
            self._harbor_env, next_state, runtime_probing=self._runtime_probing,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            extracted_action=extracted_cmd,
            resolved_action=extracted_cmd,
            info={
                "episode_step": next_step,
                "observation": obs_text,
            },
        )

    def compute_rewards(
        self,
        state: State[HarborHidden],
        action: Action,
        next_state: State[HarborHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Stop the running container."""
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass
            self._harbor_env = None
            self._text_session = None


# ── Tool-mode environment ───────────────────────────────────────


class HarborToolEnvironment(BaseToolEnvironment[HarborHidden]):
    """Tool-based MDP wrapper for Harbor containerized environments.

    Agents use structured tool calls (``execute_command``, ``read_file``,
    ``write_file``, ``submit``) instead of free-form text commands.
    Inherits tool validation, message building, and monitoring rewards
    from ``BaseToolEnvironment``.

    Example:
        >>> env = HarborToolEnvironment(tasks=tasks, harbor_env_factory=factory, ...)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> call = ToolCall(id="1", name="execute_command", arguments={"command": "ls"})
        >>> result = env.step(state, Action(tool_calls=(call,)))
    """

    def __init__(
        self,
        tasks: tuple[Any, ...],
        harbor_env_factory: Any,
        verifier_factory: Any | None = None,
        *,
        dataset_name: str = "terminal-bench",
        max_steps: int = 30,
        verify_on_truncation: bool = True,
        start_timeout: int | None = 120,
        exec_timeout: int = 120,
        extra_rewards: tuple[RewardFunction, ...] = (),
        state_capture_mode: str = "replay",
        snapshot_artifact_root: Path | str | None = None,
        snapshot_options: HarborSnapshotOptions | None = None,
    ) -> None:
        self._tasks = tasks
        self._harbor_env_factory = harbor_env_factory
        self._verifier_factory = verifier_factory
        self._dataset_name = dataset_name
        self._max_steps = max_steps
        self._verify_on_truncation = verify_on_truncation
        self._start_timeout = start_timeout
        self._exec_timeout = exec_timeout
        self._state_capture_mode = _normalize_snapshot_mode(state_capture_mode)
        self._snapshot_artifact_root = (
            None if snapshot_artifact_root is None else Path(snapshot_artifact_root).resolve()
        )
        self._snapshot_options = snapshot_options or HarborSnapshotOptions()

        self._tools = HARBOR_TOOLS
        self._executor = None  # Not used — we handle execution directly

        self._native_rewards: tuple[RewardFunction, ...] = (
            HarborReward(),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

        self._harbor_env: Any = None
        self._current_task: Any = None

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=f"harbor:{self._dataset_name}",
            adapter="harbor",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={"dataset": self._dataset_name, "tool_mode": True},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HarborHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        # Stop previous container
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass

        task = self._tasks[task_index]
        self._current_task = task

        # Create and start container
        self._harbor_env = self._harbor_env_factory(task)
        _run_with_timeout(
            self._harbor_env.start(force_build=False),
            self._start_timeout,
            "Harbor container start",
        )

        instruction = getattr(task, "instruction", str(task))

        hidden = HarborHidden(
            task_index=task_index,
            task_name=getattr(task, "name", str(task_index)),
            instruction=instruction,
            episode_step=0,
        )

        observation = Observation(
            prompt=instruction,
            available_tools=self._tools,
            task=ObservationContent(text=instruction),
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        state = _capture_state_snapshot(
            self._harbor_env,
            state,
            state_capture_mode=self._state_capture_mode,
            snapshot_artifact_root=self._snapshot_artifact_root,
            snapshot_options=self._snapshot_options,
        )
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "task_name": hidden.task_name,
        }

    def _execute_tool_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call against the Harbor container."""
        if call.name == "execute_command":
            command = call.arguments.get("command", "")
            cwd = call.arguments.get("cwd")
            timeout = call.arguments.get("timeout", self._exec_timeout)

            if cwd:
                command = f"cd {shlex.quote(cwd)} && {command}"

            result = run_async(self._harbor_env.exec(command, timeout_sec=timeout))
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=_format_exec_result(result),
            )

        elif call.name == "read_file":
            path = call.arguments.get("path", "")
            result = run_async(
                self._harbor_env.exec(f"cat {shlex.quote(path)}", timeout_sec=self._exec_timeout)
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            if stderr and not stdout:
                return ToolResult.from_error(
                    call_id=call.id,
                    tool_name=call.name,
                    error_message=stderr,
                )
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=stdout,
            )

        elif call.name == "write_file":
            path = call.arguments.get("path", "")
            content = call.arguments.get("content", "")
            # Use heredoc to write content safely
            eof_marker = "_LLENVS_EOF_"
            cmd = f"cat > {shlex.quote(path)} << '{eof_marker}'\n{content}\n{eof_marker}"
            result = run_async(self._harbor_env.exec(cmd, timeout_sec=self._exec_timeout))
            stderr = getattr(result, "stderr", "") or ""
            if stderr:
                return ToolResult.from_error(
                    call_id=call.id,
                    tool_name=call.name,
                    error_message=stderr,
                )
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output="File written successfully.",
            )

        elif call.name == "submit":
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output="Submitting for verification...",
            )

        return ToolResult.from_error(
            call_id=call.id,
            tool_name=call.name,
            error_message=f"Unknown tool: {call.name}",
        )

    def step(
        self,
        state: State[HarborHidden],
        action: Action,
    ) -> StepResult[HarborHidden]:
        self._state_tracker.validate(state, "HarborToolEnvironment")

        next_step = state.hidden.episode_step + 1
        terminated = False
        truncated = False
        tool_results: list[ToolResult] = []

        if action.has_tool_calls:
            for tc in action.tool_calls:
                validation_error = self._validate_tool_call(tc)
                if validation_error is not None:
                    tool_results.append(validation_error)
                    continue

                try:
                    result = self._execute_tool_call(tc)
                    tool_results.append(result)
                except Exception as e:
                    logger.warning(f"Harbor tool call {tc.name} failed: {e}")
                    tool_results.append(
                        ToolResult.from_error(
                            call_id=tc.id,
                            tool_name=tc.name,
                            error_message=str(e),
                        )
                    )

            # Check for terminal tools
            terminated = self._check_terminal_tools(action.tool_calls)

        # Check truncation
        if not terminated and next_step >= self._max_steps:
            truncated = True

        # Run verifier at terminal
        reward_value: float | None = None
        if terminated or (truncated and self._verify_on_truncation):
            if self._verifier_factory is not None:
                try:
                    rewards = _run_verifier(
                        self._verifier_factory, self._current_task, self._harbor_env
                    )
                    reward_value = rewards.get("reward", 0.0)
                except Exception as e:
                    cause = e.__cause__ if e.__cause__ else e
                    logger.warning("Verifier failed: %s (cause: %s)", e, cause)
                    reward_value = 0.0

        # Build next observation via BaseToolEnvironment helper
        state_text = "\n".join(
            str(tr.output) if tr.is_success else str(tr.error) for tr in tool_results
        )
        next_obs = self._build_next_observation(
            current_obs=state.observation,
            action=action,
            tool_results=tuple(tool_results),
            state_content=ObservationContent(text=state_text) if state_text else None,
        )

        # Build action text for trajectory tracking
        action_text = action.text
        if action.has_tool_calls:
            parts = []
            for tc in action.tool_calls:
                if tc.name == "execute_command":
                    parts.append(tc.arguments.get("command", tc.name))
                else:
                    parts.append(tc.name)
            action_text = "; ".join(parts)

        next_hidden = HarborHidden(
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            instruction=state.hidden.instruction,
            episode_step=next_step,
            last_action=action_text,
            trajectory=state.hidden.trajectory + ((action_text,) if action_text else ()),
        )

        info: dict[str, Any] = {
            **state.metadata.info,
            "episode_step": next_step,
        }
        if reward_value is not None:
            info["reward"] = reward_value

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info=info,
        )

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=next_metadata,
        )
        next_state = _capture_state_snapshot(
            self._harbor_env,
            next_state,
            state_capture_mode=self._state_capture_mode,
            snapshot_artifact_root=self._snapshot_artifact_root,
            snapshot_options=self._snapshot_options,
        )

        rewards = self._compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "tool_results": tuple(tool_results),
                "episode_step": next_step,
            },
        )

    def _compute_rewards(
        self,
        state: State[HarborHidden],
        action: Action,
        next_state: State[HarborHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Stop the running container."""
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass
            self._harbor_env = None


# ── Adapter ─────────────────────────────────────────────────────


class HarborAdapter:
    """Adapter for Harbor containerized evaluation environments.

    Harbor is a generic framework for containerized agent evaluation
    managing Docker containers, task discovery, and verification.
    Datasets include Terminal-Bench, aider-polyglot, swe-bench, etc.
    """

    @property
    def name(self) -> str:
        return "harbor"

    def _get_harbor_api(self) -> Any:
        """Import and return Harbor API handles."""
        try:
            from harbor.environments.factory import EnvironmentFactory
            from harbor.models.environment_type import EnvironmentType
            from harbor.models.task.paths import TaskPaths
            from harbor.models.task.task import Task
            from harbor.models.trial.paths import TrialPaths
            from harbor.registry.client.factory import RegistryClientFactory
            from harbor.tasks.client import TaskClient
            from harbor.verifier.verifier import Verifier

            return _HarborAPI(
                registry_client_factory=RegistryClientFactory,
                task_client=TaskClient,
                task_class=Task,
                task_paths_class=TaskPaths,
                environment_factory=EnvironmentFactory,
                environment_type_enum=EnvironmentType,
                trial_paths_class=TrialPaths,
                verifier_class=Verifier,
            )
        except ImportError as e:
            raise ImportError(
                "harbor is required for HarborAdapter. "
                "Install with: pip install harbor\n"
                "See: https://github.com/laude-institute/harbor"
            ) from e

    @staticmethod
    def _parse_name(name: str) -> tuple[str, str | None]:
        """Parse dataset name and optional version.

        Format: ``"dataset@version"`` or just ``"dataset"``.

        Returns:
            Tuple of (dataset_name, version_or_none).
        """
        if "@" in name:
            dataset, version = name.split("@", 1)
            return dataset, version
        return name, None

    def list_environments(self) -> list[str]:
        """List available datasets from Harbor's registry.

        Returns:
            Sorted list of dataset identifiers.

        Raises:
            ImportError: If harbor is not installed.
        """
        api = self._get_harbor_api()
        client = api.registry_client_factory.create()
        datasets = client.get_datasets()
        names = {f"{dataset.name}@{dataset.version}" for dataset in datasets}
        return sorted(names)

    def load_tasks(
        self,
        name: str = "terminal-bench@2.0",
        *,
        dataset_path: str | None = None,
    ) -> tuple[Any, ...]:
        """Load Harbor tasks without creating environments."""
        dataset_name, version = self._parse_name(name)
        if dataset_path is not None:
            cache_key = ("path", str(Path(dataset_path).resolve()))
        else:
            cache_key = ("registry", dataset_name, version)

        with _HARBOR_TASK_CACHE_LOCK:
            cached = _HARBOR_TASK_CACHE.get(cache_key)
            if cached is not None:
                return cached

            api = self._get_harbor_api()
            if dataset_path is not None:
                tasks = self._load_tasks_from_path(api, dataset_path)
            else:
                tasks = self._load_tasks_from_registry(api, dataset_name, version)
            _HARBOR_TASK_CACHE[cache_key] = tasks
            return tasks

    def inspect_snapshot_eligibility(
        self,
        name: str = "terminal-bench@2.0",
        *,
        tasks: tuple[Any, ...] | None = None,
        dataset_path: str | None = None,
        environment_type: str = "docker",
    ) -> tuple[HarborSnapshotEligibility, ...]:
        """Statically inspect which Harbor tasks support exact snapshots."""
        if tasks is None:
            tasks = self.load_tasks(name, dataset_path=dataset_path)

        if environment_type != "podman-hpc":
            return tuple(
                HarborSnapshotEligibility(
                    task_index=i,
                    task_name=task.name,
                    eligible=False,
                    reason_code="unsupported_snapshot_runtime",
                    reason_detail=(
                        "Exact Harbor snapshots are currently implemented only "
                        "for environment_type='podman-hpc'."
                    ),
                )
                for i, task in enumerate(tasks)
            )

        return tuple(
            _analyze_podman_snapshot_definition(
                task_index=i,
                task_name=task.name,
                environment_dir=Path(task.paths.environment_dir),
                task_env_config=task.config.environment,
            )
            for i, task in enumerate(tasks)
        )

    def get_environment(
        self,
        name: str = "terminal-bench@2.0",
        tasks: tuple[Any, ...] | None = None,
        env_factory: Any | None = None,
        verify_factory: Any | None = None,
        dataset_path: str | None = None,
        environment_type: str = "docker",
        tool_mode: bool = False,
        max_steps: int = 30,
        submit_keyword: str = "SUBMIT",
        start_timeout: int | None = 120,
        exec_timeout: int = 120,
        verify_on_truncation: bool = True,
        extra_rewards: tuple[RewardFunction, ...] = (),
        state_capture_mode: str = "replay",
        snapshot_artifact_root: Path | str | None = None,
        snapshot_options: HarborSnapshotOptions | None = None,
        answer_extractor: AnswerExtractor | None = None,
        runtime_probing: bool = False,
        text_exec_mode: str = "independent_exec",
        tmux_bootstrap_if_missing: bool = False,
        **kwargs: Any,
    ) -> HarborEnvironment | HarborToolEnvironment:
        """Create a Harbor environment.

        Args:
            name: Dataset name with optional version (e.g., "terminal-bench@2.0").
            tasks: Pre-loaded tuple of Harbor Task objects. If None, loaded
                from Harbor's registry or ``dataset_path``.
            env_factory: Callable ``(task) -> BaseEnvironment`` creating
                Harbor container environments. If None, built from harbor library.
            verify_factory: Callable ``(task, env) -> Verifier``. If None,
                built from harbor library.
            dataset_path: Local path to dataset directory. Used when tasks
                and factories are not provided.
            environment_type: Harbor environment type (docker, daytona, etc.).
            tool_mode: If True, returns ``HarborToolEnvironment`` with structured
                tool calls. If False (default), returns ``HarborEnvironment``
                with text-based commands.
            max_steps: Maximum steps per episode.
            submit_keyword: Text mode only — keyword triggering submission.
            start_timeout: Timeout (seconds) for container start/reset.
            exec_timeout: Per-command timeout in seconds.
            verify_on_truncation: Run verifier when truncating at max_steps.
            extra_rewards: Additional reward functions.
            state_capture_mode: ``"replay"`` or ``"snapshot_exact"``.
            snapshot_artifact_root: Snapshot artifact root for exact capture.
            snapshot_options: Runtime-specific exact snapshot options.
            answer_extractor: Text mode only — extractor for parsing agent
                responses (strips reasoning tokens, etc.).
            runtime_probing: If True, capture runtime probes at each state
                and annotate with filesystem-restore risk signals.
            text_exec_mode: Text mode execution model. ``"independent_exec"``
                runs each step in a fresh shell. ``"tmux_session"`` keeps a
                persistent tmux-backed shell inside the container.
            tmux_bootstrap_if_missing: When ``text_exec_mode="tmux_session"``,
                attempt a bounded package-manager install of tmux inside the
                task container if it is missing.
            **kwargs: Passed to Harbor constructors.

        Returns:
            HarborEnvironment or HarborToolEnvironment.
        """
        dataset_name, _version = self._parse_name(name)

        if tool_mode and runtime_probing:
            raise ValueError(
                "runtime_probing is not supported in tool mode. "
                "Use text mode (tool_mode=False) for runtime probing."
            )
        if tool_mode and _normalize_text_exec_mode(text_exec_mode) != "independent_exec":
            raise ValueError(
                "tmux_session text execution is not supported in Harbor tool mode. "
                "Use tool_mode=False when requesting text_exec_mode='tmux_session'."
            )

        # Load tasks and create factories from Harbor if not provided
        if tasks is None or env_factory is None or verify_factory is None:
            api = self._get_harbor_api()

            if tasks is None:
                tasks = self.load_tasks(name, dataset_path=dataset_path)

            trials_dir = kwargs.pop("trials_dir", None)

            if env_factory is None:

                def build_harbor_env(task: Any) -> Any:
                    if environment_type == "podman-hpc" or environment_type in _APPTAINER_ALIASES:
                        return self._create_local_environment(api, task, environment_type, trials_dir=trials_dir, **kwargs)
                    return self._create_harbor_environment(api, task, environment_type, trials_dir=trials_dir, **kwargs)

                env_factory = build_harbor_env

            if verify_factory is None:

                def build_verifier(task: Any, env: Any) -> Any:
                    return api.verifier_class(
                        task=task,
                        trial_paths=env.trial_paths,
                        environment=env,
                        logger=logger,
                    )

                verify_factory = build_verifier

        if tool_mode:
            return HarborToolEnvironment(
                tasks=tasks,
                harbor_env_factory=env_factory,
                verifier_factory=verify_factory,
                dataset_name=dataset_name,
                max_steps=max_steps,
                verify_on_truncation=verify_on_truncation,
                start_timeout=start_timeout,
                exec_timeout=exec_timeout,
                extra_rewards=extra_rewards,
                state_capture_mode=state_capture_mode,
                snapshot_artifact_root=snapshot_artifact_root,
                snapshot_options=snapshot_options,
            )

        return HarborEnvironment(
            tasks=tasks,
            harbor_env_factory=env_factory,
            verifier_factory=verify_factory,
            dataset_name=dataset_name,
            max_steps=max_steps,
            submit_keyword=submit_keyword,
            verify_on_truncation=verify_on_truncation,
            start_timeout=start_timeout,
            exec_timeout=exec_timeout,
            extra_rewards=extra_rewards,
            state_capture_mode=state_capture_mode,
            snapshot_artifact_root=snapshot_artifact_root,
            snapshot_options=snapshot_options,
            answer_extractor=answer_extractor,
            runtime_probing=runtime_probing,
            text_exec_mode=text_exec_mode,
            tmux_bootstrap_if_missing=tmux_bootstrap_if_missing,
        )

    def get_default_system_prompt(self, name: str) -> str:
        """Return a terminal-agent system prompt."""
        return (
            "You are an AI agent with access to a Linux terminal. "
            "Execute commands to complete the task described below. "
            "Work step by step, checking the output of each command "
            "before proceeding. When you have completed the task, "
            "submit your work for verification."
        )

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        dataset_name, version = self._parse_name(name)
        return {
            "name": name,
            "adapter": self.name,
            "dataset": dataset_name,
            "version": version,
            "description": f"Harbor containerized environment ({dataset_name})",
        }

    @staticmethod
    def _load_tasks_from_path(api: "_HarborAPI", dataset_path: str) -> tuple[Any, ...]:
        tasks_root = Path(dataset_path)
        if not tasks_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        tasks: list[Any] = []
        for entry in sorted(tasks_root.iterdir()):
            if not entry.is_dir():
                continue
            task_paths = api.task_paths_class(entry)
            if task_paths.is_valid():
                tasks.append(api.task_class(entry))

        if not tasks:
            raise ValueError(
                "No valid Harbor tasks found in dataset_path. "
                "Expected task directories with instruction.md, task.toml, and environment/."
            )

        return tuple(sorted(tasks, key=lambda t: t.name))

    @staticmethod
    def _load_tasks_from_registry(
        api: "_HarborAPI", dataset_name: str, version: str | None
    ) -> tuple[Any, ...]:
        client = api.registry_client_factory.create()
        spec = client.get_dataset_spec(dataset_name, version=version)
        task_ids = [task.to_source_task_id() for task in spec.tasks]
        task_dirs = api.task_client().download_tasks(task_ids=task_ids)
        tasks = [api.task_class(task_dir=task_dir) for task_dir in task_dirs]
        return tuple(sorted(tasks, key=lambda t: t.name))

    @staticmethod
    def _create_harbor_environment(
        api: "_HarborAPI", task: Any, environment_type: str, *, trials_dir: str | Path | None = None, **kwargs: Any
    ) -> Any:
        trials_base = Path(trials_dir) if trials_dir is not None else Path("trials")
        trial_paths = api.trial_paths_class(trial_dir=trials_base / str(uuid.uuid4()))
        trial_paths.mkdir()
        env_type = api.environment_type_enum(environment_type)
        env = api.environment_factory.create_environment(
            type=env_type,
            environment_dir=task.paths.environment_dir,
            environment_name=task.name,
            session_id=str(uuid.uuid4()),
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            **kwargs,
        )
        env.trial_paths = trial_paths
        return env

    @staticmethod
    def _create_local_environment(
        api: "_HarborAPI",
        task: Any,
        environment_type: str,
        *,
        trials_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        trials_base = Path(trials_dir) if trials_dir is not None else Path("trials")
        trial_paths = api.trial_paths_class(trial_dir=trials_base / str(uuid.uuid4()))
        trial_paths.mkdir()

        if environment_type == "podman-hpc":
            env = PodmanHPCEnvironment(
                environment_dir=Path(task.paths.environment_dir),
                environment_name=task.name,
                session_id=str(uuid.uuid4()),
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                logger=logger,
                **kwargs,
            )
            env.trial_paths = trial_paths
            return env

        if environment_type in _APPTAINER_ALIASES:
            env = ApptainerHPCEnvironment(
                environment_dir=Path(task.paths.environment_dir),
                environment_name=task.name,
                session_id=str(uuid.uuid4()),
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                logger=logger,
                **kwargs,
            )
            env.trial_paths = trial_paths
            return env

        raise ValueError(f"Unsupported local Harbor environment type: {environment_type}")


# ── Restore / replay utilities ──────────────────────────────────


def harbor_restore(
    env: HarborEnvironment,
    state: State[HarborHidden],
) -> State[HarborHidden]:
    """Restore a Harbor env to a saved state by replaying the trajectory prefix.

    Resets to the original task via ``task_index``, then replays each command
    from ``state.hidden.trajectory``. Validates task name to guard against
    index drift across dataset versions.

    Args:
        env: A fresh ``HarborEnvironment`` instance (new container).
        state: The target state whose ``hidden.trajectory`` is replayed.

    Returns:
        The restored state after replaying all commands.

    Raises:
        ValueError: If the task name at the given index doesn't match
            the expected task name from the saved state.
    """
    current, info = env.reset(
        options={
            "task_index": state.hidden.task_index,
            "episode_id": state.metadata.episode_id,
        }
    )

    # Validate task identity
    if state.hidden.task_name and info.get("task_name"):
        if state.hidden.task_name != info["task_name"]:
            raise ValueError(
                f"Task name mismatch: expected {state.hidden.task_name!r}, "
                f"got {info['task_name']!r} at index {state.hidden.task_index}. "
                f"Dataset version may have changed."
            )

    for cmd in state.hidden.trajectory:
        result = env.step(current, Action(text=cmd))
        current = result.next_state

    return current


def harbor_snapshot_restore(
    env: HarborEnvironment,
    state: State[HarborHidden],
    *,
    artifact_root: Path | str,
) -> State[HarborHidden]:
    """Restore a Harbor env to an exact saved state from a checkpoint artifact."""
    snapshot_ref = state.hidden.snapshot_ref
    if snapshot_ref is None:
        raise ValueError("Cannot snapshot-restore Harbor state without hidden.snapshot_ref")

    _current, info = env.reset(
        options={
            "task_index": state.hidden.task_index,
            "episode_id": state.metadata.episode_id,
        }
    )

    if state.hidden.task_name and info.get("task_name"):
        if state.hidden.task_name != info["task_name"]:
            raise ValueError(
                f"Task name mismatch: expected {state.hidden.task_name!r}, "
                f"got {info['task_name']!r} at index {state.hidden.task_index}. "
                f"Dataset version may have changed."
            )

    harbor_env = getattr(env, "_harbor_env", None)
    if harbor_env is None:
        raise RuntimeError("Harbor environment reset did not initialize a runtime")

    _restore_state_snapshot(harbor_env, snapshot_ref, artifact_root=artifact_root)
    text_session = getattr(env, "_text_session", None)
    if text_session is not None:
        try:
            text_session.resync_after_restore()
        except Exception as exc:
            raise RuntimeError(
                "Harbor tmux session could not be re-synchronized after snapshot restore"
            ) from exc
    env._state_tracker.track(state)
    return state


def validate_replay_consistency(
    env_factory: Callable[[], HarborEnvironment],
    task_index: int,
    trajectory: tuple[str, ...],
    probe_commands: tuple[str, ...] = (
        "find /app /home /etc -type f 2>/dev/null | sort | md5sum",
        "dpkg -l 2>/dev/null | awk '{print $2, $3}' | md5sum",
    ),
    reference_probes: dict[str, str] | None = None,
    num_trials: int = 3,
) -> dict[str, Any]:
    """Test whether replaying a trajectory produces consistent container state.

    Two validation modes:

    1. **Self-consistency** (``reference_probes=None``): checks that multiple
       replays produce the same state as each other.
    2. **Live-vs-restored** (``reference_probes`` provided): checks that
       restored state matches probe outputs captured from the live env
       during original data collection.

    Args:
        env_factory: Creates a fresh ``HarborEnvironment`` instance.
        task_index: Task index to reset to.
        trajectory: Commands to replay.
        probe_commands: Commands to run after replay to fingerprint state.
        reference_probes: Optional mapping of probe command → expected stdout
            from the live container. Enables live-vs-restored comparison.
        num_trials: Number of independent replay trials.

    Returns:
        Dict with keys:
            ``consistent`` (bool): All trials match each other.
            ``matches_reference`` (bool | None): Whether probes match stored
                live probes (None if ``reference_probes`` not provided).
            ``probe_outputs`` (list[dict[str, str]]): Probe results per trial.
            ``divergence_details`` (list[str]): Description of any differences.
    """
    trial_outputs: list[dict[str, str]] = []

    for _trial in range(num_trials):
        env = env_factory()
        try:
            # Reset and replay
            current, _info = env.reset(options={"task_index": task_index})
            for cmd in trajectory:
                result = env.step(current, Action(text=cmd))
                current = result.next_state

            # Run probe commands
            probes: dict[str, str] = {}
            for probe_cmd in probe_commands:
                probe_result = env.step(current, Action(text=probe_cmd))
                obs = probe_result.next_state.observation
                probes[probe_cmd] = obs.state.text if obs.state else ""
                current = probe_result.next_state

            trial_outputs.append(probes)
        finally:
            env.close()

    # Self-consistency: all trials must match the first
    divergence_details: list[str] = []
    consistent = True
    if trial_outputs:
        baseline = trial_outputs[0]
        for i, trial in enumerate(trial_outputs[1:], 1):
            for cmd in probe_commands:
                if trial.get(cmd) != baseline.get(cmd):
                    consistent = False
                    divergence_details.append(f"Trial {i} diverges from trial 0 on probe: {cmd!r}")

    # Live-vs-restored comparison
    matches_reference: bool | None = None
    if reference_probes is not None and trial_outputs:
        matches_reference = True
        baseline = trial_outputs[0]
        for cmd, expected in reference_probes.items():
            actual = baseline.get(cmd, "")
            if actual != expected:
                matches_reference = False
                divergence_details.append(
                    f"Reference mismatch on probe {cmd!r}: expected {expected!r}, got {actual!r}"
                )

    return {
        "consistent": consistent,
        "matches_reference": matches_reference,
        "probe_outputs": trial_outputs,
        "divergence_details": divergence_details,
    }
