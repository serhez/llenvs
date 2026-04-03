"""Tests for the Harbor adapter."""

import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from llenvs.core.reward import RewardType
from llenvs.core.state import Action, State
from llenvs.core.tools import ToolCall
from llenvs.core.trajectory import Trajectory

# ── Mock Harbor objects ─────────────────────────────────────────


@dataclass
class MockExecResult:
    """Mock result of executing a command in a Harbor container."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass
class MockHarborTask:
    """Mock Harbor task."""

    name: str = "crypto_01"
    instruction: str = "Decrypt the file secret.enc using AES-256."
    config: dict = field(default_factory=lambda: {"image": "harbor/crypto:latest"})


@dataclass
class MockVerifierResult:
    """Mock result from Harbor verifier."""

    rewards: dict = field(default_factory=lambda: {"reward": 1.0})


class MockHarborEnvironment:
    """Mock Harbor BaseEnvironment (async API)."""

    def __init__(
        self,
        exec_results: list[MockExecResult] | None = None,
        exec_handler: Any | None = None,
        start_error: bool = False,
        start_delay: float = 0.0,
    ):
        self._exec_results = exec_results or [MockExecResult(stdout="ok")]
        self._exec_handler = exec_handler
        self._exec_index = 0
        self._started = False
        self._stopped = False
        self._start_error = start_error
        self._start_delay = start_delay
        self._start_force_build: bool | None = None
        self._stop_delete: bool | None = None
        self._exec_history: list[str] = []
        self._uploaded_files: list[tuple[str, str]] = []
        self._checkpoint_exports: list[tuple[Path, dict[str, Any]]] = []
        self._checkpoint_restores: list[tuple[Path, dict[str, Any]]] = []
        self.is_mounted = True
        self.trial_paths: Any | None = None
        self.snapshot_runtime = "podman-hpc"

    async def start(self, force_build: bool = False) -> None:
        if self._start_delay:
            import asyncio

            await asyncio.sleep(self._start_delay)
        if self._start_error:
            raise RuntimeError("Container failed to start")
        self._start_force_build = force_build
        self._started = True

    async def stop(self, delete: bool = True) -> None:
        self._stop_delete = delete
        self._stopped = True

    async def exec(self, command: str, timeout_sec: int = 120, **kwargs: Any) -> MockExecResult:
        self._exec_history.append(command)
        if self._exec_handler is not None:
            return self._exec_handler(command, timeout_sec=timeout_sec, **kwargs)
        if self._exec_index < len(self._exec_results):
            result = self._exec_results[self._exec_index]
            self._exec_index += 1
            return result
        return MockExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        self._uploaded_files.append((local_path, remote_path))

    async def upload_dir(self, source_dir: str, target_dir: str) -> None:
        pass

    async def download_dir(self, source_dir: str, target_dir: str) -> None:
        pass

    async def download_file(self, source_path: str, target_path: str) -> None:
        pass

    async def export_checkpoint(
        self,
        export_path: Path | str,
        **kwargs: Any,
    ) -> None:
        self._checkpoint_exports.append((Path(export_path), dict(kwargs)))

    async def restore_checkpoint(
        self,
        import_path: Path | str,
        **kwargs: Any,
    ) -> None:
        self._checkpoint_restores.append((Path(import_path), dict(kwargs)))


def _make_harbor_env_factory(
    env: MockHarborEnvironment | None = None,
) -> Any:
    """Create a factory that returns mock Harbor environments."""
    created_envs: list[MockHarborEnvironment] = []

    def factory(task: Any) -> MockHarborEnvironment:
        e = env or MockHarborEnvironment()
        created_envs.append(e)
        return e

    factory._created_envs = created_envs  # type: ignore[attr-defined]
    return factory


def _make_verifier_factory(
    result: MockVerifierResult | None = None,
) -> Any:
    """Create a factory that returns mock verifiers."""

    class MockVerifier:
        def __init__(self, result: MockVerifierResult):
            self._result = result

        async def verify(self) -> MockVerifierResult:
            return self._result

    def factory(task: Any, env: Any) -> MockVerifier:
        return MockVerifier(result or MockVerifierResult())

    return factory


# ── Helpers ─────────────────────────────────────────────────────


def _make_tasks(n: int = 3) -> tuple:
    """Create a tuple of mock tasks."""
    return tuple(
        MockHarborTask(
            name=f"task_{i:02d}",
            instruction=f"Task {i} instruction",
        )
        for i in range(n)
    )


def _make_env(
    tasks: tuple | None = None,
    harbor_env: MockHarborEnvironment | None = None,
    verifier_result: MockVerifierResult | None = None,
    verifier_factory: Any | None = None,
    max_steps: int = 30,
    submit_keyword: str = "SUBMIT",
    verify_on_truncation: bool = True,
    start_timeout: int | None = 120,
    exec_timeout: int = 120,
    extra_rewards: tuple = (),
    dataset_name: str = "terminal-bench",
    state_capture_mode: str = "replay",
    snapshot_artifact_root: Path | None = None,
    text_exec_mode: str = "independent_exec",
    tmux_bootstrap_if_missing: bool = False,
    command_soft_timeout: int | None = None,
    command_timeout_budget: int | None = None,
    max_consecutive_command_timeouts: int | None = None,
    runtime_probing: bool = False,
):
    """Create a HarborEnvironment with mocks."""
    from llenvs.adapters.harbor import HarborEnvironment

    tasks = tasks or _make_tasks()
    mock_env = harbor_env or MockHarborEnvironment()
    env_factory = _make_harbor_env_factory(mock_env)
    verifier_factory = verifier_factory or _make_verifier_factory(verifier_result)

    return HarborEnvironment(
        tasks=tasks,
        harbor_env_factory=env_factory,
        verifier_factory=verifier_factory,
        dataset_name=dataset_name,
        max_steps=max_steps,
        submit_keyword=submit_keyword,
        verify_on_truncation=verify_on_truncation,
        start_timeout=start_timeout,
        exec_timeout=exec_timeout,
        extra_rewards=extra_rewards,
        state_capture_mode=state_capture_mode,
        snapshot_artifact_root=snapshot_artifact_root,
        text_exec_mode=text_exec_mode,
        tmux_bootstrap_if_missing=tmux_bootstrap_if_missing,
        command_soft_timeout=command_soft_timeout,
        command_timeout_budget=command_timeout_budget,
        max_consecutive_command_timeouts=max_consecutive_command_timeouts,
        runtime_probing=runtime_probing,
    )


def _make_tool_env(
    tasks: tuple | None = None,
    harbor_env: MockHarborEnvironment | None = None,
    verifier_result: MockVerifierResult | None = None,
    max_steps: int = 30,
    verify_on_truncation: bool = True,
    start_timeout: int | None = 120,
    exec_timeout: int = 120,
    extra_rewards: tuple = (),
    dataset_name: str = "terminal-bench",
    state_capture_mode: str = "replay",
    snapshot_artifact_root: Path | None = None,
):
    """Create a HarborToolEnvironment with mocks."""
    from llenvs.adapters.harbor import HarborToolEnvironment

    tasks = tasks or _make_tasks()
    mock_env = harbor_env or MockHarborEnvironment()
    env_factory = _make_harbor_env_factory(mock_env)
    verifier_factory = _make_verifier_factory(verifier_result)

    return HarborToolEnvironment(
        tasks=tasks,
        harbor_env_factory=env_factory,
        verifier_factory=verifier_factory,
        dataset_name=dataset_name,
        max_steps=max_steps,
        verify_on_truncation=verify_on_truncation,
        start_timeout=start_timeout,
        exec_timeout=exec_timeout,
        extra_rewards=extra_rewards,
        state_capture_mode=state_capture_mode,
        snapshot_artifact_root=snapshot_artifact_root,
    )


def _reset_env(env, task_index: int = 0):
    """Reset an environment and return (state, info)."""
    return env.reset(options={"task_index": task_index})


def _make_snapshot_task(tmp_path: Path, name: str, *, dockerfile: str | None = None, compose: str | None = None):
    task_dir = tmp_path / name
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    if dockerfile is not None:
        (env_dir / "Dockerfile").write_text(dockerfile)
    if compose is not None:
        (env_dir / "docker-compose.yaml").write_text(compose)
    config = SimpleNamespace(environment=SimpleNamespace(docker_image=None, cpus=1, memory_mb=1024))
    paths = SimpleNamespace(environment_dir=env_dir)
    return SimpleNamespace(name=name, paths=paths, config=config)


class _FakeTmuxRuntime:
    _STATUS_DIR = "/tmp/.llenvs_harbor_tmux_status"

    def __init__(
        self,
        *,
        initial_buffer: str = "bash$ ",
        full_buffers: list[str] | None = None,
        visible_buffers: list[str] | None = None,
        direct_start_error: Exception | None = None,
        missing_tmux: bool = False,
        script_available: bool = True,
        wait_timeout_once: bool = False,
        wait_recovery_fails: bool = False,
        ready_after_attempts: int | None = 1,
        hook_wait_timeout_once: bool = False,
        bang_requires_history_disable: bool = False,
        step_exit_codes: list[int | None] | None = None,
        recovery_exit_code: int | None = None,
    ) -> None:
        self.tmux_installed = not missing_tmux
        self.script_available = script_available
        self.direct_start_error = direct_start_error
        self.direct_start_attempts = 0
        self.wait_timeout_once = wait_timeout_once
        self.wait_recovery_fails = wait_recovery_fails
        self.ready_after_attempts = ready_after_attempts
        self.ready_send_attempts = 0
        self.hook_wait_timeout_once = hook_wait_timeout_once
        self.bang_requires_history_disable = bang_requires_history_disable
        self.history_expansion_disabled = False
        self.pending_bang_timeout = False
        self.pending_bang_recovery_failure = False
        self.full_buffers = [initial_buffer, *(full_buffers or [])]
        self.visible_buffers = visible_buffers or []
        self.step_exit_codes = list(step_exit_codes or [])
        self.recovery_exit_code = recovery_exit_code
        self.status_reads: list[str] = []
        self.exec_calls: list[tuple[str, Any | None]] = []
        self.install_attempts = 0
        self.files: dict[str, str] = {}
        self.staged_hook_script = ""

    def __call__(self, command: str, **kwargs: Any) -> MockExecResult:
        self.exec_calls.append((command, kwargs.get("timeout_sec")))
        if "tmux -V" in command:
            if not self.tmux_installed:
                raise RuntimeError("tmux: not found")
            return MockExecResult(stdout="tmux 3.4")

        if "apt-get" in command or "yum " in command or "dnf " in command or "apk add" in command:
            self.install_attempts += 1
            self.tmux_installed = True
            return MockExecResult(stdout="installed")

        if "command -v script" in command:
            return MockExecResult(stdout="/usr/bin/script" if self.script_available else "")

        if "tmux new-session -d -s" in command:
            if "script -q" not in command and self.direct_start_error is not None:
                self.direct_start_attempts += 1
                raise self.direct_start_error
            return MockExecResult(stdout="")

        if "tmux resize-window -t " in command:
            return MockExecResult(stdout="")

        if "mkdir -p /tmp/.llenvs_harbor_tmux_status" in command:
            return MockExecResult(stdout="")

        if "rm -f /tmp/.llenvs_harbor_tmux_ready" in command:
            self.files.pop("/tmp/.llenvs_harbor_tmux_ready", None)
            return MockExecResult(stdout="")

        if "test -r /tmp/.llenvs_harbor_tmux_ready" in command:
            return MockExecResult(
                stdout=self.files.get("/tmp/.llenvs_harbor_tmux_ready", "")
            )

        if "cat > /tmp/.llenvs_harbor_hook_init.sh <<" in command:
            marker = "cat > /tmp/.llenvs_harbor_hook_init.sh << "
            content = command.split(marker, 1)[1]
            lines = content.splitlines()
            if len(lines) >= 2:
                self.staged_hook_script = "\n".join(lines[1:-1])
                self.files["/tmp/.llenvs_harbor_hook_init.sh"] = self.staged_hook_script
            return MockExecResult(stdout="")

        if "tmux wait-for " in command and "capture-pane" in command and "-S -" in command:
            if self.pending_bang_timeout:
                self.pending_bang_timeout = False
                self.pending_bang_recovery_failure = True
                if not self.visible_buffers:
                    self.visible_buffers.append(
                        "bash: !DOCTYPE: event not found\nbash$ "
                    )
                raise RuntimeError("apptainer command timed out after 120s")
            if self.wait_timeout_once:
                self.wait_timeout_once = False
                raise RuntimeError("apptainer command timed out after 120s")
            token_match = re.search(r"(llenvs_harbor_step_[A-Za-z0-9]+)", command)
            if token_match is not None and self.step_exit_codes:
                exit_code = self.step_exit_codes.pop(0)
                if exit_code is not None:
                    self.files[f"{self._STATUS_DIR}/{token_match.group(1)}"] = str(exit_code)
            if not self.full_buffers:
                raise AssertionError(f"Unexpected full-buffer capture command: {command}")
            return MockExecResult(stdout=self.full_buffers.pop(0))

        if "tmux capture-pane" in command and "-S -" in command:
            if not self.full_buffers:
                raise AssertionError(f"Unexpected full-buffer capture command: {command}")
            return MockExecResult(stdout=self.full_buffers.pop(0))

        if "tmux capture-pane" in command:
            if not self.visible_buffers:
                raise AssertionError(f"Unexpected visible-buffer capture command: {command}")
            return MockExecResult(stdout=self.visible_buffers.pop(0))

        if "tmux send-keys" in command:
            if "/tmp/.llenvs_harbor_tmux_ready" in command:
                self.ready_send_attempts += 1
                if (
                    self.ready_after_attempts is not None
                    and self.ready_send_attempts >= self.ready_after_attempts
                ):
                    match = re.search(
                        r"(llenvs_harbor_ready_[A-Za-z0-9]+)",
                        command,
                    )
                    if match is not None:
                        self.files["/tmp/.llenvs_harbor_tmux_ready"] = match.group(1)
            if "source /tmp/.llenvs_harbor_hook_init.sh" in command:
                self.history_expansion_disabled = "set +H" in self.staged_hook_script
            if (
                self.bang_requires_history_disable
                and "tmux send-keys -l" in command
                and "!" in command
                and not self.history_expansion_disabled
            ):
                self.pending_bang_timeout = True
            return MockExecResult(stdout="")

        if self._STATUS_DIR in command:
            match = re.search(
                r"/tmp/\.llenvs_harbor_tmux_status/[A-Za-z0-9_]+",
                command,
            )
            if match is None:
                raise AssertionError(f"Unexpected status-read command: {command}")
            path = match.group(0)
            self.status_reads.append(path)
            return MockExecResult(stdout=self.files.pop(path, ""))

        if "tmux has-session" in command:
            return MockExecResult(stdout="")

        if "tmux wait-for " in command:
            if (
                self.pending_bang_recovery_failure
                and "llenvs_harbor_step_" in command
                and "tmux wait-for -U" in command
            ):
                self.pending_bang_recovery_failure = False
                raise RuntimeError("apptainer command timed out after 5s")
            if (
                self.wait_recovery_fails
                and "llenvs_harbor_step_" in command
                and "tmux wait-for -U" in command
            ):
                raise RuntimeError("apptainer command timed out after 5s")
            if "llenvs_harbor_init_" in command and self.hook_wait_timeout_once:
                self.hook_wait_timeout_once = False
                raise RuntimeError("apptainer command timed out after 120s")
            if (
                self.recovery_exit_code is not None
                and "llenvs_harbor_step_" in command
                and "tmux wait-for -U" in command
            ):
                token_match = re.search(r"(llenvs_harbor_step_[A-Za-z0-9]+)", command)
                if token_match is not None:
                    self.files[f"{self._STATUS_DIR}/{token_match.group(1)}"] = str(
                        self.recovery_exit_code
                    )
            return MockExecResult(stdout="")


class _FakeTimeoutThenExitProcess:
    def __init__(self) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.communicate_timeouts: list[float | int | None] = []

    def communicate(self, timeout: float | int | None = None) -> tuple[bytes, bytes]:
        self.communicate_timeouts.append(timeout)
        if len(self.communicate_timeouts) == 1:
            raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)
        self.returncode = -signal.SIGTERM
        return (b"", b"")

    def poll(self) -> int | None:
        return self.returncode


class _FakeNeverReapProcess:
    def __init__(self) -> None:
        self.pid = 8765
        self.returncode: int | None = None
        self.communicate_timeouts: list[float | int | None] = []

    def communicate(self, timeout: float | int | None = None) -> tuple[bytes, bytes]:
        self.communicate_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)

    def poll(self) -> int | None:
        return self.returncode

        if "tmux set-option" in command:
            return MockExecResult(stdout="")

        return MockExecResult(stdout="")


# ── TestHarborHidden ────────────────────────────────────────────


class TestHarborHidden:
    def test_creation(self):
        from llenvs.adapters.harbor import HarborHidden

        h = HarborHidden(
            task_index=0,
            task_name="crypto_01",
            instruction="Decrypt the file",
            episode_step=0,
        )
        assert h.task_index == 0
        assert h.task_name == "crypto_01"
        assert h.instruction == "Decrypt the file"
        assert h.episode_step == 0
        assert h.last_action is None


class TestHarborSnapshotEligibility:
    def test_inspect_snapshot_eligibility_accepts_single_container_task(self, tmp_path):
        from llenvs.adapters.harbor import HarborAdapter, HarborSnapshotEligibility

        task = _make_snapshot_task(
            tmp_path,
            "task_01",
            dockerfile="FROM ubuntu:latest\n",
        )

        results = HarborAdapter().inspect_snapshot_eligibility(
            tasks=(task,),
            environment_type="podman-hpc",
        )

        assert results == (
            HarborSnapshotEligibility(
                task_index=0,
                task_name="task_01",
                eligible=True,
                reason_code=None,
                reason_detail=None,
            ),
        )

    def test_inspect_snapshot_eligibility_rejects_multi_service_compose(self, tmp_path):
        from llenvs.adapters.harbor import HarborAdapter

        task = _make_snapshot_task(
            tmp_path,
            "task_01",
            compose=(
                "services:\n"
                "  main:\n"
                "    image: ubuntu:latest\n"
                "  db:\n"
                "    image: postgres:latest\n"
            ),
        )

        results = HarborAdapter().inspect_snapshot_eligibility(
            tasks=(task,),
            environment_type="podman-hpc",
        )

        assert results[0].eligible is False
        assert results[0].reason_code == "multi_service_compose"

    def test_inspect_snapshot_eligibility_rejects_unsupported_compose_fields(self, tmp_path):
        from llenvs.adapters.harbor import HarborAdapter

        task = _make_snapshot_task(
            tmp_path,
            "task_01",
            compose=(
                "services:\n"
                "  main:\n"
                "    image: ubuntu:latest\n"
                "    ports:\n"
                "      - '8080:8080'\n"
            ),
        )

        results = HarborAdapter().inspect_snapshot_eligibility(
            tasks=(task,),
            environment_type="podman-hpc",
        )

        assert results[0].eligible is False
        assert results[0].reason_code == "unsupported_compose_service_fields"

    def test_inspect_snapshot_eligibility_rejects_unsupported_runtime(self, tmp_path):
        from llenvs.adapters.harbor import HarborAdapter

        task = _make_snapshot_task(
            tmp_path,
            "task_01",
            dockerfile="FROM ubuntu:latest\n",
        )

        results = HarborAdapter().inspect_snapshot_eligibility(
            tasks=(task,),
            environment_type="docker",
        )

        assert results[0].eligible is False
        assert results[0].reason_code == "unsupported_snapshot_runtime"

    def test_frozen(self):
        from llenvs.adapters.harbor import HarborHidden

        h = HarborHidden(
            task_index=0,
            task_name="crypto_01",
            instruction="Decrypt",
            episode_step=0,
        )
        with pytest.raises(AttributeError):
            h.episode_step = 5  # type: ignore[misc]

    def test_with_trajectory(self):
        from llenvs.adapters.harbor import HarborHidden

        h = HarborHidden(
            task_index=1,
            task_name="ml_03",
            instruction="Train a model",
            episode_step=3,
            last_action="python train.py",
            trajectory=("ls", "cat data.csv", "python train.py"),
        )
        assert len(h.trajectory) == 3
        assert h.last_action == "python train.py"

    def test_defaults(self):
        from llenvs.adapters.harbor import HarborHidden

        h = HarborHidden(
            task_index=0,
            task_name="t",
            instruction="i",
            episode_step=0,
        )
        assert h.last_action is None
        assert h.trajectory == ()
        assert h.snapshot_ref is None
        assert h.command_timeout_count == 0
        assert h.consecutive_command_timeout_count == 0
        assert h.command_timeout_total_sec == 0.0


# ── TestHarborReward ────────────────────────────────────────────


class TestHarborReward:
    def test_name(self):
        from llenvs.adapters.harbor import HarborReward

        r = HarborReward()
        assert r.name == "harbor"

    def test_reward_type(self):
        from llenvs.adapters.harbor import HarborReward

        r = HarborReward()
        assert r.reward_type == RewardType.OUTCOME

    def test_non_terminal_returns_step_none(self):
        from llenvs.adapters.harbor import HarborHidden, HarborReward
        from llenvs.core.state import Observation, StateMetadata

        r = HarborReward()
        hidden = HarborHidden(0, "t", "i", 1)
        obs = Observation(prompt="test")
        state = State(obs, hidden, StateMetadata(step=0, episode_id="e"))
        next_state = State(
            obs,
            hidden,
            StateMetadata(step=1, episode_id="e", is_terminal=False),
        )
        signal = r.compute(state, Action(text="ls"), next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward is None

    def test_terminal_success(self):
        from llenvs.adapters.harbor import HarborHidden, HarborReward
        from llenvs.core.state import Observation, StateMetadata

        r = HarborReward()
        hidden = HarborHidden(0, "t", "i", 1)
        obs = Observation(prompt="test")
        state = State(obs, hidden, StateMetadata(step=0, episode_id="e"))
        next_state = State(
            obs,
            hidden,
            StateMetadata(
                step=1,
                episode_id="e",
                is_terminal=True,
                info={"reward": 1.0},
            ),
        )
        signal = r.compute(state, Action(text="SUBMIT"), next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 1.0

    def test_terminal_failure(self):
        from llenvs.adapters.harbor import HarborHidden, HarborReward
        from llenvs.core.state import Observation, StateMetadata

        r = HarborReward()
        hidden = HarborHidden(0, "t", "i", 1)
        obs = Observation(prompt="test")
        state = State(obs, hidden, StateMetadata(step=0, episode_id="e"))
        next_state = State(
            obs,
            hidden,
            StateMetadata(
                step=1,
                episode_id="e",
                is_terminal=True,
                info={"reward": 0.0},
            ),
        )
        signal = r.compute(state, Action(text="SUBMIT"), next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0

    def test_terminal_no_reward_info(self):
        from llenvs.adapters.harbor import HarborHidden, HarborReward
        from llenvs.core.state import Observation, StateMetadata

        r = HarborReward()
        hidden = HarborHidden(0, "t", "i", 1)
        obs = Observation(prompt="test")
        state = State(obs, hidden, StateMetadata(step=0, episode_id="e"))
        next_state = State(
            obs,
            hidden,
            StateMetadata(step=1, episode_id="e", is_terminal=True, info={}),
        )
        signal = r.compute(state, Action(text="SUBMIT"), next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0


# ── TestFormatExecResult ────────────────────────────────────────


class TestFormatExecResult:
    def test_stdout_only(self):
        from llenvs.adapters.harbor import _format_exec_result

        result = MockExecResult(stdout="hello world", stderr="", return_code=0)
        assert _format_exec_result(result) == "hello world"

    def test_stderr_shown_with_prefix(self):
        from llenvs.adapters.harbor import _format_exec_result

        result = MockExecResult(stdout="out", stderr="warning", return_code=0)
        formatted = _format_exec_result(result)
        assert "out" in formatted
        assert "[stderr]" in formatted
        assert "warning" in formatted

    def test_both_empty_exit_zero_shows_success_placeholder(self):
        from llenvs.adapters.harbor import _format_exec_result

        result = MockExecResult(stdout="", stderr="", return_code=0)
        assert _format_exec_result(result) == "[Command completed successfully with no output]"

    def test_both_empty_nonzero_exit(self):
        from llenvs.adapters.harbor import _format_exec_result

        result = MockExecResult(stdout="", stderr="", return_code=1)
        assert "[exit code: 1]" in _format_exec_result(result)

    def test_stderr_only(self):
        from llenvs.adapters.harbor import _format_exec_result

        result = MockExecResult(stdout="", stderr="error msg", return_code=1)
        formatted = _format_exec_result(result)
        assert "[stderr]" in formatted
        assert "error msg" in formatted


class TestTmuxHelpers:
    def test_pick_heredoc_delimiter_avoids_full_line_collisions(self):
        from llenvs.adapters.harbor import _pick_heredoc_delimiter

        command = "echo hello\nLLENVS_HARBOR_CMD_deadbeef\npwd"
        delimiter = _pick_heredoc_delimiter(command)

        assert delimiter not in command.splitlines()


# ── TestHarborEnvironment (Text Mode) ───────────────────────────


class TestHarborEnvironment:
    def test_spec(self):
        env = _make_env()
        spec = env.spec
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is False
        assert spec.adapter == "harbor"

    def test_len(self):
        env = _make_env(tasks=_make_tasks(5))
        assert len(env) == 5

    def test_available_tools_empty(self):
        env = _make_env()
        assert env.available_tools == ()

    def test_reward_functions_native(self):
        from llenvs.adapters.harbor import HarborReward

        env = _make_env()
        rfs = env.reward_functions
        assert len(rfs) == 1
        assert isinstance(rfs[0], HarborReward)

    def test_reward_functions_with_extra(self):
        mock_extra = MagicMock()
        env = _make_env(extra_rewards=(mock_extra,))
        assert len(env.reward_functions) == 2

    def test_reset_returns_state_and_info(self):
        env = _make_env()
        state, info = _reset_env(env, task_index=0)
        assert isinstance(state, State)
        assert "task_index" in info
        assert info["task_index"] == 0

    def test_reset_observation_has_instruction(self):
        tasks = _make_tasks()
        env = _make_env(tasks=tasks)
        state, _ = _reset_env(env, task_index=1)
        assert tasks[1].instruction in state.observation.prompt
        assert state.observation.task is not None
        assert tasks[1].instruction in state.observation.task.text

    def test_reset_hidden_state(self):
        from llenvs.adapters.harbor import HarborHidden

        tasks = _make_tasks()
        env = _make_env(tasks=tasks)
        state, _ = _reset_env(env, task_index=1)
        h = state.hidden
        assert isinstance(h, HarborHidden)
        assert h.task_index == 1
        assert h.task_name == tasks[1].name
        assert h.episode_step == 0
        assert h.last_action is None
        assert h.trajectory == ()

    def test_reset_snapshot_exact_captures_checkpoint(self, tmp_path):
        env = _make_env(
            harbor_env=MockHarborEnvironment(),
            state_capture_mode="snapshot_exact",
            snapshot_artifact_root=tmp_path,
        )

        state, _ = _reset_env(env)

        assert state.hidden.snapshot_ref is not None
        assert state.hidden.snapshot_ref.runtime == "podman-hpc"
        assert state.hidden.snapshot_ref.relative_path.endswith("state_0000.tar")
        assert env._harbor_env._checkpoint_exports == [
            (
                tmp_path.parent / state.hidden.snapshot_ref.relative_path,
                {
                    "file_locks": False,
                    "tcp_established": False,
                    "ignore_volumes": False,
                },
            )
        ]

    def test_reset_metadata(self):
        env = _make_env()
        state, _ = _reset_env(env)
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

    def test_reset_tmux_session_initializes_helper_and_metadata(self):
        runtime = _FakeTmuxRuntime()
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")

        state, info = _reset_env(env)

        assert state.metadata.info["text_exec_mode"] == "tmux_session"
        assert state.metadata.info["tmux_bootstrapped"] is False
        assert state.metadata.info["tmux_start_method"] == "direct"
        assert info["tmux_start_method"] == "direct"
        assert any(
            "tmux new-session -d -s" in cmd and "bash --login" in cmd
            for cmd in mock_env._exec_history
        )
        assert any("rm -f /tmp/.llenvs_harbor_tmux_ready" in cmd for cmd in mock_env._exec_history)
        assert any(
            "/tmp/.llenvs_harbor_tmux_ready" in cmd and "tmux send-keys" in cmd
            for cmd in mock_env._exec_history
        )
        init_cmd = next(
            cmd
            for cmd in mock_env._exec_history
            if "cat > /tmp/.llenvs_harbor_hook_init.sh <<" in cmd
        )
        assert 'tmux wait-for -U "$token"' in init_cmd
        assert "PROMPT_COMMAND" in init_cmd
        assert "set +H" in init_cmd
        assert any(
            "tmux wait-for -L llenvs_harbor_init_" in cmd
            and "printf '%s' llenvs_harbor_init_" in cmd
            for cmd in mock_env._exec_history
        )
        assert any(
            "tmux send-keys -t" in cmd
            and "source /tmp/.llenvs_harbor_hook_init.sh" in cmd
            for cmd in mock_env._exec_history
        )
        assert any(
            "tmux resize-window -t " in cmd and " -x 200" in cmd
            for cmd in mock_env._exec_history
        )

    def test_reset_tmux_session_bootstraps_missing_tmux(self):
        runtime = _FakeTmuxRuntime(missing_tmux=True)
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(
            harbor_env=mock_env,
            text_exec_mode="tmux_session",
            tmux_bootstrap_if_missing=True,
        )

        _state, info = _reset_env(env)

        assert info["tmux_bootstrapped"] is True
        assert runtime.install_attempts == 1

    def test_bootstrap_tmux_exports_tmpdir_env_vars(self):
        """TMPDIR/TMP/TEMP are exported before package-manager commands."""
        runtime = _FakeTmuxRuntime(missing_tmux=True)
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(
            harbor_env=mock_env,
            text_exec_mode="tmux_session",
            tmux_bootstrap_if_missing=True,
        )

        _state, _info = _reset_env(env)

        bootstrap_cmd = next(
            cmd for cmd in mock_env._exec_history
            if "apt-get" in cmd or "yum " in cmd or "dnf " in cmd or "apk add" in cmd
        )
        # TMPDIR export must appear before any package-manager invocation
        tmpdir_pos = bootstrap_cmd.find("export TMPDIR=/tmp TMP=/tmp TEMP=/tmp")
        pkg_pos = bootstrap_cmd.find("apt-get")
        assert tmpdir_pos != -1, "TMPDIR export not found in bootstrap command"
        assert tmpdir_pos < pkg_pos, "TMPDIR export must precede package-manager commands"

    def test_reset_tmux_session_uses_script_fallback(self):
        runtime = _FakeTmuxRuntime(
            direct_start_error=RuntimeError("open terminal failed: not a terminal")
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")

        _state, info = _reset_env(env)

        assert info["tmux_start_method"] == "script_fallback"
        assert runtime.direct_start_attempts == 1
        assert any(
            ("script -qc" in cmd or "script -q -c" in cmd) and "bash --login" in cmd
            for cmd in mock_env._exec_history
        )
        assert any(
            "tmux resize-window -t " in cmd and " -x 200" in cmd
            for cmd in mock_env._exec_history
        )

    def test_reset_tmux_session_readiness_timeout_includes_pane_diagnostics(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import llenvs.adapters.harbor as harbor_module

        runtime = _FakeTmuxRuntime(
            initial_buffer="full startup buffer",
            ready_after_attempts=None,
            visible_buffers=["visible startup buffer"],
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session", exec_timeout=3)

        tick = {"value": 0.0}

        def fake_monotonic() -> float:
            tick["value"] += 0.6
            return tick["value"]

        monkeypatch.setattr(harbor_module, "_now_monotonic", fake_monotonic)
        monkeypatch.setattr(harbor_module.time, "sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="did not become ready") as excinfo:
            _reset_env(env)

        message = str(excinfo.value)
        assert "visible startup buffer" in message
        assert "full startup buffer" in message
        assert runtime.ready_send_attempts >= 1
        assert any(
            "tmux capture-pane -p -t " in cmd and "-J" not in cmd
            for cmd in mock_env._exec_history
        )
        assert any(
            "tmux capture-pane -p -S - -t " in cmd and "-J" not in cmd
            for cmd in mock_env._exec_history
        )

    def test_reset_tmux_session_hook_install_timeout_includes_pane_diagnostics(self):
        runtime = _FakeTmuxRuntime(
            initial_buffer="full hook buffer",
            hook_wait_timeout_once=True,
            visible_buffers=["visible hook buffer"],
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")

        with pytest.raises(RuntimeError, match="Prompt hook installation timed out") as excinfo:
            _reset_env(env)

        message = str(excinfo.value)
        assert "visible hook buffer" in message
        assert "full hook buffer" in message

    def test_reset_runner_messages_do_not_repeat_instruction(self):
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner
        from llenvs.inference.protocol import SamplingParams

        env = _make_env()
        state, _ = _reset_env(env)
        trajectory = Trajectory.create(state)
        runner = TrajectoryRunner(
            environment=env,
            backend=MagicMock(),
            sampling_params=SamplingParams(),
            system_prompt="System prompt.",
        )

        messages = runner.build_messages(state, trajectory=trajectory)

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == state.observation.prompt

    def test_reset_requires_task_index(self):
        env = _make_env()
        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_reset_task_index_out_of_bounds(self):
        env = _make_env(tasks=_make_tasks(3))
        with pytest.raises((ValueError, IndexError)):
            env.reset(options={"task_index": 5})

    def test_reset_start_timeout_raises(self):
        mock_env = MockHarborEnvironment(start_delay=0.01)
        env = _make_env(harbor_env=mock_env, start_timeout=0.001)
        with pytest.raises(TimeoutError, match="Harbor container start timed out"):
            _reset_env(env)

    def test_step_executes_command(self):
        mock_env = MockHarborEnvironment(
            exec_results=[MockExecResult(stdout="file1.txt\nfile2.txt")]
        )
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="ls"))
        state_text = ""
        if result.next_state.observation.state is not None:
            state_text = result.next_state.observation.state.text or ""
        assert "file1.txt" in state_text
        assert result.terminated is False
        assert result.truncated is False

    def test_step_snapshot_exact_captures_checkpoint(self, tmp_path):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")])
        env = _make_env(
            harbor_env=mock_env,
            state_capture_mode="snapshot_exact",
            snapshot_artifact_root=tmp_path,
        )
        state, _ = _reset_env(env)

        result = env.step(state, Action(text="ls"))

        assert result.next_state.hidden.snapshot_ref is not None
        assert result.next_state.hidden.snapshot_ref.relative_path.endswith("state_0001.tar")
        assert mock_env._checkpoint_exports[-1] == (
            tmp_path.parent / result.next_state.hidden.snapshot_ref.relative_path,
            {
                "file_locks": False,
                "tcp_established": False,
                "ignore_volumes": False,
            },
        )

    def test_step_accumulates_messages(self):
        mock_env = MockHarborEnvironment(
            exec_results=[
                MockExecResult(stdout="output1"),
                MockExecResult(stdout="output2"),
            ]
        )
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        result1 = env.step(state, Action(text="cmd1"))
        result2 = env.step(result1.next_state, Action(text="cmd2"))

        msgs = result2.next_state.observation.messages
        assert len(msgs) == 4  # 2 pairs of (assistant, user)

    def test_step_updates_hidden(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")])
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="ls"))
        h = result.next_state.hidden
        assert h.episode_step == 1
        assert h.last_action == "ls"
        assert h.trajectory == ("ls",)

    def test_step_with_strict_tag_extractor_returns_invalid_action_feedback_without_exec(self):
        from llenvs.core.extraction import TagBasedExtractor

        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="should not run")])
        env = _make_env(harbor_env=mock_env)
        env._answer_extractor = TagBasedExtractor(tag_name="answer")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        result = env.step(state, Action(text="ls"))

        assert len(mock_env._exec_history) == before
        assert result.extracted_action is None
        assert result.resolved_action is None
        assert result.info["invalid_action_format"] is True
        assert result.info["extraction_metadata"] == {
            "found": False,
            "tag_name": "answer",
        }
        assert result.next_state.metadata.info["invalid_action_format"] is True
        assert result.next_state.metadata.info["extraction_metadata"] == {
            "found": False,
            "tag_name": "answer",
        }
        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == (
            "[Invalid action format: provide exactly one command wrapped in "
            "<answer>...</answer>. No command was executed.]"
        )
        assert result.next_state.hidden.last_action == "ls"
        assert result.next_state.hidden.trajectory == ()

    def test_step_with_strict_action_regex_uses_action_specific_feedback(self):
        from llenvs.core.extraction import RegexExtractor

        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="should not run")])
        env = _make_env(harbor_env=mock_env)
        env._answer_extractor = RegexExtractor(pattern=r"Action:\s*(.+)")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        result = env.step(state, Action(text="think\nls"))

        assert len(mock_env._exec_history) == before
        assert result.info["invalid_action_format"] is True
        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == (
            "[Invalid action format: provide exactly one action in the form "
            "'Action: ...'. No command was executed.]"
        )

    def test_step_with_strict_extractor_trims_whitespace_in_assistant_history(self):
        from llenvs.core.extraction import TagBasedExtractor

        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="should not run")])
        env = _make_env(harbor_env=mock_env)
        env._answer_extractor = TagBasedExtractor(tag_name="answer")
        state, _ = _reset_env(env)

        result = env.step(state, Action(text="\n\napt update && apt install -y python3-pgmpy\n\n"))

        assistant_turn = result.next_state.observation.messages[-2]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] == "apt update && apt install -y python3-pgmpy"

    def test_step_tmux_session_uses_two_exec_success_path(self):
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                "bash$ pwd\n/app\nbash$ ",
            ]
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        result = env.step(state, Action(text="pwd"))

        after = mock_env._exec_history[before:]
        # 3 execs: send, wait+capture, read-exit-status
        assert len(after) == 3
        assert "tmux wait-for -L " in after[0]
        assert "tmux send-keys -l " in after[0]
        assert "tmux wait-for -L " in after[1]
        assert "tmux wait-for -U " in after[1]
        assert "capture-pane -J -p -S -" in after[1]
        assert "/tmp/.llenvs_harbor_tmux_status/" in after[2]
        assert result.next_state.observation.state is not None
        assert "/app" in result.next_state.observation.state.text

    def test_step_tmux_session_uses_direct_send_keys_for_short_command(self):
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                "bash$ echo hello\nhello\nbash$ ",
            ]
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        env.step(state, Action(text="echo hello"))

        after = mock_env._exec_history[before:]
        control_cmd = after[0]
        # Direct send-keys -l with the literal command payload.
        assert "tmux send-keys -l " in control_cmd
        assert "echo hello" in control_cmd
        # No staged-file / source involved.
        assert "source " not in control_cmd

    def test_step_tmux_session_uses_staged_file_for_multiline_command(self):
        multiline_cmd = "echo line1\necho line2"
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                "bash$ source /tmp/.llenvs_harbor_tmux_command\nline1\nline2\nbash$ ",
            ]
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        env.step(state, Action(text=multiline_cmd))

        after = mock_env._exec_history[before:]
        # Staged-file path uses 3 execs: file write, control, wait+capture
        assert len(after) == 4
        # Exec 1: standalone heredoc file staging (NOT in a && chain with send-keys)
        assert "cat > " in after[0]
        assert "tmux send-keys" not in after[0]
        # Exec 2: control exec with send-keys source
        assert "source /tmp/.llenvs_harbor_tmux_command" in after[1]
        assert "tmux wait-for -L " in after[1]
        # Exec 3: wait + capture
        assert "tmux wait-for -L " in after[2]
        assert "capture-pane" in after[2]

    def test_step_tmux_session_uses_staged_file_for_oversized_command(self):
        from llenvs.adapters.harbor import _HarborTmuxTextSession

        long_cmd = "x" * (_HarborTmuxTextSession._DIRECT_SEND_KEYS_MAX_CHARS + 1)
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                f"bash$ source /tmp/.llenvs_harbor_tmux_command\nbash$ ",
            ]
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        env.step(state, Action(text=long_cmd))

        after = mock_env._exec_history[before:]
        # Even though single-line, exceeds threshold — staged-file with 4 execs.
        assert len(after) == 4
        assert "cat > " in after[0]
        assert "source /tmp/.llenvs_harbor_tmux_command" in after[1]

    def test_step_tmux_session_handles_bang_in_command(self):
        """Commands with ! (e.g., <!DOCTYPE>) work because set +H disables history expansion."""
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                'bash$ echo "<!DOCTYPE html>" > /app/out.html\nbash$ ',
            ],
            bang_requires_history_disable=True,
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)

        # This command contains ! which would trigger history expansion without set +H.
        result = env.step(state, Action(text='echo "<!DOCTYPE html>" > /app/out.html'))

        assert result.info["command_timed_out"] is False
        assert result.next_state.observation.state is not None
        # The echoed command should be stripped from the observation by sanitization.
        # echo "..." > file produces no output, so the observation should be minimal.
        obs_text = result.next_state.observation.state.text
        assert "source /tmp/.llenvs_harbor_tmux_command" not in obs_text
        # Verify the hook script disables history expansion and forces prompt sentinel.
        hook_cmd = next(
            cmd for cmd in mock_env._exec_history
            if "cat > /tmp/.llenvs_harbor_hook_init.sh <<" in cmd
        )
        assert "set +H" in hook_cmd
        assert "PS1=" in hook_cmd
        assert "VIRTUAL_ENV_DISABLE_PROMPT=1" in hook_cmd

    def test_step_tmux_session_strips_wrapped_direct_command_echo(self):
        runtime = _FakeTmuxRuntime()
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(
            "bash$  echo '<!DOCTYPE html><html><body><img src=\"x\" on\n"
            "error=\"alert(\\'xss\\')\"></body></html>' > /app/out.html\n"
            "bash: syntax error near unexpected token `)'\n"
            f"{text_session._prompt_sentinel}"
        )

        result = env.step(
            state,
            Action(
                text=(
                    "echo '<!DOCTYPE html><html><body><img src=\"x\" "
                    "onerror=\"alert(\\'xss\\')\"></body></html>' > /app/out.html"
                )
            ),
        )

        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == (
            "bash: syntax error near unexpected token `)'"
        )

    def test_step_tmux_session_rewrites_staged_file_bash_errors(self):
        runtime = _FakeTmuxRuntime()
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(
            "bash$ source /tmp/.llenvs_harbor_tmux_command\n"
            "bash: /tmp/.llenvs_harbor_tmux_command: line 1: syntax error near unexpected token `newline'\n"
            "bash: /tmp/.llenvs_harbor_tmux_command: line 1: `<answer>'\n"
            f"{text_session._prompt_sentinel}"
        )

        result = env.step(
            state,
            Action(text="x" * (text_session._DIRECT_SEND_KEYS_MAX_CHARS + 1)),
        )

        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == (
            "bash: line 1: syntax error near unexpected token `newline'\n"
            "bash: line 1: `<answer>'"
        )

    def test_step_tmux_session_silent_exit_zero_shows_success_placeholder(self):
        runtime = _FakeTmuxRuntime(step_exit_codes=[0])
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(f"bash$ true\n{text_session._prompt_sentinel}")

        result = env.step(state, Action(text="true"))

        assert result.next_state.observation.state is not None
        assert (
            result.next_state.observation.state.text
            == "[Command completed successfully with no output]"
        )
        assert len(runtime.status_reads) == 1

    def test_step_tmux_session_silent_nonzero_exit_shows_exit_code(self):
        runtime = _FakeTmuxRuntime(step_exit_codes=[1])
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(f"bash$ false\n{text_session._prompt_sentinel}")

        result = env.step(state, Action(text="false"))

        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == "[exit code: 1]"
        assert len(runtime.status_reads) == 1

    def test_step_tmux_session_status_unavailable_falls_back_to_no_output(self):
        runtime = _FakeTmuxRuntime(step_exit_codes=[None])
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(f"bash$ true\n{text_session._prompt_sentinel}")

        result = env.step(state, Action(text="true"))

        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == "[No output]"
        assert len(runtime.status_reads) == 1

    def test_step_tmux_session_falls_back_to_visible_screen(self):
        runtime = _FakeTmuxRuntime()
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append("totally different buffer")
        runtime.visible_buffers.append(f"pwd\n/app\n{text_session._prompt_sentinel}")
        before = len(mock_env._exec_history)

        result = env.step(state, Action(text="pwd"))

        after = mock_env._exec_history[before:]
        # 3 execs for direct path: send, wait+capture, read-status
        # Plus 1 extra capture from visible-screen fallback
        assert "capture-pane -J -p -t" in after[2]
        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == "/app"

    def test_step_tmux_session_keeps_real_output_that_mentions_command_file(self):
        runtime = _FakeTmuxRuntime()
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(
            "bash$ note: source /tmp/.llenvs_harbor_tmux_command\nline2\nbash$ "
        )

        result = env.step(
            state,
            Action(
                text="x" * (text_session._DIRECT_SEND_KEYS_MAX_CHARS + 1)
            ),
        )

        assert result.next_state.observation.state is not None
        obs_text = result.next_state.observation.state.text
        assert obs_text.startswith("note: source /tmp/.llenvs_harbor_tmux_command")
        assert "line2" in obs_text

    def test_step_tmux_session_preserves_output_whitespace_when_stripping_prompt(self):
        runtime = _FakeTmuxRuntime()
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        text_session = getattr(env, "_text_session")
        assert text_session is not None
        runtime.full_buffers.append(f"bash$ printf 'x '\nx {text_session._prompt_sentinel}")

        result = env.step(state, Action(text="printf 'x '"))

        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == "x "

    def test_step_tmux_session_soft_timeout_returns_observation(self):
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                "bash$ sleep 999",
            ],
            visible_buffers=["sleep 999"],
            wait_timeout_once=True,
            recovery_exit_code=130,
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(
            harbor_env=mock_env,
            text_exec_mode="tmux_session",
            exec_timeout=30,
            command_soft_timeout=5,
            command_timeout_budget=20,
            max_consecutive_command_timeouts=2,
        )
        state, _ = _reset_env(env)

        result = env.step(state, Action(text="sleep 999"))

        assert any("C-c" in cmd for cmd in mock_env._exec_history)
        assert any(
            "tmux wait-for -L llenvs_harbor_step_" in cmd
            and "tmux wait-for -U llenvs_harbor_step_" in cmd
            for cmd in mock_env._exec_history
        )
        assert result.info["command_timed_out"] is True
        assert result.info["command_timeout_elapsed_sec"] is not None
        assert result.next_state.observation.state is not None
        assert (
            result.next_state.observation.state.text
            == "[Command timed out after 5 seconds and was cancelled.]"
        )
        assert any(
            "tmux wait-for -L llenvs_harbor_step_" in cmd
            and "tmux wait-for -U llenvs_harbor_step_" in cmd
            and timeout_sec == 30
            for cmd, timeout_sec in runtime.exec_calls
        )
        assert any(
            "tmux capture-pane -p -t " in cmd and "-J" not in cmd
            for cmd in mock_env._exec_history
        )
        assert any(
            "tmux capture-pane -p -S - -t " in cmd and "-J" not in cmd
            for cmd in mock_env._exec_history
        )
        assert len(runtime.status_reads) == 1
        assert not any(path.startswith(runtime._STATUS_DIR + "/") for path in runtime.files)

    def test_step_tmux_session_unrecoverable_timeout_still_raises(self):
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                "bash$ sleep 999",
            ],
            visible_buffers=["sleep 999"],
            wait_timeout_once=True,
            wait_recovery_fails=True,
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(
            harbor_env=mock_env,
            text_exec_mode="tmux_session",
            exec_timeout=30,
            command_soft_timeout=5,
            command_timeout_budget=20,
            max_consecutive_command_timeouts=2,
        )
        state, _ = _reset_env(env)

        with pytest.raises(RuntimeError, match="sleep 999"):
            env.step(state, Action(text="sleep 999"))

    def test_step_tmux_session_stages_file_for_large_commands(self):
        runtime = _FakeTmuxRuntime(
            full_buffers=[
                "bash$ source /tmp/.llenvs_harbor_tmux_command\nbash$ ",
            ]
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        state, _ = _reset_env(env)
        before = len(mock_env._exec_history)

        large_command = "x" * 70000
        env.step(state, Action(text=large_command))

        after = mock_env._exec_history[before:]
        assert len(after) == 4
        assert "cat > " in after[0]
        assert "source /tmp/.llenvs_harbor_tmux_command" in after[1]

    def test_step_soft_timeout_adds_observation_and_history(self, monkeypatch: pytest.MonkeyPatch):
        import llenvs.adapters.harbor as harbor_module

        def timeout_handler(command: str, timeout_sec: int = 120, **_: Any) -> MockExecResult:
            raise RuntimeError(f"apptainer command timed out after {timeout_sec}s: {command}")

        mock_env = MockHarborEnvironment(exec_handler=timeout_handler)
        env = _make_env(
            harbor_env=mock_env,
            command_soft_timeout=5,
            command_timeout_budget=20,
            max_consecutive_command_timeouts=2,
        )
        state, _ = _reset_env(env)
        tick_values = [0.0, 4.5]
        last_tick = {"value": tick_values[-1]}

        def fake_monotonic() -> float:
            if tick_values:
                last_tick["value"] = tick_values.pop(0)
            return last_tick["value"]

        monkeypatch.setattr(harbor_module, "_now_monotonic", fake_monotonic)

        result = env.step(state, Action(text="sleep 999"))

        expected = "[Command timed out after 5 seconds and was cancelled by the evaluation harness.]"
        assert result.info["command_timed_out"] is True
        assert result.info["command_timeout_elapsed_sec"] == pytest.approx(4.5)
        assert result.info["command_timeout_total_sec"] == pytest.approx(4.5)
        assert result.info["timeout_policy_truncated"] is False
        assert result.next_state.hidden.command_timeout_count == 1
        assert result.next_state.hidden.consecutive_command_timeout_count == 1
        assert result.next_state.hidden.command_timeout_total_sec == pytest.approx(4.5)
        assert result.next_state.observation.state is not None
        assert result.next_state.observation.state.text == expected
        assert result.next_state.observation.messages[-2] == {
            "role": "assistant",
            "content": "sleep 999",
        }
        assert result.next_state.observation.messages[-1] == {
            "role": "user",
            "content": expected,
        }
        assert result.next_state.metadata.info["command_timed_out"] is True
        assert result.next_state.metadata.info["command_timeout_total_sec"] == pytest.approx(4.5)

    def test_step_success_resets_consecutive_timeout_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import llenvs.adapters.harbor as harbor_module

        def handler(command: str, timeout_sec: int = 120, **_: Any) -> MockExecResult:
            if command == "sleep 999":
                raise RuntimeError(
                    f"apptainer command timed out after {timeout_sec}s: {command}"
                )
            return MockExecResult(stdout=f"ran {command}")

        mock_env = MockHarborEnvironment(exec_handler=handler)
        env = _make_env(
            harbor_env=mock_env,
            command_soft_timeout=5,
            command_timeout_budget=20,
            max_consecutive_command_timeouts=2,
        )
        state, _ = _reset_env(env)
        tick_values = [0.0, 4.0]
        last_tick = {"value": tick_values[-1]}

        def fake_monotonic() -> float:
            if tick_values:
                last_tick["value"] = tick_values.pop(0)
            return last_tick["value"]

        monkeypatch.setattr(harbor_module, "_now_monotonic", fake_monotonic)

        timeout_result = env.step(state, Action(text="sleep 999"))
        success_result = env.step(timeout_result.next_state, Action(text="pwd"))

        assert timeout_result.next_state.hidden.consecutive_command_timeout_count == 1
        assert success_result.info["command_timed_out"] is False
        assert success_result.next_state.hidden.command_timeout_count == 1
        assert success_result.next_state.hidden.command_timeout_total_sec == pytest.approx(4.0)
        assert success_result.next_state.hidden.consecutive_command_timeout_count == 0

    def test_step_timeout_budget_truncates_and_runs_verifier(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import llenvs.adapters.harbor as harbor_module

        def timeout_handler(command: str, timeout_sec: int = 120, **_: Any) -> MockExecResult:
            raise RuntimeError(f"apptainer command timed out after {timeout_sec}s: {command}")

        mock_env = MockHarborEnvironment(exec_handler=timeout_handler)
        env = _make_env(
            harbor_env=mock_env,
            verifier_result=MockVerifierResult(rewards={"reward": 0.75}),
            command_soft_timeout=5,
            command_timeout_budget=7,
            max_consecutive_command_timeouts=3,
            verify_on_truncation=True,
        )
        state, _ = _reset_env(env)
        tick_values = [0.0, 4.0, 10.0, 14.0]
        last_tick = {"value": tick_values[-1]}

        def fake_monotonic() -> float:
            if tick_values:
                last_tick["value"] = tick_values.pop(0)
            return last_tick["value"]

        monkeypatch.setattr(harbor_module, "_now_monotonic", fake_monotonic)

        first = env.step(state, Action(text="sleep 999"))
        second = env.step(first.next_state, Action(text="sleep 999"))

        expected = (
            "[Command timed out after 5 seconds and was cancelled by the evaluation harness.]\n"
            "[Trajectory terminated after exceeding the command-timeout budget.]"
        )
        assert first.truncated is False
        assert second.truncated is True
        assert second.terminated is False
        assert second.info["timeout_policy_truncated"] is True
        assert second.info["command_timeout_total_sec"] == pytest.approx(8.0)
        assert second.next_state.hidden.command_timeout_total_sec == pytest.approx(8.0)
        assert second.next_state.observation.state is not None
        assert second.next_state.observation.state.text == expected
        assert second.next_state.metadata.info["reward"] == 0.75

    def test_step_consecutive_timeout_cap_truncates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import llenvs.adapters.harbor as harbor_module

        def timeout_handler(command: str, timeout_sec: int = 120, **_: Any) -> MockExecResult:
            raise RuntimeError(f"apptainer command timed out after {timeout_sec}s: {command}")

        mock_env = MockHarborEnvironment(exec_handler=timeout_handler)
        env = _make_env(
            harbor_env=mock_env,
            command_soft_timeout=5,
            command_timeout_budget=30,
            max_consecutive_command_timeouts=2,
        )
        state, _ = _reset_env(env)
        tick_values = [0.0, 3.0, 10.0, 13.0]
        last_tick = {"value": tick_values[-1]}

        def fake_monotonic() -> float:
            if tick_values:
                last_tick["value"] = tick_values.pop(0)
            return last_tick["value"]

        monkeypatch.setattr(harbor_module, "_now_monotonic", fake_monotonic)

        first = env.step(state, Action(text="sleep 999"))
        second = env.step(first.next_state, Action(text="sleep 999"))

        assert first.truncated is False
        assert second.truncated is True
        assert second.info["timeout_policy_truncated"] is True
        assert second.info["consecutive_command_timeout_count"] == 2

    def test_step_submit_keyword_terminates(self):
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env, submit_keyword="SUBMIT")
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="SUBMIT"))
        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_submit_keyword_case_sensitive(self):
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env, submit_keyword="SUBMIT")
        state, _ = _reset_env(env)
        # "submit" should NOT trigger termination (case-sensitive)
        result = env.step(state, Action(text="submit"))
        assert result.terminated is False

    def test_step_submit_runs_verifier(self):
        verifier_result = MockVerifierResult(rewards={"reward": 1.0})
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env, verifier_result=verifier_result)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="SUBMIT"))
        assert result.terminated is True
        assert result.next_state.metadata.info.get("reward") == 1.0

    def test_step_submit_verifier_failure(self):
        verifier_result = MockVerifierResult(rewards={"reward": 0.0})
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env, verifier_result=verifier_result)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="SUBMIT"))
        assert result.terminated is True
        assert result.next_state.metadata.info.get("reward") == 0.0

    def test_truncation_at_max_steps(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_env(harbor_env=mock_env, max_steps=3)
        state, _ = _reset_env(env)
        # Steps 1, 2, 3 — step 3 should truncate
        for i in range(2):
            result = env.step(state, Action(text=f"cmd{i}"))
            state = result.next_state
            assert result.truncated is False

        result = env.step(state, Action(text="cmd2"))
        assert result.truncated is True
        assert result.next_state.metadata.is_terminal is True

    def test_truncation_runs_verifier_when_enabled(self):
        verifier_result = MockVerifierResult(rewards={"reward": 0.5})
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_env(
            harbor_env=mock_env,
            max_steps=1,
            verifier_result=verifier_result,
            verify_on_truncation=True,
        )
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="cmd"))
        assert result.truncated is True
        assert result.next_state.metadata.info.get("reward") == 0.5

    def test_debug_logs_step_probe_and_verifier(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        from llenvs.adapters.harbor import RuntimeProbeSnapshot

        verifier_result = MockVerifierResult(rewards={"reward": 0.5})
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)

        async def capture_runtime_probe() -> RuntimeProbeSnapshot:
            return RuntimeProbeSnapshot(
                process_commands=frozenset(),
                mount_fingerprint="abc",
                listening_ports=frozenset(),
                staging_has_content=False,
            )

        mock_env.capture_runtime_probe = capture_runtime_probe  # type: ignore[attr-defined]
        mock_env.detect_runtime_risk = lambda current: (False, ())  # type: ignore[attr-defined]
        mock_env._probe_baseline = None  # type: ignore[attr-defined]

        env = _make_env(
            harbor_env=mock_env,
            max_steps=1,
            verifier_result=verifier_result,
            verify_on_truncation=True,
            runtime_probing=True,
        )
        state, _ = _reset_env(env)

        with caplog.at_level(logging.DEBUG, logger="llenvs.adapters.harbor"):
            result = env.step(state, Action(text="cmd"))

        assert result.truncated is True
        messages = [record.getMessage() for record in caplog.records]
        assert any("Harbor step start:" in message for message in messages)
        assert any("Harbor step command phase done:" in message for message in messages)
        assert any("Harbor verifier start:" in message for message in messages)
        assert any("Harbor verifier done:" in message for message in messages)
        assert any("Harbor runtime probe start:" in message for message in messages)
        assert any("Harbor runtime probe finished:" in message for message in messages)

    def test_debug_logs_tmux_timeout_recovery(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        runtime = _FakeTmuxRuntime(
            full_buffers=["bash$ "],
            visible_buffers=["bash$ sleep 999"],
            wait_timeout_once=True,
        )
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(
            harbor_env=mock_env,
            text_exec_mode="tmux_session",
            exec_timeout=30,
            command_soft_timeout=5,
            command_timeout_budget=20,
            max_consecutive_command_timeouts=2,
        )
        state, _ = _reset_env(env)

        with caplog.at_level(logging.DEBUG, logger="llenvs.adapters.harbor"):
            result = env.step(state, Action(text="sleep 999"))

        assert result.info["command_timed_out"] is True
        messages = [record.getMessage() for record in caplog.records]
        assert any("Harbor tmux timeout recovery start:" in message for message in messages)
        assert any("Harbor tmux timeout recovery sent Ctrl-C:" in message for message in messages)
        assert any("Harbor tmux timeout recovery succeeded:" in message for message in messages)

    def test_truncation_skips_verifier_when_disabled(self):
        verifier_result = MockVerifierResult(rewards={"reward": 0.5})
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_env(
            harbor_env=mock_env,
            max_steps=1,
            verifier_result=verifier_result,
            verify_on_truncation=False,
        )
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="cmd"))
        assert result.truncated is True
        # No reward should be set
        assert result.next_state.metadata.info.get("reward") is None

    def test_truncation_verifier_timeout_returns_zero_reward(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import llenvs.adapters.harbor as harbor_module

        class NeverVerifier:
            async def verify(self) -> MockVerifierResult:
                import asyncio

                await asyncio.Future()
                raise AssertionError("unreachable")

        def verifier_factory(task: Any, env: Any) -> NeverVerifier:
            return NeverVerifier()

        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_env(
            harbor_env=mock_env,
            max_steps=1,
            verify_on_truncation=True,
            verifier_factory=verifier_factory,
        )
        state, _ = _reset_env(env)

        original_run_with_timeout = harbor_module._run_with_timeout

        def fake_run_with_timeout(coro: Any, timeout: int | None, label: str) -> Any:
            if label == "Harbor verifier":
                close = getattr(coro, "close", None)
                if callable(close):
                    close()
                raise TimeoutError("Harbor verifier timed out after 120s")
            return original_run_with_timeout(coro, timeout, label)

        monkeypatch.setattr(harbor_module, "_run_with_timeout", fake_run_with_timeout)

        result = env.step(state, Action(text="cmd"))

        assert result.truncated is True
        assert result.next_state.metadata.info.get("reward") == 0.0

    def test_step_stderr_in_observation(self):
        mock_env = MockHarborEnvironment(
            exec_results=[MockExecResult(stdout="", stderr="permission denied", return_code=1)]
        )
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="rm /root"))
        obs_text = ""
        if result.next_state.observation.state is not None:
            obs_text = result.next_state.observation.state.text or ""
        assert "permission denied" in obs_text

    def test_step_nonzero_exit_not_terminal(self):
        mock_env = MockHarborEnvironment(
            exec_results=[MockExecResult(stdout="", stderr="error", return_code=1)]
        )
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="bad_cmd"))
        assert result.terminated is False

    def test_rewards_computed(self):
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="ls"))
        assert result.rewards is not None
        assert len(result.rewards.signals) >= 1

    def test_close_stops_container(self):
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        env.close()

    def test_state_continuity_validated(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        env.step(state, Action(text="cmd1"))
        # Using stale state should raise
        with pytest.raises((ValueError, NotImplementedError), match="stale|Stale"):
            env.step(state, Action(text="cmd2"))

    def test_reset_cleans_up_previous_episode(self):
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env)
        state1, _ = _reset_env(env, task_index=0)
        # Reset again — should not raise
        state2, _ = _reset_env(env, task_index=1)
        assert state2.hidden.task_index == 1

    def test_submit_keyword_embedded_in_text(self):
        """Submit keyword within text should trigger termination."""
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env, submit_keyword="SUBMIT")
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="I want to SUBMIT my work"))
        assert result.terminated is True

    def test_empty_action_text(self):
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        # Should handle None/empty text gracefully
        result = env.step(state, Action(text=""))
        assert result.terminated is False


# ── TestHarborToolEnvironment (Tool Mode) ───────────────────────


class TestHarborToolEnvironment:
    def test_spec(self):
        env = _make_tool_env()
        spec = env.spec
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.adapter == "harbor"

    def test_available_tools(self):
        env = _make_tool_env()
        tools = env.available_tools
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert "execute_command" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "submit" in names

    def test_submit_tool_is_terminal(self):
        env = _make_tool_env()
        submit_tool = next(t for t in env.available_tools if t.name == "submit")
        assert submit_tool.is_terminal is True

    def test_other_tools_not_terminal(self):
        env = _make_tool_env()
        for tool in env.available_tools:
            if tool.name != "submit":
                assert tool.is_terminal is False

    def test_reward_functions_include_monitoring(self):
        env = _make_tool_env()
        rfs = env.reward_functions
        # Should have HarborReward + 2 monitoring rewards
        assert len(rfs) == 3

    def test_len(self):
        env = _make_tool_env(tasks=_make_tasks(7))
        assert len(env) == 7

    def test_reset(self):
        env = _make_tool_env()
        state, info = _reset_env(env)
        assert isinstance(state, State)
        assert state.observation.available_tools == env.available_tools

    def test_reset_start_timeout_raises(self):
        mock_env = MockHarborEnvironment(start_delay=0.01)
        env = _make_tool_env(harbor_env=mock_env, start_timeout=0.001)
        with pytest.raises(TimeoutError, match="Harbor container start timed out"):
            _reset_env(env)

    def test_execute_command(self):
        mock_env = MockHarborEnvironment(
            exec_results=[MockExecResult(stdout="file1.txt\nfile2.txt")]
        )
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="execute_command",
            arguments={"command": "ls -la"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.terminated is False
        # Tool result should contain the output
        assert result.info.get("tool_results") is not None
        tool_results = result.info["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].is_success
        assert "file1.txt" in tool_results[0].output

    def test_read_file(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="file contents here")])
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="read_file",
            arguments={"path": "/etc/passwd"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.info["tool_results"][0].is_success
        assert "file contents here" in result.info["tool_results"][0].output

    def test_write_file(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="")])
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="write_file",
            arguments={"path": "/tmp/test.txt", "content": "hello"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.info["tool_results"][0].is_success

    def test_submit_tool_terminates(self):
        verifier_result = MockVerifierResult(rewards={"reward": 1.0})
        mock_env = MockHarborEnvironment()
        env = _make_tool_env(harbor_env=mock_env, verifier_result=verifier_result)
        state, _ = _reset_env(env)

        call = ToolCall(id="call_1", name="submit", arguments={})
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True
        assert result.next_state.metadata.info.get("reward") == 1.0

    def test_unknown_tool_rejected(self):
        env = _make_tool_env()
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="nonexistent_tool",
            arguments={"x": "y"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        tr = result.info["tool_results"][0]
        assert not tr.is_success
        assert "Unknown tool" in tr.error

    def test_execute_command_with_timeout(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")])
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="execute_command",
            arguments={"command": "sleep 1", "timeout": 60},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.info["tool_results"][0].is_success

    def test_truncation_at_max_steps(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_tool_env(harbor_env=mock_env, max_steps=2)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="execute_command",
            arguments={"command": "cmd1"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        state = result.next_state
        assert result.truncated is False

        call2 = ToolCall(
            id="call_2",
            name="execute_command",
            arguments={"command": "cmd2"},
        )
        result2 = env.step(state, Action(tool_calls=(call2,)))
        assert result2.truncated is True

    def test_messages_built_by_base(self):
        """Tool env should build observation messages via BaseToolEnvironment."""
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="output")])
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="execute_command",
            arguments={"command": "ls"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        msgs = result.next_state.observation.messages
        # Should have assistant + tool messages
        assert len(msgs) >= 2

    def test_state_continuity(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 5)
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        call = ToolCall(
            id="call_1",
            name="execute_command",
            arguments={"command": "ls"},
        )
        env.step(state, Action(tool_calls=(call,)))
        with pytest.raises((ValueError, NotImplementedError), match="stale|Stale"):
            env.step(state, Action(tool_calls=(call,)))

    def test_execute_command_with_cwd(self):
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")])
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="call_1",
            name="execute_command",
            arguments={"command": "ls", "cwd": "/home"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.info["tool_results"][0].is_success


# ── TestHarborAdapter ───────────────────────────────────────────


class TestHarborAdapter:
    def test_name(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        assert adapter.name == "harbor"

    def test_get_harbor_import_error(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()

        def boom() -> Any:
            raise ImportError("harbor")

        adapter._get_harbor_api = boom  # type: ignore[assignment]

        with pytest.raises(ImportError, match="harbor"):
            adapter._get_harbor_api()

    def test_get_native_answer_extractor_returns_none(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        assert adapter.get_native_answer_extractor("anything") is None

    def test_get_prompt_template_returns_none(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        assert adapter.get_prompt_template("anything") is None

    def test_get_default_system_prompt(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        prompt = adapter.get_default_system_prompt("terminal-bench")
        assert prompt is not None
        assert "terminal" in prompt.lower() or "command" in prompt.lower()

    def test_get_environment_info(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        info = adapter.get_environment_info("harbor:terminal-bench@2.0")
        assert info["adapter"] == "harbor"
        assert "terminal-bench" in info["name"]

    def test_name_parsing_dataset_version(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        dataset, version = adapter._parse_name("terminal-bench@2.0")
        assert dataset == "terminal-bench"
        assert version == "2.0"

    def test_name_parsing_no_version(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        dataset, version = adapter._parse_name("terminal-bench")
        assert dataset == "terminal-bench"
        assert version is None

    def test_load_tasks_reuses_cached_registry_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import HarborAdapter

        harbor_mod._HARBOR_TASK_CACHE.clear()
        api = object()
        calls: list[tuple[Any, str, str | None]] = []

        monkeypatch.setattr(HarborAdapter, "_get_harbor_api", lambda self: api)

        def fake_load(api_obj: Any, dataset_name: str, version: str | None) -> tuple[Any, ...]:
            calls.append((api_obj, dataset_name, version))
            return ("task-a", "task-b")

        monkeypatch.setattr(
            HarborAdapter,
            "_load_tasks_from_registry",
            staticmethod(fake_load),
        )

        assert HarborAdapter().load_tasks("terminal-bench@2.0") == ("task-a", "task-b")
        assert HarborAdapter().load_tasks("terminal-bench@2.0") == ("task-a", "task-b")
        assert calls == [(api, "terminal-bench", "2.0")]

    def test_get_environment_text_mode(self):
        """Should return HarborEnvironment when tool_mode=False."""
        from llenvs.adapters.harbor import HarborAdapter, HarborEnvironment

        adapter = HarborAdapter()
        tasks = _make_tasks()
        mock_env = MockHarborEnvironment()
        env_factory = _make_harbor_env_factory(mock_env)
        verifier_factory = _make_verifier_factory()

        env = adapter.get_environment(
            name="test",
            tasks=tasks,
            env_factory=env_factory,
            verify_factory=verifier_factory,
            tool_mode=False,
        )
        assert isinstance(env, HarborEnvironment)

    def test_get_environment_tool_mode(self):
        """Should return HarborToolEnvironment when tool_mode=True."""
        from llenvs.adapters.harbor import HarborAdapter, HarborToolEnvironment

        adapter = HarborAdapter()
        tasks = _make_tasks()
        mock_env = MockHarborEnvironment()
        env_factory = _make_harbor_env_factory(mock_env)
        verifier_factory = _make_verifier_factory()

        env = adapter.get_environment(
            name="test",
            tasks=tasks,
            env_factory=env_factory,
            verify_factory=verifier_factory,
            tool_mode=True,
        )
        assert isinstance(env, HarborToolEnvironment)

    def test_tool_mode_rejects_runtime_probing(self):
        """Should raise ValueError when both tool_mode and runtime_probing are True."""
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        tasks = _make_tasks()
        mock_env = MockHarborEnvironment()
        env_factory = _make_harbor_env_factory(mock_env)
        verifier_factory = _make_verifier_factory()

        with pytest.raises(ValueError, match="not supported in tool mode"):
            adapter.get_environment(
                name="test",
                tasks=tasks,
                env_factory=env_factory,
                verify_factory=verifier_factory,
                tool_mode=True,
                runtime_probing=True,
            )

    def test_tool_mode_rejects_tmux_text_exec_mode(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        tasks = _make_tasks()
        mock_env = MockHarborEnvironment()
        env_factory = _make_harbor_env_factory(mock_env)
        verifier_factory = _make_verifier_factory()

        with pytest.raises(ValueError, match="tool mode"):
            adapter.get_environment(
                name="test",
                tasks=tasks,
                env_factory=env_factory,
                verify_factory=verifier_factory,
                tool_mode=True,
                text_exec_mode="tmux_session",
            )

    def test_tool_mode_rejects_command_soft_timeout_policy(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        tasks = _make_tasks()
        mock_env = MockHarborEnvironment()
        env_factory = _make_harbor_env_factory(mock_env)
        verifier_factory = _make_verifier_factory()

        with pytest.raises(ValueError, match="tool mode"):
            adapter.get_environment(
                name="test",
                tasks=tasks,
                env_factory=env_factory,
                verify_factory=verifier_factory,
                tool_mode=True,
                command_soft_timeout=120,
                command_timeout_budget=240,
                max_consecutive_command_timeouts=2,
            )

    def test_text_mode_requires_complete_command_soft_timeout_policy(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        tasks = _make_tasks()
        mock_env = MockHarborEnvironment()
        env_factory = _make_harbor_env_factory(mock_env)
        verifier_factory = _make_verifier_factory()

        with pytest.raises(ValueError, match="set together"):
            adapter.get_environment(
                name="test",
                tasks=tasks,
                env_factory=env_factory,
                verify_factory=verifier_factory,
                command_soft_timeout=120,
            )

    def test_list_environments_uses_registry_client(self, monkeypatch: pytest.MonkeyPatch):
        from llenvs.adapters.harbor import HarborAdapter, _HarborAPI

        client = MagicMock()
        client.get_datasets.return_value = [
            SimpleNamespace(name="terminal-bench", version="2.0"),
            SimpleNamespace(name="swe-bench", version="1.0"),
        ]
        api = _HarborAPI(
            registry_client_factory=SimpleNamespace(create=MagicMock(return_value=client)),
            task_client=object,
            task_class=object,
            task_paths_class=object,
            environment_factory=object,
            environment_type_enum=str,
            trial_paths_class=object,
            verifier_class=object,
        )
        monkeypatch.setattr(HarborAdapter, "_get_harbor_api", lambda self: api)

        adapter = HarborAdapter()
        assert adapter.list_environments() == ["swe-bench@1.0", "terminal-bench@2.0"]

    def test_get_environment_loads_local_tasks_from_dataset_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import HarborAdapter

        task_a = tmp_path / "task_a"
        task_b = tmp_path / "task_b"
        ignored = tmp_path / "notes"
        for path in (task_a, task_b, ignored):
            path.mkdir()

        class FakeTaskPaths:
            def __init__(self, path):
                self.path = path

            def is_valid(self, disable_verification: bool = False) -> bool:
                return self.path.name.startswith("task_")

        class FakeTask:
            def __init__(self, path):
                self.name = path.name
                self.instruction = f"Instruction for {path.name}"

        api = SimpleNamespace(task_paths_class=FakeTaskPaths, task_class=FakeTask)
        monkeypatch.setattr(HarborAdapter, "_get_harbor_api", lambda self: api)

        adapter = HarborAdapter()
        env = adapter.get_environment(
            name="local-dataset",
            dataset_path=str(tmp_path),
            env_factory=_make_harbor_env_factory(),
            verify_factory=_make_verifier_factory(),
        )

        assert [task.name for task in env._tasks] == ["task_a", "task_b"]

    def test_get_environment_builds_modern_env_and_verifier_factories(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from llenvs.adapters.harbor import HarborAdapter

        mock_env = MockHarborEnvironment()
        task = SimpleNamespace(
            name="task_01",
            instruction="Solve the task.",
            paths=SimpleNamespace(environment_dir="/tmp/task_01/environment"),
            config=SimpleNamespace(environment=SimpleNamespace()),
        )
        env_factory = MagicMock(return_value=mock_env)
        created_verifiers: list[Any] = []

        class FakeTrialPaths:
            def __init__(self, trial_dir):
                self.trial_dir = trial_dir
                self.mkdir_called = False

            def mkdir(self):
                self.mkdir_called = True

        class FakeVerifier:
            def __init__(self, task: Any, trial_paths: Any, environment: Any, logger: Any = None):
                self.task = task
                self.trial_paths = trial_paths
                self.environment = environment
                self.logger = logger
                created_verifiers.append(self)

            async def verify(self):
                return MockVerifierResult(rewards={"reward": 1.0})

        api = SimpleNamespace(
            environment_type_enum=lambda value: f"ENV:{value}",
            environment_factory=SimpleNamespace(create_environment=env_factory),
            trial_paths_class=FakeTrialPaths,
            verifier_class=FakeVerifier,
        )
        monkeypatch.setattr(HarborAdapter, "_get_harbor_api", lambda self: api)

        adapter = HarborAdapter()
        env = adapter.get_environment(name="terminal-bench@2.0", tasks=(task,))

        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.prompt == "Solve the task."
        assert mock_env._started is True
        assert mock_env._start_force_build is False

        env_call = env_factory.call_args
        assert env_call.kwargs["type"] == "ENV:docker"
        assert env_call.kwargs["environment_name"] == "task_01"
        assert env_call.kwargs["environment_dir"] == "/tmp/task_01/environment"
        assert env_call.kwargs["task_env_config"] is task.config.environment
        assert env_call.kwargs["trial_paths"].mkdir_called is True
        assert mock_env.trial_paths is env_call.kwargs["trial_paths"]

        result = env.step(state, Action(text="SUBMIT"))
        assert result.terminated is True
        assert result.next_state.metadata.info["reward"] == 1.0
        assert created_verifiers[0].task is task
        assert created_verifiers[0].environment is mock_env
        assert created_verifiers[0].trial_paths is mock_env.trial_paths

        env.close()
        assert mock_env._stopped is True
        assert mock_env._stop_delete is True

    def test_get_environment_builds_local_podman_hpc_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from llenvs.adapters.harbor import HarborAdapter

        mock_env = MockHarborEnvironment()
        task = SimpleNamespace(
            name="task_01",
            instruction="Solve the task.",
            paths=SimpleNamespace(environment_dir="/tmp/task_01/environment"),
            config=SimpleNamespace(
                environment=SimpleNamespace(
                    docker_image="ubuntu:latest",
                    cpus=1,
                    memory_mb=1024,
                    allow_internet=True,
                )
            ),
        )
        created_verifiers: list[Any] = []

        class FakeTrialPaths:
            def __init__(self, trial_dir):
                self.trial_dir = trial_dir
                self.mkdir_called = False

            def mkdir(self):
                self.mkdir_called = True

        class FakeVerifier:
            def __init__(self, task: Any, trial_paths: Any, environment: Any, logger: Any = None):
                self.task = task
                self.trial_paths = trial_paths
                self.environment = environment
                self.logger = logger
                created_verifiers.append(self)

            async def verify(self):
                return MockVerifierResult(rewards={"reward": 1.0})

        podman_cls = MagicMock(return_value=mock_env)
        monkeypatch.setattr("llenvs.adapters.harbor.PodmanHPCEnvironment", podman_cls)

        api = SimpleNamespace(
            environment_type_enum=lambda value: f"ENV:{value}",
            environment_factory=SimpleNamespace(create_environment=MagicMock()),
            trial_paths_class=FakeTrialPaths,
            verifier_class=FakeVerifier,
        )
        monkeypatch.setattr(HarborAdapter, "_get_harbor_api", lambda self: api)

        adapter = HarborAdapter()
        env = adapter.get_environment(
            name="terminal-bench@2.0",
            tasks=(task,),
            environment_type="podman-hpc",
        )

        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.prompt == "Solve the task."
        podman_cls.assert_called_once()
        call = podman_cls.call_args
        assert call.kwargs["environment_dir"] == Path("/tmp/task_01/environment")
        assert call.kwargs["environment_name"] == "task_01"
        assert call.kwargs["task_env_config"] is task.config.environment
        assert call.kwargs["trial_paths"].mkdir_called is True

        result = env.step(state, Action(text="SUBMIT"))
        assert result.terminated is True
        assert result.next_state.metadata.info["reward"] == 1.0
        assert created_verifiers[0].environment is mock_env

    def test_list_environments_requires_harbor(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
        def boom() -> Any:
            raise ImportError("harbor")

        adapter._get_harbor_api = boom  # type: ignore[assignment]

        with pytest.raises(ImportError):
            adapter.list_environments()


# ── TestHarborFullEpisode ───────────────────────────────────────


class TestHarborFullEpisode:
    def test_text_mode_full_episode(self):
        """Full episode: reset → steps → submit."""
        mock_env = MockHarborEnvironment(
            exec_results=[
                MockExecResult(stdout="secret.enc  key.txt"),
                MockExecResult(stdout="SuperSecretKey123"),
                MockExecResult(stdout="Decrypted: Hello World"),
            ]
        )
        verifier_result = MockVerifierResult(rewards={"reward": 1.0})
        env = _make_env(
            harbor_env=mock_env,
            verifier_result=verifier_result,
            max_steps=10,
        )

        state, info = _reset_env(env, task_index=0)
        assert state.metadata.step == 0

        # Step 1: ls
        result = env.step(state, Action(text="ls"))
        assert result.terminated is False
        state = result.next_state
        assert state.hidden.episode_step == 1

        # Step 2: cat key.txt
        result = env.step(state, Action(text="cat key.txt"))
        assert result.terminated is False
        state = result.next_state
        assert state.hidden.episode_step == 2

        # Step 3: decrypt
        result = env.step(state, Action(text="openssl enc -d -aes-256-cbc ..."))
        assert result.terminated is False
        state = result.next_state

        # Step 4: submit
        result = env.step(state, Action(text="SUBMIT"))
        assert result.terminated is True
        assert result.next_state.metadata.info.get("reward") == 1.0

        # Check trajectory
        h = result.next_state.hidden
        assert len(h.trajectory) == 4

    def test_tool_mode_full_episode(self):
        """Full episode using tool calls."""
        mock_env = MockHarborEnvironment(
            exec_results=[
                MockExecResult(stdout="secret.enc  key.txt"),
                MockExecResult(stdout="SuperSecretKey123"),
                MockExecResult(stdout="Decrypted: Hello World"),
            ]
        )
        verifier_result = MockVerifierResult(rewards={"reward": 1.0})
        env = _make_tool_env(
            harbor_env=mock_env,
            verifier_result=verifier_result,
            max_steps=10,
        )

        state, _ = _reset_env(env, task_index=0)

        # Step 1: ls
        call1 = ToolCall(
            id="c1",
            name="execute_command",
            arguments={"command": "ls"},
        )
        result = env.step(state, Action(tool_calls=(call1,)))
        state = result.next_state
        assert "secret.enc" in result.info["tool_results"][0].output

        # Step 2: read key
        call2 = ToolCall(
            id="c2",
            name="read_file",
            arguments={"path": "/key.txt"},
        )
        result = env.step(state, Action(tool_calls=(call2,)))
        state = result.next_state

        # Step 3: decrypt
        call3 = ToolCall(
            id="c3",
            name="execute_command",
            arguments={"command": "openssl enc -d ..."},
        )
        result = env.step(state, Action(tool_calls=(call3,)))
        state = result.next_state

        # Step 4: submit
        call4 = ToolCall(id="c4", name="submit", arguments={})
        result = env.step(state, Action(tool_calls=(call4,)))
        assert result.terminated is True
        assert result.next_state.metadata.info.get("reward") == 1.0

    def test_text_mode_truncation_episode(self):
        """Episode that hits max_steps."""
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")] * 10)
        verifier_result = MockVerifierResult(rewards={"reward": 0.0})
        env = _make_env(
            harbor_env=mock_env,
            verifier_result=verifier_result,
            max_steps=3,
            verify_on_truncation=True,
        )

        state, _ = _reset_env(env)
        for i in range(2):
            result = env.step(state, Action(text=f"cmd{i}"))
            state = result.next_state
            assert not result.truncated

        result = env.step(state, Action(text="cmd2"))
        assert result.truncated is True
        assert result.next_state.metadata.info.get("reward") == 0.0

    def test_reward_signals_on_submit(self):
        """Verify reward signals at terminal step."""
        verifier_result = MockVerifierResult(rewards={"reward": 1.0})
        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env, verifier_result=verifier_result)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="SUBMIT"))

        bundle = result.rewards
        outcome_signals = [s for s in bundle.signals if s.reward_type == RewardType.OUTCOME]
        assert len(outcome_signals) >= 1
        assert outcome_signals[0].reward == 1.0

    def test_reward_signals_non_terminal(self):
        """Non-terminal steps should have STEP signals with None reward."""
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="ok")])
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="ls"))

        bundle = result.rewards
        step_signals = [s for s in bundle.signals if s.reward_type == RewardType.STEP]
        assert len(step_signals) >= 1
        assert step_signals[0].reward is None

    def test_multiple_resets(self):
        """Resetting multiple times with different tasks."""
        env = _make_env(tasks=_make_tasks(5))

        state0, _ = _reset_env(env, task_index=0)
        assert state0.hidden.task_index == 0

        state2, _ = _reset_env(env, task_index=2)
        assert state2.hidden.task_index == 2

        state4, _ = _reset_env(env, task_index=4)
        assert state4.hidden.task_index == 4

    def test_write_file_tool_content_preserved(self):
        """Write file should send content to container."""
        mock_env = MockHarborEnvironment(exec_results=[MockExecResult(stdout="")])
        env = _make_tool_env(harbor_env=mock_env)
        state, _ = _reset_env(env)

        call = ToolCall(
            id="c1",
            name="write_file",
            arguments={"path": "/tmp/test.py", "content": "print('hello')"},
        )
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.info["tool_results"][0].is_success
        # Verify the write was sent via exec
        assert len(mock_env._exec_history) > 0


# ── Restore / replay validation tests ───────────────────────────


class TestHarborRestore:
    """Tests for harbor_restore() utility."""

    def test_harbor_restore_replays_trajectory(self):
        """harbor_restore calls reset + step for each command in the trajectory."""
        from llenvs.adapters.harbor import HarborHidden, harbor_restore

        exec_results = [
            MockExecResult(stdout="file1.txt"),
            MockExecResult(stdout="hello world"),
            MockExecResult(stdout="done"),
        ]
        mock_env = MockHarborEnvironment(exec_results=exec_results)
        env = _make_env(harbor_env=mock_env)

        # Build a state with trajectory to replay
        target_hidden = HarborHidden(
            task_index=0,
            task_name="task_00",
            instruction="Task 0 instruction",
            episode_step=3,
            last_action="cat result.txt",
            trajectory=("ls", "cat hello.txt", "cat result.txt"),
        )
        target_state = State(
            observation=MagicMock(),
            hidden=target_hidden,
            metadata=MagicMock(step=3, is_terminal=False),
        )

        restored = harbor_restore(env, target_state)

        # Should have executed all 3 commands from trajectory
        assert len(mock_env._exec_history) == 3
        assert mock_env._exec_history == ["ls", "cat hello.txt", "cat result.txt"]
        # Restored state should be at step 3
        assert restored.hidden.episode_step == 3

    def test_harbor_restore_task_name_mismatch(self):
        """harbor_restore raises ValueError on task name drift."""
        from llenvs.adapters.harbor import HarborHidden, harbor_restore

        env = _make_env()

        target_hidden = HarborHidden(
            task_index=0,
            task_name="wrong_task_name",
            instruction="Task instruction",
            episode_step=0,
            trajectory=(),
        )
        target_state = State(
            observation=MagicMock(),
            hidden=target_hidden,
            metadata=MagicMock(step=0, is_terminal=False),
        )

        with pytest.raises(ValueError, match="Task name mismatch"):
            harbor_restore(env, target_state)

    def test_harbor_restore_empty_trajectory(self):
        """harbor_restore with empty trajectory just resets."""
        from llenvs.adapters.harbor import HarborHidden, harbor_restore

        env = _make_env()

        target_hidden = HarborHidden(
            task_index=0,
            task_name="task_00",
            instruction="Task 0 instruction",
            episode_step=0,
            trajectory=(),
        )
        target_state = State(
            observation=MagicMock(),
            hidden=target_hidden,
            metadata=MagicMock(step=0, is_terminal=False),
        )

        restored = harbor_restore(env, target_state)
        assert restored.hidden.episode_step == 0
        assert restored.hidden.trajectory == ()

    def test_harbor_restore_uses_hard_timeout_when_soft_policy_enabled(self):
        """Replay should bypass recoverable soft timeouts and use exec_timeout."""
        from llenvs.adapters.harbor import HarborHidden, harbor_restore

        timeout_values: list[int] = []

        def handler(command: str, timeout_sec: int = 120, **_: Any) -> MockExecResult:
            timeout_values.append(timeout_sec)
            return MockExecResult(stdout=f"ran {command}")

        env = _make_env(
            harbor_env=MockHarborEnvironment(exec_handler=handler),
            exec_timeout=17,
            command_soft_timeout=5,
            command_timeout_budget=20,
            max_consecutive_command_timeouts=2,
        )

        target_state = State(
            observation=MagicMock(),
            hidden=HarborHidden(
                task_index=0,
                task_name="task_00",
                instruction="Task 0 instruction",
                episode_step=2,
                last_action="pwd",
                trajectory=("ls", "pwd"),
            ),
            metadata=MagicMock(step=2, episode_id="episode-1", is_terminal=False),
        )

        restored = harbor_restore(env, target_state)

        assert restored.hidden.episode_step == 2
        assert timeout_values == [17, 17]

    def test_harbor_snapshot_restore_restores_checkpoint_archive(self, tmp_path):
        """harbor_snapshot_restore uses checkpoint restore instead of replay."""
        from llenvs.adapters.harbor import (
            HarborHidden,
            HarborSnapshotOptions,
            HarborSnapshotRef,
            harbor_snapshot_restore,
        )

        snapshot_root = tmp_path / "dataset_root"
        snapshot_path = snapshot_root / "snapshots/task_00/episode-1/state_0001.tar"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("checkpoint-bytes")

        mock_env = MockHarborEnvironment()
        env = _make_env(harbor_env=mock_env)

        target_state = State(
            observation=MagicMock(),
            hidden=HarborHidden(
                task_index=0,
                task_name="task_00",
                instruction="Task 0 instruction",
                episode_step=1,
                trajectory=("ls",),
                snapshot_ref=HarborSnapshotRef(
                    runtime="podman-hpc",
                    relative_path="snapshots/task_00/episode-1/state_0001.tar",
                    options=HarborSnapshotOptions(file_locks=True),
                ),
            ),
            metadata=MagicMock(step=1, episode_id="episode-1", is_terminal=False),
        )

        restored = harbor_snapshot_restore(
            env,
            target_state,
            artifact_root=snapshot_root,
        )

        assert restored is target_state
        assert mock_env._exec_history == []
        assert mock_env._checkpoint_restores == [
            (
                snapshot_path,
                {
                    "file_locks": True,
                    "tcp_established": False,
                    "tcp_close": False,
                    "ignore_volumes": False,
                },
            )
        ]

    def test_harbor_snapshot_restore_requires_snapshot_ref(self, tmp_path):
        from llenvs.adapters.harbor import HarborHidden, harbor_snapshot_restore

        env = _make_env()
        target_state = State(
            observation=MagicMock(),
            hidden=HarborHidden(
                task_index=0,
                task_name="task_00",
                instruction="Task 0 instruction",
                episode_step=0,
            ),
            metadata=MagicMock(step=0, episode_id="episode-1", is_terminal=False),
        )

        with pytest.raises(ValueError, match="snapshot_ref"):
            harbor_snapshot_restore(env, target_state, artifact_root=tmp_path)

    def test_harbor_snapshot_restore_resyncs_tmux_session_buffer(self, tmp_path):
        from llenvs.adapters.harbor import (
            HarborHidden,
            HarborSnapshotRef,
            harbor_snapshot_restore,
        )

        snapshot_root = tmp_path / "dataset_root"
        snapshot_path = snapshot_root / "snapshots/task_00/episode-1/state_0001.tar"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("checkpoint-bytes")

        runtime = _FakeTmuxRuntime(full_buffers=["bash$ ", "restored$ "])
        mock_env = MockHarborEnvironment(exec_handler=runtime)
        env = _make_env(harbor_env=mock_env, text_exec_mode="tmux_session")
        _reset_env(env)

        target_state = State(
            observation=MagicMock(),
            hidden=HarborHidden(
                task_index=0,
                task_name="task_00",
                instruction="Task 0 instruction",
                episode_step=1,
                trajectory=("pwd",),
                snapshot_ref=HarborSnapshotRef(
                    runtime="podman-hpc",
                    relative_path="snapshots/task_00/episode-1/state_0001.tar",
                ),
            ),
            metadata=MagicMock(step=1, episode_id="episode-1", is_terminal=False),
        )

        restored = harbor_snapshot_restore(
            env,
            target_state,
            artifact_root=snapshot_root,
        )

        assert restored is target_state
        assert env._text_session is not None
        assert env._text_session._previous_full_buffer == "restored$ "


class TestValidateReplayConsistency:
    """Tests for validate_replay_consistency() utility."""

    def test_capture_replay_probe_outputs_returns_probe_stdout(self):
        from llenvs.adapters.harbor import capture_replay_probe_outputs

        def env_factory():
            mock_env = MockHarborEnvironment(
                exec_results=[
                    MockExecResult(stdout="ok"),  # trajectory command
                    MockExecResult(stdout="abc123"),  # probe 1
                    MockExecResult(stdout="def456"),  # probe 2
                ]
            )
            return _make_env(harbor_env=mock_env)

        result = capture_replay_probe_outputs(
            env_factory=env_factory,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1", "probe2"),
        )

        assert result == {"probe1": "abc123", "probe2": "def456"}

    def test_consistent_replays(self):
        """Deterministic env produces consistent=True."""
        from llenvs.adapters.harbor import validate_replay_consistency

        def env_factory():
            # Each factory call returns an env with deterministic outputs
            mock_env = MockHarborEnvironment(
                exec_results=[
                    MockExecResult(stdout="ok"),  # trajectory command
                    MockExecResult(stdout="abc123"),  # probe 1
                    MockExecResult(stdout="def456"),  # probe 2
                ]
            )
            return _make_env(harbor_env=mock_env)

        result = validate_replay_consistency(
            env_factory=env_factory,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1", "probe2"),
            num_trials=3,
        )

        assert result["consistent"] is True
        assert result["matches_reference"] is None
        assert len(result["probe_outputs"]) == 3
        assert result["divergence_details"] == []

    def test_divergent_replays(self):
        """Non-deterministic env produces consistent=False."""
        from llenvs.adapters.harbor import validate_replay_consistency

        call_count = [0]

        def env_factory():
            call_count[0] += 1
            # Vary output between trials
            probe_output = f"hash_{call_count[0]}"
            mock_env = MockHarborEnvironment(
                exec_results=[
                    MockExecResult(stdout="ok"),
                    MockExecResult(stdout=probe_output),
                ]
            )
            return _make_env(harbor_env=mock_env)

        result = validate_replay_consistency(
            env_factory=env_factory,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1",),
            num_trials=3,
        )

        assert result["consistent"] is False
        assert len(result["divergence_details"]) > 0

    def test_with_reference_probes_match(self):
        """Reference probes that match produce matches_reference=True."""
        from llenvs.adapters.harbor import validate_replay_consistency

        def env_factory():
            mock_env = MockHarborEnvironment(
                exec_results=[
                    MockExecResult(stdout="ok"),
                    MockExecResult(stdout="expected_hash"),
                ]
            )
            return _make_env(harbor_env=mock_env)

        result = validate_replay_consistency(
            env_factory=env_factory,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1",),
            reference_probes={"probe1": "expected_hash"},
            num_trials=2,
        )

        assert result["consistent"] is True
        assert result["matches_reference"] is True

    def test_with_reference_probes_mismatch(self):
        """Reference probes that don't match produce matches_reference=False."""
        from llenvs.adapters.harbor import validate_replay_consistency

        def env_factory():
            mock_env = MockHarborEnvironment(
                exec_results=[
                    MockExecResult(stdout="ok"),
                    MockExecResult(stdout="actual_hash"),
                ]
            )
            return _make_env(harbor_env=mock_env)

        result = validate_replay_consistency(
            env_factory=env_factory,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1",),
            reference_probes={"probe1": "different_hash"},
            num_trials=1,
        )

        assert result["matches_reference"] is False
        assert any("Reference mismatch" in d for d in result["divergence_details"])

    def test_replay_probe_capture_does_not_trigger_verifier_on_near_horizon_state(self):
        from llenvs.adapters.harbor import validate_replay_consistency

        verifier_calls = 0

        class CountingVerifier:
            async def verify(self):
                nonlocal verifier_calls
                verifier_calls += 1
                return MockVerifierResult()

        def verifier_factory(task: Any, env: Any) -> CountingVerifier:
            return CountingVerifier()

        def env_factory():
            mock_env = MockHarborEnvironment(
                exec_results=[
                    MockExecResult(stdout="ok"),  # trajectory command
                    MockExecResult(stdout="probe_hash"),  # probe command
                ]
            )
            return _make_env(
                harbor_env=mock_env,
                verifier_factory=verifier_factory,
                max_steps=2,
                verify_on_truncation=True,
            )

        result = validate_replay_consistency(
            env_factory=env_factory,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1",),
            num_trials=1,
        )

        assert result["consistent"] is True
        assert verifier_calls == 0

    def test_replay_validation_uses_hard_timeout_when_soft_policy_enabled(self):
        from llenvs.adapters.harbor import validate_replay_consistency

        timeout_values: list[int] = []

        def make_env():
            def handler(command: str, timeout_sec: int = 120, **_: Any) -> MockExecResult:
                timeout_values.append(timeout_sec)
                return MockExecResult(stdout=f"output for {command}")

            return _make_env(
                harbor_env=MockHarborEnvironment(exec_handler=handler),
                exec_timeout=17,
                command_soft_timeout=5,
                command_timeout_budget=20,
                max_consecutive_command_timeouts=2,
            )

        result = validate_replay_consistency(
            env_factory=make_env,
            task_index=0,
            trajectory=("echo ok",),
            probe_commands=("probe1",),
            num_trials=1,
        )

        assert result["consistent"] is True
        assert timeout_values == [17, 17]


class TestHarborHpcCliRunner:
    def test_shared_cli_runner_times_out_and_reaps_after_sigterm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ):
        import llenvs.adapters.harbor as harbor_module

        process = _FakeTimeoutThenExitProcess()
        popen_kwargs: dict[str, Any] = {}
        kill_signals: list[tuple[int, signal.Signals]] = []

        def fake_popen(*args: Any, **kwargs: Any) -> _FakeTimeoutThenExitProcess:
            del args
            popen_kwargs.update(kwargs)
            return process

        def fake_killpg(pid: int, sig: signal.Signals) -> None:
            kill_signals.append((pid, sig))

        monkeypatch.setattr(harbor_module.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(harbor_module.os, "killpg", fake_killpg)

        with caplog.at_level(logging.DEBUG, logger="llenvs.adapters.harbor"):
            with pytest.raises(RuntimeError, match=r"apptainer command timed out after 3s"):
                harbor_module._run_hpc_cli_command(
                    ["apptainer", "exec", "instance://test", "bash", "-lc", "sleep 999"],
                    cwd=tmp_path,
                    env={"PATH": os.environ.get("PATH", "")},
                    check=False,
                    timeout_sec=3,
                    runtime_label="apptainer",
                    logger=logging.getLogger("llenvs.adapters.harbor"),
                )

        assert popen_kwargs["cwd"] == str(tmp_path)
        assert popen_kwargs["start_new_session"] is True
        assert process.communicate_timeouts == [3, 5]
        assert kill_signals == [(process.pid, signal.SIGTERM)]
        messages = [record.getMessage() for record in caplog.records]
        assert any("apptainer cmd[" in message and "timeout after" in message for message in messages)
        assert any("sent SIGTERM" in message for message in messages)
        assert any("reaped after SIGTERM" in message for message in messages)

    def test_shared_cli_runner_raises_cleanup_failure_after_sigkill(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ):
        import llenvs.adapters.harbor as harbor_module

        process = _FakeNeverReapProcess()
        kill_signals: list[tuple[int, signal.Signals]] = []

        def fake_popen(*args: Any, **kwargs: Any) -> _FakeNeverReapProcess:
            del args, kwargs
            return process

        def fake_killpg(pid: int, sig: signal.Signals) -> None:
            kill_signals.append((pid, sig))

        monkeypatch.setattr(harbor_module.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(harbor_module.os, "killpg", fake_killpg)

        with caplog.at_level(logging.DEBUG, logger="llenvs.adapters.harbor"):
            with pytest.raises(
                RuntimeError,
                match=r"podman-hpc command timed out after 4s and cleanup failed after SIGKILL",
            ):
                harbor_module._run_hpc_cli_command(
                    ["podman-hpc", "exec", "test", "bash", "-lc", "sleep 999"],
                    cwd=tmp_path,
                    env={"PATH": os.environ.get("PATH", "")},
                    check=False,
                    timeout_sec=4,
                    runtime_label="podman-hpc",
                    logger=logging.getLogger("llenvs.adapters.harbor"),
                )

        assert process.communicate_timeouts == [4, 5, 5]
        assert kill_signals == [
            (process.pid, signal.SIGTERM),
            (process.pid, signal.SIGKILL),
        ]
        messages = [record.getMessage() for record in caplog.records]
        assert any("sent SIGTERM" in message for message in messages)
        assert any("sent SIGKILL" in message for message in messages)
        assert any("cleanup failed after SIGKILL" in message for message in messages)


class TestPodmanHPCEnvironment:
    def _make_trial_paths(self, tmp_path):
        verifier_dir = tmp_path / "verifier"
        agent_dir = tmp_path / "agent"
        artifacts_dir = tmp_path / "artifacts"
        for path in (verifier_dir, agent_dir, artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            trial_dir=tmp_path,
            verifier_dir=verifier_dir,
            agent_dir=agent_dir,
            artifacts_dir=artifacts_dir,
        )

    def _make_task_env_config(self, **kwargs: Any):
        defaults = {
            "docker_image": "ubuntu:latest",
            "cpus": 1,
            "memory_mb": 1024,
            "allow_internet": True,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_rejects_compose_without_main_service(self, tmp_path):
        from llenvs.adapters.harbor import PodmanHPCEnvironment

        (tmp_path / "docker-compose.yaml").write_text("services:\n  db:\n    image: postgres:latest\n")

        with pytest.raises(ValueError, match="main"):
            PodmanHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id="session-1",
                trial_paths=self._make_trial_paths(tmp_path / "trial"),
                task_env_config=self._make_task_env_config(),
            )

    def test_start_uses_migrate_run_and_bootstrap_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd: list[str], *, check: bool = True, timeout_sec: int | None = None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_podman_command", fake_run)

        run_async(env.start(force_build=False))

        assert env.is_mounted is False
        assert calls[0][0] == [
            "podman-hpc",
            "migrate",
            "docker://ubuntu:latest",
        ]
        assert calls[1][0][:5] == [
            "podman-hpc",
            "run",
            "-d",
            "--name",
            env._container_name,
        ]
        assert calls[2][0] == [
            "podman-hpc",
            "exec",
            env._container_name,
            "bash",
            "-lc",
            "mkdir -p /logs/agent /logs/verifier",
        ]

    def test_start_compose_creates_network_and_starts_services(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "app").mkdir()
        (tmp_path / "docker-compose.yaml").write_text(
            """
services:
  main:
    image: ubuntu:latest
    command: ["sleep", "infinity"]
    depends_on:
      - db
    working_dir: /workspace
    environment:
      FOO: bar
    volumes:
      - ./app:/workspace:ro
      - cache:/cache
  db:
    image: postgres:latest
    healthcheck:
      test: ["CMD", "true"]
volumes:
  cache: {}
""".strip()
        )
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(docker_image=None),
        )

        calls: list[tuple[list[str], bool, int | None]] = []
        waited: list[str] = []

        async def fake_run(cmd: list[str], *, check: bool = True, timeout_sec: int | None = None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        async def fake_wait(service_name: str):
            waited.append(service_name)

        monkeypatch.setattr(env, "_run_podman_command", fake_run)
        monkeypatch.setattr(env, "_wait_for_service_health", fake_wait)

        run_async(env.start(force_build=False))

        assert calls[0][0] == ["podman-hpc", "network", "create", env._network_name]
        assert calls[1][0] == ["podman-hpc", "migrate", "docker://postgres:latest"]
        assert calls[2][0] == ["podman-hpc", "migrate", "docker://ubuntu:latest"]
        db_run_idx, db_run = next(
            (i, cmd)
            for i, (cmd, _check, _timeout) in enumerate(calls)
            if "--name" in cmd and "session-1-db" in cmd
        )
        main_run_idx, main_run = next(
            (i, cmd)
            for i, (cmd, _check, _timeout) in enumerate(calls)
            if "--name" in cmd and "session-1-main" in cmd
        )
        assert db_run_idx < main_run_idx
        assert "--network" in main_run
        assert env._network_name in main_run
        assert "--network-alias" in main_run
        assert "main" in main_run
        assert "-w" in main_run
        assert "/workspace" in main_run
        assert "-e" in main_run
        assert "FOO=bar" in main_run
        assert any(part.endswith(":/workspace:ro") for part in main_run)
        assert any(part.endswith(":/cache") for part in main_run)
        assert waited == ["db"]
        assert calls[-1][0] == [
            "podman-hpc",
            "exec",
            "session-1-main",
            "bash",
            "-lc",
            "mkdir -p /logs/agent /logs/verifier",
        ]

    def test_exec_targets_main_service_for_compose(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "docker-compose.yaml").write_text(
            """
services:
  main:
    image: ubuntu:latest
    command: ["sleep", "infinity"]
  db:
    image: postgres:latest
""".strip()
        )
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(docker_image=None),
        )

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd: list[str], *, check: bool = True, timeout_sec: int | None = None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_podman_command", fake_run)
        env._started = True

        result = run_async(env.exec("pwd"))

        assert result.stdout == "ok"
        assert calls[0][0] == [
            "podman-hpc",
            "exec",
            "session-1-main",
            "bash",
            "-lc",
            "pwd",
        ]

    def test_exec_respects_cwd_env_and_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd: list[str], *, check: bool = True, timeout_sec: int | None = None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_podman_command", fake_run)
        env._started = True

        result = run_async(
            env.exec(
                "pwd",
                cwd="/workspace",
                env={"FOO": "bar"},
                timeout_sec=17,
            )
        )

        assert result.stdout == "ok"
        assert calls[0][0] == [
            "podman-hpc",
            "exec",
            "-w",
            "/workspace",
            "-e",
            "FOO=bar",
            env._container_name,
            "bash",
            "-lc",
            "pwd",
        ]
        assert calls[0][2] == 17

    def test_run_podman_command_uses_shared_cli_runner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_module
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )

        recorded: dict[str, Any] = {}

        def fake_helper(
            cmd: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            check: bool,
            timeout_sec: int | None,
            runtime_label: str,
            logger: logging.Logger,
        ) -> MockExecResult:
            recorded.update(
                {
                    "cmd": cmd,
                    "cwd": cwd,
                    "env": env,
                    "check": check,
                    "timeout_sec": timeout_sec,
                    "runtime_label": runtime_label,
                    "logger_name": logger.name,
                }
            )
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(harbor_module, "_run_hpc_cli_command", fake_helper)

        result = run_async(
            env._run_podman_command(
                ["podman-hpc", "exec", "container", "bash", "-lc", "pwd"],
                check=False,
                timeout_sec=9,
            )
        )

        assert result.stdout == "ok"
        assert recorded["cmd"] == ["podman-hpc", "exec", "container", "bash", "-lc", "pwd"]
        assert recorded["cwd"] == tmp_path
        assert recorded["check"] is False
        assert recorded["timeout_sec"] == 9
        assert recorded["runtime_label"] == "podman-hpc"
        assert recorded["logger_name"] == "llenvs.adapters.harbor"
        assert "PATH" in recorded["env"]

    def test_export_checkpoint_uses_container_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd: list[str], *, check: bool = True, timeout_sec: int | None = None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_podman_command", fake_run)
        env._started = True

        export_path = tmp_path / "checkpoint.tar"
        run_async(
            env.export_checkpoint(
                export_path,
                file_locks=True,
                tcp_established=True,
            )
        )

        assert calls[0][0] == [
            "podman-hpc",
            "container",
            "checkpoint",
            "--export",
            str(export_path),
            "--compress",
            "none",
            "--leave-running",
            "--file-locks",
            "--tcp-established",
            env._container_name,
        ]

    def test_restore_checkpoint_uses_import_keep_and_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import PodmanHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = PodmanHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd: list[str], *, check: bool = True, timeout_sec: int | None = None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_podman_command", fake_run)

        import_path = tmp_path / "checkpoint.tar"
        run_async(
            env.restore_checkpoint(
                import_path,
                file_locks=True,
                tcp_established=True,
            )
        )

        assert calls[0][0] == [
            "podman-hpc",
            "container",
            "restore",
            "--import",
            str(import_path),
            "--name",
            env._container_name,
            "--keep",
            "--file-locks",
            "--tcp-established",
        ]


class TestApptainerHPCEnvironment:
    def _make_trial_paths(self, tmp_path):
        for name in ("verifier", "agent", "artifacts"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            trial_dir=tmp_path,
            verifier_dir=tmp_path / "verifier",
            agent_dir=tmp_path / "agent",
            artifacts_dir=tmp_path / "artifacts",
        )

    def _make_task_env_config(self, **kwargs: Any):
        defaults = {
            "docker_image": "ubuntu:latest",
            "cpus": 1,
            "memory_mb": 1024,
            "allow_internet": True,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_rejects_compose_tasks(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        (tmp_path / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ubuntu:latest\n"
        )
        with pytest.raises(NotImplementedError, match="Compose"):
            ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id="session-1",
                trial_paths=self._make_trial_paths(tmp_path / "trial"),
                task_env_config=self._make_task_env_config(),
            )

    def test_rejects_missing_container_source(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        with pytest.raises(FileNotFoundError, match="Dockerfile"):
            ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id="session-1",
                trial_paths=self._make_trial_paths(tmp_path / "trial"),
                task_env_config=self._make_task_env_config(docker_image=None),
            )

    def test_singularity_hpc_normalizes_to_apptainer_hpc(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        assert env.snapshot_runtime == "apptainer-hpc"

    def test_is_mounted_is_true(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        assert env.is_mounted is True

    def test_start_fails_fast_if_sif_missing(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(tmp_path / "sif_cache"),
        )
        with pytest.raises(FileNotFoundError, match="SIF"):
            run_async(env.start())

    def test_start_fails_if_allow_internet_false(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(allow_internet=False),
            sif_cache_dir=str(sif_dir),
        )
        # Create the SIF file so we get past that check
        env._sif_path.touch()
        with pytest.raises(RuntimeError, match="network isolation"):
            run_async(env.start())

    def test_start_calls_overlay_create_and_instance_start(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_RUNTIME_INFO_LOGGED_KEYS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            overlay_size_mb=256,
            rootfs_mode="overlay",
        )
        env._sif_path.touch()

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        seed_dir = tmp_path / "seed-cache" / "task_01"
        seed_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(env, "_prepare_app_seed_dir", lambda: seed_dir)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        commands = [cmd for cmd, _check, _timeout in calls]
        assert ["apptainer", "--version"] in commands

        overlay_cmd = next(cmd for cmd in commands if cmd[:3] == ["apptainer", "overlay", "create"])
        assert "--size" in overlay_cmd
        assert "256" in overlay_cmd

        inst_cmd = next(
            cmd for cmd in commands
            if cmd[:3] == ["apptainer", "instance", "start"]
        )
        assert inst_cmd[:3] == ["apptainer", "instance", "start"]
        assert "--overlay" in inst_cmd
        assert "--cleanenv" in inst_cmd
        assert "--contain" in inst_cmd
        assert "--no-home" in inst_cmd
        assert "--bind" in inst_cmd
        assert f"{trial / 'binds' / 'app'}:/app" in inst_cmd
        assert f"{trial / 'binds' / 'tests'}:/tests" in inst_cmd
        assert f"instance://" not in " ".join(inst_cmd)  # instance start uses plain name
        assert env._instance_name == inst_cmd[-1]

        bootstrap_cmd = next(
            cmd for cmd in commands
            if cmd[:2] == ["apptainer", "exec"]
            and "mkdir -p /logs/agent /logs/verifier" in cmd[-1]
        )
        assert "apptainer" == bootstrap_cmd[0]
        assert "exec" == bootstrap_cmd[1]
        assert "--cleanenv" in bootstrap_cmd
        assert f"instance://{env._instance_name}" in bootstrap_cmd
        assert "mkdir -p /logs/agent /logs/verifier" in bootstrap_cmd

        probe_cmd = next(
            cmd for cmd in commands
            if cmd[:2] == ["apptainer", "exec"]
            and "touch /.vb_probe && rm /.vb_probe" in cmd[-1]
        )
        assert probe_cmd[:2] == ["apptainer", "exec"]
        assert f"instance://{env._instance_name}" in probe_cmd
        assert "touch /.vb_probe && rm /.vb_probe" in probe_cmd[-1]

        assert env._started is True

    def test_start_seeds_trial_app_and_tests_binds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="overlay",
        )
        env._sif_path.touch()

        seed_calls: list[Path] = []

        def fake_prepare_seed() -> Path:
            seed_dir = tmp_path / "seed-cache" / "task_01"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "README.txt").write_text("seeded")
            seed_calls.append(seed_dir)
            return seed_dir

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_prepare_app_seed_dir", fake_prepare_seed)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        assert seed_calls == [tmp_path / "seed-cache" / "task_01"]
        assert (trial / "binds" / "app" / "README.txt").read_text() == "seeded"
        assert (trial / "binds" / "tests").is_dir()

    def test_upload_dir_to_tests_uses_bound_host_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
        )
        env._started = True
        env._active_rootfs_mode = "overlay"
        env._tests_bind_dir.mkdir(parents=True, exist_ok=True)

        source_dir = tmp_path / "tests"
        source_dir.mkdir()
        (source_dir / "test_a.sh").write_text("echo ok")

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.upload_dir(source_dir, "/tests"))

        assert calls == []
        assert (env._tests_bind_dir / "test_a.sh").read_text() == "echo ok"

    def test_start_with_fakeroot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_RUNTIME_INFO_LOGGED_KEYS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            fakeroot=True,
            rootfs_mode="overlay",
        )
        env._sif_path.touch()

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        seed_dir = tmp_path / "seed-cache" / "task_01"
        seed_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(env, "_prepare_app_seed_dir", lambda: seed_dir)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        inst_cmd = next(
            cmd for cmd, _check, _timeout in calls
            if cmd[:3] == ["apptainer", "instance", "start"]
        )
        assert "--fakeroot" in inst_cmd

    def test_start_sandbox_uses_writable_rootfs_without_app_or_tests_binds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_RUNTIME_INFO_LOGGED_KEYS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="sandbox",
        )
        env._sif_path.touch()

        rootfs_dir = trial / "rootfs"

        def fake_prepare_trial_rootfs() -> Path:
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_prepare_trial_rootfs", fake_prepare_trial_rootfs)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        inst_cmd = next(
            cmd for cmd, _check, _timeout in calls
            if cmd[:3] == ["apptainer", "instance", "start"]
        )
        assert inst_cmd[:3] == ["apptainer", "instance", "start"]
        assert "--writable" in inst_cmd
        assert "--overlay" not in inst_cmd
        assert "--writable-tmpfs" not in inst_cmd
        assert all(":/app" not in arg for arg in inst_cmd)
        assert all(":/tests" not in arg for arg in inst_cmd)
        assert str(rootfs_dir) in inst_cmd
        assert env._active_rootfs_mode == "sandbox"

    def test_start_auto_falls_back_to_sandbox_when_overlay_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_ROOTFS_PROBE_CACHE.clear()
        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
        )
        env._sif_path.touch()

        rootfs_dir = trial / "rootfs"

        def fake_prepare_seed() -> Path:
            seed_dir = tmp_path / "seed-cache" / "task_01"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "README.txt").write_text("seeded")
            return seed_dir

        def fake_prepare_trial_rootfs() -> Path:
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        async def fake_probe() -> bool:
            return False

        monkeypatch.setattr(env, "_prepare_app_seed_dir", fake_prepare_seed)
        monkeypatch.setattr(env, "_prepare_trial_rootfs", fake_prepare_trial_rootfs)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        monkeypatch.setattr(env, "_probe_root_writability", fake_probe)
        run_async(env.start())

        inst_cmds = [cmd for cmd, _check, _timeout in calls if cmd[:3] == ["apptainer", "instance", "start"]]
        assert len(inst_cmds) == 2
        assert "--overlay" in inst_cmds[0] or "--writable-tmpfs" in inst_cmds[0]
        assert "--writable" in inst_cmds[1]
        assert str(rootfs_dir) in inst_cmds[1]
        assert env._active_rootfs_mode == "sandbox"

    def test_auto_reuses_cached_overlay_probe_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_ROOTFS_PROBE_CACHE.clear()
        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        sif_path = sif_dir / "ca0c1413bbd82bab.sif"
        sif_path.touch()

        def make_env(trial_name: str) -> ApptainerHPCEnvironment:
            env = ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id=f"session-{trial_name}",
                trial_paths=self._make_trial_paths(tmp_path / trial_name),
                task_env_config=self._make_task_env_config(),
                sif_cache_dir=str(sif_dir),
            )
            env._sif_path = sif_path
            return env

        env_a = make_env("trial-a")
        env_b = make_env("trial-b")

        def fake_prepare_seed() -> Path:
            seed_dir = tmp_path / "seed-cache" / "task_01"
            seed_dir.mkdir(parents=True, exist_ok=True)
            return seed_dir

        def rootfs_for(env: ApptainerHPCEnvironment) -> Path:
            return Path(env.trial_paths.trial_dir) / "rootfs"

        calls_a: list[tuple[list[str], bool, int | None]] = []
        calls_b: list[tuple[list[str], bool, int | None]] = []

        async def fake_run_a(cmd, *, check=True, timeout_sec=None):
            calls_a.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        async def fake_run_b(cmd, *, check=True, timeout_sec=None):
            calls_b.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        async def fake_probe() -> bool:
            return False

        monkeypatch.setattr(env_a, "_prepare_app_seed_dir", fake_prepare_seed)
        monkeypatch.setattr(env_b, "_prepare_app_seed_dir", fake_prepare_seed)
        monkeypatch.setattr(env_a, "_prepare_trial_rootfs", lambda: rootfs_for(env_a))
        monkeypatch.setattr(env_b, "_prepare_trial_rootfs", lambda: rootfs_for(env_b))
        monkeypatch.setattr(env_a, "_run_apptainer_command", fake_run_a)
        monkeypatch.setattr(env_b, "_run_apptainer_command", fake_run_b)
        monkeypatch.setattr(env_a, "_probe_root_writability", fake_probe)

        run_async(env_a.start())
        run_async(env_b.start())

        inst_cmds_a = [cmd for cmd, _check, _timeout in calls_a if cmd[:3] == ["apptainer", "instance", "start"]]
        inst_cmds_b = [cmd for cmd, _check, _timeout in calls_b if cmd[:3] == ["apptainer", "instance", "start"]]
        assert len(inst_cmds_a) == 2
        assert len(inst_cmds_b) == 1
        assert "--writable" in inst_cmds_b[0]

    def test_pid_support_probe_cached_across_instances(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_PID_FLAG_CACHE.clear()
        harbor_mod._APPTAINER_PID_FLAG_EVENTS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()

        def make_env(name: str) -> ApptainerHPCEnvironment:
            env = ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id=f"session-{name}",
                trial_paths=self._make_trial_paths(tmp_path / name),
                task_env_config=self._make_task_env_config(),
                sif_cache_dir=str(sif_dir),
                rootfs_mode="sandbox",
                pid_namespace=True,
            )
            env._sif_path.touch(exist_ok=True)
            return env

        env_a = make_env("trial-a")
        env_b = make_env("trial-b")

        help_calls = 0

        async def quiet_runtime_info() -> None:
            return None

        def prepare_rootfs(env: ApptainerHPCEnvironment) -> Path:
            rootfs_dir = Path(env.trial_paths.trial_dir) / "rootfs"
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            nonlocal help_calls
            if cmd == ["apptainer", "--version"]:
                return MockExecResult(stdout="singularity-ce version 4.2.2-1.el8")
            if cmd == ["apptainer", "instance", "start", "--help"]:
                help_calls += 1
                return MockExecResult(stdout="  --containall\n  --pid-file\n")
            return MockExecResult(stdout="ok")

        for env in (env_a, env_b):
            monkeypatch.setattr(env, "_log_runtime_info", quiet_runtime_info)
            monkeypatch.setattr(env, "_prepare_trial_rootfs", lambda e=env: prepare_rootfs(e))
            monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        with caplog.at_level("INFO"):
            run_async(env_a.start())
            run_async(env_b.start())

        assert help_calls == 1
        assert [
            record.getMessage()
            for record in caplog.records
            if "Runtime lacks --pid flag" in record.getMessage()
        ] == ["Runtime lacks --pid flag; using --containall for PID namespace"]

    def test_runtime_info_logged_once_per_runtime_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_RUNTIME_INFO_LOGGED_KEYS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()

        def make_env(name: str) -> ApptainerHPCEnvironment:
            env = ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id=f"session-{name}",
                trial_paths=self._make_trial_paths(tmp_path / name),
                task_env_config=self._make_task_env_config(),
                sif_cache_dir=str(sif_dir),
            )
            env._sif_path.touch(exist_ok=True)
            return env

        env_a = make_env("trial-a")
        env_b = make_env("trial-b")

        version_calls = 0

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            nonlocal version_calls
            if cmd == ["apptainer", "--version"]:
                version_calls += 1
                return MockExecResult(stdout="singularity-ce version 4.2.2-1.el8")
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env_a, "_run_apptainer_command", fake_run)
        monkeypatch.setattr(env_b, "_run_apptainer_command", fake_run)

        with caplog.at_level("INFO"):
            run_async(env_a._log_runtime_info())
            run_async(env_b._log_runtime_info())

        assert version_calls == 1
        assert [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Harbor runtime:")
        ] == [
            "Harbor runtime: apptainer-hpc (apptainer singularity-ce version 4.2.2-1.el8)\n"
            "  fakeroot: disabled\n"
            "  rootfs mode request: auto\n"
            "  overlay mode: disk-backed overlay (512 MB)\n"
            f"  SIF cache: {sif_dir} (1 images)\n"
            "  isolation: --cleanenv --contain --no-home"
        ]

    def test_cached_overlay_probe_failure_is_silent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_ROOTFS_PROBE_CACHE.clear()
        harbor_mod._APPTAINER_ROOTFS_PROBE_EVENTS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        sif_path = sif_dir / "ca0c1413bbd82bab.sif"
        sif_path.touch()

        def make_env(name: str) -> ApptainerHPCEnvironment:
            env = ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id=f"session-{name}",
                trial_paths=self._make_trial_paths(tmp_path / name),
                task_env_config=self._make_task_env_config(),
                sif_cache_dir=str(sif_dir),
            )
            env._sif_path = sif_path
            return env

        env_a = make_env("trial-a")
        env_b = make_env("trial-b")

        async def quiet_runtime_info() -> None:
            return None

        async def quiet_pid_probe() -> None:
            return None

        async def start_overlay(env: ApptainerHPCEnvironment) -> None:
            env._started = True
            env._active_rootfs_mode = "overlay"

        async def start_sandbox(env: ApptainerHPCEnvironment) -> None:
            env._started = True
            env._active_rootfs_mode = "sandbox"

        async def fake_probe() -> bool:
            return False

        async def fake_stop(delete: bool = True) -> None:
            return None

        for env in (env_a, env_b):
            monkeypatch.setattr(env, "_log_runtime_info", quiet_runtime_info)
            monkeypatch.setattr(env, "_probe_pid_support", quiet_pid_probe)
            monkeypatch.setattr(
                env,
                "_start_overlay_instance",
                lambda e=env: start_overlay(e),
            )
            monkeypatch.setattr(
                env,
                "_start_sandbox_instance",
                lambda e=env: start_sandbox(e),
            )
            monkeypatch.setattr(env, "stop", fake_stop)
        monkeypatch.setattr(env_a, "_probe_root_writability", fake_probe)

        with caplog.at_level("INFO"):
            run_async(env_a.start())
            run_async(env_b.start())

        messages = [record.getMessage() for record in caplog.records]
        assert messages.count(
            "Apptainer overlay probe failed; falling back to writable sandbox"
        ) == 1
        assert all("cached overlay probe failure" not in message for message in messages)

    def test_overlay_mode_raises_with_remediation_when_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="overlay",
        )
        env._sif_path.touch()

        seed_dir = tmp_path / "seed-cache" / "task_01"
        seed_dir.mkdir(parents=True, exist_ok=True)

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            return MockExecResult(stdout="ok")

        async def fake_probe() -> bool:
            return False

        monkeypatch.setattr(env, "_prepare_app_seed_dir", lambda: seed_dir)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        monkeypatch.setattr(env, "_probe_root_writability", fake_probe)

        with pytest.raises(RuntimeError, match="rootfs_mode: auto or rootfs_mode: sandbox"):
            run_async(env.start())

    def test_exec_uses_instance_prefix_pwd_cleanenv_and_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        env._started = True

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        result = run_async(
            env.exec("pwd", cwd="/workspace", env={"FOO": "bar"}, timeout_sec=17)
        )

        assert result.stdout == "ok"
        cmd = calls[0][0]
        assert cmd[0] == "apptainer"
        assert cmd[1] == "exec"
        assert "--cleanenv" in cmd
        assert "--pwd" in cmd
        pwd_idx = cmd.index("--pwd")
        assert cmd[pwd_idx + 1] == "/workspace"
        assert "--env" in cmd
        env_idx = cmd.index("--env")
        assert cmd[env_idx + 1] == "FOO=bar"
        assert f"instance://{env._instance_name}" in cmd
        assert cmd[-3:] == ["bash", "-lc", "pwd"]
        assert calls[0][2] == 17

    def test_run_apptainer_command_uses_shared_cli_runner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_module
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )

        recorded: dict[str, Any] = {}

        def fake_helper(
            cmd: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            check: bool,
            timeout_sec: int | None,
            runtime_label: str,
            logger: logging.Logger,
        ) -> MockExecResult:
            recorded.update(
                {
                    "cmd": cmd,
                    "cwd": cwd,
                    "env": env,
                    "check": check,
                    "timeout_sec": timeout_sec,
                    "runtime_label": runtime_label,
                    "logger_name": logger.name,
                }
            )
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(harbor_module, "_run_hpc_cli_command", fake_helper)

        result = run_async(
            env._run_apptainer_command(
                ["apptainer", "exec", "instance://test", "bash", "-lc", "pwd"],
                check=False,
                timeout_sec=11,
            )
        )

        assert result.stdout == "ok"
        assert recorded["cmd"] == ["apptainer", "exec", "instance://test", "bash", "-lc", "pwd"]
        assert recorded["cwd"] == tmp_path
        assert recorded["check"] is False
        assert recorded["timeout_sec"] == 11
        assert recorded["runtime_label"] == "apptainer"
        assert recorded["logger_name"] == "llenvs.adapters.harbor"
        assert "PATH" in recorded["env"]

    def test_exec_defaults_to_image_workdir_when_cwd_omitted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\nWORKDIR /app\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        env._started = True

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        result = run_async(env.exec("pwd"))

        assert result.stdout == "ok"
        cmd = calls[0][0]
        assert "--pwd" in cmd
        pwd_idx = cmd.index("--pwd")
        assert cmd[pwd_idx + 1] == "/app"

    def test_exec_defaults_to_app_when_dockerfile_has_no_workdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        env._started = True

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        result = run_async(env.exec("pwd"))

        assert result.stdout == "ok"
        cmd = calls[0][0]
        assert "--pwd" in cmd
        pwd_idx = cmd.index("--pwd")
        assert cmd[pwd_idx + 1] == "/app"

    def test_stop_calls_instance_stop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        env._started = True

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.stop())

        assert calls[0][0] == [
            "apptainer", "instance", "stop", env._instance_name
        ]
        assert env._started is False

    def test_upload_dir_to_tests_uses_generic_exec_in_sandbox_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
            rootfs_mode="sandbox",
        )
        env._started = True
        env._active_rootfs_mode = "sandbox"

        source_dir = tmp_path / "tests"
        source_dir.mkdir()
        (source_dir / "test_a.sh").write_text("echo ok")

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.upload_dir(source_dir, "/tests"))

        assert len(calls) == 1
        assert calls[0][0][:2] == ["apptainer", "exec"]
        assert "mkdir -p /tests && cp -a " in calls[0][0][-1]

    def test_upload_dir_via_staging(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
        )
        env._staging_dir.mkdir(parents=True, exist_ok=True)
        env._started = True

        source_dir = tmp_path / "tests"
        source_dir.mkdir()
        (source_dir / "test_a.sh").write_text("echo ok")

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.upload_dir(source_dir, "/workspace/tests"))

        # Should have called exec with cp command
        assert len(calls) == 1
        cmd = calls[0][0]
        assert "exec" in cmd
        assert f"instance://{env._instance_name}" in cmd
        assert any("cp -a" in arg and "/workspace/tests/" in arg for arg in cmd)

    def test_has_checkpoint_methods(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        assert hasattr(env, "export_checkpoint")
        assert hasattr(env, "restore_checkpoint")
        assert hasattr(env, "capture_runtime_probe")
        assert hasattr(env, "detect_runtime_risk")

    def test_sandbox_creates_bind_mount_dirs_in_rootfs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="sandbox",
        )
        env._sif_path.touch()

        rootfs_dir = trial / "rootfs"

        def fake_prepare_trial_rootfs() -> Path:
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        dirs_at_instance_start: list[list[str]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            if cmd[:3] == ["apptainer", "instance", "start"]:
                # Record which bind-mount dirs exist at the time of start.
                dirs_at_instance_start.append([
                    d for d in ("staging", "logs/verifier", "logs/agent")
                    if (rootfs_dir / d).exists()
                ])
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_prepare_trial_rootfs", fake_prepare_trial_rootfs)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        assert dirs_at_instance_start == [["staging", "logs/verifier", "logs/agent"]]

    def test_overlay_start_binds_tmp_and_var_tmp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_RUNTIME_INFO_LOGGED_KEYS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            overlay_size_mb=256,
            rootfs_mode="overlay",
        )
        env._sif_path.touch()

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        seed_dir = tmp_path / "seed-cache" / "task_01"
        seed_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(env, "_prepare_app_seed_dir", lambda: seed_dir)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        inst_cmd = next(
            cmd for cmd, _check, _timeout in calls
            if cmd[:3] == ["apptainer", "instance", "start"]
        )
        assert f"{trial / 'tmp'}:/tmp" in inst_cmd
        assert f"{trial / 'var_tmp'}:/var/tmp" in inst_cmd
        assert (trial / "tmp").exists()
        assert (trial / "var_tmp").exists()
        assert (trial / "tmp").stat().st_mode & 0o7777 == 0o1777
        assert (trial / "var_tmp").stat().st_mode & 0o7777 == 0o1777

    def test_sandbox_start_binds_tmp_and_var_tmp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="sandbox",
        )
        env._sif_path.touch()

        rootfs_dir = trial / "rootfs"

        def fake_prepare_trial_rootfs() -> Path:
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_prepare_trial_rootfs", fake_prepare_trial_rootfs)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.start())

        inst_cmd = next(
            cmd for cmd, _check, _timeout in calls
            if cmd[:3] == ["apptainer", "instance", "start"]
        )
        assert f"{trial / 'tmp'}:/tmp" in inst_cmd
        assert f"{trial / 'var_tmp'}:/var/tmp" in inst_cmd

    def test_bootstrap_log_dirs_raises_on_unwritable_tmp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="overlay",
        )
        env._instance_name = "test-instance"

        call_count = 0

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            nonlocal call_count
            call_count += 1
            # First call is mkdir for log dirs — succeeds.
            # Second call is tmpdir probe — fails.
            if call_count == 2:
                return MockExecResult(
                    stdout="", stderr="Permission denied", return_code=1
                )
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        with pytest.raises(RuntimeError, match="/tmp is not writable"):
            run_async(env._bootstrap_log_dirs())

    def test_stop_cleans_up_tmp_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
        )
        env._started = True
        env._instance_name = "test-instance"

        # Create the tmp dirs as _prepare_runtime_dirs would.
        env._host_tmp_dir.mkdir(parents=True)
        env._host_var_tmp_dir.mkdir(parents=True)
        assert env._host_tmp_dir.exists()
        assert env._host_var_tmp_dir.exists()

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env.stop(delete=True))

        assert not env._host_tmp_dir.exists()
        assert not env._host_var_tmp_dir.exists()

    def test_concurrent_overlay_probe_runs_only_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import threading

        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_ROOTFS_PROBE_CACHE.clear()
        harbor_mod._APPTAINER_ROOTFS_PROBE_EVENTS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        sif_path = sif_dir / "ca0c1413bbd82bab.sif"
        sif_path.touch()

        def make_env(name: str) -> ApptainerHPCEnvironment:
            trial = tmp_path / name
            env = ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id=f"session-{name}",
                trial_paths=self._make_trial_paths(trial),
                task_env_config=self._make_task_env_config(),
                sif_cache_dir=str(sif_dir),
            )
            env._sif_path = sif_path
            return env

        env_a = make_env("trial-a")
        env_b = make_env("trial-b")

        probe_count = 0
        # Barrier ensures both threads are inside start() before either probes.
        entry_barrier = threading.Barrier(2, timeout=5)

        def fake_prepare_seed() -> Path:
            seed_dir = tmp_path / "seed-cache" / "task_01"
            seed_dir.mkdir(parents=True, exist_ok=True)
            return seed_dir

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            return MockExecResult(stdout="ok")

        async def fake_probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return False

        # Use _log_runtime_info as a sync point so both threads are inside
        # start() before either reaches _wait_for_overlay_probe.
        async def synced_log_a():
            entry_barrier.wait()

        async def synced_log_b():
            entry_barrier.wait()

        for env in (env_a, env_b):
            monkeypatch.setattr(env, "_prepare_app_seed_dir", fake_prepare_seed)
            monkeypatch.setattr(env, "_prepare_trial_rootfs",
                                lambda e=env: (Path(e.trial_paths.trial_dir) / "rootfs"))
            monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
            monkeypatch.setattr(env, "_probe_root_writability", fake_probe)

        monkeypatch.setattr(env_a, "_log_runtime_info", synced_log_a)
        monkeypatch.setattr(env_b, "_log_runtime_info", synced_log_b)

        errors: list[Exception] = []

        def run_start(env):
            try:
                run_async(env.start())
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=run_start, args=(env_a,))
        t_b = threading.Thread(target=run_start, args=(env_b,))

        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        assert not errors, f"Threads raised: {errors}"
        # Only one thread should have probed — the other waited on the event.
        assert probe_count == 1
        # Both should end up in sandbox mode.
        assert env_a._active_rootfs_mode == "sandbox"
        assert env_b._active_rootfs_mode == "sandbox"

    def test_overlay_probe_waiters_recover_if_first_owner_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        import threading

        import llenvs.adapters.harbor as harbor_mod
        from llenvs.adapters.harbor import ApptainerHPCEnvironment
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_ROOTFS_PROBE_CACHE.clear()
        harbor_mod._APPTAINER_ROOTFS_PROBE_EVENTS.clear()

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        sif_path = sif_dir / "ca0c1413bbd82bab.sif"
        sif_path.touch()

        def make_env(name: str) -> ApptainerHPCEnvironment:
            trial = tmp_path / name
            env = ApptainerHPCEnvironment(
                environment_dir=tmp_path,
                environment_name="task_01",
                session_id=f"session-{name}",
                trial_paths=self._make_trial_paths(trial),
                task_env_config=self._make_task_env_config(),
                sif_cache_dir=str(sif_dir),
            )
            env._sif_path = sif_path
            return env

        env_a = make_env("trial-a")
        env_b = make_env("trial-b")

        entry_barrier = threading.Barrier(2, timeout=5)
        overlay_start_count = 0
        probe_count = 0

        async def synced_log():
            entry_barrier.wait()

        async def fake_probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return False

        async def fake_start_overlay(env: ApptainerHPCEnvironment) -> None:
            nonlocal overlay_start_count
            overlay_start_count += 1
            if overlay_start_count == 1:
                raise RuntimeError("overlay start failed")
            env._started = True
            env._active_rootfs_mode = "overlay"

        async def fake_start_sandbox(env: ApptainerHPCEnvironment) -> None:
            env._started = True
            env._active_rootfs_mode = "sandbox"

        async def fake_stop(delete: bool = True) -> None:
            return None

        for env in (env_a, env_b):
            monkeypatch.setattr(env, "_log_runtime_info", synced_log)
            monkeypatch.setattr(
                env,
                "_start_overlay_instance",
                lambda e=env: fake_start_overlay(e),
            )
            monkeypatch.setattr(
                env,
                "_start_sandbox_instance",
                lambda e=env: fake_start_sandbox(e),
            )
            monkeypatch.setattr(env, "_probe_root_writability", fake_probe)
            monkeypatch.setattr(env, "stop", fake_stop)

        errors: list[Exception] = []

        def run_start(env):
            try:
                run_async(env.start())
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=run_start, args=(env_a,))
        t_b = threading.Thread(target=run_start, args=(env_b,))

        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        assert not t_a.is_alive()
        assert not t_b.is_alive()
        assert len(errors) == 1
        assert str(errors[0]) == "overlay start failed"
        assert probe_count == 1
        assert overlay_start_count == 2
        assert any(env._active_rootfs_mode == "sandbox" for env in (env_a, env_b))
        assert harbor_mod._APPTAINER_ROOTFS_PROBE_CACHE
        assert not harbor_mod._APPTAINER_ROOTFS_PROBE_EVENTS


class TestApptainerCheckpointRestore:
    """Tests for Apptainer filesystem checkpoint/restore and sandbox start helpers."""

    def _make_trial_paths(self, tmp_path):
        for name in ("verifier", "agent", "artifacts"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            trial_dir=tmp_path,
            verifier_dir=tmp_path / "verifier",
            agent_dir=tmp_path / "agent",
            artifacts_dir=tmp_path / "artifacts",
        )

    def _make_task_env_config(self, **kwargs: Any):
        defaults = {
            "docker_image": "ubuntu:latest",
            "cpus": 1,
            "memory_mb": 1024,
            "allow_internet": True,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _make_env(self, tmp_path, **kwargs):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir(exist_ok=True)
        trial = tmp_path / "trial"
        defaults = dict(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(trial),
            task_env_config=self._make_task_env_config(),
            sif_cache_dir=str(sif_dir),
            rootfs_mode="sandbox",
        )
        defaults.update(kwargs)
        env = ApptainerHPCEnvironment(**defaults)
        env._sif_path.touch()
        return env

    def test_export_checkpoint_uses_helper_exec_and_replaces_temp_tar(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path, fakeroot=True)
        env._active_rootfs_mode = "sandbox"
        env._started = True

        rootfs = env._sandbox_rootfs_dir
        rootfs.mkdir(parents=True, exist_ok=True)
        (rootfs / "file_a.txt").write_text("hello")
        sub = rootfs / "subdir"
        sub.mkdir()
        (sub / "file_b.txt").write_text("world")

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            assert (tmp_path / "exports").exists()
            shell_cmd = cmd[-1]
            prefix = "/.vb_checkpoint_out/"
            start = shell_cmd.index(prefix) + len(prefix)
            end = shell_cmd.index(" ", start)
            temp_name = shell_cmd[start:end].strip("'\"")
            (tmp_path / "exports" / temp_name).write_text("checkpoint-bytes")
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        export_path = tmp_path / "exports" / "checkpoint.tar"
        run_async(env.export_checkpoint(export_path))

        assert export_path.exists()
        assert export_path.read_text() == "checkpoint-bytes"
        temp_files = list(export_path.parent.glob(".*.tmp"))
        assert temp_files == []

        cmd = calls[0][0]
        assert cmd[:2] == ["apptainer", "exec"]
        assert "--cleanenv" in cmd
        assert "--fakeroot" in cmd
        bind_specs = [
            cmd[i + 1]
            for i, token in enumerate(cmd)
            if token == "--bind"
        ]
        assert f"{env._sandbox_rootfs_dir}:/.vb_checkpoint_src" in bind_specs
        assert f"{export_path.parent}:/.vb_checkpoint_out" in bind_specs
        assert str(env._sif_path) in cmd
        assert cmd[-3] == "bash"
        assert cmd[-2] == "-lc"
        assert "tar -cf" in cmd[-1]
        assert "-C /.vb_checkpoint_src ." in cmd[-1]
        assert "/.vb_checkpoint_out/" in cmd[-1]

    def test_restore_checkpoint_replaces_rootfs_and_restarts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path, fakeroot=True)
        env._started = True
        env._active_rootfs_mode = "sandbox"

        tar_path = tmp_path / "checkpoint.tar"
        tar_path.write_text("checkpoint-bytes")

        stop_calls: list[dict] = []
        start_sandbox_calls: list[Path] = []
        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_stop(delete: bool = True):
            stop_calls.append({"delete": delete})
            env._started = False

        async def fake_start_sandbox(rootfs_dir: Path):
            start_sandbox_calls.append(rootfs_dir)
            env._started = True
            env._active_rootfs_mode = "sandbox"

        monkeypatch.setattr(env, "stop", fake_stop)
        monkeypatch.setattr(env, "_start_sandbox_from_rootfs", fake_start_sandbox)

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            (env._sandbox_rootfs_dir / "restored.txt").write_text("restored_content")
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        run_async(env.restore_checkpoint(tar_path))

        # Verify stop(delete=False) was called so rootfs cleanup happens in-helper.
        assert len(stop_calls) == 1
        assert stop_calls[0]["delete"] is False

        # Verify rootfs was replaced with tar contents
        assert (env._sandbox_rootfs_dir / "restored.txt").read_text() == "restored_content"

        # Verify _start_sandbox_from_rootfs was called with the rootfs dir
        assert len(start_sandbox_calls) == 1
        assert start_sandbox_calls[0] == env._sandbox_rootfs_dir

        cmd = calls[0][0]
        assert cmd[:2] == ["apptainer", "exec"]
        assert "--cleanenv" in cmd
        assert "--fakeroot" in cmd
        bind_specs = [
            cmd[i + 1]
            for i, token in enumerate(cmd)
            if token == "--bind"
        ]
        assert f"{tar_path.parent}:/.vb_checkpoint_in" in bind_specs
        assert f"{env._sandbox_rootfs_dir}:/.vb_checkpoint_dst" in bind_specs
        assert str(env._sif_path) in cmd
        assert cmd[-3] == "bash"
        assert cmd[-2] == "-lc"
        assert "find /.vb_checkpoint_dst" in cmd[-1]
        assert f"tar -xf /.vb_checkpoint_in/{tar_path.name} -C /.vb_checkpoint_dst" in cmd[-1]

    def test_export_does_not_modify_host_staging_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)
        env._active_rootfs_mode = "sandbox"
        env._started = True

        # Set up staging dir with some files
        env._staging_dir.mkdir(parents=True, exist_ok=True)
        (env._staging_dir / "leftover.txt").write_text("should be removed")
        (env._staging_dir / "upload").mkdir()
        (env._staging_dir / "upload" / "data.bin").write_text("stale")

        # Set up rootfs so tar succeeds
        env._sandbox_rootfs_dir.mkdir(parents=True, exist_ok=True)
        (env._sandbox_rootfs_dir / "dummy.txt").write_text("content")

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            shell_cmd = cmd[-1]
            prefix = "/.vb_checkpoint_out/"
            start = shell_cmd.index(prefix) + len(prefix)
            end = shell_cmd.index(" ", start)
            temp_name = shell_cmd[start:end].strip("'\"")
            (tmp_path / temp_name).write_text("checkpoint-bytes")
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        export_path = tmp_path / "checkpoint.tar"
        run_async(env.export_checkpoint(export_path))

        # Host staging dir should be left untouched.
        assert env._staging_dir.exists()
        assert (env._staging_dir / "leftover.txt").read_text() == "should be removed"
        assert (env._staging_dir / "upload" / "data.bin").read_text() == "stale"

    def test_export_checkpoint_cleans_partial_temp_tar_on_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)
        env._active_rootfs_mode = "sandbox"
        env._started = True
        env._sandbox_rootfs_dir.mkdir(parents=True, exist_ok=True)
        (env._sandbox_rootfs_dir / "dummy.txt").write_text("content")

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            shell_cmd = cmd[-1]
            prefix = "/.vb_checkpoint_out/"
            start = shell_cmd.index(prefix) + len(prefix)
            end = shell_cmd.index(" ", start)
            temp_name = shell_cmd[start:end].strip("'\"")
            partial_path = (tmp_path / "exports" / temp_name)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text("partial")
            raise RuntimeError("boom")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        export_path = tmp_path / "exports" / "checkpoint.tar"
        with pytest.raises(RuntimeError, match="filesystem checkpoint export failed"):
            run_async(env.export_checkpoint(export_path))

        assert not export_path.exists()
        assert list(export_path.parent.glob(".*.tmp")) == []

    def test_checkpoint_reports_missing_tar(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)
        env._active_rootfs_mode = "sandbox"
        env._started = True
        env._sandbox_rootfs_dir.mkdir(parents=True, exist_ok=True)
        (env._sandbox_rootfs_dir / "dummy.txt").write_text("content")

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            raise RuntimeError("stderr: bash: tar: command not found")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        with pytest.raises(RuntimeError, match="requires `tar`"):
            run_async(env.export_checkpoint(tmp_path / "checkpoint.tar"))

    def test_export_requires_sandbox_mode(self, tmp_path):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)
        env._active_rootfs_mode = "overlay"
        env._started = True

        with pytest.raises(RuntimeError, match="sandbox mode"):
            run_async(env.export_checkpoint(tmp_path / "checkpoint.tar"))

    def test_pid_namespace_flag_adds_pid_to_start_cmd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """When runtime supports --pid, use it directly."""
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_PID_FLAG_CACHE.clear()
        harbor_mod._APPTAINER_PID_FLAG_EVENTS.clear()

        env = self._make_env(tmp_path, pid_namespace=True)

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            # Return --pid in help output so the probe detects support
            if "start" in cmd and "--help" in cmd:
                return MockExecResult(stdout="  --pid   run in PID namespace")
            return MockExecResult(stdout="ok")

        def fake_prepare_trial_rootfs() -> Path:
            rootfs_dir = env._sandbox_rootfs_dir
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        monkeypatch.setattr(env, "_prepare_trial_rootfs", fake_prepare_trial_rootfs)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env._start_sandbox_instance())

        # Find the instance start command
        inst_cmds = [
            cmd for cmd, _check, _timeout in calls
            if len(cmd) >= 3 and cmd[:3] == ["apptainer", "instance", "start"]
            and "--help" not in cmd
        ]
        assert len(inst_cmds) >= 1
        assert "--pid" in inst_cmds[0]

    def test_pid_namespace_falls_back_to_containall(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """When runtime lacks --pid (e.g., SingularityCE 4.3), use --containall."""
        import llenvs.adapters.harbor as harbor_mod
        from llenvs.core.async_utils import run_async

        harbor_mod._APPTAINER_VERSION_CACHE.clear()
        harbor_mod._APPTAINER_PID_FLAG_CACHE.clear()
        harbor_mod._APPTAINER_PID_FLAG_EVENTS.clear()

        env = self._make_env(tmp_path, pid_namespace=True)

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            # Realistic SingularityCE 4.3 help — contains --pid-file,
            # --pids-limit, and "--pid" in --no-init description, but
            # NOT --pid as a standalone flag.
            if "start" in cmd and "--help" in cmd:
                return MockExecResult(stdout=(
                    "  -C, --containall                    contain PID, IPC, env\n"
                    "      --no-init                       do NOT start shim process with --pid\n"
                    "      --pid-file string               write instance PID to file\n"
                    "      --pids-limit int                Limit number of container PIDs\n"
                ))
            return MockExecResult(stdout="ok")

        def fake_prepare_trial_rootfs() -> Path:
            rootfs_dir = env._sandbox_rootfs_dir
            rootfs_dir.mkdir(parents=True, exist_ok=True)
            return rootfs_dir

        monkeypatch.setattr(env, "_prepare_trial_rootfs", fake_prepare_trial_rootfs)
        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env._start_sandbox_instance())

        # Find the instance start command (not the help probe)
        inst_cmds = [
            cmd for cmd, _check, _timeout in calls
            if len(cmd) >= 3 and cmd[:3] == ["apptainer", "instance", "start"]
            and "--help" not in cmd
        ]
        assert len(inst_cmds) >= 1
        assert "--containall" in inst_cmds[0]
        assert "--pid" not in inst_cmds[0]
        # --contain should be removed (superseded by --containall)
        assert "--contain" not in inst_cmds[0]

    def test_start_sandbox_from_rootfs_reuses_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)

        # Set up a rootfs dir with content
        rootfs_dir = tmp_path / "existing_rootfs"
        rootfs_dir.mkdir()
        (rootfs_dir / "important.txt").write_text("keep me")

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(stdout="ok")

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)
        run_async(env._start_sandbox_from_rootfs(rootfs_dir))

        # Content should still be there (not re-extracted from SIF)
        assert (rootfs_dir / "important.txt").read_text() == "keep me"

        # Instance start should have used the rootfs dir
        inst_cmds = [
            cmd for cmd, _check, _timeout in calls
            if len(cmd) >= 3 and cmd[:3] == ["apptainer", "instance", "start"]
        ]
        assert len(inst_cmds) == 1
        assert str(rootfs_dir) in inst_cmds[0]
        assert env._active_rootfs_mode == "sandbox"

    def test_capture_runtime_probe_uses_internal_timeout_cap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)
        env._started = True

        calls: list[tuple[list[str], bool, int | None]] = []

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            calls.append((cmd, check, timeout_sec))
            return MockExecResult(
                stdout=(
                    "===PROCS===\n"
                    "bash\n"
                    "===MOUNTS===\n"
                    "abcdef123456  /proc/self/mountinfo\n"
                    "===SOCKETS===\n"
                    "===STAGING===\n"
                )
            )

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        probe = run_async(env.capture_runtime_probe())

        assert probe.probe_failed is False
        assert calls[0][2] == 15

    def test_capture_runtime_probe_timeout_marks_probe_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        from llenvs.core.async_utils import run_async

        env = self._make_env(tmp_path)
        env._started = True

        async def fake_run(cmd, *, check=True, timeout_sec=None):
            raise RuntimeError(
                f"apptainer command timed out after {timeout_sec}s: {' '.join(cmd)}"
            )

        monkeypatch.setattr(env, "_run_apptainer_command", fake_run)

        probe = run_async(env.capture_runtime_probe())

        assert probe.probe_failed is True
        assert "timed out" in (probe.probe_error or "")


class TestRuntimeProbing:
    """Tests for runtime probe parsing, risk detection, and state annotation."""

    def test_runtime_probe_parses_all_sections(self):
        from llenvs.adapters.harbor import RuntimeProbeSnapshot, _parse_probe_output

        stdout = (
            "===PROCS===\n"
            "bash\n"
            "python3\n"
            "nginx\n"
            "===MOUNTS===\n"
            "abcdef123456  /proc/self/mountinfo\n"
            "===SOCKETS===\n"
            "tcp   LISTEN 0 128 0.0.0.0:8080 0.0.0.0:*\n"
            "tcp   LISTEN 0 128 0.0.0.0:443 0.0.0.0:*\n"
            "===STAGING===\n"
            "upload\n"
            "download\n"
        )
        result = _parse_probe_output(stdout, has_pid_namespace=True)

        assert isinstance(result, RuntimeProbeSnapshot)
        assert result.process_commands == frozenset({"bash", "python3", "nginx"})
        assert result.mount_fingerprint == "abcdef123456"
        assert result.listening_ports == frozenset({8080, 443})
        assert result.staging_entries == frozenset({"upload", "download"})
        assert result.staging_has_content is True
        assert result.probe_failed is False

    def test_runtime_probe_handles_unavailable_tools(self):
        from llenvs.adapters.harbor import _parse_probe_output

        stdout = (
            "===PROCS===\n"
            "UNAVAILABLE\n"
            "===MOUNTS===\n"
            "UNAVAILABLE\n"
            "===SOCKETS===\n"
            "UNAVAILABLE\n"
            "===STAGING===\n"
            "UNAVAILABLE\n"
        )
        result = _parse_probe_output(stdout, has_pid_namespace=True)

        assert result.process_commands == frozenset()
        assert result.mount_fingerprint == ""
        assert result.listening_ports == frozenset()
        assert result.staging_entries == frozenset()
        assert result.staging_has_content is False

    def test_probe_skips_process_diff_without_pid_namespace(self):
        from llenvs.adapters.harbor import _parse_probe_output

        stdout = (
            "===PROCS===\n"
            "bash\n"
            "python3\n"
            "===MOUNTS===\n"
            "abc123  /proc/self/mountinfo\n"
            "===SOCKETS===\n"
            "UNAVAILABLE\n"
            "===STAGING===\n"
        )
        result = _parse_probe_output(stdout, has_pid_namespace=False)

        # Even though process data is present, it should be ignored
        assert result.process_commands == frozenset()

    def test_detect_risk_ignores_harbor_managed_staging_paths(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment, RuntimeProbeSnapshot

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        for name in ("verifier", "agent", "artifacts"):
            (trial / name).mkdir(parents=True, exist_ok=True)
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=SimpleNamespace(
                trial_dir=trial,
                verifier_dir=trial / "verifier",
                agent_dir=trial / "agent",
                artifacts_dir=trial / "artifacts",
            ),
            task_env_config=SimpleNamespace(
                docker_image="ubuntu:latest", cpus=1, memory_mb=1024, allow_internet=True
            ),
            sif_cache_dir=str(sif_dir),
        )

        env._probe_baseline = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        current = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=True,
            staging_entries=frozenset({"upload", "download"}),
        )

        risk, reasons = env.detect_runtime_risk(current)
        assert risk is False
        assert reasons == ()

    def test_detect_risk_extra_processes(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment, RuntimeProbeSnapshot

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        for name in ("verifier", "agent", "artifacts"):
            (trial / name).mkdir(parents=True, exist_ok=True)
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=SimpleNamespace(
                trial_dir=trial,
                verifier_dir=trial / "verifier",
                agent_dir=trial / "agent",
                artifacts_dir=trial / "artifacts",
            ),
            task_env_config=SimpleNamespace(
                docker_image="ubuntu:latest", cpus=1, memory_mb=1024, allow_internet=True
            ),
            sif_cache_dir=str(sif_dir),
            pid_namespace=True,
        )

        env._probe_baseline = RuntimeProbeSnapshot(
            process_commands=frozenset({"bash"}),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        current = RuntimeProbeSnapshot(
            process_commands=frozenset({"bash", "redis-server"}),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        risk, reasons = env.detect_runtime_risk(current)
        assert risk is True
        assert any("extra_processes" in r and "redis-server" in r for r in reasons)

    def test_detect_risk_new_listening_port(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment, RuntimeProbeSnapshot

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        for name in ("verifier", "agent", "artifacts"):
            (trial / name).mkdir(parents=True, exist_ok=True)
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=SimpleNamespace(
                trial_dir=trial,
                verifier_dir=trial / "verifier",
                agent_dir=trial / "agent",
                artifacts_dir=trial / "artifacts",
            ),
            task_env_config=SimpleNamespace(
                docker_image="ubuntu:latest", cpus=1, memory_mb=1024, allow_internet=True
            ),
            sif_cache_dir=str(sif_dir),
        )

        env._probe_baseline = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        current = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset({8080}),
            staging_has_content=False,
        )

        risk, reasons = env.detect_runtime_risk(current)
        assert risk is True
        assert any("new_listening_ports" in r and "8080" in r for r in reasons)

    def test_detect_risk_mount_change(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment, RuntimeProbeSnapshot

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        for name in ("verifier", "agent", "artifacts"):
            (trial / name).mkdir(parents=True, exist_ok=True)
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=SimpleNamespace(
                trial_dir=trial,
                verifier_dir=trial / "verifier",
                agent_dir=trial / "agent",
                artifacts_dir=trial / "artifacts",
            ),
            task_env_config=SimpleNamespace(
                docker_image="ubuntu:latest", cpus=1, memory_mb=1024, allow_internet=True
            ),
            sif_cache_dir=str(sif_dir),
        )

        env._probe_baseline = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        current = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="def",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        risk, reasons = env.detect_runtime_risk(current)
        assert risk is True
        assert "mount_table_changed" in reasons

    def test_detect_risk_staging_content(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment, RuntimeProbeSnapshot

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        trial = tmp_path / "trial"
        for name in ("verifier", "agent", "artifacts"):
            (trial / name).mkdir(parents=True, exist_ok=True)
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=SimpleNamespace(
                trial_dir=trial,
                verifier_dir=trial / "verifier",
                agent_dir=trial / "agent",
                artifacts_dir=trial / "artifacts",
            ),
            task_env_config=SimpleNamespace(
                docker_image="ubuntu:latest", cpus=1, memory_mb=1024, allow_internet=True
            ),
            sif_cache_dir=str(sif_dir),
        )

        env._probe_baseline = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=False,
        )

        current = RuntimeProbeSnapshot(
            process_commands=frozenset(),
            mount_fingerprint="abc",
            listening_ports=frozenset(),
            staging_has_content=True,
            staging_entries=frozenset({"upload", "scratch"}),
        )

        risk, reasons = env.detect_runtime_risk(current)
        assert risk is True
        assert "staging_content_detected" in reasons

    def test_risk_ever_is_sticky(self):
        from dataclasses import replace

        from llenvs.adapters.harbor import HarborHidden

        hidden = HarborHidden(
            task_index=0,
            task_name="t",
            instruction="i",
            episode_step=1,
            fs_restore_risk_now=True,
            fs_restore_risk_reasons=("extra_processes:redis",),
            fs_restore_risk_ever=False,
        )

        # Simulate what HarborEnvironment.step does:
        # fs_restore_risk_ever = prev.fs_restore_risk_ever or prev.fs_restore_risk_now
        next_hidden = HarborHidden(
            task_index=hidden.task_index,
            task_name=hidden.task_name,
            instruction=hidden.instruction,
            episode_step=hidden.episode_step + 1,
            fs_restore_risk_ever=hidden.fs_restore_risk_ever or hidden.fs_restore_risk_now,
        )

        assert next_hidden.fs_restore_risk_ever is True
        # Even if next step has risk_now=False by default
        assert next_hidden.fs_restore_risk_now is False

    def test_risk_now_resets_per_step(self):
        from llenvs.adapters.harbor import HarborHidden, RuntimeProbeSnapshot, _probe_and_annotate_state
        from llenvs.core.state import Observation, ObservationContent, State, StateMetadata

        hidden = HarborHidden(
            task_index=0,
            task_name="t",
            instruction="i",
            episode_step=2,
            fs_restore_risk_now=True,
            fs_restore_risk_reasons=("mount_table_changed",),
            fs_restore_risk_ever=True,
        )

        # Build next hidden as step() does
        next_hidden = HarborHidden(
            task_index=hidden.task_index,
            task_name=hidden.task_name,
            instruction=hidden.instruction,
            episode_step=hidden.episode_step + 1,
            fs_restore_risk_ever=hidden.fs_restore_risk_ever or hidden.fs_restore_risk_now,
        )

        obs = Observation(
            prompt="test",
            task=ObservationContent(text="task"),
            state=ObservationContent(text="state"),
        )
        state = State(
            observation=obs,
            hidden=next_hidden,
            metadata=StateMetadata(step=3, episode_id="ep"),
        )

        # Create a mock harbor_env with a baseline and no risk
        class FakeHarborEnv:
            _pid_namespace = False
            _probe_baseline = RuntimeProbeSnapshot(
                process_commands=frozenset(),
                mount_fingerprint="abc",
                listening_ports=frozenset(),
                staging_has_content=False,
            )

            def capture_runtime_probe(self):
                import asyncio

                async def _probe():
                    return RuntimeProbeSnapshot(
                        process_commands=frozenset(),
                        mount_fingerprint="abc",
                        listening_ports=frozenset(),
                        staging_has_content=False,
                    )
                return _probe()

            def detect_runtime_risk(self, current):
                # No risk this time
                return False, ()

        annotated = _probe_and_annotate_state(
            FakeHarborEnv(), state, runtime_probing=True,
        )

        # risk_now should be False for this step
        assert annotated.hidden.fs_restore_risk_now is False
        # risk_ever remains True (sticky from earlier steps)
        assert annotated.hidden.fs_restore_risk_ever is True

    def test_baseline_probe_failure_recorded_immediately(self):
        from llenvs.adapters.harbor import HarborHidden, RuntimeProbeSnapshot, _probe_and_annotate_state
        from llenvs.core.state import Observation, ObservationContent, State, StateMetadata

        hidden = HarborHidden(
            task_index=0,
            task_name="t",
            instruction="i",
            episode_step=0,
        )
        obs = Observation(
            prompt="test",
            task=ObservationContent(text="task"),
            state=ObservationContent(text="state"),
        )
        state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="ep"),
        )

        class FakeHarborEnv:
            _pid_namespace = False
            _probe_baseline = None  # No baseline yet

            def capture_runtime_probe(self):
                import asyncio

                async def _probe():
                    return RuntimeProbeSnapshot(
                        process_commands=frozenset(),
                        mount_fingerprint="",
                        listening_ports=frozenset(),
                        staging_has_content=False,
                        probe_failed=True,
                        probe_error="command not found: ss",
                    )
                return _probe()

        fake_env = FakeHarborEnv()
        annotated = _probe_and_annotate_state(
            fake_env, state, runtime_probing=True,
        )

        assert annotated.hidden.fs_restore_risk_now is True
        assert "baseline_probe_degraded" in annotated.hidden.fs_restore_risk_reasons
        # Baseline should have been set
        assert fake_env._probe_baseline is not None
        assert fake_env._probe_baseline.probe_failed is True


class TestRuntimeEligibility:
    def _make_task(self, tmp_path, name, *, dockerfile=None, compose=None, docker_image=None, allow_internet=True):
        env_dir = tmp_path / name
        env_dir.mkdir(parents=True, exist_ok=True)
        if dockerfile:
            (env_dir / "Dockerfile").write_text(dockerfile)
        if compose:
            (env_dir / "docker-compose.yaml").write_text(compose)
        config = SimpleNamespace(
            environment=SimpleNamespace(
                docker_image=docker_image,
                allow_internet=allow_internet,
            )
        )
        paths = SimpleNamespace(environment_dir=str(env_dir))
        return SimpleNamespace(name=name, config=config, paths=paths)

    def test_podman_hpc_all_eligible(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0", dockerfile="FROM ubuntu\n"),
            self._make_task(tmp_path, "t1", compose="services:\n  main:\n    image: x\n"),
        )
        results = inspect_harbor_runtime_eligibility(tasks, "podman-hpc")
        assert all(r.eligible for r in results)

    def test_apptainer_hpc_rejects_compose(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0", compose="services:\n  main:\n    image: x\n"),
        )
        results = inspect_harbor_runtime_eligibility(tasks, "apptainer-hpc")
        assert not results[0].eligible
        assert results[0].reason_code == "multi_service_compose"

    def test_apptainer_hpc_rejects_network_isolation(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0", dockerfile="FROM ubuntu\n", allow_internet=False),
        )
        results = inspect_harbor_runtime_eligibility(tasks, "apptainer-hpc")
        assert not results[0].eligible
        assert results[0].reason_code == "network_isolation"

    def test_apptainer_hpc_rejects_missing_sif(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0", docker_image="ubuntu:latest"),
        )
        sif_dir = tmp_path / "sif_cache"
        sif_dir.mkdir()
        results = inspect_harbor_runtime_eligibility(
            tasks, "apptainer-hpc", sif_cache_dir=str(sif_dir)
        )
        assert not results[0].eligible
        assert results[0].reason_code == "missing_sif_image"

    def test_apptainer_hpc_accepts_valid_task(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0", dockerfile="FROM ubuntu\n"),
        )
        # No sif_cache_dir = skip SIF check
        results = inspect_harbor_runtime_eligibility(tasks, "apptainer-hpc")
        assert results[0].eligible

    def test_singularity_hpc_routes_to_apptainer(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0", compose="services:\n  main:\n    image: x\n"),
        )
        results = inspect_harbor_runtime_eligibility(tasks, "singularity-hpc")
        assert not results[0].eligible
        assert results[0].reason_code == "multi_service_compose"

    def test_apptainer_hpc_rejects_missing_source(self, tmp_path):
        from llenvs.adapters.harbor import inspect_harbor_runtime_eligibility

        tasks = (
            self._make_task(tmp_path, "t0"),
        )
        results = inspect_harbor_runtime_eligibility(tasks, "apptainer-hpc")
        assert not results[0].eligible
        assert results[0].reason_code == "missing_container_source"
