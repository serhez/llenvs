"""Tests for the Harbor adapter."""

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
        start_error: bool = False,
        start_delay: float = 0.0,
    ):
        self._exec_results = exec_results or [MockExecResult(stdout="ok")]
        self._exec_index = 0
        self._started = False
        self._stopped = False
        self._start_error = start_error
        self._start_delay = start_delay
        self._start_force_build: bool | None = None
        self._stop_delete: bool | None = None
        self._exec_history: list[str] = []
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
        if self._exec_index < len(self._exec_results):
            result = self._exec_results[self._exec_index]
            self._exec_index += 1
            return result
        return MockExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        pass

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
    max_steps: int = 30,
    submit_keyword: str = "SUBMIT",
    verify_on_truncation: bool = True,
    start_timeout: int | None = 120,
    exec_timeout: int = 120,
    extra_rewards: tuple = (),
    dataset_name: str = "terminal-bench",
    state_capture_mode: str = "replay",
    snapshot_artifact_root: Path | None = None,
):
    """Create a HarborEnvironment with mocks."""
    from llenvs.adapters.harbor import HarborEnvironment

    tasks = tasks or _make_tasks()
    mock_env = harbor_env or MockHarborEnvironment()
    env_factory = _make_harbor_env_factory(mock_env)
    verifier_factory = _make_verifier_factory(verifier_result)

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

    def test_both_empty_shows_exit_code(self):
        from llenvs.adapters.harbor import _format_exec_result

        result = MockExecResult(stdout="", stderr="", return_code=0)
        assert "[exit code: 0]" in _format_exec_result(result)

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


class TestValidateReplayConsistency:
    """Tests for validate_replay_consistency() utility."""

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
            overlay_size_mb=256,
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

        # First call: --version (runtime info probe)
        assert calls[0][0] == ["apptainer", "--version"]

        # Second call: overlay create
        assert calls[1][0][:3] == ["apptainer", "overlay", "create"]
        assert "--size" in calls[1][0]
        assert "256" in calls[1][0]

        # Third call: instance start
        inst_cmd = calls[2][0]
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

        # Fourth call: bootstrap dirs
        bootstrap_cmd = calls[3][0]
        assert "apptainer" == bootstrap_cmd[0]
        assert "exec" == bootstrap_cmd[1]
        assert "--cleanenv" in bootstrap_cmd
        assert f"instance://{env._instance_name}" in bootstrap_cmd
        assert "mkdir -p /logs/agent /logs/verifier" in bootstrap_cmd

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
            fakeroot=True,
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

        # calls[0] = --version, calls[1] = overlay create, calls[2] = instance start
        inst_cmd = calls[2][0]
        assert "--fakeroot" in inst_cmd

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

    def test_no_checkpoint_methods(self, tmp_path):
        from llenvs.adapters.harbor import ApptainerHPCEnvironment

        (tmp_path / "Dockerfile").write_text("FROM ubuntu:latest\n")
        env = ApptainerHPCEnvironment(
            environment_dir=tmp_path,
            environment_name="task_01",
            session_id="session-1",
            trial_paths=self._make_trial_paths(tmp_path / "trial"),
            task_env_config=self._make_task_env_config(),
        )
        assert not hasattr(env, "export_checkpoint")
        assert not hasattr(env, "restore_checkpoint")


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
