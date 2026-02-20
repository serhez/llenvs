"""Tests for the Harbor adapter."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from llenvs.core.reward import RewardType
from llenvs.core.state import Action, State
from llenvs.core.tools import ToolCall

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
    ):
        self._exec_results = exec_results or [MockExecResult(stdout="ok")]
        self._exec_index = 0
        self._started = False
        self._stopped = False
        self._start_error = start_error
        self._exec_history: list[str] = []

    async def start(self) -> None:
        if self._start_error:
            raise RuntimeError("Container failed to start")
        self._started = True

    async def stop(self) -> None:
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
    exec_timeout: int = 120,
    extra_rewards: tuple = (),
    dataset_name: str = "terminal-bench",
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
        exec_timeout=exec_timeout,
        extra_rewards=extra_rewards,
    )


def _make_tool_env(
    tasks: tuple | None = None,
    harbor_env: MockHarborEnvironment | None = None,
    verifier_result: MockVerifierResult | None = None,
    max_steps: int = 30,
    verify_on_truncation: bool = True,
    exec_timeout: int = 120,
    extra_rewards: tuple = (),
    dataset_name: str = "terminal-bench",
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
        exec_timeout=exec_timeout,
        extra_rewards=extra_rewards,
    )


def _reset_env(env, task_index: int = 0):
    """Reset an environment and return (state, info)."""
    return env.reset(options={"task_index": task_index})


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
        assert h.trajectory == ()

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

    def test_reset_metadata(self):
        env = _make_env()
        state, _ = _reset_env(env)
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

    def test_reset_requires_task_index(self):
        env = _make_env()
        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_reset_task_index_out_of_bounds(self):
        env = _make_env(tasks=_make_tasks(3))
        with pytest.raises((ValueError, IndexError)):
            env.reset(options={"task_index": 5})

    def test_step_executes_command(self):
        mock_env = MockHarborEnvironment(
            exec_results=[MockExecResult(stdout="file1.txt\nfile2.txt")]
        )
        env = _make_env(harbor_env=mock_env)
        state, _ = _reset_env(env)
        result = env.step(state, Action(text="ls"))
        assert "file1.txt" in result.next_state.observation.state.text
        assert result.terminated is False
        assert result.truncated is False

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
        obs_text = result.next_state.observation.state.text
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
        with pytest.raises(ImportError, match="harbor"):
            adapter._get_harbor()

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
            harbor_env_factory=env_factory,
            verifier_factory=verifier_factory,
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
            harbor_env_factory=env_factory,
            verifier_factory=verifier_factory,
            tool_mode=True,
        )
        assert isinstance(env, HarborToolEnvironment)

    def test_list_environments_requires_harbor(self):
        from llenvs.adapters.harbor import HarborAdapter

        adapter = HarborAdapter()
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
