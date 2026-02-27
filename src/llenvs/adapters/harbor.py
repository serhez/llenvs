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

import logging
import shlex
import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.async_utils import run_async
from llenvs.core.environment import (
    EnvironmentSpec,
    StepResult,
    _StateContinuityTracker,
)
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
class HarborHidden:
    """Hidden state for Harbor environments.

    Attributes:
        task_index: Index into the task list.
        task_name: The Harbor task identifier.
        instruction: Task instruction text.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        trajectory: Command history (frozen tuple).
    """

    task_index: int
    task_name: str
    instruction: str
    episode_step: int
    last_action: str | None = None
    trajectory: tuple[str, ...] = ()


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


def _run_verifier(
    verifier_factory: Any,
    task: Any,
    harbor_env: Any,
) -> dict[str, float]:
    """Run the verifier and return rewards dict."""
    verifier = verifier_factory(task, harbor_env)
    result = run_async(verifier.verify())
    return result.rewards


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
        exec_timeout: int = 120,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._tasks = tasks
        self._harbor_env_factory = harbor_env_factory
        self._verifier_factory = verifier_factory
        self._dataset_name = dataset_name
        self._max_steps = max_steps
        self._submit_keyword = submit_keyword
        self._verify_on_truncation = verify_on_truncation
        self._exec_timeout = exec_timeout

        self._native_rewards: tuple[RewardFunction, ...] = (HarborReward(),)
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
            metadata={"dataset": self._dataset_name},
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
                run_async(self._harbor_env.stop())
            except Exception:
                pass

        task = self._tasks[task_index]
        self._current_task = task

        # Create and start container
        self._harbor_env = self._harbor_env_factory(task)
        run_async(self._harbor_env.start())

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
            info={"task_index": task_index},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "task_name": hidden.task_name,
        }

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

        # Check for submit keyword
        if self._submit_keyword in action_text:
            terminated = True

        # Execute command in container (even for submit, to maintain trajectory)
        if not terminated:
            exec_result = run_async(
                self._harbor_env.exec(action_text, timeout_sec=self._exec_timeout)
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
                    logger.warning(f"Verifier failed: {e}")
                    reward_value = 0.0

        # Build next hidden
        next_hidden = HarborHidden(
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            instruction=state.hidden.instruction,
            episode_step=next_step,
            last_action=action_text,
            trajectory=state.hidden.trajectory + (action_text,),
        )

        # Build messages
        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action_text},
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

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
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
                run_async(self._harbor_env.stop())
            except Exception:
                pass
            self._harbor_env = None


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
        exec_timeout: int = 120,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._tasks = tasks
        self._harbor_env_factory = harbor_env_factory
        self._verifier_factory = verifier_factory
        self._dataset_name = dataset_name
        self._max_steps = max_steps
        self._verify_on_truncation = verify_on_truncation
        self._exec_timeout = exec_timeout

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
                run_async(self._harbor_env.stop())
            except Exception:
                pass

        task = self._tasks[task_index]
        self._current_task = task

        # Create and start container
        self._harbor_env = self._harbor_env_factory(task)
        run_async(self._harbor_env.start())

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
                    logger.warning(f"Verifier failed: {e}")
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
                run_async(self._harbor_env.stop())
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

    def _get_harbor(self) -> Any:
        """Import and return the harbor module."""
        try:
            import harbor

            return harbor
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
        harbor = self._get_harbor()
        registry = harbor.get_registry()
        return sorted(registry.list_datasets())

    def get_environment(
        self,
        name: str = "terminal-bench@2.0",
        tasks: tuple[Any, ...] | None = None,
        harbor_env_factory: Any | None = None,
        verifier_factory: Any | None = None,
        dataset_path: str | None = None,
        environment_type: str = "docker",
        tool_mode: bool = False,
        max_steps: int = 30,
        submit_keyword: str = "SUBMIT",
        exec_timeout: int = 120,
        verify_on_truncation: bool = True,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> HarborEnvironment | HarborToolEnvironment:
        """Create a Harbor environment.

        Args:
            name: Dataset name with optional version (e.g., "terminal-bench@2.0").
            tasks: Pre-loaded tuple of Harbor Task objects. If None, loaded
                from Harbor's registry or ``dataset_path``.
            harbor_env_factory: Callable ``(task) -> BaseEnvironment`` creating
                Harbor container environments. If None, built from harbor library.
            verifier_factory: Callable ``(task, env) -> Verifier``. If None,
                built from harbor library.
            dataset_path: Local path to dataset directory. Used when tasks
                and factories are not provided.
            environment_type: Harbor environment type (docker, daytona, etc.).
            tool_mode: If True, returns ``HarborToolEnvironment`` with structured
                tool calls. If False (default), returns ``HarborEnvironment``
                with text-based commands.
            max_steps: Maximum steps per episode.
            submit_keyword: Text mode only — keyword triggering submission.
            exec_timeout: Per-command timeout in seconds.
            verify_on_truncation: Run verifier when truncating at max_steps.
            extra_rewards: Additional reward functions.
            **kwargs: Passed to Harbor constructors.

        Returns:
            HarborEnvironment or HarborToolEnvironment.
        """
        dataset_name, _version = self._parse_name(name)

        # Load tasks and create factories from Harbor if not provided
        if tasks is None or harbor_env_factory is None or verifier_factory is None:
            harbor = self._get_harbor()

            if tasks is None:
                if dataset_path is not None:
                    tasks = tuple(sorted(harbor.load_tasks(dataset_path), key=lambda t: t.name))
                else:
                    registry = harbor.get_registry()
                    tasks = tuple(
                        sorted(
                            registry.get_tasks(dataset_name, version=_version),
                            key=lambda t: t.name,
                        )
                    )

            if harbor_env_factory is None:

                def harbor_env_factory(task: Any) -> Any:
                    return harbor.create_environment(
                        task, environment_type=environment_type, **kwargs
                    )

            if verifier_factory is None:

                def verifier_factory(task: Any, env: Any) -> Any:
                    return harbor.create_verifier(task, env)

        if tool_mode:
            return HarborToolEnvironment(
                tasks=tasks,
                harbor_env_factory=harbor_env_factory,
                verifier_factory=verifier_factory,
                dataset_name=dataset_name,
                max_steps=max_steps,
                verify_on_truncation=verify_on_truncation,
                exec_timeout=exec_timeout,
                extra_rewards=extra_rewards,
            )

        return HarborEnvironment(
            tasks=tasks,
            harbor_env_factory=harbor_env_factory,
            verifier_factory=verifier_factory,
            dataset_name=dataset_name,
            max_steps=max_steps,
            submit_keyword=submit_keyword,
            verify_on_truncation=verify_on_truncation,
            exec_timeout=exec_timeout,
            extra_rewards=extra_rewards,
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
