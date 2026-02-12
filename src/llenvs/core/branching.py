"""Branching strategies for creating independent environment copies at checkpoints.

The branching system enables research workflows that need to estimate Q-values,
compute success rates from intermediate states, or explore multiple actions from
a decision point. Three strategies are provided:

- **DirectStrategy**: Zero-cost branching for pure-function environments.
- **ActionReplayStrategy**: Replays actions on a fresh env instance.
- **ProcessForkStrategy**: Uses ``os.fork()`` via EnvironmentServer for
  mutable-backend environments (Phase 2).

``BranchManager`` is the user-facing API that auto-resolves the best strategy.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from llenvs.core.environment import Environment
from llenvs.core.state import Action, State


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchHandle:
    """An independent environment copy at a checkpoint state.

    Attributes:
        environment: Independent env instance (or same for DirectStrategy).
        state: The checkpoint state to resume from.
        resource_id: Opaque ID for cleanup (e.g., forked PID).
    """

    environment: Any  # Environment[Any]
    state: State
    resource_id: str = ""


@dataclass(frozen=True)
class CheckpointHandle:
    """Opaque reference to a saved checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        state: The state at the checkpoint.
    """

    checkpoint_id: str
    state: State


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BranchingStrategy(Protocol):
    """Protocol for branching strategies."""

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    def can_branch(self, env: Any) -> bool:
        """Whether this strategy can branch the given environment."""
        ...

    def create_checkpoint(
        self,
        env: Any,
        state: State,
        actions: tuple[Action, ...],
        reset_options: dict[str, Any],
    ) -> CheckpointHandle:
        """Save a checkpoint at the given state.

        Args:
            env: The environment instance.
            state: Current state to checkpoint.
            actions: Actions taken to reach this state (for replay strategies).
            reset_options: Options used in the original reset() call
                (seed, task_index, etc.).

        Returns:
            Handle to the checkpoint.
        """
        ...

    def create_branch(self, handle: CheckpointHandle) -> BranchHandle:
        """Create an independent branch from a checkpoint.

        Can be called multiple times per checkpoint.

        Args:
            handle: The checkpoint to branch from.

        Returns:
            An independent environment + state pair.
        """
        ...

    def release_checkpoint(self, handle: CheckpointHandle) -> None:
        """Release resources associated with a checkpoint.

        Args:
            handle: The checkpoint to release.
        """
        ...


# ---------------------------------------------------------------------------
# DirectStrategy
# ---------------------------------------------------------------------------


class DirectStrategy:
    """Zero-cost branching for pure-function environments.

    For environments with ``spec.supports_branching=True``, ``step()`` is a
    pure function. The same environment instance can process any state, so
    branching simply returns the same env with the stored state.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, _DirectCheckpointData] = {}

    @property
    def name(self) -> str:
        return "direct"

    def can_branch(self, env: Any) -> bool:
        return getattr(getattr(env, "spec", None), "supports_branching", False)

    def create_checkpoint(
        self,
        env: Any,
        state: State,
        actions: tuple[Action, ...],
        reset_options: dict[str, Any],
    ) -> CheckpointHandle:
        cp_id = uuid.uuid4().hex
        self._checkpoints[cp_id] = _DirectCheckpointData(env=env, state=state)
        return CheckpointHandle(checkpoint_id=cp_id, state=state)

    def create_branch(self, handle: CheckpointHandle) -> BranchHandle:
        data = self._checkpoints[handle.checkpoint_id]
        return BranchHandle(environment=data.env, state=data.state)

    def release_checkpoint(self, handle: CheckpointHandle) -> None:
        self._checkpoints.pop(handle.checkpoint_id, None)


@dataclass
class _DirectCheckpointData:
    env: Any
    state: State


# ---------------------------------------------------------------------------
# ActionReplayStrategy
# ---------------------------------------------------------------------------


class ActionReplayStrategy:
    """Branching via action replay on fresh environment instances.

    Creates a new environment using ``env_factory``, resets with the same
    seed/task_index, and replays all stored actions to reach the checkpoint
    state. O(K) cost per branch where K = checkpoint depth.

    Args:
        env_factory: Callable that creates a fresh environment instance.
    """

    def __init__(self, env_factory: Callable[[], Any]) -> None:
        self._env_factory = env_factory
        self._checkpoints: dict[str, _ReplayCheckpointData] = {}

    @property
    def name(self) -> str:
        return "action_replay"

    def can_branch(self, env: Any) -> bool:
        spec = getattr(env, "spec", None)
        if spec is None:
            return False
        return (
            getattr(spec, "supports_seed", False)
            and getattr(spec, "supports_task_index", False)
        )

    def create_checkpoint(
        self,
        env: Any,
        state: State,
        actions: tuple[Action, ...],
        reset_options: dict[str, Any],
    ) -> CheckpointHandle:
        cp_id = uuid.uuid4().hex
        self._checkpoints[cp_id] = _ReplayCheckpointData(
            actions=actions,
            reset_options=reset_options,
        )
        return CheckpointHandle(checkpoint_id=cp_id, state=state)

    def create_branch(self, handle: CheckpointHandle) -> BranchHandle:
        data = self._checkpoints[handle.checkpoint_id]
        fresh_env = self._env_factory()

        # Build reset kwargs
        reset_kwargs: dict[str, Any] = {}
        seed = data.reset_options.get("seed")
        if seed is not None:
            reset_kwargs["seed"] = seed

        task_index = data.reset_options.get("task_index")
        options: dict[str, Any] = {}
        if task_index is not None:
            options["task_index"] = task_index
        if options:
            reset_kwargs["options"] = options

        state, _ = fresh_env.reset(**reset_kwargs)

        # Replay actions
        for action in data.actions:
            result = fresh_env.step(state, action)
            state = result.next_state

        return BranchHandle(environment=fresh_env, state=state)

    def release_checkpoint(self, handle: CheckpointHandle) -> None:
        self._checkpoints.pop(handle.checkpoint_id, None)


@dataclass
class _ReplayCheckpointData:
    actions: tuple[Action, ...]
    reset_options: dict[str, Any]


# ---------------------------------------------------------------------------
# ProcessForkStrategy
# ---------------------------------------------------------------------------


class ProcessForkStrategy:
    """Branching via ``os.fork()`` through an EnvironmentServer.

    On ``create_checkpoint()``, wraps the environment in an
    ``EnvironmentServer`` running in-process (threaded), then calls
    ``POST /fork`` to create a frozen "checkpoint server" (forked process
    at that exact state).

    On ``create_branch()``, calls ``POST /fork`` on the *checkpoint server*,
    producing an independent "branch server" + ``ContainerEnvironment``
    client. Each branch gets its own process with perfect state via
    ``os.fork()`` COW semantics.

    Only available on Unix (macOS/Linux).
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, _ForkCheckpointData] = {}

    @property
    def name(self) -> str:
        return "process_fork"

    def can_branch(self, env: Any) -> bool:
        # ProcessFork can branch any environment on Unix
        return sys.platform != "win32"

    def create_checkpoint(
        self,
        env: Any,
        state: State,
        actions: tuple[Action, ...],
        reset_options: dict[str, Any],
    ) -> CheckpointHandle:
        cp_id = uuid.uuid4().hex

        # Ensure we have a server wrapping this environment.
        # Start a local server in a thread, then fork it.
        # Pass hidden_type from the state so we don't need to call reset()
        # (which would mutate the environment's internal state).
        hidden_type = type(state.hidden) if state.hidden is not None else None
        server_url, http_server = _start_threaded_server(env, hidden_type=hidden_type)

        from llenvs.container.client import ContainerEnvironment

        client = ContainerEnvironment(url=server_url)

        # Fork the server to create a checkpoint snapshot
        checkpoint_url, checkpoint_pid = client.fork()
        client.close()

        # Create a client to the checkpoint server
        checkpoint_client = ContainerEnvironment(url=checkpoint_url)

        self._checkpoints[cp_id] = _ForkCheckpointData(
            checkpoint_url=checkpoint_url,
            checkpoint_pid=checkpoint_pid,
            checkpoint_client=checkpoint_client,
            local_server=http_server,
            branch_pids=[],
            state=state,
        )
        return CheckpointHandle(checkpoint_id=cp_id, state=state)

    def create_branch(self, handle: CheckpointHandle) -> BranchHandle:
        import os as _os

        data = self._checkpoints[handle.checkpoint_id]

        # Fork the checkpoint server to create an independent branch
        branch_url, branch_pid = data.checkpoint_client.fork()
        data.branch_pids.append(branch_pid)

        branch_env = _import_container_env()(url=branch_url)

        return BranchHandle(
            environment=branch_env,
            state=data.state,
            resource_id=str(branch_pid),
        )

    def release_checkpoint(self, handle: CheckpointHandle) -> None:
        import os as _os
        import signal as _signal

        data = self._checkpoints.pop(handle.checkpoint_id, None)
        if data is None:
            return

        # Kill all branch processes
        for pid in data.branch_pids:
            _kill_process(pid)

        # Kill the checkpoint process
        _kill_process(data.checkpoint_pid)

        # Shut down the local threaded server
        if data.local_server is not None:
            data.local_server.shutdown()


def _import_container_env():
    from llenvs.container.client import ContainerEnvironment
    return ContainerEnvironment


@dataclass
class _ForkCheckpointData:
    checkpoint_url: str
    checkpoint_pid: int
    checkpoint_client: Any  # ContainerEnvironment
    local_server: Any  # HTTPServer
    branch_pids: list[int]
    state: State


def _start_threaded_server(
    env: Any, *, hidden_type: type | None = None
) -> tuple[str, Any]:
    """Start an EnvironmentServer in a daemon thread, return (url, HTTPServer)."""
    import threading
    import http.client
    from http.server import HTTPServer
    from llenvs.container.server import EnvironmentHandler

    handler_class = type(
        "BoundHandler",
        (EnvironmentHandler,),
        {"environment": env, "hidden_type": hidden_type},
    )
    http_server = HTTPServer(("127.0.0.1", 0), handler_class)
    port = http_server.server_address[1]

    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()

    # Wait for ready
    import time
    for _ in range(100):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                break
        except Exception:
            time.sleep(0.05)

    return f"http://127.0.0.1:{port}", http_server


def _kill_process(pid: int) -> None:
    """Kill a process by PID, ignoring errors if already dead."""
    import os as _os
    import signal as _signal

    try:
        _os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        return
    # Wait for it to exit
    try:
        _os.waitpid(pid, 0)
    except ChildProcessError:
        pass


# ---------------------------------------------------------------------------
# resolve_strategy
# ---------------------------------------------------------------------------

# Strategy names that don't require ProcessForkStrategy (Phase 2 will add "process_fork")
_STRATEGY_NAMES = {"direct", "action_replay", "process_fork"}


def resolve_strategy(
    env: Any,
    *,
    preference: str | None = None,
    env_factory: Callable[[], Any] | None = None,
) -> BranchingStrategy:
    """Auto-resolve the best branching strategy for an environment.

    Priority (when no preference):
    1. DirectStrategy — if ``spec.supports_branching``
    2. ProcessForkStrategy — if Unix (added in Phase 2)
    3. ActionReplayStrategy — if ``spec.supports_seed`` and ``supports_task_index``

    Args:
        env: The environment to branch.
        preference: Optional strategy name override.
        env_factory: Factory for creating fresh env instances (needed for
            ActionReplayStrategy).

    Returns:
        A BranchingStrategy instance.

    Raises:
        ValueError: If preference is unknown or requirements aren't met.
        NotImplementedError: If no strategy can handle the environment.
    """
    if preference is not None:
        if preference not in _STRATEGY_NAMES:
            raise ValueError(
                f"Unknown branching strategy: {preference!r}. "
                f"Available: {sorted(_STRATEGY_NAMES)}"
            )
        return _create_strategy(preference, env, env_factory)

    # Auto-resolve
    spec = getattr(env, "spec", None)

    # 1. Direct
    if spec is not None and getattr(spec, "supports_branching", False):
        return DirectStrategy()

    # 2. ProcessForkStrategy (Unix only)
    if sys.platform != "win32":
        return ProcessForkStrategy()

    # 3. ActionReplay
    if (
        spec is not None
        and getattr(spec, "supports_seed", False)
        and getattr(spec, "supports_task_index", False)
    ):
        if env_factory is None:
            raise ValueError(
                "ActionReplayStrategy requires env_factory. "
                "Pass env_factory= to resolve_strategy() or BranchManager.create()."
            )
        return ActionReplayStrategy(env_factory=env_factory)

    raise NotImplementedError(
        f"No branching strategy available for {type(env).__name__} "
        f"(supports_branching={getattr(spec, 'supports_branching', '?')}, "
        f"supports_seed={getattr(spec, 'supports_seed', '?')}, "
        f"supports_task_index={getattr(spec, 'supports_task_index', '?')}). "
        f"Consider using a container with ProcessForkStrategy."
    )


def _create_strategy(
    name: str,
    env: Any,
    env_factory: Callable[[], Any] | None,
) -> BranchingStrategy:
    """Create a specific strategy by name."""
    if name == "direct":
        return DirectStrategy()
    elif name == "action_replay":
        if env_factory is None:
            raise ValueError(
                "ActionReplayStrategy requires env_factory. "
                "Pass env_factory= to resolve_strategy() or BranchManager.create()."
            )
        return ActionReplayStrategy(env_factory=env_factory)
    elif name == "process_fork":
        if sys.platform == "win32":
            raise NotImplementedError("ProcessForkStrategy is not available on Windows")
        return ProcessForkStrategy()
    else:
        raise ValueError(f"Unknown branching strategy: {name!r}")


# ---------------------------------------------------------------------------
# BranchManager
# ---------------------------------------------------------------------------


class BranchManager:
    """User-facing API for checkpointing and branching environments.

    Usage::

        with BranchManager.create(env) as mgr:
            state, info = env.reset(seed=42, options={"task_index": 0})
            # ... step to state_k ...
            mgr.checkpoint("step_k", state_k, actions, {"seed": 42, "task_index": 0})

            branch_env, branch_state = mgr.branch("step_k")
            result = branch_env.step(branch_state, some_action)

    Args:
        strategy: The branching strategy to use.
        env: The environment being branched.
    """

    def __init__(self, strategy: BranchingStrategy, env: Any) -> None:
        self._strategy = strategy
        self._env = env
        self._checkpoints: dict[str, CheckpointHandle] = {}

    @classmethod
    def create(
        cls,
        env: Any,
        *,
        strategy: str | None = None,
        env_factory: Callable[[], Any] | None = None,
    ) -> BranchManager:
        """Create a BranchManager with auto-resolved or explicit strategy.

        Args:
            env: The environment to manage branching for.
            strategy: Optional strategy name (``"direct"``, ``"action_replay"``,
                ``"process_fork"``). Auto-resolves if None.
            env_factory: Factory for creating fresh env instances. Required
                for ActionReplayStrategy.

        Returns:
            A configured BranchManager.
        """
        resolved = resolve_strategy(env, preference=strategy, env_factory=env_factory)
        return cls(resolved, env)

    def checkpoint(
        self,
        name: str,
        state: State,
        actions: tuple[Action, ...],
        reset_options: dict[str, Any],
    ) -> None:
        """Save a named checkpoint at the given state.

        If a checkpoint with this name already exists, it is overwritten
        (the old one is released first).

        Args:
            name: Checkpoint name (for later ``branch()`` / ``release()``).
            state: Current state to checkpoint.
            actions: Actions taken to reach this state.
            reset_options: Options used in the original ``reset()`` call.
        """
        # Release existing checkpoint with same name if present
        if name in self._checkpoints:
            self._strategy.release_checkpoint(self._checkpoints[name])

        handle = self._strategy.create_checkpoint(
            self._env, state, actions, reset_options
        )
        self._checkpoints[name] = handle

    def branch(self, name: str) -> tuple[Any, State]:
        """Create an independent branch from a named checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            Tuple of (environment, state) for the branch.

        Raises:
            KeyError: If checkpoint name is not found.
        """
        if name not in self._checkpoints:
            raise KeyError(
                f"No checkpoint named {name!r}. "
                f"Available: {sorted(self._checkpoints.keys())}"
            )
        handle = self._checkpoints[name]
        branch = self._strategy.create_branch(handle)
        return (branch.environment, branch.state)

    def release(self, name: str) -> None:
        """Release a named checkpoint and its resources.

        Args:
            name: Checkpoint name.

        Raises:
            KeyError: If checkpoint name is not found.
        """
        if name not in self._checkpoints:
            raise KeyError(
                f"No checkpoint named {name!r}. "
                f"Available: {sorted(self._checkpoints.keys())}"
            )
        handle = self._checkpoints.pop(name)
        self._strategy.release_checkpoint(handle)

    def close(self) -> None:
        """Release all checkpoints."""
        for handle in self._checkpoints.values():
            self._strategy.release_checkpoint(handle)
        self._checkpoints.clear()

    def __enter__(self) -> BranchManager:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
