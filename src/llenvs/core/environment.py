"""Environment protocol and related types.

The Environment protocol defines the MDP interface for evaluation environments.
Key design: step() takes explicit state rather than maintaining internal state.
Environments with ``pure_step=True`` are genuine pure functions;
others enforce sequential continuity and raise on stale states.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.reward import SignalBundle, RewardFunction

HiddenT = TypeVar("HiddenT")


@dataclass(frozen=True)
class StepResult(Generic[HiddenT]):
    """Result of taking a step in the environment.

    Attributes:
        next_state: The state after taking the action.
        rewards: Bundle of reward signals for this transition.
        terminated: Whether the episode ended (goal reached or failed).
        truncated: Whether the episode was cut off (max steps, etc.).
        info: Additional step metadata.
    """

    next_state: State[HiddenT]
    rewards: SignalBundle
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Whether the episode is done (terminated or truncated)."""
        return self.terminated or self.truncated


@dataclass(frozen=True)
class EnvironmentSpec:
    """Specification describing an environment's properties.

    Attributes:
        name: Environment name (e.g., "sudoku", "leg_counting").
        adapter: Adapter that provides this environment (e.g., "reasoning_gym").
        max_steps: Maximum steps per episode (None = unlimited).
        observation_type: Type hint for observations.
        action_type: Type hint for actions.
        is_multi_turn: Whether the environment supports multi-turn interaction.
        pure_step: Whether ``step()`` is a pure function that can be
            called with any prior state (True) or only the most recent
            state from ``reset()``/``step()`` (False). Pure-step
            environments enable zero-cost DirectStrategy branching.
            Non-pure environments can still branch via ProcessFork or
            ActionReplay strategies.
        metadata: Additional environment-specific metadata.
    """

    name: str
    adapter: str = ""
    max_steps: int | None = None
    observation_type: type | None = None
    action_type: type | None = None
    is_multi_turn: bool = False
    supports_task_index: bool = True
    supports_len: bool = True
    supports_seed: bool = True
    pure_step: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class _StateContinuityTracker:
    """Validates that stateful environments receive states in order.

    Stateful adapters (those wrapping mutable backends) compose with this
    tracker. It records the expected ``(episode_id, step)`` after each
    ``reset()``/``step()`` and raises ``NotImplementedError`` when a stale
    or branched state is passed.

    Usage in adapters::

        self._state_tracker = _StateContinuityTracker()

        # At end of reset():
        self._state_tracker.track(state)

        # At start of step():
        self._state_tracker.validate(state, "MyEnvironment")

        # At end of step():
        self._state_tracker.track(result.next_state)
    """

    def __init__(self) -> None:
        self._expected_episode_id: str | None = None
        self._expected_step: int | None = None

    def track(self, state: "State") -> None:
        """Record the expected state after reset/step."""
        self._expected_episode_id = state.metadata.episode_id
        self._expected_step = state.metadata.step

    def validate(self, state: "State", env_name: str = "") -> None:
        """Raise if *state* doesn't match what was last tracked.

        No-op if ``track()`` has never been called (allows pre-reset use).

        Raises:
            NotImplementedError: On episode_id or step mismatch.
        """
        if self._expected_episode_id is None:
            return  # No tracking yet

        if state.metadata.episode_id != self._expected_episode_id:
            raise NotImplementedError(
                f"{env_name} has pure_step=False and cannot step "
                f"from a state belonging to a different episode "
                f"(expected episode {self._expected_episode_id!r}, "
                f"got {state.metadata.episode_id!r}). "
                f"Call reset() to start a new episode."
            )

        if state.metadata.step != self._expected_step:
            raise NotImplementedError(
                f"{env_name} has pure_step=False and cannot step "
                f"from a stale state (expected step {self._expected_step}, "
                f"got {state.metadata.step}). "
                f"Only the most recent state from reset()/step() is valid."
            )


@runtime_checkable
class Environment(Protocol[HiddenT]):
    """Protocol for MDP-style evaluation environments.

    State is passed explicitly to ``step()``. Environments with
    ``spec.pure_step == True`` are genuine pure functions
    supporting tree search and branching. Others enforce sequential
    continuity and raise on stale states.

    Type Parameters:
        HiddenT: Hidden state type (for reward computation).
    """

    @property
    def prompts(self) -> dict[str, str]:
        """Named prompt components used internally by this environment.

        Multi-step environments expose templates they use for building
        observations. Single-turn environments return an empty dict.
        Users can override these at construction time.

        Returns:
            Dict mapping prompt names to template strings.
        """
        ...

    @property
    def available_tools(self) -> tuple:
        """Get the tools available in this environment.

        Text-only environments return an empty tuple.
        Tool-aware environments return a tuple of ToolDefinitions.

        Returns:
            Tuple of available tool definitions.
        """
        ...

    @property
    def spec(self) -> EnvironmentSpec:
        """Get the environment specification."""
        ...

    @property
    def reward_functions(self) -> tuple[RewardFunction[HiddenT], ...]:
        """Get the reward functions used by this environment."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HiddenT], dict[str, Any]]:
        """Reset the environment and return initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Environment-specific options (e.g., task_index).

        Returns:
            Tuple of (initial_state, info_dict).
        """
        ...

    def step(
        self,
        state: State[HiddenT],
        action: Action,
    ) -> StepResult[HiddenT]:
        """Take an action from the given state.

        On environments with ``pure_step=True``, this is a pure
        function — the same state and action always produce the same
        result. Non-branchable environments raise ``NotImplementedError``
        if *state* is not the most recent state from ``reset()``/``step()``.

        Args:
            state: Current state.
            action: Action to take.

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        ...

    def compute_rewards(
        self,
        state: State[HiddenT],
        action: Action,
        next_state: State[HiddenT],
    ) -> SignalBundle:
        """Compute rewards for a transition.

        Called internally by step(), but can also be called directly
        for reward probing.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.

        Returns:
            SignalBundle containing all reward signals.
        """
        ...
