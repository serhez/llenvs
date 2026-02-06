"""Environment protocol and related types.

The Environment protocol defines the MDP interface for evaluation environments.
Key design: step() takes explicit state (pure function) rather than maintaining
internal state, enabling branching and parallel exploration.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from llenvs.core.state import State
from llenvs.core.reward import RewardBundle, RewardFunction

ObsT = TypeVar("ObsT")
HiddenT = TypeVar("HiddenT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class StepResult(Generic[ObsT, HiddenT]):
    """Result of taking a step in the environment.

    Attributes:
        next_state: The state after taking the action.
        rewards: Bundle of reward signals for this transition.
        terminated: Whether the episode ended (goal reached or failed).
        truncated: Whether the episode was cut off (max steps, etc.).
        info: Additional step metadata.
    """

    next_state: State[ObsT, HiddenT]
    rewards: RewardBundle
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
        metadata: Additional environment-specific metadata.
    """

    name: str
    adapter: str = ""
    max_steps: int | None = None
    observation_type: type | None = None
    action_type: type | None = None
    is_multi_turn: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol[ObsT, HiddenT, ActionT]):
    """Protocol for MDP-style evaluation environments.

    Environments are stateless - all state is passed explicitly to step().
    This enables:
    - Safe checkpointing and branching
    - Parallel exploration of multiple trajectories
    - Deterministic replay

    Type Parameters:
        ObsT: Observation type (what the model sees).
        HiddenT: Hidden state type (for reward computation).
        ActionT: Action type (model responses).
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
    def spec(self) -> EnvironmentSpec:
        """Get the environment specification."""
        ...

    @property
    def reward_functions(self) -> tuple[RewardFunction[ObsT, HiddenT, ActionT], ...]:
        """Get the reward functions used by this environment."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[ObsT, HiddenT], dict[str, Any]]:
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
        state: State[ObsT, HiddenT],
        action: ActionT,
    ) -> StepResult[ObsT, HiddenT]:
        """Take an action from the given state.

        This is a pure function - the same state and action always
        produce the same result.

        Args:
            state: Current state.
            action: Action to take.

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        ...

    def compute_rewards(
        self,
        state: State[ObsT, HiddenT],
        action: ActionT,
        next_state: State[ObsT, HiddenT],
    ) -> RewardBundle:
        """Compute rewards for a transition.

        Called internally by step(), but can also be called directly
        for reward probing.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.

        Returns:
            RewardBundle containing all reward signals.
        """
        ...
