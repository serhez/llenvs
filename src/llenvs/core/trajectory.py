"""Trajectory tracking with checkpointing and branching.

Supports:
- Recording full episode histories
- Checkpointing at any point
- Branching from checkpoints to explore alternatives
- Querying states at any step
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
import uuid

from llenvs.core.state import Action, State
from llenvs.core.reward import SignalBundle

HiddenT = TypeVar("HiddenT")


@dataclass(frozen=True)
class Transition(Generic[HiddenT]):
    """A single state-action-reward transition.

    Attributes:
        state: State before action.
        action: Action taken.
        next_state: State after action.
        rewards: Reward bundle for this transition.
        info: Additional transition metadata.
    """

    state: State[HiddenT]
    action: Any
    next_state: State[HiddenT]
    rewards: SignalBundle
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Checkpoint(Generic[HiddenT]):
    """A saved point in a trajectory for later branching.

    Attributes:
        name: Identifier for this checkpoint.
        trajectory_id: ID of the trajectory this checkpoint belongs to.
        step_index: Index in the trajectory's transition list.
        state: The state at this checkpoint.
    """

    name: str
    trajectory_id: str
    step_index: int
    state: State[HiddenT]


@dataclass
class Trajectory(Generic[HiddenT]):
    """A sequence of transitions with checkpointing support.

    Trajectories are mutable (transitions can be appended) but individual
    transitions are immutable. Branching creates a new trajectory that
    shares history up to the branch point.

    Attributes:
        episode_id: Unique identifier for this episode.
        initial_state: Starting state of the episode.
    """

    episode_id: str
    initial_state: State[HiddenT]
    _transitions: list[Transition[HiddenT]] = field(default_factory=list)
    _checkpoints: dict[str, Checkpoint[HiddenT]] = field(default_factory=dict)
    _parent_trajectory_id: str | None = field(default=None)

    @classmethod
    def create(cls, initial_state: State[HiddenT]) -> "Trajectory[HiddenT]":
        """Create a new trajectory starting from the given state."""
        return cls(
            episode_id=initial_state.metadata.episode_id,
            initial_state=initial_state,
        )

    def add_transition(self, transition: Transition[HiddenT]) -> None:
        """Append a transition to the trajectory."""
        self._transitions.append(transition)

    def checkpoint(self, name: str) -> Checkpoint[HiddenT]:
        """Create a named checkpoint at the current position.

        Args:
            name: Unique name for this checkpoint.

        Returns:
            The created Checkpoint.

        Raises:
            ValueError: If a checkpoint with this name already exists.
        """
        if name in self._checkpoints:
            raise ValueError(f"Checkpoint '{name}' already exists")

        current_state = self.current_state
        checkpoint = Checkpoint(
            name=name,
            trajectory_id=self.episode_id,
            step_index=len(self._transitions),
            state=current_state,
        )
        self._checkpoints[name] = checkpoint
        return checkpoint

    def branch(self, checkpoint_name: str) -> "Trajectory[HiddenT]":
        """Create a new trajectory branching from a checkpoint.

        The new trajectory shares history up to the checkpoint but can
        diverge from that point.

        Args:
            checkpoint_name: Name of the checkpoint to branch from.

        Returns:
            A new Trajectory starting from the checkpoint state.

        Raises:
            KeyError: If the checkpoint doesn't exist.
        """
        if checkpoint_name not in self._checkpoints:
            raise KeyError(f"Checkpoint '{checkpoint_name}' not found")

        checkpoint = self._checkpoints[checkpoint_name]

        # Create new trajectory with copied history up to checkpoint
        new_trajectory: Trajectory[HiddenT] = Trajectory(
            episode_id=str(uuid.uuid4()),
            initial_state=self.initial_state,
            _parent_trajectory_id=self.episode_id,
        )

        # Copy transitions up to checkpoint
        for transition in self._transitions[: checkpoint.step_index]:
            new_trajectory.add_transition(transition)

        return new_trajectory

    def state_at(self, index: int) -> State[HiddenT]:
        """Get the state at a specific step index.

        Args:
            index: Step index (0 = initial state, 1 = after first action, etc.)

        Returns:
            The State at that step.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0:
            raise IndexError(f"Index {index} is negative")
        if index == 0:
            return self.initial_state
        if index > len(self._transitions):
            raise IndexError(f"Index {index} exceeds trajectory length {len(self._transitions)}")
        return self._transitions[index - 1].next_state

    @property
    def current_state(self) -> State[HiddenT]:
        """Get the current (most recent) state."""
        if not self._transitions:
            return self.initial_state
        return self._transitions[-1].next_state

    @property
    def transitions(self) -> tuple[Transition[HiddenT], ...]:
        """Get all transitions as an immutable tuple."""
        return tuple(self._transitions)

    @property
    def checkpoints(self) -> dict[str, Checkpoint[HiddenT]]:
        """Get all checkpoints."""
        return dict(self._checkpoints)

    def __len__(self) -> int:
        """Number of transitions in the trajectory."""
        return len(self._transitions)

    @property
    def total_reward(self) -> float:
        """Sum of all rewards across all transitions."""
        return sum(t.rewards.total for t in self._transitions)

    @property
    def is_terminal(self) -> bool:
        """Whether the trajectory has reached a terminal state."""
        return self.current_state.metadata.is_terminal

    def strip_images(self) -> "Trajectory[HiddenT]":
        """Return a new Trajectory with all images removed from observations.

        Useful for reducing memory usage when storing trajectories for later
        analysis. All other state (text, hidden, metadata, rewards) is preserved.

        Returns:
            New Trajectory with empty images in all observations.
        """
        from llenvs.core.state import Observation

        def _strip_obs(obs: Observation) -> Observation:
            if not obs.images:
                return obs
            return Observation(
                prompt=obs.prompt,
                messages=obs.messages,
                tool_results=obs.tool_results,
                available_tools=obs.available_tools,
                images=(),
            )

        def _strip_state(state: State[HiddenT]) -> State[HiddenT]:
            stripped_obs = _strip_obs(state.observation)
            if stripped_obs is state.observation:
                return state
            return State(
                observation=stripped_obs,
                hidden=state.hidden,
                metadata=state.metadata,
            )

        new_initial = _strip_state(self.initial_state)
        new_traj: Trajectory[HiddenT] = Trajectory(
            episode_id=self.episode_id,
            initial_state=new_initial,
        )

        for t in self._transitions:
            new_t = Transition(
                state=_strip_state(t.state),
                action=t.action,
                next_state=_strip_state(t.next_state),
                rewards=t.rewards,
                info=t.info,
            )
            new_traj.add_transition(new_t)

        return new_traj
