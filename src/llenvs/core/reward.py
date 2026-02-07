"""Multi-signal reward abstractions.

Supports multiple reward signals per step, enabling:
- Outcome rewards (correctness at episode end)
- Step rewards (per-turn feedback)
- Format rewards (did model follow instructions)
- Process rewards (intermediate reasoning quality)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypeVar

from llenvs.core.state import State

ObsT = TypeVar("ObsT")
HiddenT = TypeVar("HiddenT")
ActionT = TypeVar("ActionT")


class RewardType(Enum):
    """Type of reward signal."""

    OUTCOME = auto()  # Final correctness (binary or graded)
    STEP = auto()  # Per-turn feedback
    FORMAT = auto()  # Did model follow formatting instructions
    PROCESS = auto()  # Intermediate reasoning quality


@dataclass(frozen=True)
class RewardSignal:
    """A single reward signal.

    Attributes:
        value: The numeric reward value.
        name: Identifier for this reward (e.g., "correctness", "format").
        reward_type: Category of reward.
        metadata: Optional additional information about the reward.
    """

    value: float
    name: str
    reward_type: RewardType
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RewardBundle:
    """Collection of reward signals for a single transition.

    Enables multi-objective optimization and detailed reward analysis.

    Attributes:
        signals: Tuple of individual reward signals.
    """

    signals: tuple[RewardSignal, ...]

    @property
    def total(self) -> float:
        """Sum of all reward values."""
        return sum(s.value for s in self.signals)

    def by_name(self, name: str) -> RewardSignal | None:
        """Get a reward signal by name."""
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None

    def by_type(self, reward_type: RewardType) -> tuple[RewardSignal, ...]:
        """Get all reward signals of a given type."""
        return tuple(s for s in self.signals if s.reward_type == reward_type)

    @classmethod
    def single(
        cls,
        value: float,
        name: str = "reward",
        reward_type: RewardType = RewardType.OUTCOME,
    ) -> "RewardBundle":
        """Create a bundle with a single reward signal."""
        return cls(signals=(RewardSignal(value=value, name=name, reward_type=reward_type),))

    @classmethod
    def empty(cls) -> "RewardBundle":
        """Create an empty reward bundle."""
        return cls(signals=())


@dataclass
class FormatReward:
    """Reward for checking if an answer was properly formatted.

    Returns 1.0 if the extractor successfully extracts an answer,
    0.0 otherwise. Generic across all adapters.
    """

    _name: str = "format"
    _reward_type: RewardType = RewardType.FORMAT

    def __init__(self, extractor: Any) -> None:
        """Initialize with an extractor.

        Args:
            extractor: AnswerExtractor to check format compliance.
        """
        self._extractor = extractor

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[Any, Any],
        action: Any,
        next_state: State[Any, Any],
    ) -> RewardSignal:
        """Compute format reward (1.0 if answer extracted, 0.0 otherwise)."""
        extracted, extraction_meta = self._extractor.extract(action.text)

        return RewardSignal(
            value=1.0 if extracted is not None else 0.0,
            name=self.name,
            reward_type=self.reward_type,
            metadata={"extraction": extraction_meta},
        )


class RewardFunction(Protocol[ObsT, HiddenT, ActionT]):
    """Protocol for computing reward signals.

    Reward functions are composable - an environment can have multiple
    reward functions that each contribute signals to the RewardBundle.
    """

    @property
    def name(self) -> str:
        """Unique name for this reward function."""
        ...

    @property
    def reward_type(self) -> RewardType:
        """Type of reward this function produces."""
        ...

    def compute(
        self,
        state: State[ObsT, HiddenT],
        action: ActionT,
        next_state: State[ObsT, HiddenT],
    ) -> RewardSignal:
        """Compute the reward signal for a transition.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.

        Returns:
            A RewardSignal for this transition.
        """
        ...
