"""Multi-signal reward abstractions.

Supports multiple evaluation signals per step, enabling:
- Outcome rewards (correctness at episode end)
- Step rewards (per-turn feedback)
- Format rewards (did model follow instructions)
- Process rewards (intermediate reasoning quality)

Signals can carry numeric rewards, textual feedback, or both.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypeVar

from llenvs.core.state import State

HiddenT = TypeVar("HiddenT")


class RewardType(Enum):
    """Type of reward signal."""

    OUTCOME = auto()  # Final correctness (binary or graded)
    STEP = auto()  # Per-turn feedback
    FORMAT = auto()  # Did model follow formatting instructions
    PROCESS = auto()  # Intermediate reasoning quality


@dataclass(frozen=True)
class Signal:
    """A single evaluation signal.

    Can carry a numeric reward, textual feedback, or both:
    - Numeric-only: ``Signal(name="correctness", reward_type=OUTCOME, reward=1.0)``
    - Feedback-only: ``Signal(name="judge", reward_type=PROCESS, feedback="Consider edge cases...")``
    - Both: ``Signal(name="code_exec", reward_type=OUTCOME, reward=0.6, feedback="3/5 tests passed")``

    Attributes:
        name: Identifier for this signal (e.g., "correctness", "format").
        reward_type: Category of signal.
        reward: Optional numeric reward.
        feedback: Optional textual feedback for the agent.
        metadata: Optional additional information.
        weight: Weight for aggregation (default 1.0).
    """

    name: str
    reward_type: RewardType
    reward: float | None = None
    feedback: str | None = None
    metadata: dict[str, Any] | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class SignalBundle:
    """Collection of evaluation signals for a single transition.

    Enables multi-objective optimization, detailed reward analysis,
    and textual feedback delivery.

    Attributes:
        signals: Tuple of individual signals.
    """

    signals: tuple[Signal, ...]

    @property
    def total(self) -> float:
        """Weighted sum of all numeric rewards (skips feedback-only signals)."""
        return sum(s.reward * s.weight for s in self.signals if s.reward is not None)

    def by_name(self, name: str) -> Signal | None:
        """Get a signal by name."""
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None

    def by_type(self, reward_type: RewardType) -> tuple[Signal, ...]:
        """Get all signals of a given type."""
        return tuple(s for s in self.signals if s.reward_type == reward_type)

    def numeric_signals(self) -> tuple[Signal, ...]:
        """Signals that carry numeric rewards."""
        return tuple(s for s in self.signals if s.reward is not None)

    def feedback_texts(self) -> tuple[str, ...]:
        """Collect all non-None feedback strings."""
        return tuple(s.feedback for s in self.signals if s.feedback is not None)

    @classmethod
    def single(
        cls,
        reward: float,
        name: str = "reward",
        reward_type: RewardType = RewardType.OUTCOME,
    ) -> "SignalBundle":
        """Create a bundle with a single signal."""
        return cls(signals=(Signal(name=name, reward_type=reward_type, reward=reward),))

    @classmethod
    def empty(cls) -> "SignalBundle":
        """Create an empty signal bundle."""
        return cls(signals=())


@dataclass
class FormatReward:
    """Reward for checking if an answer was properly formatted.

    Returns 1.0 if the extractor successfully extracts an answer,
    0.0 otherwise. Generic across all adapters.
    """

    _name: str = "format"
    _reward_type: RewardType = RewardType.FORMAT

    def __init__(self, answer_extractor: Any) -> None:
        """Initialize with an answer extractor.

        Args:
            answer_extractor: AnswerExtractor to check format compliance.
        """
        self._answer_extractor = answer_extractor

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[Any],
        action: Any,
        next_state: State[Any],
    ) -> Signal:
        """Compute format reward (1.0 if answer extracted, 0.0 otherwise)."""
        extracted, extraction_meta = self._answer_extractor.extract(action.text)

        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=1.0 if extracted is not None else 0.0,
            metadata={"extraction": extraction_meta},
        )


@dataclass
class StepPenalty:
    """Per-step penalty that incentivizes efficient solutions.

    Emits a fixed negative reward on every step. When used alongside
    an OUTCOME reward for correctness, this creates pressure to solve
    tasks in fewer turns rather than exhausting the step budget.

    Attributes:
        penalty: Penalty per step (positive value, applied as negative reward).
        _name: Name of this reward function.
        _reward_type: Type of reward (STEP by default).
        _weight: Weight for the reward signal.
    """

    penalty: float = 0.1
    _name: str = "step_penalty"
    _reward_type: RewardType = RewardType.STEP
    _weight: float = 1.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[Any],
        action: Any,
        next_state: State[Any],
    ) -> Signal:
        """Compute step penalty (negative reward each step)."""
        return Signal(
            name=self._name,
            reward_type=self._reward_type,
            reward=-self.penalty,
            metadata={"step": next_state.metadata.step},
            weight=self._weight,
        )


class RewardFunction(Protocol[HiddenT]):
    """Protocol for computing evaluation signals.

    Reward functions are composable - an environment can have multiple
    reward functions that each contribute signals to the SignalBundle.
    """

    @property
    def name(self) -> str:
        """Unique name for this reward function."""
        ...

    @property
    def reward_type(self) -> RewardType:
        """Type of signal this function produces."""
        ...

    def compute(
        self,
        state: State[HiddenT],
        action: Any,
        next_state: State[HiddenT],
    ) -> Signal:
        """Compute the signal for a transition.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.

        Returns:
            A Signal for this transition.
        """
        ...
