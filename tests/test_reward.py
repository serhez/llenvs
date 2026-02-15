"""Tests for reward abstractions."""

import pytest
from llenvs.core.reward import Signal, SignalBundle, RewardType


class TestRewardType:
    """Tests for RewardType enum."""

    def test_enum_values(self):
        """Test that all reward types exist."""
        assert RewardType.OUTCOME
        assert RewardType.STEP
        assert RewardType.FORMAT
        assert RewardType.PROCESS

    def test_enum_distinct(self):
        """Test that enum values are distinct."""
        types = [RewardType.OUTCOME, RewardType.STEP, RewardType.FORMAT, RewardType.PROCESS]
        assert len(set(types)) == 4


class TestSignal:
    """Tests for Signal."""

    def test_creation(self):
        """Test basic signal creation."""
        signal = Signal(
            name="correctness",
            reward_type=RewardType.OUTCOME,
            reward=1.0,
        )
        assert signal.reward == 1.0
        assert signal.name == "correctness"
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.metadata is None

    def test_with_metadata(self):
        """Test signal with metadata."""
        signal = Signal(
            name="partial",
            reward_type=RewardType.STEP,
            reward=0.5,
            metadata={"reason": "partial match"},
        )
        assert signal.metadata == {"reason": "partial match"}

    def test_negative_value(self):
        """Test signal with negative value."""
        signal = Signal(
            name="penalty",
            reward_type=RewardType.STEP,
            reward=-1.0,
        )
        assert signal.reward == -1.0

    def test_zero_value(self):
        """Test signal with zero value."""
        signal = Signal(
            name="incorrect",
            reward_type=RewardType.OUTCOME,
            reward=0.0,
        )
        assert signal.reward == 0.0

    def test_immutability(self):
        """Test that signal is frozen."""
        signal = Signal(name="test", reward_type=RewardType.OUTCOME, reward=1.0)
        with pytest.raises(AttributeError):
            signal.reward = 0.5  # type: ignore

    def test_feedback_only(self):
        """Test signal with feedback but no numeric value."""
        signal = Signal(
            name="judge",
            reward_type=RewardType.PROCESS,
            feedback="Consider edge cases...",
        )
        assert signal.reward is None
        assert signal.feedback == "Consider edge cases..."

    def test_value_and_feedback(self):
        """Test signal with both value and feedback."""
        signal = Signal(
            name="code_exec",
            reward_type=RewardType.OUTCOME,
            reward=0.6,
            feedback="3/5 tests passed",
        )
        assert signal.reward == 0.6
        assert signal.feedback == "3/5 tests passed"

    def test_default_feedback_none(self):
        """Test that feedback defaults to None."""
        signal = Signal(name="test", reward_type=RewardType.OUTCOME, reward=1.0)
        assert signal.feedback is None

    def test_default_value_none(self):
        """Test that value defaults to None."""
        signal = Signal(name="test", reward_type=RewardType.OUTCOME)
        assert signal.reward is None


class TestSignalBundle:
    """Tests for SignalBundle."""

    def test_creation(self):
        """Test basic bundle creation."""
        signal1 = Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0)
        signal2 = Signal(name="b", reward_type=RewardType.FORMAT, reward=0.5)
        bundle = SignalBundle(signals=(signal1, signal2))

        assert len(bundle.signals) == 2

    def test_total(self):
        """Test total reward computation."""
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0),
                Signal(name="b", reward_type=RewardType.FORMAT, reward=0.5),
                Signal(name="c", reward_type=RewardType.STEP, reward=-0.2),
            )
        )
        assert bundle.total == pytest.approx(1.3)

    def test_total_empty(self):
        """Test total with no signals."""
        bundle = SignalBundle(signals=())
        assert bundle.total == 0.0

    def test_by_name_found(self):
        """Test finding signal by name."""
        signal = Signal(name="correctness", reward_type=RewardType.OUTCOME, reward=1.0)
        bundle = SignalBundle(signals=(signal,))

        found = bundle.by_name("correctness")
        assert found is not None
        assert found.reward == 1.0

    def test_by_name_not_found(self):
        """Test finding non-existent signal."""
        signal = Signal(name="correctness", reward_type=RewardType.OUTCOME, reward=1.0)
        bundle = SignalBundle(signals=(signal,))

        found = bundle.by_name("format")
        assert found is None

    def test_by_type(self):
        """Test filtering signals by type."""
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0),
                Signal(name="b", reward_type=RewardType.FORMAT, reward=0.5),
                Signal(name="c", reward_type=RewardType.OUTCOME, reward=0.8),
            )
        )

        outcome_signals = bundle.by_type(RewardType.OUTCOME)
        assert len(outcome_signals) == 2
        assert all(s.reward_type == RewardType.OUTCOME for s in outcome_signals)

        format_signals = bundle.by_type(RewardType.FORMAT)
        assert len(format_signals) == 1

        step_signals = bundle.by_type(RewardType.STEP)
        assert len(step_signals) == 0

    def test_single_factory(self):
        """Test creating bundle with single signal."""
        bundle = SignalBundle.single(
            reward=1.0,
            name="test",
            reward_type=RewardType.OUTCOME,
        )

        assert len(bundle.signals) == 1
        assert bundle.signals[0].reward == 1.0
        assert bundle.signals[0].name == "test"
        assert bundle.total == 1.0

    def test_single_factory_defaults(self):
        """Test single factory with default values."""
        bundle = SignalBundle.single(reward=0.5)

        assert bundle.signals[0].name == "reward"
        assert bundle.signals[0].reward_type == RewardType.OUTCOME

    def test_empty_factory(self):
        """Test creating empty bundle."""
        bundle = SignalBundle.empty()
        assert len(bundle.signals) == 0
        assert bundle.total == 0.0

    def test_immutability(self):
        """Test that bundle is frozen."""
        bundle = SignalBundle.single(reward=1.0)
        with pytest.raises(AttributeError):
            bundle.signals = ()  # type: ignore

    def test_total_skips_feedback_only(self):
        """Test that total skips feedback-only signals."""
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0),
                Signal(name="b", reward_type=RewardType.PROCESS, feedback="good"),
                Signal(name="c", reward_type=RewardType.STEP, reward=0.5),
            )
        )
        assert bundle.total == pytest.approx(1.5)

    def test_numeric_signals(self):
        """Test filtering to numeric-only signals."""
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0),
                Signal(name="b", reward_type=RewardType.PROCESS, feedback="good"),
                Signal(name="c", reward_type=RewardType.STEP, reward=0.5),
            )
        )
        numeric = bundle.numeric_signals()
        assert len(numeric) == 2
        assert all(s.reward is not None for s in numeric)

    def test_feedback_texts(self):
        """Test collecting feedback strings."""
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=1.0, feedback="correct"),
                Signal(name="b", reward_type=RewardType.PROCESS, feedback="good work"),
                Signal(name="c", reward_type=RewardType.STEP, reward=0.5),
            )
        )
        texts = bundle.feedback_texts()
        assert texts == ("correct", "good work")

    def test_feedback_texts_empty(self):
        """Test feedback_texts with no feedback."""
        bundle = SignalBundle.single(reward=1.0)
        assert bundle.feedback_texts() == ()

    def test_weighted_total_with_none_values(self):
        """Test weighted total correctly handles mixed None/numeric values."""
        bundle = SignalBundle(
            signals=(
                Signal(name="a", reward_type=RewardType.OUTCOME, reward=0.8, weight=2.0),
                Signal(name="b", reward_type=RewardType.PROCESS, feedback="info"),
                Signal(name="c", reward_type=RewardType.FORMAT, reward=1.0, weight=0.5),
            )
        )
        assert bundle.total == pytest.approx(0.8 * 2.0 + 1.0 * 0.5)
