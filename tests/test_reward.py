"""Tests for reward abstractions."""

import pytest
from env_evals.core.reward import RewardSignal, RewardBundle, RewardType


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


class TestRewardSignal:
    """Tests for RewardSignal."""

    def test_creation(self):
        """Test basic signal creation."""
        signal = RewardSignal(
            value=1.0,
            name="correctness",
            reward_type=RewardType.OUTCOME,
        )
        assert signal.value == 1.0
        assert signal.name == "correctness"
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.metadata is None

    def test_with_metadata(self):
        """Test signal with metadata."""
        signal = RewardSignal(
            value=0.5,
            name="partial",
            reward_type=RewardType.STEP,
            metadata={"reason": "partial match"},
        )
        assert signal.metadata == {"reason": "partial match"}

    def test_negative_value(self):
        """Test signal with negative value."""
        signal = RewardSignal(
            value=-1.0,
            name="penalty",
            reward_type=RewardType.STEP,
        )
        assert signal.value == -1.0

    def test_zero_value(self):
        """Test signal with zero value."""
        signal = RewardSignal(
            value=0.0,
            name="incorrect",
            reward_type=RewardType.OUTCOME,
        )
        assert signal.value == 0.0

    def test_immutability(self):
        """Test that signal is frozen."""
        signal = RewardSignal(value=1.0, name="test", reward_type=RewardType.OUTCOME)
        with pytest.raises(AttributeError):
            signal.value = 0.5  # type: ignore


class TestRewardBundle:
    """Tests for RewardBundle."""

    def test_creation(self):
        """Test basic bundle creation."""
        signal1 = RewardSignal(value=1.0, name="a", reward_type=RewardType.OUTCOME)
        signal2 = RewardSignal(value=0.5, name="b", reward_type=RewardType.FORMAT)
        bundle = RewardBundle(signals=(signal1, signal2))

        assert len(bundle.signals) == 2

    def test_total(self):
        """Test total reward computation."""
        bundle = RewardBundle(
            signals=(
                RewardSignal(value=1.0, name="a", reward_type=RewardType.OUTCOME),
                RewardSignal(value=0.5, name="b", reward_type=RewardType.FORMAT),
                RewardSignal(value=-0.2, name="c", reward_type=RewardType.STEP),
            )
        )
        assert bundle.total == pytest.approx(1.3)

    def test_total_empty(self):
        """Test total with no signals."""
        bundle = RewardBundle(signals=())
        assert bundle.total == 0.0

    def test_by_name_found(self):
        """Test finding signal by name."""
        signal = RewardSignal(value=1.0, name="correctness", reward_type=RewardType.OUTCOME)
        bundle = RewardBundle(signals=(signal,))

        found = bundle.by_name("correctness")
        assert found is not None
        assert found.value == 1.0

    def test_by_name_not_found(self):
        """Test finding non-existent signal."""
        signal = RewardSignal(value=1.0, name="correctness", reward_type=RewardType.OUTCOME)
        bundle = RewardBundle(signals=(signal,))

        found = bundle.by_name("format")
        assert found is None

    def test_by_type(self):
        """Test filtering signals by type."""
        bundle = RewardBundle(
            signals=(
                RewardSignal(value=1.0, name="a", reward_type=RewardType.OUTCOME),
                RewardSignal(value=0.5, name="b", reward_type=RewardType.FORMAT),
                RewardSignal(value=0.8, name="c", reward_type=RewardType.OUTCOME),
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
        bundle = RewardBundle.single(
            value=1.0,
            name="test",
            reward_type=RewardType.OUTCOME,
        )

        assert len(bundle.signals) == 1
        assert bundle.signals[0].value == 1.0
        assert bundle.signals[0].name == "test"
        assert bundle.total == 1.0

    def test_single_factory_defaults(self):
        """Test single factory with default values."""
        bundle = RewardBundle.single(value=0.5)

        assert bundle.signals[0].name == "reward"
        assert bundle.signals[0].reward_type == RewardType.OUTCOME

    def test_empty_factory(self):
        """Test creating empty bundle."""
        bundle = RewardBundle.empty()
        assert len(bundle.signals) == 0
        assert bundle.total == 0.0

    def test_immutability(self):
        """Test that bundle is frozen."""
        bundle = RewardBundle.single(value=1.0)
        with pytest.raises(AttributeError):
            bundle.signals = ()  # type: ignore
