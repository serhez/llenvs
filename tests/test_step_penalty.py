"""Tests for StepPenalty reward function."""

from unittest.mock import MagicMock

from llenvs.core.reward import RewardType, Signal, StepPenalty
from llenvs.core.state import State, StateMetadata


def _make_states(step: int = 1) -> tuple[State, State]:
    """Create dummy state and next_state for testing."""
    state = MagicMock(spec=State)
    next_state = MagicMock(spec=State)
    next_state.metadata = StateMetadata(step=step, episode_id="test")
    return state, next_state


def test_default_penalty_value():
    penalty = StepPenalty()
    state, next_state = _make_states()
    signal = penalty.compute(state, "action", next_state)
    assert signal.reward == -0.1


def test_custom_penalty():
    penalty = StepPenalty(penalty=0.5)
    state, next_state = _make_states()
    signal = penalty.compute(state, "action", next_state)
    assert signal.reward == -0.5


def test_signal_metadata():
    penalty = StepPenalty()
    state, next_state = _make_states(step=3)
    signal = penalty.compute(state, "action", next_state)
    assert signal.metadata == {"step": 3}


def test_signal_properties():
    penalty = StepPenalty()
    assert penalty.name == "step_penalty"
    assert penalty.reward_type == RewardType.STEP

    state, next_state = _make_states()
    signal = penalty.compute(state, "action", next_state)
    assert signal.name == "step_penalty"
    assert signal.reward_type == RewardType.STEP
    assert signal.weight == 1.0


def test_zero_penalty():
    penalty = StepPenalty(penalty=0.0)
    state, next_state = _make_states()
    signal = penalty.compute(state, "action", next_state)
    assert signal.reward == 0.0


def test_custom_name_and_type():
    penalty = StepPenalty(
        penalty=0.2,
        _name="turn_cost",
        _reward_type=RewardType.PROCESS,
        _weight=0.5,
    )
    assert penalty.name == "turn_cost"
    assert penalty.reward_type == RewardType.PROCESS

    state, next_state = _make_states()
    signal = penalty.compute(state, "action", next_state)
    assert signal.name == "turn_cost"
    assert signal.reward_type == RewardType.PROCESS
    assert signal.reward == -0.2
    assert signal.weight == 0.5


def test_with_extra_rewards():
    """Integration: StepPenalty works as an extra_rewards function."""
    from llenvs.core.reward import SignalBundle

    penalty = StepPenalty(penalty=0.05)
    state, next_state = _make_states(step=2)

    # Simulate what an environment does: compute all reward signals
    correctness_signal = Signal(
        name="correctness",
        reward_type=RewardType.OUTCOME,
        reward=1.0,
    )
    penalty_signal = penalty.compute(state, "action", next_state)

    bundle = SignalBundle(signals=(correctness_signal, penalty_signal))

    # Penalty signal is in the bundle
    assert bundle.by_name("step_penalty") is not None
    assert bundle.by_name("step_penalty").reward == -0.05

    # Total includes both
    assert bundle.total == 1.0 + (-0.05)

    # Can filter by type
    step_signals = bundle.by_type(RewardType.STEP)
    assert len(step_signals) == 1
    assert step_signals[0].name == "step_penalty"
