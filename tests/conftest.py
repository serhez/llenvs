"""Shared test fixtures."""

from typing import Any

import pytest

from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.trajectory import Trajectory, Transition


@pytest.fixture
def sample_metadata() -> StateMetadata:
    """Create sample state metadata."""
    return StateMetadata(
        step=0,
        episode_id="test-episode-001",
        is_terminal=False,
        info={"task_index": 0},
    )


@pytest.fixture
def sample_observation() -> Observation:
    """Create sample text observation."""
    return Observation(
        prompt="What is 2 + 2?",
        messages=(),
    )


@pytest.fixture
def sample_hidden() -> dict[str, Any]:
    """Create sample hidden state."""
    return {
        "expected_answer": "4",
        "entry": {"question": "What is 2 + 2?", "answer": "4"},
    }


@pytest.fixture
def sample_state(
    sample_observation: Observation,
    sample_hidden: dict[str, Any],
    sample_metadata: StateMetadata,
) -> State[dict[str, Any]]:
    """Create sample state."""
    return State(
        observation=sample_observation,
        hidden=sample_hidden,
        metadata=sample_metadata,
    )


@pytest.fixture
def sample_action() -> Action:
    """Create sample action."""
    return Action(text="The answer is <answer>4</answer>")


@pytest.fixture
def sample_reward_signal() -> Signal:
    """Create sample reward signal."""
    return Signal(
        name="correctness",
        reward_type=RewardType.OUTCOME,
        reward=1.0,
        metadata={"extracted": "4", "expected": "4"},
    )


@pytest.fixture
def sample_reward_bundle(sample_reward_signal: Signal) -> SignalBundle:
    """Create sample reward bundle."""
    format_signal = Signal(
        name="format",
        reward_type=RewardType.FORMAT,
        reward=1.0,
    )
    return SignalBundle(signals=(sample_reward_signal, format_signal))


@pytest.fixture
def terminal_state(
    sample_observation: Observation,
    sample_hidden: dict[str, Any],
) -> State[dict[str, Any]]:
    """Create terminal state."""
    return State(
        observation=sample_observation,
        hidden=sample_hidden,
        metadata=StateMetadata(
            step=1,
            episode_id="test-episode-001",
            is_terminal=True,
        ),
    )


@pytest.fixture
def sample_transition(
    sample_state: State,
    sample_action: Action,
    terminal_state: State,
    sample_reward_bundle: SignalBundle,
) -> Transition:
    """Create sample transition."""
    return Transition(
        state=sample_state,
        action=sample_action,
        next_state=terminal_state,
        rewards=sample_reward_bundle,
        info={},
    )


@pytest.fixture
def sample_trajectory(sample_state: State) -> Trajectory:
    """Create sample trajectory."""
    return Trajectory.create(sample_state)


class MockDataset:
    """Mock reasoning-gym dataset for testing."""

    def __init__(self, entries: list[dict[str, Any]] | None = None):
        self.entries = entries or [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 3 * 3?", "answer": "9"},
            {"question": "What is 10 / 2?", "answer": "5"},
        ]
        self.name = "mock_dataset"

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.entries[index]

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        expected = str(entry["answer"])
        return 1.0 if answer.strip() == expected.strip() else 0.0


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Create mock dataset."""
    return MockDataset()
