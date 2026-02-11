"""Tests for the Scorer integration class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType, RewardFunction
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.integrations.scoring import Scorer, ScoringResult


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockHidden:
    expected_answer: str
    task_index: int


class _MockRewardFn:
    """Gives 1.0 if action text contains the expected answer, 0.0 otherwise.

    Mimics real adapter behavior: stores extracted=None when extraction fails.
    """

    @property
    def name(self) -> str:
        return "correctness"

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self, state: State[_MockHidden], action: Any, next_state: State[_MockHidden]
    ) -> RewardSignal:
        text = action.text or ""
        expected = state.hidden.expected_answer
        # Simple "extraction": look for expected answer in text
        extracted = text if expected in text else None
        correct = extracted is not None
        return RewardSignal(
            value=1.0 if correct else 0.0,
            name=self.name,
            reward_type=self.reward_type,
            metadata={"extracted": extracted, "expected": expected},
        )


class _MockFormatRewardFn:
    """Gives 1.0 if action text starts with 'Answer:', 0.0 otherwise."""

    @property
    def name(self) -> str:
        return "format"

    @property
    def reward_type(self) -> RewardType:
        return RewardType.FORMAT

    def compute(
        self, state: State[_MockHidden], action: Any, next_state: State[_MockHidden]
    ) -> RewardSignal:
        formatted = (action.text or "").startswith("Answer:")
        return RewardSignal(
            value=1.0 if formatted else 0.0,
            name=self.name,
            reward_type=self.reward_type,
        )


class MockSingleTurnEnv:
    """Mock single-turn environment with 5 tasks."""

    def __init__(self, num_tasks: int = 5) -> None:
        self._num_tasks = num_tasks
        self._reward_fns = (_MockRewardFn(), _MockFormatRewardFn())

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock_single", adapter="mock", max_steps=1, is_multi_turn=False)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._reward_fns

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[_MockHidden], dict[str, Any]]:
        idx = (options or {}).get("task_index", 0)
        hidden = _MockHidden(expected_answer=str(idx * 10), task_index=idx)
        state = State(
            observation=Observation(prompt=f"What is {idx} * 10?"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        )
        return state, {"task_index": idx}

    def step(self, state: State[_MockHidden], action: Action) -> StepResult[_MockHidden]:
        rewards = self.compute_rewards(state, action, state)
        next_state = State(
            observation=state.observation,
            hidden=state.hidden,
            metadata=StateMetadata(
                step=1, episode_id=state.metadata.episode_id, is_terminal=True
            ),
        )
        return StepResult(next_state=next_state, rewards=rewards, terminated=True)

    def compute_rewards(
        self,
        state: State[_MockHidden],
        action: Action,
        next_state: State[_MockHidden],
    ) -> RewardBundle:
        signals = tuple(fn.compute(state, action, next_state) for fn in self._reward_fns)
        return RewardBundle(signals=signals)

    def __len__(self) -> int:
        return self._num_tasks


class MockMultiTurnEnv:
    """Mock multi-turn environment."""

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock_multi", adapter="mock", max_steps=5, is_multi_turn=True)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return ()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[Any], dict[str, Any]]:
        state = State(
            observation=Observation(prompt="multi-turn task"),
            hidden={},
            metadata=StateMetadata(step=0, episode_id="ep_0"),
        )
        return state, {}

    def step(self, state: State[Any], action: Action) -> StepResult[Any]:
        return StepResult(next_state=state, rewards=RewardBundle.empty(), terminated=False)

    def compute_rewards(
        self, state: State[Any], action: Action, next_state: State[Any]
    ) -> RewardBundle:
        return RewardBundle.empty()

    def __len__(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScoringResult:
    """Test the ScoringResult dataclass."""

    def test_frozen(self):
        result = ScoringResult(
            total=1.0,
            signals={"correctness": 1.0},
            extracted_answer="42",
            metadata={},
        )
        with pytest.raises(AttributeError):
            result.total = 2.0  # type: ignore[misc]

    def test_fields(self):
        result = ScoringResult(
            total=1.5,
            signals={"correctness": 1.0, "format": 0.5},
            extracted_answer="hello",
            metadata={"found": True},
        )
        assert result.total == 1.5
        assert result.signals == {"correctness": 1.0, "format": 0.5}
        assert result.extracted_answer == "hello"
        assert result.metadata == {"found": True}


class TestScorer:
    """Test the Scorer class."""

    def test_score_correct_response(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        # Task 0 expects "0" (0 * 10)
        result = scorer.score(0, "Answer:0")
        assert result.total == 2.0  # correctness=1.0 + format=1.0
        assert result.signals["correctness"] == 1.0
        assert result.signals["format"] == 1.0

    def test_score_incorrect_response(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        # Task 1 expects "10", but we give "wrong"
        result = scorer.score(1, "Answer:wrong")
        assert result.signals["correctness"] == 0.0
        assert result.signals["format"] == 1.0  # still formatted

    def test_score_wrong_format(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        # Task 2 expects "20" — correct content but no "Answer:" prefix
        result = scorer.score(2, "20")
        assert result.signals["correctness"] == 1.0
        assert result.signals["format"] == 0.0

    def test_score_completely_wrong(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        result = scorer.score(3, "no idea")
        assert result.total == 0.0
        assert result.signals["correctness"] == 0.0
        assert result.signals["format"] == 0.0

    def test_extracted_answer_is_string_on_success(self):
        """extracted_answer is a string when extraction succeeds."""
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        # Task 0 expects "0" — "Answer:0" contains "0"
        result = scorer.score(0, "Answer:0")
        assert isinstance(result.extracted_answer, str)

    def test_extracted_answer_is_none_on_failure(self):
        """extracted_answer is None when extraction fails, not empty string."""
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        # Task 1 expects "10" — "no idea" does not contain "10"
        result = scorer.score(1, "no idea")
        assert result.extracted_answer is None

    def test_score_batch(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        results = scorer.score_batch(
            [0, 1, 2],
            ["Answer:0", "Answer:wrong", "20"],
        )
        assert len(results) == 3
        assert results[0].signals["correctness"] == 1.0
        assert results[1].signals["correctness"] == 0.0
        assert results[2].signals["correctness"] == 1.0

    def test_score_batch_length_mismatch_raises(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        with pytest.raises(ValueError, match="must have the same length"):
            scorer.score_batch([0, 1], ["only one"])

    def test_multi_turn_raises(self):
        env = MockMultiTurnEnv()
        with pytest.raises(TypeError, match="multi-turn"):
            Scorer(env)

    def test_metadata_contains_reward_metadata(self):
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        result = scorer.score(0, "Answer:0")
        # Metadata should have per-signal metadata
        assert "correctness" in result.metadata
        assert result.metadata["correctness"]["expected"] == "0"

    def test_different_tasks(self):
        """Verify scoring different tasks uses correct expected answers."""
        env = MockSingleTurnEnv()
        scorer = Scorer(env)
        # Task 1 expects "10", task 4 expects "40"
        r1 = scorer.score(1, "10")
        r4 = scorer.score(4, "40")
        assert r1.signals["correctness"] == 1.0
        assert r4.signals["correctness"] == 1.0
        # Cross-check: wrong answer for task (mock uses `in`, so use disjoint string)
        r1_wrong = scorer.score(1, "99")
        assert r1_wrong.signals["correctness"] == 0.0
