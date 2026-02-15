"""Tests for math-verify reward function."""

import pytest
from typing import Any
from unittest.mock import patch

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import RewardType

try:
    import math_verify

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

from llenvs.core.math_verify import MathVerifyRewardFunction

# Re-use the ReasoningGymHidden type for test states
from llenvs.adapters.reasoning_gym import ReasoningGymHidden


def _make_state(expected_answer: str) -> State[ReasoningGymHidden]:
    """Create a test state with the given expected answer."""
    obs = Observation(prompt="What is 1+1?")
    hidden = ReasoningGymHidden(
        entry={"question": "What is 1+1?", "answer": expected_answer},
        expected_answer=expected_answer,
        task_index=0,
        dataset_name="test",
    )
    metadata = StateMetadata(step=0, episode_id="test", is_terminal=False)
    return State(observation=obs, hidden=hidden, metadata=metadata)


def _make_extractor(return_value: str | None):
    """Create a mock extractor that returns the given value."""

    class MockExtractor:
        def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
            return return_value, {"found": return_value is not None}

    return MockExtractor()


class TestMathVerifyRewardFunctionProperties:
    """Test basic properties of MathVerifyRewardFunction."""

    def test_name(self):
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("x"))
        assert reward.name == "math_correctness"

    def test_reward_type(self):
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("x"))
        assert reward.reward_type == RewardType.OUTCOME

    def test_custom_name(self):
        reward = MathVerifyRewardFunction(
            answer_extractor=_make_extractor("x"), name="custom_math"
        )
        assert reward.name == "custom_math"


class TestMathVerifyExtractionFailure:
    """Test behavior when extraction fails (returns None)."""

    def test_none_extraction_returns_zero(self):
        """When extractor returns None, reward should be 0.0."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor(None))
        state = _make_state("42")
        action = Action(text="I don't know")

        signal = reward.compute(state, action, state)

        assert signal.reward == 0.0
        assert signal.name == "math_correctness"
        assert signal.metadata["extracted"] is None


@pytest.mark.skipif(not HAS_MATH_VERIFY, reason="math-verify not installed")
class TestMathVerifyEquivalence:
    """Test mathematical equivalence checking (requires math-verify)."""

    def test_exact_match(self):
        """Identical strings should be equivalent."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("42"))
        state = _make_state("42")
        action = Action(text="<answer>42</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_integer_float_equivalence(self):
        """42 and 42.0 should be equivalent."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("42.0"))
        state = _make_state("42")
        action = Action(text="<answer>42.0</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_fraction_decimal_equivalence(self):
        """1/2 and 0.5 should be equivalent."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("0.5"))
        state = _make_state("1/2")
        action = Action(text="<answer>0.5</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_fraction_equivalence(self):
        """2/4 and 1/2 should be equivalent."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("2/4"))
        state = _make_state("1/2")
        action = Action(text="<answer>2/4</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_negative_number(self):
        """-3 and -3.0 should be equivalent."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("-3.0"))
        state = _make_state("-3")
        action = Action(text="<answer>-3.0</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_wrong_answer(self):
        """Clearly different answers should score 0."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("7"))
        state = _make_state("42")
        action = Action(text="<answer>7</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 0.0

    def test_latex_equivalence(self):
        """LaTeX fractions should be parsed correctly."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("\\frac{1}{2}"))
        state = _make_state("0.5")
        action = Action(text="<answer>\\frac{1}{2}</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_percentage_as_decimal(self):
        """50/100 and 1/2 should be equivalent."""
        reward = MathVerifyRewardFunction(
            answer_extractor=_make_extractor("50/100")
        )
        state = _make_state("1/2")
        action = Action(text="<answer>50/100</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_whitespace_insensitive(self):
        """Extra whitespace shouldn't matter."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("  42  "))
        state = _make_state("42")
        action = Action(text="<answer>  42  </answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0


class TestMathVerifyFallback:
    """Test fallback to normalized string comparison."""

    def test_fallback_exact_match(self):
        """When math-verify can't parse, fall back to string comparison."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("hello"))
        state = _make_state("hello")
        action = Action(text="<answer>hello</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_fallback_case_and_whitespace(self):
        """Fallback should normalize case and whitespace."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("  Hello World  "))
        state = _make_state("hello world")
        action = Action(text="<answer>  Hello World  </answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0

    def test_fallback_mismatch(self):
        """Fallback should return 0 on string mismatch."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("dog"))
        state = _make_state("cat")
        action = Action(text="<answer>dog</answer>")

        signal = reward.compute(state, action, state)
        assert signal.reward == 0.0

    def test_metadata_includes_method(self):
        """Metadata should indicate which comparison method was used."""
        reward = MathVerifyRewardFunction(answer_extractor=_make_extractor("42"))
        state = _make_state("42")
        action = Action(text="<answer>42</answer>")

        signal = reward.compute(state, action, state)
        assert "method" in signal.metadata


class TestMathVerifyWithRealExtractor:
    """Test with actual extractors instead of mocks."""

    def test_with_tag_extractor(self):
        """Test integration with TagBasedExtractor."""
        from llenvs.core.extraction import TagBasedExtractor

        extractor = TagBasedExtractor(tag_name="answer")
        reward = MathVerifyRewardFunction(answer_extractor=extractor)
        state = _make_state("42")
        action = Action(text="The answer is <answer>42</answer>.")

        signal = reward.compute(state, action, state)
        assert signal.reward == 1.0
        assert signal.metadata["extracted"] == "42"

    def test_with_tag_extractor_no_tag(self):
        """When extractor finds nothing, reward is 0."""
        from llenvs.core.extraction import TagBasedExtractor

        extractor = TagBasedExtractor(tag_name="answer")
        reward = MathVerifyRewardFunction(answer_extractor=extractor)
        state = _make_state("42")
        action = Action(text="The answer is 42, no tags here.")

        signal = reward.compute(state, action, state)
        assert signal.reward == 0.0
