"""Math-verify reward function for semantic mathematical equivalence checking.

Uses HuggingFace's math-verify package to parse and compare mathematical
expressions, handling equivalences like "1/2" vs "0.5" or "(x+1)^2" vs
"x^2+2x+1" that simple string matching would miss.
"""

from dataclasses import dataclass
from typing import Any

from llenvs.core.extraction import AnswerExtractor
from llenvs.core.reward import RewardType, Signal
from llenvs.core.state import Action, State

try:
    from math_verify import parse, verify

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def _normalize(s: str) -> str:
    """Normalize a string for fallback comparison (lowercase, strip whitespace)."""
    return " ".join(s.lower().split())


def _math_verify_check(predicted: str, expected: str) -> tuple[bool | None, str]:
    """Try math-verify equivalence check.

    Returns:
        Tuple of (result, method).
        result is True/False if math-verify could parse and compare,
        or None if parsing failed (empty parse result or exception).
        method describes which comparison path was taken.
    """
    if not HAS_MATH_VERIFY:
        return None, "unavailable"

    try:
        gold = parse(expected)
        pred = parse(predicted)
        # parse() returns [] for unparseable strings
        if not gold or not pred:
            return None, "math_verify_parse_failed"
        result = verify(gold, pred)
        return result, "math_verify"
    except Exception:
        return None, "math_verify_parse_failed"


@dataclass
class MathVerifyRewardFunction:
    """Reward function using math-verify for semantic mathematical comparison.

    Falls back to normalized string comparison when math-verify cannot parse
    the expressions (e.g., for non-mathematical text answers).

    Attributes:
        name: Identifier for this reward function.
        reward_type: Always OUTCOME.
    """

    _answer_extractor: AnswerExtractor
    _name: str = "math_correctness"
    _reward_type: RewardType = RewardType.OUTCOME

    def __init__(
        self,
        answer_extractor: AnswerExtractor,
        name: str = "math_correctness",
    ) -> None:
        self._answer_extractor = answer_extractor
        self._name = name
        self._reward_type = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> Signal:
        """Compute math-correctness reward.

        Flow:
        1. Extract answer string using the configured extractor.
        2. Try math-verify parse + verify for semantic equivalence.
        3. On parse failure, fall back to normalized string comparison.
        """
        extracted, extraction_meta = self._answer_extractor.extract(action.text)

        if extracted is None:
            return Signal(
                name=self.name,
                reward_type=self.reward_type,
                reward=0.0,
                metadata={"extracted": None, "extraction": extraction_meta},
            )

        expected = state.hidden.expected_answer

        # Try math-verify first
        result, method = _math_verify_check(extracted, expected)

        if result is not None:
            # math-verify successfully parsed and compared
            return Signal(
                name=self.name,
                reward_type=self.reward_type,
                reward=1.0 if result else 0.0,
                metadata={
                    "extracted": extracted,
                    "expected": expected,
                    "method": method,
                    "extraction": extraction_meta,
                },
            )

        # Fallback: normalized string comparison
        is_match = _normalize(extracted) == _normalize(expected)
        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=1.0 if is_match else 0.0,
            metadata={
                "extracted": extracted,
                "expected": expected,
                "method": "string_normalized",
                "extraction": extraction_meta,
            },
        )
