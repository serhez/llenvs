"""Answer extraction protocols and implementations.

Extractors parse model responses to extract the final answer,
handling various formats (XML tags, regex patterns, etc.).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AnswerExtractor(Protocol):
    """Protocol for extracting answers from model responses.

    Extractors handle the messy reality of parsing model outputs,
    which may or may not follow the requested format.
    """

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract the answer from a model response.

        Args:
            response: The raw model response text.

        Returns:
            Tuple of (extracted_answer, metadata).
            - extracted_answer is None if no answer could be extracted.
            - metadata contains extraction details (e.g., match positions).
        """
        ...


@dataclass
class TagBasedExtractor:
    """Extract answers from XML-style tags.

    Looks for content between <tag>...</tag> markers.

    Attributes:
        tag_name: The tag name to search for (default: "answer").
        strip_whitespace: Whether to strip whitespace from extracted content.
    """

    tag_name: str = "answer"
    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        # Compile pattern for efficiency
        self._pattern = re.compile(
            rf"<{re.escape(self.tag_name)}>(.*?)</{re.escape(self.tag_name)}>",
            re.DOTALL | re.IGNORECASE,
        )

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract answer from <tag>...</tag> markers."""
        matches = list(self._pattern.finditer(response))

        if not matches:
            return None, {"found": False, "tag_name": self.tag_name}

        # Take the last match (in case model wrote multiple)
        match = matches[-1]
        content = match.group(1)

        if self.strip_whitespace:
            content = content.strip()

        return content, {
            "found": True,
            "tag_name": self.tag_name,
            "match_start": match.start(),
            "match_end": match.end(),
            "num_matches": len(matches),
        }


@dataclass
class RegexExtractor:
    """Extract answers using a custom regex pattern.

    Useful for formats like GSM8K's "#### answer" pattern.

    Attributes:
        pattern: Regex pattern with a capturing group for the answer.
        group_index: Which capture group contains the answer (default: 1).
        strip_whitespace: Whether to strip whitespace from extracted content.
    """

    pattern: str
    group_index: int = 1
    strip_whitespace: bool = True
    flags: int = field(default_factory=lambda: re.DOTALL)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, self.flags)

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract answer using the configured regex pattern."""
        matches = list(self._compiled.finditer(response))

        if not matches:
            return None, {"found": False, "pattern": self.pattern}

        # Take the last match
        match = matches[-1]

        try:
            content = match.group(self.group_index)
        except IndexError:
            return None, {
                "found": False,
                "pattern": self.pattern,
                "error": f"Group {self.group_index} not found in match",
            }

        if content is None:
            return None, {"found": False, "pattern": self.pattern}

        if self.strip_whitespace:
            content = content.strip()

        return content, {
            "found": True,
            "pattern": self.pattern,
            "match_start": match.start(),
            "match_end": match.end(),
            "num_matches": len(matches),
        }


@dataclass
class GSM8KExtractor:
    """Extract answers in GSM8K format: #### <number>.

    Handles common variations like "#### 42", "####42", "#### $42".
    """

    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        # Match #### followed by optional whitespace and the answer
        # Handles numbers, possibly with $ or commas
        self._pattern = re.compile(r"####\s*\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE)

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract answer from GSM8K #### format."""
        matches = list(self._pattern.finditer(response))

        if not matches:
            return None, {"found": False, "format": "gsm8k"}

        match = matches[-1]
        content = match.group(1)

        # Remove commas from numbers
        content = content.replace(",", "")

        if self.strip_whitespace:
            content = content.strip()

        return content, {
            "found": True,
            "format": "gsm8k",
            "match_start": match.start(),
            "match_end": match.end(),
            "num_matches": len(matches),
        }


@dataclass
class MultipleChoiceExtractor:
    """Extract multiple choice answers (A, B, C, D, etc.).

    Looks for patterns like "Answer: A", "(A)", "The answer is B".
    """

    choices: str = "ABCDE"
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        choice_pattern = f"[{re.escape(self.choices)}]"
        # Match various answer formats
        self._patterns = [
            re.compile(rf"(?:answer|choice)[\s:]*\(?({choice_pattern})\)?", flags),
            re.compile(rf"\(({choice_pattern})\)", flags),
            re.compile(rf"^({choice_pattern})[\.\)\s]", flags | re.MULTILINE),
        ]

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract multiple choice answer."""
        for pattern in self._patterns:
            matches = list(pattern.finditer(response))
            if matches:
                match = matches[-1]
                content = match.group(1).upper()
                return content, {
                    "found": True,
                    "format": "multiple_choice",
                    "match_start": match.start(),
                    "match_end": match.end(),
                    "pattern_used": pattern.pattern,
                }

        return None, {"found": False, "format": "multiple_choice"}


@dataclass
class CompositeExtractor:
    """Try multiple extractors in order until one succeeds.

    Attributes:
        extractors: List of extractors to try in order.
    """

    extractors: list[AnswerExtractor] = field(default_factory=list)

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Try each extractor in order until one succeeds."""
        for i, extractor in enumerate(self.extractors):
            result, metadata = extractor.extract(response)
            if result is not None:
                metadata["extractor_index"] = i
                metadata["extractor_type"] = type(extractor).__name__
                return result, metadata

        return None, {
            "found": False,
            "tried_extractors": [type(e).__name__ for e in self.extractors],
        }


class FallbackExtractor:
    """Extractor that returns the full response if no other extraction works.

    Useful as a last resort in a CompositeExtractor.
    """

    def __init__(self, strip_whitespace: bool = True) -> None:
        self.strip_whitespace = strip_whitespace

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Return the full response as the answer."""
        content = response
        if self.strip_whitespace:
            content = content.strip()

        return content, {"found": True, "format": "fallback", "is_full_response": True}
