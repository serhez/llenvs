"""Answer extraction protocols and implementations.

Extractors parse model responses to extract the final answer,
handling various formats (XML tags, regex patterns, etc.).
"""

import re
from collections.abc import Callable
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
        tag = re.escape(self.tag_name)
        self._pattern = re.compile(
            rf"<{tag}>(.*?)</{tag}>",
            re.DOTALL | re.IGNORECASE,
        )
        self._open_pattern = re.compile(
            rf"<{tag}>",
            re.IGNORECASE,
        )

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract answer from <tag>...</tag> markers."""
        matches = list(self._pattern.finditer(response))
        opening_tags = list(self._open_pattern.finditer(response))

        if not matches and not opening_tags:
            return None, {"found": False, "tag_name": self.tag_name}

        match = matches[-1] if matches else None
        last_open = opening_tags[-1] if opening_tags else None

        closed = match is not None and (
            last_open is None or last_open.start() < match.end()
        )

        if closed:
            assert match is not None
            content = match.group(1)
            match_start = match.start()
            match_end = match.end()
        else:
            assert last_open is not None
            content = response[last_open.end() :]
            match_start = last_open.start()
            match_end = len(response)

        if self.strip_whitespace:
            content = content.strip()

        return content, {
            "found": True,
            "tag_name": self.tag_name,
            "closed": closed,
            "match_start": match_start,
            "match_end": match_end,
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


class RawGenerationExtractor:
    """Extractor that returns the full response as the answer.

    Use this for explicit "no extraction" — the raw model generation is
    treated as the answer. Useful as a last resort in a CompositeExtractor
    or when the model's entire output is the answer.
    """

    def __init__(self, strip_whitespace: bool = True) -> None:
        self.strip_whitespace = strip_whitespace

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Return the full response as the answer."""
        content = response
        if self.strip_whitespace:
            content = content.strip()

        return content, {"found": True, "format": "raw", "is_full_response": True}


@dataclass
class SingleLineExtractor:
    """Wrap another extractor and reject multi-line extracted content.

    A single non-empty line is accepted. Surrounding blank lines are ignored,
    so inputs such as ``<action>\n north \n</action>`` still resolve to
    ``"north"``. Any extraction with more than one non-empty line is rejected.
    When ``max_chars`` is set, extracted single-line content longer than that
    limit is rejected as well.
    """

    inner: AnswerExtractor
    max_chars: int | None = None

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        result, metadata = self.inner.extract(response)
        if result is None:
            return None, metadata

        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if len(lines) <= 1:
            content = lines[0] if lines else result.strip()
            if self.max_chars is not None and len(content) > self.max_chars:
                enriched = dict(metadata)
                enriched.update(
                    {
                        "found": False,
                        "rejected_too_long": True,
                        "single_line": True,
                        "num_nonempty_lines": len(lines),
                        "max_chars": self.max_chars,
                        "content_length": len(content),
                    }
                )
                return None, enriched
            enriched = dict(metadata)
            enriched["single_line"] = True
            enriched["num_nonempty_lines"] = len(lines)
            return content, enriched

        enriched = dict(metadata)
        enriched.update(
            {
                "found": False,
                "rejected_multiline": True,
                "single_line": False,
                "num_nonempty_lines": len(lines),
            }
        )
        return None, enriched


@dataclass
class BoxedExtractor:
    r"""Extract answers from LaTeX \boxed{...} format.

    Handles nested braces correctly via balanced brace matching.
    Takes the last match when multiple \boxed{} are present.

    Attributes:
        strip_whitespace: Whether to strip whitespace from extracted content.
    """

    strip_whitespace: bool = True

    def _find_all_boxed(self, text: str) -> list[tuple[str, int, int]]:
        r"""Find all \boxed{...} with balanced braces.

        Returns list of (content, start_pos, end_pos) tuples.
        """
        results: list[tuple[str, int, int]] = []
        pattern = re.compile(r"\\boxed\s*\{")

        for match in pattern.finditer(text):
            start = match.end()
            depth = 1
            pos = start

            while pos < len(text) and depth > 0:
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    depth -= 1
                pos += 1

            if depth == 0:
                content = text[start : pos - 1]
                results.append((content, match.start(), pos))

        return results

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        r"""Extract answer from \boxed{...} markers."""
        matches = self._find_all_boxed(response)

        if not matches:
            return None, {"found": False, "format": "boxed"}

        content, start, end = matches[-1]

        if self.strip_whitespace:
            content = content.strip()

        return content, {
            "found": True,
            "format": "boxed",
            "match_start": start,
            "match_end": end,
            "num_matches": len(matches),
        }


@dataclass
class NumericExtractor:
    """Extract the last number from text.

    Handles integers, decimals, negatives, and comma-separated thousands.

    Attributes:
        strip_whitespace: Whether to strip whitespace from extracted content.
    """

    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        # Match numbers with optional commas for thousands, optional decimal
        self._pattern = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract the last number from text."""
        matches = list(self._pattern.finditer(response))

        if not matches:
            return None, {"found": False, "format": "numeric"}

        match = matches[-1]
        content = match.group(0).replace(",", "")

        if self.strip_whitespace:
            content = content.strip()

        return content, {
            "found": True,
            "format": "numeric",
            "match_start": match.start(),
            "match_end": match.end(),
            "num_matches": len(matches),
        }


class LastLineExtractor:
    """Extract the last non-empty line from text."""

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Return the last non-empty line."""
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        if not lines:
            return None, {"found": False, "format": "last_line"}

        return lines[-1], {
            "found": True,
            "format": "last_line",
            "num_lines": len(lines),
        }


@dataclass
class CodeBlockExtractor:
    """Extract content from markdown code fences.

    Takes the last code block. Optionally filters by language.

    Attributes:
        language: If set, only extract blocks with this language tag.
    """

    language: str | None = None

    def __post_init__(self) -> None:
        if self.language:
            self._pattern = re.compile(
                rf"```{re.escape(self.language)}\s*\n(.*?)```",
                re.DOTALL,
            )
        else:
            self._pattern = re.compile(
                r"```(?:\w*)\s*\n(.*?)```",
                re.DOTALL,
            )

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract content from code fences."""
        matches = list(self._pattern.finditer(response))

        if not matches:
            return None, {"found": False, "format": "code_block"}

        match = matches[-1]
        content = match.group(1).strip()

        meta: dict[str, Any] = {
            "found": True,
            "format": "code_block",
            "match_start": match.start(),
            "match_end": match.end(),
            "num_matches": len(matches),
        }

        if self.language:
            meta["language"] = self.language

        return content, meta


@dataclass
class PatternAnswerExtractor:
    """Extract answers from natural language patterns.

    Looks for patterns like "the answer is X", "therefore, X", "= X".
    Takes the last match. Captures everything after the pattern to
    end-of-line.

    Attributes:
        patterns: List of regex prefix patterns. Each should end at the
            boundary where the answer begins (the answer is captured as
            everything after the pattern to end-of-line).
    """

    patterns: list[str] | None = None

    _DEFAULT_PATTERNS: list[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.patterns is None:
            self.patterns = [
                r"the\s+answer\s+is\s+",
                r"therefore,?\s+",
                r"=\s+",
            ]

        self._compiled = [
            re.compile(rf"(?:{p})(.+?)(?=\.\s|\.$|\n|$)", re.IGNORECASE | re.MULTILINE)
            for p in self.patterns
        ]

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract answer from natural language patterns."""
        last_match: re.Match[str] | None = None
        last_pattern_idx: int = -1

        for idx, pattern in enumerate(self._compiled):
            for match in pattern.finditer(response):
                # Track the match with the latest position in text
                if last_match is None or match.start() > last_match.start():
                    last_match = match
                    last_pattern_idx = idx

        if last_match is None:
            return None, {"found": False, "format": "pattern_answer"}

        content = last_match.group(1).strip()

        return content, {
            "found": True,
            "format": "pattern_answer",
            "pattern_index": last_pattern_idx,
            "pattern": self.patterns[last_pattern_idx],
            "match_start": last_match.start(),
            "match_end": last_match.end(),
        }


@dataclass
class CleanedExtractor:
    """Transparent wrapper that applies cleaning around any extractor.

    Pre-cleaners transform the raw response before extraction.
    Post-cleaners transform the extracted answer after extraction.

    Attributes:
        inner: The extractor to delegate to.
        pre_cleaners: Functions applied to the response before extraction.
        post_cleaners: Functions applied to the extracted answer after extraction.
    """

    inner: AnswerExtractor
    pre_cleaners: list[Callable[[str], str]] = field(default_factory=list)
    post_cleaners: list[Callable[[str], str]] = field(default_factory=list)

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Apply pre-cleaners, extract, then apply post-cleaners."""
        cleaned_response = response
        for cleaner in self.pre_cleaners:
            cleaned_response = cleaner(cleaned_response)

        result, metadata = self.inner.extract(cleaned_response)

        if result is not None:
            for cleaner in self.post_cleaners:
                result = cleaner(result)

        metadata["pre_cleaners_applied"] = True
        metadata["post_cleaners_applied"] = True
        return result, metadata


class NativeExtractor:
    """Wraps a third-party extraction function as an AnswerExtractor.

    Adapts a function like reasoning_gym.utils.extract_answer to
    the AnswerExtractor protocol.

    Attributes:
        fn: The extraction function. Should take a string and return
            str | None.
        name: Identifier for the native source (for metadata).
    """

    def __init__(self, fn: Any, name: str = "native") -> None:
        self._fn = fn
        self._name = name

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Call the native extraction function."""
        result = self._fn(response)

        if result is None:
            return None, {
                "found": False,
                "format": "native",
                "native_name": self._name,
            }

        return str(result), {
            "found": True,
            "format": "native",
            "native_name": self._name,
        }
