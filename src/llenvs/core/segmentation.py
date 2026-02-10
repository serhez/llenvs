"""Text segmentation strategies for multi-step environments.

Segmenters split text into logical segments (sentences, lines, numbered steps, etc.)
to enable per-step rewards and intermediate intervention in single-turn environments.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams


@runtime_checkable
class Segmenter(Protocol):
    """Protocol for text segmentation strategies.

    Segmenters provide two methods:
    - segment(): Split complete text into all segments
    - find_boundary(): Find the first segment boundary (for streaming)
    """

    def segment(self, text: str) -> list[str]:
        """Split text into segments.

        Args:
            text: Text to segment.

        Returns:
            List of text segments. Empty segments are excluded.
        """
        ...

    def find_boundary(self, text: str) -> int | None:
        """Find the first segment boundary in text.

        Used for streaming/generation-time segmentation.

        Args:
            text: Text to search for boundary.

        Returns:
            Character index after the boundary, or None if no boundary found.
        """
        ...


@dataclass
class SentenceSegmenter:
    """Split text on sentence boundaries.

    Sentences end with .!? followed by whitespace or end of string.
    Handles common abbreviations (Mr., Dr., etc.) to avoid false splits.
    """

    # Common abbreviations to avoid splitting on
    abbreviations: tuple[str, ...] = (
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "vs.",
        "etc.",
        "i.e.",
        "e.g.",
        "cf.",
        "approx.",
        "Fig.",
        "fig.",
        "Eq.",
        "eq.",
        "No.",
        "no.",
    )

    def segment(self, text: str) -> list[str]:
        """Split text into sentences."""
        if not text.strip():
            return []

        # Pattern: sentence-ending punctuation followed by whitespace or end
        # We use a positive lookbehind for the punctuation and positive lookahead for space/end
        pattern = r"(?<=[.!?])\s+"

        # Split on sentence boundaries
        raw_segments = re.split(pattern, text.strip())

        # Merge segments that were incorrectly split on abbreviations
        segments = []
        current = ""

        for seg in raw_segments:
            if current:
                current += " " + seg
            else:
                current = seg

            # Check if current segment ends with an abbreviation
            ends_with_abbrev = any(current.rstrip().endswith(abbr) for abbr in self.abbreviations)

            if not ends_with_abbrev:
                segments.append(current.strip())
                current = ""

        # Add any remaining text
        if current.strip():
            segments.append(current.strip())

        return [s for s in segments if s]

    def find_boundary(self, text: str) -> int | None:
        """Find the first sentence boundary."""
        if not text:
            return None

        # Look for sentence-ending punctuation followed by whitespace
        pattern = r"[.!?]\s+"
        match = re.search(pattern, text)

        if match:
            # Check if this is an abbreviation
            prefix = text[: match.start() + 1]  # Include the punctuation
            for abbr in self.abbreviations:
                if prefix.endswith(abbr):
                    # Try to find next boundary after this abbreviation
                    rest = text[match.end() :]
                    next_boundary = self.find_boundary(rest)
                    if next_boundary is not None:
                        return match.end() + next_boundary
                    return None

            return match.end()

        return None


@dataclass
class LineSegmenter:
    """Split text on newlines.

    Can split on single newlines or double newlines (paragraphs).
    """

    delimiter: str = "\n"

    def segment(self, text: str) -> list[str]:
        """Split text into lines or paragraphs."""
        if not text.strip():
            return []

        segments = text.split(self.delimiter)
        return [s.strip() for s in segments if s.strip()]

    def find_boundary(self, text: str) -> int | None:
        """Find the first line/paragraph boundary."""
        if not text:
            return None

        idx = text.find(self.delimiter)
        if idx == -1:
            return None

        # Return index after the delimiter
        return idx + len(self.delimiter)


@dataclass
class PatternSegmenter:
    """Split text on regex patterns.

    Useful for structured reasoning with numbered steps, transition words, etc.
    Splits BEFORE the pattern match (pattern becomes start of new segment).
    """

    patterns: tuple[str, ...] = (
        r"(?:^|\s)(?:Step\s+)?\d+[.:]\s",  # "1.", "Step 1:", "1:" at word boundary
        r"(?:^|\s)(?:Therefore|Thus|So|Hence),?\s",  # Conclusion markers
        r"(?:^|\s)(?:First|Second|Third|Finally),?\s",  # Ordinal markers
    )

    # Compiled patterns (set in __post_init__)
    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile the combined pattern."""
        combined = "|".join(f"({p})" for p in self.patterns)
        self._compiled = re.compile(combined, re.IGNORECASE)

    def segment(self, text: str) -> list[str]:
        """Split text on pattern matches."""
        if not text.strip():
            return []

        if self._compiled is None:
            return [text.strip()]

        # Find all match positions
        matches = list(self._compiled.finditer(text))

        if not matches:
            return [text.strip()]

        segments = []
        last_end = 0

        for match in matches:
            # Adjust start to not include leading whitespace in the split point
            # (the pattern may have matched leading whitespace for word boundary)
            split_start = match.start()
            if match.group().startswith((" ", "\t", "\n")):
                split_start += 1

            # Add text before this match (if any)
            if split_start > last_end:
                before = text[last_end:split_start].strip()
                if before:
                    segments.append(before)
            last_end = split_start

        # Add final segment
        if last_end < len(text):
            final = text[last_end:].strip()
            if final:
                segments.append(final)

        return segments

    def find_boundary(self, text: str) -> int | None:
        """Find the first pattern boundary."""
        if not text or self._compiled is None:
            return None

        # Find the first match
        match = self._compiled.search(text)
        if not match:
            return None

        # Adjust for leading whitespace in pattern
        split_start = match.start()
        if match.group().startswith((" ", "\t", "\n")):
            split_start += 1

        # If match is at the start, find the next one
        if split_start == 0:
            match = self._compiled.search(text, pos=match.end())
            if not match:
                return None
            split_start = match.start()
            if match.group().startswith((" ", "\t", "\n")):
                split_start += 1

        return split_start


@dataclass
class CompositeSegmenter:
    """Combine multiple segmenters.

    Applies segmenters in order, with later segmenters further splitting
    the results of earlier ones.
    """

    segmenters: tuple[Segmenter, ...]

    def segment(self, text: str) -> list[str]:
        """Apply all segmenters in sequence."""
        if not text.strip():
            return []

        segments = [text]

        for segmenter in self.segmenters:
            new_segments = []
            for seg in segments:
                new_segments.extend(segmenter.segment(seg))
            segments = new_segments

        return segments

    def find_boundary(self, text: str) -> int | None:
        """Find the earliest boundary from any segmenter."""
        if not text:
            return None

        boundaries = []
        for segmenter in self.segmenters:
            boundary = segmenter.find_boundary(text)
            if boundary is not None:
                boundaries.append(boundary)

        return min(boundaries) if boundaries else None


@dataclass
class TokenSegmenter:
    """Split text into fixed-size token chunks.

    Accepts any tokenizer with encode(str) -> list[int] and decode(list[int]) -> str
    methods (HuggingFace AutoTokenizer, vLLM tokenizers, tiktoken, etc.).

    Uses prefix decoding to find character boundaries: decodes token prefixes
    and slices the original text at the decoded length. This guarantees that
    concatenating segments exactly reconstructs the original text.
    """

    tokenizer: Any  # Object with encode(str)->list[int], decode(list[int])->str
    token_size: int = 64

    def segment(self, text: str) -> list[str]:
        """Split text into token-sized chunks."""
        if not text:
            return []

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.token_size:
            return [text]

        segments = []
        remaining = text
        offset = 0

        while offset < len(tokens):
            chunk_end = min(offset + self.token_size, len(tokens))
            if chunk_end >= len(tokens):
                # Last chunk: take all remaining text
                segments.append(remaining)
                break

            # Decode prefix up to chunk_end to find the character boundary
            prefix = self.tokenizer.decode(tokens[:chunk_end])
            char_boundary = len(prefix)

            segment = text[len(text) - len(remaining) : char_boundary]
            segments.append(segment)
            remaining = text[char_boundary:]
            offset = chunk_end

        return segments

    def find_boundary(self, text: str) -> int | None:
        """Find the first token-chunk boundary."""
        if not text:
            return None

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.token_size:
            return None

        prefix = self.tokenizer.decode(tokens[: self.token_size])
        return len(prefix)


DEFAULT_LLM_SEGMENTER_SYSTEM_PROMPT = (
    "You are a text segmentation assistant. Your job is to split text into "
    "meaningful logical segments (reasoning steps, thoughts, or conclusions). "
    "Always respond with ONLY a JSON array of strings. Each string must be "
    "copied exactly from the original text — do not rewrite, paraphrase, or "
    "correct anything."
)

DEFAULT_LLM_SEGMENTER_PROMPT = """\
Split the following text into meaningful logical segments. Return a JSON array \
of strings where each string is an exact substring copied from the original text. \
The segments should cover the entire text with no gaps or overlaps.

Text:
{raw_generation}"""


def _map_segments_to_original(original: str, proposed: list[str]) -> list[str]:
    """Map LLM-proposed segments back to exact positions in original text.

    Uses greedy substring matching: for each proposed segment, finds the
    earliest occurrence in the remaining original text. The segment text
    in the result is sliced from the original to preserve exact characters
    (including whitespace the LLM may have trimmed).

    Returns segments that concatenate to exactly reconstruct ``original``.
    """
    segments: list[str] = []
    pos = 0

    for i, seg in enumerate(proposed):
        needle = seg.strip()
        if not needle:
            continue
        idx = original.find(needle, pos)
        if idx == -1:
            # Segment not found — dump remainder as one segment
            if pos < len(original):
                segments.append(original[pos:])
            return segments

        # Determine the end of this matched region
        match_end = idx + len(needle)

        if i < len(proposed) - 1:
            # Not the last segment: include text from current pos up to
            # the end of the matched needle (captures leading whitespace
            # between segments).
            segments.append(original[pos:match_end])
            pos = match_end
        else:
            # Last segment: take everything from current pos to end
            segments.append(original[pos:])
            pos = len(original)

    # If nothing matched at all, return remainder
    if pos < len(original):
        segments.append(original[pos:])

    return segments


def default_segment_parser(original_text: str, llm_response: str) -> list[str]:
    """Parse LLM response into segments mapped to the original text.

    1. Strips markdown code fences from the response.
    2. Parses a JSON array of strings.
    3. Maps proposed segments back to exact positions in the original text
       via greedy substring matching.
    4. Falls back to ``[original_text]`` if parsing fails.
    """
    if not original_text:
        return []

    text = llm_response.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Try to find a JSON array in the text
    if not text.startswith("["):
        # Look for [...] embedded in surrounding prose
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)
        else:
            return [original_text]

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [original_text]

    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        return [original_text]

    if not parsed:
        return [original_text]

    return _map_segments_to_original(original_text, parsed)


@dataclass
class LLMSegmenter:
    """LLM-based segmentation for logical boundaries.

    Calls a ``ModelBackend`` to semantically segment text into meaningful
    reasoning steps, thoughts, or logical units. The LLM returns a JSON
    array of segment strings which are mapped back to exact positions in
    the original text via greedy substring matching.

    Attributes:
        backend: Any inference backend (vLLM, API, etc.).
        prompt_template: Format string with ``{raw_generation}`` placeholder.
        parser: Callable ``(original_text, llm_response) -> segments``.
        sampling_params: Controls LLM generation.
        system_prompt: Optional system message (``None`` to omit).
    """

    backend: ModelBackend
    prompt_template: str = DEFAULT_LLM_SEGMENTER_PROMPT
    parser: Callable[[str, str], list[str]] = default_segment_parser
    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(temperature=0.0, max_tokens=4096)
    )
    system_prompt: str | None = DEFAULT_LLM_SEGMENTER_SYSTEM_PROMPT

    def segment(self, text: str) -> list[str]:
        """Segment text using LLM-based semantic analysis."""
        if not text:
            return []

        messages: list[ChatMessage] = []
        if self.system_prompt is not None:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.append(
            ChatMessage(
                role="user",
                content=self.prompt_template.format(raw_generation=text),
            )
        )

        result = self.backend.generate_chat(messages, self.sampling_params)
        llm_response = result.text or ""

        return self.parser(text, llm_response)

    def find_boundary(self, text: str) -> int | None:
        """Not supported — LLM calls are too expensive for streaming.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LLMSegmenter does not support find_boundary(); use replay mode instead of streaming."
        )
