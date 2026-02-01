"""Text segmentation strategies for multi-step environments.

Segmenters split text into logical segments (sentences, lines, numbered steps, etc.)
to enable per-step rewards and intermediate intervention in single-turn environments.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
        "vs.", "etc.", "i.e.", "e.g.", "cf.", "approx.",
        "Fig.", "fig.", "Eq.", "eq.", "No.", "no.",
    )

    def segment(self, text: str) -> list[str]:
        """Split text into sentences."""
        if not text.strip():
            return []

        # Pattern: sentence-ending punctuation followed by whitespace or end
        # We use a positive lookbehind for the punctuation and positive lookahead for space/end
        pattern = r'(?<=[.!?])\s+'

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
            ends_with_abbrev = any(
                current.rstrip().endswith(abbr) for abbr in self.abbreviations
            )

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
        pattern = r'[.!?]\s+'
        match = re.search(pattern, text)

        if match:
            # Check if this is an abbreviation
            prefix = text[:match.start() + 1]  # Include the punctuation
            for abbr in self.abbreviations:
                if prefix.endswith(abbr):
                    # Try to find next boundary after this abbreviation
                    rest = text[match.end():]
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
            if match.group().startswith((' ', '\t', '\n')):
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
        if match.group().startswith((' ', '\t', '\n')):
            split_start += 1

        # If match is at the start, find the next one
        if split_start == 0:
            match = self._compiled.search(text, pos=match.end())
            if not match:
                return None
            split_start = match.start()
            if match.group().startswith((' ', '\t', '\n')):
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
class SemanticSegmenter:
    """LLM-based segmentation for logical boundaries.

    Uses a small language model to identify semantic boundaries in text.
    This is a placeholder for future implementation.
    """

    model: Any = None  # Small LM for boundary detection

    def segment(self, text: str) -> list[str]:
        """Segment text using semantic analysis.

        Not yet implemented. Falls back to sentence segmentation.
        """
        # Placeholder: fall back to sentence segmentation
        return SentenceSegmenter().segment(text)

    def find_boundary(self, text: str) -> int | None:
        """Find semantic boundary using LLM.

        Not yet implemented. Falls back to sentence boundary.
        """
        # Placeholder: fall back to sentence boundary
        return SentenceSegmenter().find_boundary(text)
