"""Cleaning layer for answer extraction.

Pre-cleaners transform the raw response before extraction.
Post-cleaners transform the extracted answer after extraction.
Each cleaner is a simple str -> str function.
"""

import re
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Pre-cleaners (raw response -> cleaned response)
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|endoftext\|>|<\|im_end\|>|<\|im_start\|>|<pad>|</s>|<s>"
)


def strip_special_tokens(text: str) -> str:
    """Remove common LLM special tokens from text."""
    return _SPECIAL_TOKEN_PATTERN.sub("", text)


# ---------------------------------------------------------------------------
# Post-cleaners (extracted answer -> cleaned answer)
# ---------------------------------------------------------------------------

# Matches a trailing . or , that is NOT part of a decimal number (digit.digit).
_TRAILING_PUNCT_PATTERN = re.compile(r"(?<!\d)[.,]$|(?<=\d),(?!\d)$")


_DECIMAL_NUMBER_PATTERN = re.compile(r"^-?\d+\.\d+$")


def strip_trailing_punctuation(text: str) -> str:
    """Remove a trailing period or comma from extracted answers.

    Preserves decimal points in numbers like "3.14" (where the entire
    text is a decimal number).
    """
    if not text:
        return text
    if text[-1] in (".", ","):
        # Don't strip if the entire text is a decimal number like "3.14"
        if _DECIMAL_NUMBER_PATTERN.match(text):
            return text
        return text[:-1]
    return text


def strip_surrounding_quotes(text: str) -> str:
    """Remove matched surrounding quotes from extracted answers."""
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (
            text[0] == "'" and text[-1] == "'"
        ):
            return text[1:-1]
    return text


_SINGLE_DOLLAR_PATTERN = re.compile(r"^\$([^$]+)\$$")
_DOUBLE_DOLLAR_PATTERN = re.compile(r"^\$\$(.+)\$\$$", re.DOTALL)


def strip_latex_dollars(text: str) -> str:
    r"""Remove surrounding $...$ or $$...$$ from extracted math answers."""
    # Try double dollars first (more specific)
    m = _DOUBLE_DOLLAR_PATTERN.match(text)
    if m:
        return m.group(1)
    # Then single dollars
    m = _SINGLE_DOLLAR_PATTERN.match(text)
    if m:
        return m.group(1)
    return text


# ---------------------------------------------------------------------------
# Registries and defaults
# ---------------------------------------------------------------------------

PRE_CLEANERS: dict[str, Callable[[str], str]] = {
    "strip_special_tokens": strip_special_tokens,
}

POST_CLEANERS: dict[str, Callable[[str], str]] = {
    "strip_trailing_punctuation": strip_trailing_punctuation,
    "strip_surrounding_quotes": strip_surrounding_quotes,
    "strip_latex_dollars": strip_latex_dollars,
}

DEFAULT_PRE_CLEANERS: list[str] = ["strip_special_tokens"]
DEFAULT_POST_CLEANERS: list[str] = ["strip_trailing_punctuation"]


def resolve_cleaners(
    names: list[str] | None,
    kind: str,
) -> list[Callable[[str], str]]:
    """Resolve cleaner names to functions.

    Args:
        names: List of cleaner names, or None for defaults, or [] to disable.
        kind: "pre" or "post".

    Returns:
        List of cleaner functions.

    Raises:
        ValueError: If kind is not "pre" or "post".
        KeyError: If a cleaner name is not found in the registry.
    """
    if kind == "pre":
        registry = PRE_CLEANERS
        defaults = DEFAULT_PRE_CLEANERS
    elif kind == "post":
        registry = POST_CLEANERS
        defaults = DEFAULT_POST_CLEANERS
    else:
        raise ValueError(f"Invalid cleaner kind: {kind!r}. Must be 'pre' or 'post'.")

    if names is None:
        names = defaults

    result = []
    for name in names:
        if name not in registry:
            raise KeyError(
                f"Unknown {kind}-cleaner: {name!r}. "
                f"Available: {sorted(registry.keys())}"
            )
        result.append(registry[name])
    return result
