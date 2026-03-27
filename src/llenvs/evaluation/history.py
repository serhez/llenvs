"""History control for structured message building.

Provides ``HistoryEntry`` and built-in history functions that control how
prior (action, observation) pairs are included in the model's context.

A history function receives a list of ``HistoryEntry`` objects (one per
prior turn) and returns ``ChatMessage`` objects for the history portion
of the prompt. The runner inserts these between the task description and
the current state observation.

Built-in implementations:
    - ``full_history``: Include all prior turns (default).
    - ``no_history``: Drop all prior turns.
    - ``last_n_history(n)``: Include only the last *n* turns.
    - ``sliding_window_history(max_tokens, token_counter)``: Include as
      many recent turns as fit within a token budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llenvs.inference.protocol import ChatMessage

if TYPE_CHECKING:
    from llenvs.core.state import ImageContent

# Type alias for history functions.
HistoryFn = Callable[["list[HistoryEntry]"], list[ChatMessage]]

# Type alias for budget-aware history builders.
# Receives (entries, available_tokens) and returns messages.
BudgetHistoryFn = Callable[["list[HistoryEntry]", int], list[ChatMessage]]


@dataclass(frozen=True)
class PromptBudget:
    """Token-budget-aware prompt construction for TrajectoryRunner.

    When set on a runner, the runner computes the token cost of non-history
    parts (system prompt, task, current state) and passes the remaining
    budget to ``build_history``.

    Attributes:
        max_prompt_tokens: Maximum tokens for the entire prompt
            (``max_model_len - max_generation_tokens``).
        estimate_tokens: Callable that returns the estimated token count
            for a string.
        build_history: Callable that receives ``(entries, available_tokens)``
            and returns chat messages for the history portion.
    """

    max_prompt_tokens: int
    estimate_tokens: Callable[[str], int]
    build_history: BudgetHistoryFn


@dataclass(frozen=True)
class HistoryEntry:
    """A single prior (action, observation) pair from the trajectory.

    Attributes:
        action_text: The model's action text (full response or extracted).
        observation_text: The state observation after the action.
        observation_images: Images associated with the observation.
        step: The step number of this entry.
    """

    action_text: str
    observation_text: str
    observation_images: tuple[ImageContent, ...] = ()
    step: int = 0


def _entries_to_messages(entries: list[HistoryEntry]) -> list[ChatMessage]:
    """Convert history entries to alternating assistant/user messages.

    Entries with empty ``observation_text`` produce only an assistant
    message (no user message), which is the case for the final action
    before the current state.
    """
    messages: list[ChatMessage] = []
    for entry in entries:
        messages.append(ChatMessage(role="assistant", content=entry.action_text))
        if entry.observation_text:
            messages.append(
                ChatMessage(
                    role="user",
                    content=entry.observation_text,
                    images=entry.observation_images,
                )
            )
    return messages


def full_history(entries: list[HistoryEntry]) -> list[ChatMessage]:
    """Include all prior turns in the context."""
    return _entries_to_messages(entries)


def no_history(entries: list[HistoryEntry]) -> list[ChatMessage]:
    """Drop all prior turns — model sees only task + current state."""
    return []


def last_n_history(n: int) -> HistoryFn:
    """Return a history function that keeps only the last *n* turns.

    Args:
        n: Number of most recent turns to keep.
    """

    def _fn(entries: list[HistoryEntry]) -> list[ChatMessage]:
        if n <= 0:
            return []
        return _entries_to_messages(entries[-n:])

    return _fn


def sliding_window_history(
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> HistoryFn:
    """Return a history function that fits as many recent turns as possible
    within a token budget.

    Greedily adds entries from most recent to oldest, stopping when the
    next entry would exceed the budget.

    Args:
        max_tokens: Maximum total tokens for the history portion.
        token_counter: A callable that returns the token count for a string.
    """

    def _fn(entries: list[HistoryEntry]) -> list[ChatMessage]:
        if not entries:
            return []

        # Walk backwards, accumulating entries that fit
        selected: list[HistoryEntry] = []
        budget = max_tokens

        for entry in reversed(entries):
            cost = token_counter(entry.action_text) + token_counter(entry.observation_text)
            if cost > budget:
                break
            selected.append(entry)
            budget -= cost

        # Reverse to restore chronological order
        selected.reverse()
        return _entries_to_messages(selected)

    return _fn
