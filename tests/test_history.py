"""Tests for history control: HistoryEntry, built-in history functions."""

from __future__ import annotations

import pytest

from llenvs.core.state import ImageContent
from llenvs.evaluation.history import (
    HistoryEntry,
    full_history,
    last_n_history,
    no_history,
    sliding_window_history,
)
from llenvs.inference.protocol import ChatMessage

# =============================================================================
# HistoryEntry
# =============================================================================


class TestHistoryEntry:
    def test_construction(self):
        entry = HistoryEntry(
            action_text="go north",
            observation_text="You are in room 2.",
            step=1,
        )
        assert entry.action_text == "go north"
        assert entry.observation_text == "You are in room 2."
        assert entry.step == 1
        assert entry.observation_images == ()

    def test_defaults(self):
        entry = HistoryEntry(action_text="a", observation_text="b")
        assert entry.observation_images == ()
        assert entry.step == 0

    def test_with_images(self):
        img = ImageContent(data="abc", media_type="image/png")
        entry = HistoryEntry(
            action_text="look",
            observation_text="You see:",
            observation_images=(img,),
            step=3,
        )
        assert len(entry.observation_images) == 1
        assert entry.observation_images[0].data == "abc"

    def test_frozen(self):
        entry = HistoryEntry(action_text="a", observation_text="b")
        with pytest.raises(AttributeError):
            entry.action_text = "c"  # type: ignore[misc]


# =============================================================================
# full_history
# =============================================================================


class TestFullHistory:
    def test_empty(self):
        assert full_history([]) == []

    def test_single_entry(self):
        entries = [HistoryEntry(action_text="go north", observation_text="Room 2", step=1)]
        messages = full_history(entries)
        assert len(messages) == 2
        assert messages[0] == ChatMessage(role="assistant", content="go north")
        assert messages[1] == ChatMessage(role="user", content="Room 2")

    def test_multiple_entries(self):
        entries = [
            HistoryEntry(action_text="go north", observation_text="Room 2", step=1),
            HistoryEntry(action_text="go east", observation_text="Room 3", step=2),
        ]
        messages = full_history(entries)
        assert len(messages) == 4
        assert messages[0].role == "assistant"
        assert messages[0].content == "go north"
        assert messages[1].role == "user"
        assert messages[1].content == "Room 2"
        assert messages[2].role == "assistant"
        assert messages[2].content == "go east"
        assert messages[3].role == "user"
        assert messages[3].content == "Room 3"

    def test_includes_images(self):
        img = ImageContent(data="abc", media_type="image/png")
        entries = [
            HistoryEntry(
                action_text="look",
                observation_text="You see:",
                observation_images=(img,),
                step=1,
            )
        ]
        messages = full_history(entries)
        assert len(messages) == 2
        assert messages[1].images == (img,)


# =============================================================================
# no_history
# =============================================================================


class TestNoHistory:
    def test_always_empty(self):
        entries = [
            HistoryEntry(action_text="a", observation_text="b", step=1),
            HistoryEntry(action_text="c", observation_text="d", step=2),
        ]
        assert no_history(entries) == []

    def test_empty_input(self):
        assert no_history([]) == []


# =============================================================================
# last_n_history
# =============================================================================


class TestLastNHistory:
    def test_n_greater_than_entries(self):
        entries = [HistoryEntry(action_text="a", observation_text="b", step=1)]
        fn = last_n_history(5)
        messages = fn(entries)
        assert len(messages) == 2  # all entries returned

    def test_n_equal_to_entries(self):
        entries = [
            HistoryEntry(action_text="a", observation_text="b", step=1),
            HistoryEntry(action_text="c", observation_text="d", step=2),
        ]
        fn = last_n_history(2)
        messages = fn(entries)
        assert len(messages) == 4

    def test_n_less_than_entries(self):
        entries = [
            HistoryEntry(action_text="a", observation_text="b", step=1),
            HistoryEntry(action_text="c", observation_text="d", step=2),
            HistoryEntry(action_text="e", observation_text="f", step=3),
        ]
        fn = last_n_history(2)
        messages = fn(entries)
        # Only last 2 entries
        assert len(messages) == 4
        assert messages[0].content == "c"
        assert messages[1].content == "d"
        assert messages[2].content == "e"
        assert messages[3].content == "f"

    def test_n_zero(self):
        entries = [HistoryEntry(action_text="a", observation_text="b", step=1)]
        fn = last_n_history(0)
        assert fn(entries) == []

    def test_n_one(self):
        entries = [
            HistoryEntry(action_text="a", observation_text="b", step=1),
            HistoryEntry(action_text="c", observation_text="d", step=2),
        ]
        fn = last_n_history(1)
        messages = fn(entries)
        assert len(messages) == 2
        assert messages[0].content == "c"
        assert messages[1].content == "d"

    def test_preserves_images(self):
        img = ImageContent(data="x", media_type="image/png")
        entries = [
            HistoryEntry(action_text="a", observation_text="b", step=1),
            HistoryEntry(
                action_text="c",
                observation_text="d",
                observation_images=(img,),
                step=2,
            ),
        ]
        fn = last_n_history(1)
        messages = fn(entries)
        assert messages[1].images == (img,)


# =============================================================================
# sliding_window_history
# =============================================================================


class TestSlidingWindowHistory:
    @staticmethod
    def _char_counter(text: str) -> int:
        """Simple token counter that counts characters."""
        return len(text)

    def test_all_fit(self):
        entries = [
            HistoryEntry(action_text="ab", observation_text="cd", step=1),
        ]
        fn = sliding_window_history(max_tokens=100, token_counter=self._char_counter)
        messages = fn(entries)
        assert len(messages) == 2

    def test_none_fit(self):
        entries = [
            HistoryEntry(action_text="abcdef", observation_text="ghijkl", step=1),
        ]
        fn = sliding_window_history(max_tokens=5, token_counter=self._char_counter)
        messages = fn(entries)
        assert len(messages) == 0

    def test_partial_fit(self):
        """Only the most recent entries that fit within budget are returned."""
        entries = [
            HistoryEntry(action_text="aaaa", observation_text="bbbb", step=1),  # 8 chars
            HistoryEntry(action_text="cc", observation_text="dd", step=2),  # 4 chars
            HistoryEntry(action_text="e", observation_text="f", step=3),  # 2 chars
        ]
        # Budget of 6 chars: only last two entries (4+2=6) fit
        fn = sliding_window_history(max_tokens=6, token_counter=self._char_counter)
        messages = fn(entries)
        assert len(messages) == 4
        assert messages[0].content == "cc"
        assert messages[1].content == "dd"
        assert messages[2].content == "e"
        assert messages[3].content == "f"

    def test_exact_budget(self):
        entries = [
            HistoryEntry(action_text="ab", observation_text="cd", step=1),  # 4 chars
        ]
        fn = sliding_window_history(max_tokens=4, token_counter=self._char_counter)
        messages = fn(entries)
        assert len(messages) == 2

    def test_budget_exceeded_by_one(self):
        entries = [
            HistoryEntry(action_text="ab", observation_text="cd", step=1),  # 4 chars
        ]
        fn = sliding_window_history(max_tokens=3, token_counter=self._char_counter)
        messages = fn(entries)
        assert len(messages) == 0

    def test_empty_entries(self):
        fn = sliding_window_history(max_tokens=100, token_counter=self._char_counter)
        assert fn([]) == []

    def test_greedy_from_recent(self):
        """Most recent entries are prioritized over older ones."""
        entries = [
            HistoryEntry(action_text="a", observation_text="b", step=1),  # 2 chars
            HistoryEntry(action_text="ccc", observation_text="ddd", step=2),  # 6 chars
        ]
        # Budget of 6: only last entry fits
        fn = sliding_window_history(max_tokens=6, token_counter=self._char_counter)
        messages = fn(entries)
        assert len(messages) == 2
        assert messages[0].content == "ccc"
        assert messages[1].content == "ddd"
