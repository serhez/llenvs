"""Tests for core state abstractions."""

import pytest
from env_evals.core.state import State, StateMetadata, TextObservation, TextAction


class TestStateMetadata:
    """Tests for StateMetadata."""

    def test_creation(self):
        """Test basic metadata creation."""
        meta = StateMetadata(
            step=0,
            episode_id="ep-001",
            is_terminal=False,
        )
        assert meta.step == 0
        assert meta.episode_id == "ep-001"
        assert meta.is_terminal is False
        assert meta.info == {}

    def test_with_info(self):
        """Test metadata with info dict."""
        meta = StateMetadata(
            step=5,
            episode_id="ep-002",
            is_terminal=True,
            info={"task_index": 42, "custom": "value"},
        )
        assert meta.info["task_index"] == 42
        assert meta.info["custom"] == "value"

    def test_immutability(self):
        """Test that metadata is frozen."""
        meta = StateMetadata(step=0, episode_id="ep-001", is_terminal=False)
        with pytest.raises(AttributeError):
            meta.step = 1  # type: ignore


class TestTextObservation:
    """Tests for TextObservation."""

    def test_creation(self):
        """Test basic observation creation."""
        obs = TextObservation(prompt="Hello, world!")
        assert obs.prompt == "Hello, world!"
        assert obs.messages == ()

    def test_with_messages(self):
        """Test observation with message history."""
        messages = (
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        )
        obs = TextObservation(prompt="How are you?", messages=messages)
        assert len(obs.messages) == 2
        assert obs.messages[0]["role"] == "user"

    def test_immutability(self):
        """Test that observation is frozen."""
        obs = TextObservation(prompt="test")
        with pytest.raises(AttributeError):
            obs.prompt = "changed"  # type: ignore


class TestTextAction:
    """Tests for TextAction."""

    def test_creation(self):
        """Test basic action creation."""
        action = TextAction(text="The answer is 42")
        assert action.text == "The answer is 42"

    def test_empty_text(self):
        """Test action with empty text."""
        action = TextAction(text="")
        assert action.text == ""

    def test_immutability(self):
        """Test that action is frozen."""
        action = TextAction(text="test")
        with pytest.raises(AttributeError):
            action.text = "changed"  # type: ignore


class TestState:
    """Tests for State."""

    def test_creation(self, sample_observation, sample_hidden, sample_metadata):
        """Test basic state creation."""
        state = State(
            observation=sample_observation,
            hidden=sample_hidden,
            metadata=sample_metadata,
        )
        assert state.observation == sample_observation
        assert state.hidden == sample_hidden
        assert state.metadata == sample_metadata

    def test_immutability(self, sample_state):
        """Test that state is frozen."""
        with pytest.raises(AttributeError):
            sample_state.observation = None  # type: ignore

    def test_with_metadata_update(self, sample_state):
        """Test creating new state with updated metadata."""
        new_state = sample_state.with_metadata(step=5, is_terminal=True)

        # Original unchanged
        assert sample_state.metadata.step == 0
        assert sample_state.metadata.is_terminal is False

        # New state updated
        assert new_state.metadata.step == 5
        assert new_state.metadata.is_terminal is True

        # Other fields preserved
        assert new_state.metadata.episode_id == sample_state.metadata.episode_id
        assert new_state.observation == sample_state.observation
        assert new_state.hidden == sample_state.hidden

    def test_with_metadata_preserves_info(self, sample_state):
        """Test that with_metadata preserves info dict."""
        new_state = sample_state.with_metadata(step=1)
        assert new_state.metadata.info == sample_state.metadata.info

    def test_generic_types(self):
        """Test state with different generic types."""
        # String observation, int hidden
        state: State[str, int] = State(
            observation="prompt",
            hidden=42,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )
        assert state.observation == "prompt"
        assert state.hidden == 42

        # List observation, dict hidden
        state2: State[list[str], dict[str, int]] = State(
            observation=["a", "b"],
            hidden={"count": 5},
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )
        assert state2.observation == ["a", "b"]
        assert state2.hidden["count"] == 5
