"""Tests for branching support and stale-state detection."""

import pytest

from llenvs.core.environment import EnvironmentSpec, _StateContinuityTracker
from llenvs.core.state import Observation, State, StateMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(episode_id: str = "ep-1", step: int = 0) -> State:
    return State(
        observation=Observation(prompt="test"),
        hidden=None,
        metadata=StateMetadata(
            step=step,
            episode_id=episode_id,
            is_terminal=False,
        ),
    )


# ---------------------------------------------------------------------------
# _StateContinuityTracker unit tests
# ---------------------------------------------------------------------------


class TestStateContinuityTracker:
    """Tests for _StateContinuityTracker."""

    def test_validate_passes_with_tracked_state(self):
        tracker = _StateContinuityTracker()
        state = _make_state(episode_id="ep-1", step=0)
        tracker.track(state)
        # Should not raise
        tracker.validate(state, "TestEnv")

    def test_validate_raises_on_wrong_episode_id(self):
        tracker = _StateContinuityTracker()
        state_a = _make_state(episode_id="ep-1", step=0)
        state_b = _make_state(episode_id="ep-2", step=0)

        tracker.track(state_a)

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            tracker.validate(state_b, "TestEnv")

    def test_validate_raises_on_wrong_step(self):
        tracker = _StateContinuityTracker()
        state_0 = _make_state(episode_id="ep-1", step=0)
        state_1 = _make_state(episode_id="ep-1", step=1)

        tracker.track(state_1)

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            tracker.validate(state_0, "TestEnv")

    def test_validate_is_noop_before_first_track(self):
        tracker = _StateContinuityTracker()
        state = _make_state(episode_id="ep-99", step=42)
        # Should not raise — no tracking yet
        tracker.validate(state, "TestEnv")

    def test_track_updates_expected(self):
        tracker = _StateContinuityTracker()

        state_0 = _make_state(episode_id="ep-1", step=0)
        tracker.track(state_0)
        tracker.validate(state_0, "TestEnv")  # passes

        state_1 = _make_state(episode_id="ep-1", step=1)
        tracker.track(state_1)
        tracker.validate(state_1, "TestEnv")  # passes

        # Old state now fails
        with pytest.raises(NotImplementedError, match="stale state"):
            tracker.validate(state_0, "TestEnv")

    def test_track_after_reset_invalidates_old_episode(self):
        tracker = _StateContinuityTracker()

        state_ep1 = _make_state(episode_id="ep-1", step=0)
        tracker.track(state_ep1)

        state_ep2 = _make_state(episode_id="ep-2", step=0)
        tracker.track(state_ep2)

        with pytest.raises(NotImplementedError, match="different episode"):
            tracker.validate(state_ep1, "TestEnv")

    def test_error_message_includes_env_name(self):
        tracker = _StateContinuityTracker()
        tracker.track(_make_state(episode_id="ep-1", step=0))

        with pytest.raises(NotImplementedError, match="MyAdapter"):
            tracker.validate(_make_state(episode_id="ep-2", step=0), "MyAdapter")


# ---------------------------------------------------------------------------
# EnvironmentSpec.pure_step tests
# ---------------------------------------------------------------------------


class TestEnvironmentSpecBranching:
    """Tests for EnvironmentSpec.pure_step."""

    def test_default_is_false(self):
        spec = EnvironmentSpec(name="test")
        assert spec.pure_step is False

    def test_explicit_true(self):
        spec = EnvironmentSpec(name="test", pure_step=True)
        assert spec.pure_step is True

    def test_explicit_false(self):
        spec = EnvironmentSpec(name="test", pure_step=False)
        assert spec.pure_step is False


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------


class TestBranchingSerializationRoundTrip:
    """Tests for EnvironmentSpec serialization with pure_step."""

    def test_round_trip_true(self):
        from llenvs.container.serialization import (
            deserialize_env_spec,
            serialize_env_spec,
        )

        spec = EnvironmentSpec(name="test", pure_step=True)
        data = serialize_env_spec(spec)
        restored = deserialize_env_spec(data)

        assert restored.pure_step is True

    def test_round_trip_false(self):
        from llenvs.container.serialization import (
            deserialize_env_spec,
            serialize_env_spec,
        )

        spec = EnvironmentSpec(name="test", pure_step=False)
        data = serialize_env_spec(spec)
        restored = deserialize_env_spec(data)

        assert restored.pure_step is False

    def test_backward_compat_missing_field(self):
        """Deserialization of old data without pure_step defaults to False."""
        from llenvs.container.serialization import deserialize_env_spec

        data = {"name": "legacy", "adapter": "old"}
        spec = deserialize_env_spec(data)
        assert spec.pure_step is False
