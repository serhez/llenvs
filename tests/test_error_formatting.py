"""Tests for format_action_error utility."""

from llenvs.core.environment import format_action_error


class TestFormatActionError:
    """Tests for the format_action_error utility function."""

    def test_basic_error_only(self):
        """Minimal call with just error message."""
        result = format_action_error("Could not extract action from response.")
        assert "Invalid action: Could not extract action from response." in result
        assert "Please provide a valid action." in result

    def test_with_action_hint(self):
        """Error with action format hint."""
        hint = "Choose one of the following actions:\n  left\n  right"
        result = format_action_error("Bad action.", action_hint=hint)
        assert "Expected action format:" in result
        assert "left" in result
        assert "right" in result

    def test_with_current_state(self):
        """Error with current environment state."""
        state_text = "@ F F F\nF H F H\nF F F H\nH F F G"
        result = format_action_error("Invalid.", current_state=state_text)
        assert "Current state:" in result
        assert "@ F F F" in result
        assert "H F F G" in result

    def test_with_all_params(self):
        """Error with both action hint and current state."""
        hint = "Choose one of the following actions:\n  left\n  down"
        state_text = "@ F F F\nF H F H"
        result = format_action_error("Out of range.", action_hint=hint, current_state=state_text)
        # All sections present
        assert "Invalid action: Out of range." in result
        assert "Please provide a valid action." in result
        assert "Expected action format:" in result
        assert "left" in result
        assert "Current state:" in result
        assert "@ F F F" in result

    def test_section_ordering(self):
        """Sections appear in correct order: error, guidance, hint, state."""
        hint = "Enter an integer"
        state_text = "Position: 3"
        result = format_action_error("Bad.", action_hint=hint, current_state=state_text)
        lines = result.split("\n")
        # Find section positions
        error_idx = next(i for i, line in enumerate(lines) if "Invalid action" in line)
        guidance_idx = next(i for i, line in enumerate(lines) if "Please provide" in line)
        hint_idx = next(i for i, line in enumerate(lines) if "Expected action format" in line)
        state_idx = next(i for i, line in enumerate(lines) if "Current state" in line)
        assert error_idx < guidance_idx < hint_idx < state_idx

    def test_no_action_hint_section_when_none(self):
        """No 'Expected action format' section when action_hint is None."""
        result = format_action_error("Error.", current_state="some state")
        assert "Expected action format" not in result
        assert "Current state:" in result

    def test_no_current_state_section_when_none(self):
        """No 'Current state' section when current_state is None."""
        result = format_action_error("Error.", action_hint="some hint")
        assert "Current state" not in result
        assert "Expected action format:" in result

    def test_returns_string(self):
        """Return type is always a string."""
        result = format_action_error("Error.")
        assert isinstance(result, str)
