"""Tests for text-based tool call parsing."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from llenvs.core.tools import ToolDefinition, ToolParameter, ToolParameterType
from llenvs.core.tool_parsing import (
    HermesToolCallParser,
    ParsedToolResponse,
)
from llenvs.core.state import Observation, State, StateMetadata, Action
from llenvs.core.environment import EnvironmentSpec
from llenvs.core.reward import SignalBundle, RewardType
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
    BackendCapabilities,
)


@pytest.fixture
def sample_tools() -> tuple[ToolDefinition, ...]:
    """Create sample tool definitions."""
    return (
        ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters=(
                ToolParameter(
                    name="city",
                    type=ToolParameterType.STRING,
                    description="City name",
                ),
            ),
        ),
        ToolDefinition(
            name="calculate",
            description="Evaluate a math expression",
            parameters=(
                ToolParameter(
                    name="expression",
                    type=ToolParameterType.STRING,
                    description="Math expression",
                ),
            ),
        ),
    )


class TestHermesToolCallParserFormatTools:
    """Tests for HermesToolCallParser.format_tools()."""

    def test_renders_xml_with_tools(self, sample_tools):
        """Test that format_tools renders correct XML structure."""
        parser = HermesToolCallParser()
        result = parser.format_tools(sample_tools)

        assert "<tools>" in result
        assert "</tools>" in result
        assert "get_weather" in result
        assert "calculate" in result
        assert "<tool_call>" in result
        assert "</tool_call>" in result

    def test_includes_tool_schemas(self, sample_tools):
        """Test that tool schemas are included in output."""
        parser = HermesToolCallParser()
        result = parser.format_tools(sample_tools)

        assert '"type": "function"' in result
        assert '"city"' in result
        assert '"expression"' in result

    def test_empty_tools(self):
        """Test formatting with no tools."""
        parser = HermesToolCallParser()
        result = parser.format_tools(())

        assert "<tools>" in result
        assert "[]" in result


class TestHermesToolCallParserParse:
    """Tests for HermesToolCallParser.parse()."""

    def test_single_tool_call(self, sample_tools):
        """Test parsing a single tool call."""
        parser = HermesToolCallParser()
        text = (
            'I will check the weather.\n'
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            '</tool_call>'
        )

        result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.text == "I will check the weather."

    def test_multiple_tool_calls(self, sample_tools):
        """Test parsing multiple tool calls."""
        parser = HermesToolCallParser()
        text = (
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            '</tool_call>\n'
            '<tool_call>\n'
            '{"name": "calculate", "arguments": {"expression": "2+2"}}\n'
            '</tool_call>'
        )

        result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "calculate"

    def test_remaining_text(self, sample_tools):
        """Test that remaining text is preserved when mixed with tool calls."""
        parser = HermesToolCallParser()
        text = (
            'Let me help you.\n'
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "London"}}\n'
            '</tool_call>\n'
            'I called the weather tool.'
        )

        result = parser.parse(text, sample_tools)

        assert result.text is not None
        assert "Let me help you." in result.text
        assert "I called the weather tool." in result.text
        assert "<tool_call>" not in result.text

    def test_no_tool_calls(self, sample_tools):
        """Test parsing text with no tool calls."""
        parser = HermesToolCallParser()
        text = "Just a regular response with no tools."

        result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 0
        assert result.text == "Just a regular response with no tools."

    def test_only_tool_calls_no_text(self, sample_tools):
        """Test that text is None when output is only tool calls."""
        parser = HermesToolCallParser()
        text = (
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "NYC"}}\n'
            '</tool_call>'
        )

        result = parser.parse(text, sample_tools)

        assert result.text is None
        assert len(result.tool_calls) == 1

    def test_invalid_json_skipped(self, sample_tools, caplog):
        """Test that invalid JSON in tool_call blocks is skipped."""
        parser = HermesToolCallParser()
        text = (
            '<tool_call>\n'
            'not valid json\n'
            '</tool_call>\n'
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
            '</tool_call>'
        )

        with caplog.at_level(logging.WARNING):
            result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

    def test_unknown_tool_name_still_parsed(self, sample_tools):
        """Test that unknown tool names are still parsed (validation is env's job)."""
        parser = HermesToolCallParser()
        text = (
            '<tool_call>\n'
            '{"name": "unknown_tool", "arguments": {"x": 1}}\n'
            '</tool_call>'
        )

        result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "unknown_tool"

    def test_tool_call_ids_are_unique(self, sample_tools):
        """Test that generated tool call IDs are unique."""
        parser = HermesToolCallParser()
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}}</tool_call>\n'
            '<tool_call>{"name": "get_weather", "arguments": {"city": "B"}}</tool_call>'
        )

        result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id != result.tool_calls[1].id

    def test_empty_arguments(self, sample_tools):
        """Test tool call with empty arguments."""
        parser = HermesToolCallParser()
        text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

        result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    def test_non_dict_arguments_handled(self, sample_tools, caplog):
        """Test that non-dict arguments are converted to empty dict."""
        parser = HermesToolCallParser()
        text = '<tool_call>{"name": "get_weather", "arguments": "not a dict"}</tool_call>'

        with caplog.at_level(logging.WARNING):
            result = parser.parse(text, sample_tools)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}


class TestRunnerToolParsing:
    """Tests for runner integration with tool_call_parser."""

    def _make_backend(self, supports_fc: bool = False):
        """Create a mock backend with configurable capabilities."""
        backend = MagicMock()
        backend.capabilities = BackendCapabilities(
            supports_function_calling=supports_fc,
        )
        return backend

    def _make_tool_env(self, tools):
        """Create a mock environment with tools."""
        env = MagicMock()
        env.spec = EnvironmentSpec(name="test", max_steps=5)
        obs = Observation(
            prompt="Test prompt",
            messages=(),
            available_tools=tuple(tools),
        )
        metadata = StateMetadata(step=0, episode_id="test", is_terminal=False)
        state = State(observation=obs, hidden=None, metadata=metadata)
        env.reset.return_value = (state, {})

        # Make step return a terminal state
        terminal_metadata = StateMetadata(step=1, episode_id="test", is_terminal=True)
        terminal_obs = Observation(prompt="Test prompt", messages=())
        terminal_state = State(
            observation=terminal_obs, hidden=None, metadata=terminal_metadata
        )
        from llenvs.core.environment import StepResult

        env.step.return_value = StepResult(
            next_state=terminal_state,
            rewards=SignalBundle(signals=()),
            terminated=True,
            info={},
        )
        return env

    def test_parser_used_when_no_native_fc(self, sample_tools):
        """Test that parser is used when backend lacks function calling."""
        from llenvs.evaluation.runner import TrajectoryRunner

        backend = self._make_backend(supports_fc=False)
        backend.generate_chat.return_value = GenerationResult(
            text='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
            finish_reason=StopReason.END_OF_TEXT,
        )

        parser = HermesToolCallParser()
        env = self._make_tool_env(sample_tools)

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            tool_call_parser=parser,
        )

        result = runner.run_trajectory(task_index=0)

        # Backend should have been called with generate_chat (not generate_with_tools)
        backend.generate_chat.assert_called()
        backend.generate_with_tools.assert_not_called()

        # The step should have received an action with tool_calls
        step_call = env.step.call_args
        action = step_call[0][1]
        assert action.has_tool_calls
        assert action.tool_calls[0].name == "get_weather"

    def test_native_fc_preferred_over_parser(self, sample_tools):
        """Test that native function calling is used even when parser is set."""
        from llenvs.evaluation.runner import TrajectoryRunner

        backend = self._make_backend(supports_fc=True)
        from llenvs.core.tools import ToolCall

        backend.generate_with_tools.return_value = GenerationResult(
            text=None,
            finish_reason=StopReason.TOOL_USE,
            tool_calls=(ToolCall(id="1", name="get_weather", arguments={"city": "Paris"}),),
        )

        parser = HermesToolCallParser()
        env = self._make_tool_env(sample_tools)

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            tool_call_parser=parser,
        )

        result = runner.run_trajectory(task_index=0)

        # Native path should be used
        backend.generate_with_tools.assert_called()
        backend.generate_chat.assert_not_called()

    def test_warning_when_no_parser_and_no_fc(self, sample_tools, caplog):
        """Test warning logged when tools available but no way to use them."""
        from llenvs.evaluation.runner import TrajectoryRunner

        backend = self._make_backend(supports_fc=False)
        backend.generate_chat.return_value = GenerationResult(
            text="I cannot use tools.",
            finish_reason=StopReason.END_OF_TEXT,
        )

        env = self._make_tool_env(sample_tools)

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            # No tool_call_parser
        )

        with caplog.at_level(logging.WARNING):
            result = runner.run_trajectory(task_index=0)

        assert any("tool_call_parser" in r.message for r in caplog.records)

    def test_tools_text_injected_in_system_message(self, sample_tools):
        """Test that tool definitions are injected into system message."""
        from llenvs.evaluation.runner import TrajectoryRunner

        backend = self._make_backend(supports_fc=False)
        backend.generate_chat.return_value = GenerationResult(
            text="No tools used.",
            finish_reason=StopReason.END_OF_TEXT,
        )

        parser = HermesToolCallParser()
        env = self._make_tool_env(sample_tools)

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            system_prompt="You are helpful.",
            tool_call_parser=parser,
        )

        runner.run_trajectory(task_index=0)

        # Check that the system message was augmented with tools
        call_args = backend.generate_chat.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert system_msg.role == "system"
        assert "<tools>" in system_msg.content
        assert "You are helpful." in system_msg.content

    def test_inject_tools_creates_system_message_if_none(self):
        """Test that _inject_tools_in_messages creates system msg if none exists."""
        from llenvs.evaluation.runner import TrajectoryRunner

        messages = [
            ChatMessage(role="user", content="Hello"),
        ]

        result = TrajectoryRunner._inject_tools_in_messages(messages, "TOOLS_TEXT")

        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == "TOOLS_TEXT"
        assert result[1].role == "user"

    def test_inject_tools_appends_to_existing_system(self):
        """Test that _inject_tools_in_messages appends to existing system msg."""
        from llenvs.evaluation.runner import TrajectoryRunner

        messages = [
            ChatMessage(role="system", content="Be helpful."),
            ChatMessage(role="user", content="Hello"),
        ]

        result = TrajectoryRunner._inject_tools_in_messages(messages, "TOOLS_TEXT")

        assert len(result) == 2
        assert result[0].role == "system"
        assert "Be helpful." in result[0].content
        assert "TOOLS_TEXT" in result[0].content

    def test_batch_path_with_parser(self, sample_tools):
        """Test batch path uses parser for text-based tool calling."""
        from llenvs.evaluation.runner import TrajectoryRunner

        backend = self._make_backend(supports_fc=False)
        backend.generate_chat_batch.return_value = [
            GenerationResult(
                text='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
                finish_reason=StopReason.END_OF_TEXT,
            ),
        ]

        parser = HermesToolCallParser()
        env = self._make_tool_env(sample_tools)

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            tool_call_parser=parser,
        )

        result = runner.run_batch(task_indices=[0])

        # Should use generate_chat_batch, not generate_with_tools_batch
        backend.generate_chat_batch.assert_called()

        # The step should have tool calls
        step_call = env.step.call_args
        action = step_call[0][1]
        assert action.has_tool_calls
