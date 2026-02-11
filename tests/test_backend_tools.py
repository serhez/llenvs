"""Tests for backend tool support.

Tests mock OpenAI/Anthropic tool response parsing and generate_with_tools integration.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from llenvs.core.tools import ToolCall, ToolDefinition, ToolParameter, ToolParameterType
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
)

# Check if openai/anthropic are available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    """Create sample tools for testing."""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city",
            parameters=(
                ToolParameter(
                    name="city",
                    type=ToolParameterType.STRING,
                    description="The city name",
                ),
                ToolParameter(
                    name="units",
                    type=ToolParameterType.STRING,
                    description="Temperature units",
                    required=False,
                    enum=("celsius", "fahrenheit"),
                ),
            ),
        ),
        ToolDefinition(
            name="search",
            description="Search the web",
            parameters=(
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Search query",
                ),
            ),
        ),
    ]


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """Create sample chat messages."""
    return [
        ChatMessage(role="user", content="What's the weather in Paris?"),
    ]


class TestGenerationResultToolSupport:
    """Tests for GenerationResult with tool calls."""

    def test_has_tool_calls_false(self):
        """Test has_tool_calls returns False when no calls."""
        result = GenerationResult(text="Hello")
        assert result.has_tool_calls is False

    def test_has_tool_calls_true(self):
        """Test has_tool_calls returns True when calls present."""
        result = GenerationResult(
            text="Let me check",
            tool_calls=(ToolCall(id="1", name="search", arguments={"query": "test"}),),
        )
        assert result.has_tool_calls is True

    def test_to_agent_action_text_only(self):
        """Test conversion to Action with text only."""
        result = GenerationResult(text="Hello world")
        action = result.to_agent_action()

        assert action.text == "Hello world"
        assert action.tool_calls == ()
        assert action.is_text_only

    def test_to_agent_action_with_tools(self):
        """Test conversion to Action with tools."""
        tool_calls = (
            ToolCall(id="1", name="search", arguments={"query": "test"}),
            ToolCall(id="2", name="get_weather", arguments={"city": "Paris"}),
        )
        result = GenerationResult(text="Let me help", tool_calls=tool_calls)
        action = result.to_agent_action()

        assert action.text == "Let me help"
        assert action.tool_calls == tool_calls
        assert action.has_tool_calls

    def test_to_agent_action_tools_only(self):
        """Test conversion when only tool calls (no text)."""
        tool_calls = (ToolCall(id="1", name="search", arguments={}),)
        result = GenerationResult(text=None, tool_calls=tool_calls)
        action = result.to_agent_action()

        assert action.text is None
        assert action.has_tool_calls
        assert not action.is_text_only


class TestChatMessageToolSupport:
    """Tests for ChatMessage with tool support."""

    def test_basic_message_to_dict(self):
        """Test basic message conversion."""
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()

        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert "tool_calls" not in d

    def test_assistant_message_with_tool_calls(self):
        """Test assistant message with tool calls."""
        msg = ChatMessage(
            role="assistant",
            content="Let me search",
            tool_calls=(
                ToolCall(id="call_1", name="search", arguments={"query": "python"}),
            ),
        )
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "Let me search"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "call_1"
        assert d["tool_calls"][0]["function"]["name"] == "search"

    def test_tool_result_message(self):
        """Test creating a tool result message."""
        from llenvs.core.tools import ToolResult

        result = ToolResult.success("call_1", "search", "Found 10 results")
        msg = ChatMessage.tool_result(result)

        assert msg.role == "tool"
        assert msg.content == "Found 10 results"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "search"

    def test_tool_result_message_error(self):
        """Test creating a tool result message for errors."""
        from llenvs.core.tools import ToolResult

        result = ToolResult.from_error("call_1", "search", "Connection timeout")
        msg = ChatMessage.tool_result(result)

        assert msg.role == "tool"
        assert msg.content == "Connection timeout"

    def test_tool_message_to_dict(self):
        """Test tool message dictionary conversion."""
        msg = ChatMessage(
            role="tool",
            content="Search results here",
            tool_call_id="call_123",
            name="search",
        )
        d = msg.to_dict()

        assert d["role"] == "tool"
        assert d["content"] == "Search results here"
        assert d["tool_call_id"] == "call_123"
        assert d["name"] == "search"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
class TestOpenAIBackendTools:
    """Tests for OpenAI backend tool support (mocked)."""

    def test_generate_with_tools_parses_response(self, sample_tools, sample_messages):
        """Test that tool calls are correctly parsed from OpenAI response."""
        # Create mock response structure
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Paris", "units": "celsius"}'

        mock_message = MagicMock()
        mock_message.content = "Let me check the weather"
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        mock_choice.logprobs = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.model = "gpt-4o"
        mock_response.id = "chatcmpl-123"

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            from llenvs.inference.backends.api import OpenAIBackend

            backend = OpenAIBackend(model="gpt-4o")
            result = backend.generate_with_tools(
                sample_messages, sample_tools, SamplingParams()
            )

            assert result.text == "Let me check the weather"
            assert result.finish_reason == StopReason.TOOL_USE
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_abc123"
            assert result.tool_calls[0].name == "get_weather"
            assert result.tool_calls[0].arguments == {
                "city": "Paris",
                "units": "celsius",
            }

    def test_generate_with_tools_handles_no_tools(self, sample_tools, sample_messages):
        """Test handling when model doesn't use tools."""
        mock_message = MagicMock()
        mock_message.content = "I don't need to use tools for this"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_choice.logprobs = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 10
        mock_response.model = "gpt-4o"
        mock_response.id = "chatcmpl-456"

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            from llenvs.inference.backends.api import OpenAIBackend

            backend = OpenAIBackend(model="gpt-4o")
            result = backend.generate_with_tools(
                sample_messages, sample_tools, SamplingParams()
            )

            assert result.text == "I don't need to use tools for this"
            assert result.tool_calls == ()
            assert result.finish_reason == StopReason.STOP_SEQUENCE


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic package not installed")
class TestAnthropicBackendTools:
    """Tests for Anthropic backend tool support (mocked)."""

    def test_generate_with_tools_parses_response(self, sample_tools, sample_messages):
        """Test that tool calls are correctly parsed from Anthropic response."""
        # Create mock content blocks
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "I'll check the weather"

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_abc123"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"city": "Paris", "units": "celsius"}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.id = "msg_123"

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            from llenvs.inference.backends.api import AnthropicBackend

            backend = AnthropicBackend(model="claude-sonnet-4-20250514")
            result = backend.generate_with_tools(
                sample_messages, sample_tools, SamplingParams()
            )

            assert result.text == "I'll check the weather"
            assert result.finish_reason == StopReason.TOOL_USE
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "tool_abc123"
            assert result.tool_calls[0].name == "get_weather"
            assert result.tool_calls[0].arguments == {
                "city": "Paris",
                "units": "celsius",
            }

    def test_generate_with_tools_no_text(self, sample_tools, sample_messages):
        """Test handling when only tool use blocks are returned."""
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_456"
        mock_tool_block.name = "search"
        mock_tool_block.input = {"query": "python"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 25
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.id = "msg_456"

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            from llenvs.inference.backends.api import AnthropicBackend

            backend = AnthropicBackend(model="claude-sonnet-4-20250514")
            result = backend.generate_with_tools(
                sample_messages, sample_tools, SamplingParams()
            )

            assert result.text is None
            assert len(result.tool_calls) == 1


class TestToolDefinitionSchemas:
    """Tests for tool definition schema conversions."""

    def test_openai_schema_format(self, sample_tools):
        """Test OpenAI schema format is correct."""
        tool = sample_tools[0]  # get_weather
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "get_weather"
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"

    def test_anthropic_schema_format(self, sample_tools):
        """Test Anthropic schema format is correct."""
        tool = sample_tools[0]  # get_weather
        schema = tool.to_anthropic_schema()

        assert schema["name"] == "get_weather"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"


class TestStopReasonToolUse:
    """Tests for TOOL_USE stop reason."""

    def test_stop_reason_enum(self):
        """Test TOOL_USE is a valid stop reason."""
        assert StopReason.TOOL_USE is not None
        assert StopReason.TOOL_USE.name == "TOOL_USE"

    def test_generation_result_with_tool_use(self):
        """Test GenerationResult with TOOL_USE finish reason."""
        result = GenerationResult(
            text="Using a tool",
            finish_reason=StopReason.TOOL_USE,
            tool_calls=(ToolCall(id="1", name="search", arguments={}),),
        )
        assert result.finish_reason == StopReason.TOOL_USE
