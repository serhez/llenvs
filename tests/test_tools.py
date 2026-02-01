"""Tests for core tool types."""

import pytest
from typing import Any

from llenvs.core.tools import (
    ToolParameter,
    ToolParameterType,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    SimpleToolExecutor,
)


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_basic_parameter(self):
        """Test creating a basic parameter."""
        param = ToolParameter(
            name="city",
            type=ToolParameterType.STRING,
            description="The city name",
        )
        assert param.name == "city"
        assert param.type == ToolParameterType.STRING
        assert param.description == "The city name"
        assert param.required is True
        assert param.enum is None

    def test_optional_parameter(self):
        """Test creating an optional parameter."""
        param = ToolParameter(
            name="units",
            type=ToolParameterType.STRING,
            description="Temperature units",
            required=False,
            enum=("celsius", "fahrenheit"),
        )
        assert param.required is False
        assert param.enum == ("celsius", "fahrenheit")

    def test_to_json_schema(self):
        """Test JSON schema conversion."""
        param = ToolParameter(
            name="count",
            type=ToolParameterType.INTEGER,
            description="Number of items",
        )
        schema = param.to_json_schema()
        assert schema["type"] == "integer"
        assert schema["description"] == "Number of items"
        assert "enum" not in schema

    def test_to_json_schema_with_enum(self):
        """Test JSON schema conversion with enum."""
        param = ToolParameter(
            name="color",
            type=ToolParameterType.STRING,
            description="The color",
            enum=("red", "green", "blue"),
        )
        schema = param.to_json_schema()
        assert schema["enum"] == ["red", "green", "blue"]


class TestToolDefinition:
    """Tests for ToolDefinition."""

    @pytest.fixture
    def weather_tool(self) -> ToolDefinition:
        """Create a sample weather tool."""
        return ToolDefinition(
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
        )

    def test_basic_tool(self):
        """Test creating a basic tool."""
        tool = ToolDefinition(
            name="search",
            description="Search the web",
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.parameters == ()
        assert tool.is_terminal is False

    def test_terminal_tool(self):
        """Test creating a terminal tool."""
        tool = ToolDefinition(
            name="submit_answer",
            description="Submit the final answer",
            is_terminal=True,
        )
        assert tool.is_terminal is True

    def test_to_openai_schema(self, weather_tool: ToolDefinition):
        """Test OpenAI schema conversion."""
        schema = weather_tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"] == "Get the current weather for a city"

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "city" in params["properties"]
        assert "units" in params["properties"]
        assert params["required"] == ["city"]

    def test_to_anthropic_schema(self, weather_tool: ToolDefinition):
        """Test Anthropic schema conversion."""
        schema = weather_tool.to_anthropic_schema()

        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get the current weather for a city"

        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert "city" in input_schema["properties"]
        assert "units" in input_schema["properties"]
        assert input_schema["required"] == ["city"]

    def test_schema_with_no_parameters(self):
        """Test schema conversion with no parameters."""
        tool = ToolDefinition(
            name="get_time",
            description="Get the current time",
        )

        openai = tool.to_openai_schema()
        assert openai["function"]["parameters"]["properties"] == {}
        assert openai["function"]["parameters"]["required"] == []

        anthropic = tool.to_anthropic_schema()
        assert anthropic["input_schema"]["properties"] == {}
        assert anthropic["input_schema"]["required"] == []


class TestToolCall:
    """Tests for ToolCall."""

    def test_basic_call(self):
        """Test creating a basic tool call."""
        call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Paris"},
        )
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Paris"}

    def test_empty_arguments(self):
        """Test tool call with no arguments."""
        call = ToolCall(id="call_456", name="get_time")
        assert call.arguments == {}


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolResult.success(
            call_id="call_123",
            tool_name="get_weather",
            output="Sunny, 25°C",
        )
        assert result.call_id == "call_123"
        assert result.tool_name == "get_weather"
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output == "Sunny, 25°C"
        assert result.error is None
        assert result.is_success is True
        assert result.is_error is False

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.from_error(
            call_id="call_456",
            tool_name="get_weather",
            error_message="City not found",
        )
        assert result.status == ToolResultStatus.ERROR
        assert result.error == "City not found"
        assert result.is_success is False
        assert result.is_error is True

    def test_invalid_tool_result(self):
        """Test creating invalid tool result."""
        result = ToolResult.from_error(
            call_id="call_789",
            tool_name="unknown_tool",
            error_message="Unknown tool",
            status=ToolResultStatus.INVALID_TOOL,
        )
        assert result.status == ToolResultStatus.INVALID_TOOL

    def test_dict_output(self):
        """Test result with dict output."""
        result = ToolResult.success(
            call_id="call_123",
            tool_name="search",
            output={"results": ["a", "b", "c"]},
        )
        assert isinstance(result.output, dict)


class TestSimpleToolExecutor:
    """Tests for SimpleToolExecutor."""

    @pytest.fixture
    def executor(self) -> SimpleToolExecutor:
        """Create executor with sample tools."""

        def add(a: int, b: int) -> str:
            return str(a + b)

        def multiply(a: int, b: int) -> str:
            return str(a * b)

        def get_time() -> str:
            return "12:00:00"

        def error_tool() -> str:
            raise ValueError("Intentional error")

        return SimpleToolExecutor(
            tools={
                "add": add,
                "multiply": multiply,
                "get_time": get_time,
                "error_tool": error_tool,
            }
        )

    def test_execute_success(self, executor: SimpleToolExecutor):
        """Test successful tool execution."""
        call = ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})
        result = executor.execute(call)

        assert result.is_success
        assert result.output == "5"

    def test_execute_no_args(self, executor: SimpleToolExecutor):
        """Test execution with no arguments."""
        call = ToolCall(id="2", name="get_time", arguments={})
        result = executor.execute(call)

        assert result.is_success
        assert result.output == "12:00:00"

    def test_execute_unknown_tool(self, executor: SimpleToolExecutor):
        """Test execution of unknown tool."""
        call = ToolCall(id="3", name="unknown", arguments={})
        result = executor.execute(call)

        assert result.status == ToolResultStatus.INVALID_TOOL
        assert "Unknown tool" in result.error

    def test_execute_invalid_arguments(self, executor: SimpleToolExecutor):
        """Test execution with invalid arguments."""
        call = ToolCall(id="4", name="add", arguments={"x": 1, "y": 2})
        result = executor.execute(call)

        assert result.status == ToolResultStatus.INVALID_ARGUMENTS

    def test_execute_tool_error(self, executor: SimpleToolExecutor):
        """Test execution when tool raises error."""
        call = ToolCall(id="5", name="error_tool", arguments={})
        result = executor.execute(call)

        assert result.status == ToolResultStatus.ERROR
        assert "Intentional error" in result.error

    def test_execute_batch(self, executor: SimpleToolExecutor):
        """Test batch execution."""
        calls = (
            ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="2", name="multiply", arguments={"a": 3, "b": 4}),
            ToolCall(id="3", name="unknown", arguments={}),
        )
        results = executor.execute_batch(calls)

        assert len(results) == 3
        assert results[0].is_success
        assert results[0].output == "3"
        assert results[1].is_success
        assert results[1].output == "12"
        assert results[2].status == ToolResultStatus.INVALID_TOOL
