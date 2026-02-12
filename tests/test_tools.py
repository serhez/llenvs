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


class TestFromCallable:
    """Tests for ToolDefinition.from_callable()."""

    def test_basic_typed_params(self):
        """Test from_callable with typed parameters."""

        def greet(name: str, age: int) -> str:
            """Greet a person."""
            return f"Hello {name}, you are {age}"

        tool = ToolDefinition.from_callable(greet)

        assert tool.name == "greet"
        assert tool.description == "Greet a person."
        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "name"
        assert tool.parameters[0].type == ToolParameterType.STRING
        assert tool.parameters[0].required is True
        assert tool.parameters[1].name == "age"
        assert tool.parameters[1].type == ToolParameterType.INTEGER
        assert tool.parameters[1].required is True

    def test_defaults_make_optional(self):
        """Test that params with defaults are not required."""

        def search(query: str, limit: int = 10) -> str:
            """Search for items."""
            return ""

        tool = ToolDefinition.from_callable(search)

        assert tool.parameters[0].required is True  # query
        assert tool.parameters[1].required is False  # limit

    def test_no_annotations_fallback_to_string(self):
        """Test that unannotated params default to STRING."""

        def mystery(x, y):
            return ""

        tool = ToolDefinition.from_callable(mystery)

        assert len(tool.parameters) == 2
        assert tool.parameters[0].type == ToolParameterType.STRING
        assert tool.parameters[1].type == ToolParameterType.STRING

    def test_name_and_description_overrides(self):
        """Test overriding name and description."""

        def internal_fn(x: str) -> str:
            """Original description."""
            return x

        tool = ToolDefinition.from_callable(
            internal_fn, name="my_tool", description="Custom description"
        )

        assert tool.name == "my_tool"
        assert tool.description == "Custom description"

    def test_is_terminal(self):
        """Test is_terminal flag."""

        def submit(answer: str) -> str:
            """Submit answer."""
            return answer

        tool = ToolDefinition.from_callable(submit, is_terminal=True)
        assert tool.is_terminal is True

    def test_self_cls_skipping(self):
        """Test that self and cls are skipped."""

        class MyClass:
            def method(self, x: str) -> str:
                """A method."""
                return x

            @classmethod
            def class_method(cls, x: str) -> str:
                """A classmethod."""
                return x

        tool = ToolDefinition.from_callable(MyClass.method)
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "x"

        tool2 = ToolDefinition.from_callable(MyClass.class_method)
        assert len(tool2.parameters) == 1
        assert tool2.parameters[0].name == "x"

    def test_generic_types(self):
        """Test typing generics like list[int] and dict[str, Any]."""

        def process(items: list[int], config: dict[str, Any]) -> str:
            """Process items."""
            return ""

        tool = ToolDefinition.from_callable(process)

        assert tool.parameters[0].type == ToolParameterType.ARRAY
        assert tool.parameters[1].type == ToolParameterType.OBJECT

    def test_all_basic_types(self):
        """Test all basic type mappings."""

        def typed(
            s: str, i: int, f: float, b: bool, l: list, d: dict
        ) -> str:
            """All types."""
            return ""

        tool = ToolDefinition.from_callable(typed)

        assert tool.parameters[0].type == ToolParameterType.STRING
        assert tool.parameters[1].type == ToolParameterType.INTEGER
        assert tool.parameters[2].type == ToolParameterType.NUMBER
        assert tool.parameters[3].type == ToolParameterType.BOOLEAN
        assert tool.parameters[4].type == ToolParameterType.ARRAY
        assert tool.parameters[5].type == ToolParameterType.OBJECT

    def test_docstring_args_parsing(self):
        """Test Google-style docstring Args parsing for param descriptions."""

        def fetch(url: str, timeout: int = 30) -> str:
            """Fetch a URL.

            Args:
                url: The URL to fetch.
                timeout: Request timeout in seconds.

            Returns:
                The response body.
            """
            return ""

        tool = ToolDefinition.from_callable(fetch)

        assert tool.parameters[0].description == "The URL to fetch."
        assert tool.parameters[1].description == "Request timeout in seconds."

    def test_no_docstring(self):
        """Test function with no docstring."""

        def bare(x: str) -> str:
            return x

        tool = ToolDefinition.from_callable(bare)

        assert tool.name == "bare"
        assert tool.description == "bare"  # falls back to name

    def test_lambda(self):
        """Test from_callable with a lambda (needs name override)."""
        fn = lambda x: x  # noqa: E731
        tool = ToolDefinition.from_callable(fn, name="identity", description="Return input")

        assert tool.name == "identity"
        assert tool.description == "Return input"


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
