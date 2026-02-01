"""Core tool types for function/tool calling support.

Provides the data structures for defining tools, making tool calls,
and handling tool results.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol


class ToolParameterType(Enum):
    """JSON Schema types for tool parameters."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass(frozen=True)
class ToolParameter:
    """A parameter for a tool.

    Attributes:
        name: Parameter name.
        type: JSON Schema type.
        description: Human-readable description.
        required: Whether the parameter is required.
        enum: Optional list of allowed values for string parameters.
    """

    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        return schema


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a tool that can be called by a model.

    Attributes:
        name: Unique tool name.
        description: Human-readable description of what the tool does.
        parameters: Tuple of parameter definitions.
        is_terminal: If True, calling this tool ends the episode.
    """

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    is_terminal: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function/tool schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A request to call a tool.

    Attributes:
        id: Unique identifier for matching results to calls.
        name: Name of the tool to call.
        arguments: Dictionary of argument name to value.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolResultStatus(Enum):
    """Status of a tool execution."""

    SUCCESS = auto()
    ERROR = auto()
    INVALID_TOOL = auto()
    INVALID_ARGUMENTS = auto()


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool call.

    Attributes:
        call_id: ID of the ToolCall this is a result for.
        tool_name: Name of the tool that was called.
        status: Whether the execution succeeded or failed.
        output: The tool output (string or structured data).
        error: Error message if status is not SUCCESS.
    """

    call_id: str
    tool_name: str
    status: ToolResultStatus
    output: str | dict[str, Any] = ""
    error: str | None = None

    @classmethod
    def success(
        cls,
        call_id: str,
        tool_name: str,
        output: str | dict[str, Any],
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.SUCCESS,
            output=output,
            error=None,
        )

    @classmethod
    def from_error(
        cls,
        call_id: str,
        tool_name: str,
        error_message: str,
        status: ToolResultStatus = ToolResultStatus.ERROR,
    ) -> "ToolResult":
        """Create a failed tool result."""
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=status,
            error=error_message,
        )

    @property
    def is_success(self) -> bool:
        """Check if the tool execution was successful."""
        return self.status == ToolResultStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """Check if the tool execution failed."""
        return self.status != ToolResultStatus.SUCCESS


class ToolExecutor(Protocol):
    """Protocol for executing tool calls.

    Implementations handle the actual execution of tools,
    whether backed by Python functions, external services, or MCP.
    """

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            call: The tool call to execute.

        Returns:
            ToolResult with the execution outcome.
        """
        ...

    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls.

        Default implementation executes sequentially, but implementations
        may parallelize where appropriate.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        ...


class SimpleToolExecutor:
    """Tool executor backed by a dictionary of Python callables.

    Each callable should accept keyword arguments matching the tool's
    parameters and return a string or dict result.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str | dict[str, Any]]],
    ) -> None:
        """Initialize with a mapping of tool names to callables.

        Args:
            tools: Dict mapping tool name to callable.
        """
        self._tools = tools

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        if call.name not in self._tools:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Unknown tool: {call.name}",
                status=ToolResultStatus.INVALID_TOOL,
            )

        try:
            result = self._tools[call.name](**call.arguments)
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=result,
            )
        except TypeError as e:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Invalid arguments: {e}",
                status=ToolResultStatus.INVALID_ARGUMENTS,
            )
        except Exception as e:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=str(e),
            )

    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls sequentially."""
        return tuple(self.execute(call) for call in calls)
