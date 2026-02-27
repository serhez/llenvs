"""Core tool types for function/tool calling support.

Provides the data structures for defining tools, making tool calls,
and handling tool results.
"""

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


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
        raw_schema: Optional original JSON Schema dict for full-fidelity
            passthrough. When set, ``to_openai_schema()`` and
            ``to_anthropic_schema()`` use it directly instead of
            rebuilding from flat ``parameters``. This preserves nested
            object schemas, arrays of objects, discriminated unions, etc.
    """

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    is_terminal: bool = False
    raw_schema: dict[str, Any] | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function/tool schema format."""
        if self.raw_schema is not None:
            # Use raw_schema for full fidelity
            if "type" in self.raw_schema and "function" in self.raw_schema:
                # Already wrapped: {"type": "function", "function": {...}}
                return self.raw_schema
            if "name" in self.raw_schema:
                # Bare function dict: {"name": ..., "parameters": ...}
                return {"type": "function", "function": self.raw_schema}
            # Fallback: wrap with name/description
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.raw_schema,
                },
            }

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

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        *,
        is_terminal: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> "ToolDefinition":
        """Create a ToolDefinition from a Python callable.

        Inspects the function's signature for parameter types and defaults,
        and parses Google-style docstring Args for descriptions.

        Args:
            func: The callable to create a definition from.
            is_terminal: Whether calling this tool ends the episode.
            name: Override the tool name (defaults to func.__name__).
            description: Override the description (defaults to first line of docstring).
        """
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func)
        param_docs = _parse_param_docs(docstring)

        tool_name = name or func.__name__

        if description is not None:
            tool_desc = description
        elif docstring:
            tool_desc = docstring.split("\n")[0].strip()
        else:
            tool_desc = tool_name

        parameters: list[ToolParameter] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = _python_type_to_tool_type(param.annotation)
            param_desc = param_docs.get(param_name, param_name)
            required = param.default is inspect.Parameter.empty

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_desc,
                    required=required,
                )
            )

        return cls(
            name=tool_name,
            description=tool_desc,
            parameters=tuple(parameters),
            is_terminal=is_terminal,
        )

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema format."""
        if self.raw_schema is not None:
            # Extract parameters from raw_schema
            if "type" in self.raw_schema and "function" in self.raw_schema:
                func = self.raw_schema["function"]
            elif "name" in self.raw_schema and "parameters" in self.raw_schema:
                func = self.raw_schema
            else:
                # raw_schema is the parameters object itself
                return {
                    "name": self.name,
                    "description": self.description,
                    "input_schema": self.raw_schema,
                }
            return {
                "name": func.get("name", self.name),
                "description": func.get("description", self.description),
                "input_schema": func.get("parameters", {}),
            }

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


def _python_type_to_tool_type(annotation: Any) -> ToolParameterType:
    """Map a Python type annotation to a ToolParameterType.

    Handles basic types, typing generics (list[...], dict[...]), and
    falls back to STRING for unknown types.
    """
    if annotation is inspect.Parameter.empty:
        return ToolParameterType.STRING

    # Handle typing generics (list[int], dict[str, Any], etc.)
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        if origin is list:
            return ToolParameterType.ARRAY
        if origin is dict:
            return ToolParameterType.OBJECT
        return ToolParameterType.STRING

    type_map = {
        str: ToolParameterType.STRING,
        int: ToolParameterType.INTEGER,
        float: ToolParameterType.NUMBER,
        bool: ToolParameterType.BOOLEAN,
        list: ToolParameterType.ARRAY,
        dict: ToolParameterType.OBJECT,
    }
    return type_map.get(annotation, ToolParameterType.STRING)


def _parse_param_docs(docstring: str | None) -> dict[str, str]:
    """Parse Google-style Args section from a docstring.

    Returns a mapping of parameter name to description.
    """
    if not docstring:
        return {}

    result: dict[str, str] = {}
    in_args = False
    current_name: str | None = None
    current_desc_lines: list[str] = []

    for line in docstring.splitlines():
        stripped = line.strip()

        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue

        # Detect end of Args section (next section header)
        if in_args and stripped and not stripped.startswith(" ") and stripped.endswith(":"):
            # Could be a new section like "Returns:", "Raises:", etc.
            if re.match(r"^[A-Z][a-z]+:$", stripped):
                break

        if not in_args:
            continue

        if not stripped:
            continue

        # Check for new parameter line: "param_name: description" or "param_name (type): description"
        param_match = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
        if param_match:
            # Save previous param
            if current_name is not None:
                result[current_name] = " ".join(current_desc_lines).strip()

            current_name = param_match.group(1)
            desc = param_match.group(2).strip()
            current_desc_lines = [desc] if desc else []
        elif current_name is not None:
            # Continuation line for current parameter
            current_desc_lines.append(stripped)

    # Save last param
    if current_name is not None:
        result[current_name] = " ".join(current_desc_lines).strip()

    return result


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


def format_tool_call(tc: ToolCall) -> str:
    """Format a tool call for display.

    Args:
        tc: The tool call to format.

    Returns:
        Canonical display string, e.g. ``search(query="red shoes", limit=10)``.
    """
    args = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
    return f"{tc.name}({args})"


def format_tool_result(entry: dict[str, Any]) -> str:
    """Format a single serialized tool result dict for display.

    Takes one element from the list in ``state.data["tool_results"]``
    (as produced by ``_tool_results_to_data()``).

    Args:
        entry: Serialized tool result dict with keys
            ``tool_name``, ``status``, ``output``, ``error``.

    Returns:
        Formatted string, e.g. ``search: Found 3 results``
        or ``calculate: ERROR: division by zero``.
    """
    name = entry["tool_name"]
    status = entry["status"]
    if status == "SUCCESS":
        output = entry.get("output", "")
        if isinstance(output, dict):
            return f"{name}: {json.dumps(output)}"
        return f"{name}: {output}"
    error = entry.get("error", "")
    return f"{name}: {status}: {error}"


def format_tool_result_data(result_entries: list[dict[str, Any]]) -> str:
    """Format serialized tool result dicts for display.

    Takes the list from ``state.data["tool_results"]``
    (as produced by ``_tool_results_to_data()``).

    Args:
        result_entries: List of serialized tool result dicts.

    Returns:
        Newline-joined formatted results, each prefixed with ``- ``.
    """
    return "\n".join(f"- {format_tool_result(e)}" for e in result_entries)


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


# ── OpenAI schema conversion ──────────────────────────────────────

_OAI_TYPE_MAP: dict[str, ToolParameterType] = {
    "string": ToolParameterType.STRING,
    "integer": ToolParameterType.INTEGER,
    "number": ToolParameterType.NUMBER,
    "boolean": ToolParameterType.BOOLEAN,
    "array": ToolParameterType.ARRAY,
    "object": ToolParameterType.OBJECT,
}


def oai_tools_to_definitions(
    oai_tools: list[dict[str, Any]],
) -> tuple[ToolDefinition, ...]:
    """Convert OpenAI-format tool schemas to ToolDefinitions.

    Always preserves the original schema dict as ``raw_schema`` on the
    resulting ``ToolDefinition``.  The flat ``parameters`` tuple remains
    as a best-effort parse for inspection/display, but
    ``to_openai_schema()`` / ``to_anthropic_schema()`` will use
    ``raw_schema`` when present for full-fidelity roundtrip.

    Args:
        oai_tools: List of OpenAI tool schema dicts.

    Returns:
        Tuple of ToolDefinition objects.
    """
    definitions: list[ToolDefinition] = []

    for tool in oai_tools:
        func = tool.get("function", tool)
        name = func["name"]
        description = func.get("description", "")
        params_schema = func.get("parameters", {})
        properties = params_schema.get("properties", {})
        required_names = set(params_schema.get("required", []))

        parameters: list[ToolParameter] = []
        for param_name, param_schema in properties.items():
            param_type_str = param_schema.get("type", "string")
            param_type = _OAI_TYPE_MAP.get(param_type_str, ToolParameterType.STRING)
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_schema.get("description", ""),
                    required=param_name in required_names,
                )
            )

        definitions.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=tuple(parameters),
                raw_schema=tool,
            )
        )

    return tuple(definitions)
