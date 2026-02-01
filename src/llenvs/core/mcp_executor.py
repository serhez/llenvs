"""MCP (Model Context Protocol) tool executor.

Provides an MCPToolExecutor that connects to MCP servers and executes
tools via the MCP protocol. Supports both stdio and SSE transports.
"""

import asyncio
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
)

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Error connecting to or communicating with MCP server."""


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes:
        command: Command to start the server (for stdio transport).
        args: Arguments for the command.
        env: Environment variables for the subprocess.
        url: URL for SSE transport (alternative to command).
        timeout: Connection and request timeout in seconds.
    """

    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    timeout: float = 30.0


@dataclass
class MCPToolExecutor:
    """Tool executor that communicates with MCP servers.

    Supports the Model Context Protocol for connecting to external tool
    servers. The executor manages the connection lifecycle and translates
    ToolCall/ToolResult to/from MCP messages.

    Example:
        ```python
        # Connect to an MCP server via stdio
        config = MCPServerConfig(
            command="npx",
            args=("-y", "@modelcontextprotocol/server-filesystem", "/tmp"),
        )
        executor = MCPToolExecutor(config)

        # Initialize connection and discover tools
        await executor.connect()
        tools = await executor.list_tools()

        # Execute a tool call
        call = ToolCall(id="1", name="read_file", arguments={"path": "/tmp/test.txt"})
        result = await executor.execute_async(call)

        # Clean up
        await executor.disconnect()
        ```

    For synchronous usage:
        ```python
        executor = MCPToolExecutor(config)
        executor.connect_sync()
        result = executor.execute(call)
        executor.disconnect_sync()
        ```
    """

    _config: MCPServerConfig
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _tools_cache: dict[str, ToolDefinition] = field(default_factory=dict, init=False)
    _request_id: int = field(default=0, init=False)
    _connected: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __init__(self, config: MCPServerConfig) -> None:
        """Initialize with server configuration.

        Args:
            config: MCP server connection configuration.
        """
        self._config = config
        self._process = None
        self._tools_cache = {}
        self._request_id = 0
        self._connected = False
        self._lock = asyncio.Lock()

    def _next_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_id += 1
        return str(self._request_id)

    async def connect(self) -> None:
        """Connect to the MCP server and initialize the session.

        Raises:
            MCPConnectionError: If connection fails.
        """
        if self._connected:
            return

        if self._config.command:
            await self._connect_stdio()
        elif self._config.url:
            raise NotImplementedError("SSE transport not yet implemented")
        else:
            raise MCPConnectionError("No command or URL specified in config")

        # Initialize the MCP session
        await self._initialize_session()
        self._connected = True

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport (subprocess)."""
        try:
            cmd = [self._config.command] + list(self._config.args)
            env = {**dict(__import__("os").environ), **self._config.env}

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            logger.info(f"Started MCP server process: {cmd}")
        except Exception as e:
            raise MCPConnectionError(f"Failed to start MCP server: {e}") from e

    async def _initialize_session(self) -> None:
        """Send MCP initialize request and wait for response."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "llenvs",
                    "version": "0.1.0",
                },
            },
        }

        response = await self._send_request(init_request)
        if "error" in response:
            raise MCPConnectionError(f"Initialize failed: {response['error']}")

        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        await self._send_notification(initialized_notif)

    async def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise MCPConnectionError("Not connected to MCP server")

        async with self._lock:
            try:
                # Write request
                request_bytes = json.dumps(request).encode("utf-8") + b"\n"
                self._process.stdin.write(request_bytes)
                self._process.stdin.flush()

                # Read response (with timeout)
                loop = asyncio.get_event_loop()
                response_line = await asyncio.wait_for(
                    loop.run_in_executor(None, self._process.stdout.readline),
                    timeout=self._config.timeout,
                )

                if not response_line:
                    raise MCPConnectionError("No response from server")

                return json.loads(response_line.decode("utf-8"))
            except asyncio.TimeoutError as e:
                raise MCPConnectionError(f"Request timed out after {self._config.timeout}s") from e
            except json.JSONDecodeError as e:
                raise MCPConnectionError(f"Invalid JSON response: {e}") from e

    async def _send_notification(self, notification: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Not connected to MCP server")

        notification_bytes = json.dumps(notification).encode("utf-8") + b"\n"
        self._process.stdin.write(notification_bytes)
        self._process.stdin.flush()

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None

        self._connected = False
        self._tools_cache.clear()
        logger.info("Disconnected from MCP server")

    async def list_tools(self) -> tuple[ToolDefinition, ...]:
        """Fetch available tools from the MCP server.

        Returns:
            Tuple of ToolDefinition objects.

        Raises:
            MCPConnectionError: If not connected or request fails.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list",
        }

        response = await self._send_request(request)

        if "error" in response:
            raise MCPConnectionError(f"tools/list failed: {response['error']}")

        tools = []
        for tool_data in response.get("result", {}).get("tools", []):
            tool_def = self._parse_tool_definition(tool_data)
            tools.append(tool_def)
            self._tools_cache[tool_def.name] = tool_def

        return tuple(tools)

    def _parse_tool_definition(self, data: dict[str, Any]) -> ToolDefinition:
        """Parse an MCP tool definition into ToolDefinition."""
        name = data.get("name", "")
        description = data.get("description", "")

        parameters = []
        input_schema = data.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        for param_name, param_data in properties.items():
            param_type = self._map_json_type(param_data.get("type", "string"))
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_data.get("description", ""),
                    required=param_name in required,
                    enum=tuple(param_data["enum"]) if "enum" in param_data else None,
                )
            )

        return ToolDefinition(
            name=name,
            description=description,
            parameters=tuple(parameters),
        )

    def _map_json_type(self, json_type: str) -> ToolParameterType:
        """Map JSON Schema type to ToolParameterType."""
        type_map = {
            "string": ToolParameterType.STRING,
            "number": ToolParameterType.NUMBER,
            "integer": ToolParameterType.INTEGER,
            "boolean": ToolParameterType.BOOLEAN,
            "array": ToolParameterType.ARRAY,
            "object": ToolParameterType.OBJECT,
        }
        return type_map.get(json_type, ToolParameterType.STRING)

    async def execute_async(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call via MCP.

        Args:
            call: The tool call to execute.

        Returns:
            ToolResult with the execution outcome.
        """
        if not self._connected:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message="Not connected to MCP server",
            )

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }

        try:
            response = await self._send_request(request)

            if "error" in response:
                error = response["error"]
                return ToolResult.from_error(
                    call_id=call.id,
                    tool_name=call.name,
                    error_message=error.get("message", str(error)),
                )

            result = response.get("result", {})
            content = result.get("content", [])

            # Extract text content from MCP response
            output_parts = []
            for block in content:
                if block.get("type") == "text":
                    output_parts.append(block.get("text", ""))
                elif block.get("type") == "resource":
                    output_parts.append(f"[Resource: {block.get('resource', {}).get('uri', '')}]")

            output = "\n".join(output_parts) if output_parts else str(result)

            if result.get("isError"):
                return ToolResult.from_error(
                    call_id=call.id,
                    tool_name=call.name,
                    error_message=output,
                )

            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=output,
            )

        except MCPConnectionError as e:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=str(e),
            )

    async def execute_batch_async(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls.

        Note: MCP servers typically process requests sequentially,
        so this method executes calls one at a time.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        results = []
        for call in calls:
            result = await self.execute_async(call)
            results.append(result)
        return tuple(results)

    # Synchronous convenience methods

    def connect_sync(self) -> None:
        """Connect to the MCP server synchronously."""
        asyncio.run(self.connect())

    def disconnect_sync(self) -> None:
        """Disconnect from the MCP server synchronously."""
        asyncio.run(self.disconnect())

    def list_tools_sync(self) -> tuple[ToolDefinition, ...]:
        """Fetch available tools synchronously."""
        return asyncio.run(self.list_tools())

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call synchronously."""
        return asyncio.run(self.execute_async(call))

    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls synchronously."""
        return asyncio.run(self.execute_batch_async(calls))

    # Context manager support

    async def __aenter__(self) -> "MCPToolExecutor":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def __enter__(self) -> "MCPToolExecutor":
        """Sync context manager entry."""
        self.connect_sync()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Sync context manager exit."""
        self.disconnect_sync()
