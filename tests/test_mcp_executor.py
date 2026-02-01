"""Tests for MCPToolExecutor."""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from llenvs.core.mcp_executor import (
    MCPToolExecutor,
    MCPServerConfig,
    MCPConnectionError,
)
from llenvs.core.tools import ToolCall, ToolParameterType, ToolResultStatus


class TestMCPServerConfig:
    """Tests for MCP server configuration."""

    def test_stdio_config(self):
        """Test creating stdio transport config."""
        config = MCPServerConfig(
            command="npx",
            args=("-y", "@modelcontextprotocol/server-test"),
            timeout=60.0,
        )
        assert config.command == "npx"
        assert config.args == ("-y", "@modelcontextprotocol/server-test")
        assert config.timeout == 60.0
        assert config.url is None

    def test_url_config(self):
        """Test creating URL transport config."""
        config = MCPServerConfig(
            url="http://localhost:8080/sse",
            timeout=30.0,
        )
        assert config.url == "http://localhost:8080/sse"
        assert config.command is None


class TestMCPToolExecutorUnit:
    """Unit tests for MCP executor with mocked subprocess."""

    @pytest.fixture
    def mock_process(self):
        """Create a mock subprocess."""
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdout = MagicMock()
        process.stderr = MagicMock()
        return process

    @pytest.fixture
    def executor_with_mock(self, mock_process):
        """Create executor with mocked subprocess."""
        config = MCPServerConfig(command="test-server")
        executor = MCPToolExecutor(config)
        executor._process = mock_process
        executor._connected = True
        return executor, mock_process

    def test_parse_tool_definition(self):
        """Test parsing MCP tool definition to ToolDefinition."""
        config = MCPServerConfig(command="test")
        executor = MCPToolExecutor(config)

        mcp_tool = {
            "name": "read_file",
            "description": "Read contents of a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "enum": ["utf-8", "ascii"],
                    },
                },
                "required": ["path"],
            },
        }

        tool_def = executor._parse_tool_definition(mcp_tool)

        assert tool_def.name == "read_file"
        assert tool_def.description == "Read contents of a file"
        assert len(tool_def.parameters) == 2

        path_param = next(p for p in tool_def.parameters if p.name == "path")
        assert path_param.type == ToolParameterType.STRING
        assert path_param.required is True

        encoding_param = next(p for p in tool_def.parameters if p.name == "encoding")
        assert encoding_param.required is False
        assert encoding_param.enum == ("utf-8", "ascii")

    def test_map_json_types(self):
        """Test JSON type mapping."""
        config = MCPServerConfig(command="test")
        executor = MCPToolExecutor(config)

        assert executor._map_json_type("string") == ToolParameterType.STRING
        assert executor._map_json_type("number") == ToolParameterType.NUMBER
        assert executor._map_json_type("integer") == ToolParameterType.INTEGER
        assert executor._map_json_type("boolean") == ToolParameterType.BOOLEAN
        assert executor._map_json_type("array") == ToolParameterType.ARRAY
        assert executor._map_json_type("object") == ToolParameterType.OBJECT
        # Unknown types default to string
        assert executor._map_json_type("unknown") == ToolParameterType.STRING

    def test_execute_not_connected(self):
        """Test execute fails when not connected."""
        config = MCPServerConfig(command="test")
        executor = MCPToolExecutor(config)

        call = ToolCall(id="1", name="test", arguments={})
        result = executor.execute(call)

        assert result.is_error
        assert "Not connected" in result.error


class TestMCPToolExecutorIntegration:
    """Integration-style tests with mocked MCP protocol."""

    @pytest.fixture
    def mock_mcp_responses(self):
        """Create mock MCP JSON-RPC responses."""
        return {
            "initialize": {
                "jsonrpc": "2.0",
                "id": "1",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "test-server", "version": "1.0"},
                },
            },
            "tools/list": {
                "jsonrpc": "2.0",
                "id": "2",
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo input back",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string", "description": "Message to echo"}
                                },
                                "required": ["message"],
                            },
                        }
                    ]
                },
            },
            "tools/call": {
                "jsonrpc": "2.0",
                "id": "3",
                "result": {
                    "content": [{"type": "text", "text": "echoed: hello"}],
                },
            },
        }

    def test_list_tools_mocked(self, mock_mcp_responses):
        """Test listing tools with mocked responses."""
        config = MCPServerConfig(command="test-server")
        executor = MCPToolExecutor(config)

        # Setup mock process
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()

        # Make stdout.readline return the tools/list response
        mock_process.stdout.readline.return_value = (
            json.dumps(mock_mcp_responses["tools/list"]).encode() + b"\n"
        )

        executor._process = mock_process
        executor._connected = True

        # Mock asyncio for sync method
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tools = executor.list_tools_sync()

            assert len(tools) == 1
            assert tools[0].name == "echo"
            assert tools[0].description == "Echo input back"
        finally:
            loop.close()

    def test_execute_tool_success_mocked(self, mock_mcp_responses):
        """Test executing a tool with mocked responses."""
        config = MCPServerConfig(command="test-server")
        executor = MCPToolExecutor(config)

        # Setup mock process
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.return_value = (
            json.dumps(mock_mcp_responses["tools/call"]).encode() + b"\n"
        )

        executor._process = mock_process
        executor._connected = True

        call = ToolCall(id="test-1", name="echo", arguments={"message": "hello"})
        result = executor.execute(call)

        assert result.is_success
        assert result.output == "echoed: hello"
        assert result.call_id == "test-1"

    def test_execute_tool_error_response(self):
        """Test handling error response from MCP server."""
        config = MCPServerConfig(command="test-server")
        executor = MCPToolExecutor(config)

        error_response = {
            "jsonrpc": "2.0",
            "id": "1",
            "error": {"code": -32600, "message": "Tool not found: unknown"},
        }

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.return_value = (
            json.dumps(error_response).encode() + b"\n"
        )

        executor._process = mock_process
        executor._connected = True

        call = ToolCall(id="1", name="unknown", arguments={})
        result = executor.execute(call)

        assert result.is_error
        assert "Tool not found" in result.error

    def test_execute_tool_with_is_error(self):
        """Test handling tool result with isError flag."""
        config = MCPServerConfig(command="test-server")
        executor = MCPToolExecutor(config)

        error_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "content": [{"type": "text", "text": "File not found: /missing.txt"}],
                "isError": True,
            },
        }

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.return_value = json.dumps(error_result).encode() + b"\n"

        executor._process = mock_process
        executor._connected = True

        call = ToolCall(id="1", name="read_file", arguments={"path": "/missing.txt"})
        result = executor.execute(call)

        assert result.is_error
        assert "File not found" in result.error


class TestMCPToolExecutorContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_not_implemented_url(self):
        """Test that URL transport raises NotImplementedError."""
        config = MCPServerConfig(url="http://localhost:8080")
        executor = MCPToolExecutor(config)

        with pytest.raises(NotImplementedError):
            executor.connect_sync()

    def test_no_command_or_url_raises(self):
        """Test that missing command and URL raises error."""
        config = MCPServerConfig()
        executor = MCPToolExecutor(config)

        with pytest.raises(MCPConnectionError, match="No command or URL"):
            executor.connect_sync()


class TestMCPConnectionError:
    """Tests for MCP connection error handling."""

    def test_connection_error_message(self):
        """Test MCPConnectionError contains message."""
        error = MCPConnectionError("Connection refused")
        assert str(error) == "Connection refused"

    def test_connection_error_chaining(self):
        """Test MCPConnectionError can chain exceptions."""
        original = ValueError("original error")
        error = MCPConnectionError("Wrapped error")
        error.__cause__ = original
        assert error.__cause__ is original
