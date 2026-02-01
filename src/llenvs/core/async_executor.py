"""Async tool executor for parallel tool execution.

Provides an AsyncToolExecutor that can run multiple tool calls concurrently
using asyncio, improving performance when tools are I/O-bound.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from env_evals.core.tools import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


@dataclass
class AsyncToolExecutor:
    """Tool executor that runs tool calls asynchronously.

    Supports both sync and async callables. When executing a batch of calls,
    runs them concurrently for better performance with I/O-bound tools.

    Example:
        ```python
        async def fetch_weather(city: str) -> str:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.weather.com/{city}") as resp:
                    return await resp.text()

        def calculate(expression: str) -> str:
            return str(eval(expression))  # Sync function also works

        executor = AsyncToolExecutor(tools={
            "get_weather": fetch_weather,
            "calculate": calculate,
        })

        # Execute multiple calls in parallel
        results = await executor.execute_batch_async((call1, call2, call3))
        ```
    """

    _tools: dict[str, Callable[..., Any] | Callable[..., Awaitable[Any]]]
    _timeout: float = 30.0

    def __init__(
        self,
        tools: dict[str, Callable[..., Any] | Callable[..., Awaitable[Any]]],
        timeout: float = 30.0,
    ) -> None:
        """Initialize with a mapping of tool names to callables.

        Args:
            tools: Dict mapping tool name to sync or async callable.
            timeout: Timeout in seconds for each tool call (default 30s).
        """
        self._tools = tools
        self._timeout = timeout

    async def execute_async(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call asynchronously.

        Args:
            call: The tool call to execute.

        Returns:
            ToolResult with the execution outcome.
        """
        if call.name not in self._tools:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Unknown tool: {call.name}",
                status=ToolResultStatus.INVALID_TOOL,
            )

        try:
            func = self._tools[call.name]
            # Check if the function is a coroutine function
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**call.arguments),
                    timeout=self._timeout,
                )
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**call.arguments)),
                    timeout=self._timeout,
                )

            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=result,
            )
        except asyncio.TimeoutError:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Tool execution timed out after {self._timeout}s",
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

    async def execute_batch_async(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls concurrently.

        All calls are executed in parallel using asyncio.gather().
        Results are returned in the same order as the input calls.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        if not calls:
            return ()

        tasks = [self.execute_async(call) for call in calls]
        results = await asyncio.gather(*tasks)
        return tuple(results)

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call synchronously.

        This is a convenience method that runs the async executor
        in an event loop. For multiple calls, prefer execute_batch()
        or the async methods.

        Args:
            call: The tool call to execute.

        Returns:
            ToolResult with the execution outcome.
        """
        return asyncio.run(self.execute_async(call))

    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls in parallel, synchronously.

        This is a convenience method that runs the async batch executor
        in an event loop.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        return asyncio.run(self.execute_batch_async(calls))
