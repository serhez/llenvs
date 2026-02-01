"""Tests for AsyncToolExecutor."""

import asyncio
import pytest
from unittest.mock import MagicMock

from llenvs.core.async_executor import AsyncToolExecutor
from llenvs.core.tools import ToolCall, ToolResultStatus


class TestAsyncToolExecutor:
    """Tests for the async tool executor."""

    def test_execute_sync_function(self):
        """Test executing a sync function."""

        def add(a: int, b: int) -> str:
            return str(a + b)

        executor = AsyncToolExecutor(tools={"add": add})
        call = ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})
        result = executor.execute(call)

        assert result.is_success
        assert result.output == "5"
        assert result.call_id == "1"
        assert result.tool_name == "add"

    def test_execute_async_function(self):
        """Test executing an async function."""

        async def fetch(url: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return f"fetched: {url}"

        executor = AsyncToolExecutor(tools={"fetch": fetch})
        call = ToolCall(id="2", name="fetch", arguments={"url": "https://example.com"})
        result = executor.execute(call)

        assert result.is_success
        assert result.output == "fetched: https://example.com"

    def test_execute_unknown_tool(self):
        """Test executing an unknown tool."""
        executor = AsyncToolExecutor(tools={})
        call = ToolCall(id="1", name="unknown", arguments={})
        result = executor.execute(call)

        assert result.is_error
        assert result.status == ToolResultStatus.INVALID_TOOL
        assert "Unknown tool" in result.error

    def test_execute_invalid_arguments(self):
        """Test executing with invalid arguments."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        executor = AsyncToolExecutor(tools={"greet": greet})
        call = ToolCall(id="1", name="greet", arguments={"wrong_param": "test"})
        result = executor.execute(call)

        assert result.is_error
        assert result.status == ToolResultStatus.INVALID_ARGUMENTS

    def test_execute_function_raises(self):
        """Test handling function exceptions."""

        def fail() -> str:
            raise ValueError("Something went wrong")

        executor = AsyncToolExecutor(tools={"fail": fail})
        call = ToolCall(id="1", name="fail", arguments={})
        result = executor.execute(call)

        assert result.is_error
        assert result.status == ToolResultStatus.ERROR
        assert "Something went wrong" in result.error

    def test_execute_batch_parallel(self):
        """Test batch execution runs in parallel."""
        execution_times = []

        async def slow_task(task_id: int) -> str:
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Each task takes 0.1s
            end = asyncio.get_event_loop().time()
            execution_times.append((task_id, start, end))
            return f"done: {task_id}"

        executor = AsyncToolExecutor(tools={"slow_task": slow_task})
        calls = tuple(
            ToolCall(id=str(i), name="slow_task", arguments={"task_id": i}) for i in range(3)
        )

        # Execute in batch
        results = executor.execute_batch(calls)

        # All should succeed
        assert len(results) == 3
        assert all(r.is_success for r in results)

        # Check parallel execution: total time should be ~0.1s, not ~0.3s
        # All tasks should have overlapping execution times
        starts = [t[1] for t in execution_times]
        ends = [t[2] for t in execution_times]
        total_time = max(ends) - min(starts)
        assert total_time < 0.2  # Should be ~0.1s if parallel

    def test_execute_batch_empty(self):
        """Test batch execution with empty calls."""
        executor = AsyncToolExecutor(tools={})
        results = executor.execute_batch(())
        assert results == ()

    def test_execute_batch_mixed_results(self):
        """Test batch with some successes and failures."""

        def good(x: int) -> str:
            return str(x * 2)

        def bad() -> str:
            raise RuntimeError("fail")

        executor = AsyncToolExecutor(tools={"good": good, "bad": bad})
        calls = (
            ToolCall(id="1", name="good", arguments={"x": 5}),
            ToolCall(id="2", name="bad", arguments={}),
            ToolCall(id="3", name="good", arguments={"x": 10}),
        )
        results = executor.execute_batch(calls)

        assert len(results) == 3
        assert results[0].is_success
        assert results[0].output == "10"
        assert results[1].is_error
        assert results[2].is_success
        assert results[2].output == "20"

    def test_timeout(self):
        """Test that slow functions time out."""

        async def slow() -> str:
            await asyncio.sleep(10)  # Very slow
            return "done"

        executor = AsyncToolExecutor(tools={"slow": slow}, timeout=0.1)
        call = ToolCall(id="1", name="slow", arguments={})
        result = executor.execute(call)

        assert result.is_error
        assert "timed out" in result.error.lower()


class TestAsyncToolExecutorAsync:
    """Async tests for the executor."""

    def test_execute_async_directly(self):
        """Test calling execute_async directly."""

        async def multiply(a: int, b: int) -> str:
            return str(a * b)

        async def run_test():
            executor = AsyncToolExecutor(tools={"multiply": multiply})
            call = ToolCall(id="1", name="multiply", arguments={"a": 3, "b": 4})
            result = await executor.execute_async(call)

            assert result.is_success
            assert result.output == "12"

        asyncio.run(run_test())

    def test_execute_batch_async_directly(self):
        """Test calling execute_batch_async directly."""

        def square(n: int) -> str:
            return str(n * n)

        async def run_test():
            executor = AsyncToolExecutor(tools={"square": square})
            calls = tuple(
                ToolCall(id=str(i), name="square", arguments={"n": i}) for i in range(1, 4)
            )
            results = await executor.execute_batch_async(calls)

            assert len(results) == 3
            assert results[0].output == "1"
            assert results[1].output == "4"
            assert results[2].output == "9"

        asyncio.run(run_test())

    def test_concurrent_with_other_tasks(self):
        """Test executor works well with other async tasks."""

        async def run_test():
            other_task_ran = False

            async def other_task():
                nonlocal other_task_ran
                await asyncio.sleep(0.05)
                other_task_ran = True

            async def tool_func(x: int) -> str:
                await asyncio.sleep(0.05)
                return str(x)

            executor = AsyncToolExecutor(tools={"tool": tool_func})
            call = ToolCall(id="1", name="tool", arguments={"x": 42})

            # Run tool and other task concurrently
            result, _ = await asyncio.gather(executor.execute_async(call), other_task())

            assert result.is_success
            assert result.output == "42"
            assert other_task_ran

        asyncio.run(run_test())
