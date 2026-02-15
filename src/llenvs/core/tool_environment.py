"""Base implementation for tool-aware environments.

Provides common tool execution logic that can be inherited by
concrete tool environments.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from llenvs.core.state import Observation
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
    ToolResultStatus,
)

HiddenT = TypeVar("HiddenT")


@dataclass
class BaseToolEnvironment(Generic[HiddenT]):
    """Base implementation for tool-aware environments.

    Provides common tool execution logic that can be inherited by
    concrete tool environments.

    Subclasses must implement:
    - _get_initial_state(task_index, episode_id) -> State
    - _process_action(state, action) -> tuple[next_hidden, terminated, info]
    - reward_functions property
    - spec property

    Attributes:
        _tools: Tuple of available tool definitions.
        _executor: Optional tool executor (if None, subclass must override execute_tools).
    """

    _tools: tuple[ToolDefinition, ...] = field(default_factory=tuple)
    _executor: ToolExecutor | None = None

    @staticmethod
    def _tool_monitoring_rewards() -> tuple:
        """Create weight-0 monitoring rewards for tool usage diagnostics.

        These rewards are auto-attached to tool environments. With weight=0,
        they contribute nothing to the total reward but appear in the
        SignalBundle for inspection.
        """
        from llenvs.core.tool_rewards import ToolEfficiencyReward, ToolValidityReward

        return (ToolValidityReward(_weight=0.0), ToolEfficiencyReward(_weight=0.0))

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        """Get the tools available in this environment."""
        return self._tools

    def _validate_tool_call(self, call: ToolCall) -> ToolResult | None:
        """Validate a tool call against available tools.

        Args:
            call: The tool call to validate.

        Returns:
            ToolResult with error if invalid, None if valid.
        """
        tool_names = {t.name for t in self._tools}
        if call.name not in tool_names:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Unknown tool: {call.name}. Available tools: {', '.join(sorted(tool_names))}",
                status=ToolResultStatus.INVALID_TOOL,
            )
        return None

    def execute_tools(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute tool calls and return results.

        First validates all calls, then delegates to executor.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        results: list[ToolResult] = []

        for call in calls:
            # Validate the call
            validation_error = self._validate_tool_call(call)
            if validation_error is not None:
                results.append(validation_error)
                continue

            # Execute via executor if available
            if self._executor is not None:
                results.append(self._executor.execute(call))
            else:
                # Subclass must handle execution
                results.append(
                    ToolResult.from_error(
                        call_id=call.id,
                        tool_name=call.name,
                        error_message="No executor configured",
                    )
                )

        return tuple(results)

    def _check_terminal_tools(
        self,
        calls: tuple[ToolCall, ...],
    ) -> bool:
        """Check if any tool calls are terminal (end episode).

        Args:
            calls: Tuple of tool calls.

        Returns:
            True if any called tool is marked as terminal.
        """
        terminal_tools = {t.name for t in self._tools if t.is_terminal}
        return any(call.name in terminal_tools for call in calls)

    def _build_next_observation(
        self,
        current_obs: Observation,
        action: Any,
        tool_results: tuple[ToolResult, ...],
    ) -> Observation:
        """Build the next observation including tool results.

        Args:
            current_obs: Current Observation.
            action: The Action taken.
            tool_results: Results of any tool calls.

        Returns:
            New Observation with updated messages and tool results.
        """
        # Build the new message history
        messages = list(current_obs.messages)

        # Add the assistant's action as a message
        assistant_content: dict[str, Any] = {"role": "assistant"}
        if action.text:
            assistant_content["content"] = action.text
        if action.tool_calls:
            assistant_content["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in action.tool_calls
            ]
        messages.append(assistant_content)

        # Add tool results as messages
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "name": result.tool_name,
                    "content": str(result.output) if result.is_success else result.error,
                }
            )

        return Observation(
            prompt=current_obs.prompt,
            messages=tuple(messages),
            tool_results=tool_results,
            available_tools=self._tools,
        )
