"""Tool-aware environment protocol and base implementation.

Extends the base Environment protocol to support tool/function calling,
where the environment executes tools and returns results in observations.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from env_evals.core.environment import EnvironmentSpec, StepResult
from env_evals.core.reward import RewardBundle, RewardFunction
from env_evals.core.state import State
from env_evals.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
    ToolResultStatus,
)

HiddenT = TypeVar("HiddenT")


# Forward reference for AgentObservation - import at runtime to avoid circular import
def _get_agent_observation_class():
    from env_evals.core.state import AgentObservation

    return AgentObservation


def _get_agent_action_class():
    from env_evals.core.state import AgentAction

    return AgentAction


@runtime_checkable
class ToolEnvironment(Protocol[HiddenT]):
    """Protocol for environments that support tool calling.

    Extends the base Environment protocol with tool-specific methods.
    Models receive available tools in their observations and can call
    them via AgentAction. The environment executes tools internally.

    Type Parameters:
        HiddenT: Hidden state type (for reward computation).
    """

    @property
    def spec(self) -> EnvironmentSpec:
        """Get the environment specification."""
        ...

    @property
    def reward_functions(self) -> tuple[RewardFunction[Any, HiddenT, Any], ...]:
        """Get the reward functions used by this environment."""
        ...

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        """Get the tools available in this environment."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[Any, HiddenT], dict[str, Any]]:
        """Reset the environment and return initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Environment-specific options (e.g., task_index).

        Returns:
            Tuple of (initial_state with AgentObservation, info_dict).
        """
        ...

    def step(
        self,
        state: State[Any, HiddenT],
        action: Any,
    ) -> StepResult[Any, HiddenT]:
        """Take an action from the given state.

        For tool-calling actions, this executes the tools internally
        and includes results in the next observation.

        Args:
            state: Current state.
            action: AgentAction with optional tool calls.

        Returns:
            StepResult with next state containing tool results.
        """
        ...

    def execute_tools(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute tool calls and return results.

        This is called internally by step() but can also be used
        directly for testing or probing.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        ...


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
        current_obs: Any,
        action: Any,
        tool_results: tuple[ToolResult, ...],
    ) -> Any:
        """Build the next observation including tool results.

        Args:
            current_obs: Current AgentObservation.
            action: The AgentAction taken.
            tool_results: Results of any tool calls.

        Returns:
            New AgentObservation with updated messages and tool results.
        """
        AgentObservation = _get_agent_observation_class()

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

        return AgentObservation(
            prompt=current_obs.prompt,
            messages=tuple(messages),
            tool_results=tool_results,
            available_tools=self._tools,
        )
