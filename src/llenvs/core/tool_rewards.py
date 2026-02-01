"""Tool-specific reward functions.

Provides reward functions for evaluating the quality and efficiency
of tool usage by models.
"""

from dataclasses import dataclass, field
from typing import Any

from llenvs.core.reward import RewardSignal, RewardType
from llenvs.core.state import State
from llenvs.core.tools import ToolResultStatus


@dataclass
class ToolValidityReward:
    """Reward based on validity of tool calls.

    Awards 1.0 if all tool calls are valid (correct tool name, valid arguments,
    successful execution). Gives partial credit based on fraction of valid calls.

    Attributes:
        _name: Name of this reward function.
        _reward_type: Type of reward (STEP by default).
    """

    _name: str = "tool_validity"
    _reward_type: RewardType = field(default=RewardType.STEP)

    @property
    def name(self) -> str:
        """Unique name for this reward function."""
        return self._name

    @property
    def reward_type(self) -> RewardType:
        """Type of reward this function produces."""
        return self._reward_type

    def compute(
        self,
        state: State[Any, Any],
        action: Any,
        next_state: State[Any, Any],
    ) -> RewardSignal:
        """Compute validity reward for tool calls in action.

        Args:
            state: State before action.
            action: Action taken (should be AgentAction).
            next_state: State after action.

        Returns:
            RewardSignal with value in [0, 1] based on tool validity.
        """
        # Check if action has tool calls
        if not hasattr(action, "tool_calls") or not action.tool_calls:
            # No tool calls - neutral reward
            return RewardSignal(
                value=1.0,
                name=self._name,
                reward_type=self._reward_type,
                metadata={"num_calls": 0, "num_valid": 0},
            )

        # Check if next observation has tool results
        obs = next_state.observation
        if not hasattr(obs, "tool_results") or not obs.tool_results:
            # No results - assume all valid (shouldn't happen)
            return RewardSignal(
                value=1.0,
                name=self._name,
                reward_type=self._reward_type,
                metadata={"num_calls": len(action.tool_calls), "num_valid": len(action.tool_calls)},
            )

        # Count successful tool calls
        num_calls = len(action.tool_calls)
        num_valid = sum(
            1 for result in obs.tool_results if result.status == ToolResultStatus.SUCCESS
        )

        # Partial credit
        value = num_valid / num_calls if num_calls > 0 else 1.0

        return RewardSignal(
            value=value,
            name=self._name,
            reward_type=self._reward_type,
            metadata={
                "num_calls": num_calls,
                "num_valid": num_valid,
                "statuses": [r.status.name for r in obs.tool_results],
            },
        )


@dataclass
class ToolEfficiencyReward:
    """Reward for efficient tool usage.

    Penalizes excessive or redundant tool calls. Awards higher scores
    for using fewer tools to accomplish the task.

    Attributes:
        _name: Name of this reward function.
        _reward_type: Type of reward (STEP by default).
        max_calls_per_step: Target maximum calls per step.
        penalty_per_excess: Penalty per call beyond max.
        duplicate_penalty: Extra penalty for duplicate calls.
    """

    _name: str = "tool_efficiency"
    _reward_type: RewardType = field(default=RewardType.STEP)
    max_calls_per_step: int = 5
    penalty_per_excess: float = 0.1
    duplicate_penalty: float = 0.2

    @property
    def name(self) -> str:
        """Unique name for this reward function."""
        return self._name

    @property
    def reward_type(self) -> RewardType:
        """Type of reward this function produces."""
        return self._reward_type

    def compute(
        self,
        state: State[Any, Any],
        action: Any,
        next_state: State[Any, Any],
    ) -> RewardSignal:
        """Compute efficiency reward for tool usage.

        Args:
            state: State before action.
            action: Action taken (should be AgentAction).
            next_state: State after action.

        Returns:
            RewardSignal with value in [0, 1] based on efficiency.
        """
        # Check if action has tool calls
        if not hasattr(action, "tool_calls") or not action.tool_calls:
            # No tool calls - perfect efficiency
            return RewardSignal(
                value=1.0,
                name=self._name,
                reward_type=self._reward_type,
                metadata={"num_calls": 0, "excess": 0, "duplicates": 0},
            )

        num_calls = len(action.tool_calls)
        value = 1.0

        # Penalty for excess calls
        excess = max(0, num_calls - self.max_calls_per_step)
        value -= excess * self.penalty_per_excess

        # Penalty for duplicate calls (same tool with same arguments)
        seen_calls: set[tuple[str, str]] = set()
        duplicates = 0
        for call in action.tool_calls:
            # Create a hashable representation
            args_str = str(sorted(call.arguments.items()))
            call_key = (call.name, args_str)
            if call_key in seen_calls:
                duplicates += 1
            seen_calls.add(call_key)

        value -= duplicates * self.duplicate_penalty

        # Clamp to [0, 1]
        value = max(0.0, min(1.0, value))

        return RewardSignal(
            value=value,
            name=self._name,
            reward_type=self._reward_type,
            metadata={
                "num_calls": num_calls,
                "excess": excess,
                "duplicates": duplicates,
            },
        )
