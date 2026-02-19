"""Aviary adapter — wraps fhaviary tool-calling environments as MDP environments.

Aviary is an async-first framework for language agent RL environments where
all interaction is through tool calling. Built-in environments include
GSM8K, HotPotQA, LABBench, and LFRQA.

Key design: Aviary tools have access to internal environment state (via an
injected ``state`` parameter), so tool execution is delegated to Aviary's
``step()`` rather than extracted and wrapped in our ToolExecutor. We extend
BaseToolEnvironment for the tool infrastructure (_tools, available_tools,
monitoring rewards) but build observations manually from Aviary's response
messages.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.async_utils import run_async as _run_async
from llenvs.core.environment import (
    EnvironmentSpec,
    StepResult,
    _StateContinuityTracker,
)
from llenvs.core.reward import (
    RewardFunction,
    RewardType,
    Signal,
    SignalBundle,
)
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tools import (
    ToolDefinition,
    ToolResult,
    oai_tools_to_definitions,
)

logger = logging.getLogger(__name__)


# ── Tool conversion ───────────────────────────────────────────────


def _aviary_tools_to_definitions(
    tools: list[Any],
) -> tuple[ToolDefinition, ...]:
    """Convert Aviary Tool objects to ToolDefinitions.

    Calls ``tool.model_dump(by_alias=True)`` on each Aviary Tool to get
    OpenAI-format schema, then delegates to ``oai_tools_to_definitions()``.
    Falls back to manual extraction from ``tool.info`` if ``model_dump`` fails.

    Args:
        tools: List of Aviary Tool objects.

    Returns:
        Tuple of ToolDefinition objects.
    """
    oai_schemas: list[dict[str, Any]] = []
    for tool in tools:
        try:
            schema = tool.model_dump(by_alias=True)
            # Aviary dumps to {"type": "function", "function": {...}}
            oai_schemas.append(schema)
        except (AttributeError, TypeError):
            # Fallback: build from tool.info if available
            try:
                info = tool.info
                oai_schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": info.name,
                            "description": info.description or "",
                            "parameters": info.parameters or {},
                        },
                    }
                )
            except AttributeError:
                logger.warning(f"Skipping unconvertible Aviary tool: {tool}")
                continue

    return oai_tools_to_definitions(oai_schemas)


# ── Action conversion ─────────────────────────────────────────────


def _action_to_tool_request(action: Action) -> Any:
    """Convert an llenvs Action to an Aviary ToolRequestMessage.

    Args:
        action: The llenvs action with tool_calls and/or text.

    Returns:
        An Aviary ToolRequestMessage.
    """
    from aviary.core import ToolCall as AviaryToolCall
    from aviary.core import ToolCallFunction, ToolRequestMessage

    aviary_tool_calls: list[AviaryToolCall] = []
    for tc in action.tool_calls:
        args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
        aviary_tool_calls.append(
            AviaryToolCall(
                id=tc.id,
                function=ToolCallFunction(
                    name=tc.name,
                    arguments=args_str,
                ),
            )
        )

    return ToolRequestMessage(
        content=action.text or "",
        tool_calls=aviary_tool_calls or None,
    )


# ── Message conversion ────────────────────────────────────────────


def _aviary_messages_to_observation(
    messages: list[Any],
    prompt: str,
    prior_messages: tuple[dict[str, Any], ...],
    action: Action,
    available_tools: tuple[ToolDefinition, ...],
) -> tuple[Observation, tuple[ToolResult, ...]]:
    """Convert Aviary response messages to an Observation.

    Adds the assistant action to message history, then processes Aviary
    response messages into chat-format dicts and ToolResults.

    Args:
        messages: List of Aviary Message objects from step().
        prompt: The original prompt string.
        prior_messages: Previous message history.
        action: The Action that was taken.
        available_tools: Current tool definitions.

    Returns:
        Tuple of (new Observation, tool results).
    """
    new_messages = list(prior_messages)

    # Add assistant action to history
    assistant_msg: dict[str, Any] = {"role": "assistant"}
    if action.text:
        assistant_msg["content"] = action.text
    if action.tool_calls:
        assistant_msg["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in action.tool_calls
        ]
    new_messages.append(assistant_msg)

    # Process Aviary response messages
    tool_results: list[ToolResult] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", "")

        if role == "tool":
            # Tool response message
            tool_call_id = getattr(msg, "tool_call_id", None) or str(uuid.uuid4())
            tool_name = getattr(msg, "name", None) or "unknown"
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": str(content) if content else "",
                }
            )
            tool_results.append(
                ToolResult.success(
                    call_id=tool_call_id,
                    tool_name=tool_name,
                    output=str(content) if content else "",
                )
            )
        else:
            # Regular message (user, system, etc.)
            msg_dict: dict[str, Any] = {"role": role or "assistant"}
            if content:
                msg_dict["content"] = str(content)
            new_messages.append(msg_dict)

    return (
        Observation(
            prompt=prompt,
            messages=tuple(new_messages),
            tool_results=tuple(tool_results),
            available_tools=available_tools,
        ),
        tuple(tool_results),
    )


# ── Hidden state ──────────────────────────────────────────────────


@dataclass(frozen=True)
class AviaryHidden:
    """Hidden state for Aviary environments.

    Attributes:
        task_index: Index into the task dataset.
        env_name: Name of the Aviary environment.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        cumulative_reward: Sum of all step rewards so far.
        aviary_reward: Reward from the most recent step.
    """

    task_index: int
    env_name: str
    episode_step: int = 0
    last_action: str | None = None
    cumulative_reward: float = 0.0
    aviary_reward: float = 0.0


# ── Reward function ───────────────────────────────────────────────


@dataclass
class AviaryReward:
    """Reward function that reads Aviary's native step reward.

    Returns OUTCOME when terminal, STEP otherwise.
    """

    _name: str = "aviary"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> Signal:
        aviary_reward = getattr(next_state.hidden, "aviary_reward", 0.0)
        cumulative = getattr(next_state.hidden, "cumulative_reward", 0.0)
        is_terminal = next_state.metadata.is_terminal

        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME if is_terminal else RewardType.STEP,
            reward=aviary_reward,
            metadata={
                "cumulative_reward": cumulative,
                "is_terminal": is_terminal,
            },
        )


# ── Environment ───────────────────────────────────────────────────


class AviaryEnvironment(BaseToolEnvironment[AviaryHidden]):
    """MDP wrapper for Aviary tool-calling environments.

    Aviary tools have access to internal environment state, so tool execution
    is delegated to Aviary's ``step()`` rather than extracted and wrapped.
    We extend BaseToolEnvironment for the tool infrastructure but skip
    ``execute_tools()`` entirely.
    """

    def __init__(
        self,
        dataset: Any,
        env_name: str,
        max_steps: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._dataset = dataset
        self._env_name = env_name
        self._max_steps = max_steps
        self._active_env: Any | None = None
        self._executor = None  # Not used — Aviary handles execution

        self._native_rewards: tuple[RewardFunction, ...] = (
            AviaryReward(),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_name,
            adapter="aviary",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={
                "env_name": self._env_name,
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def __len__(self) -> int:
        return len(self._dataset)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[AviaryHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._dataset):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._dataset)})")

        # Create a fresh Aviary environment for this task
        aviary_env = _run_async(self._dataset.get_new_env_by_idx(task_index))
        self._active_env = aviary_env

        # Reset the Aviary environment
        messages, tools = _run_async(aviary_env.reset())

        # Convert tools to ToolDefinitions
        self._tools = _aviary_tools_to_definitions(tools)

        # Build prompt from reset messages
        prompt_parts: list[str] = []
        for msg in messages:
            content = getattr(msg, "content", "")
            if content:
                prompt_parts.append(str(content))
        prompt = "\n".join(prompt_parts) if prompt_parts else ""

        hidden = AviaryHidden(
            task_index=task_index,
            env_name=self._env_name,
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            aviary_reward=0.0,
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        observation = Observation(
            prompt=prompt,
            available_tools=self._tools,
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        info: dict[str, Any] = {
            "task_index": task_index,
            "env_name": self._env_name,
            "num_tools": len(self._tools),
        }

        return state, info

    def step(
        self,
        state: State[AviaryHidden],
        action: Action,
    ) -> StepResult[AviaryHidden]:
        self._state_tracker.validate(state, "AviaryEnvironment")

        if self._active_env is None:
            raise RuntimeError("No active Aviary environment. Call reset() first.")

        # Convert action to Aviary ToolRequestMessage
        tool_request = _action_to_tool_request(action)

        # Step the Aviary environment
        messages, reward, done, truncated = _run_async(self._active_env.step(tool_request))

        next_step = state.hidden.episode_step + 1
        cumulative_reward = state.hidden.cumulative_reward + reward

        # Check max_steps truncation
        if self._max_steps is not None and next_step >= self._max_steps:
            truncated = True

        # Build observation from Aviary response
        next_obs, tool_results = _aviary_messages_to_observation(
            messages=messages,
            prompt=state.observation.prompt,
            prior_messages=state.observation.messages,
            action=action,
            available_tools=self._tools,
        )

        next_hidden = AviaryHidden(
            task_index=state.hidden.task_index,
            env_name=state.hidden.env_name,
            episode_step=next_step,
            last_action=action.text,
            cumulative_reward=cumulative_reward,
            aviary_reward=reward,
        )

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=done or truncated,
            info={
                **state.metadata.info,
                "aviary_reward": reward,
                "cumulative_reward": cumulative_reward,
            },
        )

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=done,
            truncated=truncated,
            info={
                "tool_results": tool_results,
                "episode_step": next_step,
                "aviary_reward": reward,
            },
        )

    def compute_rewards(
        self,
        state: State[AviaryHidden],
        action: Action,
        next_state: State[AviaryHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Close the active Aviary environment if any."""
        if self._active_env is not None:
            try:
                close = getattr(self._active_env, "close", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        _run_async(result)
            except Exception as e:
                logger.warning(f"Error closing Aviary environment: {e}")
            finally:
                self._active_env = None


# ── Presets ────────────────────────────────────────────────────────

AVIARY_PRESETS: dict[str, dict[str, str]] = {
    "gsm8k": {
        "dataset_class": "GSM8kDataset",
        "module": "aviary.envs.gsm8k",
    },
    "hotpotqa": {
        "dataset_class": "HotPotQADataset",
        "module": "aviary.envs.hotpotqa",
    },
    "labbench": {
        "dataset_class": "LABBenchDataset",
        "module": "aviary.envs.labbench",
    },
    "lfrqa": {
        "dataset_class": "LFRQADataset",
        "module": "aviary.envs.lfrqa",
    },
}


# ── Adapter ────────────────────────────────────────────────────────


class AviaryAdapter:
    """Adapter for the fhaviary library.

    Routes environment creation through Aviary's TaskDataset interface.
    """

    @property
    def name(self) -> str:
        return "aviary"

    def _get_aviary(self) -> Any:
        try:
            import aviary

            return aviary
        except ImportError as e:
            raise ImportError(
                "fhaviary is required for AviaryAdapter. Install with: pip install fhaviary"
            ) from e

    def list_environments(self) -> list[str]:
        return list(AVIARY_PRESETS.keys())

    def get_environment(
        self,
        name: str,
        dataset: Any | None = None,
        max_steps: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> AviaryEnvironment:
        """Create an Aviary environment.

        Args:
            name: Preset name (e.g., "gsm8k") or custom name.
            dataset: Pre-created TaskDataset. If None, loads from preset.
            max_steps: Maximum steps per episode.
            extra_rewards: Additional reward functions.
            **kwargs: Passed to the dataset constructor.

        Returns:
            AviaryEnvironment wrapping the dataset.

        Raises:
            ValueError: If name is not a preset and no dataset is provided.
        """
        self._get_aviary()

        if dataset is not None:
            return AviaryEnvironment(
                dataset=dataset,
                env_name=name,
                max_steps=max_steps,
                extra_rewards=extra_rewards,
            )

        if name not in AVIARY_PRESETS:
            raise ValueError(
                f"Unknown Aviary environment: {name!r}. "
                f"Available presets: {list(AVIARY_PRESETS.keys())}. "
                f"Pass a dataset= argument for custom environments."
            )

        preset = AVIARY_PRESETS[name]
        module = importlib.import_module(preset["module"])
        dataset_class = getattr(module, preset["dataset_class"])
        ds = dataset_class(**kwargs)

        return AviaryEnvironment(
            dataset=ds,
            env_name=name,
            max_steps=max_steps,
            extra_rewards=extra_rewards,
        )

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_default_system_prompt(self, name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": name,
            "adapter": self.name,
            "description": f"Aviary environment: {name}",
        }
        if name in AVIARY_PRESETS:
            info["preset"] = AVIARY_PRESETS[name]
        return info
