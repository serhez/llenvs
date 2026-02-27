"""OpenEnv adapter — wraps OpenEnv session-based environments as MDP environments.

OpenEnv environments are server-backed, session-based (no task indices,
no __len__, no seed). The adapter connects to a running server via URL
and creates fresh sessions on each reset.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
)

logger = logging.getLogger(__name__)


# ── Type mapping for MCP tool schemas ───────────────────────────────

_JSON_TYPE_MAP: dict[str, ToolParameterType] = {
    "string": ToolParameterType.STRING,
    "integer": ToolParameterType.INTEGER,
    "number": ToolParameterType.NUMBER,
    "boolean": ToolParameterType.BOOLEAN,
    "array": ToolParameterType.ARRAY,
    "object": ToolParameterType.OBJECT,
}


def _mcp_tools_to_definitions(
    tools: list[Any],
) -> tuple[ToolDefinition, ...]:
    """Convert MCP Tool objects to ToolDefinitions.

    Preserves the original ``input_schema`` as ``raw_schema`` on each
    ``ToolDefinition`` for full-fidelity schema passthrough.  The flat
    ``parameters`` tuple is a best-effort parse for inspection/display.

    Args:
        tools: List of MCP Tool objects (name, description, input_schema).

    Returns:
        Tuple of ToolDefinition objects.
    """
    definitions: list[ToolDefinition] = []

    for tool in tools:
        schema = getattr(tool, "input_schema", {}) or {}
        properties = schema.get("properties", {})
        required_names = set(schema.get("required", []))

        parameters: list[ToolParameter] = []
        for param_name, param_schema in properties.items():
            param_type_str = param_schema.get("type", "string")
            param_type = _JSON_TYPE_MAP.get(param_type_str, ToolParameterType.STRING)
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_schema.get("description", ""),
                    required=param_name in required_names,
                )
            )

        # Build OpenAI-style function dict for raw_schema passthrough
        raw_schema: dict[str, Any] = {
            "name": tool.name,
            "description": getattr(tool, "description", ""),
            "parameters": schema,
        }

        definitions.append(
            ToolDefinition(
                name=tool.name,
                description=getattr(tool, "description", ""),
                parameters=tuple(parameters),
                raw_schema=raw_schema,
            )
        )

    return tuple(definitions)


# ── Observation coercion ────────────────────────────────────────────


def _coerce_observation(obs: Any) -> str:
    """Coerce an OpenEnv observation to a string.

    Checks common keys in dict observations, falls back to JSON.

    Args:
        obs: Observation dict, string, or other type.

    Returns:
        String representation of the observation.
    """
    if isinstance(obs, str):
        return obs

    if isinstance(obs, dict):
        for key in ("text", "content", "observation", "message"):
            if key in obs:
                return str(obs[key])
        return json.dumps(obs)

    return str(obs)


# ── Hidden state ────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpenEnvHidden:
    """Hidden state for OpenEnv environments.

    Attributes:
        env_name: The environment name.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        session_info: Frozen representation of server state.
    """

    env_name: str
    episode_step: int
    last_action: str | None
    session_info: tuple[tuple[str, Any], ...]


# ── Reward function ─────────────────────────────────────────────────


@dataclass
class OpenEnvReward:
    """Reward function that reads native OpenEnv step rewards.

    Reads the reward from the step result metadata (stored in
    next_state.metadata.info["openenv_reward"]).
    Uses STEP type for intermediate steps, OUTCOME for terminal.
    """

    _name: str = "openenv_native"

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
        reward_value = next_state.metadata.info.get("openenv_reward", 0.0) or 0.0
        is_terminal = next_state.metadata.is_terminal

        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME if is_terminal else RewardType.STEP,
            reward=float(reward_value),
        )


# ── Text environment ────────────────────────────────────────────────


class OpenEnvEnvironment:
    """MDP wrapper for OpenEnv text-based environments.

    Session-based: task_index is ignored, each reset creates a fresh session.
    No __len__ support.
    """

    def __init__(
        self,
        client: Any,
        env_name: str,
        max_steps: int | None = None,
        action_format: Callable[[str], Any] | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._client = client
        self._env_name = env_name
        self._max_steps = max_steps
        self._action_format = action_format or (lambda text: {"text": text})

        self._native_rewards: tuple[RewardFunction, ...] = (OpenEnvReward(),)
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_name,
            adapter="openenv",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=False,
            supports_len=False,
            supports_seed=False,
            metadata={"env_name": self._env_name},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[OpenEnvHidden], dict[str, Any]]:
        # task_index is deliberately ignored — sessions are fresh
        step_result = self._client.reset()

        obs_text = _coerce_observation(step_result.observation)

        hidden = OpenEnvHidden(
            env_name=self._env_name,
            episode_step=0,
            last_action=None,
            session_info=(),
        )

        episode_id = str(uuid.uuid4())
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={},
        )

        observation = Observation(
            prompt=obs_text,
            task=ObservationContent(text=obs_text),
            state=ObservationContent(text=obs_text),
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        info: dict[str, Any] = {
            "env_name": self._env_name,
            "initial_observation": step_result.observation,
        }

        return state, info

    def step(
        self,
        state: State[OpenEnvHidden],
        action: Action,
    ) -> StepResult[OpenEnvHidden]:
        self._state_tracker.validate(state, "OpenEnvEnvironment")
        formatted = self._action_format(action.text)
        step_result = self._client.step(formatted)

        next_step = state.hidden.episode_step + 1
        terminated = step_result.done
        truncated = not terminated and self._max_steps is not None and next_step >= self._max_steps

        obs_text = _coerce_observation(step_result.observation)

        # Build message history
        messages = list(state.observation.messages)
        messages.append({"role": "assistant", "content": action.text})
        messages.append({"role": "user", "content": obs_text})

        next_observation = Observation(
            prompt=state.observation.prompt,
            messages=tuple(messages),
            task=state.observation.task,
            state=ObservationContent(
                text=obs_text,
                data={"raw_observation": step_result.observation},
            ),
        )

        # Get session info
        try:
            server_state = self._client.state()
            session_info = (
                tuple((k, v) for k, v in server_state.items() if isinstance(k, str))
                if isinstance(server_state, dict)
                else ()
            )
        except Exception:
            session_info = ()

        next_hidden = OpenEnvHidden(
            env_name=self._env_name,
            episode_step=next_step,
            last_action=action.text,
            session_info=session_info,
        )

        reward_value = step_result.reward

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                "openenv_reward": reward_value,
                "raw_observation": step_result.observation,
            },
        )

        next_state = State(
            observation=next_observation,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "episode_step": next_step,
                "raw_observation": step_result.observation,
            },
        )

    def compute_rewards(
        self,
        state: State[OpenEnvHidden],
        action: Action,
        next_state: State[OpenEnvHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# ── Tool environment ────────────────────────────────────────────────


class OpenEnvToolEnvironment(BaseToolEnvironment[OpenEnvHidden]):
    """MDP wrapper for OpenEnv MCP tool-enabled environments.

    Uses MCPToolClient to list_tools() and call_tool(). Session-based
    like OpenEnvEnvironment.
    """

    def __init__(
        self,
        client: Any,
        env_name: str,
        max_steps: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._client = client
        self._env_name = env_name
        self._max_steps = max_steps

        self._native_rewards: tuple[RewardFunction, ...] = (
            OpenEnvReward(),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards
        self._tools: tuple[ToolDefinition, ...] = ()
        self._state_tracker = _StateContinuityTracker()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        return self._tools

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_name,
            adapter="openenv",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=False,
            supports_len=False,
            supports_seed=False,
            metadata={"env_name": self._env_name},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def _refresh_tools(self) -> None:
        """Fetch tools from the server and update definitions."""
        try:
            mcp_tools = self._client.list_tools()
            self._tools = _mcp_tools_to_definitions(mcp_tools)
        except Exception as e:
            logger.warning(f"Failed to fetch tools from {self._env_name}: {e}")
            self._tools = ()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[OpenEnvHidden], dict[str, Any]]:
        step_result = self._client.reset()
        self._refresh_tools()

        obs_text = _coerce_observation(step_result.observation)

        hidden = OpenEnvHidden(
            env_name=self._env_name,
            episode_step=0,
            last_action=None,
            session_info=(),
        )

        episode_id = str(uuid.uuid4())
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={},
        )

        observation = Observation(
            prompt=obs_text,
            available_tools=self._tools,
            task=ObservationContent(text=obs_text),
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        info: dict[str, Any] = {
            "env_name": self._env_name,
            "initial_observation": step_result.observation,
            "num_tools": len(self._tools),
        }

        return state, info

    def execute_tools(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute tool calls via the MCP client."""
        results: list[ToolResult] = []

        for call in calls:
            # Validate against known tools
            validation_error = self._validate_tool_call(call)
            if validation_error is not None:
                results.append(validation_error)
                continue

            try:
                output = self._client.call_tool(call.name, **call.arguments)
                results.append(
                    ToolResult(
                        call_id=call.id,
                        tool_name=call.name,
                        output=str(output),
                        status=ToolResultStatus.SUCCESS,
                    )
                )
            except Exception as e:
                results.append(
                    ToolResult.from_error(
                        call_id=call.id,
                        tool_name=call.name,
                        error_message=str(e),
                    )
                )

        return tuple(results)

    def step(
        self,
        state: State[OpenEnvHidden],
        action: Action,
    ) -> StepResult[OpenEnvHidden]:
        self._state_tracker.validate(state, "OpenEnvToolEnvironment")
        next_step = state.hidden.episode_step + 1

        # Execute tool calls if present
        tool_results: tuple[ToolResult, ...] = ()
        if action.tool_calls:
            tool_results = self.execute_tools(action.tool_calls)

        # Also send text action to the server if present
        obs_text = ""
        reward_value = None
        done = False

        if action.text:
            step_result = self._client.step({"text": action.text})
            obs_text = _coerce_observation(step_result.observation)
            reward_value = step_result.reward
            done = step_result.done

        terminated = done
        truncated = not terminated and self._max_steps is not None and next_step >= self._max_steps

        # Build next observation
        if tool_results or action.tool_calls:
            state_text = "\n".join(
                str(tr.output) if tr.is_success else str(tr.error) for tr in tool_results
            )
            next_observation = self._build_next_observation(
                state.observation,
                action,
                tool_results,
                state_content=ObservationContent(text=state_text) if state_text else None,
            )
        else:
            messages = list(state.observation.messages)
            messages.append({"role": "assistant", "content": action.text})
            if obs_text:
                messages.append({"role": "user", "content": obs_text})
            next_observation = Observation(
                prompt=state.observation.prompt,
                messages=tuple(messages),
                available_tools=self._tools,
                task=state.observation.task,
                state=ObservationContent(
                    text=obs_text,
                    data={"raw_observation": step_result.observation},
                )
                if obs_text
                else None,
            )

        next_hidden = OpenEnvHidden(
            env_name=self._env_name,
            episode_step=next_step,
            last_action=action.text,
            session_info=(),
        )

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                "openenv_reward": reward_value,
            },
        )

        next_state = State(
            observation=next_observation,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "episode_step": next_step,
                "tool_results": tool_results,
            },
        )

    def compute_rewards(
        self,
        state: State[OpenEnvHidden],
        action: Action,
        next_state: State[OpenEnvHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# ── Adapter ─────────────────────────────────────────────────────────


class OpenEnvAdapter:
    """Adapter for the OpenEnv library.

    Creates OpenEnvEnvironment (text) or OpenEnvToolEnvironment (MCP tools)
    by connecting to a running server at a given URL.
    """

    @property
    def name(self) -> str:
        return "openenv"

    def _get_openenv(self) -> Any:
        try:
            import openenv

            return openenv
        except ImportError as e:
            raise ImportError(
                "openenv-core is required for OpenEnvAdapter. "
                "Install with: pip install openenv-core"
            ) from e

    def list_environments(self) -> list[str]:
        """OpenEnv environments are server-based; listing is not supported."""
        return []

    def get_environment(
        self,
        name: str,
        base_url: str | None = None,
        use_tools: bool = False,
        max_steps: int | None = None,
        action_format: Callable[[str], Any] | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> OpenEnvEnvironment | OpenEnvToolEnvironment:
        """Create an environment by connecting to a running server.

        Args:
            name: Environment name (for identification only).
            base_url: URL of the running OpenEnv server. Required.
            use_tools: If True, use MCPToolClient for tool support.
            max_steps: Maximum steps per episode.
            action_format: Optional function to format action text for the server.
            extra_rewards: Additional reward functions.
            **kwargs: Passed to the OpenEnv client constructor.

        Returns:
            Connected environment instance.

        Raises:
            ValueError: If base_url is not provided.
        """
        if base_url is None:
            raise ValueError(
                "base_url is required for OpenEnv environments. "
                "Provide the URL of a running OpenEnv server."
            )

        openenv = self._get_openenv()

        if use_tools:
            async_client = openenv.MCPToolClient(base_url=base_url, **kwargs)
            sync_client = async_client.sync()
            return OpenEnvToolEnvironment(
                client=sync_client,
                env_name=name,
                max_steps=max_steps,
                extra_rewards=extra_rewards,
            )
        else:
            async_client = openenv.GenericEnvClient(base_url=base_url, **kwargs)
            sync_client = async_client.sync()
            return OpenEnvEnvironment(
                client=sync_client,
                env_name=name,
                max_steps=max_steps,
                action_format=action_format,
                extra_rewards=extra_rewards,
            )

    def get_native_answer_extractor(self, task_name: str) -> None:
        """OpenEnv environments don't provide ground truth answers."""
        return None

    def get_default_system_prompt(self, name: str) -> None:
        """System prompts are environment-specific."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """No prompt templates."""
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "adapter": self.name,
            "type": "session_based",
            "description": f"OpenEnv environment: {name}",
        }
