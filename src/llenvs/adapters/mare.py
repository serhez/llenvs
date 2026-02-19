"""MARE adapter — wraps Meta Agents Research Environments as MDP environments.

MARE (Meta Agents Research Environments) is Facebook Research's event-driven
simulation platform for evaluating AI agents. It powers the Gaia2 benchmark
with 800 scenarios across 10 "universes" featuring 5 simulated apps (email,
calendar, contacts, shopping, file system) exposing ~101 tools.

Key design: ARE tools have internal state and the environment runs an event
loop with asynchronous notifications. Tool execution is delegated to ARE's
``tool.forward()`` rather than extracted and wrapped. We extend
BaseToolEnvironment for tool infrastructure but handle execution internally.
"""

from __future__ import annotations

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
    ToolParameter,
    ToolParameterType,
    ToolResult,
)

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────

MARE_CAPABILITIES: tuple[str, ...] = (
    "execution",
    "search",
    "ambiguity",
    "adaptability",
    "time_sensitivity",
)

# Tools whose invocations count as "write actions" for validation scoring.
_WRITE_ACTION_PREFIXES: tuple[str, ...] = (
    "send_",
    "create_",
    "delete_",
    "update_",
    "add_",
    "remove_",
    "set_",
    "move_",
    "cancel_",
    "schedule_",
    "reply_",
    "forward_",
    "archive_",
    "purchase_",
    "place_order",
)

# ── Type mapping ─────────────────────────────────────────────────

_TYPE_MAP: dict[str, ToolParameterType] = {
    "string": ToolParameterType.STRING,
    "str": ToolParameterType.STRING,
    "text": ToolParameterType.STRING,
    "number": ToolParameterType.NUMBER,
    "float": ToolParameterType.NUMBER,
    "integer": ToolParameterType.INTEGER,
    "int": ToolParameterType.INTEGER,
    "boolean": ToolParameterType.BOOLEAN,
    "bool": ToolParameterType.BOOLEAN,
    "array": ToolParameterType.ARRAY,
    "list": ToolParameterType.ARRAY,
    "object": ToolParameterType.OBJECT,
    "dict": ToolParameterType.OBJECT,
}


# ── Tool conversion ──────────────────────────────────────────────


def _mare_tools_to_definitions(
    tools: list[Any],
) -> tuple[ToolDefinition, ...]:
    """Convert ARE Tool objects to ToolDefinitions.

    ARE Tools have ``name``, ``description``, ``inputs`` (dict of param specs
    with type strings), and ``output_type``.

    Args:
        tools: List of ARE Tool objects.

    Returns:
        Tuple of ToolDefinition objects.
    """
    definitions: list[ToolDefinition] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if name is None:
            logger.warning(f"Skipping ARE tool without name: {tool}")
            continue

        description = getattr(tool, "description", "") or ""
        inputs = getattr(tool, "inputs", {}) or {}

        parameters: list[ToolParameter] = []
        for param_name, param_spec in inputs.items():
            if isinstance(param_spec, dict):
                type_str = param_spec.get("type", "string")
                param_desc = param_spec.get("description", "")
            else:
                type_str = "string"
                param_desc = str(param_spec) if param_spec else ""

            param_type = _TYPE_MAP.get(type_str.lower(), ToolParameterType.STRING)
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_desc,
                    required=True,
                )
            )

        definitions.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=tuple(parameters),
            )
        )

    return tuple(definitions)


# ── Hidden state ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MAREHidden:
    """Hidden state for MARE environments.

    Attributes:
        task_index: Index into the scenario list.
        scenario_id: The ARE scenario identifier.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        notifications: Pending notifications from the environment.
        write_actions: Tracked write actions for validation scoring.
    """

    task_index: int
    scenario_id: str
    episode_step: int = 0
    last_action: str | None = None
    notifications: tuple[str, ...] = ()
    write_actions: tuple[dict[str, Any], ...] = ()


# ── Reward function ──────────────────────────────────────────────


@dataclass
class MAREReward:
    """Reward function for MARE environments.

    Non-terminal steps return None reward (STEP type).
    Terminal steps run ARE scenario validation against oracle annotations.
    """

    validator: Any | None = None
    _name: str = "mare"

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
        is_terminal = next_state.metadata.is_terminal

        if not is_terminal:
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=None,
                metadata={"is_terminal": False},
            )

        # Terminal: run validation
        write_actions = getattr(next_state.hidden, "write_actions", ())

        if self.validator is not None:
            try:
                result = _run_async(self.validator(list(write_actions)))
                score = result.get("score", 0.0) if isinstance(result, dict) else 0.0
                metadata = result if isinstance(result, dict) else {"raw": result}
            except Exception as e:
                logger.warning(f"MARE validation failed: {e}")
                score = 0.0
                metadata = {"error": str(e)}
        else:
            score = 0.0
            metadata = {"reason": "no_validator"}

        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=score,
            metadata={"is_terminal": True, **metadata},
        )


# ── Environment ──────────────────────────────────────────────────


def _is_write_action(tool_name: str) -> bool:
    """Check if a tool call is a write action based on its name."""
    lower = tool_name.lower()
    return any(lower.startswith(prefix) for prefix in _WRITE_ACTION_PREFIXES)


class MAREEnvironment(BaseToolEnvironment[MAREHidden]):
    """MDP wrapper for MARE (Meta Agents Research Environments).

    Each scenario provides a set of tools across simulated apps. The
    environment runs an event loop that generates notifications. Tool
    execution is delegated to ARE's ``tool.forward()`` since tools
    have internal state.
    """

    def __init__(
        self,
        scenarios: list[Any],
        max_steps: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        write_action_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        self._scenarios = scenarios
        self._max_steps = max_steps
        self._active_are_env: Any | None = None
        self._active_scenario: Any | None = None
        self._tool_map: dict[str, Any] = {}
        self._executor = None  # Not used — ARE handles execution
        self._write_prefixes = write_action_prefixes or _WRITE_ACTION_PREFIXES

        self._native_rewards: tuple[RewardFunction, ...] = (
            MAREReward(),
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
            name="mare",
            adapter="mare",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=True,
            pure_step=False,
            metadata={},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def __len__(self) -> int:
        return len(self._scenarios)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[MAREHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._scenarios):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._scenarios)})")

        scenario = self._scenarios[task_index]
        self._active_scenario = scenario

        # Initialize the scenario (set up apps, load data)
        _run_async(scenario.initialize())

        # Create and start the ARE environment event loop
        are_env = self._create_are_environment(scenario)
        self._active_are_env = are_env
        _run_async(are_env.start())

        # Get tools from the scenario
        raw_tools = scenario.get_user_tools()
        self._tools = _mare_tools_to_definitions(raw_tools)
        self._tool_map = {getattr(t, "name", ""): t for t in raw_tools}

        # Build prompt from scenario
        prompt_parts: list[str] = []
        scenario_prompt = getattr(scenario, "prompt", "")
        if scenario_prompt:
            prompt_parts.append(str(scenario_prompt))
        task_desc = getattr(scenario, "task_description", "")
        if task_desc:
            prompt_parts.append(str(task_desc))
        prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

        hidden = MAREHidden(
            task_index=task_index,
            scenario_id=getattr(scenario, "id", str(task_index)),
            episode_step=0,
            last_action=None,
            notifications=(),
            write_actions=(),
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

        # Set up the validator on the native reward
        if hasattr(scenario, "validate"):
            for rf in self._native_rewards:
                if isinstance(rf, MAREReward):
                    rf.validator = scenario.validate
                    break

        info: dict[str, Any] = {
            "task_index": task_index,
            "scenario_id": hidden.scenario_id,
            "num_tools": len(self._tools),
        }

        return state, info

    def _create_are_environment(self, scenario: Any) -> Any:
        """Create an ARE Environment for a scenario.

        Tries to use the ARE library's Environment class. Falls back to
        a simple mock-compatible object if ARE isn't installed (for testing).
        """
        try:
            from meta_agents_research_environments.core import Environment as AREEnvironment

            are_env = AREEnvironment()
            # Register apps from the scenario
            for app in scenario.get_apps():
                are_env.register_app(app)
            return are_env
        except ImportError:
            # For testing without ARE installed — use the scenario's own
            # environment if it provides one, or create a minimal stub
            if hasattr(scenario, "_are_env"):
                return scenario._are_env

            # Minimal stub for testing
            class _MinimalAREEnv:
                def __init__(self):
                    self._notifications: list[Any] = []

                async def start(self):
                    pass

                async def stop(self):
                    pass

                async def tick(self):
                    pass

                def get_pending_notifications(self):
                    notifs = list(self._notifications)
                    self._notifications.clear()
                    return notifs

                def add_notification(self, notif: Any):
                    self._notifications.append(notif)

            return _MinimalAREEnv()

    def step(
        self,
        state: State[MAREHidden],
        action: Action,
    ) -> StepResult[MAREHidden]:
        self._state_tracker.validate(state, "MAREEnvironment")

        if self._active_are_env is None:
            raise RuntimeError("No active ARE environment. Call reset() first.")

        next_step = state.hidden.episode_step + 1
        terminated = False
        truncated = False
        tool_results: list[ToolResult] = []
        new_write_actions = list(state.hidden.write_actions)

        if action.has_tool_calls:
            # Execute each tool call through ARE
            for tc in action.tool_calls:
                are_tool = self._tool_map.get(tc.name)
                if are_tool is None:
                    tool_results.append(
                        ToolResult.from_error(
                            call_id=tc.id,
                            tool_name=tc.name,
                            error_message=f"Unknown tool: {tc.name}",
                        )
                    )
                    continue

                try:
                    result = are_tool.forward(**tc.arguments)
                    # Handle async tools
                    import asyncio

                    if asyncio.iscoroutine(result):
                        result = _run_async(result)

                    tool_results.append(
                        ToolResult.success(
                            call_id=tc.id,
                            tool_name=tc.name,
                            output=str(result),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Tool call {tc.name} failed: {e}")
                    tool_results.append(
                        ToolResult.from_error(
                            call_id=tc.id,
                            tool_name=tc.name,
                            error_message=str(e),
                        )
                    )

                # Track write actions
                if _is_write_action(tc.name):
                    new_write_actions.append({"tool": tc.name, "arguments": tc.arguments})
        else:
            # Text-only action = agent signals completion
            terminated = True

        # Tick the environment (advance simulation, process events)
        try:
            _run_async(self._active_are_env.tick())
        except Exception as e:
            logger.warning(f"ARE environment tick failed: {e}")

        # Collect notifications
        notifications: list[str] = []
        try:
            pending = self._active_are_env.get_pending_notifications()
            for notif in pending:
                notifications.append(str(notif))
        except Exception:
            pass

        # Check max_steps truncation
        if self._max_steps is not None and next_step >= self._max_steps:
            truncated = True

        # Build next observation
        next_obs = self._build_next_observation(
            current_obs=state.observation,
            action=action,
            tool_results=tuple(tool_results),
        )

        # Add notifications as system messages
        if notifications:
            messages = list(next_obs.messages)
            for notif in notifications:
                messages.append({"role": "system", "content": f"[Notification] {notif}"})
            next_obs = Observation(
                prompt=next_obs.prompt,
                messages=tuple(messages),
                tool_results=next_obs.tool_results,
                available_tools=next_obs.available_tools,
            )

        next_hidden = MAREHidden(
            task_index=state.hidden.task_index,
            scenario_id=state.hidden.scenario_id,
            episode_step=next_step,
            last_action=action.text,
            notifications=tuple(notifications),
            write_actions=tuple(new_write_actions),
        )

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                **state.metadata.info,
                "episode_step": next_step,
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
            terminated=terminated,
            truncated=truncated,
            info={
                "tool_results": tuple(tool_results),
                "episode_step": next_step,
                "notifications": tuple(notifications),
            },
        )

    def compute_rewards(
        self,
        state: State[MAREHidden],
        action: Action,
        next_state: State[MAREHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Stop the active ARE environment event loop."""
        if self._active_are_env is not None:
            try:
                stop = getattr(self._active_are_env, "stop", None)
                if stop is not None:
                    import asyncio

                    result = stop()
                    if asyncio.iscoroutine(result):
                        _run_async(result)
            except Exception as e:
                logger.warning(f"Error closing ARE environment: {e}")
            finally:
                self._active_are_env = None
                self._active_scenario = None
                self._tool_map.clear()


# ── Adapter ──────────────────────────────────────────────────────


class MAREAdapter:
    """Adapter for Meta Agents Research Environments (ARE).

    Routes environment creation through scenario lists. Scenarios can
    be pre-loaded or loaded via ARE's built-in scenario loaders.
    """

    @property
    def name(self) -> str:
        return "mare"

    def _get_mare(self) -> Any:
        try:
            import meta_agents_research_environments

            return meta_agents_research_environments
        except ImportError as e:
            raise ImportError(
                "meta-agents-research-environments is required for MAREAdapter. "
                "Install with: pip install git+https://github.com/facebookresearch/meta-agents-research-environments.git"
            ) from e

    def list_environments(self) -> list[str]:
        envs = ["mare"]
        for cap in MARE_CAPABILITIES:
            envs.append(f"mare:{cap}")
        return envs

    def get_environment(
        self,
        name: str,
        scenarios: list[Any] | None = None,
        scenario_loader: Any | None = None,
        max_steps: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        capability: str | None = None,
        **kwargs: Any,
    ) -> MAREEnvironment:
        """Create a MARE environment.

        Args:
            name: Environment name. Use "mare:execution" to filter by capability.
            scenarios: Pre-loaded list of Scenario objects.
            scenario_loader: Callable that returns scenarios.
            max_steps: Maximum steps per episode.
            extra_rewards: Additional reward functions.
            capability: Filter scenarios by capability type.
            **kwargs: Passed to scenario loader.

        Returns:
            MAREEnvironment wrapping the scenarios.

        Raises:
            ValueError: If neither scenarios nor scenario_loader is provided.
        """
        self._get_mare()

        # Load scenarios
        if scenarios is None and scenario_loader is not None:
            scenarios = scenario_loader(**kwargs)
        elif scenarios is None:
            raise ValueError(
                "Either scenarios= or scenario_loader= must be provided for MARE environments."
            )

        # Parse capability from name
        parsed_capability = capability
        if ":" in name:
            _, parsed_capability = name.split(":", 1)

        # Filter by capability if specified (scenario metadata filtering)
        if parsed_capability is not None:
            cap_lower = parsed_capability.lower()
            filtered = [
                s
                for s in scenarios
                if cap_lower in str(getattr(s, "capability", getattr(s, "category", ""))).lower()
            ]
            if filtered:
                scenarios = filtered

        return MAREEnvironment(
            scenarios=scenarios,
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
        return {
            "name": name,
            "adapter": self.name,
            "description": "Meta Agents Research Environments (ARE / Gaia2)",
            "capabilities": list(MARE_CAPABILITIES),
        }
