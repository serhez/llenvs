"""tau2 adapter — wraps tau2-bench customer service benchmark as MDP environments.

tau2-bench (https://github.com/sierra-research/tau2-bench) is a multi-turn
customer service benchmark evaluating LLM agents across domains (airline,
retail, telecom). It features heavy tool usage with stateful DB-backed tools,
an LLM-powered user simulator, and multi-signal evaluation.

Key design: Tools have internal DB state — tool execution is delegated to
tau2's ``Environment.make_tool_call()`` rather than extracted and called
directly. User simulation is delegated to tau2's ``UserSimulator``.
"""

from __future__ import annotations

import json
import logging
import uuid
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
    ToolDefinition,
    ToolResult,
    oai_tools_to_definitions,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

TAU2_DOMAINS: tuple[str, ...] = ("airline", "retail", "telecom")
TAU2_SPLITS: tuple[str, ...] = ("base", "train", "test")

_STOP_TOKENS: tuple[str, ...] = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


# ── Tool conversion ─────────────────────────────────────────────


def _tau2_tools_to_definitions(
    tools: list[Any],
) -> tuple[ToolDefinition, ...]:
    """Convert tau2 Tool objects to ToolDefinitions.

    Uses ``tool.openai_schema`` for full-fidelity passthrough via
    ``raw_schema``. The flat ``parameters`` tuple is a best-effort parse
    for inspection/display.

    Args:
        tools: List of tau2 Tool objects.

    Returns:
        Tuple of ToolDefinition objects.
    """
    oai_schemas: list[dict[str, Any]] = []
    for tool in tools:
        schema = getattr(tool, "openai_schema", None)
        if schema is None:
            logger.warning(f"Skipping tau2 tool without openai_schema: {tool}")
            continue
        oai_schemas.append(schema)

    return oai_tools_to_definitions(oai_schemas)


# ── Hidden state ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Tau2Hidden:
    """Hidden state for tau2 environments.

    Attributes:
        task_index: Index into the task list.
        task_id: The tau2 task identifier.
        domain: The tau2 domain (airline, retail, telecom).
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        messages: Conversation message history (frozen tuple).
        termination_reason: Why the episode ended (if terminal).
        reward_info: tau2 RewardInfo object (stored at terminal step).
    """

    task_index: int
    task_id: str
    domain: str
    episode_step: int = 0
    last_action: str | None = None
    messages: tuple[dict[str, Any], ...] = ()
    termination_reason: str | None = None
    reward_info: Any = None


# ── Reward functions ─────────────────────────────────────────────


@dataclass
class Tau2Reward:
    """Primary OUTCOME reward using tau2's aggregate evaluation score.

    Non-terminal steps return None reward (STEP type).
    Terminal steps read the overall ``reward`` from tau2's ``RewardInfo``.
    """

    _name: str = "tau2"

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

        reward_info = getattr(next_state.hidden, "reward_info", None)
        if reward_info is None:
            return Signal(
                name=self.name,
                reward_type=RewardType.OUTCOME,
                reward=0.0,
                metadata={"is_terminal": True, "reason": "no_reward_info"},
            )

        score = getattr(reward_info, "reward", 0.0)
        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=float(score),
            metadata={"is_terminal": True},
        )


@dataclass
class Tau2DetailedRewards:
    """Optional reward function emitting per-criterion breakdown.

    Emits a single OUTCOME signal with metadata containing per-criterion
    scores (db_reward, action_reward, communicate_reward, nl_reward).
    Available via ``extra_rewards`` for fine-grained analysis.
    """

    _name: str = "tau2_detailed"

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

        reward_info = getattr(next_state.hidden, "reward_info", None)
        if reward_info is None:
            return Signal(
                name=self.name,
                reward_type=RewardType.OUTCOME,
                reward=0.0,
                metadata={"is_terminal": True, "reason": "no_reward_info"},
            )

        # Extract per-criterion breakdown
        metadata: dict[str, Any] = {"is_terminal": True}

        db_check = getattr(reward_info, "db_check", None)
        if db_check is not None:
            metadata["db_reward"] = getattr(db_check, "db_reward", None)
            metadata["db_match"] = getattr(db_check, "db_match", None)

        action_checks = getattr(reward_info, "action_checks", None)
        if action_checks:
            metadata["action_checks"] = len(action_checks)
            metadata["action_reward"] = sum(
                getattr(c, "action_reward", 0.0) for c in action_checks
            ) / max(len(action_checks), 1)

        communicate_checks = getattr(reward_info, "communicate_checks", None)
        if communicate_checks:
            metadata["communicate_checks"] = len(communicate_checks)
            met = sum(1 for c in communicate_checks if getattr(c, "met", False))
            metadata["communicate_reward"] = met / max(len(communicate_checks), 1)

        nl_assertions = getattr(reward_info, "nl_assertions", None)
        if nl_assertions:
            metadata["nl_assertions"] = len(nl_assertions)
            met = sum(1 for a in nl_assertions if getattr(a, "met", False))
            metadata["nl_reward"] = met / max(len(nl_assertions), 1)

        score = getattr(reward_info, "reward", 0.0)
        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=float(score),
            metadata=metadata,
        )


# ── Environment ──────────────────────────────────────────────────


def _contains_stop_token(text: str | None) -> bool:
    """Check if text contains any tau2 stop token."""
    if text is None:
        return False
    return any(token in text for token in _STOP_TOKENS)


def _result_to_str(result: Any) -> str:
    """Convert a tau2 tool result to string."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result)
    return str(result)


class Tau2Environment(BaseToolEnvironment[Tau2Hidden]):
    """MDP wrapper for tau2-bench customer service benchmark.

    Each task provides a set of tools backed by domain databases. Tool
    execution is delegated to tau2's ``Environment.make_tool_call()``.
    User simulation (when not in solo mode) is delegated to tau2's
    ``UserSimulator``.
    """

    def __init__(
        self,
        domain: str,
        tasks: list[Any],
        tau2_env: Any,
        max_steps: int | None = None,
        solo_mode: bool = False,
        user_simulator: Any | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._domain = domain
        self._tasks = tasks
        self._tau2_env = tau2_env
        self._max_steps = max_steps
        self._solo_mode = solo_mode
        self._user_simulator = user_simulator
        self._user_state: Any = None
        self._executor = None  # Not used — tau2 handles execution

        self._native_rewards: tuple[RewardFunction, ...] = (
            Tau2Reward(),
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
            name=f"tau2:{self._domain}",
            adapter="tau2",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={"domain": self._domain},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def __len__(self) -> int:
        return len(self._tasks)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[Tau2Hidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        task = self._tasks[task_index]

        # Initialize tau2 environment state from task
        initial_state = getattr(task, "initial_state", None)
        if initial_state is not None:
            try:
                self._tau2_env.set_state(
                    initialization_data=getattr(initial_state, "initialization_data", None),
                    initialization_actions=getattr(initial_state, "initialization_actions", None),
                    message_history=getattr(initial_state, "message_history", None),
                )
            except Exception as e:
                logger.warning(f"Failed to set tau2 initial state: {e}")

        # Get tools from tau2 environment
        raw_tools = self._tau2_env.get_tools()
        self._tools = _tau2_tools_to_definitions(raw_tools)

        # Initialize user simulator state
        if self._user_simulator is not None:
            msg_history = None
            if initial_state is not None:
                msg_history = getattr(initial_state, "message_history", None)
            self._user_state = self._user_simulator.get_init_state(message_history=msg_history)

        # Build initial observation
        if self._solo_mode:
            ticket = getattr(task, "ticket", None) or ""
            prompt = ticket
        else:
            # Use user scenario instructions as the initial observation
            user_scenario = getattr(task, "user_scenario", None)
            if user_scenario is not None:
                instructions = getattr(user_scenario, "instructions", "")
                prompt = str(instructions) if instructions else ""
            else:
                prompt = ""

        hidden = Tau2Hidden(
            task_index=task_index,
            task_id=getattr(task, "id", str(task_index)),
            domain=self._domain,
            episode_step=0,
            last_action=None,
            messages=(),
            termination_reason=None,
            reward_info=None,
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
            task=ObservationContent(text=prompt),
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        info: dict[str, Any] = {
            "task_index": task_index,
            "task_id": hidden.task_id,
            "domain": self._domain,
            "num_tools": len(self._tools),
        }

        return state, info

    def step(
        self,
        state: State[Tau2Hidden],
        action: Action,
    ) -> StepResult[Tau2Hidden]:
        self._state_tracker.validate(state, "Tau2Environment")

        next_step = state.hidden.episode_step + 1
        terminated = False
        truncated = False
        termination_reason: str | None = None
        tool_results: list[ToolResult] = []
        messages = list(state.hidden.messages)

        if action.has_tool_calls:
            # Execute tool calls through tau2 environment
            for tc in action.tool_calls:
                # Validate against known tools
                validation_error = self._validate_tool_call(tc)
                if validation_error is not None:
                    tool_results.append(validation_error)
                    continue

                try:
                    result = self._tau2_env.make_tool_call(
                        tc.name, requestor="assistant", **tc.arguments
                    )
                    tool_results.append(
                        ToolResult.success(
                            call_id=tc.id,
                            tool_name=tc.name,
                            output=_result_to_str(result),
                        )
                    )
                except Exception as e:
                    logger.warning(f"tau2 tool call {tc.name} failed: {e}")
                    tool_results.append(
                        ToolResult.from_error(
                            call_id=tc.id,
                            tool_name=tc.name,
                            error_message=str(e),
                        )
                    )

            # Record tool calls in message history
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in action.tool_calls
                    ],
                }
            )
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.call_id,
                        "name": tr.tool_name,
                        "content": str(tr.output) if tr.is_success else tr.error,
                    }
                )

        elif action.text is not None:
            if self._solo_mode:
                # Solo mode: check for stop token
                if _contains_stop_token(action.text):
                    terminated = True
                    termination_reason = "agent_stop"
                # In solo mode, text without stop is just a no-op (agent thinking)
                messages.append({"role": "assistant", "content": action.text})
            else:
                # Multi-turn mode: forward text to user simulator
                messages.append({"role": "assistant", "content": action.text})

                if self._user_simulator is not None:
                    # Create a mock message for the user simulator
                    assistant_msg = type(
                        "_Msg",
                        (),
                        {"content": action.text, "role": "assistant", "tool_calls": None},
                    )()
                    user_msg, self._user_state = self._user_simulator.generate_next_message(
                        message=assistant_msg, state=self._user_state
                    )

                    user_content = getattr(user_msg, "content", str(user_msg))
                    messages.append({"role": "user", "content": user_content})

                    # Check for user stop
                    if self._user_simulator.is_stop(user_msg):
                        terminated = True
                        termination_reason = "user_stop"

        # Check max_steps truncation
        if not terminated and self._max_steps is not None and next_step >= self._max_steps:
            truncated = True

        # Build next observation
        if tool_results:
            state_text = "\n".join(
                str(tr.output) if tr.is_success else str(tr.error) for tr in tool_results
            )
            next_obs = self._build_next_observation(
                current_obs=state.observation,
                action=action,
                tool_results=tuple(tool_results),
                state_content=ObservationContent(text=state_text) if state_text else None,
            )
        else:
            # Text-based step: build observation from messages
            obs_messages = list(state.observation.messages)
            if action.text is not None:
                obs_messages.append({"role": "assistant", "content": action.text})
            # Add user response if present
            if (
                not self._solo_mode
                and self._user_simulator is not None
                and action.text is not None
                and not terminated
            ):
                # User response already in `messages`, extract last user msg
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    obs_messages.append(user_msgs[-1])
            elif (
                not self._solo_mode
                and self._user_simulator is not None
                and action.text is not None
                and terminated
            ):
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    obs_messages.append(user_msgs[-1])

            # Derive state from latest user message or agent text
            state_text = ""
            user_msgs = [m for m in obs_messages if m.get("role") == "user"]
            if user_msgs:
                state_text = str(user_msgs[-1].get("content", ""))
            elif action.text:
                state_text = action.text

            next_obs = Observation(
                prompt=state.observation.prompt,
                messages=tuple(obs_messages),
                available_tools=self._tools,
                task=state.observation.task,
                state=ObservationContent(text=state_text) if state_text else None,
            )

        next_hidden = Tau2Hidden(
            task_index=state.hidden.task_index,
            task_id=state.hidden.task_id,
            domain=self._domain,
            episode_step=next_step,
            last_action=action.text,
            messages=tuple(messages),
            termination_reason=termination_reason,
            reward_info=None,
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
                "termination_reason": termination_reason,
            },
        )

    def compute_rewards(
        self,
        state: State[Tau2Hidden],
        action: Action,
        next_state: State[Tau2Hidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# ── Adapter ──────────────────────────────────────────────────────


class Tau2Adapter:
    """Adapter for the tau2-bench customer service benchmark.

    Wraps tau2's multi-turn, tool-using customer service environments
    across airline, retail, and telecom domains.
    """

    @property
    def name(self) -> str:
        return "tau2"

    def _get_tau2(self) -> Any:
        try:
            import tau2

            return tau2
        except ImportError as e:
            raise ImportError(
                "tau2 is required for Tau2Adapter. Install with: pip install tau2"
            ) from e

    @staticmethod
    def _parse_domain(name: str) -> str:
        """Extract domain from environment name like 'tau2:airline' or 'tau2:airline:base'."""
        parts = name.split(":")
        if len(parts) >= 2:
            return parts[1]
        return ""

    @staticmethod
    def _parse_split(name: str) -> str | None:
        """Extract split from environment name like 'tau2:airline:base'."""
        parts = name.split(":")
        if len(parts) >= 3:
            return parts[2]
        return None

    def list_environments(self) -> list[str]:
        envs: list[str] = []
        for domain in TAU2_DOMAINS:
            envs.append(f"tau2:{domain}")
            for split in TAU2_SPLITS:
                envs.append(f"tau2:{domain}:{split}")
        return envs

    def get_environment(
        self,
        name: str,
        tasks: list[Any] | None = None,
        task_split: str | None = None,
        tau2_env: Any | None = None,
        max_steps: int | None = None,
        solo_mode: bool = False,
        user_simulator: Any | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> Tau2Environment:
        """Create a tau2 environment.

        Args:
            name: Environment name (e.g., "tau2:airline", "tau2:retail:base").
            tasks: Pre-loaded list of tau2 Task objects.
            task_split: Task split name (base, train, test). Overridden by name.
            tau2_env: Pre-created tau2 Environment. If None, created from registry.
            max_steps: Maximum steps per episode.
            solo_mode: If True, no user simulator (agent uses tools only).
            user_simulator: Pre-created UserSimulator. If None, created from tau2.
            extra_rewards: Additional reward functions.
            **kwargs: Passed to tau2 environment/task constructors.

        Returns:
            Tau2Environment wrapping the domain.

        Raises:
            ValueError: If tasks cannot be loaded.
        """
        domain = self._parse_domain(name)
        split = self._parse_split(name) or task_split

        # Load tasks if not provided
        if tasks is None:
            tau2 = self._get_tau2()
            try:
                tasks_loader = tau2.registry.get_tasks_loader(domain)
                tasks = tasks_loader(task_split_name=split)
            except Exception as e:
                raise ValueError(
                    f"Could not load tasks for tau2:{domain}. "
                    f"Provide tasks= directly or ensure tau2 is properly installed. "
                    f"Error: {e}"
                ) from e

        # Create tau2 environment if not provided
        if tau2_env is None:
            tau2 = self._get_tau2()
            try:
                env_constructor = tau2.registry.get_env_constructor(domain)
                tau2_env = env_constructor(solo_mode=solo_mode)
            except Exception as e:
                raise ValueError(
                    f"Could not create tau2 environment for domain '{domain}'. Error: {e}"
                ) from e

        return Tau2Environment(
            domain=domain,
            tasks=tasks,
            tau2_env=tau2_env,
            max_steps=max_steps,
            solo_mode=solo_mode,
            user_simulator=user_simulator,
            extra_rewards=extra_rewards,
        )

    def get_default_system_prompt(self, name: str, tau2_env: Any | None = None) -> str | None:
        """Get the domain policy as the default system prompt.

        Args:
            name: Environment name.
            tau2_env: tau2 Environment to read policy from.

        Returns:
            Policy string or None.
        """
        if tau2_env is not None:
            policy = getattr(tau2_env, "policy", None)
            if policy is None:
                get_policy = getattr(tau2_env, "get_policy", None)
                if get_policy is not None:
                    policy = get_policy()
            return str(policy) if policy else None
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        domain = self._parse_domain(name)
        return {
            "name": name,
            "adapter": self.name,
            "domain": domain,
            "description": f"tau2-bench customer service benchmark ({domain})",
            "domains": list(TAU2_DOMAINS),
        }
