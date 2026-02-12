"""Verifiers adapter — wraps verifiers environments as MDP environments.

Supports SingleTurnEnv (dataset-backed, single response) and ToolEnv
(multi-turn with tool calling). SandboxEnv/PythonEnv are not supported
(require Prime Sandboxes cloud infrastructure).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

import uuid

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import (
    RewardBundle,
    RewardSignal,
    RewardType,
    RewardFunction,
)
from llenvs.core.environment import Environment, StepResult, EnvironmentSpec, _StateContinuityTracker
from llenvs.core.extraction import AnswerExtractor
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
    ToolExecutor,
)
from llenvs.core.tool_environment import BaseToolEnvironment

logger = logging.getLogger(__name__)


# ── Type mapping ────────────────────────────────────────────────────

_OAI_TYPE_MAP: dict[str, ToolParameterType] = {
    "string": ToolParameterType.STRING,
    "integer": ToolParameterType.INTEGER,
    "number": ToolParameterType.NUMBER,
    "boolean": ToolParameterType.BOOLEAN,
    "array": ToolParameterType.ARRAY,
    "object": ToolParameterType.OBJECT,
}


def _oai_tools_to_definitions(
    oai_tools: list[dict[str, Any]],
) -> tuple[ToolDefinition, ...]:
    """Convert OpenAI-format tool schemas to ToolDefinitions.

    Args:
        oai_tools: List of OpenAI tool schema dicts.

    Returns:
        Tuple of ToolDefinition objects.
    """
    definitions: list[ToolDefinition] = []

    for tool in oai_tools:
        func = tool.get("function", tool)
        name = func["name"]
        description = func.get("description", "")
        params_schema = func.get("parameters", {})
        properties = params_schema.get("properties", {})
        required_names = set(params_schema.get("required", []))

        parameters: list[ToolParameter] = []
        for param_name, param_schema in properties.items():
            param_type_str = param_schema.get("type", "string")
            param_type = _OAI_TYPE_MAP.get(param_type_str, ToolParameterType.STRING)
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_schema.get("description", ""),
                    required=param_name in required_names,
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


# ── Hidden states ───────────────────────────────────────────────────


@dataclass(frozen=True)
class VerifiersHidden:
    """Hidden state for single-turn verifiers environments.

    Attributes:
        env_id: The verifiers environment identifier.
        task_index: Index into the dataset.
        expected_answer: Ground truth answer string, or None.
        dataset_item: Frozen representation of the dataset row.
    """

    env_id: str
    task_index: int
    expected_answer: str | None
    dataset_item: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class VerifiersToolHidden:
    """Hidden state for tool-enabled verifiers environments.

    Attributes:
        env_id: The verifiers environment identifier.
        task_index: Index into the dataset.
        expected_answer: Ground truth answer string, or None.
        dataset_item: Frozen representation of the dataset row.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
    """

    env_id: str
    task_index: int
    expected_answer: str | None
    dataset_item: tuple[tuple[str, Any], ...]
    episode_step: int
    last_action: str | None


# ── Rubric reward function ──────────────────────────────────────────


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already in an async context — run in a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _build_reward_kwargs(
    func: Any,
    action_text: str,
    hidden: VerifiersHidden | VerifiersToolHidden,
) -> dict[str, Any]:
    """Build keyword arguments for a verifiers reward function.

    Inspects the function signature and provides matching arguments.
    """
    dataset_dict = dict(hidden.dataset_item)
    sig = inspect.signature(func)
    available: dict[str, Any] = {
        "completion": [{"role": "assistant", "content": action_text}],
        "answer": hidden.expected_answer or "",
        "prompt": dataset_dict.get("prompt", []),
        "info": dataset_dict.get("info", {}),
        "state": {},
        "task": dataset_dict.get("task", ""),
    }
    return {
        name: available[name]
        for name in sig.parameters
        if name in available
    }


@dataclass
class VerifiersRubricReward:
    """Reward function that evaluates using a verifiers Rubric.

    Calls each rubric function with appropriate keyword arguments,
    applies weights, and returns the weighted sum as a single OUTCOME signal.
    """

    _name: str = "verifiers_rubric"
    _reward_type: RewardType = RewardType.OUTCOME

    def __init__(self, rubric: Any, env_id: str) -> None:
        self._rubric = rubric
        self._env_id = env_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> RewardSignal:
        """Compute reward by calling rubric functions."""
        try:
            total = 0.0
            func_results: dict[str, float] = {}

            for func, weight in zip(self._rubric.funcs, self._rubric.weights):
                kwargs = _build_reward_kwargs(func, action.text, state.hidden)

                result = func(**kwargs)
                # Handle async functions
                if asyncio.iscoroutine(result):
                    result = _run_async(result)

                score = float(result)
                func_name = getattr(func, "__name__", str(func))
                func_results[func_name] = score
                total += score * weight

            return RewardSignal(
                value=total,
                name=self.name,
                reward_type=self.reward_type,
                metadata={
                    "func_scores": func_results,
                    "env_id": self._env_id,
                },
            )
        except Exception as e:
            logger.warning(f"Rubric evaluation failed for {self._env_id}: {e}")
            return RewardSignal(
                value=0.0,
                name=self.name,
                reward_type=self.reward_type,
                metadata={"error": str(e)},
            )


# ── Single-turn environment ────────────────────────────────────────


class VerifiersSingleTurnEnvironment:
    """MDP wrapper for verifiers SingleTurnEnv.

    Converts verifiers' dataset + rubric interface to the Environment protocol.
    Each episode corresponds to a single dataset row.
    """

    def __init__(
        self,
        vf_env: Any,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._vf_env = vf_env
        self._answer_extractor = answer_extractor
        self._dataset = vf_env.dataset
        self._env_id = getattr(vf_env, "env_id", "verifiers") or "verifiers"

        self._native_rewards: tuple[RewardFunction, ...] = (
            VerifiersRubricReward(rubric=vf_env.rubric, env_id=self._env_id),
        )
        self._extra_rewards = extra_rewards

    @property
    def system_prompt(self) -> str | None:
        """The environment's system prompt."""
        return self._vf_env.system_prompt

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_id,
            adapter="verifiers",
            max_steps=1,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=False,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            supports_branching=True,
            metadata={
                "dataset_size": len(self._dataset),
                "env_id": self._env_id,
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
    ) -> tuple[State[VerifiersHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._dataset):
            raise ValueError(
                f"task_index {task_index} out of bounds [0, {len(self._dataset)})"
            )

        row = self._dataset[task_index]

        # Extract prompt from dataset row
        prompt_messages = row.get("prompt", [])
        user_prompt = ""
        row_system_prompt = None

        for msg in prompt_messages:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
            elif msg.get("role") == "system":
                row_system_prompt = msg.get("content")

        # Fall back to question column or env system prompt
        if not user_prompt:
            user_prompt = row.get("question", "")

        expected_answer = row.get("answer")

        # Freeze the dataset row for hidden state
        dataset_item = tuple(
            (k, v) for k, v in row.items()
            if isinstance(k, str)
        )

        hidden = VerifiersHidden(
            env_id=self._env_id,
            task_index=task_index,
            expected_answer=expected_answer,
            dataset_item=dataset_item,
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        observation = Observation(prompt=user_prompt)
        state = State(observation=observation, hidden=hidden, metadata=metadata)

        info: dict[str, Any] = {
            "task_index": task_index,
            "env_id": self._env_id,
            "system_prompt": row_system_prompt or self._vf_env.system_prompt,
        }

        return state, info

    def step(
        self,
        state: State[VerifiersHidden],
        action: Action,
    ) -> StepResult[VerifiersHidden]:
        rewards = self.compute_rewards(state, action, state)

        next_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=True,
            info={**state.metadata.info, "response": action.text},
        )
        next_state = State(
            observation=state.observation,
            hidden=state.hidden,
            metadata=next_metadata,
        )

        info: dict[str, Any] = {
            "expected_answer": state.hidden.expected_answer,
        }

        if self._answer_extractor is not None:
            extracted, extraction_meta = self._answer_extractor.extract(action.text)
            info["extracted_answer"] = extracted
            info["extraction_metadata"] = extraction_meta

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=True,
            truncated=False,
            info=info,
        )

    def compute_rewards(
        self,
        state: State[VerifiersHidden],
        action: Action,
        next_state: State[VerifiersHidden],
    ) -> RewardBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return RewardBundle(signals=tuple(signals))


# ── Tool executor ───────────────────────────────────────────────────


class VerifiersToolExecutor:
    """Executes tool calls by delegating to verifiers' Python callables."""

    def __init__(self, tool_map: dict[str, Any]) -> None:
        self._tool_map = tool_map

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name not in self._tool_map:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Unknown tool: {call.name}",
                status=ToolResultStatus.INVALID_TOOL,
            )

        func = self._tool_map[call.name]
        try:
            result = func(**call.arguments)
            if asyncio.iscoroutine(result):
                result = _run_async(result)
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                output=str(result),
                status=ToolResultStatus.SUCCESS,
            )
        except Exception as e:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=str(e),
            )


# ── Tool environment ────────────────────────────────────────────────


class VerifiersToolEnvironment(BaseToolEnvironment[VerifiersToolHidden]):
    """MDP wrapper for verifiers ToolEnv.

    Multi-turn environment with tool calling. Extracts tools from the
    verifiers env's OpenAI-format schemas, executes via the original callables.
    """

    def __init__(
        self,
        vf_env: Any,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._vf_env = vf_env
        self._answer_extractor = answer_extractor
        self._dataset = vf_env.dataset
        self._env_id = getattr(vf_env, "env_id", "verifiers") or "verifiers"
        self._max_steps = getattr(vf_env, "max_turns", 10)

        # Convert tools
        oai_tools = getattr(vf_env, "oai_tools", []) or []
        tool_map = getattr(vf_env, "tool_map", {}) or {}
        self._tools = _oai_tools_to_definitions(oai_tools)
        self._executor = VerifiersToolExecutor(tool_map=tool_map)

        self._native_rewards: tuple[RewardFunction, ...] = (
            VerifiersRubricReward(rubric=vf_env.rubric, env_id=self._env_id),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

    @property
    def system_prompt(self) -> str | None:
        return self._vf_env.system_prompt

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_id,
            adapter="verifiers",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            metadata={
                "dataset_size": len(self._dataset),
                "env_id": self._env_id,
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
    ) -> tuple[State[VerifiersToolHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._dataset):
            raise ValueError(
                f"task_index {task_index} out of bounds [0, {len(self._dataset)})"
            )

        row = self._dataset[task_index]

        prompt_messages = row.get("prompt", [])
        user_prompt = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")

        if not user_prompt:
            user_prompt = row.get("question", "")

        expected_answer = row.get("answer")
        dataset_item = tuple(
            (k, v) for k, v in row.items()
            if isinstance(k, str)
        )

        hidden = VerifiersToolHidden(
            env_id=self._env_id,
            task_index=task_index,
            expected_answer=expected_answer,
            dataset_item=dataset_item,
            episode_step=0,
            last_action=None,
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        observation = Observation(
            prompt=user_prompt,
            available_tools=self._tools,
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        info: dict[str, Any] = {
            "task_index": task_index,
            "env_id": self._env_id,
            "system_prompt": self._vf_env.system_prompt,
        }

        return state, info

    def step(
        self,
        state: State[VerifiersToolHidden],
        action: Action,
    ) -> StepResult[VerifiersToolHidden]:
        self._state_tracker.validate(state, "VerifiersToolEnvironment")
        next_step = state.hidden.episode_step + 1
        truncated = next_step >= self._max_steps

        # Execute any tool calls
        tool_results: tuple[ToolResult, ...] = ()
        if action.tool_calls:
            tool_results = self.execute_tools(action.tool_calls)

        # Build next observation with tool results
        if tool_results or action.tool_calls:
            next_observation = self._build_next_observation(
                state.observation, action, tool_results,
            )
        else:
            # Text-only — add assistant message to history
            messages = list(state.observation.messages)
            messages.append({"role": "assistant", "content": action.text})
            next_observation = Observation(
                prompt=state.observation.prompt,
                messages=tuple(messages),
                available_tools=self._tools,
            )

        terminated = False  # Only truncation ends tool episodes

        # Compute rewards on terminal steps
        if truncated or terminated:
            next_hidden = VerifiersToolHidden(
                env_id=state.hidden.env_id,
                task_index=state.hidden.task_index,
                expected_answer=state.hidden.expected_answer,
                dataset_item=state.hidden.dataset_item,
                episode_step=next_step,
                last_action=action.text,
            )
            temp_state = State(
                observation=next_observation,
                hidden=next_hidden,
                metadata=state.metadata,
            )
            rewards = self.compute_rewards(state, action, temp_state)
        else:
            rewards = RewardBundle.empty()

        next_hidden = VerifiersToolHidden(
            env_id=state.hidden.env_id,
            task_index=state.hidden.task_index,
            expected_answer=state.hidden.expected_answer,
            dataset_item=state.hidden.dataset_item,
            episode_step=next_step,
            last_action=action.text,
        )

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=truncated or terminated,
            info={**state.metadata.info},
        )

        next_state = State(
            observation=next_observation,
            hidden=next_hidden,
            metadata=next_metadata,
        )
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "tool_results": tool_results,
                "episode_step": next_step,
            },
        )

    def compute_rewards(
        self,
        state: State[VerifiersToolHidden],
        action: Action,
        next_state: State[VerifiersToolHidden],
    ) -> RewardBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return RewardBundle(signals=tuple(signals))


# ── Adapter ─────────────────────────────────────────────────────────


class VerifiersAdapter:
    """Adapter for the verifiers library.

    Routes to VerifiersSingleTurnEnvironment or VerifiersToolEnvironment
    based on the loaded environment type.
    """

    @property
    def name(self) -> str:
        return "verifiers"

    def _get_verifiers(self) -> Any:
        try:
            import verifiers
            return verifiers
        except ImportError as e:
            raise ImportError(
                "verifiers is required for VerifiersAdapter. "
                "Install with: pip install verifiers"
            ) from e

    def list_environments(self) -> list[str]:
        """List available environment names.

        verifiers uses dynamic module loading, so we return common
        built-in environments.
        """
        return [
            "gsm8k",
            "math",
            "math_python",
            "tool_test",
            "doublecheck",
            "aime2024",
            "gpqa",
            "mmlu",
        ]

    def get_environment(
        self,
        name: str,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> VerifiersSingleTurnEnvironment | VerifiersToolEnvironment:
        """Create an environment by name.

        Loads via verifiers.load_environment() and routes to the
        appropriate wrapper based on the environment type.

        Args:
            name: Environment name/ID (e.g., "gsm8k", "tool_test").
            answer_extractor: Optional extractor for parsing model responses.
            extra_rewards: Additional reward functions.
            **kwargs: Passed to verifiers.load_environment().

        Returns:
            Wrapped environment.

        Raises:
            NotImplementedError: For SandboxEnv/PythonEnv types.
        """
        vf = self._get_verifiers()
        vf_env = vf.load_environment(name, **kwargs)

        if isinstance(vf_env, vf.SingleTurnEnv):
            return VerifiersSingleTurnEnvironment(
                vf_env=vf_env,
                answer_extractor=answer_extractor,
                extra_rewards=extra_rewards,
            )
        elif isinstance(vf_env, vf.ToolEnv):
            return VerifiersToolEnvironment(
                vf_env=vf_env,
                answer_extractor=answer_extractor,
                extra_rewards=extra_rewards,
            )
        elif isinstance(vf_env, vf.MultiTurnEnv):
            # MultiTurnEnv without tools — wrap as tool env (no tools)
            return VerifiersToolEnvironment(
                vf_env=vf_env,
                answer_extractor=answer_extractor,
                extra_rewards=extra_rewards,
            )
        else:
            raise NotImplementedError(
                f"Environment type {type(vf_env).__name__} is not supported. "
                f"SandboxEnv and PythonEnv require Prime Sandboxes cloud infrastructure."
            )

    def get_native_answer_extractor(self, task_name: str) -> None:
        """verifiers doesn't provide a standalone extractor."""
        return None

    def get_default_system_prompt(self, name: str) -> None:
        """System prompt comes from the environment itself."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """verifiers questions don't need wrapping."""
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "adapter": self.name,
            "description": f"verifiers environment: {name}",
        }
