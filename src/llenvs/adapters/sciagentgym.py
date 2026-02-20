"""SciAgentGYM adapter — wraps SciAgentGYM scientific tool-use benchmark.

SciAgentGYM provides 1780+ domain-specific tools across physics, chemistry,
materials science, life science, astronomy, and statistics. Each task defines
its own set of 3-16 tools in OpenAI function-calling schema format.

Key design: SciAgentGYM tools have internal state (history tracking, filesystem
artifacts), so tool execution is delegated to MinimalSciEnv.step() rather than
extracted and wrapped. We extend BaseToolEnvironment for tool infrastructure
but build observations from SciAgentGYM's response objects.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
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
    ToolResult,
    oai_tools_to_definitions,
)

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────

SCIAGENTGYM_SUBJECTS: tuple[str, ...] = (
    "physics",
    "chemistry",
    "materials_science",
    "life_science",
    "astronomy",
    "statistics",
)


# ── Tool call conversion ────────────────────────────────────────


def _to_sci_tool_call(tc: ToolCall) -> Any:
    """Convert an llenvs ToolCall to a SciAgentGYM ToolCall.

    Args:
        tc: The llenvs tool call.

    Returns:
        A SciAgentGYM ToolCall object.
    """
    from gym.tool import ToolCall as SciToolCall

    return SciToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)


# ── Dataset loading ──────────────────────────────────────────────


def _load_dataset(data_path: str) -> list[dict[str, Any]]:
    """Load test cases from a JSON file or directory of JSON files.

    Args:
        data_path: Path to a JSON file or directory containing JSON files.

    Returns:
        List of test case dicts.

    Raises:
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    # Directory: load all JSON files
    all_data: list[dict[str, Any]] = []
    for json_file in sorted(path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_data.extend(data)
        else:
            all_data.append(data)
    return all_data


# ── Native scoring ───────────────────────────────────────────────


def _try_native_scoring(response: str, gold_answer: str) -> tuple[float, dict[str, Any]] | None:
    """Try to score using SciAgentGYM's native evaluator.

    Returns (score, metadata) or None if the evaluator isn't available.
    """
    try:
        from gym.core.evaluator import calculate_answer_score, extract_boxed_answer

        model_answer = extract_boxed_answer(response)
        if model_answer is None:
            return 0.0, {"extracted": None, "reason": "no_boxed_answer"}

        score, summary, details = calculate_answer_score(model_answer, gold_answer)
        return score, {"extracted": model_answer, "summary": summary, "details": details}
    except ImportError:
        return None


def _fallback_scoring(response: str, gold_answer: str) -> tuple[float, dict[str, Any]]:
    """Score using our own BoxedExtractor as fallback.

    Simple string comparison — returns 1.0 for exact match, 0.0 otherwise.
    """
    from llenvs.core.extraction import BoxedExtractor

    extractor = BoxedExtractor()
    extracted, _ = extractor.extract(response)

    if extracted is None:
        return 0.0, {"extracted": None, "reason": "no_boxed_answer"}

    # Try numeric comparison first
    try:
        model_val = float(extracted)
        gold_val = float(gold_answer)
        if gold_val == 0:
            match = abs(model_val) < 1e-9
        else:
            match = abs(model_val - gold_val) / abs(gold_val) < 0.05
        score = 1.0 if match else 0.0
        return score, {"extracted": extracted, "reason": "numeric_comparison"}
    except (ValueError, TypeError):
        pass

    # String comparison (case-insensitive, whitespace-stripped)
    match = extracted.strip().lower() == gold_answer.strip().lower()
    score = 1.0 if match else 0.0
    return score, {"extracted": extracted, "reason": "string_comparison"}


# ── Hidden state ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SciAgentGymHidden:
    """Hidden state for SciAgentGYM environments.

    Attributes:
        task_index: Index into the dataset.
        task_id: Original task ID from the dataset.
        question: The task question text.
        gold_answer: Expected answer for scoring.
        subject: Scientific domain (physics, chemistry, etc.).
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        tool_names_used: Names of tools used so far.
    """

    task_index: int
    task_id: int
    question: str
    gold_answer: str
    subject: str
    episode_step: int = 0
    last_action: str | None = None
    tool_names_used: tuple[str, ...] = ()


# ── Reward function ──────────────────────────────────────────────


@dataclass
class SciAgentGymReward:
    """Reward function for SciAgentGYM.

    Non-terminal steps return None reward (STEP type).
    Terminal steps extract \\boxed{} from the response and compare to
    the gold answer using SciAgentGYM's native scoring if available,
    falling back to our BoxedExtractor.
    """

    _name: str = "sciagentgym"

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

        # Terminal: score the response
        response = action.text or ""
        gold_answer = getattr(next_state.hidden, "gold_answer", "")

        # Try native scoring first
        native_result = _try_native_scoring(response, gold_answer)
        if native_result is not None:
            score, metadata = native_result
        else:
            score, metadata = _fallback_scoring(response, gold_answer)

        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=score,
            metadata={"is_terminal": True, **metadata},
        )


# ── Environment ──────────────────────────────────────────────────


class SciAgentGymEnvironment(BaseToolEnvironment[SciAgentGymHidden]):
    """MDP wrapper for SciAgentGYM scientific tool-use environments.

    Each task provides its own set of domain-specific tools. Tool execution
    is delegated to SciAgentGYM's MinimalSciEnv.step() since tools have
    internal state (history tracking, filesystem artifacts).
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        max_steps: int = 30,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._dataset = dataset
        self._max_steps = max_steps
        self._active_env: Any | None = None
        self._tool_registry: dict[str, Any] | None = None
        self._executor = None  # Not used — SciAgentGYM handles execution

        self._native_rewards: tuple[RewardFunction, ...] = (
            SciAgentGymReward(),
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
            name="sciagentgym",
            adapter="sciagentgym",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={},
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
    ) -> tuple[State[SciAgentGymHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._dataset):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._dataset)})")

        query_data = self._dataset[task_index]

        # Create environment with tools for this task
        from gym.core.tool_loader import prepare_env_from_query

        sci_env, _tool_instances, tools_schema, tool_registry = prepare_env_from_query(query_data)
        self._active_env = sci_env
        self._tool_registry = tool_registry

        # Convert tools to ToolDefinitions
        tool_protocols = query_data.get("usage_tool_protocol", [])
        self._tools = oai_tools_to_definitions(tool_protocols)

        # Build prompt from question
        question = query_data.get("question", "")
        prompt = question

        hidden = SciAgentGymHidden(
            task_index=task_index,
            task_id=query_data.get("id", task_index),
            question=question,
            gold_answer=query_data.get("answer", ""),
            subject=query_data.get("metadata", {}).get("subject", ""),
            episode_step=0,
            last_action=None,
            tool_names_used=(),
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
            "task_id": query_data.get("id", task_index),
            "subject": hidden.subject,
            "num_tools": len(self._tools),
        }

        return state, info

    def step(
        self,
        state: State[SciAgentGymHidden],
        action: Action,
    ) -> StepResult[SciAgentGymHidden]:
        self._state_tracker.validate(state, "SciAgentGymEnvironment")

        if self._active_env is None:
            raise RuntimeError("No active SciAgentGYM environment. Call reset() first.")

        next_step = state.hidden.episode_step + 1
        terminated = False
        truncated = False
        tool_results: list[ToolResult] = []
        new_tool_names: list[str] = list(state.hidden.tool_names_used)

        if action.has_tool_calls:
            # Execute each tool call through SciAgentGYM
            for tc in action.tool_calls:
                sci_tc = _to_sci_tool_call(tc)
                try:
                    step_output = self._active_env.step(sci_tc)
                    obs_str = str(step_output.observation)
                    tool_results.append(
                        ToolResult.success(
                            call_id=tc.id,
                            tool_name=tc.name,
                            output=obs_str,
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
                if tc.name not in new_tool_names:
                    new_tool_names.append(tc.name)
        else:
            # Text-only action = final answer submission
            terminated = True

        # Check max_steps truncation
        if self._max_steps is not None and next_step >= self._max_steps:
            truncated = True

        # Build next observation
        next_obs = self._build_next_observation(
            current_obs=state.observation,
            action=action,
            tool_results=tuple(tool_results),
        )

        next_hidden = SciAgentGymHidden(
            task_index=state.hidden.task_index,
            task_id=state.hidden.task_id,
            question=state.hidden.question,
            gold_answer=state.hidden.gold_answer,
            subject=state.hidden.subject,
            episode_step=next_step,
            last_action=action.text,
            tool_names_used=tuple(new_tool_names),
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
            },
        )

    def compute_rewards(
        self,
        state: State[SciAgentGymHidden],
        action: Action,
        next_state: State[SciAgentGymHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Clean up the active SciAgentGYM environment."""
        if self._active_env is not None:
            try:
                close = getattr(self._active_env, "close", None)
                if close is not None:
                    close()
            except Exception as e:
                logger.warning(f"Error closing SciAgentGYM environment: {e}")
            finally:
                self._active_env = None
                self._tool_registry = None


# ── Adapter ──────────────────────────────────────────────────────


class SciAgentGymAdapter:
    """Adapter for the SciAgentGYM benchmark.

    Routes environment creation through test case datasets with per-task
    tool sets loaded via prepare_env_from_query().
    """

    @property
    def name(self) -> str:
        return "sciagentgym"

    def _get_sciagentgym(self) -> Any:
        try:
            import gym
            import gym.core.tool_loader

            return gym
        except ImportError as e:
            raise ImportError(
                "SciAgentGYM is required for SciAgentGymAdapter. "
                "Install with: pip install git+https://github.com/CMarsRover/SciAgentGYM.git"
            ) from e

    def list_environments(self) -> list[str]:
        envs = ["sciagentgym"]
        for subject in SCIAGENTGYM_SUBJECTS:
            envs.append(f"sciagentgym:{subject}")
        return envs

    def get_environment(
        self,
        name: str,
        dataset: list[dict[str, Any]] | None = None,
        data_path: str | None = None,
        subject: str | None = None,
        max_steps: int = 30,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> SciAgentGymEnvironment:
        """Create a SciAgentGYM environment.

        Args:
            name: Environment name. Use "sciagentgym:physics" to filter by subject.
            dataset: Pre-loaded list of test case dicts.
            data_path: Path to JSON file or directory of test cases.
            subject: Filter dataset by scientific domain.
            max_steps: Maximum steps per episode.
            extra_rewards: Additional reward functions.

        Returns:
            SciAgentGymEnvironment wrapping the dataset.

        Raises:
            ValueError: If neither dataset nor data_path is provided.
        """
        self._get_sciagentgym()

        # Load dataset
        if dataset is None and data_path is not None:
            dataset = _load_dataset(data_path)
        elif dataset is None:
            raise ValueError(
                "Either dataset= or data_path= must be provided for SciAgentGYM environments."
            )

        # Parse subject from name (e.g., "sciagentgym:physics")
        parsed_subject = subject
        if ":" in name:
            _, parsed_subject = name.split(":", 1)

        # Filter by subject if specified
        if parsed_subject is not None:
            parsed_subject_lower = parsed_subject.lower()
            dataset = [
                q
                for q in dataset
                if q.get("metadata", {}).get("subject", "").lower() == parsed_subject_lower
            ]

        return SciAgentGymEnvironment(
            dataset=dataset,
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
            "description": "SciAgentGYM scientific tool-use benchmark",
            "subjects": list(SCIAGENTGYM_SUBJECTS),
        }
