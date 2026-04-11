"""Trajectory runner for orchestrating evaluations.

Handles running trajectories through environments with model backends,
collecting results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenvs.core.tool_parsing import ToolCallParser
    from llenvs.core.tools import ToolDefinition
    from llenvs.evaluation.history import HistoryFn
    from llenvs.inference.prompts import ModelProfile, PromptTemplate

from llenvs.core.environment import Environment, StepResult
from llenvs.core.reward import RewardType
from llenvs.core.segmented_environment import SegmentedEnvironment
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.evaluation.continuation import (
    _BUFFER_ONLY_RESULT,
    ContinuationStrategy,
    SegmentContext,
    select_strategy,
)
from llenvs.evaluation.logging import (
    LogConfig,
    _BatchEndEvent,
    _BatchStartEvent,
    _ErrorEvent,
    _StepEvent,
    _TrajectoryEndEvent,
)
from llenvs.evaluation.logging import (
    _EvaluationLogger as _EvalLogger,
)
from llenvs.inference.prompting import PromptPipeline, PromptTemplateTransformer
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
)

logger = logging.getLogger(__name__)


def _runner_now_monotonic() -> float:
    return time.monotonic()


def _raise_with_context(kind: str, task_index: int, error: Exception) -> None:
    message = f"Error {kind} task {task_index}: {error}"
    logger.error(message)
    error.args = (message,)
    raise error


def _raise_multi_reset(entry_index: int, task_index: int, error: Exception) -> None:
    message = f"Error resetting task {task_index} in entry {entry_index}: {error}"
    logger.error(message)
    error.args = (message,)
    raise error


@dataclass(frozen=True)
class TurnInfoConfig:
    """Configuration for injecting turn/step info into structured messages.

    When enabled on a ``TrajectoryRunner``, the runner prepends turn counters
    to state observations and appends turn-limit info to task descriptions.

    Placeholders available in format strings:
        ``{max_steps}``: Maximum steps declared by the environment spec.
        ``{turn}``: Current turn number (1-indexed).
        ``{turns_remaining}``: ``max_steps - turn``.

    Two sets of templates are provided — one for environments that declare
    ``max_steps`` and one for those that don't.

    Attributes:
        task_suffix: Appended to the task description when ``max_steps`` is known.
        state_prefix: Prepended to each state observation when ``max_steps`` is known.
        task_suffix_no_max: Appended to the task description when ``max_steps`` is None.
        state_prefix_no_max: Prepended to each state observation when ``max_steps`` is None.
    """

    task_suffix: str = "\n\nYou have a maximum of {max_steps} turns."
    state_prefix: str = "[Turn {turn}/{max_steps}]\n"
    task_suffix_no_max: str = ""
    state_prefix_no_max: str = "[Turn {turn}]\n"


def _truncate_image_history(
    messages: list[ChatMessage],
    max_images: int,
) -> list[ChatMessage]:
    """Keep only the most recent N images across all messages.

    Images beyond the limit are stripped (images field set to empty tuple).
    Text content is preserved.

    Args:
        messages: List of ChatMessages.
        max_images: Maximum number of images to keep (from the end).

    Returns:
        New list of ChatMessages with older images removed.
    """
    # Count total images and find which messages have them
    image_positions: list[tuple[int, int]] = []  # (msg_index, num_images)
    for i, msg in enumerate(messages):
        if msg.images:
            image_positions.append((i, len(msg.images)))

    total = sum(n for _, n in image_positions)
    if total <= max_images:
        return messages

    # Determine how many to keep from the end
    keep = max_images
    keep_from: dict[int, int] = {}  # msg_index -> num images to keep from this msg
    for msg_idx, n in reversed(image_positions):
        if keep <= 0:
            keep_from[msg_idx] = 0
        elif keep >= n:
            keep_from[msg_idx] = n
            keep -= n
        else:
            keep_from[msg_idx] = keep
            keep = 0

    result = []
    for i, msg in enumerate(messages):
        if i in keep_from:
            n_keep = keep_from[i]
            if n_keep == 0:
                result.append(
                    ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        tool_calls=msg.tool_calls,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                        images=(),
                    )
                )
            elif n_keep < len(msg.images):
                # Keep only the last n_keep images
                result.append(
                    ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        tool_calls=msg.tool_calls,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                        images=msg.images[-n_keep:],
                    )
                )
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


# Sentinel value: when a step_callback returns COMPLETE, the segment loop
# breaks immediately and returns a SegmentedTrajectoryHandle for later
# completion via complete_trajectory().
COMPLETE = "___COMPLETE___"


@dataclass(frozen=True)
class ForceAction:
    """Override generation at a step with predetermined text.

    When returned from a step_callback, the runner uses this text as the
    next segment instead of calling the backend. The buffer is cleared
    (stale after forced context change).
    """

    text: str


@dataclass
class TrajectoryResult:
    """Result of running a single trajectory.

    Attributes:
        trajectory: The full trajectory (sequence of state-action-reward transitions).
        total_reward: Sum of all rewards across the trajectory.
        success: Whether the trajectory was successful (based on correctness).
        metadata: Additional result metadata.
    """

    trajectory: Trajectory[Any]
    total_reward: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of running a batch of trajectories.

    Attributes:
        trajectory_results: List of individual trajectory results.
        success_rate: Fraction of successful trajectories.
        mean_reward: Mean total reward across trajectories.
        metadata: Additional batch metadata.
    """

    trajectory_results: list[TrajectoryResult]
    success_rate: float
    mean_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _task_index_for_state(state: State[Any]) -> int:
    """Extract the task index from a state's metadata or hidden fields.

    Lookup order:
    1. ``state.metadata.info["task_index"]`` (set by multi-instance resets)
    2. ``state.hidden.task_index`` (adapter-level task identity)
    3. ``0`` (single-task default)
    """
    task_index = state.metadata.info.get("task_index")
    if isinstance(task_index, int):
        return task_index
    hidden_task_index = getattr(state.hidden, "task_index", 0)
    return hidden_task_index if isinstance(hidden_task_index, int) else 0


@dataclass
class _ActiveTrajectory:
    """Internal state for tracking a trajectory during lockstep batch execution."""

    position: int
    task_index: int
    state: State[Any]
    reset_info: dict[str, Any]
    trajectory: Trajectory[Any]
    done: bool = False
    error: str | None = None
    step_count: int = 0
    failed: bool = False


def _summarize_task_indices(task_indices: Sequence[int], *, limit: int = 8) -> str:
    """Return a compact representation of task indices for debug logs."""
    if not task_indices:
        return "[]"
    shown = ", ".join(str(task_index) for task_index in task_indices[:limit])
    if len(task_indices) > limit:
        return f"[{shown}, ...] ({len(task_indices)} total)"
    return f"[{shown}]"


def _summarize_active_trajectories(
    trajectories: Sequence[_ActiveTrajectory],
    *,
    limit: int = 8,
) -> str:
    """Return ``task@step`` summaries for active trajectories."""
    if not trajectories:
        return "[]"
    shown = ", ".join(
        f"{trajectory.task_index}@{trajectory.step_count}"
        for trajectory in trajectories[:limit]
    )
    if len(trajectories) > limit:
        return f"[{shown}, ...] ({len(trajectories)} total)"
    return f"[{shown}]"


@dataclass
class _ActiveSegmentedTrajectory:
    """Internal state for tracking a segmented trajectory during batch execution."""

    position: int
    task_index: int
    state: State[Any]
    reset_info: dict[str, Any]
    trajectory: Trajectory[Any]
    messages: list[ChatMessage]
    accumulated: str = ""
    buffer: str = ""
    done: bool = False
    error: str | None = None
    step_count: int = 0
    forced_segment: str | None = None
    complete_early: bool = False
    generation_done: bool = False


def _error_metadata(task_index: int) -> StateMetadata:
    """Create dummy metadata for error cases."""
    return StateMetadata(
        step=0,
        episode_id=f"error_{task_index}",
        is_terminal=True,
        info={"error": True},
    )


def _aggregate_results(
    trajectory_results: list[TrajectoryResult],
) -> BatchResult:
    """Compute aggregate metrics for a batch of trajectory results."""
    num_successful = sum(1 for r in trajectory_results if r.success)
    success_rate = num_successful / len(trajectory_results) if trajectory_results else 0.0
    total_rewards = [r.total_reward for r in trajectory_results]
    mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

    return BatchResult(
        trajectory_results=trajectory_results,
        success_rate=success_rate,
        mean_reward=mean_reward,
        metadata={
            "num_trajectories": len(trajectory_results),
            "num_successful": num_successful,
        },
    )


def _run_in_chunks(
    run_fn: Callable[[list[int], Callable[[int, int], None] | None], BatchResult],
    task_indices: list[int],
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None,
) -> BatchResult:
    """Run batches in chunks of batch_size, aggregating results.

    Args:
        run_fn: Callable that takes (task_indices, progress_callback) and
            returns a BatchResult.
        task_indices: All task indices to process.
        batch_size: Maximum tasks per chunk.
        progress_callback: Optional callback(completed, total).

    Returns:
        Aggregated BatchResult over all chunks.
    """
    all_results: list[TrajectoryResult] = []
    total = len(task_indices)

    for start in range(0, total, batch_size):
        chunk = task_indices[start : start + batch_size]

        chunk_progress_callback: Callable[[int, int], None] | None = None
        if progress_callback:
            _offset = start

            def _chunk_progress_callback(
                done: int,
                chunk_total: int,
                _s: int = _offset,
            ) -> None:
                del chunk_total
                progress_callback(_s + done, total)

            chunk_progress_callback = _chunk_progress_callback

        chunk_result = run_fn(chunk, chunk_progress_callback)
        all_results.extend(chunk_result.trajectory_results)

    if progress_callback:
        progress_callback(total, total)

    return _aggregate_results(all_results)


def _run_in_sequence_chunks(
    items: Sequence[Any],
    *,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None,
    run_chunk: Callable[[Sequence[Any], Callable[[int, int], None] | None], list[Any]],
) -> list[Any]:
    """Run chunked sequence workloads while preserving global progress offsets."""
    results: list[Any] = []
    total = len(items)

    for start in range(0, total, batch_size):
        chunk = items[start : start + batch_size]

        chunk_progress_callback: Callable[[int, int], None] | None = None
        if progress_callback:
            _offset = start

            def _chunk_progress_callback(
                done: int,
                chunk_total: int,
                _s: int = _offset,
            ) -> None:
                del chunk_total
                progress_callback(_s + done, total)

            chunk_progress_callback = _chunk_progress_callback

        results.extend(run_chunk(chunk, chunk_progress_callback))

    if progress_callback:
        progress_callback(total, total)

    return results


def _finalize_trajectory(t: _ActiveTrajectory | _ActiveSegmentedTrajectory) -> TrajectoryResult:
    """Build a TrajectoryResult from a completed active trajectory."""
    if t.error is not None:
        return TrajectoryResult(
            trajectory=t.trajectory,
            total_reward=t.trajectory.total_reward,
            success=False,
            metadata={"error": t.error, "task_index": t.task_index},
        )

    success = False
    if t.trajectory.transitions:
        last_rewards = t.trajectory.transitions[-1].rewards
        outcome_rewards = last_rewards.by_type(RewardType.OUTCOME)
        if outcome_rewards:
            success = outcome_rewards[-1].reward >= 1.0

    return TrajectoryResult(
        trajectory=t.trajectory,
        total_reward=t.trajectory.total_reward,
        success=success,
        metadata={
            "task_index": t.task_index,
            "num_steps": len(t.trajectory),
            "reset_info": t.reset_info,
        },
    )


def _coalesce_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Merge consecutive same-role messages by joining content.

    Consecutive messages with the same role are merged into a single message
    with content joined by ``"\\n\\n"`` and images concatenated. This avoids
    issues with APIs that reject consecutive same-role messages (e.g., two
    user messages in a row).

    Only merges messages with role "user" or "assistant". System and tool
    messages are never merged.

    Args:
        messages: List of ChatMessages.

    Returns:
        New list with consecutive same-role messages merged.
    """
    if not messages:
        return messages

    result: list[ChatMessage] = []
    for msg in messages:
        if (
            result
            and msg.role == result[-1].role
            and msg.role in ("user", "assistant")
            and not msg.tool_calls
            and not msg.tool_call_id
            and not result[-1].tool_calls
            and not result[-1].tool_call_id
        ):
            prev = result[-1]
            merged_content = "\n\n".join(part for part in [prev.content, msg.content] if part)
            merged_images = prev.images + msg.images
            result[-1] = ChatMessage(
                role=prev.role,
                content=merged_content,
                images=merged_images,
            )
        else:
            result.append(msg)
    return result


def _normalize_text_for_comparison(text: str | None) -> str:
    """Normalize observation text for equality checks."""
    if not text:
        return ""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


@dataclass
class TrajectoryRunner:
    """Runs trajectories through an environment with a model backend.

    Handles the interaction loop between model and environment,
    collecting trajectories and computing results. Supports both
    text-only and tool-aware environments.

    Attributes:
        environment: The environment to run trajectories in.
        backend: The model backend for generation.
        sampling_params: Parameters for text generation.
        prompt_pipeline: Optional pipeline to transform prompts.
        system_prompt: Optional system prompt to include.
    """

    environment: Environment[Any]
    backend: ModelBackend
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    prompt_pipeline: PromptPipeline | None = None
    system_prompt: str | None = None
    prompt_template: PromptTemplate | None = None
    model_profile: ModelProfile | None = None
    tool_call_parser: ToolCallParser | None = None
    turn_info: TurnInfoConfig | bool | None = None
    log: LogConfig | None = None
    max_image_history: int | None = None
    history_fn: HistoryFn | None = None
    prompt_budget: Any | None = None  # PromptBudget — uses Any to avoid circular import
    include_reasoning_in_history: bool = False
    format_reminder: str | None = None
    env_factory: Callable[[], Environment[Any]] | None = None
    restore_fn: Callable[[Environment[Any], State[Any]], State[Any]] | None = None
    system_prompt_fn: Callable[[State[Any], int | None], str | None] | None = None
    last_environment_errors: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False,
    )

    def __post_init__(self) -> None:
        if self.system_prompt is not None and self.system_prompt_fn is not None:
            raise ValueError(
                "system_prompt and system_prompt_fn are mutually exclusive"
            )

    def _resolve_system_prompt(
        self,
        state: State[Any],
        task_index: int | None = None,
    ) -> str | None:
        """Return the system prompt for a given state.

        When ``system_prompt_fn`` is set, delegates to the callback.
        Otherwise returns the static ``system_prompt``.
        """
        if self.system_prompt_fn is not None:
            return self.system_prompt_fn(state, task_index)
        return self.system_prompt

    def _build_messages(
        self,
        state: State[Any],
        trajectory: Trajectory[Any] | None = None,
        task_index: int | None = None,
    ) -> list[ChatMessage]:
        """Build chat messages from state including tool results.

        When the observation has structured ``task``/``state`` fields and a
        trajectory is provided, uses structured mode: builds messages from
        the trajectory's task description and per-step state observations.
        Otherwise falls back to legacy mode (prompt + messages).

        Args:
            state: Current environment state.
            trajectory: Optional trajectory for structured message building.
            task_index: Optional task index for per-task system prompt
                resolution via ``system_prompt_fn``.

        Returns:
            List of ChatMessages for the model.
        """
        obs = state.observation

        # Structured mode: when task field is set, trajectory available,
        # and no tools (tool environments need legacy message formatting)
        if obs.task is not None and trajectory is not None and not obs.available_tools:
            messages = self._build_structured_messages(state, trajectory, task_index)
        else:
            messages = self._build_legacy_messages(state, task_index)

        # Apply prompt template to wrap the question
        if self.prompt_template is not None:
            transformer = PromptTemplateTransformer(template=self.prompt_template)
            messages = transformer.transform(messages)

        # Apply model profile transformers
        if self.model_profile is not None:
            for t in self.model_profile.build_transformers():
                messages = t.transform(messages)

        # Apply prompt pipeline if configured
        if self.prompt_pipeline:
            messages = self.prompt_pipeline.transform(messages)

        # Truncate image history if configured
        if self.max_image_history is not None:
            messages = _truncate_image_history(messages, self.max_image_history)

        # Coalesce consecutive same-role messages
        messages = _coalesce_messages(messages)

        return messages

    # Estimated per-message overhead tokens for chat template formatting
    # (role tags, special tokens, etc.)
    _MSG_OVERHEAD_TOKENS: int = 5

    def _build_structured_messages(
        self,
        state: State[Any],
        trajectory: Trajectory[Any],
        task_index: int | None = None,
    ) -> list[ChatMessage]:
        """Build messages from structured task/state fields.

        Sequence:
        1. [system: system_prompt] (if any)
        2. [user: task.text + task.images]
        3. history_fn(entries) — prior (action, observation) pairs
        4. [user: current state.text + state.images]

        When ``prompt_budget`` is set, computes the token cost of non-history
        parts and passes the remaining budget to ``prompt_budget.build_history``
        instead of using ``history_fn``.

        When ``history_fn`` is None (default) and ``prompt_budget`` is None,
        ``full_history`` is used. ``prompt_budget`` takes precedence over
        ``history_fn`` when both are set.

        The ``include_reasoning_in_history`` flag controls whether prior
        actions show the full model response or the extracted action.
        """
        from llenvs.evaluation.history import HistoryEntry, full_history

        messages: list[ChatMessage] = []

        resolved_prompt = self._resolve_system_prompt(state, task_index)
        if resolved_prompt:
            messages.append(ChatMessage(role="system", content=resolved_prompt))

        obs = state.observation
        task = obs.task

        # Resolve turn info config
        tic = self._resolve_turn_info()
        max_steps = self.environment.spec.max_steps if tic else None

        # Task description (with optional turn info suffix)
        task_text = task.text
        if tic is not None:
            if max_steps is not None:
                suffix = tic.task_suffix.format(
                    max_steps=max_steps,
                    turn=state.metadata.step + 1,
                    turns_remaining=max_steps - (state.metadata.step + 1),
                )
            else:
                suffix = tic.task_suffix_no_max.format(
                    turn=state.metadata.step + 1,
                )
            task_text = task_text + suffix

        messages.append(
            ChatMessage(
                role="user",
                content=task_text,
                images=task.images,
            )
        )

        # The initial state's observation is not the result of any action,
        # so it never appears in a history entry.  At step 1+, inject it
        # before history so the model retains context for the first action.
        initial_obs_msg: ChatMessage | None = None
        if state.metadata.step > 0:
            init_obs = trajectory.initial_state.observation
            init_state = init_obs.state
            init_task = init_obs.task
            if init_state is not None:
                skip_initial = (
                    init_task is not None
                    and _normalize_text_for_comparison(init_state.text)
                    == _normalize_text_for_comparison(init_task.text)
                    and init_state.images == init_task.images
                )
                if not skip_initial and init_state.text:
                    init_text = init_state.text
                    if tic is not None:
                        init_turn = trajectory.initial_state.metadata.step + 1
                        if max_steps is not None:
                            init_text = tic.state_prefix.format(
                                max_steps=max_steps,
                                turn=init_turn,
                                turns_remaining=max_steps - init_turn,
                            ) + init_text
                        else:
                            init_text = (
                                tic.state_prefix_no_max.format(turn=init_turn)
                                + init_text
                            )
                    initial_obs_msg = ChatMessage(
                        role="user",
                        content=init_text,
                        images=init_state.images,
                    )

        # Build history entries from prior transitions.
        # The last transition's next_state IS the current state, so we
        # exclude its observation from the history (it's added separately).
        transitions = trajectory.transitions
        history_entries: list[HistoryEntry] = []
        for i, transition in enumerate(transitions):
            action_text = self._resolve_action_text(transition)
            is_last = i == len(transitions) - 1

            next_obs = transition.next_state.observation
            if not is_last and next_obs.state is not None:
                history_entries.append(
                    HistoryEntry(
                        action_text=action_text,
                        observation_text=next_obs.state.text,
                        observation_images=next_obs.state.images,
                        step=transition.next_state.metadata.step,
                    )
                )
            else:
                # Last transition or no state: action only
                history_entries.append(
                    HistoryEntry(
                        action_text=action_text,
                        observation_text="",
                        step=transition.next_state.metadata.step,
                    )
                )

        # When a trajectory was created fresh from a mid-episode state, the
        # transitions only capture the rollout suffix.  Reconstruct the prefix
        # transcript from the restored state's observation.messages.
        reconstructed_prefix_msgs: list[ChatMessage] = []
        init_obs = trajectory.initial_state.observation
        if init_obs.messages and trajectory.initial_state.metadata.step > 0:
            reconstructed = self._prior_history_from_messages(
                init_obs.messages,
                split_final_user=not trajectory.transitions,
            )
            if reconstructed is not None:
                reconstructed_prefix_msgs, prior = reconstructed
                if prior:
                    history_entries = prior + history_entries
                if reconstructed_prefix_msgs or prior:
                    # Do not separately inject the restored state's observation.
                    # If there are new transitions, the restored state already
                    # lives in `prior`. If there are no new transitions, the
                    # helper kept the restored current observation out of
                    # history, and the normal current_state_text path will add
                    # it exactly once.
                    initial_obs_msg = None

        # Resolve current state content early so we can measure it for budgets
        skip_current_state = False
        if obs.state is not None and task is not None and state.metadata.step == 0:
            skip_current_state = (
                _normalize_text_for_comparison(obs.state.text)
                == _normalize_text_for_comparison(task.text)
                and obs.state.images == task.images
            )

        current_state_text: str | None = None
        if obs.state is not None and not skip_current_state:
            current_state_text = obs.state.text
            if tic is not None:
                turn = state.metadata.step + 1
                if max_steps is not None:
                    prefix = tic.state_prefix.format(
                        max_steps=max_steps,
                        turn=turn,
                        turns_remaining=max_steps - turn,
                    )
                else:
                    prefix = tic.state_prefix_no_max.format(
                        turn=turn,
                    )
                current_state_text = prefix + current_state_text
            current_state_text = self._append_format_reminder(current_state_text)
        elif self.format_reminder and messages and messages[-1].role == "user":
            last = messages[-1]
            messages[-1] = ChatMessage(
                role=last.role,
                content=self._append_format_reminder(last.content),
                images=last.images,
            )

        # Apply history function or prompt budget
        if self.prompt_budget is not None:
            budget = self.prompt_budget
            # Estimate non-history token cost
            non_history_tokens = 0
            for msg in messages:
                non_history_tokens += (
                    budget.estimate_tokens(msg.content or "")
                    + self._MSG_OVERHEAD_TOKENS
                )
            if current_state_text is not None:
                non_history_tokens += (
                    budget.estimate_tokens(current_state_text)
                    + self._MSG_OVERHEAD_TOKENS
                )
            if initial_obs_msg is not None:
                non_history_tokens += (
                    budget.estimate_tokens(initial_obs_msg.content or "")
                    + self._MSG_OVERHEAD_TOKENS
                )
            for pfx_msg in reconstructed_prefix_msgs:
                non_history_tokens += (
                    budget.estimate_tokens(pfx_msg.content or "")
                    + self._MSG_OVERHEAD_TOKENS
                )
            available = max(0, budget.max_prompt_tokens - non_history_tokens)
            history_messages = budget.build_history(history_entries, available)
            if initial_obs_msg is not None and history_messages:
                messages.append(initial_obs_msg)
            if reconstructed_prefix_msgs and history_messages:
                messages.extend(reconstructed_prefix_msgs)
            messages.extend(history_messages)

            # Phase 2: truncate current observation if still over budget
            if (
                current_state_text is not None
                and budget.min_current_observation_chars is not None
            ):
                total = non_history_tokens
                for msg in history_messages:
                    total += (
                        budget.estimate_tokens(msg.content or "")
                        + self._MSG_OVERHEAD_TOKENS
                    )
                excess = total - budget.max_prompt_tokens
                if excess > 0:
                    from llenvs.evaluation.history import middle_truncate

                    cur_tokens = budget.estimate_tokens(current_state_text)
                    cur_len = len(current_state_text)
                    # Approximate chars-per-token for this text
                    cpt = cur_len / cur_tokens if cur_tokens > 0 else 4.0
                    # Account for middle_truncate marker overhead
                    _MARKER_OVERHEAD = 45
                    target_chars = max(
                        budget.min_current_observation_chars,
                        int(cur_len - excess * cpt) - _MARKER_OVERHEAD,
                    )
                    current_state_text = middle_truncate(
                        current_state_text, target_chars,
                    )
        else:
            fn = self.history_fn if self.history_fn is not None else full_history
            history_messages = fn(history_entries)
            if initial_obs_msg is not None and history_messages:
                messages.append(initial_obs_msg)
            if reconstructed_prefix_msgs and history_messages:
                messages.extend(reconstructed_prefix_msgs)
            messages.extend(history_messages)

        # Add current state observation
        if current_state_text is not None:
            messages.append(
                ChatMessage(
                    role="user",
                    content=current_state_text,
                    images=obs.state.images if obs.state else (),
                )
            )

        return messages

    def _resolve_turn_info(self) -> TurnInfoConfig | None:
        """Resolve turn_info field to a TurnInfoConfig or None."""
        if self.turn_info is True:
            return TurnInfoConfig()
        if isinstance(self.turn_info, TurnInfoConfig):
            return self.turn_info
        return None

    def _append_format_reminder(self, text: str | None) -> str | None:
        """Append the optional format reminder to a user-facing text block."""
        if text is None or not self.format_reminder:
            return text
        return text.rstrip() + "\n\n" + self.format_reminder

    def _resolve_action_text(self, transition: Transition[Any]) -> str:
        """Get the action text for a transition, respecting reasoning stripping.

        When ``include_reasoning_in_history`` is False (default), uses the
        priority chain ``resolved_action`` → ``extracted_action`` → legacy
        step-info fields → raw text with thinking stripped.  This matches
        ``value_bench.methods.serialization.action_text_for_display``.
        """
        full_text = transition.action.text or ""

        if self.include_reasoning_in_history:
            return full_text

        # Prefer resolved_action (formatted native command, or placeholder
        # for invalid actions).
        if transition.resolved_action is not None:
            return transition.resolved_action

        # Then extracted_action (strips reasoning even on mapping failure)
        if transition.extracted_action is not None:
            return transition.extracted_action

        # Legacy fallback: extracted action/answer from step info
        step_info = transition.info.get("step", {})
        if isinstance(step_info, dict):
            extracted = step_info.get("extracted_action") or step_info.get("extracted_answer")
            if extracted:
                return extracted

        from llenvs.core.cleaning import strip_thinking_tokens

        return strip_thinking_tokens(full_text)

    def _build_legacy_messages(
        self,
        state: State[Any],
        task_index: int | None = None,
    ) -> list[ChatMessage]:
        """Build messages using legacy prompt + messages fields.

        When ``history_fn`` or ``prompt_budget`` is configured, plain text-only
        assistant/user histories are reconstructed into history entries so the
        same history shaping used by structured observations also applies to
        legacy chat environments. Tool-call histories continue to use the raw
        legacy path unchanged.
        """
        if self.prompt_budget is not None or self.history_fn is not None:
            budgeted = self._build_budgeted_legacy_messages(state, task_index)
            if budgeted is not None:
                return budgeted
        return self._build_raw_legacy_messages(state, task_index)

    def _build_raw_legacy_messages(
        self,
        state: State[Any],
        task_index: int | None = None,
    ) -> list[ChatMessage]:
        """Build legacy messages without history reconstruction."""
        messages: list[ChatMessage] = []

        resolved_prompt = self._resolve_system_prompt(state, task_index)
        if resolved_prompt:
            messages.append(ChatMessage(role="system", content=resolved_prompt))

        obs = state.observation
        messages.append(ChatMessage(role="user", content=obs.prompt, images=obs.images))

        for msg in obs.messages:
            role = msg.get("role", "user")

            if role == "assistant":
                tool_calls_data = msg.get("tool_calls", [])
                if tool_calls_data:
                    from llenvs.core.tools import ToolCall

                    tool_calls = tuple(
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc["arguments"],
                        )
                        for tc in tool_calls_data
                    )
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=msg.get("content"),
                            tool_calls=tool_calls,
                        )
                    )
                else:
                    messages.append(
                        ChatMessage(role="assistant", content=msg.get("content", ""))
                    )

            elif role == "tool":
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=msg.get("content", ""),
                        tool_call_id=msg.get("tool_call_id"),
                        name=msg.get("name"),
                    )
                )

            elif role == "user":
                messages.append(
                    ChatMessage(
                        role="user",
                        content=msg.get("content", ""),
                        images=self._legacy_message_images(msg),
                    )
                )

        if self.format_reminder and messages and messages[-1].role == "user":
            last = messages[-1]
            messages[-1] = ChatMessage(
                role=last.role,
                content=self._append_format_reminder(last.content),
                images=last.images,
            )

        return messages

    def _legacy_message_images(self, msg: dict[str, Any]) -> tuple[Any, ...]:
        """Extract image payloads from a legacy message dict."""
        if "images" not in msg:
            return ()

        from llenvs.core.state import ImageContent

        return tuple(
            ImageContent(
                data=im["data"],
                media_type=im.get("media_type", "image/png"),
            )
            for im in msg["images"]
        )

    def _prior_history_from_messages(
        self,
        messages: tuple[dict[str, Any], ...],
        *,
        split_final_user: bool,
    ) -> tuple[list[ChatMessage], list["HistoryEntry"]] | None:
        """Reconstruct structured prompt context from accumulated messages.

        Returns ``(prefix_messages, history_entries)``, or ``None`` if the
        transcript contains tool calls or unsupported roles.

        When *split_final_user* is ``True`` and the transcript ends with a
        user message, that final user message is excluded from the result so
        the caller can present it via the normal ``current_state_text`` path.
        """
        from llenvs.evaluation.history import HistoryEntry

        if not messages:
            return ([], [])

        raw = list(messages)

        # When split_final_user is requested and the last message is a user
        # message, pop it so it stays on the current_state_text path.
        if split_final_user and raw and raw[-1].get("role", "user") == "user":
            raw.pop()

        prefix_messages: list[ChatMessage] = []
        history_entries: list[HistoryEntry] = []
        pending_action: str | None = None
        step_counter = 0

        for msg in raw:
            role = msg.get("role", "user")

            if role == "assistant":
                if msg.get("tool_calls"):
                    return None
                # Flush any pending action as an empty-observation entry
                if pending_action is not None:
                    step_counter += 1
                    history_entries.append(
                        HistoryEntry(
                            action_text=pending_action,
                            observation_text="",
                            step=step_counter,
                        )
                    )
                pending_action = msg.get("content", "") or ""
                continue

            if role == "user":
                if pending_action is None:
                    # Leading user message(s) before any assistant turn
                    prefix_messages.append(
                        ChatMessage(
                            role="user",
                            content=msg.get("content", "") or "",
                            images=self._legacy_message_images(msg),
                        )
                    )
                else:
                    step_counter += 1
                    history_entries.append(
                        HistoryEntry(
                            action_text=pending_action,
                            observation_text=msg.get("content", "") or "",
                            observation_images=self._legacy_message_images(msg),
                            step=step_counter,
                        )
                    )
                    pending_action = None
                continue

            # Unexpected role (tool, system, etc.)
            return None

        # Trailing assistant with no paired user
        if pending_action is not None:
            step_counter += 1
            history_entries.append(
                HistoryEntry(
                    action_text=pending_action,
                    observation_text="",
                    step=step_counter,
                )
            )

        return (prefix_messages, history_entries)

    def _build_budgeted_legacy_messages(
        self,
        state: State[Any],
        task_index: int | None = None,
    ) -> list[ChatMessage] | None:
        """Apply history shaping to plain text-only legacy chats when possible."""
        from llenvs.evaluation.history import HistoryEntry, full_history

        obs = state.observation
        messages: list[ChatMessage] = []

        resolved_prompt = self._resolve_system_prompt(state, task_index)
        if resolved_prompt:
            messages.append(ChatMessage(role="system", content=resolved_prompt))

        messages.append(ChatMessage(role="user", content=obs.prompt, images=obs.images))

        if not obs.messages:
            return messages

        raw_messages = list(obs.messages)
        current_message: dict[str, Any] | None = None
        if raw_messages and raw_messages[-1].get("role", "user") == "user":
            current_message = raw_messages.pop()

        history_entries: list[HistoryEntry] = []
        pending_action: str | None = None

        for index, msg in enumerate(raw_messages, start=1):
            role = msg.get("role", "user")
            if role == "assistant":
                if msg.get("tool_calls"):
                    return None
                if pending_action is not None:
                    history_entries.append(
                        HistoryEntry(
                            action_text=pending_action,
                            observation_text="",
                            step=index,
                        )
                    )
                pending_action = msg.get("content", "") or ""
                continue

            if role != "user":
                return None
            if pending_action is None:
                return None

            history_entries.append(
                HistoryEntry(
                    action_text=pending_action,
                    observation_text=msg.get("content", "") or "",
                    observation_images=self._legacy_message_images(msg),
                    step=index,
                )
            )
            pending_action = None

        if pending_action is not None:
            history_entries.append(
                HistoryEntry(
                    action_text=pending_action,
                    observation_text="",
                    step=len(raw_messages) + 1,
                )
            )

        current_user_message: ChatMessage | None = None
        if current_message is not None:
            current_user_message = ChatMessage(
                role="user",
                content=self._append_format_reminder(current_message.get("content", "") or ""),
                images=self._legacy_message_images(current_message),
            )
        elif self.format_reminder and messages and messages[-1].role == "user":
            last = messages[-1]
            messages[-1] = ChatMessage(
                role=last.role,
                content=self._append_format_reminder(last.content),
                images=last.images,
            )

        if self.prompt_budget is not None:
            budget = self.prompt_budget
            non_history_tokens = 0
            for msg in messages:
                non_history_tokens += (
                    budget.estimate_tokens(msg.content or "")
                    + self._MSG_OVERHEAD_TOKENS
                )
            if current_user_message is not None:
                non_history_tokens += (
                    budget.estimate_tokens(current_user_message.content or "")
                    + self._MSG_OVERHEAD_TOKENS
                )
            available = max(0, budget.max_prompt_tokens - non_history_tokens)
            history_messages = budget.build_history(history_entries, available)
            messages.extend(history_messages)

            # Phase 2: truncate current observation if still over budget
            if (
                current_user_message is not None
                and budget.min_current_observation_chars is not None
            ):
                total = non_history_tokens
                for msg in history_messages:
                    total += (
                        budget.estimate_tokens(msg.content or "")
                        + self._MSG_OVERHEAD_TOKENS
                    )
                excess = total - budget.max_prompt_tokens
                if excess > 0:
                    from llenvs.evaluation.history import middle_truncate

                    cur_text = current_user_message.content or ""
                    cur_tokens = budget.estimate_tokens(cur_text)
                    cur_len = len(cur_text)
                    cpt = cur_len / cur_tokens if cur_tokens > 0 else 4.0
                    _MARKER_OVERHEAD = 45
                    target_chars = max(
                        budget.min_current_observation_chars,
                        int(cur_len - excess * cpt) - _MARKER_OVERHEAD,
                    )
                    current_user_message = ChatMessage(
                        role="user",
                        content=middle_truncate(cur_text, target_chars),
                        images=current_user_message.images,
                    )
        else:
            fn = self.history_fn if self.history_fn is not None else full_history
            messages.extend(fn(history_entries))

        if current_user_message is not None:
            messages.append(current_user_message)

        return messages

    def _resolve_elicitation_suffix(self) -> str:
        """Resolve the suffix for second elicitation."""
        return self.sampling_params.second_elicitation_suffix or ""

    def _build_elicitation_messages(
        self,
        messages: list[ChatMessage],
        first_result: GenerationResult,
        suffix: str,
    ) -> list[ChatMessage]:
        """Build continuation messages for second elicitation."""
        continued = list(messages)
        continued.append(ChatMessage(role="assistant", content=(first_result.text or "") + suffix))
        continued.append(ChatMessage(role="user", content="Please provide your final answer."))
        return continued

    def _elicitation_params(self) -> SamplingParams:
        """Create sampling params for the second elicitation call."""
        return replace(
            self.sampling_params,
            max_tokens=self.sampling_params.second_elicitation_max_tokens,
            second_elicitation_suffix=None,
        )

    def _merge_elicitation(
        self,
        first: GenerationResult,
        second: GenerationResult,
        suffix: str,
    ) -> GenerationResult:
        """Merge first and second elicitation results."""
        merged_text = (first.text or "") + suffix + (second.text or "")
        merged_meta = {**first.metadata, **second.metadata, "second_elicitation": True}
        return GenerationResult(
            text=merged_text,
            finish_reason=second.finish_reason,
            tool_calls=second.tool_calls,
            token_logprobs=None,
            prompt_tokens=first.prompt_tokens + second.prompt_tokens,
            completion_tokens=first.completion_tokens + second.completion_tokens,
            metadata=merged_meta,
        )

    def _second_elicitation(
        self,
        messages: list[ChatMessage],
        first_result: GenerationResult,
    ) -> GenerationResult:
        """Perform a follow-up generation to rescue a truncated output."""
        suffix = self._resolve_elicitation_suffix()
        continued = self._build_elicitation_messages(messages, first_result, suffix)
        params = self._elicitation_params()
        second = self.backend.generate_chat(continued, params)
        return self._merge_elicitation(first_result, second, suffix)

    def _generate_action(
        self,
        state: State[Any],
        trajectory: Trajectory[Any] | None = None,
    ) -> tuple[Action, GenerationResult]:
        """Generate an action (model response) for the current state.

        Uses generate_with_tools if the backend supports it and tools are available.

        Args:
            state: Current environment state.
            trajectory: Optional trajectory for structured message building.

        Returns:
            Tuple of (Action, GenerationResult).
        """
        messages = self._build_messages(state, trajectory=trajectory)
        tools = list(state.observation.available_tools)

        if tools and self.backend.capabilities.supports_function_calling:
            # Native function calling (API backends)
            result = self.backend.generate_with_tools(messages, tools, self.sampling_params)
        elif tools and self.tool_call_parser:
            # Text-based tool calling (vLLM/HF with parser)
            result = self._generate_with_text_tools(messages, tuple(tools))
        else:
            if tools:
                logger.warning(
                    "Environment provides %d tools but backend '%s' does not "
                    "support function calling and no tool_call_parser is "
                    "configured. Tools will be ignored.",
                    len(tools),
                    type(self.backend).__name__,
                )
            result = self.backend.generate_chat(messages, self.sampling_params)

        if (
            result.finish_reason == StopReason.MAX_TOKENS
            and self.sampling_params.second_elicitation_suffix is not None
        ):
            result = self._second_elicitation(messages, result)

        return result.to_agent_action(), result

    @staticmethod
    def _inject_tools_in_messages(
        messages: list[ChatMessage], tools_text: str
    ) -> list[ChatMessage]:
        """Inject tool definitions text into the message list.

        If a system message exists (first message), appends the tools
        text to it. Otherwise inserts a new system message at position 0.
        """
        messages = list(messages)
        if messages and messages[0].role == "system":
            original = messages[0].content or ""
            messages[0] = ChatMessage(
                role="system",
                content=f"{original}\n\n{tools_text}" if original else tools_text,
            )
        else:
            messages.insert(0, ChatMessage(role="system", content=tools_text))
        return messages

    def _generate_with_text_tools(
        self,
        messages: list[ChatMessage],
        tools: tuple[ToolDefinition, ...],
    ) -> GenerationResult:
        """Generate with text-based tool calling.

        Injects formatted tool definitions into the system message,
        generates text, then parses tool calls from the output.
        """
        assert self.tool_call_parser is not None

        tools_text = self.tool_call_parser.format_tools(tools)
        modified = self._inject_tools_in_messages(messages, tools_text)
        gen_result = self.backend.generate_chat(modified, self.sampling_params)

        parsed = self.tool_call_parser.parse(gen_result.text or "", tools)

        return GenerationResult(
            text=parsed.text,
            finish_reason=gen_result.finish_reason,
            tool_calls=parsed.tool_calls,
            token_logprobs=gen_result.token_logprobs,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            metadata=gen_result.metadata,
        )

    def run_trajectory(
        self,
        task_index: int,
        trajectory_id: str | None = None,
        max_steps: int | None = None,
    ) -> TrajectoryResult:
        """Run a single trajectory.

        Args:
            task_index: Index of the task in the environment.
            trajectory_id: Optional custom trajectory ID.
            max_steps: Maximum steps (overrides environment spec).

        Returns:
            TrajectoryResult with trajectory and metrics.
        """
        eval_logger: _EvalLogger | None = None
        if self.log is not None:
            eval_logger = _EvalLogger(self.log, self.environment.spec.name)

        try:
            return self._run_trajectory_impl(
                task_index,
                trajectory_id,
                max_steps,
                eval_logger,
            )
        finally:
            if eval_logger:
                eval_logger.close()

    def _run_trajectory_impl(
        self,
        task_index: int,
        trajectory_id: str | None,
        max_steps: int | None,
        eval_logger: _EvalLogger | None,
    ) -> TrajectoryResult:
        # Reset environment
        options = {"task_index": task_index}
        if trajectory_id:
            options["episode_id"] = trajectory_id

        state, reset_info = self.environment.reset(options=options)
        trajectory: Trajectory[Any] = Trajectory.create(state)

        max_steps = max_steps or self.environment.spec.max_steps or 100

        # Run trajectory loop
        step_count = 0
        while not state.metadata.is_terminal and step_count < max_steps:
            # Generate action
            action, gen_result = self._generate_action(state, trajectory=trajectory)

            # Take step (apply transition function)
            step_result = self.environment.step(state, action)

            # Record transition
            gen_info: dict[str, Any] = {
                "prompt_tokens": gen_result.prompt_tokens,
                "completion_tokens": gen_result.completion_tokens,
                "finish_reason": gen_result.finish_reason.name,
            }
            if gen_result.has_tool_calls:
                gen_info["has_tool_calls"] = True
                gen_info["num_tool_calls"] = len(gen_result.tool_calls)

            transition: Transition[Any] = Transition(
                state=state,
                action=action,
                next_state=step_result.next_state,
                rewards=step_result.rewards,
                extracted_action=step_result.extracted_action,
                resolved_action=step_result.resolved_action,
                info={
                    "generation": gen_info,
                    "step": step_result.info,
                },
            )
            trajectory.add_transition(transition)

            state = step_result.next_state
            step_count += 1

            if eval_logger:
                eval_logger.on_step(
                    _StepEvent(
                        task_index=task_index,
                        step_num=step_count,
                        reward_total=step_result.rewards.total,
                        prompt_tokens=gen_result.prompt_tokens,
                        completion_tokens=gen_result.completion_tokens,
                        has_tool_calls=gen_result.has_tool_calls,
                        num_tool_calls=len(gen_result.tool_calls)
                        if gen_result.has_tool_calls
                        else 0,
                    )
                )

            if step_result.done:
                break

        # Determine success from OUTCOME-type reward
        success = False
        if trajectory.transitions:
            last_rewards = trajectory.transitions[-1].rewards
            outcome_rewards = last_rewards.by_type(RewardType.OUTCOME)
            if outcome_rewards:
                success = outcome_rewards[-1].reward >= 1.0

        if eval_logger:
            eval_logger.on_trajectory_end(
                _TrajectoryEndEvent(
                    task_index=task_index,
                    success=success,
                    total_reward=trajectory.total_reward,
                    num_steps=len(trajectory),
                    completed_count=1,
                    total_count=1,
                )
            )

        return TrajectoryResult(
            trajectory=trajectory,
            total_reward=trajectory.total_reward,
            success=success,
            metadata={
                "task_index": task_index,
                "num_steps": len(trajectory),
                "reset_info": reset_info,
            },
        )

    def run_batch(
        self,
        task_indices: list[int],
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Run a batch of trajectories with lockstep batched generation.

        All trajectories advance one step together in each iteration,
        batching inference calls via generate_chat_batch(). Trajectories
        that finish early drop out of subsequent batches.

        Args:
            task_indices: List of task indices to run.
            batch_size: Maximum trajectories per lockstep batch. When set,
                task_indices are chunked and each chunk is processed
                independently. None means all tasks in one batch.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            BatchResult with all trajectory results and aggregate metrics.
        """
        eval_logger: _EvalLogger | None = None
        if self.log is not None:
            eval_logger = _EvalLogger(self.log, self.environment.spec.name)

        try:
            return self._run_batch_impl(
                task_indices,
                batch_size,
                progress_callback,
                eval_logger,
            )
        finally:
            if eval_logger:
                eval_logger.close()

    def _run_batch_impl(
        self,
        task_indices: list[int],
        batch_size: int | None,
        progress_callback: Callable[[int, int], None] | None,
        eval_logger: _EvalLogger | None,
    ) -> BatchResult:
        if not task_indices:
            return _aggregate_results([])

        max_steps = self.environment.spec.max_steps or 100

        if eval_logger:
            eval_logger.on_batch_start(
                _BatchStartEvent(
                    num_tasks=len(task_indices),
                    environment_name=self.environment.spec.name,
                    max_steps=max_steps,
                )
            )

        if batch_size is not None and len(task_indices) > batch_size:
            result = _run_in_chunks(
                lambda indices, cb: self._run_batch_inner(indices, cb, eval_logger),
                task_indices,
                batch_size,
                progress_callback,
            )
        else:
            result = self._run_batch_inner(task_indices, progress_callback, eval_logger)

        if eval_logger:
            eval_logger.on_batch_end(
                _BatchEndEvent(
                    success_rate=result.success_rate,
                    mean_reward=result.mean_reward,
                    num_tasks=len(task_indices),
                )
            )

        return result

    def _run_batch_inner(
        self,
        task_indices: list[int],
        progress_callback: Callable[[int, int], None] | None,
        eval_logger: _EvalLogger | None,
    ) -> BatchResult:
        if not task_indices:
            return _aggregate_results([])

        if self.env_factory is not None and not self.environment.spec.pure_step:
            return self._run_batch_inner_multi_instance(
                task_indices,
                progress_callback,
                eval_logger,
            )

        max_steps = self.environment.spec.max_steps or 100
        total = len(task_indices)
        result_slots: list[TrajectoryResult | None] = [None] * total
        logger.debug(
            "Trajectory batch start: mode=single-instance total=%d max_steps=%d tasks=%s",
            total,
            max_steps,
            _summarize_task_indices(task_indices),
        )

        # Phase 1: Reset all tasks
        active: list[_ActiveTrajectory] = []
        consecutive_reset_errors = 0
        reset_phase_started_at = _runner_now_monotonic()
        for pos, task_index in enumerate(task_indices):
            logger.debug(
                "Trajectory reset start: task=%d position=%d",
                task_index,
                pos,
            )
            reset_started_at = _runner_now_monotonic()
            try:
                state, reset_info = self.environment.reset(options={"task_index": task_index})
                trajectory: Trajectory[Any] = Trajectory.create(state)
                active.append(
                    _ActiveTrajectory(
                        position=pos,
                        task_index=task_index,
                        state=state,
                        reset_info=reset_info,
                        trajectory=trajectory,
                    )
                )
                logger.debug(
                    "Trajectory reset done: task=%d duration=%.2fs",
                    task_index,
                    max(0.0, _runner_now_monotonic() - reset_started_at),
                )
                consecutive_reset_errors = 0
            except Exception as e:
                consecutive_reset_errors += 1
                logger.error("Error resetting task %d: %s", task_index, e)
                if eval_logger:
                    eval_logger.on_error(
                        _ErrorEvent(
                            task_index=task_index,
                            phase="reset",
                            error=str(e),
                        )
                    )
                result_slots[pos] = TrajectoryResult(
                    trajectory=Trajectory(
                        episode_id=f"error_{task_index}",
                        initial_state=State(
                            observation=Observation(prompt=""),
                            hidden=None,
                            metadata=_error_metadata(task_index),
                        ),
                    ),
                    total_reward=0.0,
                    success=False,
                    metadata={"error": str(e), "task_index": task_index},
                )
                if consecutive_reset_errors >= 3:
                    raise RuntimeError(
                        f"Environment appears broken: {consecutive_reset_errors} "
                        f"consecutive reset errors (last: task {task_index}: {e})"
                    ) from e

        reset_errors = total - len(active)
        logger.debug(
            "Trajectory reset phase complete: ready=%d reset_errors=%d duration=%.2fs",
            len(active),
            reset_errors,
            max(0.0, _runner_now_monotonic() - reset_phase_started_at),
        )

        # Phase 2: Lockstep generation
        completed_count = reset_errors
        round_index = 0
        while True:
            remaining = [t for t in active if not t.done]
            if not remaining:
                break
            round_index += 1
            round_started_at = _runner_now_monotonic()
            logger.debug(
                "Trajectory round %d start: active=%d done=%d/%d tasks=%s",
                round_index,
                len(remaining),
                reset_errors + sum(1 for t in active if t.done),
                total,
                _summarize_active_trajectories(remaining),
            )

            prompt_build_started_at = _runner_now_monotonic()
            messages_batch = [
                self._build_messages(t.state, trajectory=t.trajectory, task_index=t.task_index)
                for t in remaining
            ]
            logger.debug(
                "Trajectory round %d prompt build finished in %.2fs",
                round_index,
                max(0.0, _runner_now_monotonic() - prompt_build_started_at),
            )

            # Use tool calling if tools available and backend supports it
            first_obs = remaining[0].state.observation
            tools = list(first_obs.available_tools)
            use_native_tools = tools and self.backend.capabilities.supports_function_calling
            use_text_tools = tools and not use_native_tools and self.tool_call_parser is not None

            logger.debug(
                "Trajectory round %d generation start: tasks=%s",
                round_index,
                _summarize_active_trajectories(remaining),
            )
            generation_started_at = _runner_now_monotonic()
            if use_native_tools:
                gen_results = self.backend.generate_with_tools_batch(
                    messages_batch, tools, self.sampling_params
                )
            elif use_text_tools:
                assert self.tool_call_parser is not None
                tools_text = self.tool_call_parser.format_tools(tuple(tools))
                modified_batch = [
                    self._inject_tools_in_messages(msgs, tools_text) for msgs in messages_batch
                ]
                raw_results = self.backend.generate_chat_batch(modified_batch, self.sampling_params)
                gen_results = []
                for raw in raw_results:
                    parsed = self.tool_call_parser.parse(raw.text or "", tuple(tools))
                    gen_results.append(
                        GenerationResult(
                            text=parsed.text,
                            finish_reason=raw.finish_reason,
                            tool_calls=parsed.tool_calls,
                            token_logprobs=raw.token_logprobs,
                            prompt_tokens=raw.prompt_tokens,
                            completion_tokens=raw.completion_tokens,
                            metadata=raw.metadata,
                        )
                    )
            else:
                if tools and not hasattr(self, "_batch_tool_warning_logged"):
                    logger.warning(
                        "Environment provides %d tools but backend '%s' does "
                        "not support function calling and no tool_call_parser "
                        "is configured. Tools will be ignored.",
                        len(tools),
                        type(self.backend).__name__,
                    )
                    self._batch_tool_warning_logged = True  # type: ignore[attr-defined]
                gen_results = self.backend.generate_chat_batch(messages_batch, self.sampling_params)
            logger.debug(
                "Trajectory round %d generation finished in %.2fs (results=%d)",
                round_index,
                max(0.0, _runner_now_monotonic() - generation_started_at),
                len(gen_results),
            )

            # Second elicitation for truncated outputs in batch
            if self.sampling_params.second_elicitation_suffix is not None:
                needs_elicitation = [
                    (i, gen)
                    for i, gen in enumerate(gen_results)
                    if gen.finish_reason == StopReason.MAX_TOKENS
                ]
                if needs_elicitation:
                    elicitation_started_at = _runner_now_monotonic()
                    logger.debug(
                        "Trajectory round %d second elicitation start: count=%d",
                        round_index,
                        len(needs_elicitation),
                    )
                    suffix = self._resolve_elicitation_suffix()
                    elicitation_msgs = [
                        self._build_elicitation_messages(messages_batch[i], gen, suffix)
                        for i, gen in needs_elicitation
                    ]
                    elicitation_params = self._elicitation_params()
                    elicitation_results = self.backend.generate_chat_batch(
                        elicitation_msgs, elicitation_params
                    )
                    for (i, first), second in zip(needs_elicitation, elicitation_results):
                        gen_results[i] = self._merge_elicitation(first, second, suffix)
                    logger.debug(
                        "Trajectory round %d second elicitation finished in %.2fs (count=%d)",
                        round_index,
                        max(0.0, _runner_now_monotonic() - elicitation_started_at),
                        len(needs_elicitation),
                    )

            for t, gen_result in zip(remaining, gen_results):
                step_started_at = _runner_now_monotonic()
                try:
                    action = gen_result.to_agent_action()
                    logger.debug(
                        "Trajectory round %d step start: task=%d env_step=%d",
                        round_index,
                        t.task_index,
                        t.step_count + 1,
                    )
                    step_result = self.environment.step(t.state, action)
                    step_elapsed_sec = max(0.0, _runner_now_monotonic() - step_started_at)
                    logger.debug(
                        "Trajectory round %d step done: task=%d duration=%.2fs done=%s truncated=%s reward=%.4f",
                        round_index,
                        t.task_index,
                        step_elapsed_sec,
                        step_result.done,
                        step_result.truncated,
                        step_result.rewards.total,
                    )

                    gen_info: dict[str, Any] = {
                        "prompt_tokens": gen_result.prompt_tokens,
                        "completion_tokens": gen_result.completion_tokens,
                        "finish_reason": gen_result.finish_reason.name,
                    }
                    if gen_result.has_tool_calls:
                        gen_info["has_tool_calls"] = True
                        gen_info["num_tool_calls"] = len(gen_result.tool_calls)

                    transition: Transition[Any] = Transition(
                        state=t.state,
                        action=action,
                        next_state=step_result.next_state,
                        rewards=step_result.rewards,
                        extracted_action=step_result.extracted_action,
                        resolved_action=step_result.resolved_action,
                        info={
                            "generation": gen_info,
                            "step": step_result.info,
                        },
                    )
                    t.trajectory.add_transition(transition)
                    t.state = step_result.next_state
                    t.step_count += 1

                    if eval_logger:
                        eval_logger.on_step(
                            _StepEvent(
                                task_index=t.task_index,
                                step_num=t.step_count,
                                reward_total=step_result.rewards.total,
                                prompt_tokens=gen_result.prompt_tokens,
                                completion_tokens=gen_result.completion_tokens,
                                has_tool_calls=gen_result.has_tool_calls,
                                num_tool_calls=len(gen_result.tool_calls)
                                if gen_result.has_tool_calls
                                else 0,
                            )
                        )

                    if step_result.done or t.step_count >= max_steps:
                        t.done = True
                        completed_count += 1
                        if eval_logger:
                            result = _finalize_trajectory(t)
                            eval_logger.on_trajectory_end(
                                _TrajectoryEndEvent(
                                    task_index=t.task_index,
                                    success=result.success,
                                    total_reward=result.total_reward,
                                    num_steps=len(t.trajectory),
                                    completed_count=completed_count,
                                    total_count=total,
                                )
                            )
                except Exception as e:
                    logger.error("Error stepping task %d: %s", t.task_index, e)
                    logger.debug(
                        "Trajectory round %d step failed: task=%d duration=%.2fs",
                        round_index,
                        t.task_index,
                        max(0.0, _runner_now_monotonic() - step_started_at),
                    )
                    t.done = True
                    t.error = str(e)
                    completed_count += 1
                    if eval_logger:
                        eval_logger.on_error(
                            _ErrorEvent(
                                task_index=t.task_index,
                                phase="step",
                                error=str(e),
                            )
                        )

            if progress_callback:
                done_count = reset_errors + sum(1 for t in active if t.done)
                progress_callback(done_count, total)
            else:
                done_count = reset_errors + sum(1 for t in active if t.done)
            logger.debug(
                "Trajectory round %d complete: duration=%.2fs done=%d/%d remaining=%d",
                round_index,
                max(0.0, _runner_now_monotonic() - round_started_at),
                done_count,
                total,
                sum(1 for t in active if not t.done),
            )

        # Phase 3: Build results
        for t in active:
            result_slots[t.position] = _finalize_trajectory(t)

        if progress_callback:
            progress_callback(total, total)

        return _aggregate_results([r for r in result_slots if r is not None])

    def _run_batch_inner_multi_instance(
        self,
        task_indices: list[int],
        progress_callback: Callable[[int, int], None] | None,
        eval_logger: _EvalLogger | None,
    ) -> BatchResult:
        """Run batch trajectories with one environment instance per task."""
        if not task_indices:
            return _aggregate_results([])

        assert self.env_factory is not None

        max_steps = self.environment.spec.max_steps or 100
        total = len(task_indices)
        result_slots: list[TrajectoryResult | None] = [None] * total
        logger.debug(
            "Trajectory batch start: mode=multi-instance total=%d max_steps=%d tasks=%s",
            total,
            max_steps,
            _summarize_task_indices(task_indices),
        )

        envs: list[Environment[Any]] = []
        active: list[_ActiveTrajectory] = []
        try:
            # Phase 1: Reset all tasks with dedicated env instances
            reset_phase_started_at = _runner_now_monotonic()
            for pos, task_index in enumerate(task_indices):
                logger.debug(
                    "Trajectory reset start: task=%d position=%d",
                    task_index,
                    pos,
                )
                reset_started_at = _runner_now_monotonic()
                env = self.env_factory()
                envs.append(env)
                try:
                    state, reset_info = env.reset(options={"task_index": task_index})
                    trajectory: Trajectory[Any] = Trajectory.create(state)
                    active.append(
                        _ActiveTrajectory(
                            position=pos,
                            task_index=task_index,
                            state=state,
                            reset_info=reset_info,
                            trajectory=trajectory,
                        )
                    )
                    logger.debug(
                        "Trajectory reset done: task=%d duration=%.2fs",
                        task_index,
                        max(0.0, _runner_now_monotonic() - reset_started_at),
                    )
                except Exception as e:
                    logger.error("Error resetting task %d: %s", task_index, e)
                    if eval_logger:
                        eval_logger.on_error(
                            _ErrorEvent(
                                task_index=task_index,
                                phase="reset",
                                error=str(e),
                            )
                        )
                    result_slots[pos] = TrajectoryResult(
                        trajectory=Trajectory(
                            episode_id=f"error_{task_index}",
                            initial_state=State(
                                observation=Observation(prompt=""),
                                hidden=None,
                                metadata=_error_metadata(task_index),
                            ),
                        ),
                        total_reward=0.0,
                        success=False,
                        metadata={"error": str(e), "task_index": task_index},
                    )

            reset_errors = total - len(active)
            logger.debug(
                "Trajectory reset phase complete: ready=%d reset_errors=%d duration=%.2fs",
                len(active),
                reset_errors,
                max(0.0, _runner_now_monotonic() - reset_phase_started_at),
            )

            # Phase 2: Lockstep generation
            completed_count = reset_errors
            round_index = 0
            while True:
                remaining = [t for t in active if not t.done]
                if not remaining:
                    break
                round_index += 1
                round_started_at = _runner_now_monotonic()
                logger.debug(
                    "Trajectory round %d start: active=%d done=%d/%d tasks=%s",
                    round_index,
                    len(remaining),
                    reset_errors + sum(1 for t in active if t.done),
                    total,
                    _summarize_active_trajectories(remaining),
                )

                prompt_build_started_at = _runner_now_monotonic()
                messages_batch = [
                    self._build_messages(
                        t.state, trajectory=t.trajectory, task_index=t.task_index,
                    )
                    for t in remaining
                ]
                logger.debug(
                    "Trajectory round %d prompt build finished in %.2fs",
                    round_index,
                    max(0.0, _runner_now_monotonic() - prompt_build_started_at),
                )

                # Use tool calling if tools available and backend supports it
                first_obs = remaining[0].state.observation
                tools = list(first_obs.available_tools)
                use_native_tools = tools and self.backend.capabilities.supports_function_calling
                use_text_tools = tools and not use_native_tools and self.tool_call_parser is not None

                logger.debug(
                    "Trajectory round %d generation start: tasks=%s",
                    round_index,
                    _summarize_active_trajectories(remaining),
                )
                generation_started_at = _runner_now_monotonic()
                if use_native_tools:
                    gen_results = self.backend.generate_with_tools_batch(
                        messages_batch, tools, self.sampling_params
                    )
                elif use_text_tools:
                    assert self.tool_call_parser is not None
                    tools_text = self.tool_call_parser.format_tools(tuple(tools))
                    modified_batch = [
                        self._inject_tools_in_messages(msgs, tools_text)
                        for msgs in messages_batch
                    ]
                    raw_results = self.backend.generate_chat_batch(
                        modified_batch, self.sampling_params
                    )
                    gen_results = []
                    for raw in raw_results:
                        parsed = self.tool_call_parser.parse(raw.text or "", tuple(tools))
                        gen_results.append(
                            GenerationResult(
                                text=parsed.text,
                                finish_reason=raw.finish_reason,
                                tool_calls=parsed.tool_calls,
                                token_logprobs=raw.token_logprobs,
                                prompt_tokens=raw.prompt_tokens,
                                completion_tokens=raw.completion_tokens,
                                metadata=raw.metadata,
                            )
                        )
                else:
                    if tools and not hasattr(self, "_batch_tool_warning_logged"):
                        logger.warning(
                            "Environment provides %d tools but backend '%s' does "
                            "not support function calling and no tool_call_parser "
                            "is configured. Tools will be ignored.",
                            len(tools),
                            type(self.backend).__name__,
                        )
                        self._batch_tool_warning_logged = True  # type: ignore[attr-defined]
                    gen_results = self.backend.generate_chat_batch(
                        messages_batch, self.sampling_params
                    )
                logger.debug(
                    "Trajectory round %d generation finished in %.2fs (results=%d)",
                    round_index,
                    max(0.0, _runner_now_monotonic() - generation_started_at),
                    len(gen_results),
                )

                # Second elicitation for truncated outputs in batch
                if self.sampling_params.second_elicitation_suffix is not None:
                    needs_elicitation = [
                        (i, gen)
                        for i, gen in enumerate(gen_results)
                        if gen.finish_reason == StopReason.MAX_TOKENS
                    ]
                    if needs_elicitation:
                        elicitation_started_at = _runner_now_monotonic()
                        logger.debug(
                            "Trajectory round %d second elicitation start: count=%d",
                            round_index,
                            len(needs_elicitation),
                        )
                        suffix = self._resolve_elicitation_suffix()
                        elicitation_msgs = [
                            self._build_elicitation_messages(messages_batch[i], gen, suffix)
                            for i, gen in needs_elicitation
                        ]
                        elicitation_params = self._elicitation_params()
                        elicitation_results = self.backend.generate_chat_batch(
                            elicitation_msgs, elicitation_params
                        )
                        for (i, first), second in zip(needs_elicitation, elicitation_results):
                            gen_results[i] = self._merge_elicitation(first, second, suffix)
                        logger.debug(
                            "Trajectory round %d second elicitation finished in %.2fs (count=%d)",
                            round_index,
                            max(0.0, _runner_now_monotonic() - elicitation_started_at),
                            len(needs_elicitation),
                        )

                actions_for_step = [gen_result.to_agent_action() for gen_result in gen_results]
                with ThreadPoolExecutor() as executor:
                    step_started_at_by_pos: dict[int, float] = {}
                    step_futures = {
                        t.position: executor.submit(envs[t.position].step, t.state, action)
                        for t, action in zip(remaining, actions_for_step)
                    }
                    for t in remaining:
                        step_started_at_by_pos[t.position] = _runner_now_monotonic()
                        logger.debug(
                            "Trajectory round %d step submitted: task=%d env_step=%d",
                            round_index,
                            t.task_index,
                            t.step_count + 1,
                        )

                    for t, gen_result, action in zip(
                        remaining,
                        gen_results,
                        actions_for_step,
                    ):
                        step_started_at = step_started_at_by_pos[t.position]
                        logger.debug(
                            "Trajectory round %d waiting for step result: task=%d env_step=%d",
                            round_index,
                            t.task_index,
                            t.step_count + 1,
                        )
                        try:
                            step_result = step_futures[t.position].result()
                            step_elapsed_sec = max(
                                0.0, _runner_now_monotonic() - step_started_at
                            )
                            logger.debug(
                                "Trajectory round %d step done: task=%d duration=%.2fs done=%s truncated=%s reward=%.4f",
                                round_index,
                                t.task_index,
                                step_elapsed_sec,
                                step_result.done,
                                step_result.truncated,
                                step_result.rewards.total,
                            )

                            gen_info: dict[str, Any] = {
                                "prompt_tokens": gen_result.prompt_tokens,
                                "completion_tokens": gen_result.completion_tokens,
                                "finish_reason": gen_result.finish_reason.name,
                            }
                            if gen_result.has_tool_calls:
                                gen_info["has_tool_calls"] = True
                                gen_info["num_tool_calls"] = len(gen_result.tool_calls)

                            transition: Transition[Any] = Transition(
                                state=t.state,
                                action=action,
                                next_state=step_result.next_state,
                                rewards=step_result.rewards,
                                extracted_action=step_result.extracted_action,
                                resolved_action=step_result.resolved_action,
                                info={
                                    "generation": gen_info,
                                    "step": step_result.info,
                                },
                            )
                            t.trajectory.add_transition(transition)
                            t.state = step_result.next_state
                            t.step_count += 1

                            if eval_logger:
                                eval_logger.on_step(
                                    _StepEvent(
                                        task_index=t.task_index,
                                        step_num=t.step_count,
                                        reward_total=step_result.rewards.total,
                                        prompt_tokens=gen_result.prompt_tokens,
                                        completion_tokens=gen_result.completion_tokens,
                                        has_tool_calls=gen_result.has_tool_calls,
                                        num_tool_calls=len(gen_result.tool_calls)
                                        if gen_result.has_tool_calls
                                        else 0,
                                    )
                                )

                            if step_result.done or t.step_count >= max_steps:
                                t.done = True
                                completed_count += 1
                                if eval_logger:
                                    result = _finalize_trajectory(t)
                                    eval_logger.on_trajectory_end(
                                        _TrajectoryEndEvent(
                                            task_index=t.task_index,
                                            success=result.success,
                                            total_reward=result.total_reward,
                                            num_steps=len(t.trajectory),
                                            completed_count=completed_count,
                                            total_count=total,
                                        )
                                    )
                        except Exception as e:
                            logger.error("Error stepping task %d: %s", t.task_index, e)
                            logger.debug(
                                "Trajectory round %d step failed: task=%d duration=%.2fs",
                                round_index,
                                t.task_index,
                                max(0.0, _runner_now_monotonic() - step_started_at),
                            )
                            t.done = True
                            t.error = str(e)
                            completed_count += 1
                            if eval_logger:
                                eval_logger.on_error(
                                    _ErrorEvent(
                                        task_index=t.task_index,
                                        phase="step",
                                        error=str(e),
                                    )
                                )

                if progress_callback:
                    done_count = reset_errors + sum(1 for t in active if t.done)
                    progress_callback(done_count, total)
                else:
                    done_count = reset_errors + sum(1 for t in active if t.done)
                logger.debug(
                    "Trajectory round %d complete: duration=%.2fs done=%d/%d remaining=%d",
                    round_index,
                    max(0.0, _runner_now_monotonic() - round_started_at),
                    done_count,
                    total,
                    sum(1 for t in active if not t.done),
                )

            # Phase 3: Build results
            for t in active:
                result_slots[t.position] = _finalize_trajectory(t)

            if progress_callback:
                progress_callback(total, total)

            return _aggregate_results([r for r in result_slots if r is not None])
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass

    # -----------------------------------------------------------------
    # Run-from-state API
    # -----------------------------------------------------------------

    def build_messages(
        self,
        state: State[Any],
        trajectory: Trajectory[Any] | None = None,
    ) -> list[ChatMessage]:
        """Build chat messages from a state.

        Public wrapper for the internal ``_build_messages`` pipeline.
        Applies structured/legacy mode selection, prompt template,
        model profile transformers, prompt pipeline, image truncation,
        and message coalescing.

        Args:
            state: Current environment state.
            trajectory: Optional trajectory for structured message building.

        Returns:
            List of ChatMessages for the backend.
        """
        return self._build_messages(state, trajectory=trajectory)

    def run_from_state(
        self,
        state: State[Any],
        max_steps: int | None = None,
    ) -> Trajectory[Any]:
        """Run a single rollout from an arbitrary state.

        Args:
            state: Starting state (may be mid-trajectory).
            max_steps: Maximum new steps to take. Defaults to the
                environment spec's ``max_steps``.

        Returns:
            Trajectory containing all transitions from the rollout.
        """
        env_max = self.environment.spec.max_steps or 100

        if max_steps is None:
            max_steps = env_max

        loop_max = max(0, min(max_steps, env_max - state.metadata.step))

        trajectory: Trajectory[Any] = Trajectory.create(state)
        current_state = state

        for _ in range(loop_max):
            if current_state.metadata.is_terminal or current_state.metadata.step >= env_max:
                break

            messages = self._build_messages(current_state, trajectory=trajectory)
            gen_result = self.backend.generate_chat(messages, self.sampling_params)
            action = gen_result.to_agent_action()
            step_result = self.environment.step(current_state, action)

            trajectory.add_transition(
                Transition(
                    state=current_state,
                    action=action,
                    next_state=step_result.next_state,
                    rewards=step_result.rewards,
                    extracted_action=step_result.extracted_action,
                    resolved_action=step_result.resolved_action,
                    info={
                        "step": step_result.info,
                    },
                )
            )
            current_state = step_result.next_state

        return trajectory

    def run_from_state_action(
        self,
        state: State[Any],
        action: Action,
        max_steps: int | None = None,
    ) -> Trajectory[Any]:
        """Run a single rollout with a forced first action.

        The first step uses the given ``action`` instead of generating
        one from the backend. Subsequent steps (if the episode continues)
        use normal generation.

        Args:
            state: Starting state.
            action: Action to force for the first step.
            max_steps: Maximum new steps to take. Defaults to the
                environment spec's ``max_steps``.

        Returns:
            Trajectory containing all transitions from the rollout.
        """
        env_max = self.environment.spec.max_steps or 100

        if max_steps is None:
            max_steps = env_max

        loop_max = max(0, min(max_steps, env_max - state.metadata.step))

        trajectory: Trajectory[Any] = Trajectory.create(state)
        current_state = state

        for step_i in range(loop_max):
            if current_state.metadata.is_terminal or current_state.metadata.step >= env_max:
                break

            if step_i == 0:
                step_action = action
            else:
                messages = self._build_messages(current_state, trajectory=trajectory)
                gen_result = self.backend.generate_chat(messages, self.sampling_params)
                step_action = gen_result.to_agent_action()

            step_result = self.environment.step(current_state, step_action)

            trajectory.add_transition(
                Transition(
                    state=current_state,
                    action=step_action,
                    next_state=step_result.next_state,
                    rewards=step_result.rewards,
                    extracted_action=step_result.extracted_action,
                    resolved_action=step_result.resolved_action,
                    info={
                        "step": step_result.info,
                    },
                )
            )
            current_state = step_result.next_state

        return trajectory

    def run_batch_from_states(
        self,
        states: Sequence[State[Any]],
        max_steps: int | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
        on_environment_error: Callable[[Exception], str] | None = None,
    ) -> list[Trajectory[Any] | None]:
        """Run batched rollouts from arbitrary states.

        Uses lockstep batched generation: all active rollouts advance
        one step together per iteration.

        Args:
            states: Starting states for each rollout.
            max_steps: Maximum new steps per rollout. Defaults to the
                environment spec's ``max_steps``.
            batch_size: If set, processes states in chunks of this size.
            progress_callback: Optional callback(completed, total).
            on_generation_error: Optional callback invoked when
                ``generate_chat_batch`` raises during a lockstep step.
                Must return ``"skip"`` (mark offending trajectories as
                failed and continue), ``"abort"`` (mark all active as
                failed, keep completed), or ``"raise"`` (re-raise).
                When ``None``, errors always propagate.
            on_environment_error: Optional callback invoked when a
                per-rollout restore or environment step raises in the
                multi-instance path. Must return ``"skip"`` (mark only
                the affected rollout as failed and continue) or
                ``"raise"`` (re-raise). Ignored for single-instance runs.

        Returns:
            List of Trajectory objects (or ``None`` for failed
            trajectories when *on_generation_error* is used), one per
            input state, in order.
        """
        if not states:
            self.last_environment_errors = {}
            return []

        if batch_size is not None and batch_size < len(states):
            results: list[Trajectory[Any] | None] = []
            total = len(states)
            aggregated_errors: dict[int, dict[str, Any]] = {}
            for start in range(0, total, batch_size):
                chunk = states[start : start + batch_size]

                chunk_progress_callback: Callable[[int, int], None] | None = None
                if progress_callback:
                    _offset = start

                    def _chunk_progress_callback(
                        done: int,
                        chunk_total: int,
                        _s: int = _offset,
                    ) -> None:
                        del chunk_total
                        progress_callback(_s + done, total)

                    chunk_progress_callback = _chunk_progress_callback

                results.extend(
                    self._run_batch_from_states_inner(
                        chunk,
                        max_steps=max_steps,
                        progress_callback=chunk_progress_callback,
                        on_generation_error=on_generation_error,
                        on_environment_error=on_environment_error,
                    )
                )
                for pos, info in self.last_environment_errors.items():
                    aggregated_errors[start + pos] = {
                        **info,
                        "position": start + pos,
                    }
            self.last_environment_errors = aggregated_errors
            if progress_callback:
                progress_callback(total, total)
            return results

        return self._run_batch_from_states_inner(
            states,
            max_steps=max_steps,
            progress_callback=progress_callback,
            on_generation_error=on_generation_error,
            on_environment_error=on_environment_error,
        )

    def run_batch_from_state_actions(
        self,
        states: Sequence[State[Any]],
        actions: Sequence[Action],
        max_steps: int | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
        on_environment_error: Callable[[Exception], str] | None = None,
    ) -> list[Trajectory[Any] | None]:
        """Run batched rollouts with forced first actions.

        The first step of each rollout uses the corresponding action from
        ``actions`` instead of generating from the backend. Subsequent
        steps use normal generation.

        Args:
            states: Starting states for each rollout.
            actions: Actions to force for the first step of each rollout.
            max_steps: Maximum new steps per rollout. Defaults to the
                environment spec's ``max_steps``.
            batch_size: If set, processes states in chunks of this size.
            progress_callback: Optional callback(completed, total).
            on_generation_error: Same as ``run_batch_from_states``.
            on_environment_error: Same as ``run_batch_from_states``.

        Returns:
            List of Trajectory objects (or ``None`` for failed
            trajectories), one per input state, in order.

        Raises:
            ValueError: If ``states`` and ``actions`` have different lengths.
        """
        if len(states) != len(actions):
            raise ValueError(
                f"states and actions must have the same length, "
                f"got {len(states)} and {len(actions)}"
            )

        if not states:
            self.last_environment_errors = {}
            return []

        if batch_size is not None and batch_size < len(states):
            indexed = list(zip(states, actions, strict=False))
            results: list[Trajectory[Any] | None] = []
            total = len(indexed)
            aggregated_errors: dict[int, dict[str, Any]] = {}
            for start in range(0, total, batch_size):
                chunk = indexed[start : start + batch_size]

                chunk_progress_callback: Callable[[int, int], None] | None = None
                if progress_callback:
                    _offset = start

                    def _chunk_progress_callback(
                        done: int,
                        chunk_total: int,
                        _s: int = _offset,
                    ) -> None:
                        del chunk_total
                        progress_callback(_s + done, total)

                    chunk_progress_callback = _chunk_progress_callback

                results.extend(
                    self._run_batch_from_states_inner(
                        [state for state, _action in chunk],
                        forced_actions=[action for _state, action in chunk],
                        max_steps=max_steps,
                        progress_callback=chunk_progress_callback,
                        on_generation_error=on_generation_error,
                        on_environment_error=on_environment_error,
                    )
                )
                for pos, info in self.last_environment_errors.items():
                    aggregated_errors[start + pos] = {
                        **info,
                        "position": start + pos,
                    }
            self.last_environment_errors = aggregated_errors
            if progress_callback:
                progress_callback(total, total)
            return results

        return self._run_batch_from_states_inner(
            states,
            forced_actions=actions,
            max_steps=max_steps,
            progress_callback=progress_callback,
            on_generation_error=on_generation_error,
            on_environment_error=on_environment_error,
        )

    def _run_batch_from_states_multi_instance(
        self,
        states: Sequence[State[Any]],
        max_steps: int | None = None,
        forced_actions: Sequence[Action] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
        on_environment_error: Callable[[Exception], str] | None = None,
    ) -> list[Trajectory[Any] | None]:
        """Lockstep batch loop using per-rollout environment instances.

        Each rollout gets its own environment instance (via ``env_factory``)
        restored to the target state (via ``restore_fn``). LLM generation is
        batched across all rollouts; environment steps run in parallel via
        ThreadPoolExecutor (I/O-bound container operations).

        Args:
            states: Starting states for each rollout.
            max_steps: Maximum new steps per rollout.
            forced_actions: If set, use these for the first step of each
                rollout instead of generating from the backend.
            progress_callback: Optional callback(completed, total).
        """
        assert self.env_factory is not None
        assert self.restore_fn is not None
        self.last_environment_errors = {}

        env_max = self.environment.spec.max_steps or 100

        if max_steps is None:
            max_steps = env_max

        if not states:
            return []

        loop_max = max(max(0, min(max_steps, env_max - s.metadata.step)) for s in states)
        total = len(states)

        # Create and restore environment instances
        envs: list[Environment[Any]] = []
        active: list[_ActiveTrajectory] = []

        def _record_environment_error(
            *,
            position: int,
            task_index: int,
            phase: str,
            error: Exception,
            action: Action | None = None,
        ) -> None:
            payload: dict[str, Any] = {
                "position": position,
                "task_index": task_index,
                "phase": phase,
                "error": str(error),
                "error_type": type(error).__name__,
            }
            if action is not None and action.text is not None:
                payload["action_text"] = action.text
            self.last_environment_errors[position] = payload

        def _handle_environment_error(
            *,
            trajectory: _ActiveTrajectory,
            phase: str,
            error: Exception,
            action: Action | None = None,
        ) -> str:
            _record_environment_error(
                position=trajectory.position,
                task_index=trajectory.task_index,
                phase=phase,
                error=error,
                action=action,
            )
            if on_environment_error is None:
                raise error
            decision = on_environment_error(error)
            if decision not in {"skip", "raise"}:
                raise ValueError(
                    "on_environment_error must return 'skip' or 'raise'"
                )
            if decision == "raise":
                raise error
            trajectory.done = True
            trajectory.failed = True
            trajectory.error = str(error)
            return decision

        try:
            # Create env instances
            for _i in range(total):
                envs.append(self.env_factory())

            for pos, state in enumerate(states):
                trajectory: Trajectory[Any] = Trajectory.create(state)
                active.append(
                    _ActiveTrajectory(
                        position=pos,
                        task_index=_task_index_for_state(state),
                        state=state,
                        reset_info={},
                        trajectory=trajectory,
                    )
                )

            # Restore each env to the target state in parallel (I/O-bound)
            with ThreadPoolExecutor() as executor:
                futures = {
                    pos: executor.submit(self.restore_fn, envs[pos], states[pos])
                    for pos in range(total)
                }
                for pos in range(total):
                    try:
                        restored = futures[pos].result()
                    except Exception as exc:
                        _handle_environment_error(
                            trajectory=active[pos],
                            phase="restore",
                            error=exc,
                        )
                        continue
                    active[pos].state = restored
                    active[pos].trajectory = Trajectory.create(restored)

            # Mark already-terminal states as done
            for t in active:
                if t.failed:
                    continue
                if t.state.metadata.is_terminal or t.state.metadata.step >= env_max:
                    t.done = True

            if progress_callback:
                progress_callback(sum(1 for t in active if t.done), total)

            # Lockstep loop
            for step_i in range(loop_max):
                remaining = [t for t in active if not t.done]
                if not remaining:
                    break

                # First step with forced actions: skip generation
                if step_i == 0 and forced_actions is not None:
                    with ThreadPoolExecutor() as executor:
                        step_futures = {
                            t.position: executor.submit(
                                envs[t.position].step,
                                t.state,
                                forced_actions[t.position],
                            )
                            for t in remaining
                        }
                        for t in remaining:
                            try:
                                step_result = step_futures[t.position].result()
                            except Exception as exc:
                                _handle_environment_error(
                                    trajectory=t,
                                    phase="step",
                                    error=exc,
                                    action=forced_actions[t.position],
                                )
                                continue
                            t.trajectory.add_transition(
                                Transition(
                                    state=t.state,
                                    action=forced_actions[t.position],
                                    next_state=step_result.next_state,
                                    rewards=step_result.rewards,
                                    extracted_action=step_result.extracted_action,
                                    resolved_action=step_result.resolved_action,
                                    info={"step": step_result.info},
                                )
                            )
                            t.state = step_result.next_state
                            t.step_count += 1
                            if step_result.done or t.state.metadata.step >= env_max:
                                t.done = True

                if step_i != 0 or forced_actions is None:
                    # Generate actions via batched inference
                    messages_batch = [
                        self._build_messages(
                            t.state, trajectory=t.trajectory, task_index=t.task_index,
                        )
                        for t in remaining
                    ]
                    try:
                        gen_results = self.backend.generate_chat_batch(
                            messages_batch, self.sampling_params
                        )
                    except Exception as exc:
                        if on_generation_error is None:
                            raise
                        decision = on_generation_error(exc)
                        if decision == "raise":
                            raise
                        offending = getattr(exc, "offending_indices", None)
                        if decision == "skip" and offending:
                            for idx in offending:
                                remaining[idx].done = True
                                remaining[idx].failed = True
                            remaining = [t for t in remaining if not t.failed]
                            if not remaining:
                                break
                            messages_batch = [
                                self._build_messages(
                                    t.state, trajectory=t.trajectory, task_index=t.task_index,
                                )
                                for t in remaining
                            ]
                            try:
                                gen_results = self.backend.generate_chat_batch(
                                    messages_batch, self.sampling_params,
                                )
                            except Exception as retry_exc:
                                # Re-classify: non-recoverable errors must propagate
                                if on_generation_error is not None:
                                    retry_decision = on_generation_error(retry_exc)
                                    if retry_decision == "raise":
                                        raise
                                # Recoverable retry failure — abort all remaining
                                for t in remaining:
                                    t.done = True
                                    t.failed = True
                                break
                        else:
                            for t in remaining:
                                t.done = True
                                t.failed = True
                            break

                    # Execute steps per-env in parallel
                    actions_for_step = [gen_result.to_agent_action() for gen_result in gen_results]
                    with ThreadPoolExecutor() as executor:
                        step_futures_gen = {
                            t.position: executor.submit(envs[t.position].step, t.state, action)
                            for t, action in zip(remaining, actions_for_step)
                        }
                        for t, action in zip(remaining, actions_for_step):
                            try:
                                step_result = step_futures_gen[t.position].result()
                            except Exception as exc:
                                _handle_environment_error(
                                    trajectory=t,
                                    phase="step",
                                    error=exc,
                                    action=action,
                                )
                                continue
                            t.trajectory.add_transition(
                                Transition(
                                    state=t.state,
                                    action=action,
                                    next_state=step_result.next_state,
                                    rewards=step_result.rewards,
                                    extracted_action=step_result.extracted_action,
                                    resolved_action=step_result.resolved_action,
                                    info={"step": step_result.info},
                                )
                            )
                            t.state = step_result.next_state
                            t.step_count += 1
                            if step_result.done or t.state.metadata.step >= env_max:
                                t.done = True

                if progress_callback:
                    progress_callback(sum(1 for t in active if t.done), total)

            # Return trajectories in original order
            result_slots: list[Trajectory[Any] | None] = [None] * total
            for t in active:
                if t.failed:
                    result_slots[t.position] = None
                else:
                    result_slots[t.position] = t.trajectory

            if progress_callback:
                progress_callback(total, total)

            if on_generation_error is not None or on_environment_error is not None:
                return result_slots
            return [r for r in result_slots if r is not None]

        finally:
            # Ensure all env instances are closed
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass

    def _run_batch_from_states_inner(
        self,
        states: Sequence[State[Any]],
        max_steps: int | None = None,
        forced_actions: Sequence[Action] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
        on_environment_error: Callable[[Exception], str] | None = None,
    ) -> list[Trajectory[Any] | None]:
        """Core lockstep batch loop for run-from-state methods.

        Args:
            states: Starting states for each rollout.
            max_steps: Maximum new steps per rollout.
            forced_actions: If set, use these for the first step of each
                rollout instead of generating from the backend.
        """
        # Dispatch to multi-instance path for non-pure environments with factory
        self.last_environment_errors = {}
        if self.env_factory is not None and not self.environment.spec.pure_step:
            return self._run_batch_from_states_multi_instance(
                states,
                max_steps=max_steps,
                forced_actions=forced_actions,
                progress_callback=progress_callback,
                on_generation_error=on_generation_error,
                on_environment_error=on_environment_error,
            )

        env_max = self.environment.spec.max_steps or 100

        if max_steps is None:
            max_steps = env_max

        if not states:
            loop_max = 0
        else:
            loop_max = max(max(0, min(max_steps, env_max - s.metadata.step)) for s in states)

        # Build active rollout state
        active: list[_ActiveTrajectory] = []
        for pos, state in enumerate(states):
            trajectory: Trajectory[Any] = Trajectory.create(state)
            active.append(
                _ActiveTrajectory(
                    position=pos,
                    task_index=_task_index_for_state(state),
                    state=state,
                    reset_info={},
                    trajectory=trajectory,
                )
            )

        # Mark already-terminal states as done
        for t in active:
            if t.state.metadata.is_terminal or t.state.metadata.step >= env_max:
                t.done = True

        total = len(states)
        if progress_callback and active:
            progress_callback(sum(1 for t in active if t.done), total)

        for step_i in range(loop_max):
            remaining = [t for t in active if not t.done]
            if not remaining:
                break

            # First step with forced actions: skip generation
            if step_i == 0 and forced_actions is not None:
                for t in remaining:
                    forced = forced_actions[t.position]
                    step_result = self.environment.step(t.state, forced)

                    t.trajectory.add_transition(
                        Transition(
                            state=t.state,
                            action=forced,
                            next_state=step_result.next_state,
                            rewards=step_result.rewards,
                            extracted_action=step_result.extracted_action,
                            resolved_action=step_result.resolved_action,
                            info={"step": step_result.info},
                        )
                    )
                    t.state = step_result.next_state
                    t.step_count += 1
                    if step_result.done or t.state.metadata.step >= env_max:
                        t.done = True

            if step_i != 0 or forced_actions is None:
                # Generate actions via batched inference
                messages_batch = [
                    self._build_messages(
                        t.state, trajectory=t.trajectory, task_index=t.task_index,
                    )
                    for t in remaining
                ]
                try:
                    gen_results = self.backend.generate_chat_batch(
                        messages_batch, self.sampling_params,
                    )
                except Exception as exc:
                    if on_generation_error is None:
                        raise
                    decision = on_generation_error(exc)
                    if decision == "raise":
                        raise
                    offending = getattr(exc, "offending_indices", None)
                    if decision == "skip" and offending:
                        for idx in offending:
                            remaining[idx].done = True
                            remaining[idx].failed = True
                        # Re-filter remaining and continue lockstep
                        remaining = [t for t in remaining if not t.failed]
                        if not remaining:
                            break
                        # Rebuild messages and retry this step
                        messages_batch = [
                            self._build_messages(
                                t.state, trajectory=t.trajectory, task_index=t.task_index,
                            )
                            for t in remaining
                        ]
                        try:
                            gen_results = self.backend.generate_chat_batch(
                                messages_batch, self.sampling_params,
                            )
                        except Exception as retry_exc:
                            # Re-classify: non-recoverable errors must propagate
                            if on_generation_error is not None:
                                retry_decision = on_generation_error(retry_exc)
                                if retry_decision == "raise":
                                    raise
                            # Recoverable retry failure — abort all remaining
                            for t in remaining:
                                t.done = True
                                t.failed = True
                            break
                    else:
                        # "abort" or "skip" without offending info
                        for t in remaining:
                            t.done = True
                            t.failed = True
                        break

                for t, gen_result in zip(remaining, gen_results):
                    action = gen_result.to_agent_action()
                    step_result = self.environment.step(t.state, action)

                    t.trajectory.add_transition(
                        Transition(
                            state=t.state,
                            action=action,
                            next_state=step_result.next_state,
                            rewards=step_result.rewards,
                            extracted_action=step_result.extracted_action,
                            resolved_action=step_result.resolved_action,
                            info={"step": step_result.info},
                        )
                    )
                    t.state = step_result.next_state
                    t.step_count += 1
                    if step_result.done or t.state.metadata.step >= env_max:
                        t.done = True

            if progress_callback:
                progress_callback(sum(1 for t in active if t.done), total)

        # Return trajectories in original order
        result_slots: list[Trajectory[Any] | None] = [None] * len(states)
        for t in active:
            if t.failed:
                result_slots[t.position] = None
            else:
                result_slots[t.position] = t.trajectory

        if progress_callback:
            progress_callback(total, total)

        if on_generation_error is not None:
            return result_slots
        return [r for r in result_slots if r is not None]


@dataclass(frozen=True)
class MultiEvalEntry:
    """One environment + its task indices for multi-environment batching."""

    runner: TrajectoryRunner
    task_indices: list[int]


@dataclass
class _MultiActiveTrajectory:
    """Wraps _ActiveTrajectory with its entry index and runner."""

    entry_index: int
    inner: _ActiveTrajectory
    runner: TrajectoryRunner


def _run_multi_lockstep(
    trajectories: list[_MultiActiveTrajectory],
    backend: ModelBackend,
    sampling_params: SamplingParams,
    max_steps_per_entry: dict[int, int],
    progress_callback: Callable[[int, int], None] | None = None,
    total_for_progress: int = 0,
    progress_offset: int = 0,
) -> None:
    """Run lockstep loop over pre-initialized multi-entry trajectories.

    Mutates trajectories in-place (marks done, records transitions).
    """
    while True:
        remaining = [t for t in trajectories if not t.inner.done]
        if not remaining:
            break

        messages_batch = [
            t.runner._build_messages(
                t.inner.state,
                trajectory=t.inner.trajectory,
                task_index=t.inner.task_index,
            )
            for t in remaining
        ]
        gen_results = backend.generate_chat_batch(messages_batch, sampling_params)

        for t, gen_result in zip(remaining, gen_results):
            try:
                action = gen_result.to_agent_action()
                step_result = t.runner.environment.step(t.inner.state, action)

                transition: Transition[Any] = Transition(
                    state=t.inner.state,
                    action=action,
                    next_state=step_result.next_state,
                    rewards=step_result.rewards,
                    extracted_action=step_result.extracted_action,
                    resolved_action=step_result.resolved_action,
                    info={
                        "generation": {
                            "prompt_tokens": gen_result.prompt_tokens,
                            "completion_tokens": gen_result.completion_tokens,
                            "finish_reason": gen_result.finish_reason.name,
                        },
                        "step": step_result.info,
                    },
                )
                t.inner.trajectory.add_transition(transition)
                t.inner.state = step_result.next_state
                t.inner.step_count += 1

                max_steps = max_steps_per_entry[t.entry_index]
                if step_result.done or t.inner.step_count >= max_steps:
                    t.inner.done = True
            except Exception as e:
                logger.error(f"Error stepping task {t.inner.task_index}: {e}")
                t.inner.done = True
                t.inner.error = str(e)

        if progress_callback:
            done_count = progress_offset + sum(1 for t in trajectories if t.inner.done)
            progress_callback(done_count, total_for_progress)


def run_multi_evaluation(
    entries: list[MultiEvalEntry],
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[BatchResult]:
    """Run cross-environment batched evaluation.

    Interleaves trajectories from multiple environments into a single
    lockstep loop, making one generate_chat_batch() call per step across
    all environments. This maximizes GPU utilization and API concurrency.

    All entries must share the same backend and sampling_params. Only
    TrajectoryRunner is supported (tool and segmented runners are not).

    Each entry's runner may have its own ``log`` configuration. Loggers
    are created per-entry and closed after results are finalized.

    Args:
        entries: List of MultiEvalEntry, each pairing a TrajectoryRunner
            with task indices.
        batch_size: Maximum trajectories per lockstep batch. When set,
            all trajectories (across all entries) are chunked and each
            chunk processed independently, then results merged per entry.
        progress_callback: Optional callback(completed, total) where
            total is the sum of all task_indices across entries.

    Returns:
        List of BatchResult, one per entry in the same order as entries.

    Raises:
        ValueError: If entries use different backends or sampling_params.
    """
    if not entries:
        return []

    # Validate shared backend and sampling_params
    first = entries[0].runner
    for i, entry in enumerate(entries[1:], 1):
        if entry.runner.backend is not first.backend:
            raise ValueError(
                f"All entries must share the same backend. "
                f"Entry 0 and entry {i} have different backends."
            )
        if entry.runner.sampling_params != first.sampling_params:
            raise ValueError(
                f"All entries must share the same sampling_params. "
                f"Entry 0 and entry {i} have different sampling_params."
            )

    backend = first.backend
    sampling_params = first.sampling_params

    # Create per-entry loggers
    entry_loggers: dict[int, _EvalLogger] = {}
    for i, entry in enumerate(entries):
        if entry.runner.log is not None:
            entry_loggers[i] = _EvalLogger(
                entry.runner.log,
                entry.runner.environment.spec.name,
            )

    try:
        return _run_multi_eval_impl(
            entries,
            backend,
            sampling_params,
            batch_size,
            progress_callback,
            entry_loggers,
        )
    finally:
        for el in entry_loggers.values():
            el.close()


def _run_multi_eval_impl(
    entries: list[MultiEvalEntry],
    backend: ModelBackend,
    sampling_params: SamplingParams,
    batch_size: int | None,
    progress_callback: Callable[[int, int], None] | None,
    entry_loggers: dict[int, _EvalLogger],
) -> list[BatchResult]:
    # Compute max_steps per entry
    max_steps_per_entry: dict[int, int] = {}
    for i, entry in enumerate(entries):
        max_steps_per_entry[i] = entry.runner.environment.spec.max_steps or 100

    total = sum(len(e.task_indices) for e in entries)

    # Emit batch_start for each entry
    for i, entry in enumerate(entries):
        if i in entry_loggers:
            entry_loggers[i].on_batch_start(
                _BatchStartEvent(
                    num_tasks=len(entry.task_indices),
                    environment_name=entry.runner.environment.spec.name,
                    max_steps=max_steps_per_entry[i],
                )
            )

    # Reset all tasks across all entries
    all_trajectories: list[_MultiActiveTrajectory] = []

    for entry_idx, entry in enumerate(entries):
        for task_index in entry.task_indices:
            try:
                state, reset_info = entry.runner.environment.reset(
                    options={"task_index": task_index}
                )
                trajectory: Trajectory[Any] = Trajectory.create(state)
                inner = _ActiveTrajectory(
                    position=len(all_trajectories),
                    task_index=task_index,
                    state=state,
                    reset_info=reset_info,
                    trajectory=trajectory,
                )
                all_trajectories.append(
                    _MultiActiveTrajectory(
                        entry_index=entry_idx,
                        inner=inner,
                        runner=entry.runner,
                    )
                )
            except Exception as e:
                if entry_idx in entry_loggers:
                    entry_loggers[entry_idx].on_error(
                        _ErrorEvent(
                            task_index=task_index,
                            phase="reset",
                            error=str(e),
                        )
                    )
                _raise_multi_reset(entry_idx, task_index, e)

    if batch_size is not None and len(all_trajectories) > batch_size:
        # Chunk trajectories and process each chunk
        for start in range(0, len(all_trajectories), batch_size):
            chunk = all_trajectories[start : start + batch_size]
            _run_multi_lockstep(
                chunk,
                backend,
                sampling_params,
                max_steps_per_entry,
                progress_callback=progress_callback,
                total_for_progress=total,
                progress_offset=sum(1 for t in all_trajectories[:start] if t.inner.done),
            )
    else:
        _run_multi_lockstep(
            all_trajectories,
            backend,
            sampling_params,
            max_steps_per_entry,
            progress_callback=progress_callback,
            total_for_progress=total,
            progress_offset=0,
        )

    # Partition results by entry_index
    per_entry_results: dict[int, list[TrajectoryResult]] = {i: [] for i in range(len(entries))}
    for t in all_trajectories:
        per_entry_results[t.entry_index].append(_finalize_trajectory(t.inner))

    if progress_callback:
        progress_callback(total, total)

    # Build batch results and emit batch_end events
    results = []
    for i in range(len(entries)):
        batch_result = _aggregate_results(per_entry_results[i])
        if i in entry_loggers:
            entry_loggers[i].on_batch_end(
                _BatchEndEvent(
                    success_rate=batch_result.success_rate,
                    mean_reward=batch_result.mean_reward,
                    num_tasks=len(per_entry_results[i]),
                )
            )
        results.append(batch_result)

    return results


def run_evaluation(
    environment: Environment[Any],
    backend: ModelBackend,
    num_tasks: int | None = None,
    task_indices: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
    prompt_pipeline: PromptPipeline | None = None,
    system_prompt: str | None = None,
    prompt_template: PromptTemplate | None = None,
    model_profile: ModelProfile | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    tool_call_parser: ToolCallParser | None = None,
    log: LogConfig | None = None,
    history_fn: HistoryFn | None = None,
    include_reasoning_in_history: bool = False,
    turn_info: TurnInfoConfig | bool | None = None,
) -> BatchResult:
    """Convenience function to run an evaluation.

    Args:
        environment: The environment to evaluate on.
        backend: Model backend for generation.
        num_tasks: Number of tasks to run (from start).
        task_indices: Specific task indices (overrides num_tasks).
        sampling_params: Generation parameters.
        prompt_pipeline: Optional prompt pipeline.
        system_prompt: Optional system prompt.
        prompt_template: Optional prompt template for wrapping questions.
        model_profile: Optional model profile for model-specific adjustments.
        batch_size: Maximum trajectories per lockstep batch.
        progress_callback: Optional progress callback.
        tool_call_parser: Optional text-based tool call parser for backends
            that don't support native function calling.
        log: Optional logging configuration.
        history_fn: Optional history function for structured message building.
        include_reasoning_in_history: Whether to include full model reasoning
            in prior actions (default False strips to extracted action).
        turn_info: Turn info injection config. ``True`` for defaults,
            ``TurnInfoConfig(...)`` for custom, ``None``/``False`` to disable.

    Returns:
        BatchResult with evaluation results.
    """
    if task_indices is None:
        max_tasks = len(environment) if hasattr(environment, "__len__") else 100
        num_tasks = num_tasks or max_tasks
        task_indices = list(range(min(num_tasks, max_tasks)))

    runner = TrajectoryRunner(
        environment=environment,
        backend=backend,
        sampling_params=sampling_params or SamplingParams(),
        prompt_pipeline=prompt_pipeline,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        model_profile=model_profile,
        tool_call_parser=tool_call_parser,
        turn_info=turn_info,
        log=log,
        history_fn=history_fn,
        include_reasoning_in_history=include_reasoning_in_history,
    )

    return runner.run_batch(
        task_indices,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


@dataclass
class SegmentedTrajectoryRunner:
    """Runs trajectories with segment-at-a-time generation.

    Instead of generating a full response and replaying it, this runner
    generates one segment at a time, calling SegmentedEnvironment.step()
    after each segment. This enables per-step rewards, intervention via
    step_callback, and branching at segment boundaries.

    Attributes:
        environment: A SegmentedEnvironment wrapping a single-step env.
        backend: The model backend for generation.
        sampling_params: Parameters for text generation.
        prompt_pipeline: Optional pipeline to transform prompts.
        system_prompt: Optional system prompt to include.
        prompt_template: Optional prompt template for wrapping questions.
        model_profile: Optional model profile for model-specific adjustments.
        chunk_max_tokens: Max tokens per chunk for boundary-based strategies.
    """

    environment: SegmentedEnvironment[Any]
    backend: ModelBackend
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    prompt_pipeline: PromptPipeline | None = None
    system_prompt: str | None = None
    prompt_template: PromptTemplate | None = None
    model_profile: ModelProfile | None = None
    chunk_max_tokens: int = 256
    log: LogConfig | None = None

    def _build_messages(
        self,
        state: State[Any],
    ) -> list[ChatMessage]:
        """Build chat messages from state.

        Args:
            state: Current environment state.

        Returns:
            List of ChatMessages for the model.
        """
        messages = []

        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        for msg in state.observation.messages:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
        messages.append(ChatMessage(role="user", content=state.observation.prompt))

        if self.prompt_template is not None:
            transformer = PromptTemplateTransformer(template=self.prompt_template)
            messages = transformer.transform(messages)

        if self.model_profile is not None:
            for t in self.model_profile.build_transformers():
                messages = t.transform(messages)

        if self.prompt_pipeline:
            messages = self.prompt_pipeline.transform(messages)

        return messages

    def _select_strategy(self) -> ContinuationStrategy:
        """Select the continuation strategy based on segmenter type."""
        return select_strategy(
            backend=self.backend,
            segmenter=self.environment.segmenter,
            chunk_max_tokens=self.chunk_max_tokens,
        )

    def _complete_remainder(
        self,
        env: SegmentedEnvironment[Any],
        trajectory: Trajectory[Any],
        state: State[Any],
        messages: list[ChatMessage],
        accumulated: str,
    ) -> tuple[State[Any], bool]:
        """Generate the rest of the response in one LLM call and replay segments.

        Called when the step_callback returns COMPLETE. Makes a single
        backend.generate_chat() call, segments the result, and steps each
        segment through the environment.

        Args:
            env: The segmented environment.
            trajectory: Trajectory to append transitions to.
            state: Current environment state.
            messages: Chat messages at the point COMPLETE was returned.
            accumulated: Text accumulated in the current assistant turn.

        Returns:
            Tuple of (final_state, terminal).
        """
        cont_messages = list(messages)
        if accumulated:
            cont_messages.append(ChatMessage(role="assistant", content=accumulated))

        gen_result = self.backend.generate_chat(cont_messages, self.sampling_params)
        remainder_text = gen_result.text or ""

        if remainder_text:
            segments = env.segmenter.segment(remainder_text)
            if not segments:
                segments = [remainder_text]
        else:
            segments = []

        for segment in segments:
            action = Action(text=segment)
            step_result = env.step(state, action)

            transition: Transition[Any] = Transition(
                state=state,
                action=action,
                next_state=step_result.next_state,
                rewards=step_result.rewards,
                extracted_action=step_result.extracted_action,
                resolved_action=step_result.resolved_action,
                info={
                    "generation": {
                        "prompt_tokens": gen_result.prompt_tokens,
                        "completion_tokens": gen_result.completion_tokens,
                        "finish_reason": gen_result.finish_reason.name,
                    },
                    "step": step_result.info,
                },
            )
            trajectory.add_transition(transition)
            state = step_result.next_state

            if step_result.done:
                return state, True

        return state, False

    def run_trajectory(
        self,
        task_index: int,
        trajectory_id: str | None = None,
        max_steps: int | None = None,
        step_callback: Callable[[StepResult[Any]], str | ForceAction | None] | None = None,
        prefix: str | Sequence[tuple[State[Any], Action]] | None = None,
    ) -> TrajectoryResult:
        """Run a single trajectory with segment-at-a-time generation.

        Args:
            task_index: Index of the task in the environment.
            trajectory_id: Optional custom trajectory ID.
            max_steps: Maximum generation steps (prefix steps excluded).
            step_callback: Optional callback invoked after each non-terminal
                generation step. Return values:
                - ``None``: continue generating segment-by-segment.
                - A feedback string: inject as a user message (new turn).
                - ``COMPLETE``: finish the rest in one LLM call.
                - ``ForceAction(text)``: use *text* as the next segment
                  (skip LLM call, clear buffer).
                The callback is **not** called during prefix replay.
            prefix: Predetermined content to replay before generation.
                - ``str``: auto-segmented via the environment's segmenter
                  and stepped from the reset state.
                - ``Sequence[tuple[State, Action]]``: state-action pairs
                  stepped using the provided states (for exact replay).

        Returns:
            TrajectoryResult with trajectory and metrics.
        """
        env = self.environment
        strategy = self._select_strategy()

        # Reset environment
        options: dict[str, Any] = {"task_index": task_index}
        if trajectory_id:
            options["episode_id"] = trajectory_id

        state, reset_info = env.reset(options=options)
        trajectory: Trajectory[Any] = Trajectory.create(state)

        max_steps = max_steps or 1000
        messages = self._build_messages(state)
        accumulated = ""
        buffer = ""
        step_count = 0
        terminal = False
        complete_early = False
        prefix_steps = 0

        # ── Prefix replay phase ──────────────────────────────────────
        if prefix:
            if isinstance(prefix, str):
                # Text form: auto-segment and step from reset state
                segments = env.segmenter.segment(prefix)
                for seg in segments:
                    action = Action(text=seg)
                    step_result = env.step(state, action)

                    transition: Transition[Any] = Transition(
                        state=state,
                        action=action,
                        next_state=step_result.next_state,
                        rewards=step_result.rewards,
                        extracted_action=step_result.extracted_action,
                        resolved_action=step_result.resolved_action,
                        info={
                            "replayed": True,
                            "step": step_result.info,
                        },
                    )
                    trajectory.add_transition(transition)
                    state = step_result.next_state
                    prefix_steps += 1

                    if step_result.done:
                        terminal = True
                        break

                if not terminal:
                    accumulated = prefix
            else:
                # Structured form: step each (state, action) pair
                for pair_state, pair_action in prefix:
                    step_result = env.step(pair_state, pair_action)

                    transition = Transition(
                        state=pair_state,
                        action=pair_action,
                        next_state=step_result.next_state,
                        rewards=step_result.rewards,
                        extracted_action=step_result.extracted_action,
                        resolved_action=step_result.resolved_action,
                        info={
                            "replayed": True,
                            "step": step_result.info,
                        },
                    )
                    trajectory.add_transition(transition)
                    state = step_result.next_state
                    prefix_steps += 1

                    if step_result.done:
                        terminal = True
                        break

                if not terminal:
                    accumulated = state.hidden.accumulated_text

        # ── Generation loop ──────────────────────────────────────────
        forced_segment: str | None = None

        while not terminal and step_count < max_steps:
            # Use forced segment if available, otherwise generate
            this_forced = False
            if forced_segment is not None:
                segment = forced_segment
                forced_segment = None
                this_forced = True
                buffer = ""  # Clear stale buffer
            else:
                segment, buffer, gen_result = strategy.generate_segment(
                    messages=messages,
                    accumulated_text=accumulated,
                    buffer=buffer,
                    sampling_params=self.sampling_params,
                )

                if not segment:
                    break

            # Step the environment with this segment
            action = Action(text=segment)
            step_result = env.step(state, action)

            # Build transition info
            if this_forced:
                trans_info: dict[str, Any] = {
                    "forced": True,
                    "step": step_result.info,
                }
            else:
                trans_info = {
                    "generation": {
                        "prompt_tokens": gen_result.prompt_tokens,
                        "completion_tokens": gen_result.completion_tokens,
                        "finish_reason": gen_result.finish_reason.name,
                    },
                    "step": step_result.info,
                }

            transition = Transition(
                state=state,
                action=action,
                next_state=step_result.next_state,
                rewards=step_result.rewards,
                extracted_action=step_result.extracted_action,
                resolved_action=step_result.resolved_action,
                info=trans_info,
            )
            trajectory.add_transition(transition)

            accumulated += segment
            state = step_result.next_state
            step_count += 1

            if step_result.done:
                terminal = True

            # Observation injection via callback (skip for terminal)
            if not terminal and step_callback is not None:
                feedback = step_callback(step_result)
                if feedback is COMPLETE:
                    complete_early = True
                    break
                elif isinstance(feedback, ForceAction):
                    forced_segment = feedback.text
                elif feedback is not None:
                    # Feedback becomes a user message between assistant turns
                    messages.append(ChatMessage(role="assistant", content=accumulated))
                    messages.append(ChatMessage(role="user", content=feedback))
                    accumulated = ""
                    buffer = ""

            # Check if generation is done (skip for forced segments)
            if not this_forced and strategy.is_generation_done(gen_result, buffer):
                break

        # Drain remaining buffer as final segment
        if buffer and not terminal and not complete_early:
            action = Action(text=buffer)
            step_result = env.step(state, action)

            transition = Transition(
                state=state,
                action=action,
                next_state=step_result.next_state,
                rewards=step_result.rewards,
                extracted_action=step_result.extracted_action,
                resolved_action=step_result.resolved_action,
                info={"step": step_result.info},
            )
            trajectory.add_transition(transition)

            accumulated += buffer
            state = step_result.next_state
            step_count += 1
            buffer = ""

            if step_result.done:
                terminal = True

        # One-shot completion for early exit
        if complete_early and not terminal:
            state, terminal = self._complete_remainder(
                env,
                trajectory,
                state,
                messages,
                accumulated,
            )

        # Finalize to get correctness rewards
        if not terminal:
            finalize_result = env.finalize(state)

            transition = Transition(
                state=state,
                action=Action(text=""),
                next_state=finalize_result.next_state,
                rewards=finalize_result.rewards,
                extracted_action=finalize_result.extracted_action,
                resolved_action=finalize_result.resolved_action,
                info={"step": finalize_result.info, "finalize": True},
            )
            trajectory.add_transition(transition)
            state = finalize_result.next_state

        # Determine success from OUTCOME-type reward
        success = False
        if trajectory.transitions:
            last_rewards = trajectory.transitions[-1].rewards
            outcome_rewards = last_rewards.by_type(RewardType.OUTCOME)
            if outcome_rewards:
                success = outcome_rewards[-1].reward >= 1.0

        return TrajectoryResult(
            trajectory=trajectory,
            total_reward=trajectory.total_reward,
            success=success,
            metadata={
                "task_index": task_index,
                "num_steps": len(trajectory),
                "reset_info": reset_info,
                "prefix_steps": prefix_steps,
            },
        )

    def run_batch(
        self,
        task_indices: list[int],
        step_callback: Callable[[StepResult[Any]], str | ForceAction | None] | None = None,
        max_steps: int | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Run a batch of segmented trajectories with lockstep batched generation.

        All trajectories advance one segment together in each iteration,
        batching inference calls via generate_segment_batch(). Trajectories
        that finish generation early drop out of subsequent batches.

        Args:
            task_indices: List of task indices to run.
            step_callback: Optional per-step callback (shared across trajectories).
            max_steps: Maximum segment steps per trajectory.
            batch_size: Maximum trajectories per lockstep batch. When set,
                task_indices are chunked and each chunk is processed
                independently. None means all tasks in one batch.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            BatchResult with all trajectory results and aggregate metrics.
        """
        if batch_size is not None and len(task_indices) > batch_size:
            return _run_in_chunks(
                lambda indices, cb: self.run_batch(
                    indices,
                    step_callback=step_callback,
                    max_steps=max_steps,
                    progress_callback=cb,
                ),
                task_indices,
                batch_size,
                progress_callback,
            )

        if not task_indices:
            return _aggregate_results([])

        env = self.environment
        strategy = self._select_strategy()
        max_steps = max_steps or 1000
        total = len(task_indices)
        result_slots: list[TrajectoryResult | None] = [None] * total

        # Phase 1: Reset all tasks
        active: list[_ActiveSegmentedTrajectory] = []
        for pos, task_index in enumerate(task_indices):
            try:
                state, reset_info = env.reset(options={"task_index": task_index})
                trajectory: Trajectory[Any] = Trajectory.create(state)
                messages = self._build_messages(state)
                active.append(
                    _ActiveSegmentedTrajectory(
                        position=pos,
                        task_index=task_index,
                        state=state,
                        reset_info=reset_info,
                        trajectory=trajectory,
                        messages=messages,
                    )
                )
            except Exception as e:
                _raise_with_context("resetting", task_index, e)

        # Phase 2: Lockstep segment generation
        while True:
            # Separate trajectories needing generation from those with forced segments
            need_gen: list[_ActiveSegmentedTrajectory] = []
            have_forced: list[_ActiveSegmentedTrajectory] = []
            for t in active:
                if t.done or t.generation_done:
                    continue
                if t.forced_segment is not None:
                    have_forced.append(t)
                else:
                    need_gen.append(t)

            if not need_gen and not have_forced:
                break

            # Batch generate for trajectories needing generation
            gen_map: dict[int, tuple[str, str, GenerationResult]] = {}
            if need_gen:
                contexts = [
                    SegmentContext(
                        messages=t.messages,
                        accumulated_text=t.accumulated,
                        buffer=t.buffer,
                    )
                    for t in need_gen
                ]
                seg_results = strategy.generate_segment_batch(contexts, self.sampling_params)
                for t, seg_result in zip(need_gen, seg_results):
                    gen_map[id(t)] = seg_result

            # Process all active trajectories this round
            for t in need_gen + have_forced:
                try:
                    if t.forced_segment is not None:
                        segment = t.forced_segment
                        t.forced_segment = None
                        t.buffer = ""
                        is_forced = True
                        gen_result = _BUFFER_ONLY_RESULT
                    else:
                        segment, t.buffer, gen_result = gen_map[id(t)]
                        is_forced = False
                        if not segment:
                            t.generation_done = True
                            continue

                    # Step the environment
                    action = Action(text=segment)
                    step_result = env.step(t.state, action)

                    # Build transition info
                    if is_forced:
                        trans_info: dict[str, Any] = {
                            "forced": True,
                            "step": step_result.info,
                        }
                    else:
                        trans_info = {
                            "generation": {
                                "prompt_tokens": gen_result.prompt_tokens,
                                "completion_tokens": gen_result.completion_tokens,
                                "finish_reason": gen_result.finish_reason.name,
                            },
                            "step": step_result.info,
                        }

                    transition: Transition[Any] = Transition(
                        state=t.state,
                        action=action,
                        next_state=step_result.next_state,
                        rewards=step_result.rewards,
                        extracted_action=step_result.extracted_action,
                        resolved_action=step_result.resolved_action,
                        info=trans_info,
                    )
                    t.trajectory.add_transition(transition)

                    t.accumulated += segment
                    t.state = step_result.next_state
                    t.step_count += 1

                    if step_result.done:
                        t.done = True
                        continue

                    # Handle step_callback
                    if step_callback is not None:
                        feedback = step_callback(step_result)
                        if feedback is COMPLETE:
                            t.complete_early = True
                            t.generation_done = True
                            continue
                        elif isinstance(feedback, ForceAction):
                            t.forced_segment = feedback.text
                        elif feedback is not None:
                            t.messages.append(ChatMessage(role="assistant", content=t.accumulated))
                            t.messages.append(ChatMessage(role="user", content=feedback))
                            t.accumulated = ""
                            t.buffer = ""

                    # Check if generation is done
                    if not is_forced and strategy.is_generation_done(gen_result, t.buffer):
                        t.generation_done = True

                    if t.step_count >= max_steps:
                        t.generation_done = True

                except Exception as e:
                    _raise_with_context("stepping", t.task_index, e)

            if progress_callback:
                done_count = sum(1 for t in active if t.done or t.generation_done)
                progress_callback(done_count, total)

        # Phase 3: Buffer drain
        for t in active:
            if t.buffer and not t.done and not t.complete_early:
                try:
                    action = Action(text=t.buffer)
                    step_result = env.step(t.state, action)

                    transition = Transition(
                        state=t.state,
                        action=action,
                        next_state=step_result.next_state,
                        rewards=step_result.rewards,
                        extracted_action=step_result.extracted_action,
                        resolved_action=step_result.resolved_action,
                        info={"step": step_result.info},
                    )
                    t.trajectory.add_transition(transition)

                    t.accumulated += t.buffer
                    t.state = step_result.next_state
                    t.step_count += 1
                    t.buffer = ""

                    if step_result.done:
                        t.done = True
                except Exception as e:
                    _raise_with_context("draining buffer for task", t.task_index, e)

        # Phase 4: Complete remainder for COMPLETE callbacks
        for t in active:
            if t.complete_early and not t.done:
                try:
                    state, terminal = self._complete_remainder(
                        env,
                        t.trajectory,
                        t.state,
                        t.messages,
                        t.accumulated,
                    )
                    t.state = state
                    if terminal:
                        t.done = True
                except Exception as e:
                    _raise_with_context("completing", t.task_index, e)

        # Phase 5: Finalize non-terminal trajectories
        for t in active:
            if not t.done and t.error is None:
                try:
                    finalize_result = env.finalize(t.state)

                    transition = Transition(
                        state=t.state,
                        action=Action(text=""),
                        next_state=finalize_result.next_state,
                        rewards=finalize_result.rewards,
                        extracted_action=finalize_result.extracted_action,
                        resolved_action=finalize_result.resolved_action,
                        info={"step": finalize_result.info, "finalize": True},
                    )
                    t.trajectory.add_transition(transition)
                    t.state = finalize_result.next_state
                    t.done = True
                except Exception as e:
                    _raise_with_context("finalizing", t.task_index, e)

        # Phase 6: Build results
        for t in active:
            result_slots[t.position] = _finalize_trajectory(t)

        if progress_callback:
            progress_callback(total, total)

        return _aggregate_results([r for r in result_slots if r is not None])


def run_segmented_evaluation(
    environment: SegmentedEnvironment[Any],
    backend: ModelBackend,
    num_tasks: int | None = None,
    task_indices: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
    prompt_pipeline: PromptPipeline | None = None,
    system_prompt: str | None = None,
    prompt_template: PromptTemplate | None = None,
    model_profile: ModelProfile | None = None,
    step_callback: Callable[[StepResult[Any]], str | ForceAction | None] | None = None,
    max_steps: int | None = None,
    chunk_max_tokens: int = 256,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    log: LogConfig | None = None,
) -> BatchResult:
    """Convenience function to run a segmented evaluation.

    Args:
        environment: A SegmentedEnvironment to evaluate on.
        backend: Model backend for generation.
        num_tasks: Number of tasks to run (from start).
        task_indices: Specific task indices (overrides num_tasks).
        sampling_params: Generation parameters.
        prompt_pipeline: Optional prompt pipeline.
        system_prompt: Optional system prompt.
        prompt_template: Optional prompt template for wrapping questions.
        model_profile: Optional model profile for model-specific adjustments.
        step_callback: Optional per-step callback for observation injection.
        max_steps: Maximum segment steps per trajectory.
        chunk_max_tokens: Max tokens per chunk for boundary strategies.
        batch_size: Maximum trajectories per lockstep batch.
        progress_callback: Optional progress callback.
        log: Optional logging configuration.

    Returns:
        BatchResult with evaluation results.
    """
    if task_indices is None:
        max_tasks = len(environment) if hasattr(environment, "__len__") else 100
        num_tasks = num_tasks or max_tasks
        task_indices = list(range(min(num_tasks, max_tasks)))

    runner = SegmentedTrajectoryRunner(
        environment=environment,
        backend=backend,
        sampling_params=sampling_params or SamplingParams(),
        prompt_pipeline=prompt_pipeline,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        model_profile=model_profile,
        chunk_max_tokens=chunk_max_tokens,
        log=log,
    )

    return runner.run_batch(
        task_indices,
        step_callback=step_callback,
        max_steps=max_steps,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
