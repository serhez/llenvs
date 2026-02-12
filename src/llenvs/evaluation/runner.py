"""Trajectory runner for orchestrating evaluations.

Handles running trajectories through environments with model backends,
collecting results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
import logging

if TYPE_CHECKING:
    from llenvs.core.tool_parsing import ToolCallParser
    from llenvs.inference.prompts import ModelProfile, PromptTemplate

from llenvs.core.state import State, Observation, Action
from llenvs.core.environment import Environment, StepResult
from llenvs.core.segmented_environment import SegmentedEnvironment
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.core.reward import RewardBundle, RewardType
from llenvs.inference.protocol import (
    ModelBackend,
    SamplingParams,
    ChatMessage,
    GenerationResult,
    StopReason,
)
from llenvs.inference.prompting import PromptPipeline, PromptTemplateTransformer
from llenvs.evaluation.continuation import (
    ContinuationStrategy,
    TokenContinuationStrategy,
    BoundaryContinuationStrategy,
    SegmentContext,
    select_strategy,
    _BUFFER_ONLY_RESULT,
)

logger = logging.getLogger(__name__)

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


def _error_metadata(task_index: int) -> "StateMetadata":
    """Create dummy metadata for error cases."""
    from llenvs.core.state import StateMetadata

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

        sub_cb: Callable[[int, int], None] | None = None
        if progress_callback:
            _offset = start

            def sub_cb(done: int, chunk_total: int, _s: int = _offset) -> None:
                progress_callback(_s + done, total)

        chunk_result = run_fn(chunk, sub_cb)
        all_results.extend(chunk_result.trajectory_results)

    if progress_callback:
        progress_callback(total, total)

    return _aggregate_results(all_results)


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
            success = outcome_rewards[-1].value >= 1.0

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
    prompt_template: "PromptTemplate | None" = None
    model_profile: "ModelProfile | None" = None
    tool_call_parser: "ToolCallParser | None" = None

    def _build_messages(
        self,
        state: State[Any],
    ) -> list[ChatMessage]:
        """Build chat messages from state including tool results.

        Args:
            state: Current environment state.

        Returns:
            List of ChatMessages for the model.
        """
        messages: list[ChatMessage] = []

        # Add system prompt if provided
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        obs = state.observation

        # Add initial prompt as user message
        if not obs.messages:
            messages.append(ChatMessage(role="user", content=obs.prompt))
        else:
            # First message should be user with prompt
            messages.append(ChatMessage(role="user", content=obs.prompt))

            # Then add message history
            for msg in obs.messages:
                role = msg.get("role", "user")

                if role == "assistant":
                    # Check for tool calls
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
                    messages.append(ChatMessage(role="user", content=msg.get("content", "")))

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

        return messages

    def _generate_action(
        self,
        state: State[Any],
    ) -> tuple[Action, GenerationResult]:
        """Generate an action (model response) for the current state.

        Uses generate_with_tools if the backend supports it and tools are available.

        Args:
            state: Current environment state.

        Returns:
            Tuple of (Action, GenerationResult).
        """
        messages = self._build_messages(state)
        tools = list(state.observation.available_tools)

        if tools and self.backend.capabilities.supports_function_calling:
            # Native function calling (API backends)
            result = self.backend.generate_with_tools(
                messages, tools, self.sampling_params
            )
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
        tools: tuple["ToolDefinition", ...],
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
            action, gen_result = self._generate_action(state)

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
                info={
                    "generation": gen_info,
                    "step": step_result.info,
                },
            )
            trajectory.add_transition(transition)

            state = step_result.next_state
            step_count += 1

            if step_result.done:
                break

        # Determine success from OUTCOME-type reward
        success = False
        if trajectory.transitions:
            last_rewards = trajectory.transitions[-1].rewards
            outcome_rewards = last_rewards.by_type(RewardType.OUTCOME)
            if outcome_rewards:
                success = outcome_rewards[-1].value >= 1.0

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
        if batch_size is not None and len(task_indices) > batch_size:
            return _run_in_chunks(
                lambda indices, cb: self.run_batch(indices, progress_callback=cb),
                task_indices,
                batch_size,
                progress_callback,
            )

        if not task_indices:
            return _aggregate_results([])

        max_steps = self.environment.spec.max_steps or 100
        total = len(task_indices)
        result_slots: list[TrajectoryResult | None] = [None] * total

        # Phase 1: Reset all tasks
        active: list[_ActiveTrajectory] = []
        for pos, task_index in enumerate(task_indices):
            try:
                state, reset_info = self.environment.reset(
                    options={"task_index": task_index}
                )
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
            except Exception as e:
                logger.error(f"Error resetting task {task_index}: {e}")
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

        # Phase 2: Lockstep generation
        while True:
            remaining = [t for t in active if not t.done]
            if not remaining:
                break

            messages_batch = [self._build_messages(t.state) for t in remaining]

            # Use tool calling if tools available and backend supports it
            first_obs = remaining[0].state.observation
            tools = list(first_obs.available_tools)
            use_native_tools = tools and self.backend.capabilities.supports_function_calling
            use_text_tools = tools and not use_native_tools and self.tool_call_parser is not None

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
                    parsed = self.tool_call_parser.parse(
                        raw.text or "", tuple(tools)
                    )
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

            for t, gen_result in zip(remaining, gen_results):
                try:
                    action = gen_result.to_agent_action()
                    step_result = self.environment.step(t.state, action)

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
                        info={
                            "generation": gen_info,
                            "step": step_result.info,
                        },
                    )
                    t.trajectory.add_transition(transition)
                    t.state = step_result.next_state
                    t.step_count += 1

                    if step_result.done or t.step_count >= max_steps:
                        t.done = True
                except Exception as e:
                    logger.error(f"Error stepping task {t.task_index}: {e}")
                    t.done = True
                    t.error = str(e)

            if progress_callback:
                done_count = reset_errors + sum(1 for t in active if t.done)
                progress_callback(done_count, total)

        # Phase 3: Build results
        for t in active:
            result_slots[t.position] = _finalize_trajectory(t)

        if progress_callback:
            progress_callback(total, total)

        return _aggregate_results([r for r in result_slots if r is not None])


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

        messages_batch = [t.runner._build_messages(t.inner.state) for t in remaining]
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

    # Compute max_steps per entry
    max_steps_per_entry: dict[int, int] = {}
    for i, entry in enumerate(entries):
        max_steps_per_entry[i] = entry.runner.environment.spec.max_steps or 100

    total = sum(len(e.task_indices) for e in entries)

    # Reset all tasks across all entries
    all_trajectories: list[_MultiActiveTrajectory] = []
    # Track reset errors per entry for result assembly
    reset_error_results: dict[int, list[TrajectoryResult]] = {i: [] for i in range(len(entries))}

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
                logger.error(f"Error resetting task {task_index} in entry {entry_idx}: {e}")
                reset_error_results[entry_idx].append(
                    TrajectoryResult(
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
                )

    if batch_size is not None and len(all_trajectories) > batch_size:
        # Chunk trajectories and process each chunk
        for start in range(0, len(all_trajectories), batch_size):
            chunk = all_trajectories[start:start + batch_size]
            offset = start  # reset errors already handled outside lockstep
            _run_multi_lockstep(
                chunk, backend, sampling_params, max_steps_per_entry,
                progress_callback=progress_callback,
                total_for_progress=total,
                progress_offset=sum(
                    1 for t in all_trajectories[:start] if t.inner.done
                ) + sum(len(v) for v in reset_error_results.values()),
            )
    else:
        reset_errors_total = sum(len(v) for v in reset_error_results.values())
        _run_multi_lockstep(
            all_trajectories, backend, sampling_params, max_steps_per_entry,
            progress_callback=progress_callback,
            total_for_progress=total,
            progress_offset=reset_errors_total,
        )

    # Partition results by entry_index
    per_entry_results: dict[int, list[TrajectoryResult]] = {
        i: list(reset_error_results[i]) for i in range(len(entries))
    }
    for t in all_trajectories:
        per_entry_results[t.entry_index].append(_finalize_trajectory(t.inner))

    if progress_callback:
        progress_callback(total, total)

    return [_aggregate_results(per_entry_results[i]) for i in range(len(entries))]


def run_evaluation(
    environment: Environment[Any],
    backend: ModelBackend,
    num_tasks: int | None = None,
    task_indices: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
    prompt_pipeline: PromptPipeline | None = None,
    system_prompt: str | None = None,
    prompt_template: "PromptTemplate | None" = None,
    model_profile: "ModelProfile | None" = None,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    tool_call_parser: "ToolCallParser | None" = None,
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
    )

    return runner.run_batch(
        task_indices, batch_size=batch_size, progress_callback=progress_callback,
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
    prompt_template: "PromptTemplate | None" = None
    model_profile: "ModelProfile | None" = None
    chunk_max_tokens: int = 256

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
                env, trajectory, state, messages, accumulated,
            )

        # Finalize to get correctness rewards
        if not terminal:
            finalize_result = env.finalize(state)

            transition = Transition(
                state=state,
                action=Action(text=""),
                next_state=finalize_result.next_state,
                rewards=finalize_result.rewards,
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
                success = outcome_rewards[-1].value >= 1.0

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
                logger.error(f"Error resetting task {task_index}: {e}")
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
                seg_results = strategy.generate_segment_batch(
                    contexts, self.sampling_params
                )
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
                            t.messages.append(
                                ChatMessage(role="assistant", content=t.accumulated)
                            )
                            t.messages.append(
                                ChatMessage(role="user", content=feedback)
                            )
                            t.accumulated = ""
                            t.buffer = ""

                    # Check if generation is done
                    if not is_forced and strategy.is_generation_done(gen_result, t.buffer):
                        t.generation_done = True

                    if t.step_count >= max_steps:
                        t.generation_done = True

                except Exception as e:
                    logger.error(f"Error stepping task {t.task_index}: {e}")
                    t.done = True
                    t.error = str(e)

            if progress_callback:
                done_count = reset_errors + sum(
                    1 for t in active if t.done or t.generation_done
                )
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
                    logger.error(f"Error draining buffer for task {t.task_index}: {e}")
                    t.done = True
                    t.error = str(e)

        # Phase 4: Complete remainder for COMPLETE callbacks
        for t in active:
            if t.complete_early and not t.done:
                try:
                    state, terminal = self._complete_remainder(
                        env, t.trajectory, t.state, t.messages, t.accumulated,
                    )
                    t.state = state
                    if terminal:
                        t.done = True
                except Exception as e:
                    logger.error(f"Error completing task {t.task_index}: {e}")
                    t.done = True
                    t.error = str(e)

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
                        info={"step": finalize_result.info, "finalize": True},
                    )
                    t.trajectory.add_transition(transition)
                    t.state = finalize_result.next_state
                    t.done = True
                except Exception as e:
                    logger.error(f"Error finalizing task {t.task_index}: {e}")
                    t.done = True
                    t.error = str(e)

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
    prompt_template: "PromptTemplate | None" = None,
    model_profile: "ModelProfile | None" = None,
    step_callback: Callable[[StepResult[Any]], str | ForceAction | None] | None = None,
    max_steps: int | None = None,
    chunk_max_tokens: int = 256,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
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
    )

    return runner.run_batch(
        task_indices,
        step_callback=step_callback,
        max_steps=max_steps,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )