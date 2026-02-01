"""Trajectory runner for orchestrating evaluations.

Handles running trajectories through environments with model backends,
collecting results.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging

from llenvs.core.state import State, TextObservation, TextAction, AgentObservation, AgentAction
from llenvs.core.environment import Environment, StepResult
from llenvs.core.tool_environment import ToolEnvironment
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.core.reward import RewardBundle
from llenvs.inference.protocol import (
    ModelBackend,
    SamplingParams,
    ChatMessage,
    GenerationResult,
)
from llenvs.inference.prompting import PromptPipeline

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Result of running a single trajectory.

    Attributes:
        trajectory: The full trajectory (sequence of state-action-reward transitions).
        total_reward: Sum of all rewards across the trajectory.
        success: Whether the trajectory was successful (based on correctness).
        metadata: Additional result metadata.
    """

    trajectory: Trajectory[Any, Any, Any]
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
class TrajectoryRunner:
    """Runs trajectories through an environment with a model backend.

    Handles the interaction loop between model and environment,
    collecting trajectories and computing results.

    Attributes:
        environment: The environment to run trajectories in.
        backend: The model backend for generation.
        sampling_params: Parameters for text generation.
        prompt_pipeline: Optional pipeline to transform prompts.
        system_prompt: Optional system prompt to include.
    """

    environment: Environment[Any, Any, Any]
    backend: ModelBackend
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    prompt_pipeline: PromptPipeline | None = None
    system_prompt: str | None = None

    def _build_messages(
        self,
        state: State[TextObservation, Any],
    ) -> list[ChatMessage]:
        """Build chat messages from state.

        Args:
            state: Current environment state.

        Returns:
            List of ChatMessages for the model.
        """
        messages = []

        # Add system prompt if provided
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        # Add observation as user message
        if isinstance(state.observation, TextObservation):
            # Include any message history
            for msg in state.observation.messages:
                messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

            # Add current prompt
            messages.append(ChatMessage(role="user", content=state.observation.prompt))
        else:
            # Fallback for other observation types
            messages.append(ChatMessage(role="user", content=str(state.observation)))

        # Apply prompt pipeline if configured
        if self.prompt_pipeline:
            messages = self.prompt_pipeline.transform(messages)

        return messages

    def _generate_action(
        self,
        state: State[TextObservation, Any],
    ) -> tuple[TextAction, GenerationResult]:
        """Generate an action (model response) for the current state.

        Args:
            state: Current environment state.

        Returns:
            Tuple of (TextAction, GenerationResult).
        """
        messages = self._build_messages(state)
        result = self.backend.generate_chat(messages, self.sampling_params)
        action = TextAction(text=result.text)
        return action, result

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
        trajectory: Trajectory[Any, Any, Any] = Trajectory.create(state)

        max_steps = max_steps or self.environment.spec.max_steps or 100

        # Run trajectory loop
        step_count = 0
        while not state.metadata.is_terminal and step_count < max_steps:
            # Generate action
            action, gen_result = self._generate_action(state)

            # Take step (apply transition function)
            step_result = self.environment.step(state, action)

            # Record transition
            transition: Transition[Any, Any, Any] = Transition(
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
            step_count += 1

            if step_result.done:
                break

        # Determine success from correctness reward
        success = False
        if trajectory.transitions:
            last_rewards = trajectory.transitions[-1].rewards
            correctness = last_rewards.by_name("correctness")
            if correctness:
                success = correctness.value >= 1.0

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
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Run a batch of trajectories.

        Args:
            task_indices: List of task indices to run.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            BatchResult with all trajectory results and aggregate metrics.
        """
        trajectory_results: list[TrajectoryResult] = []

        for i, task_index in enumerate(task_indices):
            if progress_callback:
                progress_callback(i, len(task_indices))

            try:
                result = self.run_trajectory(task_index)
                trajectory_results.append(result)
            except Exception as e:
                logger.error(f"Error running task {task_index}: {e}")
                # Create failed result
                trajectory_results.append(
                    TrajectoryResult(
                        trajectory=Trajectory(
                            episode_id=f"error_{task_index}",
                            initial_state=State(
                                observation=TextObservation(prompt=""),
                                hidden=None,
                                metadata=self._dummy_metadata(task_index),
                            ),
                        ),
                        total_reward=0.0,
                        success=False,
                        metadata={"error": str(e), "task_index": task_index},
                    )
                )

        if progress_callback:
            progress_callback(len(task_indices), len(task_indices))

        # Compute aggregate metrics
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

    def _dummy_metadata(self, task_index: int) -> Any:
        """Create dummy metadata for error cases."""
        from llenvs.core.state import StateMetadata

        return StateMetadata(
            step=0,
            episode_id=f"error_{task_index}",
            is_terminal=True,
            info={"error": True},
        )


def run_evaluation(
    environment: Environment[Any, Any, Any],
    backend: ModelBackend,
    num_tasks: int | None = None,
    task_indices: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
    prompt_pipeline: PromptPipeline | None = None,
    system_prompt: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
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
        progress_callback: Optional progress callback.

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
    )

    return runner.run_batch(task_indices, progress_callback=progress_callback)


@dataclass
class ToolTrajectoryRunner:
    """Runs trajectories through a tool-aware environment with a model backend.

    Similar to TrajectoryRunner but supports tool calling via the
    generate_with_tools backend method.

    Attributes:
        environment: The tool environment to run trajectories in.
        backend: The model backend for generation.
        sampling_params: Parameters for text generation.
        prompt_pipeline: Optional pipeline to transform prompts.
        system_prompt: Optional system prompt to include.
    """

    environment: ToolEnvironment[Any]
    backend: ModelBackend
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    prompt_pipeline: PromptPipeline | None = None
    system_prompt: str | None = None

    def _build_messages(
        self,
        state: State[AgentObservation, Any],
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

        # Add initial prompt as user message if no messages yet
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

        # Apply prompt pipeline if configured
        if self.prompt_pipeline:
            messages = self.prompt_pipeline.transform(messages)

        return messages

    def _generate_action(
        self,
        state: State[AgentObservation, Any],
    ) -> tuple[AgentAction, GenerationResult]:
        """Generate an action (model response) for the current state.

        Uses generate_with_tools if the backend supports it and tools are available.

        Args:
            state: Current environment state.

        Returns:
            Tuple of (AgentAction, GenerationResult).
        """
        messages = self._build_messages(state)
        tools = list(state.observation.available_tools)

        if tools and self.backend.capabilities.supports_function_calling:
            result = self.backend.generate_with_tools(
                messages, tools, self.sampling_params
            )
        else:
            result = self.backend.generate_chat(messages, self.sampling_params)

        return result.to_agent_action(), result

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
        trajectory: Trajectory[Any, Any, Any] = Trajectory.create(state)

        max_steps = max_steps or self.environment.spec.max_steps or 100

        # Run trajectory loop
        step_count = 0
        while not state.metadata.is_terminal and step_count < max_steps:
            # Generate action
            action, gen_result = self._generate_action(state)

            # Take step (apply transition function)
            step_result = self.environment.step(state, action)

            # Record transition
            transition: Transition[Any, Any, Any] = Transition(
                state=state,
                action=action,
                next_state=step_result.next_state,
                rewards=step_result.rewards,
                info={
                    "generation": {
                        "prompt_tokens": gen_result.prompt_tokens,
                        "completion_tokens": gen_result.completion_tokens,
                        "finish_reason": gen_result.finish_reason.name,
                        "has_tool_calls": gen_result.has_tool_calls,
                        "num_tool_calls": len(gen_result.tool_calls),
                    },
                    "step": step_result.info,
                },
            )
            trajectory.add_transition(transition)

            state = step_result.next_state
            step_count += 1

            if step_result.done:
                break

        # Determine success from correctness reward
        success = False
        if trajectory.transitions:
            last_rewards = trajectory.transitions[-1].rewards
            correctness = last_rewards.by_name("correctness")
            if correctness:
                success = correctness.value >= 1.0

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
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Run a batch of trajectories.

        Args:
            task_indices: List of task indices to run.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            BatchResult with all trajectory results and aggregate metrics.
        """
        trajectory_results: list[TrajectoryResult] = []

        for i, task_index in enumerate(task_indices):
            if progress_callback:
                progress_callback(i, len(task_indices))

            try:
                result = self.run_trajectory(task_index)
                trajectory_results.append(result)
            except Exception as e:
                logger.error(f"Error running task {task_index}: {e}")
                # Create failed result
                trajectory_results.append(
                    TrajectoryResult(
                        trajectory=Trajectory(
                            episode_id=f"error_{task_index}",
                            initial_state=State(
                                observation=AgentObservation(prompt=""),
                                hidden=None,
                                metadata=self._dummy_metadata(task_index),
                            ),
                        ),
                        total_reward=0.0,
                        success=False,
                        metadata={"error": str(e), "task_index": task_index},
                    )
                )

        if progress_callback:
            progress_callback(len(task_indices), len(task_indices))

        # Compute aggregate metrics
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

    def _dummy_metadata(self, task_index: int) -> Any:
        """Create dummy metadata for error cases."""
        from llenvs.core.state import StateMetadata

        return StateMetadata(
            step=0,
            episode_id=f"error_{task_index}",
            is_terminal=True,
            info={"error": True},
        )


def run_tool_evaluation(
    environment: ToolEnvironment[Any],
    backend: ModelBackend,
    num_tasks: int | None = None,
    task_indices: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
    prompt_pipeline: PromptPipeline | None = None,
    system_prompt: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Convenience function to run a tool-aware evaluation.

    Args:
        environment: The tool environment to evaluate on.
        backend: Model backend for generation.
        num_tasks: Number of tasks to run (from start).
        task_indices: Specific task indices (overrides num_tasks).
        sampling_params: Generation parameters.
        prompt_pipeline: Optional prompt pipeline.
        system_prompt: Optional system prompt.
        progress_callback: Optional progress callback.

    Returns:
        BatchResult with evaluation results.
    """
    if task_indices is None:
        max_tasks = len(environment) if hasattr(environment, "__len__") else 100
        num_tasks = num_tasks or max_tasks
        task_indices = list(range(min(num_tasks, max_tasks)))

    runner = ToolTrajectoryRunner(
        environment=environment,
        backend=backend,
        sampling_params=sampling_params or SamplingParams(),
        prompt_pipeline=prompt_pipeline,
        system_prompt=system_prompt,
    )

    return runner.run_batch(task_indices, progress_callback=progress_callback)
