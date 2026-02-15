"""Segmented environment wrapper for multi-step reasoning.

Wraps single-step environments to enable per-step rewards through text segmentation.
Supports both post-hoc replay and generation-time stepping.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
import copy

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import SignalBundle, Signal, RewardType, RewardFunction
from llenvs.core.environment import Environment, StepResult, EnvironmentSpec
from llenvs.core.segmentation import Segmenter

HiddenT = TypeVar("HiddenT")


@dataclass(frozen=True)
class SegmentedHidden(Generic[HiddenT]):
    """Extended hidden state for segmented environments.

    Wraps the base environment's hidden state with segmentation tracking.

    Attributes:
        base_hidden: Original environment's hidden state.
        accumulated_text: Text generated so far across all segments.
        segment_index: Current segment number (0-indexed).
        segments: All segments seen so far.
        total_segments: Total number of segments (known only in replay mode).
    """

    base_hidden: HiddenT
    accumulated_text: str = ""
    segment_index: int = 0
    segments: tuple[str, ...] = ()
    total_segments: int | None = None


@dataclass
class SegmentedEnvironment(Generic[HiddenT]):
    """Wrapper that segments single-step environments into multi-step.

    Takes a single-step environment and a segmenter, and exposes a multi-step
    interface where each step processes one segment of the response.

    This enables:
    - Per-step rewards from process reward models
    - Tree search / branching at segment boundaries
    - Early stopping and intervention
    - Analysis of reasoning traces

    Example (replay mode):
        >>> base_env = HuggingFaceAdapter().get_environment("gsm8k", size=1)
        >>> env = SegmentedEnvironment(base_env, SentenceSegmenter())
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> results = env.replay(state, "Step 1: Add. Step 2: Get 42. <answer>42</answer>")
        >>> print(f"Got {len(results)} step results")

    Example (generation-time):
        >>> env = SegmentedEnvironment(base_env, SentenceSegmenter())
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> while not state.metadata.is_terminal:
        ...     segment = model.generate_until_boundary(state, env.segmenter)
        ...     result = env.step(state, Action(text=segment))
        ...     state = result.next_state
    """

    _env: Environment[HiddenT]
    _segmenter: Segmenter
    _reward_functions: tuple[RewardFunction, ...] | None = None

    def __post_init__(self) -> None:
        """Initialize reward functions if not provided."""
        if self._reward_functions is None:
            object.__setattr__(self, "_reward_functions", self._env.reward_functions)

    @property
    def spec(self) -> EnvironmentSpec:
        """Return spec with is_multi_turn=True.

        The max_steps is set to None since the number of segments
        depends on the response content.
        """
        base_spec = self._env.spec
        return EnvironmentSpec(
            name=f"{base_spec.name}_segmented",
            adapter=base_spec.adapter,
            max_steps=None,  # Variable based on response
            observation_type=base_spec.observation_type,
            action_type=Action,
            is_multi_turn=True,
            metadata={
                **base_spec.metadata,
                "base_environment": base_spec.name,
                "segmenter": type(self._segmenter).__name__,
            },
        )

    @property
    def available_tools(self) -> tuple:
        """No tools available in segmented environments."""
        return ()

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        """Get reward functions used by this environment."""
        return self._reward_functions or ()

    @property
    def segmenter(self) -> Segmenter:
        """Get the segmenter used by this environment."""
        return self._segmenter

    @property
    def base_env(self) -> Environment[HiddenT]:
        """Get the underlying single-step environment."""
        return self._env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[SegmentedHidden[HiddenT]], dict[str, Any]]:
        """Reset the environment and return initial state.

        Initializes the underlying environment and wraps its hidden state
        with segmentation tracking.

        Args:
            seed: Random seed for reproducibility.
            options: Environment-specific options (e.g., task_index).

        Returns:
            Tuple of (initial_state, info_dict).
        """
        base_state, info = self._env.reset(seed=seed, options=options)

        # Wrap hidden state with segmentation tracking
        segmented_hidden = SegmentedHidden(
            base_hidden=base_state.hidden,
            accumulated_text="",
            segment_index=0,
            segments=(),
            total_segments=None,
        )

        state = State(
            observation=base_state.observation,
            hidden=segmented_hidden,
            metadata=base_state.metadata,
        )

        return state, info

    def step(
        self,
        state: State[SegmentedHidden[HiddenT]],
        action: Action,
    ) -> StepResult[SegmentedHidden[HiddenT]]:
        """Process one segment of the response.

        Accumulates the segment to the state and computes intermediate rewards.
        On the final segment (or when finalize() is called), calls the underlying
        environment's step() to get final rewards.

        The step is considered terminal when:
        - In replay mode: segment_index reaches total_segments - 1
        - In generation mode: finalize() is explicitly called

        Args:
            state: Current state.
            action: Action containing one segment of text.

        Returns:
            StepResult with next state, rewards, and done flags.
        """
        segment_text = action.text
        hidden = state.hidden

        # Update accumulated text and segments
        new_accumulated = hidden.accumulated_text + segment_text
        new_segments = hidden.segments + (segment_text,)
        new_segment_index = hidden.segment_index + 1

        # Determine if this is the final segment
        is_final = hidden.total_segments is not None and new_segment_index >= hidden.total_segments

        if is_final:
            # Final segment: call underlying environment to get outcome rewards
            return self._finalize_episode(state, new_accumulated, new_segments)

        # Intermediate step: compute step rewards
        new_hidden = SegmentedHidden(
            base_hidden=hidden.base_hidden,
            accumulated_text=new_accumulated,
            segment_index=new_segment_index,
            segments=new_segments,
            total_segments=hidden.total_segments,
        )

        next_state = State(
            observation=state.observation,
            hidden=new_hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=False,
                info={
                    **state.metadata.info,
                    "segment_index": new_segment_index,
                },
            ),
        )

        # Compute intermediate rewards
        rewards = self.compute_rewards(state, action, next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=False,
            truncated=False,
            info={
                "segment_index": new_segment_index,
                "accumulated_text": new_accumulated,
                "is_intermediate": True,
            },
        )

    def finalize(
        self,
        state: State[SegmentedHidden[HiddenT]],
    ) -> StepResult[SegmentedHidden[HiddenT]]:
        """Explicitly end the episode with accumulated text.

        Useful in generation-time mode when the model has finished generating
        and no more segments are coming.

        Args:
            state: Current state with accumulated text.

        Returns:
            StepResult with final rewards and terminal state.
        """
        return self._finalize_episode(
            state,
            state.hidden.accumulated_text,
            state.hidden.segments,
        )

    def _finalize_episode(
        self,
        state: State[SegmentedHidden[HiddenT]],
        accumulated_text: str,
        segments: tuple[str, ...],
    ) -> StepResult[SegmentedHidden[HiddenT]]:
        """Finalize the episode by calling the underlying environment.

        Args:
            state: Current state.
            accumulated_text: All accumulated text.
            segments: All segments.

        Returns:
            StepResult with final rewards.
        """
        # Create state with base hidden for underlying env
        base_state = State(
            observation=state.observation,
            hidden=state.hidden.base_hidden,
            metadata=state.metadata,
        )

        # Call underlying environment with full accumulated text
        full_action = Action(text=accumulated_text)
        base_result = self._env.step(base_state, full_action)

        # Wrap the result's hidden state
        final_hidden = SegmentedHidden(
            base_hidden=base_result.next_state.hidden,
            accumulated_text=accumulated_text,
            segment_index=len(segments),
            segments=segments,
            total_segments=len(segments),
        )

        final_state = State(
            observation=base_result.next_state.observation,
            hidden=final_hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=True,
                info={
                    **base_result.next_state.metadata.info,
                    "segment_index": len(segments),
                    "total_segments": len(segments),
                },
            ),
        )

        return StepResult(
            next_state=final_state,
            rewards=base_result.rewards,
            terminated=True,
            truncated=False,
            info={
                **base_result.info,
                "segment_index": len(segments),
                "total_segments": len(segments),
                "accumulated_text": accumulated_text,
                "is_intermediate": False,
            },
        )

    def replay(
        self,
        state: State[SegmentedHidden[HiddenT]],
        full_response: str,
    ) -> list[StepResult[SegmentedHidden[HiddenT]]]:
        """Segment a full response and step through all segments.

        Convenience method for post-hoc analysis. Segments the full response
        and returns a list of StepResults for each segment.

        Args:
            state: Initial state (from reset()).
            full_response: The complete model response to segment.

        Returns:
            List of StepResults, one for each segment. The last result
            will have is_terminal=True.
        """
        segments = self._segmenter.segment(full_response)

        if not segments:
            # No segments found, treat entire response as one segment
            segments = [full_response] if full_response.strip() else []

        if not segments:
            # Empty response: finalize immediately
            return [self.finalize(state)]

        # Set total_segments so we know when to finalize
        initial_hidden = SegmentedHidden(
            base_hidden=state.hidden.base_hidden,
            accumulated_text="",
            segment_index=0,
            segments=(),
            total_segments=len(segments),
        )

        current_state = State(
            observation=state.observation,
            hidden=initial_hidden,
            metadata=state.metadata,
        )

        results = []
        for segment in segments:
            result = self.step(current_state, Action(text=segment))
            results.append(result)
            current_state = result.next_state

        return results

    def compute_rewards(
        self,
        state: State[SegmentedHidden[HiddenT]],
        action: Action,
        next_state: State[SegmentedHidden[HiddenT]],
    ) -> SignalBundle:
        """Compute rewards for a transition.

        For intermediate steps, returns an empty reward bundle by default.
        Custom reward functions (e.g., process reward models) can be added
        via the reward_functions parameter.

        Args:
            state: State before action.
            action: Action taken (segment text).
            next_state: State after action.

        Returns:
            SignalBundle containing reward signals.
        """
        # For intermediate steps with no custom reward functions, return empty
        if not next_state.metadata.is_terminal:
            if self._reward_functions == self._env.reward_functions:
                # Using base env's reward functions, which only work at episode end
                return SignalBundle.empty()

        # Compute rewards from all reward functions
        signals = []
        for reward_fn in self.reward_functions:
            # Create base states for reward functions that expect base hidden type
            base_state = State(
                observation=state.observation,
                hidden=state.hidden.base_hidden,
                metadata=state.metadata,
            )
            base_next_state = State(
                observation=next_state.observation,
                hidden=next_state.hidden.base_hidden,
                metadata=next_state.metadata,
            )

            signal = reward_fn.compute(base_state, action, base_next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def __len__(self) -> int:
        """Number of tasks in the underlying environment."""
        if hasattr(self._env, "__len__"):
            return len(self._env)  # type: ignore
        raise TypeError(f"{type(self._env).__name__} has no len()")
