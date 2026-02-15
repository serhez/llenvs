"""Token masking for multi-turn RL training.

Converts trajectories into token-level masks that distinguish model-generated
tokens (mask=1, receives gradient) from environment-generated tokens (mask=0,
no gradient). Works with any tokenizer that has encode(str) -> list[int].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from llenvs.core.trajectory import Trajectory


@dataclass(frozen=True)
class TokenSpan:
    """A contiguous span of tokens from a single source.

    Attributes:
        text: The original text that was tokenized.
        token_ids: Token IDs for this span.
        source: Whether this span is from the model or environment.
        step_index: Which trajectory step this span belongs to.
    """

    text: str
    token_ids: tuple[int, ...]
    source: Literal["model", "environment"]
    step_index: int


@dataclass(frozen=True)
class MaskedTrajectory:
    """A trajectory with token-level source masks for RL training.

    Attributes:
        prompt_ids: Token IDs for the initial observation (not part of response).
        response_ids: All response token IDs concatenated.
        response_mask: 1=model token, 0=environment token, aligned with response_ids.
        spans: Structured span information for debugging/analysis.
        rewards: Per-step reward totals, one per trajectory transition.
    """

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    spans: tuple[TokenSpan, ...]
    rewards: tuple[float, ...]


class TrajectoryMasker:
    """Converts trajectories into masked token sequences for RL training.

    Args:
        tokenizer: Any object with encode(str) -> list[int]. Works with
            HuggingFace tokenizers, tiktoken, and vLLM tokenizers.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def mask_trajectory(self, trajectory: Trajectory[Any]) -> MaskedTrajectory:
        """Convert a trajectory into a masked token sequence.

        Walks the trajectory's transitions in order:
        1. Initial observation prompt -> prompt_ids
        2. For each transition:
           - Action text/tool calls -> model tokens (mask=1)
           - If not terminal: next observation content -> environment tokens (mask=0)
        3. Concatenate all response spans into response_ids + response_mask

        Args:
            trajectory: A completed or in-progress Trajectory.

        Returns:
            MaskedTrajectory with token-level masks.
        """
        prompt_ids = tuple(self._tokenizer.encode(trajectory.initial_state.observation.prompt))

        spans: list[TokenSpan] = []
        rewards: list[float] = []
        transitions = trajectory.transitions

        for step_idx, transition in enumerate(transitions):
            action = transition.action
            rewards.append(transition.rewards.total)

            # Model tokens: action text
            if action.text is not None:
                model_ids = tuple(self._tokenizer.encode(action.text))
                spans.append(
                    TokenSpan(
                        text=action.text,
                        token_ids=model_ids,
                        source="model",
                        step_index=step_idx,
                    )
                )

            # Model tokens: serialized tool calls
            if action.has_tool_calls:
                tc_text = self._serialize_tool_calls(action.tool_calls)
                tc_ids = tuple(self._tokenizer.encode(tc_text))
                spans.append(
                    TokenSpan(
                        text=tc_text,
                        token_ids=tc_ids,
                        source="model",
                        step_index=step_idx,
                    )
                )

            # Environment tokens: only if not the last transition
            is_last = step_idx == len(transitions) - 1
            if not is_last:
                env_text = self._get_environment_text(transition)
                if env_text:
                    env_ids = tuple(self._tokenizer.encode(env_text))
                    spans.append(
                        TokenSpan(
                            text=env_text,
                            token_ids=env_ids,
                            source="environment",
                            step_index=step_idx,
                        )
                    )

        # Concatenate all spans
        response_ids: list[int] = []
        response_mask: list[int] = []
        for span in spans:
            response_ids.extend(span.token_ids)
            mask_value = 1 if span.source == "model" else 0
            response_mask.extend([mask_value] * len(span.token_ids))

        return MaskedTrajectory(
            prompt_ids=prompt_ids,
            response_ids=tuple(response_ids),
            response_mask=tuple(response_mask),
            spans=tuple(spans),
            rewards=tuple(rewards),
        )

    def mask_batch(self, trajectories: list[Trajectory[Any]]) -> list[MaskedTrajectory]:
        """Convert multiple trajectories into masked token sequences.

        Args:
            trajectories: List of trajectories to mask.

        Returns:
            List of MaskedTrajectory instances.
        """
        return [self.mask_trajectory(t) for t in trajectories]

    def _serialize_tool_calls(self, tool_calls: tuple[Any, ...]) -> str:
        """Serialize tool calls to a text representation."""
        parts = []
        for tc in tool_calls:
            parts.append(
                json.dumps(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )
            )
        return "\n".join(parts)

    def _get_environment_text(self, transition: Any) -> str:
        """Extract environment-generated text from a transition's next state."""
        next_state = transition.next_state
        parts: list[str] = []

        # Tool results
        if next_state.observation.tool_results:
            for result in next_state.observation.tool_results:
                output = result.output
                if isinstance(output, dict):
                    output = json.dumps(output)
                parts.append(str(output))

        # New prompt content (if different from current state's prompt)
        next_prompt = next_state.observation.prompt
        current_prompt = transition.state.observation.prompt
        if next_prompt != current_prompt:
            parts.append(next_prompt)

        return "".join(parts)
