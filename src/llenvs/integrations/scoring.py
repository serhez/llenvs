"""Standalone scoring API for RL training frameworks.

Wraps an Environment to expose a simple score(task_index, response) interface
that reuses the environment's reward computation without running full episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import Environment
from llenvs.core.state import Action


@dataclass(frozen=True)
class ScoringResult:
    """Result of scoring a single response.

    Attributes:
        total: Sum of all reward signals.
        signals: Mapping from reward name to value.
        extracted_answer: The answer the extractor found, or None if extraction
            failed. None unambiguously means no answer was extracted — useful
            for diagnosing formatting issues or inadequate prompts.
        metadata: Per-signal metadata from reward computation.
    """

    total: float
    signals: dict[str, float]
    extracted_answer: str | None
    metadata: dict[str, Any]


class Scorer:
    """Scores responses against environment tasks.

    Wraps a single-turn Environment to provide a simple scoring interface
    for RL training frameworks. Reuses the environment's compute_rewards()
    to produce reward signals without running full inference.

    Args:
        environment: A single-turn Environment instance.

    Raises:
        TypeError: If the environment is multi-turn.
    """

    def __init__(self, environment: Environment[Any]) -> None:
        if environment.spec.is_multi_turn:
            raise TypeError(
                f"Scorer only supports single-turn environments, but "
                f"'{environment.spec.name}' is multi-turn. Use the AgentLoop "
                f"or rollout_func integration for multi-turn training."
            )
        if not environment.spec.supports_task_index:
            raise TypeError(
                f"Scorer requires an environment that supports task indexing, but "
                f"'{environment.spec.name}' does not (supports_task_index=False)."
            )
        self._env = environment

    def score(self, task_index: int, response: str) -> ScoringResult:
        """Score a response against a specific task.

        Resets the environment to the given task, builds an Action from the
        response text, and calls step() to get rewards and extraction info.

        Args:
            task_index: Which task to score against.
            response: The model's response text.

        Returns:
            ScoringResult with total reward, per-signal breakdown, and metadata.
        """
        state, _ = self._env.reset(options={"task_index": task_index})
        action = Action.from_text(response)
        step_result = self._env.step(state, action)

        rewards = step_result.rewards
        signals = {s.name: s.reward for s in rewards.signals}
        metadata = {
            s.name: s.metadata for s in rewards.signals if s.metadata is not None
        }

        # Get extracted answer: prefer step info, fall back to reward metadata.
        # Both sources use None to mean "extraction failed" (not empty string).
        extracted_answer = step_result.info.get("extracted_answer")
        if extracted_answer is None:
            for signal in rewards.signals:
                if signal.metadata and "extracted" in signal.metadata:
                    extracted_answer = signal.metadata["extracted"]
                    break

        return ScoringResult(
            total=rewards.total,
            signals=signals,
            extracted_answer=extracted_answer,
            metadata=metadata,
        )

    def score_batch(
        self, task_indices: list[int], responses: list[str]
    ) -> list[ScoringResult]:
        """Score multiple responses in batch.

        Args:
            task_indices: Task indices to score against.
            responses: Corresponding response texts.

        Returns:
            List of ScoringResults, one per task/response pair.

        Raises:
            ValueError: If task_indices and responses have different lengths.
        """
        if len(task_indices) != len(responses):
            raise ValueError(
                f"task_indices and responses must have the same length, "
                f"got {len(task_indices)} and {len(responses)}"
            )
        return [
            self.score(idx, resp) for idx, resp in zip(task_indices, responses)
        ]

    @classmethod
    def from_config(cls, config: Any) -> Scorer:
        """Create a Scorer from an EnvironmentConfig.

        Args:
            config: EnvironmentConfig instance.

        Returns:
            Configured Scorer.
        """
        from llenvs.core.config import EnvironmentFactory

        environment = EnvironmentFactory.create(config)
        return cls(environment)
