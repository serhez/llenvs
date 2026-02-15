"""OpenRLHF framework integration.

Provides reward functions compatible with OpenRLHF's training pipeline.
Single-turn reward function is a thin wrapper around Scorer.
"""

from __future__ import annotations

from typing import Any, Callable

from llenvs.core.environment import Environment
from llenvs.integrations.scoring import Scorer


def make_openrlhf_reward_fn(
    environment: Environment[Any],
) -> Callable[..., dict[str, Any]]:
    """Create an OpenRLHF-compatible reward function.

    Returns a callable with OpenRLHF's expected signature:
        reward_func(queries, prompts, labels, **kwargs) -> dict

    The returned dict has keys 'rewards', 'scores', and 'extra_logs'.
    The function extracts completions by stripping the prompt from queries.

    Args:
        environment: A single-turn Environment instance.

    Returns:
        A callable compatible with OpenRLHF's reward function interface.
    """
    scorer = Scorer(environment)

    def reward_func(
        queries: list[str],
        prompts: list[str],
        labels: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        task_indices = kwargs["task_indices"]

        # Extract completions by removing prompt prefix from queries
        completions = []
        for query, prompt in zip(queries, prompts):
            if query.startswith(prompt):
                completions.append(query[len(prompt) :])
            else:
                completions.append(query)

        results = scorer.score_batch(task_indices, completions)
        rewards = [r.total for r in results]

        return {
            "rewards": rewards,
            "scores": rewards,  # OpenRLHF uses scores for logging
            "extra_logs": {
                "signals": [r.signals for r in results],
            },
        }

    return reward_func
