"""TRL framework integration.

Provides reward functions, dataset helpers, and multi-turn rollout function
compatible with TRL's GRPOTrainer. Single-turn functions are thin wrappers
around Scorer; rollout function drives multi-turn environments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llenvs.core.environment import Environment
from llenvs.integrations.dataset_provider import DatasetProvider
from llenvs.integrations.scoring import Scorer


def make_trl_reward_fn(
    environment: Environment[Any],
) -> Callable[..., list[float]]:
    """Create a TRL-compatible reward function.

    Returns a callable with TRL's expected signature:
        reward_func(prompts, completions, **kwargs) -> list[float]

    The function requires task_indices in kwargs to map prompts to
    environment tasks.

    Args:
        environment: A single-turn Environment instance.

    Returns:
        A callable compatible with TRL's reward function interface.
    """
    scorer = Scorer(environment)

    def reward_func(
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        task_indices = kwargs["task_indices"]
        results = scorer.score_batch(task_indices, completions)
        return [r.total for r in results]

    return reward_func


def make_trl_dataset(
    environment: Environment[Any],
    num_tasks: int | None = None,
) -> Any:
    """Create a TRL-compatible HuggingFace Dataset.

    Returns a Dataset with 'prompt' column suitable for GRPOTrainer.
    Requires the `datasets` package.

    Args:
        environment: An Environment instance.
        num_tasks: Maximum number of tasks. None means all tasks.

    Returns:
        A datasets.Dataset with prompt, ground_truth, and task_index columns.
    """
    provider = DatasetProvider(environment)
    total = len(provider)
    count = min(num_tasks, total) if num_tasks is not None else total
    indices = list(range(count))
    return provider.to_hf_dataset(indices)


def make_trl_rollout_fn(
    environment: Environment[Any],
    tokenizer: Any,
    max_steps: int = 20,
) -> Callable[..., Any]:
    """Create a TRL-compatible rollout function for multi-turn environments.

    Returns an async callable that runs a full episode and returns
    prompt_ids, response_ids, response_mask, and rewards.

    Args:
        environment: A multi-turn Environment instance.
        tokenizer: Any object with encode(str) -> list[int].
        max_steps: Maximum environment steps before truncating.

    Returns:
        Async callable compatible with TRL's OpenEnv integration.
    """
    from llenvs.integrations.verl import LLEnvsAgentLoop

    loop = LLEnvsAgentLoop(environment, tokenizer, max_steps=max_steps)

    async def rollout_func(
        task_index: int,
        generate_fn: Callable[..., Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await loop.run(task_index=task_index, generate_fn=generate_fn)

    return rollout_func
