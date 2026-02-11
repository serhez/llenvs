"""veRL framework integration.

Provides reward functions, dataset helpers, and multi-turn AgentLoop
compatible with veRL's training pipeline. Single-turn functions are
thin wrappers around Scorer; AgentLoop drives multi-turn environments
with veRL's generation function.
"""

from __future__ import annotations

from typing import Any, Callable

from llenvs.core.environment import Environment
from llenvs.core.state import Action
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.integrations.scoring import Scorer
from llenvs.integrations.dataset_provider import DatasetProvider
from llenvs.integrations.token_mask import TrajectoryMasker


def make_verl_reward_fn(
    environment: Environment[Any],
) -> Callable[[str, str, str, dict[str, Any]], float]:
    """Create a veRL-compatible reward function.

    Returns a callable with veRL's expected signature:
        compute_score(data_source, solution_str, ground_truth, extra_info) -> float

    The function uses task_index from extra_info to score against the
    correct environment task.

    Args:
        environment: A single-turn Environment instance.

    Returns:
        A callable compatible with veRL's reward function interface.
    """
    scorer = Scorer(environment)

    def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, Any],
    ) -> float:
        task_index = extra_info["task_index"]
        result = scorer.score(task_index, solution_str)
        return result.total

    return compute_score


def make_verl_dataset(
    environment: Environment[Any],
    num_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Create a veRL-compatible dataset.

    Returns a list of dicts with keys expected by veRL's DataLoader:
    'prompt', 'ground_truth', 'data_source', 'extra_info'.

    Args:
        environment: An Environment instance.
        num_tasks: Maximum number of tasks. None means all tasks.

    Returns:
        List of dicts suitable for veRL's training pipeline.
    """
    provider = DatasetProvider(environment)
    total = len(provider)
    count = min(num_tasks, total) if num_tasks is not None else total
    indices = list(range(count))
    items = provider.get_items(indices)

    return [
        {
            "prompt": item.prompt,
            "ground_truth": item.ground_truth or "",
            "data_source": environment.spec.name,
            "extra_info": {"task_index": item.task_index, **item.metadata},
        }
        for item in items
    ]


class LLEnvsAgentLoop:
    """Runs llenvs multi-turn environments within veRL's agent loop.

    The RL framework owns generation (needs token-level logprobs for policy
    gradients). This class drives the environment loop, letting veRL's
    generate_fn produce model responses.

    Args:
        environment: A multi-turn Environment instance.
        tokenizer: Any object with encode(str) -> list[int].
        max_steps: Maximum number of environment steps before truncating.
    """

    def __init__(
        self,
        environment: Environment[Any],
        tokenizer: Any,
        max_steps: int = 20,
    ) -> None:
        self._env = environment
        self._tokenizer = tokenizer
        self._max_steps = max_steps
        self._masker = TrajectoryMasker(tokenizer)

    async def run(
        self,
        task_index: int,
        generate_fn: Callable[..., Any],
    ) -> dict[str, Any]:
        """Run one episode.

        Args:
            task_index: Which task to run.
            generate_fn: Async callable that takes messages (list[dict])
                and returns model response text (str).

        Returns:
            Dict with prompt_ids, response_ids, response_mask, rewards.
        """
        state, _ = self._env.reset(options={"task_index": task_index})
        trajectory = Trajectory.create(state)

        for _ in range(self._max_steps):
            # Get model response via framework's generation
            messages = list(state.observation.messages)
            if not messages:
                messages = [{"role": "user", "content": state.observation.prompt}]

            response_text = await generate_fn(messages)
            action = Action.from_text(response_text)

            # Step the environment
            step_result = self._env.step(state, action)

            transition = Transition(
                state=state,
                action=action,
                next_state=step_result.next_state,
                rewards=step_result.rewards,
            )
            trajectory.add_transition(transition)

            if step_result.done:
                break

            state = step_result.next_state

        # Convert trajectory to masked token sequence
        masked = self._masker.mask_trajectory(trajectory)

        return {
            "prompt_ids": list(masked.prompt_ids),
            "response_ids": list(masked.response_ids),
            "response_mask": list(masked.response_mask),
            "rewards": list(masked.rewards),
        }
