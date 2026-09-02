"""Dataset provider for RL training framework data pipelines.

Iterates over environment tasks to provide prompts and ground truths
in formats suitable for RL training dataloaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import Environment
from llenvs.core.state import ObservationImages
from llenvs.core.tools import ToolDefinition


@dataclass(frozen=True)
class TaskItem:
    """A single task from an environment, ready for RL training.

    Attributes:
        task_index: Index of this task in the environment.
        prompt: The prompt text from the observation.
        messages: Chat-format messages from the observation.
        ground_truth: Expected answer (None for multi-turn environments).
        metadata: Additional task metadata.
        images: Images from the observation, separated by source.
        available_tools: Tools advertised by the initial observation.
    """

    task_index: int
    prompt: str
    messages: tuple[dict[str, Any], ...]
    ground_truth: str | None
    metadata: dict[str, Any]
    images: ObservationImages = ObservationImages()
    available_tools: tuple[ToolDefinition, ...] = ()


class DatasetProvider:
    """Provides prompts and ground truths from an environment.

    Iterates over environment tasks to produce TaskItems suitable for
    RL training data pipelines. Works with both single-turn and multi-turn
    environments (multi-turn returns ground_truth=None).

    Args:
        environment: An Environment instance.
    """

    def __init__(self, environment: Environment[Any]) -> None:
        if not environment.spec.supports_task_index:
            raise TypeError(
                f"DatasetProvider requires an environment that supports task indexing, "
                f"but '{environment.spec.name}' does not (supports_task_index=False)."
            )
        if not environment.spec.supports_len:
            raise TypeError(
                f"DatasetProvider requires an environment that supports len(), "
                f"but '{environment.spec.name}' does not (supports_len=False)."
            )
        self._env = environment

    def __len__(self) -> int:
        """Number of tasks in the environment."""
        return len(self._env)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> TaskItem:
        """Get a single TaskItem by index.

        Args:
            index: Task index.

        Returns:
            TaskItem with prompt, messages, ground truth, and metadata.
        """
        state, info = self._env.reset(options={"task_index": index})

        # Extract ground truth from hidden state if available
        ground_truth = None
        if hasattr(state.hidden, "expected_answer"):
            ground_truth = state.hidden.expected_answer

        return TaskItem(
            task_index=index,
            prompt=state.observation.prompt,
            messages=state.observation.messages,
            ground_truth=ground_truth,
            metadata={
                "episode_id": state.metadata.episode_id,
                **info,
            },
            images=state.observation.get_images(),
            available_tools=state.observation.available_tools,
        )

    def get_items(self, indices: list[int] | None = None) -> list[TaskItem]:
        """Get multiple TaskItems.

        Args:
            indices: Specific task indices to retrieve. None means all tasks.

        Returns:
            List of TaskItems.
        """
        if indices is None:
            indices = list(range(len(self)))
        return [self[i] for i in indices]

    def to_hf_dataset(self, indices: list[int] | None = None) -> Any:
        """Convert to a HuggingFace Dataset.

        Requires the `datasets` package to be installed.

        Args:
            indices: Specific task indices. None means all tasks.

        Returns:
            A datasets.Dataset with columns: task_index, prompt, ground_truth.
        """
        from datasets import Dataset

        items = self.get_items(indices)
        data: dict[str, list[Any]] = {
            "task_index": [item.task_index for item in items],
            "prompt": [item.prompt for item in items],
            "ground_truth": [item.ground_truth for item in items],
            "messages": [list(item.messages) for item in items],
        }

        # Include images as serialized dicts (base64 data + media_type)
        has_images = any(item.images for item in items)
        if has_images:
            data["images"] = [
                [{"data": img.data, "media_type": img.media_type} for img in item.images.all]
                for item in items
            ]

        return Dataset.from_dict(data)

    @classmethod
    def from_config(cls, config: Any) -> DatasetProvider:
        """Create a DatasetProvider from an EnvironmentConfig.

        Args:
            config: EnvironmentConfig instance.

        Returns:
            Configured DatasetProvider.
        """
        from llenvs.core.config import EnvironmentFactory

        environment = EnvironmentFactory.create(config)
        return cls(environment)
