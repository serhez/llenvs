"""Tests for the DatasetProvider integration class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardBundle, RewardFunction
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem


# ---------------------------------------------------------------------------
# Mock environments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockHidden:
    expected_answer: str
    task_index: int


class MockSingleTurnEnv:
    """Mock single-turn environment with ground truth in hidden state."""

    def __init__(self, num_tasks: int = 5) -> None:
        self._num_tasks = num_tasks

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock_single", adapter="mock", max_steps=1, is_multi_turn=False)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return ()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[_MockHidden], dict[str, Any]]:
        idx = (options or {}).get("task_index", 0)
        hidden = _MockHidden(expected_answer=str(idx * 10), task_index=idx)
        state = State(
            observation=Observation(
                prompt=f"What is {idx} * 10?",
                messages=({"role": "user", "content": f"What is {idx} * 10?"},),
            ),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        )
        return state, {"task_index": idx}

    def step(self, state: State[_MockHidden], action: Action) -> StepResult[_MockHidden]:
        return StepResult(next_state=state, rewards=RewardBundle.empty(), terminated=True)

    def compute_rewards(
        self, state: State[_MockHidden], action: Action, next_state: State[_MockHidden]
    ) -> RewardBundle:
        return RewardBundle.empty()

    def __len__(self) -> int:
        return self._num_tasks


class MockMultiTurnEnv:
    """Mock multi-turn environment — no expected_answer on hidden state."""

    def __init__(self, num_tasks: int = 3) -> None:
        self._num_tasks = num_tasks

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock_multi", adapter="mock", max_steps=5, is_multi_turn=True)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return ()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[dict], dict[str, Any]]:
        idx = (options or {}).get("task_index", 0)
        state = State(
            observation=Observation(
                prompt=f"Multi-turn task {idx}",
                messages=({"role": "user", "content": f"Multi-turn task {idx}"},),
            ),
            hidden={"task_id": idx},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        )
        return state, {"task_index": idx}

    def step(self, state: State[dict], action: Action) -> StepResult[dict]:
        return StepResult(next_state=state, rewards=RewardBundle.empty(), terminated=False)

    def compute_rewards(
        self, state: State[dict], action: Action, next_state: State[dict]
    ) -> RewardBundle:
        return RewardBundle.empty()

    def __len__(self) -> int:
        return self._num_tasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaskItem:
    """Test the TaskItem dataclass."""

    def test_frozen(self):
        item = TaskItem(
            task_index=0,
            prompt="hello",
            messages=({"role": "user", "content": "hello"},),
            ground_truth="world",
            metadata={},
        )
        with pytest.raises(AttributeError):
            item.prompt = "changed"  # type: ignore[misc]

    def test_fields(self):
        item = TaskItem(
            task_index=3,
            prompt="question",
            messages=({"role": "user", "content": "question"},),
            ground_truth="answer",
            metadata={"key": "value"},
        )
        assert item.task_index == 3
        assert item.prompt == "question"
        assert len(item.messages) == 1
        assert item.ground_truth == "answer"
        assert item.metadata == {"key": "value"}


class TestDatasetProvider:
    """Test the DatasetProvider class."""

    def test_len(self):
        env = MockSingleTurnEnv(num_tasks=7)
        provider = DatasetProvider(env)
        assert len(provider) == 7

    def test_getitem(self):
        env = MockSingleTurnEnv()
        provider = DatasetProvider(env)
        item = provider[2]
        assert item.task_index == 2
        assert item.prompt == "What is 2 * 10?"
        assert item.ground_truth == "20"
        assert len(item.messages) == 1
        assert item.messages[0]["content"] == "What is 2 * 10?"

    def test_getitem_all_tasks(self):
        env = MockSingleTurnEnv(num_tasks=3)
        provider = DatasetProvider(env)
        for i in range(3):
            item = provider[i]
            assert item.task_index == i
            assert item.ground_truth == str(i * 10)

    def test_ground_truth_none_for_multi_turn(self):
        env = MockMultiTurnEnv()
        provider = DatasetProvider(env)
        item = provider[0]
        assert item.ground_truth is None

    def test_get_items_all(self):
        env = MockSingleTurnEnv(num_tasks=3)
        provider = DatasetProvider(env)
        items = provider.get_items()
        assert len(items) == 3
        assert items[0].task_index == 0
        assert items[1].task_index == 1
        assert items[2].task_index == 2

    def test_get_items_subset(self):
        env = MockSingleTurnEnv(num_tasks=5)
        provider = DatasetProvider(env)
        items = provider.get_items(indices=[1, 3])
        assert len(items) == 2
        assert items[0].task_index == 1
        assert items[1].task_index == 3

    def test_multi_turn_prompts(self):
        env = MockMultiTurnEnv()
        provider = DatasetProvider(env)
        item = provider[1]
        assert item.prompt == "Multi-turn task 1"
        assert item.messages[0]["content"] == "Multi-turn task 1"

    def test_metadata_includes_episode_id(self):
        env = MockSingleTurnEnv()
        provider = DatasetProvider(env)
        item = provider[0]
        assert "episode_id" in item.metadata

    def test_to_hf_dataset(self):
        """Test HF dataset conversion (skipped if datasets not installed)."""
        pytest.importorskip("datasets")
        env = MockSingleTurnEnv(num_tasks=3)
        provider = DatasetProvider(env)
        ds = provider.to_hf_dataset()
        assert len(ds) == 3
        assert "prompt" in ds.column_names
        assert "ground_truth" in ds.column_names
        assert "task_index" in ds.column_names
        assert ds[0]["prompt"] == "What is 0 * 10?"

    def test_to_hf_dataset_with_indices(self):
        """Test HF dataset conversion with specific indices."""
        pytest.importorskip("datasets")
        env = MockSingleTurnEnv(num_tasks=5)
        provider = DatasetProvider(env)
        ds = provider.to_hf_dataset(indices=[1, 3])
        assert len(ds) == 2
        assert ds[0]["task_index"] == 1
        assert ds[1]["task_index"] == 3
