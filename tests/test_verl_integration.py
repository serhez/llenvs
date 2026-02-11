"""Tests for veRL integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType, RewardFunction
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.integrations.verl import make_verl_reward_fn, make_verl_dataset


# ---------------------------------------------------------------------------
# Mock environment (reused pattern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockHidden:
    expected_answer: str
    task_index: int


class _MockRewardFn:
    @property
    def name(self) -> str:
        return "correctness"

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(self, state: State[_MockHidden], action: Any, next_state: State[_MockHidden]) -> RewardSignal:
        correct = (action.text or "") == state.hidden.expected_answer
        return RewardSignal(value=1.0 if correct else 0.0, name=self.name, reward_type=self.reward_type)


class MockEnv:
    def __init__(self, num_tasks: int = 5) -> None:
        self._num_tasks = num_tasks

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock", adapter="mock", max_steps=1, is_multi_turn=False)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return (_MockRewardFn(),)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        idx = (options or {}).get("task_index", 0)
        hidden = _MockHidden(expected_answer=str(idx), task_index=idx)
        state = State(
            observation=Observation(prompt=f"Q{idx}?"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        )
        return state, {"task_index": idx}

    def step(self, state, action):
        rewards = self.compute_rewards(state, action, state)
        next_state = State(
            observation=state.observation, hidden=state.hidden,
            metadata=StateMetadata(step=1, episode_id=state.metadata.episode_id, is_terminal=True),
        )
        return StepResult(next_state=next_state, rewards=rewards, terminated=True,
                          info={"extracted_answer": action.text})

    def compute_rewards(self, state, action, next_state):
        signals = tuple(fn.compute(state, action, next_state) for fn in self.reward_functions)
        return RewardBundle(signals=signals)

    def __len__(self):
        return self._num_tasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeVerlRewardFn:
    def test_returns_callable(self):
        env = MockEnv()
        fn = make_verl_reward_fn(env)
        assert callable(fn)

    def test_correct_signature(self):
        """veRL reward fn signature: compute_score(data_source, solution_str, ground_truth, extra_info) -> float"""
        env = MockEnv()
        compute_score = make_verl_reward_fn(env)
        # Task 0 expects "0"
        result = compute_score("mock", "0", "0", {"task_index": 0})
        assert isinstance(result, float)
        assert result == 1.0

    def test_incorrect_response(self):
        env = MockEnv()
        compute_score = make_verl_reward_fn(env)
        result = compute_score("mock", "wrong", "0", {"task_index": 0})
        assert result == 0.0

    def test_uses_task_index_from_extra_info(self):
        env = MockEnv()
        compute_score = make_verl_reward_fn(env)
        # Task 3 expects "3"
        result = compute_score("mock", "3", "3", {"task_index": 3})
        assert result == 1.0


class TestMakeVerlDataset:
    def test_returns_list_of_dicts(self):
        env = MockEnv(num_tasks=3)
        dataset = make_verl_dataset(env)
        assert isinstance(dataset, list)
        assert len(dataset) == 3

    def test_dict_keys(self):
        env = MockEnv()
        dataset = make_verl_dataset(env)
        item = dataset[0]
        assert "prompt" in item
        assert "ground_truth" in item
        assert "data_source" in item
        assert "extra_info" in item

    def test_num_tasks_limits(self):
        env = MockEnv(num_tasks=10)
        dataset = make_verl_dataset(env, num_tasks=3)
        assert len(dataset) == 3

    def test_extra_info_has_task_index(self):
        env = MockEnv()
        dataset = make_verl_dataset(env)
        assert dataset[2]["extra_info"]["task_index"] == 2
