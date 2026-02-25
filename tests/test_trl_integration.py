"""Tests for TRL integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Observation, State, StateMetadata
from llenvs.integrations.trl import make_trl_dataset, make_trl_reward_fn

# ---------------------------------------------------------------------------
# Mock environment
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

    def compute(
        self, state: State[_MockHidden], action: Any, next_state: State[_MockHidden]
    ) -> Signal:
        correct = (action.text or "") == state.hidden.expected_answer
        return Signal(reward=1.0 if correct else 0.0, name=self.name, reward_type=self.reward_type)


class MockEnv:
    def __init__(self, num_tasks: int = 5) -> None:
        self._num_tasks = num_tasks
        self._prompts_to_index: dict[str, int] = {}
        for i in range(num_tasks):
            self._prompts_to_index[f"Q{i}?"] = i

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
            observation=state.observation,
            hidden=state.hidden,
            metadata=StateMetadata(step=1, episode_id=state.metadata.episode_id, is_terminal=True),
        )
        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=True,
            info={"extracted_answer": action.text},
        )

    def compute_rewards(self, state, action, next_state):
        signals = tuple(fn.compute(state, action, next_state) for fn in self.reward_functions)
        return SignalBundle(signals=signals)

    def __len__(self):
        return self._num_tasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeTrlRewardFn:
    def test_returns_callable(self):
        env = MockEnv()
        fn = make_trl_reward_fn(env)
        assert callable(fn)

    def test_correct_signature(self):
        """TRL reward fn signature: reward_func(prompts, completions, **kwargs) -> list[float]"""
        env = MockEnv()
        reward_func = make_trl_reward_fn(env)
        # Task 0 expects "0", task 1 expects "1"
        results = reward_func(
            prompts=["Q0?", "Q1?"],
            completions=["0", "1"],
            task_indices=[0, 1],
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] == 1.0
        assert results[1] == 1.0

    def test_incorrect_response(self):
        env = MockEnv()
        reward_func = make_trl_reward_fn(env)
        results = reward_func(
            prompts=["Q0?"],
            completions=["wrong"],
            task_indices=[0],
        )
        assert results[0] == 0.0

    def test_batch_scoring(self):
        env = MockEnv()
        reward_func = make_trl_reward_fn(env)
        results = reward_func(
            prompts=["Q0?", "Q1?", "Q2?"],
            completions=["0", "wrong", "2"],
            task_indices=[0, 1, 2],
        )
        assert results == [1.0, 0.0, 1.0]


class TestMakeTrlDataset:
    def test_returns_hf_dataset(self):
        """make_trl_dataset returns a HF Dataset (skipped if datasets not installed)."""
        pytest.importorskip("datasets")
        env = MockEnv(num_tasks=3)
        ds = make_trl_dataset(env)
        assert len(ds) == 3
        assert "prompt" in ds.column_names

    def test_num_tasks_limits(self):
        pytest.importorskip("datasets")
        env = MockEnv(num_tasks=10)
        ds = make_trl_dataset(env, num_tasks=3)
        assert len(ds) == 3

    def test_prompt_column(self):
        pytest.importorskip("datasets")
        env = MockEnv(num_tasks=3)
        ds = make_trl_dataset(env)
        assert ds[0]["prompt"] == "Q0?"
        assert ds[2]["prompt"] == "Q2?"
