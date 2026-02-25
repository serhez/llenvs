"""Tests for OpenRLHF integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Observation, State, StateMetadata
from llenvs.integrations.openrlhf import make_openrlhf_reward_fn

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


class TestMakeOpenrlhfRewardFn:
    def test_returns_callable(self):
        env = MockEnv()
        fn = make_openrlhf_reward_fn(env)
        assert callable(fn)

    def test_correct_signature(self):
        """OpenRLHF reward fn: reward_func(queries, prompts, labels) -> dict"""
        env = MockEnv()
        reward_func = make_openrlhf_reward_fn(env)
        result = reward_func(
            queries=["Q0?0"],  # query = prompt + completion
            prompts=["Q0?"],
            labels=["0"],
            task_indices=[0],
        )
        assert isinstance(result, dict)
        assert "rewards" in result
        assert "scores" in result
        assert len(result["rewards"]) == 1

    def test_correct_scoring(self):
        env = MockEnv()
        reward_func = make_openrlhf_reward_fn(env)
        result = reward_func(
            queries=["Q0?0", "Q1?wrong"],
            prompts=["Q0?", "Q1?"],
            labels=["0", "1"],
            task_indices=[0, 1],
        )
        assert result["rewards"][0] == 1.0
        assert result["rewards"][1] == 0.0
        assert result["scores"][0] == 1.0
        assert result["scores"][1] == 0.0

    def test_extra_logs(self):
        env = MockEnv()
        reward_func = make_openrlhf_reward_fn(env)
        result = reward_func(
            queries=["Q0?0"],
            prompts=["Q0?"],
            labels=["0"],
            task_indices=[0],
        )
        assert "extra_logs" in result
        assert isinstance(result["extra_logs"], dict)
