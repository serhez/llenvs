"""Tests for multi-turn RL framework integrations (AgentLoop, rollout_func)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Observation, State, StateMetadata
from llenvs.integrations.trl import make_trl_rollout_fn
from llenvs.integrations.verl import LLEnvsAgentLoop

# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Tokenizer that assigns one token per character."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# Mock multi-turn environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockHidden:
    target_steps: int
    task_index: int


class _MockRewardFn:
    @property
    def name(self) -> str:
        return "correctness"

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(self, state, action, next_state) -> Signal:
        done = next_state.metadata.is_terminal
        return Signal(reward=1.0 if done else 0.0, name=self.name, reward_type=self.reward_type)


class MockMultiTurnEnv:
    """Multi-turn environment that takes N steps to complete."""

    def __init__(self, steps_per_task: dict[int, int] | None = None) -> None:
        self._steps = steps_per_task or {0: 2, 1: 3, 2: 1}

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock_multi", adapter="mock", max_steps=10, is_multi_turn=True)

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
        target = self._steps.get(idx, 2)
        hidden = _MockHidden(target_steps=target, task_index=idx)
        state = State(
            observation=Observation(
                prompt=f"Task {idx}: complete in {target} steps",
                messages=({"role": "user", "content": f"Task {idx}: complete in {target} steps"},),
            ),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        )
        return state, {"task_index": idx}

    def step(self, state, action):
        next_step = state.metadata.step + 1
        done = next_step >= state.hidden.target_steps
        next_state = State(
            observation=Observation(
                prompt=f"Step {next_step} feedback",
                messages=state.observation.messages
                + (
                    {"role": "assistant", "content": action.text or ""},
                    {"role": "user", "content": f"Step {next_step} feedback"},
                ),
            ),
            hidden=state.hidden,
            metadata=StateMetadata(
                step=next_step,
                episode_id=state.metadata.episode_id,
                is_terminal=done,
            ),
        )
        rewards = self.compute_rewards(state, action, next_state)
        return StepResult(next_state=next_state, rewards=rewards, terminated=done)

    def compute_rewards(self, state, action, next_state):
        signals = tuple(fn.compute(state, action, next_state) for fn in self.reward_functions)
        return SignalBundle(signals=signals)

    def __len__(self):
        return len(self._steps)


# ---------------------------------------------------------------------------
# Tests: LLEnvsAgentLoop (veRL)
# ---------------------------------------------------------------------------


class TestLLEnvsAgentLoop:
    def test_init(self):
        env = MockMultiTurnEnv()
        tokenizer = MockTokenizer()
        loop = LLEnvsAgentLoop(env, tokenizer, max_steps=20)
        assert loop is not None

    def test_run_completes_episode(self):
        """AgentLoop runs until environment terminates."""
        env = MockMultiTurnEnv({0: 2})
        tokenizer = MockTokenizer()
        loop = LLEnvsAgentLoop(env, tokenizer, max_steps=20)

        call_count = 0

        async def mock_generate_fn(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            return f"response_{call_count}"

        result = asyncio.run(loop.run(task_index=0, generate_fn=mock_generate_fn))

        assert call_count == 2  # 2 steps to complete
        assert "prompt_ids" in result
        assert "response_ids" in result
        assert "response_mask" in result
        assert "rewards" in result
        assert len(result["response_ids"]) == len(result["response_mask"])

    def test_max_steps_truncation(self):
        """AgentLoop stops at max_steps even if not terminated."""
        env = MockMultiTurnEnv({0: 100})  # Would take 100 steps
        tokenizer = MockTokenizer()
        loop = LLEnvsAgentLoop(env, tokenizer, max_steps=3)

        call_count = 0

        async def mock_generate_fn(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            return f"step_{call_count}"

        asyncio.run(loop.run(task_index=0, generate_fn=mock_generate_fn))
        assert call_count == 3

    def test_mask_correctness(self):
        """Model tokens have mask=1, environment tokens have mask=0."""
        env = MockMultiTurnEnv({0: 2})
        tokenizer = MockTokenizer()
        loop = LLEnvsAgentLoop(env, tokenizer, max_steps=20)

        async def mock_generate_fn(messages: list[dict]) -> str:
            return "reply"

        result = asyncio.run(loop.run(task_index=0, generate_fn=mock_generate_fn))

        mask = result["response_mask"]
        # Should have both 1s (model) and 0s (environment feedback)
        assert 1 in mask
        assert 0 in mask

    def test_rewards_per_step(self):
        """Rewards list has one entry per step."""
        env = MockMultiTurnEnv({0: 3})
        tokenizer = MockTokenizer()
        loop = LLEnvsAgentLoop(env, tokenizer, max_steps=20)

        async def mock_generate_fn(messages: list[dict]) -> str:
            return "step"

        result = asyncio.run(loop.run(task_index=0, generate_fn=mock_generate_fn))
        assert len(result["rewards"]) == 3


# ---------------------------------------------------------------------------
# Tests: TRL rollout function
# ---------------------------------------------------------------------------


class TestMakeTrlRolloutFn:
    def test_returns_callable(self):
        env = MockMultiTurnEnv()
        tokenizer = MockTokenizer()
        fn = make_trl_rollout_fn(env, tokenizer, max_steps=10)
        assert callable(fn)

    def test_rollout_returns_expected_keys(self):
        env = MockMultiTurnEnv({0: 2})
        tokenizer = MockTokenizer()
        rollout_fn = make_trl_rollout_fn(env, tokenizer, max_steps=10)

        call_count = 0

        async def mock_generate_fn(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            return f"step_{call_count}"

        result = asyncio.run(rollout_fn(task_index=0, generate_fn=mock_generate_fn))
        assert "prompt_ids" in result
        assert "response_ids" in result
        assert "response_mask" in result
        assert "rewards" in result

    def test_rollout_mask_alignment(self):
        env = MockMultiTurnEnv({0: 2})
        tokenizer = MockTokenizer()
        rollout_fn = make_trl_rollout_fn(env, tokenizer, max_steps=10)

        async def mock_generate_fn(messages: list[dict]) -> str:
            return "ok"

        result = asyncio.run(rollout_fn(task_index=0, generate_fn=mock_generate_fn))
        assert len(result["response_ids"]) == len(result["response_mask"])
