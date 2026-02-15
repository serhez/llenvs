"""Tests for the TrajectoryMasker integration class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.reward import SignalBundle, Signal, RewardType
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.tools import ToolCall, ToolResult, ToolResultStatus
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.integrations.token_mask import (
    MaskedTrajectory,
    TokenSpan,
    TrajectoryMasker,
)


# ---------------------------------------------------------------------------
# Simple tokenizer mock
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Tokenizer that assigns one token per character (for predictable testing)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    prompt: str = "question",
    step: int = 0,
    is_terminal: bool = False,
    tool_results: tuple[ToolResult, ...] = (),
    messages: tuple[dict[str, Any], ...] = (),
) -> State[Any]:
    return State(
        observation=Observation(
            prompt=prompt,
            messages=messages,
            tool_results=tool_results,
        ),
        hidden={"expected": "answer"},
        metadata=StateMetadata(step=step, episode_id="ep_0", is_terminal=is_terminal),
    )


def _make_transition(
    state: State[Any],
    action_text: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    next_state: State[Any] | None = None,
    reward: float = 0.0,
) -> Transition[Any]:
    if next_state is None:
        next_state = _make_state(step=state.metadata.step + 1, is_terminal=True)
    action = Action(text=action_text, tool_calls=tool_calls)
    return Transition(
        state=state,
        action=action,
        next_state=next_state,
        rewards=SignalBundle.single(reward=reward, name="correctness"),
    )


# ---------------------------------------------------------------------------
# Tests: TokenSpan
# ---------------------------------------------------------------------------


class TestTokenSpan:
    def test_frozen(self):
        span = TokenSpan(text="hi", token_ids=(1, 2), source="model", step_index=0)
        with pytest.raises(AttributeError):
            span.text = "bye"  # type: ignore[misc]

    def test_fields(self):
        span = TokenSpan(text="hello", token_ids=(1, 2, 3), source="environment", step_index=1)
        assert span.text == "hello"
        assert span.token_ids == (1, 2, 3)
        assert span.source == "environment"
        assert span.step_index == 1


# ---------------------------------------------------------------------------
# Tests: MaskedTrajectory
# ---------------------------------------------------------------------------


class TestMaskedTrajectory:
    def test_frozen(self):
        mt = MaskedTrajectory(
            prompt_ids=(1, 2),
            response_ids=(3, 4),
            response_mask=(1, 1),
            spans=(),
            rewards=(1.0,),
        )
        with pytest.raises(AttributeError):
            mt.prompt_ids = ()  # type: ignore[misc]

    def test_mask_length_matches_response_ids(self):
        mt = MaskedTrajectory(
            prompt_ids=(1, 2),
            response_ids=(3, 4, 5),
            response_mask=(1, 0, 1),
            spans=(),
            rewards=(0.5,),
        )
        assert len(mt.response_ids) == len(mt.response_mask)


# ---------------------------------------------------------------------------
# Tests: TrajectoryMasker — single-turn
# ---------------------------------------------------------------------------


class TestTrajectoryMaskerSingleTurn:
    def test_single_turn_all_model_tokens(self):
        """Single-turn: only the model's response, all mask=1."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        initial = _make_state(prompt="Q?")
        action_text = "A!"
        terminal = _make_state(prompt="Q?", step=1, is_terminal=True)

        traj = Trajectory.create(initial)
        traj.add_transition(
            _make_transition(initial, action_text=action_text, next_state=terminal, reward=1.0)
        )

        result = masker.mask_trajectory(traj)

        assert result.prompt_ids == tuple(tokenizer.encode("Q?"))
        assert result.response_ids == tuple(tokenizer.encode("A!"))
        assert all(m == 1 for m in result.response_mask)
        assert len(result.response_ids) == len(result.response_mask)
        assert result.rewards == (1.0,)
        assert len(result.spans) == 1
        assert result.spans[0].source == "model"

    def test_single_turn_batch(self):
        """mask_batch processes multiple trajectories."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        trajectories = []
        for i in range(3):
            initial = _make_state(prompt=f"Q{i}")
            terminal = _make_state(prompt=f"Q{i}", step=1, is_terminal=True)
            traj = Trajectory.create(initial)
            traj.add_transition(_make_transition(initial, action_text=f"A{i}", next_state=terminal))
            trajectories.append(traj)

        results = masker.mask_batch(trajectories)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.prompt_ids == tuple(tokenizer.encode(f"Q{i}"))
            assert r.response_ids == tuple(tokenizer.encode(f"A{i}"))


# ---------------------------------------------------------------------------
# Tests: TrajectoryMasker — multi-turn
# ---------------------------------------------------------------------------


class TestTrajectoryMaskerMultiTurn:
    def test_multi_turn_interleaved_masking(self):
        """Multi-turn: model tokens=1, environment tokens=0."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        # Step 0: initial state
        s0 = _make_state(prompt="Q?", step=0)

        # Step 0→1: model says "try1", env responds with new prompt
        s1 = _make_state(prompt="feedback1", step=1)
        t0 = _make_transition(s0, action_text="try1", next_state=s1, reward=0.0)

        # Step 1→2: model says "try2", episode ends
        s2 = _make_state(prompt="feedback1", step=2, is_terminal=True)
        t1 = _make_transition(s1, action_text="try2", next_state=s2, reward=1.0)

        traj = Trajectory.create(s0)
        traj.add_transition(t0)
        traj.add_transition(t1)

        result = masker.mask_trajectory(traj)

        # Prompt = initial observation
        assert result.prompt_ids == tuple(tokenizer.encode("Q?"))

        # Response should contain: "try1" (model) + "feedback1" (env) + "try2" (model)
        expected_model1 = tokenizer.encode("try1")
        expected_env = tokenizer.encode("feedback1")
        expected_model2 = tokenizer.encode("try2")

        all_ids = list(expected_model1) + list(expected_env) + list(expected_model2)
        all_mask = [1] * len(expected_model1) + [0] * len(expected_env) + [1] * len(expected_model2)

        assert result.response_ids == tuple(all_ids)
        assert result.response_mask == tuple(all_mask)
        assert result.rewards == (0.0, 1.0)

    def test_multi_turn_spans_track_steps(self):
        """Verify spans record correct step_index."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        s0 = _make_state(prompt="Q", step=0)
        s1 = _make_state(prompt="R", step=1)
        s2 = _make_state(prompt="R", step=2, is_terminal=True)

        traj = Trajectory.create(s0)
        traj.add_transition(_make_transition(s0, action_text="a", next_state=s1))
        traj.add_transition(_make_transition(s1, action_text="b", next_state=s2))

        result = masker.mask_trajectory(traj)

        # Should have: model span (step 0), env span (step 0), model span (step 1)
        assert len(result.spans) == 3
        assert result.spans[0].source == "model"
        assert result.spans[0].step_index == 0
        assert result.spans[1].source == "environment"
        assert result.spans[1].step_index == 0
        assert result.spans[2].source == "model"
        assert result.spans[2].step_index == 1


# ---------------------------------------------------------------------------
# Tests: TrajectoryMasker — tool calls
# ---------------------------------------------------------------------------


class TestTrajectoryMaskerToolCalls:
    def test_tool_call_serialization(self):
        """Tool calls are serialized and marked as model tokens."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        tool_call = ToolCall(id="tc_1", name="search", arguments={"q": "hi"})
        tool_result = ToolResult(
            call_id="tc_1",
            tool_name="search",
            status=ToolResultStatus.SUCCESS,
            output="result text",
        )

        s0 = _make_state(prompt="Q?", step=0)
        s1 = _make_state(
            prompt="Q?",
            step=1,
            tool_results=(tool_result,),
        )
        s2 = _make_state(prompt="Q?", step=2, is_terminal=True)

        # Step 0: model calls a tool (no text, just tool call)
        t0 = _make_transition(s0, tool_calls=(tool_call,), next_state=s1)

        # Step 1: model gives final answer
        t1 = _make_transition(s1, action_text="final answer", next_state=s2, reward=1.0)

        traj = Trajectory.create(s0)
        traj.add_transition(t0)
        traj.add_transition(t1)

        result = masker.mask_trajectory(traj)

        # All spans should be present and correctly sourced
        model_spans = [s for s in result.spans if s.source == "model"]
        env_spans = [s for s in result.spans if s.source == "environment"]

        # Model: tool call serialization + final answer
        assert len(model_spans) >= 1
        # Environment: tool result
        assert len(env_spans) >= 1

        # Verify mask alignment
        assert len(result.response_ids) == len(result.response_mask)

    def test_tool_results_marked_as_environment(self):
        """Tool results from the environment should have mask=0."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        tool_result = ToolResult(
            call_id="tc_1",
            tool_name="search",
            status=ToolResultStatus.SUCCESS,
            output="result",
        )

        s0 = _make_state(prompt="Q?", step=0)
        s1 = _make_state(prompt="Q?", step=1, tool_results=(tool_result,))
        s2 = _make_state(prompt="Q?", step=2, is_terminal=True)

        tool_call = ToolCall(id="tc_1", name="search", arguments={"q": "hi"})
        t0 = _make_transition(s0, tool_calls=(tool_call,), next_state=s1)
        t1 = _make_transition(s1, action_text="done", next_state=s2)

        traj = Trajectory.create(s0)
        traj.add_transition(t0)
        traj.add_transition(t1)

        result = masker.mask_trajectory(traj)

        # Find environment spans — they should contain tool result text
        env_spans = [s for s in result.spans if s.source == "environment"]
        assert len(env_spans) > 0
        # All environment token positions should have mask=0
        env_token_count = sum(len(s.token_ids) for s in env_spans)
        zero_mask_count = sum(1 for m in result.response_mask if m == 0)
        assert env_token_count == zero_mask_count


# ---------------------------------------------------------------------------
# Tests: Reconstruction
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_response_ids_reconstruct_text(self):
        """Decoding response_ids should reconstruct the interleaved text."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        s0 = _make_state(prompt="Q", step=0)
        s1 = _make_state(prompt="env", step=1)
        s2 = _make_state(prompt="env", step=2, is_terminal=True)

        traj = Trajectory.create(s0)
        traj.add_transition(_make_transition(s0, action_text="model1", next_state=s1))
        traj.add_transition(_make_transition(s1, action_text="model2", next_state=s2))

        result = masker.mask_trajectory(traj)

        decoded = tokenizer.decode(list(result.response_ids))
        assert "model1" in decoded
        assert "env" in decoded
        assert "model2" in decoded

    def test_empty_trajectory(self):
        """Trajectory with no transitions should produce empty response."""
        tokenizer = MockTokenizer()
        masker = TrajectoryMasker(tokenizer)

        s0 = _make_state(prompt="Q")
        traj = Trajectory.create(s0)

        result = masker.mask_trajectory(traj)

        assert result.prompt_ids == tuple(tokenizer.encode("Q"))
        assert result.response_ids == ()
        assert result.response_mask == ()
        assert result.spans == ()
        assert result.rewards == ()
