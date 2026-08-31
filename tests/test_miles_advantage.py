"""Tests for the Tier-2 turn-level advantage hook (``turn_grpo``).

``turn_grpo`` is the trainer-side target of the miles fork's
``--custom-advantage-function-path``: it broadcasts precomputed per-decision
advantages (from ``rollout_data["decision_spans"]`` / ``["decision_advantages"]``,
lifted out of sample metadata by the fork's train-data conversion) over each
decision's response token span, then slices to this CP rank's tokens.

Tests run without miles: the CP seams (``_cp_size``, ``_cp_token_offsets``)
are monkeypatched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llenvs.integrations.miles import advantage as miles_advantage

torch = pytest.importorskip("torch")


def _args(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "turn_advantage_length_weighting": "uniform",
        "qkv_format": "thd",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _rollout_data(
    *,
    response_lengths: list[int],
    decision_spans: list[list[list[int]]] | None = None,
    decision_advantages: list[list[float]] | None = None,
    rewards: list[float] | None = None,
    total_lengths: list[int] | None = None,
) -> dict[str, Any]:
    n = len(response_lengths)
    data: dict[str, Any] = {
        "response_lengths": response_lengths,
        "total_lengths": total_lengths or [r + 10 for r in response_lengths],
        "rewards": rewards or [0.0] * n,
        "loss_masks": [torch.ones(r, dtype=torch.int32) for r in response_lengths],
    }
    if decision_spans is not None:
        data["decision_spans"] = decision_spans
    if decision_advantages is not None:
        data["decision_advantages"] = decision_advantages
    return data


@pytest.fixture(autouse=True)
def _single_cp_rank(monkeypatch):
    monkeypatch.setattr(miles_advantage, "_cp_size", lambda: 1)


class TestTurnGrpo:
    def test_span_broadcast_uniform(self):
        rollout_data = _rollout_data(
            response_lengths=[8],
            decision_spans=[[[0, 4], [6, 8]]],
            decision_advantages=[[1.0, -0.5]],
        )
        kl = [torch.zeros(8)]
        advantages, returns = miles_advantage.turn_grpo(_args(), rollout_data, kl)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -0.5, -0.5])
        assert torch.equal(advantages[0], expected)
        assert torch.equal(returns[0], expected)

    def test_span_normalized_weighting(self):
        rollout_data = _rollout_data(
            response_lengths=[8],
            decision_spans=[[[0, 4], [6, 8]]],
            decision_advantages=[[1.0, -0.5]],
        )
        kl = [torch.zeros(8)]
        advantages, _ = miles_advantage.turn_grpo(
            _args(turn_advantage_length_weighting="span_normalized"), rollout_data, kl
        )
        expected = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, -0.25, -0.25])
        assert torch.equal(advantages[0], expected)

    def test_no_decisions_falls_back_to_grpo_broadcast(self):
        rollout_data = _rollout_data(response_lengths=[4], rewards=[2.0])
        kl = [torch.zeros(4)]
        advantages, returns = miles_advantage.turn_grpo(_args(), rollout_data, kl)
        expected = torch.tensor([2.0, 2.0, 2.0, 2.0])
        assert torch.equal(advantages[0], expected)
        assert torch.equal(returns[0], expected)

    def test_sample_with_empty_decisions_falls_back(self):
        rollout_data = _rollout_data(
            response_lengths=[4, 4],
            decision_spans=[[[0, 2]], []],
            decision_advantages=[[1.0], []],
            rewards=[0.0, 3.0],
        )
        kl = [torch.zeros(4), torch.zeros(4)]
        advantages, _ = miles_advantage.turn_grpo(_args(), rollout_data, kl)
        assert torch.equal(advantages[0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        assert torch.equal(advantages[1], torch.tensor([3.0, 3.0, 3.0, 3.0]))

    def test_spans_clamped_to_response_length(self):
        rollout_data = _rollout_data(
            response_lengths=[4],
            decision_spans=[[[2, 99]]],
            decision_advantages=[[1.0]],
        )
        kl = [torch.zeros(4)]
        advantages, _ = miles_advantage.turn_grpo(_args(), rollout_data, kl)
        assert torch.equal(advantages[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_cp_slicing(self, monkeypatch):
        """With CP > 1 only this rank's response-token segments are returned."""
        monkeypatch.setattr(miles_advantage, "_cp_size", lambda: 2)
        # prompt is 10 tokens; this rank holds global token ranges [10,12) and [16,18)
        monkeypatch.setattr(
            miles_advantage,
            "_cp_token_offsets",
            lambda total_len, response_len, qkv_format, max_seq_len: ((10, 12), (16, 18)),
        )
        rollout_data = _rollout_data(
            response_lengths=[8],
            total_lengths=[18],
            decision_spans=[[[0, 4], [6, 8]]],
            decision_advantages=[[1.0, -0.5]],
        )
        kl = [torch.zeros(4)]  # C_i = 4 local tokens
        advantages, _ = miles_advantage.turn_grpo(_args(), rollout_data, kl)
        # full vector [1,1,1,1,0,0,-0.5,-0.5]; local = response[0:2] + response[6:8]
        assert torch.equal(advantages[0], torch.tensor([1.0, 1.0, -0.5, -0.5]))

    def test_returns_are_independent_tensors(self):
        rollout_data = _rollout_data(
            response_lengths=[2], decision_spans=[[[0, 2]]], decision_advantages=[[1.0]]
        )
        kl = [torch.zeros(2)]
        advantages, returns = miles_advantage.turn_grpo(_args(), rollout_data, kl)
        returns[0][0] = 99.0
        assert advantages[0][0] == 1.0
