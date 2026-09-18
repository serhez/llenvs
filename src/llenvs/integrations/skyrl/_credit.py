"""Custom credit on complete real groups, before native sharding and packing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TypedDict

import torch

from llenvs.integrations.skyrl._checks import identifier, integer
from llenvs.integrations.skyrl._validation import TOKEN_ESTIMATOR, validate_credit_options


class SampledSpan(TypedDict):
    generation_id: str
    start: int
    end: int
    sampled_count: int


class AttributionRow(TypedDict):
    instance_id: str
    repetition_id: int
    response_length: int
    sampled_spans: list[SampledSpan]


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite (overflow or non-finite input)")


def _validate_ledger(
    attribution: Sequence[AttributionRow], samples: int, *, width: int, row_count: int
) -> dict[str, list[int]]:
    if (
        not isinstance(attribution, (list, tuple))
        or not attribution
        or len(attribution) > row_count
    ):
        raise ValueError(
            "attribution must contain one record per real row before appended dummy rows"
        )
    groups: dict[str, list[int]] = defaultdict(list)
    repetitions: dict[str, set[int]] = defaultdict(set)
    generation_ids: set[str] = set()
    for i, row in enumerate(attribution):
        if not isinstance(row, Mapping) or set(row) != set(AttributionRow.__annotations__):
            raise ValueError("invalid attribution row fields")
        group = identifier(row["instance_id"], "instance_id")
        repetition = integer(row["repetition_id"], "repetition_id")
        if repetition >= samples or repetition in repetitions[group]:
            raise ValueError("duplicate or out-of-range repetition_id in prompt group")
        repetitions[group].add(repetition)
        groups[group].append(i)
        length = integer(row["response_length"], "response_length", minimum=1)
        if length > width:
            raise ValueError("response_length exceeds reward tensor width")
        spans = row["sampled_spans"]
        if not isinstance(spans, (list, tuple)) or not spans:
            raise ValueError("a real training trajectory must contain sampled spans")
        previous_end = 0
        for span in spans:
            if not isinstance(span, Mapping) or set(span) != set(SampledSpan.__annotations__):
                raise ValueError("invalid sampled span fields")
            generation = identifier(span["generation_id"], "generation_id")
            if generation in generation_ids:
                raise ValueError("duplicate generation_id in attribution")
            generation_ids.add(generation)
            start = integer(span["start"], "span.start")
            end = integer(span["end"], "span.end", minimum=1)
            count = integer(span["sampled_count"], "span.sampled_count", minimum=1)
            if not previous_end <= start < end <= length or count != end - start:
                raise ValueError(
                    "sampled spans must be ordered, disjoint, in bounds, and have exact counts"
                )
            previous_end = end
    if any(len(seen) != samples for seen in repetitions.values()):
        raise ValueError("incomplete prompt group in attribution")
    return groups


@torch.no_grad()
def compute_credit(
    rewards: torch.Tensor,
    *,
    attribution: Sequence[AttributionRow],
    estimator: str,
    n_samples_per_prompt: int,
    gamma: float = 1.0,
    grpo_norm_by_std: bool = True,
    turn_weighting: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and independent return tensors from effective rewards.

    Ledger spans use unpadded response coordinates. The driver must establish
    real-row identity/coverage before padding; rows after the ledger are known
    native dummies, not inferred missing trajectories. Policy loss masks never
    enter the return clock, span lengths, or normalization population.
    """
    validate_credit_options(estimator, gamma, grpo_norm_by_std, turn_weighting)
    integer(n_samples_per_prompt, "n_samples_per_prompt", minimum=1)
    if (
        not isinstance(rewards, torch.Tensor)
        or rewards.ndim != 2
        or not rewards.is_floating_point()
    ):
        raise ValueError("rewards must be a two-dimensional floating-point tensor")
    if rewards.dtype not in (torch.float32, torch.float64):
        raise ValueError("credit arithmetic requires float32 or float64 rewards")
    _finite(rewards, "rewards")
    groups = _validate_ledger(
        attribution, n_samples_per_prompt, width=rewards.shape[1], row_count=len(rewards)
    )
    allowed_rewards = torch.zeros_like(rewards, dtype=torch.bool)
    advantages = torch.zeros_like(rewards)
    row_returns: list[torch.Tensor] = []
    width = rewards.shape[1]
    for i, row in enumerate(attribution):
        offset = width - row["response_length"]
        spans = row["sampled_spans"]
        if estimator == TOKEN_ESTIMATOR:
            positions = torch.cat(
                [
                    torch.arange(
                        offset + span["start"], offset + span["end"], device=rewards.device
                    )
                    for span in spans
                ]
            )
        else:
            positions = torch.tensor(
                [offset + span["end"] - 1 for span in spans], device=rewards.device
            )
        allowed_rewards[i, positions] = True
        immediate = rewards[i, positions]
        if gamma == 1:
            returns = immediate.flip(0).cumsum(0).flip(0)
        else:
            returns = torch.empty_like(immediate)
            running = rewards.new_zeros(())
            for t in range(len(immediate) - 1, -1, -1):
                running = immediate[t] + gamma * running
                returns[t] = running
        _finite(returns, "returns")
        row_returns.append(returns)
        if estimator == TOKEN_ESTIMATOR:
            advantages[i, positions] = returns
    if bool((rewards[~allowed_rewards] != 0).any()):
        raise ValueError(
            "nonzero reward outside sampled targets/endpoints, including context, padding or dummy rows"
        )
    if estimator != TOKEN_ESTIMATOR:
        for members in groups.values():
            pooled = torch.cat([row_returns[i] for i in members])
            # Constant/singleton pools must not amplify centering roundoff.
            if bool((pooled == pooled[0]).all()):
                continue
            mean = pooled.mean()
            _finite(mean, "return mean")
            centered = pooled - mean
            _finite(centered, "centered returns")
            if grpo_norm_by_std:
                std = (centered.square().sum() / (len(pooled) - 1)).sqrt()
                _finite(std, "return standard deviation")
                centered = centered / (std + 1e-6)
                _finite(centered, "normalized credit")
            cursor = 0
            for i in members:
                row = attribution[i]
                offset = width - row["response_length"]
                for span in row["sampled_spans"]:
                    value = centered[cursor]
                    if turn_weighting == "span_normalized":
                        value = value / span["sampled_count"]
                    advantages[i, offset + span["start"] : offset + span["end"]] = value
                    cursor += 1
    _finite(advantages, "advantages")
    return advantages, advantages.clone()
