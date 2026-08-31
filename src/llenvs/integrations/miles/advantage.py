"""Tier-2 turn-level advantage hook (``--custom-advantage-function-path``).

Requires the turn-level-credit miles fork (see
``docs/design/miles-integration.md``). Launch flags::

    --turn-level-credit --advantage-estimator grpo
    --custom-advantage-function-path llenvs.integrations.miles.advantage.turn_grpo
    --session-sample-postprocessor-path llenvs.integrations.miles.postprocess.postprocess

Division of labor (one forward pass per merged trajectory, always):

- The ``postprocess`` hook records response-relative decision spans/rewards
  in sample metadata.
- The fork's rollout-side ``turn_credit`` module computes per-decision
  advantages (return-to-go with a per-DECISION discount, group-relative
  normalization) while the whole group is still visible — after DP sharding
  a rank only sees group slices — and the fork's train-data conversion ships
  ``decision_spans``/``decision_advantages`` to the trainer.
- ``turn_grpo`` (this module) only broadcasts each decision's advantage over
  its token span and slices to the local CP rank. ``returns`` equals
  ``advantages`` (GRPO parity — KL enters the loss, not the returns).

Samples without decision data fall back to broadcasting the scalar reward —
degenerating to exactly GRPO.
"""

from __future__ import annotations

from typing import Any


def _cp_size() -> int:
    """Context-parallel world size. Module-level seam for tests."""
    from miles.backends.training_utils.parallel import get_parallel_state

    return get_parallel_state().cp.size


def _cp_token_offsets(
    total_len: int, response_len: int, qkv_format: str, max_seq_len: int | None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """This CP rank's two global token ranges. Module-level seam for tests."""
    from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp

    _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
        total_len, response_len, qkv_format, max_seq_len
    )
    return token_offsets


def _cp_slice(
    args: Any, full: Any, total_len: int, response_len: int, max_seq_len: int | None
) -> Any:
    """Slice a full response-length vector to this CP rank's tokens."""
    import torch

    if _cp_size() == 1:
        return full
    prompt_len = total_len - response_len
    (s0, e0), (s1, e1) = _cp_token_offsets(
        total_len, response_len, getattr(args, "qkv_format", "thd"), max_seq_len
    )
    parts = []
    for start, end in ((s0, e0), (s1, e1)):
        res_start = max(0, start - prompt_len)
        res_end = max(0, end - prompt_len)
        if res_end > res_start:
            parts.append(full[res_start:res_end])
    if not parts:
        return torch.tensor([], dtype=full.dtype, device=full.device)
    return torch.cat(parts)


def turn_grpo(
    args: Any, rollout_data: dict[str, Any], kl: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Span-broadcast per-decision advantages; returns == advantages.

    Args:
        args: miles argparse namespace (uses ``turn_advantage_length_weighting``
            and ``qkv_format``).
        rollout_data: Trainer-side batch with ``rewards``, ``response_lengths``,
            ``total_lengths``, optional ``max_seq_lens``, and the fork-shipped
            ``decision_spans`` / ``decision_advantages``.
        kl: Per-sample CP-local response-aligned tensors (shape ``[C_i]``);
            used only for device/dtype anchoring.

    Returns:
        ``(advantages, returns)`` — lists of per-sample ``[C_i]`` tensors.
    """
    import torch

    response_lengths: list[int] = rollout_data["response_lengths"]
    total_lengths: list[int] = rollout_data["total_lengths"]
    rewards: list[float] = rollout_data["rewards"]
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens")
    spans_per_sample = rollout_data.get("decision_spans")
    advs_per_sample = rollout_data.get("decision_advantages")
    weighting = getattr(args, "turn_advantage_length_weighting", "uniform")

    advantages: list[Any] = []
    for i, k in enumerate(kl):
        response_len = response_lengths[i]
        spans = spans_per_sample[i] if spans_per_sample else None
        decision_advs = advs_per_sample[i] if advs_per_sample else None
        full = torch.zeros(response_len, dtype=torch.float32, device=k.device)
        if spans and decision_advs:
            for (start, end), adv in zip(spans, decision_advs):
                start = max(0, int(start))
                end = min(response_len, int(end))
                if end <= start:
                    continue
                value = adv / (end - start) if weighting == "span_normalized" else adv
                full[start:end] = value
        else:
            # No decision data — degenerate to GRPO's scalar broadcast.
            full[:] = float(rewards[i])
        advantages.append(
            _cp_slice(
                args,
                full,
                total_lengths[i],
                response_len,
                max_seq_lens[i] if max_seq_lens else None,
            )
        )

    returns = [tensor.clone() for tensor in advantages]
    return advantages, returns
