"""Validation of recorded, append-only token-in/token-out generations."""

from collections.abc import Sequence

from llenvs.integrations.skyrl._checks import finite_number, integer, token_ids

# SkyRL's generate_wire helper substitutes this for unavailable probabilities.
_MISSING_LOGPROB = -9999.0


def validate_generation(
    *,
    expected_prefix: Sequence[int],
    actual_input_ids: Sequence[int],
    output_ids: Sequence[int],
    rollout_logprobs: Sequence[float],
    vocab_size: int,
) -> None:
    """Check structural provenance, without reconstructing or repairing tokens.

    The caller must capture real endpoint IDs/probabilities; numerical validity
    alone cannot distinguish fabricated IDs or prove training/inference parity.
    This text-token check does not validate multimodal features or positions.
    """
    integer(vocab_size, "vocab_size", minimum=1)
    for name, values in (
        ("expected_prefix", expected_prefix),
        ("actual_input_ids", actual_input_ids),
        ("output_ids", output_ids),
    ):
        token_ids(values, name)
        if any(value >= vocab_size for value in values):
            raise ValueError(f"{name} contains a token outside the vocabulary")
    if not actual_input_ids:
        raise ValueError("actual_input_ids must contain conditioning tokens")
    if len(actual_input_ids) < len(expected_prefix) or any(
        actual_input_ids[i] != value for i, value in enumerate(expected_prefix)
    ):
        raise ValueError("actual input does not preserve the recorded token prefix")
    if not output_ids:
        raise ValueError("empty generation has no sampled token")
    if len(rollout_logprobs) != len(output_ids):
        raise ValueError("rollout_logprobs length must equal sampled output length")
    for value in rollout_logprobs:
        logprob = finite_number(value, "rollout_logprob")
        if logprob == _MISSING_LOGPROB:
            raise ValueError("rollout_logprob is the missing-probability sentinel")
        if logprob > 0:
            raise ValueError("rollout_logprob must be <= 0")
