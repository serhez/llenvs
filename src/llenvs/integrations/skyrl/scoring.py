"""Immutable inputs/results for prefix-causal additive token scorers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from llenvs.integrations.skyrl._checks import finite_number, identifier, integer, token_ids

_SEMANTICS = "prefix_causal_additive"


def _freeze(value: Any) -> Any:
    """Own a recursively immutable JSON-like snapshot; never retain mutable views."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("scoring context keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return finite_number(value, "scoring context value")
    raise ValueError("scoring context must contain only JSON-like values")


@dataclass(frozen=True, kw_only=True)
class GenerationScoreInput:
    """One generation's actual input/output and prior conditioning only.

    IDs are immutable sequences; existing tuples are shared. Conditioning and
    provenance are owned JSON-like snapshots. Keep this request transient on the
    driver rather than retaining copies of all prefixes in a rollout ledger.
    A scorer can still read later tokens within output_ids: causality requires
    an independently checked scoring construction, not just this interface.
    """

    occurrence_id: str
    instance_id: str
    repetition_id: int
    generation_id: str
    input_ids: Sequence[int]
    output_ids: Sequence[int]
    conditioning: Sequence[Mapping[str, Any]]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("occurrence_id", "instance_id", "generation_id"):
            identifier(getattr(self, name), name)
        integer(self.repetition_id, "repetition_id")
        for name in ("input_ids", "output_ids"):
            values = getattr(self, name)
            token_ids(values, name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, tuple(values))
        if not isinstance(self.conditioning, (list, tuple)) or not all(
            isinstance(message, Mapping) for message in self.conditioning
        ):
            raise ValueError("conditioning must be a sequence of context mappings")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        for name in ("model", "tokenizer"):
            identifier(self.provenance.get(name), f"provenance.{name}")
        object.__setattr__(self, "conditioning", _freeze(self.conditioning))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True, kw_only=True)
class GenerationTokenRewards:
    """One finite immediate reward per sampled output token, including sampled EOS."""

    occurrence_id: str
    generation_id: str
    rewards: Sequence[float]

    def __post_init__(self) -> None:
        identifier(self.occurrence_id, "occurrence_id")
        identifier(self.generation_id, "generation_id")
        if isinstance(self.rewards, (str, bytes, bytearray)):
            raise ValueError("rewards must be a numeric sequence")
        object.__setattr__(
            self, "rewards", tuple(finite_number(value, "token reward") for value in self.rewards)
        )


class GenerationScorer(Protocol):
    """Async callable; resource-owning implementations also provide async aclose()."""

    reward_semantics: str

    async def __call__(self, generation: GenerationScoreInput) -> GenerationTokenRewards: ...


async def score_generation(
    scorer: GenerationScorer,
    generation: GenerationScoreInput,
    *,
    weight: float = 1.0,
) -> GenerationTokenRewards:
    """Validate and weight one scorer result, without retries or fallback rewards.

    This validates the declared contract, not the truth of the scorer's causal
    claim. Backend exceptions and cancellation propagate to the episode owner.
    """
    weight = finite_number(weight, "token scorer weight")
    if getattr(scorer, "reward_semantics", None) != _SEMANTICS:
        raise ValueError(f"scorer reward_semantics must be {_SEMANTICS}")
    if not isinstance(generation, GenerationScoreInput):
        raise ValueError("generation must be a GenerationScoreInput")
    result = await scorer(generation)
    if not isinstance(result, GenerationTokenRewards):
        raise ValueError("scorer must return GenerationTokenRewards")
    if (result.occurrence_id, result.generation_id) != (
        generation.occurrence_id,
        generation.generation_id,
    ):
        raise ValueError("scorer occurrence/generation identity does not match the request")
    if len(result.rewards) != len(generation.output_ids):
        raise ValueError("token rewards length must equal sampled output length")
    return GenerationTokenRewards(
        occurrence_id=result.occurrence_id,
        generation_id=result.generation_id,
        rewards=tuple(
            finite_number(value * weight, "weighted token reward") for value in result.rewards
        ),
    )
