"""Ordered discrete decisions, independent of text generation."""

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from llenvs.inference.records import sum_known


@dataclass(frozen=True)
class DecisionOption:
    value: str
    description: str | None = None


@dataclass(frozen=True)
class DecisionRequest:
    context: str | list | dict
    question: str
    options: tuple[DecisionOption, ...]

    def __post_init__(self):
        labels = [option.value for option in self.options]
        if not 2 <= len(labels) <= 255 or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise ValueError("A decision requires 2..255 nonempty option labels")
        if len(set(labels)) != len(labels):
            raise ValueError("Decision option labels must be unique")
        if not self.question.strip():
            raise ValueError("Decision question cannot be empty")
        object.__setattr__(self, "context", deepcopy(self.context))
        object.__setattr__(self, "options", tuple(self.options))


@dataclass(frozen=True)
class DecisionCapabilities:
    max_options: int = 255
    all_probabilities: bool = True
    version_pinned: bool = False
    deterministic: bool = False
    batch_invariant: bool = False
    supports_request_progress: bool = True


@dataclass(frozen=True)
class DecisionResult:
    probabilities: tuple[float, ...]
    labels: tuple[str, ...]
    model: str
    raw_response: dict[str, Any]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    cache_write_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self):
        return sum_known([self.prompt_tokens, self.completion_tokens])


def normalize_probabilities(request: DecisionRequest, entries) -> tuple[tuple[float, ...], dict]:
    observed = {}
    labels = [option.value for option in request.options]
    for label, probability in entries:
        if label in observed or label not in labels:
            raise ValueError(f"Unexpected or duplicated decision option {label!r}")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0
        ):
            raise ValueError(f"Invalid probability for {label!r}")
        observed[label] = probability
    if set(observed) != set(labels):
        raise ValueError("Decision response does not contain every requested option")
    raw = [observed[label] for label in labels]
    total = math.fsum(raw)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Decision distribution must have finite positive mass")
    return tuple(value / total for value in raw), {
        "raw_probabilities": raw,
        "probability_sum": total,
        "probability_sum_deviation": total - 1,
    }


class DecisionBackend(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> DecisionCapabilities: ...

    @abstractmethod
    def descriptor(self) -> dict[str, Any]: ...

    @abstractmethod
    def decide_batch(
        self, requests: list[DecisionRequest], *, observer=None
    ) -> list[DecisionResult]: ...

    def decide(self, request: DecisionRequest) -> DecisionResult:
        return self.decide_batch([request])[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass
