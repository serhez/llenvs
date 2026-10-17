"""Sampling controls must preserve caller intent before any backend dispatch."""

from dataclasses import replace

import pytest

from llenvs.inference.protocol import (
    BackendCapabilities,
    GenerationResult,
    ModelBackend,
    SamplingParams,
)


def test_default_filters_and_penalties_are_disabled():
    params = SamplingParams()

    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.min_p == 0.0
    assert params.repetition_penalty == 1.0
    assert params.presence_penalty == 0.0
    assert params.frequency_penalty == 0.0
    assert params.n == 1


@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ],
)
def test_none_preserves_an_explicit_request_to_inherit_a_default(field):
    params = SamplingParams(**{field: None})

    assert getattr(params, field) is None


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_sampling_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("temperature", -0.1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("top_k", -1),
        ("min_p", -0.1),
        ("min_p", 1.1),
        ("repetition_penalty", 0.0),
        ("repetition_penalty", -1.0),
        ("n", 0),
    ],
)
def test_values_outside_the_common_sampling_domain_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: value})


@pytest.mark.parametrize("field", ["max_tokens", "top_k", "n"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_token_and_choice_counts_require_integers(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        SamplingParams(**{field: value})


class _RecordingBackend(ModelBackend):
    @property
    def capabilities(self):
        return BackendCapabilities(supports_logprobs=True)

    @property
    def model_name(self):
        return "recording"

    def generate(self, prompts, params):
        self.received_params = params
        return [GenerationResult(text=prompt) for prompt in prompts]


def test_logprob_helper_changes_only_the_requested_logprob_settings():
    backend = _RecordingBackend()
    params = SamplingParams(
        max_tokens=71,
        temperature=0.6,
        top_p=0.83,
        top_k=13,
        min_p=0.07,
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=-0.2,
        stop_sequences=("DONE",),
        thinking_budget=23,
        second_elicitation_suffix="Answer now.",
        second_elicitation_max_tokens=9,
        extra={"backend_option": {"value": 3}},
    )

    results = backend.generate_with_logprobs(["prompt"], params, num_logprobs=3)

    assert results[0].text == "prompt"
    assert backend.received_params == replace(params, logprobs=True, num_logprobs=3)
    assert params.logprobs is False
    assert params.num_logprobs == 5


def test_custom_backend_must_declare_controls_and_its_own_identity():
    from llenvs.inference.protocol import BackendCapabilities, ModelBackend

    class Custom(ModelBackend):
        @property
        def capabilities(self):
            return BackendCapabilities()

        @property
        def model_name(self):
            return "custom"

        def generate(self, prompts, params):
            return []

    backend = Custom()
    with pytest.raises(NotImplementedError):
        backend.descriptor()
    with pytest.raises(ValueError):
        backend.validate_sampling(SamplingParams())
