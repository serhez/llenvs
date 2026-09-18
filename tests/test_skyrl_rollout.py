"""Model-free contracts at the recorded TITO generation boundary.

This validates provenance coordinates only; it does not certify a renderer,
packed attention, multimodal inputs, or inference/training probability parity.
"""

import copy
import importlib
import math

import pytest


@pytest.fixture
def trace():
    return importlib.import_module("llenvs.integrations.skyrl._trace")


@pytest.fixture
def generation():
    return {
        "expected_prefix": [10, 20, 21],
        "actual_input_ids": [10, 20, 21, 30, 31],
        "output_ids": [0, 99],
        "rollout_logprobs": [-0.7, -0.4],
        "vocab_size": 100,
    }


def test_actual_prefix_context_additions_and_sampled_eos_are_preserved(trace, generation):
    # Token zero is legitimate; 99 represents a sampled EOS. Neither is
    # inferred from decoded text (which may be empty for special tokens).
    original = copy.deepcopy(generation)
    trace.validate_generation(**generation)
    assert generation == original


@pytest.mark.parametrize("actual", [[10, 20, 22, 30], [10, 21, 20, 30], [10, 20], [20, 21, 30]])
def test_retokenization_removal_reordering_or_compaction_breaks_prefix(trace, generation, actual):
    generation["actual_input_ids"] = actual
    with pytest.raises(ValueError, match="prefix"):
        trace.validate_generation(**generation)


@pytest.mark.parametrize("logprobs", [[], [-0.7], [-0.7, -0.4, -0.2]])
def test_inner_logprob_lengths_must_match_exactly(trace, generation, logprobs):
    generation["rollout_logprobs"] = logprobs
    with pytest.raises(ValueError, match="length|logprob"):
        trace.validate_generation(**generation)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -9999.0])
def test_nonfinite_and_known_floor_probabilities_are_rejected(trace, generation, value):
    generation["rollout_logprobs"][0] = value
    with pytest.raises(ValueError, match="logprob|finite|sentinel"):
        trace.validate_generation(**generation)


@pytest.mark.parametrize("value", [-1, 100, 1.5, True])
def test_sampled_ids_must_be_integer_vocabulary_ids(trace, generation, value):
    generation["output_ids"][0] = value
    with pytest.raises(ValueError, match="token|vocab|integer"):
        trace.validate_generation(**generation)


def test_zero_sampled_tokens_are_a_protocol_error_not_a_synthetic_action(trace, generation):
    generation.update(output_ids=[], rollout_logprobs=[])
    with pytest.raises(ValueError, match="empty|sampled|token"):
        trace.validate_generation(**generation)
