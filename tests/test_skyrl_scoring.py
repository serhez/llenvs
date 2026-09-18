"""Async exact-token scoring contracts, without a reward model or network.

A generation-wide API cannot prove a scorer causal. Prefix-perturbation cases
below are adapter acceptance checks, not automatic runtime causality detection.
"""

import asyncio
import copy
import importlib
import math
from dataclasses import FrozenInstanceError, replace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def scoring():
    return importlib.import_module("llenvs.integrations.skyrl.scoring")


@pytest.fixture
def generation(scoring):
    return scoring.GenerationScoreInput(
        occurrence_id="occurrence-1",
        instance_id="task-1",
        repetition_id=0,
        generation_id="generation-1",
        input_ids=(10, 11),
        output_ids=(20, 99),
        conditioning=({"role": "user", "content": "Question"},),
        provenance={
            "model": "model@revision",
            "tokenizer": "tokenizer@revision",
            "processor": None,
        },
    )


def result(scoring, generation, rewards=(1.0, 2.0), **changes):
    values = {
        "occurrence_id": generation.occurrence_id,
        "generation_id": generation.generation_id,
        "rewards": rewards,
    }
    values.update(changes)
    return scoring.GenerationTokenRewards(**values)


def scorer_for(output):
    scorer = AsyncMock(return_value=output)
    scorer.reward_semantics = "prefix_causal_additive"
    return scorer


def test_identity_exact_length_and_component_weight_are_preserved(scoring, generation):
    scorer = scorer_for(result(scoring, generation))
    scored = asyncio.run(scoring.score_generation(scorer, generation, weight=0.5))
    assert scored.occurrence_id == generation.occurrence_id
    assert scored.generation_id == generation.generation_id
    assert scored.rewards == (0.5, 1.0)
    scorer.assert_awaited_once_with(generation)


def test_input_owns_immutable_context_and_provenance(scoring, generation):
    conditioning = [{"role": "user", "content": [{"type": "text", "text": "before"}]}]
    provenance = {"model": "fixed", "tokenizer": "fixed", "processor": None}
    copied = replace(generation, conditioning=conditioning, provenance=provenance)
    conditioning[0]["content"][0]["text"] = "after"
    provenance["model"] = "changed"
    assert copied.conditioning[0]["content"][0]["text"] == "before"
    assert copied.provenance["model"] == "fixed"
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        copied.conditioning[0]["content"][0]["text"] = "mutated"
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        copied.output_ids = (1,)
    with pytest.raises(TypeError):
        copied.provenance["model"] = "mutated"


def test_result_does_not_alias_mutable_scorer_output(scoring, generation):
    rewards = [1.0, 2.0]
    output = result(scoring, generation, rewards=rewards)
    rewards[0] = 999
    assert output.rewards == (1.0, 2.0)
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        output.rewards = (3.0, 4.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"occurrence_id": "other-occurrence"},
        {"generation_id": "other-generation"},
        {"rewards": ()},
        {"rewards": (1.0,)},
        {"rewards": (1.0, 2.0, 3.0)},
    ],
)
def test_mismatched_identity_or_ragged_scores_are_rejected(scoring, generation, changes):
    with pytest.raises(ValueError, match="identity|occurrence|generation|length|reward"):
        scorer = scorer_for(result(scoring, generation, **changes))
        asyncio.run(scoring.score_generation(scorer, generation))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_component_scores_are_rejected(scoring, generation, value):
    with pytest.raises(ValueError, match="finite"):
        scorer = scorer_for(result(scoring, generation, rewards=(value, 0.0)))
        asyncio.run(scoring.score_generation(scorer, generation))


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf])
def test_nonfinite_weights_fail_before_calling_the_scorer(scoring, generation, weight):
    scorer = scorer_for(result(scoring, generation))
    with pytest.raises(ValueError, match="finite|weight"):
        asyncio.run(scoring.score_generation(scorer, generation, weight=weight))
    scorer.assert_not_awaited()


def test_finite_weighted_components_cannot_overflow_silently(scoring, generation):
    scorer = scorer_for(result(scoring, generation, rewards=(1e308, 0.0)))
    with pytest.raises(ValueError, match="finite|overflow"):
        asyncio.run(scoring.score_generation(scorer, generation, weight=2.0))


@pytest.mark.parametrize("semantics", [None, "quality_probability", "value", "advantage"])
def test_undeclared_or_nonadditive_scores_are_not_implicitly_converted(
    scoring, generation, semantics
):
    scorer = scorer_for(result(scoring, generation))
    scorer.reward_semantics = semantics
    with pytest.raises(ValueError, match="semantics|causal|additive"):
        asyncio.run(scoring.score_generation(scorer, generation))
    scorer.assert_not_awaited()


def test_backend_failure_is_not_retried_or_replaced_with_zero(scoring, generation):
    scorer = scorer_for(None)
    scorer.side_effect = RuntimeError("scorer failed")
    with pytest.raises(RuntimeError, match="scorer failed"):
        asyncio.run(scoring.score_generation(scorer, generation))
    assert scorer.await_count == 1


def test_cancellation_is_not_converted_to_an_incomplete_score(scoring, generation):
    scorer = scorer_for(None)
    scorer.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scoring.score_generation(scorer, generation))


@pytest.mark.parametrize(
    "future_field", ["next_transition", "later_generations", "episode_outcome"]
)
def test_scoring_input_does_not_accept_future_transition_fields(scoring, generation, future_field):
    with pytest.raises(TypeError, match=future_field):
        replace(generation, **{future_field: "future information"})


def test_prefix_perturbation_accepts_causal_adapter_and_exposes_false_declaration(
    scoring, generation
):
    async def check_adapter(scorer):
        first = await scoring.score_generation(scorer, generation)
        changed_suffix = replace(generation, output_ids=(20, 98))
        second = await scoring.score_generation(scorer, changed_suffix)
        assert first.rewards[0] == second.rewards[0], "earlier reward depends on later token"

    async def causal(item):
        return result(scoring, item, rewards=tuple(float(token) for token in item.output_ids))

    async def noncausal(item):
        return result(scoring, item, rewards=(float(item.output_ids[-1]), 0.0))

    causal.reward_semantics = noncausal.reward_semantics = "prefix_causal_additive"
    before = copy.deepcopy(generation.output_ids)
    asyncio.run(check_adapter(causal))
    # The acceptance assertion, not the runtime wrapper, detects this lie.
    with pytest.raises(AssertionError, match="earlier reward"):
        asyncio.run(check_adapter(noncausal))
    assert generation.output_ids == before
