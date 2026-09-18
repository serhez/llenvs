"""Pure validation seams; real native typed-config loading is a separate suite."""

import copy
import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def validation():
    return importlib.import_module("llenvs.integrations.skyrl._validation")


@pytest.fixture
def recipe():
    # These are the selected native fields, not a substitute native config loader.
    return {
        "algorithm": SimpleNamespace(
            advantage_estimator="llenvs_turn_grpo",
            gamma=1.0,
            grpo_norm_by_std=True,
            use_kl_in_reward=False,
            advantage_batch_normalize=False,
            zero_variance_filter=False,
            dynamic_sampling=SimpleNamespace(type=None),
            loss_reduction="token_mean",
            policy_loss_type="regular",
            off_policy_correction=SimpleNamespace(
                tis_ratio_type=None,
                token_tis_ratio_clip_high=2.0,
                sequence_tis_ratio_clip_high=5.0,
                sequence_mask_metric=None,
                outlier_token_is_threshold_low=None,
                outlier_token_is_threshold_high=None,
                token_mask_is_threshold_low=None,
                token_mask_is_threshold_high=None,
            ),
        ),
        "generator": SimpleNamespace(
            zero_reward_on_non_stop=False,
            apply_overlong_filtering=False,
            step_wise_trajectories=False,
            merge_stepwise_output=False,
        ),
        "sampling_contract": "unmodified",
        "turn_weighting": "uniform",
        "token_scorer": None,
    }


def test_custom_recipe_is_valid_without_rewriting_native_settings(validation, recipe):
    before = copy.deepcopy(recipe)
    validation.validate_credit_recipe(**recipe)
    assert recipe == before


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("algorithm", "advantage_estimator", "gae"),
        ("algorithm", "use_kl_in_reward", True),
        ("algorithm", "advantage_batch_normalize", True),
        ("algorithm", "zero_variance_filter", True),
        ("algorithm", "loss_reduction", "token_mean_legacy"),
        ("algorithm", "policy_loss_type", "gspo"),
        ("generator", "zero_reward_on_non_stop", True),
        ("generator", "step_wise_trajectories", True),
        ("generator", "merge_stepwise_output", True),
    ],
)
def test_incompatible_credit_features_fail_explicitly(validation, recipe, section, field, value):
    setattr(recipe[section], field, value)
    with pytest.raises(ValueError, match=field):
        validation.validate_credit_recipe(**recipe)


def test_dynamic_outcome_selection_is_not_a_process_credit_filter(validation, recipe):
    recipe["algorithm"].dynamic_sampling.type = "filter"
    with pytest.raises(ValueError, match="dynamic_sampling"):
        validation.validate_credit_recipe(**recipe)


def test_overlong_loss_masking_is_distinct_from_reward_zeroing(validation, recipe):
    recipe["generator"].apply_overlong_filtering = True
    validation.validate_credit_recipe(**recipe)


def test_native_scalar_mode_keeps_native_outcome_modifiers(validation, recipe):
    recipe["algorithm"].advantage_estimator = "grpo"
    recipe["algorithm"].zero_variance_filter = True
    recipe["generator"].zero_reward_on_non_stop = True
    recipe["sampling_contract"] = "native"
    before = copy.deepcopy(recipe)
    validation.validate_credit_recipe(**recipe)
    assert recipe == before


def test_native_mode_does_not_silently_collapse_a_token_scorer(validation, recipe):
    recipe["algorithm"].advantage_estimator = "grpo"
    recipe["token_scorer"] = {"factory": "fixture:scorer", "revision": "fixed"}
    with pytest.raises(ValueError, match="token_scorer"):
        validation.validate_credit_recipe(**recipe)


def test_custom_training_requires_explicit_sampling_contract(validation, recipe):
    recipe["sampling_contract"] = "native"
    with pytest.raises(ValueError, match="sampling_contract"):
        validation.validate_credit_recipe(**recipe)


def test_sync_token_tis_is_an_explicit_admitted_correction(validation, recipe):
    recipe["algorithm"].off_policy_correction.tis_ratio_type = "token"
    validation.validate_credit_recipe(**recipe)


def test_rollout_anchored_loss_cannot_apply_tis_a_second_time(validation, recipe):
    recipe["algorithm"].policy_loss_type = "rollout_is"
    validation.validate_credit_recipe(**recipe)
    recipe["algorithm"].off_policy_correction.tis_ratio_type = "token"
    with pytest.raises(ValueError, match="TIS|tis|off_policy_correction"):
        validation.validate_credit_recipe(**recipe)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tis_ratio_type", "sequence"),
        ("sequence_mask_metric", "product"),
        ("outlier_token_is_threshold_low", 0.5),
        ("token_mask_is_threshold_low", 0.5),
    ],
)
def test_deferred_or_incomplete_correction_options_do_not_silently_pass(
    validation, recipe, field, value
):
    setattr(recipe["algorithm"].off_policy_correction, field, value)
    with pytest.raises(ValueError, match=field):
        validation.validate_credit_recipe(**recipe)


@pytest.fixture
def request_params():
    return {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "min_tokens": 0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 32,
        "logprobs": 1,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
    }


def test_effective_unmodified_request_is_valid_and_not_mutated(validation, request_params):
    before = copy.deepcopy(request_params)
    validation.validate_sampling_request(
        request_params, contract="unmodified", phase="train", logprobs_mode="raw_logprobs"
    )
    assert request_params == before


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", 0.0),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 20),
        ("min_p", 0.1),
        ("min_tokens", 1),
        ("repetition_penalty", 1.1),
        ("frequency_penalty", 0.2),
        ("presence_penalty", 0.2),
        ("allowed_token_ids", [1, 2]),
        ("logit_bias", {1: 2.0}),
        ("logits_processors", ["unknown-processor"]),
    ],
)
def test_probability_transformations_are_checked_on_effective_requests(
    validation, request_params, key, value
):
    request_params[key] = value
    with pytest.raises(ValueError, match=key):
        validation.validate_sampling_request(
            request_params, contract="unmodified", phase="train", logprobs_mode="raw_logprobs"
        )


def test_greedy_evaluation_does_not_inherit_stochastic_training_gate(validation, request_params):
    request_params.update(temperature=0.0, min_tokens=1)
    before = copy.deepcopy(request_params)
    validation.validate_sampling_request(
        request_params, contract="unmodified", phase="eval", logprobs_mode="raw_logprobs"
    )
    assert request_params == before


def test_native_uncorrected_sampling_remains_native(validation, request_params):
    request_params.update(temperature=0.7, top_p=0.9, min_tokens=1)
    before = copy.deepcopy(request_params)
    validation.validate_sampling_request(
        request_params, contract="native", phase="train", logprobs_mode="raw_logprobs"
    )
    assert request_params == before


def test_processed_probabilities_need_a_separately_reviewed_recipe(validation, request_params):
    with pytest.raises(ValueError, match="logprobs_mode"):
        validation.validate_sampling_request(
            request_params, contract="unmodified", phase="train", logprobs_mode="processed_logprobs"
        )


@pytest.mark.parametrize("phase", ["training", "unknown"])
def test_unknown_phase_does_not_accidentally_bypass_training_validation(
    validation, request_params, phase
):
    with pytest.raises(ValueError, match="phase"):
        validation.validate_sampling_request(
            request_params, contract="unmodified", phase=phase, logprobs_mode="raw_logprobs"
        )
