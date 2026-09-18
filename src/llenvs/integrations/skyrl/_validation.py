"""Checks for credit recipes and effective requests, without trainer imports.

These are component checks, not a substitute for native typed config validation
or the integration's full runtime/backend preflight.
"""

from collections.abc import Mapping
from typing import Any

from llenvs.integrations.skyrl._checks import finite_number, integer

TURN_ESTIMATOR = "llenvs_turn_grpo"
TOKEN_ESTIMATOR = "llenvs_token_rtg"


def validate_credit_options(
    estimator: str, gamma: float, grpo_norm_by_std: bool, turn_weighting: str
) -> None:
    if estimator not in (TURN_ESTIMATOR, TOKEN_ESTIMATOR):
        raise ValueError("custom advantage_estimator must be llenvs_turn_grpo or llenvs_token_rtg")
    gamma = finite_number(gamma, "gamma")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    if not isinstance(grpo_norm_by_std, bool):
        raise ValueError("grpo_norm_by_std must be a boolean")
    if turn_weighting not in ("uniform", "span_normalized"):
        raise ValueError("turn_weighting must be uniform or span_normalized")
    if estimator == TOKEN_ESTIMATOR:
        if gamma != 1:
            raise ValueError("llenvs_token_rtg requires gamma=1")
        if grpo_norm_by_std:
            raise ValueError("llenvs_token_rtg requires grpo_norm_by_std=False")
        if turn_weighting != "uniform":
            raise ValueError("turn_weighting=span_normalized requires llenvs_turn_grpo")


def _boolean(config: Any, name: str) -> bool:
    value = getattr(config, name, None)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def validate_credit_recipe(
    algorithm: Any,
    generator: Any,
    *,
    sampling_contract: str,
    turn_weighting: str,
    token_scorer: Any,
) -> None:
    """Check selected fields of resolved native config objects, without mutation."""
    if sampling_contract not in ("native", "unmodified"):
        raise ValueError("sampling_contract must be native or unmodified")
    estimator = algorithm.advantage_estimator
    if estimator not in ("grpo", TURN_ESTIMATOR, TOKEN_ESTIMATOR):
        raise ValueError("advantage_estimator is outside the supported credit recipes")
    if estimator == "grpo":
        if token_scorer is not None:
            raise ValueError("token_scorer would lose token attribution under native grpo")
        if turn_weighting != "uniform":
            raise ValueError("turn_weighting requires llenvs_turn_grpo")
    else:
        validate_credit_options(
            estimator, algorithm.gamma, algorithm.grpo_norm_by_std, turn_weighting
        )
        if sampling_contract != "unmodified":
            raise ValueError("custom credit requires sampling_contract=unmodified")
        if token_scorer is not None and estimator != TOKEN_ESTIMATOR:
            raise ValueError("token_scorer requires llenvs_token_rtg")
        for owner, names in (
            (algorithm, ("use_kl_in_reward", "advantage_batch_normalize", "zero_variance_filter")),
            (
                generator,
                ("zero_reward_on_non_stop", "step_wise_trajectories", "merge_stepwise_output"),
            ),
        ):
            for name in names:
                if _boolean(owner, name):
                    raise ValueError(f"{name} is incompatible with custom credit")
        if algorithm.dynamic_sampling.type is not None:
            raise ValueError("dynamic_sampling cannot select groups using scalar outcome variance")
        _boolean(generator, "apply_overlong_filtering")  # Loss masking is not reward zeroing.
        if algorithm.loss_reduction not in (
            "token_mean",
            "sequence_mean",
            "prompt_mean",
            "seq_mean_token_sum_norm",
        ):
            raise ValueError("loss_reduction is not supported for custom credit")
        if algorithm.loss_reduction == "seq_mean_token_sum_norm":
            integer(getattr(algorithm, "max_seq_len", None), "max_seq_len", minimum=1)

    if algorithm.policy_loss_type not in ("regular", "rollout_is"):
        raise ValueError("policy_loss_type is outside the supported loss recipes")
    correction = algorithm.off_policy_correction
    if correction.tis_ratio_type not in (None, "token"):
        raise ValueError("off_policy_correction.tis_ratio_type only supports token TIS")
    for name in (
        "sequence_mask_metric",
        "outlier_token_is_threshold_low",
        "outlier_token_is_threshold_high",
        "token_mask_is_threshold_low",
        "token_mask_is_threshold_high",
    ):
        if getattr(correction, name) is not None:
            raise ValueError(f"off_policy_correction.{name} is not supported")
    if getattr(algorithm, "use_tis", False) and correction.tis_ratio_type is None:
        raise ValueError("use_tis must be normalized into off_policy_correction before validation")
    if correction.tis_ratio_type is not None:
        if algorithm.policy_loss_type == "rollout_is":
            raise ValueError("off_policy_correction TIS would correct rollout_is a second time")
        cap = finite_number(correction.token_tis_ratio_clip_high, "token_tis_ratio_clip_high")
        if cap != 2:
            raise ValueError("token_tis_ratio_clip_high must be 2 for the admitted recipe")
    if (
        correction.tis_ratio_type is not None or algorithm.policy_loss_type == "rollout_is"
    ) and sampling_contract != "unmodified":
        raise ValueError("behavior-corrected training requires sampling_contract=unmodified")


def validate_sampling_request(
    params: Mapping[str, Any], *, contract: str, phase: str, logprobs_mode: str
) -> None:
    """Check an effective outgoing request without applying sampler transforms.

    Native scalar sampling and evaluation keep native behavior. Engine-wide
    processors, model overrides and modality gates require separate validation.
    """
    if contract not in ("native", "unmodified"):
        raise ValueError("sampling contract must be native or unmodified")
    if phase not in ("train", "eval"):
        raise ValueError("phase must be train or eval")
    if phase == "eval" or contract == "native":
        return
    if logprobs_mode != "raw_logprobs":
        raise ValueError("logprobs_mode must be raw_logprobs for unmodified training")
    # Values omitted from the request must have the same neutral server default.
    # min_tokens is deliberately required: SkyRL's helper otherwise inserts 1.
    neutral = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    for name, expected in neutral.items():
        if finite_number(params.get(name, expected), name) != expected:
            raise ValueError(f"{name} must be {expected} for unmodified training")
    if integer(params.get("min_tokens"), "min_tokens") != 0:
        raise ValueError("min_tokens must be 0 for unmodified training")
    integer(params.get("max_tokens"), "max_tokens", minimum=1)
    integer(params.get("logprobs"), "logprobs")
    allowed = set(neutral) | {
        "min_tokens",
        "max_tokens",
        "logprobs",
        "stop",
        "stop_token_ids",
        "seed",
        "skip_special_tokens",
        "include_stop_str_in_output",
        "ignore_eos",
        "detokenize",
        "spaces_between_special_tokens",
        "n",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unsupported sampling parameters: {', '.join(sorted(unknown))}")
    if params.get("ignore_eos", False):
        raise ValueError("ignore_eos requires a separately reviewed horizon recipe")
    if params.get("n", 1) != 1:
        raise ValueError("n must be 1; prompt repetition belongs to SkyRL grouping")
    if params.get("include_stop_str_in_output", True) is not True:
        raise ValueError("include_stop_str_in_output must retain sampled stop tokens")
