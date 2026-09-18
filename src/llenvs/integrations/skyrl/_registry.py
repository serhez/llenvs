"""Importable custom estimators registered through SkyRL's own registry."""

from typing import Any

from llenvs.integrations.skyrl._validation import TOKEN_ESTIMATOR, TURN_ESTIMATOR


def turn_credit(
    token_level_rewards: Any,
    *,
    attribution: Any,
    n_samples_per_prompt: int,
    gamma: float = 1.0,
    grpo_norm_by_std: bool = True,
    turn_weighting: str = "uniform",
    **_: Any,
) -> tuple[Any, Any]:
    from llenvs.integrations.skyrl._credit import compute_credit

    return compute_credit(
        token_level_rewards,
        attribution=attribution,
        estimator=TURN_ESTIMATOR,
        n_samples_per_prompt=n_samples_per_prompt,
        gamma=gamma,
        grpo_norm_by_std=grpo_norm_by_std,
        turn_weighting=turn_weighting,
    )


def token_credit(
    token_level_rewards: Any,
    *,
    attribution: Any,
    n_samples_per_prompt: int,
    gamma: float = 1.0,
    grpo_norm_by_std: bool = False,
    turn_weighting: str = "uniform",
    **_: Any,
) -> tuple[Any, Any]:
    from llenvs.integrations.skyrl._credit import compute_credit

    return compute_credit(
        token_level_rewards,
        attribution=attribution,
        estimator=TOKEN_ESTIMATOR,
        n_samples_per_prompt=n_samples_per_prompt,
        gamma=gamma,
        grpo_norm_by_std=grpo_norm_by_std,
        turn_weighting=turn_weighting,
    )


def register_estimators(registry: Any = None) -> None:
    """Register before validation; native Ray synchronization owns distribution."""
    if registry is None:
        from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

        registry = AdvantageEstimatorRegistry
    available = set(registry.list_available())
    estimators = {TURN_ESTIMATOR: turn_credit, TOKEN_ESTIMATOR: token_credit}
    # Resolve all conflicts before mutating the registry at all.
    for name, function in estimators.items():
        if name in available and registry.get(name) is not function:
            raise ValueError(f"custom estimator {name} is already registered by another function")
    for name, function in estimators.items():
        if name not in available:
            registry.register(name, function)
