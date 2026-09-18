"""Installed native config/registry bindings; no Ray cluster or model downloads.

Run in the separately staged SkyRL environment. Missing SkyRL is an explicit
module skip, not a successful native-runtime check. GPU tests are separate.
"""

import inspect

import pytest

pytest.importorskip("skyrl", reason="requires the staged native SkyRL environment")


def test_real_native_root_loader_builds_connector_and_optional_scorer():
    from llenvs.integrations.skyrl._config import LlenvsConfig, TokenScorerConfig
    from llenvs.integrations.skyrl._native import LlenvsSkyRLTrainConfig

    cfg = LlenvsSkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.logger=console",
            "llenvs.check_only=true",
            "llenvs.token_scorer.factory=builtins:list",
            "llenvs.token_scorer.revision=fixture",
            "llenvs.token_scorer.kwargs={}",
        ]
    )
    assert isinstance(cfg.llenvs, LlenvsConfig)
    assert isinstance(cfg.llenvs.token_scorer, TokenScorerConfig)
    assert cfg.environment.env_class == "llenvs"


def test_real_registry_validation_preserves_the_selected_custom_estimator():
    import ray
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry
    from skyrl.train.utils import validate_cfg

    from llenvs.integrations.skyrl._native import LlenvsSkyRLTrainConfig
    from llenvs.integrations.skyrl._preflight import validate_profile
    from llenvs.integrations.skyrl._registry import register_estimators, turn_credit

    assert not ray.is_initialized()
    register_estimators()
    register_estimators()
    cfg = LlenvsSkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.logger=console",
            "trainer.algorithm.advantage_estimator=llenvs_turn_grpo",
            "llenvs.sampling_contract=unmodified",
        ]
    )
    validate_cfg(cfg)
    validate_profile(cfg)
    assert cfg.trainer.algorithm.advantage_estimator == "llenvs_turn_grpo"
    assert AdvantageEstimatorRegistry.get("llenvs_turn_grpo") is turn_credit
    assert not ray.is_initialized()


@pytest.mark.parametrize("estimator", ["llenvs_turn_grpo", "llenvs_token_rtg"])
def test_stock_native_invocation_without_attribution_fails(estimator):
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

    from llenvs.integrations.skyrl._registry import register_estimators

    register_estimators()
    function = AdvantageEstimatorRegistry.get(estimator)
    with pytest.raises(TypeError, match="attribution"):
        function(token_level_rewards=None, response_mask=None, index=None, config=None)


def test_native_generator_and_trainers_use_real_extension_points():
    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
    from skyrl.train.generators.base import GeneratorInterface
    from skyrl.train.trainer import RayPPOTrainer

    from llenvs.integrations.skyrl._native import NativeAsyncTrainer, NativeGenerator, NativeTrainer

    assert issubclass(NativeGenerator, GeneratorInterface)
    assert not inspect.isabstract(NativeGenerator)
    assert issubclass(NativeTrainer, RayPPOTrainer)
    assert issubclass(NativeAsyncTrainer, FullyAsyncRayPPOTrainer)


@pytest.mark.parametrize("override", ["+llenvs.unknown=true", "llenvs.unknown=true"])
def test_native_loader_rejects_unknown_fields_and_hydra_prefix(override):
    from llenvs.integrations.skyrl._native import LlenvsSkyRLTrainConfig

    with pytest.raises((ValueError, TypeError)):
        LlenvsSkyRLTrainConfig.from_cli_overrides([override])
