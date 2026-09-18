"""Only importable actual estimators are registered; no scalar fallbacks."""

import importlib
from unittest.mock import Mock

import pytest


@pytest.fixture
def registry():
    return importlib.import_module("llenvs.integrations.skyrl._registry")


def test_registration_is_idempotent_without_overwriting_unrelated_functions(registry):
    functions = {}
    native = Mock()
    native.list_available.side_effect = lambda: list(functions)
    native.get.side_effect = functions.__getitem__
    native.register.side_effect = lambda name, fn: functions.__setitem__(name, fn)
    registry.register_estimators(native)
    assert set(functions) == {"llenvs_turn_grpo", "llenvs_token_rtg"}
    assert functions["llenvs_turn_grpo"] is registry.turn_credit
    assert functions["llenvs_token_rtg"] is registry.token_credit
    registry.register_estimators(native)
    assert native.register.call_count == 2
    functions["llenvs_turn_grpo"] = lambda **kwargs: None
    with pytest.raises(ValueError, match="already registered"):
        registry.register_estimators(native)
    assert native.register.call_count == 2


@pytest.mark.parametrize("name", ["turn_credit", "token_credit"])
def test_stock_invocation_without_ledger_fails(registry, name):
    with pytest.raises(TypeError, match="attribution"):
        getattr(registry, name)(
            token_level_rewards=None, response_mask=None, index=None, config=None
        )


def test_registered_functions_run_actual_credit_math(registry):
    torch = pytest.importorskip("torch")
    ledger = [
        {
            "instance_id": "occurrence",
            "repetition_id": 0,
            "response_length": 2,
            "sampled_spans": [{"generation_id": "g", "start": 0, "end": 2, "sampled_count": 2}],
        }
    ]
    advantages, returns = registry.token_credit(
        token_level_rewards=torch.tensor([[1.0, 2.0]]),
        attribution=ledger,
        n_samples_per_prompt=1,
        grpo_norm_by_std=False,
    )
    assert advantages.tolist() == [[3.0, 2.0]]
    assert returns.tolist() == advantages.tolist()
