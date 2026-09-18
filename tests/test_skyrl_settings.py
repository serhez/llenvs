"""Connector-only config, task selection and explicit credential forwarding."""

import dataclasses
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llenvs.core.config import BackendFactory, EnvironmentFactory


@pytest.fixture
def settings():
    return importlib.import_module("llenvs.integrations.skyrl._config")


def test_defaults_are_native_and_annotations_support_native_builder(settings):
    config = settings.LlenvsConfig()
    assert config.sampling_contract == "native"
    assert config.turn_weighting == "uniform"
    assert config.judge_timing == "decision"
    assert config.token_scorer is None
    assert config.forward_env == []
    assert config.max_active_episodes == 32
    assert not config.check_only
    assert all(not isinstance(field.type, str) for field in dataclasses.fields(config))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"config": 42},
        {"env_name": ""},
        {"sampling_contract": "temperature_scaled"},
        {"turn_weighting": "length"},
        {"judge_timing": "turn"},
        {"judge_use_images": 1},
        {"check_only": "false"},
        {"max_active_episodes": 0},
        {"max_active_episodes": True},
        {"forward_env": "API_KEY"},
        {"forward_env": ["API_KEY", "API_KEY"]},
        {"forward_env": ["bad-name"]},
        {"forward_env": ["CUDA_VISIBLE_DEVICES"]},
        {"forward_env": ["RAY_ADDRESS"]},
        {"forward_env": ["PYTHONPATH"]},
        {"forward_env": ["PYTORCH_CUDA_ALLOC_CONF"]},
        {"forward_env": ["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"]},
        {"forward_env": ["TORCH_USE_CUDA_DSA"]},
        {"token_scorer": {}},
    ],
)
def test_settings_reject_invalid_or_reserved_fields(settings, kwargs):
    with pytest.raises(ValueError):
        settings.LlenvsConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factory": "path/to/file.py"},
        {"revision": ""},
        {"weight": float("inf")},
        {"weight": True},
        {"kwargs": {"value": float("nan")}},
        {"kwargs": {1: "x"}},
    ],
)
def test_scorer_recipe_requires_an_identified_finite_json_contract(settings, kwargs):
    fields = {"factory": "fixture:score", "revision": "fixed"} | kwargs
    with pytest.raises(ValueError):
        settings.TokenScorerConfig(**fields)


def test_forwarding_contains_only_explicit_names_and_never_prints_values(settings, capsys):
    config = settings.LlenvsConfig(forward_env=["JUDGE_API_KEY"])
    forwarded = settings.forward_environment(
        config, environ={"JUDGE_API_KEY": "secret", "UNRELATED": "other"}
    )
    assert forwarded == {"JUDGE_API_KEY": "secret"}
    assert "secret" not in repr(config)
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="JUDGE_API_KEY"):
        settings.forward_environment(config, environ={})


def test_selection_does_not_create_environment_backend_or_scorer(settings, monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""environments:
  - name: indexed
    adapter: test
    seed: 42
    system_prompt: Environment prompt
    judge: []
system_prompt: Global prompt
judge:
  model:
    model: unused-judge
""")
    create_env = Mock(side_effect=AssertionError("environment constructed"))
    create_backend = Mock(side_effect=AssertionError("backend constructed"))
    monkeypatch.setattr(EnvironmentFactory, "create", create_env)
    monkeypatch.setattr(BackendFactory, "create", create_backend)
    selected = settings.load_selected_environment(settings.LlenvsConfig(config=str(path)))
    assert selected.path == path.resolve()
    assert selected.environment_config.seed == 42
    assert selected.system_prompt == "Environment prompt"
    assert selected.judges == ()
    assert len(selected.fingerprint) == 64
    create_env.assert_not_called()
    create_backend.assert_not_called()


def test_scorer_factory_resolution_does_not_construct_it(settings, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(
        importlib, "import_module", Mock(return_value=SimpleNamespace(create=factory))
    )
    resolved = settings.resolve_factory("fixture.scoring:create")
    assert resolved is factory
    factory.assert_not_called()
