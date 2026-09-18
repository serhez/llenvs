"""Thin native binding behavior against explicit interface doubles.

Installed SkyRL construction/registry tests live in test_skyrl_runtime.py.
"""

import asyncio
import dataclasses
import importlib
import sys
import types
from types import SimpleNamespace as Namespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def native(monkeypatch):
    pytest.importorskip("torch")

    @dataclasses.dataclass
    class Environment:
        env_class: str = "gsm8k"

    @dataclasses.dataclass
    class Config:
        environment: Environment = dataclasses.field(default_factory=Environment)

    class Base:
        pass

    class Trainer:
        pass

    class AsyncTrainer(Trainer):
        pass

    class Interface:
        pass

    interfaces = {
        "skyrl.train.config": {"SkyRLTrainConfig": Config, "EnvironmentConfig": Environment},
        "skyrl.train.entrypoints.main_base": {"BasePPOExp": Base},
        "skyrl.train.trainer": {"RayPPOTrainer": Trainer},
        "skyrl.train.fully_async_trainer": {"FullyAsyncRayPPOTrainer": AsyncTrainer},
        "skyrl.train.generators.base": {"GeneratorInterface": Interface},
        "skyrl.train.generators.utils": {"get_rollout_metrics": Mock(return_value={"native": 1.0})},
    }
    for name, attributes in interfaces.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    name = "llenvs.integrations.skyrl._native"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def test_root_is_importable_concrete_native_subclass_and_defaults_to_llenvs(native):
    cfg = native.LlenvsSkyRLTrainConfig()
    assert cfg.environment.env_class == "llenvs"
    assert cfg.llenvs.sampling_contract == "native"
    assert not cfg.llenvs.check_only
    assert all(not isinstance(field.type, str) for field in dataclasses.fields(cfg))


@pytest.mark.parametrize("phase", ["train", "eval"])
def test_native_generator_preserves_eval_and_keeps_ledger_away_from_numeric_aggregator(
    native, monkeypatch, phase
):
    from llenvs.integrations.skyrl._generator import EpisodeGenerator

    generator = object.__new__(native.NativeGenerator)
    generator.cfg = Namespace(llenvs=Namespace(sampling_contract="unmodified"))
    generator.train_sampling_params = {"temperature": 1.0, "min_tokens": 1}
    generator.eval_sampling_params = {"temperature": 0.0, "min_tokens": 1}
    request = {
        "batch_metadata": Namespace(training_phase=phase),
        "sampling_params": {"temperature": 1.0 if phase == "train" else 0.0, "min_tokens": 1},
    }
    output = {
        "response_ids": [[1, 2]],
        "rewards": [[0, 1]],
        "loss_masks": [[1, 1]],
        "env_metrics": [{"llenvs/attribution": {"spans": "not a metric"}}],
        "trajectory_generation_times": [0.1],
        "trajectory_time_splits": {"llm": [0.05]},
    }
    generate = AsyncMock(return_value=output)
    monkeypatch.setattr(EpisodeGenerator, "generate", generate)
    result = asyncio.run(generator.generate(request))
    passed = generate.call_args.args[0]["sampling_params"]
    assert passed["min_tokens"] == (0 if phase == "train" else 1)
    assert request["sampling_params"]["min_tokens"] == 1
    assert result["env_metrics"] == output["env_metrics"]
    assert "env_metrics" not in native.get_rollout_metrics.call_args.kwargs
    assert result["rollout_metrics"] == {"native": 1.0}


def test_native_metric_failure_closes_generator_before_propagating(native, monkeypatch):
    from llenvs.integrations.skyrl._generator import EpisodeGenerator

    generator = object.__new__(native.NativeGenerator)
    generator.cfg = Namespace(llenvs=Namespace(sampling_contract="native"))
    generator.train_sampling_params = {"min_tokens": 1}
    generator.aclose = AsyncMock()
    monkeypatch.setattr(EpisodeGenerator, "generate", AsyncMock(return_value={}))
    with pytest.raises(KeyError):
        asyncio.run(generator.generate({"batch_metadata": Namespace(training_phase="train")}))
    generator.aclose.assert_awaited_once()


def test_driver_identity_mismatch_prevents_manifest_write_and_experiment(native, monkeypatch):
    monkeypatch.setattr(native, "prepare", Mock(return_value=Namespace(identity={"actual": 1})))
    manifest = Mock()
    experiment = Mock()
    monkeypatch.setattr(native, "check_run", manifest)
    monkeypatch.setattr(native, "LlenvsPPOExp", experiment)
    with pytest.raises(ValueError, match="Ray driver"):
        native.run_experiment(Namespace(), {"expected": 1})
    manifest.assert_not_called()
    experiment.assert_not_called()


def test_rendered_budget_does_not_use_the_native_loss_normalizer(native):
    from tests.test_skyrl_rendering import Tokenizer

    experiment = object.__new__(native.LlenvsPPOExp)
    experiment.tokenizer = Tokenizer()
    experiment.cfg = Namespace(
        trainer=Namespace(max_prompt_length=128, algorithm=Namespace(max_seq_len=1)),
        generator=Namespace(max_input_length=128, chat_template_kwargs={}),
    )
    experiment.prepared = Namespace(engine={"vocab_size": 256, "model_context_length": 256})
    dataset = Namespace(_rows=[{"prompt": [{"role": "user", "content": "Task"}]}])
    assert experiment._validate_rendered_inputs(dataset) is dataset


def test_launch_uses_native_initialization_forwards_only_values_and_always_disconnects(
    native, monkeypatch
):
    events = []
    ray = types.ModuleType("ray")
    ray.is_initialized = Mock(return_value=False)
    remote = Mock()
    remote.options.return_value = remote
    remote.remote.side_effect = lambda *args: events.append("driver") or "reference"
    ray.remote = Mock(return_value=lambda function: remote)
    ray.get = Mock(side_effect=RuntimeError("driver failed"))
    ray.shutdown = Mock(side_effect=lambda: events.append("shutdown"))
    helpers = types.ModuleType("skyrl.train.utils.utils")
    helpers.initialize_ray = Mock(side_effect=lambda cfg: events.append("initialize"))
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setattr(
        native, "forward_environment", Mock(return_value={"JUDGE_API_KEY": "private"})
    )
    cfg = Namespace(llenvs=Namespace(forward_env=["JUDGE_API_KEY"]))
    with pytest.raises(RuntimeError, match="driver failed"):
        native.launch(cfg, {"runtime": "hash"})
    assert events == ["initialize", "driver", "shutdown"]
    remote.options.assert_called_once_with(runtime_env={"env_vars": {"JUDGE_API_KEY": "private"}})
    assert "private" not in repr(remote.remote.call_args)


def test_launch_refuses_existing_ray_session(native, monkeypatch):
    ray = types.ModuleType("ray")
    ray.is_initialized = Mock(return_value=True)
    helpers = types.ModuleType("skyrl.train.utils.utils")
    helpers.initialize_ray = Mock()
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    with pytest.raises(ValueError, match="fresh"):
        native.launch(Namespace(), {})
    helpers.initialize_ray.assert_not_called()
