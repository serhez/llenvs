"""Model-free guards for the expensive acceptance harness and its fixtures."""

import json

import pytest

from llenvs.core.state import Action
from tests import test_skyrl_bindings as binding_tests
from tests.skyrl_acceptance import FixtureEnvironment, acceptance_options

native = binding_tests.native


def test_gpu_harness_requires_explicit_opt_in_before_touching_paths():
    with pytest.raises(ValueError, match="explicit"):
        acceptance_options({})


@pytest.fixture
def acceptance_env(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    tolerance = tmp_path / "tolerances.json"
    tolerance.write_text(
        json.dumps(
            dict(
                model_content_hash="a" * 64,
                runtime_hash="b" * 64,
                bounds=dict(
                    logprob_max_abs=0.01,
                    gradient_relative_l2=0.01,
                    update_relative_l2=0.01,
                    rollout_logprob_max_abs=0.01,
                ),
            )
        )
    )
    return {
        "LLENVS_SKYRL_GPU": "1",
        "LLENVS_SKYRL_MODEL": str(model),
        "LLENVS_SKYRL_ARTIFACTS": str(tmp_path / "artifacts"),
        "LLENVS_SKYRL_TOLERANCES": str(tolerance),
    }


def test_acceptance_settings_are_read_only_and_do_not_invent_bounds(acceptance_env):
    settings = acceptance_options(acceptance_env)
    assert settings["thresholds"]["bounds"]["logprob_max_abs"] == 0.01
    assert not settings["artifacts"].exists()


@pytest.mark.parametrize(
    "damage",
    [
        "relative",
        "missing_model",
        "missing_bounds",
        "nan",
        "boolean",
        "negative",
        "identity",
        "not_object",
    ],
)
def test_explicit_but_invalid_acceptance_recipe_fails(acceptance_env, damage):
    from pathlib import Path

    path = Path(acceptance_env["LLENVS_SKYRL_TOLERANCES"])
    document = json.loads(path.read_text())
    if damage == "relative":
        acceptance_env["LLENVS_SKYRL_MODEL"] = "relative-model"
    elif damage == "missing_model":
        acceptance_env["LLENVS_SKYRL_MODEL"] += "-missing"
    elif damage == "missing_bounds":
        document["bounds"].pop("logprob_max_abs")
    elif damage == "identity":
        document["runtime_hash"] = "unverified"
    elif damage == "not_object":
        document = None
    else:
        document["bounds"]["logprob_max_abs"] = {
            "nan": float("nan"),
            "boolean": True,
            "negative": -1,
        }[damage]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        acceptance_options(acceptance_env)


def test_multi_turn_fixture_is_deterministic_and_rejects_stale_state():
    env = FixtureEnvironment(size=2, seed=123)
    initial, _ = env.reset(options={"task_index": 1})
    assert env.reset(options={"task_index": 1})[0] == initial
    first = env.step(initial, Action.from_text("one"))
    assert not first.done and first.rewards.total == 1.3
    with pytest.raises(ValueError, match="stale"):
        env.step(initial, Action.from_text("again"))
    second = env.step(first.next_state, Action.from_text("two"))
    assert second.done and second.rewards.total == 2.3
    env.close()
    with pytest.raises(ValueError):
        env.reset(options={"task_index": 0})


def test_fixture_exports_public_tasks_and_executes_a_real_tool_call(monkeypatch, tmp_path):
    from llenvs.core.registry import environment_registry
    from llenvs.core.tools import ToolCall, ToolResultStatus
    from llenvs.integrations.skyrl.data import export_prompt_data
    from tests.skyrl_acceptance import register_fixture

    monkeypatch.setattr(environment_registry, "_adapters", {})
    register_fixture()
    register_fixture()  # A second use must not replace the existing instance.
    config = tmp_path / "environment.yaml"
    config.write_text(
        json.dumps(
            {
                "environments": [
                    {
                        "name": "transport",
                        "adapter": "skyrl_acceptance",
                        "size": 3,
                        "seed": 7,
                    }
                ]
            }
        )
    )
    path = tmp_path / "tasks.jsonl"
    assert export_prompt_data(config, path, indices=[2, 0]) == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["llenvs"]["task_index"] for r in rows] == [2, 0]
    assert "lookup" in rows[0]["prompt"][0]["content"]
    env = FixtureEnvironment(size=3, seed=7)
    state, _ = env.reset(options={"task_index": 2})
    result = env.step(state, Action.from_tool_call(ToolCall("call-1", "lookup", {})))
    (tool_result,) = result.next_state.observation.tool_results
    assert tool_result.call_id == "call-1"
    assert tool_result.status == ToolResultStatus.SUCCESS
    assert tool_result.output == {"turn": 1}
    env.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("estimator", ["grpo", "llenvs_turn_grpo", "llenvs_token_rtg"])
@pytest.mark.parametrize("dp", [1, 2])
def test_smoke_configs_build_real_native_dataclasses_and_pass_profile(
    monkeypatch, tmp_path, asynchronous, estimator, dp
):
    from llenvs.integrations.skyrl._preflight import validate_profile
    from tests.skyrl_acceptance import smoke_config
    from tests.skyrl_source import config_namespace

    ns = config_namespace(monkeypatch)
    for resume in (False, True):
        raw = smoke_config(
            tmp_path,
            tmp_path / "model",
            asynchronous=asynchronous,
            estimator=estimator,
            dp=dp,
            resume=resume,
        )
        cfg = ns["build_nested_dataclass"](ns["LlenvsSkyRLTrainConfig"], raw)
        validate_profile(cfg)
        assert cfg.generator.eval_sampling_params.temperature == 0.0
        assert cfg.trainer.max_training_steps == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_audit_driver_selects_hooks_and_restores_experiment(
    native, monkeypatch, asynchronous, fail
):
    from types import SimpleNamespace as Namespace

    from tests import skyrl_acceptance, skyrl_acceptance_run

    monkeypatch.setattr(skyrl_acceptance, "register_fixture", lambda: None)
    observed = []

    class Trainer:
        def __init__(self, **kwargs):
            self.arguments = kwargs

    class AsyncTrainer(Trainer):
        pass

    monkeypatch.setattr(native, "NativeTrainer", Trainer)
    monkeypatch.setattr(native, "NativeAsyncTrainer", AsyncTrainer)
    original = native.LlenvsPPOExp

    def run(cfg, identity):
        exp = object.__new__(native.LlenvsPPOExp)
        trainer = exp.get_trainer(cfg, None, None, None, None, None, None, None)
        assert isinstance(trainer, skyrl_acceptance_run.AuditTrainerMixin)
        assert isinstance(trainer, AsyncTrainer) == asynchronous
        observed.append(identity)
        if fail:
            raise RuntimeError("test driver failure")

    monkeypatch.setattr(native, "run_experiment", run)
    cfg = Namespace(trainer=Namespace(fully_async=Namespace(enabled=asynchronous)))
    if fail:
        with pytest.raises(RuntimeError, match="test driver failure"):
            skyrl_acceptance_run.audited_run_experiment(cfg, {"identity": "fixture"})
    else:
        skyrl_acceptance_run.audited_run_experiment(cfg, {"identity": "fixture"})
    assert native.LlenvsPPOExp is original
    assert observed == [{"identity": "fixture"}]


@pytest.mark.parametrize("fail", [False, True])
def test_audit_lifecycle_writes_success_and_failure_without_suppressing_errors(tmp_path, fail):
    import asyncio
    from types import SimpleNamespace as Namespace

    from tests.skyrl_acceptance_run import AuditTrainerMixin

    class Native:
        async def eval(self):
            return {"eval/fixture": 1.0}

        def train_critic_and_policy(self, batch):
            return {"policy_loss": 0.25}

        async def train(self):
            self._audit_batches.append({"global_step": self.global_step})
            self.train_critic_and_policy({})
            await self.eval()
            if fail:
                raise RuntimeError("native failure")
            return "native result"

    class Trainer(AuditTrainerMixin, Native):
        pass

    trainer = Trainer()
    trainer.global_step = 1
    trainer.cfg = Namespace(trainer=Namespace(log_path=tmp_path))
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            asyncio.run(trainer.train())
    else:
        assert asyncio.run(trainer.train()) == "native result"
    report = json.loads((tmp_path / "llenvs-acceptance.json").read_text())
    assert report["completed"] == (not fail)
    assert report["evaluation_steps"] == [1] and report["final_counter"] == 1
    assert report["batches"][0]["policy_metrics"] == {"policy_loss": 0.25}


def test_audit_entry_requires_opt_in_before_native_import(monkeypatch):
    from tests.skyrl_acceptance_run import main

    monkeypatch.delenv("LLENVS_SKYRL_GPU", raising=False)
    with pytest.raises(RuntimeError, match="explicit GPU"):
        main()


@pytest.mark.parametrize("address", [None, "auto", "ray://unrelated-cluster:10001"])
def test_audit_entry_refuses_implicit_or_existing_ray_clusters(monkeypatch, address):
    from tests.skyrl_acceptance_run import main

    monkeypatch.setenv("LLENVS_SKYRL_GPU", "1")
    if address is None:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("RAY_ADDRESS", address)
    with pytest.raises(RuntimeError, match="RAY_ADDRESS=local"):
        main()


def test_audit_launcher_serializes_importable_entry_and_restores_it_on_failure(
    native, monkeypatch, tmp_path, acceptance_env
):
    import sys
    from types import SimpleNamespace as Namespace

    from llenvs.core.registry import environment_registry
    from tests import skyrl_acceptance_run

    monkeypatch.setattr(environment_registry, "_adapters", {})
    for key, value in acceptance_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setattr(sys, "argv", ["acceptance", "sync", "grpo", str(tmp_path)])
    monkeypatch.setattr(native.LlenvsSkyRLTrainConfig, "from_cli_overrides", lambda raw: raw)
    monkeypatch.setattr(native, "prepare", lambda cfg: Namespace(identity={"test": "identity"}))
    original = native.run_experiment

    def launch(cfg, identity):
        assert native.run_experiment is skyrl_acceptance_run.audited_run_experiment
        assert native.run_experiment.__module__ == "tests.skyrl_acceptance_run"
        assert identity == {"test": "identity"}
        assert (tmp_path / "train.jsonl").is_file()
        assert (tmp_path / "eval.jsonl").is_file()
        raise RuntimeError("native launch failure")

    monkeypatch.setattr(native, "launch", launch)
    with pytest.raises(RuntimeError, match="native launch failure"):
        skyrl_acceptance_run.main()
    assert native.run_experiment is original
