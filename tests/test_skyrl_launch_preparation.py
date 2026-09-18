"""Full connector preparation with pinned native builders and external fixtures.

vLLM's parser/model configuration and native validation/registration are test
boundaries, not installed-stack evidence. Connector preparation is not mocked.
"""

import copy
import dataclasses
import json
import sys
import types
from types import SimpleNamespace as Namespace
from unittest.mock import Mock

import pytest

from tests import test_skyrl_bindings, test_skyrl_preparation
from tests.skyrl_source import config_namespace, definitions

native = test_skyrl_bindings.native
model = test_skyrl_preparation.model


@pytest.fixture
def launch_preparation(native, model, tmp_path, monkeypatch):
    from llenvs.core.config import BackendFactory, EnvironmentFactory
    from llenvs.core.registry import environment_registry
    from llenvs.integrations.skyrl._config import LlenvsConfig, load_selected_environment
    from llenvs.integrations.skyrl.data import _fingerprint, _task_id

    ns = config_namespace(monkeypatch)
    path = tmp_path / "environment.yaml"
    path.write_text("environments:\n  - name: fixture\n    adapter: test\n    seed: 42\n")
    selected = load_selected_environment(LlenvsConfig(config=str(path)))
    rows = []
    for index in range(2):
        prompt = [{"role": "user", "content": f"Task {index}"}]
        rows.append(
            dict(
                prompt=prompt,
                env_class="llenvs",
                data_source="test/fixture",
                llenvs=dict(
                    schema_version=1,
                    env_fingerprint=selected.fingerprint,
                    task_index=index,
                    task_id=_task_id(selected.fingerprint, index),
                    initial_fingerprint=_fingerprint(prompt),
                ),
            )
        )
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(map(json.dumps, rows)) + "\n")
    cfg = ns["build_nested_dataclass"](
        ns["LlenvsSkyRLTrainConfig"],
        {
            "llenvs": {
                "config": str(path),
                "sampling_contract": "unmodified",
                "forward_env": ["JUDGE_FIXTURE_KEY"],
            },
            "data": {"train_data": [str(data)], "val_data": []},
            "trainer": {
                "train_batch_size": 2,
                "policy_mini_batch_size": 2,
                "eval_interval": -1,
                "logger": "console",
                "resume_mode": "none",
                "policy": {"model": {"path": str(model)}},
                "ref": {"model": {"path": str(model)}},
                "ckpt_path": str(tmp_path / "checkpoints"),
                "export_path": str(tmp_path / "exports"),
                "log_path": str(tmp_path / "logs"),
                "algorithm": {"advantage_estimator": "llenvs_turn_grpo"},
            },
        },
    )
    events = []
    monkeypatch.setenv("JUDGE_FIXTURE_KEY", "fixture-secret")
    monkeypatch.setattr(environment_registry, "get_adapter", Mock(return_value=object()))
    for factory in (EnvironmentFactory, BackendFactory):
        monkeypatch.setattr(
            factory, "create", Mock(side_effect=AssertionError("allocation during preflight"))
        )

    def module(name, **values):
        value = types.ModuleType(name)
        vars(value).update(values)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("skyrl", __file__=str(tmp_path / "skyrl/skyrl/__init__.py"))
    module("skyrl.train.utils", validate_cfg=lambda cfg: events.append("native_validation"))
    monkeypatch.setattr(native, "register_estimators", lambda: events.append("registration"))
    # runtime_identity is tested independently below with its real body. The
    # complete prepare() path receives a deterministic external-host inventory.
    monkeypatch.setattr(native, "runtime_identity", lambda root, driver=False: {"fixture_host": 1})
    sampling = module(
        "skyrl.backends.skyrl_train.inference_servers.engine_utils",
        DictConfig=dict,
        ListConfig=list,
    )
    definitions(
        "skyrl/backends/skyrl_train/inference_servers/engine_utils.py",
        {"get_vllm_sampling_params", "get_sampling_params_for_backend"},
        vars(sampling),
    )
    utilities = sys.modules["skyrl.backends.skyrl_train.inference_servers.utils"]
    vars(utilities).update(
        get_config_as_dict=ns["get_config_as_dict"],
        VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS="native-worker-fixture",
        logger=Mock(),
    )
    definitions(
        "skyrl/backends/skyrl_train/inference_servers/utils.py",
        {"build_vllm_cli_args", "_apply_serialized_fp8_weight_sync_defaults"},
        vars(utilities),
    )
    definitions(
        "skyrl/backends/skyrl_train/weight_sync/__init__.py",
        {"get_transfer_strategy"},
        vars(utilities),
    )

    defaults = dict(
        tokenizer=None,
        logprobs_mode="raw_logprobs",
        max_model_len=None,
        enable_chunked_prefill=None,
        speculative_config=None,
        quantization=None,
        kv_cache_dtype="auto",
        hf_overrides={},
        override_generation_config={},
        logits_processors=None,
    )

    class Parser:
        def parse_args(self, args):
            assert args == []
            events.append("engine_arguments")
            return Namespace(**copy.deepcopy(defaults))

    model_config = Namespace(
        max_model_len=1024, logprobs_mode="raw_logprobs", get_vocab_size=lambda: 100
    )

    class EngineArgs:
        @staticmethod
        def add_cli_args(parser):
            return parser

        @staticmethod
        def from_cli_args(args):
            events.append("model_configuration")
            return Namespace(create_model_config=lambda: model_config)

    module("vllm", AsyncEngineArgs=EngineArgs)

    @dataclasses.dataclass
    class WeightTransferConfig:
        backend: str

    module("vllm.config", WeightTransferConfig=WeightTransferConfig)
    module("vllm.entrypoints.openai.cli_args", FrontendArgs=EngineArgs)
    module("vllm.platforms", current_platform=Namespace(device_type=""))
    module("vllm.utils.argparse_utils", FlexibleArgumentParser=Parser)
    return Namespace(
        native=native,
        cfg=cfg,
        events=events,
        defaults=defaults,
        model_config=model_config,
        path=path,
        data=data,
    )


def test_full_prepare_preserves_resolved_phase_sampling_and_has_no_allocations(
    launch_preparation, tmp_path
):
    case = launch_preparation
    before = dataclasses.asdict(case.cfg)
    result = case.native.prepare(case.cfg)
    assert case.events == [
        "registration",
        "native_validation",
        "engine_arguments",
        "model_configuration",
    ]
    assert result.train_sampling["min_tokens"] == 0
    assert result.eval_sampling["min_tokens"] == 1
    assert result.eval_sampling["temperature"] == 0.0
    assert result.engine == {
        "vocab_size": 100,
        "model_context_length": 1024,
        "logprobs_mode": "raw_logprobs",
    }
    assert before == dataclasses.asdict(case.cfg)
    assert "fixture-secret" not in repr(result.identity)
    assert not (tmp_path / "checkpoints").exists()
    assert result.identity == case.native.prepare(case.cfg, driver=True).identity


@pytest.mark.parametrize(
    "change", ["engine_override", "dataset_order", "environment", "sampling", "reward"]
)
def test_complete_identity_changes_for_each_effective_recipe_change(launch_preparation, change):
    case = launch_preparation
    baseline = case.native.prepare(case.cfg).identity
    if change == "engine_override":
        case.cfg.generator.inference_engine.engine_init_kwargs["max_model_len"] = 512
    elif change == "dataset_order":
        case.data.write_text("\n".join(reversed(case.data.read_text().splitlines())) + "\n")
    elif change == "environment":
        # An env change must reject old prepared rows, not just change a hash.
        case.path.write_text(case.path.read_text().replace("seed: 42", "seed: 43"))
        with pytest.raises(ValueError, match="fingerprint"):
            case.native.prepare(case.cfg)
        return
    elif change == "sampling":
        case.cfg.generator.sampling_params.max_generate_length = 512
    else:
        case.cfg.trainer.algorithm.gamma = 0.5
    assert case.native.prepare(case.cfg).identity != baseline


@pytest.mark.parametrize(
    "damage", ["profile", "credentials", "engine", "vocabulary", "probability"]
)
def test_prepare_failures_precede_later_native_resource_boundaries(
    launch_preparation, monkeypatch, damage
):
    case = launch_preparation
    if damage == "profile":
        case.cfg.trainer.remove_microbatch_padding = "false"
    elif damage == "credentials":
        monkeypatch.delenv("JUDGE_FIXTURE_KEY")
    elif damage == "engine":
        case.defaults["logits_processors"] = ["unreviewed-plugin"]
    elif damage == "vocabulary":
        case.model_config.get_vocab_size = lambda: 101
    else:
        case.model_config.logprobs_mode = "processed_logprobs"
    with pytest.raises(ValueError):
        case.native.prepare(case.cfg)
    assert not (case.path.parent / "checkpoints").exists()
    if damage in ("profile", "credentials"):
        assert "engine_arguments" not in case.events
    if damage == "engine":
        assert "model_configuration" not in case.events


def test_secret_rotation_and_resume_locations_do_not_change_identity(
    launch_preparation, monkeypatch
):
    case = launch_preparation
    baseline = case.native.prepare(case.cfg).identity
    monkeypatch.setenv("JUDGE_FIXTURE_KEY", "rotated-fixture-secret")
    case.cfg.llenvs.check_only = True
    case.cfg.trainer.ckpt_path = str(case.path.parent / "new-output")
    case.cfg.trainer.resume_mode = "latest"
    assert case.native.prepare(case.cfg).identity == baseline
