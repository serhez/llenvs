"""Read-only task/model preparation and resolved resume recipes."""

import copy
import importlib
import json
from types import SimpleNamespace as Namespace
from unittest.mock import Mock

import pytest

from llenvs.integrations.skyrl._config import LlenvsConfig


@pytest.fixture
def preparation():
    return importlib.import_module("llenvs.integrations.skyrl._preparation")


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "vocab_size": 100,
                "max_position_embeddings": 1024,
                "attention_dropout": 0.0,
                "use_sliding_window": False,
            }
        )
    )
    (directory / "tokenizer_config.json").write_text('{"tokenizer_class": "Qwen2Tokenizer"}')
    (directory / "tokenizer.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"fixture, not loadable weights")
    return directory


def test_model_identity_reads_contents_without_transformers_or_weight_loading(preparation, model):
    result = preparation.inspect_model(model)
    assert result["vocab_size"] == 100
    assert result["model_type"] == "qwen2"
    assert len(result["content_hash"]) == 64
    (model / "model.safetensors").write_bytes(b"changed fixture")
    assert preparation.inspect_model(model)["content_hash"] != result["content_hash"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen2_moe"),
        ("architectures", ["OtherModel"]),
        ("use_sliding_window", True),
        ("attention_dropout", 0.1),
        ("vocab_size", True),
        ("quantization_config", {"quant_method": "fp8"}),
        ("auto_map", {"AutoConfig": "custom.py"}),
    ],
)
def test_unreviewed_model_recipes_fail_without_importing_custom_code(
    preparation, model, field, value
):
    path = model / "config.json"
    config = json.loads(path.read_text()) | {field: value}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        preparation.inspect_model(model)


def test_tokenizer_custom_code_and_missing_weights_fail(preparation, model):
    (model / "tokenizer_config.json").write_text('{"auto_map": {"AutoTokenizer": "custom.py"}}')
    with pytest.raises(ValueError, match="custom"):
        preparation.inspect_model(model)
    (model / "tokenizer_config.json").write_text("{}")
    (model / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="safetensors"):
        preparation.inspect_model(model)


@pytest.mark.parametrize("shard", ["missing.safetensors", "../outside.safetensors"])
def test_index_cannot_hide_missing_or_outside_model_shards(preparation, model, shard):
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )
    with pytest.raises(ValueError, match="shard"):
        preparation.inspect_model(model)


def test_common_secret_fields_are_rejected_without_printing_values(preparation):
    with pytest.raises(ValueError, match="forward_env") as error:
        preparation._no_credentials({"nested": [{"api_key": "private-value"}]})
    assert "private-value" not in str(error.value)


def test_manifest_recipe_ignores_only_run_locations_and_controls(preparation):
    cfg = {
        "trainer": {
            "seed": 42,
            "resume_mode": "none",
            "resume_path": None,
            "ckpt_path": "/run",
            "export_path": "/export",
            "log_path": "/logs",
            "logger": "console",
        },
        "data": {"train_data": ["/data/train"], "val_data": [], "dataloader": {"shuffle": True}},
        "llenvs": {
            "config": "/config",
            "check_only": True,
            "forward_env": ["JUDGE_KEY"],
            "sampling_contract": "native",
        },
        "generator": {"max_turns": 2},
        "environment": {"env_class": "llenvs"},
    }
    original = copy.deepcopy(cfg)
    recipe = preparation.execution_recipe(cfg)
    assert cfg == original
    cfg["llenvs"]["check_only"] = False
    cfg["trainer"]["resume_mode"] = "from_path"
    cfg["trainer"]["resume_path"] = "/source/global_step_1"
    cfg["trainer"]["ckpt_path"] = "/new-run"
    cfg["data"]["train_data"] = ["/same-content-elsewhere"]
    assert preparation.execution_recipe(cfg) == recipe
    cfg["generator"]["max_turns"] = 3
    assert preparation.execution_recipe(cfg) != recipe


def test_prepare_inputs_does_not_create_environment_or_judge(
    preparation, model, tmp_path, monkeypatch
):
    from llenvs.core.config import BackendFactory, EnvironmentFactory
    from llenvs.core.registry import environment_registry
    from llenvs.integrations.skyrl.data import _fingerprint, _task_id

    path = tmp_path / "config.yaml"
    path.write_text("environments:\n  - name: indexed\n    adapter: test\n    seed: 42\n")
    llconfig = LlenvsConfig(config=str(path))
    selected = preparation.load_selected_environment(llconfig)
    rows = []
    for index in range(2):
        prompt = [{"role": "user", "content": f"Task {index}"}]
        rows.append(
            {
                "prompt": prompt,
                "env_class": "llenvs",
                "data_source": "test/indexed",
                "llenvs": {
                    "schema_version": 1,
                    "env_fingerprint": selected.fingerprint,
                    "task_index": index,
                    "task_id": _task_id(selected.fingerprint, index),
                    "initial_fingerprint": _fingerprint(prompt),
                },
            }
        )
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(map(json.dumps, rows)) + "\n")
    cfg = Namespace(
        llenvs=llconfig,
        data=Namespace(train_data=[str(train)], val_data=[]),
        trainer=Namespace(
            train_batch_size=2, eval_interval=-1, policy=Namespace(model=Namespace(path=str(model)))
        ),
    )
    monkeypatch.setattr(environment_registry, "get_adapter", Mock())
    monkeypatch.setattr(
        EnvironmentFactory, "create", Mock(side_effect=AssertionError("reset during preflight"))
    )
    monkeypatch.setattr(
        BackendFactory, "create", Mock(side_effect=AssertionError("judge during preflight"))
    )
    result = preparation.prepare_inputs(cfg)
    assert len(result.train) == 2 and result.evaluation is None
    assert result.model["vocab_size"] == 100
    assert not (tmp_path / "llenvs-run.json").exists()
