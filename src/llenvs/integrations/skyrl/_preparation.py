"""Read-only launch preparation and source/model/runtime identity."""

import copy
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llenvs.integrations.skyrl._checks import finite_number, integer
from llenvs.integrations.skyrl._config import (
    SelectedEnvironment,
    load_selected_environment,
    resolve_factory,
)
from llenvs.integrations.skyrl._manifest import content_hash, local_path, tree_hash
from llenvs.integrations.skyrl._preflight import runtime_controls
from llenvs.integrations.skyrl.data import (
    LlenvsPromptDataset,
    _canonical_json,
    _unique_object,
    validate_task_splits,
)

SKYRL_REVISION = "4f5ccd8e58bbcea4804bd831fd47097c3044ff48"


@dataclass(frozen=True)
class PreparedInputs:
    selected: SelectedEnvironment
    train: LlenvsPromptDataset
    evaluation: LlenvsPromptDataset | None
    model: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    _canonical_json(value)
    if not isinstance(value, dict):
        raise ValueError("model/tokenizer configuration must be a JSON object")
    return value


def inspect_model(path: str | Path) -> dict[str, Any]:
    """Inspect a staged dense Qwen2 snapshot; do not import/load/download a model."""
    root = local_path(path, "trainer.policy.model.path")
    config = _json(root / "config.json")
    tokenizer = _json(root / "tokenizer_config.json")
    if config.get("model_type") != "qwen2" or config.get("architectures") != ["Qwen2ForCausalLM"]:
        raise ValueError("initial text recipe requires the reviewed dense Qwen2 causal model")
    if config.get("auto_map") or tokenizer.get("auto_map"):
        raise ValueError("custom model/tokenizer code requires separate review")
    if config.get("quantization_config") or config.get("use_sliding_window", False) is not False:
        raise ValueError("quantization/sliding-window model variants require separate review")
    if finite_number(config.get("attention_dropout", 0), "model attention_dropout") != 0:
        raise ValueError("initial text recipe requires zero attention dropout")
    if not (root / "tokenizer.json").is_file() or not any(root.glob("*.safetensors")):
        raise ValueError("stage the complete tokenizer.json and safetensors snapshot before launch")
    index = root / "model.safetensors.index.json"
    if index.exists():
        weights = _json(index).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("model shard index must contain a nonempty weight_map")
        for shard in weights.values():
            if (
                not isinstance(shard, str)
                or Path(shard).name != shard
                or not (root / shard).is_file()
            ):
                raise ValueError("model shard index references a missing or outside shard")
    return {
        "model_type": config["model_type"],
        "vocab_size": integer(config.get("vocab_size"), "model vocab_size", minimum=1),
        "max_position_embeddings": integer(
            config.get("max_position_embeddings"), "model positions", minimum=1
        ),
        "content_hash": tree_hash(root),
    }


def _no_credentials(value: Any) -> None:
    # This catches common accidental config leaks, not arbitrary secrets hidden
    # under innocuous keys. Configuration is public; use forward_env for values.
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {
                "api_key",
                "api_token",
                "access_token",
                "auth_token",
                "password",
                "secret",
                "authorization",
                "credentials",
            }:
                raise ValueError(
                    "token scorer credentials belong in forward_env, not configuration"
                )
            _no_credentials(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _no_credentials(child)


def prepare_inputs(cfg: Any) -> PreparedInputs:
    selected = load_selected_environment(cfg.llenvs)
    from llenvs.core.registry import environment_registry

    environment_registry.get_adapter(selected.environment_config.adapter)
    scorer = cfg.llenvs.token_scorer
    if scorer is not None:
        _no_credentials(scorer.kwargs)
        resolve_factory(scorer.factory)
    # Additional GPU judges/env-LLMs cannot safely assume the driver's resource
    # placement. Remote backends are explicit; no backend is constructed here.
    models = [judge.model for judge in selected.judges]
    if selected.environment_config.env_llm is not None:
        models.append(selected.environment_config.env_llm.model)
    dependencies = {
        "openai": "openai",
        "openrouter": "openai",
        "anthropic": "anthropic",
        "litellm": "litellm",
    }
    for model in models:
        if model.backend not in dependencies:
            raise ValueError(
                "extra judges/env-LLMs require a remote backend; driver GPU placement is not allocated"
            )
        if importlib.util.find_spec(dependencies[model.backend]) is None:
            raise ValueError(f"missing dependency for configured {model.backend} judge/env-LLM")
    train_paths = [local_path(path, "data.train_data") for path in cfg.data.train_data]
    evaluation_paths = [local_path(path, "data.val_data") for path in cfg.data.val_data]
    train = LlenvsPromptDataset(train_paths, env_fingerprint=selected.fingerprint)
    evaluation = (
        LlenvsPromptDataset(evaluation_paths, env_fingerprint=selected.fingerprint)
        if evaluation_paths
        else None
    )
    validate_task_splits(train, evaluation, train_batch_size=cfg.trainer.train_batch_size)
    if cfg.trainer.eval_interval <= 0 and evaluation is not None:
        raise ValueError("data.val_data is provided but native evaluation is disabled")
    return PreparedInputs(selected, train, evaluation, inspect_model(cfg.trainer.policy.model.path))


def execution_recipe(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude locations/resume controls, not training or reward semantics."""
    result = copy.deepcopy(config)
    for name in (
        "resume_mode",
        "resume_path",
        "ckpt_path",
        "export_path",
        "log_path",
        "logger",
        "project_name",
        "run_name",
        "tags",
    ):
        result["trainer"].pop(name, None)
    for name in ("train_data", "val_data"):
        result["data"].pop(name, None)
    for name in ("config", "check_only"):
        result["llenvs"].pop(name, None)
    return result


def runtime_identity(skyrl_root: Path, *, driver: bool = False) -> dict[str, Any]:
    """Require the audited checkout/stack and fingerprint the complete inventory.

    This is not a dependency installer or a wheel/CUDA acceptance certificate.
    Staging still records the platform wheel hashes and hardware separately.
    """
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
        raise ValueError(
            "native launch/check_only requires the staged Linux Python 3.12 SkyRL environment"
        )
    controls = runtime_controls(os.environ, driver=driver)
    revision = subprocess.check_output(
        ["git", "-C", str(skyrl_root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if revision != SKYRL_REVISION:
        raise ValueError("installed SkyRL source revision differs from the audited integration pin")
    clean = subprocess.run(
        [
            "git",
            "-C",
            str(skyrl_root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "skyrl",
            "uv.lock",
            "pyproject.toml",
        ],
        timeout=10,
        check=False,
    )
    if clean.returncode != 0:
        raise ValueError("audited SkyRL source/lock has local changes")
    required = {
        "ray": "2.57.0",
        "torch": "2.13.0+cu130",
        "vllm": "0.28.0",
        "transformers": "5.16.1",
        "omegaconf": "2.3.1",
    }
    for name, version in required.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"{name} version differs from the audited SkyRL lock")
    inventory = sorted(
        (
            dist.metadata["Name"],
            dist.version,
            content_hash(dist.read_text("RECORD")),
            content_hash(dist.read_text("direct_url.json")),
        )
        for dist in importlib.metadata.distributions()
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": inventory,
        "execution_controls": controls,
        "skyrl_revision": revision,
        "skyrl_source": tree_hash(skyrl_root / "skyrl"),
        "llenvs_source": tree_hash(Path(__file__).resolve().parents[2]),
    }
