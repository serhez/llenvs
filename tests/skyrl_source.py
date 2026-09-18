"""Test-only execution of pinned native bodies without importing the GPU stack.

No function body is rewritten. Import/registration/timing decorators are not
exercised; missing optional dependencies are explicit fixtures. This is source
evidence, never an installed-runtime or GPU substitute. No network access.
"""

import ast
import copy
import dataclasses
import enum
import heapq
import io
import json
import math
import os
import pickle
import subprocess
import sys
import types
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from unittest.mock import Mock

import pytest

PIN = "4f5ccd8e58bbcea4804bd831fd47097c3044ff48"
REPO = Path(__file__).resolve().parents[1]


def source(path):
    root = Path(os.environ.get("LLENVS_SKYRL_SOURCE", REPO / ".local/upstream/skyrl"))
    if not root.is_dir():
        if "LLENVS_SKYRL_SOURCE" in os.environ:
            pytest.fail("LLENVS_SKYRL_SOURCE does not name an existing checkout")
        pytest.skip("requires the local pinned SkyRL checkout; no automatic clone")
    original = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{PIN}:{path}"], timeout=10
    )
    current = (root / path).read_bytes()
    assert current == original, f"SkyRL source differs from audited pin: {path}"
    return current.decode()


def definitions(path, names, namespace, *, owner=None):
    nodes = ast.parse(source(path)).body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    assert {n.name for n in selected} == set(names)
    for node in selected:
        if not isinstance(node, ast.ClassDef):
            node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), f"<SkyRL {PIN}/{path}>", "exec"), namespace)


def driver_namespace(monkeypatch):
    import numpy as np
    import torch

    module = types.ModuleType("tests._skyrl_source_driver")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = vars(module)
    ns.update(vars(typing))
    ns.update(
        __name__=module.__name__,
        torch=torch,
        np=np,
        math=math,
        copy=copy,
        io=io,
        pickle=pickle,
        logger=Mock(),
        DictType=typing.TypeVar("DictType"),
        defaultdict=defaultdict,
    )
    definitions(
        "skyrl/backends/skyrl_train/training_batch.py",
        {
            "TensorList",
            "TensorBatch",
            "TrainingInput",
            "TrainingInputBatch",
            "pad_training_input_batch",
            "_serialize_tensor",
            "_deserialize_tensor",
            "_rebuild_tensor_batch",
        },
        ns,
    )
    definitions(
        "skyrl/train/dataset/preprocess.py",
        {
            "_verify_inputs",
            "_reward_to_numpy",
            "convert_prompts_responses_to_batch_tensors",
            "compute_prompt_boundaries",
            "compute_prompt_mini_batch_boundaries",
        },
        ns,
    )
    definitions(
        "skyrl/train/trainer.py",
        {"convert_to_training_input", "fwd_logprobs_values_reward", "postprocess_generator_output"},
        ns,
        owner="RayPPOTrainer",
    )
    definitions(
        "skyrl/train/generators/utils.py",
        {
            "_flatten_field",
            "_concat_optional_field",
            "_last_step_only",
            "concatenate_generator_outputs",
            "get_rollout_metrics",
            "compute_turn_token_counts",
            "_add_time_stats",
            "get_metrics_from_generator_output",
        },
        ns,
    )
    definitions("skyrl/train/utils/trainer_utils.py", {"validate_generator_output"}, ns)
    ns["MetricsOutput"] = dict
    validation = types.ModuleType("skyrl.train.utils.trainer_utils")
    validation.validate_generator_output = ns["validate_generator_output"]
    monkeypatch.setitem(sys.modules, validation.__name__, validation)
    return ns


def config_namespace(monkeypatch):
    from llenvs.integrations.skyrl._config import LlenvsConfig

    @dataclasses.dataclass
    class UnusedGymConfig:
        pass

    module = types.ModuleType("tests._skyrl_source_config")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = vars(module)
    ns.update(vars(typing))
    ns.update(
        __name__=module.__name__,
        copy=copy,
        dataclasses=dataclasses,
        json=json,
        os=os,
        typing=typing,
        ABC=ABC,
        asdict=dataclasses.asdict,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        Enum=Enum,
        DictConfig=dict,
        Text2SQLEnvConfig=UnusedGymConfig,
        SearchEnvConfig=UnusedGymConfig,
        LlenvsConfig=LlenvsConfig,
    )
    nodes = [
        n
        for n in ast.parse(source("skyrl/train/config/config.py")).body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<SkyRL config bodies>", "exec"), ns)
    root = next(
        n
        for n in ast.parse((REPO / "src/llenvs/integrations/skyrl/_native.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "LlenvsSkyRLTrainConfig"
    )
    exec(compile(ast.Module(body=[root], type_ignores=[]), "<connector root>", "exec"), ns)
    # Native InferenceEngineConfig.__post_init__ imports this helper. Execute
    # the actual helper, not a replacement Boolean decision.
    helpers = types.ModuleType("skyrl.backends.skyrl_train.inference_servers.utils")
    definitions(
        "skyrl/backends/skyrl_train/inference_servers/utils.py",
        {"_uses_lora_weight_sync"},
        vars(helpers),
    )
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    return ns


def worker_namespace(monkeypatch):
    """Native CPU logprob fallback, policy losses and whole-row token binning.

    No FlashAttention, distributed collective, registry or model loader is
    substituted for a claimed runtime test. Tests provide their own CPU model
    and an explicit collective-boundary fixture when balancing is required.
    """
    import torch

    ns = driver_namespace(monkeypatch)
    ns.update(
        nn=torch.nn,
        F=torch.nn.functional,
        FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False,
        enum=enum,
        heapq=heapq,
        ABC=ABC,
        abstractmethod=abstractmethod,
        dist=torch.distributed,
    )
    definitions(
        "skyrl/backends/skyrl_train/utils/torch_utils.py",
        {
            "masked_mean",
            "safe_exp_delta",
            "logprobs_from_logits",
            "logprobs_from_logits_v2",
            "chunked_entropy_from_logits",
        },
        ns,
    )
    definitions("skyrl/backends/skyrl_train/workers/model_wrapper.py", {"HFModelWrapper"}, ns)
    definitions(
        "skyrl/backends/skyrl_train/utils/off_policy_correction_utils.py",
        {
            "off_policy_correction_enabled",
            "compute_tis_ratio",
            "compute_token_mask",
            "compute_outlier_token_mask",
            "compute_sequence_mask",
            "compute_off_policy_correction",
            "apply_off_policy_correction",
        },
        ns,
    )
    definitions(
        "skyrl/backends/skyrl_train/utils/ppo_utils.py",
        {
            "reduce_loss",
            "ppo_policy_loss",
            "rollout_is_policy_loss",
            "apply_loss_reduction_to_advantages_minibatch",
        },
        ns,
    )
    definitions(
        "skyrl/train/dataset/bin_packing.py",
        {"PackingStrategy", "SeqPacker", "FirstFitDecreasing", "Balanced", "make_seq_packer"},
        ns,
    )
    ns["_PACKERS"] = {
        ns["PackingStrategy"].FIRST_FIT_DECREASING: ns["FirstFitDecreasing"],
        ns["PackingStrategy"].BALANCED: ns["Balanced"],
    }
    definitions(
        "skyrl/backends/skyrl_train/workers/worker_utils.py",
        {
            "BaseBatchIterator",
            "SampleBasedBatchIterator",
            "TokenBasedBatchIterator",
            "get_microbatch_iterator",
        },
        ns,
    )
    return ns
