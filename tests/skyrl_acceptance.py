"""Test-only acceptance settings and deterministic multi-turn fixtures.

Nothing here launches a runtime, downloads assets, or changes the production
adapter registry on import. Tests opt in explicitly and register per process.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Observation, ObservationContent, State, StateMetadata
from llenvs.core.tools import ToolDefinition, ToolResult
from llenvs.integrations.skyrl.data import _unique_object
from llenvs.integrations.skyrl.scoring import GenerationTokenRewards


def acceptance_options(environ):
    if environ.get("LLENVS_SKYRL_GPU") != "1":
        raise ValueError("GPU acceptance requires explicit LLENVS_SKYRL_GPU=1")
    result = {}
    for key in ("MODEL", "ARTIFACTS", "TOLERANCES"):
        value = environ.get(f"LLENVS_SKYRL_{key}")
        path = Path(value) if value else None
        if path is None or not path.is_absolute():
            raise ValueError(f"LLENVS_SKYRL_{key} must be an explicit local absolute path")
        result[key.lower()] = path
    if not result["model"].is_dir() or not result["tolerances"].is_file():
        raise ValueError("stage the model snapshot and reviewed tolerance file before acceptance")
    thresholds = json.loads(result["tolerances"].read_text(), object_pairs_hook=_unique_object)
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "model_content_hash",
        "runtime_hash",
        "bounds",
    }:
        raise ValueError("tolerances require model_content_hash, runtime_hash, and bounds")
    for key in ("model_content_hash", "runtime_hash"):
        value = thresholds[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"invalid tolerance identity: {key}")
    bounds = thresholds["bounds"]
    if not isinstance(bounds, dict) or set(bounds) != {
        "logprob_max_abs",
        "gradient_relative_l2",
        "update_relative_l2",
        "rollout_logprob_max_abs",
    }:
        raise ValueError("missing or unknown acceptance bounds")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in bounds.values()
    ):
        raise ValueError("acceptance bounds must be explicit finite nonnegative numbers")
    result["thresholds"] = thresholds
    return result


def acceptance_runtime():
    """Fail closed before Ray/model allocation; use only explicitly staged assets."""
    import os

    import skyrl
    import torch

    from llenvs.integrations.skyrl._manifest import content_hash
    from llenvs.integrations.skyrl._preparation import inspect_model, runtime_identity

    options = acceptance_options(os.environ)
    runtime = runtime_identity(Path(skyrl.__file__).resolve().parent.parent)
    model = inspect_model(options["model"])
    assert options["thresholds"]["model_content_hash"] == model["content_hash"], (
        "tolerances belong to a different model snapshot"
    )
    assert options["thresholds"]["runtime_hash"] == content_hash(runtime), (
        "tolerances belong to a different runtime"
    )
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), (
        "requires allocated BF16-capable CUDA hardware"
    )
    options["artifacts"].mkdir(parents=True, exist_ok=True)
    options["identity"] = dict(
        model=model,
        runtime=runtime,
        hardware=[
            dict(
                name=torch.cuda.get_device_name(i),
                memory=torch.cuda.get_device_properties(i).total_memory,
                capability=list(torch.cuda.get_device_capability(i)),
            )
            for i in range(torch.cuda.device_count())
        ],
    )
    return options


class FixtureEnvironment:
    """Two decisions regardless of reply quality; rewards and tools are local.

    This tests transport, not learning or task success. No external side
    effects or hidden evaluator calls. Reset identity depends only on seed/index.
    """

    spec = EnvironmentSpec("transport", "skyrl_acceptance", is_multi_turn=True)
    reward_functions = ()
    tool = ToolDefinition(name="lookup", description="Return the fixed fixture observation.")

    def __init__(self, *, size=8, seed=42, **kwargs):
        self.size, self.seed, self.closed, self.turn = size, seed, False, 0

    def __len__(self):
        return self.size

    def reset(self, *, options):
        index = options["task_index"]
        if self.closed or type(index) is not int or not 0 <= index < self.size:
            raise ValueError("invalid fixture reset")
        self.turn = 0
        observation = Observation(
            f"Fixture {self.seed}/{index}. Call lookup with no arguments, then answer briefly.",
            available_tools=(self.tool,),
        )
        return State(
            observation, hidden=None, metadata=StateMetadata(step=0, episode_id=f"fixture-{index}")
        ), {}

    def step(self, state, action):
        if self.closed or state.metadata.step != self.turn or self.turn >= 2:
            raise ValueError("stale or closed fixture session")
        self.turn += 1
        results = tuple(
            ToolResult.success(call.id, call.name, {"turn": self.turn})
            for call in action.tool_calls
        )
        observation = replace(
            state.observation,
            state=ObservationContent(text=f"Observation {self.turn}: continue briefly."),
            tool_results=results,
        )
        next_state = replace(
            state,
            observation=observation,
            metadata=replace(state.metadata, step=self.turn, is_terminal=self.turn == 2),
        )
        return StepResult(
            next_state,
            SignalBundle.single(self.turn + (len(action.text or "") % 5) / 10),
            terminated=self.turn == 2,
        )

    def close(self):
        self.closed = True


class FixtureAdapter:
    name = "skyrl_acceptance"

    def list_environments(self):
        return ["transport"]

    def get_environment(self, name, **kwargs):
        if name != "transport":
            raise ValueError(name)
        return FixtureEnvironment(**kwargs)


def register_fixture():
    from llenvs.core.registry import environment_registry

    # Own a unique test-only adapter; never replace somebody else's registration.
    existing = environment_registry._adapters.get(FixtureAdapter.name)
    if existing is None:
        environment_registry.register_adapter(FixtureAdapter())
    elif not isinstance(existing, FixtureAdapter):
        raise ValueError("acceptance fixture adapter name is already owned")


class TokenIndicator:
    reward_semantics = "prefix_causal_additive"

    async def __call__(self, generation):
        return GenerationTokenRewards(
            occurrence_id=generation.occurrence_id,
            generation_id=generation.generation_id,
            rewards=tuple((token % 3 - 1) / 10 for token in generation.output_ids),
        )


def smoke_config(root, model, *, asynchronous, estimator, dp, resume=False):
    """A fixed two-update recipe, shared by CPU preflight and GPU subprocesses."""
    if type(dp) is not int or dp not in (1, 2):
        raise ValueError("smoke acceptance admits DP1/2 only")
    if estimator not in ("grpo", "llenvs_turn_grpo", "llenvs_token_rtg"):
        raise ValueError("unknown smoke estimator")
    phase = "resumed" if resume else "fresh"
    return {
        "llenvs": {
            "config": str(root / "environment.yaml"),
            "sampling_contract": "unmodified",
            "max_active_episodes": 4,
            "forward_env": ["LLENVS_SKYRL_GPU", "LLENVS_SKYRL_TOLERANCES"],
            "token_scorer": (
                {"factory": "tests.skyrl_acceptance:TokenIndicator", "revision": "fixture-v1"}
                if estimator == "llenvs_token_rtg"
                else None
            ),
        },
        "environment": {"env_class": "llenvs", "skyrl_gym": {"max_env_workers": 2}},
        "data": {
            "train_data": [str(root / "train.jsonl")],
            "val_data": [str(root / "eval.jsonl")],
            "dataloader": {"num_workers": 0},
        },
        "trainer": {
            "logger": "console",
            "train_batch_size": 2,
            "policy_mini_batch_size": 2,
            "eval_batch_size": 2,
            "eval_interval": 1,
            "eval_before_train": False,
            "max_training_steps": 2,
            "epochs": 1,
            "ckpt_interval": 1,
            "max_ckpts_to_keep": -1,
            "resume_mode": "from_path" if resume else "none",
            "resume_path": str(root / "fresh/checkpoints/global_step_1") if resume else None,
            "ckpt_path": str(root / phase / "checkpoints"),
            "export_path": str(root / phase / "exports"),
            "log_path": str(root / phase / "logs"),
            "max_prompt_length": 1024,
            "policy": {"model": {"path": str(model)}},
            "ref": {"model": {"path": str(model)}},
            "algorithm": {
                "advantage_estimator": estimator,
                "grpo_norm_by_std": estimator != "llenvs_token_rtg",
                "policy_loss_type": "rollout_is" if asynchronous else "regular",
                "use_kl_loss": False,
            },
            "fully_async": {"enabled": asynchronous, "num_parallel_generation_workers": 2},
            "placement": {
                "policy_num_gpus_per_node": dp,
                "ref_num_gpus_per_node": dp,
                "colocate_all": not asynchronous,
            },
        },
        "generator": {
            "n_samples_per_prompt": 2,
            "eval_n_samples_per_prompt": 1,
            "max_turns": 2,
            "max_input_length": 1536,
            "sampling_params": {"max_generate_length": 128},
            # Omitting the nested object preserves native greedy evaluation
            # and inherits the training length. A partial object resets defaults.
            "inference_engine": {
                "num_engines": 1 if asynchronous else dp,
                "engine_init_kwargs": {"max_model_len": 2048},
            },
        },
    }
