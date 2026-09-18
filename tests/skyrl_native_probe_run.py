"""Owned, opt-in native GPU component probes. Never launch this on import."""

import argparse
import asyncio
import json
import os
from pathlib import Path


def worker_recipe(model, directory, *, dp, budget, clear_cache=None):
    assert type(dp) is int and dp in (1, 2)
    assert type(budget) is int and budget >= 0
    assert clear_cache is None or type(clear_cache) is bool
    return {
        "trainer": {
            "strategy": "fsdp",
            "logger": "console",
            "log_path": str(directory / "logs"),
            "seed": 731,
            "flash_attn": True,
            "gradient_checkpointing": True,
            "remove_microbatch_padding": budget > 0,
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "max_tokens_per_microbatch": budget or -1,
            "policy": {"model": {"path": str(model)}},
            "placement": {
                "policy_num_gpus_per_node": dp,
                "colocate_all": False,
                "colocate_policy_ref": False,
            },
            "algorithm": {
                "policy_loss_type": "regular" if clear_cache is None else "rollout_is",
                "use_kl_loss": False,
                "use_entropy_loss": False,
                "temperature": 1.0,
                "loss_reduction": "token_mean",
                "advantage_batch_normalize": False,
            },
            "fully_async": {
                "enabled": clear_cache is not None,
                "clear_kv_cache_on_weight_sync": bool(clear_cache),
            },
        },
        "generator": {
            "sampling_params": {"temperature": 1.0},
            "inference_engine": {
                "num_engines": 1,
                "tensor_parallel_size": 1,
                "enable_prefix_caching": True,
                "weight_sync_backend": "nccl",
                "engine_init_kwargs": {"max_model_len": 2048},
            },
        },
    }


def fixture_batch(options, estimator):
    from skyrl.backends.skyrl_train.utils.ppo_utils import (
        apply_loss_reduction_to_advantages_minibatch,
    )
    from transformers import AutoTokenizer

    from tests.skyrl_layout_fixture import layout_batch

    tokenizer = AutoTokenizer.from_pretrained(
        options["model"], local_files_only=True, trust_remote_code=False
    )
    batch, trace = layout_batch(
        tokenizer, options["identity"]["model"]["vocab_size"], estimator, "uniform"
    )
    batch["advantages"] = apply_loss_reduction_to_advantages_minibatch(
        batch["advantages"], batch["loss_mask"], "token_mean", 1, 1024, None
    )
    # Global credit is already computed. Sort the whole tensor batch, not just
    # IDs, to place two short rows on DP rank 0 and two long ones on rank 1.
    lengths = batch["attention_mask"].sum(-1)
    order = lengths.argsort()
    ordered = type(batch)({k: v[order] if v is not None else None for k, v in batch.items()})
    ordered.metadata = batch.metadata
    batch = ordered
    lengths = batch["attention_mask"].sum(-1)
    budget = int(max(lengths[:2].sum(), lengths[-1]))
    assert int(lengths[2:].sum()) > budget, "fixture must require a real balancing dummy"
    return batch, trace, tokenizer, budget


def logprobs(dispatch, batch):
    import torch

    result = dispatch.forward("policy", batch)
    values = torch.tensor([r["logprobs"] for r in result.loss_fn_outputs])
    assert values.shape == batch["loss_mask"].shape and bool(torch.isfinite(values).all())
    return values


def batch_identity(batch):
    from llenvs.integrations.skyrl._manifest import content_hash

    return content_hash(
        {
            "metadata": batch.metadata,
            "tensors": {
                k: {"dtype": str(v.dtype), "shape": list(v.shape), "values": v.tolist()}
                if v is not None
                else None
                for k, v in batch.items()
            },
        }
    )


def distributed_probe(options, root, *, estimator, dp, packed):
    import torch
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict
    from skyrl.train.utils.utils import validate_cfg

    from tests.skyrl_distributed_probe import policy_workers, train_step

    directory = root / f"dp{dp}"
    directory.mkdir()
    batch, trace, _, budget = fixture_batch(options, estimator)
    if dp == 2:
        saved = torch.load(root / "dp1/batch.pt", weights_only=True, map_location="cpu")
        expected = TrainingInputBatch(saved["tensors"])
        expected.metadata = saved["metadata"]
        for key, value in batch.items():
            counterpart = expected.get(key)
            assert isinstance(value, torch.Tensor) and isinstance(counterpart, torch.Tensor)
            assert value.dtype == counterpart.dtype and torch.equal(value, counterpart)
        batch = expected
    cfg = SkyRLTrainConfig.from_cli_overrides(
        worker_recipe(options["model"], directory, dp=dp, budget=budget if packed else 0)
    )
    validate_cfg(cfg)
    with policy_workers(cfg) as (dispatch, snapshot):
        current = logprobs(dispatch, batch)
        if dp == 1:
            batch["action_log_probs"] = current
            batch["rollout_logprobs"] = current.clone()
            with (directory / "batch.pt").open("xb") as stream:
                torch.save({"tensors": dict(batch.items()), "metadata": batch.metadata}, stream)
        loss_mask = batch.get("loss_mask")
        assert isinstance(loss_mask, torch.Tensor)
        mask = loss_mask.bool()
        parity = (current - batch["action_log_probs"])[mask].abs().max().item()
        report = train_step(dispatch, batch, snapshot, directory)
    report.update(
        config=get_config_as_dict(cfg),
        batch_hash=batch_identity(batch),
        forward_max_abs=parity,
        trace=trace,
        dp=dp,
        packed=packed,
    )
    if dp == 2 and packed:
        assert report["metrics"]["num_padding_microbatches"] > 0, "no DP balancing dummy ran"
    return report


async def cache_probe(options, root, *, clear_cache):
    import torch
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict
    from skyrl.train.utils.utils import validate_cfg

    from tests.skyrl_distributed_probe import policy_workers, train_step
    from tests.skyrl_gpu_probes import check_continuity, observed_server_class

    batch, _, tokenizer, _ = fixture_batch(options, "llenvs_token_rtg")
    cfg = SkyRLTrainConfig.from_cli_overrides(
        worker_recipe(options["model"], root, dp=1, budget=0, clear_cache=clear_cache)
    )
    validate_cfg(cfg)
    group = ServerGroup(
        build_vllm_cli_args(cfg),
        num_servers=1,
        server_actor_cls=observed_server_class(),
        enable_ray_prometheus_stats=False,
    )
    client = task = None
    try:
        group.start()
        server = group.get_actors()[0]
        urls = group.get_server_urls()
        # One native server, no routing ambiguity. Ordinary smoke tests cover
        # the router; this component probe observes the native HTTP data path.
        client = RemoteInferenceClient(
            proxy_url=urls[0],
            server_urls=urls,
            data_parallel_size=1,
            model_name=str(options["model"]),
            tokenizer=tokenizer,
        )
        with policy_workers(cfg) as (dispatch, snapshot):
            dispatch.set_inference_engine_client(client)
            dispatch.init_weight_sync_state(client)
            await dispatch.save_weights_for_sampler()
            version = client.weight_version
            batch["action_log_probs"] = logprobs(dispatch, batch)
            batch["rollout_logprobs"] = batch["action_log_probs"].clone()
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "List short words."}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            await server.arm_probe.remote()
            task = asyncio.create_task(
                client.generate(
                    {
                        "prompts": None,
                        "mm_features": None,
                        "prompt_token_ids": [prompt],
                        "session_ids": ["owned-cache-probe"],
                        "sampling_params": {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_tokens": 128,
                            "min_tokens": 128,
                            "logprobs": 0,
                            "seed": 731,
                        },
                        "cache_salt": f"llenvs-probe-{version}",
                    },
                    model=client.model_name,
                )
            )
            await server.wait_probe.remote()
            assert not task.done(), "generation finished before optimizer/weight sync"
            report = train_step(dispatch, batch, snapshot, root)
            await dispatch.save_weights_for_sampler()
            assert client.weight_version == version + 1
            wire = await asyncio.wait_for(task, 120)
            trace = await server.probe_report.remote()
            check_continuity(trace, wire, clear_cache=clear_cache)
            # A cleared KV suffix must agree with updated-model teacher forcing.
            # A retained-KV suffix is NOT assumed to be a single-policy sample.
            ids = wire["response_ids"][0]
            exact = TrainingInputBatch(
                {
                    "sequences": torch.tensor([prompt + ids]),
                    "attention_mask": torch.ones(1, len(prompt) + len(ids), dtype=torch.long),
                    "loss_mask": torch.ones(1, len(ids)),
                    "response_mask": torch.ones(1, len(ids)),
                }
            )
            exact.metadata = {"response_length": len(ids)}
            updated = logprobs(dispatch, exact)[0]
            boundary = len(trace["prefix"]["ids"])
            error = (updated[boundary:] - torch.tensor(trace["final"]["logprobs"][boundary:])).abs()
            report.update(
                config=get_config_as_dict(cfg),
                trace=trace,
                wire=wire,
                version_before=version,
                version_after=client.weight_version,
                suffix_max_abs=error.max().item(),
                clear_cache=clear_cache,
            )
            return report
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if client is not None:
            # No router in this component fixture, hence no session lease to
            # release through the router-only /finish_session endpoint.
            if client._session is not None:
                await client._session.close()
        group.shutdown()


def main():
    if os.environ.get("LLENVS_SKYRL_GPU") != "1" or not __debug__:
        raise RuntimeError("native probes require explicit GPU opt-in and enabled assertions")
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("dp1", "dp2", "cache_keep", "cache_clear"))
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--estimator", choices=("llenvs_turn_grpo", "llenvs_token_rtg"), default="llenvs_token_rtg"
    )
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    if args.scenario.startswith("cache") and (args.packed or args.estimator != "llenvs_token_rtg"):
        parser.error("cache probes use the fixed unpacked token-RTG recipe")
    assert args.directory.is_absolute() and args.directory.is_dir()
    os.environ.update(RAY_ADDRESS="local", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from tests.skyrl_acceptance import acceptance_runtime

    options = acceptance_runtime()
    import ray
    import torch

    assert torch.cuda.device_count() >= 2, "native probes require two allocated GPUs"
    assert not ray.is_initialized(), "never attach a probe to an existing job"
    report = {
        "completed": False,
        "identity": options["identity"],
        "thresholds": options["thresholds"],
        "scenario": args.scenario,
    }
    try:
        ray.init(address="local")
        if args.scenario.startswith("dp"):
            result = distributed_probe(
                options,
                args.directory,
                estimator=args.estimator,
                dp=int(args.scenario[-1]),
                packed=args.packed,
            )
        else:
            result = asyncio.run(
                cache_probe(options, args.directory, clear_cache=args.scenario == "cache_clear")
            )
        report.update(result, completed=True)
    finally:
        try:
            with (args.directory / f"{args.scenario}.json").open("x") as stream:
                json.dump(report, stream, allow_nan=False, indent=2)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
