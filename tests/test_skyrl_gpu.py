"""Opt-in native GPU acceptance, with local assets and reviewed numeric bounds.

No imports of SkyRL/vLLM, model allocation or download on ordinary collection.
Layout tests isolate policy computation; they do not certify FSDP collectives,
async caches, auxiliary losses, or VLM. Training smoke tests are separate below.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from tests.skyrl_acceptance import acceptance_runtime
from tests.skyrl_acceptance_checks import (
    check_smoke_progress,
    native_probe_process,
    training_process,
)
from tests.skyrl_layout_fixture import layout_batch as _layout_batch

pytestmark = pytest.mark.skyrl_gpu
if os.environ.get("LLENVS_SKYRL_GPU") != "1":
    pytest.skip(
        "set LLENVS_SKYRL_GPU=1 plus explicit local assets/bounds to run GPU acceptance",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def acceptance():
    # Explicitly requested acceptance must fail, not skip, on missing/broken
    # dependencies, wrong runtime identity, unsupported hardware or assets.
    return acceptance_runtime()


def _delta(actual, expected):
    import torch

    error = (actual.detach().cpu().double() - expected.detach().cpu().double()).abs().flatten()
    assert error.numel() and bool(torch.isfinite(error).all())
    return {
        "max_abs": error.max().item(),
        "p99_abs": error.quantile(0.99).item(),
        "median_abs": error.median().item(),
    }


def _gradient_delta(model, reference):
    error, norm = 0.0, 0.0
    seen = set()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            assert name not in reference, f"lost gradient: {name}"
            continue
        seen.add(name)
        assert name in reference, f"unexpected gradient: {name}"
        # Keep only one parameter-sized FP32 CPU comparison live at a time.
        current = parameter.grad.detach().float().cpu()
        target = reference[name].float()
        error += (current - target).square().double().sum().item()
        norm += target.square().double().sum().item()
    assert seen == reference.keys() and norm > 0 and math.isfinite(error)
    return math.sqrt(error / norm)


@pytest.mark.parametrize("estimator", ["llenvs_turn_grpo", "llenvs_token_rtg"])
@pytest.mark.parametrize("packed", [False, True])
def test_native_dp1_dp2_same_batch_gradients_and_updates(acceptance, estimator, packed):
    from tests.skyrl_gpu_probes import check_distributed_reports

    artifact = Path(
        tempfile.mkdtemp(prefix=f"dp-{estimator}-{packed}-", dir=acceptance["artifacts"])
    )
    reports = [
        native_probe_process(artifact, f"dp{dp}", estimator=estimator, packed=packed)
        for dp in (1, 2)
    ]
    comparisons = check_distributed_reports(artifact, reports, acceptance["thresholds"]["bounds"])
    with (artifact / "comparison.json").open("x") as stream:
        json.dump(comparisons, stream, allow_nan=False, indent=2)


@pytest.mark.parametrize("clear_cache", [False, True])
def test_native_weight_change_during_generation(acceptance, clear_cache):
    from tests.skyrl_gpu_probes import check_continuity

    artifact = Path(tempfile.mkdtemp(prefix=f"cache-{clear_cache}-", dir=acceptance["artifacts"]))
    result = native_probe_process(artifact, "cache_clear" if clear_cache else "cache_keep")
    check_continuity(result["trace"], result["wire"], clear_cache=clear_cache)
    assert result["update_norm"] > 0 and math.isfinite(result["update_norm"])
    assert result["version_after"] == result["version_before"] + 1
    assert math.isfinite(result["suffix_max_abs"])
    if clear_cache:
        assert (
            result["suffix_max_abs"]
            <= acceptance["thresholds"]["bounds"]["rollout_logprob_max_abs"]
        ), artifact


LAYOUT_CASES = [
    ("grpo", "token_mean", "regular", None, "uniform"),
    ("llenvs_turn_grpo", "sequence_mean", "regular", None, "uniform"),
    ("llenvs_token_rtg", "token_mean", "regular", None, "uniform"),
    ("llenvs_turn_grpo", "prompt_mean", "regular", "token", "span_normalized"),
    ("llenvs_token_rtg", "seq_mean_token_sum_norm", "rollout_is", None, "uniform"),
]


@pytest.mark.parametrize(("estimator", "reduction", "loss_type", "tis", "weighting"), LAYOUT_CASES)
def test_native_packing_microbatch_gradients_and_attention_isolation(
    acceptance, estimator, reduction, loss_type, tis, weighting
):
    import torch
    from skyrl.backends.skyrl_train.training_batch import TensorBatch
    from skyrl.backends.skyrl_train.utils.ppo_utils import (
        PolicyLossRegistry,
        apply_loss_reduction_to_advantages_minibatch,
    )
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.backends.skyrl_train.workers.worker_utils import get_microbatch_iterator
    from skyrl.train.config import AlgorithmConfig
    from transformers import AutoTokenizer

    artifact = Path(
        tempfile.mkdtemp(prefix=f"layout-{estimator}-{reduction}-", dir=acceptance["artifacts"])
    )
    tokenizer = AutoTokenizer.from_pretrained(
        acceptance["model"], local_files_only=True, trust_remote_code=False
    )
    wrapper = None
    report = {
        "identity": acceptance["identity"],
        "thresholds": acceptance["thresholds"],
        "case": [estimator, reduction, loss_type, tis, weighting],
        "comparisons": {},
    }
    try:
        wrapper = HFModelWrapper(
            str(acceptance["model"]),
            use_flash_attention_2=True,
            bf16=True,
            remove_microbatch_padding=False,
            use_torch_compile=False,
        ).to("cuda")
        wrapper.model.set_attn_implementation("eager")
        wrapper.attn_implementation = "eager"
        batch, trace = _layout_batch(
            tokenizer, acceptance["identity"]["model"]["vocab_size"], estimator, weighting
        )
        report["trace"] = trace
        batch.to("cuda")
        wrapper.eval()
        # Independent single-row forwards are the fixed-weight old-policy
        # reference. No rollout probability is invented for inference parity.
        with torch.no_grad():
            old = torch.cat(
                [
                    wrapper(
                        batch["sequences"][i : i + 1],
                        batch.metadata["response_length"],
                        batch["attention_mask"][i : i + 1],
                    )
                    for i in range(4)
                ]
            )
        batch["action_log_probs"] = old.detach()
        # A controlled loss-only behavior distribution exercises the TIS cap;
        # real rollout/train probability comparisons belong to the smoke test.
        batch["rollout_logprobs"] = old.detach() - (math.log(3) if tis else 0)
        batch["advantages"] = apply_loss_reduction_to_advantages_minibatch(
            batch["advantages"],
            batch["loss_mask"],
            reduction,
            micro_batch_size=1,
            max_seq_len=13,
            prompt_boundaries=[(0, 2), (2, 4)],
        )
        algorithm = AlgorithmConfig()
        algorithm.policy_loss_type = loss_type
        algorithm.off_policy_correction.tis_ratio_type = tis
        algorithm.off_policy_correction.token_tis_ratio_clip_high = 2.0
        loss_function = PolicyLossRegistry.get(loss_type)

        def evaluate(*, packing, count, budget, checkpointing=False):
            wrapper.remove_microbatch_padding = packing
            if checkpointing:
                wrapper.gradient_checkpointing_enable({"use_reentrant": False})
            else:
                wrapper.gradient_checkpointing_disable()
            wrapper.train()  # Dense Qwen dropout is checked to be zero.
            wrapper.zero_grad(set_to_none=True)
            iterator = get_microbatch_iterator(
                batch, micro_batch_size=count, max_tokens_per_microbatch=budget
            )
            outputs = []
            for microbatch in iterator:
                assert not microbatch.metadata.get("is_padding_batch", False), (
                    "single-device diagnostic unexpectedly needs DP padding"
                )
                logprobs = wrapper(
                    microbatch["sequences"],
                    batch.metadata["response_length"],
                    microbatch["attention_mask"],
                )
                loss, _ = loss_function(
                    logprobs,
                    microbatch["action_log_probs"],
                    microbatch["advantages"],
                    algorithm,
                    loss_mask=microbatch["loss_mask"],
                    rollout_logprobs=microbatch["rollout_logprobs"],
                )
                assert bool(torch.isfinite(loss))
                loss.backward()
                outputs.append(
                    TensorBatch(
                        {
                            "logprobs": logprobs.detach(),
                            "row_ids": microbatch["row_ids"],
                            "advantages": microbatch["advantages"],
                        }
                    )
                )
            result = iterator.reorder_and_combine_batches(outputs)
            assert torch.equal(result["row_ids"], batch["row_ids"])
            assert torch.equal(result["advantages"], batch["advantages"])
            return result["logprobs"].cpu()

        baseline = evaluate(packing=False, count=1, budget=-1)
        mask = batch["loss_mask"].bool().cpu()
        report["old_forward_train_forward"] = _delta(baseline[mask], old.cpu()[mask])
        reference = {
            name: p.grad.detach().cpu().clone()
            for name, p in wrapper.named_parameters()
            if p.grad is not None
        }
        wrapper.model.set_attn_implementation("flash_attention_2")
        wrapper.attn_implementation = "flash_attention_2"
        maximum = int(batch["attention_mask"].sum(-1).max().item())
        for label, count, budget, checkpointing in [
            ("packed", 4, -1, False),
            ("count_microbatches", 2, -1, False),
            ("token_bins", 1, maximum, False),
            ("checkpoint_recompute", 2, -1, True),
        ]:
            actual = evaluate(packing=True, count=count, budget=budget, checkpointing=checkpointing)
            mask = batch["loss_mask"].bool().cpu()
            report["comparisons"][label] = _delta(actual[mask], baseline[mask]) | {
                "gradient_relative_l2": _gradient_delta(wrapper, reference)
            }
        del reference
        wrapper.eval()
        wrapper.gradient_checkpointing_disable()
        changed = batch["sequences"].clone()
        changed[1:] = torch.where(batch["attention_mask"][1:].bool(), 42, changed[1:])
        with torch.no_grad():
            perturbed = wrapper(
                changed, batch.metadata["response_length"], batch["attention_mask"]
            ).cpu()
        mask = batch["loss_mask"][0].bool().cpu()
        report["attention_isolation"] = _delta(perturbed[0, mask], baseline[0, mask])
    finally:
        # Persist measurements before acceptance assertions; failed candidates
        # must remain inspectable, never auto-retune their tolerance file.
        (artifact / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        wrapper = None
        torch.cuda.empty_cache()
    bounds = acceptance["thresholds"]["bounds"]
    for values in report["comparisons"].values():
        assert values["max_abs"] <= bounds["logprob_max_abs"], str(artifact)
        assert values["gradient_relative_l2"] <= bounds["gradient_relative_l2"], str(artifact)
    assert report["attention_isolation"]["max_abs"] <= bounds["logprob_max_abs"], str(artifact)
    assert report["old_forward_train_forward"]["max_abs"] <= bounds["logprob_max_abs"], str(
        artifact
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("estimator", ["grpo", "llenvs_turn_grpo", "llenvs_token_rtg"])
def test_native_training_and_explicit_periodic_checkpoint_resume(acceptance, mode, estimator):
    import torch

    dp = int(os.environ.get("LLENVS_SKYRL_DP", "1"))
    assert dp in (1, 2), "acceptance admits LLENVS_SKYRL_DP=1 or 2"
    assert torch.cuda.device_count() >= dp + (mode == "async"), (
        "sync requires DP GPUs; async requires DP training GPUs plus one inference GPU"
    )
    artifact = Path(
        tempfile.mkdtemp(prefix=f"smoke-{mode}-{estimator}-dp{dp}-", dir=acceptance["artifacts"])
    )
    (artifact / "acceptance-identity.json").write_text(
        json.dumps(
            {
                "identity": acceptance["identity"],
                "thresholds": acceptance["thresholds"],
                "mode": mode,
                "estimator": estimator,
                "dp": dp,
            },
            indent=2,
        )
    )
    fresh = training_process(artifact, mode, estimator, dp, resume=False)
    check_smoke_progress(artifact, "fresh", fresh, [1, 2], asynchronous=mode == "async")
    if mode == "sync":
        # Before the FIRST update only: the inference engine and training
        # policy have the same weights. Later/async differences can be staleness.
        assert (
            fresh["batches"][0]["rollout_train_delta"]["max_abs"]
            <= acceptance["thresholds"]["bounds"]["rollout_logprob_max_abs"]
        ), str(artifact)
    resumed = training_process(artifact, mode, estimator, dp, resume=True)
    check_smoke_progress(artifact, "resumed", resumed, [2], asynchronous=mode == "async")
    assert set(resumed["batches"][0]["uids"]) == set(fresh["batches"][1]["uids"])
    assert set(resumed["batches"][0]["uids"]).isdisjoint(fresh["batches"][0]["uids"])
    if estimator != "grpo":

        def occurrences(result):
            return {
                d["llenvs/attribution"]["instance_id"]
                for batch in result["batches"]
                for d in batch["diagnostics"]
            }

        assert occurrences(fresh).isdisjoint(occurrences(resumed))
    manifests = [
        json.loads((artifact / phase / "checkpoints/llenvs-run.json").read_text())
        for phase in ("fresh", "resumed")
    ]
    assert manifests[0] == manifests[1], str(artifact)
