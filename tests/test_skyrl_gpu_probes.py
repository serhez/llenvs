"""CPU checks of the GPU probes; these do not certify collectives or KV caches."""

import asyncio
import copy
from types import SimpleNamespace as N

import pytest

from tests import test_skyrl_tokenizer as tokenizer_tests
from tests.skyrl_gpu_probes import GenerationProbe, check_continuity, vector_error

qwen_tokenizer = tokenizer_tests.qwen_tokenizer


def state(ids=(10,), *, maximum=8):
    return N(
        external_req_id="request",
        max_tokens_param=maximum,
        detokenizer=N(output_token_ids=list(ids)),
        logprobs_processor=N(logprobs=[{i: N(logprob=-0.25)} for i in ids]),
    )


def test_probe_observes_partial_output_and_real_pause_result():
    async def run():
        current = state()
        events = []

        async def pause(**kwargs):
            events.append(kwargs)

        async def paused():
            return True

        engine = N(
            pause_generation=pause,
            is_paused=paused,
            output_processor=N(request_states={"internal": current}),
        )
        probe = GenerationProbe(engine)
        probe.arm()
        probe.observe(current, finished=False)
        prefix = await probe.wait_paused()
        assert events == [{"mode": "keep", "clear_cache": False}]
        assert prefix == {"request_id": "request", "ids": [10], "logprobs": [-0.25]}
        current.detokenizer.output_token_ids.append(11)
        assert prefix["ids"] == [10], "the audit must own its token snapshot"

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["finished", "missing", "not_paused", "pause_error"])
def test_probe_cannot_confuse_inflight_http_with_partial_generation(failure):
    async def run():
        current = state()

        async def pause(**kwargs):
            if failure == "pause_error":
                raise RuntimeError("pause failed")
            if failure == "finished":
                current.detokenizer.output_token_ids *= 8
                current.logprobs_processor.logprobs *= 8

        async def paused():
            return failure != "not_paused"

        engine = N(
            pause_generation=pause,
            is_paused=paused,
            output_processor=N(request_states={} if failure == "missing" else {0: current}),
        )
        probe = GenerationProbe(engine)
        probe.arm()
        probe.observe(current, finished=False)
        with pytest.raises((AssertionError, RuntimeError)):
            await probe.wait_paused()

    asyncio.run(run())


def valid_trace():
    prefix = {"request_id": "request", "ids": [10], "logprobs": [-0.25]}
    final = {"request_id": "request", "ids": [10, 11], "logprobs": [-0.25, -0.75]}
    return {
        "prefix": prefix,
        "final": final,
        "events": [
            {"event": "paused"},
            {"event": "transfer"},
            {"event": "reset", "running": True, "success": True},
            {"event": "resume"},
            {"event": "finished"},
        ],
    }, {
        "response_ids": [[10, 11]],
        "response_logprobs": [[-0.25, -0.75]],
        "stop_reasons": ["length"],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "prefix_id",
        "prefix_lp",
        "wire_id",
        "wire_lp",
        "nonfinite",
        "no_suffix",
        "abort",
        "request",
        "no_transfer",
        "late_reset",
        "reset_false",
        "not_running",
        "unexpected_reset",
    ],
)
def test_continuity_negative_controls(mutation):
    trace, wire = valid_trace()
    if mutation == "prefix_id":
        trace["prefix"]["ids"][0] += 1
    if mutation == "prefix_lp":
        trace["prefix"]["logprobs"][0] -= 1
    if mutation == "wire_id":
        wire["response_ids"][0][1] += 1
    if mutation == "wire_lp":
        wire["response_logprobs"][0][1] -= 1
    if mutation == "nonfinite":
        trace["final"]["logprobs"][1] = float("nan")
    if mutation == "no_suffix":
        trace["final"] = copy.deepcopy(trace["prefix"])
    if mutation == "abort":
        wire["stop_reasons"][0] = "abort"
    if mutation == "request":
        trace["final"]["request_id"] = "other"
    if mutation == "no_transfer":
        trace["events"].pop(1)
    if mutation == "late_reset":
        trace["events"].append(trace["events"].pop(2))
    if mutation == "reset_false":
        trace["events"][2]["success"] = False
    if mutation == "not_running":
        trace["events"][2]["running"] = False
    if mutation == "none":
        check_continuity(trace, wire, clear_cache=True)
    else:
        with pytest.raises(AssertionError):
            check_continuity(trace, wire, clear_cache=mutation != "unexpected_reset")


@pytest.mark.parametrize("mutation", ["none", "scale", "missing", "shape", "nan", "zero"])
def test_distributed_comparator_rejects_wrong_scaling_and_invalid_vectors(mutation):
    torch = pytest.importorskip("torch")
    reference = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([-3.0])}
    actual = copy.deepcopy(reference)
    if mutation == "scale":
        actual["a"] *= 2
    if mutation == "missing":
        del actual["b"]
    if mutation == "shape":
        actual["a"] = actual["a"].reshape(1, 2)
    if mutation == "nan":
        actual["a"][0] = float("nan")
    if mutation == "zero":
        reference = {k: torch.zeros_like(v) for k, v in reference.items()}
    if mutation in ("none", "scale"):
        assert (vector_error(actual, reference) == 0) == (mutation == "none")
    else:
        with pytest.raises(AssertionError):
            vector_error(actual, reference)


def test_native_step_observer_order_and_no_optimizer_after_invalid_gradients(tmp_path):
    from tests.skyrl_distributed_probe import train_step

    events = []
    batch = object()

    def forward(model, data, **kwargs):
        assert model == "policy" and data is batch
        assert kwargs == {"return_per_token_outputs": False}
        events.append("backward")
        return N(metrics={"loss": 0.5})

    def optim(model):
        events.append("optimizer")
        return 1.0

    def snapshot(phase, directory):
        assert directory == tmp_path
        events.append(phase)
        return "hash" if phase == "initial" else 1.0

    dispatch = N(forward_backward=forward, optim_step=optim)
    assert train_step(dispatch, batch, snapshot, tmp_path)["initial"] == "hash"
    assert events == ["initial", "backward", "gradient", "optimizer", "update"]
    events.clear()

    def bad_snapshot(phase, directory):
        if phase == "gradient":
            raise AssertionError("non-finite gradient")
        return snapshot(phase, directory)

    with pytest.raises(AssertionError, match="non-finite"):
        train_step(dispatch, batch, bad_snapshot, tmp_path)
    assert "optimizer" not in events


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "batch",
        "initial",
        "identity",
        "zero_update",
        "no_dummy",
        "forward",
        "grad_scale",
        "update_scale",
    ],
)
def test_exact_gpu_distributed_acceptance_checker(tmp_path, damage):
    torch = pytest.importorskip("torch")
    from tests.skyrl_gpu_probes import check_distributed_reports

    reports = []
    for dp in (1, 2):
        directory = tmp_path / f"dp{dp}"
        directory.mkdir()
        for phase in ("gradient", "update"):
            torch.save({"weight": torch.tensor([0.25, -0.5])}, directory / f"{phase}.pt")
        reports.append(
            dict(
                dp=dp,
                completed=True,
                identity={"model": "fixture"},
                initial="same",
                batch_hash="same",
                packed=True,
                forward_max_abs=0.0,
                gradient_norm=1.0,
                update_norm=1.0,
                clip_grad_norm=1.0,
                metrics={"num_padding_microbatches": 0.5},
            )
        )
    other = reports[1]
    if damage == "batch":
        other["batch_hash"] = "different"
    if damage == "initial":
        other["initial"] = "different"
    if damage == "identity":
        other["identity"] = {}
    if damage == "zero_update":
        other["update_norm"] = 0
    if damage == "no_dummy":
        other["metrics"]["num_padding_microbatches"] = 0
    if damage == "forward":
        other["forward_max_abs"] = 100
    if damage in ("grad_scale", "update_scale"):
        phase = "gradient" if damage == "grad_scale" else "update"
        torch.save({"weight": torch.tensor([0.5, -1.0])}, tmp_path / f"dp2/{phase}.pt")
    bounds = dict(logprob_max_abs=0.01, gradient_relative_l2=0.01, update_relative_l2=0.01)
    if damage == "none":
        assert check_distributed_reports(tmp_path, reports, bounds) == {"gradient": 0, "update": 0}
    else:
        with pytest.raises(AssertionError):
            check_distributed_reports(tmp_path, reports, bounds)


@pytest.mark.parametrize("scenario", ["dp1", "dp2", "cache_keep", "cache_clear"])
def test_gpu_probe_process_uses_owned_harness(monkeypatch, tmp_path, scenario):
    from tests import skyrl_acceptance_checks as checks

    calls = []
    monkeypatch.setattr(checks, "_owned_process", lambda command, log: calls.append((command, log)))
    (tmp_path / f"{scenario}.json").write_text('{"completed": true}')
    packed = scenario.startswith("dp")
    assert checks.native_probe_process(tmp_path, scenario, packed=packed) == {"completed": True}
    command, log = calls[0]
    assert command[1:] == [
        "-m",
        "tests.skyrl_native_probe_run",
        scenario,
        str(tmp_path),
        "--estimator",
        "llenvs_token_rtg",
    ] + (["--packed"] if packed else [])
    assert log == tmp_path / f"{scenario}.log"


@pytest.mark.parametrize(
    "dp,packed,clear", [(1, False, None), (2, True, None), (1, False, False), (1, False, True)]
)
def test_probe_recipe_through_actual_native_config_bodies(monkeypatch, tmp_path, dp, packed, clear):
    import json

    from tests.skyrl_native_probe_run import worker_recipe
    from tests.skyrl_source import config_namespace

    ns = config_namespace(monkeypatch)
    cfg = ns["build_nested_dataclass"](
        ns["SkyRLTrainConfig"],
        worker_recipe(
            tmp_path / "model", tmp_path, dp=dp, budget=128 if packed else 0, clear_cache=clear
        ),
    )
    assert cfg.trainer.strategy == "fsdp"
    assert cfg.trainer.placement.policy_num_gpus_per_node == dp
    assert cfg.trainer.remove_microbatch_padding is packed
    assert cfg.trainer.fully_async.enabled is (clear is not None)
    assert cfg.trainer.fully_async.clear_kv_cache_on_weight_sync is bool(clear)
    assert cfg.trainer.policy.optimizer_config.num_warmup_steps == 0
    assert not cfg.trainer.algorithm.use_kl_loss and not cfg.trainer.algorithm.use_entropy_loss
    json.dumps(ns["get_config_as_dict"](cfg), allow_nan=False)


@pytest.mark.parametrize("estimator", ["llenvs_turn_grpo", "llenvs_token_rtg"])
def test_gpu_batch_fixture_with_native_cpu_tensors_and_cached_tokenizer(
    monkeypatch, qwen_tokenizer, estimator
):
    import sys
    from types import ModuleType

    from tests.skyrl_native_probe_run import fixture_batch
    from tests.skyrl_source import definitions, worker_namespace

    ns = worker_namespace(monkeypatch)
    definitions(
        "skyrl/backends/skyrl_train/utils/ppo_utils.py", {"compute_grpo_outcome_advantage"}, ns
    )
    # Execute native math/batch bodies with only import boundaries substituted.
    # No model, Ray, OmegaConf, FA2 or collective is represented as installed.
    for module_name, names in {
        "skyrl.backends.skyrl_train.training_batch": ["TrainingInputBatch"],
        "skyrl.backends.skyrl_train.utils.ppo_utils": [
            "compute_grpo_outcome_advantage",
            "apply_loss_reduction_to_advantages_minibatch",
        ],
        "skyrl.train.dataset.preprocess": ["convert_prompts_responses_to_batch_tensors"],
    }.items():
        module = ModuleType(module_name)
        vars(module).update({name: ns[name] for name in names})
        monkeypatch.setitem(sys.modules, module_name, module)
    tokenizer, vocabulary = qwen_tokenizer
    options = {"model": tokenizer.name_or_path, "identity": {"model": {"vocab_size": vocabulary}}}
    batch, trace, _, budget = fixture_batch(options, estimator)
    counts = [
        len(
            ns["get_microbatch_iterator"](
                batch[start : start + 2], micro_batch_size=1, max_tokens_per_microbatch=budget
            )
        )
        for start in (0, 2)
    ]
    assert counts == [1, 2], "DP2 must need a balancing dummy, not just another partition"
    assert batch.metadata == {"response_length": batch["loss_mask"].shape[1]}
    for i, row_id in enumerate(batch["row_ids"].tolist()):
        ids = batch["sequences"][i][batch["attention_mask"][i].bool()].tolist()
        assert ids == trace["prompts"][row_id] + trace["responses"][row_id]
        n = len(trace["responses"][row_id])
        # Native response windows are right-aligned within the padded batch.
        response = batch["response_mask"][i].bool()
        assert int(response.sum()) == n
        assert batch["loss_mask"][i][response].tolist() == trace["loss_masks"][row_id]


def test_parameter_audit_measures_actual_updates_not_parameter_magnitudes(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from tests.skyrl_distributed_probe import ParameterAudit

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    worker = ParameterAudit()
    worker.model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        worker.model.weight.fill_(10.0)
    fingerprint = worker.audit_snapshot("initial", str(tmp_path))
    assert len(fingerprint) == 64
    worker.model.weight.grad = torch.ones_like(worker.model.weight)
    assert worker.audit_snapshot("gradient", str(tmp_path)) > 0
    with torch.no_grad():
        worker.model.weight.add_(0.5)
    worker.audit_snapshot("update", str(tmp_path))
    delta = torch.load(tmp_path / "update.pt", weights_only=True)
    assert torch.equal(delta["weight"], torch.full((1, 2), 0.5))
    assert not worker._audit_initial


def test_server_observation_hooks_forward_original_operations_unchanged(monkeypatch):
    import sys
    from types import ModuleType

    from tests.skyrl_gpu_probes import observed_server_class

    calls, sentinel = [], object()

    class Server:
        def __init__(self, *args, **kwargs):
            calls.append(("constructor", args, kwargs))

        @staticmethod
        def _add_custom_endpoints(*args):
            calls.append(("endpoints", args))

    class RequestState:
        def make_request_output(self, *args, **kwargs):
            calls.append(("output", args, kwargs))
            return sentinel

    for module_name, name, value in [
        (
            "skyrl.backends.skyrl_train.inference_servers.vllm_server_actor",
            "VLLMServerActor",
            Server,
        ),
        ("vllm.v1.engine.output_processor", "RequestState", RequestState),
    ]:
        module = ModuleType(module_name)
        setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    async def run():
        current = RequestState()
        vars(current).update(vars(state(maximum=2)))

        async def pause(**kwargs):
            calls.append(("pause", kwargs))

        async def paused():
            return True

        async def reset(*args, **kwargs):
            calls.append(("reset", args, kwargs))
            return True

        async def resume():
            calls.append(("resume",))

        async def rpc(*args, **kwargs):
            calls.append(("rpc", args, kwargs))
            return sentinel

        engine = N(
            pause_generation=pause,
            is_paused=paused,
            reset_prefix_cache=reset,
            resume_generation=resume,
            collective_rpc=rpc,
            output_processor=N(request_states={"internal": current}),
        )
        server = observed_server_class()("argument", setting=True)
        Server._add_custom_endpoints("app", engine, "cli")
        server.arm_probe()
        assert current.make_request_output([10], None, None) is sentinel
        await server.wait_probe()
        assert await engine.collective_rpc("update_weights_nccl", kwargs={"chunk": 1}) is sentinel
        assert await engine.reset_prefix_cache(reset_running_requests=True) is True
        await engine.resume_generation()
        current.detokenizer.output_token_ids.append(11)
        current.logprobs_processor.logprobs.append({11: N(logprob=-0.75)})
        assert current.make_request_output([11], None, "length") is sentinel
        _, wire = valid_trace()
        check_continuity(server.probe_report(), wire, clear_cache=True)
        assert calls[0] == ("constructor", ("argument",), {"setting": True})
        assert ("rpc", ("update_weights_nccl",), {"kwargs": {"chunk": 1}}) in calls
        assert [c[1] for c in calls if c[0] == "output"] == [
            ([10], None, None),
            ([11], None, "length"),
        ]

    asyncio.run(run())
