"""Test-only observations and comparisons; imports allocate no runtime or model."""

import asyncio
import copy
import math


def token_snapshot(state):
    ids = list(state.detokenizer.output_token_ids)
    entries = state.logprobs_processor.logprobs
    assert len(ids) == len(entries), "sampled token/logprob alignment changed"
    logprobs = [entry[token].logprob for token, entry in zip(ids, entries, strict=True)]
    assert all(math.isfinite(p) for p in logprobs), "non-finite sampled logprob"
    return {"request_id": state.external_req_id, "ids": ids, "logprobs": logprobs}


class GenerationProbe:
    """Pause after observed decode, without changing sampling or output delivery.

    The server-side output processor calls observe synchronously. The pause runs
    on its event loop, then requires the SAME request to remain registered with
    a nonempty, incomplete output. No timer is accepted as decode evidence.
    """

    def __init__(self, engine):
        self.engine = engine
        self.armed = False

    def arm(self):
        assert not self.armed
        self.armed = True
        self.events = []
        self.ready = asyncio.Event()
        self.pause_task = None
        self.error = None
        self.target = self.prefix = self.final = None

    def observe(self, state, *, finished):
        if not self.armed:
            return
        snapshot = token_snapshot(state)
        if self.target is None:
            self.target = snapshot["request_id"]
        assert snapshot["request_id"] == self.target, "probe requires one generation request"
        if finished:
            self.final = snapshot
            self.events.append({"event": "finished"})
        elif snapshot["ids"] and self.pause_task is None:
            self.pause_task = asyncio.create_task(self._pause())

    async def _pause(self):
        try:
            await self.engine.pause_generation(mode="keep", clear_cache=False)
            assert await self.engine.is_paused(), "pause did not freeze the scheduler"
            states = [
                s
                for s in self.engine.output_processor.request_states.values()
                if s.external_req_id == self.target
            ]
            assert len(states) == 1, "request completed before the pause"
            self.prefix = token_snapshot(states[0])
            assert 0 < len(self.prefix["ids"]) < states[0].max_tokens_param
            assert self.final is None, "generation finished before weight sync"
            self.events.append({"event": "paused"})
        except BaseException as error:
            self.error = error
        finally:
            self.ready.set()

    async def wait_paused(self):
        await asyncio.wait_for(self.ready.wait(), 60)
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.prefix)

    def report(self):
        return copy.deepcopy({"prefix": self.prefix, "final": self.final, "events": self.events})


def check_continuity(trace, wire, *, clear_cache):
    prefix, final = trace["prefix"], trace["final"]
    assert prefix and final and prefix["request_id"] == final["request_id"]
    boundary = len(prefix["ids"])
    assert 0 < boundary < len(final["ids"]), "no post-sync suffix"
    for record in (prefix, final):
        assert len(record["ids"]) == len(record["logprobs"])
        assert all(math.isfinite(p) for p in record["logprobs"])
    assert final["ids"][:boundary] == prefix["ids"]
    assert final["logprobs"][:boundary] == prefix["logprobs"]
    assert wire["response_ids"] == [final["ids"]]
    assert wire["response_logprobs"] == [final["logprobs"]]
    assert wire["stop_reasons"] == ["length"]
    events = trace["events"]
    names = [e["event"] for e in events]
    assert names.count("paused") == names.count("resume") == names.count("finished") == 1
    paused, resumed, finished = map(names.index, ("paused", "resume", "finished"))
    transfers = [i for i, name in enumerate(names) if name == "transfer"]
    assert transfers and all(paused < i < resumed < finished for i in transfers)
    resets = [(i, e) for i, e in enumerate(events) if e["event"] == "reset"]
    assert len(resets) == int(clear_cache)
    for index, reset in resets:
        assert paused < index < resumed and reset["running"] is True
        assert reset["success"] is True, "HTTP success alone is not cache-reset success"


def vector_error(actual, reference):
    """Full named-vector relative L2, rejecting missing, non-finite or zero evidence."""
    import torch

    assert actual and actual.keys() == reference.keys()
    error = norm = 0.0
    for name, target in reference.items():
        current = actual[name]
        assert current.shape == target.shape and current.dtype == target.dtype, name
        assert bool(torch.isfinite(current).all()) and bool(torch.isfinite(target).all()), name
        delta = current.double() - target.double()
        error += delta.square().sum().item()
        norm += target.double().square().sum().item()
    assert norm > 0 and math.isfinite(error) and math.isfinite(norm)
    return math.sqrt(error / norm)


def check_distributed_reports(root, reports, bounds):
    import torch

    assert [r["dp"] for r in reports] == [1, 2]
    assert reports[0]["identity"] == reports[1]["identity"]
    assert reports[0]["initial"] == reports[1]["initial"], "initial parameters differ"
    assert reports[0]["batch_hash"] == reports[1]["batch_hash"], "global training batch differs"
    assert reports[0]["packed"] == reports[1]["packed"]
    for report in reports:
        assert report["completed"] is True
        for key in ("gradient_norm", "update_norm", "clip_grad_norm"):
            assert math.isfinite(report[key]) and report[key] > 0
        assert math.isfinite(report["forward_max_abs"])
        assert report["forward_max_abs"] <= bounds["logprob_max_abs"]
    if reports[1]["packed"]:
        assert reports[1]["metrics"]["num_padding_microbatches"] > 0
    result = {}
    for phase in ("gradient", "update"):
        # Locally generated owned tensor-only evidence; never arbitrary pickle.
        reference = torch.load(root / f"dp1/{phase}.pt", weights_only=True, map_location="cpu")
        actual = torch.load(root / f"dp2/{phase}.pt", weights_only=True, map_location="cpu")
        result[phase] = vector_error(actual, reference)
        del actual, reference
        assert result[phase] <= bounds[f"{phase}_relative_l2"], (phase, result[phase], root)
    return result


def observed_server_class():
    """Install observations only in the owned native server actor process.

    Native HTTP handlers, output kinds, sampled IDs/LPs and cache operations are
    unchanged. The extra initial KEEP pause is the test's deterministic barrier.
    Private vLLM observations are version-specific and fail closed on drift.
    """
    from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import VLLMServerActor

    class ObservedServer(VLLMServerActor):
        probe: GenerationProbe

        def __init__(self, *args, **kwargs):
            from vllm.v1.engine.output_processor import RequestState

            super().__init__(*args, **kwargs)
            original_add = VLLMServerActor._add_custom_endpoints
            original_output = RequestState.make_request_output

            def add(app, engine, cli_args):
                original_add(app, engine, cli_args)
                self.probe = probe = GenerationProbe(engine)

                def output(state, new_token_ids, pooling_output, finish_reason, *args, **kwargs):
                    probe.observe(state, finished=finish_reason is not None)
                    return original_output(
                        state, new_token_ids, pooling_output, finish_reason, *args, **kwargs
                    )

                RequestState.make_request_output = output
                original_reset = engine.reset_prefix_cache
                original_resume = engine.resume_generation
                original_rpc = engine.collective_rpc

                async def reset(*args, **kwargs):
                    result = await original_reset(*args, **kwargs)
                    if probe.armed:
                        probe.events.append(
                            {
                                "event": "reset",
                                "success": result,
                                "running": kwargs.get("reset_running_requests", False),
                            }
                        )
                    return result

                async def resume(*args, **kwargs):
                    # Record before awaiting: resumed output can arrive before
                    # the resume acknowledgement returns to the HTTP handler.
                    if probe.armed:
                        probe.events.append({"event": "resume"})
                    return await original_resume(*args, **kwargs)

                async def rpc(method, *args, **kwargs):
                    result = await original_rpc(method, *args, **kwargs)
                    if probe.armed and method == "update_weights_nccl":
                        probe.events.append({"event": "transfer"})
                    return result

                engine.reset_prefix_cache = reset
                engine.resume_generation = resume
                engine.collective_rpc = rpc

            setattr(VLLMServerActor, "_add_custom_endpoints", staticmethod(add))

        def arm_probe(self):
            self.probe.arm()

        async def wait_probe(self):
            return await self.probe.wait_paused()

        def probe_report(self):
            return self.probe.report()

    return ObservedServer
