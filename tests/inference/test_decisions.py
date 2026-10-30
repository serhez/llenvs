"""Decision responses must contain the complete requested distribution."""

import asyncio
import math
from types import SimpleNamespace

import pytest

from llenvs.inference.decisions import DecisionOption, DecisionRequest, normalize_probabilities
from llenvs.inference.decisions.d3 import D3Backend
from llenvs.inference.decisions.openai import OpenAIDecisionBackend
from llenvs.inference.protocol import RefusedByPolicyError


def request():
    return DecisionRequest(
        "context", "Which range?", (DecisionOption("b", "Upper"), DecisionOption("a", "Lower"))
    )


def test_normalization_restores_requested_order_and_retains_original_mass():
    probabilities, metadata = normalize_probabilities(request(), [("a", 0.2), ("b", 0.6)])
    assert probabilities == pytest.approx((0.75, 0.25))
    assert metadata["raw_probabilities"] == [0.6, 0.2]
    assert metadata["probability_sum"] == pytest.approx(0.8)
    assert metadata["probability_sum_deviation"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "entries",
    [
        [("a", 1)],
        [("a", 0.5), ("a", 0.5)],
        [("a", 0.5), ("c", 0.5)],
        [("a", -1), ("b", 2)],
        [("a", math.nan), ("b", 1)],
        [("a", math.inf), ("b", 1)],
        [("a", 0), ("b", 0)],
        [("a", True), ("b", 1)],
    ],
)
def test_invalid_distributions_raise(entries):
    with pytest.raises(ValueError):
        normalize_probabilities(request(), entries)


def test_openai_wire_contract_and_usage():
    backend = OpenAIDecisionBackend.__new__(OpenAIDecisionBackend)
    backend._model = "gpt-6-luna"
    body = backend._request_kwargs(request())
    assert body == {
        "model": "gpt-6-luna",
        "input": "context",
        "questions": [
            {
                "type": "choice",
                "name": "decision",
                "instructions": "Which range?",
                "choices": [
                    {"value": "b", "description": "Upper"},
                    {"value": "a", "description": "Lower"},
                ],
            }
        ],
    }
    result = backend.reparse_receipt(
        {
            "request": request(),
            "response": {
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "type": "choice",
                        "name": "decision",
                        "probabilities": [
                            {"value": "a", "probability": 0.25},
                            {"value": "b", "probability": 0.75},
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    )
    assert result.probabilities == (0.75, 0.25)
    assert result.prompt_tokens == 10 and result.completion_tokens == 0
    assert backend.capabilities.version_pinned is False


def test_openai_refusal_is_an_explicit_failure():
    backend = OpenAIDecisionBackend.__new__(OpenAIDecisionBackend)
    with pytest.raises(RefusedByPolicyError):
        backend.reparse_receipt(
            {
                "request": request(),
                "response": {
                    "model": "gpt-6-luna",
                    "answers": [{"type": "refusal", "name": "decision", "refusal": "No"}],
                },
            }
        )


def test_openai_refusal_keeps_the_attempts_it_was_billed_for():
    """A refused decision is billed for its input, so its error carries the
    attempts with their usage, as a parsed result's metadata does."""
    usage = {
        "input_tokens": 10,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 8},
    }

    async def create(**kwargs):
        return {
            "model": "gpt-6-luna",
            "answers": [{"type": "refusal", "name": "decision", "refusal": "No"}],
            "usage": usage,
        }

    backend = OpenAIDecisionBackend.__new__(OpenAIDecisionBackend)
    backend._model = "gpt-6-luna"
    backend._async_client = SimpleNamespace(decisions=SimpleNamespace(create=create))
    with pytest.raises(RefusedByPolicyError) as raised:
        asyncio.run(backend._decide(0, request(), None))
    (attempt,) = raised.value.attempts
    assert attempt["status_code"] == 200 and attempt["usage"] == usage


def test_d3_uses_systemone_contract_and_label_mapping():
    backend = D3Backend.__new__(D3Backend)
    backend._model = "vllm-sr/d3"
    body = backend._request_kwargs(request())
    assert body["state"] == "context"
    assert body["questions"]["decision"]["criteria"] == {"b": "Upper", "a": "Lower"}
    assert body["options"] == {"return_meta": True}
    result = backend.reparse_receipt(
        {
            "request": request(),
            "response": {
                "model": "vllm-sr/d3",
                "answers": {
                    "decision": {"type": "choice", "probabilities": {"a": 0.25, "b": 0.75}}
                },
                "usage": {"input_tokens": 8},
                "meta": {"precision": "bf16"},
            },
        }
    )
    assert result.probabilities == (0.75, 0.25)
    assert result.metadata["runtime_meta"] == {"precision": "bf16"}


def test_duplicate_option_labels_are_rejected():
    with pytest.raises(ValueError):
        DecisionRequest("context", "question", (DecisionOption("a"), DecisionOption("a")))


@pytest.fixture
def native_worker(monkeypatch, tmp_path):
    import atexit
    import copy
    import signal
    import subprocess

    import httpx

    import llenvs.inference.decisions.d3 as mod

    card = {
        "id": "vllm-sr/d3",
        "family": "decision3",
        "revision": mod.D3_REVISION,
        "ready": True,
        "model_sha256": "model-digest",
        "manifest_sha256": "manifest-digest",
        "question_types": ["choice"],
        "dtype": "bfloat16",
        "numerics": "approximate",
        "limits": {"min_options": 2, "max_options": 255, "max_input_tokens": 32768},
        "engine_args": {
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "batch_window_ms": 1,
            "dtype": "bfloat16",
            "enable_shared_context": False,
        },
        "golden": {"status": "matched"},
    }
    state = SimpleNamespace(
        card=card, launches=[], signals=[], health_replies=None, health_requests=0
    )
    exit_callbacks = []
    monkeypatch.setattr(atexit, "register", lambda callback: exit_callbacks.append(callback))
    monkeypatch.setattr(atexit, "unregister", lambda callback: None)

    class Process:
        pid = 987654
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if signal.SIGKILL not in state.signals:
                raise subprocess.TimeoutExpired("owned", timeout)
            self.returncode = -9

    process = Process()

    def launch(argv, **kwargs):
        state.launches.append((argv, kwargs))
        return process

    def killpg(pid, sig):
        assert pid == process.pid
        state.signals.append(sig)

    class Client:
        def __init__(self, **kwargs):
            pass

        def get(self, path):
            if path == "/health":
                index = state.health_requests
                state.health_requests += 1
                if state.health_replies:
                    response = state.health_replies[min(index, len(state.health_replies) - 1)]
                    if isinstance(response, Exception):
                        raise response
                    return response
            body = (
                {"status": "ready", "api_version": "3.0.0"}
                if path == "/health"
                else {"api_version": "3.0.0", "data": [state.card]}
            )
            return SimpleNamespace(
                status_code=200, json=lambda: copy.deepcopy(body), raise_for_status=lambda: None
            )

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, **kwargs):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(
        mod,
        "native_runtime_identity",
        lambda *args: {
            "source": {"vcs_info": {"commit_id": mod.RUNTIME_REVISION}},
            "content_sha256": "runtime-digest",
        },
    )
    monkeypatch.setattr(mod.subprocess, "Popen", launch)
    monkeypatch.setattr(mod.os, "killpg", killpg)
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    state.create = lambda **kwargs: D3Backend(
        log_path=tmp_path / "d3.log", shutdown_timeout=0.01, **kwargs
    )
    yield state
    for callback in exit_callbacks:
        callback()


def startup_clock(monkeypatch):
    import llenvs.inference.decisions.d3 as mod

    clock = SimpleNamespace(seconds=0)

    def advance(seconds):
        clock.seconds += seconds

    monkeypatch.setattr(
        mod, "time", SimpleNamespace(monotonic=lambda: clock.seconds, sleep=advance)
    )
    return clock


def native_health(status, reason=None, *, api_version="3.0.0"):
    import httpx

    body = {"api_version": api_version, "status": status, "reason": reason, "model": "d3"}
    return httpx.Response(
        200 if status == "ready" else 503,
        json=body,
        request=httpx.Request("GET", "http://127.0.0.1:12345/health"),
    )


@pytest.mark.parametrize(
    "status,reason",
    [("loading", "verifying the package"), ("warming", "running the golden check")],
)
def test_native_d3_startup_timeout_retains_last_native_health(
    native_worker, monkeypatch, status, reason
):
    import json
    import signal

    startup_clock(monkeypatch)
    reply = native_health(status, reason)
    native_worker.health_replies = [reply]
    with pytest.raises(TimeoutError) as caught:
        native_worker.create(startup_timeout=0.5)
    assert reason in str(caught.value)
    assert "d3.log" in str(caught.value)
    assert caught.value.startup_health == {"status_code": 503, "body": reply.json()}
    json.dumps(caught.value.startup_health, allow_nan=False)
    assert native_worker.signals == [signal.SIGTERM, signal.SIGKILL]


def test_native_d3_terminal_native_health_failure_raises_without_waiting_for_timeout(
    native_worker, monkeypatch
):
    import signal

    clock = startup_clock(monkeypatch)
    reply = native_health("failed", "PackageError: checkpoint digest differs")
    native_worker.health_replies = [reply]
    with pytest.raises(RuntimeError, match="checkpoint digest differs") as caught:
        native_worker.create(startup_timeout=1200)
    assert native_worker.health_requests == 1
    assert clock.seconds == 0
    assert caught.value.startup_health == {"status_code": 503, "body": reply.json()}
    assert native_worker.signals == [signal.SIGTERM, signal.SIGKILL]


def test_native_d3_startup_health_does_not_expose_url_credentials(native_worker, monkeypatch):
    import json

    startup_clock(monkeypatch)
    native_worker.health_replies = [
        native_health(
            "failed",
            "download failed: https://name:synthetic-password@example.test/file"
            "?token=synthetic-token&part=1",
        )
    ]
    with pytest.raises(RuntimeError) as caught:
        native_worker.create(startup_timeout=0.5)
    text = str(caught.value) + json.dumps(caught.value.startup_health)
    assert "synthetic-password" not in text and "synthetic-token" not in text
    assert "https://example.test/file?part=1" in text


def test_native_d3_startup_transport_error_keeps_last_observed_health(native_worker, monkeypatch):
    import httpx

    startup_clock(monkeypatch)
    reply = native_health("loading", "loading weights on cuda:0")
    native_worker.health_replies = [reply, httpx.ReadTimeout("synthetic read timeout")]
    with pytest.raises(TimeoutError) as caught:
        native_worker.create(startup_timeout=0.5)
    assert "loading weights on cuda:0" in str(caught.value)
    assert caught.value.startup_health == {"status_code": 503, "body": reply.json()}


def test_native_d3_non_json_startup_reply_keeps_last_observed_health(native_worker, monkeypatch):
    import httpx

    startup_clock(monkeypatch)
    reply = native_health("loading", "verifying the package")
    native_worker.health_replies = [reply, httpx.Response(503, text="not a native health body")]
    with pytest.raises(TimeoutError) as caught:
        native_worker.create(startup_timeout=0.5)
    assert caught.value.startup_health == {"status_code": 503, "body": reply.json()}


def test_native_d3_startup_timeout_without_health_keeps_it_unknown(native_worker, monkeypatch):
    import httpx

    startup_clock(monkeypatch)
    native_worker.health_replies = [httpx.ConnectError("synthetic connection unavailable")]
    with pytest.raises(TimeoutError) as caught:
        native_worker.create(startup_timeout=0.5)
    assert caught.value.startup_health is None
    assert "d3.log" in str(caught.value)


def test_native_d3_nonready_health_still_enforces_api_version(native_worker, monkeypatch):
    startup_clock(monkeypatch)
    native_worker.health_replies = [native_health("loading", "resolving", api_version="2.0.0")]
    with pytest.raises(ValueError, match="API"):
        native_worker.create(startup_timeout=0.5)
    assert native_worker.health_requests == 1


def test_native_d3_retries_transient_startup_states_and_keeps_native_arguments(
    native_worker, monkeypatch
):
    import httpx

    startup_clock(monkeypatch)
    native_worker.health_replies = [
        httpx.ConnectError("not listening yet"),
        native_health("loading", "retrying after transient failure (attempt 1 of 5)"),
        native_health("warming", "running the golden check"),
        native_health("ready"),
    ]
    backend = native_worker.create(startup_timeout=1)
    try:
        assert native_worker.health_requests == 4
        assert native_worker.launches[0][0][:3] == ["vllm-srun", "serve", "vllm-sr/d3"]
        assert "--dtype" not in native_worker.launches[0][0]
        assert "--max-num-seqs" not in native_worker.launches[0][0]
    finally:
        backend.close()


def test_native_d3_binds_loopback_and_terminates_only_its_owned_process_group(native_worker):
    import signal

    backend = native_worker.create()
    argv, kwargs = native_worker.launches[0]
    assert argv[:3] == ["vllm-srun", "serve", "vllm-sr/d3"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert kwargs["start_new_session"] is True
    assert backend.descriptor()["identity"]["runtime"]["data"][0]["numerics"] == "approximate"
    backend.close()
    backend.close()
    assert native_worker.signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize("numerics", ["exact", "approximate"])
@pytest.mark.parametrize("golden_status", ["matched", "unverified"])
@pytest.mark.parametrize("explicit_args", [False, True], ids=["native-defaults", "explicit-args"])
def test_native_d3_accepts_ready_unverified_workers_and_preserves_native_contract(
    native_worker, numerics, golden_status, explicit_args
):
    from copy import deepcopy

    golden = {
        "status": golden_status,
        "checked": 0 if golden_status == "unverified" else 5,
        "matched": 0 if golden_status == "unverified" else 5,
        "reference": None if golden_status == "unverified" else "rocm",
    }
    native_worker.card.update(numerics=numerics, golden=deepcopy(golden))
    engine_args = ("--max-num-seqs", "1", "--dtype", "bfloat16") if explicit_args else ()
    backend = native_worker.create(engine_args=engine_args)
    try:
        argv = native_worker.launches[0][0]
        if explicit_args:
            assert tuple(argv[-len(engine_args) :]) == engine_args
        else:
            assert "--max-num-seqs" not in argv and "--dtype" not in argv
        card = backend.descriptor()["identity"]["runtime"]["data"][0]
        assert card["golden"] == golden
        assert card["numerics"] == numerics
        assert backend.capabilities.deterministic is False
        assert backend.capabilities.batch_invariant is False
        native_worker.card["golden"]["status"] = "failed"
        card["golden"]["status"] = "pending"
        assert backend.descriptor()["identity"]["runtime"]["data"][0]["golden"] == golden
    finally:
        backend.close()


@pytest.mark.parametrize(
    "golden", [None, {}, {"status": "pending"}, {"status": "failed"}, {"status": "unknown"}]
)
def test_native_d3_still_rejects_missing_failed_or_incomplete_golden_checks(native_worker, golden):
    native_worker.card["golden"] = golden
    with pytest.raises(ValueError, match="golden"):
        native_worker.create()
    assert native_worker.signals


def test_native_d3_unverified_golden_does_not_override_failed_readiness(native_worker):
    native_worker.card.update(ready=False, golden={"status": "unverified"})
    with pytest.raises(ValueError, match="readiness"):
        native_worker.create()
    assert native_worker.signals


@pytest.mark.parametrize(
    "served_id,repo,meta_id,response_model,rejected",
    [
        ("d3", "vllm-sr/d3", "vllm-sr/d3", "d3", False),
        ("experiment-d3", "vllm-sr/d3", "vllm-sr/d3", "experiment-d3", False),
        ("d3", "vllm-sr/d3", "d3", "d3", True),
        ("d3", "vllm-sr/d3", "other/model", "d3", True),
        ("d3", "vllm-sr/d3", "vllm-sr/d3", "vllm-sr/d3", True),
        ("vllm-sr/d3", None, "vllm-sr/d3", "vllm-sr/d3", False),
    ],
)
def test_native_d3_distinguishes_served_response_name_from_metadata_repository(
    native_worker, served_id, repo, meta_id, response_model, rejected
):
    native_worker.card.update(id=served_id, repo=repo, golden={"status": "unverified"})
    backend = native_worker.create()
    receipt = {
        "request": request(),
        "response": {
            "model": response_model,
            "answers": {"decision": {"type": "choice", "probabilities": {"a": 0.25, "b": 0.75}}},
            "usage": {"input_tokens": 8},
            "meta": {
                "model_id": meta_id,
                "revision": native_worker.card["revision"],
                "model_sha256": native_worker.card["model_sha256"],
                "numerics": native_worker.card["numerics"],
            },
        },
    }
    try:
        if rejected:
            with pytest.raises(ValueError, match="identity"):
                backend.reparse_receipt(receipt)
        else:
            result = backend.reparse_receipt(receipt)
            assert result.probabilities == (0.75, 0.25)
            assert result.prompt_tokens == 8
            assert result.model == response_model
            assert result.metadata["runtime_meta"] == receipt["response"]["meta"]
    finally:
        backend.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "wrong"),
        ("family", "other"),
        ("limits", {}),
        ("engine_args", {}),
        ("golden", {"status": "failed"}),
        ("question_types", ["predicate"]),
        ("numerics", "unknown"),
        ("ready", False),
    ],
)
def test_native_d3_rejects_unverified_or_incompatible_runtime_metadata(native_worker, field, value):
    native_worker.card[field] = value
    with pytest.raises(ValueError):
        native_worker.create()
    assert native_worker.signals


def test_decisions_uses_real_sdk_transport_and_excludes_inputs_from_public_metadata():
    import json

    import httpx
    from openai import AsyncOpenAI

    if not hasattr(AsyncOpenAI, "decisions"):
        pytest.skip("Run in the isolated decisions environment with OpenAI SDK >=3.26")
    from llenvs.inference.backends.api import OpenAIBackend, _AsyncRunner

    sent = []

    def respond(http_request):
        sent.append(json.loads(http_request.content))
        assert http_request.url.path == "/v1/decisions"
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "type": "choice",
                        "name": "decision",
                        "choice": "b",
                        "confidence": 0.75,
                        "probabilities": [
                            {"value": "a", "probability": 0.25},
                            {"value": "b", "probability": 0.75},
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                },
            },
        )

    transport = OpenAIBackend.__new__(OpenAIBackend)
    transport._async_runner = _AsyncRunner()
    transport._max_concurrency = 2
    transport._async_client = AsyncOpenAI(
        api_key="offline-test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    transport._rate_limit_wait = 0
    transport._rate_limit_max_retries = 0
    backend = OpenAIDecisionBackend.__new__(OpenAIDecisionBackend)
    backend._transport, backend._model = transport, "gpt-6-luna"
    backend._async_client = transport._async_client
    try:
        result = backend.decide_batch([request()])[0]
        assert sent == [backend._request_kwargs(request())]
        assert result.probabilities == (0.75, 0.25)
        assert result.cached_prompt_tokens == 4 and result.cache_write_prompt_tokens == 2
        serialized = json.dumps(result.metadata["request"])
        assert (
            "Which range?" not in serialized
            and "Upper" not in serialized
            and "context" not in serialized
        )
    finally:
        transport._async_runner.close(transport._async_client.close())


def test_decisions_without_sdk_support_gives_installation_guidance(monkeypatch):
    import openai

    monkeypatch.delattr(openai.AsyncOpenAI, "decisions", raising=False)
    with pytest.raises(ImportError, match="decisions.*environment"):
        OpenAIDecisionBackend(api_key="offline")


def test_native_d3_rejects_response_identity_mismatch(native_worker):
    backend = native_worker.create()
    try:
        with pytest.raises(ValueError, match="identity"):
            backend.reparse_receipt(
                {
                    "request": request(),
                    "response": {
                        "model": backend._served_model,
                        "answers": {
                            "decision": {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
                        },
                        "meta": {
                            "model_id": backend._served_model,
                            "revision": "wrong",
                            "model_sha256": "model-digest",
                            "numerics": "approximate",
                        },
                    },
                }
            )
    finally:
        backend.close()
