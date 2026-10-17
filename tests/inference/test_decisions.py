"""Decision responses must contain the complete requested distribution."""

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
    state = SimpleNamespace(card=card, launches=[], signals=[])
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
    state.create = lambda: D3Backend(log_path=tmp_path / "d3.log", shutdown_timeout=0.01)
    yield state
    for callback in exit_callbacks:
        callback()


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
