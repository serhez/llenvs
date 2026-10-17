"""Request and retry contracts through the real SDK, with no network access."""

import asyncio
import json
import time
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from llenvs.core.tools import ToolDefinition
from llenvs.inference.backends.api import OpenAIBackend
from llenvs.inference.protocol import (
    ChatMessage,
    PartialBatchError,
    QuotaExhaustedError,
    SamplingParams,
)

MESSAGES = [ChatMessage(role="user", content="Return probabilities for these rows.")]
TOOLS = [ToolDefinition(name="noop", description="Test tool", parameters=())]


def _completion(body):
    return {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }


@pytest.fixture
def transport_backend(monkeypatch):
    sdk = pytest.importorskip("openai")
    real_sync, real_async = sdk.OpenAI, sdk.AsyncOpenAI
    state = SimpleNamespace(requests=[], replies=[], waits=[])
    backends, sync_clients, async_clients = [], [], []

    def handle(request):
        body = json.loads(request.content)
        state.requests.append(body)
        if state.replies:
            reply = state.replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            status, payload, headers = reply
        else:
            status, payload, headers = 200, _completion(body), {}
        return httpx.Response(status, json=payload, headers=headers)

    def sync_client(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handle))
        sync_clients.append(client)
        return real_sync(**kwargs, http_client=client)

    def async_client(**kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        async_clients.append(client)
        return real_async(**kwargs, http_client=client)

    monkeypatch.setattr(sdk, "OpenAI", sync_client)
    monkeypatch.setattr(sdk, "AsyncOpenAI", async_client)

    def create(**kwargs):
        backend = OpenAIBackend(
            model="gpt-4o",
            api_key="offline-test-key",
            base_url="https://offline.invalid/v1",
            max_retries=0,
            **kwargs,
        )
        backends.append(backend)
        return backend

    state.create = create
    yield state

    for backend in backends:
        backend.close()
    for client in sync_clients:
        client.close()
    for client in async_clients:
        if not client.is_closed:
            asyncio.run(client.aclose())


@pytest.fixture(params=["chat", "chat_batch", "tools", "tools_batch"])
def operation(request):
    return request.param


def _invoke(backend, operation, params):
    if operation == "chat":
        return backend.generate_chat(MESSAGES, params)
    if operation == "chat_batch":
        return backend.generate_chat_batch([MESSAGES], params)[0]
    if operation == "tools":
        return backend.generate_with_tools(MESSAGES, TOOLS, params)
    return backend.generate_with_tools_batch([MESSAGES], TOOLS, params)[0]


def _invoke_error(backend, operation, params, error_type):
    with pytest.raises((error_type, PartialBatchError)) as caught:
        _invoke(backend, operation, params)
    if isinstance(caught.value, PartialBatchError):
        assert operation.endswith("batch")
        assert set(caught.value.failures) == {0}
        assert isinstance(caught.value.failures[0], error_type)
        return caught.value.failures[0]
    return caught.value


def test_supported_sampling_values_reach_the_serialized_request(transport_backend, operation):
    backend = transport_backend.create()
    params = SamplingParams(
        max_tokens=71,
        temperature=0.6,
        top_p=0.83,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop_sequences=("DONE",),
    )

    result = _invoke(backend, operation, params)

    assert result.text == "answer"
    assert len(transport_backend.requests) == 1
    body = transport_backend.requests[0]
    assert body["max_tokens"] == 71
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.83
    assert body["presence_penalty"] == 0.0
    assert body["frequency_penalty"] == 0.0
    assert body["n"] == 1
    assert body["stop"] == ["DONE"]
    assert not {"top_k", "min_p", "repetition_penalty"} & body.keys()
    if operation.startswith("tools"):
        assert body["tools"][0]["function"]["name"] == "noop"
        assert body["tool_choice"] == "auto"


def test_none_is_omitted_rather_than_serialized_as_null(transport_backend, operation):
    backend = transport_backend.create()
    params = SamplingParams(
        max_tokens=None,
        temperature=None,
        top_p=None,
        presence_penalty=None,
        frequency_penalty=None,
    )

    _invoke(backend, operation, params)

    assert len(transport_backend.requests) == 1
    body = transport_backend.requests[0]
    assert (
        not {
            "max_tokens",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
        }
        & body.keys()
    )


def test_vllm_dialect_disables_native_filters_explicitly(transport_backend, operation):
    backend = transport_backend.create(api_flavor="vllm")

    _invoke(backend, operation, SamplingParams(temperature=0.6, top_p=0.83))

    body = transport_backend.requests[0]
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.83
    assert body["top_k"] == 0
    assert body["min_p"] == 0.0
    assert body["repetition_penalty"] == 1.0
    assert body["presence_penalty"] == 0.0
    assert body["frequency_penalty"] == 0.0
    assert "extra_body" not in body


def test_vllm_dialect_applies_non_neutral_controls(transport_backend, operation):
    backend = transport_backend.create(api_flavor="vllm")
    params = SamplingParams(top_k=17, min_p=0.08, repetition_penalty=1.3)

    _invoke(backend, operation, params)

    body = transport_backend.requests[0]
    assert body["top_k"] == 17
    assert body["min_p"] == 0.08
    assert body["repetition_penalty"] == 1.3


@pytest.mark.parametrize(
    "field,value", [("top_k", 17), ("min_p", 0.08), ("repetition_penalty", 1.3)]
)
def test_openai_rejects_unsupported_non_neutral_controls_before_dispatch(
    transport_backend, operation, field, value
):
    backend = transport_backend.create()

    _invoke_error(backend, operation, SamplingParams(**{field: value}), ValueError)

    assert transport_backend.requests == []


@pytest.mark.parametrize("n", [2, 3])
def test_single_result_operations_refuse_multiple_choices_before_dispatch(
    transport_backend, operation, n
):
    backend = transport_backend.create()

    _invoke_error(backend, operation, SamplingParams(n=n), ValueError)

    assert transport_backend.requests == []


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 0.9},
        {"model": "another-model"},
        {"n": 2},
        {"extra_body": {"temperature": 0.9}},
        {"extra_body": {"n": 2}},
        {"extra_body": {"messages": [{"role": "user", "content": "replacement"}]}},
    ],
)
def test_extras_cannot_replace_controls_or_inputs(transport_backend, operation, extra):
    backend = transport_backend.create()

    _invoke_error(backend, operation, SamplingParams(extra=extra), ValueError)

    assert transport_backend.requests == []


def test_request_metadata_matches_the_body_after_extra_body_merging(transport_backend, operation):
    backend = transport_backend.create()
    params = SamplingParams(
        extra={
            "extra_body": {"metadata": {"experiment": "pad"}},
            "extra_headers": {"Authorization": "Bearer offline-extra-secret"},
        }
    )
    before = deepcopy(params)

    result = _invoke(backend, operation, params)

    expected = {k: v for k, v in transport_backend.requests[0].items() if k != "messages"}
    assert result.metadata["request"] == expected
    assert "offline-test-key" not in json.dumps(result.metadata["request"])
    assert "offline-extra-secret" not in json.dumps(result.metadata["request"])
    assert params == before
    params.extra["extra_body"]["metadata"]["experiment"] = "changed"
    assert result.metadata["request"]["metadata"]["experiment"] == "pad"


def test_sync_and_async_paths_have_identical_requests_and_records(transport_backend):
    backend = transport_backend.create()
    params = SamplingParams(temperature=0.6, top_p=0.83)

    results = [_invoke(backend, operation, params) for operation in ["chat", "chat_batch"]]
    results += [_invoke(backend, operation, params) for operation in ["tools", "tools_batch"]]

    bodies = transport_backend.requests
    assert bodies[0] == bodies[1]
    assert bodies[2] == bodies[3]
    assert results[0].metadata["request"] == results[1].metadata["request"]
    assert results[2].metadata["request"] == results[3].metadata["request"]


@pytest.mark.parametrize("operation", ["tools", "tools_batch"])
def test_tool_only_response_preserves_missing_text(transport_backend, operation):
    backend = transport_backend.create()
    payload = _completion({"model": "gpt-4o"})
    payload["choices"][0].update(
        finish_reason="tool_calls",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-offline",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ],
        },
    )
    transport_backend.replies[:] = [(200, payload, {})]

    result = _invoke(backend, operation, SamplingParams())

    assert result.text is None
    assert result.tool_calls[0].name == "noop"
    assert result.tool_calls[0].arguments == {}


@pytest.fixture
def no_wait(monkeypatch, transport_backend):
    def sleep(delay):
        transport_backend.waits.append(delay)

    async def async_sleep(delay):
        sleep(delay)

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(asyncio, "sleep", async_sleep)


def _error(status, code="rate_limit_exceeded", *, headers=None):
    return (
        status,
        {"error": {"message": "offline provider error", "type": code, "code": code}},
        headers or {},
    )


def test_rate_limit_retries_are_bounded_and_attempts_are_recorded(
    transport_backend, operation, no_wait
):
    backend = transport_backend.create(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport_backend.replies[:] = [_error(429), _error(429)]

    result = _invoke(backend, operation, SamplingParams())

    assert len(transport_backend.requests) == 3
    assert len(transport_backend.waits) == 2
    attempts = result.metadata["attempts"]
    assert [attempt["status_code"] for attempt in attempts] == [429, 429, 200]
    assert all(attempt["elapsed_seconds"] >= 0 for attempt in attempts)
    assert result.metadata["elapsed_seconds"] >= 0


def test_retry_after_is_honored_without_an_sdk_retry_layer(transport_backend, operation, no_wait):
    backend = transport_backend.create(rate_limit_wait=0.01, rate_limit_max_retries=1)
    transport_backend.replies[:] = [_error(429, headers={"retry-after": "2"})]

    _invoke(backend, operation, SamplingParams())

    assert len(transport_backend.requests) == 2
    assert len(transport_backend.waits) == 1
    assert transport_backend.waits[0] >= 2.0


def test_exhausted_rate_limits_do_not_send_extra_attempts(transport_backend, operation, no_wait):
    sdk = pytest.importorskip("openai")
    backend = transport_backend.create(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport_backend.replies[:] = [_error(429)] * 4

    _invoke_error(backend, operation, SamplingParams(), sdk.RateLimitError)

    assert len(transport_backend.requests) == 3
    assert len(transport_backend.waits) == 2


def test_zero_rate_limit_wait_disables_backend_retries(transport_backend, operation, no_wait):
    sdk = pytest.importorskip("openai")
    backend = transport_backend.create(rate_limit_wait=0.0, rate_limit_max_retries=2)
    transport_backend.replies[:] = [_error(429)]

    _invoke_error(backend, operation, SamplingParams(), sdk.RateLimitError)

    assert len(transport_backend.requests) == 1
    assert transport_backend.waits == []


@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_provider_errors_are_not_retried(transport_backend, operation, no_wait, status):
    sdk = pytest.importorskip("openai")
    backend = transport_backend.create(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport_backend.replies[:] = [_error(status, "invalid_request_error")]

    _invoke_error(backend, operation, SamplingParams(), sdk.APIStatusError)

    assert len(transport_backend.requests) == 1
    assert transport_backend.waits == []


def test_quota_exhaustion_is_not_treated_as_a_transient_rate_limit(
    transport_backend, operation, no_wait
):
    backend = transport_backend.create(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport_backend.replies[:] = [_error(429, "insufficient_quota")]

    _invoke_error(backend, operation, SamplingParams(), QuotaExhaustedError)

    assert len(transport_backend.requests) == 1
    assert transport_backend.waits == []


def test_ambiguous_read_timeout_is_not_automatically_resubmitted(
    transport_backend, operation, no_wait
):
    sdk = pytest.importorskip("openai")
    backend = transport_backend.create(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport_backend.replies[:] = [httpx.ReadTimeout("offline timeout after dispatch")]

    _invoke_error(backend, operation, SamplingParams(), sdk.APITimeoutError)

    assert len(transport_backend.requests) == 1
    assert transport_backend.waits == []


def test_persistence_replays_a_receipt_without_another_http_call(transport_backend, tmp_path):
    from llenvs.inference.persistence import CallStore, PersistentBackend

    backend = transport_backend.create()
    parse = backend._parse_response
    backend._parse_response = lambda *a: (_ for _ in ()).throw(ValueError("interrupted parser"))
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(PartialBatchError):
            PersistentBackend(backend, store).generate_chat(
                MESSAGES, SamplingParams(), repeat_id="draw"
            )
    backend._parse_response = parse
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        result = PersistentBackend(backend, store).generate_chat(
            MESSAGES, SamplingParams(), repeat_id="draw"
        )
        assert result.text == "answer"
        assert result.metadata["raw_usage"]["prompt_tokens"] == 7
    assert len(transport_backend.requests) == 1


def test_persistence_never_caches_refusals_as_success(transport_backend, tmp_path):
    from llenvs.inference.persistence import CallStore, PersistentBackend

    payload = _completion({"model": "gpt-4o"})
    payload["choices"][0]["message"].update(content=None, refusal="Cannot answer")
    transport_backend.replies[:] = [(200, payload, {})]
    backend = transport_backend.create()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(PartialBatchError):
            PersistentBackend(backend, store).generate_chat(MESSAGES, SamplingParams(), repeat_id=1)
        assert next(iter(store._records.values()))["receipt"]["response"]["choices"][0]["message"][
            "refusal"
        ]


def test_closed_api_logprobs_do_not_invent_native_token_ids(transport_backend):
    backend = transport_backend.create()
    payload = _completion({"model": "gpt-4o"})
    payload["choices"][0]["logprobs"] = {
        "content": [{"token": "answer", "logprob": -0.2, "bytes": [97], "top_logprobs": []}]
    }
    transport_backend.replies[:] = [(200, payload, {})]
    result = backend.generate_chat(MESSAGES, SamplingParams(logprobs=True))
    assert result.token_logprobs[0].token_id is None


def test_vllm_template_effort_is_applied_without_rewriting_requested_extras(transport_backend):
    backend = transport_backend.create(api_flavor="vllm", reasoning_family="qwen3.8")
    params = SamplingParams(reasoning_effort="medium")
    result = backend.generate_chat(MESSAGES, params)
    assert transport_backend.requests[0]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "preserve_thinking": True,
    }
    assert result.metadata["requested_sampling"]["extra"] == {}


def test_reasoning_logprobs_keep_native_coordinates_and_slice_only_with_exact_ids(
    transport_backend,
):
    backend = transport_backend.create(api_flavor="vllm", reasoning_family="qwen3.8")

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [99 if text == "</think>" else 98]

        def decode(self, ids, **kwargs):
            return "".join(str(i) for i in ids if i != 99)

    backend._reasoning_tokenizer = Tokenizer()
    payload = _completion({"model": "gpt-4o"})
    payload["choices"][0]["message"].update(content="5", reasoning_content="4")
    payload["choices"][0]["token_ids"] = [4, 99, 5]
    payload["choices"][0]["logprobs"] = {
        "content": [
            {"token": str(i), "logprob": -0.1 * i, "bytes": [97], "top_logprobs": []}
            for i in [4, 99, 5]
        ]
    }
    transport_backend.replies[:] = [(200, payload, {})]
    result = backend.generate_chat(MESSAGES, SamplingParams(reasoning_effort="low", logprobs=True))
    assert result.text == "5"
    assert [lp.token_id for lp in result.token_logprobs] == [5]
    assert [lp["token_id"] for lp in result.metadata["raw_token_logprobs"]] == [4, 99, 5]
    assert result.metadata["answer_span"] == [2, 3]


def test_persisted_quota_failure_replays_its_classification_without_dispatch(
    transport_backend, tmp_path
):
    from llenvs.inference.persistence import CallStore, PersistentBackend
    from llenvs.inference.protocol import QuotaExhaustedError

    transport_backend.replies[:] = [
        (429, {"error": {"message": "quota", "code": "insufficient_quota"}}, {})
    ]
    backend = transport_backend.create()
    for _ in range(2):
        with CallStore.open(tmp_path, run_descriptor={}) as store:
            with pytest.raises(PartialBatchError) as caught:
                PersistentBackend(backend, store).generate_chat(
                    MESSAGES, SamplingParams(), repeat_id=1
                )
            assert isinstance(caught.value.failures[0], QuotaExhaustedError)
            error = next(iter(store._records.values()))["error"]
            assert error["type"] == "QuotaExhaustedError" and error["transient"] is False
            assert len(error["attempts"]) == 1
    assert len(transport_backend.requests) == 1
