"""OpenAI's Responses API: stateless conversations and compaction, through the real
SDK over an offline transport."""

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from llenvs.core.state import ImageContent
from llenvs.core.tools import ToolCall, ToolDefinition
from llenvs.inference.backends.api import OpenAIBackend
from llenvs.inference.protocol import (
    ChatMessage,
    MalformedResponseError,
    PartialBatchError,
    PromptTooLongError,
    QuotaExhaustedError,
    RefusedByPolicyError,
    SamplingParams,
    StopReason,
    encode_backend_error,
)

LUNA = "gpt-6-luna"
# Reasoning models take no temperature or top_p, as on Chat Completions.
PARAMS = SamplingParams(max_tokens=900, temperature=None, top_p=None, reasoning_effort="low")
SYSTEM = ChatMessage("system", "You write programs that forecast held-out values.")
FIRST = ChatMessage("user", "Write the program for this summary.")
SECOND = ChatMessage("user", "Finding 1: update the program for this finding.")
USAGE = {
    "input_tokens": 3348,
    "input_tokens_details": {"cached_tokens": 1024, "cache_write_tokens": 2300},
    "output_tokens": 49,
    "output_tokens_details": {"reasoning_tokens": 42},
    "total_tokens": 3397,
}


def _reasoning_item(n=1):
    # As the API can return it: an explicit null status, and no content field, which
    # the SDK's model defaults to null.
    return {
        "id": f"rs_{n}",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": f"gAAAA-encrypted-{n}",
        "status": None,
    }


def _message_item(text, n=1):
    return {
        "id": f"msg_{n}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        # A field this SDK does not know, with a null inside, must survive too.
        "future_field": {"kept": True, "nullable": None},
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }


def _response(body, *, output=None, status="completed", incomplete=None, error=None):
    return {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0,
        "model": body["model"],
        "status": status,
        "incomplete_details": incomplete,
        "error": error,
        "output": [_reasoning_item(), _message_item("answer")] if output is None else output,
        "reasoning": {"effort": "low", "context": "all_turns", "summary": None},
        "store": False,
        "usage": USAGE,
    }


def _compaction(body, *, output=None):
    """Messages kept verbatim and every model turn replaced by one encrypted item, as
    the probe observed."""
    kept = [
        {
            "id": f"msg_kept_{i}",
            "type": "message",
            "role": entry["role"],
            "status": "completed",
            "phase": None,
            "content": [{"type": "input_text", "text": entry["content"], "annotations": None}],
        }
        for i, entry in enumerate(body["input"])
        if entry.get("role") in {"system", "developer", "user"} and "type" not in entry
    ]
    compacted = {
        "id": "cmp_1",
        "type": "compaction",
        "encrypted_content": "gAAAA-compacted",
        "created_by": None,
    }
    return {
        "id": "resp_compaction",
        "object": "response.compaction",
        "created_at": 0,
        "output": [*kept, compacted] if output is None else output,
        "usage": {
            "input_tokens": 3417,
            "input_tokens_details": {"cached_tokens": 3345, "cache_write_tokens": 0},
            "output_tokens": 39,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3456,
        },
    }


@pytest.fixture
def transport(monkeypatch):
    sdk = pytest.importorskip("openai")
    real_sync, real_async = sdk.OpenAI, sdk.AsyncOpenAI
    state = SimpleNamespace(requests=[], paths=[], replies=[], waits=[])
    backends, clients = [], []

    def handle(request):
        body = json.loads(request.content)
        state.requests.append(body)
        state.paths.append(request.url.path)
        if state.replies:
            reply = state.replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            status, payload = reply
            if callable(payload):
                payload = payload(body)
        elif request.url.path.endswith("/responses/compact"):
            status, payload = 200, _compaction(body)
        elif request.url.path.endswith("/responses"):
            status, payload = 200, _response(body)
        else:
            raise AssertionError(f"Unexpected endpoint {request.url.path}")
        return httpx.Response(status, json=payload)

    def sync_client(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handle))
        clients.append(client)
        return real_sync(**kwargs, http_client=client)

    def async_client(**kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        clients.append(client)
        return real_async(**kwargs, http_client=client)

    monkeypatch.setattr(sdk, "OpenAI", sync_client)
    monkeypatch.setattr(sdk, "AsyncOpenAI", async_client)

    def create(**kwargs):
        backend = OpenAIBackend(
            api_key="offline-test-key",
            base_url="https://offline.invalid/v1",
            max_retries=0,
            **{"model": LUNA, **kwargs},
        )
        backends.append(backend)
        return backend

    state.create = create
    state.responses = lambda **kwargs: create(
        **{"api": "responses", "reasoning_context": "all_turns", **kwargs}
    )
    yield state

    for backend in backends:
        backend.close()
    for client in clients:
        if isinstance(client, httpx.AsyncClient):
            if not client.is_closed:
                asyncio.run(client.aclose())
        else:
            client.close()


@pytest.fixture(params=["chat", "chat_batch"])
def operation(request):
    return request.param


def _generate(backend, operation, messages, params=PARAMS):
    if operation == "chat":
        return backend.generate_chat(messages, params)
    return backend.generate_chat_batch([messages], params)[0]


def _generate_error(backend, operation, messages, params, error_type):
    with pytest.raises((error_type, PartialBatchError)) as caught:
        _generate(backend, operation, messages, params)
    if isinstance(caught.value, PartialBatchError):
        assert operation == "chat_batch"
        assert isinstance(caught.value.failures[0], error_type)
        return caught.value.failures[0]
    return caught.value


def _turn(result):
    """The assistant's turn as a caller continues the conversation with it."""
    return ChatMessage(
        "assistant",
        content=result.text,
        reasoning_details=tuple(result.metadata["reasoning_details"]),
    )


# --- Configuration ------------------------------------------------------------------


def test_chat_completions_stays_the_default(transport):
    backend = transport.create()
    transport.replies[:] = [
        (
            200,
            {
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 0,
                "model": LUNA,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    ]

    backend.generate_chat([FIRST], PARAMS)

    assert transport.paths == ["/v1/chat/completions"]


def test_the_chat_completions_identity_is_unchanged_so_stored_keys_are_kept(transport):
    """Stored Chat Completions calls are keyed by this identity: the Responses path
    must not add fields to it."""
    identity = transport.create().descriptor()["identity"]

    assert set(identity) == {
        "backend",
        "protocol_version",
        "api_flavor",
        "model",
        "version_pinned",
        "endpoint",
        "sdk_max_retries",
        "dependencies",
        "rate_limit_wait",
        "rate_limit_max_retries",
    }


def test_the_api_and_the_reasoning_context_enter_the_identity(transport):
    responses = transport.responses().descriptor()["identity"]

    assert responses != transport.create().descriptor()["identity"]
    assert (
        responses != transport.responses(reasoning_context="current_turn").descriptor()["identity"]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"api": "assistants"},
        {"reasoning_context": "all-turns"},
        {"reasoning_context": ""},
        {"api": "chat_completions", "reasoning_context": "all_turns"},
        {"api_flavor": "vllm"},
    ],
)
def test_invalid_configurations_are_refused_at_construction(transport, options):
    with pytest.raises(ValueError):
        transport.responses(**options)


@pytest.mark.parametrize("context", ["auto", "current_turn", "all_turns"])
def test_every_mode_of_the_api_is_accepted(transport, context):
    backend = transport.responses(reasoning_context=context)

    backend.generate_chat([FIRST], PARAMS)

    assert transport.requests[0]["reasoning"]["context"] == context


def test_batch_refuses_a_responses_backend(transport, tmp_path):
    """The Batch path sends Chat Completions requests; a Responses backend's requests
    would go to the wrong endpoint."""
    from llenvs.inference import CallStore
    from llenvs.inference.backends.openai_batch import OpenAIBatchBackend

    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(ValueError):
            OpenAIBatchBackend(transport.responses(), store)


# --- Requests -----------------------------------------------------------------------


def test_requests_are_stateless_and_ask_for_encrypted_reasoning(transport, operation):
    backend = transport.responses()

    _generate(backend, operation, [SYSTEM, FIRST])

    assert transport.paths == ["/v1/responses"]
    body = transport.requests[0]
    assert body["model"] == LUNA
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "low", "context": "all_turns"}
    assert body["max_output_tokens"] == 900
    assert body["input"] == [
        {"role": "system", "content": SYSTEM.content},
        {"role": "user", "content": FIRST.content},
    ]
    # Controls the API does not take are left out, as their neutral values say.
    assert set(body) == {"model", "input", "store", "include", "reasoning", "max_output_tokens"}


def test_sync_and_async_requests_are_identical(transport):
    backend = transport.responses()

    results = [
        _generate(backend, operation, [SYSTEM, FIRST]) for operation in ("chat", "chat_batch")
    ]

    assert transport.requests[0] == transport.requests[1]
    assert results[0].metadata["request"] == results[1].metadata["request"]


def test_omitted_effort_and_context_are_left_to_the_model(transport):
    backend = transport.responses(reasoning_context=None)

    backend.generate_chat([FIRST], SamplingParams(max_tokens=None, temperature=None, top_p=None))

    body = transport.requests[0]
    assert "reasoning" not in body
    assert "max_output_tokens" not in body


def test_decoding_controls_reach_the_request_when_reasoning_is_off(transport):
    backend = transport.responses()

    backend.generate_chat(
        [FIRST], SamplingParams(temperature=0.6, top_p=0.83, reasoning_effort="none")
    )

    body = transport.requests[0]
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.83
    assert body["reasoning"]["effort"] == "none"


def test_a_schema_is_sent_as_the_text_format(transport):
    from llenvs.inference.protocol import ResponseSchema

    schema = ResponseSchema(
        {"type": "object", "properties": {"p": {"type": "integer"}}, "required": ["p"]},
        name="prediction",
    )

    transport.responses().generate_chat([FIRST], replace(PARAMS, response_schema=schema))

    assert transport.requests[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "prediction",
            "schema": schema.schema,
            "strict": True,
        }
    }


@pytest.mark.parametrize(
    "model,effort",
    [("gpt-6.1-sol", "none"), (LUNA, "max"), ("gpt-unknown", "low")],
)
def test_efforts_are_checked_per_model_before_dispatch(transport, model, effort):
    backend = transport.responses(model=model)

    with pytest.raises(ValueError):
        backend.generate_chat(
            [FIRST], SamplingParams(temperature=None, top_p=None, reasoning_effort=effort)
        )

    assert transport.requests == []


def test_reasoning_still_refuses_decoding_controls(transport, operation):
    backend = transport.responses()

    _generate_error(backend, operation, [FIRST], SamplingParams(reasoning_effort="low"), ValueError)

    assert transport.requests == []


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=None, top_p=None, stop_sequences=("DONE",)),
        SamplingParams(temperature=None, top_p=None, reasoning_effort="none", logprobs=True),
        SamplingParams(temperature=None, top_p=None, top_k=5),
        SamplingParams(temperature=None, top_p=None, presence_penalty=0.5),
        SamplingParams(temperature=None, top_p=None, frequency_penalty=0.5),
    ],
)
def test_controls_the_responses_api_lacks_are_refused_before_dispatch(transport, operation, params):
    backend = transport.responses()

    _generate_error(backend, operation, [FIRST], params, ValueError)

    assert transport.requests == []


def test_tools_are_refused_before_dispatch(transport):
    backend = transport.responses()
    tools = [ToolDefinition(name="noop", description="Test tool", parameters=())]

    with pytest.raises(ValueError):
        backend.generate_with_tools([FIRST], tools, PARAMS)

    assert transport.requests == []


def test_extras_pass_through(transport):
    backend = transport.responses()

    backend.generate_chat(
        [FIRST], replace(PARAMS, extra={"prompt_cache_key": "state-7", "service_tier": "default"})
    )

    assert transport.requests[0]["prompt_cache_key"] == "state-7"
    assert transport.requests[0]["service_tier"] == "default"


@pytest.mark.parametrize(
    "extra",
    [
        {"store": True},
        {"include": []},
        {"previous_response_id": "resp_earlier"},
        {"conversation": "conv_1"},
        {"background": True},
        {"reasoning": {"context": "current_turn"}},
        {"input": [{"role": "user", "content": "replacement"}]},
        {"instructions": "Another system prompt"},
        {"max_output_tokens": 5},
        {"text": {"format": {"type": "text"}}},
        {"truncation": "auto"},
        {"context_management": [{"type": "compaction", "compact_threshold": 1000}]},
        {"extra_body": {"store": True}},
        {"extra_body": {"reasoning": {"context": "current_turn"}}},
    ],
)
def test_extras_cannot_make_a_request_stateful_or_change_its_input(transport, extra):
    backend = transport.responses()

    with pytest.raises(ValueError):
        backend.generate_chat([FIRST], replace(PARAMS, extra=extra))

    assert transport.requests == []


def test_an_earlier_turn_is_sent_back_as_exactly_its_items(transport, operation):
    backend = transport.responses()
    first = _generate(backend, operation, [SYSTEM, FIRST])

    _generate(backend, operation, [SYSTEM, FIRST, _turn(first), SECOND])

    served = [_reasoning_item(), _message_item("answer")]
    assert transport.requests[1]["input"] == [
        {"role": "system", "content": SYSTEM.content},
        {"role": "user", "content": FIRST.content},
        *served,
        {"role": "user", "content": SECOND.content},
    ]


def test_items_are_sent_unchanged_after_a_json_round_trip(transport):
    """A caller keeps the items as JSON and sends them back from there."""
    backend = transport.responses()
    first = backend.generate_chat([FIRST], PARAMS)
    kept = json.loads(json.dumps(first.metadata["reasoning_details"]))

    backend.generate_chat(
        [FIRST, ChatMessage("assistant", first.text, reasoning_details=tuple(kept)), SECOND],
        PARAMS,
    )

    assert transport.requests[1]["input"][1:-1] == first.metadata["reasoning_details"]


def test_an_assistant_message_without_items_is_sent_as_its_text(transport):
    backend = transport.responses()

    backend.generate_chat([FIRST, ChatMessage("assistant", "Earlier program"), SECOND], PARAMS)

    assert transport.requests[0]["input"][1] == {"role": "assistant", "content": "Earlier program"}


@pytest.mark.parametrize(
    "message",
    [
        # Reasoning text has no place in a Responses request.
        ChatMessage("assistant", "answer", reasoning="earlier thoughts"),
        # Its items carry another text than the message claims.
        ChatMessage(
            "assistant",
            "another answer",
            reasoning_details=(_reasoning_item(), _message_item("answer")),
        ),
        ChatMessage(
            "assistant",
            None,
            tool_calls=(ToolCall(id="call_1", name="noop", arguments={}),),
        ),
        ChatMessage("tool", "output", tool_call_id="call_1", name="noop"),
        ChatMessage("user", "Look", images=(ImageContent(data="aGk=", media_type="image/png"),)),
        ChatMessage("user", content_blocks=("Look",)),
    ],
)
def test_messages_the_path_cannot_send_are_refused_before_dispatch(transport, message):
    backend = transport.responses()

    with pytest.raises(ValueError):
        backend.generate_chat([FIRST, message, SECOND], PARAMS)

    assert transport.requests == []


# --- Results ------------------------------------------------------------------------


def test_results_carry_the_text_usage_items_and_reported_context(transport, operation):
    backend = transport.responses()
    served = [
        _reasoning_item(1),
        _message_item("first part, ", 1),
        _reasoning_item(2),
        _message_item("second part", 2),
    ]

    def reply(body):
        payload = _response(body, output=served)
        # The mode the response reports, not the one requested, is what counts.
        payload["reasoning"]["context"] = "current_turn"
        return payload

    transport.replies[:] = [(200, reply)]

    result = _generate(backend, operation, [FIRST])

    assert result.text == "first part, second part"
    assert result.finish_reason == StopReason.END_OF_TEXT
    assert (
        result.prompt_tokens,
        result.cached_prompt_tokens,
        result.cache_write_prompt_tokens,
        result.completion_tokens,
        result.reasoning_tokens,
    ) == (3348, 1024, 2300, 49, 42)
    assert result.metadata["reasoning_details"] == served
    assert (
        json.loads(json.dumps(result.metadata["reasoning_details"]))
        == (result.metadata["reasoning_details"])
    )
    assert result.metadata["reasoning_context"] == "current_turn"
    assert result.metadata["id"] == "resp_offline"
    assert result.metadata["model"] == LUNA


def test_an_unreported_context_stays_unknown(transport):
    backend = transport.responses()

    def reply(body):
        payload = _response(body)
        del payload["reasoning"]
        return payload

    transport.replies[:] = [(200, reply)]

    assert backend.generate_chat([FIRST], PARAMS).metadata["reasoning_context"] is None


def test_records_describe_the_request_without_its_input_and_every_attempt(transport, operation):
    backend = transport.responses()

    result = _generate(backend, operation, [SYSTEM, FIRST])

    record = result.metadata["request"]
    assert record == {k: v for k, v in transport.requests[0].items() if k != "input"}
    assert SYSTEM.content not in json.dumps(record)
    assert [attempt["status_code"] for attempt in result.metadata["attempts"]] == [200]
    assert result.metadata["attempts"][0]["response_id"] == "resp_offline"
    assert result.metadata["attempts"][0]["usage"]["input_tokens"] == 3348


@pytest.mark.parametrize(
    "output,text",
    [
        ([_reasoning_item()], ""),
        ([_reasoning_item(), _message_item("def forecast(")], "def forecast("),
    ],
)
def test_an_answer_cut_at_its_token_limit_is_truncated(transport, operation, output, text):
    backend = transport.responses()
    transport.replies[:] = [
        (
            200,
            lambda body: _response(
                body,
                output=output,
                status="incomplete",
                incomplete={"reason": "max_output_tokens"},
            ),
        )
    ]

    result = _generate(backend, operation, [FIRST])

    assert result.finish_reason == StopReason.MAX_TOKENS
    assert result.text == text
    assert result.metadata["reasoning_details"] == output


def _refusal_item():
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
    }


@pytest.mark.parametrize(
    "reply",
    [
        lambda body: _response(body, output=[_reasoning_item(), _refusal_item()]),
        lambda body: _response(
            body, output=[], status="incomplete", incomplete={"reason": "content_filter"}
        ),
        lambda body: _response(
            body,
            output=[],
            status="failed",
            error={"code": "bio_policy", "message": "Refused by policy."},
        ),
    ],
    ids=["refusal", "content_filter", "policy_error"],
)
def test_refusals_raise_refused_by_policy(transport, operation, reply):
    backend = transport.responses()
    transport.replies[:] = [(200, reply)]

    _generate_error(backend, operation, [FIRST], PARAMS, RefusedByPolicyError)


@pytest.mark.parametrize(
    "code,transient",
    [("server_error", True), ("rate_limit_exceeded", True), ("invalid_prompt", False)],
)
def test_a_failed_response_is_malformed_and_server_errors_and_rate_limits_stay_transient(
    transport, operation, code, transient
):
    backend = transport.responses()
    transport.replies[:] = [
        (
            200,
            lambda body: _response(
                body, output=[], status="failed", error={"code": code, "message": "failed"}
            ),
        )
    ]

    error = _generate_error(backend, operation, [FIRST], PARAMS, MalformedResponseError)

    assert encode_backend_error(error)["transient"] is transient


# --- Errors over HTTP, as on Chat Completions ---------------------------------------


@pytest.fixture
def no_wait(monkeypatch, transport):
    def sleep(delay):
        transport.waits.append(delay)

    async def async_sleep(delay):
        sleep(delay)

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(asyncio, "sleep", async_sleep)


def _error(status, code, message="offline provider error"):
    return status, {"error": {"message": message, "type": code, "code": code}}


def _call(backend, call):
    if call == "compact":
        return backend.compact_chat([SYSTEM, FIRST])
    if call == "compact_batch":
        return backend.compact_chat_batch([[SYSTEM, FIRST]])[0]
    return _generate(backend, call, [FIRST])


@pytest.fixture(params=["chat", "chat_batch", "compact", "compact_batch"])
def call(request):
    return request.param


def _call_error(backend, call, error_type):
    with pytest.raises((error_type, PartialBatchError)) as caught:
        _call(backend, call)
    if isinstance(caught.value, PartialBatchError):
        assert call.endswith("batch")
        assert isinstance(caught.value.failures[0], error_type)
        return caught.value.failures[0]
    return caught.value


def test_rate_limits_are_retried_and_every_attempt_is_recorded(transport, call, no_wait):
    backend = transport.responses(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport.replies[:] = [_error(429, "rate_limit_exceeded")] * 2

    result = _call(backend, call)

    assert len(transport.requests) == 3
    assert len(transport.waits) == 2
    assert [a["status_code"] for a in result.metadata["attempts"]] == [429, 429, 200]


def test_quota_exhaustion_is_not_retried(transport, call, no_wait):
    backend = transport.responses(rate_limit_wait=0.01, rate_limit_max_retries=2)
    transport.replies[:] = [_error(429, "insufficient_quota")]

    _call_error(backend, call, QuotaExhaustedError)

    assert len(transport.requests) == 1


def test_a_conversation_too_long_for_the_model_is_a_prompt_too_long(transport, call, no_wait):
    backend = transport.responses(rate_limit_wait=0.01)
    transport.replies[:] = [
        _error(400, "context_length_exceeded", "Your input exceeds the context length.")
    ]

    _call_error(backend, call, PromptTooLongError)

    assert len(transport.requests) == 1


def test_server_errors_are_raised_as_transient_without_a_retry(transport, call, no_wait):
    sdk = pytest.importorskip("openai")
    backend = transport.responses(rate_limit_wait=0.01)
    transport.replies[:] = [_error(500, "server_error")]

    error = _call_error(backend, call, sdk.InternalServerError)

    assert encode_backend_error(error)["transient"] is True
    assert len(transport.requests) == 1


def test_a_read_timeout_is_not_sent_again(transport, call, no_wait):
    sdk = pytest.importorskip("openai")
    backend = transport.responses(rate_limit_wait=0.01)
    transport.replies[:] = [httpx.ReadTimeout("offline timeout after dispatch")]

    _call_error(backend, call, sdk.APITimeoutError)

    assert len(transport.requests) == 1


# --- Compaction ---------------------------------------------------------------------


def _conversation(backend):
    first = backend.generate_chat([SYSTEM, FIRST], PARAMS)
    return [SYSTEM, FIRST, _turn(first), SECOND]


def test_compaction_sends_the_conversation_to_the_compaction_endpoint(transport):
    backend = transport.responses()
    conversation = _conversation(backend)

    backend.compact_chat(conversation)

    assert transport.paths[-1] == "/v1/responses/compact"
    assert transport.requests[-1] == {
        "model": LUNA,
        "input": [
            {"role": "system", "content": SYSTEM.content},
            {"role": "user", "content": FIRST.content},
            _reasoning_item(),
            _message_item("answer"),
            {"role": "user", "content": SECOND.content},
        ],
    }


@pytest.mark.parametrize("call", ["compact", "compact_batch"])
def test_compaction_keeps_messages_and_returns_compacted_turns_as_opaque_items(transport, call):
    backend = transport.responses()

    from llenvs.inference import CompactionResult

    window = _call(backend, call)

    assert isinstance(window, CompactionResult)
    assert window.messages == (
        ChatMessage("system", SYSTEM.content),
        ChatMessage("user", FIRST.content),
        ChatMessage(
            "assistant",
            None,
            reasoning_details=(
                {
                    "id": "cmp_1",
                    "type": "compaction",
                    "encrypted_content": "gAAAA-compacted",
                    "created_by": None,
                },
            ),
        ),
    )
    assert (
        window.prompt_tokens,
        window.cached_prompt_tokens,
        window.cache_write_prompt_tokens,
        window.completion_tokens,
        window.reasoning_tokens,
    ) == (3417, 3345, 0, 39, 0)
    assert window.metadata["id"] == "resp_compaction"
    assert window.metadata["attempts"][0]["usage"]["input_tokens"] == 3417


def test_a_compacted_window_is_sent_back_unchanged(transport):
    backend = transport.responses()
    served = [
        {
            "id": "m0",
            "type": "message",
            "role": "developer",
            "status": "completed",
            "content": [{"type": "input_text", "text": "Rules."}],
        },
        {
            "id": "m1",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": "Task one."}],
        },
        {"id": "cmp_1", "type": "compaction", "encrypted_content": "gAAAA-one"},
        {
            "id": "m2",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": "Task two."}],
        },
        {"id": "cmp_2", "type": "compaction", "encrypted_content": "gAAAA-two"},
        _reasoning_item(3),
        _message_item("kept answer", 3),
    ]
    transport.replies[:] = [(200, lambda body: _compaction(body, output=served))]

    window = backend.compact_chat([SYSTEM, FIRST])
    backend.generate_chat([*window.messages, SECOND], PARAMS)

    assert [m.role for m in window.messages] == [
        "developer",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert window.messages[-1].content == "kept answer"
    assert transport.requests[-1]["input"] == [
        {"role": "developer", "content": "Rules."},
        {"role": "user", "content": "Task one."},
        served[2],
        {"role": "user", "content": "Task two."},
        served[4],
        served[5],
        served[6],
        {"role": "user", "content": SECOND.content},
    ]


def test_compaction_batch_keeps_the_order_of_its_conversations(transport):
    backend = transport.responses()
    other = ChatMessage("user", "Another state's program.")

    windows = backend.compact_chat_batch([[SYSTEM, FIRST], [SYSTEM, other]])

    assert [w.messages[1].content for w in windows] == [FIRST.content, other.content]


def test_a_compaction_batch_keeps_its_windows_when_one_fails(transport):
    backend = transport.responses()
    other = ChatMessage("user", "Another state's program.")

    def reply(body):
        failing = any(entry.get("content") == other.content for entry in body["input"])
        return _compaction(body, output=[] if failing else None)

    transport.replies[:] = [(200, reply)] * 3

    with pytest.raises(PartialBatchError) as caught:
        backend.compact_chat_batch([[SYSTEM, FIRST], [SYSTEM, other], [SYSTEM, SECOND]])

    assert list(caught.value.failures) == [1]
    assert isinstance(caught.value.failures[1], MalformedResponseError)
    assert [caught.value.results[i].messages[1].content for i in (0, 2)] == [
        FIRST.content,
        SECOND.content,
    ]


def test_an_empty_compaction_is_malformed(transport):
    backend = transport.responses()
    transport.replies[:] = [(200, lambda body: _compaction(body, output=[]))]

    with pytest.raises(MalformedResponseError):
        backend.compact_chat([SYSTEM, FIRST])


def test_compaction_needs_the_responses_api(transport):
    backend = transport.create()

    with pytest.raises(ValueError):
        backend.compact_chat([SYSTEM, FIRST])

    assert transport.requests == []


def test_compaction_refuses_what_the_path_cannot_send(transport):
    backend = transport.responses()

    with pytest.raises(ValueError):
        backend.compact_chat([FIRST, ChatMessage("assistant", "a", reasoning="thoughts")])

    assert transport.requests == []


# --- The store of calls -------------------------------------------------------------


def _store(path):
    from llenvs.inference import CallStore

    return CallStore.open(path, run_descriptor={"experiment": "conversation"})


def test_a_continued_conversation_is_replayed_after_a_restart(transport, tmp_path):
    from llenvs.inference import PersistentBackend

    backend = transport.responses()
    with _store(tmp_path) as store:
        persistent = PersistentBackend(backend, store)
        first = persistent.generate_chat([SYSTEM, FIRST], PARAMS, repeat_id="r0")
        second = persistent.generate_chat(
            [SYSTEM, FIRST, _turn(first), SECOND], PARAMS, repeat_id="r0"
        )
    kept = json.loads(json.dumps(first.metadata["reasoning_details"]))

    with _store(tmp_path) as store:
        replayed = PersistentBackend(backend, store).generate_chat(
            [
                SYSTEM,
                FIRST,
                ChatMessage("assistant", first.text, reasoning_details=tuple(kept)),
                SECOND,
            ],
            PARAMS,
            repeat_id="r0",
        )

    assert len(transport.requests) == 2
    assert replayed.metadata["persistence"]["source"] == "store"
    assert (
        replayed.metadata["persistence"]["call_key"] == (second.metadata["persistence"]["call_key"])
    )
    assert replayed.metadata["reasoning_details"] == second.metadata["reasoning_details"]
    assert replayed.metadata["reasoning_context"] == "all_turns"


def test_other_replayed_items_make_another_call(transport, tmp_path):
    from llenvs.inference import PersistentBackend

    backend = transport.responses()
    other = dict(_reasoning_item(), encrypted_content="gAAAA-other")
    with _store(tmp_path) as store:
        persistent = PersistentBackend(backend, store)
        for reasoning in (_reasoning_item(), other):
            turn = ChatMessage(
                "assistant", "answer", reasoning_details=(reasoning, _message_item("answer"))
            )
            persistent.generate_chat([FIRST, turn, SECOND], PARAMS, repeat_id="r0")

    assert len(transport.requests) == 2


def test_the_reasoning_context_separates_calls(transport, tmp_path):
    from llenvs.inference import PersistentBackend

    with _store(tmp_path) as store:
        for context in ("all_turns", "current_turn"):
            PersistentBackend(transport.responses(reasoning_context=context), store).generate_chat(
                [FIRST], PARAMS, repeat_id="r0"
            )

    assert len(transport.requests) == 2


def test_a_compaction_is_kept_and_replayed_like_an_answer(transport, tmp_path):
    from llenvs.inference import PersistentBackend

    backend = transport.responses()
    with _store(tmp_path) as store:
        window = PersistentBackend(backend, store).compact_chat([SYSTEM, FIRST], repeat_id="w0")
    with _store(tmp_path) as store:
        persistent = PersistentBackend(backend, store)
        [replayed] = persistent.compact_chat_batch([[SYSTEM, FIRST]], repeat_ids=["w0"])

    assert len(transport.requests) == 1
    assert window.metadata["persistence"]["source"] == "backend"
    assert replayed.metadata["persistence"]["source"] == "store"
    assert replayed.messages == window.messages
    assert replayed.prompt_tokens == 3417
    assert replayed.metadata["attempts"] == window.metadata["attempts"]


@pytest.mark.parametrize("call", ["response", "compaction"])
def test_a_receipt_is_parsed_again_without_another_call(transport, tmp_path, call):
    """A parser interrupted after the answer arrived: the next run parses the kept
    receipt into the same result a fresh parse gives."""
    from llenvs.inference import PersistentBackend

    def send(persistent):
        if call == "compaction":
            return persistent.compact_chat([SYSTEM, FIRST], repeat_id="r0")
        return persistent.generate_chat([SYSTEM, FIRST], PARAMS, repeat_id="r0")

    backend = transport.responses()
    fresh = (
        backend.compact_chat([SYSTEM, FIRST])
        if call == "compaction"
        else (backend.generate_chat([SYSTEM, FIRST], PARAMS))
    )
    parse = backend._parse_response
    backend._parse_response = lambda *a: (_ for _ in ()).throw(ValueError("interrupted parser"))
    with _store(tmp_path) as store:
        with pytest.raises(PartialBatchError):
            send(PersistentBackend(backend, store))
    backend._parse_response = parse
    with _store(tmp_path) as store:
        reparsed = send(PersistentBackend(backend, store))

    assert len(transport.requests) == 2
    assert reparsed.metadata["persistence"]["source"] == "receipt"
    if call == "compaction":
        assert reparsed.messages == fresh.messages
    else:
        assert reparsed.text == fresh.text
        assert reparsed.metadata["reasoning_details"] == fresh.metadata["reasoning_details"]
        assert reparsed.metadata["reasoning_context"] == fresh.metadata["reasoning_context"]
    assert reparsed.prompt_tokens == fresh.prompt_tokens


@pytest.mark.parametrize("source", ["backend", "receipt"])
@pytest.mark.parametrize("call", ["response", "compaction"])
def test_items_go_back_with_the_nulls_the_api_sent_and_no_others(transport, tmp_path, call, source):
    """As the SDK sends ``response.output`` back: the nulls the API sent, nested ones
    included, are kept, and the fields it left out, which the SDK's models default to
    null, stay out. A result parsed from a kept receipt is no different."""
    from llenvs.inference import PersistentBackend

    served = [_reasoning_item(), _message_item("answer")]
    if call == "compaction":
        served.insert(
            0,
            {
                "id": "cmp_1",
                "type": "compaction",
                "encrypted_content": "gAAAA-compacted",
                "created_by": None,
            },
        )
        transport.replies[:] = [(200, lambda body: _compaction(body, output=served))]
    assert "content" not in served[-2] and served[-2]["status"] is None
    assert served[-1]["future_field"]["nullable"] is None

    def send(persistent):
        if call == "compaction":
            return persistent.compact_chat([SYSTEM, FIRST], repeat_id="r0")
        return persistent.generate_chat([SYSTEM, FIRST], PARAMS, repeat_id="r0")

    backend = transport.responses()
    if source == "receipt":
        parse = backend._parse_response
        backend._parse_response = lambda *a: (_ for _ in ()).throw(ValueError("interrupted"))
        with _store(tmp_path) as store:
            with pytest.raises(PartialBatchError):
                send(PersistentBackend(backend, store))
        backend._parse_response = parse
    with _store(tmp_path) as store:
        result = send(PersistentBackend(backend, store))
    assert result.metadata["persistence"]["source"] == source

    if call == "compaction":
        [turn] = result.messages
        assert turn.reasoning_details == tuple(served)
        backend.generate_chat([turn, SECOND], PARAMS)
    else:
        assert result.metadata["reasoning_details"] == served
        backend.generate_chat([SYSTEM, FIRST, _turn(result), SECOND], PARAMS)

    sent = transport.requests[-1]["input"]
    assert sent[-1 - len(served) : -1] == served


def test_a_refusal_is_kept_and_replayed_without_another_call(transport, tmp_path):
    from llenvs.inference import PersistentBackend

    backend = transport.responses()
    transport.replies[:] = [
        (200, lambda body: _response(body, output=[_reasoning_item(), _refusal_item()]))
    ]
    for _ in range(2):
        with _store(tmp_path) as store:
            with pytest.raises(PartialBatchError) as caught:
                PersistentBackend(backend, store).generate_chat([FIRST], PARAMS, repeat_id="r0")
            assert isinstance(caught.value.failures[0], RefusedByPolicyError)

    assert len(transport.requests) == 1
