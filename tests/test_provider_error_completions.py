"""HTTP-success responses must not hide explicit provider failures.

SDK endpoints are mocked; real backend conversion and batch handling are used.
No external requests, model downloads, or credentials are needed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import BadRequestError

from llenvs.inference.backends.api import OpenRouterBackend
from llenvs.inference.protocol import (
    ChatMessage,
    MalformedResponseError,
    PartialBatchError,
    PromptTooLongError,
    RecoverableInputError,
    SamplingParams,
)


def completion(text="<answer>0.5</answer>", finish="stop", *, error=None, location="top"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=[]),
        finish_reason=finish,
        native_finish_reason=finish,
        logprobs=None,
    )
    result = SimpleNamespace(
        choices=[choice],
        model="test-model",
        id="gen-test",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )
    if error is not None:
        setattr(choice if location == "choice" else result, "error", error)
    return result


@pytest.fixture
def backend():
    backend = OpenRouterBackend(
        model="test-model", api_key="dummy", max_retries=0, rate_limit_wait=0, max_concurrency=4
    )
    yield backend
    backend.close()


MESSAGES = [ChatMessage(role="user", content="Return a value in answer tags.")]
PARAMS = SamplingParams(max_tokens=8192)


@pytest.mark.parametrize("operation", ["chat", "batch", "tools", "tools_batch"])
@pytest.mark.parametrize("envelope", ["http", "top", "choice", "no_choices"])
@pytest.mark.parametrize(
    "message,error_type",
    [
        ("maximum context length exceeded", PromptTooLongError),
        ("invalid image input", RecoverableInputError),
    ],
)
def test_input_error_normalization_matches_http_and_body(
    backend,
    monkeypatch,
    operation,
    envelope,
    message,
    error_type,
):
    payload = {"code": 400, "message": message}
    if envelope == "http":
        error = BadRequestError(
            message,
            response=httpx.Response(400, request=httpx.Request("POST", "https://offline.invalid")),
            body=payload,
        )
        sync, asynchronous = Mock(side_effect=error), AsyncMock(side_effect=error)
    else:
        response = completion(error=payload, location="choice" if envelope == "choice" else "top")
        if envelope == "no_choices":
            response.choices = []
        sync, asynchronous = Mock(return_value=response), AsyncMock(return_value=response)
    monkeypatch.setattr(backend._client.chat.completions, "create", sync)
    monkeypatch.setattr(backend._async_client.chat.completions, "create", asynchronous)
    with pytest.raises(PartialBatchError if "batch" in operation else error_type) as caught:
        if operation == "chat":
            backend.generate_chat(MESSAGES, PARAMS)
        elif operation == "batch":
            backend.generate_chat_batch([MESSAGES], PARAMS)
        elif operation == "tools":
            backend.generate_with_tools(MESSAGES, [], PARAMS)
        else:
            backend.generate_with_tools_batch([MESSAGES], [], PARAMS)
    error = caught.value.failures[0] if "batch" in operation else caught.value
    assert isinstance(error, error_type)
    assert error.offending_indices == [0]
    assert message in str(error)
    assert sync.call_count + asynchronous.call_count == 1


@pytest.mark.parametrize(
    "status,message",
    [
        (401, "invalid API key; maximum context length"),
        (402, "insufficient credits; maximum context length"),
        (403, "permission denied; invalid image"),
        (400, "content policy violation: maximum context length"),
        (400, "unrecognized invalid request"),
    ],
)
@pytest.mark.parametrize("location", ["top", "choice"])
def test_body_errors_do_not_broaden_safe_input_skipping(
    backend, monkeypatch, status, message, location
):
    payload = {"code": status, "message": message}
    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        Mock(return_value=completion(error=payload, location=location)),
    )
    with pytest.raises(MalformedResponseError) as caught:
        backend.generate_chat(MESSAGES, PARAMS)
    assert caught.value.status_code == status
    assert caught.value.provider_error == payload
    assert not isinstance(caught.value, RecoverableInputError)


@pytest.mark.parametrize("operation", ["chat", "batch", "tools", "tools_batch"])
@pytest.mark.parametrize("text", ["", "unfinished explanation", "<answer>0.5</answer>"])
def test_explicit_error_is_never_an_ordinary_result(backend, monkeypatch, operation, text):
    response = completion(text, "error")
    monkeypatch.setattr(backend._client.chat.completions, "create", Mock(return_value=response))
    monkeypatch.setattr(
        backend._async_client.chat.completions, "create", AsyncMock(return_value=response)
    )
    with pytest.raises(Exception) as caught:
        if operation == "chat":
            backend.generate_chat(MESSAGES, PARAMS)
        elif operation == "batch":
            backend.generate_chat_batch([MESSAGES], PARAMS)
        elif operation == "tools":
            backend.generate_with_tools(MESSAGES, [], PARAMS)
        else:
            backend.generate_with_tools_batch([MESSAGES], [], PARAMS)
    exc = caught.value
    if "batch" in operation:
        assert isinstance(exc, PartialBatchError)
        assert set(exc.failures) == {0}
        exc = exc.failures[0]
    assert "error" in str(exc).lower()
    assert getattr(exc, "model_name", None) == "test-model"


@pytest.mark.parametrize("location", ["top", "choice"])
@pytest.mark.parametrize("code", [400, 401, 402, 403, 429, 502, 503])
def test_provider_error_payload_survives_even_with_choices(backend, monkeypatch, location, code):
    payload = {"code": code, "message": "provider rejected this request"}
    # Even a superficially successful finish/text must not override an error body.
    response = completion(error=payload, location=location)
    monkeypatch.setattr(backend._client.chat.completions, "create", Mock(return_value=response))
    with pytest.raises(Exception) as caught:
        backend.generate_chat(MESSAGES, PARAMS)
    assert getattr(caught.value, "provider_error", None) == payload
    assert getattr(caught.value, "status_code", None) == code


def test_mixed_batch_retains_good_siblings_and_original_indices(backend, monkeypatch):
    async def create(**kwargs):
        text = kwargs["messages"][0]["content"]
        return completion(text, "error" if text == "bad" else "stop")

    monkeypatch.setattr(backend._async_client.chat.completions, "create", create)
    prompts = [[ChatMessage(role="user", content=text)] for text in ["left", "bad", "right"]]
    with pytest.raises(PartialBatchError) as caught:
        backend.generate_chat_batch(prompts, PARAMS)
    assert set(caught.value.failures) == {1}
    assert caught.value.results[0].text == "left"
    assert caught.value.results[2].text == "right"


@pytest.mark.parametrize("finish,text", [("stop", "<answer>0.5</answer>"), ("length", "")])
def test_normal_and_token_limited_results_keep_their_meaning(backend, monkeypatch, finish, text):
    monkeypatch.setattr(
        backend._client.chat.completions, "create", Mock(return_value=completion(text, finish))
    )
    result = backend.generate_chat(MESSAGES, PARAMS)
    assert result.text == text
    assert result.finish_reason.name == ("MAX_TOKENS" if finish == "length" else "STOP_SEQUENCE")
