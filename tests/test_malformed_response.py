"""Contract: backends raise ``MalformedResponseError`` on HTTP-200 responses
with missing ``choices``.

OpenRouter routes to multiple upstream providers and sometimes returns a
200 OK response whose body has ``choices: null`` (or an empty list) plus
an ``error`` field describing the upstream failure — e.g., when the
selected provider was rate-limited, timed out, or declined the request.
The SDK parses this into a response object with ``response.choices=None``.

Previously, ``_chat_result`` indexed ``response.choices[0]`` unchecked,
turning this into a raw ``TypeError: 'NoneType' object is not
subscriptable`` that bubbled up past the retry layer and aborted the
whole batch. These tests pin down the new contract: any such response
must raise a typed ``MalformedResponseError`` carrying the provider's
error payload so callers can retry or fail cleanly.

Also covers Fix 2: errors raised *from inside* ``_chat_result`` must go
through ``_normalize_provider_error`` (like SDK-layer errors already do),
so a ``status=400`` context-limit response surfaced post-parse is
rewritten to ``PromptTooLongError``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llenvs.inference.backends.api import OpenRouterBackend
from llenvs.inference.protocol import (
    ChatMessage,
    MalformedResponseError,
    PartialBatchError,
    PromptTooLongError,
    SamplingParams,
)
from openai import APIStatusError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openrouter_backend(monkeypatch: pytest.MonkeyPatch) -> OpenRouterBackend:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    backend = OpenRouterBackend(model="openrouter/stub-model")
    backend._client = MagicMock()
    backend._async_client = MagicMock()
    return backend


def _no_choices_response(
    *, choices: object, error: object | None = None
) -> SimpleNamespace:
    """Build a mocked OpenAI ChatCompletion-shaped response with a missing
    or empty ``choices`` field.

    ``error`` mirrors OpenRouter's extra top-level ``error`` field when an
    upstream provider failed but the HTTP call itself succeeded.
    """
    return SimpleNamespace(
        choices=choices,
        error=error,
        model="openrouter/stub-model",
        id="chatcmpl-stub",
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
    )


def _valid_response() -> SimpleNamespace:
    """A well-formed response — used when we want the SDK call to succeed
    but ``_chat_result`` to raise for other reasons."""
    message = SimpleNamespace(content="ok")
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    return SimpleNamespace(
        choices=[choice],
        model="openrouter/stub-model",
        id="chatcmpl-stub",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


_DUMMY_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _make_status_error(status_code: int, message: str) -> APIStatusError:
    response = httpx.Response(status_code, request=_DUMMY_REQUEST)
    return APIStatusError(message, response=response, body={"error": {"message": message}})


# ---------------------------------------------------------------------------
# Fix 1: choices=None / [] must raise MalformedResponseError (sync)
# ---------------------------------------------------------------------------


class TestMalformedResponseSync:
    def test_raises_when_choices_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._client.chat.completions.create = MagicMock(
            return_value=_no_choices_response(choices=None)
        )

        with pytest.raises(MalformedResponseError):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )

    def test_raises_when_choices_is_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._client.chat.completions.create = MagicMock(
            return_value=_no_choices_response(choices=[])
        )

        with pytest.raises(MalformedResponseError):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )

    def test_surfaces_openrouter_error_body_in_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        err = {
            "code": "provider_error",
            "message": "Upstream provider rate-limited",
        }
        backend._client.chat.completions.create = MagicMock(
            return_value=_no_choices_response(choices=None, error=err)
        )

        with pytest.raises(MalformedResponseError) as exc_info:
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )
        text = str(exc_info.value)
        assert "Upstream provider rate-limited" in text
        assert "provider_error" in text

    def test_exception_carries_backend_and_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._client.chat.completions.create = MagicMock(
            return_value=_no_choices_response(choices=None)
        )

        with pytest.raises(MalformedResponseError) as exc_info:
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )
        assert exc_info.value.backend_name == "OpenRouterBackend"
        assert exc_info.value.model_name == "openrouter/stub-model"


# ---------------------------------------------------------------------------
# Fix 1: choices=None must also surface through the concurrent batch path
# ---------------------------------------------------------------------------


class TestMalformedResponseAsync:
    def test_batch_surfaces_malformed_in_partial_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        async_create = AsyncMock(return_value=_no_choices_response(choices=None))
        backend._async_client.chat.completions.create = async_create

        with pytest.raises(PartialBatchError) as exc_info:
            backend.generate_chat_batch(
                [[ChatMessage(role="user", content="hi")]], SamplingParams()
            )
        failures = exc_info.value.failures
        assert len(failures) == 1
        failure = next(iter(failures.values()))
        assert isinstance(failure, MalformedResponseError)
        assert failure.backend_name == "OpenRouterBackend"
        assert failure.model_name == "openrouter/stub-model"


# ---------------------------------------------------------------------------
# Fix 2: errors raised from inside _chat_result must be normalized
# ---------------------------------------------------------------------------


class TestChatResultErrorsAreNormalized:
    def test_status_400_context_error_from_chat_result_becomes_prompt_too_long(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK returns a valid response, but ``_chat_result`` raises an
        SDK-style error. With fix 2, this goes through
        ``_normalize_provider_error`` and comes out as ``PromptTooLongError``.
        Without fix 2, the raw ``APIStatusError`` would escape.
        """
        backend = _openrouter_backend(monkeypatch)
        backend._client.chat.completions.create = MagicMock(
            return_value=_valid_response()
        )

        def _raising_chat_result(response: object) -> object:
            raise _make_status_error(400, "maximum context length exceeded")

        monkeypatch.setattr(backend, "_chat_result", _raising_chat_result)

        with pytest.raises(PromptTooLongError):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )
