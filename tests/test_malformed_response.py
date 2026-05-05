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

from llenvs.inference.backends.api import (
    OpenRouterBackend,
    _is_rate_limit_malformed,
)
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


def _reasoning_response() -> SimpleNamespace:
    """OpenRouter response with hidden-reasoning diagnostics."""
    message = SimpleNamespace(
        content="",
        reasoning="private chain of thought",
        reasoning_details=[
            {"type": "reasoning.text", "text": "step one"},
            {"type": "reasoning.encrypted", "data": "opaque"},
        ],
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason="length",
        native_finish_reason="MAX_TOKENS",
        logprobs=None,
    )
    usage = SimpleNamespace(
        prompt_tokens=9,
        completion_tokens=8192,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=6144),
    )
    return SimpleNamespace(
        choices=[choice],
        model="google/gemma-4-26b-a4b-it",
        id="gen-stub",
        usage=usage,
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


class TestIsRateLimitMalformed:
    """Classification of rate-limit-shaped MalformedResponseError instances.

    Some upstream providers (e.g. Alibaba, observed live on 2026-04-27)
    surface their own 429 as an HTTP-200 with ``choices=null`` plus an
    OpenRouter top-level ``error`` of HTTP 502 wrapping a textual rate-limit
    message. The retry loop only catches ``RateLimitError``, so these slip
    through. ``_is_rate_limit_malformed`` is the helper that classifies
    those cases for the retry path.
    """

    def test_returns_false_for_non_malformed(self) -> None:
        assert _is_rate_limit_malformed(ValueError("rate limit")) is False

    def test_detects_alibaba_rate_increased_too_quickly(self) -> None:
        exc = MalformedResponseError(
            "Provider returned no choices: 502 — Upstream error from Alibaba: "
            "Request rate increased too quickly. To ensure system stability, "
            "please adjust your client logic to scale requests more smoothly "
            "over time.",
            backend_name="OpenRouterBackend",
            model_name="qwen/qwen3.5-27b",
            provider_error={
                "code": 502,
                "message": (
                    "Upstream error from Alibaba: Request rate increased too "
                    "quickly. To ensure system stability, please adjust your "
                    "client logic to scale requests more smoothly over time."
                ),
            },
        )
        assert _is_rate_limit_malformed(exc) is True

    def test_detects_too_many_requests(self) -> None:
        exc = MalformedResponseError(
            "no choices",
            provider_error={"message": "429 Too Many Requests"},
        )
        assert _is_rate_limit_malformed(exc) is True

    def test_does_not_match_top_logprobs_cap_error(self) -> None:
        """The cap-mismatch error from Alibaba must NOT be retried — it's
        a permanent input rejection that retrying would just re-trigger."""
        exc = MalformedResponseError(
            "Provider returned no choices: 502 — Upstream error from Alibaba: "
            "<400> InternalError.Algo.InvalidParameter: Range of top_logprobs "
            "should be [0, 5]",
            provider_error={
                "code": 502,
                "message": (
                    "Upstream error from Alibaba: <400> "
                    "InternalError.Algo.InvalidParameter: Range of top_logprobs "
                    "should be [0, 5]"
                ),
            },
        )
        assert _is_rate_limit_malformed(exc) is False

    def test_does_not_match_unrelated_failure(self) -> None:
        exc = MalformedResponseError(
            "Provider returned no choices: upstream timeout",
            provider_error={"code": 503, "message": "upstream timeout"},
        )
        assert _is_rate_limit_malformed(exc) is False


class TestRateLimitMalformedRetry:
    """OpenRouterBackend retry loop must treat rate-limit-shaped
    MalformedResponseError the same as RateLimitError: sleep and retry up to
    ``rate_limit_max_retries`` times. Non-rate-limit MalformedResponseError
    must still propagate immediately, since retrying a 4xx-equivalent is
    wasteful.
    """

    def _rate_limit_response(self) -> SimpleNamespace:
        err = {
            "code": 502,
            "message": (
                "Upstream error from Alibaba: Request rate increased too "
                "quickly. Please scale requests more smoothly."
            ),
        }
        return _no_choices_response(choices=None, error=err)

    def _cap_mismatch_response(self) -> SimpleNamespace:
        err = {
            "code": 502,
            "message": (
                "Upstream error from Alibaba: <400> "
                "InternalError.Algo.InvalidParameter: Range of top_logprobs "
                "should be [0, 5]"
            ),
        }
        return _no_choices_response(choices=None, error=err)

    def test_sync_retries_until_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._rate_limit_wait = 0.01
        backend._rate_limit_max_retries = 2

        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

        responses = [self._rate_limit_response(), _valid_response()]
        backend._client.chat.completions.create = MagicMock(side_effect=responses)

        result = backend.generate_chat(
            [ChatMessage(role="user", content="hi")], SamplingParams()
        )
        assert result.text == "ok"
        assert backend._client.chat.completions.create.call_count == 2
        assert sleeps == [0.01]

    def test_sync_does_not_retry_non_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._rate_limit_wait = 0.01
        backend._rate_limit_max_retries = 2

        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

        backend._client.chat.completions.create = MagicMock(
            return_value=self._cap_mismatch_response()
        )

        with pytest.raises(MalformedResponseError):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )
        assert backend._client.chat.completions.create.call_count == 1
        assert sleeps == []

    def test_sync_exhausts_retries_then_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._rate_limit_wait = 0.01
        backend._rate_limit_max_retries = 2

        monkeypatch.setattr("time.sleep", lambda s: None)

        backend._client.chat.completions.create = MagicMock(
            return_value=self._rate_limit_response()
        )

        with pytest.raises(MalformedResponseError):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")], SamplingParams()
            )
        # Initial attempt + 2 retries = 3 total calls.
        assert backend._client.chat.completions.create.call_count == 3

    def test_async_retries_until_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._rate_limit_wait = 0.01
        backend._rate_limit_max_retries = 2

        sleeps: list[float] = []

        async def _fake_async_sleep(s: float) -> None:
            sleeps.append(s)

        monkeypatch.setattr("asyncio.sleep", _fake_async_sleep)

        responses = [self._rate_limit_response(), _valid_response()]
        backend._async_client.chat.completions.create = AsyncMock(side_effect=responses)

        # The batch path goes through _generate_chat_async, which has its
        # own copy of the retry loop. One item, one rate-limit response,
        # one success.
        results = backend.generate_chat_batch(
            [[ChatMessage(role="user", content="hi")]], SamplingParams()
        )
        assert len(results) == 1
        assert results[0].text == "ok"
        assert backend._async_client.chat.completions.create.call_count == 2
        assert sleeps == [0.01]

    def test_async_does_not_retry_non_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._rate_limit_wait = 0.01
        backend._rate_limit_max_retries = 2

        sleeps: list[float] = []

        async def _fake_async_sleep(s: float) -> None:
            sleeps.append(s)

        monkeypatch.setattr("asyncio.sleep", _fake_async_sleep)

        backend._async_client.chat.completions.create = AsyncMock(
            return_value=self._cap_mismatch_response()
        )

        with pytest.raises(PartialBatchError) as exc_info:
            backend.generate_chat_batch(
                [[ChatMessage(role="user", content="hi")]], SamplingParams()
            )
        failure = next(iter(exc_info.value.failures.values()))
        assert isinstance(failure, MalformedResponseError)
        assert backend._async_client.chat.completions.create.call_count == 1
        assert sleeps == []


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


class TestOpenRouterReasoningMetadata:
    def test_chat_result_records_finish_and_reasoning_diagnostics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._client.chat.completions.create = MagicMock(
            return_value=_reasoning_response()
        )

        result = backend.generate_chat(
            [ChatMessage(role="user", content="hi")], SamplingParams()
        )

        assert result.text == ""
        assert result.prompt_tokens == 9
        assert result.completion_tokens == 8192
        assert result.metadata == {
            "model": "google/gemma-4-26b-a4b-it",
            "id": "gen-stub",
            "finish_reason": "length",
            "native_finish_reason": "MAX_TOKENS",
            "reasoning_present": True,
            "reasoning_chars": 24,
            "reasoning_details_present": True,
            "reasoning_details_count": 2,
            "reasoning_details_types": ("reasoning.text", "reasoning.encrypted"),
            "completion_tokens_details": {"reasoning_tokens": 6144},
            "reasoning_tokens": 6144,
        }

    def test_chat_result_omits_absent_reasoning_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        backend._client.chat.completions.create = MagicMock(
            return_value=_valid_response()
        )

        result = backend.generate_chat(
            [ChatMessage(role="user", content="hi")], SamplingParams()
        )

        assert result.metadata == {
            "model": "openrouter/stub-model",
            "id": "chatcmpl-stub",
            "finish_reason": "stop",
        }
