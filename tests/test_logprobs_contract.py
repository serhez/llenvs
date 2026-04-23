"""Cross-backend tests for the logprobs request/response contract.

Every backend that declares ``supports_logprobs=True`` must honour this
rule: when ``params.logprobs=True``, either return
``GenerationResult.token_logprobs`` populated with ``TokenLogprob`` entries,
or raise ``LogprobsNotReturnedError``. These tests verify that contract
for the two cloud backends (OpenAI and OpenRouter), plus the
OpenRouter-specific ``top_logprobs`` range enforcement.

For vLLM and HuggingFace, the guard is defensive — both local backends
set the upstream flag correctly — and coverage is implicit in the
existing logprob tests in ``test_vllm_backend.py`` and
``test_huggingface_backend.py``. Adding a guard unit test for them would
require mocking the full framework-specific pipeline, which is more
churn than the risk warrants.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenvs.inference.backends.api import OpenAIBackend, OpenRouterBackend
from llenvs.inference.protocol import (
    ChatMessage,
    LogprobsNotReturnedError,
    SamplingParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_completion(
    *,
    content: str = "hello",
    finish_reason: str = "stop",
    logprobs: object | None = None,
    model: str = "test-model",
    usage_prompt: int = 1,
    usage_completion: int = 1,
) -> SimpleNamespace:
    """Build a mocked OpenAI ChatCompletion-shaped response.

    Matches the shape the OpenAI SDK returns (and OpenRouter mirrors) —
    ``choices[0].{message.content, finish_reason, logprobs}``, plus
    ``model``, ``id``, ``usage``.
    """
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )
    usage = SimpleNamespace(
        prompt_tokens=usage_prompt, completion_tokens=usage_completion
    )
    return SimpleNamespace(
        choices=[choice],
        model=model,
        id="chatcmpl-stub",
        usage=usage,
    )


def _token_info(token: str, logprob: float, top: list[tuple[str, float]]) -> SimpleNamespace:
    """Build a ChatTokenLogprob-shaped entry.

    Matches the OpenAPI schema at openrouter.ai/openapi.json — each
    content item carries ``token``, ``logprob``, ``top_logprobs[]`` where
    each top entry has ``token`` and ``logprob``.
    """
    return SimpleNamespace(
        token=token,
        logprob=logprob,
        top_logprobs=[SimpleNamespace(token=t, logprob=lp) for t, lp in top],
    )


def _logprobs_block(entries: list[SimpleNamespace]) -> SimpleNamespace:
    """Shape of ``choice.logprobs`` in an OpenAI/OpenRouter response."""
    return SimpleNamespace(content=entries)


def _openrouter_backend(monkeypatch: pytest.MonkeyPatch) -> OpenRouterBackend:
    """Construct an ``OpenRouterBackend`` without touching real OpenAI SDK."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    backend = OpenRouterBackend(model="openrouter/stub-model")
    backend._client = MagicMock()
    backend._async_client = MagicMock()
    return backend


def _openai_backend(monkeypatch: pytest.MonkeyPatch) -> OpenAIBackend:
    """Construct an ``OpenAIBackend`` without touching real OpenAI SDK."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAIBackend(model="openai/stub-model")
    backend._client = MagicMock()
    backend._async_client = MagicMock()
    return backend


# ---------------------------------------------------------------------------
# OpenRouter: parsing
# ---------------------------------------------------------------------------


class TestOpenRouterLogprobParsing:
    def test_populates_token_logprobs_when_response_has_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        entries = [
            _token_info("A", -0.1, [("A", -0.1), ("B", -2.3)]),
            _token_info(" B", -0.4, [(" B", -0.4)]),
        ]
        response = _chat_completion(
            content="A B",
            logprobs=_logprobs_block(entries),
        )
        backend._client.chat.completions.create = MagicMock(return_value=response)

        result = backend.generate_chat(
            [ChatMessage(role="user", content="go")],
            SamplingParams(logprobs=True, num_logprobs=5),
        )

        assert result.token_logprobs is not None
        assert len(result.token_logprobs) == 2
        assert result.token_logprobs[0].token == "A"
        assert result.token_logprobs[0].logprob == pytest.approx(-0.1)
        assert result.token_logprobs[0].top_logprobs == {"A": -0.1, "B": -2.3}

    def test_forwards_logprobs_and_top_logprobs_in_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        response = _chat_completion(
            logprobs=_logprobs_block([_token_info("A", -0.1, [])])
        )
        create = MagicMock(return_value=response)
        backend._client.chat.completions.create = create

        backend.generate_chat(
            [ChatMessage(role="user", content="go")],
            SamplingParams(logprobs=True, num_logprobs=7),
        )

        kwargs = create.call_args.kwargs
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 7

    def test_does_not_send_logprobs_when_not_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        response = _chat_completion()
        create = MagicMock(return_value=response)
        backend._client.chat.completions.create = create

        backend.generate_chat(
            [ChatMessage(role="user", content="go")],
            SamplingParams(),
        )

        kwargs = create.call_args.kwargs
        assert "logprobs" not in kwargs
        assert "top_logprobs" not in kwargs


# ---------------------------------------------------------------------------
# OpenRouter: contract violations
# ---------------------------------------------------------------------------


class TestOpenRouterRaisesWhenLogprobsMissing:
    def test_raises_when_response_has_no_logprobs_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        response = _chat_completion(logprobs=None)
        backend._client.chat.completions.create = MagicMock(return_value=response)

        with pytest.raises(LogprobsNotReturnedError) as exc_info:
            backend.generate_chat(
                [ChatMessage(role="user", content="go")],
                SamplingParams(logprobs=True, num_logprobs=5),
            )
        assert exc_info.value.backend_name == "OpenRouterBackend"
        assert exc_info.value.model_name == "openrouter/stub-model"

    def test_raises_when_logprobs_content_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        response = _chat_completion(logprobs=_logprobs_block([]))
        backend._client.chat.completions.create = MagicMock(return_value=response)

        with pytest.raises(LogprobsNotReturnedError):
            backend.generate_chat(
                [ChatMessage(role="user", content="go")],
                SamplingParams(logprobs=True, num_logprobs=5),
            )


class TestOpenRouterStrictTopLogprobsCap:
    def test_raises_when_num_logprobs_exceeds_20(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)

        with pytest.raises(LogprobsNotReturnedError) as exc_info:
            backend.generate_chat(
                [ChatMessage(role="user", content="go")],
                SamplingParams(logprobs=True, num_logprobs=21),
            )
        assert "caps top_logprobs at 20" in str(exc_info.value)
        assert exc_info.value.backend_name == "OpenRouterBackend"

    def test_allows_exactly_20(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _openrouter_backend(monkeypatch)
        response = _chat_completion(
            logprobs=_logprobs_block([_token_info("A", -0.1, [])])
        )
        create = MagicMock(return_value=response)
        backend._client.chat.completions.create = create

        result = backend.generate_chat(
            [ChatMessage(role="user", content="go")],
            SamplingParams(logprobs=True, num_logprobs=20),
        )

        assert result.token_logprobs is not None
        assert create.call_args.kwargs["top_logprobs"] == 20


# ---------------------------------------------------------------------------
# OpenAI: contract violations
# ---------------------------------------------------------------------------


class TestOpenAIRaisesWhenLogprobsMissing:
    def test_raises_when_response_has_no_logprobs_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openai_backend(monkeypatch)
        response = _chat_completion(logprobs=None)
        backend._client.chat.completions.create = MagicMock(return_value=response)

        with pytest.raises(LogprobsNotReturnedError) as exc_info:
            backend.generate_chat(
                [ChatMessage(role="user", content="go")],
                SamplingParams(logprobs=True, num_logprobs=5),
            )
        assert exc_info.value.backend_name == "OpenAIBackend"
        assert exc_info.value.model_name == "openai/stub-model"

    def test_does_not_raise_when_logprobs_not_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openai_backend(monkeypatch)
        response = _chat_completion(logprobs=None)
        backend._client.chat.completions.create = MagicMock(return_value=response)

        result = backend.generate_chat(
            [ChatMessage(role="user", content="go")],
            SamplingParams(),  # logprobs=False by default
        )
        assert result.token_logprobs is None
