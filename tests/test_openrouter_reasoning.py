"""OpenRouter reasoning-budget request shape.

OpenRouter's chat-completion API exposes a unified ``reasoning`` block at
the top level of the request body
(`docs <https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>`_).
``reasoning.max_tokens`` is the cross-provider budget knob — it maps to
Anthropic ``thinking.budget_tokens``, Gemini's thinking budget, and
Qwen's ``thinking_budget``; for OpenAI/Grok-style ``reasoning_effort``
models OpenRouter derives an effort bucket from the token count.

These tests cover ``OpenRouterBackend._chat_kwargs`` mapping
``SamplingParams.thinking_budget`` onto ``extra_body.reasoning.max_tokens``
and the merge semantics with caller-supplied ``params.extra``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenvs.inference.backends.api import OpenRouterBackend
from llenvs.inference.protocol import ChatMessage, SamplingParams


def _chat_completion() -> SimpleNamespace:
    """Minimal ChatCompletion-shaped response (no logprobs needed here)."""
    message = SimpleNamespace(content="hi")
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(
        choices=[choice], model="test-model", id="chatcmpl-stub", usage=usage,
    )


def _openrouter_backend(monkeypatch: pytest.MonkeyPatch) -> OpenRouterBackend:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    backend = OpenRouterBackend(model="openrouter/stub-model")
    backend._client = MagicMock()
    backend._async_client = MagicMock()
    return backend


def _capture_kwargs(
    backend: OpenRouterBackend,
    params: SamplingParams,
) -> dict:
    """Drive a single ``generate_chat`` and return the kwargs forwarded
    to ``client.chat.completions.create``."""
    create = MagicMock(return_value=_chat_completion())
    backend._client.chat.completions.create = create
    backend.generate_chat([ChatMessage(role="user", content="go")], params)
    return create.call_args.kwargs


class TestThinkingBudgetMapping:
    """``thinking_budget`` -> ``extra_body.reasoning.max_tokens``."""

    def test_thinking_budget_populates_reasoning_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        kwargs = _capture_kwargs(
            backend, SamplingParams(thinking_budget=2048),
        )
        assert kwargs.get("extra_body") == {"reasoning": {"max_tokens": 2048}}

    def test_no_thinking_budget_omits_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _openrouter_backend(monkeypatch)
        kwargs = _capture_kwargs(backend, SamplingParams())
        assert "extra_body" not in kwargs

    def test_thinking_budget_does_not_set_top_level_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``reasoning`` must go through ``extra_body`` because the OpenAI
        SDK does not know the field at the top level."""
        backend = _openrouter_backend(monkeypatch)
        kwargs = _capture_kwargs(
            backend, SamplingParams(thinking_budget=512),
        )
        assert "reasoning" not in kwargs


class TestExtraMergeSemantics:
    """``params.extra`` interacts cleanly with the derived ``reasoning``."""

    def test_user_extra_body_reasoning_overrides_thinking_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a caller explicitly puts ``reasoning`` in
        ``extra.extra_body``, that escape hatch wins over the derived
        ``reasoning.max_tokens`` mapped from ``thinking_budget``."""
        backend = _openrouter_backend(monkeypatch)
        params = SamplingParams(
            thinking_budget=2048,
            extra={"extra_body": {"reasoning": {"effort": "high"}}},
        )
        kwargs = _capture_kwargs(backend, params)
        assert kwargs["extra_body"] == {"reasoning": {"effort": "high"}}

    def test_user_extra_body_other_keys_coexist_with_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``extra.extra_body.provider`` (provider routing) must coexist
        with the derived ``reasoning`` block — both are top-level keys
        inside ``extra_body``."""
        backend = _openrouter_backend(monkeypatch)
        params = SamplingParams(
            thinking_budget=1024,
            extra={"extra_body": {"provider": {"order": ["anthropic"]}}},
        )
        kwargs = _capture_kwargs(backend, params)
        assert kwargs["extra_body"] == {
            "reasoning": {"max_tokens": 1024},
            "provider": {"order": ["anthropic"]},
        }

    def test_user_extra_top_level_keys_pass_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-``extra_body`` keys in ``params.extra`` keep flowing as
        top-level kwargs to the OpenAI SDK call (existing behavior)."""
        backend = _openrouter_backend(monkeypatch)
        params = SamplingParams(extra={"user": "u-123"})
        kwargs = _capture_kwargs(backend, params)
        assert kwargs.get("user") == "u-123"
        assert "extra_body" not in kwargs

    def test_user_extra_body_alone_without_thinking_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-existing ``extra.extra_body`` plumbing keeps working when
        ``thinking_budget`` is unset."""
        backend = _openrouter_backend(monkeypatch)
        params = SamplingParams(
            extra={"extra_body": {"provider": {"order": ["anthropic"]}}},
        )
        kwargs = _capture_kwargs(backend, params)
        assert kwargs["extra_body"] == {"provider": {"order": ["anthropic"]}}

    def test_disable_thinking_without_other_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``disable_thinking`` explicitly disables OpenRouter reasoning."""
        backend = _openrouter_backend(monkeypatch)
        params = SamplingParams(disable_thinking=True)
        kwargs = _capture_kwargs(backend, params)
        assert kwargs["extra_body"] == {"reasoning": {"effort": "none"}}

    def test_disable_thinking_overrides_reasoning_and_preserves_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-call disable wins over all reasoning knobs, without dropping routing."""
        backend = _openrouter_backend(monkeypatch)
        params = SamplingParams(
            thinking_budget=2048,
            disable_thinking=True,
            extra={
                "extra_body": {
                    "provider": {"order": ["anthropic"]},
                    "reasoning": {"effort": "high"},
                }
            },
        )
        kwargs = _capture_kwargs(backend, params)
        assert kwargs["extra_body"] == {
            "provider": {"order": ["anthropic"]},
            "reasoning": {"effort": "none"},
        }

    def test_default_provider_coexists_with_derived_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backend-level provider routing is preserved with thinking budgets."""
        backend = _openrouter_backend(monkeypatch)
        backend._default_provider = {"order": ["anthropic"]}
        kwargs = _capture_kwargs(backend, SamplingParams(thinking_budget=1024))
        assert kwargs["extra_body"] == {
            "provider": {"order": ["anthropic"]},
            "reasoning": {"max_tokens": 1024},
        }
