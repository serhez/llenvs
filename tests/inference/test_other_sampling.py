"""Unsupported and hidden controls never silently acquire neutral semantics."""

import pytest

from llenvs.inference.backends.api import AnthropicBackend, OpenRouterBackend
from llenvs.inference.protocol import ChatMessage, SamplingParams

MESSAGES = [ChatMessage("user", "question")]


def test_anthropic_sends_accepted_neutral_controls_and_disables_reasoning():
    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._model = "claude"
    body = backend._request_kwargs(MESSAGES, SamplingParams(reasoning_effort="none"))
    assert body["temperature"] == 0 and body["top_p"] == 1
    assert body["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(
    "values",
    [
        {"n": 2},
        {"min_p": 0.2},
        {"repetition_penalty": 1.2},
        {"presence_penalty": 1},
        {"max_tokens": None},
        {"reasoning_effort": "unknown"},
    ],
)
def test_anthropic_rejects_unapplied_controls(values):
    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._model = "claude"
    with pytest.raises(ValueError):
        backend._request_kwargs(MESSAGES, SamplingParams(**values))


def test_router_effort_and_nullable_controls():
    backend = OpenRouterBackend.__new__(OpenRouterBackend)
    backend._model = "provider/model"
    body = backend._chat_kwargs(
        MESSAGES,
        SamplingParams(
            max_tokens=None,
            temperature=None,
            top_p=None,
            reasoning_effort="high",
            min_p=0.2,
            repetition_penalty=1.2,
            top_k=3,
        ),
    )
    assert "max_tokens" not in body and "temperature" not in body and "top_p" not in body
    assert body["extra_body"]["reasoning"] == {"effort": "high"}
    assert body["extra_body"]["min_p"] == 0.2


def test_router_extra_reasoning_collision_is_rejected():
    backend = OpenRouterBackend.__new__(OpenRouterBackend)
    backend._model = "provider/model"
    with pytest.raises(ValueError):
        backend._chat_kwargs(
            MESSAGES,
            SamplingParams(
                reasoning_effort="none", extra={"extra_body": {"reasoning": {"effort": "high"}}}
            ),
        )


@pytest.mark.parametrize("name", ["CodexCLIBackend", "ClaudeCodeBackend"])
def test_cli_requires_omission_of_hidden_controls(name):
    from llenvs.inference import backends

    instance = getattr(backends, name).__new__(getattr(backends, name))
    with pytest.raises(ValueError):
        instance._validate_params(SamplingParams())
    instance._validate_params(
        SamplingParams(
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
        )
    )


def test_vllm_http_dialect_rejects_unimplemented_local_thinking_budget():
    from llenvs.inference.backends.api import OpenAIBackend

    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend._api_flavor, backend._model = "vllm", "local"
    with pytest.raises(ValueError, match="thinking_budget"):
        backend._request_kwargs(MESSAGES, SamplingParams(thinking_budget=10))


def test_router_retry_owner_disables_hidden_sdk_retries(monkeypatch):
    from unittest.mock import MagicMock

    import openai

    sync, async_ = MagicMock(), MagicMock()
    monkeypatch.setattr(openai, "OpenAI", sync)
    monkeypatch.setattr(openai, "AsyncOpenAI", async_)
    OpenRouterBackend(api_key="offline", rate_limit_wait=0.1)
    assert sync.call_args.kwargs["max_retries"] == 0
    assert async_.call_args.kwargs["max_retries"] == 0
    with pytest.raises(ValueError):
        OpenRouterBackend(api_key="offline", rate_limit_wait=0.1, max_retries=2)
