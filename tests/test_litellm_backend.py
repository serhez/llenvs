"""Tests for the LiteLLM backend.

The backend calls the module-level ``litellm.completion`` /
``litellm.acompletion`` functions (resolved at call time), so tests
monkeypatch those on the ``litellm`` module — the analog of overwriting
``backend._client`` on the OpenAI-SDK backends. Responses are
``SimpleNamespace`` trees since the parsers are getattr-based.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenvs.core.state import ImageContent
from llenvs.core.tools import ToolDefinition, ToolParameter, ToolParameterType
from llenvs.inference.protocol import (
    ChatMessage,
    LogprobsNotReturnedError,
    MalformedResponseError,
    PartialBatchError,
    PromptTooLongError,
    RefusedByPolicyError,
    SamplingParams,
    StopReason,
)

try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

pytestmark = pytest.mark.skipif(not HAS_LITELLM, reason="litellm not installed")

if HAS_LITELLM:
    from llenvs.inference.backends.litellm import LiteLLMBackend


def _completion_response(
    content: str | None = "hi",
    finish_reason: str = "stop",
    logprobs: SimpleNamespace | None = None,
    tool_calls: list | None = None,
    usage: SimpleNamespace | None = None,
    **response_fields,
) -> SimpleNamespace:
    """Minimal ModelResponse-shaped stub (OpenAI chat-completion format)."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, logprobs=logprobs)
    if usage is None:
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(
        choices=[choice],
        model="stub-model",
        id="chatcmpl-stub",
        usage=usage,
        **response_fields,
    )


def _backend(**kwargs) -> "LiteLLMBackend":
    return LiteLLMBackend(model="gemini/gemini-2.5-flash", **kwargs)


def _capture_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    backend: "LiteLLMBackend",
    params: SamplingParams,
    messages: list[ChatMessage] | None = None,
    response: SimpleNamespace | None = None,
) -> dict:
    """Drive one ``generate_chat`` and return the kwargs sent to
    ``litellm.completion``."""
    create = MagicMock(return_value=response or _completion_response())
    monkeypatch.setattr(litellm, "completion", create)
    backend.generate_chat(messages or [ChatMessage(role="user", content="go")], params)
    return create.call_args.kwargs


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_weather",
            description="Get the weather for a city.",
            parameters=(
                ToolParameter(
                    name="city",
                    type=ToolParameterType.STRING,
                    description="City name",
                    required=True,
                ),
            ),
        )
    ]


class TestConstruction:
    def test_model_name(self):
        backend = _backend()
        assert backend.model_name == "gemini/gemini-2.5-flash"

    def test_capabilities(self):
        backend = _backend(max_concurrency=8)
        caps = backend.capabilities
        assert caps.supports_logprobs is True
        assert caps.supports_batching is True
        assert caps.supports_streaming is True
        assert caps.supports_chat is True
        assert caps.supports_function_calling is True
        assert caps.supports_vision is True
        assert caps.supports_prefix_continuation is False
        assert caps.max_concurrency == 8

    def test_close_is_idempotent(self):
        backend = _backend()
        backend.close()
        backend.close()

    def test_close_safe_on_uninitialized_instance(self):
        backend = LiteLLMBackend.__new__(LiteLLMBackend)
        backend.close()  # must not raise

    def test_context_manager(self):
        with _backend() as backend:
            assert backend.model_name == "gemini/gemini-2.5-flash"


class TestRequestKwargs:
    def test_core_sampling_params_forwarded(self, monkeypatch):
        backend = _backend()
        params = SamplingParams(
            max_tokens=123,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        kwargs = _capture_kwargs(monkeypatch, backend, params)
        assert kwargs["model"] == "gemini/gemini-2.5-flash"
        assert kwargs["messages"] == [{"role": "user", "content": "go"}]
        assert kwargs["max_tokens"] == 123
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["presence_penalty"] == 0.1
        assert kwargs["frequency_penalty"] == 0.2
        assert "n" not in kwargs

    def test_stop_sequences_only_when_set(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert "stop" not in kwargs
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams(stop_sequences=("END",)))
        assert kwargs["stop"] == ["END"]

    def test_top_k_only_when_positive(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert "top_k" not in kwargs
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams(top_k=40))
        assert kwargs["top_k"] == 40

    def test_drop_params_default_true(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert kwargs["drop_params"] is True

    def test_drop_params_can_be_disabled(self, monkeypatch):
        backend = _backend(drop_params=False)
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert kwargs["drop_params"] is False

    def test_connection_config_forwarded_only_when_set(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        for key in ("api_key", "api_base", "timeout", "num_retries", "extra_headers"):
            assert key not in kwargs

        backend = _backend(
            api_key="sk-virtual",
            api_base="http://localhost:4000",
            timeout=30.0,
            num_retries=3,
            extra_headers={"X-Team": "bethgelab"},
        )
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert kwargs["api_key"] == "sk-virtual"
        assert kwargs["api_base"] == "http://localhost:4000"
        assert kwargs["timeout"] == 30.0
        assert kwargs["num_retries"] == 3
        assert kwargs["extra_headers"] == {"X-Team": "bethgelab"}

    def test_constructor_completion_kwargs_ride_along(self, monkeypatch):
        backend = _backend(metadata={"trace_id": "t-1"})
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert kwargs["metadata"] == {"trace_id": "t-1"}

    def test_params_extra_passes_through_top_level(self, monkeypatch):
        backend = _backend()
        params = SamplingParams(extra={"user": "u-123"})
        kwargs = _capture_kwargs(monkeypatch, backend, params)
        assert kwargs["user"] == "u-123"


class TestThinkingMapping:
    def test_thinking_budget_maps_to_thinking_param(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams(thinking_budget=2048))
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "reasoning_effort" not in kwargs

    def test_no_thinking_budget_omits_thinking(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert "thinking" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_disable_thinking_sets_reasoning_effort_none(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams(disable_thinking=True))
        assert kwargs["reasoning_effort"] == "none"
        assert "thinking" not in kwargs

    def test_disable_thinking_wins_over_thinking_budget(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(
            monkeypatch,
            backend,
            SamplingParams(thinking_budget=2048, disable_thinking=True),
        )
        assert kwargs["reasoning_effort"] == "none"
        assert "thinking" not in kwargs

    def test_disable_thinking_wins_over_user_extra(self, monkeypatch):
        backend = _backend()
        params = SamplingParams(disable_thinking=True, extra={"reasoning_effort": "high"})
        kwargs = _capture_kwargs(monkeypatch, backend, params)
        assert kwargs["reasoning_effort"] == "none"

    def test_user_extra_reasoning_effort_passthrough(self, monkeypatch):
        backend = _backend()
        params = SamplingParams(extra={"reasoning_effort": "high"})
        kwargs = _capture_kwargs(monkeypatch, backend, params)
        assert kwargs["reasoning_effort"] == "high"

    def test_user_extra_thinking_overrides_derived(self, monkeypatch):
        backend = _backend()
        params = SamplingParams(
            thinking_budget=2048,
            extra={"thinking": {"type": "enabled", "budget_tokens": 64}},
        )
        kwargs = _capture_kwargs(monkeypatch, backend, params)
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 64}


class TestResponseParsing:
    def test_text_and_finish_reason(self, monkeypatch):
        backend = _backend()
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=_completion_response(content="hello")),
        )
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.text == "hello"
        assert result.finish_reason == StopReason.STOP_SEQUENCE
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.metadata["model"] == "stub-model"
        assert result.metadata["id"] == "chatcmpl-stub"
        assert result.metadata["finish_reason"] == "stop"

    @pytest.mark.parametrize(
        "finish_reason,expected",
        [
            ("stop", StopReason.STOP_SEQUENCE),
            ("length", StopReason.MAX_TOKENS),
            ("tool_calls", StopReason.TOOL_USE),
            ("content_filter", StopReason.ERROR),
        ],
    )
    def test_finish_reason_mapping(self, monkeypatch, finish_reason, expected):
        backend = _backend()
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=_completion_response(finish_reason=finish_reason)),
        )
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.finish_reason == expected

    def test_reasoning_content_metadata(self, monkeypatch):
        backend = _backend()
        response = _completion_response()
        response.choices[0].message.reasoning_content = "let me think..."
        monkeypatch.setattr(litellm, "completion", MagicMock(return_value=response))
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.metadata["reasoning_present"] is True
        assert result.metadata["reasoning_chars"] == len("let me think...")

    def test_reasoning_tokens_metadata(self, monkeypatch):
        backend = _backend()
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=_completion_response(usage=usage)),
        )
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.metadata["reasoning_tokens"] == 3
        assert result.metadata["completion_tokens_details"] == {"reasoning_tokens": 3}

    def test_response_cost_metadata(self, monkeypatch):
        backend = _backend()
        response = _completion_response(_hidden_params={"response_cost": 0.0012})
        monkeypatch.setattr(litellm, "completion", MagicMock(return_value=response))
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.metadata["response_cost"] == 0.0012

    @pytest.mark.parametrize("choices", [None, []])
    def test_missing_choices_raises_malformed(self, monkeypatch, choices):
        backend = _backend()
        response = SimpleNamespace(
            choices=choices,
            error={"code": 429, "message": "upstream exploded"},
            model="stub-model",
            id="chatcmpl-stub",
            usage=None,
        )
        monkeypatch.setattr(litellm, "completion", MagicMock(return_value=response))
        with pytest.raises(MalformedResponseError) as exc_info:
            backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert exc_info.value.backend_name == "LiteLLMBackend"
        assert exc_info.value.model_name == "gemini/gemini-2.5-flash"
        assert exc_info.value.provider_error == {
            "code": 429,
            "message": "upstream exploded",
        }

    def test_generate_wraps_prompts_as_user_messages(self, monkeypatch):
        backend = _backend()
        create = MagicMock(
            side_effect=[
                _completion_response(content="a"),
                _completion_response(content="b"),
            ]
        )
        monkeypatch.setattr(litellm, "completion", create)
        results = backend.generate(["one", "two"], SamplingParams())
        assert [r.text for r in results] == ["a", "b"]
        first_call_messages = create.call_args_list[0].kwargs["messages"]
        assert first_call_messages == [{"role": "user", "content": "one"}]


class TestLogprobs:
    def _logprobs_response(self) -> SimpleNamespace:
        token_info = SimpleNamespace(
            token="hi",
            logprob=-0.1,
            top_logprobs=[
                SimpleNamespace(token="hi", logprob=-0.1),
                SimpleNamespace(token="yo", logprob=-2.3),
            ],
        )
        return _completion_response(logprobs=SimpleNamespace(content=[token_info]))

    def test_logprobs_request_kwargs(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(
            monkeypatch,
            backend,
            SamplingParams(logprobs=True, num_logprobs=7),
            response=self._logprobs_response(),
        )
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 7

    def test_logprobs_parsed(self, monkeypatch):
        backend = _backend()
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=self._logprobs_response()),
        )
        result = backend.generate_chat(
            [ChatMessage(role="user", content="go")],
            SamplingParams(logprobs=True),
        )
        assert result.token_logprobs is not None
        assert len(result.token_logprobs) == 1
        lp = result.token_logprobs[0]
        assert lp.token == "hi"
        assert lp.logprob == -0.1
        assert lp.top_logprobs == {"hi": -0.1, "yo": -2.3}

    def test_logprobs_requested_but_absent_raises(self, monkeypatch):
        backend = _backend()
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=_completion_response()),
        )
        with pytest.raises(LogprobsNotReturnedError) as exc_info:
            backend.generate_chat(
                [ChatMessage(role="user", content="go")],
                SamplingParams(logprobs=True),
            )
        assert exc_info.value.backend_name == "LiteLLMBackend"
        assert exc_info.value.model_name == "gemini/gemini-2.5-flash"

    def test_logprobs_not_requested_no_kwargs(self, monkeypatch):
        backend = _backend()
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams())
        assert "logprobs" not in kwargs
        assert "top_logprobs" not in kwargs


class TestErrorMapping:
    def _raise_and_expect(self, monkeypatch, side_effect, expected):
        backend = _backend()
        monkeypatch.setattr(litellm, "completion", MagicMock(side_effect=side_effect))
        with pytest.raises(expected) as exc_info:
            backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        return exc_info.value

    def test_context_window_exceeded_maps_to_prompt_too_long(self, monkeypatch):
        original = litellm.ContextWindowExceededError(
            message="way too long", model="m", llm_provider="openai"
        )
        error = self._raise_and_expect(monkeypatch, original, PromptTooLongError)
        assert error.model_name == "gemini/gemini-2.5-flash"
        assert error.offending_indices == [0]
        assert error.__cause__ is original

    def test_content_policy_maps_to_refused_by_policy(self, monkeypatch):
        original = litellm.ContentPolicyViolationError(
            message="not allowed", model="m", llm_provider="openai"
        )
        error = self._raise_and_expect(monkeypatch, original, RefusedByPolicyError)
        assert error.offending_indices == [0]
        assert error.__cause__ is original

    def test_rate_limit_reraised_without_wait(self, monkeypatch):
        original = litellm.RateLimitError(message="slow down", llm_provider="openai", model="m")
        self._raise_and_expect(monkeypatch, original, litellm.RateLimitError)

    def test_authentication_error_passes_through(self, monkeypatch):
        original = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="m")
        self._raise_and_expect(monkeypatch, original, litellm.AuthenticationError)

    def test_transient_errors_pass_through(self, monkeypatch):
        for original in (
            litellm.Timeout(message="t", model="m", llm_provider="openai"),
            litellm.InternalServerError(message="ise", llm_provider="openai", model="m"),
            litellm.APIConnectionError(message="conn", llm_provider="openai", model="m"),
        ):
            self._raise_and_expect(monkeypatch, original, type(original))

    def test_bad_request_with_context_text_normalized(self, monkeypatch):
        original = litellm.BadRequestError(
            message="this model's maximum context length is 8192 tokens",
            model="m",
            llm_provider="openai",
        )
        self._raise_and_expect(monkeypatch, original, PromptTooLongError)

    def test_generic_bad_request_passes_through(self, monkeypatch):
        original = litellm.BadRequestError(
            message="unknown parameter", model="m", llm_provider="openai"
        )
        self._raise_and_expect(monkeypatch, original, litellm.BadRequestError)


class TestRateLimitRetry:
    def test_retry_succeeds_after_wait(self, monkeypatch):
        backend = _backend(rate_limit_wait=15.0, rate_limit_max_retries=2)
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", sleeps.append)
        create = MagicMock(
            side_effect=[
                litellm.RateLimitError(message="slow down", llm_provider="openai", model="m"),
                _completion_response(content="recovered"),
            ]
        )
        monkeypatch.setattr(litellm, "completion", create)
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.text == "recovered"
        assert sleeps == [15.0]

    def test_retries_exhausted_reraises(self, monkeypatch):
        backend = _backend(rate_limit_wait=1.0, rate_limit_max_retries=2)
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", sleeps.append)
        create = MagicMock(
            side_effect=litellm.RateLimitError(
                message="slow down", llm_provider="openai", model="m"
            )
        )
        monkeypatch.setattr(litellm, "completion", create)
        with pytest.raises(litellm.RateLimitError):
            backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert create.call_count == 3  # initial + 2 retries
        assert sleeps == [1.0, 1.0]

    def test_no_wait_means_no_retry(self, monkeypatch):
        backend = _backend(rate_limit_wait=0.0)
        create = MagicMock(
            side_effect=litellm.RateLimitError(
                message="slow down", llm_provider="openai", model="m"
            )
        )
        monkeypatch.setattr(litellm, "completion", create)
        with pytest.raises(litellm.RateLimitError):
            backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert create.call_count == 1

    def test_rate_limit_shaped_malformed_is_retried(self, monkeypatch):
        backend = _backend(rate_limit_wait=1.0, rate_limit_max_retries=2)
        monkeypatch.setattr("time.sleep", lambda _: None)
        malformed = SimpleNamespace(
            choices=None,
            error={"code": "429", "message": "rate limit exceeded"},
            model="stub-model",
            id="x",
            usage=None,
        )
        create = MagicMock(side_effect=[malformed, _completion_response(content="recovered")])
        monkeypatch.setattr(litellm, "completion", create)
        result = backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert result.text == "recovered"

    def test_non_rate_limit_malformed_raises_immediately(self, monkeypatch):
        backend = _backend(rate_limit_wait=1.0, rate_limit_max_retries=2)
        malformed = SimpleNamespace(
            choices=None,
            error={"code": "500", "message": "upstream exploded"},
            model="stub-model",
            id="x",
            usage=None,
        )
        create = MagicMock(return_value=malformed)
        monkeypatch.setattr(litellm, "completion", create)
        with pytest.raises(MalformedResponseError):
            backend.generate_chat([ChatMessage(role="user", content="go")], SamplingParams())
        assert create.call_count == 1

    def test_async_retry_in_batch(self, monkeypatch):
        backend = _backend(rate_limit_wait=0.01, rate_limit_max_retries=2)
        calls = {"count": 0}

        async def flaky_acompletion(**kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise litellm.RateLimitError(message="slow down", llm_provider="openai", model="m")
            return _completion_response(content="recovered")

        monkeypatch.setattr(litellm, "acompletion", flaky_acompletion)
        results = backend.generate_chat_batch(
            [[ChatMessage(role="user", content="go")]], SamplingParams()
        )
        assert results[0].text == "recovered"
        assert calls["count"] == 2


class TestBatching:
    def test_batch_preserves_order(self, monkeypatch):
        backend = _backend()

        async def echo_acompletion(**kwargs):
            text = kwargs["messages"][0]["content"]
            return _completion_response(content=f"echo:{text}")

        monkeypatch.setattr(litellm, "acompletion", echo_acompletion)
        batch = [[ChatMessage(role="user", content=str(i))] for i in range(5)]
        results = backend.generate_chat_batch(batch, SamplingParams())
        assert [r.text for r in results] == [f"echo:{i}" for i in range(5)]

    def test_partial_batch_failure(self, monkeypatch):
        backend = _backend()

        async def flaky_acompletion(**kwargs):
            text = kwargs["messages"][0]["content"]
            if text == "bad":
                raise litellm.AuthenticationError(
                    message="bad key", llm_provider="openai", model="m"
                )
            return _completion_response(content=f"echo:{text}")

        monkeypatch.setattr(litellm, "acompletion", flaky_acompletion)
        batch = [
            [ChatMessage(role="user", content="ok")],
            [ChatMessage(role="user", content="bad")],
            [ChatMessage(role="user", content="fine")],
        ]
        with pytest.raises(PartialBatchError) as exc_info:
            backend.generate_chat_batch(batch, SamplingParams())
        error = exc_info.value
        assert list(error.failures) == [1]
        assert error.results[0].text == "echo:ok"
        assert error.results[2].text == "echo:fine"
        assert isinstance(error.results[1], litellm.AuthenticationError)

    def test_concurrency_bounded_by_max_concurrency(self, monkeypatch):
        backend = _backend(max_concurrency=2)
        state = {"current": 0, "peak": 0}

        async def slow_acompletion(**kwargs):
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0.01)
            state["current"] -= 1
            return _completion_response()

        monkeypatch.setattr(litellm, "acompletion", slow_acompletion)
        batch = [[ChatMessage(role="user", content=str(i))] for i in range(6)]
        backend.generate_chat_batch(batch, SamplingParams())
        assert state["peak"] <= 2


class TestTools:
    def _tool_call_response(self, arguments: str) -> SimpleNamespace:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="get_weather", arguments=arguments),
        )
        return _completion_response(
            content=None, finish_reason="tool_calls", tool_calls=[tool_call]
        )

    def test_tools_in_request(self, monkeypatch, sample_tools):
        backend = _backend()
        create = MagicMock(return_value=_completion_response())
        monkeypatch.setattr(litellm, "completion", create)
        backend.generate_with_tools(
            [ChatMessage(role="user", content="weather?")],
            sample_tools,
            SamplingParams(),
        )
        kwargs = create.call_args.kwargs
        assert kwargs["tools"] == [t.to_openai_schema() for t in sample_tools]
        assert kwargs["tool_choice"] == "auto"

    @pytest.mark.parametrize(
        "tool_choice,expected",
        [
            ("auto", "auto"),
            ("none", "none"),
            ("required", "required"),
            (
                "get_weather",
                {"type": "function", "function": {"name": "get_weather"}},
            ),
        ],
    )
    def test_tool_choice_mapping(self, monkeypatch, sample_tools, tool_choice, expected):
        backend = _backend()
        create = MagicMock(return_value=_completion_response())
        monkeypatch.setattr(litellm, "completion", create)
        backend.generate_with_tools(
            [ChatMessage(role="user", content="weather?")],
            sample_tools,
            SamplingParams(),
            tool_choice=tool_choice,
        )
        assert create.call_args.kwargs["tool_choice"] == expected

    def test_tool_calls_parsed(self, monkeypatch, sample_tools):
        backend = _backend()
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=self._tool_call_response('{"city": "Bern"}')),
        )
        result = backend.generate_with_tools(
            [ChatMessage(role="user", content="weather?")],
            sample_tools,
            SamplingParams(),
        )
        assert result.finish_reason == StopReason.TOOL_USE
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.id == "call-1"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Bern"}

    def test_malformed_tool_arguments_fall_back_to_raw(self, monkeypatch, sample_tools):
        backend = _backend()
        monkeypatch.setattr(
            litellm,
            "completion",
            MagicMock(return_value=self._tool_call_response("not json {")),
        )
        result = backend.generate_with_tools(
            [ChatMessage(role="user", content="weather?")],
            sample_tools,
            SamplingParams(),
        )
        assert result.tool_calls[0].arguments == {"raw": "not json {"}

    def test_thinking_budget_flows_into_tools_path(self, monkeypatch, sample_tools):
        backend = _backend()
        create = MagicMock(return_value=_completion_response())
        monkeypatch.setattr(litellm, "completion", create)
        backend.generate_with_tools(
            [ChatMessage(role="user", content="weather?")],
            sample_tools,
            SamplingParams(thinking_budget=1024),
        )
        assert create.call_args.kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 1024,
        }

    def test_tools_batch_preserves_order(self, monkeypatch, sample_tools):
        backend = _backend()

        async def echo_acompletion(**kwargs):
            text = kwargs["messages"][0]["content"]
            return _completion_response(content=f"echo:{text}")

        monkeypatch.setattr(litellm, "acompletion", echo_acompletion)
        batch = [[ChatMessage(role="user", content=str(i))] for i in range(3)]
        results = backend.generate_with_tools_batch(batch, sample_tools, SamplingParams())
        assert [r.text for r in results] == [f"echo:{i}" for i in range(3)]


class TestVision:
    def test_images_serialized_as_image_url_parts(self, monkeypatch):
        backend = _backend()
        message = ChatMessage(
            role="user",
            content="what is this?",
            images=(ImageContent(data="aGk=", media_type="image/png"),),
        )
        kwargs = _capture_kwargs(monkeypatch, backend, SamplingParams(), messages=[message])
        content = kwargs["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "what is this?"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGk="},
        }
