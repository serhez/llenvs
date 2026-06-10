"""LiteLLM backend: unified access to many providers via the litellm SDK.

The ``litellm`` import is deferred into methods — it is an optional
dependency and is slow to import, so ``import llenvs`` must not pay for
it. Python's absolute imports guarantee ``import litellm`` here resolves
to the installed package, not this module (same pattern as
``backends/vllm.py``).
"""

import json
import logging
import time
from typing import Any

from llenvs.core.tools import ToolCall, ToolDefinition
from llenvs.inference.backends.api import (
    _format_no_choices_message,
    _get_field,
    _is_rate_limit_malformed,
    _normalize_provider_error,
    _openai_stop_reason,
    _run_concurrent,
    _to_plain_mapping,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    LogprobsNotReturnedError,
    MalformedResponseError,
    ModelBackend,
    PromptTooLongError,
    RefusedByPolicyError,
    SamplingParams,
    TokenLogprob,
)

logger = logging.getLogger(__name__)


class LiteLLMBackend(ModelBackend):
    """LiteLLM API backend.

    Routes requests through the ``litellm`` SDK, providing unified access
    to many providers via one interface. Model strings use litellm's
    ``provider/model`` format, e.g. ``"gemini/gemini-2.5-flash"``,
    ``"bedrock/anthropic.claude-sonnet-4"``, or
    ``"litellm_proxy/<model>"`` for a LiteLLM proxy server accessed with
    a virtual key. Capabilities vary by underlying model.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        api_key: str | None = None,
        api_base: str | None = None,
        max_concurrency: int = 64,
        rate_limit_wait: float = 0.0,
        rate_limit_max_retries: int = 2,
        timeout: float | None = None,
        num_retries: int | None = None,
        drop_params: bool = True,
        extra_headers: dict[str, str] | None = None,
        **completion_kwargs: Any,
    ) -> None:
        """Initialize LiteLLM backend.

        Args:
            model: Model name in litellm's ``provider/model`` format.
            api_key: Explicit API key. When ``None``, litellm reads the
                provider's native environment variable (``GEMINI_API_KEY``,
                ``ANTHROPIC_API_KEY``, ``LITELLM_PROXY_API_KEY``, ...).
            api_base: Custom endpoint URL — e.g. a LiteLLM proxy server.
            max_concurrency: Maximum concurrent requests for batch generation.
            rate_limit_wait: Seconds to wait before retrying after a
                rate-limit error. When set to ``0`` (the default) no
                backend-level retry happens. The retry path covers both
                ``litellm.RateLimitError`` and ``MalformedResponseError``
                whose provider message is rate-limit-shaped (some providers
                surface their own 429 as an HTTP-200 with ``choices=null``).
            rate_limit_max_retries: Maximum number of rate-limit retries
                before giving up and re-raising the error.
            timeout: Per-request timeout in seconds, passed to litellm.
            num_retries: litellm's in-SDK retries for transient errors.
                ``None`` (the default) disables them — the evaluation
                runner already retries transient failures per slot, and
                stacking both obscures failure accounting.
            drop_params: Drop provider-unsupported request params instead
                of erroring (passed per-call, never via the global
                ``litellm.drop_params``). Logprobs keep a hard guarantee:
                a dropped/ignored logprobs request still raises
                ``LogprobsNotReturnedError`` when no logprobs come back.
            extra_headers: Extra HTTP headers sent with every request.
            **completion_kwargs: Additional kwargs forwarded verbatim on
                every ``litellm.completion`` call.

        Raises:
            ImportError: If the litellm package is not installed.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMBackend. Install with: pip install litellm"
            ) from e

        # Tame litellm's process-global noise. Behavior-relevant config
        # (drop_params) stays per-call so nothing leaks across instances.
        litellm.suppress_debug_info = True
        litellm.telemetry = False
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._max_concurrency = max_concurrency
        self._rate_limit_wait = rate_limit_wait
        self._rate_limit_max_retries = rate_limit_max_retries
        self._timeout = timeout
        self._num_retries = num_retries
        self._drop_params = drop_params
        self._extra_headers = dict(extra_headers) if extra_headers else None
        self._completion_kwargs = completion_kwargs
        self._return_partial_batch = True
        self._closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        """LiteLLM capabilities.

        Optimistic/static, like OpenRouter: whether a given model actually
        supports tools, vision, or logprobs is provider-dependent and
        surfaces as a typed call-time error (e.g.
        ``LogprobsNotReturnedError``) rather than a capability probe —
        litellm's model registry can be stale and knows nothing about
        models behind a proxy.
        """
        return BackendCapabilities(
            supports_logprobs=True,
            supports_prefix_continuation=False,
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=True,
            supports_vision=True,
            max_batch_size=None,
            max_context_length=None,  # Varies by model
            max_concurrency=self._max_concurrency,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    def close(self) -> None:
        """Mark the backend closed.

        litellm's module-level functions share global client pools across
        all instances — there is no per-instance session to release.
        """
        self._closed = True

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate completions for text prompts."""
        results = []
        for prompt in prompts:
            messages = [ChatMessage(role="user", content=prompt)]
            results.append(self.generate_chat(messages, params))
        return results

    def _request_kwargs(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Build the kwargs dict for a ``litellm.completion`` call.

        Shared by the chat and tools paths, so thinking budgets and
        logprobs apply to both.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
            "drop_params": self._drop_params,
        }

        if params.top_k > 0:
            kwargs["top_k"] = params.top_k
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        if params.logprobs:
            # No provider-side cap pre-check (caps vary per provider);
            # absence is caught post-hoc by _ensure_logprobs_if_requested.
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = params.num_logprobs

        if tools is not None:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]
            if tool_choice in ("auto", "none", "required"):
                kwargs["tool_choice"] = tool_choice
            else:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        # litellm's cross-provider budget knob (Anthropic ``thinking``,
        # Gemini thinking budget, ...). Providers without a budget concept
        # drop it via ``drop_params``.
        if params.thinking_budget is not None and not params.disable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.thinking_budget,
            }

        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._num_retries is not None:
            kwargs["num_retries"] = self._num_retries
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers
        kwargs.update(self._completion_kwargs)

        if params.extra:
            kwargs.update(params.extra)

        # Applied after user extras so an explicit disable always wins.
        # ``"none"`` is the cross-provider disable token: Anthropic's
        # reasoning_effort mapper rejects ``"disable"`` while accepting
        # ``"none"``; Gemini accepts both.
        if params.disable_thinking:
            kwargs["reasoning_effort"] = "none"
            kwargs.pop("thinking", None)

        return kwargs

    def _parse_response(self, response: Any) -> GenerationResult:
        """Convert a litellm ModelResponse (OpenAI format) to GenerationResult."""
        choices = getattr(response, "choices", None)
        if not choices:
            provider_error = getattr(response, "error", None)
            raise MalformedResponseError(
                _format_no_choices_message(provider_error),
                backend_name="LiteLLMBackend",
                model_name=self._model,
                provider_error=provider_error,
            )
        choice = choices[0]
        message = choice.message

        token_logprobs = None
        if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
            logprobs: list[TokenLogprob] = []
            for token_info in choice.logprobs.content:
                top_lps: dict[str, float] | None = None
                if token_info.top_logprobs:
                    top_lps = {lp.token: lp.logprob for lp in token_info.top_logprobs}
                logprobs.append(
                    TokenLogprob(
                        token=token_info.token,
                        token_id=0,  # OpenAI-format responses carry no token IDs
                        logprob=token_info.logprob,
                        top_logprobs=top_lps,
                    )
                )
            token_logprobs = tuple(logprobs)

        tool_calls: tuple[ToolCall, ...] = ()
        if getattr(message, "tool_calls", None):
            parsed_calls = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}
                parsed_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )
            tool_calls = tuple(parsed_calls)

        usage = getattr(response, "usage", None)
        metadata: dict[str, Any] = {
            "model": response.model,
            "id": response.id,
        }

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason

        completion_tokens_details = _to_plain_mapping(
            _get_field(usage, "completion_tokens_details")
        )
        if completion_tokens_details:
            metadata["completion_tokens_details"] = completion_tokens_details
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens")
            if reasoning_tokens is not None:
                metadata["reasoning_tokens"] = reasoning_tokens

        # ``reasoning_content`` is litellm's normalized field; check
        # ``reasoning`` first for proxy passthrough responses.
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None:
            reasoning_text = str(reasoning)
            metadata["reasoning_present"] = bool(reasoning_text)
            metadata["reasoning_chars"] = len(reasoning_text)

        hidden_params = getattr(response, "_hidden_params", None)
        if hidden_params is not None:
            response_cost = _get_field(hidden_params, "response_cost")
            if response_cost is not None:
                metadata["response_cost"] = response_cost

        text = message.content
        if text is None and not tool_calls:
            text = ""

        return GenerationResult(
            text=text,
            finish_reason=_openai_stop_reason(getattr(choice, "finish_reason", None)),
            tool_calls=tool_calls,
            token_logprobs=token_logprobs,
            prompt_tokens=_get_field(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=(_get_field(usage, "completion_tokens", 0) if usage else 0),
            metadata=metadata,
        )

    def _ensure_logprobs_if_requested(
        self,
        result: GenerationResult,
        params: SamplingParams,
    ) -> GenerationResult:
        """Raise ``LogprobsNotReturnedError`` if logprobs were asked for but absent."""
        if params.logprobs and not result.token_logprobs:
            raise LogprobsNotReturnedError(
                "LiteLLM returned no token logprobs despite logprobs=True. "
                "The underlying provider likely does not expose logprobs "
                f"for model {self._model!r}, or drop_params removed the "
                "request fields.",
                backend_name="LiteLLMBackend",
                model_name=self._model,
            )
        return result

    def _normalize_error(self, exc: BaseException) -> BaseException:
        """Map litellm exceptions to llenvs error types.

        litellm sub-types the interesting 400s, so the typed checks come
        first; remaining 400s go through api.py's text heuristics. Errors
        the runner's transient classifier already understands (timeouts,
        connection errors, 5xx, auth) pass through unchanged.
        """
        import litellm

        if isinstance(exc, litellm.ContextWindowExceededError):
            return PromptTooLongError(
                str(exc),
                model_name=self._model,
                offending_indices=[0],
            )
        if isinstance(exc, litellm.ContentPolicyViolationError):
            return RefusedByPolicyError(str(exc), offending_indices=[0])
        return _normalize_provider_error(exc, model_name=self._model)

    def _completion_sync(
        self,
        kwargs: dict[str, Any],
        params: SamplingParams,
    ) -> GenerationResult:
        """Call ``litellm.completion`` with error normalization and
        rate-limit retries."""
        import litellm

        last_exc: BaseException | None = None
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                try:
                    response = litellm.completion(**kwargs)
                    return self._ensure_logprobs_if_requested(
                        self._parse_response(response), params
                    )
                except (litellm.RateLimitError, MalformedResponseError):
                    raise
                except Exception as exc:
                    normalized = self._normalize_error(exc)
                    if normalized is not exc:
                        raise normalized from exc
                    raise
            except (litellm.RateLimitError, MalformedResponseError) as exc:
                if isinstance(exc, MalformedResponseError) and not _is_rate_limit_malformed(exc):
                    raise
                if self._rate_limit_wait <= 0 or attempt == self._rate_limit_max_retries:
                    raise
                last_exc = exc
                logger.warning(
                    "Rate limited (%s), waiting %.0fs (attempt %d/%d)",
                    type(exc).__name__,
                    self._rate_limit_wait,
                    attempt + 1,
                    self._rate_limit_max_retries,
                )
                time.sleep(self._rate_limit_wait)

        raise last_exc  # unreachable, but satisfies type checker

    async def _completion_async(
        self,
        kwargs: dict[str, Any],
        params: SamplingParams,
    ) -> GenerationResult:
        """Async variant of ``_completion_sync`` for concurrent batches."""
        import asyncio

        import litellm

        last_exc: BaseException | None = None
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                try:
                    response = await litellm.acompletion(**kwargs)
                    return self._ensure_logprobs_if_requested(
                        self._parse_response(response), params
                    )
                except (litellm.RateLimitError, MalformedResponseError):
                    raise
                except Exception as exc:
                    normalized = self._normalize_error(exc)
                    if normalized is not exc:
                        raise normalized from exc
                    raise
            except (litellm.RateLimitError, MalformedResponseError) as exc:
                if isinstance(exc, MalformedResponseError) and not _is_rate_limit_malformed(exc):
                    raise
                if self._rate_limit_wait <= 0 or attempt == self._rate_limit_max_retries:
                    raise
                last_exc = exc
                logger.warning(
                    "Rate limited (%s), waiting %.0fs (attempt %d/%d)",
                    type(exc).__name__,
                    self._rate_limit_wait,
                    attempt + 1,
                    self._rate_limit_max_retries,
                )
                await asyncio.sleep(self._rate_limit_wait)

        raise last_exc  # unreachable, but satisfies type checker

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation."""
        return self._completion_sync(self._request_kwargs(messages, params), params)

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Async version of generate_chat for concurrent batch execution."""
        return await self._completion_async(self._request_kwargs(messages, params), params)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate responses for multiple conversations concurrently."""
        return _run_concurrent(
            lambda msgs: self._generate_chat_async(msgs, params),
            messages_batch,
            self._max_concurrency,
            return_partial=getattr(self, "_return_partial_batch", True),
        )

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Generate a response with tool calling capability."""
        kwargs = self._request_kwargs(messages, params, tools=tools, tool_choice=tool_choice)
        return self._completion_sync(kwargs, params)

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Async version of generate_with_tools for concurrent batch execution."""
        kwargs = self._request_kwargs(messages, params, tools=tools, tool_choice=tool_choice)
        return await self._completion_async(kwargs, params)

    def generate_with_tools_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> list[GenerationResult]:
        """Generate tool-calling responses for multiple conversations concurrently."""
        return _run_concurrent(
            lambda msgs: self._generate_with_tools_async(msgs, tools, params, tool_choice),
            messages_batch,
            self._max_concurrency,
            return_partial=getattr(self, "_return_partial_batch", True),
        )
