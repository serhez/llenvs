"""API-based backends for OpenAI, Anthropic, and OpenRouter.

Each backend wraps the respective API client with the ModelBackend interface.
"""

import asyncio
import json
from typing import Any

from llenvs.core.tools import ToolCall, ToolDefinition
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    LogprobsNotReturnedError,
    MalformedResponseError,
    ModelBackend,
    PartialBatchError,
    PromptTooLongError,
    RecoverableInputError,
    SamplingParams,
    StopReason,
    TokenLogprob,
)


def _openai_stop_reason(reason: str | None) -> StopReason:
    """Convert OpenAI finish reason to StopReason."""
    if reason is None:
        return StopReason.UNKNOWN
    reason = reason.lower()
    if reason == "stop":
        return StopReason.STOP_SEQUENCE
    elif reason == "length":
        return StopReason.MAX_TOKENS
    elif reason == "tool_calls":
        return StopReason.TOOL_USE
    elif reason == "content_filter":
        return StopReason.ERROR
    return StopReason.UNKNOWN


def _anthropic_stop_reason(reason: str | None) -> StopReason:
    """Convert Anthropic stop reason to StopReason."""
    if reason is None:
        return StopReason.UNKNOWN
    reason = reason.lower()
    if reason == "end_turn":
        return StopReason.END_OF_TEXT
    elif reason == "stop_sequence":
        return StopReason.STOP_SEQUENCE
    elif reason == "max_tokens":
        return StopReason.MAX_TOKENS
    elif reason == "tool_use":
        return StopReason.TOOL_USE
    return StopReason.UNKNOWN


def _error_status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _error_body(error: BaseException) -> Any:
    body = getattr(error, "body", None)
    if body is not None:
        return body
    response = getattr(error, "response", None)
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _flatten_error_text(error: BaseException) -> str:
    parts: list[str] = [str(error)]
    body = _error_body(error)

    def _walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            parts.append(value)
            return
        if isinstance(value, dict):
            for inner in value.values():
                _walk(inner)
            return
        if isinstance(value, (list, tuple)):
            for inner in value:
                _walk(inner)

    _walk(body)
    return " ".join(part for part in parts if part)


def _is_policy_error(error: BaseException) -> bool:
    text = _flatten_error_text(error).lower()
    return any(
        needle in text
        for needle in (
            "content_policy_violation",
            "content policy",
            "safety policy",
            "safety system",
            "policy violation",
        )
    )


def _is_context_limit_error(error: BaseException) -> bool:
    text = _flatten_error_text(error).lower()
    return any(
        needle in text
        for needle in (
            "maximum context length",
            "context length",
            "prompt is too long",
            "input is too long",
            "request is too large",
            "too many tokens",
            "reduce the length",
        )
    )


def _is_invalid_input_error(error: BaseException) -> bool:
    text = _flatten_error_text(error).lower()
    return any(
        needle in text
        for needle in (
            "invalid image",
            "image input",
            "unsupported image",
            "unsupported file type",
            "unsupported media type",
            "invalid base64",
            "image exceeds",
            "maximum allowed size",
            "invalid audio",
            "unsupported audio",
        )
    )


def _format_no_choices_message(provider_error: Any) -> str:
    """Build a diagnostic message for an HTTP-200 response with no choices.

    If the provider returned a top-level ``error`` object (as OpenRouter
    does when routing-time upstream failures can't be surfaced as an HTTP
    error), include its ``code`` and ``message`` so the classifier and
    the operator both see why the request was rejected.
    """
    if isinstance(provider_error, dict):
        code = provider_error.get("code")
        message = provider_error.get("message")
        parts = [p for p in (code, message) if p]
        if parts:
            return "Provider returned no choices: " + " — ".join(str(p) for p in parts)
    if provider_error is not None:
        return f"Provider returned no choices: {provider_error}"
    return "Provider returned a response with no choices."


def _normalize_provider_error(error: BaseException, *, model_name: str) -> BaseException:
    status = _error_status_code(error)
    if status != 400 or _is_policy_error(error):
        return error

    message = _flatten_error_text(error)
    if _is_context_limit_error(error):
        return PromptTooLongError(
            message,
            model_name=model_name,
            offending_indices=[0],
        )
    if _is_invalid_input_error(error):
        return RecoverableInputError(message, offending_indices=[0])
    return error


def _run_concurrent(
    coro_fn: Any,
    items: list[Any],
    max_concurrency: int,
    *,
    return_partial: bool = False,
) -> list[Any]:
    """Run an async function concurrently over a list of items.

    Uses asyncio.gather with a semaphore to limit concurrency.
    Handles being called from both sync and async contexts.

    Args:
        coro_fn: Async callable that takes one item and returns a result.
        items: Items to process.
        max_concurrency: Maximum number of concurrent calls.

    Returns:
        List of results in the same order as items.
    """
    if not items:
        return []

    async def _run() -> list[Any]:
        sem = asyncio.Semaphore(max_concurrency)

        async def _limited(item: Any) -> Any:
            async with sem:
                return await coro_fn(item)

        results = await asyncio.gather(
            *[_limited(item) for item in items], return_exceptions=True,
        )
        failures = {
            idx: failure
            for idx, failure in enumerate(results)
            if isinstance(failure, BaseException)
        }
        if failures:
            import logging

            logging.getLogger(__name__).warning(
                "Concurrent batch: %d/%d tasks failed (%s); "
                "%s",
                len(failures),
                len(results),
                type(next(iter(failures.values()))).__name__,
                "raising PartialBatchError"
                if return_partial
                else "re-raising first exception",
            )
            if return_partial:
                raise PartialBatchError(list(results), failures)
            raise next(iter(failures.values()))
        return list(results)

    try:
        asyncio.get_running_loop()
        # Already in an async context — run in a separate thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(_run())


def _run_async_close(coro: Any) -> None:
    """Run an async close coroutine from sync code."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        asyncio.run(coro)


class OpenAIBackend(ModelBackend):
    """OpenAI API backend.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    Includes logprob support for compatible models.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        max_concurrency: int = 64,
        timeout: float | None = None,
        max_retries: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize OpenAI backend.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo").
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            base_url: Custom API base URL.
            organization: OpenAI organization ID.
            max_concurrency: Maximum concurrent requests for batch generation.
            timeout: Connect timeout in seconds.  The overall read timeout
                stays at the SDK default (600s).  ``None`` uses SDK defaults.
            max_retries: Maximum SDK-level retries for transient errors
                (timeouts, connection errors, 5xx).  ``None`` uses SDK
                default (2).
            **client_kwargs: Additional kwargs for OpenAI client.

        Raises:
            ImportError: If openai package is not installed.
        """
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIBackend. Install with: pip install openai"
            ) from e

        self._model = model
        self._max_concurrency = max_concurrency
        self._return_partial_batch = True

        client_args: dict[str, Any] = {**client_kwargs}
        if api_key is not None:
            client_args["api_key"] = api_key
        if base_url is not None:
            client_args["base_url"] = base_url
        if organization is not None:
            client_args["organization"] = organization
        if timeout is not None:
            import httpx
            client_args["timeout"] = httpx.Timeout(timeout=600.0, connect=timeout)
        if max_retries is not None:
            client_args["max_retries"] = max_retries

        self._client = OpenAI(**client_args)
        self._async_client = AsyncOpenAI(**client_args)
        self._closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        """OpenAI capabilities."""
        return BackendCapabilities(
            supports_logprobs=True,
            supports_prefix_continuation=False,
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=True,
            supports_vision=True,
            max_batch_size=None,
            max_context_length=128000,  # Varies by model
            max_concurrency=self._max_concurrency,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    def close(self) -> None:
        """Close reusable OpenAI clients."""
        if getattr(self, "_closed", False):
            return

        client = getattr(self, "_client", None)
        async_client = getattr(self, "_async_client", None)
        if client is not None:
            client.close()
        if async_client is not None:
            _run_async_close(async_client.close())
        self._client = None
        self._async_client = None
        self._closed = True

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate completions for text prompts.

        Uses the chat API with a single user message.
        """
        results = []
        for prompt in prompts:
            messages = [ChatMessage(role="user", content=prompt)]
            result = self.generate_chat(messages, params)
            results.append(result)
        return results

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation."""
        message_dicts = [m.to_dict() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
            "n": params.n,
        }

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = params.num_logprobs

        # Merge backend-specific extra params
        if params.extra:
            kwargs.update(params.extra)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc
        choice = response.choices[0]

        # Extract logprobs if available
        token_logprobs = None
        if choice.logprobs and choice.logprobs.content:
            logprobs = []
            for token_info in choice.logprobs.content:
                top_lps = None
                if token_info.top_logprobs:
                    top_lps = {lp.token: lp.logprob for lp in token_info.top_logprobs}
                logprobs.append(
                    TokenLogprob(
                        token=token_info.token,
                        token_id=0,  # OpenAI doesn't provide token IDs
                        logprob=token_info.logprob,
                        top_logprobs=top_lps,
                    )
                )
            token_logprobs = tuple(logprobs)

        if params.logprobs and not token_logprobs:
            raise LogprobsNotReturnedError(
                "OpenAI returned no token logprobs despite logprobs=True for "
                f"model {self._model!r}.",
                backend_name="OpenAIBackend",
                model_name=self._model,
            )

        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            token_logprobs=token_logprobs,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Async version of generate_chat for concurrent batch execution."""
        message_dicts = [m.to_dict() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
            "n": params.n,
        }

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = params.num_logprobs

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = await self._async_client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc
        choice = response.choices[0]

        token_logprobs = None
        if choice.logprobs and choice.logprobs.content:
            logprobs = []
            for token_info in choice.logprobs.content:
                top_lps = None
                if token_info.top_logprobs:
                    top_lps = {lp.token: lp.logprob for lp in token_info.top_logprobs}
                logprobs.append(
                    TokenLogprob(
                        token=token_info.token,
                        token_id=0,
                        logprob=token_info.logprob,
                        top_logprobs=top_lps,
                    )
                )
            token_logprobs = tuple(logprobs)

        if params.logprobs and not token_logprobs:
            raise LogprobsNotReturnedError(
                "OpenAI returned no token logprobs despite logprobs=True for "
                f"model {self._model!r}.",
                backend_name="OpenAIBackend",
                model_name=self._model,
            )

        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            token_logprobs=token_logprobs,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

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
        message_dicts = [m.to_dict() for m in messages]
        tool_schemas = [t.to_openai_schema() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "tools": tool_schemas,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
            "n": params.n,
        }

        # Handle tool_choice
        if tool_choice == "auto":
            kwargs["tool_choice"] = "auto"
        elif tool_choice == "none":
            kwargs["tool_choice"] = "none"
        elif tool_choice == "required":
            kwargs["tool_choice"] = "required"
        else:
            # Specific tool name
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc
        choice = response.choices[0]

        # Parse tool calls from response
        tool_calls: tuple[ToolCall, ...] = ()
        if choice.message.tool_calls:
            parsed_calls = []
            for tc in choice.message.tool_calls:
                # Parse arguments from JSON string
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

        return GenerationResult(
            text=choice.message.content,
            finish_reason=_openai_stop_reason(choice.finish_reason),
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Async version of generate_with_tools for concurrent batch execution."""
        message_dicts = [m.to_dict() for m in messages]
        tool_schemas = [t.to_openai_schema() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "tools": tool_schemas,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
            "n": params.n,
        }

        if tool_choice == "auto":
            kwargs["tool_choice"] = "auto"
        elif tool_choice == "none":
            kwargs["tool_choice"] = "none"
        elif tool_choice == "required":
            kwargs["tool_choice"] = "required"
        else:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = await self._async_client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc
        choice = response.choices[0]

        tool_calls: tuple[ToolCall, ...] = ()
        if choice.message.tool_calls:
            parsed_calls = []
            for tc in choice.message.tool_calls:
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

        return GenerationResult(
            text=choice.message.content,
            finish_reason=_openai_stop_reason(choice.finish_reason),
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

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


class AnthropicBackend(ModelBackend):
    """Anthropic API backend.

    Supports Claude models. Includes prefix continuation via assistant prefill.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_retries: int = 2,
        max_concurrency: int = 64,
        **client_kwargs: Any,
    ) -> None:
        """Initialize Anthropic backend.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514").
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
            max_retries: Number of retries for failed requests.
            max_concurrency: Maximum concurrent requests for batch generation.
            **client_kwargs: Additional kwargs for Anthropic client.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "anthropic is required for AnthropicBackend. Install with: pip install anthropic"
            ) from e

        self._model = model
        self._max_concurrency = max_concurrency
        self._return_partial_batch = True

        client_args: dict[str, Any] = {
            "max_retries": max_retries,
            **client_kwargs,
        }
        if api_key is not None:
            client_args["api_key"] = api_key

        self._client = Anthropic(**client_args)
        self._async_client = AsyncAnthropic(**client_args)
        self._closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        """Anthropic capabilities."""
        return BackendCapabilities(
            supports_logprobs=False,
            supports_prefix_continuation=True,  # Via assistant prefill
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=True,
            supports_vision=True,
            max_batch_size=None,
            max_context_length=200000,  # Claude 3 context
            max_concurrency=self._max_concurrency,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    def close(self) -> None:
        """Close reusable Anthropic clients."""
        if getattr(self, "_closed", False):
            return

        client = getattr(self, "_client", None)
        async_client = getattr(self, "_async_client", None)
        if client is not None:
            client.close()
        if async_client is not None:
            _run_async_close(async_client.close())
        self._client = None
        self._async_client = None
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
            result = self.generate_chat(messages, params)
            results.append(result)
        return results

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation."""
        # Separate system message if present
        system_content = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_dict())

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": params.max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content

        # Anthropic uses temperature differently
        if params.temperature > 0:
            kwargs["temperature"] = params.temperature

        if params.top_p < 1.0:
            kwargs["top_p"] = params.top_p

        if params.top_k > 0:
            kwargs["top_k"] = params.top_k

        if params.stop_sequences:
            kwargs["stop_sequences"] = list(params.stop_sequences)

        # Merge backend-specific extra params
        if params.extra:
            kwargs.update(params.extra)

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc

        # Extract text from content blocks
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return GenerationResult(
            text=text,
            finish_reason=_anthropic_stop_reason(response.stop_reason),
            token_logprobs=None,  # Anthropic doesn't provide logprobs
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Async version of generate_chat for concurrent batch execution."""
        system_content = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_dict())

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": params.max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature

        if params.top_p < 1.0:
            kwargs["top_p"] = params.top_p

        if params.top_k > 0:
            kwargs["top_k"] = params.top_k

        if params.stop_sequences:
            kwargs["stop_sequences"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = await self._async_client.messages.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return GenerationResult(
            text=text,
            finish_reason=_anthropic_stop_reason(response.stop_reason),
            token_logprobs=None,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

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

    def continue_from_prefix(
        self,
        prefix: str,
        params: SamplingParams,
        num_continuations: int = 1,
    ) -> list[GenerationResult]:
        """Continue from a prefix using assistant prefill.

        Anthropic supports prefilling the assistant's response.
        """
        results = []
        for _ in range(num_continuations):
            # Use assistant prefill
            messages = [
                {"role": "user", "content": "Continue the following:"},
                {"role": "assistant", "content": prefix},
            ]

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": params.max_tokens,
            }

            if params.temperature > 0:
                kwargs["temperature"] = params.temperature

            if params.stop_sequences:
                kwargs["stop_sequences"] = list(params.stop_sequences)

            try:
                response = self._client.messages.create(**kwargs)
            except Exception as exc:
                raise _normalize_provider_error(exc, model_name=self._model) from exc

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            results.append(
                GenerationResult(
                    text=text,
                    finish_reason=_anthropic_stop_reason(response.stop_reason),
                    token_logprobs=None,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    metadata={"is_continuation": True, "prefix": prefix},
                )
            )

        return results

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Generate a response with tool calling capability."""
        # Separate system message if present
        system_content = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "tool":
                # Anthropic uses tool_result blocks
                chat_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                chat_messages.append({"role": "assistant", "content": content})
            else:
                chat_messages.append(msg.to_dict())

        tool_schemas = [t.to_anthropic_schema() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "tools": tool_schemas,
            "max_tokens": params.max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content

        # Handle tool_choice
        if tool_choice == "auto":
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "none":
            # Don't include tools at all
            del kwargs["tools"]
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        else:
            # Specific tool name
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature

        if params.top_p < 1.0:
            kwargs["top_p"] = params.top_p

        if params.top_k > 0:
            kwargs["top_k"] = params.top_k

        if params.stop_sequences:
            kwargs["stop_sequences"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc

        # Extract text and tool calls from content blocks
        text_parts: list[str] = []
        parsed_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                parsed_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        text = "".join(text_parts) if text_parts else None

        return GenerationResult(
            text=text,
            finish_reason=_anthropic_stop_reason(response.stop_reason),
            tool_calls=tuple(parsed_calls),
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Async version of generate_with_tools for concurrent batch execution."""
        system_content = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "tool":
                chat_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                chat_messages.append({"role": "assistant", "content": content})
            else:
                chat_messages.append(msg.to_dict())

        tool_schemas = [t.to_anthropic_schema() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "tools": tool_schemas,
            "max_tokens": params.max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content

        if tool_choice == "auto":
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "none":
            del kwargs["tools"]
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        else:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature

        if params.top_p < 1.0:
            kwargs["top_p"] = params.top_p

        if params.top_k > 0:
            kwargs["top_k"] = params.top_k

        if params.stop_sequences:
            kwargs["stop_sequences"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = await self._async_client.messages.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc

        text_parts: list[str] = []
        parsed_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                parsed_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        text = "".join(text_parts) if text_parts else None

        return GenerationResult(
            text=text,
            finish_reason=_anthropic_stop_reason(response.stop_reason),
            tool_calls=tuple(parsed_calls),
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

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


class OpenRouterBackend(ModelBackend):
    """OpenRouter API backend.

    Provides access to multiple model providers through a unified API.
    Capabilities vary by underlying model.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        max_concurrency: int = 64,
        rate_limit_wait: float = 0.0,
        rate_limit_max_retries: int = 2,
        timeout: float | None = None,
        max_retries: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize OpenRouter backend.

        Args:
            model: Model name in provider/model format.
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var).
            site_url: Your site URL for rankings.
            app_name: Your app name for rankings.
            max_concurrency: Maximum concurrent requests for batch generation.
            rate_limit_wait: Seconds to wait before retrying after a rate-limit
                (429) error.  When set to ``0`` (the default) the SDK's
                built-in retry/backoff handles 429s.  Set to e.g. ``60`` for
                free-tier models with strict per-minute caps.
            rate_limit_max_retries: Maximum number of rate-limit retries before
                giving up and re-raising the error.
            timeout: Connect timeout in seconds.  The overall read timeout
                stays at the SDK default (600s).  ``None`` uses SDK defaults.
            max_retries: Maximum SDK-level retries for transient errors
                (timeouts, connection errors, 5xx).  ``None`` uses SDK
                default (2).
            **client_kwargs: Additional kwargs for OpenAI client.

        Raises:
            ImportError: If openai package is not installed.
        """
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenRouterBackend. Install with: pip install openai"
            ) from e

        import os

        self._model = model
        self._max_concurrency = max_concurrency
        self._rate_limit_wait = rate_limit_wait
        self._rate_limit_max_retries = rate_limit_max_retries
        self._return_partial_batch = True

        # OpenRouter uses OpenAI-compatible API
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        client_args: dict[str, Any] = {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            **client_kwargs,
        }
        if headers:
            client_args["default_headers"] = headers
        if timeout is not None:
            import httpx
            client_args["timeout"] = httpx.Timeout(timeout=600.0, connect=timeout)
        if max_retries is not None:
            client_args["max_retries"] = max_retries

        self._client = OpenAI(**client_args)
        self._async_client = AsyncOpenAI(**client_args)
        self._closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        """OpenRouter capabilities.

        ``supports_logprobs=True`` because OpenRouter accepts the OpenAI
        logprobs fields (``logprobs`` bool + ``top_logprobs`` int, cap 20).
        Whether the upstream model actually returns logprobs is
        provider-dependent — when it doesn't, the backend raises
        ``LogprobsNotReturnedError`` at call time.
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
        """Close reusable OpenRouter clients."""
        if getattr(self, "_closed", False):
            return

        client = getattr(self, "_client", None)
        async_client = getattr(self, "_async_client", None)
        if client is not None:
            client.close()
        if async_client is not None:
            _run_async_close(async_client.close())
        self._client = None
        self._async_client = None
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
            result = self.generate_chat(messages, params)
            results.append(result)
        return results

    # OpenRouter documents an explicit cap of 20 on ``top_logprobs`` (OpenAPI
    # schema ``ChatRequest.top_logprobs`` at openrouter.ai/openapi.json).
    _TOP_LOGPROBS_MAX = 20

    def _chat_kwargs(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> dict[str, Any]:
        """Build kwargs dict for chat completions."""
        message_dicts = [m.to_dict() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.logprobs:
            if params.num_logprobs > self._TOP_LOGPROBS_MAX:
                raise LogprobsNotReturnedError(
                    f"OpenRouter caps top_logprobs at {self._TOP_LOGPROBS_MAX}; "
                    f"requested num_logprobs={params.num_logprobs}.",
                    backend_name="OpenRouterBackend",
                    model_name=self._model,
                )
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = params.num_logprobs
            # OpenRouter's default router picks any provider with
            # capacity — several (e.g. DeepInfra for Gemma) silently
            # drop ``logprobs`` and return text only. ``provider.
            # require_parameters`` restricts routing to providers that
            # actually honour every requested param, so the request
            # either gets logprobs back or fails with a routing error
            # we can surface instead of silently losing them.
            extra_body = kwargs.setdefault("extra_body", {})
            provider_cfg = extra_body.setdefault("provider", {})
            provider_cfg.setdefault("require_parameters", True)

        if params.extra:
            kwargs.update(params.extra)

        return kwargs

    def _chat_result(self, response: Any) -> GenerationResult:
        """Convert an OpenAI-compatible ChatCompletion to GenerationResult.

        Logprobs follow the OpenAI response shape — OpenRouter's OpenAPI
        schema ``ChatTokenLogprobs`` at ``choices[i].logprobs.content[]``
        matches OpenAI field-for-field (``token``, ``logprob``,
        ``top_logprobs: [{token, logprob, bytes}]``), so the same
        extraction works.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            provider_error = getattr(response, "error", None)
            raise MalformedResponseError(
                _format_no_choices_message(provider_error),
                backend_name="OpenRouterBackend",
                model_name=self._model,
                provider_error=provider_error,
            )
        choice = choices[0]

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
                        token_id=0,  # OpenRouter/OpenAI do not return token IDs
                        logprob=token_info.logprob,
                        top_logprobs=top_lps,
                    )
                )
            token_logprobs = tuple(logprobs)

        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            token_logprobs=token_logprobs,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    def _ensure_logprobs_if_requested(
        self,
        result: GenerationResult,
        params: SamplingParams,
    ) -> GenerationResult:
        """Raise ``LogprobsNotReturnedError`` if logprobs were asked for but absent."""
        if params.logprobs and not result.token_logprobs:
            raise LogprobsNotReturnedError(
                "OpenRouter returned no token logprobs despite logprobs=True. "
                "The underlying provider likely does not expose logprobs for "
                f"model {self._model!r}.",
                backend_name="OpenRouterBackend",
                model_name=self._model,
            )
        return result

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation."""
        import logging
        import time

        from openai import RateLimitError

        kwargs = self._chat_kwargs(messages, params)
        last_exc: RateLimitError | None = None

        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                try:
                    response = self._client.chat.completions.create(**kwargs)
                    return self._ensure_logprobs_if_requested(
                        self._chat_result(response), params
                    )
                except RateLimitError:
                    raise
                except Exception as exc:
                    raise _normalize_provider_error(exc, model_name=self._model) from exc
            except RateLimitError as exc:
                if self._rate_limit_wait <= 0 or attempt == self._rate_limit_max_retries:
                    raise
                last_exc = exc
                logging.getLogger(__name__).warning(
                    "Rate limited, waiting %.0fs (attempt %d/%d)",
                    self._rate_limit_wait, attempt + 1, self._rate_limit_max_retries,
                )
                time.sleep(self._rate_limit_wait)

        raise last_exc  # unreachable, but satisfies type checker

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Async version of generate_chat for concurrent batch execution."""
        import asyncio
        import logging

        from openai import RateLimitError

        kwargs = self._chat_kwargs(messages, params)
        last_exc: RateLimitError | None = None

        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                try:
                    response = await self._async_client.chat.completions.create(**kwargs)
                    return self._ensure_logprobs_if_requested(
                        self._chat_result(response), params
                    )
                except RateLimitError:
                    raise
                except Exception as exc:
                    raise _normalize_provider_error(exc, model_name=self._model) from exc
            except RateLimitError as exc:
                if self._rate_limit_wait <= 0 or attempt == self._rate_limit_max_retries:
                    raise
                last_exc = exc
                logging.getLogger(__name__).warning(
                    "Rate limited, waiting %.0fs (attempt %d/%d)",
                    self._rate_limit_wait, attempt + 1, self._rate_limit_max_retries,
                )
                await asyncio.sleep(self._rate_limit_wait)

        raise last_exc  # unreachable, but satisfies type checker

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
        """Generate a response with tool calling capability.

        Uses OpenAI-compatible format since OpenRouter is OpenAI-compatible.
        """
        message_dicts = [m.to_dict() for m in messages]
        tool_schemas = [t.to_openai_schema() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "tools": tool_schemas,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }

        # Handle tool_choice
        if tool_choice == "auto":
            kwargs["tool_choice"] = "auto"
        elif tool_choice == "none":
            kwargs["tool_choice"] = "none"
        elif tool_choice == "required":
            kwargs["tool_choice"] = "required"
        else:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc
        choice = response.choices[0]

        # Parse tool calls from response
        tool_calls: tuple[ToolCall, ...] = ()
        if choice.message.tool_calls:
            parsed_calls = []
            for tc in choice.message.tool_calls:
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

        return GenerationResult(
            text=choice.message.content,
            finish_reason=_openai_stop_reason(choice.finish_reason),
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Async version of generate_with_tools for concurrent batch execution."""
        message_dicts = [m.to_dict() for m in messages]
        tool_schemas = [t.to_openai_schema() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": message_dicts,
            "tools": tool_schemas,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }

        if tool_choice == "auto":
            kwargs["tool_choice"] = "auto"
        elif tool_choice == "none":
            kwargs["tool_choice"] = "none"
        elif tool_choice == "required":
            kwargs["tool_choice"] = "required"
        else:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        try:
            response = await self._async_client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _normalize_provider_error(exc, model_name=self._model) from exc
        choice = response.choices[0]

        tool_calls: tuple[ToolCall, ...] = ()
        if choice.message.tool_calls:
            parsed_calls = []
            for tc in choice.message.tool_calls:
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

        return GenerationResult(
            text=choice.message.content,
            finish_reason=_openai_stop_reason(choice.finish_reason),
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

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
