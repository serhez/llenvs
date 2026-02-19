"""API-based backends for OpenAI, Anthropic, and OpenRouter.

Each backend wraps the respective API client with the ModelBackend interface.
"""

import asyncio
import json
import uuid
from typing import Any

from llenvs.core.tools import ToolCall, ToolDefinition
from llenvs.inference.protocol import (
    ModelBackend,
    BackendCapabilities,
    SamplingParams,
    GenerationResult,
    ChatMessage,
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


def _run_concurrent(coro_fn: Any, items: list[Any], max_concurrency: int) -> list[Any]:
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

        return list(await asyncio.gather(*[_limited(item) for item in items]))

    try:
        asyncio.get_running_loop()
        # Already in an async context — run in a separate thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(_run())


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
        **client_kwargs: Any,
    ) -> None:
        """Initialize OpenAI backend.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo").
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            base_url: Custom API base URL.
            organization: OpenAI organization ID.
            max_concurrency: Maximum concurrent requests for batch generation.
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

        client_args: dict[str, Any] = {**client_kwargs}
        if api_key is not None:
            client_args["api_key"] = api_key
        if base_url is not None:
            client_args["base_url"] = base_url
        if organization is not None:
            client_args["organization"] = organization

        self._client = OpenAI(**client_args)
        self._async_client = AsyncOpenAI(**client_args)

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

        response = self._client.chat.completions.create(**kwargs)
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

        response = await self._async_client.chat.completions.create(**kwargs)
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

        response = self._client.chat.completions.create(**kwargs)
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

        client_args: dict[str, Any] = {
            "max_retries": max_retries,
            **client_kwargs,
        }
        if api_key is not None:
            client_args["api_key"] = api_key

        self._client = Anthropic(**client_args)
        self._async_client = AsyncAnthropic(**client_args)

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

        response = self._client.messages.create(**kwargs)

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

        response = await self._async_client.messages.create(**kwargs)

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

            response = self._client.messages.create(**kwargs)

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

        response = self._client.messages.create(**kwargs)

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
        **client_kwargs: Any,
    ) -> None:
        """Initialize OpenRouter backend.

        Args:
            model: Model name in provider/model format.
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var).
            site_url: Your site URL for rankings.
            app_name: Your app name for rankings.
            max_concurrency: Maximum concurrent requests for batch generation.
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

        # OpenRouter uses OpenAI-compatible API
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers if headers else None,
            **client_kwargs,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers if headers else None,
            **client_kwargs,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """OpenRouter capabilities (conservative estimate)."""
        return BackendCapabilities(
            supports_logprobs=False,  # Varies by model
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

        # Merge backend-specific extra params
        if params.extra:
            kwargs.update(params.extra)

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            token_logprobs=None,
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
        }

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.extra:
            kwargs.update(params.extra)

        response = await self._async_client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            token_logprobs=None,
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

        response = self._client.chat.completions.create(**kwargs)
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
