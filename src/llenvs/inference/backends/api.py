"""API-based backends for OpenAI, Anthropic, and OpenRouter.

Each backend wraps the respective API client with the ModelBackend interface.
"""

import asyncio
import inspect
import json
import math
import random
import threading
import time
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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
    QuotaExhaustedError,
    RecoverableInputError,
    RefusedByPolicyError,
    SamplingParams,
    StopReason,
    TokenLogprob,
)
from llenvs.inference.reasoning import local_chat_messages
from llenvs.inference.records import dependency_versions, normalize_usage, plain
from llenvs.inference.sampling import sampling_metadata, validate_sampling

# Callers that persist normalized OpenRouter responses should include this in
# checkpoint identity. Bump it when response acceptance semantics change.
OPENROUTER_RESPONSE_PROTOCOL_VERSION = "require-termination-v1"


def _configure_openai_timeout(client_args, connect, read):
    if read is not None and (
        isinstance(read, bool)
        or not isinstance(read, (int, float))
        or not math.isfinite(read)
        or read <= 0
    ):
        raise ValueError("read_timeout must be finite and positive")
    if connect is None and read is None:
        return
    import httpx
    from openai import DEFAULT_TIMEOUT

    values = DEFAULT_TIMEOUT.as_dict()
    if connect is not None:
        values["connect"] = connect
    if read is not None:
        values["read"] = read
    client_args["timeout"] = httpx.Timeout(**values)


async def _notify(observer, method, index, value):
    if observer is None:
        return
    if hasattr(observer, "notify"):
        result = observer.notify(method, index, value)
    else:
        result = getattr(observer, method)(index, value)
    if inspect.isawaitable(result):
        await result


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
    provider_error = getattr(error, "provider_error", None)
    if provider_error is not None:
        return provider_error
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


def _is_rate_limit_text(text: str) -> bool:
    text = text.lower()
    return any(
        needle in text
        for needle in (
            "rate increased too quickly",
            "too many requests",
            "rate limit",
            "rate_limit",
            "throttl",
            "request limit",
        )
    )


def _is_rate_limit_malformed(exc: BaseException) -> bool:
    """True iff ``exc`` is a ``MalformedResponseError`` whose provider error
    text matches a rate-limit pattern.

    Some providers (notably Alibaba) surface their own 429 as an HTTP-200
    body with ``choices=null`` plus a textual rate-limit message in the
    OpenRouter top-level ``error``. The OpenAI SDK's ``RateLimitError``
    path therefore never fires, and the retry loop would otherwise let
    these abort the batch immediately.
    """
    if not isinstance(exc, MalformedResponseError):
        return False
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if status is not None and 400 <= status < 500:
        return False
    parts = [str(exc)]
    provider_error = getattr(exc, "provider_error", None)
    if isinstance(provider_error, dict):
        msg = provider_error.get("message")
        if isinstance(msg, str):
            parts.append(msg)
        code = provider_error.get("code")
        if isinstance(code, str):
            parts.append(code)
    elif isinstance(provider_error, str):
        parts.append(provider_error)
    return _is_rate_limit_text(" ".join(parts))


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


def _to_plain_mapping(value: Any) -> dict[str, Any]:
    """Return non-None fields from SDK or dict-like objects."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if v is not None}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return {
            str(k): v for k, v in vars(value).items() if not k.startswith("_") and v is not None
        }
    return {}


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from either a mapping or an SDK object."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _reasoning_detail_types(reasoning_details: Any) -> list[str]:
    """Extract reasoning detail ``type`` values without retaining payloads."""
    if not isinstance(reasoning_details, (list, tuple)):
        return []
    types: list[str] = []
    for detail in reasoning_details:
        detail_type = _get_field(detail, "type")
        if detail_type is not None:
            types.append(str(detail_type))
    return types


def _reasoning_metadata(message: Any) -> dict[str, Any]:
    """Retain separately returned reasoning without treating it as answer text.

    Reasoning blocks may include signatures and opaque provider fields. Keep
    their order and full contents, including nulls, as detached plain data.
    """
    metadata: dict[str, Any] = {}
    reasoning = _get_field(message, "reasoning")
    if reasoning is None:
        reasoning = _get_field(message, "reasoning_content")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise ValueError("Expected assistant reasoning to be a string")
        metadata.update(
            reasoning=reasoning, reasoning_present=bool(reasoning), reasoning_chars=len(reasoning)
        )

    details = _get_field(message, "reasoning_details")
    if details is not None:
        if not isinstance(details, (list, tuple)):
            raise ValueError("Expected reasoning_details to be a list of blocks")
        blocks = []
        for block in details:
            if not isinstance(block, dict):
                dump = getattr(block, "model_dump", None)
                if not callable(dump):
                    raise ValueError("Expected reasoning_details blocks to be mappings")
                block = dump(mode="json")
            if not isinstance(block, dict):
                raise ValueError("Expected reasoning_details blocks to be mappings")
            blocks.append(deepcopy(block))
        metadata.update(
            reasoning_details=blocks,
            reasoning_details_present=bool(blocks),
            reasoning_details_count=len(blocks),
        )
        detail_types = _reasoning_detail_types(blocks)
        if detail_types:
            metadata["reasoning_details_types"] = tuple(detail_types)
    return metadata


def _normalize_provider_error(error: BaseException, *, model_name: str) -> BaseException:
    status = _error_status_code(error)
    if status == 429:
        body = _error_body(error)
        detail = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(detail, dict) and detail.get("code") in {
            "insufficient_quota",
            "billing_hard_limit_reached",
        }:
            return QuotaExhaustedError(f"Provider quota exhausted for model {model_name!r}")
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


class _AsyncRunner:
    """Own one loop for a backend's pooled async connections until close().

    Synchronous callers may come from different threads or an active event loop.
    Calls share the loop and connection pool while progressing independently.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    @staticmethod
    def _serve(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def run(self, coro: Any) -> Any:
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("Cannot synchronously call a backend from its own API loop")
        with self._lock:
            if self._closed:
                coro.close()
                raise RuntimeError("API backend is closed")
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._serve,
                    args=(self._loop,),
                    name="llenvs-api",
                    daemon=True,
                )
                self._thread.start()
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def close(self, finalizer: Any = None) -> None:
        if threading.current_thread() is self._thread:
            if finalizer is not None:
                finalizer.close()
            raise RuntimeError("Cannot synchronously close a backend from its own API loop")
        with self._lock:
            if self._closed:
                if finalizer is not None:
                    finalizer.close()
                return
            try:
                if finalizer is not None:
                    self.run(finalizer)
            finally:
                self._closed = True
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    self._thread.join()


def _close_api_clients(backend: Any) -> None:
    if getattr(backend, "_closed", False):
        return
    runner = getattr(backend, "_async_runner", None) or _AsyncRunner()
    client = getattr(backend, "_client", None)
    async_client = getattr(backend, "_async_client", None)
    try:
        try:
            if client is not None:
                client.close()
        finally:
            runner.close(async_client.close() if async_client is not None else None)
    finally:
        backend._client = None
        backend._async_client = None
        backend._closed = True


def _run_concurrent(
    coro_fn: Any,
    items: list[Any],
    max_concurrency: int,
    *,
    return_partial: bool = False,
    runner: _AsyncRunner | None = None,
) -> list[Any]:
    """Run an async function concurrently over a list of items.

    Uses asyncio.gather with a semaphore to limit concurrency.
    Handles both sync and async callers. Supply the client owner's runner when
    reusing async HTTP connections; omission creates a one-call loop.

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
        fatal = None

        async def _limited(item: Any) -> Any:
            nonlocal fatal
            async with sem:
                if fatal is not None:
                    raise fatal
                try:
                    return await coro_fn(item)
                except Exception as exc:
                    from llenvs.inference.persistence import CheckpointError

                    if isinstance(exc, CheckpointError):
                        fatal = exc
                    raise

        results = await asyncio.gather(
            *[_limited(item) for item in items],
            return_exceptions=True,
        )
        failures = {
            idx: failure
            for idx, failure in enumerate(results)
            if isinstance(failure, BaseException)
        }
        if failures:
            if fatal is not None:
                raise fatal
            import logging

            logging.getLogger(__name__).warning(
                "Concurrent batch: %d/%d tasks failed (%s); %s",
                len(failures),
                len(results),
                type(next(iter(failures.values()))).__name__,
                "raising PartialBatchError" if return_partial else "re-raising first exception",
            )
            if return_partial:
                raise PartialBatchError(list(results), failures)
            raise next(iter(failures.values()))
        return list(results)

    if runner is not None:
        return runner.run(_run())
    temporary = _AsyncRunner()
    try:
        return temporary.run(_run())
    finally:
        temporary.close()


class _APIRequests:
    """Shared request receipts, normalization boundary, and bounded retry ownership."""

    _model: str
    _api_flavor: str
    _client: Any
    _async_client: Any
    _rate_limit_wait: float
    _rate_limit_max_retries: int

    def _parse_response(
        self, response: Any, params: SamplingParams, metadata: dict[str, Any]
    ) -> GenerationResult:
        raise NotImplementedError

    def descriptor(self):
        from llenvs.inference.sampling import _record_value

        return {
            "identity": {
                "backend": f"{type(self).__module__}.{type(self).__qualname__}",
                "protocol_version": 1,
                "api_flavor": self._api_flavor,
                "model": self._model,
                "version_pinned": False,
                "endpoint": _record_value({"url": str(self._client.base_url)})["url"],
                "sdk_max_retries": self._client.max_retries,
                "dependencies": dependency_versions(
                    "openai" if self._api_flavor != "anthropic" else "anthropic"
                ),
            },
            "provenance": {},
        }

    def _dispatch(self, kwargs):
        resource = (
            self._client.messages
            if self._api_flavor == "anthropic"
            else self._client.chat.completions
        )
        return resource.create(**kwargs)

    async def _dispatch_async(self, kwargs):
        resource = (
            self._async_client.messages
            if self._api_flavor == "anthropic"
            else self._async_client.chat.completions
        )
        return await resource.create(**kwargs)

    @staticmethod
    def _serialized_body(kwargs: dict[str, Any]) -> dict[str, Any]:
        body = {
            key: value
            for key, value in kwargs.items()
            if key not in {"extra_body", "extra_headers", "extra_query", "timeout"}
            and type(value).__name__ not in {"NotGiven", "Omit"}
        }

        def merge(target, extra):
            for key, value in extra.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = deepcopy(value)

        body = deepcopy(body)
        merge(body, kwargs.get("extra_body", {}))
        try:
            body = json.loads(json.dumps(body, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Request body must contain finite JSON values") from exc
        return body

    def _request_metadata(self, kwargs: dict[str, Any], params: SamplingParams) -> dict[str, Any]:
        body = self._serialized_body(kwargs)
        omitted = {
            "openai": ("top_k", "min_p", "repetition_penalty"),
            "anthropic": ("min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"),
        }.get(self._api_flavor, ())
        return sampling_metadata(params, body, omitted=omitted, capture="serialized_body")

    def _failed_attempt(
        self,
        exc: Exception,
        started: float,
        attempts: list[dict[str, Any]],
        call_started: float,
    ) -> float:
        normalized = _normalize_provider_error(exc, model_name=self._model)
        if normalized is not exc:
            setattr(normalized, "status_code", _error_status_code(exc))
        attempt = {
            "status_code": _error_status_code(exc),
            "elapsed_seconds": time.monotonic() - started,
            "request_id": getattr(exc, "request_id", None),
            "error_type": type(normalized).__name__,
            "usage": _to_plain_mapping(_get_field(_error_body(exc), "usage")),
        }
        if getattr(exc, "reported_attempt", None) is not None:
            attempt.update(getattr(exc, "reported_attempt"))
            attempt["error_type"] = type(normalized).__name__
        attempts.append(attempt)
        delay = None
        if (
            normalized is exc
            and (_error_status_code(exc) == 429 or _is_rate_limit_malformed(exc))
            and self._rate_limit_wait > 0
            and len(attempts) <= self._rate_limit_max_retries
        ):
            delay = min(
                60.0,
                self._rate_limit_wait * 2 ** min(len(attempts) - 1, 16) * random.uniform(1.0, 1.25),
            )
            headers = getattr(getattr(exc, "response", None), "headers", {})
            retry_after = headers.get("retry-after")
            if retry_after is not None:
                try:
                    provider_wait = float(retry_after)
                except (ValueError, TypeError):
                    try:
                        provider_wait = (
                            parsedate_to_datetime(retry_after) - datetime.now(UTC)
                        ).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        provider_wait = 0.0
                if math.isfinite(provider_wait):
                    delay = max(delay, provider_wait)
                else:
                    delay = None
            if delay is not None and delay > 300:
                delay = None
        if delay is None:
            setattr(normalized, "attempts", deepcopy(attempts))
            setattr(normalized, "elapsed_seconds", time.monotonic() - call_started)
            if normalized is exc:
                raise exc
            raise normalized from exc
        attempt["backoff_seconds"] = delay
        return delay

    async def _failed_attempt_async(self, exc, started, attempts, call_started, observer, index):
        try:
            delay = self._failed_attempt(exc, started, attempts, call_started)
        except Exception as normalized:
            await _notify(observer, "failed", index, normalized)
            raise
        setattr(exc, "attempts", deepcopy(attempts))
        setattr(exc, "elapsed_seconds", time.monotonic() - call_started)
        await _notify(observer, "failed", index, exc)
        return delay

    def _completed_request(
        self,
        response: Any,
        params: SamplingParams,
        metadata: dict[str, Any],
        attempts: list[dict[str, Any]],
        started: float,
        call_started: float,
    ) -> GenerationResult:
        attempts.append(
            {
                "status_code": 200,
                "elapsed_seconds": time.monotonic() - started,
                "request_id": getattr(response, "_request_id", None),
                "response_id": response.id,
                "usage": _to_plain_mapping(getattr(response, "usage", None)),
            }
        )
        metadata.update(attempts=attempts, elapsed_seconds=time.monotonic() - call_started)
        try:
            return self._parse_response(response, params, metadata)
        except Exception as exc:
            normalized = _normalize_provider_error(exc, model_name=self._model)
            setattr(normalized, "attempts", deepcopy(attempts))
            setattr(normalized, "elapsed_seconds", metadata["elapsed_seconds"])
            if normalized is exc:
                raise
            raise normalized from exc

    def _request(self, kwargs: dict[str, Any], params: SamplingParams) -> GenerationResult:
        metadata = self._request_metadata(kwargs, params)
        attempts: list[dict[str, Any]] = []
        call_started = time.monotonic()
        while True:
            started = time.monotonic()
            try:
                response = self._dispatch(kwargs)
            except Exception as exc:
                time.sleep(self._failed_attempt(exc, started, attempts, call_started))
            else:
                try:
                    return self._completed_request(
                        response, params, metadata, attempts, started, call_started
                    )
                except MalformedResponseError as exc:
                    if not _is_rate_limit_malformed(exc):
                        raise
                    setattr(exc, "reported_attempt", attempts.pop())
                    time.sleep(self._failed_attempt(exc, started, attempts, call_started))

    async def _request_async(
        self, kwargs: dict[str, Any], params: SamplingParams, *, observer=None, index=0
    ) -> GenerationResult:
        metadata = self._request_metadata(kwargs, params)
        attempts: list[dict[str, Any]] = []
        call_started = time.monotonic()
        while True:
            await _notify(observer, "started", index, len(attempts))
            started = time.monotonic()
            try:
                response = await self._dispatch_async(kwargs)
            except Exception as exc:
                await asyncio.sleep(
                    await self._failed_attempt_async(
                        exc, started, attempts, call_started, observer, index
                    )
                )
            else:
                receipt = {
                    "response": plain(response),
                    "metadata": {
                        **metadata,
                        "attempts": [
                            *attempts,
                            {
                                "status_code": 200,
                                "request_id": getattr(response, "_request_id", None),
                                "response_id": response.id,
                                "usage": plain(response.usage),
                                "elapsed_seconds": time.monotonic() - started,
                            },
                        ],
                        "elapsed_seconds": time.monotonic() - call_started,
                    },
                }
                await _notify(observer, "received", index, receipt)
                try:
                    result = self._completed_request(
                        response, params, metadata, attempts, started, call_started
                    )
                except Exception as exc:
                    await _notify(observer, "failed", index, exc)
                    if not _is_rate_limit_malformed(exc):
                        raise
                    setattr(exc, "reported_attempt", attempts.pop())
                    await asyncio.sleep(self._failed_attempt(exc, started, attempts, call_started))
                    continue
                await _notify(observer, "completed", index, result)
                return result


class OpenAIBackend(_APIRequests, ModelBackend):
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
        api_flavor: str = "openai",
        reasoning_family: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        rate_limit_wait: float = 0.0,
        rate_limit_max_retries: int = 2,
        read_timeout: float | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize OpenAI backend.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo").
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            base_url: Custom API base URL.
            organization: OpenAI organization ID.
            max_concurrency: Maximum concurrent requests for batch generation.
            timeout: Connect timeout in seconds. ``None`` uses the SDK default.
            read_timeout: Read timeout in seconds, independently of connection
                establishment. ``None`` keeps the SDK default (600s).
            max_retries: Maximum SDK-level retries for transient errors
                (timeouts, connection errors, 5xx).  ``None`` uses SDK
                defaults unless backend rate-limit retries are enabled.
                Use 0 to prevent hidden retries and ambiguous resubmissions.
            api_flavor: Explicit endpoint dialect, "openai" or "vllm".
            rate_limit_wait: Initial rate-limit retry delay in seconds;
                0 disables backend retries. Enabling it disables SDK retries.
            rate_limit_max_retries: Maximum number of backend rate-limit retries.
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
        if api_flavor not in {"openai", "vllm"}:
            raise ValueError("api_flavor must be 'openai' or 'vllm'")
        if not math.isfinite(rate_limit_wait) or rate_limit_wait < 0:
            raise ValueError("rate_limit_wait must be finite and nonnegative")
        if (
            isinstance(rate_limit_max_retries, bool)
            or not isinstance(rate_limit_max_retries, int)
            or rate_limit_max_retries < 0
        ):
            raise ValueError("rate_limit_max_retries must be a nonnegative integer")
        if rate_limit_wait > 0:
            if max_retries not in (None, 0):
                raise ValueError("backend rate-limit retries require SDK max_retries=0")
            max_retries = 0
        self._api_flavor = api_flavor
        if reasoning_family is not None and api_flavor != "vllm":
            raise ValueError("reasoning_family is a local vLLM template mapping")
        self._reasoning_family = reasoning_family
        self._reasoning_tokenizer = None
        self._chat_template_kwargs = deepcopy(chat_template_kwargs or {})
        if reasoning_family == "qwen3.8":
            self._chat_template_kwargs.setdefault("preserve_thinking", True)
        self._rate_limit_wait = rate_limit_wait
        self._rate_limit_max_retries = rate_limit_max_retries
        self._max_concurrency = max_concurrency
        self._return_partial_batch = True

        client_args: dict[str, Any] = {**client_kwargs}
        if api_key is not None:
            client_args["api_key"] = api_key
        if base_url is not None:
            client_args["base_url"] = base_url
        if organization is not None:
            client_args["organization"] = organization
        _configure_openai_timeout(client_args, timeout, read_timeout)
        if max_retries is not None:
            client_args["max_retries"] = max_retries

        self._client = OpenAI(**client_args)
        self._async_client = AsyncOpenAI(**client_args)
        self._async_runner = _AsyncRunner()
        self._closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        """OpenAI capabilities."""
        return BackendCapabilities(
            supports_explicit_sampling=True,
            supports_logprobs=True,
            supports_prefix_continuation=False,
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_response_schema=self._api_flavor == "vllm"
            or self._model.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "gpt-6-luna")),
            supports_schema_after_reasoning=self._model == "gpt-6-luna"
            or (self._api_flavor == "vllm" and self._reasoning_family == "qwen3.8"),
            supports_request_progress=True,
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
        _close_api_clients(self)

    def generate(self, prompts, params, *, observer=None):
        if observer is None:
            return [self.generate_chat([ChatMessage("user", prompt)], params) for prompt in prompts]
        return self.generate_chat_batch(
            [[ChatMessage("user", prompt)] for prompt in prompts], params, observer=observer
        )

    def _request_kwargs(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        unsupported = (
            {"top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
            if self._api_flavor == "openai"
            else {}
        )
        validate_sampling(params, unsupported=unsupported)
        if params.after_reasoning_text is not None:
            raise ValueError("after_reasoning_text requires a configured local token boundary")
        if params.thinking_budget is not None:
            raise ValueError(
                "thinking_budget requires local token generation or controlled insertion"
            )
        luna = self._api_flavor == "openai" and self._model == "gpt-6-luna"
        if self._api_flavor == "openai":
            if params.reasoning_effort is not None and not luna:
                raise ValueError("reasoning_effort needs a known model capability mapping")
            if luna:
                if params.reasoning_effort not in {
                    None,
                    "none",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                }:
                    raise ValueError("Unsupported Luna reasoning_effort")
                if params.reasoning_effort != "none" and (
                    params.temperature is not None or params.top_p is not None or params.logprobs
                ):
                    raise ValueError(
                        "Luna reasoning requires temperature=None, top_p=None and logprobs=False"
                    )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": local_chat_messages(messages, getattr(self, "_reasoning_family", None))
            if self._api_flavor == "vllm"
            else [m.to_dict() for m in messages],
            "n": 1,
        }
        for name in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"):
            value = getattr(params, name)
            if value is not None:
                kwargs[name] = value
        if luna and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if params.reasoning_effort is not None and self._api_flavor == "openai":
            kwargs["reasoning_effort"] = params.reasoning_effort
        if params.response_schema is not None:
            spec = params.response_schema
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": spec.name,
                    "schema": deepcopy(spec.schema),
                    "strict": spec.strict,
                },
            }
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        if params.logprobs:
            kwargs.update(logprobs=True, top_logprobs=params.num_logprobs)
        if tools is not None:
            kwargs["tools"] = [tool.to_openai_schema() for tool in tools]
            kwargs["tool_choice"] = (
                tool_choice
                if tool_choice in {"auto", "none", "required"}
                else {"type": "function", "function": {"name": tool_choice}}
            )
        kwargs.update(deepcopy(params.extra))
        if self._api_flavor == "vllm":
            body = kwargs.setdefault("extra_body", {})
            for name in ("top_k", "min_p", "repetition_penalty"):
                value = getattr(params, name)
                if value is not None:
                    body[name] = value
            if (
                params.reasoning_effort is not None
                or getattr(self, "_reasoning_family", None) is not None
                or getattr(self, "_chat_template_kwargs", {})
            ):
                from llenvs.inference.reasoning import local_template_kwargs

                body["chat_template_kwargs"] = local_template_kwargs(
                    getattr(self, "_reasoning_family", None),
                    params.reasoning_effort,
                    getattr(self, "_chat_template_kwargs", {}),
                    body.get("chat_template_kwargs"),
                )
            if getattr(self, "_reasoning_family", None) == "qwen3.8":
                body["return_token_ids"] = True
                body["skip_special_tokens"] = False
        return kwargs

    def reparse_receipt(self, receipt: dict[str, Any], params: SamplingParams) -> GenerationResult:
        """Normalize a durable response locally without another provider call."""
        from openai.types.chat import ChatCompletion

        return self._parse_response(
            ChatCompletion.construct(**receipt["response"]), params, deepcopy(receipt["metadata"])
        )

    def _parse_response(
        self,
        response: Any,
        params: SamplingParams,
        metadata: dict[str, Any],
    ) -> GenerationResult:
        if not response.choices:
            raise MalformedResponseError(
                _format_no_choices_message(getattr(response, "error", None)),
                backend_name="OpenAIBackend",
                model_name=self._model,
            )
        choice = response.choices[0]
        if (
            isinstance(getattr(choice.message, "refusal", None), str)
            and choice.message.refusal
            or choice.finish_reason == "content_filter"
        ):
            raise RefusedByPolicyError(
                getattr(choice.message, "refusal", None) or "Content filtered"
            )
        token_logprobs = None
        choice_logprobs = getattr(choice, "logprobs", None)
        raw_ids = getattr(choice, "token_ids", None) if self._api_flavor == "vllm" else None
        if raw_ids is not None and any(
            type(token_id) is not int or token_id < 0 for token_id in raw_ids
        ):
            raise ValueError("Provider returned invalid native token IDs")
        if choice_logprobs and choice_logprobs.content:
            token_logprobs = tuple(
                TokenLogprob(
                    token=item.token,
                    token_id=None,
                    logprob=item.logprob,
                    top_logprobs={lp.token: lp.logprob for lp in item.top_logprobs}
                    if item.top_logprobs
                    else None,
                )
                for item in choice_logprobs.content
            )
            if raw_ids is not None:
                if len(raw_ids) != len(token_logprobs):
                    raise ValueError("Native token IDs and logprobs have different lengths")
                token_logprobs = tuple(
                    replace(lp, token_id=token_id)
                    for token_id, lp in zip(raw_ids, token_logprobs, strict=True)
                )
        if params.logprobs and not token_logprobs:
            raise LogprobsNotReturnedError(
                f"OpenAI returned no token logprobs despite logprobs=True for model {self._model!r}.",
                backend_name="OpenAIBackend",
                model_name=self._model,
            )
        parsed_calls = []
        for call in getattr(choice.message, "tool_calls", None) or ():
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {"raw": call.function.arguments}
            parsed_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        result = GenerationResult(
            text=choice.message.content if parsed_calls else choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            tool_calls=tuple(parsed_calls),
            token_logprobs=token_logprobs,
            **normalize_usage(response.usage),
            metadata={
                "model": response.model,
                "id": response.id,
                "raw_usage": plain(response.usage),
                "raw_text": choice.message.content,
                "reasoning": getattr(choice.message, "reasoning_content", None)
                or getattr(choice.message, "reasoning", None),
                "service_tier": getattr(response, "service_tier", None),
                "request_id": getattr(response, "_request_id", None),
                **metadata,
            },
        )
        tokenizer = getattr(self, "_reasoning_tokenizer", None)
        if raw_ids is not None and tokenizer is not None:
            from llenvs.inference.reasoning import extract_local_reasoning

            template = metadata.get("request", {}).get("chat_template_kwargs", {})
            return extract_local_reasoning(
                result,
                raw_ids,
                tokenizer,
                reasoning_enabled=params.reasoning_effort != "none"
                and template.get("enable_thinking", True),
            )
        reasoning = result.metadata.get("reasoning")
        raw = {
            "raw_token_ids": list(raw_ids) if raw_ids is not None else None,
            "raw_token_logprobs": [asdict(lp) for lp in token_logprobs or ()],
        }
        if reasoning:
            return replace(
                result,
                token_logprobs=None,
                metadata={
                    **result.metadata,
                    **raw,
                    "answer_span": None,
                    "reasoning_truncated": result.finish_reason == StopReason.MAX_TOKENS
                    and not result.text,
                },
            )
        return replace(result, metadata={**result.metadata, **raw})

    def descriptor(self) -> dict[str, Any]:
        from llenvs.inference.sampling import _record_value

        return {
            "identity": {
                **super().descriptor()["identity"],
                "api_flavor": self._api_flavor,
                "endpoint": _record_value({"base_url": str(self._client.base_url)})["base_url"],
                "dependencies": dependency_versions("openai"),
                "rate_limit_wait": self._rate_limit_wait,
                "rate_limit_max_retries": self._rate_limit_max_retries,
                "sdk_max_retries": self._client.max_retries,
            },
            "provenance": {},
        }

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        """Generate one response, recording the request and every backend attempt."""
        return self._request(self._request_kwargs(messages, params), params)

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        return await self._request_async(self._request_kwargs(messages, params), params)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
        *,
        observer=None,
    ) -> list[GenerationResult]:
        requests = [self._request_kwargs(messages, params) for messages in messages_batch]
        return _run_concurrent(
            lambda item: self._request_async(item[1], params, observer=observer, index=item[0]),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=getattr(self, "_return_partial_batch", True),
            runner=self._async_runner,
        )

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._request(self._request_kwargs(messages, params, tools, tool_choice), params)

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return await self._request_async(
            self._request_kwargs(messages, params, tools, tool_choice), params
        )

    def generate_with_tools_batch(
        self, messages_batch, tools, params, tool_choice="auto", *, observer=None
    ):
        requests = [
            self._request_kwargs(messages, params, tools, tool_choice)
            for messages in messages_batch
        ]
        return _run_concurrent(
            lambda item: self._request_async(item[1], params, observer=observer, index=item[0]),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=True,
            runner=self._async_runner,
        )


class AnthropicBackend(_APIRequests, ModelBackend):
    """Anthropic API backend.

    Supports Claude models. Includes prefix continuation via assistant prefill.
    """

    _api_flavor = "anthropic"

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
        self._api_flavor = "anthropic"
        self._rate_limit_wait = 0.0
        self._rate_limit_max_retries = 0
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
        self._async_runner = _AsyncRunner()
        self._closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        """Anthropic capabilities."""
        return BackendCapabilities(
            supports_explicit_sampling=True,
            supports_logprobs=False,
            supports_prefix_continuation=True,  # Via assistant prefill
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_request_progress=True,
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
        _close_api_clients(self)

    def _request_kwargs(self, messages, params, tools=None, tool_choice="auto"):
        validate_sampling(
            params,
            unsupported={
                "min_p": 0,
                "repetition_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
        )
        if params.max_tokens is None:
            raise ValueError("Anthropic requires max_tokens")
        if (
            params.logprobs
            or params.response_schema is not None
            or params.after_reasoning_text is not None
        ):
            raise ValueError(
                "Anthropic does not support the requested logprobs/schema/insertion control"
            )
        if params.reasoning_effort not in {None, "none"}:
            raise ValueError("No Anthropic model mapping for this reasoning_effort")
        system = [
            message.content for message in messages if message.role == "system" and message.content
        ]
        kwargs = {
            "model": self._model,
            "messages": [
                message.to_anthropic_dict() for message in messages if message.role != "system"
            ],
            "max_tokens": params.max_tokens,
        }
        if system:
            kwargs["system"] = "\n\n".join(system)
        for name in ("temperature", "top_p", "top_k"):
            if (value := getattr(params, name)) is not None:
                kwargs[name] = value
        if params.stop_sequences:
            kwargs["stop_sequences"] = list(params.stop_sequences)
        if tools is not None and tool_choice != "none":
            kwargs["tools"] = [tool.to_anthropic_schema() for tool in tools]
            kwargs["tool_choice"] = (
                {"type": "any" if tool_choice == "required" else "auto"}
                if tool_choice in {"auto", "required"}
                else {"type": "tool", "name": tool_choice}
            )
        if params.reasoning_effort == "none":
            kwargs["thinking"] = {"type": "disabled"}
        elif params.thinking_budget is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": params.thinking_budget}
        kwargs.update(deepcopy(params.extra))
        return kwargs

    def _parse_response(self, response, params, metadata):
        text = []
        calls = []
        reasoning = []
        for block in response.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
            elif block.type in {"thinking", "redacted_thinking"}:
                reasoning.append(plain(block))
        return GenerationResult(
            text="".join(text) if text or not calls else None,
            tool_calls=tuple(calls),
            finish_reason=_anthropic_stop_reason(response.stop_reason),
            **normalize_usage(response.usage, provider="anthropic"),
            metadata={
                **metadata,
                "model": response.model,
                "id": response.id,
                "raw_usage": plain(response.usage),
                "reasoning_details": reasoning,
                "raw_content": plain(response.content),
            },
        )

    def reparse_receipt(self, receipt, params):
        from anthropic.types import Message

        return self._parse_response(
            Message.construct(**receipt["response"]), params, receipt["metadata"]
        )

    def generate(self, prompts, params, *, observer=None):
        if observer is None:
            return [self.generate_chat([ChatMessage("user", prompt)], params) for prompt in prompts]
        return self.generate_chat_batch(
            [[ChatMessage("user", prompt)] for prompt in prompts], params, observer=observer
        )

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        """Generate one response, recording the request and every backend attempt."""
        return self._request(self._request_kwargs(messages, params), params)

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        return await self._request_async(self._request_kwargs(messages, params), params)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
        *,
        observer=None,
    ) -> list[GenerationResult]:
        requests = [self._request_kwargs(messages, params) for messages in messages_batch]
        return _run_concurrent(
            lambda item: self._request_async(item[1], params, observer=observer, index=item[0]),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=getattr(self, "_return_partial_batch", True),
            runner=self._async_runner,
        )

    def continue_from_prefix(self, prefix, params, num_continuations=1):
        """Return every requested draw using native assistant prefill."""
        from dataclasses import replace

        if num_continuations < 1:
            raise ValueError("num_continuations must be positive")
        validate_sampling(params)
        messages = [
            ChatMessage("user", "Continue the following:"),
            ChatMessage("assistant", prefix),
        ]
        results = [self.generate_chat(messages, params) for _ in range(num_continuations)]
        return [
            replace(result, metadata={**result.metadata, "is_continuation": True})
            for result in results
        ]

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._request(self._request_kwargs(messages, params, tools, tool_choice), params)

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return await self._request_async(
            self._request_kwargs(messages, params, tools, tool_choice), params
        )

    def generate_with_tools_batch(
        self, messages_batch, tools, params, tool_choice="auto", *, observer=None
    ):
        requests = [
            self._request_kwargs(messages, params, tools, tool_choice)
            for messages in messages_batch
        ]
        return _run_concurrent(
            lambda item: self._request_async(item[1], params, observer=observer, index=item[0]),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=True,
            runner=self._async_runner,
        )


class OpenRouterBackend(_APIRequests, ModelBackend):
    """OpenRouter API backend.

    Provides access to multiple model providers through a unified API.
    Capabilities vary by underlying model.
    """

    _api_flavor = "openrouter"

    def _request_kwargs(self, messages, params, tools=None, tool_choice="auto"):
        kwargs = self._chat_kwargs(messages, params)
        if tools is not None:
            kwargs["tools"] = [tool.to_openai_schema() for tool in tools]
            kwargs["tool_choice"] = (
                tool_choice
                if tool_choice in {"auto", "none", "required"}
                else {"type": "function", "function": {"name": tool_choice}}
            )
        return kwargs

    def _parse_response(self, response, params, metadata):
        from dataclasses import replace

        result = self._ensure_logprobs_if_requested(self._chat_result(response), params)
        return replace(result, metadata={**result.metadata, **metadata})

    def _failed_attempt(self, exc, started, attempts, call_started):
        # Preserve capability diagnostics for provider routing failures.
        if _error_status_code(exc) == 404 and "logprob" in str(exc).lower():
            raise LogprobsNotReturnedError(
                str(exc), backend_name="OpenRouterBackend", model_name=self._model
            ) from exc
        return super()._failed_attempt(exc, started, attempts, call_started)

    def reparse_receipt(self, receipt, params):
        from openai.types.chat import ChatCompletion

        return self._parse_response(
            ChatCompletion.construct(**receipt["response"]), params, receipt["metadata"]
        )

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
        provider: dict[str, Any] | None = None,
        read_timeout: float | None = None,
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
                error.  When set to ``0`` (the default) the SDK's built-in
                retry/backoff handles 429s.  Set to e.g. ``60`` for free-tier
                models with strict per-minute caps. The retry path covers both
                ``RateLimitError`` (HTTP 429) and ``MalformedResponseError``
                whose provider message is rate-limit-shaped — some providers
                (notably Alibaba) surface their own 429 as an HTTP-200 with
                ``choices=null`` plus a textual rate-limit message in the
                OpenRouter top-level ``error``, which would otherwise bypass
                the SDK's rate-limit handling.
            rate_limit_max_retries: Maximum number of rate-limit retries before
                giving up and re-raising the error.
            timeout: Connect timeout in seconds. ``None`` uses the SDK default.
            read_timeout: Read timeout in seconds, independently of connection
                establishment. ``None`` keeps the SDK default (600s).
            max_retries: Maximum SDK-level retries for transient errors
                (timeouts, connection errors, 5xx).  ``None`` uses SDK
                default (2).
            provider: Default ``extra_body.provider`` routing dict sent with
                every request (e.g. ``{"order": ["Parasail"],
                "allow_fallbacks": true}``).  Per-request overrides via
                ``params.extra`` still take precedence.  Useful when a model
                has several OpenRouter providers with diverging per-prompt
                image caps and we want to pin to one that accepts our
                workload.
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

        if rate_limit_wait > 0:
            if max_retries not in (None, 0):
                raise ValueError("Backend retries require SDK max_retries=0")
            max_retries = 0
        self._model = model
        self._max_concurrency = max_concurrency
        self._rate_limit_wait = rate_limit_wait
        self._rate_limit_max_retries = rate_limit_max_retries
        self._default_provider = dict(provider) if provider else None
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
        _configure_openai_timeout(client_args, timeout, read_timeout)
        if max_retries is not None:
            client_args["max_retries"] = max_retries

        self._client = OpenAI(**client_args)
        self._async_client = AsyncOpenAI(**client_args)
        self._async_runner = _AsyncRunner()
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
            supports_explicit_sampling=True,
            supports_logprobs=True,
            supports_prefix_continuation=False,
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_request_progress=True,
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
        _close_api_clients(self)

    def generate(self, prompts, params, *, observer=None):
        if observer is None:
            return [self.generate_chat([ChatMessage("user", prompt)], params) for prompt in prompts]
        return self.generate_chat_batch(
            [[ChatMessage("user", prompt)] for prompt in prompts], params, observer=observer
        )

    # OpenRouter documents an explicit cap of 20 on ``top_logprobs`` (OpenAPI
    # schema ``ChatRequest.top_logprobs`` at openrouter.ai/openapi.json).
    _TOP_LOGPROBS_MAX = 20

    def _chat_kwargs(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> dict[str, Any]:
        """Build kwargs dict for chat completions."""
        validate_sampling(params)
        if params.after_reasoning_text is not None:
            raise ValueError("after_reasoning_text requires a configured local backend")
        if params.reasoning_effort is not None and params.reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("Unknown OpenRouter reasoning_effort")
        if params.reasoning_effort is not None and any(
            "reasoning" in mapping for mapping in (params.extra, params.extra.get("extra_body", {}))
        ):
            raise ValueError("reasoning_effort conflicts with per-request reasoning extras")
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
        extra_body: dict[str, Any] = {}
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        for name in ("top_k", "min_p", "repetition_penalty"):
            if (value := getattr(params, name)) is not None:
                extra_body[name] = value
        if params.response_schema is not None:
            schema = params.response_schema
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.name,
                    "schema": deepcopy(schema.schema),
                    "strict": schema.strict,
                },
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
            provider_cfg = extra_body.setdefault("provider", {})
            provider_cfg.setdefault("require_parameters", True)

        # Backend-level default provider routing (e.g. pinning to a
        # vision-capable provider with a generous per-prompt image cap).
        # Per-request ``params.extra`` still wins via the update() below.
        default_provider = getattr(self, "_default_provider", None)
        if default_provider:
            provider_cfg = extra_body.setdefault("provider", {})
            for key, value in default_provider.items():
                provider_cfg.setdefault(key, value)

        # OpenRouter's unified ``reasoning`` block lives at the top level of
        # the request body, but the OpenAI SDK doesn't know that field — it
        # has to ride through ``extra_body``. Map ``thinking_budget`` onto
        # ``reasoning.max_tokens`` (covers Anthropic ``thinking.budget_tokens``,
        # Gemini thinking budget, Qwen ``thinking_budget``; effort-only models
        # get a derived effort bucket on the OpenRouter side). Caller-supplied
        # ``extra.extra_body`` shallow-merges on top so the explicit escape
        # hatch wins on key collisions.
        # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        if params.thinking_budget is not None and params.reasoning_effort != "none":
            extra_body["reasoning"] = {"max_tokens": params.thinking_budget}

        user_extras = dict(params.extra) if params.extra else {}
        user_extra_body = user_extras.pop("extra_body", None)
        if user_extra_body:
            extra_body = {**extra_body, **user_extra_body}
        if params.reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": params.reasoning_effort}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if user_extras:
            kwargs.update(user_extras)

        return kwargs

    def _validated_choice(self, response: Any) -> Any:
        """Reject provider failures or incomplete completions before parsing."""
        choices = getattr(response, "choices", None)
        provider_error = getattr(response, "error", None)
        choice = choices[0] if choices else None
        if provider_error is None and choice is not None:
            provider_error = getattr(choice, "error", None)
        error_stop = choice is not None and any(
            str(getattr(choice, field, "")).lower() == "error"
            for field in ("finish_reason", "native_finish_reason")
        )
        if not choices or provider_error is not None or error_stop:
            message = (
                _format_no_choices_message(provider_error)
                if not choices
                else f"OpenRouter returned a provider error completion: {provider_error!r}"
            )
            error = MalformedResponseError(
                message,
                backend_name="OpenRouterBackend",
                model_name=self._model,
                provider_error=provider_error,
            )
            normalized = _normalize_provider_error(error, model_name=self._model)
            if normalized is error:
                raise error
            raise normalized from error
        finish_reason = getattr(choice, "finish_reason", None)
        native_finish_reason = getattr(choice, "native_finish_reason", None)
        if not finish_reason and not native_finish_reason:
            completion_id = getattr(response, "id", "unknown")
            raise MalformedResponseError(
                "OpenRouter returned a completion without termination metadata; "
                f"the visible response may be incomplete (id={completion_id})",
                backend_name="OpenRouterBackend",
                model_name=self._model,
            )
        return choice

    def _chat_result(self, response: Any) -> GenerationResult:
        """Convert an OpenAI-compatible ChatCompletion to GenerationResult.

        Logprobs follow the OpenAI response shape — OpenRouter's OpenAPI
        schema ``ChatTokenLogprobs`` at ``choices[i].logprobs.content[]``
        matches OpenAI field-for-field (``token``, ``logprob``,
        ``top_logprobs: [{token, logprob, bytes}]``), so the same
        extraction works.
        """
        choice = self._validated_choice(response)

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
                        token_id=None,  # OpenRouter/OpenAI do not return token IDs
                        logprob=token_info.logprob,
                        top_logprobs=top_lps,
                    )
                )
            token_logprobs = tuple(logprobs)

        usage = getattr(response, "usage", None)
        message = choice.message
        metadata: dict[str, Any] = {
            "model": response.model,
            "id": response.id,
        }

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        native_finish_reason = getattr(choice, "native_finish_reason", None)
        if native_finish_reason is not None:
            metadata["native_finish_reason"] = native_finish_reason

        completion_tokens_details = _to_plain_mapping(
            _get_field(usage, "completion_tokens_details")
        )
        if completion_tokens_details:
            metadata["completion_tokens_details"] = completion_tokens_details
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens")
            if reasoning_tokens is not None:
                metadata["reasoning_tokens"] = reasoning_tokens

        metadata.update(_reasoning_metadata(message))
        metadata["raw_usage"] = plain(usage)

        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=_openai_stop_reason(choice.finish_reason),
            token_logprobs=token_logprobs,
            **normalize_usage(usage),
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
                "OpenRouter returned no token logprobs despite logprobs=True. "
                "The underlying provider likely does not expose logprobs for "
                f"model {self._model!r}.",
                backend_name="OpenRouterBackend",
                model_name=self._model,
            )
        return result

    def _translate_router_404_for_logprobs(
        self,
        exc: BaseException,
        params: SamplingParams,
    ) -> BaseException:
        """Translate OpenRouter's "no endpoints can handle params" 404 into
        :class:`LogprobsNotReturnedError` when logprobs were requested.

        With ``provider.require_parameters=True`` set for every logprobs
        request, OpenRouter returns a 404 if no provider supports the full
        set of requested parameters.  From the caller's perspective this
        is the same failure mode as a provider silently dropping
        logprobs, so we surface it with the graceful-skip exception type
        instead of a raw :class:`openai.NotFoundError`.
        """
        try:
            from openai import NotFoundError
        except ImportError:
            return exc
        if (
            params.logprobs
            and isinstance(exc, NotFoundError)
            and "no endpoints found" in str(exc).lower()
        ):
            return LogprobsNotReturnedError(
                "OpenRouter could not route this request: no provider "
                "supports the full set of requested parameters (logprobs "
                f"+ everything else in the payload) for model {self._model!r}. "
                "Original error: " + str(exc),
                backend_name="OpenRouterBackend",
                model_name=self._model,
            )
        return exc

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        """Generate one response, recording the request and every backend attempt."""
        return self._request(self._request_kwargs(messages, params), params)

    async def _generate_chat_async(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        return await self._request_async(self._request_kwargs(messages, params), params)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
        *,
        observer=None,
    ) -> list[GenerationResult]:
        requests = [self._request_kwargs(messages, params) for messages in messages_batch]
        return _run_concurrent(
            lambda item: self._request_async(item[1], params, observer=observer, index=item[0]),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=getattr(self, "_return_partial_batch", True),
            runner=self._async_runner,
        )

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._request(self._request_kwargs(messages, params, tools, tool_choice), params)

    async def _generate_with_tools_async(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return await self._request_async(
            self._request_kwargs(messages, params, tools, tool_choice), params
        )

    def generate_with_tools_batch(
        self, messages_batch, tools, params, tool_choice="auto", *, observer=None
    ):
        requests = [
            self._request_kwargs(messages, params, tools, tool_choice)
            for messages in messages_batch
        ]
        return _run_concurrent(
            lambda item: self._request_async(item[1], params, observer=observer, index=item[0]),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=True,
            runner=self._async_runner,
        )
