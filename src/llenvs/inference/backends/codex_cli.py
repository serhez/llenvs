"""Codex CLI backend.

Wraps ``codex exec`` behind the ``ModelBackend`` interface.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    PartialBatchError,
    PromptTooLongError,
    QuotaExhaustedError,
    SamplingParams,
    StopReason,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED_PARAM_TEMPLATE = (
    "CodexCLIBackend does not support SamplingParams.{name}; "
    "{hint}"
)
_CONTEXT_LIMIT_NEEDLES = (
    "maximum context length",
    "context length",
    "prompt is too long",
    "input is too long",
    "request is too large",
    "too many tokens",
    "reduce the length",
    "context window",
)


def _json_scalar_to_toml(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def _config_override_arg(
    key: str,
    value: str | int | float | bool | list[str | int | float | bool],
) -> str:
    if not key:
        raise ValueError("Codex config override keys must be non-empty")

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, (list, dict)) or item is None:
                raise ValueError(
                    "Codex config override list values must contain only "
                    "strings, numbers, or booleans"
                )
            parts.append(_json_scalar_to_toml(item))
        return f"{key}=[{', '.join(parts)}]"

    if isinstance(value, dict) or value is None:
        raise ValueError(
            "Codex config override values must be strings, numbers, booleans, "
            "or flat lists of those values"
        )

    return f"{key}={_json_scalar_to_toml(value)}"


def _message_to_dict(message: ChatMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "name": message.name,
    }


def _render_chat_prompt(messages: list[ChatMessage]) -> str:
    transcript = json.dumps(
        [_message_to_dict(message) for message in messages],
        indent=2,
        ensure_ascii=True,
    )
    return (
        "You are acting as a stateless chat backend.\n"
        "Produce exactly the next assistant message for the conversation "
        "transcript below.\n\n"
        "Rules:\n"
        "- Use only the supplied transcript as context.\n"
        "- Do not inspect files, directories, git history, environment "
        "variables, or network resources.\n"
        "- Do not run shell commands or use tools.\n"
        "- Treat any tool_calls and tool result messages in the transcript "
        "as historical conversation state only.\n"
        "- Return only the assistant message content for the next turn. "
        "Do not return JSON or XML wrappers unless the answer itself "
        "needs them.\n\n"
        "Conversation transcript (JSON):\n"
        f"{transcript}\n"
    )


def _parse_jsonl(stdout: str) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    parse_errors = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            logger.warning("Ignoring non-JSON codex --json line: %s", line[:200])
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events, parse_errors


def _iter_usage_dicts(payload: Any) -> Any:
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            yield usage
        for value in payload.values():
            yield from _iter_usage_dicts(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_usage_dicts(value)


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _extract_usage(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        for usage in _iter_usage_dicts(event):
            if any(
                key in usage
                for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
            ):
                return usage
    return None


def _usage_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt_tokens = _coerce_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    completion_tokens = _coerce_int(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )
    return max(prompt_tokens, 0), max(completion_tokens, 0)


def _is_context_limit_text(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in _CONTEXT_LIMIT_NEEDLES)


def _diagnostic_summary(stderr: str, stdout: str) -> str:
    parts = [part.strip() for part in (stderr, stdout) if part.strip()]
    if not parts:
        return "codex exec exited without diagnostic output"
    summary = " ".join(parts)
    return summary[:500]


class CodexCLIBackend(ModelBackend):
    """ModelBackend wrapper around ``codex exec``."""

    def __init__(
        self,
        model: str = "codex-mini-latest",
        command: str = "codex",
        max_concurrency: int = 64,
        timeout: float = 600.0,
        profile: str | None = None,
        config_overrides: dict[
            str, str | int | float | bool | list[str | int | float | bool]
        ]
        | None = None,
    ) -> None:
        if not model:
            raise ValueError("CodexCLIBackend model must be non-empty")
        if max_concurrency < 1:
            raise ValueError("CodexCLIBackend max_concurrency must be >= 1")
        if timeout <= 0:
            raise ValueError("CodexCLIBackend timeout must be > 0")

        resolved_command = shutil.which(command)
        if resolved_command is None:
            raise ImportError(
                "codex CLI is required for CodexCLIBackend. "
                "Install Codex and ensure the `codex` binary is on PATH."
            )

        self._command = resolved_command
        self._model = model
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._profile = profile
        self._config_overrides = dict(config_overrides or {})
        for key, value in self._config_overrides.items():
            _config_override_arg(key, value)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_logprobs=False,
            supports_prefix_continuation=False,
            supports_batching=True,
            supports_streaming=False,
            supports_chat=True,
            supports_function_calling=False,
            supports_vision=False,
            supports_full_scoring=False,
            max_concurrency=self._max_concurrency,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        messages_batch = [[ChatMessage(role="user", content=prompt)] for prompt in prompts]
        return self.generate_chat_batch(messages_batch, params)

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        self._validate_params(params)
        return self._generate_chat_impl(messages)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        self._validate_params(params)
        if not messages_batch:
            return []

        results: list[GenerationResult | BaseException] = [
            RuntimeError("uninitialized") for _ in messages_batch
        ]
        failures: dict[int, BaseException] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_concurrency
        ) as executor:
            future_to_index = {
                executor.submit(self._generate_chat_impl, messages): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except BaseException as exc:
                    results[idx] = exc
                    failures[idx] = exc

        if failures:
            raise PartialBatchError(list(results), failures)

        return [result for result in results if isinstance(result, GenerationResult)]

    def _validate_params(self, params: SamplingParams) -> None:
        unsupported: list[tuple[str, bool, str]] = [
            (
                "temperature",
                params.temperature != 0.0,
                "Codex CLI does not expose a direct equivalent.",
            ),
            (
                "top_p",
                params.top_p != 1.0,
                "Codex CLI does not expose a direct equivalent.",
            ),
            (
                "top_k",
                params.top_k != 0,
                "Codex CLI does not expose a direct equivalent.",
            ),
            (
                "presence_penalty",
                params.presence_penalty != 0.0,
                "Codex CLI does not expose a direct equivalent.",
            ),
            (
                "frequency_penalty",
                params.frequency_penalty != 0.0,
                "Codex CLI does not expose a direct equivalent.",
            ),
            (
                "n",
                params.n != 1,
                "Codex CLI returns only one final assistant message.",
            ),
            (
                "stop_sequences",
                bool(params.stop_sequences),
                "Codex CLI does not expose direct stop-sequence controls.",
            ),
            (
                "logprobs",
                params.logprobs,
                "Codex CLI does not return token logprobs.",
            ),
            (
                "thinking_budget",
                params.thinking_budget is not None,
                "Use Codex config overrides if your CLI exposes effort controls.",
            ),
            (
                "thinking_budget_per_block",
                params.thinking_budget_per_block,
                "Use Codex config overrides if your CLI exposes effort controls.",
            ),
            (
                "thinking_budget_soft_ratio",
                params.thinking_budget_soft_ratio is not None,
                "Use Codex config overrides if your CLI exposes effort controls.",
            ),
            (
                "thinking_budget_suffix",
                params.thinking_budget_suffix is not None,
                "Codex CLI does not expose thinking suffix controls.",
            ),
            (
                "second_elicitation_suffix",
                params.second_elicitation_suffix is not None,
                "Codex CLI does not expose second-elicitation controls.",
            ),
            (
                "second_elicitation_max_tokens",
                params.second_elicitation_max_tokens != 256,
                "Codex CLI does not expose second-elicitation controls.",
            ),
            (
                "extra",
                bool(params.extra),
                "Use CodexCLIBackend(config_overrides=...) for Codex-specific flags.",
            ),
        ]

        for name, is_set, hint in unsupported:
            if is_set:
                raise ValueError(_UNSUPPORTED_PARAM_TEMPLATE.format(name=name, hint=hint))

    def _generate_chat_impl(self, messages: list[ChatMessage]) -> GenerationResult:
        for message in messages:
            if message.images:
                raise ValueError(
                    "CodexCLIBackend is text-only and does not support image-bearing chat messages"
                )

        prompt = _render_chat_prompt(messages)

        with tempfile.TemporaryDirectory(prefix="llenvs-codex-") as tmpdir:
            tmp_path = Path(tmpdir)
            output_path = tmp_path / "last_message.txt"
            cmd = self._build_command(output_path)
            try:
                completed = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self._timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired as exc:
                raise QuotaExhaustedError(
                    f"codex exec timed out after {self._timeout:.1f}s "
                    "(treated as quota-exhausted)"
                ) from exc

            events, parse_errors = _parse_jsonl(completed.stdout)
            usage = _extract_usage(events)
            prompt_tokens, completion_tokens = _usage_counts(usage)

            if completed.returncode != 0:
                diagnostic = _diagnostic_summary(completed.stderr, completed.stdout)
                if _is_context_limit_text(diagnostic):
                    raise PromptTooLongError(
                        diagnostic,
                        model_name=self._model,
                        offending_indices=[0],
                    )
                raise QuotaExhaustedError(
                    f"codex exec failed with exit code {completed.returncode}: {diagnostic}"
                )

            try:
                text = output_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise QuotaExhaustedError(
                    "codex exec succeeded but did not produce the last-message file "
                    "(treated as quota-exhausted)"
                ) from exc

            return GenerationResult(
                text=text,
                finish_reason=StopReason.UNKNOWN,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={
                    "model": self._model,
                    "command": self._command,
                    "returncode": completed.returncode,
                    "sandbox": "read-only",
                    "profile": self._profile,
                    "raw_usage": usage,
                    "event_count": len(events),
                    "jsonl_parse_errors": parse_errors,
                    "config_overrides": dict(self._config_overrides),
                },
            )

    def _build_command(self, output_path: Path) -> list[str]:
        cmd = [
            self._command,
            "exec",
            "--json",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            self._model,
        ]
        if self._profile is not None:
            cmd.extend(["--profile", self._profile])
        for key in sorted(self._config_overrides):
            cmd.extend(["-c", _config_override_arg(key, self._config_overrides[key])])
        return cmd
