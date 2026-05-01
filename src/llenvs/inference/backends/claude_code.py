"""Claude Code CLI backend.

Wraps ``claude --print --output-format json`` behind the ``ModelBackend``
interface.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import shutil
import subprocess
import tempfile
from typing import Any

from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    PartialBatchError,
    PromptTooLongError,
    QuotaExhaustedError,
    RefusedByPolicyError,
    SamplingParams,
    StopReason,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED_PARAM_TEMPLATE = "ClaudeCodeBackend does not support SamplingParams.{name}; {hint}"

_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_VALID_PERMISSION_MODES = (
    "acceptEdits",
    "auto",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
)

_CONTEXT_LIMIT_NEEDLES = (
    "prompt is too long",
    "input is too long",
    "context length",
    "maximum context length",
    "context window",
    "request is too large",
    "too many tokens",
    "reduce the length",
)
_AUTH_ERROR_NEEDLES = (
    "invalid api key",
    "authentication_error",
    "authentication failed",
    "api key not found",
    "api key is invalid",
    "not logged in",
    "please run /login",
    "unauthorized",
)
_AUP_REFUSAL_NEEDLES = (
    "usage policy",
    "anthropic.com/legal/aup",
)


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


def _usage_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt_tokens = _coerce_int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    completion_tokens = _coerce_int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
    return max(prompt_tokens, 0), max(completion_tokens, 0)


def _is_context_limit_text(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in _CONTEXT_LIMIT_NEEDLES)


def _is_auth_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in _AUTH_ERROR_NEEDLES)


def _is_aup_refusal_text(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in _AUP_REFUSAL_NEEDLES)


def _diagnostic_summary(stderr: str, stdout: str) -> str:
    parts = [part.strip() for part in (stderr, stdout) if part.strip()]
    if not parts:
        return "claude --print exited without diagnostic output"
    summary = " ".join(parts)
    return summary[:500]


def _claude_stop_reason(reason: Any) -> StopReason:
    if not isinstance(reason, str):
        return StopReason.UNKNOWN
    lowered = reason.lower()
    if lowered == "end_turn":
        return StopReason.END_OF_TEXT
    if lowered == "max_tokens":
        return StopReason.MAX_TOKENS
    if lowered == "stop_sequence":
        return StopReason.STOP_SEQUENCE
    if lowered == "tool_use":
        return StopReason.TOOL_USE
    return StopReason.UNKNOWN


def _extract_text(payload: dict[str, Any]) -> str | None:
    """Extract assistant text from a ``--output-format json`` payload.

    The documented and observed shape is ``{"result": "<text>", ...}``.
    Defensive fallbacks for ``content`` (string or block list) are kept
    so a future schema drift does not silently lose output.
    """
    result = payload.get("result")
    if isinstance(result, str):
        return result
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text
    return None


class ClaudeCodeBackend(ModelBackend):
    """ModelBackend wrapper around ``claude --print``."""

    def __init__(
        self,
        model: str = "sonnet",
        command: str = "claude",
        max_concurrency: int = 4,
        timeout: float = 600.0,
        *,
        bare: bool = False,
        tools: str | None = "",
        disallowed_tools: str | None = None,
        setting_sources: str | None = None,
        permission_mode: str | None = None,
        effort: str | None = None,
        max_budget_usd: float | None = None,
        fallback_model: str | None = None,
        system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        exclude_dynamic_system_prompt_sections: bool = False,
        no_session_persistence: bool = True,
        strict_mcp_config: bool = True,
        extra_args: list[str] | None = None,
    ) -> None:
        if not model:
            raise ValueError("ClaudeCodeBackend model must be non-empty")
        if max_concurrency < 1:
            raise ValueError("ClaudeCodeBackend max_concurrency must be >= 1")
        if timeout <= 0:
            raise ValueError("ClaudeCodeBackend timeout must be > 0")
        if effort is not None and effort not in _VALID_EFFORTS:
            raise ValueError(
                f"ClaudeCodeBackend effort must be one of {_VALID_EFFORTS}, got {effort!r}"
            )
        if permission_mode is not None and permission_mode not in _VALID_PERMISSION_MODES:
            raise ValueError(
                "ClaudeCodeBackend permission_mode must be one of "
                f"{_VALID_PERMISSION_MODES}, got {permission_mode!r}"
            )
        if max_budget_usd is not None and max_budget_usd <= 0:
            raise ValueError("ClaudeCodeBackend max_budget_usd must be > 0")

        if extra_args is not None:
            for entry in extra_args:
                if not isinstance(entry, str):
                    raise ValueError("ClaudeCodeBackend extra_args must contain only strings")

        resolved_command = shutil.which(command)
        if resolved_command is None:
            raise ImportError(
                "claude CLI is required for ClaudeCodeBackend. "
                "Install Claude Code and ensure the `claude` binary is on PATH."
            )

        self._command = resolved_command
        self._model = model
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._bare = bare
        self._tools = tools
        self._disallowed_tools = disallowed_tools
        self._setting_sources = setting_sources
        self._permission_mode = permission_mode
        self._effort = effort
        self._max_budget_usd = max_budget_usd
        self._fallback_model = fallback_model
        self._system_prompt = system_prompt
        self._append_system_prompt = append_system_prompt
        self._exclude_dynamic_system_prompt_sections = exclude_dynamic_system_prompt_sections
        self._no_session_persistence = no_session_persistence
        self._strict_mcp_config = strict_mcp_config
        self._extra_args: list[str] = list(extra_args or [])

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_concurrency) as executor:
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
                "Claude Code CLI does not expose a direct equivalent.",
            ),
            (
                "top_p",
                params.top_p != 1.0,
                "Claude Code CLI does not expose a direct equivalent.",
            ),
            (
                "top_k",
                params.top_k != 0,
                "Claude Code CLI does not expose a direct equivalent.",
            ),
            (
                "presence_penalty",
                params.presence_penalty != 0.0,
                "Claude Code CLI does not expose a direct equivalent.",
            ),
            (
                "frequency_penalty",
                params.frequency_penalty != 0.0,
                "Claude Code CLI does not expose a direct equivalent.",
            ),
            (
                "n",
                params.n != 1,
                "Claude Code CLI returns only one final assistant message.",
            ),
            (
                "stop_sequences",
                bool(params.stop_sequences),
                "Claude Code CLI does not expose direct stop-sequence controls.",
            ),
            (
                "logprobs",
                params.logprobs,
                "Claude Code CLI does not return token logprobs.",
            ),
            (
                "thinking_budget",
                params.thinking_budget is not None,
                "Use the `effort` constructor argument to control reasoning level.",
            ),
            (
                "thinking_budget_per_block",
                params.thinking_budget_per_block,
                "Use the `effort` constructor argument to control reasoning level.",
            ),
            (
                "thinking_budget_soft_ratio",
                params.thinking_budget_soft_ratio is not None,
                "Use the `effort` constructor argument to control reasoning level.",
            ),
            (
                "thinking_budget_suffix",
                params.thinking_budget_suffix is not None,
                "Claude Code CLI does not expose thinking suffix controls.",
            ),
            (
                "second_elicitation_suffix",
                params.second_elicitation_suffix is not None,
                "Claude Code CLI does not expose second-elicitation controls.",
            ),
            (
                "second_elicitation_max_tokens",
                params.second_elicitation_max_tokens != 256,
                "Claude Code CLI does not expose second-elicitation controls.",
            ),
            (
                "extra",
                bool(params.extra),
                "Use ClaudeCodeBackend(extra_args=...) for Claude Code-specific flags.",
            ),
        ]

        for name, is_set, hint in unsupported:
            if is_set:
                raise ValueError(_UNSUPPORTED_PARAM_TEMPLATE.format(name=name, hint=hint))

    def _generate_chat_impl(self, messages: list[ChatMessage]) -> GenerationResult:
        for message in messages:
            if message.images:
                raise ValueError(
                    "ClaudeCodeBackend is text-only and does not support image-bearing chat messages"
                )

        prompt = _render_chat_prompt(messages)
        cmd = self._build_command()

        with tempfile.TemporaryDirectory(prefix="llenvs-claude-") as tmpdir:
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
                    f"claude --print timed out after {self._timeout:.1f}s "
                    "(treated as quota-exhausted)"
                ) from exc

        return self._classify_and_parse(completed)

    def _classify_and_parse(self, completed: subprocess.CompletedProcess[str]) -> GenerationResult:
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        try:
            payload = json.loads(stdout) if stdout.strip() else None
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            return self._handle_json_payload(payload, completed.returncode, stderr)

        diagnostic = _diagnostic_summary(stderr, stdout)
        if completed.returncode == 0:
            raise QuotaExhaustedError(f"claude --print returned non-JSON output: {diagnostic}")
        raise self._classify_error_text(diagnostic, completed.returncode, subtype=None)

    def _handle_json_payload(
        self,
        payload: dict[str, Any],
        returncode: int,
        stderr: str,
    ) -> GenerationResult:
        text = _extract_text(payload)
        is_error = bool(payload.get("is_error"))

        if is_error:
            error_text = text or stderr or "<unknown>"
            subtype = payload.get("subtype")
            subtype_str = subtype if isinstance(subtype, str) else None
            if subtype_str is not None:
                error_text = f"[subtype={subtype_str}] {error_text}"
            raise self._classify_error_text(error_text, returncode, subtype=subtype_str)

        if text is None:
            keys = ", ".join(sorted(payload)[:20])
            raise QuotaExhaustedError(
                f"claude --print succeeded but JSON had no recognized text field; got keys=[{keys}]"
            )

        usage_dict: dict[str, Any] | None = None
        for usage in _iter_usage_dicts(payload):
            usage_dict = usage
            break
        prompt_tokens, completion_tokens = _usage_counts(usage_dict)
        finish_reason = _claude_stop_reason(payload.get("stop_reason"))

        metadata = {
            "model": self._model,
            "command": self._command,
            "returncode": returncode,
            "session_id": payload.get("session_id"),
            "subtype": payload.get("subtype"),
            "stop_reason": payload.get("stop_reason"),
            "is_error": False,
            "raw_usage": usage_dict,
            "total_cost_usd": payload.get("total_cost_usd"),
            "duration_ms": payload.get("duration_ms"),
            "duration_api_ms": payload.get("duration_api_ms"),
            "num_turns": payload.get("num_turns"),
        }

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata=metadata,
        )

    def _classify_error_text(
        self,
        text: str,
        returncode: int,
        *,
        subtype: str | None,
    ) -> BaseException:
        if _is_context_limit_text(text):
            return PromptTooLongError(
                text,
                model_name=self._model,
                offending_indices=[0],
            )
        if _is_auth_error_text(text):
            return RuntimeError(f"ClaudeCodeBackend: authentication failed: {text}")
        if _is_aup_refusal_text(text):
            return RefusedByPolicyError(
                f"ClaudeCodeBackend: refused by Anthropic Usage Policy: {text}",
                subtype=subtype,
                offending_indices=[0],
            )
        if returncode != 0:
            return QuotaExhaustedError(f"claude --print failed with exit code {returncode}: {text}")
        return QuotaExhaustedError(f"claude --print reported is_error=true: {text}")

    def _build_command(self) -> list[str]:
        cmd: list[str] = [
            self._command,
            "--print",
            "--output-format",
            "json",
            "--input-format",
            "text",
            "--model",
            self._model,
        ]
        if self._bare:
            cmd.append("--bare")
        if self._tools is not None:
            cmd.extend(["--tools", self._tools])
        if self._disallowed_tools is not None:
            cmd.extend(["--disallowedTools", self._disallowed_tools])
        if self._permission_mode is not None:
            cmd.extend(["--permission-mode", self._permission_mode])
        if self._setting_sources is not None:
            cmd.extend(["--setting-sources", self._setting_sources])
        if self._effort is not None:
            cmd.extend(["--effort", self._effort])
        if self._max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self._max_budget_usd)])
        if self._fallback_model is not None:
            cmd.extend(["--fallback-model", self._fallback_model])
        if self._system_prompt is not None:
            cmd.extend(["--system-prompt", self._system_prompt])
        if self._append_system_prompt is not None:
            cmd.extend(["--append-system-prompt", self._append_system_prompt])
        if self._exclude_dynamic_system_prompt_sections:
            cmd.append("--exclude-dynamic-system-prompt-sections")
        if self._no_session_persistence:
            cmd.append("--no-session-persistence")
        if self._strict_mcp_config:
            cmd.append("--strict-mcp-config")
        cmd.extend(self._extra_args)
        return cmd
