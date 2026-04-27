"""Tests for the Claude Code CLI backend."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from llenvs.core.state import ImageContent
from llenvs.core.tools import ToolCall
from llenvs.inference import (
    ChatMessage,
    GenerationResult,
    PartialBatchError,
    PromptTooLongError,
    QuotaExhaustedError,
    SamplingParams,
    StopReason,
)
from llenvs.inference.backends import ClaudeCodeBackend


def _make_backend(**kwargs: object) -> ClaudeCodeBackend:
    with patch(
        "llenvs.inference.backends.claude_code.shutil.which",
        return_value="/usr/bin/claude",
    ):
        return ClaudeCodeBackend(**kwargs)


def _success_payload(text: str = "hello", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "session_id": "session-abc",
        "total_cost_usd": 0.001,
        "duration_ms": 123,
        "duration_api_ms": 100,
        "num_turns": 1,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    payload.update(overrides)
    return payload


def _error_payload(text: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "result": text,
        "stop_reason": "stop_sequence",
        "session_id": "session-err",
        "total_cost_usd": 0,
        "duration_ms": 5,
        "duration_api_ms": 0,
        "num_turns": 1,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    payload.update(overrides)
    return payload


class TestClaudeCodeBackendInit:
    def test_exported_from_backends_package(self) -> None:
        assert ClaudeCodeBackend is not None

    def test_missing_binary_raises_import_error(self) -> None:
        with patch(
            "llenvs.inference.backends.claude_code.shutil.which",
            return_value=None,
        ):
            with pytest.raises(ImportError, match="claude CLI is required"):
                ClaudeCodeBackend()

    def test_capabilities_and_model_name(self) -> None:
        backend = _make_backend(model="claude-sonnet-4-6", max_concurrency=8)

        caps = backend.capabilities

        assert backend.model_name == "claude-sonnet-4-6"
        assert caps.supports_chat is True
        assert caps.supports_batching is True
        assert caps.supports_function_calling is False
        assert caps.supports_logprobs is False
        assert caps.supports_prefix_continuation is False
        assert caps.supports_streaming is False
        assert caps.supports_vision is False
        assert caps.supports_full_scoring is False
        assert caps.max_concurrency == 8

    def test_invalid_effort_raises(self) -> None:
        with pytest.raises(ValueError, match="effort must be one of"):
            _make_backend(effort="ludicrous")

    def test_invalid_permission_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="permission_mode must be one of"):
            _make_backend(permission_mode="bogus")

    def test_negative_max_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="max_budget_usd must be > 0"):
            _make_backend(max_budget_usd=-1.0)

    def test_extra_args_must_be_strings(self) -> None:
        with pytest.raises(ValueError, match="extra_args must contain only strings"):
            _make_backend(extra_args=["--ok", 5])  # type: ignore[list-item]

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model must be non-empty"):
            _make_backend(model="")


class TestClaudeCodeBackendCommandAssembly:
    def test_minimal_command_includes_safe_defaults(self) -> None:
        backend = _make_backend(model="sonnet")

        cmd = backend._build_command()

        assert cmd[:2] == ["/usr/bin/claude", "--print"]
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert "--input-format" in cmd
        assert cmd[cmd.index("--input-format") + 1] == "text"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert "--no-session-persistence" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--tools" in cmd
        assert cmd[cmd.index("--tools") + 1] == ""
        assert "--bare" not in cmd

    def test_command_includes_optional_flags_when_set(self) -> None:
        backend = _make_backend(
            model="claude-sonnet-4-6",
            bare=True,
            tools="Bash,Read",
            disallowed_tools="Edit",
            permission_mode="plan",
            setting_sources="project",
            effort="high",
            max_budget_usd=2.5,
            fallback_model="sonnet",
            system_prompt="You are X.",
            append_system_prompt="Always Y.",
            exclude_dynamic_system_prompt_sections=True,
        )

        cmd = backend._build_command()

        assert "--bare" in cmd
        assert cmd[cmd.index("--tools") + 1] == "Bash,Read"
        assert cmd[cmd.index("--disallowedTools") + 1] == "Edit"
        assert cmd[cmd.index("--permission-mode") + 1] == "plan"
        assert cmd[cmd.index("--setting-sources") + 1] == "project"
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert cmd[cmd.index("--max-budget-usd") + 1] == "2.5"
        assert cmd[cmd.index("--fallback-model") + 1] == "sonnet"
        assert cmd[cmd.index("--system-prompt") + 1] == "You are X."
        assert cmd[cmd.index("--append-system-prompt") + 1] == "Always Y."
        assert "--exclude-dynamic-system-prompt-sections" in cmd

    def test_extra_args_appended_verbatim(self) -> None:
        backend = _make_backend(extra_args=["--betas", "interleaved-thinking"])

        cmd = backend._build_command()

        assert cmd[-2:] == ["--betas", "interleaved-thinking"]

    def test_tools_none_omits_flag(self) -> None:
        backend = _make_backend(tools=None)

        cmd = backend._build_command()

        assert "--tools" not in cmd

    def test_tools_empty_string_emits_flag(self) -> None:
        backend = _make_backend(tools="")

        cmd = backend._build_command()

        idx = cmd.index("--tools")
        assert cmd[idx + 1] == ""

    def test_strict_mcp_config_can_be_disabled(self) -> None:
        backend = _make_backend(strict_mcp_config=False)

        cmd = backend._build_command()

        assert "--strict-mcp-config" not in cmd


class TestClaudeCodeBackendPromptRendering:
    def test_render_includes_stateless_backend_instructions(self) -> None:
        backend = _make_backend()
        messages = [
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(
                role="assistant",
                content="Earlier answer.",
                tool_calls=(ToolCall(id="tc_1", name="lookup", arguments={"q": "x"}),),
            ),
            ChatMessage(
                role="tool",
                content="tool output",
                tool_call_id="tc_1",
                name="lookup",
            ),
            ChatMessage(role="user", content="What next?"),
        ]
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured.update(kwargs)
            captured["cwd_exists_during_run"] = Path(kwargs["cwd"]).exists()
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(_success_payload("final reply")), stderr=""
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(messages, SamplingParams(max_tokens=17))

        prompt = captured["input"]
        assert isinstance(prompt, str)
        assert "stateless chat backend" in prompt
        assert "Do not inspect files" in prompt
        assert '"role": "system"' in prompt
        assert '"role": "tool"' in prompt
        assert '"tool_calls": [' in prompt
        assert '"tool_call_id": "tc_1"' in prompt
        assert captured["text"] is True
        assert captured["capture_output"] is True
        assert captured["timeout"] == backend._timeout
        assert captured["cwd"] != os.getcwd()
        assert captured["cwd_exists_during_run"] is True
        assert result.text == "final reply"


class TestClaudeCodeBackendGeneration:
    def test_generate_chat_parses_result_field(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(_success_payload("hello world")),
                stderr="",
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(
                [ChatMessage(role="user", content="Hi")], SamplingParams()
            )

        assert result.text == "hello world"
        assert result.finish_reason == StopReason.END_OF_TEXT
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 7
        assert result.metadata["session_id"] == "session-abc"
        assert result.metadata["total_cost_usd"] == 0.001
        assert result.metadata["is_error"] is False
        assert result.metadata["raw_usage"] == {"input_tokens": 11, "output_tokens": 7}

    def test_generate_chat_parses_content_string_fallback(self) -> None:
        backend = _make_backend()
        payload = {
            "is_error": False,
            "content": "fallback text",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(
                [ChatMessage(role="user", content="Hi")], SamplingParams()
            )

        assert result.text == "fallback text"

    def test_generate_chat_parses_content_block_list_fallback(self) -> None:
        backend = _make_backend()
        payload = {
            "is_error": False,
            "content": [{"type": "text", "text": "blocked text"}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(
                [ChatMessage(role="user", content="Hi")], SamplingParams()
            )

        assert result.text == "blocked text"

    def test_generate_chat_propagates_stop_reason(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(_success_payload(stop_reason="max_tokens")),
                stderr="",
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(
                [ChatMessage(role="user", content="Hi")], SamplingParams()
            )

        assert result.finish_reason == StopReason.MAX_TOKENS
        assert result.metadata["stop_reason"] == "max_tokens"

    def test_generate_wraps_prompts_as_user_messages(self) -> None:
        backend = _make_backend(max_concurrency=1)
        calls: list[list[ChatMessage]] = []

        def fake_impl(messages: list[ChatMessage]) -> GenerationResult:
            calls.append(messages)
            return GenerationResult(
                text=messages[0].content or "",
                finish_reason=StopReason.UNKNOWN,
            )

        with patch.object(backend, "_generate_chat_impl", side_effect=fake_impl):
            results = backend.generate(["first", "second"], SamplingParams())

        assert [r.text for r in results] == ["first", "second"]
        assert [m[0].role for m in calls] == ["user", "user"]
        assert [m[0].content for m in calls] == ["first", "second"]


class TestClaudeCodeBackendValidation:
    @pytest.mark.parametrize(
        ("params", "needle"),
        [
            (SamplingParams(temperature=0.1), "SamplingParams.temperature"),
            (SamplingParams(top_p=0.9), "SamplingParams.top_p"),
            (SamplingParams(top_k=40), "SamplingParams.top_k"),
            (SamplingParams(presence_penalty=0.1), "SamplingParams.presence_penalty"),
            (SamplingParams(frequency_penalty=0.1), "SamplingParams.frequency_penalty"),
            (SamplingParams(n=2), "SamplingParams.n"),
            (SamplingParams(stop_sequences=("X",)), "SamplingParams.stop_sequences"),
            (SamplingParams(logprobs=True), "SamplingParams.logprobs"),
            (SamplingParams(thinking_budget=100), "SamplingParams.thinking_budget"),
            (SamplingParams(extra={"foo": "bar"}), "SamplingParams.extra"),
        ],
    )
    def test_unsupported_sampling_params_raise(self, params: SamplingParams, needle: str) -> None:
        backend = _make_backend()

        with pytest.raises(ValueError, match=needle):
            backend.generate_chat([ChatMessage(role="user", content="Hello")], params)

    def test_max_tokens_is_accepted_but_ignored(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(_success_payload()), stderr=""
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(
                [ChatMessage(role="user", content="Hi")],
                SamplingParams(max_tokens=4096),
            )

        assert isinstance(result, GenerationResult)

    def test_image_messages_are_rejected_before_subprocess(self) -> None:
        backend = _make_backend()
        message = ChatMessage(
            role="user",
            content="Look at this",
            images=(ImageContent(data="abcd"),),
        )

        with patch("llenvs.inference.backends.claude_code.subprocess.run") as mock_run:
            with pytest.raises(ValueError, match="text-only"):
                backend.generate_chat([message], SamplingParams())

        mock_run.assert_not_called()


class TestClaudeCodeBackendErrorClassification:
    def test_prompt_too_long_via_is_error_payload(self) -> None:
        backend = _make_backend()
        payload = _error_payload("Prompt is too long: 250000 tokens")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps(payload), stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(PromptTooLongError, match="Prompt is too long"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_prompt_too_long_via_nonzero_exit(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="API Error: maximum context length exceeded",
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(PromptTooLongError, match="maximum context length"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_quota_via_youve_hit(self) -> None:
        backend = _make_backend()
        payload = _error_payload("You've hit your weekly limit · resets Mon 2:00am")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps(payload), stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError, match="weekly limit"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_quota_via_429_in_stderr(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="API Error: Request rejected (429) · capacity issue",
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError, match="exit code 1"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_quota_via_repeated_529(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="API Error: Repeated 529 Overloaded errors",
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_auth_error_is_runtime_error_not_quota(self) -> None:
        backend = _make_backend()
        payload = _error_payload("Not logged in · Please run /login")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps(payload), stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="authentication failed"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_auth_error_invalid_api_key_is_runtime_error(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="Invalid API key · please check"
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="authentication failed"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_timeout_is_quota_exhausted(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=backend._timeout)

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError, match="timed out"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_generic_nonzero_exit_is_quota_exhausted(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="something unexpected went wrong"
            )

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError, match="exit code 1"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_non_json_stdout_is_quota_exhausted(self) -> None:
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="not json at all", stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError, match="non-JSON output"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_missing_text_field_is_quota_exhausted(self) -> None:
        backend = _make_backend()
        payload = {"is_error": False, "session_id": "abc", "usage": {}}

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        with patch("llenvs.inference.backends.claude_code.subprocess.run", side_effect=fake_run):
            with pytest.raises(QuotaExhaustedError, match="no recognized text field"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())


class TestClaudeCodeBackendBatching:
    def test_batch_partial_failures_raise_partial_batch_error(self) -> None:
        backend = _make_backend(max_concurrency=2)

        def fake_impl(messages: list[ChatMessage]) -> GenerationResult:
            content = messages[0].content or ""
            if content == "bad":
                raise PromptTooLongError(
                    "maximum context length",
                    model_name=backend.model_name,
                    offending_indices=[0],
                )
            return GenerationResult(text=content.upper(), finish_reason=StopReason.UNKNOWN)

        messages_batch = [
            [ChatMessage(role="user", content="ok")],
            [ChatMessage(role="user", content="bad")],
        ]

        with patch.object(backend, "_generate_chat_impl", side_effect=fake_impl):
            with pytest.raises(PartialBatchError) as exc_info:
                backend.generate_chat_batch(messages_batch, SamplingParams())

        error = exc_info.value
        assert isinstance(error.results[0], GenerationResult)
        assert error.results[0].text == "OK"
        assert isinstance(error.results[1], PromptTooLongError)
