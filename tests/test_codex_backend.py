"""Tests for the Codex CLI backend."""

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
    SamplingParams,
    StopReason,
)
from llenvs.inference.backends import CodexCLIBackend


def _make_backend(**kwargs: object) -> CodexCLIBackend:
    with patch("llenvs.inference.backends.codex_cli.shutil.which", return_value="/usr/bin/codex"):
        return CodexCLIBackend(**kwargs)


class TestCodexCLIBackendInit:
    def test_exported_from_backends_package(self):
        assert CodexCLIBackend is not None

    def test_missing_binary_raises_import_error(self):
        with patch("llenvs.inference.backends.codex_cli.shutil.which", return_value=None):
            with pytest.raises(ImportError, match="codex CLI is required"):
                CodexCLIBackend()

    def test_capabilities_and_model_name(self):
        backend = _make_backend(model="codex-test", max_concurrency=4)

        caps = backend.capabilities

        assert backend.model_name == "codex-test"
        assert caps.supports_chat is True
        assert caps.supports_batching is True
        assert caps.supports_function_calling is False
        assert caps.supports_logprobs is False
        assert caps.supports_prefix_continuation is False
        assert caps.supports_streaming is False
        assert caps.supports_vision is False
        assert caps.supports_full_scoring is False
        assert caps.max_concurrency == 4

    def test_invalid_config_override_value_raises(self):
        with pytest.raises(ValueError, match="Codex config override values"):
            _make_backend(config_overrides={"bad": {"nested": True}})


class TestCodexCLIBackendGeneration:
    def test_generate_chat_builds_expected_command_and_prompt(self):
        backend = _make_backend(
            model="codex-model",
            profile="research",
            config_overrides={
                "z_flag": True,
                "a_list": ["x", 2],
            },
        )
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
            output_path = Path(cmd[cmd.index("--output-last-message") + 1])
            output_path.write_text("final reply", encoding="utf-8")
            stdout = "\n".join(
                [
                    json.dumps({"type": "turn.started"}),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {"input_tokens": 11, "output_tokens": 7},
                        }
                    ),
                ]
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        with patch("llenvs.inference.backends.codex_cli.subprocess.run", side_effect=fake_run):
            result = backend.generate_chat(messages, SamplingParams(max_tokens=17))

        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert cmd[:2] == ["/usr/bin/codex", "exec"]
        assert "--json" in cmd
        assert "--color" in cmd
        assert "--ephemeral" in cmd
        assert "--skip-git-repo-check" in cmd
        assert "--sandbox" in cmd
        assert "read-only" in cmd
        assert "--profile" in cmd
        assert "research" in cmd
        assert "--model" in cmd
        assert "codex-model" in cmd
        assert cmd.count("-c") == 2
        assert cmd[cmd.index("-c") + 1] == 'a_list=["x", 2]'
        assert cmd[cmd.index("-c", cmd.index("-c") + 1) + 1] == "z_flag=true"

        assert captured["text"] is True
        assert captured["capture_output"] is True
        assert captured["timeout"] == backend._timeout
        assert captured["cwd"] != os.getcwd()
        assert captured["cwd_exists_during_run"] is True

        prompt = captured["input"]
        assert isinstance(prompt, str)
        assert "stateless chat backend" in prompt
        assert "Do not inspect files" in prompt
        assert '"role": "system"' in prompt
        assert '"role": "tool"' in prompt
        assert '"tool_calls": [' in prompt
        assert '"tool_call_id": "tc_1"' in prompt

        assert result.text == "final reply"
        assert result.finish_reason == StopReason.UNKNOWN
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 7
        assert result.metadata["raw_usage"] == {"input_tokens": 11, "output_tokens": 7}
        assert result.metadata["event_count"] == 2
        assert result.metadata["config_overrides"] == {"z_flag": True, "a_list": ["x", 2]}

    def test_generate_wraps_prompts_as_user_messages(self):
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

        assert [result.text for result in results] == ["first", "second"]
        assert [messages[0].role for messages in calls] == ["user", "user"]
        assert [messages[0].content for messages in calls] == ["first", "second"]

    @pytest.mark.parametrize(
        ("params", "needle"),
        [
            (SamplingParams(temperature=0.1), "SamplingParams.temperature"),
            (SamplingParams(extra={"reasoning_effort": "high"}), "SamplingParams.extra"),
        ],
    )
    def test_unsupported_sampling_params_raise(self, params: SamplingParams, needle: str):
        backend = _make_backend()

        with pytest.raises(ValueError, match=needle):
            backend.generate_chat([ChatMessage(role="user", content="Hello")], params)

    def test_image_messages_are_rejected_before_subprocess(self):
        backend = _make_backend()
        message = ChatMessage(
            role="user",
            content="Look at this",
            images=(ImageContent(data="abcd"),),
        )

        with patch("llenvs.inference.backends.codex_cli.subprocess.run") as mock_run:
            with pytest.raises(ValueError, match="text-only"):
                backend.generate_chat([message], SamplingParams())

        mock_run.assert_not_called()

    def test_prompt_too_long_errors_are_normalized(self):
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="maximum context length exceeded",
            )

        with patch("llenvs.inference.backends.codex_cli.subprocess.run", side_effect=fake_run):
            with pytest.raises(PromptTooLongError, match="maximum context length exceeded"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_missing_output_file_raises_runtime_error(self):
        backend = _make_backend()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"type": "turn.completed"}),
                stderr="",
            )

        with patch("llenvs.inference.backends.codex_cli.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="last-message file"):
                backend.generate_chat([ChatMessage(role="user", content="Hello")], SamplingParams())

    def test_batch_partial_failures_raise_partial_batch_error(self):
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
