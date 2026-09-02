"""Tests for ``llenvs_env._relay`` — pure llenvs <-> chat translation helpers.

This module imports no verifiers symbols, so it runs in the base venv. The
feedback-selection rules mirror the miles connector's agent function.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.tool_parsing import HermesToolCallParser, ParsedToolResponse
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)
from llenvs_env import _relay

LOOKUP_TOOL = ToolDefinition(
    name="lookup",
    description="Look something up.",
    parameters=(ToolParameter(name="q", type=ToolParameterType.STRING, description="Query."),),
)

SUBMIT_TOOL = ToolDefinition(name="submit", description="Submit an answer.")


def _state(**obs_kwargs: Any) -> State[Any]:
    obs_kwargs.setdefault("prompt", "What is 2+2?")
    return State(
        observation=Observation(**obs_kwargs),
        hidden=None,
        metadata=StateMetadata(step=0, episode_id="ep"),
    )


def _tool_call_block(name: str, arguments: dict[str, Any]) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def _parse_blocks(text: str) -> list[dict[str, Any]]:
    """Split ``<tool_response>`` blocks and JSON-decode their payloads."""
    blocks = []
    remainder = text
    while "<tool_response>" in remainder:
        start = remainder.index("<tool_response>") + len("<tool_response>")
        end = remainder.index("</tool_response>")
        blocks.append(json.loads(remainder[start:end].strip()))
        remainder = remainder[end + len("</tool_response>") :]
    assert remainder.strip() == ""
    return blocks


# ---------------------------------------------------------------------------
# refuse_images
# ---------------------------------------------------------------------------


class TestRefuseImages:
    def test_text_only_state_passes(self):
        _relay.refuse_images(_state(state=ObservationContent(text="look")))

    def test_state_images_refused(self):
        img = ImageContent(data="aGk=", media_type="image/png")
        with pytest.raises(NotImplementedError, match="text-only"):
            _relay.refuse_images(_state(state=ObservationContent(text="look", images=(img,))))

    def test_task_images_refused(self):
        img = ImageContent(data="aGk=", media_type="image/png")
        with pytest.raises(NotImplementedError, match="images"):
            _relay.refuse_images(_state(task=ObservationContent(text="goal", images=(img,))))


# ---------------------------------------------------------------------------
# feedback_text
# ---------------------------------------------------------------------------


class TestFeedbackText:
    def test_state_text_preferred(self):
        state = _state(
            messages=({"role": "user", "content": "Ignored."},),
            state=ObservationContent(text="You see a door."),
        )
        assert _relay.feedback_text(state) == "You see a door."

    def test_empty_state_text_falls_through(self):
        state = _state(
            messages=({"role": "user", "content": "Next?"},),
            state=ObservationContent(text=""),
        )
        assert _relay.feedback_text(state) == "Next?"

    def test_last_user_message_when_no_state(self):
        state = _state(
            messages=(
                {"role": "assistant", "content": "Earlier."},
                {"role": "user", "content": "Next?"},
            )
        )
        assert _relay.feedback_text(state) == "Next?"

    def test_trailing_assistant_message_falls_back_to_prompt(self):
        state = _state(
            prompt="The prompt.",
            messages=({"role": "assistant", "content": "Earlier."},),
        )
        assert _relay.feedback_text(state) == "The prompt."

    def test_prompt_fallback(self):
        assert _relay.feedback_text(_state(prompt="The prompt.")) == "The prompt."

    def test_tool_results_win_over_state_text(self):
        result = ToolResult.success(call_id="c1", tool_name="lookup", output="found")
        state = _state(tool_results=(result,), state=ObservationContent(text="ignored"))
        text = _relay.feedback_text(state)
        assert "<tool_response>" in text
        assert "ignored" not in text
        assert text == _relay.tool_response_text((result,))


# ---------------------------------------------------------------------------
# tool_response_text
# ---------------------------------------------------------------------------


class TestToolResponseText:
    def test_single_string_output(self):
        result = ToolResult.success(call_id="c1", tool_name="lookup", output="found it")
        text = _relay.tool_response_text((result,))
        assert text.startswith("<tool_response>")
        assert text.endswith("</tool_response>")
        assert _parse_blocks(text) == [{"name": "lookup", "content": "found it"}]

    def test_dict_output_embedded_as_object(self):
        result = ToolResult.success(call_id="c1", tool_name="lookup", output={"value": 42})
        assert _parse_blocks(_relay.tool_response_text((result,))) == [
            {"name": "lookup", "content": {"value": 42}}
        ]

    def test_error_result_uses_error_message(self):
        result = ToolResult.from_error(call_id="c1", tool_name="lookup", error_message="boom")
        assert _parse_blocks(_relay.tool_response_text((result,))) == [
            {"name": "lookup", "content": "boom"}
        ]

    def test_empty_output_without_error_placeholder(self):
        result = ToolResult.success(call_id="c1", tool_name="lookup", output="")
        assert _parse_blocks(_relay.tool_response_text((result,))) == [
            {"name": "lookup", "content": "(no output)"}
        ]

    def test_multiple_results_keep_order(self):
        results = (
            ToolResult.success(call_id="c1", tool_name="lookup", output="first"),
            ToolResult.success(call_id="c2", tool_name="submit", output="second"),
        )
        blocks = _parse_blocks(_relay.tool_response_text(results))
        assert [b["name"] for b in blocks] == ["lookup", "submit"]
        assert [b["content"] for b in blocks] == ["first", "second"]


# ---------------------------------------------------------------------------
# action_from_reply
# ---------------------------------------------------------------------------


class TestActionFromReply:
    def test_no_tools_keeps_text_verbatim(self):
        # Without advertised tools the reply is plain text, markup included.
        reply = "Sure.\n" + _tool_call_block("lookup", {"q": "x"})
        action = _relay.action_from_reply(reply, ())
        assert action == Action.from_text(reply)

    def test_plain_text_with_tools_advertised(self):
        action = _relay.action_from_reply("The answer is 4.", (LOOKUP_TOOL,))
        assert action == Action.from_text("The answer is 4.")

    def test_tool_call_parsed(self):
        reply = "Let me look.\n" + _tool_call_block("lookup", {"q": "x"})
        action = _relay.action_from_reply(reply, (LOOKUP_TOOL,))
        assert action.text == "Let me look."
        assert len(action.tool_calls) == 1
        assert action.tool_calls[0].name == "lookup"
        assert action.tool_calls[0].arguments == {"q": "x"}
        assert action.tool_calls[0].id

    def test_multiple_tool_calls_keep_order(self):
        reply = _tool_call_block("lookup", {"q": "a"}) + "\n" + _tool_call_block("submit", {})
        action = _relay.action_from_reply(reply, (LOOKUP_TOOL, SUBMIT_TOOL))
        assert [tc.name for tc in action.tool_calls] == ["lookup", "submit"]
        ids = [tc.id for tc in action.tool_calls]
        assert len(set(ids)) == 2

    def test_parser_is_injectable(self):
        class FakeParser:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[ToolDefinition, ...]]] = []

            def format_tools(self, tools):  # pragma: no cover - protocol completeness
                return ""

            def parse(self, text, available_tools):
                self.calls.append((text, available_tools))
                return ParsedToolResponse(
                    text=None,
                    tool_calls=(ToolCall(id="fake", name="submit", arguments={}),),
                )

        parser = FakeParser()
        action = _relay.action_from_reply("anything", (SUBMIT_TOOL,), parser=parser)
        assert parser.calls == [("anything", (SUBMIT_TOOL,))]
        assert action.tool_calls == (ToolCall(id="fake", name="submit", arguments={}),)
        assert action.text is None

    def test_default_parser_is_hermes(self):
        assert isinstance(_relay.default_parser(), HermesToolCallParser)


# ---------------------------------------------------------------------------
# history_messages
# ---------------------------------------------------------------------------


class TestHistoryMessages:
    def test_prompt_only(self):
        assert _relay.history_messages("Q?", ()) == [{"role": "user", "content": "Q?"}]

    def test_user_and_assistant_replayed(self):
        history = (
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hi."},
        )
        assert _relay.history_messages("Q?", history) == [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hi."},
        ]

    def test_assistant_tool_calls_take_openai_shape(self):
        history = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
            },
        )
        messages = _relay.history_messages("Q?", history)
        assistant = messages[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] is None
        assert assistant["tool_calls"] == [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps({"q": "x"})},
            }
        ]

    def test_tool_message_shape(self):
        history = ({"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "found"},)
        assert _relay.history_messages("Q?", history)[1] == {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "lookup",
            "content": "found",
        }

    def test_unknown_role_treated_as_user(self):
        history = ({"role": "narrator", "content": "Meanwhile..."},)
        assert _relay.history_messages("Q?", history)[1] == {
            "role": "user",
            "content": "Meanwhile...",
        }

    def test_missing_role_defaults_to_user(self):
        assert _relay.history_messages("Q?", ({"content": "x"},))[1] == {
            "role": "user",
            "content": "x",
        }


# ---------------------------------------------------------------------------
# close_environment
# ---------------------------------------------------------------------------


class TestCloseEnvironment:
    def test_calls_close(self):
        calls: list[str] = []

        class Env:
            def close(self):
                calls.append("close")

        _relay.close_environment(Env())
        assert calls == ["close"]

    def test_falls_back_to_shutdown(self):
        calls: list[str] = []

        class Env:
            def shutdown(self):
                calls.append("shutdown")

        _relay.close_environment(Env())
        assert calls == ["shutdown"]

    def test_prefers_close_over_shutdown(self):
        calls: list[str] = []

        class Env:
            def close(self):
                calls.append("close")

            def shutdown(self):
                calls.append("shutdown")

        _relay.close_environment(Env())
        assert calls == ["close"]

    def test_no_method_is_a_noop(self):
        _relay.close_environment(object())

    def test_failure_is_swallowed_with_warning(self, caplog):
        class Env:
            def close(self):
                raise RuntimeError("boom")

        with caplog.at_level("WARNING"):
            _relay.close_environment(Env())
        assert "close" in caplog.text
