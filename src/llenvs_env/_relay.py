"""Pure llenvs <-> chat translation helpers for the ``llenvs-env`` relay.

Feedback selection mirrors the miles connector's agent function
(``llenvs.integrations.miles.agent``): tool results first, then the dynamic
state text, then the last user message, then the prompt. Tool results are
rendered as Hermes ``<tool_response>`` blocks inside ONE user message: the
relay path has no native tool-call assistant node, and a dangling
``role: tool`` message breaks chat templates.

No verifiers imports: this module is tested in the base venv.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from llenvs.core.state import Action, State
from llenvs.core.tool_parsing import HermesToolCallParser, ToolCallParser
from llenvs.core.tools import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


def default_parser() -> HermesToolCallParser:
    """The text-based tool-call parser used when none is injected."""
    return HermesToolCallParser()


def refuse_images(state: State[Any]) -> None:
    """Reject observations carrying images: the relay is text-only."""
    if state.observation.get_images():
        raise NotImplementedError(
            "Observation contains images; the llenvs-env relay is text-only, so vision "
            "environments are not supported."
        )


def _tool_result_content(result: ToolResult) -> str | dict[str, Any]:
    output = result.output
    if isinstance(output, dict):
        return output
    content = str(output)
    if not content:
        content = result.error or "(no output)"
    return content


def tool_response_text(results: Sequence[ToolResult]) -> str:
    """Render tool results as Hermes ``<tool_response>`` blocks, in call order."""
    blocks = []
    for result in results:
        payload = json.dumps(
            {"name": result.tool_name, "content": _tool_result_content(result)}, default=str
        )
        blocks.append(f"<tool_response>\n{payload}\n</tool_response>")
    return "\n".join(blocks)


def feedback_text(state: State[Any]) -> str:
    """Text to send the policy after an env step."""
    obs = state.observation
    if obs.tool_results:
        return tool_response_text(obs.tool_results)
    if obs.state is not None and obs.state.text:
        return obs.state.text
    if obs.messages and obs.messages[-1].get("role") == "user":
        return obs.messages[-1].get("content", "")
    return obs.prompt


def action_from_reply(
    reply: str,
    tools: Sequence[ToolDefinition],
    parser: ToolCallParser | None = None,
) -> Action:
    """Convert the policy's reply text into an llenvs Action.

    Without advertised tools the reply is plain text, markup included. With
    tools, ``<tool_call>`` blocks are parsed into tool calls and the remaining
    prose becomes the action text.
    """
    if not tools:
        return Action.from_text(reply)
    parsed = (parser or default_parser()).parse(reply, tuple(tools))
    if not parsed.tool_calls:
        return Action.from_text(reply)
    return Action(text=parsed.text, tool_calls=tuple(parsed.tool_calls))


def history_messages(prompt: str, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build OpenAI-shaped chat messages: the user prompt plus ``obs.messages``.

    llenvs tool-call dicts (``id``/``name``/``arguments``) become OpenAI
    ``function`` entries with JSON-encoded arguments.
    """
    out: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content"),
                        "tool_calls": [_openai_tool_call(tc) for tc in tool_calls],
                    }
                )
            else:
                out.append({"role": "assistant", "content": msg.get("content", "")})
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "name": msg.get("name"),
                    "content": msg.get("content", ""),
                }
            )
        else:
            out.append({"role": "user", "content": msg.get("content", "")})
    return out


def _openai_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    arguments = tc.get("arguments", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": tc["id"],
        "type": "function",
        "function": {"name": tc["name"], "arguments": arguments},
    }


def close_environment(env: Any) -> None:
    """Best-effort cleanup: the Environment protocol has no close()."""
    for method_name in ("close", "shutdown"):
        method = getattr(env, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                logger.warning("Environment %s() failed during cleanup", method_name)
            return
