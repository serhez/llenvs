"""Text-based tool call parsing for non-function-calling backends.

Provides a protocol for parsing tool calls from plain text output
(e.g., vLLM/HF backends) and the Hermes format implementation used
by many open-source tool-calling models.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from llenvs.core.tools import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedToolResponse:
    """Result of parsing tool calls from model text output.

    Attributes:
        text: Remaining text with tool call blocks removed. None if
            the entire output was tool calls with no surrounding text.
        tool_calls: Parsed tool calls.
    """

    text: str | None
    tool_calls: tuple[ToolCall, ...]


class ToolCallParser(Protocol):
    """Protocol for text-based tool call parsing.

    Implementations format tool definitions as text for inclusion in
    prompts and parse tool calls from model text output.
    """

    def format_tools(self, tools: tuple[ToolDefinition, ...]) -> str:
        """Format tool definitions as text for inclusion in the prompt."""
        ...

    def parse(
        self, text: str, available_tools: tuple[ToolDefinition, ...]
    ) -> ParsedToolResponse:
        """Parse tool calls from model text output."""
        ...


_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


class HermesToolCallParser:
    """Hermes-style tool call parser.

    Parses the ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
    format used by NousResearch Hermes, Qwen, and many open-source
    tool-calling fine-tunes.

    ``format_tools()`` renders definitions as JSON inside
    ``<tools>...</tools>`` XML with instructions for the model.
    """

    def format_tools(self, tools: tuple[ToolDefinition, ...]) -> str:
        """Render tool definitions as Hermes-style XML."""
        tool_specs = []
        for tool in tools:
            spec = tool.to_openai_schema()
            tool_specs.append(spec)

        tools_json = json.dumps(tool_specs, indent=2)

        return (
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, respond with a <tool_call> block containing "
            "a JSON object with \"name\" and \"arguments\" keys:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
            "</tool_call>"
        )

    def parse(
        self, text: str, available_tools: tuple[ToolDefinition, ...]
    ) -> ParsedToolResponse:
        """Parse Hermes-style tool calls from text."""
        tool_calls: list[ToolCall] = []
        remaining = _TOOL_CALL_PATTERN.sub("", text).strip()

        for match in _TOOL_CALL_PATTERN.finditer(text):
            raw = match.group(1)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in tool_call block: %s", raw)
                continue

            name = data.get("name", "")
            arguments = data.get("arguments", {})

            if not isinstance(arguments, dict):
                logger.warning(
                    "tool_call arguments is not a dict: %s", type(arguments)
                )
                arguments = {}

            call_id = f"tc_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                ToolCall(id=call_id, name=name, arguments=arguments)
            )

        return ParsedToolResponse(
            text=remaining or None,
            tool_calls=tuple(tool_calls),
        )
