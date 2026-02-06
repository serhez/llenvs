"""Composable prompt transformers.

Provides a pipeline of transformations to build prompts from
base messages. Transformers can be composed with >> operator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from llenvs.inference.protocol import ChatMessage

if TYPE_CHECKING:
    from llenvs.inference.prompts import PromptTemplate


@runtime_checkable
class PromptTransformer(Protocol):
    """Protocol for prompt transformers.

    Transformers take a list of messages and return a modified list.
    They can be composed using the >> operator.
    """

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Transform the message list.

        Args:
            messages: Input messages.

        Returns:
            Transformed messages.
        """
        ...

    def __rshift__(self, other: "PromptTransformer") -> "PromptPipeline":
        """Compose transformers with >> operator."""
        ...


@dataclass
class PromptPipeline:
    """A pipeline of prompt transformers.

    Applies transformers in order. Can be extended with >> operator.
    """

    transformers: list[PromptTransformer] = field(default_factory=list)

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Apply all transformers in sequence."""
        result = messages
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def __rshift__(self, other: PromptTransformer) -> "PromptPipeline":
        """Add a transformer to the pipeline."""
        return PromptPipeline(transformers=self.transformers + [other])

    def __call__(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Allow calling the pipeline directly."""
        return self.transform(messages)


class BaseTransformer(ABC):
    """Base class for transformers with >> operator support."""

    @abstractmethod
    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Transform the message list."""
        ...

    def __rshift__(self, other: PromptTransformer) -> PromptPipeline:
        """Compose with another transformer."""
        return PromptPipeline(transformers=[self, other])

    def __call__(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Allow calling the transformer directly."""
        return self.transform(messages)


@dataclass
class SystemPromptInjector(BaseTransformer):
    """Injects or replaces the system prompt.

    Attributes:
        content: System prompt content.
        replace_existing: If True, replaces existing system message.
            If False, prepends to existing.
    """

    content: str
    replace_existing: bool = True

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Inject system prompt at the beginning."""
        result = []
        has_system = False

        for msg in messages:
            if msg.role == "system":
                has_system = True
                if self.replace_existing:
                    result.append(ChatMessage(role="system", content=self.content))
                else:
                    # Prepend to existing
                    combined = f"{self.content}\n\n{msg.content}"
                    result.append(ChatMessage(role="system", content=combined))
            else:
                result.append(msg)

        if not has_system:
            result.insert(0, ChatMessage(role="system", content=self.content))

        return result


@dataclass
class FewShotInjector(BaseTransformer):
    """Injects few-shot examples into the conversation.

    Examples are inserted after the system message and before user messages.

    Attributes:
        examples: List of (user, assistant) message pairs.
    """

    examples: list[tuple[str, str]]

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Inject few-shot examples."""
        if not self.examples:
            return messages

        result = []
        examples_inserted = False

        for msg in messages:
            # Insert examples after system message
            if msg.role == "system":
                result.append(msg)
                for user_content, assistant_content in self.examples:
                    result.append(ChatMessage(role="user", content=user_content))
                    result.append(ChatMessage(role="assistant", content=assistant_content))
                examples_inserted = True
            else:
                # If no system message, insert before first user message
                if not examples_inserted and msg.role == "user":
                    for user_content, assistant_content in self.examples:
                        result.append(ChatMessage(role="user", content=user_content))
                        result.append(ChatMessage(role="assistant", content=assistant_content))
                    examples_inserted = True
                result.append(msg)

        return result


@dataclass
class ChainOfThoughtWrapper(BaseTransformer):
    """Wraps messages to encourage chain-of-thought reasoning.

    Attributes:
        style: Style of CoT prompt ("think_step_by_step", "show_work", "explain").
    """

    style: str = "think_step_by_step"

    _PROMPTS = {
        "think_step_by_step": "Think through this step by step before giving your final answer.",
        "show_work": "Show your work and reasoning before providing the answer.",
        "explain": "Explain your thought process as you work through this problem.",
        "reflect": "Consider the problem carefully, reflect on your approach, then provide your answer.",
    }

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Add CoT instruction to the last user message."""
        if not messages:
            return messages

        cot_prompt = self._PROMPTS.get(self.style, self.style)
        result = list(messages)

        # Find and modify last user message
        for i in range(len(result) - 1, -1, -1):
            if result[i].role == "user":
                new_content = f"{result[i].content}\n\n{cot_prompt}"
                result[i] = ChatMessage(role="user", content=new_content)
                break

        return result


@dataclass
class AnswerFormatInjector(BaseTransformer):
    """Injects answer formatting instructions.

    Attributes:
        format_type: Type of format ("xml_answer", "json", "markdown").
        tag_name: Tag name for XML format (default: "answer").
    """

    format_type: str = "xml_answer"
    tag_name: str = "answer"

    _FORMATS = {
        "xml_answer": "Put your final answer in <{tag_name}>...</{tag_name}> tags.",
        "json": "Provide your answer as a JSON object with an 'answer' field.",
        "markdown": "Format your final answer as a markdown code block.",
        "boxed": "Put your final answer in \\boxed{{}}.",
        "gsm8k": "End your response with '#### ' followed by your numerical answer.",
    }

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Add answer format instruction."""
        format_template = self._FORMATS.get(self.format_type, self.format_type)
        format_instruction = format_template.format(tag_name=self.tag_name)

        result = list(messages)

        # Add to system message if present, otherwise to last user message
        for i, msg in enumerate(result):
            if msg.role == "system":
                new_content = f"{msg.content}\n\n{format_instruction}"
                result[i] = ChatMessage(role="system", content=new_content)
                return result

        # No system message, add to last user message
        for i in range(len(result) - 1, -1, -1):
            if result[i].role == "user":
                new_content = f"{result[i].content}\n\n{format_instruction}"
                result[i] = ChatMessage(role="user", content=new_content)
                break

        return result


@dataclass
class MessageTrimmer(BaseTransformer):
    """Trims conversation to fit within token limits.

    Keeps the system message and most recent messages.

    Attributes:
        max_messages: Maximum number of messages to keep.
        keep_system: Whether to always keep the system message.
    """

    max_messages: int = 10
    keep_system: bool = True

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Trim messages to fit limit."""
        if len(messages) <= self.max_messages:
            return messages

        result = []
        system_msg = None

        # Extract system message
        non_system = []
        for msg in messages:
            if msg.role == "system" and self.keep_system:
                system_msg = msg
            else:
                non_system.append(msg)

        # Keep most recent messages
        reserved = 1 if system_msg else 0
        max_non_system = self.max_messages - reserved
        trimmed = non_system[-max_non_system:] if max_non_system > 0 else []

        if system_msg:
            result.append(system_msg)
        result.extend(trimmed)

        return result


@dataclass
class RoleMapper(BaseTransformer):
    """Maps message roles for different APIs.

    Some APIs expect different role names (e.g., "developer" instead of "system").

    Attributes:
        mapping: Dictionary mapping old roles to new roles.
    """

    mapping: dict[str, str] = field(default_factory=dict)

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Apply role mapping."""
        if not self.mapping:
            return messages

        result = []
        for msg in messages:
            new_role = self.mapping.get(msg.role, msg.role)
            result.append(ChatMessage(role=new_role, content=msg.content))

        return result


@dataclass
class ContentWrapper(BaseTransformer):
    """Wraps message content with prefix/suffix.

    Useful for adding delimiters or formatting.

    Attributes:
        prefix: Text to prepend.
        suffix: Text to append.
        roles: Roles to apply to (empty = all).
    """

    prefix: str = ""
    suffix: str = ""
    roles: tuple[str, ...] = ()

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Wrap content for specified roles."""
        result = []
        for msg in messages:
            if not self.roles or msg.role in self.roles:
                new_content = f"{self.prefix}{msg.content}{self.suffix}"
                result.append(ChatMessage(role=msg.role, content=new_content))
            else:
                result.append(msg)
        return result


@dataclass
class PromptTemplateTransformer(BaseTransformer):
    """Applies a PromptTemplate to the last user message.

    Wraps the last user message content using the template's {question}
    placeholder. Used by the runner to apply adapter/config templates.

    Attributes:
        template: The PromptTemplate to apply.
    """

    template: "PromptTemplate"

    def transform(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Apply template to the last user message."""
        if not messages:
            return messages

        result = list(messages)

        for i in range(len(result) - 1, -1, -1):
            if result[i].role == "user":
                new_content = self.template.apply(result[i].content)
                result[i] = ChatMessage(role="user", content=new_content)
                break

        return result


def build_standard_pipeline(
    system_prompt: str | None = None,
    examples: list[tuple[str, str]] | None = None,
    use_cot: bool = False,
    answer_format: str = "xml_answer",
    tag_name: str = "answer",
) -> PromptPipeline:
    """Build a standard prompting pipeline.

    Args:
        system_prompt: Optional system prompt.
        examples: Optional few-shot examples.
        use_cot: Whether to use chain-of-thought.
        answer_format: Answer format type.
        tag_name: Tag name for XML format.

    Returns:
        Configured PromptPipeline.
    """
    transformers: list[PromptTransformer] = []

    if system_prompt:
        transformers.append(SystemPromptInjector(content=system_prompt))

    if examples:
        transformers.append(FewShotInjector(examples=examples))

    if use_cot:
        transformers.append(ChainOfThoughtWrapper())

    transformers.append(AnswerFormatInjector(format_type=answer_format, tag_name=tag_name))

    return PromptPipeline(transformers=transformers)
