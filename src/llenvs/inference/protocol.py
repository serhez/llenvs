"""Model backend protocol and related types.

Defines the interface for inference backends (vLLM, OpenAI, Anthropic, etc.)
and common data structures for generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenvs.core.state import Action
    from llenvs.core.tools import ToolCall, ToolDefinition, ToolResult


class StopReason(Enum):
    """Reason why generation stopped."""

    MAX_TOKENS = auto()  # Hit token limit
    STOP_SEQUENCE = auto()  # Hit a stop sequence
    END_OF_TEXT = auto()  # Natural end of text
    TOOL_USE = auto()  # Stopped to make tool calls
    ERROR = auto()  # Generation error
    UNKNOWN = auto()  # Unknown reason


@dataclass(frozen=True)
class TokenLogprob:
    """Log probability information for a single token.

    Attributes:
        token: The token string.
        token_id: Token ID in vocabulary.
        logprob: Log probability of this token.
        top_logprobs: Optional top-k alternatives with their logprobs.
    """

    token: str
    token_id: int
    logprob: float
    top_logprobs: dict[str, float] | None = None


@dataclass(frozen=True)
class SamplingParams:
    """Parameters controlling text generation.

    Common parameters are defined as typed fields for discoverability.
    Backend-specific parameters can be passed via `extra` and will be
    forwarded to the underlying backend (e.g., HuggingFace transformers,
    vLLM, OpenAI API).

    Attributes:
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling (0 = disabled).
        stop_sequences: Sequences that stop generation.
        presence_penalty: Penalty for token presence.
        frequency_penalty: Penalty for token frequency.
        n: Number of completions to generate.
        logprobs: Whether to return log probabilities.
        num_logprobs: Number of top logprobs to return per token.
        extra: Backend-specific parameters passed through to the underlying
            inference library. For HuggingFace: repetition_penalty, do_sample,
            num_beams, etc. For vLLM: best_of, use_beam_search, etc.
    """

    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: tuple[str, ...] = ()
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    logprobs: bool = False
    num_logprobs: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Result of a generation request.

    Attributes:
        text: Generated text (may be None if only tool calls).
        finish_reason: Why generation stopped.
        tool_calls: Tuple of tool calls (if any).
        token_logprobs: Per-token log probabilities (if requested).
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of generated tokens.
        metadata: Backend-specific metadata.
    """

    text: str | None = None
    finish_reason: StopReason = StopReason.UNKNOWN
    tool_calls: tuple["ToolCall", ...] = ()
    token_logprobs: tuple[TokenLogprob, ...] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Check if this result contains tool calls."""
        return len(self.tool_calls) > 0

    def to_agent_action(self) -> "Action":
        """Convert to Action for use with environments."""
        from llenvs.core.state import Action

        return Action(text=self.text, tool_calls=self.tool_calls)


@dataclass(frozen=True)
class ChatMessage:
    """A message in a chat conversation.

    Supports both text messages and tool-related messages.

    Attributes:
        role: Message role (system, user, assistant, tool).
        content: Message content (may be None for tool calls).
        tool_calls: Tool calls made by the assistant.
        tool_call_id: ID of the tool call this message responds to (for role=tool).
        name: Tool name (for role=tool).
    """

    role: str
    content: str | None = None
    tool_calls: tuple["ToolCall", ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API calls."""
        result: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id

        if self.name is not None:
            result["name"] = self.name

        return result

    @classmethod
    def tool_result(cls, result: "ToolResult") -> "ChatMessage":
        """Create a tool result message from a ToolResult."""
        content = str(result.output) if result.is_success else (result.error or "Unknown error")
        return cls(
            role="tool",
            content=content,
            tool_call_id=result.call_id,
            name=result.tool_name,
        )


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities supported by a backend.

    Attributes:
        supports_logprobs: Can return token log probabilities.
        supports_prefix_continuation: Can continue from a partial response.
        supports_batching: Can process multiple requests efficiently.
        supports_streaming: Can stream tokens as generated.
        supports_chat: Supports chat message format.
        supports_function_calling: Supports function/tool calling.
        max_batch_size: Maximum batch size (None = unlimited).
        max_context_length: Maximum context window.
    """

    supports_logprobs: bool = False
    supports_prefix_continuation: bool = False
    supports_batching: bool = False
    supports_streaming: bool = False
    supports_chat: bool = True
    supports_function_calling: bool = False
    max_batch_size: int | None = None
    max_context_length: int | None = None
    max_concurrency: int | None = None


class ModelBackend(ABC):
    """Abstract base class for model inference backends.

    Backends provide a uniform interface for different inference methods
    (local vLLM, API services, etc.).
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Get the capabilities of this backend."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate completions for text prompts.

        Args:
            prompts: List of prompt strings.
            params: Sampling parameters.

        Returns:
            List of GenerationResults, one per prompt.
        """
        ...

    def generate_single(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate completion for a single prompt.

        Args:
            prompt: Prompt string.
            params: Sampling parameters.

        Returns:
            GenerationResult for the prompt.
        """
        results = self.generate([prompt], params)
        return results[0]

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation.

        Args:
            messages: List of chat messages.
            params: Sampling parameters.

        Returns:
            GenerationResult for the conversation.

        Raises:
            NotImplementedError: If backend doesn't support chat.
        """
        raise NotImplementedError("This backend does not support chat generation")

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate responses for multiple chat conversations.

        Enables efficient batched inference across independent conversations.
        The default implementation calls generate_chat() sequentially.
        Backends should override for better performance (e.g., GPU batching
        for vLLM/HF, concurrent HTTP requests for API backends).

        Args:
            messages_batch: List of conversations, each a list of ChatMessages.
            params: Sampling parameters (shared across all conversations).

        Returns:
            List of GenerationResults, one per conversation, in the same order.
        """
        return [self.generate_chat(msgs, params) for msgs in messages_batch]

    def generate_with_tools_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        tools: list["ToolDefinition"],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> list[GenerationResult]:
        """Generate tool-calling responses for multiple conversations.

        The default implementation calls generate_with_tools() sequentially.
        Backends should override for better performance.

        Args:
            messages_batch: List of conversations.
            tools: Available tool definitions (shared across all conversations).
            params: Sampling parameters.
            tool_choice: Tool selection mode.

        Returns:
            List of GenerationResults, one per conversation, in the same order.
        """
        return [
            self.generate_with_tools(msgs, tools, params, tool_choice)
            for msgs in messages_batch
        ]

    def generate_with_logprobs(
        self,
        prompts: list[str],
        params: SamplingParams,
        num_logprobs: int = 5,
    ) -> list[GenerationResult]:
        """Generate completions with log probabilities.

        Args:
            prompts: List of prompt strings.
            params: Sampling parameters.
            num_logprobs: Number of top logprobs to return per token.

        Returns:
            List of GenerationResults with logprobs populated.

        Raises:
            NotImplementedError: If backend doesn't support logprobs.
        """
        if not self.capabilities.supports_logprobs:
            raise NotImplementedError("This backend does not support logprobs")

        # Create params with logprobs enabled
        params_with_logprobs = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop_sequences=params.stop_sequences,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            n=params.n,
            logprobs=True,
            num_logprobs=num_logprobs,
        )
        return self.generate(prompts, params_with_logprobs)

    def continue_from_prefix(
        self,
        prefix: str,
        params: SamplingParams,
        num_continuations: int = 1,
    ) -> list[GenerationResult]:
        """Continue generation from a partial response.

        Useful for branching and exploring multiple continuations.

        Args:
            prefix: The partial response to continue from.
            params: Sampling parameters.
            num_continuations: Number of different continuations to generate.

        Returns:
            List of GenerationResults, one per continuation.

        Raises:
            NotImplementedError: If backend doesn't support prefix continuation.
        """
        if not self.capabilities.supports_prefix_continuation:
            raise NotImplementedError(
                "This backend does not support prefix continuation"
            )
        raise NotImplementedError("Subclass must implement continue_from_prefix")

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list["ToolDefinition"],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        """Generate a response with tool/function calling capability.

        Args:
            messages: List of chat messages.
            tools: List of available tool definitions.
            params: Sampling parameters.
            tool_choice: How to handle tool selection:
                - "auto": Model decides whether to use tools
                - "none": Don't use tools
                - "required": Must use at least one tool
                - A tool name: Use that specific tool

        Returns:
            GenerationResult with optional tool_calls populated.

        Raises:
            NotImplementedError: If backend doesn't support function calling.
        """
        if not self.capabilities.supports_function_calling:
            raise NotImplementedError(
                "This backend does not support function calling"
            )
        raise NotImplementedError("Subclass must implement generate_with_tools")
