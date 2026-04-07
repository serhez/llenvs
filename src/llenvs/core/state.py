"""Immutable state abstractions for MDP-style environments.

The State class is the core data structure that flows through the environment.
It separates what the model sees (observation) from internal state needed for
reward computation (hidden).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from llenvs.core.tools import ToolCall, ToolDefinition, ToolResult

HiddenT = TypeVar("HiddenT")


@dataclass(frozen=True)
class ImageContent:
    """Base64-encoded image for multimodal observations.

    Attributes:
        data: Base64-encoded image data.
        media_type: MIME type (e.g., "image/png", "image/jpeg").
        source: Origin of this image — ``"state"`` for per-step visual
            observations, ``"task"`` for goal/reference images, or ``""``
            when unspecified.
    """

    data: str
    media_type: str = "image/png"
    source: str = ""


@dataclass(frozen=True)
class ObservationImages:
    """Images from an observation, separated by source.

    Keeps task images (goal / reference) distinct from state images
    (per-step visual observation) so callers can decide ordering and
    placement in prompts.

    Attributes:
        task: Static goal or reference images (from ``obs.task``).
        state: Per-step visual observations (from ``obs.state``).
    """

    task: tuple[ImageContent, ...] = ()
    state: tuple[ImageContent, ...] = ()

    @property
    def all(self) -> tuple[ImageContent, ...]:
        """All images, task first then state."""
        return self.task + self.state

    def __bool__(self) -> bool:
        return bool(self.task or self.state)


@dataclass(frozen=True)
class StateMetadata:
    """Metadata associated with a state.

    Attributes:
        step: Current step number in the episode (0-indexed).
        episode_id: Unique identifier for this episode.
        is_terminal: Whether this state is terminal (episode ended).
        info: Additional metadata (environment-specific).
    """

    step: int
    episode_id: str
    is_terminal: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class State[HiddenT]:
    """Immutable state representing a point in an episode.

    The separation of observation and hidden state enables:
    - Models to only see what they should (observation)
    - Reward functions to access ground truth (hidden)
    - Safe checkpointing and branching (immutability)

    Attributes:
        observation: What the model sees (prompt, messages, etc.).
        hidden: Internal state for reward computation (ground truth, etc.).
        metadata: Step count, episode ID, terminal flag, etc.
    """

    observation: Observation
    hidden: HiddenT
    metadata: StateMetadata

    def with_metadata(self, **kwargs: Any) -> State[HiddenT]:
        """Create a new state with updated metadata fields."""
        current = {
            "step": self.metadata.step,
            "episode_id": self.metadata.episode_id,
            "is_terminal": self.metadata.is_terminal,
            "info": self.metadata.info,
        }
        current.update(kwargs)
        return State(
            observation=self.observation,
            hidden=self.hidden,
            metadata=StateMetadata(**current),
        )


@dataclass(frozen=True)
class ObservationContent:
    """Structured content for task descriptions or state observations.

    Attributes:
        text: Text content (empty string for primarily-structured content).
        images: Optional image content.
        data: Optional structured data (tool results, game state, etc.).
    """

    text: str = ""
    images: tuple[ImageContent, ...] = ()
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class Observation:
    """Unified observation for all environments.

    Supports both text-only and tool-aware environments. Text-only
    environments leave tool_results and available_tools as empty tuples.

    Images live on the structured ``state`` and ``task`` fields
    (:class:`ObservationContent`).  ``state.images`` carries per-step
    visual observations (e.g. screenshots); ``task.images`` carries
    static goal or reference images.

    Attributes:
        prompt: The question or prompt text.
        messages: Chat message history (including tool calls/results).
        tool_results: Results from the most recent tool calls.
        available_tools: Tools the model can call.
        task: Static task description (same every turn). None = not supported.
        state: Dynamic state observation (changes each turn). None = not supported.
    """

    prompt: str
    messages: tuple[dict[str, Any], ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    available_tools: tuple[ToolDefinition, ...] = ()
    task: ObservationContent | None = None
    state: ObservationContent | None = None

    def get_images(self) -> ObservationImages:
        """Extract images from structured fields, separated by source.

        Tags each :class:`ImageContent` with ``source="task"`` or
        ``source="state"`` so downstream consumers can distinguish them.
        """
        task_imgs = tuple(
            ImageContent(data=img.data, media_type=img.media_type, source="task")
            for img in (self.task.images if self.task is not None else ())
        )
        state_imgs = tuple(
            ImageContent(data=img.data, media_type=img.media_type, source="state")
            for img in (self.state.images if self.state is not None else ())
        )
        return ObservationImages(task=task_imgs, state=state_imgs)


@dataclass(frozen=True)
class Action:
    """Unified action supporting both text and tool calls.

    Models can respond with just text, just tool calls, or both.
    Text-only environments use Action(text="...") with empty tool_calls.

    Attributes:
        text: Optional text response.
        tool_calls: Optional tuple of tool calls.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @classmethod
    def from_text(cls, text: str) -> Action:
        """Create an action with only text."""
        return cls(text=text, tool_calls=())

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> Action:
        """Create an action with a single tool call."""
        return cls(text=None, tool_calls=(call,))

    @classmethod
    def from_tool_calls(cls, calls: tuple[ToolCall, ...]) -> Action:
        """Create an action with multiple tool calls."""
        return cls(text=None, tool_calls=calls)

    @property
    def is_text_only(self) -> bool:
        """Check if this action contains only text."""
        return self.text is not None and len(self.tool_calls) == 0

    @property
    def has_tool_calls(self) -> bool:
        """Check if this action contains any tool calls."""
        return len(self.tool_calls) > 0

    @property
    def has_text(self) -> bool:
        """Check if this action contains text."""
        return self.text is not None
