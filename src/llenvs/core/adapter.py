"""Adapter protocol for third-party task libraries.

Adapters bridge between external libraries (like reasoning-gym) and
our common environment interface. Each adapter knows how to:
- List available environments from that library
- Create Environment instances from environment names
- Handle library-specific details (scoring, dataset creation, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llenvs.core.environment import Environment
from llenvs.core.extraction import AnswerExtractor

if TYPE_CHECKING:
    from llenvs.inference.prompts import PromptTemplate


@runtime_checkable
class Adapter(Protocol):
    """Protocol for third-party library adapters.

    An adapter provides access to environments from a specific library
    without exposing library-specific details to the rest of the system.

    Example:
        adapter = ReasoningGymAdapter()
        env = adapter.get_environment("sudoku", size=100, seed=42)
    """

    @property
    def name(self) -> str:
        """Unique identifier for this adapter (e.g., 'reasoning_gym')."""
        ...

    def list_environments(self) -> list[str]:
        """List all environment names available from this adapter.

        Returns:
            List of environment names that can be passed to get_environment().
        """
        ...

    def get_environment(
        self,
        name: str,
        **kwargs: Any,
    ) -> Environment[Any, Any, Any]:
        """Create an environment by name.

        Args:
            name: Environment name (from list_environments()).
            **kwargs: Environment-specific configuration (size, seed, etc.).

        Returns:
            Configured Environment instance.

        Raises:
            ValueError: If the environment name is not recognized.
            ImportError: If the underlying library is not installed.
        """
        ...

    def get_native_extractor(self, task_name: str) -> AnswerExtractor | None:
        """Return native extraction for a specific task, or None.

        Adapters should:
        1. Return a task-specific native extractor if the library provides one.
        2. Fall back to a generic native extractor if the library has one.
        3. Return None if the library provides no extraction at all.

        Args:
            task_name: The task/environment name.

        Returns:
            An AnswerExtractor wrapping the library's native extraction,
            or None if not available.
        """
        ...

    def get_default_system_prompt(self, name: str) -> str | None:
        """Return default system prompt for an environment, or None.

        Adapters return the prompt that their library recommends for this
        environment. The resolution chain respects this over llenvs defaults,
        but user config always wins.

        Args:
            name: Environment/task name.

        Returns:
            System prompt string or None if not available.
        """
        ...

    def get_prompt_template(self, name: str) -> PromptTemplate | None:
        """Return default prompt template for wrapping this task's questions.

        Adapters return None by default. Override to provide task-specific
        question wrapping.

        Args:
            name: Environment/task name.

        Returns:
            A PromptTemplate or None if no wrapping needed.
        """
        ...

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about an environment without creating it.

        Args:
            name: Environment name.

        Returns:
            Dictionary with environment metadata (description, category, etc.).
            Returns empty dict if no metadata available.
        """
        ...
