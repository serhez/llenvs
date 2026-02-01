"""Adapter protocol for third-party task libraries.

Adapters bridge between external libraries (like reasoning-gym) and
our common environment interface. Each adapter knows how to:
- List available environments from that library
- Create Environment instances from environment names
- Handle library-specific details (scoring, dataset creation, etc.)
"""

from typing import Any, Protocol, runtime_checkable

from env_evals.core.environment import Environment


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

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about an environment without creating it.

        Args:
            name: Environment name.

        Returns:
            Dictionary with environment metadata (description, category, etc.).
            Returns empty dict if no metadata available.
        """
        ...
