"""Registry with lazy loading and decorators.

Provides a generic registry pattern for environments, extractors,
backends, and other pluggable components.
"""

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry[T]:
    """A registry for named components with lazy loading support.

    Components can be registered either directly or via factory functions
    that are only called when the component is first accessed.

    Example:
        >>> registry = Registry[AnswerExtractor]("extractor")
        >>> @registry.register("tag_based")
        ... class TagBasedExtractor:
        ...     pass
        >>> extractor = registry.get("tag_based")
    """

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: Human-readable name for this registry (for error messages).
        """
        self.name = name
        self._items: dict[str, T] = {}
        self._factories: dict[str, Callable[[], T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], T] | None = None,
    ) -> Callable[[type[T]], type[T]] | None:
        """Register a component or factory.

        Can be used as a decorator or called directly.

        Args:
            name: Unique name for the component.
            factory: Optional factory function. If None, used as decorator.

        Returns:
            Decorator function if factory is None, else None.

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._items or name in self._factories:
            raise ValueError(f"{self.name} '{name}' is already registered")

        if factory is not None:
            self._factories[name] = factory
            return None

        def decorator(cls: type[T]) -> type[T]:
            # Register a factory that instantiates the class
            self._factories[name] = cls  # type: ignore[assignment]
            return cls

        return decorator

    def register_instance(self, name: str, instance: T) -> None:
        """Register an already-instantiated component.

        Args:
            name: Unique name for the component.
            instance: The component instance.

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._items or name in self._factories:
            raise ValueError(f"{self.name} '{name}' is already registered")
        self._items[name] = instance

    def get(self, name: str, **kwargs: Any) -> T:
        """Get a component by name, instantiating if needed.

        Args:
            name: Name of the component.
            **kwargs: Arguments passed to factory if instantiating.

        Returns:
            The component instance.

        Raises:
            KeyError: If name is not registered.
        """
        # Return cached instance if available
        if name in self._items:
            return self._items[name]

        # Instantiate from factory
        if name in self._factories:
            factory = self._factories[name]
            instance = factory(**kwargs) if kwargs else factory()
            self._items[name] = instance
            return instance

        raise KeyError(f"{self.name} '{name}' not found. Available: {list(self.keys())}")

    def create(self, name: str, **kwargs: Any) -> T:
        """Create a new instance without caching.

        Always calls the factory, even if an instance is cached.

        Args:
            name: Name of the component.
            **kwargs: Arguments passed to factory.

        Returns:
            A new component instance.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._factories:
            if name in self._items:
                raise KeyError(f"{self.name} '{name}' was registered as instance, not factory")
            raise KeyError(f"{self.name} '{name}' not found. Available: {list(self.keys())}")

        factory = self._factories[name]
        return factory(**kwargs) if kwargs else factory()

    def keys(self) -> list[str]:
        """Get all registered component names."""
        return list(set(self._items.keys()) | set(self._factories.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._items or name in self._factories

    def __iter__(self):
        """Iterate over registered names."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Number of registered components."""
        return len(self.keys())


class EnvironmentRegistry:
    """Registry for environments with adapter support.

    Environments are accessed by name and adapter. Adapters must be
    registered first, then environments can be created through them.

    Example:
        registry = EnvironmentRegistry()
        registry.register_adapter(ReasoningGymAdapter())
        env = registry.get(name="sudoku", adapter="reasoning_gym", size=100)
    """

    def __init__(self) -> None:
        from llenvs.core.adapter import Adapter

        self._adapters: dict[str, Adapter] = {}

    def register_adapter(self, adapter: Any) -> None:
        """Register an adapter.

        Args:
            adapter: Adapter instance implementing the Adapter protocol.

        Raises:
            ValueError: If an adapter with this name is already registered.
        """
        name = adapter.name
        if name in self._adapters:
            raise ValueError(f"Adapter '{name}' is already registered")
        self._adapters[name] = adapter

    def unregister_adapter(self, name: str) -> None:
        """Unregister an adapter by name.

        Args:
            name: Adapter name to unregister.
        """
        self._adapters.pop(name, None)

    def get_adapter(self, name: str) -> Any:
        """Get a registered adapter by name.

        Args:
            name: Adapter name.

        Returns:
            The adapter instance.

        Raises:
            KeyError: If adapter is not registered.
        """
        if name not in self._adapters:
            raise KeyError(
                f"Adapter '{name}' not registered. Available: {list(self._adapters.keys())}"
            )
        return self._adapters[name]

    def get(
        self,
        name: str,
        adapter: str,
        **kwargs: Any,
    ) -> Any:
        """Get an environment by name and adapter.

        Args:
            name: Environment name (e.g., "sudoku").
            adapter: Adapter name (e.g., "reasoning_gym").
            **kwargs: Additional arguments passed to the adapter.

        Returns:
            Configured Environment instance.

        Raises:
            KeyError: If adapter is not registered.
            ValueError: If environment name is not recognized by adapter.
        """
        adapter_instance = self.get_adapter(adapter)
        return adapter_instance.get_environment(name, **kwargs)

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    def list_environments(self, adapter: str | None = None) -> list[tuple[str, str]]:
        """List available environments.

        Args:
            adapter: If provided, only list environments from this adapter.
                Otherwise, list from all adapters.

        Returns:
            List of (adapter_name, environment_name) tuples.
        """
        result: list[tuple[str, str]] = []

        adapters_to_check = [self._adapters[adapter]] if adapter else self._adapters.values()

        for adp in adapters_to_check:
            for env_name in adp.list_environments():
                result.append((adp.name, env_name))

        return result

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check if (adapter, name) is available.

        Args:
            key: Tuple of (adapter_name, environment_name).
        """
        adapter_name, env_name = key
        if adapter_name not in self._adapters:
            return False
        return env_name in self._adapters[adapter_name].list_environments()


# Global registries for common component types
environment_registry = EnvironmentRegistry()
answer_extractor_registry: Registry[Any] = Registry("extractor")
backend_registry: Registry[Any] = Registry("backend")
reward_function_registry: Registry[Any] = Registry("reward_function")
adapter_registry: Registry[Any] = Registry("adapter")


def register_defaults() -> None:
    """Register default implementations in global registries.

    Called automatically on import, but can be called again to reset.
    """
    from llenvs.core.extraction import (
        BoxedExtractor,
        CodeBlockExtractor,
        GSM8KExtractor,
        LastLineExtractor,
        MultipleChoiceExtractor,
        NumericExtractor,
        PatternAnswerExtractor,
        RawGenerationExtractor,
        RegexExtractor,
        TagBasedExtractor,
    )

    # Clear and re-register extractors
    answer_extractor_registry._items.clear()
    answer_extractor_registry._factories.clear()

    answer_extractor_registry.register("tag_based", TagBasedExtractor)
    answer_extractor_registry.register("regex", RegexExtractor)
    answer_extractor_registry.register("gsm8k", GSM8KExtractor)
    answer_extractor_registry.register("multiple_choice", MultipleChoiceExtractor)
    answer_extractor_registry.register("raw", RawGenerationExtractor)
    answer_extractor_registry.register("boxed", BoxedExtractor)
    answer_extractor_registry.register("numeric", NumericExtractor)
    answer_extractor_registry.register("last_line", LastLineExtractor)
    answer_extractor_registry.register("code_block", CodeBlockExtractor)
    answer_extractor_registry.register("pattern_answer", PatternAnswerExtractor)


# Register defaults on import
register_defaults()

# Import adapters to trigger auto-registration with environment_registry
# This must come after environment_registry is created
import llenvs.adapters  # noqa: E402, F401
