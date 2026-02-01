"""Environment adapters for different libraries.

Adapters bridge between third-party libraries and our common interface.
Available adapters are automatically registered with the environment_registry.
"""

from env_evals.adapters.reasoning_gym import (
    ReasoningGymEnvironment,
    ReasoningGymHidden,
    ReasoningGymAdapter,
    create_reasoning_gym_environment,
)
from env_evals.adapters.huggingface import (
    HuggingFaceEnvironment,
    HuggingFaceHidden,
    HuggingFaceAdapter,
    create_huggingface_environment,
    DATASET_PRESETS,
)
from env_evals.adapters.gem import (
    GemEnvironment,
    GemHidden,
    GemToolEnvironment,
    GemToolHidden,
    GemToolExecutor,
    GemAdapter,
    create_gem_environment,
    create_gem_tool_environment,
    MULTI_TURN_ENVS,
    GEM_PYTHON_TOOL,
    GEM_SEARCH_TOOL,
    GEM_SUBMIT_ANSWER_TOOL,
)
from env_evals.adapters.webshop import (
    WebShopEnvironment,
    WebShopHidden,
    WebShopAdapter,
    WebShopReward,
    create_webshop_environment,
)

__all__ = [
    # ReasoningGym
    "ReasoningGymEnvironment",
    "ReasoningGymHidden",
    "ReasoningGymAdapter",
    "create_reasoning_gym_environment",
    # HuggingFace
    "HuggingFaceEnvironment",
    "HuggingFaceHidden",
    "HuggingFaceAdapter",
    "create_huggingface_environment",
    "DATASET_PRESETS",
    # GEM
    "GemEnvironment",
    "GemHidden",
    "GemToolEnvironment",
    "GemToolHidden",
    "GemToolExecutor",
    "GemAdapter",
    "create_gem_environment",
    "create_gem_tool_environment",
    "MULTI_TURN_ENVS",
    "GEM_PYTHON_TOOL",
    "GEM_SEARCH_TOOL",
    "GEM_SUBMIT_ANSWER_TOOL",
    # WebShop
    "WebShopEnvironment",
    "WebShopHidden",
    "WebShopAdapter",
    "WebShopReward",
    "create_webshop_environment",
]


def _register_adapters() -> None:
    """Register available adapters with the environment registry.

    Called automatically on import. Adapters for libraries that aren't
    installed are silently skipped.
    """
    from env_evals.core.registry import environment_registry

    # Register reasoning-gym adapter if available
    try:
        adapter = ReasoningGymAdapter()
        # Test that reasoning-gym is actually importable
        adapter._get_reasoning_gym()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # reasoning-gym not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register HuggingFace datasets adapter if available
    try:
        adapter = HuggingFaceAdapter()
        # Test that datasets library is actually importable
        adapter._get_datasets_library()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # datasets not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register GEM adapter if available
    try:
        adapter = GemAdapter()
        # Test that gem is actually importable
        adapter._get_gem()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # gem not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register WebShop adapter if available
    try:
        adapter = WebShopAdapter()
        # Test that webshop is actually importable
        adapter._get_webshop()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # webshop not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)


# Auto-register adapters on import
_register_adapters()
