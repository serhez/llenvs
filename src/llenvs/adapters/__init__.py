"""Environment adapters for different libraries.

Adapters bridge between third-party libraries and our common interface.
Available adapters are automatically registered with the environment_registry.
"""

from llenvs.adapters.reasoning_gym import (
    ReasoningGymEnvironment,
    ReasoningGymHidden,
    ReasoningGymAdapter,
)
from llenvs.adapters.huggingface import (
    HuggingFaceEnvironment,
    HuggingFaceHidden,
    HuggingFaceAdapter,
    DATASET_PRESETS,
)
from llenvs.adapters.gem import (
    GemEnvironment,
    GemHidden,
    GemToolEnvironment,
    GemToolHidden,
    GemToolExecutor,
    GemAdapter,
    MULTI_TURN_ENVS,
    GEM_PYTHON_TOOL,
    GEM_SEARCH_TOOL,
    GEM_SUBMIT_ANSWER_TOOL,
)
from llenvs.adapters.webshop import (
    WebShopEnvironment,
    WebShopHidden,
    WebShopAdapter,
    WebShopReward,
)
from llenvs.adapters.agentgym import (
    AgentGymEnvironment,
    AgentGymHidden,
    AgentGymAdapter,
    AgentGymReward,
)
from llenvs.adapters.verifiers import (
    VerifiersSingleTurnEnvironment,
    VerifiersToolEnvironment,
    VerifiersHidden,
    VerifiersToolHidden,
    VerifiersToolExecutor,
    VerifiersRubricReward,
    VerifiersAdapter,
)
from llenvs.adapters.openenv import (
    OpenEnvEnvironment,
    OpenEnvToolEnvironment,
    OpenEnvHidden,
    OpenEnvReward,
    OpenEnvAdapter,
)
from llenvs.adapters.dialogue import (
    DialogueEnvironment,
    DialogueHidden,
    DialogueTask,
    DialogueAdapter,
    DIALOGUE_PRESETS,
)
from llenvs.adapters.gymnasium import (
    GymnasiumEnvironment,
    GymnasiumHidden,
    GymnasiumReward,
    GymnasiumAdapter,
    AutoObservationMapper,
    AutoActionMapper,
    GridObservationMapper,
    GYMNASIUM_PRESETS,
)

__all__ = [
    # ReasoningGym
    "ReasoningGymEnvironment",
    "ReasoningGymHidden",
    "ReasoningGymAdapter",
    # HuggingFace
    "HuggingFaceEnvironment",
    "HuggingFaceHidden",
    "HuggingFaceAdapter",
    "DATASET_PRESETS",
    # GEM
    "GemEnvironment",
    "GemHidden",
    "GemToolEnvironment",
    "GemToolHidden",
    "GemToolExecutor",
    "GemAdapter",
    "MULTI_TURN_ENVS",
    "GEM_PYTHON_TOOL",
    "GEM_SEARCH_TOOL",
    "GEM_SUBMIT_ANSWER_TOOL",
    # WebShop
    "WebShopEnvironment",
    "WebShopHidden",
    "WebShopAdapter",
    "WebShopReward",
    # AgentGym
    "AgentGymEnvironment",
    "AgentGymHidden",
    "AgentGymAdapter",
    "AgentGymReward",
    # Verifiers
    "VerifiersSingleTurnEnvironment",
    "VerifiersToolEnvironment",
    "VerifiersHidden",
    "VerifiersToolHidden",
    "VerifiersToolExecutor",
    "VerifiersRubricReward",
    "VerifiersAdapter",
    # OpenEnv
    "OpenEnvEnvironment",
    "OpenEnvToolEnvironment",
    "OpenEnvHidden",
    "OpenEnvReward",
    "OpenEnvAdapter",
    # Dialogue
    "DialogueEnvironment",
    "DialogueHidden",
    "DialogueTask",
    "DialogueAdapter",
    "DIALOGUE_PRESETS",
    # Gymnasium
    "GymnasiumEnvironment",
    "GymnasiumHidden",
    "GymnasiumReward",
    "GymnasiumAdapter",
    "AutoObservationMapper",
    "AutoActionMapper",
    "GridObservationMapper",
    "GYMNASIUM_PRESETS",
]


def _register_adapters() -> None:
    """Register available adapters with the environment registry.

    Called automatically on import. Adapters for libraries that aren't
    installed are silently skipped.
    """
    from llenvs.core.registry import environment_registry

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

    # Register AgentGym adapter if available
    try:
        adapter = AgentGymAdapter()
        adapter._get_agentenv()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # agentenv not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register verifiers adapter if available
    try:
        adapter = VerifiersAdapter()
        adapter._get_verifiers()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # verifiers not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register OpenEnv adapter if available
    try:
        adapter = OpenEnvAdapter()
        adapter._get_openenv()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # openenv-core not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register dialogue adapter (no third-party deps)
    try:
        environment_registry.register_adapter(DialogueAdapter())
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register gymnasium adapter if available
    try:
        adapter = GymnasiumAdapter()
        adapter._get_gymnasium()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # gymnasium not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)


# Auto-register adapters on import
_register_adapters()
