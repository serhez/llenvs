"""Environment adapters for different libraries.

Adapters bridge between third-party libraries and our common interface.
Available adapters are automatically registered with the environment_registry.
"""

from llenvs.adapters.agentgym import (
    AgentGymAdapter,
    AgentGymEnvironment,
    AgentGymHidden,
    AgentGymReward,
)
from llenvs.adapters.alfworld import (
    ALFWORLD_TASK_TYPES,
    AlfWorldAdapter,
    AlfWorldEnvironment,
    AlfWorldHidden,
    AlfWorldReward,
)
from llenvs.adapters.craftax import (
    CRAFTAX_PRESETS,
    CraftaxActionMapper,
    CraftaxAdapter,
    CraftaxAchievementReward,
    CraftaxEnvironment,
    CraftaxHidden,
    CraftaxReward,
)
from llenvs.adapters.aviary import (
    AVIARY_PRESETS,
    AviaryAdapter,
    AviaryEnvironment,
    AviaryHidden,
    AviaryReward,
)
from llenvs.adapters.dialogue import (
    DIALOGUE_PRESETS,
    DialogueAdapter,
    DialogueEnvironment,
    DialogueHidden,
    DialogueTask,
)
from llenvs.adapters.gem import (
    GEM_PYTHON_TOOL,
    GEM_SEARCH_TOOL,
    GEM_SUBMIT_ANSWER_TOOL,
    MULTI_TURN_ENVS,
    GemAdapter,
    GemEnvironment,
    GemHidden,
    GemToolEnvironment,
    GemToolExecutor,
    GemToolHidden,
)
from llenvs.adapters.gymnasium import (
    GYMNASIUM_PRESETS,
    AutoActionMapper,
    AutoObservationMapper,
    GridObservationMapper,
    GymnasiumAdapter,
    GymnasiumEnvironment,
    GymnasiumHidden,
    GymnasiumReward,
    ImageObservationMapper,
    MultimodalObservationMapper,
)
from llenvs.adapters.huggingface import (
    DATASET_PRESETS,
    HuggingFaceAdapter,
    HuggingFaceEnvironment,
    HuggingFaceHidden,
)
from llenvs.adapters.intercode import (
    INTERCODE_PRESETS,
    InterCodeAdapter,
    InterCodeEnvironment,
    InterCodeHidden,
    InterCodeReward,
)
from llenvs.adapters.jericho import (
    JerichoAdapter,
    JerichoEnvironment,
    JerichoHidden,
    JerichoReward,
)
from llenvs.adapters.lmrl import (
    LMRL_PRESETS,
    LMRLAdapter,
    LMRLEnvironment,
    LMRLHidden,
    LMRLReward,
)
from llenvs.adapters.mare import (
    MARE_CAPABILITIES,
    MAREAdapter,
    MAREEnvironment,
    MAREHidden,
    MAREReward,
)
from llenvs.adapters.molmospaces import (
    MOLMOSPACES_PRESETS,
    MOLMOSPACES_TASKS,
    MolmoSpacesAdapter,
    MolmoSpacesEnvironment,
    MolmoSpacesHidden,
    MolmoSpacesReward,
    MolmoSpacesSuccessReward,
    MolmoSpacesToolExecutor,
)
from llenvs.adapters.openenv import (
    OpenEnvAdapter,
    OpenEnvEnvironment,
    OpenEnvHidden,
    OpenEnvReward,
    OpenEnvToolEnvironment,
)
from llenvs.adapters.reasoning_gym import (
    ReasoningGymAdapter,
    ReasoningGymEnvironment,
    ReasoningGymHidden,
)
from llenvs.adapters.sciagentgym import (
    SCIAGENTGYM_SUBJECTS,
    SciAgentGymAdapter,
    SciAgentGymEnvironment,
    SciAgentGymHidden,
    SciAgentGymReward,
)
from llenvs.adapters.tau2 import (
    TAU2_DOMAINS,
    TAU2_SPLITS,
    Tau2Adapter,
    Tau2DetailedRewards,
    Tau2Environment,
    Tau2Hidden,
    Tau2Reward,
)
from llenvs.adapters.verifiers import (
    VerifiersAdapter,
    VerifiersHidden,
    VerifiersRubricReward,
    VerifiersSingleTurnEnvironment,
    VerifiersToolEnvironment,
    VerifiersToolExecutor,
    VerifiersToolHidden,
)
from llenvs.adapters.webshop import (
    WebShopAdapter,
    WebShopEnvironment,
    WebShopHidden,
    WebShopReward,
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
    # AlfWorld
    "AlfWorldEnvironment",
    "AlfWorldHidden",
    "AlfWorldAdapter",
    "AlfWorldReward",
    "ALFWORLD_TASK_TYPES",
    # InterCode
    "InterCodeEnvironment",
    "InterCodeHidden",
    "InterCodeAdapter",
    "InterCodeReward",
    "INTERCODE_PRESETS",
    # Jericho
    "JerichoEnvironment",
    "JerichoHidden",
    "JerichoAdapter",
    "JerichoReward",
    # LMRL-Gym
    "LMRLEnvironment",
    "LMRLHidden",
    "LMRLAdapter",
    "LMRLReward",
    "LMRL_PRESETS",
    # Aviary
    "AviaryEnvironment",
    "AviaryHidden",
    "AviaryAdapter",
    "AviaryReward",
    "AVIARY_PRESETS",
    # SciAgentGYM
    "SciAgentGymEnvironment",
    "SciAgentGymHidden",
    "SciAgentGymAdapter",
    "SciAgentGymReward",
    "SCIAGENTGYM_SUBJECTS",
    # MARE
    "MAREEnvironment",
    "MAREHidden",
    "MAREAdapter",
    "MAREReward",
    "MARE_CAPABILITIES",
    # Craftax
    "CraftaxEnvironment",
    "CraftaxHidden",
    "CraftaxAdapter",
    "CraftaxReward",
    "CraftaxAchievementReward",
    "CraftaxActionMapper",
    "CRAFTAX_PRESETS",
    # MolmoSpaces
    "MolmoSpacesEnvironment",
    "MolmoSpacesHidden",
    "MolmoSpacesAdapter",
    "MolmoSpacesReward",
    "MolmoSpacesSuccessReward",
    "MolmoSpacesToolExecutor",
    "MOLMOSPACES_PRESETS",
    "MOLMOSPACES_TASKS",
    # tau2
    "Tau2Environment",
    "Tau2Hidden",
    "Tau2Adapter",
    "Tau2Reward",
    "Tau2DetailedRewards",
    "TAU2_DOMAINS",
    "TAU2_SPLITS",
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

    # Register AlfWorld adapter if available
    try:
        adapter = AlfWorldAdapter()
        adapter._get_alfworld()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # alfworld not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register InterCode adapter if available
    try:
        adapter = InterCodeAdapter()
        adapter._get_intercode()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # intercode not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register Jericho adapter if available
    try:
        adapter = JerichoAdapter()
        adapter._get_jericho()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # jericho not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register LMRL-Gym adapter if available
    try:
        adapter = LMRLAdapter()
        adapter._get_lmrl()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # LLM_RL not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register Aviary adapter if available
    try:
        adapter = AviaryAdapter()
        adapter._get_aviary()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # fhaviary not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register SciAgentGYM adapter if available
    try:
        adapter = SciAgentGymAdapter()
        adapter._get_sciagentgym()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # SciAgentGYM not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register MARE adapter if available
    try:
        adapter = MAREAdapter()
        adapter._get_mare()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # meta-agents-research-environments not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register MolmoSpaces adapter if available
    try:
        adapter = MolmoSpacesAdapter()
        adapter._get_molmospaces()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # molmospaces not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register tau2 adapter if available
    try:
        adapter = Tau2Adapter()
        adapter._get_tau2()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # tau2 not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)

    # Register Craftax adapter if available
    try:
        adapter = CraftaxAdapter()
        adapter._get_craftax()
        environment_registry.register_adapter(adapter)
    except ImportError:
        pass  # craftax not installed, skip registration
    except ValueError:
        pass  # Already registered (e.g., during testing)


# Auto-register adapters on import
_register_adapters()
