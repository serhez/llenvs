"""Environment adapters for different libraries.

Adapters bridge between third-party libraries and our common interface.
Available adapters are automatically registered with the environment_registry.
"""

import logging

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
from llenvs.adapters.aviary import (
    AVIARY_PRESETS,
    AviaryAdapter,
    AviaryEnvironment,
    AviaryHidden,
    AviaryReward,
)
from llenvs.adapters.craftax import (
    CRAFTAX_PRESETS,
    CraftaxAchievementReward,
    CraftaxActionMapper,
    CraftaxAdapter,
    CraftaxEnvironment,
    CraftaxHidden,
    CraftaxReward,
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
    FrozenLakeObservationMapper,
    GridObservationMapper,
    GymnasiumAdapter,
    GymnasiumEnvironment,
    GymnasiumHidden,
    GymnasiumReward,
    ImageObservationMapper,
    MultimodalObservationMapper,
)
from llenvs.adapters.harbor import (
    HarborAdapter,
    HarborEnvironment,
    HarborHidden,
    HarborReward,
    HarborToolEnvironment,
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
from llenvs.adapters.open_apps import (
    OPEN_APPS_MODULES,
    OPEN_APPS_TASKS,
    OpenAppsAdapter,
    OpenAppsEnvironment,
    OpenAppsHidden,
    OpenAppsReward,
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
from llenvs.adapters.tau import (
    TAU_DOMAINS,
    TAU_DOMAINS_WITH_SPLITS,
    TAU_SPLITS,
    TauAdapter,
    TauDetailedRewards,
    TauEnvironment,
    TauHidden,
    TauReward,
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
from llenvs.adapters.verifiers_v1 import (
    VerifiersV1Adapter,
    VerifiersV1Hidden,
    VerifiersV1SingleTurnEnvironment,
    VerifiersV1TraceRewards,
)
from llenvs.adapters.webshop import (
    WebShopAdapter,
    WebShopEnvironment,
    WebShopHidden,
    WebShopReward,
    webshop_restore,
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
    # Harbor
    "HarborEnvironment",
    "HarborToolEnvironment",
    "HarborHidden",
    "HarborReward",
    "HarborAdapter",
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
    # OpenApps
    "OpenAppsEnvironment",
    "OpenAppsHidden",
    "OpenAppsAdapter",
    "OpenAppsReward",
    "OPEN_APPS_TASKS",
    "OPEN_APPS_MODULES",
    # WebShop
    "WebShopEnvironment",
    "WebShopHidden",
    "WebShopAdapter",
    "WebShopReward",
    "webshop_restore",
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
    # Verifiers v1
    "VerifiersV1SingleTurnEnvironment",
    "VerifiersV1Hidden",
    "VerifiersV1TraceRewards",
    "VerifiersV1Adapter",
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
    "FrozenLakeObservationMapper",
    "GridObservationMapper",
    "ImageObservationMapper",
    "MultimodalObservationMapper",
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
    # Tau
    "TauEnvironment",
    "TauHidden",
    "TauAdapter",
    "TauReward",
    "TauDetailedRewards",
    "TAU_DOMAINS",
    "TAU_DOMAINS_WITH_SPLITS",
    "TAU_SPLITS",
]


logger = logging.getLogger(__name__)


def _register_adapters() -> None:
    """Register available adapters with the environment registry.

    Called automatically on import. Each optional adapter is probed by
    importing its third-party stack; a probe that fails for any reason —
    missing package, broken install, or an unavailable system dependency
    (e.g. pyjnius raising ``RuntimeError`` when no JVM is present) — skips
    that adapter instead of breaking ``import llenvs``.
    """
    from llenvs.core.registry import environment_registry

    # (display name, adapter class, probe importing the third-party stack)
    optional_adapters: tuple[tuple[str, type, str], ...] = (
        ("reasoning-gym", ReasoningGymAdapter, "_get_reasoning_gym"),
        ("HuggingFace datasets", HuggingFaceAdapter, "_get_datasets_library"),
        ("GEM", GemAdapter, "_get_gem"),
        ("WebShop", WebShopAdapter, "_get_webshop"),
        ("AgentGym", AgentGymAdapter, "_get_agentenv"),
        ("verifiers", VerifiersAdapter, "_get_verifiers"),
        ("verifiers v1", VerifiersV1Adapter, "_get_verifiers_v1"),
        ("OpenEnv", OpenEnvAdapter, "_get_openenv"),
        ("gymnasium", GymnasiumAdapter, "_get_gymnasium"),
        ("AlfWorld", AlfWorldAdapter, "_get_alfworld"),
        ("InterCode", InterCodeAdapter, "_get_intercode"),
        ("Jericho", JerichoAdapter, "_get_jericho"),
        ("LMRL-Gym", LMRLAdapter, "_get_lmrl"),
        ("Aviary", AviaryAdapter, "_get_aviary"),
        ("SciAgentGYM", SciAgentGymAdapter, "_get_sciagentgym"),
        ("MARE", MAREAdapter, "_get_mare"),
        ("MolmoSpaces", MolmoSpacesAdapter, "_get_molmospaces"),
        ("tau", TauAdapter, "_get_tau"),
        ("OpenApps", OpenAppsAdapter, "_get_open_apps"),
        ("Harbor", HarborAdapter, "_get_harbor_api"),
        ("Craftax", CraftaxAdapter, "_get_craftax"),
    )

    for display_name, adapter_cls, probe_name in optional_adapters:
        try:
            adapter = adapter_cls()
            getattr(adapter, probe_name)()
            environment_registry.register_adapter(adapter)
        except ValueError:
            pass  # Already registered (e.g., during testing)
        except Exception as exc:
            logger.debug("Skipping %s adapter registration: %s", display_name, exc)

    # The dialogue adapter has no third-party deps, so any failure other than
    # double registration is a real bug and must surface.
    try:
        environment_registry.register_adapter(DialogueAdapter())
    except ValueError:
        pass  # Already registered (e.g., during testing)


# Auto-register adapters on import
_register_adapters()
