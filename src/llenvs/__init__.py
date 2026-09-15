"""llenvs: MDP-style access to evaluation environments for LLM research.

Key capabilities:
- State history & rollback (checkpoint and branch trajectories)
- Step-level rewards (per-turn signals, not just outcomes)
- Partial generation control (stop mid-response for reward probing)
- Multiple inference backends (vLLM, OpenAI, Anthropic, OpenRouter, Codex CLI)
"""

from llenvs.adapters.iterative import IterativeEnvironment
from llenvs.core.branching import BranchManager
from llenvs.core.config import EvalConfig
from llenvs.core.environment import Environment, EnvironmentSpec, StepResult
from llenvs.core.extraction import (
    AnswerExtractor,
    GSM8KExtractor,
    JSONFieldExtractor,
    MultipleChoiceExtractor,
    RegexExtractor,
    TagBasedExtractor,
)
from llenvs.core.judge import JudgeReward
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    ObservationImages,
    State,
    StateMetadata,
)
from llenvs.core.trajectory import Checkpoint, Trajectory, Transition
from llenvs.environments.coding import IterativeCodingEnvironment
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem
from llenvs.integrations.scoring import Scorer, ScoringResult

__version__ = "0.1.0"

__all__ = [
    # State
    "State",
    "StateMetadata",
    "Observation",
    "ObservationContent",
    "ObservationImages",
    "Action",
    "ImageContent",
    # Rewards
    "Signal",
    "SignalBundle",
    "RewardType",
    "RewardFunction",
    # Trajectory
    "Trajectory",
    "Transition",
    "Checkpoint",
    # Environment
    "Environment",
    "StepResult",
    "EnvironmentSpec",
    # Extraction
    "AnswerExtractor",
    "TagBasedExtractor",
    "RegexExtractor",
    "GSM8KExtractor",
    "JSONFieldExtractor",
    "MultipleChoiceExtractor",
    # Config
    "EvalConfig",
    # Judge
    "JudgeReward",
    # Branching
    "BranchManager",
    # Iterative
    "IterativeEnvironment",
    "IterativeCodingEnvironment",
    # Integrations
    "Scorer",
    "ScoringResult",
    "DatasetProvider",
    "TaskItem",
]
