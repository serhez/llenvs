"""llenvs: MDP-style access to evaluation environments for LLM research.

Key capabilities:
- State history & rollback (checkpoint and branch trajectories)
- Step-level rewards (per-turn signals, not just outcomes)
- Partial generation control (stop mid-response for reward probing)
- Multiple inference backends (vLLM, OpenAI, Anthropic, OpenRouter)
"""

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import RewardSignal, RewardBundle, RewardType, RewardFunction
from llenvs.core.trajectory import Trajectory, Transition, Checkpoint
from llenvs.core.environment import Environment, StepResult, EnvironmentSpec
from llenvs.core.extraction import (
    AnswerExtractor,
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
)
from llenvs.core.config import EvalConfig
from llenvs.core.judge import JudgeReward
from llenvs.core.branching import BranchManager
from llenvs.integrations.scoring import Scorer, ScoringResult
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem

__version__ = "0.1.0"

__all__ = [
    # State
    "State",
    "StateMetadata",
    "Observation",
    "Action",
    # Rewards
    "RewardSignal",
    "RewardBundle",
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
    "MultipleChoiceExtractor",
    # Config
    "EvalConfig",
    # Judge
    "JudgeReward",
    # Branching
    "BranchManager",
    # Integrations
    "Scorer",
    "ScoringResult",
    "DatasetProvider",
    "TaskItem",
]
