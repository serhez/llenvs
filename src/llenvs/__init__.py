"""llenvs: MDP-style access to evaluation environments for LLM research.

Key capabilities:
- State history & rollback (checkpoint and branch trajectories)
- Step-level rewards (per-turn signals, not just outcomes)
- Partial generation control (stop mid-response for reward probing)
- Multiple inference backends (vLLM, OpenAI, Anthropic, OpenRouter)
"""

from env_evals.core.state import State, StateMetadata, TextObservation, TextAction
from env_evals.core.reward import RewardSignal, RewardBundle, RewardType, RewardFunction
from env_evals.core.trajectory import Trajectory, Transition, Checkpoint
from env_evals.core.environment import Environment, StepResult, EnvironmentSpec
from env_evals.core.extraction import (
    AnswerExtractor,
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
)
from env_evals.core.config import EvalConfig

__version__ = "0.1.0"

__all__ = [
    # State
    "State",
    "StateMetadata",
    "TextObservation",
    "TextAction",
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
]
