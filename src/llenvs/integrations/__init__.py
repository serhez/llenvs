"""RL training framework integrations.

Provides standalone scoring, dataset provision, and token masking
that work with any RL framework. Framework-specific adapters
(veRL, TRL, OpenRLHF) are thin wrappers around these primitives.
"""

from llenvs.integrations.scoring import Scorer, ScoringResult
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem
from llenvs.integrations.token_mask import TrajectoryMasker, MaskedTrajectory, TokenSpan

__all__ = [
    "Scorer",
    "ScoringResult",
    "DatasetProvider",
    "TaskItem",
    "TrajectoryMasker",
    "MaskedTrajectory",
    "TokenSpan",
]
