"""Evaluation orchestration and metrics."""

from llenvs.evaluation.continuation import (
    BoundaryContinuationStrategy,
    ContinuationStrategy,
    TokenContinuationStrategy,
    select_strategy,
)
from llenvs.evaluation.history import (
    BudgetHistoryFn,
    HistoryEntry,
    HistoryFn,
    PromptBudget,
    full_history,
    last_n_history,
    no_history,
    sliding_window_history,
)
from llenvs.evaluation.logging import LogConfig
from llenvs.evaluation.metrics import (
    BinaryStatistics,
    ContinuousStatistics,
    Metric,
    MetricsBundle,
    aggregate_binary_metrics,
    aggregate_continuous_metrics,
    compute_accuracy,
    compute_action_reward,
    compute_all_metrics,
    compute_binary_statistics,
    compute_continuous_statistics,
    compute_format_compliance,
    compute_trajectory_reward,
)
from llenvs.evaluation.results import (
    EvaluationMetadata,
    EvaluationResult,
    create_evaluation_result,
    format_trajectory_result,
    print_summary,
)
from llenvs.evaluation.runner import (
    COMPLETE,
    BatchResult,
    ForceAction,
    MultiEvalEntry,
    SegmentedTrajectoryRunner,
    TrajectoryResult,
    TrajectoryRunner,
    TurnInfoConfig,
    run_evaluation,
    run_multi_evaluation,
    run_segmented_evaluation,
)

__all__ = [
    # History control
    "BudgetHistoryFn",
    "HistoryEntry",
    "HistoryFn",
    "PromptBudget",
    "full_history",
    "no_history",
    "last_n_history",
    "sliding_window_history",
    # Logging
    "LogConfig",
    # Runner
    "COMPLETE",
    "ForceAction",
    "MultiEvalEntry",
    "TrajectoryRunner",
    "TrajectoryResult",
    "TurnInfoConfig",
    "SegmentedTrajectoryRunner",
    "BatchResult",
    "run_evaluation",
    "run_multi_evaluation",
    "run_segmented_evaluation",
    # Continuation strategies
    "ContinuationStrategy",
    "TokenContinuationStrategy",
    "BoundaryContinuationStrategy",
    "select_strategy",
    # Metrics
    "ContinuousStatistics",
    "BinaryStatistics",
    "Metric",
    "MetricsBundle",
    "compute_continuous_statistics",
    "compute_binary_statistics",
    "compute_action_reward",
    "compute_trajectory_reward",
    "compute_accuracy",
    "compute_format_compliance",
    "compute_all_metrics",
    "aggregate_continuous_metrics",
    "aggregate_binary_metrics",
    # Results
    "EvaluationMetadata",
    "EvaluationResult",
    "format_trajectory_result",
    "create_evaluation_result",
    "print_summary",
]
