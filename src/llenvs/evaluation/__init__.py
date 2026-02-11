"""Evaluation orchestration and metrics."""

from llenvs.evaluation.runner import (
    COMPLETE,
    ForceAction,
    MultiEvalEntry,
    TrajectoryRunner,
    TrajectoryResult,
    SegmentedTrajectoryRunner,
    BatchResult,
    run_evaluation,
    run_multi_evaluation,
    run_segmented_evaluation,
)
from llenvs.evaluation.continuation import (
    ContinuationStrategy,
    TokenContinuationStrategy,
    BoundaryContinuationStrategy,
    select_strategy,
)
from llenvs.evaluation.metrics import (
    ContinuousStatistics,
    BinaryStatistics,
    Metric,
    MetricsBundle,
    compute_continuous_statistics,
    compute_binary_statistics,
    compute_action_reward,
    compute_trajectory_reward,
    compute_accuracy,
    compute_format_compliance,
    compute_all_metrics,
    aggregate_continuous_metrics,
    aggregate_binary_metrics,
)
from llenvs.evaluation.results import (
    EvaluationMetadata,
    EvaluationResult,
    format_trajectory_result,
    create_evaluation_result,
    print_summary,
)

__all__ = [
    # Runner
    "COMPLETE",
    "ForceAction",
    "MultiEvalEntry",
    "TrajectoryRunner",
    "TrajectoryResult",

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
