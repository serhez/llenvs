"""Result formatting and output utilities.

Handles serialization of evaluation results to JSON and other formats.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llenvs.evaluation.metrics import (
    BinaryStatistics,
    ContinuousStatistics,
    MetricsBundle,
    compute_all_metrics,
)
from llenvs.evaluation.runner import BatchResult, TrajectoryResult


@dataclass
class EvaluationMetadata:
    """Metadata about an evaluation run.

    Attributes:
        timestamp: When the evaluation started.
        model: Model name/path.
        environment: Environment name.
        duration_seconds: Total runtime.
        num_trajectories: Number of trajectories run.
        config: Configuration used.
    """

    timestamp: str
    model: str
    environment: str
    duration_seconds: float = 0.0
    num_trajectories: int = 0
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result with metadata and metrics.

    Attributes:
        metadata: Evaluation metadata.
        metrics: Computed metrics.
        results: Per-trajectory results (optional, can be large).
        summary: Summary statistics.
    """

    metadata: EvaluationMetadata
    metrics: MetricsBundle
    results: list[dict[str, Any]] | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_results: bool = True) -> dict[str, Any]:
        """Convert to dictionary format.

        Args:
            include_results: Whether to include per-trajectory results.

        Returns:
            Dictionary representation.
        """
        data: dict[str, Any] = {
            "metadata": asdict(self.metadata),
            "metrics": self.metrics.to_dict(),
            "summary": self.summary,
        }

        if include_results and self.results:
            data["results"] = self.results

        return data

    def to_json(self, include_results: bool = True) -> str:
        """Convert to JSON string.

        Args:
            include_results: Whether to include per-trajectory results.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(include_results), indent=2, default=str)

    def save(
        self,
        path: str | Path,
        include_results: bool = True,
    ) -> Path:
        """Save to a JSON file.

        Args:
            path: Output file path.
            include_results: Whether to include per-trajectory results.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_json(include_results))

        return path


def format_trajectory_result(traj_result: TrajectoryResult) -> dict[str, Any]:
    """Format a trajectory result for serialization.

    Args:
        traj_result: The trajectory result.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    trajectory = traj_result.trajectory

    # Extract key information from trajectory
    transitions_data = []
    for t in trajectory.transitions:
        # Get action text
        action_text = t.action.text if hasattr(t.action, "text") else str(t.action)

        # Get rewards
        rewards = {s.name: s.reward for s in t.rewards.signals}

        transitions_data.append(
            {
                "step": t.state.metadata.step,
                "action": action_text[:1000],  # Truncate long responses
                "rewards": rewards,
                "total_reward": t.rewards.total,
                "info": t.info,
            }
        )

    # Get observation from initial state
    observation = trajectory.initial_state.observation
    prompt = observation.prompt if hasattr(observation, "prompt") else str(observation)

    # Get expected answer if available
    expected = None
    if hasattr(trajectory.initial_state.hidden, "expected_answer"):
        expected = trajectory.initial_state.hidden.expected_answer

    return {
        "trajectory_id": trajectory.episode_id,
        "task_index": traj_result.metadata.get("task_index"),
        "prompt": prompt[:2000],  # Truncate
        "expected_answer": expected,
        "success": traj_result.success,
        "total_reward": traj_result.total_reward,
        "num_steps": len(trajectory),
        "transitions": transitions_data,
        "metadata": traj_result.metadata,
    }


def create_evaluation_result(
    batch_result: BatchResult,
    model_name: str,
    environment_name: str,
    start_time: datetime,
    config: dict[str, Any] | None = None,
    include_detailed_results: bool = True,
) -> EvaluationResult:
    """Create an EvaluationResult from a BatchResult.

    Args:
        batch_result: The batch evaluation result.
        model_name: Name of the model used.
        environment_name: Name of the environment.
        start_time: When evaluation started.
        config: Optional configuration dict.
        include_detailed_results: Whether to include per-trajectory details.

    Returns:
        Formatted EvaluationResult.
    """
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Create metadata
    metadata = EvaluationMetadata(
        timestamp=start_time.isoformat(),
        model=model_name,
        environment=environment_name,
        duration_seconds=duration,
        num_trajectories=len(batch_result.trajectory_results),
        config=config or {},
    )

    # Compute metrics
    metrics = compute_all_metrics(batch_result)

    # Format results if requested
    results = None
    if include_detailed_results:
        results = [format_trajectory_result(tr) for tr in batch_result.trajectory_results]

    # Create summary
    accuracy = metrics.get("accuracy")
    summary = {
        "success_rate": batch_result.success_rate,
        "mean_reward": batch_result.mean_reward,
        "accuracy": accuracy.statistics.mean if accuracy else batch_result.success_rate,
        "num_trajectories": len(batch_result.trajectory_results),
        "num_successful": sum(1 for r in batch_result.trajectory_results if r.success),
    }

    return EvaluationResult(
        metadata=metadata,
        metrics=metrics,
        results=results,
        summary=summary,
    )


def print_summary(result: EvaluationResult) -> None:
    """Print a human-readable summary of evaluation results.

    Args:
        result: The evaluation result.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results: {result.metadata.environment}")
    print(f"{'=' * 60}")
    print(f"Model: {result.metadata.model}")
    print(f"Timestamp: {result.metadata.timestamp}")
    print(f"Duration: {result.metadata.duration_seconds:.1f}s")
    print(f"Trajectories: {result.metadata.num_trajectories}")
    print()
    print("Metrics:")
    print("-" * 40)

    for name, metric in result.metrics.metrics.items():
        stats = metric.statistics
        parts = [f"  {name}: {stats.mean:.4f}"]
        if isinstance(stats, ContinuousStatistics):
            if stats.std_dev is not None:
                parts.append(f" (std: {stats.std_dev:.4f})")
        elif isinstance(stats, BinaryStatistics):
            parts.append(f" ({stats.count}/{stats.n})")
        if stats.std_error is not None:
            parts.append(f" +/- {stats.std_error:.4f}")
        print("".join(parts))

    print()
    print(
        f"Summary: {result.summary['num_successful']}/{result.summary['num_trajectories']} successful"
    )
    print(f"{'=' * 60}\n")
