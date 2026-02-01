"""Metric aggregation functions.

Computes accuracy, Pass@k, and other evaluation metrics
from episode results.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from llenvs.evaluation.runner import EpisodeResult, BatchResult


@dataclass(frozen=True)
class MetricValue:
    """A computed metric with optional confidence interval.

    Attributes:
        name: Metric name.
        value: Metric value.
        std_error: Standard error (optional).
        ci_lower: Lower confidence bound (optional).
        ci_upper: Upper confidence bound (optional).
        n: Sample size.
    """

    name: str
    value: float
    std_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n: int = 0


@dataclass
class MetricsBundle:
    """Collection of computed metrics.

    Attributes:
        metrics: Dictionary of metric name to MetricValue.
    """

    metrics: dict[str, MetricValue] = field(default_factory=dict)

    def add(self, metric: MetricValue) -> None:
        """Add a metric to the bundle."""
        self.metrics[metric.name] = metric

    def get(self, name: str) -> MetricValue | None:
        """Get a metric by name."""
        return self.metrics.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        for name, metric in self.metrics.items():
            result[name] = {
                "value": metric.value,
                "std_error": metric.std_error,
                "ci_lower": metric.ci_lower,
                "ci_upper": metric.ci_upper,
                "n": metric.n,
            }
        return result


def compute_accuracy(
    results: list[EpisodeResult],
    confidence_level: float = 0.95,
) -> MetricValue:
    """Compute accuracy with confidence interval.

    Args:
        results: List of episode results.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        MetricValue with accuracy and confidence bounds.
    """
    if not results:
        return MetricValue(name="accuracy", value=0.0, n=0)

    n = len(results)
    successes = sum(1 for r in results if r.success)
    accuracy = successes / n

    # Wilson score confidence interval for binomial proportion
    z = _z_score(confidence_level)
    denominator = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((accuracy * (1 - accuracy) + z * z / (4 * n)) / n) / denominator

    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)

    # Standard error
    std_error = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0.0

    return MetricValue(
        name="accuracy",
        value=accuracy,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
    )


def compute_mean_reward(results: list[EpisodeResult]) -> MetricValue:
    """Compute mean total reward across episodes.

    Args:
        results: List of episode results.

    Returns:
        MetricValue with mean reward and standard error.
    """
    if not results:
        return MetricValue(name="mean_reward", value=0.0, n=0)

    rewards = [r.total_reward for r in results]
    n = len(rewards)
    mean = sum(rewards) / n

    # Standard deviation and error
    if n > 1:
        variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)
    else:
        std_error = 0.0

    return MetricValue(
        name="mean_reward",
        value=mean,
        std_error=std_error,
        n=n,
    )


def compute_pass_at_k(
    results_by_task: dict[int, list[EpisodeResult]],
    k: int = 1,
) -> MetricValue:
    """Compute Pass@k metric.

    Pass@k measures the probability that at least one of k samples
    for a problem is correct.

    Args:
        results_by_task: Dictionary mapping task index to list of results.
        k: Number of samples to consider.

    Returns:
        MetricValue with Pass@k.
    """
    if not results_by_task:
        return MetricValue(name=f"pass@{k}", value=0.0, n=0)

    pass_at_k_values = []

    for task_idx, task_results in results_by_task.items():
        n = len(task_results)
        c = sum(1 for r in task_results if r.success)

        if n < k:
            # Not enough samples, use what we have
            pass_at_k_values.append(1.0 if c > 0 else 0.0)
        else:
            # Exact Pass@k calculation
            # P(at least one correct in k samples) = 1 - P(all k wrong)
            # = 1 - C(n-c, k) / C(n, k)
            if c == n:
                pass_at_k_values.append(1.0)
            elif c == 0:
                pass_at_k_values.append(0.0)
            else:
                pass_at_k_values.append(1.0 - _comb(n - c, k) / _comb(n, k))

    if not pass_at_k_values:
        return MetricValue(name=f"pass@{k}", value=0.0, n=0)

    mean_pass_at_k = sum(pass_at_k_values) / len(pass_at_k_values)

    return MetricValue(
        name=f"pass@{k}",
        value=mean_pass_at_k,
        n=len(results_by_task),
    )


def compute_format_compliance(results: list[EpisodeResult]) -> MetricValue:
    """Compute fraction of episodes where answer was properly formatted.

    Args:
        results: List of episode results.

    Returns:
        MetricValue with format compliance rate.
    """
    if not results:
        return MetricValue(name="format_compliance", value=0.0, n=0)

    compliant = 0
    for result in results:
        if result.trajectory.transitions:
            last_rewards = result.trajectory.transitions[-1].rewards
            format_reward = last_rewards.by_name("format")
            if format_reward and format_reward.value >= 1.0:
                compliant += 1

    n = len(results)
    rate = compliant / n

    return MetricValue(
        name="format_compliance",
        value=rate,
        n=n,
    )


def compute_all_metrics(
    batch_result: BatchResult,
    results_by_task: dict[int, list[EpisodeResult]] | None = None,
    k_values: list[int] | None = None,
) -> MetricsBundle:
    """Compute all standard metrics from a batch result.

    Args:
        batch_result: The batch evaluation result.
        results_by_task: Optional grouping for Pass@k.
        k_values: List of k values for Pass@k (default [1, 5, 10]).

    Returns:
        MetricsBundle with all computed metrics.
    """
    bundle = MetricsBundle()
    results = batch_result.episode_results

    # Basic metrics
    bundle.add(compute_accuracy(results))
    bundle.add(compute_mean_reward(results))
    bundle.add(compute_format_compliance(results))

    # Pass@k if we have grouped results
    if results_by_task:
        k_values = k_values or [1, 5, 10]
        for k in k_values:
            bundle.add(compute_pass_at_k(results_by_task, k=k))

    return bundle


def _z_score(confidence_level: float) -> float:
    """Get z-score for a given confidence level."""
    # Common values
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    return z_scores.get(confidence_level, 1.96)


def _comb(n: int, k: int) -> float:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    # Use smaller k for efficiency
    k = min(k, n - k)

    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)

    return result
