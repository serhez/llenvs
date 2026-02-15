"""Metric computation with summary statistics.

This module separates metrics (what we measure) from statistics (how we summarize).

- **Metrics**: Measurable quantities like action_reward, trajectory_reward, accuracy
- **Statistics**: Summary computations like mean, std_dev, quantiles, confidence intervals

Two statistics classes exist for different metric types:
- **ContinuousStatistics**: For continuous-valued metrics (rewards, etc.)
- **BinaryStatistics**: For binary (success/failure) metrics with pass_at_k support
"""

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from llenvs.evaluation.runner import TrajectoryResult, BatchResult


@dataclass(frozen=True)
class ContinuousStatistics:
    """Summary statistics for continuous-valued metrics.

    All statistics are computed from a collection of values. This class
    provides a comprehensive summary including central tendency, dispersion,
    and distribution measures.

    Attributes:
        n: Sample size.
        mean: Arithmetic mean.
        std_dev: Sample standard deviation (None if n < 2).
        std_error: Standard error of the mean (None if n < 2).
        min: Minimum value.
        max: Maximum value.
        median: Median (50th percentile).
        q25: 25th percentile.
        q75: 75th percentile.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
    """

    n: int
    mean: float
    std_dev: float | None = None
    std_error: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass(frozen=True)
class BinaryStatistics:
    """Summary statistics for binary (success/failure) metrics.

    Optimized for binary data where values are either 0 or 1. Includes
    Wilson score confidence intervals and pass_at_k computation.

    Attributes:
        n: Sample size.
        mean: Success rate (proportion of successes).
        count: Number of successes.
        std_error: Standard error sqrt(p(1-p)/n).
        ci_lower: Lower bound of Wilson score confidence interval.
        ci_upper: Upper bound of Wilson score confidence interval.
    """

    n: int
    mean: float
    count: int
    std_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None

    def pass_at_k(self, k: int) -> float:
        """Compute probability of at least one success in k samples.

        Uses the exact formula: 1 - C(n-c, k) / C(n, k)
        where n is total samples and c is number of successes.

        Args:
            k: Number of samples to consider.

        Returns:
            Probability of at least one success in k samples.

        Raises:
            ValueError: If k < 1 or k > n.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > self.n:
            raise ValueError(f"k ({k}) cannot exceed n ({self.n})")
        if self.count == self.n:
            return 1.0
        if self.count == 0:
            return 0.0
        return 1.0 - _comb(self.n - self.count, k) / _comb(self.n, k)


@dataclass(frozen=True)
class Metric:
    """A named metric with computed statistics.

    A metric represents a measurable quantity (e.g., action_reward,
    trajectory_reward, format_compliance). Statistics are computed
    from the samples of this metric.

    Attributes:
        name: Metric name (e.g., "action_reward", "accuracy").
        statistics: Summary statistics computed from samples.
    """

    name: str
    statistics: ContinuousStatistics | BinaryStatistics


@dataclass
class MetricsBundle:
    """Collection of computed metrics.

    Attributes:
        metrics: Dictionary of metric name to Metric.
    """

    metrics: dict[str, Metric] = field(default_factory=dict)

    def add(self, metric: Metric) -> None:
        """Add a metric to the bundle."""
        self.metrics[metric.name] = metric

    def get(self, name: str) -> Metric | None:
        """Get a metric by name."""
        return self.metrics.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result = {}
        for name, metric in self.metrics.items():
            stats = metric.statistics
            if isinstance(stats, ContinuousStatistics):
                result[name] = {
                    "n": stats.n,
                    "mean": stats.mean,
                    "std_dev": stats.std_dev,
                    "std_error": stats.std_error,
                    "min": stats.min,
                    "max": stats.max,
                    "median": stats.median,
                    "q25": stats.q25,
                    "q75": stats.q75,
                    "ci_lower": stats.ci_lower,
                    "ci_upper": stats.ci_upper,
                }
            else:  # BinaryStatistics
                result[name] = {
                    "n": stats.n,
                    "mean": stats.mean,
                    "count": stats.count,
                    "std_error": stats.std_error,
                    "ci_lower": stats.ci_lower,
                    "ci_upper": stats.ci_upper,
                }
        return result


def compute_continuous_statistics(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> ContinuousStatistics:
    """Compute summary statistics from a sequence of continuous values.

    Args:
        values: The values to compute statistics for.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        ContinuousStatistics object with computed measures.
    """
    n = len(values)

    if n == 0:
        return ContinuousStatistics(n=0, mean=0.0)

    # Convert to list for indexing
    sorted_values = sorted(values)
    mean = sum(values) / n

    # Min, max, median
    min_val = sorted_values[0]
    max_val = sorted_values[-1]
    median = _percentile(sorted_values, 0.5)

    # Quantiles
    q25 = _percentile(sorted_values, 0.25)
    q75 = _percentile(sorted_values, 0.75)

    # Standard deviation and error
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)
    else:
        std_dev = 0.0
        std_error = 0.0

    # Confidence interval
    ci_lower = None
    ci_upper = None
    if n > 1 and std_error > 0:
        z = _z_score(confidence_level)
        ci_lower = mean - z * std_error
        ci_upper = mean + z * std_error

    return ContinuousStatistics(
        n=n,
        mean=mean,
        std_dev=std_dev,
        std_error=std_error,
        min=min_val,
        max=max_val,
        median=median,
        q25=q25,
        q75=q75,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compute_binary_statistics(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> BinaryStatistics:
    """Compute summary statistics from a sequence of binary values.

    Binary values should be 0.0 or 1.0 (or truthy/falsy convertible to such).
    Uses Wilson score interval for confidence bounds.

    Args:
        values: The binary values (0.0 or 1.0) to compute statistics for.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        BinaryStatistics object with computed measures.
    """
    n = len(values)

    if n == 0:
        return BinaryStatistics(n=0, mean=0.0, count=0)

    count = sum(1 for v in values if v)
    mean = count / n

    # Standard error for binary variable: sqrt(p(1-p)/n)
    if n > 1:
        std_error = math.sqrt(mean * (1 - mean) / n)
    else:
        std_error = 0.0

    # Wilson score confidence interval (better for proportions)
    z = _z_score(confidence_level)
    denominator = 1 + z * z / n
    center = (mean + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((mean * (1 - mean) + z * z / (4 * n)) / n) / denominator

    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)

    return BinaryStatistics(
        n=n,
        mean=mean,
        count=count,
        std_error=std_error if n > 1 else None,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compute_action_reward(results: list[TrajectoryResult]) -> Metric:
    """Compute action reward statistics (action-level metric).

    Collects the reward for each individual action across all trajectories
    and computes summary statistics.

    Args:
        results: List of trajectory results.

    Returns:
        Metric with action reward statistics.
    """
    values = []
    for result in results:
        for transition in result.trajectory.transitions:
            values.append(transition.rewards.total)

    return Metric(name="action_reward", statistics=compute_continuous_statistics(values))


def compute_trajectory_reward(results: list[TrajectoryResult]) -> Metric:
    """Compute trajectory reward statistics (trajectory-level metric).

    Collects the total reward for each trajectory and computes summary statistics.

    Args:
        results: List of trajectory results.

    Returns:
        Metric with trajectory reward statistics.
    """
    values = [r.total_reward for r in results]
    return Metric(name="trajectory_reward", statistics=compute_continuous_statistics(values))


def compute_accuracy(
    results: list[TrajectoryResult],
    confidence_level: float = 0.95,
) -> Metric:
    """Compute accuracy statistics (trajectory-level binary metric).

    Accuracy is the fraction of successful trajectories. Each trajectory
    contributes a binary value (1 for success, 0 for failure).

    Uses Wilson score interval for confidence bounds on proportions.

    Args:
        results: List of trajectory results.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        Metric with accuracy statistics (BinaryStatistics).
    """
    values = [1.0 if r.success else 0.0 for r in results]
    return Metric(
        name="accuracy",
        statistics=compute_binary_statistics(values, confidence_level),
    )


def compute_format_compliance(
    results: list[TrajectoryResult],
    confidence_level: float = 0.95,
) -> Metric:
    """Compute format compliance statistics (action-level binary metric).

    Counts format compliance across ALL actions in all trajectories.
    Each action contributes a binary value (1 if compliant, 0 otherwise).

    Args:
        results: List of trajectory results.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        Metric with format compliance statistics (BinaryStatistics).
    """
    values = []
    for result in results:
        for transition in result.trajectory.transitions:
            format_reward = transition.rewards.by_name("format")
            compliant = 1.0 if (format_reward and format_reward.reward >= 1.0) else 0.0
            values.append(compliant)

    return Metric(
        name="format_compliance",
        statistics=compute_binary_statistics(values, confidence_level),
    )


def compute_all_metrics(batch_result: BatchResult) -> MetricsBundle:
    """Compute all standard metrics from a batch result.

    Args:
        batch_result: The batch evaluation result.

    Returns:
        MetricsBundle with all computed metrics.
    """
    bundle = MetricsBundle()
    results = batch_result.trajectory_results

    # Trajectory-level metrics
    bundle.add(compute_accuracy(results))
    bundle.add(compute_trajectory_reward(results))

    # Action-level metrics
    bundle.add(compute_action_reward(results))
    bundle.add(compute_format_compliance(results))

    return bundle


def _z_score(confidence_level: float) -> float:
    """Get z-score for a given confidence level."""
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    return z_scores.get(confidence_level, 1.96)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation.

    Args:
        sorted_values: Pre-sorted list of values.
        p: Percentile as fraction (0.0 to 1.0).

    Returns:
        Interpolated percentile value.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Linear interpolation
    idx = p * (n - 1)
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = idx - lower_idx

    return sorted_values[lower_idx] + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])


def _comb(n: int, k: int) -> float:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)

    return result


def aggregate_continuous_metrics(
    metrics: Sequence[Metric],
    name: str | None = None,
) -> Metric:
    """Aggregate multiple continuous metrics into one.

    Combines statistics from multiple metrics using appropriate formulas:
    - n: sum of all sample sizes
    - mean: weighted mean
    - std_dev: pooled standard deviation
    - min/max: min/max of all values
    - median, q25, q75: None (cannot reconstruct from summaries)
    - ci: recomputed from aggregated statistics

    Args:
        metrics: Sequence of Metric objects with ContinuousStatistics.
        name: Name for the aggregated metric. Defaults to first metric's name.

    Returns:
        New Metric with aggregated ContinuousStatistics.

    Raises:
        ValueError: If metrics is empty.
        TypeError: If any metric has non-continuous statistics.
    """
    if not metrics:
        raise ValueError("Cannot aggregate empty sequence of metrics")

    for m in metrics:
        if not isinstance(m.statistics, ContinuousStatistics):
            raise TypeError(
                f"Expected ContinuousStatistics, got {type(m.statistics).__name__}"
            )

    stats_list = [m.statistics for m in metrics]
    result_name = name if name is not None else metrics[0].name

    # Aggregate n
    n_total = sum(s.n for s in stats_list)

    if n_total == 0:
        return Metric(name=result_name, statistics=ContinuousStatistics(n=0, mean=0.0))

    # Weighted mean
    mean_total = sum(s.n * s.mean for s in stats_list) / n_total

    # Aggregate min/max
    mins = [s.min for s in stats_list if s.min is not None]
    maxs = [s.max for s in stats_list if s.max is not None]
    min_total = min(mins) if mins else None
    max_total = max(maxs) if maxs else None

    # Pooled variance
    std_dev_total = None
    std_error_total = None
    ci_lower = None
    ci_upper = None

    if n_total > 1:
        # Check if we have std_dev for all non-trivial stats
        can_pool = all(s.std_dev is not None for s in stats_list if s.n > 1)

        if can_pool:
            ss_within = sum(
                (s.n - 1) * s.std_dev**2
                for s in stats_list
                if s.n > 1 and s.std_dev is not None
            )
            ss_between = sum(s.n * (s.mean - mean_total) ** 2 for s in stats_list)
            var_pooled = (ss_within + ss_between) / (n_total - 1)
            std_dev_total = math.sqrt(var_pooled)
            std_error_total = std_dev_total / math.sqrt(n_total)

            # Confidence interval
            z = _z_score(0.95)
            ci_lower = mean_total - z * std_error_total
            ci_upper = mean_total + z * std_error_total

    return Metric(
        name=result_name,
        statistics=ContinuousStatistics(
            n=n_total,
            mean=mean_total,
            std_dev=std_dev_total,
            std_error=std_error_total,
            min=min_total,
            max=max_total,
            median=None,
            q25=None,
            q75=None,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ),
    )


def aggregate_binary_metrics(
    metrics: Sequence[Metric],
    name: str | None = None,
    confidence_level: float = 0.95,
) -> Metric:
    """Aggregate multiple binary metrics into one.

    Combines statistics from multiple metrics:
    - n: sum of all sample sizes
    - count: sum of all success counts
    - mean: total successes / total samples
    - std_error, ci: recomputed from aggregated statistics

    Args:
        metrics: Sequence of Metric objects with BinaryStatistics.
        name: Name for the aggregated metric. Defaults to first metric's name.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        New Metric with aggregated BinaryStatistics.

    Raises:
        ValueError: If metrics is empty.
        TypeError: If any metric has non-binary statistics.
    """
    if not metrics:
        raise ValueError("Cannot aggregate empty sequence of metrics")

    for m in metrics:
        if not isinstance(m.statistics, BinaryStatistics):
            raise TypeError(
                f"Expected BinaryStatistics, got {type(m.statistics).__name__}"
            )

    stats_list = [m.statistics for m in metrics]
    result_name = name if name is not None else metrics[0].name

    n_total = sum(s.n for s in stats_list)
    count_total = sum(s.count for s in stats_list)

    if n_total == 0:
        return Metric(
            name=result_name, statistics=BinaryStatistics(n=0, mean=0.0, count=0)
        )

    mean_total = count_total / n_total

    # Standard error
    std_error = (
        math.sqrt(mean_total * (1 - mean_total) / n_total) if n_total > 1 else None
    )

    # Wilson score CI
    z = _z_score(confidence_level)
    denominator = 1 + z * z / n_total
    center = (mean_total + z * z / (2 * n_total)) / denominator
    spread = (
        z
        * math.sqrt((mean_total * (1 - mean_total) + z * z / (4 * n_total)) / n_total)
        / denominator
    )
    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)

    return Metric(
        name=result_name,
        statistics=BinaryStatistics(
            n=n_total,
            mean=mean_total,
            count=count_total,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ),
    )
