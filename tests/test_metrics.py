"""Tests for metric computation."""

import pytest
from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.core.reward import SignalBundle, Signal, RewardType
from llenvs.evaluation.runner import TrajectoryResult, BatchResult
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


def make_state(
    step: int = 0,
    trajectory_id: str = "traj_0",
    is_terminal: bool = False,
) -> State[dict]:
    """Helper to create states for testing."""
    return State(
        observation=Observation(prompt="test"),
        hidden={},
        metadata=StateMetadata(
            step=step,
            episode_id=trajectory_id,
            is_terminal=is_terminal,
        ),
    )


def make_transition(
    correctness_reward: float = 0.0,
    format_reward: float = 0.0,
    step: int = 0,
    trajectory_id: str = "traj_0",
) -> Transition:
    """Helper to create transitions for testing."""
    state = make_state(step=step, trajectory_id=trajectory_id)
    next_state = make_state(step=step + 1, trajectory_id=trajectory_id, is_terminal=True)

    return Transition(
        state=state,
        action=Action(text="test response"),
        next_state=next_state,
        rewards=SignalBundle(
            signals=(
                Signal(
                    name="correctness",
                    reward_type=RewardType.OUTCOME,
                    reward=correctness_reward,
                ),
                Signal(
                    name="format",
                    reward_type=RewardType.FORMAT,
                    reward=format_reward,
                ),
            )
        ),
        info={},
    )


def make_trajectory_result(
    success: bool = False,
    total_reward: float | None = None,
    transitions: list[Transition] | None = None,
    task_index: int = 0,
) -> TrajectoryResult:
    """Helper to create trajectory results for testing."""
    trajectory_id = f"traj_{task_index}"
    state = make_state(trajectory_id=trajectory_id)
    trajectory = Trajectory.create(state)

    if transitions is None:
        correctness_value = 1.0 if success else 0.0
        format_value = 1.0
        transition = make_transition(
            correctness_reward=correctness_value,
            format_reward=format_value,
            trajectory_id=trajectory_id,
        )
        transitions = [transition]

    for t in transitions:
        trajectory.add_transition(t)

    if total_reward is None:
        total_reward = sum(t.rewards.total for t in transitions)

    return TrajectoryResult(
        trajectory=trajectory,
        total_reward=total_reward,
        success=success,
        metadata={"task_index": task_index},
    )


class TestContinuousStatistics:
    """Tests for ContinuousStatistics and compute_continuous_statistics."""

    def test_compute_basic(self):
        """Test basic statistics computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_continuous_statistics(values)

        assert stats.n == 5
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_compute_std_dev(self):
        """Test standard deviation computation."""
        # [10, 20, 30] -> mean=20, sample std_dev=10
        values = [10.0, 20.0, 30.0]
        stats = compute_continuous_statistics(values)

        assert stats.mean == 20.0
        assert stats.std_dev == pytest.approx(10.0)

    def test_compute_std_error(self):
        """Test standard error computation."""
        values = [10.0, 20.0, 30.0]
        stats = compute_continuous_statistics(values)

        # std_error = std_dev / sqrt(n) = 10 / sqrt(3)
        expected_std_error = 10.0 / (3**0.5)
        assert stats.std_error == pytest.approx(expected_std_error)

    def test_compute_quantiles(self):
        """Test quantile computation."""
        values = list(range(1, 101))  # 1 to 100
        stats = compute_continuous_statistics(values)

        assert stats.q25 is not None
        assert stats.q75 is not None
        # For 1-100, q25 ≈ 25.75, q75 ≈ 75.25 (linear interpolation)
        assert 25 <= stats.q25 <= 26
        assert 75 <= stats.q75 <= 76

    def test_compute_confidence_interval(self):
        """Test confidence interval computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_continuous_statistics(values, confidence_level=0.95)

        assert stats.ci_lower is not None
        assert stats.ci_upper is not None
        assert stats.ci_lower < stats.mean < stats.ci_upper

    def test_empty_values(self):
        """Test with empty values."""
        stats = compute_continuous_statistics([])

        assert stats.n == 0
        assert stats.mean == 0.0
        assert stats.std_dev is None
        assert stats.min is None
        assert stats.max is None

    def test_single_value(self):
        """Test with single value."""
        stats = compute_continuous_statistics([5.0])

        assert stats.n == 1
        assert stats.mean == 5.0
        assert stats.std_dev == 0.0
        assert stats.min == 5.0
        assert stats.max == 5.0
        assert stats.median == 5.0

    def test_statistics_immutable(self):
        """Test that ContinuousStatistics is immutable."""
        stats = compute_continuous_statistics([1.0, 2.0, 3.0])

        with pytest.raises(AttributeError):
            stats.mean = 10.0


class TestBinaryStatistics:
    """Tests for BinaryStatistics and compute_binary_statistics."""

    def test_compute_basic(self):
        """Test basic binary statistics computation."""
        values = [1.0, 1.0, 0.0, 1.0, 0.0]
        stats = compute_binary_statistics(values)

        assert stats.n == 5
        assert stats.mean == 0.6
        assert stats.count == 3

    def test_all_successes(self):
        """Test with all successes."""
        values = [1.0, 1.0, 1.0]
        stats = compute_binary_statistics(values)

        assert stats.mean == 1.0
        assert stats.count == 3
        assert stats.n == 3

    def test_no_successes(self):
        """Test with no successes."""
        values = [0.0, 0.0, 0.0]
        stats = compute_binary_statistics(values)

        assert stats.mean == 0.0
        assert stats.count == 0

    def test_empty_values(self):
        """Test with empty values."""
        stats = compute_binary_statistics([])

        assert stats.n == 0
        assert stats.mean == 0.0
        assert stats.count == 0

    def test_confidence_interval(self):
        """Test Wilson score confidence interval."""
        values = [1.0] * 8 + [0.0] * 2  # 80% success rate
        stats = compute_binary_statistics(values)

        assert stats.ci_lower is not None
        assert stats.ci_upper is not None
        assert 0.0 <= stats.ci_lower <= stats.mean
        assert stats.mean <= stats.ci_upper <= 1.0

    def test_std_error(self):
        """Test standard error for binary data."""
        values = [1.0, 1.0, 0.0, 0.0]  # p = 0.5
        stats = compute_binary_statistics(values)

        # sqrt(0.5 * 0.5 / 4) = 0.25
        assert stats.std_error == pytest.approx(0.25)

    def test_immutable(self):
        """Test that BinaryStatistics is immutable."""
        stats = compute_binary_statistics([1.0, 0.0])

        with pytest.raises(AttributeError):
            stats.mean = 0.5

    def test_pass_at_k_all_correct(self):
        """Test pass_at_k when all attempts are successful."""
        stats = BinaryStatistics(n=5, mean=1.0, count=5)

        assert stats.pass_at_k(1) == 1.0
        assert stats.pass_at_k(5) == 1.0

    def test_pass_at_k_none_correct(self):
        """Test pass_at_k when no attempts are successful."""
        stats = BinaryStatistics(n=5, mean=0.0, count=0)

        assert stats.pass_at_k(1) == 0.0
        assert stats.pass_at_k(5) == 0.0

    def test_pass_at_k_partial(self):
        """Test pass_at_k with partial success rate."""
        # 1 success out of 5 attempts
        stats = BinaryStatistics(n=5, mean=0.2, count=1)

        # P(at least 1 correct in k=1) = 1/5 = 0.2
        assert stats.pass_at_k(1) == pytest.approx(0.2)

        # P(at least 1 correct in k=5) = 1 - C(4,5)/C(5,5) = 1 - 0 = 1.0
        assert stats.pass_at_k(5) == 1.0

    def test_pass_at_k_increases_with_k(self):
        """Test that pass_at_k increases with k."""
        # 2 successes out of 10 attempts
        stats = BinaryStatistics(n=10, mean=0.2, count=2)

        pass_1 = stats.pass_at_k(1)
        pass_3 = stats.pass_at_k(3)
        pass_5 = stats.pass_at_k(5)

        assert pass_1 < pass_3 < pass_5

    def test_pass_at_k_invalid_k_too_small(self):
        """Test pass_at_k raises for k < 1."""
        stats = BinaryStatistics(n=5, mean=0.5, count=2)

        with pytest.raises(ValueError, match="k must be >= 1"):
            stats.pass_at_k(0)

        with pytest.raises(ValueError, match="k must be >= 1"):
            stats.pass_at_k(-1)

    def test_pass_at_k_invalid_k_too_large(self):
        """Test pass_at_k raises for k > n."""
        stats = BinaryStatistics(n=5, mean=0.5, count=2)

        with pytest.raises(ValueError, match="cannot exceed n"):
            stats.pass_at_k(6)


class TestMetric:
    """Tests for Metric class."""

    def test_creation_with_continuous(self):
        """Test metric creation with continuous statistics."""
        stats = compute_continuous_statistics([1.0, 2.0, 3.0])
        metric = Metric(name="action_reward", statistics=stats)

        assert metric.name == "action_reward"
        assert metric.statistics.mean == 2.0
        assert metric.statistics.n == 3

    def test_creation_with_binary(self):
        """Test metric creation with binary statistics."""
        stats = compute_binary_statistics([1.0, 0.0, 1.0])
        metric = Metric(name="accuracy", statistics=stats)

        assert metric.name == "accuracy"
        assert metric.statistics.mean == pytest.approx(2 / 3)
        assert isinstance(metric.statistics, BinaryStatistics)
        assert metric.statistics.count == 2

    def test_immutable(self):
        """Test that Metric is immutable."""
        stats = compute_continuous_statistics([1.0, 2.0, 3.0])
        metric = Metric(name="test", statistics=stats)

        with pytest.raises(AttributeError):
            metric.name = "changed"


class TestMetricsBundle:
    """Tests for MetricsBundle."""

    def test_add_and_get(self):
        """Test adding and retrieving metrics."""
        bundle = MetricsBundle()
        stats = compute_continuous_statistics([0.8, 0.9, 1.0])
        metric = Metric(name="reward", statistics=stats)

        bundle.add(metric)
        retrieved = bundle.get("reward")

        assert retrieved is not None
        assert retrieved.statistics.mean == pytest.approx(0.9)

    def test_get_nonexistent(self):
        """Test getting non-existent metric."""
        bundle = MetricsBundle()
        assert bundle.get("nonexistent") is None

    def test_to_dict_continuous(self):
        """Test conversion to dictionary for continuous statistics."""
        bundle = MetricsBundle()
        stats = compute_continuous_statistics([0.8, 0.9, 1.0])
        bundle.add(Metric(name="reward", statistics=stats))

        result = bundle.to_dict()

        assert "reward" in result
        assert result["reward"]["mean"] == pytest.approx(0.9)
        assert result["reward"]["n"] == 3
        assert "std_dev" in result["reward"]
        assert "min" in result["reward"]
        assert "max" in result["reward"]
        assert "count" not in result["reward"]

    def test_to_dict_binary(self):
        """Test conversion to dictionary for binary statistics."""
        bundle = MetricsBundle()
        stats = compute_binary_statistics([1.0, 1.0, 0.0])
        bundle.add(Metric(name="accuracy", statistics=stats))

        result = bundle.to_dict()

        assert "accuracy" in result
        assert result["accuracy"]["mean"] == pytest.approx(2 / 3)
        assert result["accuracy"]["n"] == 3
        assert result["accuracy"]["count"] == 2
        assert "std_dev" not in result["accuracy"]
        assert "min" not in result["accuracy"]


class TestComputeActionReward:
    """Tests for compute_action_reward (action-level metric)."""

    def test_single_action_trajectories(self):
        """Test with single-action trajectories."""
        t1 = make_transition(correctness_reward=1.0, format_reward=0.0)  # total=1.0
        t2 = make_transition(correctness_reward=0.5, format_reward=0.5)  # total=1.0
        t3 = make_transition(correctness_reward=0.0, format_reward=0.0)  # total=0.0

        results = [
            make_trajectory_result(transitions=[t1]),
            make_trajectory_result(transitions=[t2]),
            make_trajectory_result(transitions=[t3]),
        ]
        metric = compute_action_reward(results)

        assert metric.name == "action_reward"
        assert metric.statistics.mean == pytest.approx(2.0 / 3.0)
        assert metric.statistics.n == 3
        assert isinstance(metric.statistics, ContinuousStatistics)

    def test_multi_action_trajectories(self):
        """Test with multi-action trajectories."""
        # Trajectory 1: 2 actions with rewards [2.0, 4.0]
        t1_a1 = make_transition(correctness_reward=2.0, format_reward=0.0)
        t1_a2 = make_transition(correctness_reward=4.0, format_reward=0.0, step=1)

        # Trajectory 2: 1 action with reward [3.0]
        t2_a1 = make_transition(correctness_reward=3.0, format_reward=0.0)

        results = [
            make_trajectory_result(transitions=[t1_a1, t1_a2]),
            make_trajectory_result(transitions=[t2_a1]),
        ]
        metric = compute_action_reward(results)

        # Mean of [2.0, 4.0, 3.0] = 3.0
        assert metric.statistics.mean == pytest.approx(3.0)
        assert metric.statistics.n == 3

    def test_statistics_computed(self):
        """Test that full statistics are computed."""
        t1 = make_transition(correctness_reward=1.0, format_reward=0.0)
        t2 = make_transition(correctness_reward=2.0, format_reward=0.0, step=1)
        t3 = make_transition(correctness_reward=3.0, format_reward=0.0, step=2)

        results = [make_trajectory_result(transitions=[t1, t2, t3])]
        metric = compute_action_reward(results)

        stats = metric.statistics
        assert stats.mean == pytest.approx(2.0)
        assert stats.std_dev == pytest.approx(1.0)
        assert stats.min == 1.0
        assert stats.max == 3.0
        assert stats.median == 2.0

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_action_reward([])

        assert metric.statistics.n == 0
        assert metric.statistics.mean == 0.0


class TestComputeTrajectoryReward:
    """Tests for compute_trajectory_reward (trajectory-level metric)."""

    def test_basic(self):
        """Test basic trajectory reward computation."""
        results = [
            make_trajectory_result(success=True, total_reward=1.0),
            make_trajectory_result(success=True, total_reward=2.0),
            make_trajectory_result(success=False, total_reward=0.0),
        ]
        metric = compute_trajectory_reward(results)

        assert metric.name == "trajectory_reward"
        assert metric.statistics.mean == pytest.approx(1.0)
        assert metric.statistics.n == 3
        assert isinstance(metric.statistics, ContinuousStatistics)

    def test_statistics_computed(self):
        """Test that full statistics are computed."""
        # Rewards: [10, 20, 30] -> mean=20, std_dev=10
        results = [
            make_trajectory_result(success=True, total_reward=10.0),
            make_trajectory_result(success=True, total_reward=20.0),
            make_trajectory_result(success=True, total_reward=30.0),
        ]
        metric = compute_trajectory_reward(results)

        stats = metric.statistics
        assert stats.mean == pytest.approx(20.0)
        assert stats.std_dev == pytest.approx(10.0)
        assert stats.min == 10.0
        assert stats.max == 30.0

    def test_single_result(self):
        """Test with single result."""
        results = [make_trajectory_result(success=True, total_reward=5.0)]
        metric = compute_trajectory_reward(results)

        assert metric.statistics.mean == 5.0
        assert metric.statistics.std_dev == 0.0

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_trajectory_reward([])

        assert metric.statistics.n == 0
        assert metric.statistics.mean == 0.0


class TestComputeAccuracy:
    """Tests for compute_accuracy (trajectory-level binary metric)."""

    def test_all_correct(self):
        """Test accuracy with all successful trajectories."""
        results = [make_trajectory_result(success=True) for _ in range(10)]
        metric = compute_accuracy(results)

        assert metric.name == "accuracy"
        assert metric.statistics.mean == 1.0
        assert metric.statistics.n == 10
        assert isinstance(metric.statistics, BinaryStatistics)
        assert metric.statistics.count == 10

    def test_all_incorrect(self):
        """Test accuracy with all failed trajectories."""
        results = [make_trajectory_result(success=False) for _ in range(10)]
        metric = compute_accuracy(results)

        assert metric.statistics.mean == 0.0
        assert metric.statistics.count == 0

    def test_mixed(self):
        """Test accuracy with mixed results."""
        results = [
            make_trajectory_result(success=True),
            make_trajectory_result(success=True),
            make_trajectory_result(success=False),
            make_trajectory_result(success=False),
        ]
        metric = compute_accuracy(results)

        assert metric.statistics.mean == 0.5
        assert metric.statistics.n == 4
        assert metric.statistics.count == 2

    def test_confidence_interval(self):
        """Test that confidence interval is computed."""
        results = [make_trajectory_result(success=(i < 8)) for i in range(10)]
        metric = compute_accuracy(results)

        stats = metric.statistics
        assert stats.ci_lower is not None
        assert stats.ci_upper is not None
        assert stats.ci_lower <= stats.mean <= stats.ci_upper

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_accuracy([])

        assert metric.statistics.mean == 0.0
        assert metric.statistics.n == 0
        assert metric.statistics.count == 0

    def test_pass_at_k_via_statistics(self):
        """Test that pass_at_k is available via BinaryStatistics."""
        results = [make_trajectory_result(success=(i < 2)) for i in range(10)]
        metric = compute_accuracy(results)

        # 2 successes out of 10
        assert metric.statistics.pass_at_k(1) == pytest.approx(0.2)
        assert metric.statistics.pass_at_k(10) == 1.0


class TestComputeFormatCompliance:
    """Tests for compute_format_compliance (action-level binary metric)."""

    def test_all_compliant(self):
        """Test with all format-compliant actions."""
        t1 = make_transition(format_reward=1.0)
        t2 = make_transition(format_reward=1.0)

        results = [
            make_trajectory_result(transitions=[t1]),
            make_trajectory_result(transitions=[t2]),
        ]
        metric = compute_format_compliance(results)

        assert metric.name == "format_compliance"
        assert metric.statistics.mean == 1.0
        assert metric.statistics.n == 2
        assert isinstance(metric.statistics, BinaryStatistics)
        assert metric.statistics.count == 2

    def test_none_compliant(self):
        """Test with no format-compliant actions."""
        t1 = make_transition(format_reward=0.0)
        t2 = make_transition(format_reward=0.0)

        results = [
            make_trajectory_result(transitions=[t1]),
            make_trajectory_result(transitions=[t2]),
        ]
        metric = compute_format_compliance(results)

        assert metric.statistics.mean == 0.0
        assert metric.statistics.count == 0

    def test_counts_all_actions(self):
        """Format compliance counts ALL actions across all trajectories."""
        # Trajectory 1: 2 actions, both formatted
        t1_a1 = make_transition(format_reward=1.0)
        t1_a2 = make_transition(format_reward=1.0, step=1)

        # Trajectory 2: 3 actions, only 1 formatted
        t2_a1 = make_transition(format_reward=1.0)
        t2_a2 = make_transition(format_reward=0.0, step=1)
        t2_a3 = make_transition(format_reward=0.0, step=2)

        results = [
            make_trajectory_result(transitions=[t1_a1, t1_a2]),
            make_trajectory_result(transitions=[t2_a1, t2_a2, t2_a3]),
        ]
        metric = compute_format_compliance(results)

        # 3 formatted out of 5 total actions = 0.6
        assert metric.statistics.mean == pytest.approx(0.6)
        assert metric.statistics.n == 5
        assert metric.statistics.count == 3

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_format_compliance([])

        assert metric.statistics.mean == 0.0
        assert metric.statistics.n == 0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_computes_basic_metrics(self):
        """Test that all basic metrics are computed."""
        t1 = make_transition(correctness_reward=1.0, format_reward=1.0)
        t2 = make_transition(correctness_reward=0.0, format_reward=0.5)

        results = [
            make_trajectory_result(success=True, transitions=[t1]),
            make_trajectory_result(success=False, transitions=[t2]),
        ]
        batch_result = BatchResult(
            trajectory_results=results,
            success_rate=0.5,
            mean_reward=0.75,
            metadata={},
        )

        metrics = compute_all_metrics(batch_result)

        assert metrics.get("accuracy") is not None
        assert metrics.get("trajectory_reward") is not None
        assert metrics.get("action_reward") is not None
        assert metrics.get("format_compliance") is not None

    def test_correct_statistics_types(self):
        """Test that correct statistics types are used."""
        t1 = make_transition(correctness_reward=1.0, format_reward=1.0)

        results = [make_trajectory_result(success=True, transitions=[t1])]
        batch_result = BatchResult(
            trajectory_results=results,
            success_rate=1.0,
            mean_reward=1.0,
            metadata={},
        )

        metrics = compute_all_metrics(batch_result)

        # Binary metrics
        assert isinstance(metrics.get("accuracy").statistics, BinaryStatistics)
        assert isinstance(metrics.get("format_compliance").statistics, BinaryStatistics)

        # Continuous metrics
        assert isinstance(metrics.get("trajectory_reward").statistics, ContinuousStatistics)
        assert isinstance(metrics.get("action_reward").statistics, ContinuousStatistics)

    def test_statistics_available(self):
        """Test that statistics are available for all metrics."""
        t1 = make_transition(correctness_reward=1.0, format_reward=0.0)
        t2 = make_transition(correctness_reward=2.0, format_reward=0.0)
        t3 = make_transition(correctness_reward=3.0, format_reward=0.0)

        results = [
            make_trajectory_result(transitions=[t1]),
            make_trajectory_result(transitions=[t2]),
            make_trajectory_result(transitions=[t3]),
        ]
        batch_result = BatchResult(
            trajectory_results=results,
            success_rate=0.0,
            mean_reward=2.0,
            metadata={},
        )

        metrics = compute_all_metrics(batch_result)

        traj_reward = metrics.get("trajectory_reward")
        action_reward = metrics.get("action_reward")

        assert traj_reward is not None
        assert traj_reward.statistics.std_dev is not None
        assert traj_reward.statistics.min is not None
        assert traj_reward.statistics.max is not None

        assert action_reward is not None
        assert action_reward.statistics.std_dev is not None


class TestAggregateContinuousMetrics:
    """Tests for aggregate_continuous_metrics."""

    def test_basic_aggregation(self):
        """Test aggregating two continuous metrics with equal weights."""
        m1 = Metric(
            "reward",
            ContinuousStatistics(n=10, mean=5.0, std_dev=2.0, min=1.0, max=9.0, std_error=0.632),
        )
        m2 = Metric(
            "reward",
            ContinuousStatistics(n=10, mean=7.0, std_dev=2.0, min=3.0, max=11.0, std_error=0.632),
        )
        result = aggregate_continuous_metrics([m1, m2])

        assert result.name == "reward"
        assert result.statistics.n == 20
        assert result.statistics.mean == pytest.approx(6.0)  # weighted mean
        assert result.statistics.min == 1.0
        assert result.statistics.max == 11.0

    def test_weighted_mean(self):
        """Test that weighted mean is computed correctly for unequal n."""
        m1 = Metric("reward", ContinuousStatistics(n=10, mean=10.0, std_dev=1.0, std_error=0.316))
        m2 = Metric("reward", ContinuousStatistics(n=30, mean=20.0, std_dev=1.0, std_error=0.183))
        result = aggregate_continuous_metrics([m1, m2])

        # Weighted mean: (10*10 + 30*20) / 40 = 700/40 = 17.5
        assert result.statistics.n == 40
        assert result.statistics.mean == pytest.approx(17.5)

    def test_custom_name(self):
        """Test custom name for aggregated metric."""
        m1 = Metric("math_reward", ContinuousStatistics(n=5, mean=1.0))
        m2 = Metric("logic_reward", ContinuousStatistics(n=5, mean=2.0))
        result = aggregate_continuous_metrics([m1, m2], name="all_reasoning_reward")

        assert result.name == "all_reasoning_reward"

    def test_single_metric(self):
        """Test with single metric returns equivalent."""
        m = Metric(
            "reward",
            ContinuousStatistics(
                n=10,
                mean=5.0,
                std_dev=2.0,
                std_error=0.632,
                min=1.0,
                max=9.0,
                ci_lower=3.76,
                ci_upper=6.24,
            ),
        )
        result = aggregate_continuous_metrics([m])

        assert result.statistics.n == 10
        assert result.statistics.mean == 5.0
        assert result.statistics.min == 1.0
        assert result.statistics.max == 9.0

    def test_single_metric_with_new_name(self):
        """Test single metric with custom name."""
        m = Metric("old_name", ContinuousStatistics(n=5, mean=3.0))
        result = aggregate_continuous_metrics([m], name="new_name")

        assert result.name == "new_name"
        assert result.statistics.mean == 3.0

    def test_empty_raises(self):
        """Test empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            aggregate_continuous_metrics([])

    def test_type_mismatch_raises(self):
        """Test mixing binary with continuous raises TypeError."""
        m1 = Metric("reward", ContinuousStatistics(n=10, mean=5.0))
        m2 = Metric("accuracy", BinaryStatistics(n=10, mean=0.8, count=8))

        with pytest.raises(TypeError, match="Expected ContinuousStatistics"):
            aggregate_continuous_metrics([m1, m2])

    def test_min_max_aggregation(self):
        """Test min/max are properly aggregated."""
        m1 = Metric("reward", ContinuousStatistics(n=5, mean=5.0, min=2.0, max=8.0))
        m2 = Metric("reward", ContinuousStatistics(n=5, mean=15.0, min=10.0, max=20.0))
        m3 = Metric("reward", ContinuousStatistics(n=5, mean=0.0, min=-5.0, max=5.0))
        result = aggregate_continuous_metrics([m1, m2, m3])

        assert result.statistics.min == -5.0
        assert result.statistics.max == 20.0

    def test_quantiles_are_none(self):
        """Test that median, q25, q75 are None (cannot reconstruct)."""
        m1 = Metric(
            "reward",
            ContinuousStatistics(n=10, mean=5.0, std_dev=2.0, median=5.0, q25=4.0, q75=6.0),
        )
        m2 = Metric(
            "reward",
            ContinuousStatistics(n=10, mean=7.0, std_dev=2.0, median=7.0, q25=6.0, q75=8.0),
        )
        result = aggregate_continuous_metrics([m1, m2])

        assert result.statistics.median is None
        assert result.statistics.q25 is None
        assert result.statistics.q75 is None

    def test_pooled_variance_identical_groups(self):
        """Test variance pooling with identical groups."""
        # Two groups with same mean and std_dev
        m1 = Metric("reward", ContinuousStatistics(n=10, mean=5.0, std_dev=2.0, std_error=0.632))
        m2 = Metric("reward", ContinuousStatistics(n=10, mean=5.0, std_dev=2.0, std_error=0.632))
        result = aggregate_continuous_metrics([m1, m2])

        # SS_within = (10-1)*4 + (10-1)*4 = 72
        # SS_between = 0 (means are equal)
        # var_pooled = 72 / 19 ≈ 3.789, std ≈ 1.947
        assert result.statistics.std_dev == pytest.approx(1.947, rel=0.01)

    def test_pooled_variance_different_means(self):
        """Test pooled variance captures between-group variance."""
        # Two groups: [0,0,0] mean=0, std=0 and [10,10,10] mean=10, std=0
        # Combined variance comes from between-group differences
        m1 = Metric("reward", ContinuousStatistics(n=3, mean=0.0, std_dev=0.0))
        m2 = Metric("reward", ContinuousStatistics(n=3, mean=10.0, std_dev=0.0))
        result = aggregate_continuous_metrics([m1, m2])

        # Overall mean = 5.0
        # SS_between = 3*(0-5)^2 + 3*(10-5)^2 = 75 + 75 = 150
        # SS_within = 0
        # var = 150 / 5 = 30, std = sqrt(30) ≈ 5.477
        assert result.statistics.mean == pytest.approx(5.0)
        assert result.statistics.std_dev == pytest.approx(5.477, rel=0.01)

    def test_confidence_interval_computed(self):
        """Test that CI is computed for aggregated metrics."""
        m1 = Metric("reward", ContinuousStatistics(n=10, mean=5.0, std_dev=2.0, std_error=0.632))
        m2 = Metric("reward", ContinuousStatistics(n=10, mean=7.0, std_dev=2.0, std_error=0.632))
        result = aggregate_continuous_metrics([m1, m2])

        assert result.statistics.ci_lower is not None
        assert result.statistics.ci_upper is not None
        assert result.statistics.ci_lower < result.statistics.mean
        assert result.statistics.ci_upper > result.statistics.mean

    def test_all_n_zero(self):
        """Test aggregating metrics where all n=0."""
        m1 = Metric("reward", ContinuousStatistics(n=0, mean=0.0))
        m2 = Metric("reward", ContinuousStatistics(n=0, mean=0.0))
        result = aggregate_continuous_metrics([m1, m2])

        assert result.statistics.n == 0
        assert result.statistics.mean == 0.0

    def test_mixed_n_with_zeros(self):
        """Test aggregating when some metrics have n=0."""
        m1 = Metric("reward", ContinuousStatistics(n=0, mean=0.0))
        m2 = Metric("reward", ContinuousStatistics(n=10, mean=5.0, std_dev=2.0))
        result = aggregate_continuous_metrics([m1, m2])

        assert result.statistics.n == 10
        assert result.statistics.mean == 5.0


class TestAggregateBinaryMetrics:
    """Tests for aggregate_binary_metrics."""

    def test_basic_aggregation(self):
        """Test aggregating two binary metrics."""
        m1 = Metric("accuracy", BinaryStatistics(n=10, mean=0.8, count=8))
        m2 = Metric("accuracy", BinaryStatistics(n=10, mean=0.6, count=6))
        result = aggregate_binary_metrics([m1, m2])

        assert result.name == "accuracy"
        assert result.statistics.n == 20
        assert result.statistics.count == 14
        assert result.statistics.mean == pytest.approx(0.7)

    def test_custom_name(self):
        """Test custom name for aggregated metric."""
        m1 = Metric("math_acc", BinaryStatistics(n=10, mean=0.8, count=8))
        m2 = Metric("logic_acc", BinaryStatistics(n=10, mean=0.6, count=6))
        result = aggregate_binary_metrics([m1, m2], name="all_reasoning_accuracy")

        assert result.name == "all_reasoning_accuracy"

    def test_single_metric(self):
        """Test with single metric returns equivalent."""
        m = Metric(
            "accuracy",
            BinaryStatistics(
                n=10, mean=0.8, count=8, std_error=0.126, ci_lower=0.49, ci_upper=0.94
            ),
        )
        result = aggregate_binary_metrics([m])

        assert result.statistics.n == 10
        assert result.statistics.count == 8
        assert result.statistics.mean == 0.8

    def test_empty_raises(self):
        """Test empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            aggregate_binary_metrics([])

    def test_type_mismatch_raises(self):
        """Test mixing continuous with binary raises TypeError."""
        m1 = Metric("accuracy", BinaryStatistics(n=10, mean=0.8, count=8))
        m2 = Metric("reward", ContinuousStatistics(n=10, mean=5.0))

        with pytest.raises(TypeError, match="Expected BinaryStatistics"):
            aggregate_binary_metrics([m1, m2])

    def test_pass_at_k_works(self):
        """Test pass_at_k works on aggregated binary stats."""
        m1 = Metric("accuracy", BinaryStatistics(n=5, mean=0.4, count=2))
        m2 = Metric("accuracy", BinaryStatistics(n=5, mean=0.2, count=1))
        result = aggregate_binary_metrics([m1, m2])

        # 3 successes out of 10
        assert result.statistics.count == 3
        assert result.statistics.n == 10
        assert result.statistics.pass_at_k(1) == pytest.approx(0.3)
        assert result.statistics.pass_at_k(10) == 1.0

    def test_confidence_interval_computed(self):
        """Test that Wilson CI is computed for aggregated metrics."""
        m1 = Metric("accuracy", BinaryStatistics(n=10, mean=0.8, count=8))
        m2 = Metric("accuracy", BinaryStatistics(n=10, mean=0.6, count=6))
        result = aggregate_binary_metrics([m1, m2])

        assert result.statistics.ci_lower is not None
        assert result.statistics.ci_upper is not None
        assert 0.0 <= result.statistics.ci_lower <= result.statistics.mean
        assert result.statistics.mean <= result.statistics.ci_upper <= 1.0

    def test_std_error_computed(self):
        """Test that std_error is computed for aggregated metrics."""
        m1 = Metric("accuracy", BinaryStatistics(n=10, mean=0.8, count=8))
        m2 = Metric("accuracy", BinaryStatistics(n=10, mean=0.6, count=6))
        result = aggregate_binary_metrics([m1, m2])

        # Combined: n=20, p=0.7
        # std_error = sqrt(0.7 * 0.3 / 20) = sqrt(0.0105) ≈ 0.1025
        assert result.statistics.std_error == pytest.approx(0.1025, rel=0.01)

    def test_all_n_zero(self):
        """Test aggregating metrics where all n=0."""
        m1 = Metric("accuracy", BinaryStatistics(n=0, mean=0.0, count=0))
        m2 = Metric("accuracy", BinaryStatistics(n=0, mean=0.0, count=0))
        result = aggregate_binary_metrics([m1, m2])

        assert result.statistics.n == 0
        assert result.statistics.count == 0
        assert result.statistics.mean == 0.0

    def test_mixed_n_with_zeros(self):
        """Test aggregating when some metrics have n=0."""
        m1 = Metric("accuracy", BinaryStatistics(n=0, mean=0.0, count=0))
        m2 = Metric("accuracy", BinaryStatistics(n=10, mean=0.8, count=8))
        result = aggregate_binary_metrics([m1, m2])

        assert result.statistics.n == 10
        assert result.statistics.count == 8
        assert result.statistics.mean == 0.8

    def test_all_successes(self):
        """Test aggregating all-success metrics."""
        m1 = Metric("accuracy", BinaryStatistics(n=10, mean=1.0, count=10))
        m2 = Metric("accuracy", BinaryStatistics(n=5, mean=1.0, count=5))
        result = aggregate_binary_metrics([m1, m2])

        assert result.statistics.n == 15
        assert result.statistics.count == 15
        assert result.statistics.mean == 1.0

    def test_no_successes(self):
        """Test aggregating no-success metrics."""
        m1 = Metric("accuracy", BinaryStatistics(n=10, mean=0.0, count=0))
        m2 = Metric("accuracy", BinaryStatistics(n=5, mean=0.0, count=0))
        result = aggregate_binary_metrics([m1, m2])

        assert result.statistics.n == 15
        assert result.statistics.count == 0
        assert result.statistics.mean == 0.0
