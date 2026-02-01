"""Tests for metric computation."""

import pytest
from llenvs.core.state import State, StateMetadata, TextObservation
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType
from llenvs.evaluation.runner import EpisodeResult, BatchResult
from llenvs.evaluation.metrics import (
    compute_accuracy,
    compute_mean_reward,
    compute_pass_at_k,
    compute_format_compliance,
    compute_all_metrics,
    MetricValue,
    MetricsBundle,
)


def make_episode_result(
    success: bool,
    total_reward: float,
    format_compliant: bool = True,
    task_index: int = 0,
) -> EpisodeResult:
    """Helper to create episode results for testing."""
    state = State(
        observation=TextObservation(prompt="test"),
        hidden={},
        metadata=StateMetadata(step=0, episode_id=f"ep_{task_index}", is_terminal=False),
    )
    trajectory = Trajectory.create(state)

    # Add a transition with rewards
    format_value = 1.0 if format_compliant else 0.0
    correctness_value = 1.0 if success else 0.0

    transition = Transition(
        state=state,
        action=None,  # type: ignore
        next_state=state.with_metadata(is_terminal=True),
        rewards=RewardBundle(
            signals=(
                RewardSignal(value=correctness_value, name="correctness", reward_type=RewardType.OUTCOME),
                RewardSignal(value=format_value, name="format", reward_type=RewardType.FORMAT),
            )
        ),
        info={},
    )
    trajectory.add_transition(transition)

    return EpisodeResult(
        trajectory=trajectory,
        total_reward=total_reward,
        success=success,
        metadata={"task_index": task_index},
    )


class TestMetricValue:
    """Tests for MetricValue."""

    def test_creation(self):
        """Test basic metric value creation."""
        metric = MetricValue(
            name="accuracy",
            value=0.85,
            std_error=0.05,
            ci_lower=0.75,
            ci_upper=0.95,
            n=100,
        )

        assert metric.name == "accuracy"
        assert metric.value == 0.85
        assert metric.std_error == 0.05
        assert metric.n == 100

    def test_optional_fields(self):
        """Test metric with optional fields."""
        metric = MetricValue(name="test", value=0.5, n=10)

        assert metric.std_error is None
        assert metric.ci_lower is None
        assert metric.ci_upper is None


class TestMetricsBundle:
    """Tests for MetricsBundle."""

    def test_add_and_get(self):
        """Test adding and retrieving metrics."""
        bundle = MetricsBundle()
        metric = MetricValue(name="accuracy", value=0.9, n=100)

        bundle.add(metric)
        retrieved = bundle.get("accuracy")

        assert retrieved is not None
        assert retrieved.value == 0.9

    def test_get_nonexistent(self):
        """Test getting non-existent metric."""
        bundle = MetricsBundle()
        assert bundle.get("nonexistent") is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        bundle = MetricsBundle()
        bundle.add(MetricValue(name="accuracy", value=0.9, std_error=0.05, n=100))
        bundle.add(MetricValue(name="reward", value=1.5, n=100))

        result = bundle.to_dict()

        assert "accuracy" in result
        assert result["accuracy"]["value"] == 0.9
        assert result["accuracy"]["std_error"] == 0.05
        assert "reward" in result


class TestComputeAccuracy:
    """Tests for compute_accuracy."""

    def test_all_correct(self):
        """Test accuracy with all correct answers."""
        results = [make_episode_result(success=True, total_reward=1.0) for _ in range(10)]
        metric = compute_accuracy(results)

        assert metric.value == 1.0
        assert metric.n == 10

    def test_all_incorrect(self):
        """Test accuracy with all incorrect answers."""
        results = [make_episode_result(success=False, total_reward=0.0) for _ in range(10)]
        metric = compute_accuracy(results)

        assert metric.value == 0.0

    def test_mixed(self):
        """Test accuracy with mixed results."""
        results = [
            make_episode_result(success=True, total_reward=1.0),
            make_episode_result(success=True, total_reward=1.0),
            make_episode_result(success=False, total_reward=0.0),
            make_episode_result(success=False, total_reward=0.0),
        ]
        metric = compute_accuracy(results)

        assert metric.value == 0.5
        assert metric.n == 4

    def test_confidence_interval(self):
        """Test that confidence interval is computed."""
        results = [make_episode_result(success=i < 8, total_reward=1.0 if i < 8 else 0.0) for i in range(10)]
        metric = compute_accuracy(results)

        assert metric.ci_lower is not None
        assert metric.ci_upper is not None
        assert metric.ci_lower <= metric.value <= metric.ci_upper

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_accuracy([])

        assert metric.value == 0.0
        assert metric.n == 0


class TestComputeMeanReward:
    """Tests for compute_mean_reward."""

    def test_basic(self):
        """Test basic mean reward computation."""
        results = [
            make_episode_result(success=True, total_reward=1.0),
            make_episode_result(success=True, total_reward=2.0),
            make_episode_result(success=False, total_reward=0.0),
        ]
        metric = compute_mean_reward(results)

        assert metric.value == pytest.approx(1.0)
        assert metric.n == 3

    def test_std_error(self):
        """Test that standard error is computed."""
        results = [make_episode_result(success=True, total_reward=float(i)) for i in range(10)]
        metric = compute_mean_reward(results)

        assert metric.std_error is not None
        assert metric.std_error > 0

    def test_single_result(self):
        """Test with single result (no std error)."""
        results = [make_episode_result(success=True, total_reward=5.0)]
        metric = compute_mean_reward(results)

        assert metric.value == 5.0
        assert metric.std_error == 0.0

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_mean_reward([])

        assert metric.value == 0.0
        assert metric.n == 0


class TestComputePassAtK:
    """Tests for compute_pass_at_k."""

    def test_pass_at_1_all_correct(self):
        """Test Pass@1 when all attempts are correct."""
        results_by_task = {
            0: [make_episode_result(success=True, total_reward=1.0, task_index=0) for _ in range(5)],
            1: [make_episode_result(success=True, total_reward=1.0, task_index=1) for _ in range(5)],
        }
        metric = compute_pass_at_k(results_by_task, k=1)

        assert metric.value == 1.0

    def test_pass_at_1_none_correct(self):
        """Test Pass@1 when no attempts are correct."""
        results_by_task = {
            0: [make_episode_result(success=False, total_reward=0.0, task_index=0) for _ in range(5)],
        }
        metric = compute_pass_at_k(results_by_task, k=1)

        assert metric.value == 0.0

    def test_pass_at_k_increases_with_k(self):
        """Test that Pass@k increases with k for mixed results."""
        # One success out of 5 attempts per task
        results_by_task = {
            i: [
                make_episode_result(success=(j == 0), total_reward=1.0 if j == 0 else 0.0, task_index=i)
                for j in range(5)
            ]
            for i in range(10)
        }

        pass_at_1 = compute_pass_at_k(results_by_task, k=1)
        pass_at_5 = compute_pass_at_k(results_by_task, k=5)

        # Pass@5 should be higher (guaranteed since each task has 1 success in 5)
        assert pass_at_5.value >= pass_at_1.value

    def test_pass_at_k_all_correct_in_5(self):
        """Test Pass@5 when every task has at least one success."""
        results_by_task = {
            i: [
                make_episode_result(success=True, total_reward=1.0, task_index=i),
                make_episode_result(success=False, total_reward=0.0, task_index=i),
                make_episode_result(success=False, total_reward=0.0, task_index=i),
                make_episode_result(success=False, total_reward=0.0, task_index=i),
                make_episode_result(success=False, total_reward=0.0, task_index=i),
            ]
            for i in range(5)
        }
        metric = compute_pass_at_k(results_by_task, k=5)

        assert metric.value == 1.0  # At least one correct per task

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_pass_at_k({}, k=1)
        assert metric.value == 0.0


class TestComputeFormatCompliance:
    """Tests for compute_format_compliance."""

    def test_all_compliant(self):
        """Test with all format-compliant results."""
        results = [make_episode_result(success=True, total_reward=1.0, format_compliant=True) for _ in range(10)]
        metric = compute_format_compliance(results)

        assert metric.value == 1.0

    def test_none_compliant(self):
        """Test with no format-compliant results."""
        results = [make_episode_result(success=False, total_reward=0.0, format_compliant=False) for _ in range(10)]
        metric = compute_format_compliance(results)

        assert metric.value == 0.0

    def test_mixed(self):
        """Test with mixed compliance."""
        results = [
            make_episode_result(success=True, total_reward=1.0, format_compliant=True),
            make_episode_result(success=True, total_reward=1.0, format_compliant=True),
            make_episode_result(success=False, total_reward=0.0, format_compliant=False),
            make_episode_result(success=False, total_reward=0.0, format_compliant=False),
        ]
        metric = compute_format_compliance(results)

        assert metric.value == 0.5

    def test_empty_results(self):
        """Test with empty results."""
        metric = compute_format_compliance([])
        assert metric.value == 0.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_computes_basic_metrics(self):
        """Test that all basic metrics are computed."""
        results = [
            make_episode_result(success=True, total_reward=1.0),
            make_episode_result(success=False, total_reward=0.5),
        ]
        batch_result = BatchResult(
            episode_results=results,
            success_rate=0.5,
            mean_reward=0.75,
            metadata={},
        )

        metrics = compute_all_metrics(batch_result)

        assert metrics.get("accuracy") is not None
        assert metrics.get("mean_reward") is not None
        assert metrics.get("format_compliance") is not None

    def test_computes_pass_at_k(self):
        """Test Pass@k computation when results_by_task provided."""
        results = [
            make_episode_result(success=True, total_reward=1.0, task_index=0),
            make_episode_result(success=False, total_reward=0.0, task_index=0),
        ]
        batch_result = BatchResult(
            episode_results=results,
            success_rate=0.5,
            mean_reward=0.5,
            metadata={},
        )
        results_by_task = {0: results}

        metrics = compute_all_metrics(batch_result, results_by_task=results_by_task, k_values=[1, 5])

        assert metrics.get("pass@1") is not None
        assert metrics.get("pass@5") is not None
