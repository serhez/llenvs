# Running Evaluations

This guide covers running evaluations, computing metrics, and saving results.

## TrajectoryRunner

The `TrajectoryRunner` orchestrates evaluation trajectories:

```python
from llenvs.adapters import create_reasoning_gym_environment
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import SamplingParams, build_standard_pipeline
from llenvs.evaluation import TrajectoryRunner

env = create_reasoning_gym_environment("leg_counting", size=100, seed=42)
backend = OpenAIBackend(model="gpt-4o")

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
    prompt_pipeline=build_standard_pipeline(
        system_prompt="You are a helpful assistant.",
        use_cot=True,
        answer_format="xml_answer",
    ),
)

# Run single trajectory
result = runner.run_trajectory(task_index=0)
print(f"Success: {result.success}")
print(f"Total reward: {result.total_reward}")

# Run batch
batch_result = runner.run_batch(
    task_indices=list(range(50)),
    progress_callback=lambda c, t: print(f"\r{c}/{t}", end=""),
)
print(f"\nSuccess rate: {batch_result.success_rate:.2%}")
```

## ToolTrajectoryRunner

For tool-enabled environments:

```python
from llenvs.adapters import create_gem_tool_environment
from llenvs.evaluation import ToolTrajectoryRunner

env = create_gem_tool_environment("math:GSM8K", tool_types=("python",))

runner = ToolTrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Use Python to solve. Submit your final answer.",
)

result = runner.run_trajectory(task_index=0)
```

## Batched Evaluation

`run_batch()` automatically batches inference calls using lockstep execution. All active trajectories advance one step together, and the backend's `generate_chat_batch()` (or `generate_with_tools_batch()`) is called once per step with all active conversations. Trajectories that finish early drop out of subsequent batches.

This applies to all three runner types:
- **`TrajectoryRunner`**: Batches `generate_chat_batch()` calls per step.
- **`ToolTrajectoryRunner`**: Batches `generate_with_tools_batch()` (or `generate_chat_batch()` fallback) per step.
- **`SegmentedTrajectoryRunner`**: Batches segment generation via `generate_segment_batch()` on the continuation strategy. Callbacks (`step_callback`) still run per-trajectory after each step.

This gives significant speedups:
- **vLLM / HuggingFace**: All prompts go through the GPU in one batched call per step.
- **API backends (OpenAI, Anthropic, OpenRouter)**: Concurrent async HTTP requests limited by `max_concurrency`.

Single-trajectory `run_trajectory()` is unchanged.

```python
# This automatically batches inference across all 100 tasks
batch_result = runner.run_batch(
    task_indices=list(range(100)),
    progress_callback=lambda c, t: print(f"\r{c}/{t}", end=""),
)
```

Use `batch_size` to limit how many trajectories run in each lockstep batch (useful for GPU memory management):

```python
# Process 32 tasks at a time instead of all 100
batch_result = runner.run_batch(
    task_indices=list(range(100)),
    batch_size=32,
)
```

For cross-environment batching (interleaving trajectories from multiple environments into a single lockstep loop), see `run_multi_evaluation()` in the [Parallelization guide](parallelization.md#cross-environment-batching).

See the [Parallelization guide](parallelization.md) for architecture details, `max_concurrency` tuning, and performance tips.

## Metrics and Statistics

The evaluation module separates **metrics** (what we measure) from **statistics** (how we summarize):

- **Metrics**: Measurable quantities like `action_reward`, `trajectory_reward`, `accuracy`
- **Statistics**: Summary computations like mean, std_dev, quantiles, confidence intervals

Two statistics types exist for different metric categories:

- **ContinuousStatistics**: For continuous-valued metrics (rewards, scores, etc.)
- **BinaryStatistics**: For binary metrics (success/failure) with pass@k support

### Computing Metrics

```python
from llenvs.evaluation import (
    compute_accuracy,
    compute_trajectory_reward,
    compute_action_reward,
    compute_format_compliance,
)

# Binary metrics (accuracy, format_compliance) return BinaryStatistics
accuracy = compute_accuracy(batch_result.trajectory_results)
stats = accuracy.statistics
print(f"Accuracy: {stats.mean:.3f} ({stats.count}/{stats.n})")
print(f"95% CI: [{stats.ci_lower:.3f}, {stats.ci_upper:.3f}]")

# Continuous metrics (rewards) return ContinuousStatistics
trajectory_reward = compute_trajectory_reward(batch_result.trajectory_results)
stats = trajectory_reward.statistics
print(f"Mean trajectory reward: {stats.mean:.3f}")
print(f"Std dev: {stats.std_dev:.3f}")
print(f"Min: {stats.min:.3f}, Max: {stats.max:.3f}")

# Action-level metrics
action_reward = compute_action_reward(batch_result.trajectory_results)
stats = action_reward.statistics
print(f"Mean action reward: {stats.mean:.3f}")
print(f"Median: {stats.median:.3f}")
print(f"Q25-Q75: [{stats.q25:.3f}, {stats.q75:.3f}]")

format_compliance = compute_format_compliance(batch_result.trajectory_results)
stats = format_compliance.statistics
print(f"Format compliance: {stats.mean:.1%} ({stats.count}/{stats.n} actions)")
```

### Statistics Types

**ContinuousStatistics** (for rewards, scores, etc.):

| Field | Description |
|-------|-------------|
| `n` | Sample size |
| `mean` | Arithmetic mean |
| `std_dev` | Sample standard deviation |
| `std_error` | Standard error of the mean |
| `min` | Minimum value |
| `max` | Maximum value |
| `median` | Median (50th percentile) |
| `q25` | 25th percentile |
| `q75` | 75th percentile |
| `ci_lower` | Lower bound of 95% CI |
| `ci_upper` | Upper bound of 95% CI |

**BinaryStatistics** (for success/failure metrics):

| Field | Description |
|-------|-------------|
| `n` | Sample size |
| `mean` | Success rate (proportion of successes) |
| `count` | Number of successes |
| `std_error` | Standard error sqrt(p(1-p)/n) |
| `ci_lower` | Lower bound of Wilson score CI |
| `ci_upper` | Upper bound of Wilson score CI |

BinaryStatistics also provides the `pass_at_k(k)` method.

### Computing Statistics Directly

You can compute statistics from any sequence of values:

```python
from llenvs.evaluation import compute_continuous_statistics, compute_binary_statistics

# Continuous data
values = [0.8, 0.85, 0.9, 0.78, 0.92]
stats = compute_continuous_statistics(values, confidence_level=0.95)
print(f"Mean: {stats.mean:.3f}")
print(f"Std dev: {stats.std_dev:.3f}")
print(f"Range: [{stats.min:.3f}, {stats.max:.3f}]")

# Binary data
successes = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # 7 successes out of 10
stats = compute_binary_statistics(successes)
print(f"Success rate: {stats.mean:.1%} ({stats.count}/{stats.n})")
```

### Pass@k

Pass@k measures the probability of at least one success in k samples. It's available as a method on `BinaryStatistics`:

```python
# Run multiple times per task
results = []
for _ in range(10):  # 10 samples
    result = runner.run_trajectory(task_index=0)
    results.append(result)

# Get accuracy statistics (BinaryStatistics)
accuracy = compute_accuracy(results)
stats = accuracy.statistics

# Compute Pass@k for different values of k
print(f"Pass@1: {stats.pass_at_k(1):.3f}")
print(f"Pass@5: {stats.pass_at_k(5):.3f}")
print(f"Pass@10: {stats.pass_at_k(10):.3f}")
```

The `pass_at_k(k)` method uses the exact formula: `1 - C(n-c, k) / C(n, k)` where `n` is the total number of samples and `c` is the number of successes.

### All Metrics at Once

```python
from llenvs.evaluation import compute_all_metrics, ContinuousStatistics, BinaryStatistics

metrics = compute_all_metrics(batch_result)

for name, metric in metrics.metrics.items():
    stats = metric.statistics
    if isinstance(stats, ContinuousStatistics):
        print(f"{name}: {stats.mean:.4f} (std: {stats.std_dev:.4f})")
    else:  # BinaryStatistics
        print(f"{name}: {stats.mean:.4f} ({stats.count}/{stats.n})")
```

### Aggregating Metrics

Combine metrics from multiple evaluations:

```python
from llenvs.evaluation import (
    aggregate_continuous_metrics,
    aggregate_binary_metrics,
)

# Aggregate trajectory rewards from multiple task groups
math_rewards = compute_trajectory_reward(math_results)
logic_rewards = compute_trajectory_reward(logic_results)

combined = aggregate_continuous_metrics(
    [math_rewards, logic_rewards],
    name="all_reasoning_reward",
)
print(f"Combined: {combined.statistics.mean:.3f} (n={combined.statistics.n})")

# Aggregate accuracy across task groups
math_acc = compute_accuracy(math_results)
logic_acc = compute_accuracy(logic_results)

combined_acc = aggregate_binary_metrics([math_acc, logic_acc])
print(f"Combined accuracy: {combined_acc.statistics.mean:.1%}")
print(f"Pass@5: {combined_acc.statistics.pass_at_k(5):.3f}")
```

**Note:** Aggregated continuous metrics have `median`, `q25`, `q75` set to `None` because these cannot be accurately reconstructed from summary statistics alone.

## Saving Results

```python
from datetime import datetime
from llenvs.evaluation.results import create_evaluation_result, print_summary

# Create formatted result
eval_result = create_evaluation_result(
    batch_result=batch_result,
    model_name="gpt-4o",
    environment_name="leg_counting",
    start_time=datetime.now(),
    config={"temperature": 0.0, "max_tokens": 1024},
    include_detailed_results=True,
)

# Print summary
print_summary(eval_result)

# Save to JSON
eval_result.save("results/eval_20240115.json")

# Save without per-trajectory details (smaller file)
eval_result.save("results/eval_summary.json", include_results=False)
```

### Result Structure

```python
{
    "metadata": {
        "timestamp": "2024-01-15T10:30:00",
        "model": "gpt-4o",
        "environment": "leg_counting",
        "duration_seconds": 123.4,
        "num_trajectories": 100,
        "config": {...},
    },
    "metrics": {
        # Binary metrics (BinaryStatistics)
        "accuracy": {"n": 100, "mean": 0.85, "count": 85, "std_error": 0.036, ...},
        "format_compliance": {"n": 100, "mean": 0.95, "count": 95, ...},
        # Continuous metrics (ContinuousStatistics)
        "trajectory_reward": {"n": 100, "mean": 0.85, "std_dev": 0.12, "min": 0.0, ...},
        "action_reward": {"n": 100, "mean": 0.82, "std_dev": 0.15, ...},
    },
    "summary": {
        "success_rate": 0.85,
        "mean_reward": 0.85,
        "num_trajectories": 100,
    },
    "results": [...]  # Per-trajectory details if included
}
```

## Prompt Templates and Model Profiles

The runner supports prompt templates (wrapping questions) and model profiles (model-specific adjustments):

```python
from llenvs.inference import TEMPLATE_REGISTRY, PROFILE_REGISTRY

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="You are an expert mathematician.",
    prompt_template=TEMPLATE_REGISTRY["math"],          # Wraps questions
    model_profile=PROFILE_REGISTRY["deepseek_r1"],      # Model-specific
)
```

The message build order is: system prompt → observation → prompt template → model profile → prompt pipeline. See the [Prompts guide](prompts.md) for full details.

## Convenience Function

```python
from llenvs.evaluation import run_evaluation
from llenvs.inference import TEMPLATE_REGISTRY

result = run_evaluation(
    environment=env,
    backend=backend,
    num_tasks=50,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Think step by step. Use <answer>...</answer> tags.",
    prompt_template=TEMPLATE_REGISTRY["math"],  # Optional
)

print(f"Accuracy: {result.success_rate:.2%}")
```

## Error Handling

The runner handles errors gracefully:

```python
batch_result = runner.run_batch(task_indices)

# Check for errors
for result in batch_result.trajectory_results:
    if "error" in result.metadata:
        print(f"Task {result.metadata['task_index']} failed: {result.metadata['error']}")
```

## Custom Reward Analysis

```python
from llenvs.core import RewardType

for result in batch_result.trajectory_results:
    for transition in result.trajectory.transitions:
        # Get rewards by type
        outcome_rewards = transition.rewards.by_type(RewardType.OUTCOME)
        format_rewards = transition.rewards.by_type(RewardType.FORMAT)

        # Get specific reward
        correctness = transition.rewards.by_name("correctness")
        if correctness:
            print(f"Correctness: {correctness.value}")
```
