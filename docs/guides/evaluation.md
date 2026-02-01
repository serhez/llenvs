# Running Evaluations

This guide covers running evaluations, computing metrics, and saving results.

## EpisodeRunner

The `EpisodeRunner` orchestrates evaluation episodes:

```python
from llenvs.adapters import create_reasoning_gym_environment
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import SamplingParams, build_standard_pipeline
from llenvs.evaluation import EpisodeRunner

env = create_reasoning_gym_environment("leg_counting", size=100, seed=42)
backend = OpenAIBackend(model="gpt-4o")

runner = EpisodeRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
    prompt_pipeline=build_standard_pipeline(
        system_prompt="You are a helpful assistant.",
        use_cot=True,
        answer_format="xml_answer",
    ),
)

# Run single episode
result = runner.run_episode(task_index=0)
print(f"Success: {result.success}")
print(f"Total reward: {result.total_reward}")

# Run batch
batch_result = runner.run_batch(
    task_indices=list(range(50)),
    progress_callback=lambda c, t: print(f"\r{c}/{t}", end=""),
)
print(f"\nSuccess rate: {batch_result.success_rate:.2%}")
```

## ToolEpisodeRunner

For tool-enabled environments:

```python
from llenvs.adapters import create_gem_tool_environment
from llenvs.evaluation import ToolEpisodeRunner

env = create_gem_tool_environment("math:GSM8K", tool_types=("python",))

runner = ToolEpisodeRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Use Python to solve. Submit your final answer.",
)

result = runner.run_episode(task_index=0)
```

## Computing Metrics

### Basic Metrics

```python
from llenvs.evaluation.metrics import (
    compute_accuracy,
    compute_mean_reward,
    compute_format_compliance,
    compute_all_metrics,
)

# Individual metrics
accuracy = compute_accuracy(batch_result.episode_results)
print(f"Accuracy: {accuracy.value:.3f} ± {accuracy.std_error:.3f}")
print(f"95% CI: [{accuracy.ci_lower:.3f}, {accuracy.ci_upper:.3f}]")

mean_reward = compute_mean_reward(batch_result.episode_results)
print(f"Mean reward: {mean_reward.value:.3f}")

format_rate = compute_format_compliance(batch_result.episode_results)
print(f"Format compliance: {format_rate.value:.1%}")
```

### Pass@k

For multiple samples per task:

```python
from llenvs.evaluation.metrics import compute_pass_at_k

# Run multiple times per task
results_by_task = {}
for task_idx in range(10):
    results_by_task[task_idx] = []
    for _ in range(5):  # 5 samples per task
        result = runner.run_episode(task_idx)
        results_by_task[task_idx].append(result)

# Compute Pass@k
pass_at_1 = compute_pass_at_k(results_by_task, k=1)
pass_at_5 = compute_pass_at_k(results_by_task, k=5)

print(f"Pass@1: {pass_at_1.value:.3f}")
print(f"Pass@5: {pass_at_5.value:.3f}")
```

### All Metrics at Once

```python
metrics = compute_all_metrics(
    batch_result,
    results_by_task=results_by_task,
    k_values=[1, 5, 10],
)

for name, metric in metrics.metrics.items():
    print(f"{name}: {metric.value:.4f}")
```

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

# Save without per-episode details (smaller file)
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
        "num_episodes": 100,
        "config": {...},
    },
    "metrics": {
        "accuracy": {"value": 0.85, "std_error": 0.036, ...},
        "mean_reward": {"value": 0.85, ...},
    },
    "summary": {
        "success_rate": 0.85,
        "mean_reward": 0.85,
        "num_episodes": 100,
    },
    "results": [...]  # Per-episode details if included
}
```

## Convenience Function

```python
from llenvs.evaluation import run_evaluation

result = run_evaluation(
    environment=env,
    backend=backend,
    num_tasks=50,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Think step by step. Use <answer>...</answer> tags.",
)

print(f"Accuracy: {result.success_rate:.2%}")
```

## Error Handling

The runner handles errors gracefully:

```python
batch_result = runner.run_batch(task_indices)

# Check for errors
for result in batch_result.episode_results:
    if "error" in result.metadata:
        print(f"Task {result.metadata['task_index']} failed: {result.metadata['error']}")
```

## Custom Reward Analysis

```python
from llenvs.core import RewardType

for result in batch_result.episode_results:
    for transition in result.trajectory.transitions:
        # Get rewards by type
        outcome_rewards = transition.rewards.by_type(RewardType.OUTCOME)
        format_rewards = transition.rewards.by_type(RewardType.FORMAT)

        # Get specific reward
        correctness = transition.rewards.by_name("correctness")
        if correctness:
            print(f"Correctness: {correctness.value}")
```
