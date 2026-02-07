# Quick Start

Get up and running with llenvs in minutes.

## Using the CLI

The simplest way to run evaluations:

```bash
# Run evaluation from config file
llenvs run config.yaml

# Limit number of tasks
llenvs run config.yaml --limit 10

# Run specific environment only
llenvs run config.yaml --environment leg_counting

# Custom output directory
llenvs run config.yaml --output-dir ./my_results
```

### Configuration File

```yaml
# config.yaml
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 100
    seed: 42
    answer_extractor: tag_based

model:
  backend: openai
  model: gpt-4o

inference:
  temperature: 0.0
  max_tokens: 2048

system_prompt: general_reasoning  # Pre-built prompt (see Prompts guide)
# Or use a literal string:
# system_prompt: |
#   You are a helpful assistant. Think step by step.
#   Put your final answer in <answer>...</answer> tags.

output_dir: ./results
```

## Programmatic Usage

### Basic Evaluation Loop

```python
from llenvs.core.registry import environment_registry
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import SamplingParams
from llenvs.core import TextAction

# Create environment via registry
env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=100,
    seed=42,
)

# Create backend
backend = OpenAIBackend(model="gpt-4o")
params = SamplingParams(temperature=0.0, max_tokens=1024)

# Run single trajectory
state, info = env.reset(options={"task_index": 0})
print(f"Question: {state.observation.prompt}")

# Generate response
from llenvs.inference import ChatMessage
messages = [ChatMessage(role="user", content=state.observation.prompt)]
result = backend.generate_chat(messages, params)
print(f"Response: {result.text}")

# Step environment
action = TextAction(text=result.text)
step_result = env.step(state, action)

# Check results
print(f"Rewards: {step_result.rewards.total}")
print(f"Extracted answer: {step_result.info['extracted_answer']}")
```

### Using the TrajectoryRunner

For running multiple trajectories with proper orchestration:

```python
from llenvs.adapters import create_reasoning_gym_environment
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import SamplingParams, build_standard_pipeline
from llenvs.evaluation import TrajectoryRunner

# Create environment and backend
env = create_reasoning_gym_environment("leg_counting", size=100, seed=42)
backend = OpenAIBackend(model="gpt-4o")

# Create prompt pipeline
pipeline = build_standard_pipeline(
    system_prompt="You are a helpful assistant.",
    use_cot=True,
    answer_format="xml_answer",
    tag_name="answer",
)

# Create runner
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
    prompt_pipeline=pipeline,
)

# Run single trajectory
result = runner.run_trajectory(task_index=0)
print(f"Success: {result.success}")
print(f"Total reward: {result.total_reward}")

# Run batch with progress
batch_result = runner.run_batch(
    task_indices=list(range(10)),
    progress_callback=lambda c, t: print(f"\rProgress: {c}/{t}", end=""),
)
print(f"\nSuccess rate: {batch_result.success_rate:.2%}")
```

### Convenience Function

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
print(f"Mean reward: {result.mean_reward:.3f}")
```

## Next Steps

- **[Environments Guide](../guides/environments.md)** - Learn about different environment types
- **[Prompts Guide](../guides/prompts.md)** - Pre-built system prompts, templates, and model profiles
- **[Tools Guide](../guides/tools.md)** - Add tool/function calling to evaluations
- **[Evaluation Guide](../guides/evaluation.md)** - Compute metrics and save results
