# ReasoningGym Adapter

The [reasoning-gym](https://github.com/open-thought/reasoning-gym) adapter provides access to procedural reasoning tasks.

## Installation

```bash
pip install reasoning-gym
# or
pip install llenvs[reasoning-gym]
```

## Quick Start

```python
from env_evals.adapters import create_reasoning_gym_environment
from env_evals.core import TextAction

env = create_reasoning_gym_environment(
    dataset_name="leg_counting",
    size=100,
    seed=42,
)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
# "How many legs do 2 dogs and 3 birds have in total?"

action = TextAction(text="<answer>14</answer>")
result = env.step(state, action)
print(f"Correct: {result.rewards.by_name('correctness').value == 1.0}")
```

## Using the Adapter

```python
from env_evals.adapters import ReasoningGymAdapter

adapter = ReasoningGymAdapter()

# List available environments
envs = adapter.list_environments()
print(envs[:10])
# ["sudoku", "leg_counting", "simple_arithmetic", ...]

# Get environment info
info = adapter.get_environment_info("sudoku")
print(info)
# {"name": "sudoku", "adapter": "reasoning_gym", "type": "single_turn"}

# Create with full configuration
env = adapter.get_environment(
    name="simple_arithmetic",
    size=500,
    seed=42,
    extractor=None,  # Use default TagBasedExtractor
    include_format_reward=True,
)
```

## Configuration Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Number of samples to generate |
| `seed` | `int` | Random seed for reproducibility |
| `extractor` | `AnswerExtractor` | Custom answer extractor |
| `include_format_reward` | `bool` | Include format compliance reward |
| `**dataset_kwargs` | | Passed to dataset constructor |

## Available Datasets

reasoning-gym provides many procedural datasets:

- **Arithmetic**: `simple_arithmetic`, `chain_sum`
- **Counting**: `leg_counting`, `object_counting`
- **Logic**: `propositional_logic`, `syllogisms`
- **Puzzles**: `sudoku`, `tower_of_hanoi`
- **Spatial**: `maze_solving`, `path_finding`

Get the full list:

```python
adapter = ReasoningGymAdapter()
print(adapter.list_environments())
```

## Hidden State

```python
@dataclass(frozen=True)
class ReasoningGymHidden:
    entry: dict[str, Any]      # Original dataset entry
    expected_answer: str       # entry["answer"] as string
    task_index: int
    dataset_name: str
```

## Rewards

| Reward | Type | Description |
|--------|------|-------------|
| `correctness` | OUTCOME | 1.0 if answer matches expected |
| `format` | FORMAT | 1.0 if answer can be extracted |

## Example: Running Evaluation

```python
from env_evals.adapters import create_reasoning_gym_environment
from env_evals.inference.backends import OpenAIBackend
from env_evals.evaluation import EpisodeRunner
from env_evals.inference import SamplingParams

env = create_reasoning_gym_environment("leg_counting", size=100, seed=42)
backend = OpenAIBackend(model="gpt-4o")

runner = EpisodeRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=512),
    system_prompt="Answer with <answer>...</answer> tags.",
)

batch = runner.run_batch(list(range(100)))
print(f"Accuracy: {batch.success_rate:.2%}")
```
