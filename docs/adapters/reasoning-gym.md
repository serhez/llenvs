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
from llenvs.core.registry import environment_registry
from llenvs.core import Action

env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=100,
    seed=42,
)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
# "How many legs do 2 dogs and 3 birds have in total?"

action = Action(text="<answer>14</answer>")
result = env.step(state, action)
print(f"Correct: {result.rewards.by_name('correctness').value == 1.0}")
```

## Using the Adapter

```python
from llenvs.adapters import ReasoningGymAdapter

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
    answer_extractor=None,  # Use default TagBasedExtractor
)

# Add optional extra rewards (e.g., format compliance)
from llenvs.core.reward import FormatReward
env_with_format = adapter.get_environment(
    name="simple_arithmetic",
    size=500,
    extra_rewards=(FormatReward(env._answer_extractor),),
)
```

## Configuration Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Number of samples to generate |
| `seed` | `int` | Random seed for reproducibility |
| `answer_extractor` | `AnswerExtractor` | Custom answer extractor |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions appended after native rewards |
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

By default, only native rewards are included. Extra rewards (like format checking) are opt-in via `extra_rewards`.

| Reward | Type | Default | Description |
|--------|------|---------|-------------|
| `correctness` | OUTCOME | Yes | 1.0 if answer matches expected |
| `format` | FORMAT | No | 1.0 if answer can be extracted (add via `FormatReward`) |

## Example: Running Evaluation

```python
from llenvs.core.registry import environment_registry
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

env = environment_registry.get(name="leg_counting", adapter="reasoning_gym", size=100, seed=42)
backend = OpenAIBackend(model="gpt-4o")

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=512),
    system_prompt="Answer with <answer>...</answer> tags.",
)

batch = runner.run_batch(list(range(100)))
print(f"Accuracy: {batch.success_rate:.2%}")
```
