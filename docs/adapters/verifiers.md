# Verifiers Adapter

Wraps [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments as llenvs MDP environments.

## Installation

```bash
uv pip install -e ".[verifiers]"
```

## Environment Types

The adapter supports two verifiers environment types:

| verifiers Type | llenvs Wrapper | Turn Type |
|---|---|---|
| `SingleTurnEnv` | `VerifiersSingleTurnEnvironment` | Single-turn |
| `ToolEnv` / `MultiTurnEnv` | `VerifiersToolEnvironment` | Multi-turn with tools |
| `SandboxEnv` / `PythonEnv` | Not supported | Requires Prime Sandboxes |

## Quick Start

### Single-Turn (GSM8K, MATH, etc.)

```python
from llenvs.adapters.verifiers import VerifiersAdapter

adapter = VerifiersAdapter()
env = adapter.get_environment("gsm8k")

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)  # Math question
print(info["system_prompt"])     # From verifiers env

from llenvs.core import Action
result = env.step(state, Action(text="The answer is \\boxed{42}"))
print(result.rewards.total)  # Rubric-computed reward
```

### Tool Environment

```python
env = adapter.get_environment("tool_test")

state, info = env.reset(options={"task_index": 0})
print(env.available_tools)  # Tool definitions from verifiers

# Use tools
from llenvs.core.tools import ToolCall
call = ToolCall(id="c1", name="tool_A", arguments={"x": 5})
action = Action(text="", tool_calls=(call,))
result = env.step(state, action)
```

### With Evaluation Runner

```python
from llenvs.evaluation.runner import run_evaluation

result = run_evaluation(
    environment=env,
    backend=backend,
    task_indices=list(range(100)),
)
print(f"Success rate: {result.success_rate}")
```

## How It Works

### Dataset Mapping

verifiers datasets are HuggingFace `Dataset` objects with columns:

- `prompt`: List of chat message dicts (system + user messages)
- `answer`: Ground truth string
- `example_id`: Integer index

The adapter extracts the user message as the observation prompt, stores the full row in hidden state for reward computation.

### Rubric Rewards

verifiers rubrics contain weighted async reward functions. The adapter:

1. Calls each rubric function with appropriate keyword arguments (`completion`, `answer`, `prompt`, etc.)
2. Applies the rubric's weights to each score
3. Returns the weighted sum as a single `OUTCOME`-type `RewardSignal` named `"verifiers_rubric"`

```python
# Rubric with two functions at different weights
rubric = vf.Rubric(
    funcs=[exact_match, format_check],
    weights=[2.0, 0.5],
)
# -> RewardSignal(value=2.0*1.0 + 0.5*0.8, name="verifiers_rubric", type=OUTCOME)
```

### Tool Conversion

For `ToolEnv`, the adapter converts OpenAI-format tool schemas to llenvs `ToolDefinition` objects and wraps the original Python callables in a `VerifiersToolExecutor`.

## Parameters

### `VerifiersAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Environment name/ID (e.g., `"gsm8k"`, `"tool_test"`) |
| `answer_extractor` | `AnswerExtractor \| None` | Optional extractor for parsing responses |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |
| `**kwargs` | | Passed to `verifiers.load_environment()` |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes (dataset size) |
| `task_index` | Yes (dataset index) |
| `seed` | No |
| `compute_rewards` | Yes (via rubric) |
| `Scorer` / `DatasetProvider` | Yes (single-turn) |
| System prompt | Yes (from `env.system_prompt`) |
| Answer extraction | Optional (pass `answer_extractor`) |

## Limitations

- **SandboxEnv / PythonEnv** are not supported — they require Prime Sandboxes cloud infrastructure for Docker container management
- **Group reward functions** (operating across multiple rollouts) are not supported — the adapter evaluates each response independently
- **`env_response()`** from `MultiTurnEnv` is not called — the adapter only uses the dataset and rubric, bypassing verifiers' own rollout loop
