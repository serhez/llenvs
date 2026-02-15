# Aviary Adapter

Wraps [Aviary](https://github.com/Future-House/aviary) (`fhaviary`) tool-calling environments as llenvs MDP environments.

## Installation

```bash
uv pip install -e ".[aviary]"
```

## Quick Start

```python
from llenvs.adapters.aviary import AviaryAdapter

adapter = AviaryAdapter()
env = adapter.get_environment("gsm8k")

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
print([t.name for t in state.observation.available_tools])

from llenvs.core import Action
from llenvs.core.tools import ToolCall

call = ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
action = Action(text="", tool_calls=(call,))
result = env.step(state, action)
print(result.rewards.total)
```

### With Custom Dataset

```python
from aviary.envs.gsm8k import GSM8kDataset

dataset = GSM8kDataset(split="test")
env = adapter.get_environment("my-gsm8k", dataset=dataset, max_steps=15)
```

### With Evaluation Runner

```python
from llenvs.evaluation.runner import run_tool_evaluation

result = run_tool_evaluation(
    environment=env,
    backend=backend,
    task_indices=list(range(50)),
)
print(f"Success rate: {result.success_rate}")
```

## How It Works

### Tool Execution Delegation

Aviary tools have access to internal environment state via an injected `state` parameter. This means tool execution **must** be delegated to Aviary's `step()` rather than extracted and called directly. The adapter extends `BaseToolEnvironment` for tool infrastructure (definitions, monitoring rewards) but lets Aviary handle all tool execution internally.

The flow:

1. `reset()` calls Aviary's async `reset()` which returns `(messages, tools)`
2. Tools are converted to `ToolDefinition` objects via `oai_tools_to_definitions()`
3. `step()` converts the llenvs `Action` to an Aviary `ToolRequestMessage`
4. Aviary's async `step()` executes tools internally and returns `(messages, reward, done, truncated)`
5. Response messages are converted back to an `Observation` with `ToolResult` objects

### Task Dataset

Aviary uses `TaskDataset.get_new_env_by_idx(idx)` to create a fresh environment per task. Each reset creates a new Aviary environment instance, ensuring clean state.

### Rewards

Aviary provides a numeric reward on each step. The adapter wraps this as:

- `RewardType.STEP` for intermediate steps
- `RewardType.OUTCOME` for terminal steps

Cumulative reward is tracked in hidden state metadata.

## Presets

| Preset | Dataset Class | Description |
|---|---|---|
| `gsm8k` | `GSM8kDataset` | Grade school math with calculator tool |
| `hotpotqa` | `HotPotQADataset` | Multi-hop question answering with search |
| `labbench` | `LABBenchDataset` | Laboratory benchmark tasks |
| `lfrqa` | `LFRQADataset` | Long-form retrieval QA |

## Parameters

### `AviaryAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Preset name or custom name |
| `dataset` | `TaskDataset \| None` | Pre-created dataset (bypasses preset lookup) |
| `max_steps` | `int \| None` | Maximum steps per episode |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |
| `**kwargs` | | Passed to the dataset constructor |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes |
| `task_index` | Yes |
| `seed` | No |
| `compute_rewards` | Yes |
| `pure_step` | No (Aviary envs are stateful) |
| Tool calling | Yes (delegated to Aviary) |
| `Scorer` / `DatasetProvider` | No (multi-turn) |

## Limitations

- **Async-first**: Aviary is async-first; the adapter runs coroutines synchronously via `asyncio.run()` with a thread pool fallback for nested event loops
- **No seed support**: Aviary's `TaskDataset` doesn't expose a seeding mechanism
- **Tool execution**: Tools cannot be called independently of `step()` since they access internal environment state
- **Python >= 3.11**: Aviary requires Python 3.11+, though llenvs itself requires 3.12+
