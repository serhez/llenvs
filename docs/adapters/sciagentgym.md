# SciAgentGYM Adapter

Wraps [SciAgentGYM](https://github.com/CMarsRover/SciAgentGYM) as llenvs MDP environments for multi-step scientific tool-use evaluation.

## Installation

SciAgentGYM is not on PyPI. Install from git:

```bash
pip install git+https://github.com/CMarsRover/SciAgentGYM.git
```

## Quick Start

```python
from llenvs.adapters.sciagentgym import SciAgentGymAdapter

adapter = SciAgentGymAdapter()
env = adapter.get_environment("sciagentgym", data_path="path/to/test_cases.json")

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
print([t.name for t in state.observation.available_tools])

from llenvs.core import Action
from llenvs.core.tools import ToolCall

# Use a scientific tool
call = ToolCall(id="c1", name="analyze_threshold", arguments={"energy": 0.001})
action = Action(tool_calls=(call,))
result = env.step(state, action)

# Submit final answer
action = Action(text="The threshold energy is \\boxed{2.6e5 GeV}")
result = env.step(result.next_state, action)
print(result.rewards.total)
```

### Filter by Domain

```python
# Only physics tasks
env = adapter.get_environment("sciagentgym:physics", data_path="path/to/data.json")

# Only chemistry tasks
env = adapter.get_environment("sciagentgym:chemistry", data_path="path/to/data.json")
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

### Per-Task Tool Sets

Unlike fixed-tool environments, each SciAgentGYM task defines its own set of 3-16 domain-specific tools. On `reset()`, the adapter loads the task's tools from `usage_tool_protocol` (OpenAI function-calling format) and converts them to `ToolDefinition` objects.

### Tool Execution Delegation

SciAgentGYM tools have internal state (history tracking, filesystem artifacts), so tool execution is delegated to `MinimalSciEnv.step()`. The adapter extends `BaseToolEnvironment` for tool infrastructure but lets SciAgentGYM handle all execution internally.

The flow:

1. `reset()` calls `prepare_env_from_query()` to create a `MinimalSciEnv` with tools for the task
2. Tools from `usage_tool_protocol` are converted to `ToolDefinition` via `oai_tools_to_definitions()`
3. `step()` converts llenvs `ToolCall` objects to SciAgentGYM `ToolCall` objects
4. Each tool call is executed via `MinimalSciEnv.step()` sequentially
5. Results are converted back to `ToolResult` objects

### Termination

SciAgentGYM's `MinimalSciEnv.step()` never sets `done=True`. The adapter terminates on:

- **Text-only action**: Treated as final answer submission (`terminated=True`)
- **Max steps**: Truncation after `max_steps` tool-calling turns (default 30)

### Rewards

Terminal steps extract `\boxed{}` from the model's response and compare to the gold answer:

- **Native scoring** (when SciAgentGYM is installed): Uses `calculate_answer_score()` which supports numeric tolerance (5%), structural comparison, and partial scores
- **Fallback scoring**: Uses llenvs `BoxedExtractor` with numeric tolerance and string comparison

Non-terminal steps return `Signal(reward=None, reward_type=STEP)`.

## Domains

| Subject | Examples |
|---|---|
| Physics | Particle physics, optics, thermodynamics |
| Chemistry | Molecular analysis, reaction kinetics |
| Materials Science | Crystal structure, material properties |
| Life Science | Molecular biology, genetics |
| Astronomy | Stellar physics, cosmology |
| Statistics | Statistical analysis, hypothesis testing |

## Parameters

### `SciAgentGymAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | `"sciagentgym"` or `"sciagentgym:<subject>"` for domain filtering |
| `dataset` | `list[dict] \| None` | Pre-loaded test case dicts |
| `data_path` | `str \| None` | Path to JSON file or directory of test cases |
| `subject` | `str \| None` | Filter by scientific domain |
| `max_steps` | `int` | Maximum steps per episode (default 30) |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes |
| `task_index` | Yes |
| `seed` | No |
| `compute_rewards` | Yes |
| `pure_step` | No (SciAgentGYM envs are stateful) |
| Tool calling | Yes (delegated to SciAgentGYM) |
| `Scorer` / `DatasetProvider` | No (multi-turn) |

## Limitations

- **Git-only install**: SciAgentGYM is not on PyPI; must install from the GitHub repository
- **No seed support**: SciAgentGYM doesn't expose a seeding mechanism
- **Tool execution**: Tools cannot be called independently of `step()` since they access internal environment state
- **Namespace conflict**: SciAgentGYM uses the `gym` package name, which may conflict with OpenAI Gym/Gymnasium
