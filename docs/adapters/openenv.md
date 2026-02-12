# OpenEnv Adapter

Wraps [OpenEnv](https://github.com/meta-pytorch/OpenEnv) session-based environments as llenvs MDP environments.

## Installation

```bash
uv pip install -e ".[openenv]"
```

## Key Design: Session-Based

OpenEnv environments are **server-backed sessions**, not indexed datasets. There are no task indices, no `__len__`, and no seed support. Each `reset()` creates a fresh session on the server regardless of any `task_index` passed.

To run N episodes, pass `task_indices=list(range(N))` — the indices serve only as episode identifiers, not dataset lookups.

## Quick Start

### Text Environment

```python
from llenvs.adapters.openenv import OpenEnvAdapter

adapter = OpenEnvAdapter()
env = adapter.get_environment(
    "my-env",
    base_url="http://localhost:8000",
)

state, info = env.reset()
print(state.observation.prompt)  # Server's initial observation

from llenvs.core import Action
result = env.step(state, Action(text="go north"))
print(result.next_state.observation.messages[-1])  # Server response
```

### MCP Tool Environment

```python
env = adapter.get_environment(
    "tool-env",
    base_url="http://localhost:8000",
    use_tools=True,
)

state, info = env.reset()
print(env.available_tools)  # Fetched from server via list_tools()

from llenvs.core.tools import ToolCall
call = ToolCall(id="c1", name="search", arguments={"query": "hello"})
action = Action(text="", tool_calls=(call,))
result = env.step(state, action)
```

### Running Multiple Episodes

```python
from llenvs.evaluation.runner import TrajectoryRunner

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
)

# Run 10 fresh episodes (indices are just identifiers)
result = runner.run_batch(task_indices=list(range(10)))
print(f"Mean reward: {result.mean_reward}")
```

## How It Works

### Server Connection

The adapter connects to a running OpenEnv server via URL. You start the server separately (via Docker, OpenEnv CLI, or manually). The adapter uses:

- **`GenericEnvClient`** for text-based environments (simulation mode)
- **`MCPToolClient`** for tool-enabled environments (MCP/production mode)

Both are wrapped in synchronous clients internally.

### Observation Coercion

OpenEnv returns observation dicts. The adapter checks common keys in priority order:

1. `text` key
2. `content` key
3. `observation` key
4. `message` key
5. Falls back to JSON serialization

### Rewards

`OpenEnvReward` reads `StepResult.reward` from the server response:

- **Non-terminal steps**: `RewardType.STEP`
- **Terminal steps**: `RewardType.OUTCOME`

The reward value is stored in `state.metadata.info["openenv_reward"]`.

### Action Formatting

By default, actions are sent as `{"text": action_text}`. Use `action_format` for custom formatting:

```python
env = OpenEnvEnvironment(
    client=client,
    env_name="custom",
    action_format=lambda text: {"command": text, "type": "text"},
)
```

## Parameters

### `OpenEnvAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Environment name (identification only) |
| `base_url` | `str` | URL of running OpenEnv server. **Required.** |
| `use_tools` | `bool` | Use MCPToolClient for MCP tool support |
| `max_steps` | `int \| None` | Maximum steps per episode |
| `action_format` | `Callable` | Transform action text before sending to server |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | No (session-based) |
| `task_index` | No (indices ignored, fresh sessions) |
| `seed` | No |
| `compute_rewards` | Yes (from native step reward) |
| `Scorer` / `DatasetProvider` | No (no task indices or ground truth) |
| State snapshots / branching | No (mutable server-side state) |
| MCP tools | Yes (via `list_tools()` / `call_tool()`) |

## Limitations

- **No Docker management**: The adapter only connects to running servers. Start servers separately via OpenEnv CLI or Docker.
- **No state snapshots**: Server-side state is mutable; branching and checkpointing are not supported.
- **No ground truth**: OpenEnv environments don't expose expected answers, so `Scorer` and `DatasetProvider` cannot be used.
- **Requires running server**: Unlike other adapters, you must have a server running before creating the environment.
