# Harbor Adapter

Wraps [Harbor](https://github.com/laude-institute/harbor) containerized evaluation environments as llenvs MDP environments. Harbor is a generic framework by the Laude Institute for containerized agent evaluation. It manages Docker containers, task discovery via a JSON registry, and verification (test scripts produce binary pass/fail rewards).

By wrapping Harbor (not individual benchmarks), this adapter provides access to Terminal-Bench, aider-polyglot, swe-bench, and other datasets through a single interface.

## Installation

```bash
pip install harbor
```

Docker is required for running containers.

## Quick Start

### Text Mode

Agents send shell commands as plain text and receive stdout/stderr:

```python
from llenvs.adapters.harbor import HarborAdapter

adapter = HarborAdapter()
env = adapter.get_environment("terminal-bench@2.0", max_steps=30)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)  # Task instruction

from llenvs.core import Action

# Execute commands
result = env.step(state, Action(text="ls -la"))
state = result.next_state

result = env.step(state, Action(text="cat secret.enc"))
state = result.next_state

# Submit for verification
result = env.step(state, Action(text="SUBMIT"))
print(result.rewards.total)  # 1.0 or 0.0
```

### Tool Mode

Agents use structured tool calls for better model compatibility:

```python
env = adapter.get_environment("terminal-bench@2.0", tool_mode=True, max_steps=30)

state, info = env.reset(options={"task_index": 0})
print([t.name for t in state.observation.available_tools])
# ['execute_command', 'read_file', 'write_file', 'submit']

from llenvs.core.tools import ToolCall

call = ToolCall(id="c1", name="execute_command", arguments={"command": "ls -la"})
result = env.step(state, Action(tool_calls=(call,)))
state = result.next_state

# Submit
call = ToolCall(id="c2", name="submit", arguments={})
result = env.step(state, Action(tool_calls=(call,)))
print(result.rewards.total)
```

### With Pre-loaded Tasks

```python
import harbor

tasks = tuple(sorted(harbor.load_tasks("/path/to/dataset"), key=lambda t: t.name))
env_factory = lambda task: harbor.create_environment(task, environment_type="docker")
verifier_factory = lambda task, env: harbor.create_verifier(task, env)

env = adapter.get_environment(
    "custom-dataset",
    tasks=tasks,
    harbor_env_factory=env_factory,
    verifier_factory=verifier_factory,
    max_steps=50,
)
```

## How It Works

### Interaction Modes

The adapter provides two interaction modes, both sharing the same hidden state, reward function, and adapter class:

- **Text mode** (`HarborEnvironment`): Agent sends shell commands as `Action(text="...")`. Natural for terminal interaction. Submit via keyword.
- **Tool mode** (`HarborToolEnvironment`): Agent uses structured `ToolCall` objects. Better for models with strong function-calling capabilities. Inherits `BaseToolEnvironment` for validation, message building, and monitoring.

The `tool_mode` parameter on `get_environment()` selects the mode.

### Docker

Container lifecycle is delegated entirely to Harbor. Harbor supports multiple providers (Docker, Daytona, E2B, Modal). The adapter creates and starts containers via `harbor_env_factory`, executes commands via `env.exec()`, and stops containers on `close()` or `reset()`.

### Task Discovery

Tasks are loaded from Harbor's registry by dataset name and optional version. They are sorted alphabetically by name for deterministic task indexing.

### Termination

Episodes end in one of three ways:

1. **Submit** — text mode: action contains the submit keyword; tool mode: agent calls the `submit` tool.
2. **Truncation** — episode reaches `max_steps`.
3. Both trigger verification (truncation verification is configurable via `verify_on_truncation`).

### Verification

At terminal steps, the verifier runs test scripts inside the container and produces binary pass/fail rewards. The reward value (0.0 or 1.0) is stored in `next_state.metadata.info["reward"]`.

### Rewards

| Signal | Type | When | Value |
|---|---|---|---|
| `harbor` | `STEP` | Non-terminal steps | `None` |
| `harbor` | `OUTCOME` | Terminal steps | Verifier result (0.0 or 1.0) |

Tool mode additionally includes weight-0 monitoring rewards (`ToolValidityReward`, `ToolEfficiencyReward`).

## Tool Definitions

Available in tool mode:

| Tool | Parameters | Terminal | Description |
|---|---|---|---|
| `execute_command` | `command` (str), `cwd` (str, optional), `timeout` (int, optional) | No | Run a shell command |
| `read_file` | `path` (str) | No | Read file contents |
| `write_file` | `path` (str), `content` (str) | No | Write file to container |
| `submit` | *(none)* | Yes | Signal task completion |

## Hidden State

`HarborHidden` is a frozen dataclass:

| Field | Type | Description |
|---|---|---|
| `task_index` | `int` | Index into the task list |
| `task_name` | `str` | Harbor task identifier |
| `instruction` | `str` | Task instruction text |
| `episode_step` | `int` | Current step in the episode |
| `last_action` | `str \| None` | Text of the last action |
| `trajectory` | `tuple[str, ...]` | Command history |

## Parameters

### `HarborAdapter.get_environment()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"terminal-bench@2.0"` | Dataset name with optional version (`"dataset@version"`) |
| `tasks` | `tuple` | `None` | Pre-loaded Harbor Task objects |
| `harbor_env_factory` | callable | `None` | `(task) -> BaseEnvironment` factory |
| `verifier_factory` | callable | `None` | `(task, env) -> Verifier` factory |
| `dataset_path` | `str` | `None` | Local path to dataset directory |
| `environment_type` | `str` | `"docker"` | Harbor environment type |
| `tool_mode` | `bool` | `False` | Use structured tools instead of text |
| `max_steps` | `int` | `30` | Maximum steps per episode |
| `submit_keyword` | `str` | `"SUBMIT"` | Text mode submit keyword |
| `exec_timeout` | `int` | `120` | Per-command timeout in seconds |
| `verify_on_truncation` | `bool` | `True` | Run verifier when truncating |
| `extra_rewards` | `tuple` | `()` | Additional reward functions |

## Capabilities

| Feature | Status |
|---|---|
| Multi-turn | Yes |
| Task indexing | Yes |
| `__len__` | Yes |
| Seed support | No (tasks are deterministic) |
| History control | Automatic (via runner) |
| Branching | `ProcessForkStrategy` or `ActionReplayStrategy` |
| Judge rewards | Via `extra_rewards` |
| RL training | `DatasetProvider` works; `Scorer` rejects multi-turn |
| Tool monitoring | Automatic in tool mode (weight=0.0) |
| Evaluation logging | Automatic via runner |

## Available Datasets

Harbor's registry provides access to multiple datasets. Common ones include:

- **terminal-bench** — 92 containerized terminal/shell tasks (cryptography, ML, sysadmin, data processing)
- **aider-polyglot** — Multi-language coding tasks
- **swe-bench** — Software engineering benchmark

Use `adapter.list_environments()` to query the registry for all available datasets.

## Limitations

- **No seed support** — Harbor tasks are deterministic (fixed Dockerfiles/test scripts).
- **Docker required** — Container runtime must be available.
- **Network-dependent** — Registry queries and image pulls require network access.
- **Binary rewards** — Native verifiers produce pass/fail only; use `extra_rewards` for finer-grained scoring (e.g., `JudgeReward`).
