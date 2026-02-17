# InterCode Adapter

Wraps [InterCode](https://github.com/princeton-nlp/intercode) (Princeton NLP, NeurIPS 2023) as llenvs MDP environments for interactive code generation evaluation.

## Installation

InterCode is not included as a formal pip extra due to legacy dependency pins. Install separately:

```bash
pip install intercode-bench
# OR from git:
pip install git+https://github.com/princeton-nlp/intercode.git
```

## Quick Start

```python
from llenvs.adapters.intercode import InterCodeAdapter

adapter = InterCodeAdapter()

# With a pre-created InterCode environment
from intercode.envs import BashEnv
ic_env = BashEnv(data_path="path/to/tasks.json", image_name="intercode-nl2bash")
env = adapter.get_environment("intercode:bash", intercode_env=ic_env)

# Or let the adapter create one
env = adapter.get_environment("intercode:bash", data_path="path/to/tasks.json")

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)

from llenvs.core import Action

# Execute commands
action = Action(text="ls -la /home")
result = env.step(state, action)
print(result.next_state.observation.messages[-1]["content"])

# Submit final answer
action = Action(text="submit wc -l file.txt")
result = env.step(result.next_state, action)
print(result.rewards.total)
```

### With Evaluation Runner

```python
from llenvs.evaluation.runner import run_evaluation

result = run_evaluation(
    environment=env,
    backend=backend,
    task_indices=list(range(50)),
)
print(f"Success rate: {result.success_rate}")
```

## How It Works

### Text-Based Actions

InterCode uses free-form text commands — there is no tool schema. The agent sends a text string (e.g., `ls -la`, `SELECT * FROM users`, `print(42)`) which gets executed in a Docker container. This matches InterCode's native API and avoids unnecessary tool-call wrapping.

### Docker Containers

InterCode manages its own Docker containers internally (one per env type, persistent across steps). This is separate from llenvs' container infrastructure (`EnvironmentServer`, `DockerRuntime`), which runs llenvs environments remotely. These are orthogonal concerns.

### Termination

Episodes end when:

- **InterCode signals done**: The `step()` call returns `done=True` (e.g., after a `submit` command)
- **Max steps**: Truncation after `max_steps` turns (default 10)

### Rewards

- **Non-terminal steps**: `Signal(reward=None, reward_type=STEP)` — InterCode only scores on completion
- **Terminal steps**: `Signal(reward=float, reward_type=OUTCOME)` — reward value comes from InterCode's info dict

### Hidden State

`InterCodeHidden` tracks:

| Field | Description |
|---|---|
| `task_index` | Index in the dataset |
| `env_type` | Environment type (bash, sql, python, ctf) |
| `query` | Task description from the dataset |
| `gold` | Reference solution |
| `episode_step` | Current step count |
| `last_action` | Last command executed |
| `cumulative_reward` | Sum of rewards across steps |
| `trajectory` | Tuple of all actions taken |

## Environment Types

| Type | Description | Docker Image |
|---|---|---|
| `bash` | Bash shell commands | `intercode-nl2bash` |
| `sql` | SQL queries against databases | `intercode-nl2sql` |
| `python` | Python code execution | `intercode-python` |
| `ctf` | Capture-the-flag challenges | `intercode-ctf` |

## Parameters

### `InterCodeAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | `"intercode:<type>"` where type is bash, sql, python, or ctf |
| `intercode_env` | `Any \| None` | Pre-created InterCode environment instance |
| `data_path` | `str \| None` | Path to task dataset (required if `intercode_env` not provided) |
| `image_name` | `str \| None` | Docker image name override |
| `max_steps` | `int` | Maximum steps per episode (default 10) |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes |
| `task_index` | Yes |
| `seed` | No |
| `compute_rewards` | Yes |
| `pure_step` | No (Docker containers are stateful) |
| Tool calling | No (text-based commands) |
| `Scorer` / `DatasetProvider` | No (multi-turn) |

## Limitations

- **Docker required**: InterCode environments run in Docker containers that must be available
- **Git/pip install**: InterCode may have legacy dependency pins; install separately
- **No seed support**: InterCode doesn't expose a seeding mechanism
- **No tool calling**: Despite executing code, InterCode uses text commands, not structured tool calls
