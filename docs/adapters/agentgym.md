# AgentGym

[AgentGym](https://github.com/THUDM/AgentGym) (Xi et al., 2024) provides 14 diverse multi-turn agent environments — web navigation, text games, API tasks, and more — behind a unified client-server architecture. Each environment runs as a FastAPI server, and Python clients communicate via REST.

## Installation

```bash
# Install the agentenv client library
uv pip install -e ".[agentgym]"

# Install server packages for the environments you need
pip install agentenv-alfworld agentenv-babyai agentenv-lmrlgym
pip install agentenv-sciworld agentenv-sqlgym agentenv-textcraft
pip install agentenv-webarena agentenv-webshop agentenv-searchqa
pip install agentenv-tool  # movie, weather, academia, todo, sheet
```

## Quick Start

```python
from llenvs.adapters import AgentGymAdapter
from llenvs.core import Action

adapter = AgentGymAdapter()
env = adapter.get_environment("maze", max_steps=20)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
print(f"Dataset size: {len(env)}")

action = Action(text="go north")
result = env.step(state, action)
print(result.next_state.observation.prompt)
print(f"Reward: {result.rewards.total}")
```

## Environments

| Name | Description | Server Package |
|------|-------------|----------------|
| `webshop` | E-commerce product search and purchase | `agentenv-webshop` |
| `alfworld` | Embodied household tasks (TextWorld) | `agentenv-alfworld` |
| `babyai` | Grid-world navigation with language goals | `agentenv-babyai` |
| `maze` | Maze navigation | `agentenv-lmrlgym` |
| `wordle` | Word guessing game | `agentenv-lmrlgym` |
| `sciworld` | Science experiments in a virtual lab | `agentenv-sciworld` |
| `sqlgym` | SQL query generation | `agentenv-sqlgym` |
| `textcraft` | Minecraft-like crafting in text | `agentenv-textcraft` |
| `webarena` | Real-website navigation tasks | `agentenv-webarena` |
| `searchqa` | Information retrieval via search | `agentenv-searchqa` |
| `movie` | Movie database API tasks | `agentenv-tool` |
| `weather` | Weather API tasks | `agentenv-tool` |
| `academia` | Academic search API tasks | `agentenv-tool` |
| `todo` | Todo list management via API | `agentenv-tool` |
| `sheet` | Spreadsheet manipulation via API | `agentenv-tool` |

All environments are **multi-turn** — the agent receives text observations and responds with text actions until the episode ends or `max_steps` is reached.

## Server Setup

AgentGym uses a client-server architecture. There are two ways to manage servers:

### Automatic (default)

The adapter auto-starts servers from installed server packages. It finds a free port, launches the CLI command, and waits for the server to become ready.

```python
# Adapter handles server lifecycle automatically
env = adapter.get_environment("maze")
```

### Manual (pre-running servers)

Point the adapter at an already-running server with `env_server_base`:

```python
# Start server manually first:
#   alfworld --port 8080

env = adapter.get_environment(
    "alfworld",
    env_server_base="http://localhost:8080",
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Environment name (e.g. `"maze"`, `"alfworld"`) |
| `env_server_base` | `str \| None` | `None` | URL of pre-running server; `None` to auto-start |
| `data_len` | `int` | `100` | Number of tasks to load on the server |
| `max_steps` | `int` | `20` | Maximum steps per episode before truncation |
| `timeout` | `int` | `300` | HTTP request timeout in seconds |
| `action_format` | `str` | `"react"` | Action format: `"react"`, `"function_calling"`, or `"code_as_action"` |
| `extra_rewards` | `tuple` | `()` | Additional reward functions |
| `prompts` | `dict \| None` | `None` | Override prompt components (merged with auto-extracted prompts) |

## Action Formats

AgentGym clients support three action formats that change the system prompt and expected action structure:

| Format | Description |
|--------|-------------|
| `"react"` | ReAct-style reasoning and acting (default) |
| `"function_calling"` | Structured function call format |
| `"code_as_action"` | Python code as actions |

The adapter auto-extracts conversation prompts (`system_prompt`, `assistant_ack`) from the client's `_conversation_start` attribute for the selected format. User-provided `prompts` take precedence over auto-extracted values.

## Rewards

AgentGym provides a native reward signal from each environment's `step()` call:

| Signal | Type | Description |
|--------|------|-------------|
| `agentgym_native` | `STEP` / `OUTCOME` | Native reward; `STEP` for intermediate, `OUTCOME` for terminal |

## Hidden State

```python
@dataclass(frozen=True)
class AgentGymHidden:
    task_index: int                        # Current task index
    env_name: str                          # Environment name
    episode_step: int                      # Step within episode
    last_action: str | None                # Last action taken
    available_actions: tuple[str, ...] = () # Actions available (if provided by client)
```

The `available_actions` field is populated from `client.info["available_actions"]` when present. AlfWorld populates this on every reset/step; other environments return an empty tuple.

## Info Dicts

### `reset()` info

| Key | Description |
|-----|-------------|
| `task_index` | Task index used |
| `env_name` | Environment name |
| `action_format` | Active action format |
| `dataset_size` | Total number of tasks |
| `reset_*` | Keys from the client's `reset()` return value (prefixed) |
| `client_*` | Keys from `client.info` (prefixed, `"observation"` excluded) |
| `available_actions` | Available actions tuple (when non-empty) |

### `step()` info

| Key | Description |
|-----|-------------|
| `agentgym_reward` | Native reward value |
| `action` | Action taken |
| `done` | Whether the environment signaled done |
| `truncated` | Whether the episode was truncated by max_steps |
| `episode_step` | Current step number |
| `env_name` | Environment name |
| `client_*` | Keys from `client.info` (prefixed) |
| `available_actions` | Available actions tuple (when non-empty) |

## Using with TrajectoryRunner

```python
from llenvs.adapters import AgentGymAdapter
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

adapter = AgentGymAdapter()
env = adapter.get_environment("alfworld", max_steps=30)
backend = OpenAIBackend(model="gpt-4o")

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
    system_prompt="You are a household robot. Complete the given task.",
)

result = runner.run_trajectory(task_index=0)
print(f"Success: {result.success}")
print(f"Reward: {result.total_reward}")
```

## Limitations

| Limitation | Reason |
|------------|--------|
| No ground truth / expected answers | Interactive environments — reward is the only feedback signal |
| No deterministic `step()` | Client-server is inherently stateful; no snapshot/restore API |
| No seed support | `reset(idx)` is the only control; no seed parameter |
| No native answer extraction | Not applicable for multi-turn interactive environments |
