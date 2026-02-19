# MARE Adapter

Wraps [Meta Agents Research Environments (ARE)](https://github.com/facebookresearch/meta-agents-research-environments) as llenvs MDP environments. ARE powers the [Gaia2 benchmark](https://arxiv.org/abs/2602.11964) with 800 scenarios testing execution, search, ambiguity, adaptability, and time-sensitivity across 10 "universes" featuring 5 simulated apps (email, calendar, contacts, shopping, file system) with ~101 tools.

## Installation

```bash
pip install git+https://github.com/facebookresearch/meta-agents-research-environments.git
```

## Quick Start

```python
from llenvs.adapters.mare import MAREAdapter

adapter = MAREAdapter()

# Load scenarios (from ARE's built-in loaders or custom)
from meta_agents_research_environments.scenarios import load_scenarios
scenarios = load_scenarios("gaia2")

env = adapter.get_environment("mare", scenarios=scenarios, max_steps=30)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
print([t.name for t in state.observation.available_tools])

from llenvs.core import Action
from llenvs.core.tools import ToolCall

call = ToolCall(id="c1", name="send_email", arguments={"to": "alice@example.com", "subject": "Meeting", "body": "See you at 3pm"})
action = Action(text="", tool_calls=(call,))
result = env.step(state, action)
print(result.rewards.total)
```

### With Capability Filter

```python
# Filter scenarios by capability type
env = adapter.get_environment("mare:execution", scenarios=scenarios)
env = adapter.get_environment("mare:search", scenarios=scenarios)
env = adapter.get_environment("mare:time_sensitivity", scenarios=scenarios)
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

ARE tools have access to internal environment state (app instances, event queues). Tool execution **must** be delegated to ARE's `tool.forward()` rather than extracted and called directly. The adapter extends `BaseToolEnvironment` for tool infrastructure (definitions, monitoring rewards) but lets ARE handle all tool execution internally.

The flow:

1. `reset()` loads a scenario, initializes apps, starts the event loop
2. Tools are converted from ARE format to `ToolDefinition` objects via `_mare_tools_to_definitions()`
3. `step()` executes tool calls via `tool.forward(**args)`, ticks the environment, and collects notifications
4. Notifications from the environment (e.g., new emails, calendar reminders) are delivered as system messages
5. Write actions (send, create, delete, etc.) are tracked for validation scoring

### Event Loop and Notifications

ARE environments run an asynchronous event loop. After each step, the adapter calls `tick()` to advance the simulation and collects pending notifications. These appear in the next observation as system messages, simulating real-time events the agent must respond to.

### Write Action Tracking

The adapter automatically classifies tool calls as "write actions" based on name prefixes (e.g., `send_`, `create_`, `delete_`). These are tracked in hidden state and used for validation scoring at episode end.

### Scenario Validation Scoring

Terminal rewards use ARE's scenario validation, which compares tracked write actions against oracle annotations. This includes both hard matching (exact action comparison) and soft matching (LLM judge for semantic equivalence).

## Capabilities

| Capability | Description |
|---|---|
| `execution` | Direct task completion (send email, create event) |
| `search` | Information retrieval across apps |
| `ambiguity` | Tasks with underspecified requirements |
| `adaptability` | Tasks requiring adjustment to changing conditions |
| `time_sensitivity` | Tasks with time pressure from notifications |

## Parameters

### `MAREAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | `"mare"` or `"mare:<capability>"` |
| `scenarios` | `list \| None` | Pre-loaded scenario objects |
| `scenario_loader` | `callable \| None` | Function returning scenarios |
| `max_steps` | `int \| None` | Maximum steps per episode |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |
| `capability` | `str \| None` | Filter scenarios by capability |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes |
| `task_index` | Yes |
| `seed` | Yes (scenarios have seed field) |
| `compute_rewards` | Yes (validation-based) |
| `pure_step` | No (ARE envs are stateful) |
| Tool calling | Yes (delegated to ARE) |
| Notifications | Yes (event-driven) |
| `Scorer` / `DatasetProvider` | No (multi-turn) |

## Limitations

- **Async-first**: ARE is async-first; the adapter runs coroutines synchronously via `run_async()` with a thread pool fallback for nested event loops
- **No native answer extraction**: ARE uses write-action validation, not text answer extraction
- **Tool execution**: Tools cannot be called independently of the adapter since they access internal app state
- **External dependency**: ARE is not on PyPI; install from GitHub
