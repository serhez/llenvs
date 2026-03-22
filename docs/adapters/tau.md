# Tau Adapter

Wraps [tau-bench](https://github.com/sierra-research/tau2-bench) as llenvs MDP environments. tau-bench is a multi-turn customer service benchmark evaluating LLM agents across domains (airline, retail, telecom, banking_knowledge). It features heavy tool usage with stateful DB-backed tools, an LLM-powered user simulator, and multi-signal evaluation.

## Installation

```bash
pip install tau2
```

## Quick Start

```python
from llenvs.adapters.tau import TauAdapter

adapter = TauAdapter()

# Load tasks and environment from tau2's registry
env = adapter.get_environment("tau:airline", max_steps=50)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
print([t.name for t in state.observation.available_tools])

from llenvs.core import Action
from llenvs.core.tools import ToolCall

# Make a tool call
call = ToolCall(id="c1", name="get_user_details", arguments={"user_id": "USR001"})
action = Action(tool_calls=(call,))
result = env.step(state, action)
print(result.rewards.total)
```

### With Specific Task Split

```python
# Load specific splits (base, train, test) — available for airline, retail, telecom
env = adapter.get_environment("tau:airline:base", max_steps=50)
env = adapter.get_environment("tau:retail:test", max_steps=50)
```

### With Pre-loaded Tasks

```python
from tau2.registry import registry

tasks = registry.get_tasks_loader("airline")(task_split_name="base")
env_constructor = registry.get_env_constructor("airline")
tau2_env = env_constructor()

env = adapter.get_environment(
    "tau:airline",
    tasks=tasks,
    tau2_env=tau2_env,
    max_steps=50,
)
```

### With User Simulator

```python
from tau2.user.user_simulator import UserSimulator

user_sim = UserSimulator(llm="gpt-4o", llm_args={"temperature": 0.0})
env = adapter.get_environment(
    "tau:airline",
    user_simulator=user_sim,
    max_steps=50,
)
```

### Solo Mode

Solo mode disables user simulation. The agent interacts only via tool calls and terminates by including `###STOP###` in a text action.

```python
env = adapter.get_environment(
    "tau:airline",
    solo_mode=True,
    max_steps=50,
)
state, _ = env.reset(options={"task_index": 0})
# In solo mode, the prompt contains the task ticket
print(state.observation.prompt)  # "Fix order #12345..."

# Agent uses tools, then stops
action = Action(text="###STOP###")
result = env.step(state, action)
assert result.terminated
```

### Banking Knowledge Domain

The `banking_knowledge` domain features RAG-based knowledge retrieval tools. Extra kwargs are forwarded to tau2's environment constructor:

```python
env = adapter.get_environment(
    "tau:banking_knowledge",
    max_steps=30,
    retrieval_variant="qwen_embeddings_grep",
)
```

Note: `banking_knowledge` has no task splits and does not support solo mode.

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

tau-bench tools have access to internal domain databases (flight bookings, order records, telecom accounts, banking knowledge bases). Tool execution **must** be delegated to tau2's `Environment.make_tool_call()` since tools mutate shared state.

The flow:

1. `reset()` loads a task, initializes the domain environment and DB state
2. Tools are converted from tau2's OpenAI-format schemas to `ToolDefinition` objects with full `raw_schema` passthrough
3. `step()` with tool calls executes via `make_tool_call()`, converting results back to `ToolResult`
4. `step()` with text forwards to the user simulator and returns the user's response

### User Simulation

In multi-turn mode, the agent communicates with an LLM-powered user simulator. The user has a persona, instructions, and known/unknown information. The user signals end-of-conversation via `###STOP###`, `###TRANSFER###`, or `###OUT-OF-SCOPE###` tokens.

### Schema Fidelity

tau-bench tools have complex Pydantic-generated JSON schemas (arrays of objects, nested properties, discriminated unions). The adapter uses `raw_schema` passthrough on `ToolDefinition` to preserve these schemas exactly when converting to OpenAI or Anthropic format for the model backend.

### Evaluation

tau-bench provides multi-signal evaluation:

| Signal | Description |
|---|---|
| **DB** | Database state matches expected state after actions |
| **Action** | Expected tool calls were made with correct arguments |
| **Communicate** | Agent communicated required information to user |
| **ENV_ASSERTION** | Programmatic environment state assertions |
| **NL Assertion** | LLM-judged natural language assertions |

The default `TauReward` returns the aggregate score. For per-criterion breakdown, use `TauDetailedRewards` via `extra_rewards`.

```python
from llenvs.adapters.tau import TauDetailedRewards

env = adapter.get_environment(
    "tau:airline",
    extra_rewards=(TauDetailedRewards(),),
)
# After terminal step, inspect detailed reward metadata:
# result.rewards.by_name("tau_detailed").metadata
# => {"db_reward": 1.0, "action_reward": 0.8, "communicate_reward": 1.0, ...}
```

## Domains

| Domain | Tools | Description |
|---|---|---|
| `airline` | Flight booking, cancellation, seat changes, baggage | Customer service for airline reservations |
| `retail` | Order management, returns, exchanges, account updates | E-commerce customer support |
| `telecom` | Account management, billing, tech support, plan changes | Telecommunications customer service |
| `banking_knowledge` | Knowledge retrieval, document search, account tools | Banking support with RAG-based knowledge base |

## Parameters

### `TauAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | `"tau:<domain>"` or `"tau:<domain>:<split>"` |
| `tasks` | `list \| None` | Pre-loaded tau2 Task objects |
| `task_split` | `str \| None` | Task split name (base, train, test) |
| `tau2_env` | `Any \| None` | Pre-created tau2 Environment |
| `max_steps` | `int \| None` | Maximum steps per episode |
| `solo_mode` | `bool` | Disable user simulator (tool-only) |
| `user_simulator` | `Any \| None` | Pre-created UserSimulator |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |
| `**kwargs` | | Forwarded to tau2's env constructor (e.g., `retrieval_variant`) |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes |
| `task_index` | Yes |
| `seed` | No |
| `compute_rewards` | Yes (multi-signal) |
| `pure_step` | No (DB-backed state) |
| Tool calling | Yes (delegated to tau2) |
| User simulation | Yes (LLM-powered) |
| Solo mode | Yes (not banking_knowledge) |
| `Scorer` / `DatasetProvider` | No (multi-turn) |

## Limitations

- **User simulator requires LLM**: The user simulator needs an LLM API (e.g., GPT-4o) for realistic user responses
- **DB state**: Tools mutate shared database state; environments are not pure
- **Evaluation requires LLM**: NL assertion checks use an LLM judge
- **No native answer extraction**: tau-bench uses tool-action + DB-state evaluation, not text answer extraction
- **banking_knowledge**: Does not support solo mode; has no task splits
