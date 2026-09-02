# Verifiers v1 Adapter

Wraps [verifiers](https://github.com/PrimeIntellect-ai/verifiers) **v1** tasksets and environments as llenvs MDP environments — including multi-agent envs, whose untrainable seats are driven by `env_llm`. The legacy v0 API is served by the sibling [Verifiers adapter](verifiers.md); both APIs ship in the same installed `verifiers` package.

The opposite direction — exposing llenvs environments *to* verifiers, GEPA, and prime-rl — is the `llenvs_env` plugin package described in the [prime-rl / verifiers v1 Training](../guides/prime-rl.md) guide.

## Installation

```bash
uv pip install -e ".[verifiers]"
```

Requires a verifiers release that ships the `verifiers.v1` API. Note that verifiers v1 has no stability contract; the adapter funnels every touchpoint through one internal seam, but a verifiers upgrade can still require adapter updates.

## Environment Types

verifiers v1 resolves a taskset id to an installed Python module (dashes become underscores) exporting one `Taskset` subclass — and optionally one `Env` subclass — via `__all__`. The adapter routes on the resolved `Env` class:

| verifiers routing | llenvs Wrapper | Turn Type |
|---|---|---|
| `SingleAgentEnv` (plain tasksets) | `VerifiersV1SingleTurnEnvironment` | Single-turn, pure-step |
| Custom `Env` subclass (taskset-exported or explicitly paired) | `VerifiersV1MultiTurnEnvironment` | Multi-turn relay via an episode bridge |

The `name` argument is a taskset id, optionally paired as `"<env_id>+<taskset_id>"` (verifiers' own env-id format); an explicit `env_id` kwarg wins over the name's prefix.

## Quick Start

### Single-Turn Taskset

```python
from llenvs.adapters.verifiers_v1 import VerifiersV1Adapter

adapter = VerifiersV1Adapter()
env = adapter.get_environment("my-taskset", task_params={"judges": []})

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)  # Task prompt
print(info["system_prompt"])     # Task/taskset system prompt

from llenvs.core import Action
result = env.step(state, Action(text="The answer is 42"))
print(result.rewards.total)              # Weighted sum of the task's rewards
print(result.info["verifiers_metrics"])  # @vf.metric values
```

### Multi-Turn Env with an Untrainable Seat

A custom v1 `Env` runs its own episode loop (`Env.run`). The adapter inverts it into reset/step on a per-episode thread; env-driven seats (e.g. a simulated user) need `env_llm`:

```python
env = adapter.get_environment(
    "user-sim-taskset",
    env_llm=backend,             # drives seats left untrainable by Env.setup()
    system_prompt="You are the user.",  # fallback if the seat's task has none
    max_steps=20,                # truncates the episode after 20 policy turns
    step_timeout=120.0,          # seconds a step may wait on the episode thread
)

state, info = env.reset(options={"task_index": 0})
result = env.step(state, Action(text="Hello!"))
while not result.done:
    result = env.step(result.next_state, Action(text=...))
print(result.rewards.total)               # Terminal-only, per-name signals
print(result.info["all_trace_rewards"])   # Every seat's scored rewards
```

### YAML Configuration

```yaml
environments:
  - name: user-sim-taskset
    adapter: verifiers_v1
    params:
      task_params:
        judges: []
      max_steps: 20
    env_llm:
      model:
        backend: openai
        model: gpt-4o-mini
      inference:
        temperature: 0.0
        max_tokens: 512
```

## How It Works

### Offline Scoring

The adapter never runs verifiers' rollout/interception server. Traces are token-free message graphs (a sanctioned verifiers mode): each policy turn commits the prompt messages plus the assistant reply, and scoring is `Task.score(trace, runtime=None)` — verifiers' offline mode, in which runtime-requiring reward/metric hooks are skipped (their names surface in `info["runtime_skipped_signals"]`).

Each named trace reward becomes one `OUTCOME` Signal carrying its native weight, so `rewards.total` reproduces verifiers' weighted `trace.reward` exactly. `@vf.metric` values go to `info["verifiers_metrics"]` (metadata, not signals). `extra_rewards` are appended after the native signals.

### The Episode Bridge (multi-turn)

A custom `Env` owns its loop, so the adapter runs each episode on a daemon thread: `reset()` starts the env's real `setup -> run -> finalize` sequence against stub seats, and every trainable-seat turn parks on a future until the next `step()` supplies the policy's reply. Non-terminal steps carry an empty reward bundle; the terminal step carries the scored rewards, `info["stop_condition"]`, `info["episode_ok"]`, and every seat's rewards in `info["all_trace_rewards"]`.

Exactly one seat may remain trainable after `Env.setup()`. Untrainable seats are driven by `env_llm`: the seat's conversation is rebuilt from its own trace, its task's `system_prompt` wins over the adapter-level fallback, and a seat's `AgentConfig.sampling` (temperature/top_p/max_tokens) overlays `sampling_params`.

Because verifiers enforces `max_turns` and `@vf.stop` hooks in its interception server, the adapter replicates that check before each turn: `max_steps` maps to it, and Trace-boundary `@vf.stop` hooks run offline. A limit-triggered end (`max_turns`, token caps) maps to `truncated`; every other stop terminates.

Episodes are stateful (`pure_step=False`): only the latest state can be stepped, and a stale state raises `NotImplementedError`. `close()` abandons an in-flight episode; `reset()` and a GC finalizer call it implicitly. An episode abandoned mid-turn leaves its daemon thread parked until process exit if a hung `env_llm` call cannot be interrupted — `step_timeout` bounds the caller, not the env.

### Task Materialization

Finite tasksets are materialized to a list for indexed access (`__len__`, `task_index`). `INFINITE` tasksets require `size` (applied as `head(size)`); `seed` opts into `Taskset.shuffle(seed)`. `taskset_params` feed the taskset's typed config, and `task_params` nest under its `task` field — e.g. `task_params={"judges": []}` disables judge rewards.

## Parameters

### `VerifiersV1Adapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Taskset id, or `"<env_id>+<taskset_id>"` pair |
| `env_id` | `str` | Explicit env plugin id (overrides the name's prefix) |
| `taskset_params` | `dict \| None` | Extra fields for the taskset's config |
| `task_params` | `dict \| None` | Task config fields (nested under the taskset config's `task`) |
| `env_params` | `dict \| None` | Extra fields for the env's config (multi-turn only) |
| `env_llm` | `ModelBackend \| None` | Backend driving untrainable seats (multi-turn only) |
| `sampling_params` | `SamplingParams \| None` | Sampling for `env_llm` (seat configs overlay it) |
| `system_prompt` | `str \| None` | Fallback system prompt for `env_llm` seats |
| `size` | `int \| None` | Materialize at most this many tasks (required for INFINITE tasksets) |
| `seed` | `int \| None` | Opt-in `Taskset.shuffle(seed)` |
| `max_steps` | `int \| None` | Cap on policy turns per episode (exceeding it truncates) |
| `step_timeout` | `float \| None` | Seconds a reset/step may wait on the episode thread |
| `answer_extractor` | `AnswerExtractor \| None` | Optional extractor (single-turn only) |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions, appended after native |

`list_environments()` returns the taskset ids shipped inside verifiers itself; third-party taskset plugins are installed packages and are passed by id directly.

### Environment Capabilities

| Feature | Single-Turn | Multi-Turn |
|---|---|---|
| `__len__` / `task_index` | Yes | Yes |
| `seed` (reset) | No (use the `seed` kwarg) | No (use the `seed` kwarg) |
| `pure_step` | Yes | No (live episode thread) |
| Terminal rewards | Per-name native signals | Per-name native signals |
| System prompt | Yes | Yes |
| Answer extraction | Optional | No |
| Tool calls | No (chat relay) | No (chat relay) |
| `close()` | Not needed | Yes (also runs on GC/reset) |

## Limitations

The adapter refuses, with explicit errors, features that require verifiers' interception server or container runtimes:

- **`NEEDS_CONTAINER` tasks** (containerized environments) — no runtime provisioning
- **Taskset `toolsets()`** (MCP tool servers) and **`Agent.provision()`**
- **`@intercept` hooks** and **Request/Response-boundary `@vf.stop` hooks** (Trace-boundary stops run offline)
- **Image tasks** — the adapter is text-only
- **More than one trainable seat**, and **concurrent trainable turns** within an episode

Further fidelity notes:

- **Token caps are inert**: traces are token-free, so `max_input_tokens`-style limits never fire; only `max_turns` (via `max_steps`) is enforced offline.
- **Task `setup()`/`finalize()` overrides are skipped** (they need a Runtime); a one-time warning names them. Env-level `setup`/`run`/`finalize` run normally.
- **Judge rewards call real endpoints**: judge-backed `@vf.reward` hooks build their own API clients (`PRIME_API_KEY`/`PRIME_INFERENCE_URL`); pass `task_params={"judges": []}` to disable them.
- **Runtime-requiring reward/metric hooks are skipped** in offline scoring and surfaced in `info["runtime_skipped_signals"]`.
