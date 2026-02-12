# Branching

Branching enables creating independent environment copies from intermediate states. Given a state at step K, you can create N independent `(environment, state)` pairs that can each be stepped, branched further, or run to completion independently.

## Use Cases

- **Q-value estimation**: Try many continuations from the same state to estimate expected reward
- **Action exploration**: Try different actions from a decision point and compare outcomes
- **Success rate computation**: Branch N times from a checkpoint and measure how often a strategy succeeds
- **Recursive tree search**: Branch from branches to build exploration trees

## Strategy Tiers

llenvs auto-selects the best strategy for each environment:

| Rank | Strategy | Overhead | Fidelity | Applicability |
|------|----------|----------|----------|---------------|
| 1 | **Direct** | Zero | Perfect | `supports_branching=True` envs (ReasoningGym, HF, Verifiers ST, Dialogue, GEM) |
| 2 | **ProcessFork** | Fork cost (fast, COW) | Perfect | Any env on Unix (macOS/Linux) |
| 3 | **ActionReplay** | O(K) per branch | Perfect if deterministic | Envs with seed + task_index support |

## Quick Start

```python
from llenvs.core.branching import BranchManager

env = my_environment()
state, info = env.reset(seed=42, options={"task_index": 0})

# Step to an interesting state
actions = []
for i in range(5):
    action = Action.from_text(f"step-{i}")
    actions.append(action)
    result = env.step(state, action)
    state = result.next_state

# Checkpoint and branch
with BranchManager.create(env) as mgr:
    mgr.checkpoint("step5", state, actions=tuple(actions),
                   reset_options={"seed": 42, "task_index": 0})

    # Try 100 continuations
    results = []
    for _ in range(100):
        branch_env, branch_state = mgr.branch("step5")
        # ... run trajectory on branch_env from branch_state ...
```

## Strategies

### DirectStrategy

For environments with `spec.supports_branching=True`. `step()` is a pure function, so the same environment instance handles all branches. Zero cost.

```python
with BranchManager.create(env) as mgr:  # auto-selects Direct
    mgr.checkpoint("s0", state, actions=(), reset_options={})
    e1, s1 = mgr.branch("s0")  # e1 is env (same instance)
    e2, s2 = mgr.branch("s0")  # e2 is env (same instance)
```

### ProcessForkStrategy

For any environment on Unix. Uses `os.fork()` to snapshot the entire process, including mutable backend state. Each branch gets its own process with copy-on-write memory.

```python
with BranchManager.create(env, strategy="process_fork") as mgr:
    mgr.checkpoint("s0", state, actions=(), reset_options={})
    e1, s1 = mgr.branch("s0")  # e1 is a ContainerEnvironment proxy
    e2, s2 = mgr.branch("s0")  # e2 is a separate proxy
```

Each branch creates a forked server process. Resources are cleaned up on `release()`, `close()`, or context manager exit.

### ActionReplayStrategy

For deterministic environments with seed and task_index support. Creates a fresh environment, resets with the same parameters, and replays all stored actions.

```python
with BranchManager.create(env, strategy="action_replay",
                          env_factory=MyEnvClass) as mgr:
    mgr.checkpoint("s0", state, actions=tuple(actions),
                   reset_options={"seed": 42, "task_index": 0})
    e1, s1 = mgr.branch("s0")  # fresh env, replayed to step K
```

Requires `env_factory` (callable that creates a fresh env instance). Cost is O(K) per branch where K is the checkpoint depth.

## BranchManager API

### Creating

```python
# Auto-resolve strategy
mgr = BranchManager.create(env)

# Explicit strategy
mgr = BranchManager.create(env, strategy="direct")
mgr = BranchManager.create(env, strategy="process_fork")
mgr = BranchManager.create(env, strategy="action_replay", env_factory=MyEnv)
```

### Checkpointing

```python
mgr.checkpoint(name, state, actions, reset_options)
```

- `name`: String identifier for later `branch()` / `release()` calls
- `state`: The `State` to checkpoint
- `actions`: Tuple of `Action`s taken to reach this state (used by ActionReplay)
- `reset_options`: Dict with `seed` and `task_index` used in the original `reset()` call

Checkpointing the same name again overwrites the previous checkpoint.

### Branching

```python
branch_env, branch_state = mgr.branch("checkpoint_name")
```

Returns an independent `(environment, state)` pair. Can be called multiple times per checkpoint.

### Resource Management

```python
mgr.release("checkpoint_name")  # Release one checkpoint
mgr.close()                     # Release all checkpoints

# Or use as context manager (recommended)
with BranchManager.create(env) as mgr:
    ...  # All checkpoints released on exit
```

## Recursive Branching

Branches can be branched further using nested `BranchManager` instances:

```python
with BranchManager.create(env) as mgr:
    mgr.checkpoint("A", state_a, actions_a, opts)
    for i in range(3):
        b_env, b_state = mgr.branch("A")
        result = b_env.step(b_state, some_action)

        with BranchManager.create(b_env) as sub_mgr:
            sub_mgr.checkpoint("B", result.next_state, ..., ...)
            for j in range(3):
                bb_env, bb_state = sub_mgr.branch("B")
                # 9 total branches at depth 2
```

## Configuration

Set `branching_strategy` on `EnvironmentConfig` to specify a preference:

```yaml
environments:
  - name: webshop
    adapter: webshop
    branching_strategy: process_fork
```

```python
from llenvs.core.config import EnvironmentConfig

config = EnvironmentConfig(
    name="webshop",
    adapter="webshop",
    branching_strategy="process_fork",  # or "direct", "action_replay", None
)
```

When `None` (default), `BranchManager.create()` auto-resolves the best strategy.

## Per-Adapter Capabilities

| Adapter | `supports_branching` | Direct | ProcessFork | ActionReplay |
|---------|---------------------|--------|-------------|--------------|
| reasoning_gym | True | Yes | Yes | Yes |
| huggingface | True | Yes | Yes | Yes |
| gem | True | Yes | Yes | Yes |
| dialogue | True | Yes | Yes | Yes |
| verifiers (single-turn) | True | Yes | Yes | Yes |
| verifiers (tool) | False | No | Yes | No |
| webshop | False | No | Yes | No |
| agentgym | False | No | Yes | No |
| openenv | False | No | Yes | No |

ProcessForkStrategy works for all environments on Unix, making it the universal fallback for mutable-backend environments.

## Resource Management

ProcessForkStrategy creates server processes for each checkpoint and branch. These are cleaned up:

1. On `mgr.release("name")` for a specific checkpoint
2. On `mgr.close()` or context manager exit for all checkpoints
3. Via `atexit` handler as a safety net

Always use `BranchManager` as a context manager to ensure cleanup.
