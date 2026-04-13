# Multi-Instance Runner

When estimating Q-values via Monte Carlo rollouts from saved mid-trajectory states, the standard runner assumes `pure_step=True` — a single environment instance can replay from any state. For container-based environments (Harbor, Docker runtimes) where `pure_step=False`, each rollout needs its own independent environment instance restored to the target state.

The multi-instance runner path handles this transparently: create per-rollout environment instances, restore each to the target state via trajectory replay, then run lockstep batched rollouts with shared LLM inference.

## When to Use

Use `env_factory` + `restore_fn` when:

- The environment has `pure_step=False` (mutable backend state)
- You need `run_batch_from_states()` or `run_batch_from_state_actions()` (MC rollouts, Q-value estimation)
- Each rollout must operate on an independent environment instance

## Configuration

```python
from llenvs.evaluation.runner import TrajectoryRunner
from llenvs.adapters.harbor import HarborAdapter, harbor_restore

adapter = HarborAdapter()

# Primary env for trajectory collection
env = adapter.get_environment("terminal-bench@2.0", max_steps=30)

# Factory creates fresh, independent instances
def env_factory():
    return adapter.get_environment("terminal-bench@2.0", max_steps=30)

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=sampling_params,
    env_factory=env_factory,
    restore_fn=harbor_restore,
)
```

## How It Works

When `run_batch_from_states()` is called and the environment is non-pure with an `env_factory`:

1. **Create instances**: One fresh environment per rollout in the batch
2. **Restore state**: Each instance is restored to the target state via `restore_fn` (in parallel via `ThreadPoolExecutor`)
3. **Lockstep loop**: All active rollouts advance one step together — LLM generation is batched (single GPU call), environment steps run in parallel (I/O-bound container operations)
4. **Cleanup**: All instances are closed in a `try/finally` block, even on error

## Writing a Restore Function

A restore function takes `(env, state)` and returns the restored state. For Harbor, prefer the built-in `harbor_restore` helper instead of open-coding the replay loop:

```python
from llenvs.adapters.harbor import harbor_restore

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=sampling_params,
    env_factory=env_factory,
    restore_fn=harbor_restore,
)
```

The built-in helper validates task name consistency to guard against index drift across dataset versions, aborts on shell continuation prompts, replays the prefix with the live `trajectory_timeout` budget disabled, and re-anchors that budget before the restored continuation begins. A naive loop over `env.step(...)` is incomplete for Harbor because it can silently spend the continuation budget during replay.

## Concurrency Tuning

Concurrency is controlled by `batch_size` on `run_batch_from_states()`. Each item in a batch gets its own environment instance, so `batch_size` directly caps concurrent containers:

```python
# At most 4 containers running simultaneously
trajectories = runner.run_batch_from_states(
    states,
    batch_size=4,
)
```

When `batch_size` is set and there are more states than `batch_size`, states are processed in sequential chunks. Within each chunk, environment operations (restore, step) run in parallel threads while LLM generation is batched.

## Forced Actions (Q-Value Estimation)

`run_batch_from_state_actions()` works identically — the first step of each rollout uses the forced action instead of generating from the backend:

```python
# Q(s, a) estimation: force first action, then rollout
trajectories = runner.run_batch_from_state_actions(
    states=[saved_state] * num_rollouts,
    actions=[Action(text=candidate_action)] * num_rollouts,
    batch_size=8,
)
```

## Replay Validation

For container-based environments, replay consistency is not guaranteed (network-dependent commands, non-deterministic outputs). Use `validate_replay_consistency()` to test whether replaying a trajectory produces consistent state:

```python
from llenvs.adapters.harbor import validate_replay_consistency

result = validate_replay_consistency(
    env_factory=env_factory,
    task_index=0,
    trajectory=("apt-get update", "pip install pandas"),
    probe_commands=("dpkg -l | md5sum",),
    num_trials=3,
)

if result["consistent"]:
    print("Task is replay-safe")
```

See the [Harbor adapter docs](../adapters/harbor.md) for details on replay validation modes.

## Pure-Step Bypass

When `env_factory` is set but the environment has `pure_step=True`, the multi-instance path is bypassed — the standard single-instance lockstep loop is used instead. This means you can safely set `env_factory` on a runner that may be used with both pure and non-pure environments.
