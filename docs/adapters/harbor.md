# Harbor Adapter

Wraps [Harbor](https://github.com/laude-institute/harbor) containerized evaluation environments as llenvs MDP environments. Harbor is a generic framework by the Laude Institute for containerized agent evaluation. It manages Docker containers, task discovery via a JSON registry, and verification (test scripts produce binary pass/fail rewards).

By wrapping Harbor (not individual benchmarks), this adapter provides access to Terminal-Bench, aider-polyglot, swe-bench, and other datasets through a single interface.

## Installation

```bash
pip install harbor
```

Docker is required for Harbor's built-in container backends. On clusters where
Docker is unavailable, `llenvs` also supports a local `podman-hpc` Harbor
runtime.

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
from pathlib import Path
import uuid

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

tasks = tuple(
    sorted(
        (Task(p) for p in Path("/path/to/dataset").iterdir()),
        key=lambda t: t.name,
    )
)

def env_factory(task):
    trial_paths = TrialPaths(trial_dir=Path("trials") / str(uuid.uuid4()))
    trial_paths.mkdir()
    env = EnvironmentFactory.create_environment(
        type=EnvironmentType.DOCKER,
        environment_dir=task.paths.environment_dir,
        environment_name=task.name,
        session_id=str(uuid.uuid4()),
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )
    env.trial_paths = trial_paths
    return env

def verifier_factory(task, env):
    return Verifier(task=task, trial_paths=env.trial_paths, environment=env)

env = adapter.get_environment(
    "custom-dataset",
    tasks=tasks,
    env_factory=env_factory,
    verify_factory=verifier_factory,
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

Container lifecycle is delegated entirely to Harbor. Harbor supports multiple providers (Docker, Daytona, E2B, Modal). The adapter creates and starts containers via `env_factory`, executes commands via `env.exec()`, and stops containers on `close()` or `reset()`.

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
| `snapshot_ref` | `HarborSnapshotRef \| None` | Optional exact runtime snapshot artifact for this state |
| `fs_restore_risk_now` | `bool` | Whether the current state has filesystem-restore risk signals (resets per step) |
| `fs_restore_risk_reasons` | `tuple[str, ...]` | Specific risk reasons detected at this step |
| `fs_restore_risk_ever` | `bool` | Sticky flag — `True` if any prior state in this trajectory had risk |

## Parameters

### `HarborAdapter.get_environment()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"terminal-bench@2.0"` | Dataset name with optional version (`"dataset@version"`) |
| `tasks` | `tuple` | `None` | Pre-loaded Harbor Task objects |
| `env_factory` | callable | `None` | `(task) -> BaseEnvironment` factory |
| `verify_factory` | callable | `None` | `(task, env) -> Verifier` factory |
| `dataset_path` | `str` | `None` | Local path to dataset directory |
| `environment_type` | `str` | `"docker"` | Harbor environment type. Built-in Harbor values include `docker`, `daytona`, `e2b`, `modal`, etc. `llenvs` also accepts `podman-hpc` for a local user-space runtime. |
| `tool_mode` | `bool` | `False` | Use structured tools instead of text |
| `max_steps` | `int` | `30` | Maximum steps per episode |
| `submit_keyword` | `str` | `"SUBMIT"` | Text mode submit keyword |
| `exec_timeout` | `int` | `120` | Per-command timeout in seconds |
| `command_soft_timeout` | `int \| None` | `None` | Text mode only — recoverable per-command timeout for live model-issued commands. On timeout, `Ctrl-C` / `Ctrl-\` recovery is attempted. Disabled when `None`. Trajectory-level limits (`max_steps`, `exec_timeout`) control total episode length |
| `verify_on_truncation` | `bool` | `True` | Run verifier when truncating |
| `extra_rewards` | `tuple` | `()` | Additional reward functions |
| `state_capture_mode` | `str` | `"replay"` | Harbor state capture mode: `replay` or `snapshot_exact` |
| `snapshot_artifact_root` | `Path \| str \| None` | `None` | Artifact root used when `state_capture_mode="snapshot_exact"` |
| `snapshot_options` | `HarborSnapshotOptions \| None` | `None` | Exact snapshot options (`file_locks`, `tcp_established`, `tcp_close`, `ignore_volumes`) |
| `runtime_probing` | `bool` | `False` | Enable runtime probing to annotate states with filesystem-restore risk signals |
| `text_exec_mode` | `str` | `"independent_exec"` | Text-mode execution model: `independent_exec` runs each step in a fresh shell; `tmux_session` keeps a persistent tmux-backed shell inside the container |
| `tmux_bootstrap_if_missing` | `bool` | `False` | When `text_exec_mode="tmux_session"`, attempt a bounded package-manager install of `tmux` inside the task image if it is missing |

When both `command_soft_timeout` and `exec_timeout` are set, they apply to
different phases:

- `command_soft_timeout` governs live model-issued text commands only and can
  recover with a timeout observation after a successful `Ctrl-C`.
- Harbor's extra runtime-probe execs and verifier execution are internally
  bounded without exposing separate public timeout knobs.
- `exec_timeout` remains the hard timeout for non-recoverable Harbor runtime
  operations such as replay/restore and any verifier/probe call that does not
  override it.

### `HarborAdapter.load_tasks()`

Loads Harbor task definitions without creating environments. This is useful when another layer wants to inspect or filter the task set before collection.

### `HarborAdapter.inspect_snapshot_eligibility()`

Returns per-task static exact-snapshot eligibility for a given runtime. The current implementation is runtime-specific:

- `environment_type="podman-hpc"` inspects the Harbor task definition and reports whether exact snapshots are supported for that task.
- Other runtimes currently report snapshot eligibility as unsupported.

The result objects include `task_index`, `task_name`, `eligible`, `reason_code`, and `reason_detail`.

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

## Monte Carlo Rollouts

Harbor environments have `pure_step=False` — container state is mutable and cannot be cheaply reset to a prior state. Replay-based restore remains the default path for MC rollouts:

```python
from llenvs.adapters.harbor import HarborAdapter, harbor_restore

adapter = HarborAdapter()
env = adapter.get_environment("terminal-bench@2.0", max_steps=30)

def env_factory():
    return adapter.get_environment("terminal-bench@2.0", max_steps=30)

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=sampling_params,
    env_factory=env_factory,
    restore_fn=harbor_restore,
)

# MC rollouts from a saved state
trajectories = runner.run_batch_from_states(
    [saved_state] * num_rollouts,
    batch_size=4,  # max concurrent containers
)
```

### `harbor_restore()`

Restores a Harbor environment to a saved state by replaying the trajectory prefix. Resets to the original task via `task_index`, then replays each command from `state.hidden.trajectory`. Validates task name to guard against index drift across dataset versions.

Replay uses `command_soft_timeout` (the same per-command timeout as normal collection). If a replayed command times out and recovery succeeds, replay continues to the next command — the timeout observation is preserved in the state's message history. Only shell continuation prompts (incomplete syntax) abort replay.

### `harbor_snapshot_restore()`

For datasets collected with exact checkpoints, `harbor_snapshot_restore()` restores a fresh Harbor environment from `state.hidden.snapshot_ref` instead of replaying the command prefix.

```python
from llenvs.adapters.harbor import HarborAdapter, harbor_snapshot_restore

adapter = HarborAdapter()
env = adapter.get_environment(
    "terminal-bench@2.0",
    environment_type="podman-hpc",
    max_steps=30,
)

def env_factory():
    return adapter.get_environment(
        "terminal-bench@2.0",
        environment_type="podman-hpc",
        max_steps=30,
    )

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=sampling_params,
    env_factory=env_factory,
    restore_fn=lambda env, state: harbor_snapshot_restore(
        env,
        state,
        artifact_root="/path/to/dataset_dir",
    ),
)
```

### Replay Validation

Not all Harbor tasks produce consistent state on replay — network-dependent commands, non-deterministic outputs, or time-sensitive operations can cause divergence. Use `validate_replay_consistency()` to identify replay-safe tasks:

```python
from llenvs.adapters.harbor import validate_replay_consistency

result = validate_replay_consistency(
    env_factory=env_factory,
    task_index=0,
    trajectory=("apt-get update", "pip install pandas"),
    probe_commands=(
        "find /app /home /etc -type f 2>/dev/null | sort | md5sum",
        "dpkg -l 2>/dev/null | awk '{print $2, $3}' | md5sum",
    ),
    reference_probes=stored_live_probes,  # from data collection
    num_trials=3,
)
```

Two validation modes:

1. **Self-consistency** (`reference_probes=None`): multiple replays produce the same state as each other.
2. **Live-vs-restored** (`reference_probes` provided): restored state matches probe outputs captured from the live container during data collection. This is the stronger check.

Probe commands are executed out-of-band against the restored Harbor runtime, not through `env.step(...)`, so validation does not consume episode steps or trigger verifier/truncation side effects on near-horizon states.

Returns a dict with `consistent` (bool), `matches_reference` (bool | None), `probe_outputs` (per-trial), and `divergence_details`.

### Text Execution Modes

Harbor text mode supports two execution models:

- `text_exec_mode="independent_exec"` keeps the original Harbor `exec()` behavior. Each step runs in a fresh shell, so shell-local state such as `cd`, `export`, aliases, and background jobs does not persist across steps.
- `text_exec_mode="tmux_session"` starts a persistent tmux-backed login shell inside the task container. Commands are pasted into that shell, completion is detected through a `PROMPT_COMMAND` hook, and model-facing observations come from a joined (`capture-pane -J`) tmux pane capture after Harbor strips its own staged-file echo, direct-command echo, helper-file-path bash prefixes, and prompt sentinel. The session window is resized to 200 columns to reduce visual wrapping at the source.

`tmux_session` is the preferred mode for TerminalBench-style terminal tasks because it preserves shell state, allows `cmd &` background jobs to continue across steps, and more closely matches the official TerminalBench execution model.

The tmux-backed bash shell disables history expansion (`set +H`), so `!` in pasted commands (for example `<!DOCTYPE html>`) is treated literally instead of triggering `event not found` errors.

Timeout and startup diagnostics still use raw (non-joined) pane captures so the debugging surface remains as close as possible to tmux's rendered state.

In both text exec modes, commands that produce no visible output get an explicit observation instead of an empty string.  In ``tmux_session`` mode, the shell exit code is captured via the ``PROMPT_COMMAND`` hook: silent successful commands show ``[Command completed successfully with no output]``, silent failures show ``[exit code: N]``, and if the exit status is unavailable (e.g. after a timeout) the fallback is ``[No output]``.  ``independent_exec`` mode has the same behaviour using the exec return code directly.

When an ``answer_extractor`` is configured and it cannot extract an action, Harbor does not execute the raw model text. Instead it returns a format-specific invalid-action observation (for example, reminding the model to use ``<answer>...</answer>``) and records the invalid-format flag plus extraction metadata in step info.

In ``tmux_session`` mode, Harbor also detects when bash has entered the continuation prompt because a direct command is syntactically incomplete (for example, unmatched quotes). Live steps are cancelled immediately and receive a synthetic observation explaining that the shell is waiting for more input. Replay and replay-probe restore paths treat the same condition as an immediate restore failure instead of waiting for the full command timeout.

If `tmux` is missing inside the image and `tmux_bootstrap_if_missing=True`, the adapter attempts a bounded package-manager install. Production runs still benefit from preinstalled `tmux`, especially when replay or fresh-container restores are frequent.

When `command_soft_timeout` is set, live model-issued text commands become recoverable on timeout: Harbor interrupts the command with `Ctrl-C` (escalating to `Ctrl-\` if needed), appends a standard timeout observation to the trajectory, and keeps the session alive. Trajectory-level limits (`max_steps`, `exec_timeout`) control total episode length. Replay, restore, and replay-validation commands stay on the hard `exec_timeout` path, and tool mode rejects this kwarg entirely.

See the [multi-instance runner guide](../guides/multi-instance-runner.md) for architecture details.

## `podman-hpc` Runtime

For HPC clusters where Docker is unavailable, `llenvs` can route Harbor tasks through a local `podman-hpc` runtime:

```python
adapter = HarborAdapter()
env = adapter.get_environment(
    "terminal-bench@2.0",
    environment_type="podman-hpc",
    max_steps=30,
)
```

This path preserves the existing Harbor replay model in `llenvs`: replay remains the default restore method. It also adds an opt-in exact snapshot mode for single-container tasks, where states captured during live collection can later be restored with `harbor_snapshot_restore()`.

Current v1 behavior:

- Single-container Harbor tasks are supported.
- Apptainer-backed single-container tasks default `exec()` to the image workdir, falling back to `/app` when no `WORKDIR` is declared.
- `apptainer-hpc` supports `rootfs_mode="auto" | "overlay" | "sandbox"`. `auto` probes whether the overlay path yields a writable root filesystem and falls back to writable per-trial sandbox copies when it does not.
- Harbor task loading is cached per process by dataset source, so repeated `load_tasks()` calls for the same registry dataset or local dataset path reuse the previously loaded task set.
- Overlay mode keeps `/app` and `/tests` writable with host-backed binds; sandbox mode uses the writable rootfs directly and does not bind those paths.
- Task-local `docker-compose.yaml` is supported for a constrained subset centered on a required `main` service plus sidecars.
- `exec()`, upload, and download operations target `main`; sidecars are runtime-only support services.
- Supported compose features are limited to common TerminalBench-style fields (`image`, `build.context`, `build.dockerfile`, `command`, `entrypoint`, `environment`, `working_dir`, `volumes`, `depends_on`, `healthcheck`).
- Unsupported compose features fail fast (`ports`, custom `networks`, `secrets`, `configs`, `profiles`, `devices`, external volumes).
- The runtime expects `podman-hpc` to be available on the host.
- Replay remains text-mode only, exactly like `harbor_restore()`.
- Exact snapshot export/restore is currently limited to single-container tasks. Compose-backed tasks still use replay.
- Exact snapshots are additive. If you do not opt in to `state_capture_mode="snapshot_exact"`, Harbor behavior is unchanged.
- Static snapshot eligibility inspection is available via `HarborAdapter.inspect_snapshot_eligibility()` so callers can filter task sets before starting collection.

## `apptainer-hpc` Filesystem Checkpoint/Restore

`ApptainerHPCEnvironment` supports tar-based filesystem checkpoint/restore in sandbox mode. These are filesystem-level snapshots, not process-level snapshots — background processes, in-memory state, and open sockets are not preserved across restore:

- `export_checkpoint(path)` — runs a one-shot Apptainer helper against the SIF image, bind-mounts the sandbox rootfs and the destination directory, and creates the tar from inside the container context. This avoids host-side permission failures on fakeroot-created paths.
- `restore_checkpoint(path)` — stops the running instance without deleting the sandbox, runs a one-shot Apptainer helper to clear and untar the checkpoint into the sandbox rootfs, then restarts the instance from that restored rootfs.
- `pid_namespace=True` enables `--pid` namespace isolation, required for process-level runtime probing.
- Checkpointing requires `bash` and `tar` inside the image used for the helper commands.

### Runtime Probing

Apptainer runtime discovery is cached per process. The runtime version string,
PID-namespace flag support, and overlay rootfs probe results are reused across
environment instances with the same effective runtime key. Cached hits do not
emit repeated `INFO` logs.

When `runtime_probing=True` is passed to `get_environment()`, each `reset()` and `step()` captures a runtime probe snapshot and annotates the state with risk signals:

- **Processes** (requires `pid_namespace=True`): detects new background processes not in the baseline.
- **Mounts**: detects mount table changes via `/proc/self/mountinfo` fingerprint.
- **Listening ports**: detects new listening sockets via `ss`/`netstat`.
- **Staging content**: detects unexpected top-level content in `/staging`. Harbor-managed helper entries such as `upload` and `download` are ignored.

Risk signals are stored on `HarborHidden`:
- `fs_restore_risk_now` — resets each step (current probe vs baseline).
- `fs_restore_risk_ever` — sticky OR of all prior `risk_now` values.
- `fs_restore_risk_reasons` — tuple of reason strings (e.g., `"extra_processes:redis-server"`, `"new_listening_ports:8080"`).

## Limitations

- **No seed support** — Harbor tasks are deterministic (fixed Dockerfiles/test scripts).
- **Container runtime required** — Docker or an alternative Harbor-supported runtime must be available.
- **Network-dependent** — Registry queries and image pulls require network access.
- **Binary rewards** — Native verifiers produce pass/fail only; use `extra_rewards` for finer-grained scoring (e.g., `JudgeReward`).
- **Text-mode replay only** — `harbor_restore` supports text mode; tool-mode replay requires storing full `Action` objects (deferred).
- **`PROMPT_COMMAND` clobbering** — `tmux_session` relies on a shell hook installed through `PROMPT_COMMAND`. Commands that explicitly overwrite `PROMPT_COMMAND` are unsupported and can break completion detection.
- **Exact snapshot runtime coverage** — v1 exact snapshots (process-level CRIU checkpoints) require `podman-hpc`. `inspect_snapshot_eligibility()` only reports eligibility for podman-hpc.
- **Apptainer filesystem checkpoints** — `apptainer-hpc` supports filesystem-level tar snapshots (sandbox mode only) via `export_checkpoint`/`restore_checkpoint`. These are used by the prediction-time replay cache but are not exact snapshots — background processes, in-memory state, and open sockets are not preserved.
