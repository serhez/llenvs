# prime-rl / verifiers v1 Training

[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) trains on environments written against the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 API (tasksets, envs, traces). llenvs plugs in from the outside as a verifiers **plugin package**: the `llenvs_env` module (plugin id `llenvs-env`) ships in the llenvs distribution and exports one `Taskset` and one `Env`. Anything that loads verifiers v1 plugins consumes llenvs environments unchanged: standalone `uv run eval`, GEPA prompt optimization, and prime-rl training sources. No prime-rl fork is needed for the environment side.

Install the extra:

```bash
uv pip install -e ".[verifiers]"   # verifiers>=0.3.1 (ships the v1 API)
```

For prime-rl training, match the verifiers version prime-rl vendors as a submodule (the v1 API has no stability contract); the plugin is tested against the commit range noted in `docs/design/prime-rl-integration.md`.

The import direction — running verifiers v1 tasksets *inside* llenvs — is the [Verifiers v1 adapter](../adapters/verifiers-v1.md).

## How configuration reaches the plugin

verifiers resolves plugin ids to installed modules (`llenvs-env` → `llenvs_env`) and narrows the run's typed config to the plugin's own `TasksetConfig` subclass, so the llenvs config is a first-class CLI/TOML field — no environment variables.

| `--env.taskset.*` field | Type | Meaning |
| --- | --- | --- |
| `config` | path (required to load) | llenvs `EvalConfig` YAML: environments, extractors, system prompt |
| `env_name` | str | Which `environments[]` entry to use; required when the YAML lists several |
| `num_tasks` | int | Keep only the first `num_tasks` tasks (applied after shuffling) |
| `shuffle_seed` | int | Shuffle task indices with this seed first; prime-rl samples sources in order, so shuffling is the taskset's job |
| `system_prompt` | path | verifiers' own override: a file whose text replaces every task's system prompt (e.g. a GEPA `best_system_prompt.txt`) |

The seat is `--env.agent.*` (model, sampling, `max_turns`, timeouts, harness). The relay's own knob is `--env.max_steps`: environment steps per episode, defaulting to the llenvs spec's `max_steps`, else 100.

!!! note "Default harness"
    verifiers runs an unpinned seat under the taskset's bundled harness when the taskset ships one, else under its `bash` coding-agent harness (shell and edit tools in a sandbox). `llenvs-env` bundles `LLEnvsHarness`, a tool-less chat loop that makes exactly one model call per turn, so no harness flag is needed: llenvs environments get plain replies and turn-level credit gets its one-node-per-turn traces. A pinned seat (`--env.agent.harness.id bash`, TOML `env.agent.harness.id`) still wins.

## Recipe A — standalone evaluation with `uv run eval`

From a verifiers checkout (or any venv with `verifiers` and `llenvs[verifiers]` installed):

```bash
uv run eval llenvs-env \
    --env.taskset.config /abs/path/config.yaml \
    --env.taskset.env_name leg_counting \
    --env.taskset.num_tasks 100 \
    --env.agent.model <served model> --env.agent.client.base_url <endpoint>
```

`llenvs-env` resolves to `LLEnvsTaskset` and, because the module also exports `LLEnvsEnv`, to that env for every run of the taskset unless `--env.id` names another. GEPA (`uv run gepa llenvs-env ...`) works the same way and writes the optimized prompt to a file you can hand back through `--env.taskset.system_prompt`.

## Recipe B — a prime-rl training source

```toml
[[orchestrator.train.source]]
name = "llenvs"
env.taskset.id = "llenvs-env"
env.taskset.config = "/abs/path/config.yaml"
env.taskset.env_name = "leg_counting"      # only when the YAML lists several
env.taskset.shuffle_seed = 0
```

Two processes touch the plugin: the orchestrator loads the taskset (it needs the YAML and every llenvs adapter the config names), and the env-server workers run episodes and score traces (they need the same). Install llenvs with the same optional adapters into both venvs, and use an absolute config path that both hosts can read.

Rewards flow through the normal prime-rl pipeline: `trace.reward` is the trajectory scalar GRPO uses. Turn-level credit is an optional algorithm patch (below).

## What the relay does per episode

1. Creates a **fresh environment** from the config (llenvs environments enforce state continuity, so episodes never share one); `reset`, `step`, creation and cleanup run under `asyncio.to_thread`.
2. Resets to the task's `task_index` and discards the observation: the framework sends the task prompt itself, so the relay opens with a bare `turn()`. This relies on the core invariant that `reset(task_index)` is deterministic.
3. Loops: policy reply → llenvs `Action` (Hermes `<tool_call>` blocks parsed when the observation advertises tools, else the reply text verbatim) → `env.step` → feedback message: `<tool_response>` blocks for tool results, else the observation's state text, else the last user message, else the prompt (same precedence as the miles connector).
4. Stops with a named condition and closes the environment (`close()` / `shutdown()` when the environment has them).

| `trace.stop_condition` | Meaning |
| --- | --- |
| `env_terminated` | the environment ended the episode |
| `env_truncated` | the environment truncated the episode |
| `max_steps` | the relay's step cap was hit |
| `empty_action` | the policy returned an empty reply; nothing was stepped |
| framework conditions (`max_turns`, token caps, `user_closed`) | the framework ended the run first; the relay records no stop of its own |

Environment exceptions **raise** out of `run()`: the framework records the episode error, retries per `--env.retries`, and never trains on the trace.

## Rewards and metrics

Every step's `SignalBundle` is folded into the trace once the loop ends:

| llenvs signal | recorded as |
| --- | --- |
| weight ≠ 0, same weight on every turn | `trace.rewards["llenvs/<name>"]` = sum of raw values at the native weight |
| weight ≠ 0, weight varies across turns | `trace.rewards["llenvs/<name>"]` = weighted sum at weight 1.0 (logged warning) |
| weight = 0 (monitoring signals) | `trace.metrics["llenvs/<name>"]` — never touches `trace.reward` |
| `reward is None` (feedback-only) | skipped, as in `SignalBundle.total` |
| — | `trace.metrics["llenvs/env_steps"]` |

`trace.info["llenvs"]` carries the per-turn breakdown: `turn_rewards` (each step's `SignalBundle.total`), `turn_signals` (raw per-signal values per turn), `signal_weights` (`null` for a signal whose weight varied), `env_steps`, `stop`, `task_index`, `env_name`. **Invariant: `trace.reward == sum(turn_rewards)`** — the relay records exactly the llenvs weighted total, and the task's own `llenvs_total` reward hook contributes 0.0 whenever `trace.info["llenvs"]` is present.

## Single-turn tasks

Single-turn llenvs environments work under the bundled env too (one step, `env_terminated`). Alternatively run them under verifiers' plain single-agent env with `--env.id single-agent`: the task's `@vf.stop single_turn` hook ends the run after the first reply and its `@vf.reward llenvs_total` hook scores `trace.last_reply` through a process-cached llenvs `Scorer` (the config's extractors and reward functions), recording each signal under `trace.metrics["llenvs/<name>"]` and returning the weighted total as `trace.reward`. Multi-turn tasks refuse that path with an explicit error.

Task data (`LLEnvsData`): `idx` is the yield position, `name` is `<env_name>#<task_index>`, `task_index`, `multi_turn`, `answer` (ground truth when the environment exposes one), `config_path`, `env_name`, and `info` (JSON-serializable reset metadata without `episode_id`). `Task.key` is `llenvs:<env_name>:<task_index>`, a durable identity that survives shuffling. Observations with a message history become typed verifiers `Messages` (user prompt first, then the history); otherwise the prompt is a plain string.

## Tools

Tool use is host-side text parsing, no MCP: when the initial observation advertises tools, the taskset appends the Hermes `<tools>` preamble (`HermesToolCallParser.format_tools()`) to the system prompt, the relay parses `<tool_call>` blocks from replies into `Action.tool_calls`, and tool results return as `<tool_response>` blocks inside one user message (never `role: tool` — the relay path has no native tool-call assistant node, and a dangling tool message breaks chat templates). Tools are snapshotted at reset; per-step changes to `available_tools` are honored when parsing but not re-advertised in the prompt.

## Hard constraints

- **Text-only.** Task or observation images raise `NotImplementedError` (at taskset load and at reset/step).
- **Verbatim, append-only history.** The relay only ever calls `turn()`; no history rewriting.
- **Deterministic `reset(task_index)`.** The prompt the model sees comes from the taskset's reset; the episode's live environment is reset again to the same index.
- **Fresh environment per episode.** Expensive (containerized) environments pay creation per episode; pooling through verifiers' `Env.start()`/`stop()` is future work.
- **One model call per turn for turn-level credit.** The bundled `LLEnvsHarness` (and verifiers' `null` harness) satisfy this; a tool-looping harness produces several sampled nodes per turn and the credit algorithm falls back to plain GRPO for that trace.

## Turn-level credit

Stock prime-rl trains on `trace.reward` (GRPO broadcasts one scalar over the trajectory's tokens). Per-decision credit **without any extra forward passes** is a small prime-rl patch: an `llenvs_turn_grpo` algorithm whose `score_group` reads `turn_rewards`, computes per-decision return-to-go with a per-decision discount, normalizes group-relative over the pooled decision returns, and broadcasts each decision's advantage over its turn's sampled tokens through prime-rl's own `assign_advantages`. Nothing on the transport or trainer side changes. The patch (module source, config class, registry entry, tests, launch snippet) is specified in `docs/design/prime-rl-integration.md`, and `docs/design/patches/prime-rl-llenvs-turn-grpo.patch` is the same change as a `git apply`-ready diff against the prime-rl commit named in its header.

## Testing

`tests/test_llenvs_env_config.py` and `tests/test_llenvs_env_relay.py` cover the verifiers-free halves (config loading and caching, feedback/tool text, action parsing) and run in the base venv. `tests/test_llenvs_env_taskset.py` and `tests/test_llenvs_env_env.py` drive the taskset and the relay loop with stub `Agents`/`Interaction` objects from `tests/llenvs_env_stubs.py` (token-free trace commits, a scripted mock environment) and are skipped unless `verifiers.v1` is importable — run them in a venv with the `verifiers` extra installed. A model-free smoke that exercises the plugin loader, CLI narrowing, taskset loading and task setup is verifiers' validate CLI:

```bash
uv run validate llenvs-env --taskset.config path/to/eval.yaml --runtime.type subprocess -o runs/
```

llenvs tasks carry no model-free gold check, so every task ends as `unchecked`; what the run proves is that the taskset resolves, its tasks load with the expected keys and system prompts, and nothing errors. End-to-end validation is `uv run eval llenvs-env ...` in a verifiers checkout against a served model.
