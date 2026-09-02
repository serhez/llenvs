# prime-rl Integration — Deep Analysis

Status: both directions are implemented on branch `verifiers-v1-integration` — the export
plugin `src/llenvs_env/` (plugin id `llenvs-env`; user guide `docs/guides/prime-rl.md`) and
the import adapter `src/llenvs/adapters/verifiers_v1.py` (`docs/adapters/verifiers-v1.md`).
The `llenvs_turn_grpo` prime-rl patch is specified apply-ready at the end of this document
and applies to a prime-rl checkout, not to this repo. Companion to `miles-integration.md`
(same three questions, same llenvs baseline), written after the miles connector shipped
and informed by that experience.

Verified against source: prime-rl `main` @ `3fc28dd` (2026-08-28) and verifiers `main` @
`c51c094` (2026-08-29). prime-rl vendors verifiers as an editable submodule pinned to
`b2e4e81` — re-verify the verifiers surface at that exact pin before implementing
(the API has no stability contract; see Risks).

## What prime-rl is

[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) is PrimeIntellect's async RL
post-training stack. `uv run rl` is a pure launcher that spawns four kinds of processes
(`entrypoints/rl.py:116`):

1. **inference** — vLLM (0.28), OpenAI-compatible plus a token-in renderer route.
2. **one env server per configured source** — a verifiers v1 `serve_env` process: a ZMQ
   ROUTER broker over an elastically scaled pool of `EnvServer` worker processes, each
   running rollouts as asyncio tasks on its own loop (`verifiers/v1/serve/pool.py`).
3. **orchestrator** — task dispatch, episode collection, credit assignment, batch
   assembly (importable as a library: `run_orchestrator(config)`,
   `orchestrator.py:1045`).
4. **trainer** — torch-native FSDP2 via torchrun (`prime_rl.trainer.rl.train`).

Transports: orchestrator↔env servers ZMQ; orchestrator→trainer ZMQ or filesystem;
trainer→inference weights via filesystem/NCCL/NIXL; env workers call inference over
HTTP directly (the orchestrator ships a `vf.ClientConfig` with every episode request).

**The orchestrator never runs an environment.** It owns only the *taskset*
(`vf.load_taskset`, `orchestrator/envs.py:77` — loaded once, client-side) and ships each
episode's `task_data` to the env server, which pydantic-validates it into the taskset's
declared `TaskData` type and runs the episode (`envs.py:1-18`). Every environment —
including any llenvs connector — executes inside env-server worker processes.

Everything environment-shaped is **verifiers v1**: `Taskset` + `Task` + (optionally)
`Env`, loaded by id. There is no other env interface.

## How environments plug in (no fork needed)

This is the decisive finding, and it corrects the first-pass analysis: **verifiers v1
has a real plugin system**, so the env side of an llenvs connector requires no fork of
verifiers or prime-rl.

- A taskset/env/harness/judge **id resolves to an installed module**:
  `verifiers/v1/utils/loaders.py:70-96` strips hub prefixes, normalizes
  (`-`→`_`, lowercase), tries the built-in namespace (`verifiers.v1.tasksets.<module>`),
  then imports the **bare top-level module**. The module exports its subclass via
  `__all__` — exactly one `Taskset`, optionally one `Env` and/or `Harness`
  (`loaders.py:99-126`). If the taskset package exports an `Env`, it is auto-paired;
  otherwise the run gets `SingleAgentEnv` (`loaders.py:168-181`).
- Packaging convention (scaffold: `uv run init <name>` in verifiers): distribution named
  `<dash-id>`, module `<snake_id>`, `__init__.py` with
  `__all__ = ["MyEnv", "MyTaskset"]`. Explicitly **no** `load_environment()` function —
  that is the removed v0 idiom (`skills/create-environments/SKILL.md:56`).
- prime-rl references the env purely by id in TOML:

  ```toml
  [[orchestrator.train.source]]
  name = "llenvs-env"
  env.taskset.id = "llenvs-env"
  env.taskset.config = "/path/to/llenvs-eval.yaml"   # our typed TasksetConfig field
  env.agent.harness.id = "null"
  env.agent.runtime.type = "subprocess"
  ```

  The `env` block is validated against the config class the id's package declares
  (`vf.resolve_env_field` narrowing, `prime_rl/configs/orchestrator.py:133-137`).
  **There is no free-form kwargs channel** — `docs/configuration.md`'s
  `source.args '{...}'` snippet is stale (no such field exists in the config classes).
  All env configuration must be typed pydantic fields on our `TasksetConfig` /
  `TaskConfig` / `EnvConfig` subclasses. This is *better* than miles for us: no
  `LLENVS_MILES_CONFIG` env-var side channel — the llenvs YAML path and env-name
  selection become first-class TOML/CLI fields.
- Operational consequence: the llenvs plugin module (and llenvs itself, plus whichever
  adapter extras the config needs) must be importable in **both** the orchestrator
  process (taskset load) and the env-server processes (episode execution). In prime-rl's
  workspace-driven setup that means installing it into the same venv
  (`uv sync --package prime-rl --package <env>`).

## The authoring surface we'd implement

A `Taskset` yields typed `Task`s; an `Env` turns one task into one episode by driving
agents; rewards land on the `Trace`.

- **`Taskset.load() -> Iterable[Task]`** is the only abstract method; may be a generator;
  `INFINITE = True` supported. `head(n)`/`shuffle(seed)` are lazy views. The taskset is
  the data half — one task type per taskset.
- **`TaskData`** (frozen pydantic, extensible): `idx`, `prompt` (str | Messages |
  `None` = the env's loop opens the conversation), `system_prompt`, per-task `timeout` /
  `resources` / container `image` / network policy. Task identity: `Task.key` (override
  for durable ids) and `Task.hash`.
- **`Task`** is a behavior class: `setup`/`finalize` hooks (with a `Runtime` handle),
  scoring via `@vf.reward` / `@vf.metric` decorated methods plus config-plugged
  functions (dotted-path loadable, `loaders.py:215-233`) and judges; `toolsets()` for
  per-rollout MCP tool servers; `NEEDS_CONTAINER`.
- **`Env[ConfigT]`**: agents are declared as `AgentConfig` fields on the `EnvConfig`
  subclass (field name = agent name); `async run(task, agents)` is the one abstract
  method; `setup(agents)` sets standing (e.g. `agents.judge.trainable = False`);
  `finalize(task, episode)` is the cross-agent judgement surface; `start()`/`stop()`
  hold env-owned shared resources per worker. Episode timeout, whole-episode retries,
  and per-agent retries are framework-owned config.
- **The turn relay** (`agent.py:142-230`): `agents.player.interaction(task)` opens a
  rollout with the caller as the run's user; `await interaction.turn(message)` sends one
  user turn (a string, or full `Messages` — **multimodal allowed**) and runs one harness
  segment, returning `Segment(messages, last_reply, terminated)`. `interaction.trace` is
  live; `trace.record_reward(name, value, weight)` / `record_metric` / `trace.info` /
  `trace.stop("<named condition>")` are the recording surface.
- **In-tree templates that are almost the llenvs connector already**:
  - `verifiers/v1/tasksets/textarena/taskset.py` — seeded external game engine stepped
    host-side; the engine plays the user; reward read off the engine post-loop. Also the
    canonical trick for non-reentrant libraries (seed→make→reset in one `await`-free
    block).
  - `verifiers/v1/tasksets/openenv/taskset.py` — external env client; JSON action
    parsing from `segment.last_reply`; empty-action guard via
    `trace.stop("empty_action")`; per-step rewards summed to one recorded reward.
  - `verifiers/v1/envs/user_sim/env.py` — a second, untrainable LLM agent
    (`trainable = False` in `setup()`, its own trace) — the native pattern for
    llenvs dialogue envs' env-side LLM.
  - `environments/color_codeword` — image content injected through `turn(Messages)`.
  - `verifiers/v1/tasksets/nemo_gym/` — `Env.start()` hosting a side server per worker,
    and a **dynamic MCP toolset** (`register()` override proxying `list_tools`/
    `call_tool` to an external service, with per-rollout state through `vf.State`).
- **Concurrency/blocking**: `run()` executes on the worker's event loop (default 128
  concurrent episodes per run, `max_concurrent_agents=1` per episode) — every blocking
  llenvs call (`reset`, `step`, env construction) must go through `asyncio.to_thread`,
  exactly as the miles connector already does. No per-turn timeout exists; bound turns
  via `AgentConfig.max_turns` and episodes via `--env.timeout.episode`.

## Credit assignment and the training pipeline (verified)

- **Trace = message graph.** Nodes carry `token_ids`, `mask` (model-sampled), per-token
  `advantages`, named `loss_weights` streams (`rl`/`ce`/`ref_kl`), `logprobs`,
  `multi_modal_data`, `routed_experts`. A `Branch` (root→leaf) is one training sample;
  `iter_trainable_branches` trains a node shared by several branches **once**
  (`orchestrator/trajectories.py:69-89`) — forks never double-train and never force
  prefix recomputation. Tokenization is renderer-owned at the interception proxy —
  the connector has no miles-style token-exactness burden.
- **Rewards**: `Trace.rewards: dict[str, Reward(score, weight)]`, `trace.reward` =
  weighted sum; `metrics` unweighted; `trace.info` free-form JSON scratch. A native
  trajectory-level `SignalBundle` equivalent.
- **Algorithms are episode-native** (API changed since the first pass): `Algorithm` has
  `score_episode(episode)` (rollout-local, at arrival) and `score_group(episodes)`
  (cohort-relative, when the group completes) — both receive **native verifiers
  episodes/traces**, so anything the env recorded (rewards, metrics, `info`, node
  structure) is visible to credit assignment (`orchestrator/algo/base.py:45-80`).
- **`assign_advantages(trace, float | list[float])`** writes per-token credit in compact
  sampled-token order across the graph (`algo/routing.py:11-34`); the list-valued path
  is first-class and validated. Built-in GRPO passes a scalar (reward minus group mean,
  optional length shaping); no sibling-rewards-must-match assertion exists anywhere.
- **Pipeline** (`orchestrator/train_sink.py`): episodes arrive → `score_episode` →
  buffered into groups of `group_size` episodes per task → `score_group` → curriculum
  admission → `trace_to_samples` → `TrainingSample(token_ids, mask, logprobs,
  advantages, rl/ce/ref_kl weights, mm_kwargs)` → packing → trainer. Staleness is
  enforced by a `max_off_policy_steps` sweep; `filter_zero_advantages` can prune
  zero-credit tokens from the RL loss. The trainer consumes per-token advantages and
  weight streams as-is — **nothing trainer- or transport-side needs to change for
  turn-level credit**.
- **Registration is the one closed door**: `ALGORITHM_CLASSES` is a hard-coded dict and
  `AlgoConfig` a closed `Literal`-discriminated union (`algo/__init__.py:42-51`,
  `prime-rl-configs/.../algorithm.py:367-377`); prime-rl's docs are emphatic that a new
  scheme is a new in-repo class. Two dotted-path precedents exist elsewhere (custom
  trainer loss `CustomLossConfig.import_path`; echo's token filter), so an upstream PR
  making algorithms dotted-path loadable would be small and thematically consistent —
  otherwise a custom algorithm is a ~3-file fork patch (algorithm module + config class
  in the union + registry entry), with zero transport/trainer changes.
- **Task sampling** (`orchestrator/train_source.py`, `curriculum/`): multi-source mixing
  by `ratio`-weighted choice; per-source `StandardSampler` is `itertools.cycle` in
  **source order — no shuffling** (shuffle inside our `Taskset.load()` if desired, we
  own it); optional difficulty-pool sampler; resume state is an integer cursor.
- **Eval**: same `Env`/`EnvClient` interface — `[[orchestrator.eval.source]]` with
  `num_examples`, or standalone `uv run evals`. No exporter, no separate scorer path.
- **Multimodal is end-to-end**: branch `multi_modal_data` → `mm_kwargs` encode →
  transport → VLM trainer forward (`trajectories.py:35-49`, `trainer/rl/train.py:384+`).

## Question 1: prime-rl into llenvs, or llenvs into prime-rl?

**llenvs-into-prime-rl**, same verdict as miles, now with stronger evidence:

- prime-rl is a launched multi-process system (launcher + ZMQ mesh + torchrun). The
  orchestrator is importable, but env execution is wire-only (`EnvClient` → env server);
  there is no in-process env path to embed into llenvs.
- verifiers v1's plugin loader makes the reverse trivial by comparison: llenvs ships one
  installed module and becomes a first-class citizen of the entire verifiers ecosystem —
  `vf eval`, GEPA, the env hub, *and* prime-rl — from a single artifact.
- The opposite direction (wrapping verifiers-v1 tasksets *into* llenvs as an adapter,
  the way `VerifiersAdapter` wraps the legacy API) is a separate, additive idea for
  consuming their env catalog — unrelated to training integration and not blocked by
  this work.

## Question 2: what's missing in prime-rl that becomes an issue for llenvs?

| llenvs capability | prime-rl today | Severity |
|---|---|---|
| Turn-level STEP rewards (per-transition `SignalBundle`) | Trace-level rewards; built-in algos broadcast scalars. Transport + trainer already carry per-token advantages; the missing piece is only an in-repo `Algorithm` registration | **Medium** (miles: High, needed a 4-file fork incl. transport) |
| Multi-signal weighted rewards | `record_reward(name, score, weight)` — direct mapping | None |
| Weight-0 monitoring signals | `record_metric` | None |
| Feedback text → next observation (iterative envs) | `run()` owns the loop; feedback is just the next `turn()` | None |
| Env-side LLM (dialogue, judges) | Native: second agent, `trainable=False`, own trace, own model/client pin | None — better than miles' config-discipline guard |
| In-process Python tools (`ToolDefinition`) | Tools are MCP-mediated (`vf.Toolset` servers; null harness executes them in-segment). llenvs tools execute through `env.step()` | **Medium** — the one real design problem (see below) |
| History rewriting (`history_fn`, `prompt_budget`, reasoning stripping — llenvs defaults) | `turn()` is append-only; the graph *represents* rewrites as branches (trained once, no recompute) but the relay API doesn't expose them | Medium-low — default to verbatim mode, same as the miles connector |
| Env truncation vs token truncation | Named `trace.stop(condition)` + framework `is_truncated`; staleness admission separate | Low — better than miles |
| Multi-turn VLM | Images through `turn(Messages)`; renderer→`mm_kwargs`→VLM trainer verified e2e | None — miles blocks this entirely |
| Branching / tree exploration | Branches first-class; shared nodes trained once | Low |
| Config-driven setup | Typed pydantic fields only, no kwargs dict | None — cleaner than miles' env-var discovery, but our config surface must be declared as typed fields |
| Sub-response segment rewards | Expressible via per-token advantages in a custom Algorithm | Medium-low |
| Blocking sync envs | `run()` on the worker event loop | Low — `asyncio.to_thread` discipline (already our pattern) |
| Container-based llenvs envs | Env-server host needs Docker; or reach them via llenvs' own `EnvironmentServer` HTTP | Low-medium — ops, not API |

**The tools gap, concretely.** llenvs tool semantics = a tool call advances env state
through `env.step()` and yields per-step rewards. prime-rl's native path executes tools
harness-side against MCP servers within one segment. Three options, in order:

1. **Host-side text parsing (phase 1)**: no MCP; the env advertises tool syntax in the
   prompt and parses calls from `segment.last_reply` with llenvs' `HermesToolCallParser`,
   stepping the env host-side — exactly the OpenEnv template. Cost: no native
   function-calling format for the policy. Benefit: zero moving parts, full llenvs
   semantics preserved.
2. **MCP facade (phase 2)**: a `vf.Toolset` whose `register()` dynamically advertises the
   task's `ToolDefinition`s (nemo_gym's `list_tools`/`call_tool` proxy is the exact
   pattern) and routes calls into the live env. Requires the env instance to be
   reachable from the toolset server — via llenvs' `EnvironmentServer` with per-rollout
   coordinates in `vf.State` (nemo_gym does precisely this handoff). Native tool_calls,
   more infrastructure.
3. Tool interception (verifiers' interception layer) — only if 2 proves awkward.

## Question 3: integration shape

`llenvs` ships a second top-level module in the same distribution (import hygiene: the
plugin imports `verifiers.v1`, so it cannot live under `llenvs/`'s import surface):

```
src/llenvs_env/            # plugin id "llenvs-env" → module "llenvs_env"
├── __init__.py            # lazy (PEP 562) __all__ = ["LLEnvsTaskset", "LLEnvsEnv"]
├── _vf.py                 # guarded `verifiers.v1` import with an install hint
├── _config.py             # EvalConfig load/select, Scorer/DatasetProvider caches (no vf imports)
├── _relay.py              # feedback/tool text, Hermes parsing → Action, cleanup (no vf imports)
├── taskset.py             # LLEnvsTasksetConfig, LLEnvsData, LLEnvsTask, LLEnvsTaskset
└── env.py                 # LLEnvsEnvConfig (one `agent` seat, max_steps), LLEnvsEnv relay
```

- **`LLEnvsTasksetConfig(vf.TasksetConfig)`**: `config: Path` (llenvs `EvalConfig`
  YAML), `env_name: str | None`, `num_tasks`, `shuffle_seed`. First-class TOML/CLI
  fields — no env-var discovery layer needed (retire the miles-style
  `LLENVS_MILES_CONFIG` pattern here).
- **`LLEnvsData(vf.TaskData)`**: `task_index: int`, ground truth for scoring,
  the prompt from `DatasetProvider` (the framework sends it; the relay opens with a
  bare `turn()`), `multi_turn`, `answer`, `config_path`, `env_name`, `info`;
  `Task.key = llenvs:<env_name>:<task_index>`.
- **`LLEnvsTaskset.load()`**: `DatasetProvider.get_items()` → yield tasks (drop
  `episode_id` etc. as in the miles exporter; identity via `Task.key`).
- **`LLEnvsEnv(vf.Env[LLEnvsEnvConfig])`** (one `agent: vf.AgentConfig` seat plus `max_steps`):
  `run()` = fresh llenvs env via `EnvironmentFactory` in `to_thread`, `reset(task_index)`,
  then the textarena/openenv relay loop — `turn(observation)` → llenvs extraction /
  tool parsing → `to_thread(env.step)` → feedback message; named stops
  (`env_terminated`, `env_truncated`, `max_steps`, `empty_action`; environment exceptions
  raise so the framework records the episode error and retries); post-loop:
  weighted signals → `record_reward`, weight-0 → `record_metric`, per-turn totals →
  `trace.info["llenvs"]["turn_rewards"]` (the channel a turn-credit algorithm reads;
  `info` not `metrics`, to keep logging clean), plus per-turn signal breakdowns.
- **Single-turn RLVR needs no custom Env at all**: taskset + `@vf.reward` on the Task
  calling llenvs extraction/scoring over `trace.last_reply`, under `SingleAgentEnv`.
- **Turn-level credit — `llenvs_turn_grpo` Algorithm** (the miles-Tier-2 analogue,
  much smaller): `score_group(episodes)` computes per-decision return-to-go from
  `trace.info["llenvs"]["turn_rewards"]` (γ per decision), group-relative normalization
  over the pooled decision returns of the cohort, then broadcasts each decision's
  advantage over its turn's sampled tokens — turn *i* ↔ the *i*-th sampled node, spans
  from `node.mask` — and ships via `assign_advantages(trace, per_token_list)`.
  Traces without turn data degenerate to exact GRPO. Registration: a ~3-file prime-rl
  patch (module + `AlgoConfig` union entry + `ALGORITHM_CLASSES` entry), carried as a
  fork-with-patch or proposed upstream as a dotted-path `import_path` registry
  (precedent: `CustomLossConfig`). **No transport or trainer changes** — the decisive
  advantage over miles' Tier 2.
- **Phasing**: (1) taskset + env + text relay loop + reward mapping (unit-testable
  against verifiers locally — it's a real package, no GPU); (2) tools via host-side
  parsing; (3) the turn-credit algorithm + fork patch; (4) MCP toolset facade; (5) VLM
  passthrough. Factor the loop core (action extraction, feedback formatting, env
  lifecycle, signal aggregation) into a driver shared with
  `llenvs.integrations.miles.agent` — the two loops are near-isomorphic.

## Comparison vs the miles connector (experience-informed)

What miles made us hand-build that prime-rl/verifiers owns natively: episode/rollout
timeouts and retries, eval plumbing, data-source semantics (grouping, buffering,
save/load), the postprocessor span join, token-exactness (verbatim echo), the isolation
guard (env-side LLMs are first-class agents here), the JSONL exporter (taskset is the
dataset). What prime-rl adds that miles didn't have: plugin packaging against a
fast-moving API, env-server ops (llenvs installed in the training venv, Docker on env
hosts for container envs), typed-config declaration, and the MCP question for tools.

| Dimension | miles connector (shipped) | prime-rl connector (projected) |
|---|---|---|
| Entry surface | agent fn + RM + DataSource + postprocessor + advantage hook (5 dotted paths) | one plugin package (taskset+env), optional algorithm patch |
| Turn-level credit | 4-file fork incl. transport allowlist + trainer dispatch | ~3-file in-repo registration, zero transport/trainer changes |
| Token exactness | connector's burden (verbatim echo, `max_retries=0`) | framework's burden (renderer + interception) |
| Multi-signal / monitoring | metadata dict + postprocess join | native `record_reward`/`record_metric` |
| VLM | impossible on TITO path | supported |
| Env-side LLM / dialogue | isolation guard, must not touch session URL | native second agent, untrainable |
| Tools | native function calling via `--tito-model` parser | text-parse first, MCP facade later |
| Very large models | Megatron backend | FSDP2 only |
| API stability | miles v2 experimental but self-contained | verifiers v1 unversioned churn, vendored pin |
| e2e test cost | Linux+GPU only | env package unit-testable locally; training e2e Linux+GPU |

**Verdict unchanged in direction, stronger in degree**: prime-rl is decisively better for
multi-turn expressiveness *and* — corrected from the first pass — its integration surface
is smaller than it looked: the plugin system, in-tree templates, and framework-owned
lifecycle remove most of what the miles connector had to build by hand. The remaining
structural costs are operational (env servers, venv co-installation) and ecosystemic
(API churn), not architectural.

## Risks & constraints

- **API churn is the top risk.** verifiers v1 has no stability contract, no changelog,
  and rename scars in-tree (`default`→`bash` harness, `max_concurrent`→
  `max_concurrent_agents`); prime-rl pins a submodule commit and resolves deps with a
  rolling 7-day `exclude-newer`. Pin verifiers to prime-rl's exact submodule commit and
  expect connector maintenance. The `EnvConfig`/CLI field surface is the churn-prone
  part.
- **Typed-config-only** means llenvs config exposure is deliberate API design, not a
  passthrough — decide early which llenvs `EnvironmentConfig` knobs surface as taskset
  fields vs stay in the YAML.
- Stack pins are aggressive: Python ~=3.12, torch 2.13+cu130, vLLM 0.28,
  transformers ==5.6.2 — the connector package itself must stay dependency-light so it
  co-installs cleanly.
- `run()` shares an event loop with up to 128 episodes: any accidentally-sync llenvs
  path (judge HTTP calls, container waits) must be audited for `to_thread`.
- Fresh-env-per-episode again leaks container envs; `Env.start()/stop()` per-worker
  pooling is the natural future home (better than miles, which had no lifecycle hook).
- Group-relative turn credit assumes linear traces (one branch); multi-branch traces
  should degenerate to scalar GRPO explicitly.
- prime-rl trains through renderers (token-in): the policy's chat template must be in
  `renderers`' `MODEL_RENDERER_MAP`.

## Open items to verify at implementation time

- Re-read the verifiers surface at prime-rl's pinned submodule commit (`b2e4e81`) —
  the analysis above reads verifiers `main` (`c51c094`), one day apart.
- `env_config_data` / `serve_env` construction path end-to-end with an out-of-tree
  package (smoke: `uv run eval llenvs-env` in a verifiers checkout).
- Whether `trace.info` survives the env-server wire intact for the algorithm to read
  (it is part of the serialized trace; confirm on the msgpack path).
- `Segment.last_reply` vs full `segment.messages` for reasoning-bearing models (what the
  extractor should see).
- Where `group_size` interacts with `ratio`-mixed multi-source runs when only one source
  is llenvs.

## `llenvs_turn_grpo` fork patch specification

The llenvs side of turn-level credit ships in the plugin: `LLEnvsEnv` records each step's
`SignalBundle.total` in `trace.info["llenvs"]["turn_rewards"]` (decision *i* = the *i*-th
policy turn) and keeps `trace.reward == sum(turn_rewards)`. The prime-rl side is a
three-file patch plus tests, applied to a **prime-rl checkout** (verified against `main` @
`3fc28dd`), not to this repo. It relies on two verified transport facts: `trace.info` is a
regular serialized `Trace` field (the env-server msgpack wire trims only tensors), and
`assign_advantages(trace, list[float])` takes exactly `Σ sum(node.mask)` values over the
masked nodes in `trace.nodes` order — so decision *i* ↔ the *i*-th masked node on a linear
relay trace. `Branch.advantages` spreads node advantages onto the flat sample and
`trace_to_samples` ships them verbatim: **zero transport or trainer changes.** Stock prime-rl
already trains llenvs-env traces with trajectory-level GRPO; this patch only adds
per-decision credit.

Discounting is per **decision**, never per token (a per-token γ would decay across a
1000-token turn), and there is one forward pass per trajectory — credit rides the advantage
vector.

**1. New `src/prime_rl/orchestrator/algo/llenvs_turn_grpo.py`**

```python
"""llenvs turn-level GRPO: per-decision credit from ``trace.info["llenvs"]``.

The llenvs-env relay records each environment step's reward total in
``trace.info["llenvs"]["turn_rewards"]`` (decision i = the i-th policy turn) and
keeps ``trace.reward == sum(turn_rewards)``. This algorithm turns those per-turn
rewards into per-decision advantages — return-to-go with a per-DECISION discount,
normalized group-relative over the pooled decision returns of the cohort — and
broadcasts each decision's advantage over its turn's sampled tokens. Traces
without usable turn data (no llenvs payload, a forked trace, or a harness that
sampled several nodes per turn) fall back to the stock GRPO scalar, so a group
with no turn data trains exactly like ``grpo``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import LLEnvsTurnGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def turn_rewards(trace: vf.Trace) -> list[float] | None:
    """The relay's per-turn reward totals, or None when the trace carries none."""
    payload = trace.info.get("llenvs")
    if not isinstance(payload, dict):
        return None
    rewards = payload.get("turn_rewards")
    if not isinstance(rewards, list) or not rewards:
        return None
    return [float(r) for r in rewards]


def decision_spans(trace: vf.Trace) -> list[int]:
    """Sampled-token count of each masked node in ``trace.nodes`` order — the
    layout ``assign_advantages`` expects."""
    return [sum(node.mask) for node in trace.nodes if any(node.mask)]


def returns_to_go(rewards: list[float], gamma: float) -> list[float]:
    """G_i = sum_{j >= i} gamma^(j - i) * r_j, discounted per decision."""
    returns = [0.0] * len(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        returns[i] = acc
    return returns


def align_turn_rewards(trace: vf.Trace) -> tuple[list[float], list[int]] | None:
    """Per-decision rewards aligned to the trace's masked nodes, or None when the
    trace is not eligible for turn-level credit (it then gets the GRPO scalar)."""
    rewards = turn_rewards(trace)
    if rewards is None:
        return None
    logger = get_logger()
    if trace.num_branches != 1:
        logger.warning(
            f"llenvs_turn_grpo: trace {trace.id!r} has {trace.num_branches} branches; falling back to GRPO"
        )
        return None
    spans = decision_spans(trace)
    missing = len(spans) - len(rewards)
    if missing not in (0, 1):
        logger.warning(
            f"llenvs_turn_grpo: trace {trace.id!r} has {len(spans)} sampled nodes for {len(rewards)} "
            "turn rewards (use a one-call-per-turn harness such as `null`); falling back to GRPO"
        )
        return None
    # A trailing sampled turn the env never stepped (e.g. an empty reply) earns 0.
    return rewards + [0.0] * missing, spans


class LLEnvsTurnGRPOAlgorithm(Algorithm):
    """GRPO with per-decision credit for llenvs-env relay traces."""

    def __init__(self, config: LLEnvsTurnGRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.gamma = config.gamma
        self.std_normalization = config.std_normalization
        self.length_weighting = config.length_weighting

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return
        aligned = {
            i: alignment
            for i, trace in enumerate(traces)
            if (alignment := align_turn_rewards(trace)) is not None
        }
        returns = {i: returns_to_go(rewards, self.gamma) for i, (rewards, _) in aligned.items()}
        pooled = [g for gs in returns.values() for g in gs]
        mean = sum(pooled) / len(pooled) if pooled else 0.0
        std = 1.0
        if self.std_normalization and len(pooled) > 1:
            std = math.sqrt(sum((g - mean) ** 2 for g in pooled) / len(pooled)) or 1.0
        scalar_mean = sum(trace.reward for trace in traces) / len(traces)
        for i, trace in enumerate(traces):
            if i not in aligned:
                assign_advantages(trace, trace.reward - scalar_mean)
                continue
            _, spans = aligned[i]
            per_token: list[float] = []
            for g, span in zip(returns[i], spans, strict=True):
                advantage = (g - mean) / std
                if self.length_weighting == "span_normalized" and span:
                    advantage /= span
                per_token.extend([advantage] * span)
            assign_advantages(trace, per_token)
```

Eligibility per trace: a non-empty `turn_rewards` list, a single branch, and
`len(masked nodes) − len(turn_rewards) ∈ {0, 1}` (the trailing un-stepped turn — an empty
reply — is padded with 0.0; any other mismatch means the harness sampled several nodes per
turn, so the decision↔node mapping is unknown and the trace falls back). Ineligible traces
get the stock GRPO scalar `trace.reward − group mean`; a group with no eligible trace is
`GRPOAlgorithm` exactly. The connector invariant `trace.reward == Σ turn_rewards` keeps
mixed groups on one reward scale (pooled-decision baseline for eligible traces, scalar
baseline for the rest). `filter_zero_advantages` (default on) then drops zero-credit
decisions from the RL loss — intended.

**2. `packages/prime-rl-configs/src/prime_rl/configs/algorithm.py` — the config class**

Add before the `AlgoConfig` union:

```python
class LLEnvsTurnGRPOAlgoConfig(BaseAlgoConfig):
    type: Literal["llenvs_turn_grpo"] = "llenvs_turn_grpo"
    """Turn-level GRPO for llenvs-env relay traces: per-decision return-to-go
    from ``trace.info["llenvs"]["turn_rewards"]``, normalized over the group's
    pooled decision returns and broadcast over each turn's sampled tokens
    (``rl`` loss). Traces without turn data get the stock GRPO scalar."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    gamma: float = Field(1.0, ge=0.0, le=1.0)
    """Per-DECISION discount of the return-to-go (never per token)."""

    std_normalization: bool = False
    """Also divide by the std of the pooled decision returns (stock GRPO
    parity is mean-only)."""

    length_weighting: Literal["uniform", "span_normalized"] = "uniform"
    """``uniform``: every sampled token of a decision carries its advantage;
    ``span_normalized``: divide by the decision's token count so each decision
    weighs the same regardless of its length."""

    def validate_env(self, env_config: vf.EnvConfig) -> None:
        """Require the llenvs-env relay: it is what records the per-turn rewards."""
        try:
            from llenvs_env.env import LLEnvsEnvConfig
        except ImportError as e:
            raise ValueError(
                "algorithm 'llenvs_turn_grpo' needs the llenvs-env plugin "
                "(`pip install 'llenvs[verifiers]'` in the orchestrator venv)."
            ) from e
        if not isinstance(env_config, LLEnvsEnvConfig):
            raise ValueError(
                "algorithm 'llenvs_turn_grpo' reads the llenvs-env relay's per-turn rewards, "
                f"but this env resolved to {type(env_config).__name__}. Use 'grpo' for other envs."
            )
```

Then add `| LLEnvsTurnGRPOAlgoConfig` to the `AlgoConfig` union and a bullet to its
docstring list:
``- ``llenvs_turn_grpo`` — GRPO with per-decision credit from llenvs-env turn rewards.``

**3. `src/prime_rl/orchestrator/algo/__init__.py` — registry**

```python
from prime_rl.orchestrator.algo.llenvs_turn_grpo import LLEnvsTurnGRPOAlgorithm
...
ALGORITHM_CLASSES["llenvs_turn_grpo"] = LLEnvsTurnGRPOAlgorithm   # in the dict literal
...
__all__ += ["LLEnvsTurnGRPOAlgorithm"]                            # in the list literal
```

**4. `tests/unit/orchestrator/test_algorithms.py`**

- Add `("llenvs_turn_grpo", {}, "policy", "rl")` to the vetted-defaults parametrize row;
  assert `_build(type="llenvs_turn_grpo").gamma == 1.0` and that `validate_env` rejects a
  plain `vf.EnvConfig()` (`match="llenvs-env"`).
- A `_relay_episode(turn_rewards, spans)` helper modeled on the file's `_make_episode`:
  alternating user/assistant `MessageNode`s, the *k*-th assistant node with `span[k]`
  masked tokens, `trace.info={"llenvs": {"turn_rewards": turn_rewards}}`,
  `rewards={"llenvs/progress": vf.Reward(score=sum(turn_rewards))}`; build the algorithm
  with `LLEnvsTurnGRPOAlgorithm(_build(type="llenvs_turn_grpo", **kw), MagicMock())` and
  drive `asyncio.run(algo.score_group([...]))`, reading `node.advantages` per masked node.
- Behaviors: (a) return-to-go with γ=1 over a two-episode group (`[1, 0]` and `[0, 0]` →
  pooled mean 0.25 → node advantages `0.75, -0.25` / `-0.25, -0.25`); (b) γ=0.5 on
  `[0, 1]` → returns `[0.5, 1.0]`; (c) broadcast lengths equal `sum(node.mask)` per node
  and `trace_to_samples` ships a full-length advantage stream; (d) trailing un-stepped
  turn padded with 0.0; (e) three sampled nodes for one reward → scalar fallback equal to
  `GRPOAlgorithm`'s; (f) multi-branch trace → fallback; (g) a group with no turn data →
  per-token advantages identical to `GRPOAlgorithm`; (h) `span_normalized` divides by the
  span; (i) `std_normalization` divides by the pooled std, with the zero-std guard.

**Launch** (per-source algorithm, verified at `configs/orchestrator.py` `TrainSourceConfig.algo`):

```toml
[[orchestrator.train.source]]
name = "llenvs"
env.taskset.id = "llenvs-env"
env.taskset.config = "/abs/path/config.yaml"
env.agent.harness.id = "null"          # one model call per turn — required for the node↔turn mapping
algo.type = "llenvs_turn_grpo"
algo.gamma = 1.0
```

Verification in the checkout: `uv run pytest tests/unit/orchestrator/test_algorithms.py -q`.

**Upstream alternative.** A dotted-path algorithm registry (an `import_path` on
`BaseAlgoConfig`, mirroring `CustomLossConfig.import_path` and echo's token filter) would
let this module live in the llenvs distribution and shrink the patch to zero files.

**Apply-time verifications.** Re-verify at the verifiers commit prime-rl vendors
(`b2e4e81` at analysis time): `assign_advantages`' masked-node order, `Trace.info` on the
msgpack wire (statically verified: a regular field; `EXCLUDE_FIELDS` trims only tensors),
`Segment.last_reply` vs full messages for reasoning models (what the llenvs extractors see),
and how `group_size` interacts with `ratio`-mixed sources when only one source is llenvs.

**Estimator caveat.** Turn-GRPO with a pooled group baseline is a pragmatic first
estimator; per-turn-index baselines and turn-PPO (decision-boundary value extraction) are
research knobs on top of the same channel, mirroring the miles Tier-2 caveat.
