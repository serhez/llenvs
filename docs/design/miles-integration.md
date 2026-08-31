# Miles Integration — Research & Direction

Status: research phase. No implementation yet.

## What miles is

[Miles](https://github.com/radixark/miles) (radixark, v0.1 released 2026-08) is an
enterprise RL post-training framework for LLMs/VLMs, forked from and co-evolving with
THUDM/slime. It composes **SGLang** for rollout, **Megatron-LM** (or FSDP2) for
training, and **Ray** for orchestration. It is launched as a distributed job
(`train.py` / `train_async.py` under Ray), configured through hundreds of CLI flags,
Linux + GPU only, installed from source or docker. Supports GRPO/GSPO/PPO/REINFORCE++,
SFT, on-policy distillation, fully-async rollout, low-precision recipes, LoRA.

## How miles consumes environments

Miles is explicitly designed to be agnostic about where environments come from
(`docs/user-guide/environments.md`). Its rollout stack has **three nested plug-in
layers**, each replaceable by a dotted import path — no fork required:

| Layer | Flag | Takes over | Existing connectors there |
|---|---|---|---|
| Agent function (innermost) | `--custom-agent-function-path` | agent–env loop only; miles records tokens via TITO session server | Harbor, NeMo Gym, OpenEnv |
| Generate function | `--custom-generate-function-path` | one sample's generation + token recording (raw SGLang `/generate`) | HUD, Strands, τ-bench |
| Rollout function (outermost) | `--rollout-function-path` | everything: data source, loop, rewards, batching | Verifiers (Prime Intellect) |

Supporting hooks: `--custom-rm-path` (rewards; typically reads
`sample.metadata["reward"]` that the agent function returned), `--data-source-path`
(custom task iteration), `--eval-function-path`, filter hooks, and an optional
`abort()` teardown hook on the agent-function module.

**The TITO session server** is the key mechanism for the agent-function layer: the
agent sends ordinary OpenAI-compatible `/v1/chat/completions` requests with the full
message history each turn; miles renders the chat template, preserves the exact token
ids/logprobs sampled during inference, and assembles the training sample (loss masks
included). Constraints: message history must be replayed verbatim (a
`--session-message-matcher` can loosen this), a `--tito-model` family must be
registered for the checkpoint, **no VLM support in the session path yet**, and
`--partial-rollout` is unsupported for live episodes.

All environment connectors are marked **experimental** by miles.

## The two candidate directions

### Direction A — llenvs-into-miles

llenvs ships plug-in modules (in `src/llenvs/integrations/miles/`) that a miles launch
script references by dotted path. Miles remains the training orchestrator; llenvs is
one more environment ecosystem it can train on, alongside Harbor/OpenEnv/Verifiers.

| Pros | Cons |
|---|---|
| Uses miles' *designed* extension surface; zero miles fork, connector is a small amount of code (the agent-function contract is one async function + a reward hook) | Training UX stays in miles-land: bash launch scripts, Ray cluster, Linux+GPU; llenvs config does not drive training |
| Matches llenvs' existing precedent exactly — verl/trl/openrlhf integrations (and `LLEnvsAgentLoop`) are all "trainer owns generation, llenvs drives the env" | TITO constraints leak into env usage: verbatim history replay conflicts with `history_fn` summarization / reasoning stripping; needs a "verbatim mode" or looser matcher |
| Miles keeps the hard parts (token fidelity, batching, weight sync, async scheduling, fault tolerance) — inherited for free, forward-compatible as miles evolves | llenvs envs are sync; miles rollout is high-fan-out asyncio → needs `asyncio.to_thread` + per-episode env instances (or env pooling / container server) |
| One connector exposes all ~24 llenvs adapters + machinery (extraction, cleaning, judges, iterative, dialogue, containers) that miles' per-ecosystem examples hand-roll today (e.g. its OpenEnv example reimplements fence parsing, obs capping, system prompts) | No VLM in the TITO path → vision envs need the lower-level generate-function layer; tool-calling depends on the tito-model family's parser |
| llenvs stays dependency-light: the connector imports nothing heavyweight; unit-testable against a fake session server; miles has inference-only debug modes | End-to-end validation needs a Linux GPU box; miles is fast-moving (rollout v1→v2 in transition), so the connector is maintenance-exposed |
| | Partially redundant: for Verifiers/OpenEnv/Harbor tasks, miles' direct connectors are more battle-tested than proxying through llenvs |

### Direction B — miles-into-llenvs

llenvs wraps miles as a training backend: llenvs config constructs and launches a
miles run (`llenvs train ...` → generates flags → `ray job submit`).

| Pros | Cons |
|---|---|
| Single-facade UX: envs + training in one llenvs config | Miles is not an importable library — it is a launched distributed system. "Wrapping" it means generating CLI flag arrays for a fast-churning, hundreds-of-flags surface: a brittle façade |
| Could swap trainers (verl vs miles) behind one llenvs API | llenvs would absorb trainer-scope responsibility (cluster topology, Megatron checkpoints, precision recipes, version pins like `transformers==5.12.1`) — far beyond an env library; against the lightweight/wrapper-fidelity ethos |
| | **B requires A anyway**: even when llenvs launches miles, miles' rollout still needs the agent-function/RM connector to reach llenvs envs. B is A plus a launcher wrapper, not an alternative |
| | Cannot be developed or smoke-tested on macOS; poor CI story |

## Recommendation

**Direction A: llenvs-into-miles**, primarily at the **agent-function layer**, with the
reward hook for single-turn. Direction B is strictly a superset of A whose extra layer
(launch orchestration) is the part that fits llenvs worst; if launch convenience is
ever wanted, a thin recipe script (like miles' own `examples/*/run.py`) delivers most
of it without wrapping miles.

### The connector

Implemented in `src/llenvs/integrations/miles/` (usage: `docs/guides/miles.md`):

- `config` — EvalConfig discovery via `LLENVS_MILES_CONFIG` (+ per-row metadata
  overrides), process caches, and the session-isolation guard.
- `reward` — polymorphic `reward_func` for `--custom-rm-path` wrapping `Scorer`.
- `data` — prompt-data JSONL exporter (`python -m llenvs.integrations.miles.data`).
- `agent` — `run`/`abort` for `--custom-agent-function-path`: fresh env per episode,
  verbatim append-only history, tool schemas/results, failure taxonomy
  (`completed | max_steps | context_overflow | env_error | timeout`), per-turn
  `reward_events` keyed by chat-completion response id.
- `source` — `LLEnvsDataSource` for `--data-source-path`.
- `postprocess` — v2 `--session-sample-postprocessor-path`: default postprocessing +
  `decision_spans`/`decision_rewards` attachment (the Tier-2 data foundation).
- `advantage` — `turn_grpo` for the fork's `--custom-advantage-function-path`.

Still open: a launcher recipe script, upstreaming an example to miles.

### Honest scoping notes

- For environments miles already connects to natively (Verifiers, OpenEnv, Harbor,
  τ-bench, HUD), prefer miles' direct connectors. The llenvs connector's value is the
  long tail miles has no connector for (gymnasium, craftax, jericho, webshop, alfworld,
  agentgym, gem, lmrl, dialogue, iterative coding, reasoning_gym, HF datasets, ...)
  plus llenvs' unified config/extraction/reward tooling.
- Vision environments are blocked on the TITO path; revisit via the generate-function
  layer or when miles' session server carries images.
- Everything on the miles side is experimental-tier; expect connector maintenance as
  miles evolves (it co-evolves with slime; improvements get upstreamed both ways).

## Multi-turn gap analysis: what miles cannot express that llenvs does

Verified against miles source (clone of 2026-08-30). The reward data path in miles:

- A training sample carries **one scalar reward** (`Sample.reward`; a dict is allowed
  but `--reward-key` selects a single scalar for training — `miles/utils/types.py`
  `get_reward_value`).
- `compute_advantages` receives `rewards: list[float]` — one scalar per sample. GRPO/
  GSPO broadcast it uniformly over all trainable tokens
  (`get_grpo_returns: ones_like(kl_i) * reward_i`); REINFORCE++ discounts from the
  terminal scalar. **The PPO/GAE path already has a real per-token reward channel**
  (`get_advantages_and_returns_batch(rewards_list=...)`, masked env tokens skipped,
  terminal reward at last trainable token) — but it is fed only the KL penalty today
  (`miles/backends/training_utils/loss_hub/advantages.py`).
- TITO session assembly builds **per-turn Samples internally**
  (`session/samples/merge.py`), then `merge_samples` fuses them into one contiguous
  sample: env-observation spans get loss-mask 0, per-turn structure survives only as
  the loss-mask's runs of 1s (each contiguous 1-run = one assistant turn; merges
  assert `obs_len > 0` between turns) plus timing-only `lifecycle` metadata.
- **Session v2 is materially better here**: tree metadata carries per-node
  `completion_span`s and response ids (`path_node_ids` per leaf), and the default
  postprocessor (`session/v2/postprocessor_hub/default_postprocess.py`) already does
  shared-prefix ownership masking — a shared completion is trainable only in the
  earliest committed kept leaf. It still assigns **one trajectory scalar to every
  kept leaf**, but its own comment invites replacing that via
  `--session-sample-postprocessor-path` for finer-grained assignment. So under v2,
  turn boundaries are exact node spans, not recovered loss-mask runs.
- Default reward normalization **asserts all samples sharing a `rollout_id` carry the
  same reward** (`train_data_conversion.py::_normalize_rewards_by_rollout`), i.e.
  per-turn/per-leaf rewards are actively rejected on the default path.
- Session v2 (experimental) returns one Sample per tree leaf sharing `rollout_id`;
  miles' own docs name "multi-turn with removing thinking tokens" as a
  multiple-samples-per-generate case. The Verifiers connector rejects multi-branch
  traces because flattening loses the group boundary for group-relative advantages.
- A custom generate function may return `list[Sample]` (`GenerateFnOutput`), and
  `--custom-reward-post-process-path` replaces normalization entirely. But
  `_package_shards` partitions a **fixed key list** — a novel per-token array added by
  a custom convert does not ship to the trainer without an upstream change; even
  `train_data["metadata"]` (from `train_metadata`) is built but not in the partition
  list. `--rollout-data-postprocess-path` receives only `args` (megatron actor), so it
  cannot inject rewards.

### Gap table

| llenvs multi-turn capability | miles today | Severity |
|---|---|---|
| Turn-level STEP rewards (per-transition `SignalBundle`) | One scalar per trajectory; GRPO smears it over all tokens | **High** — the headline gap |
| Multi-signal rewards (named/typed/weighted, e.g. OUTCOME + FORMAT + StepPenalty) | Single training scalar; dict rides along, one key trains | Medium — aggregate llenvs-side, breakdown for logging/filtering |
| Sub-response segment rewards (`SegmentedEnvironment`) | Nothing below sample granularity | Medium-low — needs the token-level channel |
| Feedback text signals → next observation (iterative envs) | No gap: agent function owns the loop; feedback enters as env-role tokens, masked 0 | None |
| Weight-0 monitoring signals (tool validity/efficiency) | No native concept | Low — metadata + custom log/filter hooks |
| History rewriting (`history_fn`, `prompt_budget`, `include_reasoning_in_history=False` **default**) | TITO v1 requires verbatim replay; v2 branches (per-leaf samples) | Medium — connector must run a verbatim mode by default |
| Env-truncation vs token-truncation (`terminated`/`truncated`) | `Sample.Status.TRUNCATED` means token-length truncation only | Low — pass env truncation in metadata, filter hook if needed |
| Env-side LLM (dialogue envs, judges) | Anything sent to the session URL is recorded as policy tokens | Low — route env-LLM/judge to a non-session backend (config discipline) |
| Tool-calling envs | Works: native FC via `--tito-model` parser, or llenvs Hermes text parsing inline | None (per-model constraint) |
| Branching / tree exploration (`BranchManager`) | v2 tree sessions exist, but group-relative advantage over tree leaves is unsolved (Verifiers connector rejects it) | Deferred — research-tier |
| Multi-turn VLM envs | No images on the TITO session path | Deferred — upstream limitation |

### Remedies: the reward-granularity ladder

**Tier 0 — trajectory scalar (default; zero miles changes).** The connector reduces
each episode: weighted OUTCOME total + (optionally discounted) sum of STEP/FORMAT/
PROCESS totals → `Sample.reward`. Full per-signal breakdown and the per-turn reward
list ride in `sample.metadata["signals"]` for logging (`--custom-rollout-log-function-path`),
filtering (`--dynamic-sampling-filter-path`, `--rollout-sample-filter-path`), and
offline analysis. Aggregation policy (discount γ, which types count) is a connector
knob.

**Tier 1 — per-turn sample splitting: REJECTED.** Splitting each episode into
per-turn Samples sharing one `rollout_id` (sample *t* = prefix context + turn *t*
trainable, return-to-go reward) works through documented hooks, but the trainer then
re-forwards shared prefixes — total processed context ~quadratic in turns. The user
has explicitly ruled out any design that adds forward passes per turn. Keep episodes
as **one merged sample with one forward pass**; do credit assignment on advantage
vectors, never by sample cloning.

**Tier 2 — turn/token-level credit via transport + advantage hook (upstream PR).**
The clean shape (one forward per root-to-leaf trajectory):
1. Agent function returns reward events keyed by **response id** in metadata (already
   crosses the session boundary).
2. A custom v2 `--session-sample-postprocessor-path` joins events to node
   `completion_span`s and attaches turn spans + per-turn rewards to the merged sample
   (the default postprocessor already has the span/ownership machinery to reuse).
3. Upstream additions: response-aligned fields (`turn_ids`, `env_rewards`, or
   precomputed `advantages`) in `ROLLOUT_DATA_VALUE_SPEC` + the `_package_shards`
   allowlist, and a `--custom-advantage-function-path` hook (the conspicuous missing
   dotted-path hook — rollout, rewards, conversion, and loss all have one; the
   advantage dispatcher is closed and raises on unknown estimators).
4. The existing policy loss consumes token-aligned advantages unchanged.

**Algorithmic caveat**: do NOT naively feed turn rewards into the existing per-token
GAE (`get_advantages_and_returns_batch`) — it discounts per trainable *token*, so a
reward on the last token of a 1000-token turn discounts γ^1000 within one macro-action.
Turn-level credit must discount per *decision* and broadcast each turn's advantage
over its completion span (turn-GRPO/turn-REINFORCE first; turn-PPO needs
decision-boundary value extraction). Length weighting (uniform vs span-normalized)
should be an explicit knob. This also covers segment-level rewards.

### Non-reward remedies

- **Verbatim history mode**: the miles connector forces `include_reasoning_in_history=True`
  and rejects `history_fn`/`prompt_budget` under session v1; under session v2 rewritten
  history is supportable (per-leaf samples, one shared episode reward — miles' own
  answer for thinking-token removal). Document `--session-message-matcher` choices.
- **Env-LLM / judge isolation**: `EnvironmentLLMConfig` and judge backends must point
  at an external endpoint, never the session `base_url`. Enforce in the connector
  (refuse to construct a dialogue/judge env whose backend is the session URL).
- **Env truncation**: agent function writes `env_truncated: bool` into metadata; ship
  an optional sample filter that zeroes loss or drops those episodes.
- **Monitoring signals & eval**: map `SignalBundle` breakdowns to metrics in custom
  rollout/eval log functions (per-signal means, tool-validity rates).
- **Deferred**: tree-based training over v2 branches (group-boundary problem),
  multi-turn VLM (TITO limitation), generation-time segmentation.

## Tier-2 fork patch specification

The llenvs side of Tier 2 ships in the connector (`postprocess` records
`decision_spans`/`decision_rewards` in sample metadata; `advantage.turn_grpo` is the
trainer-side hook target). The miles side is a four-file fork patch, applied to a
separate miles checkout — not to this repo. Division of labor is dictated by two
verified transport facts: `sample.metadata` crosses the v2 samples wire while
`train_metadata` does not, and group-relative normalization must run rollout-side
**before** DP sharding (post-shard ranks only see group slices).

**1. `miles/utils/arguments.py` — new flags**

```
--custom-advantage-function-path   dotted path; when set, compute_advantages_and_returns
                                   dispatches to it instead of the built-in estimators
--turn-level-credit                enable decision-span transport + rollout-side credit
--turn-gamma FLOAT                 per-DECISION discount for return-to-go (default 1.0)
--turn-grpo-std-normalization      divide by group std of pooled decision returns (default on)
--turn-advantage-length-weighting  {uniform, span_normalized} (default uniform)
```

**2. New `miles/rollout/turn_credit.py` — rollout-side per-decision advantages**

`compute_decision_advantages(args, samples)`: for each sample read
`metadata["decision_spans"]` / `metadata["decision_rewards"]`; compute per-decision
return-to-go with `turn_gamma` discounted per decision (never per token — the existing
per-token GAE would discount γ^1000 across a 1000-token turn); normalize group-relative
(mean, optionally std) over the **pooled decision returns of the group**, using the same
grouping as `_reward_group_segments`; write `metadata["decision_advantages"]`. Runs in
the rollout flow right before conversion, where the whole group is visible.

**3. `miles/ray/rollout/train_data_conversion.py` — transport**

When `--turn-level-credit`: copy `decision_spans` and `decision_advantages` from sample
metadata into `train_data`; add both keys to `ROLLOUT_DATA_VALUE_SPEC` and to the
per-sample key list in `_package_shards`. `_normalize_rewards_by_rollout` stays
unchanged (sibling leaves still share the trajectory scalar; the scalar path remains
intact as fallback).

**4. `miles/backends/training_utils/loss.py` — the hook site**

In `compute_advantages_and_returns`, when `args.custom_advantage_function_path` is set,
dispatch `load_function(path)(args, rollout_data, kl)` in place of `compute_advantages`
(this is the one place that holds `rollout_data`); OPD application and
`normalize_advantages` on top stay unchanged. The hook returns `(advantages, returns)`
as lists of per-sample CP-local response-aligned tensors, exactly like
`compute_advantages`.

**Launch** (on the fork):

```
--turn-level-credit --advantage-estimator grpo
--custom-advantage-function-path llenvs.integrations.miles.advantage.turn_grpo
--session-sample-postprocessor-path llenvs.integrations.miles.postprocess.postprocess
```

Estimator caveat: turn-GRPO with a pooled group baseline is a pragmatic first
estimator; per-turn-index baselines and turn-PPO (decision-boundary value extraction)
are research knobs on top of the same transport.
