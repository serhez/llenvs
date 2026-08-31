# miles Training

[miles](https://github.com/radixark/miles) is an RL post-training framework (SGLang rollouts + Megatron/FSDP training). llenvs plugs into it from the outside: `llenvs.integrations.miles` ships entry-point modules that miles loads via its dotted-path CLI flags. miles stays the training orchestrator; no miles code is imported at llenvs import time.

Install the extra:

```bash
uv pip install -e ".[miles]"   # openai + httpx (session client)
```

## How configuration reaches the plug-ins

miles loads the entry points by dotted path inside its own worker processes, so they cannot receive Python objects. The llenvs `EvalConfig` YAML is discovered through an environment variable — export it in the miles launch script so every worker inherits it:

```bash
export LLENVS_MILES_CONFIG=/path/to/config.yaml
```

Per-row overrides in sample metadata take precedence: `metadata["llenvs_config"]` (config path) and `metadata["llenvs_env_name"]` (which `environments:` entry to use; also settable process-wide via `LLENVS_MILES_ENV`). A config with a single environment needs no selection.

## Recipe A — single-turn RLVR (custom RM + prompt data)

miles generates completions itself; llenvs scores them through the environment's extractors and reward functions (a cached `Scorer` under the hood). No session server needed.

Export the task set as prompt-data JSONL:

```bash
python -m llenvs.integrations.miles.data \
    --config config.yaml --output tasks.jsonl [--num-tasks N] [--env NAME]
```

Each row is `{"prompt": [<chat messages>], "label": "<ground truth or ''>", "metadata": {"task_index": ...}}`. The system prompt from the config (env-level overrides eval-level) is prepended to the messages. `metadata.task_index` is how the RM and the agent find the row's task; rows are deterministic for a given config. Image tasks are refused (the TITO path is text-only).

Launch flags:

```bash
--prompt-data tasks.jsonl
--custom-rm-path llenvs.integrations.miles.reward.reward_func
```

`reward_func` is polymorphic (single sample or batch), short-circuits on a precomputed `metadata["reward"]`, and otherwise scores `sample.response` against `metadata["task_index"]`. Scoring is serialized with a lock because the cached Scorer holds one shared environment.

## Recipe B — multi-turn agentic (TITO session server v2)

The agent function drives `Environment.reset/step` against a per-episode session URL. Every chat request through that URL is recorded token-exact as the trainable trajectory — no re-tokenization, no chat-template drift.

```bash
export LLENVS_MILES_CONFIG=/path/to/config.yaml

--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path llenvs.integrations.miles.agent.run
--use-session-server v2
--session-sample-postprocessor-path llenvs.integrations.miles.postprocess.postprocess
--tito-model <model family>          # tool-call parser for the policy model
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
--prompt-data tasks.jsonl            # exported as in Recipe A
```

Under v2 with the default postprocessing, the agent's returned `reward` is assigned server-side and `--custom-rm-path` is skipped; installing `reward_func` anyway is harmless (it short-circuits on `metadata["reward"]`).

What the agent does per episode:

1. Creates a **fresh environment** (llenvs environments enforce state continuity; sync `reset`/`step` run under `asyncio.to_thread`).
2. Resets to `metadata["task_index"]`, builds the opening messages ([system?] + user prompt + replay of `obs.messages`), and loops: chat request → echo the assistant message **verbatim** (`model_dump(exclude_none=True)`, reasoning fields included) → llenvs `Action` (tool calls parsed, else text) → `env.step` → feedback appended as `tool`/`user` messages.
3. Tool schemas from `observation.available_tools` are sent on every request; results come back as `role: "tool"` messages with matching `tool_call_id`.
4. Stops on `terminated`/`truncated`, `max_steps` (metadata override, else `spec.max_steps`, else 100), context overflow (HTTP 400/409 from the session), or the episode timeout (`LLENVS_MILES_EPISODE_TIMEOUT` seconds, default 3600).

The returned dict:

| key | meaning |
| --- | --- |
| `reward` | Tier-0 scalar: sum of transition totals (equals `Trajectory.total_reward`); 0.0 on `timeout`/`env_error` |
| `exit_status` | `completed` \| `max_steps` \| `context_overflow` \| `env_error` \| `timeout` |
| `env_truncated` | environment truncation flag |
| `num_steps`, `turn_rewards`, `signals` | per-episode diagnostics (signals are per-name sums) |
| `reward_events` | per-turn `{response_id, value, signals}` — consumed by the postprocessor (see below) |
| `agent_metrics.total_tool_time` | seconds spent in `env.step` |

Misconfigurations fail loudly instead of training on garbage: a missing `task_index`, an image observation, or a judge/env-LLM whose `base_url` points at the session endpoint all raise, miles marks the sample ABORTED, and `check_no_aborted` regenerates it.

### Hard constraints on the TITO path

- **Verbatim, append-only history.** No `history_fn`, no prompt budgets, no reasoning stripping — a modified history silently branches the v2 session tree into junk samples.
- **Text-only.** Vision environments are refused; use a generate-function-level integration for VLM training.
- **Judges and env-internal LLMs need their own serving endpoint.** Traffic against the session URL would be recorded as trainable tokens; the isolation guard refuses configs whose judge/env-LLM `base_url` matches the session endpoint.
- Tool-call quality depends on `--tito-model`'s parser matching the policy model family.

### The v2 postprocessor

`postprocess` runs miles' `default_postprocess` unchanged (shared-prefix ownership masking, agent-metadata merge, reward assignment), then joins `reward_events` to the session tree's node completion spans and writes response-relative `decision_spans` / `decision_rewards` into each sample's metadata. Without reward events the join is a no-op, so the hook is always safe to install. This is the data foundation for turn-level credit (below).

## Recipe C — direct DataSource (skip the JSONL export)

Serve tasks straight from the environment instead of a pre-exported file, so the task set and the episodes always come from the same config:

```bash
--data-source-path llenvs.integrations.miles.source.LLEnvsDataSource
--disable-rollout-global-dataset
```

`LLEnvsDataSource` mirrors the stock source's semantics: groups of `n_samples_per_prompt` deepcopies, buffer drained first, seeded per-epoch shuffle (`--rollout-shuffle` / `--rollout-seed`), epoch wraparound, and JSON save/load state under `<save>/rollout/`. Rows are identical to the exporter's.

## Evaluation

Export a held-out task file and pass it as `--eval-prompt-data`; the same RM (Recipe A) or agent function (Recipe B) scores it.

## Turn-level credit (Tier 2, requires a miles fork)

Tier 0 trains on one scalar per trajectory (GRPO broadcasts it over all trainable tokens). For per-decision credit **without any extra forward passes** — one forward per merged trajectory, credit carried by advantage vectors — llenvs ships the trainer-side hook:

```bash
--turn-level-credit --advantage-estimator grpo
--custom-advantage-function-path llenvs.integrations.miles.advantage.turn_grpo
--session-sample-postprocessor-path llenvs.integrations.miles.postprocess.postprocess
```

`turn_grpo` broadcasts precomputed per-decision advantages over each decision's response-token span and slices to the local CP rank; samples without decision data degenerate to exactly GRPO. The per-decision advantages themselves (return-to-go with a per-decision discount, group-relative normalization) are computed rollout-side by the fork, where the whole group is still visible. The fork patch (new flags, rollout-side `turn_credit`, the transport allowlist additions, and the `compute_advantages_and_returns` dispatch) is specified in `docs/design/miles-integration.md`.

## Testing

The integration is covered by mock-transport tests (`tests/test_miles_agent.py` fakes the session server with `httpx.MockTransport`; `tests/test_miles_integration.py` covers config discovery, RM, exporter, DataSource, and postprocessor; `tests/test_miles_advantage.py` covers the Tier-2 hook). End-to-end training requires a Linux + GPU box with miles installed; miles' inference-only debug mode is the cheapest full-stack validation.
