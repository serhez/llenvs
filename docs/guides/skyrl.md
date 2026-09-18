# SkyRL text integration

`llenvs.integrations.skyrl` supplies indexed environments, text/tool episodes, optional judges/token scorers, and native or custom credit to SkyRL. SkyRL owns training, inference engines, placement, optimization, and checkpoints. A native launch entrypoint and resume identity checks are implemented, but **installed-runtime and GPU acceptance have not been completed**. Do not treat component tests as certification of sampling parity, packing, distributed execution, or vision-language training. VLM rendering is not implemented.

## Prepare tasks

Install llenvs and the selected environment adapter. Task preparation does not require a SkyRL installation or construct the YAML's policy or judge backends.

Plain llenvs and data/scoring imports do not probe optional ML/GPU dependencies. Adapter dependency probes run when an adapter is selected or availability is explicitly listed. Actual environment construction can import that adapter's dependencies.

```python
from llenvs.core.config import EvalConfig
from llenvs.integrations.skyrl.data import export_prompt_data

config = EvalConfig.from_dict({
    "environments": [{
        "name": "leg_counting",
        "adapter": "reasoning_gym",
        "size": 144,
        "seed": 42,
    }],
    "system_prompt": "Give your answer in <answer>...</answer> tags.",
})

export_prompt_data(config, "train.jsonl", indices=range(128))
export_prompt_data(config, "eval.jsonl", indices=range(128, 144))
```

`config` can also be a YAML path. Choose `env_name` when several environments are configured. Use either `indices` (preserving their order) or positive `num_tasks` (starting at zero); omitting both exports all tasks. Empty selections, duplicates, booleans, and out-of-range indices fail.

For an equivalent YAML configuration, the CLI accepts contiguous ranges:

```sh
python -m llenvs.integrations.skyrl.data --config env.yaml --output train.jsonl --num-tasks 128
python -m llenvs.integrations.skyrl.data --config env.yaml --output eval.jsonl --start 128 --num-tasks 16
```

`--start` defaults to zero; omitting `--num-tasks` selects the remaining tasks. Use `--env NAME` to select among multiple configured environments. Existing output files are never overwritten.

The adapter must support finite task indexing and synchronous reset/cleanup. The constructor receives the configured seed; resets use only `options={"task_index": index}`. Preparation owns and closes its environment. Construction/reset can have side effects, including calls to an environment-owned LLM if configured. Existing output paths are refused; a completed temporary file is published without overwriting, only after cleanup succeeds. Filesystems must support same-directory hard links for this atomic publication.

Policy model, inference, batch, and evaluation-limit settings do not control preparation. Environment system prompts override the global system prompt; named prompts/fragments use the core resolver. Runner `model_profile`, global/environment `prompt_template`, and environment `branching_strategy` settings are rejected rather than silently applied.

### Rows and identity

Each JSONL row contains exactly:

| Field | Contents |
| --- | --- |
| `prompt` | Optional resolved system message, user task prompt, then validated initial history. Available tools add the core Hermes preamble to the system message. |
| `env_class` | `"llenvs"` |
| `data_source` | `"adapter/environment-name"` |
| `llenvs` | `schema_version: 1`, `env_fingerprint`, `task_index`, `task_id`, and `initial_fingerprint`. |

SHA-256 identities cover canonical, finite JSON. The environment fingerprint includes the selected environment configuration, constructor seed, resolved system prompt, and opening-message format. Connector-added judge settings and unused runner settings are excluded. Task identity adds the index; the initial fingerprint covers the complete opening conversation, including ordered images and tool definitions.

No hidden answer, arbitrary reset metadata, volatile episode ID, or configuration override is exported. This is not prompt redaction: anything the environment intentionally places in visible messages remains visible. Pin external task-data revisions separately; equal visible prompts do not prove equal hidden tasks.

Initial history preserves `system`, `user`, `assistant`, and `tool` roles. Unsupported fields or modalities fail. Tool histories require OpenAI function-call records, serialized argument strings, and matching tool results. Images use inline base64 PNG/JPEG/WebP/GIF URLs, with task images before state images; validation checks MIME/base64 structure, not image decoding or model compatibility.

The internal `LlenvsPromptDataset(paths, *, env_fingerprint)` validates every row, its identity, and duplicates across files. It returns copied `(messages, "llenvs", extras, task_id)` items with list-of-dictionaries collation. It stores only plain JSON and does not filter tasks or tokenize prompts. The expected fingerprint must come from the independently selected configuration, not be trusted merely because a dataset row supplies it. `validate_initial_messages(row, messages)` checks fresh-reset inputs against a prepared row. `validate_task_splits(train, evaluation, *, train_batch_size)` rejects overlapping task IDs and a non-divisible training tail; it never drops or rearranges rows. Rendering and native runtime preflight remain separate requirements.

## Internal text episode boundary

The internal generator drives fresh environment instances through SkyRL-shaped raw-token requests. It preserves sampled token IDs and original chosen-token logprobs, parses tool calls only for environment execution, and records observation/template additions with zero policy mask and reward. Multiple tools in one reply are one policy decision. Changes to the advertised tool schema fail. Empty generations, missing requested logprobs, sentinels, and protocol misalignment abort the group rather than inventing training data.

Text observations use the tokenizer's fixed-base multi-turn convention, with prefix equality checked before extracting the new context suffix. Prior sampled assistant text is never re-encoded. The component admits only the `enable_thinking` template kwarg; custom-template re-tokenization, continuation-only mode, and image expansion require separate support. Exporting image-bearing tasks does not imply that this text generator can train on them. Synthetic EOS is context only; unused terminal/over-budget observations are omitted.

Opening tool histories retain OpenAI argument strings in the exported/fingerprinted rows. Rendering converts these strings to JSON objects on an owned copy, as required by Hugging Face tool templates; otherwise Qwen double-encodes the arguments. Null assistant content is accepted only with tool calls and renders as empty text. Invalid, duplicate-key, non-object, or non-finite argument JSON fails before inference. This conversion never touches a sampled reply. [Template argument contract](https://huggingface.co/docs/transformers/chat_extras).

Native preparation resolves logprob mode, context length, and model/output vocabulary from the native vLLM argument builder and local model configuration. These are not inferred from tokenizer length or invented typed configuration fields. `algorithm.max_seq_len` is a loss normalizer and native warning threshold: it does **not** cap or truncate episodes. Generation uses the actual engine context limit, per-call generation/input limits, and decision count. Effective GPU execution still requires runtime validation.

Native environment rewards retain their weighted-sum semantics. Explicit extra judges use existing templates, score extraction, normalization, and weights through a strict path: parse errors, backend errors, and truncated responses fail. Decision judging attaches to that generation's final sampled token; episode judging sees the included transcript and attaches only to the episode's final sampled token. Judge image access is opt-in and does not add images to policy history. Duplicate configured/native judge names are rejected.

One driver-local semaphore bounds live episodes across concurrent calls. Blocking environment/judge work runs on a shared executor, serially per owned resource. Cleanup waits for pending operations and uses a bounded async deadline; it cannot stop a blocked external call or guarantee cleanup after process death. Such adapters need their own operation timeouts/leases, and thread-affine adapters need separate verification.

## Token-scoring contract

`scoring.GenerationScoreInput` supplies unique occurrence/generation IDs, the native task `instance_id` and repetition ID, actual input/output token IDs, prior conditioning, and model/tokenizer/processor provenance. Token IDs become immutable tuples; conditioning and provenance become recursively immutable, owned JSON-like snapshots. These requests are transient driver inputs, not serialized trajectory buffers. Native task IDs are distinct from sampling-occurrence group IDs used for custom normalization.

An async `GenerationScorer` declares `reward_semantics = "prefix_causal_additive"` and returns `GenerationTokenRewards` with matching occurrence/generation IDs and exactly one finite immediate reward per sampled output token—including sampled EOS. For example, a deliberately simple, causal scoring rule is:

```python
from llenvs.integrations.skyrl.scoring import (
    GenerationScoreInput,
    GenerationTokenRewards,
)

class TokenIndicator:
    reward_semantics = "prefix_causal_additive"

    def __init__(self, target_token_id: int):
        self.target_token_id = target_token_id

    async def __call__(self, generation: GenerationScoreInput) -> GenerationTokenRewards:
        return GenerationTokenRewards(
            occurrence_id=generation.occurrence_id,
            generation_id=generation.generation_id,
            rewards=tuple(float(token == self.target_token_id) for token in generation.output_ids),
        )
```

This example defines a token-count objective, not a recommended task reward. A real scorer must have a fixed, independently tested recipe. Each label may depend on actual conditioning and tokens through its own action, but not later sampled actions, later feedback, or final episode outcomes. The interface exposes the whole current generation, so immutability and a declared semantics string cannot establish causality. Test scorer adapters with future-token perturbations. Quality scores, values, advantages, and future-conditioned labels are not automatically additive immediate rewards.

The internal `score_generation` boundary validates identity, length, finite components, and finite weighted results. It does not retry, substitute zero on failure, or suppress cancellation. The internal episode driver constructs one fixed scorer on its running loop, scores each generation before its environment transition, and closes optional async `aclose()` resources on that loop. A cancelled initialization waiter does not discard ownership. Scorers must support concurrent calls and avoid blocking the driver loop; an in-process GPU reward model has no implicit resource allocation.

## Credit and token correctness

Internal credit helpers operate on effective, left-padded response rewards before native sharding/packing. They require complete real prompt groups and explicit generation spans in unpadded response coordinates. Observation gaps, padding, and synthetic rows receive zero credit. Original sampled membership—not the effective policy-loss mask—defines the return clock and normalization population.

The internal trainer mixin validates complete real rows, invokes native tensorization/boundary construction, clears rewards on known appended DP dummies, and computes custom credit once before forwarding. Native dummy padding copies row 0's rewards; a zero loss mask alone is insufficient. Occurrence IDs keep repeated task visits separate for prompt boundaries; stable task IDs remain available for native accounting. The ledger is consumed locally, never transported as unsliced worker metadata. The later advantage hook checks the precomputed tensors without replacing them with a stock estimator.

- `llenvs_turn_grpo`: rewards occupy generation endpoints; discounted returns advance once per generation. Returns are centered across all generations in a complete prompt group, optionally divided by sample standard deviation plus `1e-6`, then broadcast across each generation's sampled tokens. Constant/singleton pools produce zero. Optional `span_normalized` weighting divides by the original sampled length.
- `llenvs_token_rtg`: additive rewards occupy sampled tokens; undiscounted reverse sums advance only over sampled actions. No group centering, standard-deviation normalization, or span normalization is permitted. Equal episode totals can produce different token credit.

The entrypoint registers these actual functions in SkyRL's existing registry before native configuration validation. Calling either through a stock trainer without the required attribution ledger fails. No native scalar-GRPO implementation, loss, packing logic, or worker code is replaced. The helpers reject malformed attribution, unsupported reward placement, non-finite values, and overflow; they preserve float32/float64 dtype, device, and independent outputs.

Recorded-generation checks require exact append-only token prefixes, valid vocabulary IDs, and one finite chosen-token logprob per sampled output. Empty generations and missing-probability sentinels fail. They never decode/re-encode, fabricate EOS, repair boundaries, or construct packed offsets. Structural validation cannot establish actual endpoint provenance or training/inference probability equality.

Validation distinguishes native scalar sampling from explicitly unmodified custom-credit training. The latter requires raw logprobs, neutral sampling transforms, and an effective `min_tokens=0` request; incompatible reward rewrites and duplicate importance corrections fail. The native generator wrapper applies this one opt-in change after native request construction, for training only. Evaluation retains its separate native sampling parameters, including greedy defaults. Structured attribution and reward diagnostics never enter the native numeric environment-metric aggregator.

Leave `generator.eval_sampling_params` unset to inherit native greedy evaluation and the training generation limit. If you override any nested evaluation field, explicitly include `generator.eval_sampling_params.temperature=0.0` when you want greedy evaluation: the native builder constructs a new sampling object, whose temperature defaults to 1.0.

## Native launch and checks

Use the separately staged Linux/Python 3.12 SkyRL environment at source revision `4f5ccd8e58bbcea4804bd831fd47097c3044ff48`. Preserve its native lock/source overrides when adding llenvs and the selected adapter dependencies; there is no `llenvs[skyrl]` GPU-stack installer. Preflight requires the audited Ray 2.57.0, Torch 2.13.0+cu130, vLLM 0.28.0, Transformers 5.16.1, and OmegaConf 2.3.1 versions. Platform wheels, CUDA/driver compatibility, and hardware capacity still need deployment acceptance.

The bounded launch recipe admits staged dense Qwen2 causal-model snapshots (such as Qwen2.5 Instruct), FSDP/BF16/FlashAttention 2, training DP 1 or 2, and inference TP/PP/DP/EP 1 on one node. Model/tokenizer files and weights must already exist locally; preflight never downloads them. It hashes file contents, rejects custom model/tokenizer code, MoE/quantized/sliding-window variants, and missing indexed weight shards. Other model families and VLM require separate implementation/acceptance.

Paths below are illustrative staged deployment paths, not a verified training run:

```sh
python -m llenvs.integrations.skyrl.entrypoint \
  llenvs.config=/opt/run/llenvs.yaml \
  trainer.policy.model.path=/opt/models/qwen-text \
  data.train_data='["/opt/run/data/train.jsonl"]' \
  data.val_data='["/opt/run/data/eval.jsonl"]' \
  data.dataloader.num_workers=0 \
  trainer.train_batch_size=4 trainer.policy_mini_batch_size=4 \
  trainer.eval_batch_size=4 generator.n_samples_per_prompt=2 \
  trainer.logger=console trainer.resume_mode=none \
  trainer.max_training_steps=2 trainer.ckpt_interval=1 trainer.eval_interval=1 \
  trainer.ckpt_path=/opt/run/smoke/checkpoints \
  trainer.export_path=/opt/run/smoke/exports \
  trainer.log_path=/opt/run/smoke/logs \
  llenvs.check_only=true
```

Use native `key.path=value` syntax, not Hydra `+` overrides. `--help` does not import SkyRL. `llenvs.check_only=true` validates native/connector configuration, selected imports, local data/model identity, and resolved engine arguments without Ray initialization, environment reset, model allocation, scoring, or checkpoint writes. Hashing a model reads its weight files; it is not a quick file-existence check. Rendered prompt budgets are checked on real launch, after local tokenizer loading and before placement allocation; oversized tasks fail rather than being filtered out.

Set `llenvs.check_only=false` only in the staged runtime when ready to launch. The driver rechecks code/data/model/runtime identity before creating the experiment. Configured output paths must be local absolute paths. The YAML's policy/evaluation runner does not control this run.

Select credit with `trainer.algorithm.advantage_estimator` alone:

| Estimator | Required connector controls |
| --- | --- |
| `grpo` (default) | Native scalar behavior; no token scorer. |
| `llenvs_turn_grpo` | `llenvs.sampling_contract=unmodified`; optional `llenvs.turn_weighting=span_normalized`. |
| `llenvs_token_rtg` | `llenvs.sampling_contract=unmodified`, gamma 1, and `trainer.algorithm.grpo_norm_by_std=false`. |

Synchronous training uses native `regular` loss. Fully async selects the native async trainer, requires `trainer.fully_async.enabled=true`, `trainer.placement.colocate_all=false`, unmodified sampling, and `trainer.algorithm.policy_loss_type=rollout_is`. Keep train/policy prompt batch sizes equal; explicitly size `num_parallel_generation_workers` within the native staleness window. Additional TIS with `rollout_is` fails. Ordinary native packing/count-based or whole-row token-budget microbatching remains native; positive token budgets are not truncation limits. Entropy with token-budget batching is rejected, old/update forward microbatch settings must match, and real minibatch rows must be divisible by DP.

Step-wise/merged/batched generation, unreviewed losses/corrections, history compaction, custom template replacement, inference model parallelism, Megatron, LoRA, MTP/speculation, R3, FP8/QAT, compiled/fused training, external/PD-split engines, and alternative weight transports are rejected. Engine kwargs use an explicit reviewed allowlist. Ambient execution overrides are checked too: leave `VLLM_USE_V1` and `VLLM_ENABLE_V1_MULTIPROCESSING` unset on the launching shell so native runtime injection supplies both; unknown kernel/plugin controls fail instead of silently changing the recipe.

### Judges, credentials, and resume

`llenvs.config` selects the YAML; `llenvs.env_name` disambiguates multiple environments. Environment-level extra judges override global judges, and an empty list disables extras. `llenvs.judge_timing` is `decision` or `episode`; `llenvs.judge_use_images` defaults false. The launch profile permits remote OpenAI/Anthropic/OpenRouter/LiteLLM judges and env-LLMs, not implicit driver-local GPU placement. Token scorers use `llenvs.token_scorer={factory,revision,kwargs,weight}` and need their own independently tested causal recipe and resource discipline.

Supply credential **names** through `llenvs.forward_env`, with values in the launching environment. Missing names and reserved runtime controls fail. Values travel through the driver runtime environment, never the native configuration or run manifest. Configuration is public: do not hide secrets in arbitrary scorer kwargs; common credential keys are rejected, but this is not a general secret detector. `llenvs.max_active_episodes` defaults to 32 and bounds live resources across concurrent calls.

`llenvs-run.json` stores immutable hashes of ordered task content/IDs, selected environment/prompt, reward/scorer/judge recipes, model contents, execution settings, source code, and dependency inventory. It stores no trajectories, task text, live state, or forwarded credentials. Fresh runs never overwrite it; `latest` compares it, and `from_path` checks the manifest beside the selected `global_step_N` directory. Existing unidentified checkpoint directories and changed identities fail. Pin external task/scorer revisions: a manifest cannot detect an external service changing semantics behind an unchanged declared revision.

Checkpoint selection/loading and consumed-task bookkeeping remain native. Resume may regenerate unconsumed work in fresh sessions; it neither restores queued rollouts nor makes external tool effects exactly-once. Installed-runtime checkpoint/resume, Ray cancellation/hard-exit, and numerical acceptance remain pending.

Generator failures drain owned episodes and close scorer/judge resources before propagating. Cleanup failures are retained even after finished tasks leave the live-task set, without relabeling ordinary rollout failures. An uncooperative operation can exceed the cleanup deadline; deferred close remains queued but cannot survive process death. Failures elsewhere in the native worker can hard-exit without entering generator cleanup at all. Resource-owning adapters still need operation timeouts and external leases.

At the audited pin, CPU execution of the native loop/checkpoint bodies confirms an async resume bug: the final save records the next-step counter, which can skip untrained tasks and inflate accepted-rollout accounting on resume. This affects both `latest` and `from_path` selecting that final checkpoint. Resuming a periodic checkpoint at the configured step limit can also execute one extra update. Do not use these paths as accepted recipes; the connector does not repair or automatically reject them. The acceptance harness uses periodic `global_step_1`, below its limit, while general async resume awaits an upstream correction/re-audited pin. [Native async checkpoint ordering](https://github.com/NovaSky-AI/SkyRL/blob/4f5ccd8e58bbcea4804bd831fd47097c3044ff48/skyrl/train/fully_async_trainer.py#L648).

## Acceptance tests

Ordinary `tests/test_skyrl_*.py` tests do not launch a native runtime or download models. Independent reward oracles cover randomized spans/gaps/padding and exact tiny-policy gradients. A tiny in-memory CPU model compares prefix-by-prefix chosen probabilities with the native wrapper, then connects production credit to native losses/corrections and independent parameter gradients. Actual token bins, whole oversized rows, restored tensor order and zero-gradient balancing dummies have CPU tests. Source-boundary tests also execute pinned collection, tensorization, config and checkpoint-control bodies with explicit external-boundary fixtures; they do not certify the installed stack or GPU kernels. Set `LLENVS_SKYRL_SOURCE` to an already staged checkout if it is not at `.local/upstream/skyrl`. Extra correction audit cases do not expand the admitted configuration.

`test_skyrl_tokenizer.py` uses the cached Qwen2.5-1.5B-Instruct tokenizer at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, verifying five file hashes. It checks actual template suffixes, byte-level non-round-tripping, and opening tool histories. `LLENVS_SKYRL_TOKENIZER` can select an explicit local snapshot of those same files. Missing default assets skip; an explicitly selected invalid snapshot fails. No weights are required. Results on the development Transformers version do not establish behavior on the pinned training runtime.

Seven owned CPU subprocess cases compose the unchanged native async fatal/cancellation body with production generator cleanup. They cover concurrent failures, a blocked operation released before its deadline, an uncooperative operation, session-release failure, cancellation and a failure outside the generator. The last case explicitly proves cleanup can be bypassed. These use fixture resources, not Ray actors or external-resource leases.

`test_skyrl_runtime.py` requires the installed SkyRL stack. The separate `test_skyrl_gpu.py` contains 17 authored, unrun cases: five layout/gradient cases, six sync/async × reward-mode training/resume cases, four DP1/DP2 comparisons and two mid-generation cache/weight-sync cases.

Run GPU tests serially, without pytest-xdist, in an allocated Linux/CUDA job with the pinned runtime, this checkout importable on all workers, pytest, and a fully staged admitted model. Select GPU visibility through the job allocation; merely starting a local Ray instance does not reserve otherwise busy hardware. Set these test-only controls:

| Variable | Requirement |
| --- | --- |
| `LLENVS_SKYRL_GPU` | Exactly `1`; otherwise the entire GPU module skips before native imports. |
| `LLENVS_SKYRL_MODEL` | Absolute staged model directory, including weights and tokenizer. |
| `LLENVS_SKYRL_ARTIFACTS` | Absolute writable artifact parent; each case creates its own directory. |
| `LLENVS_SKYRL_TOLERANCES` | Absolute reviewed JSON file, described below. |
| `LLENVS_SKYRL_DP` | Smoke-training DP `1` (default) or `2`. Sync needs DP GPUs; async needs DP plus one inference GPU. Layout cases use one GPU. DP-comparison and cache component probes always require two GPUs, independently of this variable. |
| `LLENVS_SKYRL_TIMEOUT_SECONDS` | Per-native-subprocess deadline, default 1800; allowed range 1–7200. |

The tolerance JSON must contain exactly `model_content_hash`, `runtime_hash` (64-character lowercase SHA-256 strings), and `bounds`. Bounds must contain exactly `logprob_max_abs`, `gradient_relative_l2`, `update_relative_l2`, and `rollout_logprob_max_abs`, each a finite nonnegative number. The update bound applies to optimizer deltas, not parameter magnitudes. No numerical defaults are supplied. Use independent baseline measurements and review them for the checkpoint, kernels, dtype, hardware and intended layouts; never loosen a bound just to pass the candidate. Model/runtime hashes must match, while recorded hardware must also be reviewed separately.

Obtain the identity hashes read-only in the staged runtime using `inspect_model(Path(model_path))["content_hash"]` and `content_hash(runtime_identity(Path(skyrl.__file__).resolve().parent.parent))` from the connector's internal `_preparation` and `_manifest` modules. These inspect local files/inventory without allocating a model; hashing reads weight bytes. This does not determine acceptable numerical bounds.

With those settings present, run from the checkout using the staged interpreter:

```sh
python -m pytest tests/test_skyrl_runtime.py -o addopts='-q --tb=short'
python -m pytest tests/test_skyrl_gpu.py -m skyrl_gpu -k packing -o addopts='-q --tb=short'
python -m pytest tests/test_skyrl_gpu.py -m skyrl_gpu -k dp1_dp2 -o addopts='-q --tb=short'
python -m pytest tests/test_skyrl_gpu.py -m skyrl_gpu -k weight_change -o addopts='-q --tb=short'
python -m pytest tests/test_skyrl_gpu.py -m skyrl_gpu -k 'periodic and not async' -o addopts='-q --tb=short'
python -m pytest tests/test_skyrl_gpu.py -m skyrl_gpu -k 'periodic and async' -o addopts='-q --tb=short'
```

Explicit opt-in fails on missing dependencies, identity mismatch or unsuitable hardware; it does not silently skip, download, or substitute a backend. Training subprocesses force offline model loading and `RAY_ADDRESS=local`, preventing Ray's automatic attachment to an existing cluster. They use the native launcher and loops with audit-only hooks, deterministic two-decision text/tool tasks, and a causal token-indicator scorer for token mode. They check raw IDs/logprobs/masks, independent credit, actual optimizer/evaluation calls, periodic checkpoint contents, unconsumed task IDs and fresh occurrence IDs after resume. [Ray address resolution](https://github.com/ray-project/ray/blob/ray-2.57.0/python/ray/_private/worker.py#L1558).

Layout cases compare independent eager single-row forwards/gradients with native packed attention, fixed-count/token-budget microbatches, and checkpoint recomputation; other-sequence perturbations check attention isolation. Synthetic loss-only probabilities exercise a TIS cap, not inference provenance. Only the first synchronous training batch asserts actual rollout/train probability parity at fixed weights; async staleness is not treated as a parity failure.

DP comparisons use native FSDP workers/dispatch, one fixed global batch with frozen behavior logprobs, identical initial parameters, and actual native backward, clipping and optimizer steps. Both custom estimators run count-based and packed/token-budget variants. The packed DP2 fixture must exercise balancing dummies. Every rank participates in full-parameter gradient reconstruction; comparisons cover named gradients and optimizer deltas, not just gradient norms. These are policy-component tests, with auxiliary losses disabled.

Cache probes use one native server and one native FSDP worker. Test-only server observations establish an incomplete, same-request token prefix after a confirmed KEEP pause, then record actual NCCL reception, reset results and resumed output without changing sampling or output delivery. Retained-cache and cleared-cache branches must preserve exact prefix/final wire token-logprob pairs. Only the cleared-cache suffix is compared with updated-model teacher forcing; retained KV is not assumed to represent a single behavior policy. The extra initial pause is a controlled test barrier, not evidence for arbitrary async scheduling or KV offload. Private vLLM observations must be rechecked when changing the pin.

Budget host RAM and disk explicitly: component probes retain full FP32 trainable-parameter evidence, roughly 16 bytes per parameter per DP pair and 8 bytes per cache case, plus traces and other tests' checkpoints. Multiple full model-sized CPU tensors are live during observation/comparison. Artifacts are retained, never silently deleted.

Artifacts retain identities, chosen thresholds, raw diagnostic traces, max/percentile logprob errors, relative gradient/update errors, logs and native checkpoints. Failed conversions retain the offending input, stage and exception. CPU fault injection checks token/reward transport, checkpoint/report failures and real deadlines; component-probe negative controls reject wrong scaling, changed prefixes/probabilities, ineffective resets and missing suffixes/dummies. A timeout targets only the newly created subprocess group. Installed execution of all GPU cases, detached Ray actors/external leases, safe general async resume, other execution profiles, remote judges, VLM and learning-quality acceptance remain separate gates.
