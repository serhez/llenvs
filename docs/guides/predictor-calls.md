# Persistent predictor calls

Use explicit sampling settings, stable backend identities and retained receipts
when model draws must survive a restart. Language-model calls return
`GenerationResult`; decision calls return an ordered probability distribution.
Both interfaces can use `CallStore` and `PersistentBackend`.

## Qualification

Small live probes cover Luna reasoning/schema requests, Decisions distributions,
prefix-cache usage and OpenAI Batch submission/collection. These do not establish
application-scale quotas, production cache savings or reconciled billing.

Qwen/vLLM and native d3 have offline tests; their target GPU runtimes remain
unqualified. Before using them for experiments, validate model loading, reasoning
and schema behavior, token/logprob alignment, memory use, numerical correctness
and process cleanup in the actual runtime. Container deployment also needs
application termination and restart checks against its persistent storage.

## Reasoning and answer format

`reasoning_effort=None` inherits the model or template default. An explicit
`"none"` disables reasoning where the configured backend supports it. Unsupported
efforts raise before dispatch.

For Luna Chat Completions:

```python
from llenvs.inference import ChatMessage, ResponseSchema, SamplingParams
from llenvs.inference.backends import OpenAIBackend

schema = ResponseSchema({
    "type": "object",
    "properties": {"percentage": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["percentage"],
    "additionalProperties": False,
}, name="prediction")

params = SamplingParams(
    max_tokens=4096,
    reasoning_effort="medium",
    temperature=None,
    top_p=None,
    response_schema=schema,
)
with OpenAIBackend(model="gpt-6-luna", max_retries=0) as backend:
    result = backend.generate_chat([ChatMessage("user", "Estimate the percentage.")], params)
```

The qualified Luna Chat Completions endpoint accepts `none`, `low`, `medium`,
`high` and `xhigh`; it rejects `max` as unsupported. Its inherited
reasoning default is `medium`: omitting effort still requires
`temperature=None`, `top_p=None` and `logprobs=False`. `none` permits explicit
decoding controls. The backend uses `max_completion_tokens`, covering reasoning
and answer output. A schema does not eliminate refusals, truncation or the need
to validate the returned distribution.

For local Qwen3.8 templates, configure
`reasoning_family="qwen3.8"`. The accepted explicit efforts are
`none`, `low`, `medium` and `xhigh`. Per-request effort overrides backend template
defaults. Qwen3.8 retains historical reasoning by default. Raw prompts passed to
in-process `generate()` cannot apply a chat-template effort; use `generate_chat()`.
Chat, prefix and continuation-scoring paths translate `ChatMessage.reasoning`
to the template's native `reasoning_content` field. Opaque `reasoning_details`
cannot be rendered by this template and raise before dispatch.

`response_schema` uses native OpenAI/vLLM structured outputs. In-process vLLM's
configured Qwen path selects xgrammar, the qwen3 reasoning parser and
`enable_in_reasoning=False`. The owned vLLM server uses the corresponding engine
settings. Hugging Face rejects schema requests.

### Text before the reasoning closure

Configured local families also accept `after_reasoning_text`. Supply text in
the model's voice without `<think>` or `</think>` delimiters:

```python
from dataclasses import replace

local_params = replace(
    params,
    temperature=0.6,
    top_p=0.95,
    reasoning_effort="xhigh",
    after_reasoning_text="I will now return the requested JSON object.",
)
```

The backend renders the initial prompt once, samples free reasoning until its
verified closing token or explicit thinking budget, then supplies:

```text
initial prompt IDs + sampled reasoning body IDs + supplied text IDs
                  + supplied closing marker IDs + answer separator IDs
```

The answer is a second generation, optionally constrained by the schema. The
sampled phase-one closing token and its original logprob remain in the record,
although the supplied closing marker appears at a different position in phase
two. Supplied text has no sampled-token logprobs. With reasoning disabled, the
backend inserts before the template's empty-block closing marker and skips the
reasoning generation.

`max_tokens` limits total sampled output across both phases, including a sampled
closing marker. Supplied tokens consume context and phase-two input usage, but
not the sampled-output budget. Exhausting the total budget yields a truncated
empty answer. EOS or a caller stop before the boundary does not authorize an
answer phase. Insertion requires an explicit output budget and known context
limit; it rejects images, per-block budgets and soft-budget controls.

`metadata["phases"]` retains each phase's IDs, logprobs, usage, finish information
and effective request. `phase_two_prompt_ids` and `context_transformation` record
the actual second context and supplied spans. These calls are not append-only
sampled trajectories. Training consumers must use the separate contexts and
token provenance.

## Records and usage

Every result has a detached, sanitized `metadata["request"]`, a `requests` list,
`requested_sampling`, and inherited/omitted control names. API request records
describe the serialized body after `extra_body` merging; local records describe
native generation and template options. LiteLLM records are labeled
`request_capture="litellm_input"` because its SDK can transform them. Public
request records exclude model inputs, credentials, callbacks and clients.

`GenerationResult.text` contains the answer. Native reasoning, raw generated IDs
and original logprobs remain in metadata. Public `token_logprobs` are sliced to
the answer only when its exact native span is known. Closed APIs can supply
token strings and probabilities without IDs; those IDs are `None`. An absent
reasoning trace is not reconstructed. An unclosed local reasoning block has an
empty answer and explicit truncation metadata.

The nullable usage fields are:

| Field | Meaning |
|---|---|
| `prompt_tokens` | Total input tokens |
| `completion_tokens` | Total generated output tokens |
| `cached_prompt_tokens` | Cache-read subset of input |
| `cache_write_prompt_tokens` | Cache-write subset of input |
| `reasoning_tokens` | Reasoning subset of output |

Missing counts stay `None`; reported zero stays zero. `total_tokens` and phase
sums propagate unknowns. Reasoning and cache subsets are not added to totals
again. Anthropic's ordinary input excludes cache categories, so its normalized
input sums ordinary input, cache reads and cache writes. Raw usage, including
provider-specific write-duration details, is retained.

Evaluation transitions retain these usage fields, finish reason and detached
result metadata in `transition.info["generation"]`. With second elicitation,
`metadata["elicitation_results"]` retains both generations and their original
logprobs; `requests` includes both requests. Available first-pass reasoning is
carried into the elicitation history separately from answer text.

Provider-reported reasoning counts take precedence. Exact counts from verified
local reasoning spans are labeled `reasoning_tokens_source="token_ids"`.
Per-phase input includes reasoning reprocessed in phase two. Attempts retain
their own usage and elapsed/backoff time; successful-result totals do not also
add failed-attempt expenditure. Prices belong to the caller.

### Backend identity and lifecycle

`backend.descriptor()` returns JSON-compatible `identity` and `provenance`.
Replay hashes only `identity`: model/revision or local content manifest,
tokenizer/processor/template digests, runtime content, dependency versions,
precision, hardware and relevant engine/batching/cache settings. An owned
server's random port and installation paths are provenance. Changing runtime,
precision or template changes identity.

Owned vLLM resolves a Hub revision before launch and uses it for server and host
tokenizer loads. Offline resolution uses the cached snapshot revision. Its
descriptor reads the reported engine configuration, including effective prefix
caching; it does not infer settings from library defaults. Container identity
uses the actual SIF content digest. Native container digests are explicitly null.

Hosted aliases have `version_pinned=False`. A retained response is replayable,
but does not establish immutable hosted weights. Local model descriptors require
an immutable revision or a content manifest. Runtime content identities require
non-editable wheel or source builds; editable
loader metadata is not accepted as a digest of the executed code.
Instance startup/provenance and observed shutdown are separate lifecycle records. A hard kill can leave shutdown
time unknown; GPU allocation accounting belongs to the caller.

## Retaining individual draws

```python
from llenvs.inference import CallStore, PersistentBackend

with CallStore.open("./model-calls", run_descriptor={"experiment": "pilot"}) as store:
    with PersistentBackend(
        OpenAIBackend(model="gpt-6-luna", max_retries=0), store,
    ) as backend:
        results = backend.generate_chat_batch(
            [[ChatMessage("user", "Question A")], [ChatMessage("user", "Question B")]],
            params,
            repeat_ids=["unit-a:replicate-0", "unit-b:replicate-0"],
        )
    store.mark_complete()  # Retains records, receipts and lifecycle events.
```

The logical key includes backend identity, operation, exact inputs and settings,
selection scope and repeat ID. Batch position, siblings and batch size are
excluded. Reusing a request and explicit repeat ID retrieves the same draw across
regrouping, subset retries and process restarts. Duplicate explicit positions are
coalesced and restored to their original positions. Distinct repeat IDs identify
independent draws.

Single methods use `repeat_id=...`; batch methods use aligned `repeat_ids=...`.
Without explicit IDs, occurrence counters distinguish identical calls, and replay
requires the original invocation order. Use explicit IDs when regrouping work.
`scope=...` separates phases/selections; `persist=lambda scope: ...` can exclude
selected scopes from storage.

### Result identity and reuse

Persisted generation, decision and scoring results carry
`result.metadata["persistence"]`:

```python
{
    "run_fingerprint": "...",  # Store run identity.
    "call_key": "...",         # Logical draw, also used in the journal.
    "source": "store",
}
```

| `source` | How this invocation obtained the result |
|---|---|
| `backend` | This invocation delegated the call to the backend |
| `store` | A completed outcome was already retained locally |
| `receipt` | A retained receipt was parsed locally |
| `shared` | This position joined an already reserved call, including a duplicate batch position |
| `batch` | This collection obtained an outcome from the retained Batch submission |

The identity stays the same across replay, regrouping and process restarts.
Provenance describes each returned position, so a mixed batch can contain new,
replayed and shared results. Controlled reasoning/answer phases carry their own
keys and sources in each phase's `metadata`; a backend invocation can reuse a
previous phase. A completed parent replay preserves the original phase records
and marks the parent result as `store`. Batch sources describe collection, not
submission: collecting an earlier submission does not buy another model draw.
With `wait=True`, sources cover the entire collection call across all polls.

Usage, raw receipts, sampled tokens and stored outcomes retain their original
values. The wrapper adds provenance to detached return values. An excluded
`persist` scope has no store provenance. `ScoringResult.metadata` is empty by
default and carries the same provenance when scoring is persisted.

Do not sum returned usage across repeated invocations as new spending. Use
`(run_fingerprint, call_key)` to identify repeated outcomes, and consult retained
attempt records for retries, failures and unknown outcomes. Phase costs require
the phase records rather than counting both their usage and the parent's total.
The source is not a billing assertion; provider prompt caching still represents
a new request and differs from a local replay.

The store takes an exclusive process lock and validates the manifest, retained
records, journals and file checksums before dispatch. Pickles are trusted local
artifacts: checksums detect corruption, not malicious changes. The private store
contains full model inputs and raw responses. Keep it in an appropriate private
location.

Nonsemantic callbacks and transport clients are excluded from private request
identity. Custom policy callables, such as supplied logits processors, cannot
establish stable identity and are rejected by persistence before dispatch.

### Completion and recovery boundaries

The default `checkpoint_granularity="request"` requires individual completion
events. API and owned-server requests persist intent immediately before
dispatch, then a detached receipt before parsing, and a normalized result before
releasing the slot. Writes use a bounded worker outside the HTTP event loop.
Storage failures are fatal and poison the store.

In-process vLLM requires `request_progress=True` for native AsyncLLM completion
events. It retains native batching and a synchronous public interface. That mode
currently supports text chat/completions; prefix continuation and continuation
scoring require `request_progress=False`.

Hugging Face and synchronous native vLLM do not expose individual completion of
ordinary batches. Opt into `checkpoint_granularity="batch"` for the weaker
guarantee that only returned batches are durable. Controlled reasoning/answer
phases use stable child keys, including in this explicit batch mode.

A completed result replays. A durable receipt is reparsed locally. A definite
failure replays its saved error classification until an explicit recovery
decision; a legitimate model answer that downstream code cannot parse remains a
completed paid outcome. An attempt started without a durable receipt raises
`UnknownOutcomeError` and is not automatically resubmitted.

After investigating an uncertain outcome, a caller can explicitly accept possible
duplicate expenditure:

```python
store.acknowledge_resubmission(
    key,
    reason="Provider investigation could not recover the response.",
    accept_duplicate_expenditure=True,
)
```

The decision and prior diagnostics remain recorded. This is not exactly-once
remote execution: a process can die after remote execution and before durable
receipt. SDK retries must be disabled (`max_retries=0`, or LiteLLM
`num_retries=0`); the backend owns bounded, recorded rate-limit retries. Ambiguous
network failures are not silently retried. Quota exhaustion is not a transient
rate limit.

`mark_complete()` retains everything. Explicit `mark_complete(delete=True)` is
allowed only when requests are completed or definitely failed; unresolved
submissions and unknown outcomes prevent deletion. Close a persistent backend
before its store to retain its observed shutdown event.

## Decision backends

Decision calls return all requested option probabilities in request order.
Missing, extra or duplicate labels, negative/nonfinite values and zero total mass
raise. Valid non-unit distributions are normalized, retaining original values,
sum and deviation. Confidence is not substituted for the distribution.

```python
from llenvs.inference.decisions import DecisionOption, DecisionRequest
from llenvs.inference.decisions.openai import OpenAIDecisionBackend

question = DecisionRequest(
    context="Dataset summary and held-out rows...",
    question="Which range contains the hidden value?",
    options=(DecisionOption("low", "Below 10"), DecisionOption("high", "At least 10")),
)
with OpenAIDecisionBackend(model="gpt-6-luna", max_retries=0) as backend:
    result = backend.decide(question)
    print(dict(zip(result.labels, result.probabilities)))
```

Install the `decisions` extra in a separate environment from LiteLLM and `all`;
see [installation](../getting-started/install.md#decisions-environment). It uses
OpenAI SDK >=3.26 and one named choice question at `/v1/decisions`. It supports
2..255 unique labels. Hosted version pinning and deterministic/batch-invariant
answers are not claimed. `decide_batch()` runs independent requests concurrently;
it does not use OpenAI's Batch service. Wrap it with `PersistentBackend` and use
`decide(..., repeat_id=...)` or `decide_batch(..., repeat_ids=...)` for persistence.

### Native d3 worker

Install the `d3` client extra. In a dedicated worker environment, install
GPU-compatible PyTorch and the pinned native runtime:

```bash
uv pip install --python /path/to/d3-worker/bin/python \
  'vllm-srun @ git+https://github.com/vllm-project/semantic-router.git@72ce0e55f47a5f5b632719a28232308c57c78bc5#subdirectory=src/model-runtime'
```

```python
from llenvs.inference.decisions.d3 import D3Backend

with D3Backend(
    executable="/path/to/d3-worker/bin/vllm-srun",
    revision="156a6d9330277b7e1a6571a1d881b516e7e1ea15",
    device="cuda:0",
    log_path="./d3-worker.log",
) as backend:
    result = backend.decide(question)
```

This starts an owned loopback-only `vllm-srun serve` process without a container
frontend. It checks runtime source identity, health/API version, loaded revision,
choice limits, effective engine arguments, numerical contract and native golden
status. Requests use the native `/v1/systemone` mapping and request runtime
metadata. The served model ID can differ from its Hub repository name; responses
are checked against the verified worker identity. Teardown signals only the owned
process group, with bounded SIGTERM/SIGKILL waits.

The native decision3 family loads verified model artifacts through its own code.
The model package's Python reference server is a separate parity-check path,
not a fallback backend. Pin the resolved worker dependencies and qualify the
actual accelerator, memory use, correctness and batching behavior before an
experiment. Deterministic decoding does not imply invariance to batch composition.

## OpenAI Batch submission and collection

```python
import json
from llenvs.inference.backends.openai_batch import BatchHandle, OpenAIBatchBackend

with CallStore.open("./model-calls", run_descriptor={"experiment": "tree"}) as store:
    with OpenAIBackend(model="gpt-6-luna", max_retries=0) as backend:
        batch = OpenAIBatchBackend(backend, store)
        handle = batch.submit_chat_batch(
            [[ChatMessage("user", "Question A")]], params,
            repeat_ids=["unit-a:replicate-0"],
        )
        saved_handle = json.dumps(handle.to_dict())

# Reopen the same store under exclusive ownership in another process.
with CallStore.open("./model-calls", run_descriptor={"experiment": "tree"}) as store:
    with OpenAIBackend(model="gpt-6-luna", max_retries=0) as backend:
        batch = OpenAIBatchBackend(backend, store)
        collection = batch.collect(BatchHandle.from_dict(json.loads(saved_handle)), wait=False)
```

The Batch backend uses the same Chat Completions builder/parser. It splits exact
UTF-8 JSONL bytes at 50,000 requests and 200,000,000 bytes, with one endpoint/model
per file. Per-request transport options are unsupported. Account-specific queued
token limits remain the caller's responsibility.

Before upload, the store retains exact file bytes/digest, logical keys and a
submission UUID. It saves the upload ID before creation, includes the run/UUID/
digest in provider metadata, and saves the returned Batch ID immediately. Pending
or unknown submissions are not enqueued again.

When a creation acknowledgment is lost, collection paginates provider listings
and matches metadata, endpoint and input-file ID. One match is adopted; zero
matches remain unknown, and multiple matches require reconciliation. An absent
listing match does not authorize another creation.

After investigating an unknown submission, an explicit
`batch.acknowledge_resubmission(submission_id, reason=...,
accept_duplicate_expenditure=True)` records a recovery decision. This alone does
not create a Batch. A subsequent explicit submission makes a new attempt under
the same logical keys, preserving the unresolved attempt's diagnostics and
unknown expenditure.

`collect()` reports aligned results, failure indices, pending keys and provider
statuses. `wait=True` polls existing submissions and never creates replacements
for unknown or still-pending work. A provably undispatched retained intent is
eligible for its original submission. Output/error files are retained before
parsing; entries are matched by logical `custom_id`, rejecting unknown/duplicate
IDs. Durable receipts and files can be recovered locally after parser/process
failure. Completed draws replay without provider access.

Terminal expiry/cancellation/failure collects every available success before
marking unfinished slots failed. A later explicit `submit_chat_batch()` with the
same keys submits only definitely failed/unfinished slots under a new submission
UUID. A local parse failure keeps the original paid receipt. Missing/expired
provider output files do not trigger regeneration. The convenience
`generate_chat_batch(..., repeat_ids=...)` submits, waits and uses
`PartialBatchError` for failed slots.

Result/attempt metadata records Batch ID, endpoint and pricing tier. Raw provider
Batch usage summaries are retained for reconciliation and are not added again to
per-request totals. Missing usage remains unknown expenditure. Pricing tables and
cache-hit assumptions stay outside the backend. Decisions has no Batch backend.
