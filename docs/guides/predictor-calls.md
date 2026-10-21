# Persistent predictor calls

Use explicit sampling settings, stable backend identities and retained receipts
when model draws must survive a restart. Language-model calls return
`GenerationResult`; decision calls return an ordered probability distribution.
Both interfaces can use `CallStore` and `PersistentBackend`.

## Qualification

Small live probes cover Luna reasoning/schema requests, Decisions distributions,
prefix-cache usage and OpenAI Batch submission/collection. These do not establish
application-scale quotas, production cache savings or reconciled billing.

Qwen/vLLM and native d3 need GPU qualification for the exact model, runtime,
hardware and settings being deployed. Offline tests alone do not establish
model loading, reasoning and schema behavior, token/logprob alignment, memory
use, numerical correctness or process cleanup. Validate these in the actual
runtime. Container deployment also needs application termination and restart
checks against its persistent storage.

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

Tokenized template rendering explicitly requests a plain token-ID list; tokenizer
metadata dictionaries are never passed as the phase's input tokens.

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

`OpenAIBackend` and `OpenRouterBackend` accept independent connect `timeout` and
`read_timeout` values, in seconds. `read_timeout` must be finite and positive;
omitting it keeps the SDK default of 600 seconds. Both synchronous and async
clients use these settings. `OpenAIDecisionBackend` accepts them through the same
transport. They do not change model or call identity. A read timeout after
dispatch leaves an unknown outcome in a persistent store; a shorter timeout does
not make that draw safe to repeat. Use `max_retries=0` to disable SDK retries.

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

### GCS journals

Install `llenvs[gcs]` and pass a `GCSJournal` instead of a filesystem path:

```python
from llenvs.inference.persistence import CallStore
from llenvs.inference.persistence.storage.gcs import GCSJournal

journal = GCSJournal(
    bucket="experiment-artifacts",
    prefix="model-calls/pilot",
    cache_dir="./batch-file-cache",
    max_concurrency=128,
)
with CallStore.open(
    journal, run_descriptor={"experiment": "pilot", "shard": 0}, validation="lazy",
) as store:
    # Use PersistentBackend or OpenAIBatchBackend with this store.
    pass
```

The bucket must already exist. Authentication uses the Storage SDK's normal
mechanism; `client=...` accepts an already authorized client without constructing
another one. Grant object read, create, replace and delete access for the journal,
plus the required object-listing permission. Ordinary bucket IAM prefix
conditions do not restrict the scope of listing; choose the bucket/access scope
accordingly. See [GCS IAM conditions](https://docs.cloud.google.com/storage/docs/access-control/iam#conditions).
The extra is independent of the inference backend extras.

`max_concurrency` is a positive integer, defaulting to 128. It bounds concurrent
record downloads and uploads, including reads for cold replay and full startup
validation. It controls storage I/O independently of model-request concurrency;
changing it does not change run or call identities. Lower it for a smaller I/O
budget. Discovery, queued updates and the decoded-record cache are bounded;
active requests and submission payloads require additional memory.

Clients created by `GCSJournal` have connection pools at least as large as
`max_concurrency`. Supplied clients keep their transports by default. To enlarge
the pools of a supplied native Requests session, pass
`connection_pool_size=256` (or another integer at least `max_concurrency`). This
preserves its authentication, retry policy and pool blocking settings, and never
shrinks larger pools. Configure pools before concurrent use of that client.
Custom adapters and wrapped transports reject explicit automatic tuning; configure
those transports yourself and omit `connection_pool_size`.

Independent call updates share a rolling window of at most `max_concurrency`
admitted records. Each bounded cohort checks ownership before its uploads.
While those uploads run, later queued writes can join the checked cohort for
at most 100 ms after its first record is admitted. The cohort ends sooner on
its first native completion, its reserved record limit or a fatal failure.
The deadline does not extend when more writes arrive. Each admitted write starts
immediately; the coordinator does not wait to fill a cohort before uploading.
In a healthy store, each confirmed write is acknowledged and releases its window
slot without waiting for another write in its cohort. Ownership reads for later
cohorts do not block those acknowledgements. Queued updates and observer work are bounded
by twice `max_concurrency` (one for a single storage worker). Record transfers
also obey the shared I/O limit. A single storage worker and filesystem journals
keep writes serial.
Close stops waiting for new arrivals and drains queued and admitted writes.
Each caller waits for its own confirmed durable state. A started intent is
durable before model dispatch, an answer is durable before it is returned, and
updates of one key remain ordered. Batch retains its individual call records and
state transitions; submission and collection write those records concurrently.
Active calls retain their observed generations to avoid redundant metadata reads.
The absence read for a new call supplies its generation-zero creation precondition;
its first write needs no separate metadata read.
Existing journals use the same object layout and remain readable.

Call records, receipts, Batch journals and exact Batch file bytes are authoritative
GCS objects. The local directory only caches Batch files; losing it does not lose
call state. Without `cache_dir`, the open store uses a temporary directory and
removes it on close. Batch transfers stream through disk with bounded transfer
buffers. `blob_path()` materializes authoritative bytes and verifies them by
default. `store.run_dir` is `None` for GCS; it remains a filesystem path for local
stores. Use the store's methods rather than manipulating cloud objects directly.

Each run has one persistent owner. Ownership has no timeout: a paused worker must
not become eligible for replacement. Normal close conditionally releases exactly
that owner. If a worker dies or ownership release cannot be confirmed, first
confirm through the orchestrator that the old worker has terminated, then recover
the observed owner token explicitly:

```python
owner = journal.current_owner(run_descriptor={"experiment": "pilot", "shard": 0})
# Only after external confirmation that this owner's worker has terminated:
journal.acknowledge_owner_termination(
    run_descriptor={"experiment": "pilot", "shard": 0},
    expected_owner=owner,
    reason="Orchestrator confirmed termination of the previous worker",
    confirmed_terminated=True,
)
```

Recovery retains the assertion and reason before releasing ownership. It rejects
a stale token and does not reset call attempts or authorize resubmission of an
unknown outcome. The library does not query a scheduler. This contract requires
termination; a separate ownership object is not transactional fencing against a
still-running writer. Give concurrent shards
distinct stable run descriptors.

The owner check and record writes are separate operations. Already checked
records can finish if ownership changes during their uploads, bounded by the
`max_concurrency` window across all cohorts. Each subsequent cohort checks again
and refuses a different owner. This does not provide fencing for a paused
worker. Recovery still requires confirmed termination.

Writes use generation preconditions and bounded retries. Lost acknowledgements
are reconciled against the operation's exact bytes. A fatal storage, decoding or
serialization failure rejects new requests and stops admission of new writes.
Already accepted writes drain and publish confirmed outcomes before
unresolved outcome waiters fail. This also applies when the failure is in a
separate Batch journal, file transfer or lifecycle operation: a confirmed call
still returns with its original source. Intent acknowledgements are held during
the drain, preventing new model dispatch from those writes. The failing operation
can report its error before other admitted writes finish; close waits for them.
Batch collection errors retain confirmed progress in `partial_collection`.
Generation checks apply to
ownership, call state, journals, artifacts and deletions. Storage retries never
authorize another model draw. Keep the prefix private and trusted: journal pickles
are executable Python artifacts, and checksums detect accidental corruption.
External deletion or expiration of live journal objects invalidates replay
guarantees. `mark_complete(delete=True)` removes the run's objects after checking
for unresolved calls; ownership and recovery assertions remain separate.
Finalization and close wait for active journal operations and accepted native
writes, including when further admission fails, and run exclusively.
A handle whose run was deleted refuses subsequent calls; close it normally to
release ownership.
Close also releases active generation metadata and fails unresolved local waiters
with `CheckpointError`; it does not resolve an uncertain provider outcome.

Lazy startup does not read or list completed calls. It still discovers submission
journals and lifecycle records; full validation reads all retained records and
referenced Batch files. GCS request, storage and transfer costs are additional to
inference costs. Offline storage tests do not establish live-cloud performance or
platform restart behavior.

### Call identity and reuse

Persisted generation, decision and scoring results, and individual call errors,
carry `metadata["persistence"]`:

```python
{
    "run_fingerprint": "...",  # Store run identity.
    "call_key": "...",         # Logical draw, also used in the journal.
    "source": "store",
    "previous_attempts": 0,    # Attempt-history length at invocation entry.
}
```

| `source` | How this invocation obtained the result |
|---|---|
| `backend` | This invocation delegated the call to the backend |
| `store` | A completed result or failed/unknown outcome was already retained in the journal |
| `receipt` | A retained receipt was parsed locally |
| `shared` | This position joined an already reserved call, including a duplicate batch position |
| `batch` | This collection obtained an outcome from the retained Batch submission |

The identity stays the same across replay, regrouping and process restarts.
Provenance describes each returned position, so a mixed batch can contain new,
replayed and shared outcomes. Controlled reasoning/answer phases carry their own
keys and sources in each phase's `metadata`; a backend invocation can reuse a
previous phase. A completed parent replay preserves the original phase records
and marks the parent result as `store`. Batch sources describe collection, not
submission: collecting an earlier submission does not buy another model draw.
With `wait=True`, sources cover the entire collection call across all polls.

`previous_attempts` snapshots the call's attempt-history length when this
invocation reserves the key. Retries during the invocation do not change its
returned count; their diagnostics remain in the attempt records. A subsequent
invocation can therefore report a larger count. Reading a stored outcome or
reparsing a receipt does not add attempts. Batch collection snapshots history at
collection entry and retains it across polls. Controlled phases snapshot at
their own reservation boundary; a joined parent reporting an unknown child
retains that child's snapshot. Counts are local observations, not proof of
provider execution or charges. An aggregate controlled call has separate child
histories rather than a sum of phase attempts.

For a `PartialBatchError`, read each exception in `error.failures[index]` (also
present in the corresponding `error.results[index]` slot). Those exceptions
retain their native type, status and diagnostics on the initial invocation;
replayed errors retain the saved classification as described below. Each
position has detached provenance even when requests were coalesced. For
`OpenAIBatchBackend.collect()`, use `collection.failures[index]`.

An `UnknownOutcomeError` for an individual call also carries this record and
retains its recovery key in `error.key`. An unknown controlled phase identifies
the child call requiring recovery. A Batch submission error can affect several
calls: it carries their records in `error.metadata["persistence_calls"]`, while
`error.key` remains the submission ID used for Batch recovery. Local validation
and storage errors without an identified call have no call provenance.

Usage, raw receipts, sampled tokens and stored outcomes retain their original
values. The wrapper adds provenance to detached results and exceptions. An excluded
`persist` scope has no store provenance. `ScoringResult.metadata` is empty by
default and carries the same provenance when scoring is persisted.

Do not sum returned usage across repeated invocations as new spending. Use
`(run_fingerprint, call_key)` to identify repeated outcomes, and consult retained
attempt records for retries, failures and unknown outcomes. Phase costs require
the phase records rather than counting both their usage and the parent's total.
The source is not a billing assertion; provider prompt caching still represents
a new request and differs from a local replay.

The store takes an exclusive process lock. By default, opening it validates the
manifest, retained records, journals and file checksums before dispatch. For a
large store, opt into validation on access:

```python
with CallStore.open(
    "./model-calls", run_descriptor={"experiment": "pilot"}, validation="lazy",
) as store:
    ...
```

`validation="lazy"` checks the manifest, lifecycle events and submission journals
at open, then checks call payloads and Batch files when needed. It skips the
startup scan of call files. Requested records still undergo checksum, identity
and outcome validation before reuse or recovery. Detected corruption poisons
the store and prevents further dispatch. Corruption in unrelated, unread
payloads can remain undetected until access; close and reopen with
`validation="full"` (the default) for a complete integrity check. Lazy validation
does not reduce the number of call files or the size of Batch journals held in
memory. It does not change call keys or authorize resubmission of unknown work.

Pickles are trusted local artifacts: checksums detect corruption, not malicious
changes. The private store contains full model inputs and raw responses. Keep
it in an appropriate private location.

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
are checked against the verified worker identity. The response's `model` names
the served model, while `meta.model_id` names the card's repository, falling back
to its model ID when no repository is reported. Teardown signals only the owned
process group, with bounded SIGTERM/SIGKILL waits.

Loading and warming health responses are retried until the startup deadline;
terminal native `failed` health raises `RuntimeError` promptly. That error and
a startup `TimeoutError` expose `startup_health`: the last valid, credential-safe
`{"status_code": ..., "body": ...}` health snapshot, or `None` if no valid reply
was received. Their messages include the last native reason, when present, and
the worker log path. Transient connection failures or malformed nonready replies
do not replace an earlier snapshot. Health responses must use API major 3.

Ready workers with native golden status `matched` or `unverified` are accepted.
`unverified` can mean the runtime has no reference for that accelerator; the
original status and numerical contract remain in the descriptor's runtime card.
Failed, pending, missing or unknown golden status is rejected. Native dtype and
batching defaults are preserved, including `approximate` numerics when reported.
Readiness does not establish independent numerical parity or batch invariance.

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

All pending call records are durable before the journal enters `creating`,
immediately before the create call. An `intent`, including one with a saved upload
ID or partially updated call records, resumes through submission or collection
without a recovery acknowledgment; a saved upload ID is reused. A hard kill
between the durable `creating` marker and HTTP dispatch remains uncertain: those operations
cannot be atomic. Existing `creating` journals retain this conservative rule,
even when some call records are not pending. A stop observed before invoking
create, or a native client already known to be closed at that boundary, leaves
the submission at `intent`. Arbitrary connection errors do not establish that
creation was undispatched.

Saving a handle is optional when the requests can be reconstructed. Reopen the
same durable store and call `submit_chat_batch()` with the same requests,
parameters and explicit repeat IDs to rebuild a handle. Request order and file
windows may change: existing keys reuse their submissions or stored outcomes,
and only eligible new attempts are uploaded. Preserve the run descriptor and
store across restarts.

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

Collection normalizes and returns only the handle's keys, while retaining the
complete provider output/error files. Outcomes outside the handle are deferred
until a collection names those keys, even when the provider batch is already
terminal. Later subsets use the retained files without retrieving the provider
batch or downloading files again. The first collection returning a key's result
or failure reports `source="batch"`; later collections report `source="store"`.
This applies across overlapping handles, reopened stores and process ownership
changes, including refusals, expired requests and missing outputs. Local receipt
reparsing keeps `source="receipt"`. Attempt history and `previous_attempts` do not
change merely because requests are collected in different groups.

If collection raises after obtaining any outcomes, the same exception carries
`partial_collection`, a `BatchCollection` aligned with the handle. It includes
results and per-slot failures already committed or replayed, their usual sources
and attempt usage, the latest observed submission statuses, and uncollected keys
as pending. This includes progress from earlier `wait=True` polls and progress
before a retrieval, download or storage error. The exception's type and other
attributes are unchanged. There is no `partial_collection` when no outcome was
obtained, except for the cooperative stop described below.

```python
try:
    collection = batch.collect(handle, wait=True)
except Exception as error:
    partial = getattr(error, "partial_collection", None)
    if partial is not None:
        record_outcomes(partial)  # The same ledger path as a normal return.
    raise
else:
    record_outcomes(collection)
```

Carried outcomes count as returned: a later collection replays them with
`source="store"`, while still-uncollected outcomes retain their usual first
collection source. The collection-level exception does not turn pending keys
into slot failures or authorize resubmission. Successful returns keep the same
`BatchCollection` fields and alignment. Carried outcomes are detached from the
store, just like normal returns.

Outcome persistence and the caller's ledger are separate writes. A process can
stop without a caught exception after committing an outcome, or before the caller
records a returned or carried outcome; a later
collection then replays it from the store. Reconcile ledger entries by run,
call key and attempt rather than assuming a replay proves the ledger already
contains the outcome. A `batch` source identifies collection, not a charge:
expired unexecuted requests were not run, and missing-output usage is unknown.

### Cooperative stopping

`submit_chat_batch(..., stop=event)` and `collect(..., stop=event)` accept a
caller-owned `threading.Event`. With no event, ordinary behavior is unchanged.
After a check observes the event, the operation starts no more provider calls.
Checks occur during preparation, before upload/create/list/retrieve/download,
between submissions and between collection entries. Polling uses an
interruptible wait. Already admitted storage work drains before the operation
raises `BatchStopped(RuntimeError)`, exported from `llenvs.inference.backends`
and its `openai_batch` module. A completed upload or download is retained before
stopping. Already submitted remote batches are not cancelled.

`BatchStopped` exposes:

- `handle`: the recoverable `BatchHandle`, or `None` when submission stopped
  before retaining recoverable work. A stopped submission can cover only a
  subset of its inputs; use the original requests and repeat IDs to reconstruct
  the full handle on restart. A stopped collection keeps its original handle.
- `submission_states`: a snapshot mapping retained submission UUIDs to journal
  states such as `intent`, `creating`, `pending` or `terminal`. These are local
  journal states, rather than provider statuses or proof of remote execution.
- `partial_collection`: for collection, an aligned `BatchCollection`, including
  an empty one when nothing was obtained. For submission, this is `None`.
  Collected results and failures keep their usual sources, usage and attempts;
  uncollected keys remain pending.

```python
from threading import Event
from llenvs.inference.backends import BatchStopped

stop = Event()  # The application sets this from its shutdown path.
try:
    handle = batch.submit_chat_batch(messages, params, repeat_ids=repeat_ids, stop=stop)
    collection = batch.collect(handle, wait=True, stop=stop)
except BatchStopped as error:
    if error.partial_collection is not None:
        record_outcomes(error.partial_collection)
    if error.handle is not None:
        save_handle(error.handle.to_dict())
    # Close the store only after the stopped operation has finished.
else:
    record_outcomes(collection)
```

A stop does not create slot failures, authorize resubmission or poison the store.
Resume with the same durable store and requests/repeat IDs, or collect the saved
handle. Retained collection files are reused, and committed outcomes are replayed
without persisting them again. Clear or replace the event only after the stopped
operation has finished.

Stopping is cooperative, with no absolute shutdown deadline. An upload, download,
create or storage operation already in progress can finish or raise its original
error. The event cannot interrupt a blocked synchronous network call; configure
SDK/storage timeouts and the application's drain allowance accordingly. Closing
the SDK client can cause an in-flight call to fail, but does not prove that a
create request was never sent. Store ownership is released only after admitted
writes have drained. Qualify the complete shutdown path against the deployment's
network latency and termination deadline.

### Failed slots and resubmission

Terminal expiry/cancellation/failure collects the handle's available successes
before marking its unfinished slots failed. Unrequested keys still need
collection before the store can be marked complete. A later
`submit_chat_batch()` replays stored failures with `source="store"`, including
refusals, malformed responses, missing
logprobs and HTTP/per-request errors. A generated response may already be billed
even when parsing fails. Replaying a failure preserves its receipt and attempt
history without buying another draw.

For failures with a retained Batch entry, the exception exposes `status_code`
(including 200 for response-parsing failures), the optional provider error
`code`, and a detached `attempts` list with request/response IDs, pricing tier
and reported usage. Codes are taken from the per-request error or the response
body's error. These fields survive saved-error replay, including typed refusals
and malformed-response errors. `transient` follows the standard classification:
rate-limit 429 and server statuses of 500 or above are transient; quota exhaustion
is not. A missing output raises `BatchUnfinished(RuntimeError)` on first
collection, exported from `llenvs.inference.backends` and its `openai_batch`
module. Its saved replay remains `ReplayedBackendError` with
`original_error_type="BatchUnfinished"`. It has no HTTP status or known attempt
usage and is not transient. Missing usage is unknown, not zero. Classification
does not authorize dispatch; the resubmission rules below still apply.

The only automatically eligible failed slot is a per-request `batch_expired`
error with no response in a terminal submission: the provider explicitly reports
that this request was not executed. A later explicit `submit_chat_batch()` sends
that key under a new submission UUID, preserving the previous attempt. A batch's
expired/cancelled/failed status, a missing output row or an arbitrary non-200
status does not establish that an individual request is safe to send again.
These failures continue to replay until the caller explicitly authorizes another
attempt:

```python
store.acknowledge_resubmission(
    call_key,
    reason="Another draw is intended for this failed request.",
    accept_duplicate_expenditure=True,
)
# Only a subsequent submit containing this key dispatches the authorized attempt.
```

This per-call decision is separate from the Batch-level authorization for an
unknown creation outcome. `collect()` never creates replacement attempts. A local
parser interruption keeps the original receipt for reparsing without provider
access. Missing/expired provider output files do not trigger regeneration. The
convenience `generate_chat_batch(..., repeat_ids=...)` submits, waits and uses
`PartialBatchError` for failed slots.

Result/attempt metadata records Batch ID, endpoint and pricing tier. Raw provider
Batch usage summaries are retained for reconciliation and are not added again to
per-request totals. Missing usage remains unknown expenditure. Pricing tables and
cache-hit assumptions stay outside the backend. Decisions has no Batch backend.
