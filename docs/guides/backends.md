# Backends

`llenvs` supports multiple inference backends with a unified interface.

## Lifecycle

Backends are closeable and may be used as context managers:

```python
from llenvs.inference.backends import OpenAIBackend

with OpenAIBackend(model="gpt-4o") as backend:
    result = backend.generate_chat(...)
```

Callers own backend lifecycle. Local backends release heavyweight model
resources best-effort on `close()`, and API backends close reusable client
sessions. Repeated `close()` calls are safe.

## Available Backends

| Backend | Package | Features |
|---------|---------|----------|
| vLLM | `vllm` | Logprobs, batching, prefix continuation, streaming, vision (auto-detected VLMs) |
| HuggingFace | `transformers` | Logprobs, batching, prefix continuation, chat templates, vision (auto-detected VLMs) |
| OpenAI | `openai` | Logprobs, batching (concurrent), streaming, function calling, vision |
| Anthropic | `anthropic` | Batching (concurrent), prefix continuation (prefill), streaming, vision |
| OpenRouter | `openai` | Batching (concurrent), access to multiple models, vision |
| LiteLLM | `litellm` | Batching (concurrent), 100+ providers via one SDK, function calling, vision, logprobs (provider-dependent), thinking budgets |
| Codex CLI | `codex` | Chat, batching (subprocess concurrency), isolated temp workspace, read-only sandbox |
| Claude Code CLI | `claude` (Claude Code v2.1+) | Chat, batching (subprocess concurrency), isolated temp workspace, tools disabled, reasoning effort |

## vLLM (Local Inference)

vLLM runs in-process — no external server needed. Just point it at a model and a GPU.

### YAML Config

```yaml
model:
  backend: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  params:
    tensor_parallel_size: 2
    dtype: bfloat16
    max_model_len: 4096
    gpu_memory_utilization: 0.9
```

### Python API

```python
from llenvs.inference.backends import VLLMBackend

backend = VLLMBackend(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# Batch generation
results = backend.generate(
    ["Question 1?", "Question 2?", "Question 3?"],
    SamplingParams(temperature=0.0),
)

# With logprobs
results = backend.generate_with_logprobs(
    ["What is 2+2?"],
    SamplingParams(temperature=0.0),
    num_logprobs=5,
)
for token_lp in results[0].token_logprobs:
    print(f"{token_lp.token}: {token_lp.logprob:.3f}")
```

### Capabilities

| Feature | Supported | Notes |
|---------|-----------|-------|
| logprobs | ✓ | Native support |
| prefix_continuation | ✓ | Native support |
| batching | ✓ | Single batched GPU call |
| streaming | ✓ | Via async iterator |
| chat | ✓ | Via `tokenizer.apply_chat_template()` |
| function_calling | ✗ | Not supported |
| vision | Auto | VLMs detected automatically |

> **Tip:** If you already have a remote `vllm serve` instance running, use `OpenAIBackend(model="your-model", base_url="http://host:port/v1")` instead — vLLM's server exposes an OpenAI-compatible API.

## HuggingFace Transformers (Local Inference)

A lightweight alternative to vLLM for running HuggingFace models directly with the `transformers` library.

```python
from llenvs.inference import SamplingParams
from llenvs.inference.backends import HuggingFaceBackend

backend = HuggingFaceBackend(
    model_path="gpt2",  # or any HuggingFace model
    device="auto",  # "cuda", "mps", "cpu", or "auto"
    dtype="auto",  # "float16", "bfloat16", "float32", or "auto"
)

# Basic generation
results = backend.generate(
    ["Hello, world!"],
    SamplingParams(max_tokens=50, temperature=0.7),
)
print(results[0].text)

# With logprobs
results = backend.generate_with_logprobs(
    ["The capital of France is"],
    SamplingParams(temperature=0.0),
    num_logprobs=5,
)
for token_lp in results[0].token_logprobs:
    print(f"{token_lp.token}: {token_lp.logprob:.3f}")
```

### Advanced Options

```python
# Multi-GPU with device_map
backend = HuggingFaceBackend(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",  # Automatically distribute across GPUs
    dtype="bfloat16",
)

# With torch.compile optimization
backend = HuggingFaceBackend(
    model_path="gpt2",
    torch_compile=True,  # Enable torch.compile for faster inference
)

# Pass additional kwargs to from_pretrained
backend = HuggingFaceBackend(
    model_path="gpt2",
    trust_remote_code=True,
    revision="main",
)
```

### Chat Generation

```python
from llenvs.inference import ChatMessage

# Uses tokenizer.apply_chat_template() internally
result = backend.generate_chat(
    [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello!"),
    ],
    SamplingParams(temperature=0.7),
)
```

### Prefix Continuation

```python
# Generate multiple continuations from a prefix
continuations = backend.continue_from_prefix(
    prefix="Once upon a time",
    params=SamplingParams(temperature=0.8, max_tokens=100),
    num_continuations=3,
)
for i, cont in enumerate(continuations):
    print(f"Continuation {i + 1}: {cont.text}")
```

### Capabilities

| Feature | Supported | Notes |
|---------|-----------|-------|
| logprobs | ✓ | Via `output_scores=True` |
| prefix_continuation | ✓ | Native support |
| batching | ✓ | With attention masks |
| streaming | ✗ | Would require TextIteratorStreamer |
| chat | ✓ | Via `tokenizer.apply_chat_template()` |
| function_calling | ✗ | Not directly supported |
| vision | Auto | VLMs detected and loaded with `AutoProcessor` + `AutoModelForVision2Seq` |

## OpenAI

```python
from llenvs.inference.backends import OpenAIBackend

# Uses OPENAI_API_KEY env var by default
backend = OpenAIBackend(model="gpt-4o")

# Or explicit key with concurrency control
backend = OpenAIBackend(
    model="gpt-4o",
    api_key="sk-...",
    organization="org-...",
    max_concurrency=32,  # Max concurrent API requests (default: 64)
)

# Chat generation
from llenvs.inference import ChatMessage

result = backend.generate_chat(
    [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello!"),
    ],
    SamplingParams(temperature=0.7),
)
```

## Anthropic

```python
from llenvs.inference.backends import AnthropicBackend

backend = AnthropicBackend(
    model="claude-sonnet-4-20250514",
    max_concurrency=32,  # Max concurrent API requests (default: 64)
)

# Supports prefix continuation via assistant prefill
continuations = backend.continue_from_prefix(
    prefix="Let me solve this step by step:\n1.",
    params=SamplingParams(temperature=0.5),
    num_continuations=3,
)
```

## OpenRouter

```python
from llenvs.inference.backends import OpenRouterBackend

backend = OpenRouterBackend(
    model="anthropic/claude-sonnet-4-20250514",
    site_url="https://mysite.com",
    app_name="MyApp",
    max_concurrency=32,  # Max concurrent API requests (default: 64)
)
```

### Reasoning / Thinking Budget

`SamplingParams.thinking_budget` is forwarded as
`extra_body.reasoning.max_tokens` on the chat-completion request, which
OpenRouter routes to the upstream provider's native budget knob
(Anthropic `thinking.budget_tokens`, Gemini thinking budget, Qwen
`thinking_budget`); for effort-only models (OpenAI o-series, Grok)
OpenRouter derives an effort bucket from the token count. Non-reasoning
models silently ignore the field. The other thinking-budget knobs
(`thinking_budget_per_block`, `thinking_budget_suffix`,
`thinking_budget_soft_ratio`) require token-level logits intervention
and are inert on this backend — they only take effect for `vllm` and
`huggingface`.

To override the derived `reasoning` block (e.g. send `effort` instead
of `max_tokens`, or add `exclude: true`), pass
`extra={"extra_body": {"reasoning": {...}}}` on `SamplingParams`. The
caller-supplied dict shallow-merges over the derived one and wins on
key collisions, so it is also the escape hatch for
`reasoning.effort`/`reasoning.exclude`/`reasoning.enabled`.

OpenRouter returns hidden reasoning separately from visible assistant
content. `GenerationResult.text` contains only `message.content`, while
diagnostics are preserved in `GenerationResult.metadata`: normalized
`finish_reason`, raw `native_finish_reason` when present,
`completion_tokens_details`, top-level `reasoning_tokens`, and lightweight
presence/count/length fields for `message.reasoning` and
`message.reasoning_details`. The full reasoning payload is not copied into
metadata.

## LiteLLM

Routes requests through the [litellm](https://docs.litellm.ai) SDK,
giving unified access to 100+ providers through one interface. Models
use litellm's `provider/model` format:

```python
from llenvs.inference.backends import LiteLLMBackend

# Direct provider access — API key read from the provider's native
# env var (GEMINI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, ...)
backend = LiteLLMBackend(model="gemini/gemini-2.5-flash")

# Through a LiteLLM proxy/gateway with a virtual key
backend = LiteLLMBackend(
    model="litellm_proxy/Qwen/Qwen3.6-35B-A3B",
    api_base="https://your-gateway.example.com",
    api_key="sk-your-virtual-key",
)
```

For the `litellm_proxy/` prefix, litellm also reads
`LITELLM_PROXY_API_KEY` and `LITELLM_PROXY_API_BASE` natively, so both
constructor params can be omitted.

YAML config:

```yaml
model:
  backend: litellm
  model: litellm_proxy/Qwen/Qwen3.6-35B-A3B
  max_concurrency: 32
  params:
    api_base: https://your-gateway.example.com
    rate_limit_wait: 30.0
```

### Parameter Dropping

`drop_params=True` (the default) makes litellm silently drop request
parameters the target provider does not support (e.g. `presence_penalty`
on Anthropic) instead of erroring. Set `drop_params=False` on the
constructor to fail loudly instead. Logprobs keep a hard guarantee
either way: if `logprobs=True` was requested and the provider returns
none, the backend raises `LogprobsNotReturnedError`.

### Reasoning / Thinking Budget

`SamplingParams.thinking_budget` maps to litellm's cross-provider
`thinking={"type": "enabled", "budget_tokens": N}` parameter (Anthropic
`thinking.budget_tokens`, Gemini thinking budget, Bedrock/Vertex
equivalents); providers without a budget concept drop it via
`drop_params`. `disable_thinking=True` sends `reasoning_effort="none"`
and wins over any other reasoning knob. For effort-style control, pass
`extra={"reasoning_effort": "high"}` on `SamplingParams` — `params.extra`
keys flow as top-level kwargs into `litellm.completion()` and override
derived values. The logits-level thinking knobs
(`thinking_budget_per_block`, `thinking_budget_suffix`,
`thinking_budget_soft_ratio`) are inert on this backend — they only take
effect for `vllm` and `huggingface`.

Reasoning diagnostics land in `GenerationResult.metadata`:
`completion_tokens_details`, `reasoning_tokens`, and presence/length
fields for `reasoning_content`. When litellm computes a request cost it
is surfaced as `metadata["response_cost"]`.

### Rate-Limit Retries

Same knobs as OpenRouter: `rate_limit_wait` (seconds to wait before
retrying a rate-limited call; `0` disables backend-level retries) and
`rate_limit_max_retries`. The retry path also covers providers that
surface their own 429 as an HTTP-200 body with `choices=null` and a
rate-limit message. litellm's in-SDK retries are available via
`num_retries` but default to off — the evaluation runner already
retries transient failures per slot.

## Codex CLI

```python
from llenvs.inference.backends import CodexCLIBackend

backend = CodexCLIBackend(
    model="codex-mini-latest",
    max_concurrency=4,  # Recommended: keep this small; each request spawns a CLI process
    timeout=600.0,
    profile="default",
)
```

### Notes

- Runs `codex exec` in a fresh temporary directory for every request.
- Locks Codex to `--sandbox read-only` and disables session persistence with `--ephemeral`.
- Supports chat generation only. Native tool calling, vision, logprobs, prefix continuation, and full scoring are not available.
- `SamplingParams.max_tokens` is accepted but ignored because `codex exec` does not expose a direct equivalent.
- Use the backend constructor's `config_overrides` argument to pass Codex-specific `-c key=value` overrides when your local Codex CLI/config supports them.

### Capabilities

| Feature | Supported | Notes |
|---------|-----------|-------|
| logprobs | ✗ | Not exposed by the CLI |
| prefix_continuation | ✗ | Stateless transcript replay only |
| batching | ✓ | Parallel subprocess calls |
| streaming | ✗ | Returns one final assistant message |
| chat | ✓ | Transcript rendered into each request |
| function_calling | ✗ | Native tool calls unsupported |
| vision | ✗ | Text-only |

## Claude Code CLI

```python
from llenvs.inference.backends import ClaudeCodeBackend

backend = ClaudeCodeBackend(
    model="sonnet",       # alias, or a full name like "claude-sonnet-4-6"
    max_concurrency=4,    # Recommended: keep this small; each request spawns a CLI process
    timeout=600.0,
    effort="low",         # "low" | "medium" | "high" | "xhigh" | "max"
    max_budget_usd=0.10,  # per-call USD safety cap
)
```

### Notes

- Runs `claude --print --output-format json` in a fresh temporary directory for every request.
- Always passes `--no-session-persistence` and `--strict-mcp-config`. By default `--tools ""` is sent so the spawned CLI cannot run Bash/Edit/Read tools — the backend behaves as a pure text-completion wrapper.
- `--bare` is **off by default** so the CLI can use the same OAuth/keychain login as the interactive TUI (e.g. a Claude Pro/Max subscription). Pass `bare=True` to force `ANTHROPIC_API_KEY` / `apiKeyHelper` auth and skip auto-discovery — recommended for reproducible CI.
- Supports chat generation only. Native tool calling, vision, logprobs, prefix continuation, and full scoring are not available.
- `SamplingParams.max_tokens` is accepted but ignored because the Claude Code CLI does not expose a direct equivalent. Use `effort` to control reasoning level and `max_budget_usd` as a per-call USD cap.
- Errors from the underlying Anthropic API are classified into `PromptTooLongError` (context-limit messages), `RuntimeError` (auth failures — not retryable), or `QuotaExhaustedError` (everything else, including `is_error: true` payloads, rate limits, server overload, timeouts, and unparseable output). The CLI itself retries 10× internally before surfacing errors (configurable via `CLAUDE_CODE_MAX_RETRIES`).
- Use the backend constructor's `extra_args` argument (a list of strings) to pass any Claude Code flag this backend does not model directly.

### Capabilities

| Feature | Supported | Notes |
|---------|-----------|-------|
| logprobs | ✗ | Not exposed by the CLI |
| prefix_continuation | ✗ | Stateless transcript replay only |
| batching | ✓ | Parallel subprocess calls |
| streaming | ✗ | Returns one final assistant message |
| chat | ✓ | Transcript rendered into each request |
| function_calling | ✗ | Native tool calls unsupported |
| vision | ✗ | Text-only |

## Sampling Parameters

```python
from llenvs.inference import SamplingParams

params = SamplingParams(
    max_tokens=2048,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    stop_sequences=("</answer>",),
    presence_penalty=0.0,
    frequency_penalty=0.0,
    n=1,
    logprobs=False,
)
```

### Backend-Specific Parameters

Use the `extra` parameter to pass backend-specific options that aren't part of the common interface. These are forwarded directly to the underlying library.

```python
# HuggingFace-specific (transformers)
params = SamplingParams(
    temperature=0.7,
    extra={
        "repetition_penalty": 1.2,
        "num_beams": 4,
        "do_sample": True,
        "length_penalty": 1.0,
    },
)

# vLLM-specific
params = SamplingParams(
    temperature=0.7,
    extra={
        "best_of": 3,
        "use_beam_search": True,
        "length_penalty": 1.0,
    },
)

# OpenAI-specific
params = SamplingParams(
    temperature=0.7,
    extra={
        "seed": 42,
        "response_format": {"type": "json_object"},
        "logit_bias": {123: -100},
    },
)

# Anthropic-specific
params = SamplingParams(
    temperature=0.7,
    extra={
        "metadata": {"user_id": "user-123"},
    },
)
```

> **Note:** Parameters in `extra` take precedence over the common parameters if there's a conflict. Unknown parameters may cause errors from the underlying API.

### Chat Template Parameters

Local backends (vLLM and HuggingFace) support passing extra keyword arguments to `tokenizer.apply_chat_template()` via `chat_template_kwargs`. This is useful for models like Qwen3 that use special template parameters (e.g., `enable_thinking=True`).

```yaml
model:
  backend: vllm
  model: Qwen/Qwen3-8B
  params:
    chat_template_kwargs:
      enable_thinking: true
```

```python
from llenvs.inference.backends import VLLMBackend

backend = VLLMBackend(
    model_path="Qwen/Qwen3-8B",
    chat_template_kwargs={"enable_thinking": True},
)
```

The kwargs are spread into every `apply_chat_template()` call — both `generate_chat()` and `generate_chat_batch()`.

### Thinking Budget

This section describes the in-process logits-processor implementation
used by `vllm` and `huggingface` — those backends own token generation
and can intervene per-step. For OpenRouter the same `thinking_budget`
field is accepted but flows through the provider's reasoning API
(`reasoning.max_tokens`); see the OpenRouter section above for the
behavior and limitations there. Pure HTTP backends without a reasoning
API (OpenAI, Anthropic direct, Codex CLI, Claude Code CLI, vllm_singularity) ignore
these fields entirely.

For models that produce `<think>...</think>` reasoning blocks (e.g., Qwen3 with `enable_thinking=True`), you can cap the number of tokens generated inside each thinking block using `thinking_budget` on `InferenceConfig`:

```python
from llenvs.core.config import InferenceConfig

config = InferenceConfig(
    max_tokens=4096,
    thinking_budget=512,
)
```

```yaml
inference:
  max_tokens: 4096
  thinking_budget: 512
```

When the budget is reached, the processor forces the model to close the thinking block and continue with its answer. By default, the budget is **shared** across all `<think>` blocks — if the model opens multiple thinking blocks, their token counts accumulate toward the same limit.

#### Per-Block Budget

To give each `<think>` block its own independent budget (counter resets on each new block), set `thinking_budget_per_block`:

```python
config = InferenceConfig(
    max_tokens=4096,
    thinking_budget=512,
    thinking_budget_per_block=True,
)
```

```yaml
inference:
  max_tokens: 4096
  thinking_budget: 512
  thinking_budget_per_block: true
```

#### Early Stopping Text

When the budget is hit, the processor forces a natural-language suffix before `</think>` to help the model transition cleanly from reasoning to answering. This is controlled via `thinking_budget_suffix`. By default it is `None` (bare `</think>`). Pass `DEFAULT_EARLY_STOPPING_SUFFIX` to enable the built-in natural-language transition text.

```python
from llenvs.core.config import InferenceConfig
from llenvs.inference import DEFAULT_EARLY_STOPPING_SUFFIX

# Default: bare </think> (no suffix)
config = InferenceConfig(
    max_tokens=4096,
    thinking_budget=512,
)

# Enable natural-language transition suffix
config = InferenceConfig(
    max_tokens=4096,
    thinking_budget=512,
    thinking_budget_suffix=DEFAULT_EARLY_STOPPING_SUFFIX,
)

# Custom suffix
config = InferenceConfig(
    max_tokens=4096,
    thinking_budget=512,
    thinking_budget_suffix="\n\nLet me answer directly.\n</think>\n\n",
)
```

```yaml
# Default: bare </think> (no suffix)
inference:
  max_tokens: 4096
  thinking_budget: 512

# Custom suffix
inference:
  max_tokens: 4096
  thinking_budget: 512
  thinking_budget_suffix: "\n\nLet me answer directly.\n</think>\n\n"
```

The suffix should end with `</think>` so the model naturally transitions out of the thinking block.

#### Soft Budget Transition

The hard cutoff can cause reasoning to leak into the response when the model is mid-thought. To mitigate this, set `thinking_budget_soft_ratio` to begin boosting the `</think>` logit before the hard cutoff:

```python
config = InferenceConfig(
    max_tokens=4096,
    thinking_budget=512,
    thinking_budget_soft_ratio=0.9,
)
```

```yaml
inference:
  max_tokens: 4096
  thinking_budget: 512
  thinking_budget_soft_ratio: 0.9
```

With `soft_ratio=0.9` and `budget=512`, the `</think>` logit boost begins at token 460 (90% of 512) and ramps linearly up to a max boost of ~5.0 at token 511. At token 512 the hard cutoff forces `</think>` as before. This gives the model a chance to close its thinking block naturally before being forced.

The feature is opt-in — omitting `thinking_budget_soft_ratio` preserves the hard-cutoff-only behavior. The ratio must be between 0 and 1 (exclusive).

Requirements:
- The tokenizer must have `<think>` and `</think>` as single dedicated tokens
- Works with both vLLM (`logits_processors`) and HuggingFace (`logits_processor`) backends
- The vLLM processor is stateful (O(1) per token); the HuggingFace processor derives state from the full token history for batch safety
- **vLLM V1 support**: On vLLM's V1 engine (default since 0.8), the thinking budget processor is registered as an `AdapterLogitsProcessor` at engine init, and per-request budgets are passed via `SamplingParams.extra_args`. On V0, per-request `logits_processors` callables are used instead. Both paths are transparent — just set the `thinking_budget*` fields on `InferenceConfig`

## Second Elicitation

When a generation hits the `max_tokens` limit, the output is abruptly truncated — often mid-thought for reasoning models, never producing an answer. Second elicitation performs a follow-up generation with the truncated output in context, using a smaller token budget, to coax a final answer.

Disabled by default. When enabled, it only triggers for generations with `finish_reason == MAX_TOKENS`.

### Configuration

```python
from llenvs.core.config import InferenceConfig
from llenvs.inference import DEFAULT_EARLY_STOPPING_SUFFIX

config = InferenceConfig(
    max_tokens=2048,
    second_elicitation_suffix=DEFAULT_EARLY_STOPPING_SUFFIX,  # enables the feature
    second_elicitation_max_tokens=256,                         # budget for the follow-up call
)
```

```yaml
inference:
  max_tokens: 2048
  second_elicitation_suffix: "\n\nPlease answer now.\n"  # any non-null value enables the feature
  second_elicitation_max_tokens: 256
```

When a generation is truncated (`MAX_TOKENS`), the runner:

1. Appends the truncated output + suffix as an assistant message
2. Appends `"Please provide the final answer now. Follow the formatting instructions specified above exactly."` as a user message
3. Calls the backend with `second_elicitation_max_tokens` as the token budget
4. Merges the two results: concatenated text, summed token counts, second call's finish reason

The merged result's metadata includes `"second_elicitation": True`.

### Custom Suffix

The suffix you set on `second_elicitation_suffix` is appended to the truncated output in the follow-up call. `DEFAULT_EARLY_STOPPING_SUFFIX` provides a natural-language transition that helps the model shift from reasoning to answering. To customize:

```python
config = InferenceConfig(
    second_elicitation_suffix="\n\nI need to give my final answer now.\n",
)
```

### `max_model_len` Sizing

For vLLM, your `max_model_len` must accommodate the full sequence: `prompt + max_tokens + suffix + second_elicitation_max_tokens`. The runner does not auto-adjust `max_model_len` — set it via `ModelConfig.params`:

```yaml
model:
  backend: vllm
  model: Qwen/Qwen3-8B
  params:
    max_model_len: 8192  # Enough for prompt + first gen + elicitation
```

### Interaction with Thinking Budget

Second elicitation and thinking budget are complementary for the first generation, but the follow-up call is forced into no-thinking mode. The runner clears local thinking-budget fields, sets `SamplingParams.disable_thinking=True`, and backends that support per-call thinking control honor that request:

- vLLM and HuggingFace call the chat template with `enable_thinking=False` and skip thinking-budget logits processors.
- The Singularity-hosted vLLM server backend injects `chat_template_kwargs={"enable_thinking": false}` into the request body (the server renders the chat template, so the toggle must travel with the request).
- OpenRouter sends `extra_body.reasoning = {"effort": "none"}` while preserving unrelated request body fields such as provider routing.

This keeps the rescue call focused on producing the final answer in the required format instead of spending its smaller token budget on another reasoning trace.

## Batch Generation

All backends support `generate_chat_batch()` for processing multiple independent conversations efficiently. This is the foundation for parallel evaluation.

```python
from llenvs.inference import ChatMessage, SamplingParams

messages_batch = [
    [ChatMessage(role="user", content="What is 2+2?")],
    [ChatMessage(role="user", content="What is 3*3?")],
    [ChatMessage(role="user", content="What is 10/2?")],
]

results = backend.generate_chat_batch(messages_batch, SamplingParams())
for result in results:
    print(result.text)
```

How each backend implements batching:

- **vLLM / HuggingFace**: Converts all conversations to prompt strings and calls the model in a single batched GPU call. Maximum throughput.
- **API backends (OpenAI, Anthropic, OpenRouter)**: Fires concurrent async HTTP requests with a semaphore limiting parallelism to `max_concurrency`. No API provider supports sending multiple prompts in a single call, so concurrency is the only way to parallelize.

Tool-calling has an equivalent `generate_with_tools_batch()`.

## Capabilities

Check what a backend supports:

```python
caps = backend.capabilities
print(f"Logprobs: {caps.supports_logprobs}")
print(f"Batching: {caps.supports_batching}")
print(f"Prefix continuation: {caps.supports_prefix_continuation}")
print(f"Function calling: {caps.supports_function_calling}")
```

## Prompt Engineering

For pre-built system prompts, question templates, and model profiles, see the [Prompts guide](prompts.md). The pipeline system below is the low-level API that those higher-level tools build on.

### Building Pipelines

```python
from llenvs.inference.prompting import (
    AnswerFormatInjector,
    ChainOfThoughtWrapper,
    FewShotInjector,
    SystemPromptInjector,
)

# Compose with >> operator
pipeline = (
    SystemPromptInjector("You are an expert mathematician.")
    >> FewShotInjector(
        [
            ("What is 2+3?", "2 + 3 = 5\n<answer>5</answer>"),
        ]
    )
    >> ChainOfThoughtWrapper("think_step_by_step")
    >> AnswerFormatInjector("xml_answer", tag_name="answer")
)

# Apply to messages
messages = [ChatMessage(role="user", content="What is 7*8?")]
transformed = pipeline.transform(messages)
```

### Standard Pipeline Helper

```python
from llenvs.inference.prompting import build_standard_pipeline

pipeline = build_standard_pipeline(
    system_prompt="You are a helpful assistant.",
    examples=[("Q1", "A1"), ("Q2", "A2")],
    use_cot=True,
    answer_format="xml_answer",
    tag_name="answer",
)
```

### Chain of Thought Styles

```python
from llenvs.inference.prompting import ChainOfThoughtWrapper

ChainOfThoughtWrapper("think_step_by_step")
# → "Think through this step by step..."

ChainOfThoughtWrapper("show_work")
# → "Show your work and reasoning..."

ChainOfThoughtWrapper("explain")
# → "Explain your thought process..."
```

### Answer Format Options

```python
from llenvs.inference.prompting import AnswerFormatInjector

AnswerFormatInjector("xml_answer", tag_name="answer")
# → "Put your final answer in <answer>...</answer> tags."

AnswerFormatInjector("json")
# → "Provide your answer as a JSON object..."

AnswerFormatInjector("boxed")
# → "Put your final answer in \\boxed{}."

AnswerFormatInjector("gsm8k")
# → "End your response with '#### ' followed by..."
```

## Answer Extraction

### Built-in Extractors

```python
from llenvs.core.extraction import (
    CompositeExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
    RegexExtractor,
    TagBasedExtractor,
)

# XML tags
extractor = TagBasedExtractor(tag_name="answer")
answer, _ = extractor.extract("Result: <answer>42</answer>")
# answer = "42"

# GSM8K format
extractor = GSM8KExtractor()
answer, _ = extractor.extract("So the answer is #### 42")
# answer = "42"

# Multiple choice
extractor = MultipleChoiceExtractor(choices="ABCD")
answer, _ = extractor.extract("The answer is (B)")
# answer = "B"

# Try multiple extractors
extractor = CompositeExtractor(
    extractors=[
        TagBasedExtractor(tag_name="answer"),
        GSM8KExtractor(),
        RegexExtractor(pattern=r"(\d+)$"),
    ]
)
```

### Custom Extractor

```python
class JsonExtractor:
    def extract(self, response: str) -> tuple[str | None, dict]:
        import json

        try:
            data = json.loads(response)
            return str(data.get("answer")), {"parsed": True}
        except json.JSONDecodeError:
            return None, {"parsed": False}
```
