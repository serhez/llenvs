# Configuration

This document covers configuration options for llenvs.

## CLI Configuration

```yaml
# config.yaml
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 100
    seed: 42
    answer_extractors:
      - type: tag_based
        config: {tag_name: answer}
      - type: pattern_answer
      - type: numeric

model:
  backend: openai
  model: gpt-4o

inference:
  temperature: 0.0
  max_tokens: 2048

system_prompt: math_reasoning          # Pre-built prompt by name
model_profile: auto                    # Detect from model name
prompt_template: math                  # Global default template

output_dir: ./results
```

Run with:

```bash
llenvs run config.yaml
llenvs run config.yaml --limit 10
llenvs run config.yaml --environment leg_counting
```

## Sampling Parameters

```python
from llenvs.inference import SamplingParams

params = SamplingParams(
    max_tokens=2048,        # Maximum tokens to generate
    temperature=0.0,        # Sampling temperature (0 = greedy)
    top_p=1.0,              # Nucleus sampling parameter
    top_k=0,                # Top-k sampling (0 = disabled)
    stop_sequences=(),      # Stop generation at these strings
    presence_penalty=0.0,   # Penalize token presence
    frequency_penalty=0.0,  # Penalize token frequency
    n=1,                    # Number of completions
    logprobs=False,         # Return token logprobs
    num_logprobs=5,         # How many logprobs per token
)
```

## Backend Configuration

### OpenAI

```python
from llenvs.inference.backends import OpenAIBackend

backend = OpenAIBackend(
    model="gpt-4o",
    api_key="sk-...",         # Optional, uses OPENAI_API_KEY env var
    organization="org-...",   # Optional
    base_url=None,            # Custom endpoint
    max_concurrency=64,       # Max concurrent batch requests
)
```

### Anthropic

```python
from llenvs.inference.backends import AnthropicBackend

backend = AnthropicBackend(
    model="claude-sonnet-4-20250514",
    api_key="...",            # Optional, uses ANTHROPIC_API_KEY env var
    max_concurrency=64,       # Max concurrent batch requests
)
```

### vLLM

```python
from llenvs.inference.backends import VLLMBackend

backend = VLLMBackend(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,   # Number of GPUs
    dtype="bfloat16",         # Model dtype
    max_model_len=4096,       # Context length
    gpu_memory_utilization=0.9,
)
```

### OpenRouter

```python
from llenvs.inference.backends import OpenRouterBackend

backend = OpenRouterBackend(
    model="anthropic/claude-sonnet-4-20250514",
    api_key="...",            # Optional, uses OPENROUTER_API_KEY env var
    site_url="https://mysite.com",
    app_name="MyApp",
    max_concurrency=64,       # Max concurrent batch requests
)
```

### Codex CLI

```python
from llenvs.inference.backends import CodexCLIBackend

backend = CodexCLIBackend(
    model="codex-mini-latest",
    max_concurrency=4,        # Recommended: each request spawns a local CLI process
    timeout=600.0,
    profile="default",        # Optional Codex profile
    # config_overrides={...}, # Optional `codex exec -c key=value` passthrough
)
```

YAML:

```yaml
model:
  backend: codex
  model: codex-mini-latest
  max_concurrency: 4
  params:
    timeout: 600.0
    profile: default
```

Notes:

- Each request runs in a fresh temporary directory with `codex exec --sandbox read-only --ephemeral --skip-git-repo-check`.
- `SamplingParams.max_tokens` is accepted but ignored because the Codex CLI does not expose a direct equivalent.
- Use `model.params.config_overrides` to pass Codex-specific `-c key=value` overrides when your local Codex CLI/config supports them.

## Backend Capabilities

| Backend | Logprobs | Prefix Continuation | Batching | Tool Calling |
|---------|----------|---------------------|----------|--------------|
| vLLM | ✅ | ✅ | ✅ (GPU) | ✅ |
| HuggingFace | ✅ | ✅ | ✅ (GPU) | ❌ |
| OpenAI | ✅ | ❌ | ✅ (concurrent) | ✅ |
| Anthropic | ❌ | ✅ (prefill) | ✅ (concurrent) | ✅ |
| OpenRouter | varies | ❌ | ✅ (concurrent) | varies |
| Codex CLI | ❌ | ❌ | ✅ (subprocess) | ❌ |

Check programmatically:

```python
caps = backend.capabilities
print(f"Logprobs: {caps.supports_logprobs}")
print(f"Batching: {caps.supports_batching}")
print(f"Tools: {caps.supports_function_calling}")
```

## Environment Configuration

### Reasoning-Gym

```python
from llenvs.core.registry import environment_registry

env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=100,                      # Number of samples
    seed=42,                       # Random seed
    answer_extractor=None,         # Use default TagBasedExtractor
    # Additional dataset-specific kwargs passed through
)

# Add optional extra rewards (e.g., format compliance)
from llenvs.core.reward import FormatReward
env_with_format = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=100,
    extra_rewards=(FormatReward(env._answer_extractor),),
)
```

### HuggingFace

```python
from llenvs.core.registry import environment_registry

env = environment_registry.get(
    name="gsm8k",
    adapter="huggingface",
    subset="main",                 # Dataset subset/config
    split="test",                  # train, test, validation
    question_column="question",    # Column with questions
    answer_column="answer",        # Column with answers
    ground_truth_extractor="numeric",  # boxed, numeric, last_line, direct
    scoring="numeric",             # exact, numeric, numeric_tolerance
    size=100,                      # Limit to N examples
    seed=42,                       # Shuffle seed
)
```

### GEM

```python
from llenvs.core.registry import environment_registry

# Basic environment
env = environment_registry.get(
    name="game:Wordle-v0",
    adapter="gem",
    max_steps=6,
)

# Tool-enabled environment
env = environment_registry.get_environment(
    name="math:GSM8K",
    adapter="gem",
    tool_types=("python",),        # Tools to enable
    max_steps=10,
    # For search tool:
    search_url="http://localhost:8000/retrieve",
    search_topk=3,
)
```

### WebShop

```python
from llenvs.core.registry import environment_registry

env = environment_registry.get(
    name="webshop:text_rich",
    adapter="webshop",
    max_steps=15,
    num_products=1000,             # None for full dataset
    human_goals=True,              # Use human-written goals
)
```

## Prompt Configuration

Configure system prompts, question templates, and model profiles. See the [Prompts guide](../guides/prompts.md) for full details.

### system_prompt

A string or list of strings. Each string is resolved by looking up in the system prompt registry, then the fragment registry, then treating as literal text.

```yaml
# Pre-built prompt by name
system_prompt: math_reasoning

# Composed from fragments
system_prompt:
  - math_expert
  - think_step_by_step
  - xml_answer

# Literal string
system_prompt: "You are a helpful assistant. Think step by step."
```

### prompt_template

A string referencing a registered template name (`plain`, `math`, `coding`, `reasoning`) or a literal template with a `{question}` placeholder. Applied to the last user message at runtime.

```yaml
prompt_template: math
```

### model_profile

A string referencing a registered profile name (`deepseek_r1`, `o1`, `llama3_instruct`, `qwen_chat`) or `"auto"` to detect from the model name.

```yaml
model_profile: deepseek_r1
model_profile: auto          # Detect from model name
```

### Per-Environment Overrides

`system_prompt` and `prompt_template` can be set per-environment to override the eval-level defaults:

```yaml
system_prompt: general_reasoning
prompt_template: reasoning

environments:
  - name: simple_arithmetic
    # Uses eval-level defaults

  - name: polynomial_equations
    system_prompt: math_reasoning    # Override for this env
    prompt_template: math            # Override for this env
```

### prompts

A dict of named prompt component overrides for multi-step environments. Keys and their meaning are environment-specific. Single-turn environments ignore this field.

```yaml
environments:
  - name: webshop
    adapter: webshop
    prompts:
      instruction_prefix: "Your goal: {instruction}"
      action_hint: "Navigate using search[keywords] or click[element]."
```

WebShop prompt components:

| Key | Default | Description |
|-----|---------|-------------|
| `instruction_prefix` | `Instruction: {instruction}` | Template prepended to each observation |
| `action_hint` | `Actions: search[keywords] or click[element]` | Available actions hint |

## Extraction Configuration

### Extractor Chains

Configure an ordered list of extractors to try. The first extractor that succeeds is used:

```yaml
environments:
  - name: polynomial_equations
    adapter: reasoning_gym
    answer_extractors:
      - type: tag_based
        config: {tag_name: answer}
      - type: boxed
      - type: pattern_answer
      - type: numeric
```

Each entry has a `type` (registry name) and optional `config` (kwargs passed to the extractor constructor). Available types: `tag_based`, `regex`, `gsm8k`, `multiple_choice`, `boxed`, `numeric`, `last_line`, `code_block`, `pattern_answer`, `raw`, `native`.

The `native` type uses the adapter's built-in extraction (only supported by `reasoning_gym`).

As a shorthand, a single extractor can be specified:

```yaml
environments:
  - name: test
    answer_extractor: tag_based
    answer_extractor_config: {tag_name: answer}
```

### Cleaning Layer

Pre-cleaners run on the raw response before extraction. Post-cleaners run on the extracted answer after extraction. `EnvironmentFactory` applies cleaning automatically.

```yaml
environments:
  - name: math_task
    answer_extractors:
      - type: boxed
      - type: numeric
    # Defaults: strip_special_tokens pre-cleaner, strip_trailing_punctuation post-cleaner

  - name: code_generation
    answer_extractors:
      - type: code_block
        config: {language: python}
    pre_cleaners: [strip_special_tokens]
    post_cleaners: []  # Disable post-cleaning for code
```

Semantics:
- **Not specified** (`None`) — use defaults (`strip_special_tokens` pre, `strip_trailing_punctuation` post)
- **Empty list** (`[]`) — disable cleaning entirely
- **Explicit list** — use exactly those cleaners

Available pre-cleaners: `strip_special_tokens`, `strip_thinking_tokens`

Available post-cleaners: `strip_trailing_punctuation`, `strip_surrounding_quotes`, `strip_latex_dollars`

### Parameterized Cleaners

Cleaners that take arguments are specified as dicts with `type` and optional `config`. They can be used in both `pre_cleaners` and `post_cleaners`:

```yaml
environments:
  - name: math_task
    post_cleaners:
      - strip_trailing_punctuation
      - type: truncate_tail
        config:
          max_chars: 512
```

Available parameterized cleaners:

| Name | Config | Default | Description |
|------|--------|---------|-------------|
| `truncate_tail` | `max_chars` | 256 | Keep only the last N characters (strips whitespace first) |

Parameterized cleaners can be mixed freely with simple string cleaner names.

## Prompt Pipeline Configuration

```python
from llenvs.inference.prompting import build_standard_pipeline

pipeline = build_standard_pipeline(
    system_prompt="You are a helpful assistant.",
    examples=[("Q1", "A1"), ("Q2", "A2")],  # Few-shot examples
    use_cot=True,                           # Add chain-of-thought
    answer_format="xml_answer",             # xml_answer, json, boxed, gsm8k
    tag_name="answer",                      # For xml_answer format
)
```

Or compose manually:

```python
from llenvs.inference.prompting import (
    SystemPromptInjector,
    FewShotInjector,
    ChainOfThoughtWrapper,
    AnswerFormatInjector,
)

pipeline = (
    SystemPromptInjector("You are an expert.")
    >> FewShotInjector([("Q", "A")])
    >> ChainOfThoughtWrapper("think_step_by_step")
    >> AnswerFormatInjector("xml_answer", tag_name="answer")
)
```

## Log Configuration

```python
from llenvs.evaluation import LogConfig

LogConfig(
    targets=("console",),        # Log targets: "console", "file", "wandb"
    log_dir=".logs",             # Directory for JSONL file logs
    strip_images=False,          # Strip images from logged events
    wandb_run=None,              # Existing wandb.Run (skips init)
    wandb_project=None,          # W&B project name (auto-creates run)
    wandb_name=None,             # W&B run name (auto-generated if None)
    wandb_config=None,           # Extra config dict for W&B
)
```

Pass `log=LogConfig(...)` to `TrajectoryRunner`, `SegmentedTrajectoryRunner`, `run_evaluation()`, or `run_segmented_evaluation()`. See the [Evaluation guide](../guides/evaluation.md#logging) for usage examples.

CLI usage:

```bash
llenvs run config.yaml --log console,file
llenvs run config.yaml --log wandb
```

## Evaluation Runner Configuration

```python
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import TEMPLATE_REGISTRY, PROFILE_REGISTRY

# TrajectoryRunner handles both text-only and tool environments.
# It auto-detects available tools and uses generate_with_tools when tools are present.
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=params,
    system_prompt="...",             # System prompt string
    prompt_template=TEMPLATE_REGISTRY["math"],  # Optional template
    model_profile=PROFILE_REGISTRY["deepseek_r1"],  # Optional profile
    prompt_pipeline=pipeline,        # Optional low-level pipeline
    max_image_history=None,          # Limit images in message history (None = keep all)
)

# Run evaluation
result = runner.run_trajectory(task_index=0)
batch = runner.run_batch(
    task_indices=list(range(100)),
    progress_callback=lambda c, t: print(f"{c}/{t}"),
)
```

## Container Configuration

Run any environment inside a container or subprocess by adding a `container` field:

```yaml
environments:
  - name: sudoku
    adapter: reasoning_gym
    size: 100
    container:
      runtime: docker           # or "process"
      image: llenvs-rg:latest   # required for docker runtime
      timeout: 60
      env_vars:
        CACHE_DIR: /data/cache
      volumes:
        /host/data: /data
```

```python
from llenvs.container.config import ContainerConfig

ContainerConfig(
    runtime="docker",           # "docker" or "process"
    image="llenvs-rg:latest",   # Docker image (required for docker)
    port=None,                  # Host port (None = auto-select)
    timeout=60.0,               # Startup timeout in seconds
    env_vars={},                # Environment variables
    volumes={},                 # Volume mounts (host -> container)
    docker_command="docker",    # Path to docker CLI
)
```

When `container` is set, `EnvironmentFactory.create()` starts the runtime and returns a `ContainerEnvironment` proxy. See the **[Containers Guide](../guides/containers.md)** for details.

## Judge Configuration

Add LLM-as-a-judge scoring at the eval level or per-environment. See the **[Judge guide](../guides/judge.md)** for full details.

```yaml
judge:
  model:
    backend: openai
    model: gpt-4o-mini
  template: correctness             # correctness, helpfulness, safety, or literal
  system_prompt: null               # Override template default
  score_range: [1, 10]              # For normalization to [0,1]
  name: judge                       # Signal name
  reward_type: outcome              # outcome, step, format, process
  weight: 1.0                       # Signal weight
  normalize: true                   # Normalize to [0,1]
  inference:                        # Judge sampling params
    temperature: 0.0
    max_tokens: 512
```

```python
from llenvs.core.config import JudgeConfig, ModelConfig

JudgeConfig(
    model=ModelConfig(backend="openai", model="gpt-4o-mini"),
    template="correctness",
    system_prompt=None,
    score_range=(1.0, 10.0),
    name="judge",
    reward_type="outcome",
    weight=1.0,
    normalize=True,
    inference=None,
)
```

Per-env overrides eval-level. Set `judge` on `EnvironmentConfig` or `EvalConfig`. Supports a single `JudgeConfig` or a list for multiple judges.

## Environment LLM Configuration

Configure an environment-internal LLM for environments that use an LLM in their transition function (e.g., `DialogueEnvironment`). See the **[Dialogue guide](../guides/dialogue.md)** for full details.

```yaml
environments:
  - name: twenty_questions
    adapter: dialogue
    params:
      words: [cat, dog, elephant]
    env_llm:
      model:
        backend: openai
        model: gpt-4o-mini
      system_prompt: ""               # Base system prompt (preset provides default)
      inference:                      # Env LLM sampling params
        temperature: 0.0
        max_tokens: 256
```

```python
from llenvs.core.config import EnvironmentLLMConfig, ModelConfig, InferenceConfig

EnvironmentLLMConfig(
    model=ModelConfig(backend="openai", model="gpt-4o-mini"),
    system_prompt="",                     # Base system prompt
    inference=InferenceConfig(            # Defaults: temp=0, max_tokens=512
        temperature=0.0,
        max_tokens=256,
    ),
)
```

When `env_llm` is set, `EnvironmentFactory.create()` creates a `ModelBackend` and passes it as `env_llm` along with `sampling_params` and `system_prompt` to the adapter.

## Branching Configuration

Set a branching strategy preference per environment:

```yaml
environments:
  - name: webshop
    adapter: webshop
    branching_strategy: process_fork   # direct, action_replay, process_fork, or null
```

```python
from llenvs.core.config import EnvironmentConfig

config = EnvironmentConfig(
    name="webshop",
    adapter="webshop",
    branching_strategy="process_fork",
)
```

When `None` (default), `BranchManager.create()` auto-resolves the best strategy based on environment capabilities. See the **[Branching Guide](../guides/branching.md)** for details.

## Iterative Configuration

Wrap any single-turn environment in an iterative refinement loop. See the **[Iterative Refinement Guide](../guides/iterative.md)** for full details.

```yaml
environments:
  - name: openai/openai_humaneval
    adapter: huggingface
    iterative:
      max_turns: 5                           # Maximum refinement turns
      include_history: true                  # Include previous attempts in feedback
      summarize_history: false               # Use env LLM to summarize history
      submit_keyword: SUBMIT                 # Keyword for early termination (null to disable)
      submission_extractor: code_block       # Extractor name for parsing submissions
      submission_extractor_config:
        language: python
      solved_threshold: 1.0                  # OUTCOME score threshold for solved
      code_execution:                        # Optional code execution
        timeout: 30.0                        # Execution timeout in seconds
```

```python
from llenvs.core.config import EnvironmentConfig, IterativeConfig, CodeExecutionConfig

config = EnvironmentConfig(
    name="openai/openai_humaneval",
    adapter="huggingface",
    iterative=IterativeConfig(
        max_turns=5,
        submission_extractor="code_block",
        submission_extractor_config={"language": "python"},
        code_execution=CodeExecutionConfig(timeout=30.0),
    ),
)
```

When `iterative` is set, `EnvironmentFactory.create()` wraps the base environment in an `IterativeEnvironment`. The `submission_extractor` is resolved from the answer extractor registry, and `code_execution` creates a `CodeExecutionReward` with a `SubprocessCodeExecutor`.

## MCP Server Configuration

```python
from llenvs.core import MCPServerConfig, MCPToolExecutor

config = MCPServerConfig(
    command="npx",                      # Command to start server
    args=("-y", "@mcp/server", "/tmp"), # Command arguments
    env={"VAR": "value"},               # Environment variables
    timeout=30.0,                       # Request timeout
)

executor = MCPToolExecutor(config)
```
