# Configuration Reference

This document covers configuration options for llenvs.

## CLI Configuration

```yaml
# config.yaml
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 100
    seed: 42
    extractor: tag_based

model:
  backend: openai
  model: gpt-4o

inference:
  temperature: 0.0
  max_tokens: 2048

system_prompt: |
  You are a helpful assistant. Think step by step.
  Put your final answer in <answer>...</answer> tags.

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
)
```

### Anthropic

```python
from llenvs.inference.backends import AnthropicBackend

backend = AnthropicBackend(
    model="claude-sonnet-4-20250514",
    api_key="...",            # Optional, uses ANTHROPIC_API_KEY env var
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
)
```

## Backend Capabilities

| Backend | Logprobs | Prefix Continuation | Batching | Tool Calling |
|---------|----------|---------------------|----------|--------------|
| vLLM | ✅ | ✅ | ✅ | ✅ |
| OpenAI | ✅ | ❌ | ❌ | ✅ |
| Anthropic | ❌ | ✅ (prefill) | ❌ | ✅ |
| OpenRouter | varies | ❌ | ❌ | varies |

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
from llenvs.adapters import create_reasoning_gym_environment

env = create_reasoning_gym_environment(
    dataset_name="leg_counting",
    size=100,                      # Number of samples
    seed=42,                       # Random seed
    extractor=None,                # Use default TagBasedExtractor
    include_format_reward=True,    # Include format compliance reward
    # Additional dataset-specific kwargs passed through
)
```

### HuggingFace

```python
from llenvs.adapters import HuggingFaceAdapter

adapter = HuggingFaceAdapter()
env = adapter.get_environment(
    name="gsm8k",
    subset="main",                 # Dataset subset/config
    split="test",                  # train, test, validation
    question_column="question",    # Column with questions
    answer_column="answer",        # Column with answers
    answer_extraction="numeric",   # boxed, numeric, last_line, direct
    scoring="numeric",             # exact, numeric, numeric_tolerance
    size=100,                      # Limit to N examples
    seed=42,                       # Shuffle seed
)
```

### GEM

```python
from llenvs.adapters import create_gem_environment, create_gem_tool_environment

# Basic environment
env = create_gem_environment(
    env_id="game:Wordle-v0",
    max_steps=6,
    include_format_reward=True,
)

# Tool-enabled environment
env = create_gem_tool_environment(
    env_id="math:GSM8K",
    tool_types=("python",),        # Tools to enable
    max_steps=10,
    # For search tool:
    search_url="http://localhost:8000/retrieve",
    search_topk=3,
)
```

### WebShop

```python
from llenvs.adapters import create_webshop_environment

env = create_webshop_environment(
    observation_mode="text_rich",  # text_rich, text, html
    max_steps=15,
    num_products=1000,             # None for full dataset
    human_goals=True,              # Use human-written goals
)
```

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

## Evaluation Runner Configuration

```python
from llenvs.evaluation import EpisodeRunner, ToolEpisodeRunner

# Basic runner
runner = EpisodeRunner(
    environment=env,
    backend=backend,
    sampling_params=params,
    prompt_pipeline=pipeline,    # Optional
    system_prompt="...",         # Alternative to pipeline
)

# Tool-aware runner
runner = ToolEpisodeRunner(
    environment=env,
    backend=backend,
    sampling_params=params,
    system_prompt="Use tools to solve problems.",
)

# Run evaluation
result = runner.run_episode(task_index=0)
batch = runner.run_batch(
    task_indices=list(range(100)),
    progress_callback=lambda c, t: print(f"{c}/{t}"),
)
```

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
