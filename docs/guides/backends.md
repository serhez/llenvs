# Inference Backends

llenvs supports multiple inference backends with a unified interface.

## Available Backends

| Backend | Package | Features |
|---------|---------|----------|
| vLLM | `vllm` | Logprobs, batching, prefix continuation |
| OpenAI | `openai` | Logprobs, streaming, function calling |
| Anthropic | `anthropic` | Prefix continuation (prefill), streaming |
| OpenRouter | - | Access to multiple models |

## vLLM (Local Inference)

```python
from llenvs.inference.backends import VLLMBackend

backend = VLLMBackend(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# Full feature support
print(backend.capabilities.supports_logprobs)  # True
print(backend.capabilities.supports_batching)  # True

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

## OpenAI

```python
from llenvs.inference.backends import OpenAIBackend

# Uses OPENAI_API_KEY env var by default
backend = OpenAIBackend(model="gpt-4o")

# Or explicit key
backend = OpenAIBackend(
    model="gpt-4o",
    api_key="sk-...",
    organization="org-...",
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

backend = AnthropicBackend(model="claude-sonnet-4-20250514")

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
)
```

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

### Building Pipelines

```python
from llenvs.inference.prompting import (
    SystemPromptInjector,
    FewShotInjector,
    ChainOfThoughtWrapper,
    AnswerFormatInjector,
)

# Compose with >> operator
pipeline = (
    SystemPromptInjector("You are an expert mathematician.")
    >> FewShotInjector([
        ("What is 2+3?", "2 + 3 = 5\n<answer>5</answer>"),
    ])
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
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
    CompositeExtractor,
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
extractor = CompositeExtractor(extractors=[
    TagBasedExtractor(tag_name="answer"),
    GSM8KExtractor(),
    RegexExtractor(pattern=r"(\d+)$"),
])
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
