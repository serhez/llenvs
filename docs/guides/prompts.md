# Prompts

llenvs provides composable building blocks for prompt engineering: reusable fragments, pre-built system prompts, question templates, and model-family profiles.

## Quick Start

### YAML Configuration

```yaml
# Use a pre-built system prompt by name
system_prompt: math_reasoning

# Or compose from fragments
system_prompt:
  - math_expert
  - think_step_by_step
  - xml_answer

# Per-environment overrides
environments:
  - name: leg_counting
  - name: polynomial_equations
    system_prompt: math_reasoning
    prompt_template: math

# Model-specific adjustments
model_profile: deepseek_r1   # or "auto" for detection
```

### Programmatic API

```python
from llenvs.inference import (
    PromptFragment,
    SystemPrompt,
    PromptTemplate,
    ModelProfile,
    compose_system_prompt,
    resolve_system_prompt,
    FRAGMENT_REGISTRY,
    SYSTEM_PROMPT_REGISTRY,
    TEMPLATE_REGISTRY,
    PROFILE_REGISTRY,
)
```

## Fragments

`PromptFragment` is a small, reusable instruction snippet with a name, content, and category.

```python
from llenvs.inference import FRAGMENT_REGISTRY

# Browse available fragments
for name, frag in FRAGMENT_REGISTRY.items():
    print(f"[{frag.category}] {name}: {frag.content[:60]}...")
```

### Built-in Fragments

**Reasoning:**

| Name | Content |
|------|---------|
| `think_step_by_step` | Think through this step by step before giving your final answer. |
| `show_your_work` | Show your work and reasoning before providing the answer. |
| `reflect_then_answer` | Consider the problem carefully, reflect on your approach, then provide your answer. |
| `verify_your_answer` | After finding your answer, verify it by checking your work. |
| `direct_answer` | Provide only the final answer with no reasoning or explanation. |

**Format:**

| Name | Content |
|------|---------|
| `xml_answer` | Put your final answer in \<answer\>...\</answer\> tags. |
| `boxed_answer` | Put your final answer in \boxed{}. |
| `gsm8k_answer` | End your response with '#### ' followed by your numerical answer. |
| `json_answer` | Provide your answer as a JSON object with an 'answer' field. |

**Persona:**

| Name | Content |
|------|---------|
| `math_expert` | You are an expert mathematician. Solve problems rigorously and precisely. |
| `coding_expert` | You are an expert programmer. Write clean, correct, and efficient code. |
| `general_assistant` | You are a helpful assistant. |

**Constraint:**

| Name | Content |
|------|---------|
| `be_concise` | Be concise and avoid unnecessary verbosity. |
| `no_apologies` | Do not apologize or hedge. Be direct and confident. |
| `single_answer` | Provide exactly one answer. Do not offer alternatives. |

## System Prompts

`SystemPrompt` is a complete system prompt, typically composed from fragments.

```python
from llenvs.inference import SYSTEM_PROMPT_REGISTRY

prompt = SYSTEM_PROMPT_REGISTRY["math_reasoning"]
print(prompt.content)
```

### Built-in System Prompts

| Name | Composed From |
|------|---------------|
| `general_reasoning` | general_assistant + think_step_by_step + xml_answer |
| `math_reasoning` | math_expert + show_your_work + verify_your_answer + xml_answer |
| `math_boxed` | math_expert + show_your_work + boxed_answer |
| `math_gsm8k` | general_assistant + show_your_work + gsm8k_answer |
| `coding_problem` | coding_expert + think_step_by_step + xml_answer |
| `concise_reasoning` | general_assistant + think_step_by_step + be_concise + single_answer + xml_answer |
| `direct_answer` | general_assistant + direct_answer + xml_answer |

## Composition API

### compose_system_prompt

Join any mix of strings, fragments, and system prompts:

```python
from llenvs.inference import compose_system_prompt, FRAGMENT_REGISTRY

# Mix fragments with custom text
prompt = compose_system_prompt(
    FRAGMENT_REGISTRY["math_expert"],
    "Focus on algebraic problems.",
    FRAGMENT_REGISTRY["xml_answer"],
)
```

### resolve_system_prompt

Resolve a config value (string or list) to final text. Looks up names in the system prompt registry first, then fragment registry, then treats as literal:

```python
from llenvs.inference import resolve_system_prompt

# Single name → registry lookup
text = resolve_system_prompt("math_reasoning")

# List → resolve each, join with double newlines
text = resolve_system_prompt(["math_expert", "think_step_by_step", "xml_answer"])

# Unknown name → literal passthrough
text = resolve_system_prompt("You are a custom assistant.")
```

## Prompt Templates

`PromptTemplate` wraps a task question with context. The template string uses `{question}` as a placeholder.

```python
from llenvs.inference import TEMPLATE_REGISTRY

template = TEMPLATE_REGISTRY["math"]
wrapped = template.apply("What is 2+2?")
# → "Solve the following math problem.\n\nWhat is 2+2?"
```

### Built-in Templates

| Name | Template |
|------|----------|
| `plain` | `{question}` |
| `math` | `Solve the following math problem.\n\n{question}` |
| `coding` | `Solve the following programming problem.\n\n{question}` |
| `reasoning` | `Answer the following question. Think carefully.\n\n{question}` |

Templates are applied at the runner level via `PromptTemplateTransformer`, which wraps the last user message. This keeps environments as pure data sources.

### Custom Templates

In YAML, the `prompt_template` field can reference a registry name or a literal template:

```yaml
# Registry name
prompt_template: math

# Literal template (must contain {question})
prompt_template: "Read the problem carefully.\n\n{question}\n\nShow your work."
```

## Model Profiles

`ModelProfile` captures model-family-specific prompting adjustments. Profiles can prepend/append to the system prompt, remap message roles, and specify stop sequences.

```python
from llenvs.inference import PROFILE_REGISTRY

profile = PROFILE_REGISTRY["deepseek_r1"]
transformers = profile.build_transformers()
# Returns list of PromptTransformers to apply
```

### Built-in Profiles

| Name | Adjustments |
|------|-------------|
| `deepseek_r1` | Appends `<think>`/`<answer>` tag instructions to system prompt |
| `o1` | Remaps `system` role to `developer` |
| `llama3_instruct` | No overrides (standard) |
| `qwen_chat` | No overrides (standard) |

### Auto-Detection

Set `model_profile: auto` in config to detect the profile from the model name:

```yaml
model:
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
model_profile: auto  # → detects "deepseek_r1"
```

```python
from llenvs.inference import detect_model_profile

profile = detect_model_profile("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
# → ModelProfile(name="deepseek_r1", ...)
```

## Resolution Order

When running an evaluation, prompts are resolved via `resolve_prompt_config()`:

| Setting | Resolution chain (first non-None wins) |
|---------|----------------------------------------|
| `system_prompt` | env_config → eval_config → adapter default → library fallback (single-turn only) |
| `prompt_template` | env_config → eval_config → adapter default → None |
| `model_profile` | eval_config only → None |

Per-environment settings always override the eval-level defaults.

### Adapter Defaults

Adapters can provide default system prompts and prompt templates for their environments via `get_default_system_prompt()` and `get_prompt_template()`. These sit between user config and the library fallback in the resolution chain. Currently all built-in adapters return `None` (no adapter-level defaults), but custom adapters can override this.

### Library Fallback

For **single-turn** environments where no system prompt is configured (neither by the user nor by the adapter), llenvs falls back to the `general_reasoning` system prompt. This ensures the model gets `<answer>` tag instructions matching the default `tag_based` extractor, so zero-config evaluations work out of the box.

**Multi-turn** environments get no library fallback — they manage their own prompts through observations.

## Runner Integration

The runner applies prompts in this order:

1. System prompt → system message
2. Observation → user message
3. **Prompt template** → wraps last user message
4. **Model profile** → model-specific transformers (role mapping, system prompt prefix/suffix)
5. **Prompt pipeline** → user-configured transformers (few-shot, CoT, etc.)

```python
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import TEMPLATE_REGISTRY, PROFILE_REGISTRY

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    system_prompt="You are an expert mathematician.",
    prompt_template=TEMPLATE_REGISTRY["math"],
    model_profile=PROFILE_REGISTRY["deepseek_r1"],
)
```

Or via the convenience function:

```python
from llenvs.evaluation import run_evaluation
from llenvs.inference import TEMPLATE_REGISTRY, PROFILE_REGISTRY

result = run_evaluation(
    environment=env,
    backend=backend,
    system_prompt="You are an expert mathematician.",
    prompt_template=TEMPLATE_REGISTRY["math"],
    model_profile=PROFILE_REGISTRY["deepseek_r1"],
)
```

## Multi-Step Environment Prompts

Multi-step environments (like WebShop) use internal prompt templates to build observations. These are exposed as named prompt components via the `prompts` property and can be overridden at construction time.

### WebShop Prompt Components

| Name | Default | Description |
|------|---------|-------------|
| `instruction_prefix` | `Instruction: {instruction}` | Prefixed to each observation with the shopping goal |
| `step_format` | `[Step {step}]` | Step counter |
| `action_hint` | `Actions: search[keywords] or click[element]` | Available actions hint |

### Overriding via YAML

```yaml
environments:
  - name: webshop
    adapter: webshop
    prompts:
      action_hint: "Navigate using search[keywords] or click[element]."
      instruction_prefix: "Your goal: {instruction}"
```

### Overriding Programmatically

```python
from llenvs.adapters.webshop import WebShopEnvironment, DEFAULT_WEBSHOP_PROMPTS

# Check defaults
print(DEFAULT_WEBSHOP_PROMPTS)

# Override specific components
env = WebShopEnvironment(
    webshop_env=raw_env,
    prompts={"action_hint": "Use search[q] or click[e]."},
)

# Read back active prompts
print(env.prompts)
```

### Single-Turn Environments

Single-turn environments (reasoning-gym, HuggingFace, GEM single-turn) return an empty `prompts` dict — they don't use internal prompt templates.

## Combining with Prompt Pipelines

The prompt library works alongside the existing `PromptPipeline` system. Use `PromptTemplateTransformer` directly in a pipeline:

```python
from llenvs.inference import PromptTemplateTransformer, TEMPLATE_REGISTRY
from llenvs.inference.prompting import SystemPromptInjector, AnswerFormatInjector

pipeline = (
    SystemPromptInjector("You are an expert.")
    >> PromptTemplateTransformer(template=TEMPLATE_REGISTRY["math"])
    >> AnswerFormatInjector("xml_answer")
)
```
