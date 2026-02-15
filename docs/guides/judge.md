# LLM-as-a-Judge

Use an LLM to score model responses for open-ended tasks where no ground truth or deterministic metric exists.

`JudgeReward` implements the `RewardFunction` protocol and works with any environment via the `extra_rewards` pattern.

## Quick Start

```python
from llenvs.core.judge import JudgeReward
from llenvs.inference.backends.api import OpenAIBackend

# Create judge backend
judge_backend = OpenAIBackend(model="gpt-4o-mini")

# Create judge reward
judge = JudgeReward(
    backend=judge_backend,
    template="correctness",  # Built-in template
)

# Use as extra_reward on any environment
from llenvs.core.registry import environment_registry

env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    extra_rewards=(judge,),
)
```

## Built-in Templates

| Template | Placeholders | Use Case |
|----------|-------------|----------|
| `"correctness"` | question, response, ground_truth | Reference-based correctness scoring |
| `"helpfulness"` | question, response | Reference-free helpfulness scoring |
| `"safety"` | question, response | Reference-free safety scoring |
| `"iterative_feedback"` | question, response, ground_truth | Actionable feedback for iterative refinement |

All templates use the `[[score]]` output format (MT-Bench convention) on a 1-10 scale.

```python
from llenvs.core.judge import JUDGE_TEMPLATES

# Inspect a template
t = JUDGE_TEMPLATES["correctness"]
print(t.template)
print(t.score_range)       # (1.0, 10.0)
print(t.system_prompt)     # "You are a fair and accurate judge..."
```

## Custom Templates

```python
from llenvs.core.judge import JudgeReward, JudgePromptTemplate

template = JudgePromptTemplate(
    template=(
        "Evaluate this code review response.\n\n"
        "Question: {question}\n"
        "Response: {response}\n\n"
        "Rate from 1-5. Output [[score]]."
    ),
    name="code_review",
    score_range=(1.0, 5.0),
    system_prompt="You are an expert code reviewer.",
)

judge = JudgeReward(backend=backend, template=template)
```

## Score Extraction

The default extractor (`extract_judge_score`) tries two patterns:

1. `[[7]]` or `[[7.5]]` — bracket format (preferred, what templates ask for)
2. `Score: 7` / `Rating: 7` — fallback for non-compliant models

Last match wins. If no score is found, returns `default_score` (default: 0.0).

For non-standard judge output, provide a custom extractor:

```python
def custom_extractor(text: str) -> float | None:
    """Parse SCORE=N format."""
    if "SCORE=" in text:
        return float(text.split("SCORE=")[1].strip())
    return None

judge = JudgeReward(
    backend=backend,
    template="correctness",
    score_extractor=custom_extractor,
)
```

## Normalization

By default, scores are normalized to [0, 1] using the template's `score_range`:

```
normalized = (raw_score - min) / (max - min)
```

Values are clamped to [0, 1]. Disable with `normalize=False` to get raw scores.

## Configuration Options

```python
judge = JudgeReward(
    backend=backend,
    template="correctness",           # Built-in name or JudgePromptTemplate
    sampling_params=SamplingParams(    # Defaults: temperature=0.0, max_tokens=512
        temperature=0.0,
        max_tokens=512,
    ),
    name="judge",                     # Signal name
    reward_type=RewardType.OUTCOME,   # OUTCOME, STEP, FORMAT, PROCESS
    weight=1.0,                       # Signal weight in SignalBundle
    normalize=True,                   # Normalize to [0,1]
    default_score=0.0,                # Fallback on extraction failure
    score_extractor=None,             # Custom parser (None = default)
)
```

## YAML Configuration

Add judge config at the eval level (applies to all environments) or per-environment:

```yaml
# Eval-level judge
judge:
  model:
    backend: openai
    model: gpt-4o-mini
  template: correctness

environments:
  - name: leg_counting
    adapter: reasoning_gym
    # Inherits eval-level judge

  - name: creative_writing
    adapter: huggingface
    judge:                            # Per-env override
      model:
        backend: anthropic
        model: claude-sonnet-4-20250514
      template: helpfulness
```

### Multiple Judges

```yaml
environments:
  - name: safety_eval
    adapter: huggingface
    judge:
      - model: {backend: openai, model: gpt-4o}
        template: correctness
        name: judge_correctness
        weight: 0.5
      - model: {backend: openai, model: gpt-4o}
        template: safety
        name: judge_safety
        weight: 0.5
```

### JudgeConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | ModelConfig | required | Judge LLM backend |
| `template` | str | `"correctness"` | Built-in name or literal template |
| `system_prompt` | str \| None | None | Override template's system prompt |
| `score_range` | [float, float] | [1, 10] | Scoring scale for normalization |
| `name` | str | `"judge"` | Signal name |
| `reward_type` | str | `"outcome"` | outcome, step, format, process |
| `weight` | float | 1.0 | Signal weight |
| `normalize` | bool | true | Normalize to [0,1] |
| `inference` | InferenceConfig \| None | None | Judge sampling params |

## Context Gathering

`JudgeReward` automatically extracts context from the environment state:

| Field | Source |
|-------|--------|
| `{question}` | `state.observation.prompt` |
| `{response}` | `action.text` |
| `{ground_truth}` | Duck-typed from `state.hidden` — tries `expected_answer`, `answer`, `ground_truth`, `target` attributes in order |

This works across all adapters without adapter-specific code.

## Feedback

`JudgeReward` sets the `feedback` field on every signal it produces, containing the full judge LLM response text. This is used by `IterativeEnvironment` to show the judge's feedback to the agent on each refinement turn:

```python
signal = judge.compute(state, action, next_state)
print(signal.reward)       # Normalized numeric score
print(signal.feedback)    # Full judge response text (always set)
```

In iterative environments, `SignalBundle.feedback_texts()` collects all non-None feedback strings from signals to build the agent's next observation.

## Error Handling

On any failure (backend error, unparseable score), `JudgeReward` logs a warning and returns a `Signal` with `default_score` and error metadata. The `feedback` field is set to the error message when available:

```python
signal = judge.compute(state, action, next_state)
if signal.metadata and "error" in signal.metadata:
    print(f"Judge failed: {signal.metadata['error']}")
```

## Metadata

Successful judge signals include rich metadata:

```python
signal.metadata["raw_score"]          # Numeric score before normalization
signal.metadata["judge_response"]     # Full judge LLM output
signal.metadata["prompt_tokens"]      # Judge prompt token count
signal.metadata["completion_tokens"]  # Judge completion token count
```
