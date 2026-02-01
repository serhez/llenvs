# HuggingFace Adapter

Access thousands of datasets on the HuggingFace Hub, including AIME, GSM8K, MATH, and more.

## Installation

```bash
pip install datasets huggingface-hub
# or
pip install llenvs[huggingface]
```

## Quick Start with Presets

```python
from llenvs.adapters import create_huggingface_environment

# AIME 2024 - Competition math (30 problems)
env = create_huggingface_environment("HuggingFaceH4/aime_2024")

# GSM8K - Grade school math (~1300 test problems)
env = create_huggingface_environment("gsm8k")

# Historical AIME (1983-2024)
env = create_huggingface_environment("di-zhang-fdu/AIME_1983_2024")
```

## Custom Configuration

```python
from llenvs.adapters import HuggingFaceAdapter

adapter = HuggingFaceAdapter()

env = adapter.get_environment(
    name="gsm8k",
    subset="main",              # Dataset subset/config
    split="test",               # train, test, validation
    question_column="question", # Column with questions
    answer_column="answer",     # Column with answers
    answer_extraction="numeric",# How to extract final answer
    scoring="numeric",          # How to score answers
    size=100,                   # Limit to N examples
    seed=42,                    # Shuffle seed
)
```

## Answer Extraction Methods

| Method | Description | Example |
|--------|-------------|---------|
| `"boxed"` | LaTeX `\boxed{...}` | `\boxed{42}` → `"42"` |
| `"numeric"` | Last number in text | `"answer is 42"` → `"42"` |
| `"last_line"` | Last non-empty line | Multi-line → last |
| `"direct"` | Use column directly | No extraction |

Custom extraction:

```python
def my_extractor(text: str) -> str | None:
    import re
    match = re.search(r"ANSWER: (\d+)", text)
    return match.group(1) if match else None

env = adapter.get_environment(..., answer_extraction=my_extractor)
```

## Scoring Options

| Scoring | Description |
|---------|-------------|
| `"exact"` | Case-insensitive string match |
| `"numeric"` | Numeric equivalence (42 == 42.0) |
| `"numeric_tolerance"` | Numeric with relative tolerance |

Custom scoring:

```python
def my_scorer(predicted: str, expected: str) -> float:
    return 1.0 if predicted.lower() == expected.lower() else 0.0

env = adapter.get_environment(..., scoring=my_scorer)
```

## Dataset Presets

Common datasets have preconfigured settings:

```python
from llenvs.adapters.huggingface import DATASET_PRESETS

print(DATASET_PRESETS.keys())
# ['HuggingFaceH4/aime_2024', 'gsm8k', 'di-zhang-fdu/AIME_1983_2024', ...]
```

## Example: AIME Evaluation

```python
from llenvs.adapters import create_huggingface_environment
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import SamplingParams
from llenvs.core import TextAction

env = create_huggingface_environment("HuggingFaceH4/aime_2024")
backend = OpenAIBackend(model="gpt-4o")
params = SamplingParams(temperature=0.0, max_tokens=4096)

correct = 0
for i in range(len(env)):
    state, _ = env.reset(options={"task_index": i})

    prompt = f"{state.observation.prompt}\n\nPut your answer in <answer>...</answer>."
    result = backend.generate_single(prompt, params)

    action = TextAction(text=result.text)
    step_result = env.step(state, action)

    if step_result.rewards.by_name("correctness").value == 1.0:
        correct += 1

    print(f"Problem {i+1}: {'✓' if step_result.rewards.by_name('correctness').value == 1.0 else '✗'}")

print(f"\nAccuracy: {correct}/{len(env)} = {correct/len(env):.1%}")
```

## Hidden State

```python
@dataclass(frozen=True)
class HuggingFaceHidden:
    row: dict[str, Any]        # Original dataset row
    expected_answer: str       # Extracted expected answer
    task_index: int
    dataset_name: str
```
