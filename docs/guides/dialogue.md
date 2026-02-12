# Dialogue Environments

Dialogue environments use an LLM as part of the environment's transition function. Instead of rule-based backends, `step()` calls a `ModelBackend` directly to generate observations. This enables scenarios like:

- **20-questions**: Agent asks yes/no questions, LLM oracle answers
- **Student-teacher**: Agent produces answer, LLM teacher gives feedback, agent revises
- **Debate/negotiation**: Agent and LLM counterpart take turns

## Quick Start

```python
from llenvs.adapters.dialogue import DialogueAdapter
from llenvs.inference.backends.api import OpenAIBackend

# Create the environment LLM backend
env_backend = OpenAIBackend(model="gpt-4o-mini")

# Create a 20-questions environment
adapter = DialogueAdapter()
env = adapter.get_environment(
    "twenty_questions",
    env_llm=env_backend,
    words=["cat", "dog", "elephant"],
)

# Run an episode
state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)  # "I'm thinking of something..."

from llenvs.core.state import Action

result = env.step(state, Action.from_text("Is it an animal?"))
print(result.next_state.observation.messages[-1]["content"])  # "Yes."
```

## Architecture

`DialogueEnvironment` implements the standard `Environment` protocol. Each `step()`:

1. Builds a system prompt with task-specific context
2. Reconstructs the conversation from observation message history
3. Calls the env LLM backend via `generate_chat()`
4. Returns the LLM response as the next observation

This means it composes with all existing features:

| Feature | Works? | How |
|---------|--------|-----|
| `extra_rewards` | Yes | Standard `reward_functions` returns `extra_rewards` |
| `JudgeReward` | Yes | `DialogueHidden.ground_truth` enables context gathering |
| `FormatReward` | Yes | Via `extra_rewards` |
| `SegmentedEnvironment` | Yes | Wraps any `Environment` |
| `TrajectoryRunner` | Yes | No runner changes needed |
| Batch evaluation | Yes | Runner batches agent generation; env LLM calls are per-episode |

## Tasks

Each task is a `DialogueTask` with:

```python
from llenvs.adapters.dialogue import DialogueTask

task = DialogueTask(
    prompt="I'm thinking of something. Ask yes/no questions.",  # Initial observation
    context="The secret word is: cat.",  # Injected into env LLM system prompt
    ground_truth="cat",  # For reward computation
    metadata={"difficulty": "easy"},  # Arbitrary metadata
)
```

- **prompt**: What the agent sees as the initial observation
- **context**: Private information for the env LLM (e.g., the secret word)
- **ground_truth**: Accessible via `state.hidden.ground_truth` for reward functions

## Built-in Presets

### twenty_questions

Agent asks yes/no questions to guess a secret word. Terminates when the env LLM responds with "Correct!".

```python
env = adapter.get_environment(
    "twenty_questions",
    env_llm=backend,
    words=["cat", "dog", "piano"],  # One task per word
    max_steps=20,  # Default
)
```

### teacher

Agent solves a problem, env LLM provides feedback. No automatic termination — runs until max_steps.

```python
env = adapter.get_environment(
    "teacher",
    env_llm=backend,
    questions=[
        {"question": "What is the integral of x^2?", "answer": "x^3/3 + C"},
        {"question": "Solve: 2x + 3 = 7", "answer": "x = 2"},
    ],
    max_steps=5,  # Default
)
```

### Custom Tasks

Provide tasks directly for full control:

```python
env = adapter.get_environment(
    "twenty_questions",  # Or any preset name
    env_llm=backend,
    tasks=[
        {"prompt": "Start!", "context": "Secret: cat", "ground_truth": "cat"},
        {"prompt": "Start!", "context": "Secret: dog", "ground_truth": "dog"},
    ],
)
```

## YAML Configuration

```yaml
environments:
  - name: twenty_questions
    adapter: dialogue
    params:
      words: [cat, dog, elephant, piano, mountain]
    env_llm:
      model:
        backend: openai
        model: gpt-4o-mini
      inference:
        temperature: 0.0
        max_tokens: 256

  - name: teacher
    adapter: dialogue
    params:
      questions:
        - question: "What is the integral of x^2?"
          answer: "x^3/3 + C"
    env_llm:
      model:
        backend: anthropic
        model: claude-sonnet-4-20250514
    judge:
      model:
        backend: openai
        model: gpt-4o
      template: correctness
```

The `env_llm` field creates a `ModelBackend` for the environment's internal LLM. See the [Config reference](../reference/config.md) for `EnvironmentLLMConfig` details.

## Custom System Prompts

Override the preset system prompt:

```python
env = adapter.get_environment(
    "twenty_questions",
    env_llm=backend,
    words=["cat"],
    system_prompt="You are a game host. {context}\nOnly answer Yes or No.",
)
```

The `{context}` placeholder is replaced with the task's `context` field.

## Composing with JudgeReward

`DialogueHidden` exposes a `ground_truth` property, so `JudgeReward` works out of the box:

```python
from llenvs.core.judge import JudgeReward

judge = JudgeReward(
    backend=judge_backend,
    template="correctness",
)

env = adapter.get_environment(
    "twenty_questions",
    env_llm=env_backend,
    words=["cat"],
    extra_rewards=(judge,),
)
```

## Custom Sampling Parameters

```python
from llenvs.inference.protocol import SamplingParams

env = adapter.get_environment(
    "twenty_questions",
    env_llm=backend,
    words=["cat"],
    sampling_params=SamplingParams(temperature=0.3, max_tokens=128),
)
```

## Programmatic Construction

For full control, construct `DialogueEnvironment` directly:

```python
from llenvs.adapters.dialogue import DialogueEnvironment, DialogueTask

env = DialogueEnvironment(
    backend=backend,
    tasks=(
        DialogueTask(prompt="Guess!", context="Word: cat", ground_truth="cat"),
        DialogueTask(prompt="Guess!", context="Word: dog", ground_truth="dog"),
    ),
    system_prompt="You are playing 20 Questions. {context}",
    max_steps=20,
    is_terminal=lambda resp, action, step: "correct" in resp.lower(),
)
```
