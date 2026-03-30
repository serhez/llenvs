# History Control

Multi-turn environments accumulate conversation history that can grow large over many steps. History control lets you manage what prior turns the model sees, reducing token usage and shaping agent behavior.

History control works with **structured observations** and with **plain text-only legacy chat histories**. Environments that set `task` and `state` use structured mode directly. Environments that expose only `prompt + messages` still get history shaping when their legacy history is an ordinary assistant/user chat transcript. Tool-based environments continue to use legacy message building without history shaping so tool call structures are preserved exactly.

## How It Works

In structured message building mode, the model's context is assembled as:

1. **System prompt** (if any)
2. **Task description** — static, set once at reset
3. **History** — prior (action, observation) pairs, controlled by `history_fn`
4. **Current state** — the latest observation. On step 0, this is omitted when it is identical to the task description, so the initial prompt is not duplicated.

The `history_fn` parameter controls step 3. It receives a list of `HistoryEntry` objects and returns `ChatMessage` objects for the history portion.

## Built-in History Functions

### `full_history` (default)

Includes all prior turns. This is the default behavior when no `history_fn` is specified.

```python
from llenvs.evaluation import run_evaluation, full_history

result = run_evaluation(
    environment=env,
    backend=backend,
    history_fn=full_history,  # same as omitting it
)
```

### `no_history`

Drops all prior turns. The model sees only the task description and current state, except for the step-0 dedupe case where an identical initial state is omitted. Useful for environments where the current observation is self-contained.

```python
from llenvs.evaluation import run_evaluation, no_history

result = run_evaluation(
    environment=env,
    backend=backend,
    history_fn=no_history,
)
```

### `last_n_history(n)`

Keeps only the most recent *n* turns. A sliding window over the conversation.

```python
from llenvs.evaluation import run_evaluation, last_n_history

result = run_evaluation(
    environment=env,
    backend=backend,
    history_fn=last_n_history(5),  # keep last 5 turns
)
```

### `sliding_window_history(max_tokens, token_counter)`

Fits as many recent turns as possible within a token budget. Takes a `token_counter` callable that returns the token count for a string.

```python
from llenvs.evaluation import run_evaluation, sliding_window_history

# Simple character-based counter
result = run_evaluation(
    environment=env,
    backend=backend,
    history_fn=sliding_window_history(
        max_tokens=4096,
        token_counter=lambda text: len(text) // 4,  # rough estimate
    ),
)
```

For accurate token counting, use a tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

result = run_evaluation(
    environment=env,
    backend=backend,
    history_fn=sliding_window_history(
        max_tokens=4096,
        token_counter=lambda text: len(tokenizer.encode(text)),
    ),
)
```

## Custom History Functions

A history function has the signature:

```python
def my_history_fn(entries: list[HistoryEntry]) -> list[ChatMessage]:
    ...
```

Each `HistoryEntry` has:
- `action_text`: The model's action (full or extracted, depending on `include_reasoning_in_history`)
- `observation_text`: The state observation after the action
- `observation_images`: Any images associated with the observation
- `step`: The step number

Example — summarize old turns:

```python
from llenvs.evaluation.history import HistoryEntry
from llenvs.inference.protocol import ChatMessage

def summarize_old_keep_recent(entries: list[HistoryEntry]) -> list[ChatMessage]:
    if len(entries) <= 3:
        # Few enough to keep all
        messages = []
        for e in entries:
            messages.append(ChatMessage(role="assistant", content=e.action_text))
            if e.observation_text:
                messages.append(ChatMessage(role="user", content=e.observation_text))
        return messages

    # Summarize old entries, keep last 3
    old = entries[:-3]
    summary = "Previous actions: " + ", ".join(e.action_text for e in old)
    messages = [ChatMessage(role="user", content=summary)]

    for e in entries[-3:]:
        messages.append(ChatMessage(role="assistant", content=e.action_text))
        if e.observation_text:
            messages.append(ChatMessage(role="user", content=e.observation_text))

    return messages
```

## Reasoning Stripping

By default (`include_reasoning_in_history=False`), prior actions in the history show the **extracted action** rather than the full model response. This strips chain-of-thought reasoning from the history, reducing token usage and preventing models from copying verbose patterns.

```python
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(),
    include_reasoning_in_history=True,  # keep full reasoning in history
)
```

When `False`, the runner looks for `extracted_action` or `extracted_answer` in the transition's step info. If not found, falls back to the full action text. Gymnasium and craftax adapters store this automatically.

## Runner Integration

Both `TrajectoryRunner` and `run_evaluation` accept `history_fn` and `include_reasoning_in_history`:

```python
from llenvs.evaluation import TrajectoryRunner, run_evaluation, last_n_history

# Via TrajectoryRunner
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(),
    history_fn=last_n_history(10),
    include_reasoning_in_history=False,
)

# Via convenience function
result = run_evaluation(
    environment=env,
    backend=backend,
    history_fn=last_n_history(10),
    include_reasoning_in_history=False,
)
```

History control takes effect in structured message building mode and in plain text-only legacy chat mode. It does not rewrite legacy tool-call conversations.

## Current Observation Truncation

`PromptBudget` supports an optional `min_current_observation_chars` field. When set, the runner truncates the current state observation as a last resort if the prompt still exceeds the budget after history truncation. The truncation uses middle-truncation (keeping the beginning and end of the text with a `[... N characters omitted ...]` marker).

The truncation order is:

1. **History** — truncated via the `build_history` callback
2. **Current observation** — truncated only if `min_current_observation_chars` is set and the prompt still exceeds `max_prompt_tokens` after step 1. Truncated down to the floor but no further. If the prompt is still too long, the backend (e.g., vLLM) handles the final validation.

This applies to both structured mode and plain text-only legacy chat mode. In legacy mode, the final user message is treated as the current observation.

```python
budget = PromptBudget(
    max_prompt_tokens=4096,
    estimate_tokens=my_estimator,
    build_history=my_history_builder,
    min_current_observation_chars=1024,  # last-resort truncation floor
)
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    prompt_budget=budget,
)
```

Turn/step counters are injected separately via `TurnInfoConfig` on the runner. History entries are not modified by turn info — only the task description and current state observation are affected. See the [Evaluation guide](evaluation.md#turn-info) for details.
