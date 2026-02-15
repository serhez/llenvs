# Iterative Refinement Environments

Iterative environments turn any single-turn environment into a multi-turn refinement loop. The agent submits a solution, receives feedback from evaluation signals, and can revise — repeating until solved, max turns reached, or the agent submits early.

## Quick Start

```python
from llenvs.adapters.iterative import IterativeEnvironment, IterativeTask
from llenvs.core.state import Action

# Create tasks directly
tasks = (
    IterativeTask(prompt="Write a function that adds two numbers.", ground_truth="def add(a, b): return a + b"),
    IterativeTask(prompt="Write a function that reverses a string.", ground_truth="def reverse(s): return s[::-1]"),
)

env = IterativeEnvironment(tasks=tasks, max_turns=3)

# Run an episode
state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)  # Task + "You have 3 turn(s) to solve this task."

result = env.step(state, Action(text="def add(a, b): return a - b"))
print(result.next_state.observation.prompt)  # Feedback + turns remaining
```

## Ready-to-Use Coding Environments

`IterativeCodingEnvironment` provides one-line factories for iterative coding with code execution feedback:

```python
from llenvs.environments import IterativeCodingEnvironment

# HumanEval with 5 refinement turns
env = IterativeCodingEnvironment.from_humaneval(max_turns=5)
state, info = env.reset(options={"task_index": 0})

# MBPP with 3 turns
env = IterativeCodingEnvironment.from_mbpp(max_turns=3)

# Any HF coding dataset
env = IterativeCodingEnvironment.create(
    dataset="bigcode/humanevalpack",
    language="python",
    max_turns=5,
    question_column="prompt",
    answer_column="canonical_solution",
    test_column="test",
)
```

Each factory:
1. Creates an `HuggingFaceEnvironment` with the dataset
2. Creates a `SubprocessCodeExecutor` for running tests
3. Creates a `CodeExecutionReward` for test feedback
4. Wraps in `IterativeEnvironment`

### Adding LLM Judge Feedback

```python
from llenvs.inference.backends.api import OpenAIBackend

judge_backend = OpenAIBackend(model="gpt-4o-mini")

env = IterativeCodingEnvironment.from_humaneval(
    max_turns=5,
    judge_backend=judge_backend,  # Adds JudgeReward with "iterative_feedback" template
)
```

## Architecture

`IterativeEnvironment` wraps a single-turn inner environment (or standalone tasks). On each `step()`:

1. Checks for early submit keyword at start of response
2. Extracts the submission via `submission_extractor`
3. Evaluates via inner environment rewards (re-evaluates from initial state each turn)
4. Computes extra rewards (code execution, judge, etc.)
5. Collects feedback from `Signal.feedback` fields
6. Determines termination (solved, early submit, or max turns)
7. Builds next observation with feedback and optional history

### Termination Conditions

| Condition | Trigger | Result |
|-----------|---------|--------|
| **Solved** | Best OUTCOME signal reward >= `solved_threshold` | `terminated=True` |
| **Early submit** | Response starts with `submit_keyword` | `terminated=True` |
| **Max turns** | `turn >= max_turns` | `truncated=True` |

### Signal Feedback

`Signal` carries both numeric rewards and textual feedback:

```python
Signal(
    name="code_execution",
    reward_type=RewardType.OUTCOME,
    reward=0.6,                      # Numeric reward
    feedback="3/5 tests passed.\nFailed: test_edge_case",  # Shown to agent
)
```

The iterative environment collects all non-None `feedback` strings via `SignalBundle.feedback_texts()` and formats them into the observation.

## Wrapping an Existing Environment

Any single-turn environment can become iterative:

```python
from llenvs.adapters.iterative import IterativeEnvironment
from llenvs.core.judge import JudgeReward

# Get a single-turn environment
from llenvs.core.config import EnvironmentFactory, EnvironmentConfig

inner_env = EnvironmentFactory.create(
    EnvironmentConfig(name="polynomial_equations", adapter="reasoning_gym", size=50)
)

# Add a judge for feedback
judge = JudgeReward(backend=judge_backend, template="iterative_feedback")

# Wrap it
env = IterativeEnvironment(
    inner=inner_env,
    max_turns=5,
    extra_rewards=(judge,),
)
```

The inner environment must have `pure_step=True` (most single-turn environments do). The iterative wrapper re-evaluates submissions against the stored initial state on every turn.

## Submission Extraction

By default, the full response is used as the submission (`RawGenerationExtractor`). Use `submission_extractor` to parse responses:

```python
from llenvs.core.extraction import CodeBlockExtractor

env = IterativeEnvironment(
    inner=inner_env,
    submission_extractor=CodeBlockExtractor(language="python"),
    max_turns=5,
)
```

If extraction returns `None`, the raw response is used as fallback.

## Early Submit

Agents can finalize early by starting their response with the submit keyword:

```python
env = IterativeEnvironment(tasks=tasks, submit_keyword="SUBMIT")  # Default

# Agent response: "SUBMIT my final answer is 42"
# → terminated=True, submission="SUBMIT my final answer is 42"
```

Disable with `submit_keyword=None`.

## History

By default, previous attempts and feedback are included in the observation after the first turn:

```
## Feedback on your previous submission (Turn 2/5)

3/5 tests passed.

## Previous attempts

### Turn 1
**Submission:**
def add(a, b): return a - b

**Feedback:**
0/5 tests passed.

You have 3 turn(s) remaining.
```

Control with:
- `include_history=False` — only show current turn feedback
- `history_summarizer` — custom callable to format history

### Custom History Summarizer

```python
def my_summarizer(
    submissions: tuple[str, ...],
    feedback_history: tuple[tuple[str, ...], ...],
) -> str:
    return f"You have made {len(submissions)} attempt(s) so far."

env = IterativeEnvironment(
    tasks=tasks,
    history_summarizer=my_summarizer,
)
```

## Custom Prompts

Override any prompt template:

```python
env = IterativeEnvironment(
    tasks=tasks,
    prompts={
        "initial": "{task}\n\nYou have {turns_remaining} tries. Say DONE when finished.",
        "feedback": "Turn {turn}/{max_turns}: {feedback}\n{history_section}{turns_remaining} left.",
    },
)
```

Available templates: `"initial"`, `"feedback"`, `"history_entry"`.

## Code Execution

`CodeExecutionReward` runs code and tests in a subprocess, returning both a score and feedback:

```python
from llenvs.core.code_execution import CodeExecutionReward, SubprocessCodeExecutor
from llenvs.core.extraction import CodeBlockExtractor

executor = SubprocessCodeExecutor()
code_reward = CodeExecutionReward(
    executor=executor,
    code_extractor=CodeBlockExtractor(language="python"),
)

env = IterativeEnvironment(
    inner=inner_env,
    extra_rewards=(code_reward,),
)
```

For sandboxed execution, wrap the entire environment in a `ContainerEnvironment`:

```yaml
environments:
  - name: openai/openai_humaneval
    adapter: huggingface
    container:
      runtime: docker
      image: python:3.12-slim
    iterative:
      max_turns: 5
      code_execution:
        timeout: 30
```

## YAML Configuration

```yaml
environments:
  - name: openai/openai_humaneval
    adapter: huggingface
    params:
      split: test
      question_column: prompt
      answer_column: canonical_solution
    iterative:
      max_turns: 5
      submission_extractor: code_block
      submission_extractor_config:
        language: python
      code_execution:
        timeout: 30
      solved_threshold: 1.0
      submit_keyword: SUBMIT
    judge:
      model:
        backend: openai
        model: gpt-4o-mini
      template: iterative_feedback
```

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turns` | int | 3 | Maximum refinement turns |
| `include_history` | bool | true | Include previous attempts in feedback |
| `summarize_history` | bool | false | Use env LLM to summarize history |
| `submit_keyword` | str \| null | "SUBMIT" | Keyword for early termination |
| `submission_extractor` | str \| null | null | Extractor name for parsing submissions |
| `submission_extractor_config` | dict | {} | Config for the submission extractor |
| `solved_threshold` | float | 1.0 | OUTCOME score threshold for solved |
| `code_execution` | object \| null | null | Code execution config |
| `code_execution.timeout` | float | 30.0 | Execution timeout in seconds |

## Composability

| Feature | Works? | How |
|---------|--------|-----|
| `extra_rewards` | Yes | Passed through, computed on iterative state |
| `JudgeReward` | Yes | `IterativeHidden.ground_truth` enables context gathering |
| `FormatReward` | Yes | Via `extra_rewards` |
| `ContainerEnvironment` | Yes | Wrap inner env in container for sandboxed execution |
| `TrajectoryRunner` | Yes | Multi-turn loop via `observation.messages` |
| Batch evaluation | Yes | Standard runner batching |
| `CodeExecutionReward` | Yes | Via `extra_rewards`, provides score + feedback |

## Programmatic Construction

For full control, construct `IterativeEnvironment` directly:

```python
from llenvs.adapters.iterative import IterativeEnvironment, IterativeTask

env = IterativeEnvironment(
    tasks=(
        IterativeTask(prompt="Task 1", ground_truth="answer1"),
        IterativeTask(prompt="Task 2", ground_truth="answer2"),
    ),
    max_turns=5,
    include_history=True,
    submit_keyword="SUBMIT",
    solved_threshold=1.0,
    extra_rewards=(my_reward,),
)
```

Or wrap an existing environment:

```python
env = IterativeEnvironment(
    inner=existing_env,      # Any single-turn environment
    max_turns=5,
    submission_extractor=my_extractor,
    extra_rewards=(code_reward, judge_reward),
)
```
