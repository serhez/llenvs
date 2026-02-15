# Segmentation

The segmentation system turns single-step environments into multi-step environments by breaking responses into segments. This enables:

- **Per-step rewards** with Process Reward Models (PRMs)
- **Tree search** at segment boundaries
- **Early stopping** based on intermediate quality
- **Reasoning trace analysis**

## Basic Usage (Replay Mode)

Segment a complete response and analyze per-step:

```python
from llenvs.core.registry import environment_registry
from llenvs.core import SegmentedEnvironment, SentenceSegmenter

# Create single-step environment
base_env = environment_registry.get(name="leg_counting", adapter="reasoning_gym", size=100, seed=42)

# Wrap with segmentation
env = SegmentedEnvironment(base_env, SentenceSegmenter())

# Reset (same as base env)
state, _ = env.reset(options={"task_index": 0})

# Model generates full response
full_response = """First, count the animals. There are 2 dogs.
Dogs have 4 legs each. So 2 × 4 = 8. <answer>8</answer>"""

# Replay to get per-step results
results = env.replay(state, full_response)

for i, result in enumerate(results):
    print(f"Step {i}: terminal={result.terminated}, reward={result.rewards.total}")
# Step 0: terminal=False, reward=0
# Step 1: terminal=False, reward=0
# Step 2: terminal=False, reward=0
# Step 3: terminal=True, reward=2.0
```

## Generation-Time Segmentation

`SegmentedTrajectoryRunner` generates one segment at a time, calling `env.step()` after each segment. This enables per-step rewards, intervention, and branching during live generation — unlike replay mode which analyzes a complete response after the fact.

### Basic usage with TokenSegmenter

```python
from llenvs.core import SegmentedEnvironment, TokenSegmenter
from llenvs.evaluation import SegmentedTrajectoryRunner, run_segmented_evaluation
from llenvs.inference.protocol import SamplingParams

base_env = environment_registry.get(name="leg_counting", adapter="reasoning_gym", size=100, seed=42)

tokenizer = ...  # Any tokenizer with encode/decode
segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=64)
env = SegmentedEnvironment(base_env, segmenter)

# Run a single trajectory
runner = SegmentedTrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(max_tokens=2048),
)
result = runner.run_trajectory(task_index=0)
print(f"Steps: {len(result.trajectory)}, Success: {result.success}")
```

### With SentenceSegmenter

For text-pattern segmenters, the runner generates text in chunks and splits at segment boundaries:

```python
from llenvs.core import SentenceSegmenter

env = SegmentedEnvironment(base_env, SentenceSegmenter())

result = run_segmented_evaluation(
    environment=env,
    backend=backend,
    num_tasks=10,
)
print(f"Accuracy: {result.success_rate:.1%}")
```

### Batched segmented evaluation

`run_batch()` on `SegmentedTrajectoryRunner` uses lockstep batched generation. All active trajectories generate one segment together via `generate_segment_batch()`, then step their environments. Trajectories that finish generation early drop out. Callbacks (`step_callback`) run per-trajectory after each step.

```python
from llenvs.evaluation import LogConfig

result = run_segmented_evaluation(
    environment=env,
    backend=backend,
    num_tasks=100,
    step_callback=my_callback,  # Runs per-trajectory
    progress_callback=lambda c, t: print(f"\r{c}/{t}", end=""),
    log=LogConfig(targets=("console", "file")),  # Structured logging
)
```

### Observation injection with step_callback

Inject feedback between segments to create a multi-turn conversation:

```python
def reward_feedback(step_result):
    """Return feedback string or None to continue without feedback."""
    reward = step_result.rewards.total
    if reward < 0.5:
        return f"Score: {reward:.2f}. Please reconsider your approach."
    return None  # No feedback, continue single assistant turn

result = runner.run_trajectory(
    task_index=0,
    step_callback=reward_feedback,
)
```

When `step_callback` returns a string, the conversation becomes multi-turn:

```
[system, user(question)]
→ assistant: "Step 1: count legs..."     (segments 1-3)
→ user: "Score: 0.3, reconsider"         (feedback from callback)
→ assistant: "Actually, let me..."       (segments 4-N)
```

When callback returns `None` (or no callback is provided), generation continues as a single assistant turn split into segments.

### Early exit with `COMPLETE`

For probing or intervention studies, you can step segment-by-segment until a point of interest, then finish the rest cheaply in one LLM call — all within `run_trajectory()`:

```python
from llenvs.evaluation import SegmentedTrajectoryRunner, COMPLETE

runner = SegmentedTrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(max_tokens=2048),
)

def probe_callback(step_result):
    """Step 5 segments, then finish in one shot."""
    if step_result.info.get("segment_index", 0) >= 5:
        return COMPLETE  # finish remainder in a single LLM call
    return None

result = runner.run_trajectory(task_index=0, step_callback=probe_callback)
# result.trajectory has all transitions (probed + remainder)
# result.success reflects final correctness
```

The callback can return:
- `None` — continue generating segment-by-segment (same assistant turn)
- A feedback string — inject as a user message (new assistant turn)
- `COMPLETE` — stop segment-by-segment generation, generate the rest in one LLM call, segment it, replay through the environment, and finalize
- `ForceAction(text)` — use the given text as the next segment instead of calling the backend (see [Forcing actions](#forcing-actions))

If the episode ends naturally before the callback returns `COMPLETE`, the behavior is identical to a normal `run_trajectory()` call.

### Prefix replay

Replay predetermined content through the environment before generation starts. Useful for resuming from a partial trace or studying continuations from a specific prefix.

The `prefix` parameter on `run_trajectory()` accepts two forms:

**Text form** — raw text, auto-segmented and stepped from the reset state:

```python
result = runner.run_trajectory(
    task_index=0,
    prefix="Step 1: count animals.\nStep 2: 2 dogs.",
)
# Prefix is segmented via env.segmenter, stepped through env,
# then generation continues with the prefix as assistant prefill.
```

**Structured form** — state-action pairs from a prior trajectory, stepped using the provided states:

```python
# Run an initial trajectory
old_result = runner.run_trajectory(task_index=0)

# Extract the first 5 state-action pairs
prefix_pairs = [
    (t.state, t.action)
    for t in old_result.trajectory.transitions[:5]
]

# Resume from that prefix
result = runner.run_trajectory(task_index=0, prefix=prefix_pairs)
```

Both forms:
- Reset the environment normally, then step each prefix action through it
- Record transitions with `info["replayed"] = True`
- Do **not** invoke `step_callback` during replay
- Do **not** count prefix steps toward `max_steps`
- Set `result.metadata["prefix_steps"]` to the number of replayed steps

The text form derives state from each step for the next step. The structured form uses the **provided** state for each `env.step()` call, allowing exact replay of non-deterministic trajectories.

Prefix + one-shot completion:

```python
result = runner.run_trajectory(
    task_index=0,
    prefix="My partial reasoning trace...",
    step_callback=lambda _: COMPLETE,
)
```

### Forcing actions

Override generation at specific steps with predetermined text using `ForceAction`:

```python
from llenvs.evaluation import ForceAction

def my_callback(step_result):
    if step_result.next_state.hidden.segment_index == 3:
        return ForceAction("Therefore, x = 42.")
    return None

result = runner.run_trajectory(task_index=0, step_callback=my_callback)
```

When `ForceAction` is returned, the runner:
- Uses the provided text as the next segment (no backend call)
- Clears the buffer (stale after context change)
- Records the transition with `info["forced"] = True`
- Includes the forced text in the accumulated context for subsequent generation

`ForceAction` composes with other callback returns — you can alternate between forcing actions, injecting feedback, and continuing normally:

```python
def complex_callback(step_result):
    idx = step_result.next_state.hidden.segment_index
    if idx == 2:
        return ForceAction("Injected reasoning step.")
    if idx == 5:
        return COMPLETE
    return None
```

### Continuation strategies

The runner auto-selects the appropriate strategy based on segmenter type:

- **`TokenContinuationStrategy`** — For `TokenSegmenter`: sets `max_tokens = token_size`, each generation call produces exactly one segment.
- **`BoundaryContinuationStrategy`** — For `SentenceSegmenter`, `LineSegmenter`, `PatternSegmenter`: generates text in chunks, uses `find_boundary()` to detect segment boundaries, buffers overflow.

### Manual generation-time control

For custom generation loops without the runner:

```python
from llenvs.core import Action

env = SegmentedEnvironment(base_env, SentenceSegmenter())
state, _ = env.reset(options={"task_index": 0})

accumulated = ""
while not state.metadata.is_terminal:
    # Generate until next segment boundary
    partial = model.generate_until(
        state.observation.prompt + accumulated,
        stop_at=env.segmenter.find_boundary,
    )

    result = env.step(state, Action(text=partial))
    accumulated += partial
    state = result.next_state

    # Can intervene here: branch, stop early, adjust generation
    if some_quality_check(result):
        break

# Finalize when done generating
result = env.finalize(state)
print(f"Final reward: {result.rewards.total}")
```

## Segmenters

### SentenceSegmenter

Splits on sentence boundaries (.!? + whitespace). Handles common abbreviations:

```python
from llenvs.core import SentenceSegmenter

segmenter = SentenceSegmenter()
segments = segmenter.segment("First step. Second step! Third?")
# ["First step.", "Second step!", "Third?"]
```

### LineSegmenter

Splits on newlines:

```python
from llenvs.core import LineSegmenter

# Single newlines
segmenter = LineSegmenter(delimiter="\n")

# Paragraphs (double newlines)
segmenter = LineSegmenter(delimiter="\n\n")
```

### PatternSegmenter

Splits on regex patterns:

```python
from llenvs.core import PatternSegmenter

# Default patterns: numbered steps, transition words
segmenter = PatternSegmenter()
# Matches: "1.", "Step 1:", "Therefore,", "First,", etc.

# Custom patterns
segmenter = PatternSegmenter(patterns=(
    r"(?:^|\s)\d+[.:]\s",             # Numbered steps
    r"(?:^|\s)(?:So|Thus|Hence),?\s",  # Conclusions
))
```

### TokenSegmenter

Splits text into fixed-size token chunks. Accepts any tokenizer with `encode(str) -> list[int]` and `decode(list[int]) -> str` methods:

```python
from llenvs.core import TokenSegmenter

# With HuggingFace tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=64)

# With tiktoken
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-4")
segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=64)
```

Segments concatenate to exactly reconstruct the original text — no characters are lost or added:

```python
segments = segmenter.segment(response)
assert "".join(segments) == response
```

### LLMSegmenter

Uses a `ModelBackend` to semantically segment text into meaningful reasoning steps. The LLM returns a JSON array of segment strings, which are mapped back to exact positions in the original text via greedy substring matching.

```python
from llenvs.core import LLMSegmenter

# Any ModelBackend works (API, vLLM, HuggingFace, etc.)
segmenter = LLMSegmenter(backend=backend)

segments = segmenter.segment("Let me think about this. The key insight is...")
# ["Let me think about this. ", "The key insight is..."]
```

Custom prompt template (must include `{raw_generation}`):

```python
segmenter = LLMSegmenter(
    backend=backend,
    prompt_template="Split into steps:\n{raw_generation}",
)
```

Custom parser — receives `(original_text, llm_response)` and returns segments:

```python
def my_parser(original: str, llm_response: str) -> list[str]:
    # Custom parsing logic
    return [original]  # fallback: single segment

segmenter = LLMSegmenter(backend=backend, parser=my_parser)
```

Limitations:
- `find_boundary()` raises `NotImplementedError` — LLM calls are too expensive for streaming. Use replay mode only.
- Segments always concatenate to exactly reconstruct the original text (greedy matching preserves original characters).

### CompositeSegmenter

Combine multiple segmenters:

```python
from llenvs.core import CompositeSegmenter

segmenter = CompositeSegmenter(segmenters=(
    LineSegmenter(),
    SentenceSegmenter(),
))
```

## With Process Reward Model

Add custom per-step rewards:

```python
from llenvs.core import Signal, RewardType

class PRMRewardFunction:
    """Process Reward Model for intermediate step quality."""

    def __init__(self, prm_model):
        self.prm = prm_model

    @property
    def name(self) -> str:
        return "prm"

    @property
    def reward_type(self) -> RewardType:
        return RewardType.PROCESS

    def compute(self, state, action, next_state) -> Signal:
        if not next_state.metadata.is_terminal:
            accumulated = state.hidden.accumulated_text + action.text
            score = self.prm.score(accumulated)
            return Signal(
                name=self.name,
                reward_type=self.reward_type,
                reward=score,
            )
        return Signal(name=self.name, reward_type=self.reward_type, reward=0.0)

# Use with SegmentedEnvironment
env = SegmentedEnvironment(
    base_env,
    SentenceSegmenter(),
    reward_functions=(PRMRewardFunction(prm), *base_env.reward_functions),
)
```

## SegmentedHidden

The extended hidden state tracks segmentation progress:

```python
@dataclass(frozen=True)
class SegmentedHidden(Generic[HiddenT]):
    base_hidden: HiddenT           # Original env's hidden state
    accumulated_text: str          # Text generated so far
    segment_index: int             # Current segment number
    segments: tuple[str, ...]      # All segments seen
    total_segments: int | None     # Known only in replay mode
```

## Example: AIME with Step Analysis

```python
from llenvs.core.registry import environment_registry
from llenvs.core import SegmentedEnvironment, PatternSegmenter

# Load AIME problems
base_env = environment_registry.get(name="HuggingFaceH4/aime_2024", adapter="huggingface")

# Wrap with step-based segmentation
env = SegmentedEnvironment(base_env, PatternSegmenter())

state, _ = env.reset(options={"task_index": 0})

response = """Step 1: Identify the given information.
Step 2: Set up the equation.
Step 3: Solve for x = 42.
<answer>42</answer>"""

results = env.replay(state, response)
print(f"Number of reasoning steps: {len(results)}")
```
