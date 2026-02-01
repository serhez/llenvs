# Multi-Step Segmentation

The segmentation system turns single-step environments into multi-step environments by breaking responses into segments. This enables:

- **Per-step rewards** with Process Reward Models (PRMs)
- **Tree search** at segment boundaries
- **Early stopping** based on intermediate quality
- **Reasoning trace analysis**

## Basic Usage (Replay Mode)

Segment a complete response and analyze per-step:

```python
from env_evals.adapters import create_reasoning_gym_environment
from env_evals.core import SegmentedEnvironment, SentenceSegmenter

# Create single-step environment
base_env = create_reasoning_gym_environment("leg_counting", size=100, seed=42)

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

## Generation-Time Control

Step through segments during generation for intervention:

```python
env = SegmentedEnvironment(base_env, SentenceSegmenter())
state, _ = env.reset(options={"task_index": 0})

accumulated = ""
while not state.metadata.is_terminal:
    # Generate until next segment boundary
    partial = model.generate_until(
        state.observation.prompt + accumulated,
        stop_at=env.segmenter.find_boundary,
    )

    result = env.step(state, TextAction(text=partial))
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
from env_evals.core import SentenceSegmenter

segmenter = SentenceSegmenter()
segments = segmenter.segment("First step. Second step! Third?")
# ["First step.", "Second step!", "Third?"]
```

### LineSegmenter

Splits on newlines:

```python
from env_evals.core import LineSegmenter

# Single newlines
segmenter = LineSegmenter(delimiter="\n")

# Paragraphs (double newlines)
segmenter = LineSegmenter(delimiter="\n\n")
```

### PatternSegmenter

Splits on regex patterns:

```python
from env_evals.core import PatternSegmenter

# Default patterns: numbered steps, transition words
segmenter = PatternSegmenter()
# Matches: "1.", "Step 1:", "Therefore,", "First,", etc.

# Custom patterns
segmenter = PatternSegmenter(patterns=(
    r"(?:^|\s)\d+[.:]\s",             # Numbered steps
    r"(?:^|\s)(?:So|Thus|Hence),?\s",  # Conclusions
))
```

### CompositeSegmenter

Combine multiple segmenters:

```python
from env_evals.core import CompositeSegmenter

segmenter = CompositeSegmenter(segmenters=(
    LineSegmenter(),
    SentenceSegmenter(),
))
```

## With Process Reward Model

Add custom per-step rewards:

```python
from env_evals.core import RewardSignal, RewardType

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

    def compute(self, state, action, next_state) -> RewardSignal:
        if not next_state.metadata.is_terminal:
            accumulated = state.hidden.accumulated_text + action.text
            score = self.prm.score(accumulated)
            return RewardSignal(
                value=score,
                name=self.name,
                reward_type=self.reward_type
            )
        return RewardSignal(value=0.0, name=self.name, reward_type=self.reward_type)

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
from env_evals.adapters import create_huggingface_environment
from env_evals.core import SegmentedEnvironment, PatternSegmenter

# Load AIME problems
base_env = create_huggingface_environment("HuggingFaceH4/aime_2024")

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
