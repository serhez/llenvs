# Core API

This document covers the core abstractions in llenvs.

## State

```python
@dataclass(frozen=True)
class State(Generic[HiddenT]):
    observation: Observation
    hidden: HiddenT
    metadata: StateMetadata
```

**Location**: `llenvs/core/state.py`

States are immutable (frozen dataclass). The generic type parameter allows environments to define their own hidden state type.

### StateMetadata

```python
@dataclass(frozen=True)
class StateMetadata:
    step: int              # Current step number (0-indexed)
    episode_id: str        # Unique identifier for this episode
    is_terminal: bool      # Whether episode has ended
    info: dict[str, Any]   # Additional metadata
```

### Observation

```python
@dataclass(frozen=True)
class Observation:
    prompt: str                                          # The question/prompt text
    messages: tuple[dict[str, Any], ...] = ()            # Chat history (including tool calls/results)
    tool_results: tuple[ToolResult, ...] = ()            # Results from most recent tool calls
    available_tools: tuple[ToolDefinition, ...] = ()     # Tools the model can call
```

All environments use `Observation`. Text-only environments return observations with `tool_results=()` and `available_tools=()`. Tool-enabled environments populate these fields.

### Action

```python
@dataclass(frozen=True)
class Action:
    text: str | None = None                # Optional text response
    tool_calls: tuple[ToolCall, ...] = ()  # Optional tool calls

    @classmethod
    def from_text(cls, text: str) -> Action: ...
    @classmethod
    def from_tool_call(cls, call: ToolCall) -> Action: ...

    @property
    def is_text_only(self) -> bool: ...
    @property
    def has_tool_calls(self) -> bool: ...
```

All environments accept `Action`. For text-only environments, use `Action.from_text(...)`. For tool environments, actions can include `tool_calls`.

## Environment Protocol

```python
class Environment(Protocol[HiddenT]):
    @property
    def spec(self) -> EnvironmentSpec: ...

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]: ...

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HiddenT], dict[str, Any]]: ...

    def step(
        self,
        state: State[HiddenT],
        action: Action,
    ) -> StepResult[HiddenT]: ...

    def compute_rewards(
        self,
        state: State[HiddenT],
        action: Action,
        next_state: State[HiddenT],
    ) -> RewardBundle: ...
```

**Location**: `llenvs/core/environment.py`

| Member | Description |
|--------|-------------|
| `available_tools` | Tools available in this environment; text-only environments return `()` |
| `reset()` | Initialize a new episode, returns initial state |
| `step()` | Take action from state, returns `StepResult` with next state and rewards |
| `compute_rewards()` | Compute rewards for a transition |

### StepResult

```python
@dataclass(frozen=True)
class StepResult(Generic[HiddenT]):
    next_state: State[HiddenT]
    rewards: RewardBundle
    terminated: bool  # Episode ended naturally
    truncated: bool   # Episode cut off (max steps, etc.)
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
```

### EnvironmentSpec

```python
@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    adapter: str                 # Which adapter (e.g., "reasoning_gym")
    max_steps: int | None        # None = unlimited
    is_multi_turn: bool
    metadata: dict[str, Any]
```

## Rewards

**Location**: `llenvs/core/reward.py`

### RewardType

```python
class RewardType(Enum):
    OUTCOME = auto()   # Final correctness (binary or graded)
    STEP = auto()      # Per-turn feedback
    FORMAT = auto()    # Did model follow formatting instructions
    PROCESS = auto()   # Intermediate reasoning quality
```

### RewardSignal

```python
@dataclass(frozen=True)
class RewardSignal:
    value: float
    name: str           # e.g., "correctness", "format"
    reward_type: RewardType
    metadata: dict[str, Any] | None = None
```

### RewardBundle

```python
@dataclass(frozen=True)
class RewardBundle:
    signals: tuple[RewardSignal, ...]

    @property
    def total(self) -> float:
        """Sum of all signal values."""

    def by_name(self, name: str) -> RewardSignal | None:
        """Get signal by name."""

    def by_type(self, reward_type: RewardType) -> tuple[RewardSignal, ...]:
        """Get all signals of a given type."""

    @classmethod
    def single(cls, value: float, name: str = "reward", ...) -> RewardBundle:
        """Create bundle with single signal."""

    @classmethod
    def empty(cls) -> RewardBundle:
        """Create empty bundle."""
```

### RewardFunction Protocol

```python
class RewardFunction(Protocol[HiddenT]):
    @property
    def name(self) -> str: ...

    @property
    def reward_type(self) -> RewardType: ...

    def compute(
        self,
        state: State[HiddenT],
        action: Action,
        next_state: State[HiddenT],
    ) -> RewardSignal: ...
```

## Trajectory

**Location**: `llenvs/core/trajectory.py`

```python
@dataclass
class Trajectory(Generic[HiddenT]):
    episode_id: str
    initial_state: State[HiddenT]

    @classmethod
    def create(cls, initial_state: State) -> Trajectory: ...

    def add_transition(self, transition: Transition) -> None: ...

    def checkpoint(self, name: str) -> Checkpoint: ...

    def branch(self, checkpoint_name: str) -> Trajectory: ...

    def state_at(self, index: int) -> State: ...

    @property
    def current_state(self) -> State: ...

    @property
    def transitions(self) -> tuple[Transition, ...]: ...

    @property
    def total_reward(self) -> float: ...

    @property
    def is_terminal(self) -> bool: ...
```

### Transition

```python
@dataclass(frozen=True)
class Transition(Generic[HiddenT]):
    state: State[HiddenT]
    action: Action
    next_state: State[HiddenT]
    rewards: RewardBundle
    info: dict[str, Any]
```

### Checkpoint

```python
@dataclass(frozen=True)
class Checkpoint(Generic[HiddenT]):
    name: str
    trajectory_id: str
    step_index: int
    state: State[HiddenT]
```

## Answer Extraction

**Location**: `llenvs/core/extraction.py`

```python
class AnswerExtractor(Protocol):
    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Extract answer from response.

        Returns:
            (extracted_answer, metadata)
            extracted_answer is None if extraction failed
        """
```

### Built-in Extractors

| Extractor | Registry Name | Description | Example Pattern |
|-----------|---------------|-------------|-----------------|
| `TagBasedExtractor` | `tag_based` | XML-style tags | `<answer>42</answer>` |
| `RegexExtractor` | `regex` | Custom regex with capture group | Any pattern |
| `GSM8KExtractor` | `gsm8k` | GSM8K format | `#### 42` |
| `MultipleChoiceExtractor` | `multiple_choice` | A/B/C/D answers | `Answer: B`, `(A)` |
| `BoxedExtractor` | `boxed` | LaTeX `\boxed{...}` with balanced braces | `\boxed{x^{2}+1}` |
| `NumericExtractor` | `numeric` | Last number in text | `The answer is 42` |
| `LastLineExtractor` | `last_line` | Last non-empty line | Any text |
| `CodeBlockExtractor` | `code_block` | Markdown code fences | ` ```python\n...\n``` ` |
| `PatternAnswerExtractor` | `pattern_answer` | "the answer is X", "therefore X", "= X" | Natural language |
| `CompositeExtractor` | - | Try multiple extractors in order | - |
| `RawGenerationExtractor` | `raw` | Return full response | - |
| `NativeExtractor` | - | Wraps a third-party extraction function | - |

All extractors follow the **last match wins** convention when multiple matches exist.

```python
# Tag-based (default)
extractor = TagBasedExtractor(tag_name="answer")
answer, meta = extractor.extract("The answer is <answer>42</answer>")
# answer = "42"

# GSM8K format
extractor = GSM8KExtractor()
answer, meta = extractor.extract("So the total is #### 42")
# answer = "42"

# Composite fallback chain
extractor = CompositeExtractor(extractors=[
    TagBasedExtractor(),
    BoxedExtractor(),
    PatternAnswerExtractor(),
    NumericExtractor(),
    RawGenerationExtractor(),
])
```

### Cleaning Layer

**Location**: `llenvs/core/cleaning.py`

`CleanedExtractor` wraps any extractor with **pre-cleaners** (applied to the raw response before extraction) and **post-cleaners** (applied to the extracted answer after extraction).

```python
from llenvs.core.extraction import CleanedExtractor
from llenvs.core.cleaning import strip_special_tokens, strip_trailing_punctuation

extractor = CleanedExtractor(
    inner=TagBasedExtractor(),
    pre_cleaners=[strip_special_tokens],
    post_cleaners=[strip_trailing_punctuation],
)
answer, meta = extractor.extract("<answer>42.</answer><|endoftext|>")
# answer = "42"
```

**Pre-cleaners** (raw response → cleaned response):

| Name | Default | Description |
|------|---------|-------------|
| `strip_special_tokens` | Yes | Remove `<\|endoftext\|>`, `<pad>`, `</s>`, `<\|im_end\|>`, `<\|im_start\|>`, `<s>` |

**Post-cleaners** (extracted answer → cleaned answer):

| Name | Default | Description |
|------|---------|-------------|
| `strip_trailing_punctuation` | Yes | Remove trailing `.` or `,` (preserves decimal numbers) |
| `strip_surrounding_quotes` | No | Remove matched surrounding `"..."` or `'...'` |
| `strip_latex_dollars` | No | Remove surrounding `$...$` or `$$...$$` |

When using YAML configuration, the cleaning layer is applied automatically by `EnvironmentFactory`. See [Configuration](config.md) for details.

## Registry

**Location**: `llenvs/core/registry.py`

```python
from llenvs.core.registry import environment_registry

# List all registered adapters
adapters = environment_registry.list_adapters()
# ["reasoning_gym", "huggingface", "gem", "webshop"]

# List all environments
all_envs = environment_registry.list_environments()
for adapter, name in all_envs:
    print(f"  {adapter}/{name}")

# Get environment with configuration
env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=100,
    seed=42,
)

# Register custom adapter
environment_registry.register_adapter(MyCustomAdapter())
```

## RL Training Integrations

**Location**: `llenvs/integrations/`

### Scorer

Scores responses against environment tasks for RL training.

```python
from llenvs.integrations import Scorer, ScoringResult

scorer = Scorer(environment)
result = scorer.score(task_index=0, response="<answer>42</answer>")
# result.total, result.signals, result.extracted_answer, result.metadata

results = scorer.score_batch([0, 1, 2], ["a", "b", "c"])
```

Raises `TypeError` for multi-turn environments.

### DatasetProvider

Provides prompts and ground truths from environment tasks.

```python
from llenvs.integrations import DatasetProvider, TaskItem

provider = DatasetProvider(environment)
item = provider[0]       # TaskItem
items = provider.get_items([0, 1, 2])
hf_ds = provider.to_hf_dataset()  # requires datasets package
```

### TrajectoryMasker

Converts trajectories into token-level masks for multi-turn RL training.

```python
from llenvs.integrations import TrajectoryMasker, MaskedTrajectory

masker = TrajectoryMasker(tokenizer)
masked = masker.mask_trajectory(trajectory)
# masked.prompt_ids, masked.response_ids, masked.response_mask, masked.rewards
```

See **[RL Training Guide](../guides/rl-training.md)** for framework-specific recipes.
