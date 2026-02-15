# RL Training

llenvs environments can serve as reward functions and data sources for RL training frameworks like veRL, TRL, and OpenRLHF.

## Overview

RL frameworks don't call `env.step()` themselves — they generate completions with their own policy and need **scoring functions** to evaluate responses. For multi-turn environments, they need the full environment loop but with *their* model driving generation, plus **token-level masks** distinguishing model tokens from environment tokens.

llenvs provides three layers:

1. **`Scorer`** — Score responses against environment tasks (single-turn)
2. **`DatasetProvider`** — Source prompts and ground truths for training dataloaders
3. **`TrajectoryMasker`** — Convert multi-turn trajectories into token-level masks

Framework-specific adapters (veRL, TRL, OpenRLHF) are thin wrappers around these primitives.

## Single-Turn Scoring

The `Scorer` wraps any single-turn environment to provide a `score()` interface that reuses the environment's reward computation.

```python
from llenvs.core.registry import environment_registry
from llenvs.integrations import Scorer

env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=1000,
    seed=42,
)

scorer = Scorer(env)

# Score a single response
result = scorer.score(task_index=0, response="<answer>4</answer>")
print(f"Total reward: {result.total}")        # Sum of all signals
print(f"Signals: {result.signals}")            # {"correctness": 1.0}
print(f"Answer: {result.extracted_answer}")    # "4"

# Batch scoring
results = scorer.score_batch(
    task_indices=[0, 1, 2],
    responses=["<answer>4</answer>", "<answer>wrong</answer>", "<answer>6</answer>"],
)
```

### ScoringResult

```python
@dataclass(frozen=True)
class ScoringResult:
    total: float                    # Weighted sum of numeric signal rewards
    signals: dict[str, float]      # name -> reward (only numeric signals)
    extracted_answer: str | None   # What the extractor found
    metadata: dict[str, Any]       # Per-signal metadata
```

## Dataset Provider

The `DatasetProvider` iterates over environment tasks to provide prompts and ground truths.

```python
from llenvs.integrations import DatasetProvider

provider = DatasetProvider(env)

# Iterate tasks
for i in range(len(provider)):
    item = provider[i]
    print(f"Task {item.task_index}: {item.prompt}")
    print(f"Expected: {item.ground_truth}")

# Get specific tasks
items = provider.get_items(indices=[0, 5, 10])

# Convert to HuggingFace Dataset (requires `datasets` package)
hf_dataset = provider.to_hf_dataset()
# Columns: task_index, prompt, ground_truth, messages
```

### TaskItem

```python
@dataclass(frozen=True)
class TaskItem:
    task_index: int
    prompt: str
    messages: tuple[dict[str, Any], ...]
    ground_truth: str | None    # None for multi-turn environments
    metadata: dict[str, Any]
```

## Token Masking for Multi-Turn

For RL training on multi-turn trajectories, frameworks need a `response_mask` indicating which tokens are model-generated (mask=1, receives gradient) vs environment-generated (mask=0, no gradient).

```python
from llenvs.integrations import TrajectoryMasker

masker = TrajectoryMasker(tokenizer)  # Any tokenizer with encode(str) -> list[int]

masked = masker.mask_trajectory(trajectory)
print(f"Prompt tokens: {len(masked.prompt_ids)}")
print(f"Response tokens: {len(masked.response_ids)}")
print(f"Model tokens: {sum(masked.response_mask)}")
print(f"Env tokens: {len(masked.response_mask) - sum(masked.response_mask)}")
print(f"Per-step rewards: {masked.rewards}")
```

### MaskedTrajectory

```python
@dataclass(frozen=True)
class MaskedTrajectory:
    prompt_ids: tuple[int, ...]        # Initial observation tokens
    response_ids: tuple[int, ...]      # All response tokens concatenated
    response_mask: tuple[int, ...]     # 1=model, 0=environment
    spans: tuple[TokenSpan, ...]       # Structured span information
    rewards: tuple[float, ...]         # Per-step reward totals
```

### TokenSpan

```python
@dataclass(frozen=True)
class TokenSpan:
    text: str
    token_ids: tuple[int, ...]
    source: Literal["model", "environment"]
    step_index: int
```

## Framework Recipes

### veRL

**Single-turn reward function:**

```python
from llenvs.integrations.verl import make_verl_reward_fn, make_verl_dataset

# Create reward function with veRL's expected signature
compute_score = make_verl_reward_fn(env)
# compute_score(data_source, solution_str, ground_truth, extra_info) -> float

# Create dataset for veRL's DataLoader
dataset = make_verl_dataset(env, num_tasks=1000)
# Returns list[dict] with 'prompt', 'ground_truth', 'data_source', 'extra_info' keys
```

**Multi-turn with AgentLoop:**

```python
from llenvs.integrations.verl import LLEnvsAgentLoop

loop = LLEnvsAgentLoop(multi_turn_env, tokenizer, max_steps=20)

async def generate_fn(messages):
    # Your veRL generation logic here
    return model.generate(messages)

result = await loop.run(task_index=0, generate_fn=generate_fn)
# result: {"prompt_ids", "response_ids", "response_mask", "rewards"}
```

### TRL (GRPOTrainer)

**Single-turn reward function:**

```python
from llenvs.integrations.trl import make_trl_reward_fn, make_trl_dataset

# Create reward function with TRL's expected signature
reward_func = make_trl_reward_fn(env)
# reward_func(prompts, completions, task_indices=...) -> list[float]

# Create HuggingFace Dataset for GRPOTrainer
dataset = make_trl_dataset(env, num_tasks=1000)
# Returns datasets.Dataset with 'prompt' column
```

**Multi-turn with rollout function:**

```python
from llenvs.integrations.trl import make_trl_rollout_fn

rollout_fn = make_trl_rollout_fn(multi_turn_env, tokenizer, max_steps=20)

async def generate_fn(messages):
    return trainer.model.generate(messages)

result = await rollout_fn(task_index=0, generate_fn=generate_fn)
```

### OpenRLHF

**Single-turn reward function:**

```python
from llenvs.integrations.openrlhf import make_openrlhf_reward_fn

reward_func = make_openrlhf_reward_fn(env)
# reward_func(queries, prompts, labels, task_indices=...) -> dict
# Returns {"rewards": list[float], "scores": list[float], "extra_logs": dict}
```

For multi-turn OpenRLHF training, use the `TrajectoryMasker` directly as a building block.

## Configuration-Driven Setup

All integration classes support `from_config()` for YAML-driven setup:

```python
from llenvs.core.config import EnvironmentConfig
from llenvs.integrations import Scorer, DatasetProvider

config = EnvironmentConfig(
    name="leg_counting",
    adapter="reasoning_gym",
    size=1000,
    seed=42,
    answer_extractor="tag_based",
)

scorer = Scorer.from_config(config)
provider = DatasetProvider.from_config(config)
```

## Installation

The core integration classes (`Scorer`, `DatasetProvider`, `TrajectoryMasker`) have no extra dependencies. Framework-specific dataset functions that return HuggingFace Datasets need the `datasets` package:

```bash
uv pip install -e ".[rl]"        # datasets for to_hf_dataset() and make_trl_dataset()
uv pip install -e ".[trl]"       # TRL + datasets
```
