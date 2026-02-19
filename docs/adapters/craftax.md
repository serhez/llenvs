# Craftax

The Craftax adapter wraps [Craftax](https://github.com/MichaelTMatworthy/Craftax) (ICML 2024), a fast JAX-based open-ended survival benchmark combining Crafter and NetHack mechanics. Procedurally-generated worlds with crafting, combat, dungeons, and magic.

Four variants exist: Full/Classic x Symbolic/Pixels. The adapter supports text, symbolic, and pixel observation modes.

## Installation

```bash
pip install craftax  # requires JAX
```

Craftax depends on JAX, which has its own platform-specific installation. See the [JAX installation guide](https://github.com/google/jax#installation).

## Quick Start

```python
from llenvs.adapters import CraftaxAdapter, CraftaxEnvironment
from llenvs.core import Action

adapter = CraftaxAdapter()
env = adapter.get_environment("craftax", num_tasks=50)

state, _ = env.reset(seed=42)
print(state.observation.prompt)
# Game description + visible map, inventory, stats

result = env.step(state, Action(text="left"))
print(result.rewards.total)
```

## Observation Modes

### Text Mode (`"text"`)

Uses Craftax's built-in text renderer (Full variant only). Produces natural-language descriptions of the visible world.

```python
env = adapter.get_environment("craftax", observation_mode="text")
```

### Symbolic Mode (`"symbolic"`)

Parses the flat symbolic observation array into structured text sections: visible map (grid), inventory, player stats, and dungeon info. Works for both Full and Classic variants.

```python
env = adapter.get_environment("craftax-classic", observation_mode="symbolic")
```

Output:

```
=== Map (9x11 visible grid) ===
G G G S S W W G G G G
G G S S S S W G T G G
...

=== Inventory ===
Wood: 0, Stone: 0, Coal: 0, Iron: 0, Diamond: 0, ...

=== Player Stats ===
Health: 9/9, Food: 9/9, Drink: 9/9, Energy: 9/9
Direction: east
```

### Pixel Mode (`"pixels"`)

Renders the environment as an image. Returns an `ImageContent` object in the observation's `images` field. Requires a vision-capable model (GPT-4o, Claude, etc.).

```python
env = adapter.get_environment("craftax-pixels", num_tasks=50)

state, _ = env.reset(seed=42)
# state.observation.images contains the rendered frame as base64 PNG
```

## Action Space

Actions are discrete. Classic has 17 actions, Full has 43. The model can respond with either the action name or its integer index:

```python
result = env.step(state, Action(text="left"))       # by name
result = env.step(state, Action(text="1"))           # by index
result = env.step(state, Action(text="make_stone_pickaxe"))  # crafting
```

### Classic Actions (17)

| Index | Action |
|-------|--------|
| 0 | noop |
| 1–4 | left, right, up, down |
| 5 | do (interact) |
| 6 | sleep |
| 7–10 | place_stone, place_table, place_furnace, place_plant |
| 11–16 | make_{wood,stone,iron}_{pickaxe,sword} |

### Full Actions (43)

All Classic actions plus: diamond tools, armour, arrows, torches, rubies, sapphires, potions, books, enchanting, and more.

### Invalid Actions

Invalid actions (unrecognized name, out-of-range index) waste a turn: the step counter advances and an error message is returned, but no environment step occurs.

## Achievements

Craftax tracks in-game achievements (crafting milestones, combat kills, exploration progress). These are available in the hidden state and step info:

```python
result = env.step(state, action)
new_achievements = result.info["new_achievements"]  # list of achievement indices
```

### Achievement Reward

For fine-grained per-achievement signals, use `CraftaxAchievementReward` as an extra reward:

```python
from llenvs.adapters import CraftaxAchievementReward

env = adapter.get_environment(
    "craftax",
    extra_rewards=(CraftaxAchievementReward(),),
    num_tasks=50,
)
```

## Presets

| Preset | Variant | Observation Mode |
|--------|---------|-----------------|
| `craftax` | Full | `text` |
| `craftax-classic` | Classic | `symbolic` |
| `craftax-pixels` | Full | `pixels` |
| `craftax-classic-pixels` | Classic | `pixels` |

```python
adapter = CraftaxAdapter()
env = adapter.get_environment("craftax-classic", num_tasks=50)
# Classic variant with symbolic observations
```

Presets can be overridden:

```python
env = adapter.get_environment(
    "craftax-classic",
    observation_mode="pixels",  # override to pixel mode
    max_steps=500,
    num_tasks=100,
)
```

## Pure Step

Craftax is purely functional (immutable JAX pytrees), so `pure_step=True`. This means:

- The same `(state, action)` pair always produces the same result
- Zero-cost `DirectStrategy` branching — no need to copy environments
- Safe for parallel evaluation without state interference

This is one of the few multi-turn environments with this property.

## Seeds and Task Indexing

```python
# Fixed number of tasks
env = adapter.get_environment("craftax", num_tasks=100)
state, _ = env.reset(options={"task_index": 42})

# Explicit seed
state, _ = env.reset(seed=12345)

# Both — task_index generates a deterministic seed, explicit seed overrides
state, _ = env.reset(seed=42, options={"task_index": 0})
```

## Using with TrajectoryRunner

```python
from llenvs.adapters import CraftaxAdapter
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

adapter = CraftaxAdapter()
env = adapter.get_environment("craftax", num_tasks=50, max_steps=200)

backend = OpenAIBackend(model="gpt-4o")
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
)

result = runner.run_trajectory(task_index=0, seed=42)
print(f"Total reward: {result.trajectory.total_reward}")
```

For pixel mode, use a vision-capable backend:

```python
env = adapter.get_environment("craftax-pixels", num_tasks=50, max_steps=200)
backend = OpenAIBackend(model="gpt-4o")  # supports_vision=True
```

## Hidden State

```python
@dataclass(frozen=True)
class CraftaxHidden:
    task_index: int | None
    seed: int | None
    episode_step: int
    last_action: str | None
    craftax_state: Any      # JAX EnvState pytree
    rng_key: Any            # JAX PRNG key
    cumulative_reward: float
    achievements: Any       # JAX boolean array
    is_classic: bool
```

## Extra Rewards

```python
from llenvs.core.reward import FormatReward
from llenvs.core.extraction import TagBasedExtractor
from llenvs.adapters import CraftaxAchievementReward

extractor = TagBasedExtractor(tag_name="action")
env = adapter.get_environment(
    "craftax",
    answer_extractor=extractor,
    extra_rewards=(FormatReward(extractor), CraftaxAchievementReward()),
    num_tasks=50,
)
```

## Limitations

- **JAX dependency**: Craftax requires JAX, which needs platform-specific installation
- **Container serialization**: JAX pytrees in `CraftaxHidden` are not JSON-serializable. Container proxying is not the primary use case since `pure_step=True` enables direct branching
- **Classic text mode**: Classic variant has no built-in text renderer; use `"symbolic"` mode instead
- **Pixel encoding**: Uses PIL if available, falls back to minimal PNG encoder
