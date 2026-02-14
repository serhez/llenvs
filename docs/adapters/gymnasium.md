# Gymnasium

The Gymnasium adapter wraps any [gymnasium](https://gymnasium.farama.org/)-compatible environment for text-based LLM agents. It bridges numeric observations and actions to text via configurable mapper protocols.

## Installation

```bash
pip install gymnasium
# or
pip install llenvs[gymnasium]
```

## Quick Start

```python
import gymnasium
from llenvs.adapters import GymnasiumAdapter, GymnasiumEnvironment
from llenvs.core import Action

# Via adapter
adapter = GymnasiumAdapter()
env = adapter.get_environment(
    "CartPole-v1",
    action_names={0: "left", 1: "right"},
    num_tasks=100,
)

# Or wrap directly
gym_env = gymnasium.make("CartPole-v1")
env = GymnasiumEnvironment(
    gym_env=gym_env,
    action_names={0: "left", 1: "right"},
    observation_labels={0: "Cart Position", 1: "Cart Velocity", 2: "Pole Angle", 3: "Pole Angular Velocity"},
    num_tasks=100,
)

state, _ = env.reset(seed=42)
print(state.observation.prompt)
# Shows observation/action space descriptions + initial observation

result = env.step(state, Action(text="left"))
print(result.rewards.total)
```

## Observation Mapping

The `ObservationMapper` protocol converts gymnasium observations to text:

```python
class ObservationMapper(Protocol):
    def map(self, obs: Any, info: dict[str, Any]) -> str: ...
    def describe(self) -> str: ...
```

### AutoObservationMapper

Built-in mapper that handles common space types:

| Space Type | Rendering |
|-----------|-----------|
| `Discrete(n)` | `"State: 3"` or `"State: moving_left"` (with labels) |
| `Box(k,)` where k <= 20 | `"Cart Position: 1.20, Cart Velocity: -0.53, ..."` |
| `Box(k,)` where k > 20 | `"[50-dim vector, min=-2.31, max=4.12]"` |
| `Box` with ndim >= 2 | `ValueError` (use custom mapper) |
| `Dict` / `Tuple` | Recursive rendering |
| `MultiDiscrete` / `MultiBinary` | Array with labels |
| `Text` | Passthrough |

```python
env = GymnasiumEnvironment(
    gym_env=gym_env,
    observation_labels={
        0: "Cart Position",
        1: "Cart Velocity",
        2: "Pole Angle",
        3: "Pole Angular Velocity",
    },
    num_tasks=100,
)
```

### GridObservationMapper

For grid-world environments with 2D/3D matrix observations:

```python
from llenvs.adapters import GridObservationMapper

mapper = GridObservationMapper(
    value_map={0.0: ".", 0.3: " ", 0.6: "R", 1.0: "#"},
)

env = GymnasiumEnvironment(
    gym_env=grid_env,
    observation_mapper=mapper,
    action_names={0: "right", 1: "left", 2: "up", 3: "down"},
    num_tasks=50,
)
```

Output:

```
R....
.....
...#.
.....
.....
```

### Custom Observation Mapper

Implement the protocol for complex observations:

```python
class MyObservationMapper:
    def map(self, obs, info):
        return f"Position: ({obs['x']}, {obs['y']}), Health: {obs['hp']}"

    def describe(self):
        return "Position coordinates and health points."
```

## Action Mapping

The two-step action pipeline:

1. **Extract** — `AnswerExtractor` extracts action text from LLM response (default: `RawGenerationExtractor`, entire response is the action)
2. **Map** — `ActionMapper` converts extracted text to gymnasium action value

### AutoActionMapper

| Space Type | Parsing |
|-----------|---------|
| `Discrete(n)` | Integer or case-insensitive name match |
| `Box(k,)` | Comma/whitespace-separated floats, clipped to bounds |
| `MultiDiscrete` | Comma-separated integers, range-validated |
| `MultiBinary` | Comma-separated binary values |
| `Text` | Passthrough |

```python
env = GymnasiumEnvironment(
    gym_env=gym_env,
    action_names={0: "left", 1: "right"},
    num_tasks=100,
)

# LLM can respond with either:
result = env.step(state, Action(text="left"))   # by name
result = env.step(state, Action(text="0"))      # by number
```

### Invalid Actions

Invalid actions (extraction failure or mapping error) waste a turn: the step counter advances, an error message is returned as the observation, but no gymnasium step occurs.

### Using AnswerExtractor

Configure extraction for structured LLM responses:

```python
from llenvs.core.extraction import TagBasedExtractor

env = GymnasiumEnvironment(
    gym_env=gym_env,
    answer_extractor=TagBasedExtractor(tag_name="action"),
    action_names={0: "left", 1: "right"},
    num_tasks=100,
)

# LLM responds: "I should go left. <action>left</action>"
# Extractor pulls "left", mapper converts to 0
```

## ANSI Render

When `use_ansi_render=True`, the environment calls `gym_env.render()` for observations instead of the mapper. Falls back to the mapper if render returns `None`.

```python
env = adapter.get_environment(
    "Taxi-v3",
    use_ansi_render=True,
    num_tasks=100,
)
# render_mode="ansi" is automatically set
```

## Seeds and Task Indexing

Control reproducibility with seeds or task counts:

```python
# Fixed seed list — each task_index maps to a seed
env = GymnasiumEnvironment(
    gym_env=gym_env,
    seeds=[42, 123, 456, 789],
)
assert len(env) == 4
state, _ = env.reset(options={"task_index": 2})  # uses seed 456

# Or specify task count without fixed seeds
env = GymnasiumEnvironment(
    gym_env=gym_env,
    num_tasks=100,
)
assert len(env) == 100
state, _ = env.reset(seed=42)  # explicit seed always takes priority
```

## Presets

Built-in presets for popular gymnasium-compatible environments:

### Gym4Real

| Preset ID | Description | Action Type |
|-----------|-------------|-------------|
| `gym4real/dam-v0` | Water release control | Continuous |
| `gym4real/elevator-v0` | Elevator control | Discrete (up/down/open) |
| `gym4real/microgrid-v0` | Energy grid management | Continuous |
| `gym4real/TradingEnv-v0` | Trading decisions | Discrete (short/flat/long) |
| `gym4real/wds-v0` | Water distribution | Binary (pump on/off) |

### MarsExplorer

| Preset ID | Description | Action Type |
|-----------|-------------|-------------|
| `mars_explorer` | Grid exploration | Discrete (right/left/up/down) |

```python
adapter = GymnasiumAdapter()
env = adapter.get_environment("gym4real/elevator-v0", num_tasks=50)
# Preset provides action_names and description automatically
```

Presets can be overridden:

```python
env = adapter.get_environment(
    "gym4real/elevator-v0",
    max_steps=100,  # override default
    num_tasks=50,
)
```

## Using with TrajectoryRunner

```python
from llenvs.adapters import GymnasiumAdapter
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

adapter = GymnasiumAdapter()
env = adapter.get_environment(
    "CartPole-v1",
    action_names={0: "left", 1: "right"},
    num_tasks=100,
    max_steps=200,
)

backend = OpenAIBackend(model="gpt-4o")
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
)

result = runner.run_trajectory(task_index=0, seed=42)
print(f"Total reward: {result.trajectory.total_reward}")
```

## Using the Adapter Directly

```python
from llenvs.adapters import GymnasiumAdapter

adapter = GymnasiumAdapter()

# List presets
print(adapter.list_environments())

# Get environment info
info = adapter.get_environment_info("gym4real/elevator-v0")
print(info)

# Pass pre-created gym env
import gymnasium
gym_env = gymnasium.make("LunarLander-v3")
env = adapter.get_environment(
    "LunarLander-v3",
    gym_env=gym_env,
    action_names={0: "noop", 1: "left engine", 2: "main engine", 3: "right engine"},
    num_tasks=200,
)
```

## Extra Rewards

Add format or other reward signals:

```python
from llenvs.core.reward import FormatReward
from llenvs.core.extraction import TagBasedExtractor

extractor = TagBasedExtractor(tag_name="action")
env = GymnasiumEnvironment(
    gym_env=gym_env,
    answer_extractor=extractor,
    extra_rewards=(FormatReward(extractor),),
    num_tasks=100,
)
```

## Hidden State

```python
@dataclass(frozen=True)
class GymnasiumHidden:
    task_index: int | None
    seed: int | None
    episode_step: int
    last_action: str | None
    raw_observation: Any    # raw gymnasium observation
    gym_reward: float       # cumulative episode reward
```

## Limitations

- Image observations (`Box` with ndim >= 2) require a custom `ObservationMapper`
- Non-pure (`pure_step=False`) — cannot branch with `DirectStrategy`; use `ActionReplay` or `ProcessFork` strategies
- State snapshotting depends on the underlying gymnasium environment's behavior
