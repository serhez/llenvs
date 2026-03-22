# Jericho

The Jericho adapter wraps [Jericho](https://github.com/microsoft/jericho) (Microsoft) to provide access to 50+ classic interactive fiction text adventure games for LLM agent evaluation. Games include Zork, Hitchhiker's Guide to the Galaxy, Detective, Planetfall, and many more.

## Installation

```bash
pip install jericho
# or
pip install llenvs[jericho]
```

Jericho bundles game ROM files, so no additional data download is needed.

## Quick Start

```python
from llenvs.adapters import JerichoAdapter
from llenvs.core import Action

adapter = JerichoAdapter()
env = adapter.get_environment("jericho:zork1", pure_step=True)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
# "ZORK I: The Great Underground Empire..."

result = env.step(state, Action(text="open mailbox"))
print(result.next_state.observation.messages[-1]["content"])
```

## Available Games

Jericho bundles 50+ classic games. Notable examples:

| Game | Description |
|------|-------------|
| `zork1` | The Great Underground Empire |
| `zork2` | The Wizard of Frobozz |
| `zork3` | The Dungeon Master |
| `detective` | A murder mystery |
| `hhgg` | Hitchhiker's Guide to the Galaxy |
| `planetfall` | Sci-fi adventure |
| `trinity` | Nuclear age adventure |
| `wishbringer` | Fantasy adventure |
| `enchanter` | Magic-themed adventure |
| `lurking` | Lurking Horror |

List all available games:

```python
adapter = JerichoAdapter()
games = adapter.list_environments()
# ["jericho:acorncourt", "jericho:anchor", "jericho:detective", ...]
```

## Action Format

Actions are free-form text commands understood by the Z-Machine:

| Action | Description |
|--------|-------------|
| `go north/south/east/west/up/down` | Navigate |
| `open {obj}` / `close {obj}` | Open or close something |
| `take {obj}` / `drop {obj}` | Pick up or drop an object |
| `put {obj} in/on {obj}` | Place an object |
| `examine {obj}` | Look at something |
| `turn on {obj}` / `turn off {obj}` | Operate a device |
| `look` | Look around |
| `inventory` | Check held items |

## Valid Actions

Jericho can compute valid actions at each step. By default, these are **not** shown in the observation (wrapper fidelity — the original games don't show valid actions to players), but they are always tracked in the hidden state:

```python
# Not in observation (default, wrapper fidelity)
env = adapter.get_environment("jericho:zork1", include_valid_actions=False)

# Included in observation (research/debug mode)
env = adapter.get_environment("jericho:zork1", include_valid_actions=True)

state, _ = env.reset(options={"task_index": 0})
print(state.hidden.valid_actions)  # always available
```

## Adapter Configuration

### Game Selection

Select games by name, file path, or get all bundled games:

```python
# Single game from name
env = adapter.get_environment("jericho:zork1")

# Multiple games by name
env = adapter.get_environment(games=["zork1", "detective", "hhgg"])

# All bundled games
env = adapter.get_environment("jericho")

# Direct ROM file path
env = adapter.get_environment(game_files=["/path/to/custom_game.z5"])
```

### Parameters

```python
env = adapter.get_environment(
    name="jericho:zork1",
    max_steps=100,                         # default: 100
    include_valid_actions=False,           # default: False (wrapper fidelity)
    pure_step=True,                        # default: False; enable for MC rollouts
    prompts={                              # override prompt templates
        "valid_actions_prefix": "Available commands:",
    },
)
```

## Rewards

Jericho provides score-based rewards with two modes depending on step type:

| Reward | Type | Value |
|--------|------|-------|
| `game_score` | `STEP` (intermediate) | Score delta since last step |
| `game_score` | `OUTCOME` (terminal) | Normalized score: `score / max_score` |

```python
result = env.step(state, Action(text="take leaflet"))
signal = result.rewards.by_name("game_score")
print(signal.reward)       # e.g., 5 (score delta)
print(signal.reward_type)  # RewardType.STEP
```

On terminal steps (game over or truncation):

```python
signal = result.rewards.by_name("game_score")
print(signal.reward)       # e.g., 0.5 (175/350 normalized)
print(signal.reward_type)  # RewardType.OUTCOME
```

Add extra reward signals:

```python
env = JerichoEnvironment(
    game_files=game_files,
    game_names=game_names,
    extra_rewards=(my_custom_reward,),
)
```

## Hidden State

```python
@dataclass(frozen=True)
class JerichoHidden:
    task_index: int                    # index into game_files
    game_name: str                     # e.g., "zork1"
    game_file: str                     # path to ROM file
    episode_step: int                  # current step count
    last_action: str | None            # last action taken
    score: int                         # current cumulative score
    max_score: int                     # maximum achievable score
    moves: int                         # move counter from Jericho
    valid_actions: tuple[str, ...]     # currently valid actions
    prev_score: int = 0               # score at previous step (for delta)
    frotz_state: Any = None           # Z-Machine snapshot (pure_step only)
```

## Using with TrajectoryRunner

```python
from llenvs.adapters import JerichoAdapter
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

adapter = JerichoAdapter()
env = adapter.get_environment(
    "jericho:zork1",
    max_steps=100,
)

backend = OpenAIBackend(model="gpt-4o")
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
)

result = runner.run_trajectory(task_index=0)
print(f"Score: {result.trajectory.final_state.hidden.score}")
print(f"Max: {result.trajectory.final_state.hidden.max_score}")
```

## State Snapshots (pure_step)

With `pure_step=True`, the adapter uses Jericho's native `get_state()/set_state()` to save and restore Z-Machine state. This enables stepping from any previously visited state (branching), which is required for Monte Carlo rollout-based ground truth estimation.

```python
env = adapter.get_environment("jericho:zork1", pure_step=True)
state_0, _ = env.reset(options={"task_index": 0})

# Branch A
result_a = env.step(state_0, Action(text="go north"))

# Branch B from the same state
result_b = env.step(state_0, Action(text="open mailbox"))
```

State snapshots are compact (native Z-Machine tuples, a few KB) and much faster than pickle-based approaches.

## Limitations

- Seed support is available but behavior may vary by game
- Some games may have quirks in their Z-Machine implementations
