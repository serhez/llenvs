# AlfWorld

The AlfWorld adapter wraps [AlfWorld](https://github.com/alfworld/alfworld) household environments for text-based LLM agents. Agents navigate rooms and manipulate objects to complete everyday tasks like "put a clean mug on the desk".

## Installation

```bash
pip install alfworld
# or
pip install llenvs[alfworld]
```

Follow AlfWorld's setup instructions to download game data:

```bash
export ALFWORLD_DATA=<path_to_data>
alfworld-download
```

## Quick Start

```python
from llenvs.adapters import AlfWorldAdapter
from llenvs.core import Action

adapter = AlfWorldAdapter()
env = adapter.get_environment("alfworld:eval_out_of_distribution")

state, info = env.reset(options={"task_index": 0})
print(info["objective"])  # e.g., "put a clean mug on desk 1"
print(state.observation.prompt)

result = env.step(state, Action(text="go to shelf 1"))
print(result.next_state.observation.messages[-1]["content"])
```

## Task Types

AlfWorld has 6 task types, each requiring a different sequence of interactions:

| ID | Task Type | Description |
|----|-----------|-------------|
| 1 | `pick_and_place_simple` | Pick up an object and place it somewhere |
| 2 | `look_at_obj_in_light` | Examine an object under a light source |
| 3 | `pick_clean_then_place_in_recep` | Clean an object then place it |
| 4 | `pick_heat_then_place_in_recep` | Heat an object then place it |
| 5 | `pick_cool_then_place_in_recep` | Cool an object then place it |
| 6 | `pick_two_obj_and_place` | Pick up two objects and place them |

Filter to specific task types:

```python
# Only pick-and-place tasks
env = adapter.get_environment(task_types=[1])

# Pick-and-place + cleaning tasks
env = adapter.get_environment(task_types=[1, 3])
```

## Action Format

Actions are free-form text commands understood by the TextWorld backend:

| Action | Description |
|--------|-------------|
| `go to {recep}` | Navigate to a receptacle |
| `take {obj} from {recep}` | Pick up an object |
| `put {obj} in/on {recep}` | Place an object |
| `open {recep}` / `close {recep}` | Open or close a container |
| `use {obj}` | Use an object (e.g., a lamp) |
| `clean {obj} with {recep}` | Clean an object |
| `heat {obj} with {recep}` | Heat an object |
| `cool {obj} with {recep}` | Cool an object |
| `examine {obj/recep}` | Look at something |
| `inventory` | Check held items |
| `look` | Look around the room |

## Admissible Commands

AlfWorld provides a list of valid commands at each step. By default, these are included in the observation text and always stored in the hidden state:

```python
# Included in observation (default)
env = adapter.get_environment(include_admissible_commands=True)

# Hidden from observation, but still in hidden state
env = adapter.get_environment(include_admissible_commands=False)
state, _ = env.reset(options={"task_index": 0})
print(state.hidden.admissible_commands)  # always available
```

## Adapter Configuration

### Splits

Three data splits are available, specified via the environment name:

```python
env = adapter.get_environment("alfworld:train")
env = adapter.get_environment("alfworld:eval_in_distribution")
env = adapter.get_environment("alfworld:eval_out_of_distribution")  # default
```

### Config

Config resolution priority: packaged adapter defaults < `config` dict < `config_path`.

```python
# Use llenvs' packaged ALFWorld defaults
env = adapter.get_environment()

# Override parts of the default config
env = adapter.get_environment(config={"dataset": {"num_eval_games": 100}})

# Config from YAML file
env = adapter.get_environment(config_path="/path/to/config.yaml")
```

Use `name=` or `split=` to choose the split. The adapter always applies the resolved
split after loading config overrides.

### Parameters

```python
env = adapter.get_environment(
    name="alfworld:eval_out_of_distribution",
    max_steps=50,  # default: 50
    include_admissible_commands=True,  # default: True
    include_objective_in_obs=True,  # default: True
    task_types=[1, 2, 3],  # filter task types (None = all)
    answer_extractor=my_extractor,  # extract command from raw model output
    invalid_action_text="[invalid action]",  # default history placeholder
    invalid_action_observation=None,  # custom reminder prefix
    advance_on_invalid="__invalid_action_noop__",  # default sentinel command
    prompts={  # override prompt templates
        "objective_prefix": "Goal: {objective}",
        "admissible_commands_prefix": "Valid actions:",
    },
)
```

### Answer Extractor

TextWorld requires exact command strings (e.g., `go to shelf 1`). When a model produces free-form text with reasoning or formatting around the command, pass an `answer_extractor` to strip the noise before forwarding to TextWorld:

```python
from llenvs.core.extraction import TagBasedExtractor, CleanedExtractor, resolve_cleaners

extractor = CleanedExtractor(
    TagBasedExtractor(tag_name="answer"),
    pre_cleaners=resolve_cleaners(["strip_thinking_tokens"], kind="pre"),
)
env = adapter.get_environment(answer_extractor=extractor)
```

When an extractor is provided:
- The extracted command is passed to TextWorld for stepping and stored in `hidden.trajectory` for replay
- The assistant turn in the conversation history contains the extracted command (not raw text), keeping the history clean
- `StepResult.extracted_action` and `StepResult.resolved_action` are set to the extracted command
- If extraction fails (returns `None`), the adapter executes the configured invalid-action command (by default `__invalid_action_noop__`), prepends an invalid-format reminder to the observation, stores the sentinel in replay state, and shows `[invalid action]` in history display

## Rewards

AlfWorld provides binary sparse rewards:

| Reward | Type | Value |
|--------|------|-------|
| `task_completion` | `OUTCOME` | `1.0` if task completed, `0.0` otherwise |

```python
result = env.step(state, Action(text="put mug 1 in/on desk 1"))
signal = result.rewards.by_name("task_completion")
print(signal.reward)  # 1.0 if won
```

Add extra reward signals:

```python
env = AlfWorldEnvironment(
    game_files=game_files,
    config=config,
    extra_rewards=(my_custom_reward,),
)
```

## Hidden State

```python
@dataclass(frozen=True)
class AlfWorldHidden:
    task_index: int  # index into game_files
    task_type: str  # e.g., "pick_and_place_simple"
    objective: str  # extracted task objective
    game_file: str  # path to current game file
    episode_step: int  # current step count
    last_action: str | None  # last action taken
    admissible_commands: tuple[str, ...]  # valid commands at this step
    trajectory: tuple[str, ...] = ()  # actions taken to reach this state
```

## Using with TrajectoryRunner

```python
from llenvs.adapters import AlfWorldAdapter
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams
from llenvs.inference.backends import OpenAIBackend

adapter = AlfWorldAdapter()
env = adapter.get_environment(
    "alfworld:eval_out_of_distribution",
    max_steps=50,
)

backend = OpenAIBackend(model="gpt-4o")
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
)

result = runner.run_trajectory(task_index=0)
print(f"Won: {result.trajectory.final_state.metadata.info['won']}")
```

## Visual Mode (AI2-THOR)

AlfWorld can render ego-centric RGB frames via AI2-THOR's 3D simulator. When `visual=True`, each observation includes an image alongside the text:

```python
adapter = AlfWorldAdapter()
env = adapter.get_environment(
    "alfworld:eval_out_of_distribution",
    visual=True,
)

state, info = env.reset(options={"task_index": 0})
print(len(state.observation.images))  # 1 (ImageContent with RGB frame)
print(state.observation.prompt)  # Text observation still present
```

Visual mode requires:
- `ai2thor` package (GPU + OpenGL + ~2GB download)
- A VLM backend for meaningful evaluation

Text mode (default) is unchanged — `visual=False` uses the lightweight `AlfredTWEnv` text-only backend with no additional dependencies.

```python
# Text-only (default) — no ai2thor needed
env = adapter.get_environment()

# Visual — includes RGB frames as ImageContent
env = adapter.get_environment(visual=True)
```

Images flow through the full multimodal pipeline (see [Multimodal Observations](../guides/multimodal.md)).

## Pure Step (Replay-Based)

AlfWorld uses `pure_step=True` — stepping is stateless and supports branching. Each `step()` call creates a fresh TextWorld environment, replays the action trajectory stored in the hidden state, and applies the new action. This enables:

- **Branching**: Step the same state with different actions to explore multiple paths
- **Replay**: Any saved state can be stepped at any time, in any order
- **Compatibility**: Works with `DirectStrategy` for branching workflows

```python
state, _ = env.reset(options={"task_index": 0})

# Branch from the same state
result_a = env.step(state, Action(text="go to shelf 1"))
result_b = env.step(state, Action(text="go to desk 1"))

# Both succeed — different observations, different trajectories
print(result_a.next_state.hidden.trajectory)  # ("go to shelf 1",)
print(result_b.next_state.hidden.trajectory)  # ("go to desk 1",)
```

The `trajectory` field on `AlfWorldHidden` stores the complete action history needed for replay. Observations use demangled entity names via AlfWorld's `AlfredDemangler` wrapper for readable text.

## Troubleshooting

### "ALFWorld data directory does not exist"

The adapter validates that the game data directory exists before loading environments. If you see this error, run:

```bash
alfworld-download
```

This downloads ~3.5 GB of game data into `$ALFWORLD_DATA` (defaults to `~/.cache/alfworld/`). The adapter's error message includes the resolved `ALFWORLD_DATA` path and the specific data directory it expected.

### "No ALFWorld games found"

The data directory exists but contains no valid game files for the requested split/task-type combination. The error message includes the scanned data path. Verify that the directory has the expected layout (e.g., `json_2.1.1/valid_unseen/` for `eval_out_of_distribution`).

## Limitations

- No seed support — AlfWorld games are deterministic per game file
- Requires AlfWorld data download (`alfworld-download`) before use
- Visual mode requires `ai2thor` with GPU/OpenGL support
