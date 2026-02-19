# MolmoSpaces Adapter

Wraps [MolmoSpaces](https://github.com/allenai/molmospaces) as llenvs MDP environments. MolmoSpaces is a large-scale robot manipulation and navigation research platform built on MuJoCo. It provides 8 benchmark tasks (pick, place, navigate, open/close, etc.) with multi-view cameras, proprioceptive sensors, and continuous robot control.

The adapter provides **temporal abstraction**: each llenvs `step()` = one high-level VLM decision, executed over many physics steps by MolmoSpaces' IK solver and low-level controller.

## Installation

```bash
pip install git+https://github.com/allenai/molmospaces.git
```

MolmoSpaces requires MuJoCo. See the [MolmoSpaces repository](https://github.com/allenai/molmospaces) for full installation instructions.

## Quick Start

```python
from llenvs.adapters.molmospaces import MolmoSpacesAdapter

adapter = MolmoSpacesAdapter()

env = adapter.get_environment("molmospaces:pick", max_steps=30)

state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)          # Task instruction + proprioception
print(len(state.observation.images))     # 2 (front + wrist cameras)
print([t.name for t in state.observation.available_tools])
# ['move_end_effector', 'grasp', 'release']

from llenvs.core import Action
from llenvs.core.tools import ToolCall

# Move the end-effector to the object
call = ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3})
result = env.step(state, Action(tool_calls=(call,)))

# Grasp the object
call = ToolCall(id="c2", name="grasp", arguments={})
result = env.step(result.next_state, Action(tool_calls=(call,)))
```

### With Evaluation Runner

```python
from llenvs.evaluation.runner import run_tool_evaluation

result = run_tool_evaluation(
    environment=env,
    backend=backend,
    task_indices=list(range(50)),
)
print(f"Success rate: {result.success_rate}")
```

## Robot Control Tools

The VLM interacts with the robot through function-calling tools. Each tool call triggers many internal physics steps.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `move_end_effector` | x, y, z, qw?, qx?, qy?, qz? | Move end-effector via IK solver |
| `grasp` | (none) | Close gripper |
| `release` | (none) | Open gripper |
| `navigate_to` | x, y | Move robot base (mobile robots only) |
| `set_joints` | positions (array) | Direct joint control (joint controller only) |

`navigate_to` is only available for mobile robots (e.g., RBY1). `set_joints` is only available when `controller_type="joint"`.

### Tool Execution

Each tool call is executed through MolmoSpaces' IK solver and controller:

1. `move_end_effector` → IK solver computes joint targets → controller steps physics until convergence
2. `grasp`/`release` → gripper actuated directly
3. `navigate_to` → base controller steps physics to target position
4. `set_joints` → controller steps physics to target joint positions

The `max_physics_steps_per_action` parameter (default: 200) limits the physics budget per tool call.

## Observation Space

Each observation includes:

- **Multi-view camera images**: RGB images from configurable cameras (default: front + wrist)
- **Depth images** (optional): Depth maps from each camera
- **Proprioception** (optional): End-effector position/orientation, gripper state, joint positions as text
- **Task instruction**: Natural language description of the task

```python
# Configure observations
env = adapter.get_environment(
    "molmospaces:pick",
    cameras=("front", "wrist", "overhead"),
    image_format="jpeg",
    image_quality=80,
    include_proprioception=True,
    include_depth=True,
)
```

## Camera Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cameras` | `tuple[str, ...]` | `("front", "wrist")` | Camera views to render |
| `image_format` | `str` | `"jpeg"` | `"jpeg"` or `"png"` |
| `image_quality` | `int` | `80` | JPEG quality (1-100) |
| `include_depth` | `bool` | `False` | Include depth maps as additional images |

JPEG is the default since robot camera images are natural photos where JPEG compression is efficient.

## Task Presets

| Task | Max Steps | Description |
|------|-----------|-------------|
| `navigate-to` | 20 | Navigate the robot base to a target location |
| `pick` | 30 | Pick up a specified object |
| `pick-and-place` | 40 | Pick up an object and place it at a target |
| `pick-and-place-next-to` | 40 | Pick up an object and place it next to another |
| `pick-and-place-color` | 40 | Pick up a color-specified object and place it |
| `open` | 30 | Open a container or drawer |
| `close` | 30 | Close a container or drawer |
| `open-door` | 30 | Open a door by manipulating its handle |

## Rewards

### Native Reward

`MolmoSpacesReward` wraps `task.get_reward()`. Uses STEP type for intermediate dense rewards and OUTCOME type at terminal steps.

### Success Reward (opt-in)

`MolmoSpacesSuccessReward` provides binary 0/1 from `task.judge_success()`. OUTCOME type, only at terminal. Available via `extra_rewards`:

```python
from llenvs.adapters.molmospaces import MolmoSpacesSuccessReward

env = adapter.get_environment(
    "molmospaces:pick",
    extra_rewards=(MolmoSpacesSuccessReward(),),
)
```

## Controller Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `controller_type` | `str` | `"ik"` | `"ik"` or `"joint"` |
| `robot_type` | `str` | `"franka"` | `"franka"` (stationary) or `"rby1"` (mobile) |
| `max_physics_steps_per_action` | `int` | `200` | Physics step budget per tool call |
| `is_mobile` | `bool` | `False` | Auto-set from `robot_type` |

```python
# Mobile robot with joint control
env = adapter.get_environment(
    "molmospaces:navigate-to",
    robot_type="rby1",         # auto-enables is_mobile
    controller_type="joint",   # enables set_joints tool
)
```

## Container Deployment

MolmoSpaces requires MuJoCo and GPU rendering. For isolated deployment:

```python
from llenvs.container import create_container_environment, ContainerConfig

config = ContainerConfig(
    runtime="docker",
    image="molmospaces-env:latest",  # custom image with MuJoCo + GPU support
    port=8080,
)
env = create_container_environment(env_config, container_config=config)
```

## Parameters

### `MolmoSpacesAdapter.get_environment()`

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | `"molmospaces:<task>"` (e.g., `"molmospaces:pick"`) |
| `molmospaces_env` | `Any \| None` | Pre-created MolmoSpaces environment |
| `tasks` | `list \| None` | Pre-loaded task list |
| `robot` | `Any \| None` | Pre-created robot instance |
| `ik_solver` | `Any \| None` | Pre-created IK solver |
| `controller` | `Any \| None` | Pre-created controller |
| `cameras` | `tuple[str, ...]` | Camera names to render |
| `image_format` | `str` | Image format ("jpeg" or "png") |
| `image_quality` | `int` | JPEG quality |
| `include_proprioception` | `bool` | Include robot state as text |
| `include_depth` | `bool` | Include depth images |
| `max_steps` | `int \| None` | Max high-level steps |
| `max_physics_steps_per_action` | `int` | Physics budget per tool call |
| `robot_type` | `str` | "franka" or "rby1" |
| `controller_type` | `str` | "ik" or "joint" |
| `extra_rewards` | `tuple[RewardFunction, ...]` | Additional reward functions |

### Environment Capabilities

| Feature | Supported |
|---|---|
| `__len__` | Yes |
| `task_index` | Yes |
| `seed` | Yes |
| `compute_rewards` | Yes (dense + success) |
| `pure_step` | No (MuJoCo state) |
| Tool calling | Yes (IK + controller) |
| Multi-image | Yes (configurable cameras) |
| Depth images | Yes (optional) |
| Proprioception | Yes (optional text) |
| `Scorer` / `DatasetProvider` | No (multi-turn) |

## Hidden State

`MolmoSpacesHidden` contains:

| Field | Type | Description |
|---|---|---|
| `task_index` | `int` | Task index |
| `task_name` | `str` | Task name |
| `episode_step` | `int` | High-level step count |
| `physics_steps` | `int` | Total physics steps this episode |
| `last_action` | `str \| None` | Last tool action name |
| `ee_position` | `tuple[float, ...]` | End-effector position |
| `ee_orientation` | `tuple[float, ...]` | End-effector orientation (quaternion) |
| `gripper_state` | `str` | "open" or "closed" |
| `joint_positions` | `tuple[float, ...]` | Current joint angles |
| `cumulative_reward` | `float` | Cumulative episode reward |
| `task_instruction` | `str` | Natural language instruction |
| `success` | `bool` | Task completion status |

## Limitations

- **Requires MuJoCo**: MolmoSpaces depends on MuJoCo for physics simulation
- **GPU rendering**: Camera rendering benefits from GPU acceleration
- **Non-pure step**: MuJoCo state is mutable; environments use `_StateContinuityTracker`
- **Temporal abstraction**: Each high-level step may execute hundreds of physics steps; IK failures result in tool errors
- **No native answer extraction**: Robot manipulation uses tool-action evaluation
