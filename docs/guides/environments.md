# Environments

llenvs provides a unified interface to multiple environment sources through adapters.

## Available Adapters

| Adapter | Package | Environments |
|---------|---------|--------------|
| `reasoning_gym` | `reasoning-gym` | Procedural reasoning tasks |
| `huggingface` | `datasets` | AIME, GSM8K, MATH, etc. |
| `gem` | `gem-llm` | Multi-turn games, math, QA |
| `webshop` | `webshop` | E-commerce simulation |
| `agentgym` | `agentenv` | 15 multi-turn agent environments |
| `verifiers` | `verifiers` | Single-turn and tool envs with rubric scoring |
| `openenv` | `openenv-core` | Session-based server environments with MCP tools |

## Using the Registry

The environment registry provides adapter-based lookup:

```python
from llenvs.core.registry import environment_registry

# List all registered adapters
adapters = environment_registry.list_adapters()
print(f"Available adapters: {adapters}")
# ["reasoning_gym", "huggingface", "gem", "webshop", "agentgym", "verifiers", "openenv"]

# List all environments across all adapters
all_envs = environment_registry.list_environments()
for adapter, name in all_envs[:5]:
    print(f"  {adapter}/{name}")

# Get environment with configuration
env = environment_registry.get(
    name="leg_counting",
    adapter="reasoning_gym",
    size=100,
    seed=42,
)

# Check if environment exists
if ("reasoning_gym", "sudoku") in environment_registry:
    print("Sudoku is available!")
```

## Working with Adapters Directly

```python
from llenvs.adapters import ReasoningGymAdapter

adapter = ReasoningGymAdapter()

# List available environments
envs = adapter.list_environments()
print(envs)

# Get environment info without creating it
info = adapter.get_environment_info("sudoku")
print(info)
# {"name": "sudoku", "adapter": "reasoning_gym", "type": "single_turn", ...}

# Create environment
env = adapter.get_environment("sudoku", size=50)
```

## Environment Types

### Single-Turn

Agent provides one response, receives final reward:

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action

env = environment_registry.get(name="leg_counting", adapter="reasoning_gym")
state, _ = env.reset(options={"task_index": 0})

# One action completes the episode
action = Action(text="<answer>8</answer>")
result = env.step(state, action)
assert result.terminated  # Episode done
```

### Multi-Turn

Agent interacts over multiple steps:

```python
from llenvs.core.registry import environment_registry

env = environment_registry.get(name="game:GuessTheNumber-v0", adapter="gem")
state, _ = env.reset(seed=42)

while not state.metadata.is_terminal:
    action = Action(text="50")
    result = env.step(state, action)
    state = result.next_state
    print(f"Response: {state.observation.prompt}")
```

## Common Operations

### Reset with Options

```python
state, info = env.reset(
    seed=42,                              # Reproducibility
    options={
        "task_index": 5,                  # Select specific task
        "episode_id": "custom-id-123",    # Override episode ID
    },
)
```

### Access Environment Spec

```python
spec = env.spec
print(f"Name: {spec.name}")
print(f"Adapter: {spec.adapter}")
print(f"Max steps: {spec.max_steps}")
print(f"Multi-turn: {spec.is_multi_turn}")
```

### Iterate Tasks

```python
for i in range(len(env)):
    state, info = env.reset(options={"task_index": i})
    # Process task...
```

## Working with Trajectories

Record and replay episodes:

```python
from llenvs.core import Trajectory, Transition

# Initialize trajectory
state, _ = env.reset(options={"task_index": 0})
trajectory = Trajectory.create(state)

# Run trajectory and record
while not trajectory.is_terminal:
    action = Action(text=model_response)
    result = env.step(trajectory.current_state, action)

    transition = Transition(
        state=trajectory.current_state,
        action=action,
        next_state=result.next_state,
        rewards=result.rewards,
        info={},
    )
    trajectory.add_transition(transition)

# Access trajectory data
print(f"Total reward: {trajectory.total_reward}")
print(f"Steps: {len(trajectory)}")
```

### Checkpointing and Branching

```python
# Create checkpoint before critical decision
trajectory.checkpoint("before_answer")

# Continue one path
trajectory.add_transition(transition_a)

# Branch to try alternative
alt_trajectory = trajectory.branch("before_answer")
alt_trajectory.add_transition(transition_b)

# Compare outcomes
print(f"Path A: {trajectory.total_reward}")
print(f"Path B: {alt_trajectory.total_reward}")
```

## Next Steps

- **[ReasoningGym Adapter](../adapters/reasoning-gym.md)**
- **[HuggingFace Adapter](../adapters/huggingface.md)**
- **[GEM Adapter](../adapters/gem.md)**
- **[WebShop Adapter](../adapters/webshop.md)**
- **[AgentGym Adapter](../adapters/agentgym.md)**
- **[Verifiers Adapter](../adapters/verifiers.md)**
- **[OpenEnv Adapter](../adapters/openenv.md)**
