# GEM

The GEM (General Experience Maker) adapter provides access to ~40+ environments for training and evaluating LLMs, including native multi-turn games and single-turn benchmarks.

## Installation

```bash
pip install gem-llm
# or
pip install llenvs[gem]
```

## Quick Start

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action

# Create a multi-turn game
env = environment_registry.get(name="game:GuessTheNumber-v0", adapter="gem")

# Create a single-turn math problem
env = environment_registry.get(name="math:GSM8K", adapter="gem")
```

## Multi-Turn Games

GEM provides native multi-turn games where the model interacts across multiple steps:

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action

# GuessTheNumber - binary search game
env = environment_registry.get(name="game:GuessTheNumber-v0", adapter="gem")
state, info = env.reset(seed=42)

print(f"Game: {state.observation.prompt}")
# "I'm thinking of a number between 1 and 100..."

# Play the game
while not state.metadata.is_terminal:
    # Model generates a guess
    action = Action(text="50")
    result = env.step(state, action)

    # Step feedback is in messages; prompt stays as initial instructions
    print(f"Response: {result.next_state.observation.messages[-1]['content']}")
    # "50 is too high." or "Correct!"

    state = result.next_state

print(f"Final reward: {result.rewards.total}")
```

### Available Multi-Turn Games

| Environment | Description |
|-------------|-------------|
| `game:GuessTheNumber-v0` | Binary search number guessing |
| `game:Sudoku-v0` | Sudoku puzzle solving |
| `game:Wordle-v0` | Word guessing game |
| `game:Mastermind-v0` | Code-breaking game |
| `game:Minesweeper-v0` | Mine avoidance |
| `game:Game2048-v0` | Tile merging puzzle |
| `game:Hangman-v0` | Letter guessing |
| `game:TowerOfHanoi-v0` | Disk stacking puzzle |

Most games have difficulty variants: `-easy`, `-hard`, `-random`.

## Single-Turn Benchmarks

GEM also wraps standard benchmarks as single-turn environments:

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action

# Math benchmark
env = environment_registry.get(name="math:GSM8K", adapter="gem")
state, _ = env.reset(options={"task_index": 0})

print(f"Problem: {state.observation.prompt}")

# Model solves the problem
action = Action(text="Let me solve this step by step... <answer>42</answer>")
result = env.step(state, action)

print(f"Correct: {result.rewards.by_name('correctness').reward == 1.0}")
```

### Available Single-Turn Benchmarks

**Math:**
- `math:GSM8K` - Grade school math
- `math:MATH500` - Competition math
- `math:AIME24` - AIME 2024 problems
- `math:AMC` - AMC competition
- `math:OlympiadBench` - Olympiad problems

**Code:**
- `code:CodeContest` - Programming challenges
- `code:Taco8k` - Code generation
- `code:PrimeIntellect15k` - Reasoning + code

**QA:**
- `qa:NaturalQuestions` - Open-domain QA
- `qa:HotpotQA` - Multi-hop reasoning
- `qa:TriviaQA` - Trivia questions
- `qa:PopQA` - Popular knowledge QA

## Tool-Enabled Environments

GEM environments like `math:*` and `qa:*` support tools (Python execution, search). Use `environment_registry.get()` with `tool_types` for structured function calling:

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action, ToolCall

# Create tool-enabled environment
env = environment_registry.get(
    name="math:GSM8K",
    adapter="gem",
    tool_types=("python",),  # Enable Python execution
    max_steps=10,
)

# Reset returns Observation with available tools
state, _ = env.reset(options={"task_index": 0})
print(f"Tools: {[t.name for t in state.observation.available_tools]}")
# ['python', 'submit_answer']

# Use Python tool to compute
call = ToolCall(id="1", name="python", arguments={"code": "print(0.15 * 80)"})
action = Action(tool_calls=(call,))
result = env.step(state, action)

print(f"Output: {result.info['tool_results'][0].output}")
# '12.0'

# Submit final answer
call = ToolCall(id="2", name="submit_answer", arguments={"answer": "12"})
action = Action(tool_calls=(call,))
result = env.step(result.next_state, action)

print(f"Correct: {result.info['gem_reward'] == 1.0}")
```

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `python` | Execute Python code | `code: str` |
| `search` | Search for information | `query: str` |
| `submit_answer` | Submit final answer (terminal) | `answer: str` |

### QA Environments with Search

```python
env = environment_registry.get(
    name="qa:HotpotQA",
    adapter="gem",
    tool_types=("search",),
    search_url="http://localhost:8000/retrieve",
    search_topk=3,
)

state, _ = env.reset()

# Search for information
call = ToolCall(id="1", name="search", arguments={"query": "capital of France"})
action = Action(tool_calls=(call,))
result = env.step(state, action)
# Tool result contains search results
```

### Using with TrajectoryRunner

```python
from llenvs.core.registry import environment_registry
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

env = environment_registry.get(name="math:GSM8K", adapter="gem", tool_types=("python",))
backend = OpenAIBackend(model="gpt-4o")

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Use Python to solve math problems. Submit your final answer.",
)

result = runner.run_trajectory(task_index=0)
print(f"Success: {result.success}")
```

## Using the Adapter Directly

```python
from llenvs.adapters import GemAdapter

adapter = GemAdapter()

# List all available environments
envs = adapter.list_environments()
print(f"Available: {len(envs)} environments")

# Get environment info
info = adapter.get_environment_info("game:Sudoku-v0")
print(info)
# {"name": "game:Sudoku-v0", "adapter": "gem", "type": "multi_turn", ...}

# Create with custom options
env = adapter.get_environment(
    "game:Wordle-v0",
    max_steps=6,
)

# Add optional extra rewards (e.g., format compliance)
from llenvs.core.reward import FormatReward
from llenvs.core.extraction import TagBasedExtractor
env_with_format = adapter.get_environment(
    "game:Wordle-v0",
    extra_rewards=(FormatReward(TagBasedExtractor()),),
)
```

## GEM vs ReasoningGym

GEM internally wraps reasoning-gym, so some environments are available through both adapters:

```python
from llenvs.core.registry import environment_registry

# Via GEM adapter
env1 = environment_registry.get(name="reasoning_gym:leg_counting", adapter="gem")

# Via ReasoningGym adapter (native)
env2 = environment_registry.get(name="leg_counting", adapter="reasoning_gym")
```

Use whichever adapter fits your workflow. GEM provides a unified interface across all environment types.

## State Snapshotting and Branching

All GEM environments have `pure_step=True` and support `DirectStrategy` branching:

- **Native state envs** (with `get_state()`/`set_state()`): Use the native methods for efficient snapshotting
- **Stateless envs** (without `get_state()`/`set_state()`): Use a transparent `deepcopy` fallback that captures and restores the underlying env's `__dict__`

Both approaches guarantee **pure function semantics**: `step(state, action)` always produces the same result for the same inputs.

```python
from llenvs import BranchManager

env = adapter.get_environment("game:GuessTheNumber-v0")

# All GEM envs support DirectStrategy — no special configuration needed
with BranchManager.create(env) as mgr:
    state, _ = env.reset(seed=42)
    # ... checkpoint and branch as usual
```

## Hidden State

```python
@dataclass(frozen=True)
class GemHidden:
    env_id: str
    gem_state: Any  # Frozen tuple (native state) or dict (deepcopy fallback)
    task_index: int
    is_multi_turn: bool
    episode_step: int
```
