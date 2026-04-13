# WebShop

[WebShop](https://github.com/princeton-nlp/WebShop) (Yao et al., NeurIPS 2022) is a simulated e-commerce environment with 1.18M real products and 12K instructions. Agents navigate a website to find and purchase products matching natural language instructions.

## Installation

```bash
pip install webshop
# Then run setup to download product data and build search index
# See: https://github.com/princeton-nlp/WebShop#setup
```

## Quick Start

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action

# Create environment (use num_products=1000 for fast preview)
env = environment_registry.get(
    name="webshop:text_rich",
    adapter="webshop",
    max_steps=15,
    num_products=1000,
)

# Reset with a specific task
state, info = env.reset(options={"task_index": 0})
print(f"Instruction: {info['instruction']}")
# "Find a red wireless headphone under $50"

# Search for products
action = Action(text="search[red wireless headphones]")
result = env.step(state, action)
print(result.next_state.observation.messages[-1]["content"])
# Shows search results with clickable products

# Click on a product
action = Action(text="click[Product 1 - Red Headphones $45]")
result = env.step(result.next_state, action)
# Conversation history accumulates in messages (2 per turn)
print(f"History: {len(result.next_state.observation.messages)} messages")

# Complete purchase
action = Action(text="click[Buy Now]")
result = env.step(result.next_state, action)
print(f"Reward: {result.info['webshop_reward']}")
# Reward based on how well purchase matches instruction
```

## Action Format

WebShop uses text-based actions (not structured tool calls):

| Action | Description | Example |
|--------|-------------|---------|
| `search[keywords]` | Search for products | `search[red wireless headphones]` |
| `click[element]` | Click on a page element | `click[Buy Now]` |

## Observation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `text_rich` | Tagged buttons `[button]...[button_]` and clickables | LLM agents (recommended) |
| `text` | Simple text with `[SEP]` separators | Simpler models |
| `html` | Raw HTML | Web-aware models |

## Using the Adapter Directly

```python
from llenvs.adapters import WebShopAdapter

adapter = WebShopAdapter()

# Get environment with custom configuration
env = adapter.get_environment(
    name="webshop:text_rich",
    observation_mode="text_rich",
    max_steps=15,
    num_products=None,  # Use full dataset
    human_goals=True,   # Use human-written goals
    invalid_action_text="[invalid action]",  # default history placeholder
    invalid_action_observation=None,  # custom reminder prefix
    advance_on_invalid="__invalid_action_noop__",  # default sentinel action
)

# Check environment info
info = adapter.get_environment_info("webshop:text_rich")
print(info)
# {"name": "webshop:text_rich", "adapter": "webshop", "type": "multi_turn"}
```

## Running with TrajectoryRunner

```python
from llenvs.core.registry import environment_registry
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

env = environment_registry.get(name="webshop:text_rich", adapter="webshop", max_steps=15)
backend = OpenAIBackend(model="gpt-4o")

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
    system_prompt="""You are a shopping assistant. Navigate the website to find
and purchase products matching the given instruction.

Use these actions:
- search[keywords] to search for products
- click[element] to click on buttons, products, or options

Always click "Buy Now" when you find a matching product.""",
)

result = runner.run_trajectory(task_index=0)
print(f"Success: {result.success}")
print(f"Reward: {result.total_reward}")
```

## Rewards

WebShop provides a graded reward based on attribute matching:

| Reward | Description |
|--------|-------------|
| `webshop_reward` | 0.0-1.0 based on how well the purchased product matches the instruction |

The reward considers:
- Price constraints
- Category match
- Required attributes (color, size, etc.)
- Option selections

## Hidden State

```python
@dataclass(frozen=True)
class WebShopHidden:
    instruction: str              # Shopping instruction
    session_id: str              # WebShop session ID
    task_index: int
    episode_step: int
    last_action: str | None      # Last action taken
    available_actions: tuple[str, ...]  # Valid actions
```

## Multi-Turn Nature

WebShop is a multi-turn environment. After each action, the agent receives updated observations reflecting the current page state:

1. **Search results page**: Shows product list
2. **Product page**: Shows product details and options
3. **Purchase confirmation**: Shows reward

The agent must navigate multiple pages to complete the task. Conversation history accumulates in `observation.messages` — each turn appends an assistant message (the action) and a user message (the formatted observation). The initial observation stays in `observation.prompt`.

When an `answer_extractor` is configured and extraction fails, WebShop no longer
forwards the raw model text to the environment. By default it executes the fixed
sentinel `__invalid_action_noop__`, prepends an invalid-format reminder to the
observation, and stores `[invalid action]` in history display while keeping the
hidden replay state aligned with the real step that was executed.
