# LLEnvs

**MDP-style environments for LLM evaluation and training research.**

LLEnvs wraps evaluation benchmarks — reasoning tasks, math datasets, multi-turn games, e-commerce simulations — in a unified `Environment` protocol inspired by Gymnasium. Every benchmark becomes a stateless MDP with typed observations, actions, and multi-signal rewards, so you can run evaluations, build RL training loops, or do fine-grained reasoning analysis with the same interface.

```python
from llenvs.adapters import create_reasoning_gym_environment
from llenvs.core import TextAction

env = create_reasoning_gym_environment("leg_counting", size=100, seed=42)
state, _ = env.reset(options={"task_index": 0})

print(state.observation.prompt)
# "A farmer has 3 cows and 2 chickens. How many legs in total?"

result = env.step(state, TextAction(text="<answer>16</answer>"))
print(result.rewards.total)  # 1.0
```

## Why LLEnvs?

**Evaluation benchmarks weren't designed for training.** Most come as static datasets with a scoring script. If you want to use them for RL, MCTS, process reward models, or anything beyond single-pass inference, you end up writing glue code for every benchmark. LLEnvs provides the missing abstraction layer:

- **`reset()` / `step()` / `compute_rewards()`** — the same interface whether you're evaluating on GSM8K, playing a GEM game, or navigating WebShop
- **Stateless `step()`** — pass the state in, get a new state out. Branch, checkpoint, and explore in parallel without worrying about internal mutation
- **Observation / hidden split** — models see `state.observation` (the prompt); ground truth lives in `state.hidden` (never leaked to the model)
- **Multi-signal rewards** — each step produces a `RewardBundle` with named, typed signals (outcome, format, process, step) rather than a single scalar

## Adapters

LLEnvs connects to multiple benchmark sources through adapters:

| Adapter | What it wraps | Examples |
|---------|--------------|----------|
| **ReasoningGym** | Procedural reasoning tasks | `leg_counting`, `sudoku`, `arc_1d` |
| **HuggingFace** | Any HF dataset with Q/A columns | AIME 2024, GSM8K, MATH |
| **GEM** | Multi-turn games and benchmarks | GuessTheNumber, 20 Questions, GSM8K with tools |
| **WebShop** | E-commerce product search | Navigate, search, purchase |

```python
from llenvs.core.registry import environment_registry

# All adapters, one interface
env = environment_registry.get(name="leg_counting", adapter="reasoning_gym", size=100)
env = environment_registry.get(name="HuggingFaceH4/aime_2024", adapter="huggingface")
env = environment_registry.get(name="game:GuessTheNumber-v0", adapter="gem")
```

## Inference Backends

Run models locally or through APIs — same evaluation code either way:

```python
from llenvs.inference.backends import OpenAIBackend, VLLMBackend, AnthropicBackend

backend = OpenAIBackend(model="gpt-4o")
backend = VLLMBackend(model_path="meta-llama/Llama-3.1-8B-Instruct")
backend = AnthropicBackend(model="claude-sonnet-4-20250514")
```

Backends support chat generation, batching, logprobs, prefix continuation, and tool/function calling where available.

## Evaluation

Run batched evaluations with a few lines:

```python
from llenvs.evaluation import run_evaluation
from llenvs.inference import SamplingParams

result = run_evaluation(
    environment=env,
    backend=backend,
    num_tasks=100,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=2048),
    system_prompt="Think step by step. Put your answer in <answer>...</answer> tags.",
)

print(f"Accuracy: {result.success_rate:.1%}")
print(f"Mean reward: {result.mean_reward:.3f}")
```

Or use YAML configuration with the CLI:

```yaml
# config.yaml
environments:
  - name: leg_counting
    adapter: reasoning_gym
    size: 100
    answer_extractor: tag_based

model:
  backend: openai
  model: gpt-4o

system_prompt: general_reasoning
output_dir: ./results
```

```bash
llenvs run config.yaml
```

## Tools and Function Calling

First-class support for tool-using agents, including MCP server integration:

```python
from llenvs.adapters import create_gem_tool_environment
from llenvs.evaluation import ToolTrajectoryRunner

env = create_gem_tool_environment("math:GSM8K", tool_types=("python",))

state, _ = env.reset(options={"task_index": 0})
print([t.name for t in state.observation.available_tools])
# ['python', 'submit_answer']
```

## Segmentation

Break responses into segments for per-step analysis — useful for process reward models, tree search, and reasoning trace studies:

```python
from llenvs.core import SegmentedEnvironment, SentenceSegmenter

env = SegmentedEnvironment(base_env, SentenceSegmenter())
state, _ = env.reset(options={"task_index": 0})

# Replay a complete response to get per-step rewards
results = env.replay(state, "First, count the animals. There are 2 dogs. ...")

# Or generate segment-by-segment with intervention
from llenvs.evaluation import SegmentedTrajectoryRunner
runner = SegmentedTrajectoryRunner(environment=env, backend=backend, ...)
result = runner.run_trajectory(task_index=0, step_callback=my_prm_callback)
```

## Installation

```bash
pip install llenvs[openai,reasoning-gym]   # API + reasoning tasks
pip install llenvs[vllm,huggingface]       # Local inference + HF datasets
pip install llenvs[all]                    # Everything
```

See the [full documentation](docs/README.md) for detailed guides on environments, prompts, tools, segmentation, evaluation, backends, and configuration.

## Documentation

Install dev dependencies:

```bash
uv pip install -e ".[dev]"
```

Serve locally with live reload:

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000`.

Build a static site into `site/`:

```bash
uv run mkdocs build
```
