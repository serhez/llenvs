# LLEnvs

**MDP-style environments for LLM evaluation and training research.**

LLEnvs wraps evaluation benchmarks — reasoning tasks, math datasets, multi-turn games, tool-using agents, e-commerce simulations — in a unified `Environment` protocol inspired by Gymnasium. Every benchmark becomes a stateless MDP with typed observations, actions, and multi-signal rewards, so you can run evaluations, build RL training loops, or do fine-grained reasoning analysis with the same interface.

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action

env = environment_registry.get(name="leg_counting", adapter="reasoning_gym", size=100, seed=42)
state, _ = env.reset(options={"task_index": 0})

print(state.observation.prompt)
# "A farmer has 3 cows and 2 chickens. How many legs in total?"

result = env.step(state, Action(text="<answer>16</answer>"))
print(result.rewards.total)  # 1.0
```

## Why LLEnvs?

**Evaluation benchmarks weren't designed for training.** Most come as static datasets with a scoring script. If you want to use them for RL, MCTS, process reward models, or anything beyond single-pass inference, you end up writing glue code for every benchmark. LLEnvs provides the missing abstraction layer:

- **`reset()` / `step()` / `compute_rewards()`** — the same interface whether you're evaluating on GSM8K, playing a GEM game, or navigating WebShop
- **Stateless `step()`** — pass the state in, get a new state out. Branch, checkpoint, and explore in parallel without worrying about internal mutation
- **Observation / hidden split** — models see `state.observation` (the prompt); ground truth lives in `state.hidden` (never leaked to the model)
- **Multi-signal rewards** — each step produces a `SignalBundle` with named, typed signals carrying numeric rewards, textual feedback, or both

## Adapters

LLEnvs connects to multiple benchmark sources through adapters:

| Adapter | What it wraps | Examples |
|---------|--------------|----------|
| **ReasoningGym** | Procedural reasoning tasks | `leg_counting`, `sudoku`, `arc_1d` |
| **HuggingFace** | Any HF dataset with Q/A columns | AIME 2024, GSM8K, MATH |
| **GEM** | Multi-turn games and benchmarks | GuessTheNumber, Wordle, Mastermind |
| **WebShop** | E-commerce product search | Navigate, search, purchase |
| **AlfWorld** | Text-based household tasks | pick & place, clean, heat, cool, examine |
| **Jericho** | Classic interactive fiction games | Zork, Hitchhiker's Guide, Detective |
| **AgentGym** | 15 multi-turn agent environments | ALFWorld, BabyAI, SciWorld, TextCraft |
| **Verifiers** | Verifiers library environments | Rubric-scored tasks, tool-enabled envs |
| **OpenEnv** | OpenEnv session-based environments | MCP tool servers, generic clients |

```python
from llenvs.core.registry import environment_registry

# All adapters, one interface
env = environment_registry.get(name="leg_counting", adapter="reasoning_gym", size=100)
env = environment_registry.get(name="HuggingFaceH4/aime_2024", adapter="huggingface")
env = environment_registry.get(name="game:GuessTheNumber-v0", adapter="gem")
env = environment_registry.get(name="alfworld:eval_out_of_distribution", adapter="alfworld")
```

## Inference Backends

Run models locally or through APIs — same evaluation code either way:

```python
from llenvs.inference.backends import OpenAIBackend, VLLMBackend, HuggingFaceBackend, AnthropicBackend

backend = VLLMBackend(model_path="meta-llama/Llama-3.1-8B-Instruct")
backend = HuggingFaceBackend(model_path="meta-llama/Llama-3.1-8B-Instruct")
backend = OpenAIBackend(model="gpt-4o")
backend = AnthropicBackend(model="claude-sonnet-4-20250514")
```

Backends support chat generation, batching, logprobs, prefix continuation, and tool/function calling. API backends run concurrent async requests with configurable concurrency for parallel batch processing.

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
    batch_size=32,  # process in chunks for memory efficiency
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
llenvs run config.yaml --log console,file
```

Evaluation supports logging to console, JSONL files, and Weights & Biases for tracking runs.

## Tools and Function Calling

First-class support for tool-using agents across multiple adapters:

```python
from llenvs.core.registry import environment_registry

# GEM environments with Python execution and search tools
env = environment_registry.get(name="math:GSM8K", adapter="gem", tool_types=("python",))

state, _ = env.reset(options={"task_index": 0})
print([t.name for t in state.observation.available_tools])
# ['python', 'submit_answer']
```

Tool definitions can be auto-generated from Python functions via `ToolDefinition.from_callable()`. For open-source models without native function calling, the `HermesToolCallParser` enables tool use through text-based `<tool_call>` XML format. Tool validity and efficiency are automatically monitored as reward signals.

## Containers

Run environments in isolated Docker containers or subprocesses:

```python
from llenvs.container import create_container_environment, ContainerConfig

config = ContainerConfig(runtime="docker", image="llenvs-env:latest", port=8080)
env = create_container_environment(env_config, container_config=config)

# Same interface — reset/step/compute_rewards proxied over HTTP
state, _ = env.reset(options={"task_index": 0})
```

Or start a server directly:

```bash
python -m llenvs.container --config '{"name": "leg_counting", "adapter": "reasoning_gym"}' --port 8080
```

## RL Training Integrations

Use environments directly in RL training pipelines:

```python
from llenvs import Scorer, DatasetProvider

# Score model responses against any environment
scorer = Scorer(env)
result = scorer.score(task_index=0, response="The answer is 42")
print(result.reward, result.correct)

# Provide prompts and ground truth for dataset construction
provider = DatasetProvider(env)
for item in provider:
    print(item.prompt, item.ground_truth)
```

Framework-specific adapters for **veRL**, **TRL**, and **OpenRLHF** provide ready-made reward functions and dataset helpers. A `TrajectoryMasker` converts multi-turn trajectories into per-token source masks for training.

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

Built-in segmenters: `SentenceSegmenter`, `LineSegmenter`, `NumberedStepSegmenter`, `CustomSegmenter`.

## Installation

```bash
pip install llenvs[openai,reasoning-gym]   # API + reasoning tasks
pip install llenvs[vllm,huggingface]       # Local inference + HF datasets
pip install llenvs[alfworld]               # AlfWorld household tasks
pip install llenvs[jericho]               # Jericho interactive fiction
pip install llenvs[agentgym]               # AgentGym environments
pip install llenvs[verifiers]              # Verifiers environments
pip install llenvs[openenv]                # OpenEnv environments
pip install llenvs[trl]                    # TRL integration
pip install llenvs[all]                    # Everything
```

See the [full documentation](docs/README.md) for detailed guides on environments, prompts, tools, containers, segmentation, evaluation, RL training, backends, and configuration.

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
