# llenvs Documentation

A library providing MDP-style access to evaluation environments for LLM research.

## Getting Started

- **[Installation](getting-started/install.md)** - Install llenvs and dependencies
- **[Quick Start](getting-started/quickstart.md)** - Run your first evaluation in minutes

## Guides

- **[Environments](guides/environments.md)** - Working with different environment types
- **[Tools & Function Calling](guides/tools.md)** - Using tools in evaluations
- **[Segmentation](guides/segmentation.md)** - Multi-step reasoning with segmenters
- **[Evaluation](guides/evaluation.md)** - Running evaluations and computing metrics
- **[Inference Backends](guides/backends.md)** - Configure model backends and prompting

## Environment Adapters

- **[ReasoningGym](adapters/reasoning-gym.md)** - Procedural reasoning tasks
- **[HuggingFace](adapters/huggingface.md)** - AIME, GSM8K, and other HF datasets
- **[GEM](adapters/gem.md)** - Multi-turn games and benchmarks with tool support
- **[WebShop](adapters/webshop.md)** - E-commerce product search and purchase

## Reference

- **[Core Abstractions](reference/core.md)** - State, Environment, Rewards, Trajectory
- **[Tools Reference](reference/tools.md)** - ToolDefinition, executors, MCP
- **[Configuration](reference/config.md)** - YAML config and programmatic setup

## Design Principles

### Stateless Environments

Traditional RL environments maintain internal state. llenvs uses **stateless environments** where `step()` is a pure function:

```python
result = env.step(state, action)  # state is not modified
```

This enables branching, checkpointing, and parallel exploration.

### Observation/Hidden Split

Each `State` separates what the model sees from what's needed for rewards:

```python
State(
    observation: ObsT,   # Model sees this (prompt, messages)
    hidden: HiddenT,     # For reward computation (ground truth)
    metadata: StateMetadata
)
```

### Multi-Signal Rewards

Transitions produce `RewardBundle` with multiple named signals:

```python
RewardBundle(signals=(
    RewardSignal(value=1.0, name="correctness", type=OUTCOME),
    RewardSignal(value=1.0, name="format", type=FORMAT),
))
```
