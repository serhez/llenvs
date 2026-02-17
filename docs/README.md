# llenvs Documentation

A library providing MDP-style access to evaluation environments for LLM research.

## Getting Started

- **[Installation](getting-started/install.md)** - Install llenvs and dependencies
- **[Quick Start](getting-started/quickstart.md)** - Run your first evaluation in minutes

## Guides

- **[Environments](guides/environments.md)** - Working with different environment types
- **[Prompts](guides/prompts.md)** - Fragments, system prompts, templates, and model profiles
- **[Tools & Function Calling](guides/tools.md)** - Using tools in evaluations
- **[Segmentation](guides/segmentation.md)** - Multi-step reasoning with segmenters
- **[Evaluation](guides/evaluation.md)** - Running evaluations and computing metrics
- **[Inference Backends](guides/backends.md)** - Configure model backends and prompting
- **[RL Training](guides/rl-training.md)** - Integrate with veRL, TRL, and OpenRLHF
- **[Containers](guides/containers.md)** - Run environments in Docker or isolated subprocesses
- **[Iterative Refinement](guides/iterative.md)** - Multi-turn refinement with code execution and judge feedback
- **[Library Landscape](guides/landscape.md)** - Comparison with OpenEnv, verifiers, and similar libraries

## Environment Adapters

- **[ReasoningGym](adapters/reasoning-gym.md)** - Procedural reasoning tasks
- **[HuggingFace](adapters/huggingface.md)** - AIME, GSM8K, and other HF datasets
- **[GEM](adapters/gem.md)** - Multi-turn games and benchmarks with tool support
- **[WebShop](adapters/webshop.md)** - E-commerce product search and purchase
- **[AgentGym](adapters/agentgym.md)** - 15 multi-turn agent environments
- **[Verifiers](adapters/verifiers.md)** - Single-turn and tool environments with rubric scoring
- **[OpenEnv](adapters/openenv.md)** - Session-based server environments with MCP tools
- **[SciAgentGYM](adapters/sciagentgym.md)** - Multi-step scientific tool-use across 6 domains
- **[InterCode](adapters/intercode.md)** - Interactive code generation in Bash, SQL, Python, CTF

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

Transitions produce a `SignalBundle` with multiple named, weighted `Signal` instances. Each signal can carry a numeric reward, textual feedback, or both:

```python
SignalBundle(signals=(
    Signal(name="correctness", reward_type=OUTCOME, reward=1.0, weight=1.0),
    Signal(name="format", reward_type=FORMAT, reward=1.0, weight=0.5),
    Signal(name="judge", reward_type=PROCESS, feedback="Consider edge cases..."),
))
# bundle.total = 1.0*1.0 + 1.0*0.5 = 1.5  (feedback-only signals don't contribute)
# bundle.feedback_texts() = ("Consider edge cases...",)
```
