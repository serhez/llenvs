---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# LLEnvs

**MDP-style access to evaluation environments for LLM research.**

Stateless environments, multi-signal rewards, and 17+ adapters — one unified interface for benchmarking, evaluating, and training language model agents.

<div class="hero-buttons" markdown>

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/serhez/llenvs){ .md-button }

</div>

</div>

---

<div class="grid cards" markdown>

-   :material-flask-outline:{ .lg .middle } **Stateless Environments**

    ---

    `step()` is a pure function — state in, state out. Enables branching, checkpointing, and parallel exploration with no side effects.

    [:octicons-arrow-right-24: Core abstractions](reference/core.md)

-   :material-signal-variant:{ .lg .middle } **Multi-Signal Rewards**

    ---

    Each step produces a `SignalBundle` with named, weighted signals carrying numeric rewards, textual feedback, or both.

    [:octicons-arrow-right-24: Evaluation guide](guides/evaluation.md)

-   :material-puzzle-outline:{ .lg .middle } **17+ Adapters**

    ---

    ReasoningGym, HuggingFace, GEM, Gymnasium, WebShop, AlfWorld, Jericho, AgentGym, LMRL-Gym, Aviary, Verifiers, OpenEnv, and more.

    [:octicons-arrow-right-24: Browse adapters](adapters/index.md)

-   :material-tools:{ .lg .middle } **Tool & Function Calling**

    ---

    Built-in tool environments with auto-monitoring, text-based tool parsers, and MCP support for open-source and API models.

    [:octicons-arrow-right-24: Tools guide](guides/tools.md)

-   :material-brain:{ .lg .middle } **RL Training Integration**

    ---

    Drop-in scoring, dataset provision, and token masking for veRL, TRL, and OpenRLHF training loops.

    [:octicons-arrow-right-24: RL training guide](guides/rl-training.md)

-   :material-source-branch:{ .lg .middle } **Branching & Containers**

    ---

    Checkpoint and branch environment states with zero-copy, process fork, or action replay. Run environments in Docker or isolated subprocesses.

    [:octicons-arrow-right-24: Branching guide](guides/branching.md)

</div>

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
