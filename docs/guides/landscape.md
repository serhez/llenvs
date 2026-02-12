# Library Landscape

An analysis of llenvs relative to other libraries in the LLM environment and evaluation space, focusing on architectural differences and positioning.

The two most comparable libraries are **OpenEnv** (Meta + HuggingFace) and **verifiers** (PrimeIntellect). All three are in early stages but approach the problem from different angles.

## Overview

| | llenvs | OpenEnv | verifiers |
|---|---|---|---|
| **Core idea** | Unified eval & training lib with built-in inference | Isolated sandbox environments for agentic RL | Environment-as-package with OpenAI-protocol inference |
| **Architecture** | In-process, explicit-state MDP | Client-server, Docker-isolated | In-process, rollout-based |
| **Backing** | Internal | Meta + HuggingFace | PrimeIntellect |
| **Repo** | — | [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) | [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) |

## Inference Backends

**llenvs** bundles inference backends (vLLM, HuggingFace Transformers, OpenAI, Anthropic, OpenRouter). A single library handles both the environment and the model — a complete evaluation can run with zero external services.

**OpenEnv** includes no inference layer. It is environment-only; the agent and model are entirely external (typically Forge/TorchForge, TRL, or similar).

**verifiers** also includes no inference engine. All model communication goes through an external OpenAI-compatible API. Running evaluations always requires a separate vLLM server or cloud API endpoint.

| Backend | llenvs | OpenEnv | verifiers |
|---------|--------|---------|-----------|
| vLLM (in-process) | Yes | — | — |
| HuggingFace Transformers | Yes | — | — |
| OpenAI API | Yes | — | Via external server |
| Anthropic API | Yes | — | Via external server |
| OpenRouter | Yes | — | Via external server |
| Logprobs | Yes (vLLM, HF, OpenAI) | — | Depends on external server |
| Prefix continuation | Yes (vLLM, HF, Anthropic) | — | — |

llenvs is the only library where `pip install` and a local model path are sufficient to run end-to-end evaluations. For researchers who need to quickly swap models, adjust sampling parameters, or inspect logprobs, the integrated backend avoids the overhead of orchestrating separate services.

## Environment Architecture

The three libraries use fundamentally different paradigms.

### llenvs — Explicit-State MDP

State is immutable and passed explicitly to `step()`:

```python
result = env.step(state, action)  # state is not modified
```

Environments with `spec.supports_branching == True` (single-turn adapters, GEM via snapshot/restore, Dialogue via message reconstruction) are genuine pure functions — any state can be snapshotted and resumed from, and multiple branches can diverge from the same checkpoint.

Multi-turn adapters wrapping stateful backends (WebShop, AgentGym, OpenEnv, verifiers tool) have `supports_branching=False` and enforce sequential continuity: only the most recent state from `reset()`/`step()` is valid. Passing a stale state raises `NotImplementedError`. However, the `BranchManager` system provides universal branching for all environments via a tiered strategy system: `DirectStrategy` (zero-cost for pure-function envs), `ProcessForkStrategy` (via `os.fork()` for mutable-backend envs on Unix), and `ActionReplayStrategy` (replays actions on a fresh env for deterministic envs).

### OpenEnv — Client-Server with Docker Isolation

Environments run inside Docker containers. The client communicates via persistent WebSocket connections. State lives server-side; the client only sees observations. Two separate interfaces exist:

- **MCP interface** (agent-facing): tool calls only — the agent never sees `reset()` or `step()` directly
- **HTTP interface** (orchestration): simulation control (`reset`, `step`, `state`)

The deliberate hiding of `reset()` from the agent is a training-production alignment mechanism: agents cannot learn that mistakes are recoverable.

### verifiers — Rollout-Driven

The environment owns the full rollout loop. It calls the model, accumulates trajectory steps, and scores at the end:

```python
outputs = env.generate(model="...", dataset=dataset)
```

The environment *drives* the conversation rather than being stepped through externally. This simplifies trainer integration but gives less control over the inference loop.

### Implications

| Property | llenvs | OpenEnv | verifiers |
|----------|--------|---------|-----------|
| Tree search / MCTS from any state | All environments (via branching strategies) | Not possible (server-side state) | Not possible (env owns loop) |
| Trajectory branching / checkpointing | All environments (Direct, ProcessFork, or ActionReplay) | Not supported | Not supported |
| Custom sampling strategies mid-episode | Full control | Agent brings own | Limited (env calls model) |
| Training-production environment parity | Not a goal | Core design goal | Not a goal |
| Sandboxed execution | Not built-in | Docker isolation | Prime Sandboxes |

## Multi-Turn & Tool Support

All three have first-class multi-turn support, but the mechanisms differ.

**llenvs**: Multi-turn via adapters (GEM games, WebShop, AgentGym's 15 environments). Tool support via `ToolDefinition`, `AsyncToolExecutor`, and `MCPToolExecutor`. Also offers **segmented environments** — decomposing single-turn responses into steps for per-segment rewards, process reward models, tree search at segment boundaries, and early exit.

**OpenEnv**: Multi-turn is the default paradigm (reset/step loop). MCP-native tool system with a `@tool()` decorator for mode-aware registration (simulation-only, production-only, or both). 24 reference environments spanning games, web/browser, code execution, robotics, and finance.

**verifiers**: Deep environment hierarchy: `MultiTurnEnv` → `ToolEnv` → `StatefulToolEnv` → `SandboxEnv` → `PythonEnv`. OpenAI function-calling native. Docker sandbox for code execution. MCP support via `MCPEnv`.

| Capability | llenvs | OpenEnv | verifiers |
|------------|--------|---------|-----------|
| Multi-turn games | Yes (GEM, AgentGym) | Yes (OpenSpiel, TextArena) | Yes (TextArena, Wordle) |
| E-commerce | Yes (WebShop) | Yes (browsergym) | — |
| Code sandbox | — | Yes (Docker) | Yes (Prime Sandboxes) |
| Browser automation | — | Yes (browsergym) | Yes (BrowserEnv) |
| Robotics / simulation | — | Yes (Unity, SUMO) | — |
| MCP tool protocol | Yes | Yes (native) | Yes |
| Segmented environments | Yes | — | — |

OpenEnv and verifiers have broader coverage for agentic environments (browser, code sandbox, robotics). llenvs's segmented environments — breaking single-turn responses into steps for PRM scoring, tree search, or intervention — are unique to the library.

## Answer Extraction

**llenvs** has a dedicated extraction system with 11 built-in extractors, composable fallback chains, and a cleaning layer:

| Extractor | Pattern |
|-----------|---------|
| `tag_based` | `<answer>42</answer>` |
| `boxed` | `\boxed{x^2+1}` (balanced brace matching) |
| `gsm8k` | `#### 42` |
| `numeric` | Last number in text |
| `multiple_choice` | `(A)`, `Answer: B` |
| `pattern_answer` | "the answer is X", "therefore X" |
| `code_block` | Fenced code blocks |
| `last_line` | Last non-empty line |
| `regex` | Custom pattern with capture group |
| `raw` | Full response (explicit no-extraction) |
| `CompositeExtractor` | Try multiple in order (fallback chain) |

Extractors compose with pre-cleaners (strip special tokens from raw response) and post-cleaners (strip punctuation, quotes, LaTeX from extracted answer). The `CleanedExtractor` wrapper and config integration (`answer_extractors` list, `pre_cleaners`/`post_cleaners` fields) make this a first-class subsystem.

**OpenEnv** has no answer extraction. Scoring happens inside the server-side `Rubric`; any parsing logic is written into the rubric by the environment author.

**verifiers** has lightweight parsers — `XMLParser` (extracts XML-tagged fields), `ThinkParser` (content after `</think>`), `MaybeThinkParser`. Reward functions receive raw completions and handle their own matching. Parsers can generate format compliance rewards via `parser.get_format_reward_func()`.

For math and reasoning evaluation where parsing `\boxed{}`, `#### answers`, multiple-choice patterns, and similar formats is critical, llenvs provides the most structured tooling. verifiers takes a more ad-hoc approach. OpenEnv delegates it entirely to environment authors.

## Reward System

### llenvs

Multi-signal typed rewards. Each transition produces a `RewardBundle` with named, weighted `RewardSignal` instances:

```python
RewardBundle(signals=(
    RewardSignal(value=1.0, name="correctness", type=OUTCOME, weight=2.0),
    RewardSignal(value=0.5, name="format", type=FORMAT, weight=0.5),
))
# bundle.total = 1.0*2.0 + 0.5*0.5 = 2.25
```

Adapters provide only native rewards by default (wrapper fidelity). Additional rewards (`FormatReward`, `ToolValidityReward`, `ToolEfficiencyReward`) are opt-in via `extra_rewards`. Signals are queryable by name or type. The `weight` field (default 1.0) allows expressing relative importance.

### OpenEnv

PyTorch `nn.Module`-inspired `Rubric` system. Server-side execution during `env.step()`:

```python
class Rubric(ABC):
    def forward(self, action, observation) -> float: ...
```

Composable via containers: `Sequential` (fail-fast), `Gate` (threshold), `WeightedSum` (parallel), `RubricList`, `RubricDict`. Supports hooks (pre/post), state dicts, and `TrajectoryRubric` for delayed rewards with discounting. `LLMJudge` for remote LLM-as-judge evaluation.

### verifiers

`Rubric` class combining weighted reward functions:

```python
rubric = Rubric(
    reward_funcs=[exact_match, format_reward],
    weights=[1.0, 0.5],
)
```

Functions can be individual (`completion, answer -> float`) or group-level (`completions, answers -> list[float]`) for relative scoring. Arguments are auto-injected by signature. Specialized rubrics: `JudgeRubric` (LLM-as-judge), `MathRubric` (symbolic math verification). Monitor rubrics auto-track metrics (turn counts, tool calls, timing).

| Property | llenvs | OpenEnv | verifiers |
|----------|--------|---------|-----------|
| Multiple named signals per step | Yes | No (single float) | No (single float) |
| Signal types (OUTCOME, STEP, FORMAT, PROCESS) | Yes | Reward field on Observation | Via separate reward functions |
| Composable reward containers | Via `extra_rewards` tuple | nn.Module-style nesting | Weighted function list |
| Trajectory-level rewards | Via trajectory total | `TrajectoryRubric` | After-rollout scoring |
| LLM-as-judge | — | `LLMJudge` | `JudgeRubric` |
| Group-relative scoring | — | — | Yes (group reward functions) |

## RL Training Integration

### llenvs

Three integration layers plus framework-specific adapters:

- **`Scorer`**: Wraps a single-turn environment, exposes `score(task_index, response) -> ScoringResult`
- **`DatasetProvider`**: Iterates tasks, provides prompts and ground truths, converts to HuggingFace `Dataset`
- **`TrajectoryMasker`**: Converts multi-turn `Trajectory` to `MaskedTrajectory` with per-token source masks (model tokens = 1, environment tokens = 0)

Framework adapters for veRL, TRL, and OpenRLHF provide reward functions and rollout helpers that plug into each trainer's API.

### OpenEnv

Positions itself as the environment standard that RL frameworks connect to. The environment is the same in training, evaluation, and production. Confirmed integrations: Forge (Meta), TRL, VeRL, Unsloth, SkyRL, ART, Oumi. The primary demonstrated integration is with TorchForge for GRPO training.

No built-in scoring API or dataset provision — those responsibilities belong to the training framework.

### verifiers

`prime-rl` (PrimeIntellect's production RL trainer) is the primary path: multi-node async training, MoE support, LoRA, online difficulty filtering, NCCL-based weight sync between training and vLLM inference. Hosted training available in private beta.

A legacy single-node `RLTrainer` extends HuggingFace `Trainer` with GRPO-style training and vLLM weight synchronization. Also works with external trainers (Tinker, SkyRL, rLLM).

| Capability | llenvs | OpenEnv | verifiers |
|------------|--------|---------|-----------|
| Single-turn scoring API | `Scorer` | — | Via rubric |
| Dataset provision | `DatasetProvider` | — | Built into environment |
| Token-level trajectory masking | `TrajectoryMasker` | — | — |
| veRL integration | Yes | Yes | Via rLLM (third-party) |
| TRL integration | Yes | Yes | — |
| OpenRLHF integration | Yes | — | — |
| Own RL trainer | — | — | `prime-rl` (production) |
| Weight sync (training ↔ inference) | — | — | NCCL-based |
| Hosted training | — | — | Private beta |

llenvs's `TrajectoryMasker` — providing per-token source attribution (which tokens came from the model vs. the environment) for multi-turn RL — is unique among the three.

## Batching & Parallelization

**llenvs**: Lockstep batching — all active trajectories advance one step together, finished ones drop out. GPU backends (vLLM, HF) do a single batched forward pass; API backends use concurrent async HTTP with semaphore (`max_concurrency`). `batch_size` parameter chunks large task sets for GPU memory management. Cross-environment batching (`run_multi_evaluation()`) interleaves trajectories from multiple environments in one loop.

**OpenEnv**: Per-session concurrency via `max_concurrent_envs` on the HTTP server. Each environment instance handles one trajectory. A batch orchestration layer (`EnvPool`) is planned but not yet implemented.

**verifiers**: Async semaphore-based concurrency (`max_workers`, default 512) for parallel rollouts. In RL training, the `Orchestrator` runs asynchronous batch generation in a separate thread parallel with training steps. `ZMQEnvServer`/`ZMQEnvClient` for distributed environment execution across processes.

| Property | llenvs | OpenEnv | verifiers |
|----------|--------|---------|-----------|
| GPU batch inference | Yes (vLLM, HF) | — | — |
| API concurrent requests | Yes (semaphore) | — | Yes (semaphore) |
| Cross-environment batching | Yes | — | — |
| Batch memory chunking | Yes (`batch_size`) | — | Yes (`micro_batch_size`) |
| Distributed environments | — | Docker Swarm | ZMQ workers |

## Configuration & Prompt Management

**llenvs**: Centralized YAML and Python config (`EvalConfig`, `EnvironmentConfig`, `ModelConfig`). Factory pattern for environments and backends. Composable prompt system: fragments (reusable snippets), system prompts (complete pre-built), templates (wrap questions), model profiles (family-specific adjustments like DeepSeek-R1, O1, Llama3, Qwen). Per-environment overrides for prompts and templates.

**OpenEnv**: Decentralized. Each environment has its own `openenv.yaml` manifest. No unified eval config or prompt management — prompt formatting is the agent's responsibility. Training config (e.g., GRPO hyperparameters) is external, typically via OmegaConf YAML.

**verifiers**: TOML configs and CLI (`prime eval run`). `EvalConfig` and `RLConfig` (extends HuggingFace `TrainingArguments`). Provides modified chat templates for specific models (Qwen3, DeepSeek) that preserve `<think>` sections. GEPA (Genetic-Pareto Prompt Optimization) for automatic system prompt optimization. No general prompt template/fragment system.

## Ecosystem & Community

**OpenEnv**: Meta + HuggingFace institutional backing. RFC-driven development process. HuggingFace Hub for environment distribution (`openenv push`, `AutoEnv.from_env()`). 24 reference environments. Community contributions via Hub.

**verifiers**: ~3.8k GitHub stars, 65 contributors. Community-driven Environments Hub with 400+ environments. Bounty program ($100–$5,000 per environment). RL residency program with 500+ applicants. Active ecosystem with categories spanning autonomous research, browser automation, theorem proving, legal/finance, and more.

**llenvs**: Internal library, comprehensive documentation and test suite (1100+ tests), 7 adapters wrapping established libraries (reasoning-gym, HuggingFace datasets, GEM, WebShop, AgentGym, verifiers, OpenEnv). The verifiers and OpenEnv adapters enable using those ecosystems' environments through llenvs's stateless MDP interface, extraction system, and evaluation runners.

## Positioning Summary

Each library occupies a distinct point in the design space:

**llenvs** is a **research-oriented evaluation and training library** with integrated inference. Its explicit-state environment architecture (with full branching on supported environments), built-in backends, answer extraction system, segmented environments, and token-level trajectory masking are aimed at researchers who need fine-grained control over the full pipeline — from sampling to scoring — without orchestrating external services. The tradeoff is a smaller environment catalog and no sandboxed execution.

**OpenEnv** is an **environment creation and deployment standard** with a production-grade isolation model. Its client-server architecture, Docker sandboxing, MCP-native tools, and simulation/production duality target the workflow of building, training on, and deploying agentic environments. The tradeoff is that inference, RL training, and answer parsing are all external concerns.

**verifiers** is an **RL training ecosystem** centered on environment packaging and community contribution. Its rollout-driven design, production RL trainer (prime-rl), weight synchronization, Environments Hub, and hosted training target teams running RL at scale. The tradeoff is that it requires external inference servers and gives less control over the inference loop.
