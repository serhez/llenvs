# Installation

## Installation

### Using pip

```bash
# Basic installation
pip install llenvs

# With specific backends/adapters
pip install llenvs[huggingface]    # HuggingFace datasets (AIME, GSM8K, etc.)
pip install llenvs[reasoning-gym]  # reasoning-gym datasets
pip install llenvs[vllm]           # Local inference with vLLM
pip install llenvs[transformers]   # Local inference with HuggingFace Transformers
pip install llenvs[openai]         # OpenAI API
pip install llenvs[anthropic]      # Anthropic API

# Everything
pip install llenvs[all]
```

### Using uv

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install with extras
uv pip install llenvs[openai,reasoning-gym]

# Or install from source
uv pip install -e ".[all]"
```

### From Source

```bash
git clone https://github.com/example/llenvs.git
cd llenvs

# With pip
pip install -e ".[dev]"

# With uv
uv pip install -e ".[dev]"
```

---

## Dependencies

### Core Dependencies

The base package only requires:

- `pyyaml>=6.0` - Configuration file parsing

### Optional Dependencies

| Extra | Package | Purpose |
|-------|---------|---------|
| `huggingface` | `datasets>=2.14`, `huggingface-hub>=0.20` | HuggingFace datasets (AIME, GSM8K, MATH) |
| `reasoning-gym` | `reasoning-gym>=0.1` | reasoning-gym dataset access |
| `vllm` | `vllm>=0.4` | Local GPU inference with vLLM |
| `transformers` | `transformers>=4.36`, `torch>=2.0`, `accelerate>=0.25` | Local inference with HuggingFace Transformers |
| `openai` | `openai>=1.0` | OpenAI API access |
| `anthropic` | `anthropic>=0.20` | Anthropic API access |
| `dev` | pytest, mypy, ruff | Development tools |

---

## Environment Variables

### API Keys

Set these environment variables for API backends:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
```

Or pass keys directly in code:

```python
from llenvs.inference.backends import OpenAIBackend

backend = OpenAIBackend(
    model="gpt-4o",
    api_key="sk-...",  # Explicit key
)
```

### vLLM Requirements

For local inference with vLLM:

- CUDA-compatible GPU with sufficient VRAM
- CUDA toolkit installed
- For multi-GPU: `tensor_parallel_size` parameter

```python
from llenvs.inference.backends import VLLMBackend

backend = VLLMBackend(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.9,
)
```

### HuggingFace Transformers Requirements

For local inference with HuggingFace Transformers:

- PyTorch installed (CPU, CUDA, or MPS)
- For multi-GPU: use `device_map="auto"` (requires `accelerate`)

```python
from llenvs.inference.backends import HuggingFaceBackend

# Auto-detect device (CUDA > MPS > CPU)
backend = HuggingFaceBackend(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    device="auto",
    dtype="bfloat16",
)

# Multi-GPU with accelerate
backend = HuggingFaceBackend(
    model_path="meta-llama/Llama-3.1-70B-Instruct",
    device_map="auto",  # Distribute across GPUs
)
```

---

## Verifying Installation

```python
# Test core imports
from llenvs import State, Environment, Trajectory
print("Core imports: OK")

# Test tool imports
from llenvs.core import (
    ToolDefinition, ToolParameter, ToolParameterType,
    ToolCall, ToolResult, Observation, Action,
    SimpleToolExecutor, AsyncToolExecutor,
    MCPToolExecutor, MCPServerConfig,
)
print("Tool imports: OK")

# Test extraction
from llenvs.core.extraction import TagBasedExtractor
extractor = TagBasedExtractor()
answer, _ = extractor.extract("<answer>42</answer>")
assert answer == "42"
print("Extraction: OK")

# Test backends (if installed)
try:
    from llenvs.inference.backends import OpenAIBackend
    print("OpenAI backend: OK")
except ImportError:
    print("OpenAI backend: Not installed")

try:
    from llenvs.inference.backends import HuggingFaceBackend
    print("HuggingFace Transformers backend: OK")
except ImportError:
    print("HuggingFace Transformers backend: Not installed")

# Test HuggingFace adapter (if installed)
try:
    from llenvs.adapters import HuggingFaceAdapter
    print("HuggingFace adapter: OK")
except ImportError:
    print("HuggingFace adapter: Not installed")

# Test reasoning-gym adapter (if installed)
try:
    from llenvs.adapters import ReasoningGymAdapter
    print("reasoning-gym adapter: OK")
except ImportError:
    print("reasoning-gym adapter: Not installed")
```

---

## Project Structure

After installation, the package provides:

```
llenvs/
├── core/           # Core abstractions
│   ├── state.py            # State, Observation, Action
│   ├── environment.py      # Environment protocol
│   ├── tools.py            # ToolDefinition, ToolCall, ToolResult, SimpleToolExecutor
│   ├── async_executor.py   # AsyncToolExecutor for parallel execution
│   ├── mcp_executor.py     # MCPToolExecutor for MCP server integration
│   ├── tool_environment.py # BaseToolEnvironment base class
│   ├── tool_rewards.py     # ToolValidityReward, ToolEfficiencyReward
│   ├── adapter.py
│   ├── trajectory.py
│   ├── reward.py
│   ├── extraction.py
│   ├── registry.py
│   └── config.py
├── adapters/       # Environment adapters
│   ├── reasoning_gym.py   # reasoning-gym datasets
│   ├── huggingface.py     # HuggingFace Hub datasets
│   └── gem.py             # GEM environments
├── inference/      # Model backends
│   ├── protocol.py        # ModelBackend, ChatMessage, GenerationResult
│   ├── prompting.py
│   └── backends/
│       ├── vllm.py        # vLLM backend
│       ├── huggingface.py # HuggingFace Transformers backend
│       └── api.py         # OpenAI, Anthropic, OpenRouter (with tool support)
├── evaluation/     # Evaluation tools
│   ├── runner.py          # TrajectoryRunner, SegmentedTrajectoryRunner
│   ├── metrics.py
│   └── results.py
└── cli/            # Command-line interface
    └── run.py
```

---

## CLI Setup

After installation, the `llenvs` command is available:

```bash
# Verify CLI
llenvs --help

# List available commands
llenvs list

# Run evaluation
llenvs run config.yaml
```
