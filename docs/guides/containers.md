# Container Support

Run any environment inside a Docker container or isolated subprocess. The runner, inference backend, and training code stay on the host; only the environment (and its rewards, tool execution, and state) runs inside the container.

## Why Containers?

- **Safety**: Agent actions execute in a sandbox and cannot affect the host
- **Reproducibility**: Pinned Docker images guarantee identical environment behavior
- **Compatibility**: Run Linux-only or dependency-heavy environments from any host
- **Transparency**: Users who don't use containers are completely unaffected

## Quick Start: Process Runtime

The simplest way to isolate an environment — no Docker required:

```python
from llenvs.core.config import EnvironmentConfig, EnvironmentFactory
from llenvs.container.config import ContainerConfig

config = EnvironmentConfig(
    name="sudoku",
    adapter="reasoning_gym",
    size=100,
    container=ContainerConfig(runtime="process"),
)
env = EnvironmentFactory.create(config)

# Use exactly like a local environment
state, info = env.reset(options={"task_index": 0})
print(state.observation.prompt)
```

The factory starts a subprocess running the environment server, connects a proxy client, and returns it. All `reset()`, `step()`, and `compute_rewards()` calls are forwarded over HTTP.

## Docker Setup

### Build an Image

Use the base Dockerfile:

```bash
docker build -t llenvs-base -f docker/Dockerfile.base .
```

Or create a minimal per-adapter image:

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir llenvs reasoning-gym
EXPOSE 8080
ENTRYPOINT ["python", "-m", "llenvs.container"]
```

### Use in Config

```python
config = EnvironmentConfig(
    name="sudoku",
    adapter="reasoning_gym",
    size=100,
    container=ContainerConfig(
        runtime="docker",
        image="llenvs-base:latest",
    ),
)
env = EnvironmentFactory.create(config)
```

## YAML Configuration

```yaml
environments:
  - name: sudoku
    adapter: reasoning_gym
    size: 100
    container:
      runtime: docker
      image: llenvs-reasoning-gym:latest
      timeout: 60
      env_vars:
        REASONING_GYM_CACHE: /data/cache
      volumes:
        /host/data: /data
```

Process runtime (no Docker):

```yaml
environments:
  - name: sudoku
    adapter: reasoning_gym
    size: 100
    container:
      runtime: process
```

## ContainerConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runtime` | `str` | `"docker"` | `"docker"` or `"process"` |
| `image` | `str \| None` | `None` | Docker image (required for docker runtime) |
| `port` | `int \| None` | `None` | Host port (None = auto-select) |
| `timeout` | `float` | `60.0` | Max seconds to wait for server startup |
| `env_vars` | `dict[str, str]` | `{}` | Environment variables for the container |
| `volumes` | `dict[str, str]` | `{}` | Volume mounts (host_path -> container_path) |
| `docker_command` | `str` | `"docker"` | Path to docker CLI |

## How It Works

```
Host Process                           Container / Subprocess
┌──────────────────────┐              ┌──────────────────────┐
│  TrajectoryRunner     │              │  EnvironmentServer    │
│  InferenceBackend     │    HTTP      │    ┌──────────────┐  │
│  ContainerEnvironment │◄────────────►│    │ Environment   │  │
│  (proxy)              │  JSON over   │    │ (real impl)   │  │
│                       │  localhost   │    └──────────────┘  │
│  Scorer               │              │  Rewards, tools,     │
│  DatasetProvider      │              │  state — all here     │
└──────────────────────┘              └──────────────────────┘
```

1. `EnvironmentFactory.create()` detects the `container` field
2. A runtime (Docker or subprocess) starts the environment server
3. The server creates the real environment via the same factory (without container config)
4. A `ContainerEnvironment` proxy is returned, forwarding all protocol calls over HTTP
5. Hidden state crosses as JSON; the client wraps it as `OpaqueHidden` (supports attribute access)
6. Rewards are computed server-side; `reward_functions` returns `()` on the proxy

### Server Protocol

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/spec` | Environment spec |
| `GET` | `/len` | Number of tasks |
| `GET` | `/tools` | Available tool definitions |
| `GET` | `/prompts` | Prompt components |
| `POST` | `/reset` | Reset environment |
| `POST` | `/step` | Take a step |
| `POST` | `/compute_rewards` | Compute rewards for a transition |

## Integrations

`Scorer` and `DatasetProvider` work through the proxy:

```python
from llenvs.integrations.scoring import Scorer
from llenvs.integrations.dataset_provider import DatasetProvider

scorer = Scorer(env)
result = scorer.score(task_index=0, response="<answer>42</answer>")

provider = DatasetProvider(env)
item = provider[0]
```

## Limitations

- **SegmentedEnvironment**: Don't wrap a `ContainerEnvironment` in `SegmentedEnvironment` on the host — each segment would be an HTTP round-trip. If segmentation is needed, configure it inside the container.
- **info dicts** must be JSON-serializable when using containers.
- **reward_functions** returns `()` on the proxy. All reward computation happens server-side via `step()` and `compute_rewards()`.
- **One trajectory at a time** for mutable-backend environments (WebShop, AgentGym, OpenEnv) — same constraint as local mode.

## Building Custom Docker Images

For environments with heavy dependencies, create a focused image:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir llenvs agentenv

EXPOSE 8080
ENTRYPOINT ["python", "-m", "llenvs.container"]
```

The entrypoint accepts `--config` (JSON) and `--port` arguments:

```bash
docker run -p 9090:8080 my-image \
    --config '{"name":"webshop","adapter":"agentgym"}' \
    --port 8080
```
