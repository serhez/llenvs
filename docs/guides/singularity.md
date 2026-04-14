# Running vLLM via Singularity (`vllm_singularity` backend)

llenvs' default [`VLLMBackend`](../../src/llenvs/inference/backends/vllm.py) runs `vllm` **in-process** against whatever the cluster's native venv provides. Sometimes that native vLLM is too old or its torch/CUDA stack conflicts with the model you want to run. For those cases, llenvs ships a second backend, [`SingularityVLLMBackend`](../../src/llenvs/inference/backends/vllm_singularity.py), which spawns `vllm serve` inside a Singularity image as a sibling subprocess and talks to it over HTTP.

The caller (value-bench, a notebook, any other llenvs user) stays on bare metal — its own venv, its own `uv run`, no wrapping. Projects that don't need `vllm_singularity` get zero new code paths, no extra dependencies, no surprise setup steps.

## When to use it

- The model's architecture / tokenizer / vision head lands in a vLLM release newer than your cluster's native install.
- Your transformers or torch version is pinned to something older than what the model needs.
- You want `vllm serve`'s HTTP interface for external tools.

If your model works with the in-process `VLLMBackend`, keep using that — it's cheaper (no HTTP hop, no subprocess lifecycle). Each image version supports a particular range of models; you can build several `.sif`s side by side and point `LLENVS_SIF` (or the per-call `sif=` kwarg) at whichever one matches.

## How it works

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│ bare-metal python           │         │ singularity container        │
│  value-bench / llenvs       │         │  vllm/vllm-openai:<tag>      │
│  native .venv/              │         │                              │
│                             │         │   vllm serve <model>         │
│  SingularityVLLMBackend ────┼─Popen─▶│   --host 127.0.0.1            │
│   │                         │         │   --port <dynamic>           │
│   │ openai.OpenAI           │         │                              │
│   ▼    base_url=...         │         │   POST /v1/chat/completions  │
│   http://127.0.0.1:<port>/v1├─HTTP──▶│   GET  /health                │
│                             │         │                              │
│  close() ─────────────────▶│ SIGTERM/SIGKILL on the pgroup           │
└─────────────────────────────┘         └──────────────────────────────┘
```

`SingularityVLLMBackend.__init__`:
1. Picks a free TCP port with `bind(('127.0.0.1', 0))`.
2. Builds `["singularity", "exec", "--nv", *binds, <sif>, "vllm", "serve", <model>, "--port", <port>, ...]`.
3. `subprocess.Popen(argv, env=<with SINGULARITYENV_* forwarding>, start_new_session=True)` — the container process tree is isolated in its own process group so we can SIGTERM-pgroup it cleanly.
4. Polls `http://127.0.0.1:<port>/health` until 200 (or the subprocess dies, or we hit `startup_timeout`).
5. Instantiates an inner [`OpenAIBackend`](../../src/llenvs/inference/backends/api.py) pointing at `http://127.0.0.1:<port>/v1` with `api_key="EMPTY"`.
6. Delegates every `generate` / `generate_chat` / `generate_chat_batch` call to that inner client.

`close()` terminates the process group (SIGTERM → 30s wait → SIGKILL) and shuts down the HTTP client. `__enter__`/`__exit__`/`__del__`/`atexit` all route to `close()` so crashes don't leak vllm servers.

Vision works automatically: llenvs' [`ChatMessage.to_dict()`](../../src/llenvs/inference/protocol.py) already emits OpenAI-compatible multimodal content blocks (`{"type": "text"}` / `{"type": "image_url"}` with base64 data URLs), which vLLM's server understands natively.

## One-time setup

### 1. Configure your cluster profile

All cluster-specific defaults live in one place: [`llenvs/bin/_cluster.sh`](../../bin/_cluster.sh). It exports:

- `LLENVS_SIF` — where the `.sif` file lives on this cluster
- `LLENVS_HF_HOME` — HuggingFace cache root (auto-bound into the container)
- `LLENVS_BINDS` — space-separated `singularity --bind` specs
- `LLENVS_HF_OFFLINE` — `1` if compute nodes have no outbound internet (forces offline HF/transformers inside the container)

The file auto-detects a known cluster from filesystem fingerprints and falls back to a `generic` profile otherwise. To add a new cluster: add a `case` arm with the four variables above, and add matching branches to `llenvs_profile_universal` / `llenvs_profile_cluster` if you want the new cluster's sbatch profiles. Or just export the variables manually if you don't want to edit the file.

```bash
source llenvs/bin/_cluster.sh
# If auto-detect picked your cluster, you're done. Otherwise:
#   export LLENVS_CLUSTER=my_cluster
#   and edit bin/_cluster.sh
```

### 2. Build a `.sif`

Pick a `vllm-openai` docker tag that supports your model and CUDA. Browse the available tags at [hub.docker.com/r/vllm/vllm-openai/tags](https://hub.docker.com/r/vllm/vllm-openai/tags) — there are stable releases (`v0.19.0`, `v0.20.0`, …), per-model branches (e.g. `gemma4-<date>-x86_64-cu129` for gemma-4), and nightlies. Pin an exact tag via `LLENVS_VLLM_IMAGE_TAG`:

```bash
# Run directly on a login node (has internet + singularity installed):
LLENVS_VLLM_IMAGE_TAG=v0.19.0 bash llenvs/bin/build_container.sh

# Or submit to a serial queue — partition/time/mem are yours to pick:
LLENVS_VLLM_IMAGE_TAG=v0.19.0 sbatch \
    --partition=<serial-cpu> --time=02:00:00 --mem=30G \
    --output=llenvs_build_%j.log \
    llenvs/bin/build_container.sh
```

The build typically takes 1–1.5 h. `singularity build --remote` is tried first; if it hits the Sylabs remote time limit, the script falls back to `singularity pull` automatically. The resulting `.sif` lands at `$LLENVS_SIF`. If the default scratch dir is too small (the build unpacks docker layers into it), set `LLENVS_BUILD_SCRATCH_DIR=/path/to/big/disk`.

You can keep multiple `.sif`s side by side (e.g. `llenvs-vllm-v0.19.0.sif`, `llenvs-vllm-gemma4.sif`) and select one at call time by setting `LLENVS_SIF=/path/to/llenvs-vllm-gemma4.sif` before running, or by passing `sif=...` to the backend directly.

### 3. Prefetch model weights (only if compute nodes are offline)

If `LLENVS_HF_OFFLINE=1` in your cluster profile, compute nodes can't reach the HuggingFace Hub, so the container must read weights from a shared filesystem (`$HF_HOME`, bound in automatically). Prefetch from a login node:

```bash
HF_HOME="$LLENVS_HF_HOME" huggingface-cli download <HF_REPO_ID>
```

Any Python env with `huggingface-hub` installed works for this — you don't need to be inside the container.

## Using it

### From Python

```python
from llenvs.inference.backends import SingularityVLLMBackend
from llenvs.inference.protocol import ChatMessage, SamplingParams

with SingularityVLLMBackend(
    model_path="<HF_REPO_ID_OR_LOCAL_PATH>",
    tensor_parallel_size=2,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
) as backend:
    result = backend.generate_chat(
        [ChatMessage(role="user", content="Say hello in five words.")],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    print(result.text)
```

This runs bare-metal. Your Python process is unchanged; `SingularityVLLMBackend` spawns `singularity exec … vllm serve …` on the same node and tears it down when the `with` block exits. Pass `sif="/path/to/other.sif"` if you want to target a specific `.sif` (e.g. when you have several for different vLLM versions).

### From value-bench

Add an entry to [`value-bench/configs/backends.yaml`](../../../value-bench/configs/backends.yaml). For example, gemma-4-31B needs a vLLM build new enough to know the `gemma4` model type, so you'd point `LLENVS_SIF` at a corresponding `.sif` and use:

```yaml
gemma4_31b_vision_local:
  type: vllm_singularity
  model: google/gemma-4-31B-it
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.85
  max_model_len: 8192
  dtype: bfloat16
  sampling:
    temperature: 0.4
    top_p: 0.95
    max_tokens: 1024
```

Any HuggingFace-format model vLLM supports works the same way — change `model:`, `tensor_parallel_size:`, and (optionally) point `singularity_sif:` at a different `.sif`.

Optional fields on `BackendConfig`:
- `singularity_sif` — override the `.sif` path (defaults to `$LLENVS_SIF`). Useful when you keep multiple `.sif`s for different models.
- `singularity_binds` — list of extra `--bind` specs (defaults to `$LLENVS_BINDS`).
- `singularity_extra_vllm_args` — list of extra flags appended to `vllm serve` (e.g. `["--swap-space", "4"]`).
- `singularity_startup_timeout` — seconds to wait for `/health` (default 900).
- `singularity_cuda_visible_devices` — pin `vllm serve` to a GPU subset (e.g. `"0,1"`). Lets other workloads on the same node use the remaining GPUs.

Then run value-bench normally — **no container wrapping, no special bin scripts**:

```bash
source llenvs/bin/_cluster.sh   # picks up LLENVS_SIF / LLENVS_BINDS
cd value-bench
uv run python scripts/pipeline/collect_dataset.py \
    --config configs/experiments/<env>/collection/<variant>.yaml
```

(The backend is selected inside the YAML config via `backend_names:`.)

For GPU jobs, submit with plain `sbatch` — pick the partition and resources that fit your cluster:

```bash
sbatch --partition=<gpu-partition> --gres=gpu:2 --time=02:00:00 --mem=120G \
    --output=collect_%j.log \
    --wrap 'source llenvs/bin/_cluster.sh && \
            cd value-bench && uv run python scripts/pipeline/collect_dataset.py ...'
```

The `source llenvs/bin/_cluster.sh` line inside the wrap is what injects `LLENVS_SIF` / `LLENVS_BINDS` into the job environment.

## GPU pinning (running other work alongside)

Pass `singularity_cuda_visible_devices` to reserve only part of the node's GPUs for `vllm serve`:

```yaml
my_backend:
  type: vllm_singularity
  model: <HF_REPO_ID>
  tensor_parallel_size: 2
  singularity_cuda_visible_devices: "0,1"   # vLLM sees only GPUs 0 and 1
  ...
```

The parent process's own `CUDA_VISIBLE_DEVICES` is untouched, so MC rollouts, rerankers, or any other GPU code in the same sbatch can freely use the remaining devices.

## Troubleshooting

- **`RuntimeError: no .sif path provided`** — source `llenvs/bin/_cluster.sh` or export `LLENVS_SIF` manually.
- **`vllm serve exited during startup (rc=…)`** — read `.llenvs-vllm-serve-<port>.log` next to your cwd. The most common causes are missing weights (HF_HOME not bound / wrong path), out of VRAM (lower `gpu_memory_utilization`), and transformers/vllm version mismatch for bleeding-edge models (rebuild the `.sif` from a newer tag).
- **`vllm serve health check timed out`** — first-time model loads with torch.compile warmup can take 10+ minutes. Raise `singularity_startup_timeout` (or `startup_timeout=` from Python) to 1800+ for very large models.
- **Port-in-use races** — rare, but possible if something else grabs the picked port between `bind(0)` and `vllm serve`'s own bind. Either retry, or pass `port=<fixed>` to the backend.
- **Leaked `vllm serve` after a parent crash** — the backend registers an `atexit` safety net and uses process groups, so a normal crash cleans up. A `kill -9` on the parent *will* leave the child alive; find it with `ps -ef | grep vllm | grep <port>` and `kill -TERM -<pgid>`.

## Adding another cluster

Nothing to code — just teach `llenvs/bin/_cluster.sh` about it:

```sh
mycluster)
    : "${LLENVS_SIF:=/path/to/containers/llenvs-vllm.sif}"
    : "${LLENVS_HF_HOME:=/path/to/hf_cache}"
    : "${LLENVS_BINDS:=/path/to/work /path/to/scratch ${LLENVS_HF_HOME}}"
    : "${LLENVS_HF_OFFLINE:=1}"   # 1 if compute nodes are offline
    ;;
```

Plus one `case` arm each in `llenvs_profile_universal` / `llenvs_profile_cluster` if you want sbatch profiles on that cluster. Everything else is cluster-agnostic.
