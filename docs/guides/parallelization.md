# Parallelization

Both runners (`TrajectoryRunner`, `SegmentedTrajectoryRunner`) automatically batch inference calls when using `run_batch()`. No code changes are needed to benefit from parallelism.

## How It Works

### Lockstep Batching

`run_batch()` advances all active trajectories one step together:

1. Build messages for all active trajectories
2. Call `generate_chat_batch()` (or `generate_with_tools_batch()`) once with all messages
3. Step each environment with its response
4. Drop finished trajectories, repeat

This maximizes batch utilization at each step. As trajectories finish, the batch shrinks.

### Backend Implementation

How each backend handles `generate_chat_batch()`:

| Backend | Strategy | Notes |
|---------|----------|-------|
| **vLLM** | Single GPU batch call | Converts all messages to prompts, passes to vLLM's batched `generate()` |
| **HuggingFace** | Single GPU batch call | Same approach, uses attention masks for variable-length inputs |
| **OpenAI** | Concurrent async HTTP | Fires `max_concurrency` requests in parallel via `asyncio.gather` + semaphore |
| **Anthropic** | Concurrent async HTTP | Same concurrency approach as OpenAI |
| **OpenRouter** | Concurrent async HTTP | Same concurrency approach as OpenAI |

GPU backends get true batching (one forward pass for all prompts). API backends get concurrent HTTP requests limited by a semaphore.

### Segmented Batching

`SegmentedTrajectoryRunner` extends the lockstep pattern to segment-level generation. The continuation strategy's `generate_segment_batch()` produces one segment per trajectory per round. Trajectories with forced segments (`ForceAction`) skip the backend call. Trajectories that signal `COMPLETE` finish via individual `generate_chat()` calls.

## Configuration

### `max_concurrency` (API backends)

Controls how many HTTP requests are in-flight simultaneously for API backends. Set on the backend constructor or via `ModelConfig`:

```python
# Direct
backend = OpenAIBackend(model="gpt-4o", max_concurrency=32)

# Via config
model:
  backend: openai
  model: gpt-4o
  max_concurrency: 32  # Default: 64
```

Higher values increase throughput but may trigger rate limits. Start with the default (64) and reduce if you see 429 errors.

### `batch_size` (memory management)

Limits how many trajectories run in a single lockstep batch. Useful for GPU backends where all prompts must fit in memory:

```python
# Direct on runner
result = runner.run_batch(task_indices, batch_size=32)

# Via convenience function
result = run_evaluation(
    environment=env,
    backend=backend,
    num_tasks=1000,
    batch_size=32,  # Process 32 at a time
)

# Via config
batch_size: 32  # Top-level in EvalConfig YAML
```

When `batch_size` is set, `run_batch()` chunks the task list and processes each chunk independently, aggregating results at the end. Progress callbacks report global offsets across chunks. The same global-offset behavior applies to `run_batch_from_states()` and `run_batch_from_state_actions()`.

When `batch_size` is `None` (default), all tasks run in a single lockstep batch.

## Cross-Environment Batching

When evaluating across multiple environments (e.g., `leg_counting` + `math` + `logic`), `run_multi_evaluation()` interleaves trajectories from all environments into a single lockstep loop. Instead of processing each environment sequentially, one `generate_chat_batch()` call per step covers all active trajectories across all environments. This maximizes GPU utilization and API concurrency.

### Usage

```python
from llenvs.evaluation import MultiEvalEntry, run_multi_evaluation

result_list = run_multi_evaluation(
    entries=[
        MultiEvalEntry(runner=runner_a, task_indices=list(range(100))),
        MultiEvalEntry(runner=runner_b, task_indices=list(range(50))),
    ],
    batch_size=64,
)

# result_list[0] is the BatchResult for runner_a
# result_list[1] is the BatchResult for runner_b
```

### Requirements

- All entries must share the same `backend` and `sampling_params` (enforced at runtime with `ValueError`).
- Only `TrajectoryRunner` is supported. Tool and segmented runners have different step mechanics that complicate interleaving.

### How it works

1. **Reset** all tasks across all entries, creating trajectory state for each.
2. **Lockstep loop**: each iteration collects remaining (not-done) trajectories from all entries, builds messages via each trajectory's own runner, makes a single `generate_chat_batch()` call, then steps each trajectory through its own environment.
3. **Dropout**: trajectories that finish (terminal state or max_steps) drop out of subsequent batches, so the batch shrinks as trajectories complete.
4. **Finalize**: results are partitioned by entry and returned as one `BatchResult` per entry.

When `batch_size` is set, all trajectories (across all entries) are chunked and each chunk processed independently, then results are merged back per entry.

## Performance Tips

### GPU Backends (vLLM, HuggingFace)

- **Maximize batch size**: Larger batches amortize overhead. Use the largest `batch_size` that fits in GPU memory.
- **vLLM is preferred**: vLLM's continuous batching and PagedAttention handle variable-length sequences more efficiently than HuggingFace.
- **Watch memory**: If you get OOM errors, reduce `batch_size`. A good starting point is 32-64 for 8B models on an 80GB GPU.

### API Backends (OpenAI, Anthropic, OpenRouter)

- **Tune `max_concurrency`**: The default (64) works well for most rate limits. If you hit 429s, reduce it. If your tier allows more, increase it.
- **`batch_size` is less important**: API backends don't load all prompts into memory at once, so `batch_size` mainly affects how many environments are reset simultaneously. Usually leave it at `None`.
- **Throughput scales linearly**: With `max_concurrency=64`, you get ~64x speedup over sequential requests (minus overhead).

### Multi-Turn Environments

For multi-turn environments (tool use, multi-step reasoning), trajectories finish at different times. The lockstep approach handles this naturally: finished trajectories drop out, and the batch shrinks each round. This is optimal because short trajectories don't block long ones.

## Example: Full Evaluation with Config

```yaml
environments:
  - name: leg_counting
    size: 500
    seed: 42

model:
  backend: openai
  model: gpt-4o
  max_concurrency: 32

inference:
  temperature: 0.0
  max_tokens: 1024

batch_size: 64
system_prompt: "Think step by step."
```

```python
from llenvs.core.config import EvalConfig, BackendFactory, EnvironmentFactory, create_sampling_params
from llenvs.evaluation import run_evaluation

config = EvalConfig.from_yaml("eval_config.yaml")
env = EnvironmentFactory.create(config.environments[0])
backend = BackendFactory.create(config.model)
params = create_sampling_params(config.inference)

result = run_evaluation(
    environment=env,
    backend=backend,
    sampling_params=params,
    system_prompt=config.system_prompt,
    batch_size=config.batch_size,
)
print(f"Accuracy: {result.success_rate:.1%}")
```
