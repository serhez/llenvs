"""GPU smoke test for SingularityVLLMBackend.

Runs bare-metal (NO singularity wrapping of this script). The backend
itself spawns ``vllm serve`` inside the .sif as a sibling subprocess and
talks to it over http://127.0.0.1:<port>/v1.

Usage (once your cluster profile in ``bin/_cluster.sh`` is set up and
``LLENVS_SIF`` / ``LLENVS_BINDS`` are exported)::

    source llenvs/bin/_cluster.sh
    sbatch --partition=<gpu> --gres=gpu:2 --time=01:00:00 \\
        --output=llenvs_smoke_%j.log \\
        --wrap 'source llenvs/bin/_cluster.sh && \\
                cd llenvs && uv run python bin/smoke_vllm_singularity.py'

or directly on an interactive GPU shell::

    uv run python bin/smoke_vllm_singularity.py google/gemma-4-31B-it 2

First positional arg is the HF model id (default: google/gemma-4-31B-it).
Second positional arg is tensor_parallel_size (default: 2).
"""

from __future__ import annotations

import sys
import time

from llenvs.inference.backends import SingularityVLLMBackend
from llenvs.inference.protocol import ChatMessage, SamplingParams


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-31B-it"
    tp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"[smoke] model={model_id} tp={tp_size}", flush=True)
    t0 = time.time()
    with SingularityVLLMBackend(
        model_path=model_id,
        tensor_parallel_size=tp_size,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    ) as backend:
        startup = time.time() - t0
        print(
            f"[smoke] ready in {startup:.1f}s "
            f"(pid={backend.pid}, port={backend.port}, log={backend.server_log_path})",
            flush=True,
        )
        print(f"[smoke] capabilities={backend.capabilities}", flush=True)

        messages = [ChatMessage(role="user", content="Say hello in five words.")]
        params = SamplingParams(max_tokens=32, temperature=0.0)

        t1 = time.time()
        result = backend.generate_chat(messages, params)
        infer_time = time.time() - t1

        print(f"[smoke] first-call latency: {infer_time:.2f}s", flush=True)
        print(f"[smoke] text: {result.text!r}", flush=True)
        print(f"[smoke] finish_reason: {result.finish_reason}", flush=True)

        if not result.text.strip():
            print("[smoke] ERROR: empty response", file=sys.stderr)
            return 1

    print("[smoke] OK — close() clean", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
