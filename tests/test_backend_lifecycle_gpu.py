"""GPU acceptance test for vLLM lifecycle teardown.

This test is skipped unless both CUDA and a validation model path are
available. Set ``LLENVS_VLLM_LIFECYCLE_MODEL`` to enable it.
"""

from __future__ import annotations

import gc
import os

import pytest

from llenvs.inference.protocol import ChatMessage, SamplingParams


_VALIDATION_MODEL = os.environ.get("LLENVS_VLLM_LIFECYCLE_MODEL")


@pytest.mark.skipif(
    not _VALIDATION_MODEL,
    reason="set LLENVS_VLLM_LIFECYCLE_MODEL to run the vLLM lifecycle acceptance test",
)
def test_vllm_close_recovers_gpu_memory() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for vLLM lifecycle validation")

    from llenvs.inference.backends import VLLMBackend

    def _measure_close_cycle() -> tuple[int, int]:
        baseline_free, _ = torch.cuda.mem_get_info()
        backend = VLLMBackend(
            model_path=_VALIDATION_MODEL,
            gpu_memory_utilization=0.5,
            max_model_len=512,
        )
        backend.generate_chat(
            [ChatMessage(role="user", content="Say hello.")],
            SamplingParams(max_tokens=4, temperature=0.0),
        )
        loaded_free, _ = torch.cuda.mem_get_info()
        backend.close()
        gc.collect()
        torch.cuda.empty_cache()
        closed_free, _ = torch.cuda.mem_get_info()
        consumed = baseline_free - loaded_free
        recovered = closed_free - loaded_free
        assert recovered >= 0.9 * consumed
        return baseline_free, closed_free

    _, first_closed = _measure_close_cycle()
    _, second_closed = _measure_close_cycle()
    assert abs(second_closed - first_closed) <= 1024**3
