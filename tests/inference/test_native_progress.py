"""Native async engine completion is observed before a blocked sibling returns."""

import asyncio
from types import SimpleNamespace

from llenvs.inference.backends.api import _AsyncRunner
from llenvs.inference.backends.vllm import VLLMBackend
from llenvs.inference.protocol import ChatMessage, SamplingParams


def test_native_async_completion_callbacks_are_individual():
    events = []

    class Engine:
        async def generate(self, prompt, params, request_id):
            if prompt == "slow":
                while ("completed", 0) not in events:
                    await asyncio.sleep(0)
            yield SimpleNamespace(
                finished=True,
                request_id=request_id,
                prompt_token_ids=[1],
                outputs=[
                    SimpleNamespace(token_ids=[2], text=prompt, finish_reason="stop", logprobs=None)
                ],
            )

    class Observer:
        def started(self, index, attempt):
            events.append(("started", index))

        def received(self, index, receipt):
            events.append(("received", index))

        def completed(self, index, outcome):
            events.append(("completed", index))

        def failed(self, index, error):
            raise error

    backend = VLLMBackend.__new__(VLLMBackend)
    backend._request_progress = True
    backend._llm = Engine()
    backend._to_vllm_params = lambda params: SimpleNamespace()
    backend._native_request = lambda native, params: {"request": {}}
    backend._reasoning_family = None
    backend._tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: messages[0]["content"]
    )
    backend._chat_template_kwargs = {}
    backend._async_runner = _AsyncRunner()
    try:
        results = backend.generate_chat_batch(
            [[ChatMessage("user", "fast")], [ChatMessage("user", "slow")]],
            SamplingParams(),
            observer=Observer(),
        )
        assert [result.text for result in results] == ["fast", "slow"]
        assert events.index(("completed", 0)) < events.index(("received", 1))
    finally:
        backend._async_runner.close()


def test_vllm_logprob_is_selected_by_sampled_id_not_mapping_order():
    backend = VLLMBackend.__new__(VLLMBackend)
    output = SimpleNamespace(
        token_ids=[7],
        logprobs=[
            {
                3: SimpleNamespace(logprob=-0.1, decoded_token="other"),
                7: SimpleNamespace(logprob=-2, decoded_token="sampled"),
            }
        ],
    )
    result = backend._extract_logprobs(output)
    assert result[0].token_id == 7 and result[0].logprob == -2


def test_api_runner_allows_distinct_invocations_to_progress_concurrently():
    from concurrent.futures import ThreadPoolExecutor

    runner = _AsyncRunner()

    async def one():
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=2)
        return arrived

    async def initialize():
        return asyncio.Event()

    ready = runner.run(initialize())
    arrived = 0
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(runner.run, one()) for _ in range(2)]
            assert [future.result(timeout=5) for future in futures] == [2, 2]
    finally:
        runner.close()
