"""Exercise pooled HTTP connections across public synchronous batch calls.

The loopback server keeps connections alive; mocked SDK methods do not expose
the event-loop ownership of real HTTP connections. No external API is contacted.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from llenvs.core.tools import ToolDefinition
from llenvs.inference.backends.api import AnthropicBackend, OpenAIBackend, OpenRouterBackend
from llenvs.inference.protocol import ChatMessage, PartialBatchError, SamplingParams


@pytest.fixture
def http_server():
    requests = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append((self.path, body))
            content = body.get("messages", [{"content": "completion"}])[-1]["content"]
            if isinstance(content, list):
                content = "".join(block.get("text", "") for block in content)
            status = 400 if content == "reject" else 200
            if status == 400:
                payload = {
                    "error": {"message": "Rejected by test server", "type": "invalid_request_error"}
                }
            elif self.path.endswith("/messages"):
                payload = {
                    "id": "fake",
                    "type": "message",
                    "role": "assistant",
                    "model": "fake",
                    "content": [{"type": "text", "text": content}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            elif self.path.endswith("/chat/completions"):
                payload = {
                    "id": "fake",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fake",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                ids = body["prompt"]
                payload = {
                    "id": "fake",
                    "object": "text_completion",
                    "created": 0,
                    "model": "fake",
                    "choices": [
                        {
                            "index": 0,
                            "text": "x",
                            "finish_reason": "stop",
                            "prompt_logprobs": [None]
                            + [
                                {
                                    str(token): {
                                        "logprob": -0.1,
                                        "rank": 1,
                                        "decoded_token": chr(token),
                                    }
                                }
                                for token in ids[1:]
                            ],
                        }
                    ],
                }
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}/v1", requests=requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.fixture(params=[OpenAIBackend, OpenRouterBackend, AnthropicBackend])
def backend(request, http_server):
    pytest.importorskip("anthropic" if request.param is AnthropicBackend else "openai")
    instance = request.param(
        model="fake",
        api_key="dummy",
        base_url=http_server.url,
        max_retries=0,
        max_concurrency=4,
        timeout=2,
    )
    yield instance
    instance.close()


def messages(text):
    return [ChatMessage(role="user", content=text)]


PARAMS = SamplingParams(max_tokens=16)


@pytest.mark.parametrize("backend_type", [OpenAIBackend, OpenRouterBackend])
@pytest.mark.parametrize("connect,read", [(None, None), (2.0, None), (None, 0.25), (2.0, 0.25)])
def test_read_timeout_is_independent_of_connect_and_preserves_sdk_defaults(
    backend_type, connect, read
):
    sdk = pytest.importorskip("openai")
    kwargs = {}
    if connect is not None:
        kwargs["timeout"] = connect
    if read is not None:
        kwargs["read_timeout"] = read
    instance = backend_type(
        model="fake", api_key="dummy", base_url="http://127.0.0.1:1/v1", max_retries=0, **kwargs
    )
    try:
        for client in (instance._client, instance._async_client):
            assert client.timeout.read == (sdk.DEFAULT_TIMEOUT.read if read is None else read)
            assert client.timeout.connect == (
                sdk.DEFAULT_TIMEOUT.connect if connect is None else connect
            )
            assert client.timeout.write == sdk.DEFAULT_TIMEOUT.write
            assert client.timeout.pool == sdk.DEFAULT_TIMEOUT.pool
            assert client.max_retries == 0
    finally:
        instance.close()


@pytest.mark.parametrize("backend_type", [OpenAIBackend, OpenRouterBackend])
@pytest.mark.parametrize("read", [0, -1, True, "2", float("nan"), float("inf")])
def test_invalid_read_timeouts_are_rejected_before_constructing_clients(
    backend_type, read, monkeypatch
):
    sdk = pytest.importorskip("openai")

    def forbidden(**kwargs):
        raise AssertionError("Invalid timeout reached the SDK constructor")

    monkeypatch.setattr(sdk, "OpenAI", forbidden)
    monkeypatch.setattr(sdk, "AsyncOpenAI", forbidden)
    with pytest.raises(ValueError):
        backend_type(model="fake", api_key="dummy", read_timeout=read)


def test_read_timeout_does_not_change_model_or_call_identity():
    first = OpenAIBackend(model="fake", api_key="dummy", read_timeout=1, max_retries=0)
    second = OpenAIBackend(model="fake", api_key="dummy", read_timeout=3, max_retries=0)
    try:
        assert first.descriptor()["identity"] == second.descriptor()["identity"]
    finally:
        first.close()
        second.close()


def test_decisions_uses_the_configured_read_timeout():
    sdk = pytest.importorskip("openai")
    if not hasattr(sdk.AsyncOpenAI, "decisions"):
        pytest.skip("Decisions requires the isolated SDK 3 environment")
    from llenvs.inference.decisions.openai import OpenAIDecisionBackend

    instance = OpenAIDecisionBackend(api_key="dummy", timeout=2, read_timeout=0.25)
    try:
        for client in (instance._client, instance._async_client):
            assert client.timeout.read == 0.25 and client.timeout.connect == 2
            assert client.max_retries == 0
    finally:
        instance.close()


def test_real_read_timeout_keeps_a_durable_unknown_without_another_request(tmp_path):
    from llenvs.inference.persistence import CallStore, PersistentBackend, UnknownOutcomeError

    entered, release = threading.Event(), threading.Event()
    calls = []

    class StalledResponse(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            entered.set()
            release.wait(5)
            self.close_connection = True

    with ThreadingHTTPServer(("127.0.0.1", 0), StalledResponse) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        instance = None
        try:
            instance = OpenAIBackend(
                model="fake",
                api_key="dummy",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                timeout=1,
                read_timeout=0.05,
                max_retries=0,
            )
            with CallStore.open(tmp_path / "calls", run_descriptor={}) as store:
                wrapped = PersistentBackend(instance, store)
                with pytest.raises(UnknownOutcomeError) as first:
                    wrapped.generate_chat(messages("stalled"), PARAMS, repeat_id=0)
                assert entered.is_set() and len(calls) == 1
                with pytest.raises(UnknownOutcomeError) as second:
                    wrapped.generate_chat(messages("stalled"), PARAMS, repeat_id=0)
                assert second.value.key == first.value.key and len(calls) == 1
            with CallStore.open(tmp_path / "calls", run_descriptor={}) as reopened:
                with pytest.raises(UnknownOutcomeError):
                    PersistentBackend(instance, reopened).generate_chat(
                        messages("stalled"), PARAMS, repeat_id=0
                    )
            assert len(calls) == 1
        finally:
            release.set()
            if instance is not None:
                instance.close()
            server.shutdown()
            worker.join(timeout=3)
        assert not worker.is_alive()


def test_loopback_server_handles_reused_connections_on_one_event_loop(http_server):
    """Positive control: no server failure or SDK retry is needed."""
    sdk = pytest.importorskip("openai")

    async def run():
        async with sdk.AsyncOpenAI(
            api_key="dummy", base_url=http_server.url, max_retries=0
        ) as client:
            for i in range(6):
                result = await client.chat.completions.create(
                    model="fake",
                    messages=[{"role": "user", "content": str(i)}],
                    max_tokens=16,
                )
                assert result.choices[0].message.content == str(i)

    asyncio.run(run())
    assert len(http_server.requests) == 6


def test_repeated_batches_preserve_order_without_hidden_retries(backend, http_server):
    for batch in range(6):
        expected = [f"batch-{batch}-item-{i}" for i in range(4)]
        results = backend.generate_chat_batch([messages(s) for s in expected], PARAMS)
        assert [r.text for r in results] == expected
    assert len(http_server.requests) == 24


def test_alternating_chat_and_tool_batches_share_valid_connections(backend, http_server):
    tools = [ToolDefinition(name="noop", description="Test tool", parameters=())]
    for batch in range(6):
        prompt = messages(f"item-{batch}")
        if batch % 2:
            results = backend.generate_with_tools_batch([prompt], tools, PARAMS)
        else:
            results = backend.generate_chat_batch([prompt], PARAMS)
        assert results[0].text == f"item-{batch}"
    assert len(http_server.requests) == 6


def test_sync_batch_api_works_inside_a_running_event_loop(backend, http_server):
    async def run():
        for batch in range(4):
            assert backend.generate_chat_batch([messages(str(batch))], PARAMS)[0].text == str(batch)
        backend.close()
        backend.close()

    asyncio.run(run())
    assert len(http_server.requests) == 4


def test_backend_can_be_reused_from_different_caller_threads(backend, http_server):
    for batch in range(4):
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                backend.generate_chat_batch, [messages(str(batch))], PARAMS
            ).result(timeout=10)
        assert result[0].text == str(batch)
    assert len(http_server.requests) == 4


def test_close_after_used_connection_is_idempotent(backend):
    assert backend.generate_chat_batch([messages("ok")], PARAMS)[0].text == "ok"
    sync_client, async_client = backend._client, backend._async_client
    runner = backend._async_runner
    backend.close()
    backend.close()
    assert sync_client.is_closed()
    assert async_client.is_closed()
    assert runner._loop.is_closed()
    assert not runner._thread.is_alive()
    with pytest.raises(RuntimeError, match="API backend is closed"):
        backend.generate_chat_batch([messages("after close")], PARAMS)


def test_runtime_error_in_async_caller_does_not_repeat_the_operation():
    from llenvs.inference.backends.api import _run_concurrent

    calls = []

    async def fail(item):
        calls.append(item)
        raise RuntimeError("operation failed")

    async def run():
        with pytest.raises(RuntimeError, match="operation failed"):
            _run_concurrent(fail, ["once"], max_concurrency=1)

    asyncio.run(run())
    assert calls == ["once"]


def test_real_http_failure_preserves_partial_results_and_backend_reusability(backend, http_server):
    with pytest.raises(PartialBatchError) as caught:
        backend.generate_chat_batch(
            [messages("left"), messages("reject"), messages("right")], PARAMS
        )
    assert set(caught.value.failures) == {1}
    assert caught.value.results[0].text == "left"
    assert caught.value.results[2].text == "right"
    for text in ("retry", "next"):
        assert backend.generate_chat_batch([messages(text)], PARAMS)[0].text == text
    assert len(http_server.requests) == 5


@pytest.mark.parametrize("tools", [False, True])
def test_transport_exception_keeps_original_cause(backend, monkeypatch, tools):
    sdk = pytest.importorskip("anthropic" if isinstance(backend, AnthropicBackend) else "openai")
    import httpx

    cause = RuntimeError("transport diagnostic sentinel")
    error = sdk.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1/unused"))

    async def fail(**kwargs):
        raise error from cause

    endpoint = (
        backend._async_client.messages
        if isinstance(backend, AnthropicBackend)
        else backend._async_client.chat.completions
    )
    monkeypatch.setattr(endpoint, "create", fail)
    with pytest.raises(PartialBatchError) as caught:
        if tools:
            backend.generate_with_tools_batch([messages("test")], [], PARAMS)
        else:
            backend.generate_chat_batch([messages("test")], PARAMS)
    assert caught.value.failures[0] is error
    assert error.__cause__ is cause  # Never replace the cause with the exception itself.


def test_local_scoring_and_chat_share_connection_lifetime(http_server):
    """Real /completions transport; no GPU, container or model download."""
    pytest.importorskip("openai")
    from llenvs.inference.backends.vllm_singularity import SingularityVLLMBackend

    inner = OpenAIBackend(
        model="fake", api_key="dummy", base_url=http_server.url, max_retries=0, timeout=2
    )
    outer = SingularityVLLMBackend.__new__(SingularityVLLMBackend)
    outer._openai = inner
    # This HTTP-only shell stands in for a healthy owned inference process.
    outer._proc = SimpleNamespace(poll=lambda: None)
    outer._server_failure = None
    outer._served_model_name = "fake"
    outer._model_path = "fake"
    outer._max_concurrency = 2
    outer._chat_template_kwargs = {}
    outer._tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: "PROMPT|",
        encode=lambda text, **kwargs: list(map(ord, text)),
    )
    try:
        for _ in range(3):
            assert inner.generate_chat_batch([messages("chat")], PARAMS)[0].text == "chat"
            results = outer.score_chat_batch([messages("a"), messages("b")], ["AB", ""])
            assert results[0].prompt_tokens == len("PROMPT|")
            assert [t.token_id for t in results[0].token_scores] == [ord("A"), ord("B")]
            assert [t.logprob for t in results[0].token_scores] == pytest.approx([-0.1, -0.1])
            assert results[1].scored_tokens == 0
        assert len(http_server.requests) == 6
        inner.close()
    finally:
        inner.close()
        # The outer shell did not start a server, so it must not run its destructor.
        outer._openai = None


def test_owned_server_refusal_survives_real_httpx_exception_wrapping(tmp_path):
    """A closed loopback listener models a dead server; no external requests."""
    pytest.importorskip("openai")
    from llenvs.inference.backends.vllm_singularity import SingularityVLLMBackend
    from llenvs.inference.protocol import BackendProcessExitedError

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        listener.close()
        inner = OpenAIBackend(
            model="fake",
            api_key="dummy",
            base_url=f"http://127.0.0.1:{port}/v1",
            max_retries=0,
            timeout=2,
        )
        outer = SingularityVLLMBackend.__new__(SingularityVLLMBackend)
        outer._openai = inner
        outer._proc = SimpleNamespace(poll=lambda: None)
        outer._log_path = str(tmp_path / "server.log")
        outer._server_failure = None
        outer._reasoning_family = None
        outer._chat_template_kwargs = {}
        try:
            with pytest.raises(PartialBatchError) as caught:
                outer.generate_chat_batch([messages("test")], PARAMS)
            assert isinstance(caught.value.failures[0], BackendProcessExitedError)
            assert caught.value.failures[0].__cause__ is not None
            # The service stays failed even while the launcher reports alive.
            with pytest.raises(BackendProcessExitedError):
                outer.generate_chat_batch([messages("again")], PARAMS)
        finally:
            inner.close()
            outer._openai = None
