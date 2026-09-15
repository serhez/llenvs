"""Tests for SingularityVLLMBackend (fully mocked, no GPU or .sif required)."""

from __future__ import annotations

import errno
import json
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeProc:
    """Stand-in for ``subprocess.Popen`` return value."""

    def __init__(self, argv, env, **kwargs):
        self.argv = argv
        self.env = env
        self.pid = 424242
        self.returncode: int | None = None
        self._killed_signals: list[int] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode if self.returncode is not None else 0


class _FakeResp:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_sif(tmp_path):
    sif = tmp_path / "fake.sif"
    sif.write_bytes(b"")
    return str(sif)


@pytest.fixture
def patched(tmp_path, fake_sif, monkeypatch):
    """Patch Popen, urlopen, OpenAIBackend, and os.killpg.

    Yields a dict with handles to the fakes so tests can assert on them.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLENVS_SIF", fake_sif)
    monkeypatch.setenv("LLENVS_BINDS", "")
    monkeypatch.delenv("LLENVS_HF_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("LLENVS_HF_HOME", raising=False)

    from llenvs.inference.backends import vllm_singularity as mod

    fake_proc_holder: dict[str, _FakeProc] = {}

    def fake_popen(argv, env=None, **kwargs):
        p = _FakeProc(argv, env, **kwargs)
        fake_proc_holder["p"] = p
        return p

    def fake_urlopen(url, timeout=None):
        return _FakeResp(status=200)

    killed: list[tuple[int, int]] = []

    def fake_killpg(pid, sig):
        killed.append((pid, sig))
        # Mark the recorded proc as exited so subsequent .wait() returns
        if "p" in fake_proc_holder:
            fake_proc_holder["p"].returncode = -sig

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    from llenvs.inference.backends.api import _AsyncRunner

    # Mock the SDK, but retain the real connection-loop lifetime used by scoring.
    runner = _AsyncRunner()
    mock_openai = MagicMock()
    mock_openai.return_value._async_runner = runner
    monkeypatch.setattr(mod, "OpenAIBackend", mock_openai)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(mod.os, "killpg", fake_killpg)
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)

    yield {
        "module": mod,
        "proc_holder": fake_proc_holder,
        "killed": killed,
    }
    runner.close()


class TestSingularityVLLMBackendLifecycle:
    def test_spawn_argv_contains_expected_flags(self, patched, fake_sif):
        mod = patched["module"]

        backend = mod.SingularityVLLMBackend(
            model_path="google/gemma-4-31B-it",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            dtype="bfloat16",
        )
        try:
            argv = patched["proc_holder"]["p"].argv
            assert argv[0] == "singularity"
            assert argv[1] == "exec"
            assert "--nv" in argv
            assert fake_sif in argv
            assert "vllm" in argv and "serve" in argv
            assert "google/gemma-4-31B-it" in argv
            # tensor-parallel-size flag is followed by "2"
            tp_idx = argv.index("--tensor-parallel-size")
            assert argv[tp_idx + 1] == "2"
            assert "--max-model-len" in argv
            assert "--dtype" in argv
            assert "--host" in argv and "127.0.0.1" in argv
            # a dynamic port was picked
            port_idx = argv.index("--port")
            assert int(argv[port_idx + 1]) > 0
        finally:
            backend.close()

    def test_cuda_visible_devices_passed_via_env(self, patched):
        mod = patched["module"]

        backend = mod.SingularityVLLMBackend(
            model_path="google/gemma-4-31B-it",
            cuda_visible_devices="0,1",
        )
        try:
            env = patched["proc_holder"]["p"].env
            assert env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "0,1"
        finally:
            backend.close()

    def test_hf_offline_env_injection(self, patched, monkeypatch):
        mod = patched["module"]
        monkeypatch.setenv("LLENVS_HF_OFFLINE", "1")

        backend = mod.SingularityVLLMBackend(
            model_path="google/gemma-4-31B-it",
        )
        try:
            env = patched["proc_holder"]["p"].env
            assert env["SINGULARITYENV_HF_HUB_OFFLINE"] == "1"
            assert env["SINGULARITYENV_TRANSFORMERS_OFFLINE"] == "1"
        finally:
            backend.close()

    def test_missing_sif_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LLENVS_SIF", raising=False)
        monkeypatch.delenv("LLENVS_VLLM_SIF", raising=False)

        from llenvs.inference.backends.vllm_singularity import (
            SingularityVLLMBackend,
        )

        with pytest.raises(RuntimeError, match="no .sif path"):
            SingularityVLLMBackend(model_path="google/gemma-4-31B-it")

    def test_sif_path_not_a_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LLENVS_SIF", str(tmp_path / "does-not-exist.sif"))

        from llenvs.inference.backends.vllm_singularity import (
            SingularityVLLMBackend,
        )

        with pytest.raises(RuntimeError, match="\\.sif not found"):
            SingularityVLLMBackend(model_path="google/gemma-4-31B-it")

    def test_close_sigterm_pgroup(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        pid = patched["proc_holder"]["p"].pid
        backend.close()
        assert (pid, signal.SIGTERM) in patched["killed"]

    def test_double_close_is_noop(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        backend.close()
        n_kills = len(patched["killed"])
        backend.close()
        assert len(patched["killed"]) == n_kills

    def test_context_manager(self, patched):
        mod = patched["module"]
        with mod.SingularityVLLMBackend(model_path="m") as b:
            assert b.pid == 424242
            assert b.port > 0
        assert len(patched["killed"]) >= 1

    def test_subprocess_dies_during_startup(self, patched, monkeypatch):
        mod = patched["module"]

        # Override Popen so the process is already-dead when health check runs.
        def fake_popen_dead(argv, env=None, **kwargs):
            p = _FakeProc(argv, env, **kwargs)
            p.returncode = 1
            patched["proc_holder"]["p"] = p
            return p

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_dead)

        with pytest.raises(RuntimeError, match="vllm serve exited"):
            mod.SingularityVLLMBackend(model_path="m")

    def test_health_check_times_out(self, patched, monkeypatch):
        mod = patched["module"]
        import urllib.error

        def raise_urlerror(url, timeout=None):
            raise urllib.error.URLError("refused")

        monkeypatch.setattr("urllib.request.urlopen", raise_urlerror)

        with pytest.raises(RuntimeError, match="health check timed out"):
            mod.SingularityVLLMBackend(
                model_path="m",
                startup_timeout=0.01,
                health_poll_interval=0.001,
            )

    def test_generate_chat_delegates_to_openai(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            from llenvs.inference.protocol import ChatMessage, SamplingParams

            msgs = [ChatMessage(role="user", content="hi")]
            params = SamplingParams(max_tokens=8)
            backend.generate_chat(msgs, params)
            assert backend._openai.generate_chat.call_count == 1  # type: ignore[union-attr]
            backend._openai.generate_chat.assert_called_with(msgs, params)  # type: ignore[union-attr]
        finally:
            backend.close()

    def test_capabilities_inherit_logprobs_from_inner_openai(self, patched):
        """Chat calls proxy to an inner OpenAIBackend that supports logprobs;
        the wrapper must surface that capability, not mask it. Mirrors the
        inner client both ways so a closed/non-logprob client reads False."""
        mod = patched["module"]
        from llenvs.inference.protocol import BackendCapabilities

        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            backend._openai.capabilities = BackendCapabilities(supports_logprobs=True)
            assert backend.capabilities.supports_logprobs is True

            backend._openai.capabilities = BackendCapabilities(supports_logprobs=False)
            assert backend.capabilities.supports_logprobs is False
        finally:
            backend.close()


class TestSingularityVLLMBackendThinkingToggle:
    """``disable_thinking`` must reach the server as ``chat_template_kwargs``.

    The server renders the chat template itself, so hybrid-reasoning models
    (Qwen3) default to thinking unless the request carries
    ``chat_template_kwargs={"enable_thinking": False}``.
    """

    def _params(self, **kwargs):
        from llenvs.inference.protocol import SamplingParams
        return SamplingParams(max_tokens=8, **kwargs)

    def test_generate_chat_injects_template_kwargs(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            msgs = _msgs("hi")
            backend.generate_chat(msgs, self._params(disable_thinking=True))
            (sent_msgs, sent_params), _ = backend._openai.generate_chat.call_args
            assert sent_msgs == msgs
            assert sent_params.extra["extra_body"]["chat_template_kwargs"] == {
                "enable_thinking": False
            }
            assert sent_params.disable_thinking is True
        finally:
            backend.close()

    def test_generate_chat_batch_injects_template_kwargs(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            batch = [_msgs("a"), _msgs("b")]
            backend.generate_chat_batch(batch, self._params(disable_thinking=True))
            (sent_batch, sent_params), _ = (
                backend._openai.generate_chat_batch.call_args
            )
            assert sent_batch == batch
            assert sent_params.extra["extra_body"]["chat_template_kwargs"] == {
                "enable_thinking": False
            }
        finally:
            backend.close()

    def test_no_disable_thinking_forwards_params_unchanged(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            params = self._params()
            backend.generate_chat(_msgs("hi"), params)
            (_, sent_params), _ = backend._openai.generate_chat.call_args
            assert sent_params is params
        finally:
            backend.close()

    def test_caller_extra_preserved_and_wins_on_collision(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            original_extra = {
                "top_level": "kept",
                "extra_body": {
                    "other_field": "kept",
                    "chat_template_kwargs": {"enable_thinking": True, "custom": 1},
                },
            }
            params = self._params(disable_thinking=True, extra=original_extra)
            backend.generate_chat(_msgs("hi"), params)
            (_, sent_params), _ = backend._openai.generate_chat.call_args
            assert sent_params.extra["top_level"] == "kept"
            assert sent_params.extra["extra_body"]["other_field"] == "kept"
            # Explicit caller values win over the injected default.
            assert sent_params.extra["extra_body"]["chat_template_kwargs"] == {
                "enable_thinking": True,
                "custom": 1,
            }
            # The caller's params object and dicts are never mutated.
            assert params.extra == original_extra
            assert params.extra["extra_body"]["chat_template_kwargs"] == {
                "enable_thinking": True,
                "custom": 1,
            }
        finally:
            backend.close()

    def test_raw_generate_is_not_transformed(self, patched):
        """``generate`` takes pre-rendered text prompts — no chat template,
        so there is nothing to toggle and params pass through untouched."""
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            params = self._params(disable_thinking=True)
            backend.generate(["raw prompt"], params)
            (_, sent_params), _ = backend._openai.generate.call_args
            assert sent_params is params
        finally:
            backend.close()


class _CharTokenizer:
    """Deterministic fake tokenizer: fixed prompt, encode = code points."""

    PROMPT = "PROMPT|"

    def __init__(self) -> None:
        self.template_kwargs_seen: list[dict] = []

    def apply_chat_template(self, conversation, *, tokenize=False,
                            add_generation_prompt=True, **kwargs) -> str:
        self.template_kwargs_seen.append(kwargs)
        return self.PROMPT

    def encode(self, text: str, **kwargs) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


class _FakeCompletions:
    """Async /v1/completions stub that echoes prompt_logprobs for the prompt.

    Builds an HTTP-shaped prompt_logprobs list from the token-id prompt it
    receives (one entry per position, string token-id keys), so tests can
    assert exact span extraction. Records every call for request assertions.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, *, model, prompt, max_tokens, temperature,
                     extra_body=None, **kwargs):
        self.calls.append({
            "model": model, "prompt": list(prompt), "max_tokens": max_tokens,
            "temperature": temperature, "extra_body": extra_body, **kwargs,
        })
        full_ids = list(prompt)
        plps: list = [None]
        for pos in range(1, len(full_ids)):
            tid = full_ids[pos]
            plps.append({
                str(tid): {"logprob": round(-0.1 * pos, 4),
                           "decoded_token": chr(tid), "rank": 1},
            })
        return SimpleNamespace(choices=[SimpleNamespace(prompt_logprobs=plps)])


def _msgs(text: str):
    from llenvs.inference.protocol import ChatMessage
    return [ChatMessage(role="user", content=text)]


def _wire_scoring(backend, tokenizer=None):
    """Attach a fake tokenizer and fake async completions client."""
    tok = tokenizer or _CharTokenizer()
    backend._tokenizer = tok
    fake = _FakeCompletions()
    backend._openai._async_client.completions = fake
    return tok, fake


@pytest.mark.parametrize("operation", ["generate", "generate_chat", "generate_chat_batch", "score_chat_batch"])
@pytest.mark.parametrize("returncode", [0, 1, -9])
def test_exited_server_rejects_calls_without_http(patched, operation, returncode):
    from llenvs.inference.protocol import BackendProcessExitedError, SamplingParams

    backend = patched["module"].SingularityVLLMBackend(model_path="m")
    try:
        _, fake = _wire_scoring(backend)
        patched["proc_holder"]["p"].returncode = returncode
        args = {
            "generate": (["hello"], SamplingParams()),
            "generate_chat": (_msgs("hello"), SamplingParams()),
            "generate_chat_batch": ([_msgs("hello")], SamplingParams()),
            "score_chat_batch": ([_msgs("hello")], ["A"]),
        }[operation]
        with pytest.raises(BackendProcessExitedError, match=f"rc={returncode}") as caught:
            getattr(backend, operation)(*args)
        assert backend._log_path in str(caught.value)
        assert not fake.calls
        for name in ("generate", "generate_chat", "generate_chat_batch"):
            getattr(backend._openai, name).assert_not_called()
    finally:
        backend.close()


@pytest.mark.parametrize("dead", [False, True])
def test_server_death_during_scoring_preserves_success_and_failure_cause(patched, dead):
    from llenvs.inference.protocol import BackendProcessExitedError, PartialBatchError

    backend = patched["module"].SingularityVLLMBackend(model_path="m", max_concurrency=1)
    try:
        _, fake = _wire_scoring(backend)
        original_create = fake.create
        failure = ConnectionError("temporarily unavailable")

        async def create(**kwargs):
            if kwargs["prompt"][-1] == ord("B"):
                if dead:
                    patched["proc_holder"]["p"].returncode = 1
                raise failure
            return await original_create(**kwargs)

        fake.create = create
        with pytest.raises(PartialBatchError) as caught:
            backend.score_chat_batch([_msgs("a")] * 4, ["", "A", "B", "C"])
        exc = caught.value
        assert exc.results[0].scored_tokens == 0
        assert exc.results[1].token_scores[0].token_id == ord("A")
        if dead:
            assert set(exc.failures) == {2, 3}
            assert all(isinstance(e, BackendProcessExitedError) for e in exc.failures.values())
            assert exc.failures[2].__cause__ is failure
            assert len(fake.calls) == 1  # C must not reach the dead server.
        else:
            assert exc.failures == {2: failure}
            assert exc.results[3].token_scores[0].token_id == ord("C")
    finally:
        backend.close()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("dead", [False, True])
def test_generation_failure_checks_owned_process_without_losing_siblings(patched, partial, dead):
    from llenvs.inference.protocol import (
        BackendProcessExitedError,
        GenerationResult,
        PartialBatchError,
        SamplingParams,
    )

    backend = patched["module"].SingularityVLLMBackend(model_path="m")
    good = GenerationResult(text="A")
    failure = ConnectionError("temporarily unavailable")
    original = PartialBatchError([good, failure], {1: failure}) if partial else failure

    def fail(*args):
        if dead:
            patched["proc_holder"]["p"].returncode = 1
        raise original

    try:
        backend._openai.generate_chat_batch.side_effect = fail
        expected = PartialBatchError if partial else BackendProcessExitedError if dead else ConnectionError
        with pytest.raises(expected) as caught:
            backend.generate_chat_batch([_msgs("a"), _msgs("b")], SamplingParams())
        if not dead:
            assert caught.value is original
        elif partial:
            assert caught.value.results[0] is good
            assert set(caught.value.failures) == {1}
            assert isinstance(caught.value.failures[1], BackendProcessExitedError)
            assert caught.value.failures[1].__cause__ is failure
    finally:
        backend.close()


def test_successful_batch_is_not_discarded_if_server_exits_after_reply(patched):
    from llenvs.inference.protocol import GenerationResult, SamplingParams

    backend = patched["module"].SingularityVLLMBackend(model_path="m")
    good = [GenerationResult(text="A")]

    def complete(*args):
        patched["proc_holder"]["p"].returncode = 1
        return good

    try:
        backend._openai.generate_chat_batch.side_effect = complete
        assert backend.generate_chat_batch([_msgs("a")], SamplingParams()) is good
    finally:
        backend.close()


@pytest.mark.parametrize("partial", [False, True])
def test_refused_owned_server_is_fatal_even_while_launcher_is_alive(patched, partial):
    import asyncio

    from llenvs.inference.protocol import (
        BackendProcessExitedError,
        GenerationResult,
        PartialBatchError,
        SamplingParams,
    )

    backend = patched["module"].SingularityVLLMBackend(model_path="m")
    good = GenerationResult(text="A")
    cancellation = asyncio.CancelledError()
    refused = OSError(errno.ECONNREFUSED, "connection refused")
    failure = ConnectionError("All connection attempts failed")
    failure.__cause__ = ExceptionGroup("connection attempts", [refused])
    original = PartialBatchError([good, failure, cancellation], {1: failure, 2: cancellation}) if partial else failure
    try:
        backend._openai.generate_chat_batch.side_effect = original
        with pytest.raises(PartialBatchError if partial else BackendProcessExitedError) as caught:
            backend.generate_chat_batch([_msgs("a")] * 3, SamplingParams())
        if partial:
            assert caught.value.results[0] is good
            assert caught.value.failures[2] is cancellation
            error = caught.value.failures[1]
        else:
            error = caught.value
        assert isinstance(error, BackendProcessExitedError)
        assert error.__cause__ is failure
        assert backend._log_path in str(error)
        assert patched["proc_holder"]["p"].poll() is None
        backend._openai.generate_chat_batch.reset_mock()
        with pytest.raises(BackendProcessExitedError):
            backend.generate_chat_batch([_msgs("again")], SamplingParams())
        backend._openai.generate_chat_batch.assert_not_called()
    finally:
        backend.close()


@pytest.mark.parametrize("failure", [
    TimeoutError("connection refused is mentioned, but this is a timeout"),
    OSError(errno.ECONNRESET, "peer reset the connection"),
    ConnectionError("connection refused"),
])
def test_live_server_transient_errors_are_not_misclassified_by_message(patched, failure):
    from llenvs.inference.protocol import SamplingParams

    backend = patched["module"].SingularityVLLMBackend(model_path="m")
    try:
        # Cyclic exception context must not hang failure inspection.
        failure.__context__ = failure
        backend._openai.generate_chat.side_effect = failure
        with pytest.raises(type(failure)) as caught:
            backend.generate_chat(_msgs("a"), SamplingParams())
        assert caught.value is failure
        backend._openai.generate_chat.side_effect = None
        backend.generate_chat(_msgs("b"), SamplingParams())
        assert backend._openai.generate_chat.call_count == 2
    finally:
        backend.close()


def test_refused_server_during_scoring_retains_siblings_and_skips_later_http(patched):
    from llenvs.inference.protocol import BackendProcessExitedError, PartialBatchError

    backend = patched["module"].SingularityVLLMBackend(model_path="m", max_concurrency=1)
    try:
        _, fake = _wire_scoring(backend)
        original_create = fake.create
        failure = ConnectionRefusedError(errno.ECONNREFUSED, "refused")
        attempted = []

        async def create(**kwargs):
            attempted.append(kwargs["prompt"][-1])
            if kwargs["prompt"][-1] == ord("B"):
                raise failure
            return await original_create(**kwargs)

        fake.create = create
        with pytest.raises(PartialBatchError) as caught:
            backend.score_chat_batch([_msgs("a")] * 4, ["", "A", "B", "C"])
        assert caught.value.results[0].scored_tokens == 0
        assert caught.value.results[1].token_scores[0].token_id == ord("A")
        assert set(caught.value.failures) == {2, 3}
        assert all(isinstance(e, BackendProcessExitedError) for e in caught.value.failures.values())
        assert caught.value.failures[2].__cause__ is failure
        assert attempted == [ord("A"), ord("B")]
        assert patched["proc_holder"]["p"].poll() is None
    finally:
        backend.close()


@pytest.mark.parametrize("partial", [False, True])
def test_server_exit_does_not_reclassify_cancellation(patched, partial):
    import asyncio

    from llenvs.inference.protocol import PartialBatchError, SamplingParams

    backend = patched["module"].SingularityVLLMBackend(model_path="m")
    cancellation = asyncio.CancelledError()

    def cancel(*args):
        patched["proc_holder"]["p"].returncode = 1
        if partial:
            raise PartialBatchError([cancellation], {0: cancellation})
        raise cancellation

    try:
        backend._openai.generate_chat_batch.side_effect = cancel
        with pytest.raises(PartialBatchError if partial else asyncio.CancelledError) as caught:
            backend.generate_chat_batch([_msgs("a")], SamplingParams())
        if partial:
            assert caught.value.failures == {0: cancellation}
        else:
            assert caught.value is cancellation
    finally:
        backend.close()


def test_host_tokenizer_recovers_from_legacy_extra_special_tokens_shape(
    tmp_path, monkeypatch
):
    """Transformers 4.x expects a mapping where Gemma's v5 metadata has a list.

    The fallback must preserve those entries as ordinary additional special
    tokens instead of requiring a host Transformers upgrade or mutating the HF
    cache shared with the container.
    """
    from llenvs.inference.backends import vllm_singularity as mod

    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"extra_special_tokens": ["<|video|>"]})
    )
    sentinel = object()

    class _FakeAutoTokenizer:
        calls: list[tuple[str, dict]] = []

        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            cls.calls.append((model_path, kwargs))
            if len(cls.calls) == 1:
                raise AttributeError("'list' object has no attribute 'keys'")
            return sentinel

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        mod,
        "_resolve_tokenizer_config_path",
        lambda _model_path: tokenizer_config,
        raising=False,
    )

    assert mod._load_hf_tokenizer("google/gemma-4-31B-it") is sentinel
    assert _FakeAutoTokenizer.calls == [
        ("google/gemma-4-31B-it", {}),
        (
            "google/gemma-4-31B-it",
            {
                "extra_special_tokens": {},
                "additional_special_tokens": ["<|video|>"],
            },
        ),
    ]


class TestSingularityVLLMBackendScoring:
    @pytest.mark.parametrize("continuations", [
        ["A", "B", "A"], ["", "A", "", "B", "A", ""],
        ["B"], ["", "B", "", "B"],
    ])
    @pytest.mark.parametrize("failure_kind", ["input", "connection", "fatal"])
    def test_partial_scoring_preserves_successes_and_original_indices(
        self, patched, continuations, failure_kind,
    ):
        from llenvs.inference.protocol import (
            PartialBatchError,
            RecoverableInputError,
            ScoringResult,
            TokenScore,
        )

        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        error_type = {
            "input": RecoverableInputError,
            "connection": ConnectionError,
            "fatal": RuntimeError,
        }[failure_kind]
        failure = error_type("synthetic scoring failure")
        calls = []

        async def score_one(prompt_len, ids):
            calls.append(ids[-1])
            if ids[-1] == ord("B"):
                raise failure
            return ScoringResult(
                token_scores=(TokenScore("A", ord("A"), -2),), scored_tokens=1,
            )

        try:
            _wire_scoring(backend)
            backend._score_one_async = score_one
            with pytest.raises(PartialBatchError) as caught:
                backend.score_chat_batch(
                    [_msgs(str(i)) for i in range(len(continuations))], continuations,
                )
            exc = caught.value
            assert exc.failures == {
                i: failure for i, c in enumerate(continuations) if c == "B"
            }
            assert len(exc.results) == len(continuations)
            for i, continuation in enumerate(continuations):
                if continuation == "B":
                    assert exc.results[i] is failure
                elif continuation == "":
                    assert exc.results[i].scored_tokens == 0
                else:
                    assert exc.results[i].token_scores[0].logprob == -2
            assert calls == [ord(c) for c in continuations if c]
        finally:
            backend.close()

    def test_supports_full_scoring_true_when_open_false_when_closed(self, patched):
        mod = patched["module"]
        from llenvs.inference.protocol import BackendCapabilities

        backend = mod.SingularityVLLMBackend(model_path="m")
        backend._openai.capabilities = BackendCapabilities(supports_logprobs=True)
        assert backend.capabilities.supports_full_scoring is True
        backend.close()
        assert backend.capabilities.supports_full_scoring is False

    def test_score_chat_batch_extracts_spans_in_order(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            _wire_scoring(backend)
            results = backend.score_chat_batch(
                [_msgs("a"), _msgs("b")], ["AB", "C"],
            )
            assert len(results) == 2
            # item 0: continuation "AB" -> positions 7,8 -> logprobs -0.7,-0.8
            assert results[0].prompt_tokens == len("PROMPT|")
            assert results[0].scored_tokens == 2
            assert [t.logprob for t in results[0].token_scores] == [-0.7, -0.8]
            assert [t.token_id for t in results[0].token_scores] == [ord("A"), ord("B")]
            # item 1: continuation "C" -> position 7 -> logprob -0.7
            assert results[1].scored_tokens == 1
            assert results[1].token_scores[0].logprob == -0.7
            assert results[1].token_scores[0].token_id == ord("C")
        finally:
            backend.close()

    def test_request_uses_token_id_prompt_and_prompt_logprobs(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            _tok, fake = _wire_scoring(backend)
            backend.score_chat_batch([_msgs("a")], ["AB"])
            assert len(fake.calls) == 1
            call = fake.calls[0]
            assert call["extra_body"] == {"prompt_logprobs": 0}
            assert call["max_tokens"] == 1
            assert call["temperature"] == 0
            assert call["prompt"] == [ord(c) for c in "PROMPT|AB"]
            assert call["model"] == backend._served_model_name
        finally:
            backend.close()

    def test_thinking_disabled_for_scoring(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            tok, _ = _wire_scoring(backend)
            backend.score_chat_batch([_msgs("a")], ["AB"])
            assert tok.template_kwargs_seen
            assert all(kw.get("enable_thinking") is False
                       for kw in tok.template_kwargs_seen)
        finally:
            backend.close()

    def test_empty_continuation_is_not_sent(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            _tok, fake = _wire_scoring(backend)
            results = backend.score_chat_batch(
                [_msgs("a"), _msgs("b")], ["", "AB"],
            )
            # Only the non-empty continuation hits the server.
            assert len(fake.calls) == 1
            assert results[0].token_scores == ()
            assert results[0].scored_tokens == 0
            assert results[1].scored_tokens == 2
        finally:
            backend.close()

    def test_all_empty_continuations_make_no_requests(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            _tok, fake = _wire_scoring(backend)
            results = backend.score_chat_batch([_msgs("a"), _msgs("b")], ["", ""])
            assert fake.calls == []
            assert all(r.scored_tokens == 0 for r in results)
        finally:
            backend.close()

    def test_score_chat_single_matches_batch(self, patched):
        mod = patched["module"]
        backend = mod.SingularityVLLMBackend(model_path="m")
        try:
            _wire_scoring(backend)
            single = backend.score_chat(_msgs("a"), "AB")
            assert single.scored_tokens == 2
            assert [t.logprob for t in single.token_scores] == [-0.7, -0.8]
        finally:
            backend.close()

    def test_tokenizer_lazy_loaded_and_memoized(self, patched, monkeypatch):
        mod = patched["module"]
        loaded: list[str] = []
        fake_tok = _CharTokenizer()
        monkeypatch.setattr(
            mod, "_load_hf_tokenizer",
            lambda model_path: (loaded.append(model_path) or fake_tok),
        )
        backend = mod.SingularityVLLMBackend(model_path="some/model")
        try:
            fake = _FakeCompletions()
            backend._openai._async_client.completions = fake
            backend.score_chat_batch([_msgs("a")], ["AB"])
            backend.score_chat_batch([_msgs("b")], ["CD"])
            assert loaded == ["some/model"]  # loaded once, then memoized
        finally:
            backend.close()
