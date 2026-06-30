"""Tests for SingularityVLLMBackend (fully mocked, no GPU or .sif required)."""

from __future__ import annotations

import io
import os
import signal
from unittest.mock import MagicMock, patch

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
    monkeypatch.setattr(mod, "OpenAIBackend", MagicMock())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(mod.os, "killpg", fake_killpg)
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)

    yield {
        "module": mod,
        "proc_holder": fake_proc_holder,
        "killed": killed,
    }


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
