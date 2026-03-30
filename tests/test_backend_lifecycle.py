"""Tests for backend lifecycle hooks and best-effort teardown."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from llenvs.inference.protocol import (
    BackendCapabilities,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
)


class _MinimalBackend(ModelBackend):
    def __init__(self) -> None:
        self.closed = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True)

    @property
    def model_name(self) -> str:
        return "minimal"

    def generate(self, prompts: list[str], params: SamplingParams) -> list[GenerationResult]:
        return [GenerationResult(text="ok", finish_reason=StopReason.END_OF_TEXT)]

    def close(self) -> None:
        self.closed += 1
        super().close()


class _SyncCloseClient:
    def __init__(self) -> None:
        self.calls = 0

    def close(self) -> None:
        self.calls += 1


class _AsyncCloseClient:
    def __init__(self) -> None:
        self.calls = 0

    async def close(self) -> None:
        self.calls += 1


class TestModelBackendLifecycle:
    def test_default_close_is_callable(self):
        backend = _MinimalBackend()

        backend.close()

        assert backend.closed == 1

    def test_context_manager_calls_close(self):
        backend = _MinimalBackend()

        with backend as managed:
            assert managed is backend

        assert backend.closed == 1


class TestAPIBackendLifecycle:
    def test_openai_backend_closes_sync_and_async_clients(self):
        from llenvs.inference.backends.api import OpenAIBackend

        backend = OpenAIBackend.__new__(OpenAIBackend)
        sync_client = _SyncCloseClient()
        async_client = _AsyncCloseClient()
        backend._client = sync_client
        backend._async_client = async_client

        backend.close()
        backend.close()

        assert sync_client.calls == 1
        assert async_client.calls == 1

    def test_close_bridges_async_client_inside_running_loop(self):
        from llenvs.inference.backends.api import OpenRouterBackend

        async def _run() -> None:
            backend = OpenRouterBackend.__new__(OpenRouterBackend)
            sync_client = _SyncCloseClient()
            async_client = _AsyncCloseClient()
            backend._client = sync_client
            backend._async_client = async_client
            backend.close()
            assert sync_client.calls == 1
            assert async_client.calls == 1

        asyncio.run(_run())

    def test_missing_clients_are_tolerated(self):
        from llenvs.inference.backends.api import AnthropicBackend

        backend = AnthropicBackend.__new__(AnthropicBackend)

        backend.close()
        backend.close()


class TestHuggingFaceBackendLifecycle:
    def test_close_clears_refs_and_is_idempotent(self):
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._closed = False
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._processor = MagicMock()
        backend._torch = MagicMock()
        backend._torch.cuda.is_available.return_value = True
        backend._torch.backends.mps.is_available.return_value = True

        backend.close()
        backend.close()

        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._processor is None
        backend._torch.cuda.empty_cache.assert_called_once()
        backend._torch.mps.empty_cache.assert_called_once()


class TestVLLMBackendLifecycle:
    def test_close_prefers_engine_core_shutdown_and_clears_refs(self):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._closed = False
        backend._tokenizer = MagicMock()
        backend._scoring_model = MagicMock()
        llm = MagicMock()
        llm.llm_engine.engine_core.shutdown = MagicMock()
        llm.llm_engine.model_executor.shutdown = MagicMock()
        backend._llm = llm

        backend.close()
        backend.close()

        llm.llm_engine.engine_core.shutdown.assert_called_once()
        llm.llm_engine.model_executor.shutdown.assert_not_called()
        assert backend._llm is None
        assert backend._tokenizer is None
        assert backend._scoring_model is None

    def test_close_falls_back_to_model_executor_shutdown(self):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._closed = False
        backend._tokenizer = MagicMock()
        backend._scoring_model = MagicMock()
        llm = MagicMock()
        del llm.llm_engine.engine_core
        llm.llm_engine.model_executor.shutdown = MagicMock()
        backend._llm = llm

        backend.close()
        backend.close()

        llm.llm_engine.model_executor.shutdown.assert_called_once()
        assert backend._llm is None
        assert backend._tokenizer is None
        assert backend._scoring_model is None

    def test_close_without_known_shutdown_path_still_clears_refs(self):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._closed = False
        backend._tokenizer = MagicMock()
        backend._scoring_model = MagicMock()
        llm = MagicMock()
        del llm.llm_engine.engine_core
        del llm.llm_engine.model_executor
        backend._llm = llm

        backend.close()
        backend.close()

        assert backend._llm is None
        assert backend._tokenizer is None
        assert backend._scoring_model is None

    def test_close_does_not_call_sleep(self):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._closed = False
        backend._tokenizer = MagicMock()
        backend._scoring_model = None
        llm = MagicMock()
        llm.llm_engine.engine_core.shutdown = MagicMock()
        llm.llm_engine.model_executor.shutdown = MagicMock()
        backend._llm = llm

        backend.close()

        llm.llm_engine.engine_core.shutdown.assert_called_once()
        llm.sleep.assert_not_called()
