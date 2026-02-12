"""Tests for vLLM backend (fully mocked, no GPU required)."""

from unittest.mock import MagicMock, patch

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
)


class TestVLLMChatTemplateKwargs:
    """Tests for chat_template_kwargs on VLLMBackend."""

    def _create_mock_backend(self, chat_template_kwargs=None):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "test-model"
        backend._tokenizer = MagicMock()
        backend._tokenizer.apply_chat_template.return_value = "formatted"
        backend._VLLMSamplingParams = MagicMock()
        backend._llm = MagicMock()
        backend._max_context_length = 4096
        backend._chat_template_kwargs = chat_template_kwargs or {}
        return backend

    def test_stored_correctly(self):
        """chat_template_kwargs stored on the backend instance."""
        backend = self._create_mock_backend({"enable_thinking": True})
        assert backend._chat_template_kwargs == {"enable_thinking": True}

    def test_passed_to_generate_chat(self):
        """chat_template_kwargs spread into apply_chat_template in generate_chat."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(backend, "generate", return_value=[GenerationResult(
            text="ok", finish_reason=StopReason.END_OF_TEXT,
        )]):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")],
                SamplingParams(max_tokens=10),
            )

        backend._tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is True

    def test_passed_to_generate_chat_batch(self):
        """chat_template_kwargs spread into apply_chat_template in generate_chat_batch."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(backend, "generate", return_value=[GenerationResult(
            text="ok", finish_reason=StopReason.END_OF_TEXT,
        )]):
            backend.generate_chat_batch(
                [[ChatMessage(role="user", content="hi")]],
                SamplingParams(max_tokens=10),
            )

        backend._tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is True


class TestVLLMThinkingBudget:
    """Tests for thinking_budget interception in VLLMBackend."""

    def _create_mock_backend(self):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "test-model"
        backend._tokenizer = MagicMock()
        backend._tokenizer.get_vocab.return_value = {"<think>": 100, "</think>": 101}
        backend._VLLMSamplingParams = MagicMock()
        backend._llm = MagicMock()
        backend._max_context_length = 4096
        backend._chat_template_kwargs = {}
        return backend

    def test_thinking_budget_popped_from_extra(self):
        """thinking_budget is removed from extra and not forwarded to vLLM."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            extra={"thinking_budget": 512, "some_other": "value"},
        )
        backend._to_vllm_params(params)

        # Check that VLLMSamplingParams was called without thinking_budget
        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "thinking_budget" not in call_kwargs
        assert call_kwargs["some_other"] == "value"

    def test_thinking_budget_adds_logits_processor(self):
        """thinking_budget creates a logits_processors entry in vLLM params."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            extra={"thinking_budget": 512},
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "logits_processors" in call_kwargs
        assert len(call_kwargs["logits_processors"]) == 1

    def test_thinking_budget_preserves_existing_processors(self):
        """thinking_budget appends to existing logits_processors list."""
        backend = self._create_mock_backend()
        existing_proc = lambda token_ids, logits: logits
        params = SamplingParams(
            max_tokens=100,
            extra={"thinking_budget": 512, "logits_processors": [existing_proc]},
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert len(call_kwargs["logits_processors"]) == 2
        assert call_kwargs["logits_processors"][0] is existing_proc
