"""Tests for vLLM backend (fully mocked, no GPU required)."""

from unittest.mock import MagicMock, patch

import pytest

from llenvs.inference import PromptTooLongError
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
        backend._is_vlm = False
        backend._has_v1_thinking_processor = False
        return backend

    def test_stored_correctly(self):
        """chat_template_kwargs stored on the backend instance."""
        backend = self._create_mock_backend({"enable_thinking": True})
        assert backend._chat_template_kwargs == {"enable_thinking": True}

    def test_passed_to_generate_chat(self):
        """chat_template_kwargs spread into apply_chat_template in generate_chat."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "generate",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")],
                SamplingParams(max_tokens=10),
            )

        backend._tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is True

    def test_default_false_preserves_disabled_chat_template_kwargs(self):
        """disable_thinking=False preserves backend-level no-thinking config."""
        backend = self._create_mock_backend({"enable_thinking": False})

        with patch.object(
            backend,
            "generate",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")],
                SamplingParams(max_tokens=10),
            )

        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is False

    def test_disable_thinking_overrides_chat_template_kwargs(self):
        """Per-call disable_thinking overrides backend-level thinking config."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "generate",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")],
                SamplingParams(max_tokens=10, disable_thinking=True),
            )

        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is False

    def test_passed_to_generate_chat_batch(self):
        """chat_template_kwargs spread into apply_chat_template in generate_chat_batch."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "generate",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat_batch(
                [[ChatMessage(role="user", content="hi")]],
                SamplingParams(max_tokens=10),
            )

        backend._tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is True

    def test_disable_thinking_overrides_batch_chat_template_kwargs(self):
        """Batched second elicitation also disables chat-template thinking."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "generate",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat_batch(
                [[ChatMessage(role="user", content="hi")]],
                SamplingParams(max_tokens=10, disable_thinking=True),
            )

        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is False


class TestVLLMThinkingBudget:
    """Tests for thinking_budget interception in VLLMBackend."""

    def _create_mock_backend(self, *, is_v1=False, has_v1_thinking_processor=False):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "test-model"
        backend._tokenizer = MagicMock()
        backend._tokenizer.get_vocab.return_value = {"<think>": 100, "</think>": 101}
        backend._VLLMSamplingParams = MagicMock()
        backend._llm = MagicMock()
        backend._max_context_length = 4096
        backend._chat_template_kwargs = {}
        backend._is_vlm = False
        backend._is_v1 = is_v1
        backend._has_v1_thinking_processor = has_v1_thinking_processor
        return backend

    def test_thinking_budget_not_forwarded_to_vllm(self):
        """thinking_budget field is not forwarded to vLLM SamplingParams."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            extra={"some_other": "value"},
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "thinking_budget" not in call_kwargs
        assert call_kwargs["some_other"] == "value"

    def test_thinking_budget_adds_logits_processor(self):
        """thinking_budget creates a logits_processors entry in vLLM params."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "logits_processors" in call_kwargs
        assert len(call_kwargs["logits_processors"]) == 1

    def test_disable_thinking_skips_thinking_budget_processor(self):
        """disable_thinking suppresses local thinking-budget intervention."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            disable_thinking=True,
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "logits_processors" not in call_kwargs

    def test_thinking_budget_preserves_existing_processors(self):
        """thinking_budget appends to existing logits_processors list."""
        backend = self._create_mock_backend()

        def existing_proc(token_ids, logits):
            return logits

        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            extra={"logits_processors": [existing_proc]},
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert len(call_kwargs["logits_processors"]) == 2
        assert call_kwargs["logits_processors"][0] is existing_proc

    def test_thinking_budget_v1_uses_extra_args(self):
        """thinking_budget routes via extra_args on V1 with processor registered."""
        backend = self._create_mock_backend(is_v1=True, has_v1_thinking_processor=True)
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "logits_processors" not in call_kwargs
        assert call_kwargs["extra_args"]["thinking_budget"] == 512

    def test_thinking_budget_v1_without_processor_raises(self):
        """thinking_budget raises when V1 is active but processor not registered."""
        backend = self._create_mock_backend(is_v1=True, has_v1_thinking_processor=False)
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
        )
        with pytest.raises(ValueError, match="V1"):
            backend._to_vllm_params(params)

    def test_thinking_budget_v1_preserves_existing_extra_args(self):
        """thinking_budget merges into existing extra_args on V1."""
        backend = self._create_mock_backend(is_v1=True, has_v1_thinking_processor=True)
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=256,
            extra={"extra_args": {"some_key": "value"}},
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert call_kwargs["extra_args"]["thinking_budget"] == 256
        assert call_kwargs["extra_args"]["some_key"] == "value"

    def test_no_thinking_budget_works_on_v1(self):
        """Normal params work fine on V1 engine."""
        backend = self._create_mock_backend(is_v1=True)
        params = SamplingParams(max_tokens=100, extra={"some_key": "value"})
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert call_kwargs["some_key"] == "value"
        assert "logits_processors" not in call_kwargs


class TestVLLMV1Detection:
    """Tests for V1 engine detection."""

    def _create_mock_backend_with_engine_module(self, module_name):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)

        # Create a real class with the specified module for type() detection
        engine_class = type("MockEngine", (), {})
        engine_class.__module__ = module_name
        mock_engine = engine_class()

        backend._model_path = "test-model"
        backend._tokenizer = MagicMock()
        backend._VLLMSamplingParams = MagicMock()
        backend._llm = MagicMock()
        backend._llm.llm_engine = mock_engine
        backend._max_context_length = 4096
        backend._chat_template_kwargs = {}
        backend._is_vlm = False

        # Run detection logic (same as in __init__)
        engine_module = type(backend._llm.llm_engine).__module__ or ""
        backend._is_v1 = ".v1." in engine_module or engine_module.startswith("v1.")
        return backend

    def test_detects_v1_engine(self):
        """V1 engine is detected from module path."""
        backend = self._create_mock_backend_with_engine_module("vllm.v1.engine.llm_engine")
        assert backend._is_v1 is True

    def test_detects_v0_engine(self):
        """V0 engine is not flagged as V1."""
        backend = self._create_mock_backend_with_engine_module("vllm.engine.llm_engine")
        assert backend._is_v1 is False


class TestVLLMSoftBudgetRatio:
    """Tests for thinking_budget_soft_ratio forwarding in VLLMBackend."""

    def _create_mock_backend(self, *, is_v1=False, has_v1_thinking_processor=False):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "test-model"
        backend._tokenizer = MagicMock()
        backend._tokenizer.get_vocab.return_value = {"<think>": 100, "</think>": 101}
        backend._VLLMSamplingParams = MagicMock()
        backend._llm = MagicMock()
        backend._max_context_length = 4096
        backend._chat_template_kwargs = {}
        backend._is_vlm = False
        backend._is_v1 = is_v1
        backend._has_v1_thinking_processor = has_v1_thinking_processor
        return backend

    def test_soft_ratio_not_forwarded_to_vllm(self):
        """thinking_budget_soft_ratio is not forwarded directly to vLLM."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            thinking_budget_soft_ratio=0.9,
            extra={"some_other": "value"},
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "thinking_budget_soft_ratio" not in call_kwargs
        assert call_kwargs["some_other"] == "value"

    def test_soft_ratio_forwarded_to_processor_v0(self):
        """soft_ratio is passed to ThinkingBudgetProcessor on V0."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            thinking_budget_soft_ratio=0.9,
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        # The processor should be in the logits_processors list
        assert "logits_processors" in call_kwargs
        assert len(call_kwargs["logits_processors"]) == 1

    def test_soft_ratio_forwarded_v1_extra_args(self):
        """soft_ratio forwarded via extra_args on V1."""
        backend = self._create_mock_backend(is_v1=True, has_v1_thinking_processor=True)
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            thinking_budget_soft_ratio=0.9,
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert call_kwargs["extra_args"]["thinking_budget"] == 512
        assert call_kwargs["extra_args"]["thinking_budget_soft_ratio"] == 0.9

    def test_soft_ratio_alone_without_budget_ignored(self):
        """soft_ratio without thinking_budget does not create a processor."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget_soft_ratio=0.9,
        )
        backend._to_vllm_params(params)

        call_kwargs = backend._VLLMSamplingParams.call_args[1]
        assert "logits_processors" not in call_kwargs


class TestVLLMPromptTooLong:
    def _create_mock_backend(self):
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "test-model"
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=False: [0] * (len(text) + (10 if add_special_tokens else 0))
        )
        backend._VLLMSamplingParams = MagicMock()
        backend._llm = MagicMock()
        backend._max_context_length = 50
        backend._chat_template_kwargs = {}
        backend._is_vlm = False
        backend._has_v1_thinking_processor = False
        return backend

    def test_prompt_too_long_uses_rendered_prompt_lengths_without_special_tokens(self):
        backend = self._create_mock_backend()
        backend._llm.generate.side_effect = ValueError(
            "The decoder prompt (length 60) is longer than the maximum model length of 50."
        )

        with pytest.raises(PromptTooLongError) as exc_info:
            backend.generate(["abcd", "x" * 60], SamplingParams(max_tokens=4))

        err = exc_info.value
        assert err.prompt_token_lengths == [4, 60]
        assert err.offending_indices == [1]
        assert err.offending_prompts == ["x" * 60]


class TestInferenceExports:
    def test_prompt_too_long_error_is_exported(self):
        from llenvs import inference

        assert "PromptTooLongError" in inference.__all__
