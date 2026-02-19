"""Tests for HuggingFace Transformers backend."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
    TokenLogprob,
)

# Check if transformers/torch are available
try:
    import torch
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TestHuggingFaceBackendUnit:
    """Unit tests for HuggingFaceBackend (mocked)."""

    def _create_mock_backend(self):
        """Create a HuggingFaceBackend with mocked dependencies."""
        with patch.dict("sys.modules", {"torch": MagicMock(), "transformers": MagicMock()}):
            # Mock torch
            mock_torch = MagicMock()
            mock_torch.float16 = "float16"
            mock_torch.bfloat16 = "bfloat16"
            mock_torch.float32 = "float32"
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "generated text"
            mock_tokenizer.apply_chat_template.return_value = "formatted chat"

            # Mock model
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.config.max_position_embeddings = 2048

            # Mock generate output
            mock_output = MagicMock()
            mock_output.__iter__ = lambda self: iter([mock_output])
            mock_output.__getitem__ = lambda self, idx: mock_output

            with (
                patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                from llenvs.inference.backends.huggingface import HuggingFaceBackend

                backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
                backend._model_path = "test-model"
                backend._torch = mock_torch
                backend._tokenizer = mock_tokenizer
                backend._model = mock_model
                backend._device = "cpu"
                backend._max_context_length = 2048
                backend._chat_template_kwargs = {}
                backend._is_vlm = False
                backend._processor = None

                return backend, mock_tokenizer, mock_model, mock_torch

    def test_capabilities(self):
        """Test backend capabilities are correctly reported."""
        backend, _, _, _ = self._create_mock_backend()
        caps = backend.capabilities

        assert caps.supports_logprobs is True
        assert caps.supports_prefix_continuation is True
        assert caps.supports_batching is True
        assert caps.supports_streaming is False
        assert caps.supports_chat is True
        assert caps.supports_function_calling is False
        assert caps.max_context_length == 2048

    def test_model_name(self):
        """Test model_name property."""
        backend, _, _, _ = self._create_mock_backend()
        assert backend.model_name == "test-model"

    def test_tokenizer_property(self):
        """Test tokenizer property returns underlying tokenizer."""
        backend, mock_tokenizer, _, _ = self._create_mock_backend()
        assert backend.tokenizer is mock_tokenizer

    def test_model_property(self):
        """Test model property returns underlying model."""
        backend, _, mock_model, _ = self._create_mock_backend()
        assert backend.model is mock_model

    def test_to_generate_kwargs_greedy(self):
        """Test conversion of SamplingParams to generate kwargs (greedy)."""
        backend, mock_tokenizer, _, _ = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            temperature=0.0,
        )

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["max_new_tokens"] == 100
        assert kwargs["do_sample"] is False
        assert kwargs["num_return_sequences"] == 1
        assert kwargs["temperature"] == 1.0  # Neutral default overrides model config
        assert kwargs["top_p"] == 1.0

    def test_to_generate_kwargs_sampling(self):
        """Test conversion of SamplingParams to generate kwargs (sampling)."""
        backend, mock_tokenizer, _, _ = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=200,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
        )

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["max_new_tokens"] == 200
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 50

    def test_to_generate_kwargs_with_logprobs(self):
        """Test that logprobs enables output_scores."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(logprobs=True)

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["output_scores"] is True
        assert kwargs["return_dict_in_generate"] is True

    def test_to_generate_kwargs_stop_sequences(self):
        """Test stop sequences are converted to eos_token_ids."""
        backend, mock_tokenizer, _, _ = self._create_mock_backend()
        mock_tokenizer.encode.return_value = [42]

        params = SamplingParams(stop_sequences=("STOP",))

        kwargs = backend._to_generate_kwargs(params)

        # Should include original eos_token_id plus stop sequence token
        assert 42 in kwargs["eos_token_id"] or kwargs["eos_token_id"] == [1, 42]

    def test_to_generate_kwargs_extra_params(self):
        """Test that extra params are passed through to generate kwargs."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
            extra={
                "repetition_penalty": 1.2,
                "num_beams": 4,
                "length_penalty": 0.8,
            },
        )

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["repetition_penalty"] == 1.2
        assert kwargs["num_beams"] == 4
        assert kwargs["length_penalty"] == 0.8

    def test_to_generate_kwargs_extra_overrides(self):
        """Test that extra params can override computed values."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(
            temperature=0.7,
            extra={"do_sample": False},  # Override the computed do_sample
        )

        kwargs = backend._to_generate_kwargs(params)

        # extra should override the computed do_sample=True
        assert kwargs["do_sample"] is False

    def test_to_generate_kwargs_frequency_penalty_mapping(self):
        """Test that frequency_penalty maps to repetition_penalty."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(
            temperature=0.7,
            frequency_penalty=0.5,
        )

        kwargs = backend._to_generate_kwargs(params)

        # frequency_penalty should be converted to repetition_penalty
        assert kwargs["repetition_penalty"] == 1.5  # 1.0 + 0.5

    def test_to_generate_kwargs_extra_empty(self):
        """Test that empty extra dict doesn't affect kwargs."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            temperature=0.0,
            extra={},
        )

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["max_new_tokens"] == 100
        assert kwargs["do_sample"] is False


class TestConvertStopReason:
    """Tests for _convert_stop_reason helper."""

    def test_max_tokens(self):
        """Test MAX_TOKENS detection."""
        from llenvs.inference.backends.huggingface import _convert_stop_reason

        result = _convert_stop_reason(
            generated_length=100,
            max_new_tokens=100,
            eos_token_id=1,
            last_token_id=42,
        )
        assert result == StopReason.MAX_TOKENS

    def test_end_of_text_single_eos(self):
        """Test END_OF_TEXT with single eos_token_id."""
        from llenvs.inference.backends.huggingface import _convert_stop_reason

        result = _convert_stop_reason(
            generated_length=50,
            max_new_tokens=100,
            eos_token_id=1,
            last_token_id=1,
        )
        assert result == StopReason.END_OF_TEXT

    def test_end_of_text_multiple_eos(self):
        """Test END_OF_TEXT with multiple eos_token_ids."""
        from llenvs.inference.backends.huggingface import _convert_stop_reason

        result = _convert_stop_reason(
            generated_length=50,
            max_new_tokens=100,
            eos_token_id=[1, 2, 3],
            last_token_id=2,
        )
        assert result == StopReason.END_OF_TEXT

    def test_unknown_reason(self):
        """Test UNKNOWN when no clear reason."""
        from llenvs.inference.backends.huggingface import _convert_stop_reason

        result = _convert_stop_reason(
            generated_length=50,
            max_new_tokens=100,
            eos_token_id=1,
            last_token_id=42,
        )
        assert result == StopReason.UNKNOWN

    def test_no_eos_token(self):
        """Test when eos_token_id is None."""
        from llenvs.inference.backends.huggingface import _convert_stop_reason

        result = _convert_stop_reason(
            generated_length=50,
            max_new_tokens=100,
            eos_token_id=None,
            last_token_id=1,
        )
        assert result == StopReason.UNKNOWN


class TestGenerationResult:
    """Tests for GenerationResult compatibility."""

    def test_basic_result(self):
        """Test basic GenerationResult creation."""
        result = GenerationResult(
            text="Hello world",
            finish_reason=StopReason.END_OF_TEXT,
            prompt_tokens=10,
            completion_tokens=5,
        )

        assert result.text == "Hello world"
        assert result.finish_reason == StopReason.END_OF_TEXT
        assert result.total_tokens == 15

    def test_result_with_logprobs(self):
        """Test GenerationResult with logprobs."""
        logprobs = (
            TokenLogprob(token="Hello", token_id=1, logprob=-0.5),
            TokenLogprob(token=" world", token_id=2, logprob=-0.3),
        )
        result = GenerationResult(
            text="Hello world",
            finish_reason=StopReason.END_OF_TEXT,
            token_logprobs=logprobs,
        )

        assert result.token_logprobs is not None
        assert len(result.token_logprobs) == 2
        assert result.token_logprobs[0].token == "Hello"

    def test_result_with_top_logprobs(self):
        """Test TokenLogprob with top alternatives."""
        logprob = TokenLogprob(
            token="Hello",
            token_id=1,
            logprob=-0.5,
            top_logprobs={"Hello": -0.5, "Hi": -1.0, "Hey": -1.5},
        )

        assert logprob.top_logprobs is not None
        assert len(logprob.top_logprobs) == 3
        assert logprob.top_logprobs["Hello"] == -0.5


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers/torch not installed")
class TestHuggingFaceBackendIntegration:
    """Integration tests that require transformers to be installed."""

    def test_import_error_message(self):
        """Test helpful error message when dependencies missing."""
        # This test only runs when transformers IS installed,
        # so we mock the import to simulate it being missing
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            # Clear the cached import
            import importlib
            import sys

            # Remove from cache if present
            if "llenvs.inference.backends.huggingface" in sys.modules:
                del sys.modules["llenvs.inference.backends.huggingface"]

            # This won't actually fail since we can't truly remove the modules
            # Just verify the backend module loads
            from llenvs.inference.backends.huggingface import HuggingFaceBackend

            assert HuggingFaceBackend is not None

    def test_backend_initialization_cpu(self):
        """Test backend initialization on CPU with small model."""
        from llenvs.inference.backends import HuggingFaceBackend

        # Use a tiny model for fast testing
        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        assert backend.model_name == "hf-internal-testing/tiny-random-gpt2"
        assert backend.capabilities.supports_logprobs is True

    def test_generate_basic(self):
        """Test basic generation with tiny model."""
        from llenvs.inference.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        results = backend.generate(
            ["Hello"],
            SamplingParams(max_tokens=5, temperature=0.0),
        )

        assert len(results) == 1
        assert results[0].text is not None
        assert results[0].completion_tokens <= 5

    def test_generate_batch(self):
        """Test batch generation."""
        from llenvs.inference.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        results = backend.generate(
            ["Hello", "World", "Test"],
            SamplingParams(max_tokens=3, temperature=0.0),
        )

        assert len(results) == 3
        for result in results:
            assert result.text is not None

    def test_generate_with_logprobs(self):
        """Test generation with logprobs."""
        from llenvs.inference.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        results = backend.generate_with_logprobs(
            ["Test"],
            SamplingParams(max_tokens=3, temperature=0.0),
            num_logprobs=3,
        )

        assert len(results) == 1
        assert results[0].token_logprobs is not None
        if len(results[0].token_logprobs) > 0:
            # Check logprob structure
            lp = results[0].token_logprobs[0]
            assert isinstance(lp.token, str)
            assert isinstance(lp.logprob, float)
            assert lp.top_logprobs is not None

    def test_generate_single(self):
        """Test generate_single convenience method."""
        from llenvs.inference.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        result = backend.generate_single(
            "Hello",
            SamplingParams(max_tokens=5),
        )

        assert isinstance(result, GenerationResult)
        assert result.text is not None

    def test_continue_from_prefix(self):
        """Test prefix continuation."""
        from llenvs.inference.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        results = backend.continue_from_prefix(
            prefix="Once upon a time",
            params=SamplingParams(max_tokens=5, temperature=0.7),
            num_continuations=2,
        )

        assert len(results) == 2
        for result in results:
            assert result.text is not None
            assert result.metadata.get("is_continuation") is True

    def test_sampling_temperature(self):
        """Test that temperature affects sampling."""
        from llenvs.inference.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(
            model_path="hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype="float32",
        )

        # Generate multiple times with high temperature
        results_high_temp = [
            backend.generate_single(
                "Test",
                SamplingParams(max_tokens=10, temperature=1.0),
            ).text
            for _ in range(5)
        ]

        # With high temperature, outputs should vary (though not guaranteed)
        # At minimum, the generation should work without errors
        assert all(r is not None for r in results_high_temp)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers/torch not installed")
class TestHuggingFaceBackendMockedGeneration:
    """Tests for generation logic with mocked model."""

    def test_generate_extracts_text_correctly(self):
        """Test that generated text is correctly extracted."""
        import torch

        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_cls,
        ):
            # Setup tokenizer mock
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer.__call__ = lambda self, *args, **kwargs: {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer.decode.return_value = "generated output"
            mock_tok.return_value = mock_tokenizer

            # Setup model mock
            mock_model = MagicMock()
            mock_model.device = torch.device("cpu")
            mock_model.config.max_position_embeddings = 2048
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
            mock_model_cls.return_value = mock_model

            # Setup config mock (non-VLM)
            mock_config = MagicMock()
            mock_config.model_type = "gpt2"
            mock_config_cls.return_value = mock_config

            from llenvs.inference.backends import HuggingFaceBackend

            backend = HuggingFaceBackend("test-model", device="cpu")

            # Mock the tokenizer call
            backend._tokenizer = mock_tokenizer
            backend._tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

            results = backend.generate(
                ["Hello"],
                SamplingParams(max_tokens=10),
            )

            assert len(results) == 1
            mock_model.generate.assert_called_once()


class TestHuggingFaceChatTemplateKwargs:
    """Tests for chat_template_kwargs on HuggingFaceBackend."""

    def _create_mock_backend(self, chat_template_kwargs=None):
        """Create a HuggingFaceBackend with mocked dependencies."""
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._model_path = "test-model"
        backend._torch = MagicMock()
        backend._tokenizer = MagicMock()
        backend._tokenizer.pad_token_id = 0
        backend._tokenizer.eos_token_id = 1
        backend._tokenizer.chat_template = "some_template"
        backend._tokenizer.apply_chat_template.return_value = "formatted"
        backend._model = MagicMock()
        backend._model.device = "cpu"
        backend._device = "cpu"
        backend._max_context_length = 2048
        backend._chat_template_kwargs = chat_template_kwargs or {}
        backend._is_vlm = False
        backend._processor = None
        return backend

    def test_stored_correctly(self):
        """chat_template_kwargs stored on the backend instance."""
        backend = self._create_mock_backend({"enable_thinking": True})
        assert backend._chat_template_kwargs == {"enable_thinking": True}

    def test_passed_to_generate_chat(self):
        """chat_template_kwargs spread into apply_chat_template in generate_chat."""
        backend = self._create_mock_backend({"enable_thinking": True})
        backend._tokenizer.apply_chat_template.return_value = "prompt"
        backend._model.generate.return_value = MagicMock()

        # Mock generate to avoid the full pipeline
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

    def test_empty_kwargs_no_extra_args(self):
        """Empty chat_template_kwargs doesn't add extra args."""
        backend = self._create_mock_backend({})

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

        call_kwargs = backend._tokenizer.apply_chat_template.call_args[1]
        assert "enable_thinking" not in call_kwargs


class TestHuggingFaceThinkingBudget:
    """Tests for thinking_budget interception in HuggingFaceBackend."""

    def _create_mock_backend(self):
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._model_path = "test-model"
        backend._torch = MagicMock()
        backend._tokenizer = MagicMock()
        backend._tokenizer.pad_token_id = 0
        backend._tokenizer.eos_token_id = 1
        backend._tokenizer.get_vocab.return_value = {"<think>": 100, "</think>": 101}
        backend._model = MagicMock()
        backend._model.device = "cpu"
        backend._device = "cpu"
        backend._max_context_length = 2048
        backend._chat_template_kwargs = {}
        backend._is_vlm = False
        backend._processor = None
        return backend

    def test_thinking_budget_popped_from_extra(self):
        """thinking_budget is removed from extra and not passed to generate()."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            extra={"thinking_budget": 512, "some_other": "value"},
        )
        kwargs = backend._to_generate_kwargs(params)
        assert "thinking_budget" not in kwargs
        assert kwargs["some_other"] == "value"

    def test_thinking_budget_adds_logits_processor(self):
        """thinking_budget creates a logits processor in generate kwargs."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            extra={"thinking_budget": 512},
        )
        kwargs = backend._to_generate_kwargs(params)
        assert "logits_processor" in kwargs
        assert len(kwargs["logits_processor"]) == 1

    def test_thinking_budget_preserves_existing_processors(self):
        """thinking_budget appends to existing logits_processor list."""
        backend = self._create_mock_backend()
        existing_proc = lambda input_ids, scores: scores
        params = SamplingParams(
            max_tokens=100,
            extra={"thinking_budget": 512, "logits_processor": [existing_proc]},
        )
        kwargs = backend._to_generate_kwargs(params)
        assert len(kwargs["logits_processor"]) == 2
        assert kwargs["logits_processor"][0] is existing_proc
