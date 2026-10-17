"""Tests for HuggingFace Transformers backend."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
    TokenLogprob,
)

# Check if transformers/torch are available
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

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
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs

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
        """Whole strings use native stopping; their first tokens are not EOS."""
        backend, mock_tokenizer, _, _ = self._create_mock_backend()
        mock_tokenizer.encode.return_value = [42, 43]

        params = SamplingParams(stop_sequences=("STOP",))

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["eos_token_id"] == 1
        assert kwargs["stop_strings"] == ["STOP"]
        assert kwargs["tokenizer"] is mock_tokenizer

    def test_to_generate_kwargs_extra_params(self):
        """Test that extra params are passed through to generate kwargs."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
            repetition_penalty=1.2,
            extra={
                "num_beams": 4,
                "length_penalty": 0.8,
            },
        )

        kwargs = backend._to_generate_kwargs(params)

        assert kwargs["repetition_penalty"] == 1.2
        assert kwargs["num_beams"] == 4
        assert kwargs["length_penalty"] == 0.8

    @pytest.mark.parametrize(
        "extra",
        [
            {"do_sample": False},
            {"temperature": 0.9},
            {"top_k": 5},
            {"repetition_penalty": 1.2},
            {"max_new_tokens": 999},
            {"num_return_sequences": 2},
            {"input_ids": [[1, 2]]},
        ],
    )
    def test_to_generate_kwargs_rejects_owned_keys_in_extras(self, extra):
        """Typed controls and prompt inputs cannot be overridden through extras."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(temperature=0.7, extra=extra)

        with pytest.raises(ValueError):
            backend._to_generate_kwargs(params)

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [-0.5, 0.5])
    def test_unsupported_additive_penalties_are_rejected(self, field, value):
        """Multiplicative repetition penalties cannot approximate additive penalties."""
        backend, _, _, _ = self._create_mock_backend()
        params = SamplingParams(temperature=0.7, **{field: value})

        with pytest.raises(ValueError, match=field):
            backend._to_generate_kwargs(params)

    def test_multiple_choices_are_rejected_before_generation(self):
        backend, _, model, _ = self._create_mock_backend()

        with pytest.raises(ValueError, match="n"):
            backend.generate(["prompt"], SamplingParams(n=2))

        model.generate.assert_not_called()

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


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers/torch not installed")
class TestNativeGenerationDefaults:
    """Exercise native configuration merging with a tiny, offline CPU model."""

    @pytest.fixture
    def backend(self):
        from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel

        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=8, n_layer=1, n_head=1, n_embd=8, eos_token_id=7)
        ).eval()
        model.generation_config = GenerationConfig(
            max_new_tokens=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.81,
            top_k=4,
            min_p=0.12,
            repetition_penalty=1.3,
            pad_token_id=0,
            eos_token_id=7,
        )
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        tokenizer.pad_token_id, tokenizer.eos_token_id = 0, 7
        tokenizer.__len__.return_value = 8
        tokenizer.decode.return_value = "answer"
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._model_path = "offline-tiny-model"
        backend._model, backend._tokenizer = model, tokenizer
        backend._torch, backend._device = torch, "cpu"
        backend._chat_template_kwargs = {}
        backend._is_vlm = False
        yield backend
        backend.close()

    @pytest.mark.parametrize("inherit", [False, True])
    def test_native_defaults_are_inherited_only_when_requested(self, backend, inherit):
        original = deepcopy(backend.model.generation_config.to_dict())
        params = (
            SamplingParams(
                max_tokens=None,
                temperature=None,
                top_p=None,
                top_k=None,
                min_p=None,
                repetition_penalty=None,
                presence_penalty=None,
                frequency_penalty=None,
            )
            if inherit
            else SamplingParams(max_tokens=1, temperature=0.6, top_p=0.83)
        )
        with patch.object(
            backend.model,
            "_get_logits_processor",
            wraps=backend.model._get_logits_processor,
        ) as native_processors:
            result = backend.generate(["prompt"], params)[0]

        effective = native_processors.call_args.kwargs["generation_config"]
        expected = (
            {
                key: original[key]
                for key in (
                    "max_new_tokens",
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                )
            }
            if inherit
            else {
                "max_new_tokens": 1,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.83,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            }
        )
        assert {key: getattr(effective, key) for key in expected} == expected
        assert backend.model.generation_config.to_dict() == original
        assert result.metadata["request"]["top_k"] == expected["top_k"]
        assert result.metadata["request"]["repetition_penalty"] == expected["repetition_penalty"]

    def test_post_eos_batch_padding_is_not_sampled_output_or_scored(self, backend):
        from types import SimpleNamespace

        backend.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2], [1, 2]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        }
        outputs = SimpleNamespace(
            sequences=torch.tensor([[1, 2, 3, 7, 0], [1, 2, 4, 5, 7]]),
            scores=tuple(torch.zeros(2, 8) for _ in range(3)),
        )
        with patch.object(backend.model, "generate", return_value=outputs):
            results = backend.generate(
                ["first", "second"], SamplingParams(max_tokens=3, logprobs=True)
            )
        assert [result.completion_tokens for result in results] == [2, 3]
        assert results[0].metadata["raw_token_ids"] == [3, 7]
        assert [lp.token_id for lp in results[0].token_logprobs] == [3, 7]
        assert results[0].finish_reason == StopReason.END_OF_TEXT

    def test_multiple_greedy_continuations_keep_requested_decoding(self, backend):
        backend.tokenizer.side_effect = lambda prompts, **kwargs: {
            "input_ids": torch.tensor([[1, 2]] * len(prompts)),
            "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
        }
        with patch.object(backend.model, "generate", wraps=backend.model.generate) as generate:
            results = backend.continue_from_prefix(
                "prefix", SamplingParams(max_tokens=1, temperature=0), 2
            )
        assert len(results) == 2
        assert generate.call_args.kwargs["do_sample"] is False
        assert "temperature" not in generate.call_args.kwargs
        assert all(result.metadata["is_continuation"] for result in results)

    def test_token_phase_receipt_precedes_decoding_and_can_be_reparsed(self, backend):
        events, receipts = [], []

        class Observer:
            def started(self, index, attempt):
                events.append("started")

            def received(self, index, receipt):
                receipts.append(receipt)
                events.append("received")

            def completed(self, index, result):
                events.append("completed")

        def decode(*args, **kwargs):
            assert events[-1] == "received"
            return "answer"

        backend.tokenizer.decode.side_effect = decode
        with patch.object(backend.model, "generate", wraps=backend.model.generate) as generate:
            result = backend._generate_tokens(
                [1, 2],
                SamplingParams(),
                1,
                phase_name="answer",
                stop_id=None,
                schema=None,
                observer=Observer(),
            )
        assert events == ["started", "received", "completed"]
        backend.tokenizer.decode.side_effect = None
        assert backend.reparse_token_receipt(receipts[0]) == result
        assert generate.call_count == 1

    def test_phase_eos_at_thinking_budget_boundary_does_not_authorize_answer(self, backend):
        with patch.object(backend.model, "generate", return_value=torch.tensor([[1, 2, 7]])):
            phase = backend._generate_tokens(
                [1, 2], SamplingParams(), 1, phase_name="reasoning", stop_id=99, schema=None
            )
        assert phase.token_ids == (7,)
        assert phase.result.finish_reason == StopReason.END_OF_TEXT


def test_qwen_vlm_loader_uses_one_resolved_revision_without_legacy_vision_import():
    from types import SimpleNamespace

    from llenvs.inference.backends.huggingface import HuggingFaceBackend

    revision = "a" * 40
    config = SimpleNamespace(model_type="qwen3_5", _commit_hash=revision)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096), dtype="bfloat16")
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos", padding_side="right")
    native = SimpleNamespace(
        AutoModelForCausalLM=MagicMock(),
        AutoConfig=MagicMock(),
        AutoTokenizer=MagicMock(),
        AutoProcessor=MagicMock(),
        AutoModelForImageTextToText=MagicMock(),
    )
    native.AutoConfig.from_pretrained.return_value = config
    native.AutoTokenizer.from_pretrained.return_value = tokenizer
    native.AutoModelForImageTextToText.from_pretrained.return_value = model
    tensor_module = MagicMock()
    tensor_module.cuda.is_available.return_value = False
    tensor_module.backends.mps.is_available.return_value = False
    with patch.dict("sys.modules", {"transformers": native, "torch": tensor_module}):
        backend = HuggingFaceBackend("Qwen/model", revision="main", device="cpu")
        for loader in (
            native.AutoTokenizer,
            native.AutoProcessor,
            native.AutoModelForImageTextToText,
        ):
            assert loader.from_pretrained.call_args.kwargs["revision"] == revision
        native.AutoModelForCausalLM.from_pretrained.assert_not_called()
        backend.close()


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
            "_generate_rendered",
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
        """reasoning_effort=None preserves backend-level no-thinking config."""
        backend = self._create_mock_backend({"enable_thinking": False})

        with patch.object(
            backend,
            "_generate_rendered",
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

    def test_reasoning_effort_none_overrides_chat_template_kwargs(self):
        """Per-call reasoning_effort_none overrides backend-level thinking config."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "_generate_rendered",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat(
                [ChatMessage(role="user", content="hi")],
                SamplingParams(max_tokens=10, reasoning_effort="none"),
            )

        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is False

    def test_passed_to_generate_chat_batch(self):
        """chat_template_kwargs spread into apply_chat_template in generate_chat_batch."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "_generate_rendered",
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

    def test_reasoning_effort_none_overrides_batch_chat_template_kwargs(self):
        """Batched second elicitation also disables chat-template thinking."""
        backend = self._create_mock_backend({"enable_thinking": True})

        with patch.object(
            backend,
            "_generate_rendered",
            return_value=[
                GenerationResult(
                    text="ok",
                    finish_reason=StopReason.END_OF_TEXT,
                )
            ],
        ):
            backend.generate_chat_batch(
                [[ChatMessage(role="user", content="hi")]],
                SamplingParams(max_tokens=10, reasoning_effort="none"),
            )

        call_kwargs = backend._tokenizer.apply_chat_template.call_args
        assert call_kwargs[1].get("enable_thinking") is False

    def test_empty_kwargs_no_extra_args(self):
        """Empty chat_template_kwargs doesn't add extra args."""
        backend = self._create_mock_backend({})

        with patch.object(
            backend,
            "_generate_rendered",
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

    def test_thinking_budget_not_forwarded_to_generate(self):
        """thinking_budget field is not forwarded directly to generate()."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            extra={"some_other": "value"},
        )
        kwargs = backend._to_generate_kwargs(params)
        assert "thinking_budget" not in kwargs
        assert kwargs["some_other"] == "value"

    def test_thinking_budget_adds_logits_processor(self):
        """thinking_budget creates a logits processor in generate kwargs."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
        )
        kwargs = backend._to_generate_kwargs(params)
        assert "logits_processor" in kwargs
        assert len(kwargs["logits_processor"]) == 1

    def test_reasoning_effort_none_skips_thinking_budget_processor(self):
        """reasoning_effort_none suppresses local thinking-budget intervention."""
        backend = self._create_mock_backend()
        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            reasoning_effort="none",
        )
        kwargs = backend._to_generate_kwargs(params)
        assert "logits_processor" not in kwargs

    def test_thinking_budget_preserves_existing_processors(self):
        """thinking_budget appends to existing logits_processor list."""
        backend = self._create_mock_backend()

        def existing_proc(input_ids, scores):
            return scores

        params = SamplingParams(
            max_tokens=100,
            thinking_budget=512,
            extra={"logits_processor": [existing_proc]},
        )
        kwargs = backend._to_generate_kwargs(params)
        assert len(kwargs["logits_processor"]) == 2
        assert kwargs["logits_processor"][0] is existing_proc
