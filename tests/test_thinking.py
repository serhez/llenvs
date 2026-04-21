"""Tests for ThinkingBudgetProcessor."""

from unittest.mock import MagicMock

import pytest

from llenvs.inference.thinking import DEFAULT_EARLY_STOPPING_SUFFIX, ThinkingBudgetProcessor


def _make_tokenizer(vocab=None):
    """Create a mock tokenizer with the given vocab mapping."""
    tok = MagicMock()
    vocab = vocab or {"<think>": 100, "</think>": 101}
    tok.get_vocab.return_value = vocab
    return tok


class TestTokenResolution:
    """Tests for resolving <think>/<think> token IDs."""

    def test_resolve_from_vocab(self):
        """Token IDs resolved via get_vocab()."""
        tok = _make_tokenizer({"<think>": 100, "</think>": 101, "hello": 200})
        proc = ThinkingBudgetProcessor(tok, budget=64)
        assert proc._think_id == 100
        assert proc._end_think_id == 101

    def test_resolve_via_convert_tokens_to_ids(self):
        """Falls back to convert_tokens_to_ids when tokens not in vocab."""
        tok = MagicMock()
        tok.get_vocab.return_value = {"hello": 200}

        def convert(token):
            mapping = {"<think>": 100, "</think>": 101}
            return mapping.get(token)

        tok.convert_tokens_to_ids = convert
        proc = ThinkingBudgetProcessor(tok, budget=64)
        assert proc._think_id == 100
        assert proc._end_think_id == 101

    def test_resolve_via_encode_fallback(self):
        """Falls back to encode() when convert_tokens_to_ids returns None."""
        tok = MagicMock()
        tok.get_vocab.return_value = {}
        tok.convert_tokens_to_ids.return_value = None

        def _encode(s, add_special_tokens=True):
            special = {"<think>": [100], "</think>": [101]}
            if s in special:
                return special[s]
            # Default: return multi-token encoding for any other string
            return [200, 201, 202]

        tok.encode.side_effect = _encode
        proc = ThinkingBudgetProcessor(tok, budget=64)
        assert proc._think_id == 100
        assert proc._end_think_id == 101

    def test_multi_token_encode_raises(self):
        """Raises ValueError if encode() returns multiple tokens."""
        tok = MagicMock()
        tok.get_vocab.return_value = {}
        tok.convert_tokens_to_ids.return_value = None
        tok.encode.side_effect = lambda s, add_special_tokens=True: [100, 200]
        with pytest.raises(ValueError, match="single token"):
            ThinkingBudgetProcessor(tok, budget=64)

    def test_missing_token_raises(self):
        """Raises ValueError when <think> token cannot be resolved."""
        tok = MagicMock()
        tok.get_vocab.return_value = {}
        tok.convert_tokens_to_ids.return_value = None
        tok.encode.return_value = []
        with pytest.raises(ValueError, match="<think>"):
            ThinkingBudgetProcessor(tok, budget=64)


class TestCountThinking:
    """Tests for _count_thinking state derivation."""

    def test_not_in_thinking(self):
        """Returns (False, 0) when no <think> token present."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64)
        in_thinking, count = proc._count_thinking([1, 2, 3])
        assert in_thinking is False
        assert count == 0

    def test_in_thinking_block(self):
        """Returns (True, count) inside a thinking block."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64)
        # <think>=100, then 3 tokens of thinking
        in_thinking, count = proc._count_thinking([100, 5, 6, 7])
        assert in_thinking is True
        assert count == 3

    def test_after_closed_block(self):
        """Returns (False, count) after a closed thinking block (shared mode)."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64)
        # <think>=100, tokens, </think>=101
        in_thinking, count = proc._count_thinking([100, 5, 6, 101])
        assert in_thinking is False
        assert count == 2  # shared mode preserves cumulative count

    def test_second_thinking_block_shared(self):
        """Counter accumulates across blocks in shared mode (default)."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64)
        # First block: <think> 5 6 </think>, second block: <think> 7
        in_thinking, count = proc._count_thinking([100, 5, 6, 101, 100, 7])
        assert in_thinking is True
        assert count == 3  # 2 from first block + 1 from second

    def test_second_thinking_block_per_block(self):
        """Counter resets for a second thinking block in per-block mode."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64, per_block=True)
        # First block: <think> 5 6 </think>, second block: <think> 7
        in_thinking, count = proc._count_thinking([100, 5, 6, 101, 100, 7])
        assert in_thinking is True
        assert count == 1


class TestBudgetEnforcement:
    """Tests for _apply_budget logit masking."""

    def _make_logits(self, vocab_size=200):
        """Create a simple list-based logits tensor mock."""

        return [1.0] * vocab_size

    def test_passthrough_when_not_in_thinking(self):
        """Logits unchanged when not in thinking block."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=10)
        logits = self._make_logits()
        result = proc._apply_budget(False, 0, logits)
        assert result is logits  # same object, unmodified

    def test_passthrough_when_under_budget(self):
        """Logits unchanged when in thinking but under budget."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=10)
        logits = self._make_logits()
        result = proc._apply_budget(True, 5, logits)
        assert result is logits

    def test_masking_when_budget_exceeded(self):
        """All logits masked except </think> when budget exceeded."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=10)
        logits = [1.0] * 200
        result = proc._apply_budget(True, 10, logits)
        # </think> at index 101 should be 0.0
        assert result[101] == 0.0
        # All other tokens should be -inf
        for i, v in enumerate(result):
            if i != 101:
                assert v == float("-inf"), f"logits[{i}] = {v}, expected -inf"

    def test_budget_zero_forces_end_think(self):
        """budget=0 immediately forces </think> when thinking starts."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=0)
        logits = [1.0] * 200
        result = proc._apply_budget(True, 0, logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")


class TestVLLMProcessor:
    """Tests for the vLLM processor wrapper."""

    def test_signature_and_passthrough(self):
        """vllm_processor takes (list[int], logits) and returns logits."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=10)
        logits = [1.0] * 200
        # No <think> token in sequence — passthrough
        result = proc.vllm_processor([1, 2, 3], logits)
        assert result is logits

    def test_budget_enforcement(self):
        """vllm_processor enforces budget when in thinking block."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=2)
        # <think>=100, then 2 tokens (= budget), should force </think>
        logits = [1.0] * 200
        result = proc.vllm_processor([100, 5, 6], logits)
        assert result[101] == 0.0  # </think> allowed
        assert result[0] == float("-inf")  # other tokens masked


class TestHFProcessor:
    """Tests for the HuggingFace processor wrapper."""

    def test_batch_independence(self):
        """Each batch element is processed independently."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=2)

        # Batch of 2:
        # - Sequence 0: in thinking, over budget → should mask
        # - Sequence 1: not in thinking → passthrough
        input_ids = [
            [100, 5, 6],  # <think> + 2 tokens (at budget)
            [1, 2, 3],  # no thinking
        ]
        scores = [
            [1.0] * 200,
            [1.0] * 200,
        ]

        result = proc.hf_processor(input_ids, scores)

        # Sequence 0: masked
        assert result[0][101] == 0.0
        assert result[0][0] == float("-inf")

        # Sequence 1: untouched
        assert result[1][0] == 1.0
        assert result[1][101] == 1.0

    def test_passthrough_under_budget(self):
        """HF processor passes through when under budget."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=10)

        input_ids = [[100, 5]]  # 1 thinking token, budget=10
        scores = [[1.0] * 200]

        result = proc.hf_processor(input_ids, scores)
        assert result[0][0] == 1.0  # unchanged


class TestFromTokenIds:
    """Tests for ThinkingBudgetProcessor.from_token_ids()."""

    def test_creation(self):
        """Creates processor with pre-resolved token IDs."""
        proc = ThinkingBudgetProcessor.from_token_ids(think_id=100, end_think_id=101, budget=64)
        assert proc._think_id == 100
        assert proc._end_think_id == 101
        assert proc._budget == 64

    def test_enforcement(self):
        """Budget enforcement works with from_token_ids-created processor."""
        proc = ThinkingBudgetProcessor.from_token_ids(think_id=100, end_think_id=101, budget=2)
        # <think>=100, then 2 tokens (= budget), should force </think>
        logits = [1.0] * 200
        result = proc.vllm_processor([100, 5, 6], logits)
        assert result[101] == 0.0  # </think> allowed
        assert result[0] == float("-inf")  # other tokens masked

    def test_zero_budget(self):
        """budget=0 immediately forces </think> when thinking starts."""
        proc = ThinkingBudgetProcessor.from_token_ids(think_id=100, end_think_id=101, budget=0)
        logits = [1.0] * 200
        result = proc.vllm_processor([100], logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")


class TestV1ProcessorClass:
    """Tests for make_v1_thinking_processor_class()."""

    def _make_v1_class(self):
        """Create V1 processor class with a mocked AdapterLogitsProcessor base."""
        import sys
        from unittest.mock import MagicMock

        # Create a mock AdapterLogitsProcessor base class
        class MockAdapterLogitsProcessor:
            pass

        # Mock the vllm module
        mock_module = MagicMock()
        mock_module.AdapterLogitsProcessor = MockAdapterLogitsProcessor
        sys.modules["vllm.v1.sample.logits_processor"] = mock_module

        from llenvs.inference.thinking import _build_v1_thinking_processor_class

        cls = _build_v1_thinking_processor_class()

        # Clean up
        del sys.modules["vllm.v1.sample.logits_processor"]

        return cls, MockAdapterLogitsProcessor

    def test_returns_class(self):
        """Factory returns a class subclassing AdapterLogitsProcessor."""
        cls, base = self._make_v1_class()
        assert cls is not None
        assert issubclass(cls, base)

    def test_returns_none_without_vllm(self):
        """Returns None when vllm V1 API is not importable."""
        from llenvs.inference.thinking import make_v1_thinking_processor_class

        cls = make_v1_thinking_processor_class()
        assert cls is None

    def test_class_has_pickle_compatible_qualname(self):
        """V1ThinkingBudgetProcessor carries a qualname pickle can resolve.

        vLLM's spawn-based multiprocessing pickles ``logits_processors`` to
        send them to the EngineCore subprocess. Pickle serializes classes by
        ``(__module__, __qualname__)``, so a closure-local qualname like
        ``make_v1_thinking_processor_class.<locals>.V1ThinkingBudgetProcessor``
        would fail. The build helper must rewrite these to match the
        module-level attribute.
        """
        cls, _ = self._make_v1_class()
        assert cls.__qualname__ == "V1ThinkingBudgetProcessor"
        assert cls.__module__ == "llenvs.inference.thinking"

    def test_factory_returns_cached_module_attribute(self):
        """Public factory returns the module-level cached class.

        Pickle's ``save_global`` verifies that the class being pickled is the
        same object as ``getattr(module, qualname)``. If the factory returned
        a fresh class per call, that identity check would fail with
        ``"not the same object as ..."`` when vLLM pickles logits processors
        across a spawn subprocess.
        """
        from llenvs.inference import thinking

        cls1 = thinking.make_v1_thinking_processor_class()
        cls2 = thinking.make_v1_thinking_processor_class()
        assert cls1 is cls2
        assert cls1 is thinking.V1ThinkingBudgetProcessor

    def test_validate_params_accepts_valid(self):
        """validate_params accepts valid thinking_budget int."""
        cls, _ = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {"thinking_budget": 512}
        # Should not raise
        cls.validate_params(params)

    def test_validate_params_accepts_absent(self):
        """validate_params accepts missing thinking_budget."""
        cls, _ = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {}
        cls.validate_params(params)

    def test_validate_params_rejects_invalid_type(self):
        """validate_params rejects non-int thinking_budget."""
        cls, _ = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {"thinking_budget": "not_an_int"}
        with pytest.raises(ValueError, match="must be an integer"):
            cls.validate_params(params)

    def test_is_argmax_invariant(self):
        """is_argmax_invariant returns False."""
        cls, _ = self._make_v1_class()
        instance = cls.__new__(cls)
        assert instance.is_argmax_invariant() is False

    def test_new_req_returns_none_without_budget(self):
        """new_req_logits_processor returns None when no budget in extra_args."""
        cls, _ = self._make_v1_class()
        instance = cls.__new__(cls)
        instance._available = True
        instance._think_id = 100
        instance._end_think_id = 101

        params = MagicMock()
        params.extra_args = {}
        assert instance.new_req_logits_processor(params) is None

    def test_new_req_returns_callable_with_budget(self):
        """new_req_logits_processor returns a callable when budget is set."""
        cls, _ = self._make_v1_class()
        instance = cls.__new__(cls)
        instance._available = True
        instance._think_id = 100
        instance._end_think_id = 101
        instance._tokenizer = _make_tokenizer()
        instance._tokenizer.encode.return_value = [50, 51, 101]

        params = MagicMock()
        params.extra_args = {"thinking_budget": 512}
        result = instance.new_req_logits_processor(params)
        assert callable(result)

    def test_new_req_raises_when_unavailable(self):
        """new_req_logits_processor raises when tokens couldn't be resolved."""
        cls, _ = self._make_v1_class()
        instance = cls.__new__(cls)
        instance._available = False

        params = MagicMock()
        params.extra_args = {"thinking_budget": 512}
        with pytest.raises(ValueError, match="think.*token"):
            instance.new_req_logits_processor(params)

    def test_new_req_forwards_soft_ratio(self):
        """new_req_logits_processor creates processor with soft_budget_ratio."""
        cls, _ = self._make_v1_class()
        instance = cls.__new__(cls)
        instance._available = True
        instance._think_id = 100
        instance._end_think_id = 101
        instance._tokenizer = _make_tokenizer()
        instance._tokenizer.encode.return_value = [50, 51, 101]

        params = MagicMock()
        params.extra_args = {
            "thinking_budget": 512,
            "thinking_budget_soft_ratio": 0.9,
        }
        result = instance.new_req_logits_processor(params)
        assert callable(result)

    def test_new_req_ignores_soft_ratio_without_budget(self):
        """soft_ratio alone without budget still returns None."""
        cls, _ = self._make_v1_class()
        instance = cls.__new__(cls)
        instance._available = True
        instance._think_id = 100
        instance._end_think_id = 101

        params = MagicMock()
        params.extra_args = {"thinking_budget_soft_ratio": 0.9}
        assert instance.new_req_logits_processor(params) is None

    def test_validate_params_accepts_soft_ratio(self):
        """validate_params accepts valid thinking_budget_soft_ratio float."""
        cls, _ = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {"thinking_budget": 512, "thinking_budget_soft_ratio": 0.9}
        cls.validate_params(params)

    def test_validate_params_rejects_invalid_soft_ratio_type(self):
        """validate_params rejects non-float thinking_budget_soft_ratio."""
        cls, _ = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {"thinking_budget_soft_ratio": "not_a_float"}
        with pytest.raises(ValueError, match="must be a float"):
            cls.validate_params(params)

    def test_validate_params_rejects_out_of_range_soft_ratio(self):
        """validate_params rejects soft_ratio outside (0, 1)."""
        cls, _ = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {"thinking_budget_soft_ratio": 1.5}
        with pytest.raises(ValueError, match="between 0 and 1"):
            cls.validate_params(params)


class TestVectorizedApplyBudget:
    """Tests for vectorized _apply_budget with tensor-like objects."""

    def test_tensor_masking_uses_fill(self):
        """When logits has fill_(), uses vectorized path."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=5)

        class FakeTensor:
            def __init__(self, size):
                self._data = [1.0] * size

            def fill_(self, val):
                for i in range(len(self._data)):
                    self._data[i] = val
                return self

            def __setitem__(self, idx, val):
                self._data[idx] = val

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)

        logits = FakeTensor(200)
        result = proc._apply_budget(True, 5, logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")
        assert result[50] == float("-inf")

    def test_list_masking_still_works(self):
        """Plain list logits still work (fallback path)."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=5)
        logits = [1.0] * 200
        result = proc._apply_budget(True, 5, logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")


class TestSoftBudgetTransition:
    """Tests for soft_budget_ratio feature."""

    def test_default_no_soft_budget(self):
        """By default, soft_budget_ratio is None."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100)
        assert proc._soft_budget_ratio is None

    def test_soft_budget_ratio_stored(self):
        """soft_budget_ratio is stored on the processor."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        assert proc._soft_budget_ratio == 0.9

    def test_constructor_soft_budget_ratio(self):
        """soft_budget_ratio works via main constructor."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=100, soft_budget_ratio=0.8)
        assert proc._soft_budget_ratio == 0.8

    def test_no_boost_before_soft_threshold(self):
        """Logits unchanged before soft threshold (ratio * budget)."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        logits = [1.0] * 200
        # At count=80, threshold=90 → no boost
        result = proc._apply_budget(True, 80, logits)
        assert result[101] == 1.0  # unchanged

    def test_boost_after_soft_threshold(self):
        """</think> logit is boosted between soft threshold and hard budget."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        logits = [1.0] * 200
        # At count=95, threshold=90, budget=100 → should boost </think>
        result = proc._apply_budget(True, 95, logits)
        assert result[101] > 1.0  # boosted
        # Other tokens unchanged
        assert result[0] == 1.0

    def test_boost_increases_linearly(self):
        """Boost grows linearly from threshold to budget."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        # Threshold = 90, budget = 100
        # At count=90: progress=0.0 → no boost
        logits_90 = [1.0] * 200
        proc._apply_budget(True, 90, logits_90)

        # At count=95: progress=0.5 → medium boost
        logits_95 = [1.0] * 200
        proc._apply_budget(True, 95, logits_95)

        # At count=99: progress=0.9 → near-max boost
        logits_99 = [1.0] * 200
        proc._apply_budget(True, 99, logits_99)

        assert logits_90[101] <= logits_95[101] <= logits_99[101]

    def test_hard_cutoff_still_works_with_soft(self):
        """At budget, hard cutoff still forces </think> even with soft ratio."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        logits = [1.0] * 200
        result = proc._apply_budget(True, 100, logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")

    def test_soft_budget_no_effect_when_not_thinking(self):
        """Soft budget has no effect when not in a thinking block."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        logits = [1.0] * 200
        result = proc._apply_budget(False, 95, logits)
        assert result[101] == 1.0  # unchanged

    def test_soft_budget_via_vllm_processor(self):
        """Soft budget works through the vllm_processor interface."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=10, soft_budget_ratio=0.8)
        # threshold=8, at count=9 (in soft zone)
        # <think>=100 then 9 tokens
        logits = [1.0] * 200
        result = proc.vllm_processor([100] + list(range(9)), logits)
        assert result[101] > 1.0  # boosted

    def test_max_boost_capped_at_5(self):
        """Maximum boost is approximately 5.0."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, soft_budget_ratio=0.9)
        logits = [1.0] * 200
        # At count=99, progress = 9/10 = 0.9 → boost = 0.9 * 5.0 = 4.5
        proc._apply_budget(True, 99, logits)
        # Boost should be original + boost_value, where boost_value <= 5.0
        assert logits[101] <= 1.0 + 5.0 + 0.01  # small epsilon


class TestEarlyStoppingText:
    """Tests for early stopping text multi-token forcing."""

    def test_forces_first_token_of_suffix(self):
        """When budget exhausted with early_stopping_tokens, forces first suffix token."""
        # early_stopping_tokens: [50, 51, 101] (101 = </think>)
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        logits = [1.0] * 200
        result = proc._apply_budget(True, 2, logits)
        # Should force token 50 (first in suffix)
        assert result[50] == 0.0
        assert result[0] == float("-inf")
        assert result[101] == float("-inf")  # not </think> yet

    def test_forces_sequence_incrementally(self):
        """Subsequent calls force the next tokens in the suffix sequence."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        # First call: forces token 50
        logits1 = [1.0] * 200
        proc._apply_budget(True, 2, logits1)
        assert logits1[50] == 0.0

        # Second call: forces token 51
        logits2 = [1.0] * 200
        proc._apply_budget(True, 2, logits2)
        assert logits2[51] == 0.0

        # Third call: forces token 101 (</think>)
        logits3 = [1.0] * 200
        proc._apply_budget(True, 2, logits3)
        assert logits3[101] == 0.0

    def test_falls_back_to_end_think_after_suffix(self):
        """After full suffix emitted, falls back to forcing </think>."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51]
        )
        # Exhaust the suffix
        for _ in range(2):
            proc._apply_budget(True, 2, [1.0] * 200)

        # Now past the suffix — should force bare </think>
        logits = [1.0] * 200
        proc._apply_budget(True, 2, logits)
        assert logits[101] == 0.0

    def test_none_early_stopping_forces_bare_end_think(self):
        """early_stopping_tokens=None forces bare </think> (backward compat)."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=2)
        logits = [1.0] * 200
        result = proc._apply_budget(True, 2, logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")

    def test_via_vllm_processor_incremental(self):
        """Early stopping works through vllm_processor with incremental tokens."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        # Build up: <think>=100, 2 thinking tokens
        token_ids = [100, 5, 6]

        # Budget exhausted — should force token 50
        logits1 = [1.0] * 200
        result1 = proc.vllm_processor(token_ids, logits1)
        assert result1[50] == 0.0

        # Next token is the forced 50
        token_ids.append(50)
        logits2 = [1.0] * 200
        result2 = proc.vllm_processor(token_ids, logits2)
        assert result2[51] == 0.0

        # Next token is the forced 51
        token_ids.append(51)
        logits3 = [1.0] * 200
        result3 = proc.vllm_processor(token_ids, logits3)
        assert result3[101] == 0.0

    def test_via_constructor_with_tokenizer(self):
        """early_stopping_text encodes via tokenizer at init time."""
        tok = _make_tokenizer()
        tok.encode.return_value = [50, 51, 101]
        proc = ThinkingBudgetProcessor(tok, budget=5, early_stopping_text="some text")
        assert proc._early_stopping_tokens == [50, 51, 101]

    def test_default_early_stopping_text_applied(self):
        """Constructor uses DEFAULT_EARLY_STOPPING_SUFFIX when not specified."""
        tok = _make_tokenizer()
        tok.encode.return_value = [50, 51, 52, 101]
        proc = ThinkingBudgetProcessor(tok, budget=5)
        # encode was called with DEFAULT_EARLY_STOPPING_SUFFIX
        assert proc._early_stopping_tokens == [50, 51, 52, 101]
        tok.encode.assert_called_with(DEFAULT_EARLY_STOPPING_SUFFIX, add_special_tokens=False)

    def test_none_early_stopping_text_disables(self):
        """Passing early_stopping_text=None disables early stopping."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=5, early_stopping_text=None)
        assert proc._early_stopping_tokens is None

    def test_forcing_index_resets_on_new_think_block(self):
        """_forcing_index resets when a new <think> block starts (per-block mode)."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=1, early_stopping_tokens=[50, 101], per_block=True
        )
        # First block: <think>=100, 1 token → budget exhausted → forces 50
        logits = [1.0] * 200
        proc.vllm_processor([100, 5], logits)
        assert logits[50] == 0.0  # forced first suffix token
        assert proc._forcing_index == 1

        # Suffix token emitted, then </think>, then new <think>
        proc.vllm_processor([100, 5, 50, 101, 100], [1.0] * 200)
        assert proc._forcing_index == 0
        assert proc._in_thinking is True

    def test_default_early_stopping_text_constant(self):
        """DEFAULT_EARLY_STOPPING_SUFFIX is defined and contains </think>."""
        assert "</think>" in DEFAULT_EARLY_STOPPING_SUFFIX


class TestStatefulVLLMProcessor:
    """Tests for stateful vllm_processor tracking."""

    def test_incremental_matches_full_scan(self):
        """Stateful vllm_processor produces same results as _count_thinking."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100)
        tokens = [100, 5, 6, 7, 101, 100, 8, 9]

        # Process incrementally
        for i, _tok in enumerate(tokens):
            # Call vllm_processor with growing prefix
            proc.vllm_processor(tokens[: i + 1], [1.0] * 200)

        # Final state should match full scan
        in_thinking, count = proc._count_thinking(tokens)
        assert proc._in_thinking == in_thinking
        assert proc._count == count

    def test_state_tracks_correctly_through_blocks(self):
        """State correctly transitions through open/close/reopen."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100)

        # <think>
        proc.vllm_processor([100], [1.0] * 200)
        assert proc._in_thinking is True
        assert proc._count == 0

        # <think> 5
        proc.vllm_processor([100, 5], [1.0] * 200)
        assert proc._in_thinking is True
        assert proc._count == 1

        # <think> 5 6
        proc.vllm_processor([100, 5, 6], [1.0] * 200)
        assert proc._in_thinking is True
        assert proc._count == 2

        # <think> 5 6 </think>
        proc.vllm_processor([100, 5, 6, 101], [1.0] * 200)
        assert proc._in_thinking is False
        assert proc._count == 2  # shared mode: count preserved

    def test_budget_enforcement_with_stateful(self):
        """Stateful processor correctly enforces budget."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=2)
        # <think> + 2 tokens = at budget
        logits = [1.0] * 200
        result = proc.vllm_processor([100, 5, 6], logits)
        assert result[101] == 0.0
        assert result[0] == float("-inf")

    def test_empty_token_ids(self):
        """Empty token_ids is handled gracefully."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=10)
        logits = [1.0] * 200
        result = proc.vllm_processor([], logits)
        assert result[0] == 1.0  # unchanged


class TestHFEarlyStopping:
    """Tests for early stopping via hf_processor (stateless derivation)."""

    def test_hf_forces_first_suffix_token(self):
        """hf_processor forces first early stopping token when budget exhausted."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        # Sequence: <think>=100, 2 thinking tokens (at budget)
        input_ids = [[100, 5, 6]]
        scores = [[1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][50] == 0.0
        assert result[0][0] == float("-inf")

    def test_hf_forces_second_suffix_token(self):
        """hf_processor derives forcing_index=1 when first suffix token already emitted."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        # Sequence: <think>=100, 2 thinking tokens, then forced 50
        input_ids = [[100, 5, 6, 50]]
        scores = [[1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][51] == 0.0

    def test_hf_forces_third_suffix_token(self):
        """hf_processor derives forcing_index=2 when first two suffix tokens emitted."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        input_ids = [[100, 5, 6, 50, 51]]
        scores = [[1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][101] == 0.0

    def test_hf_no_early_stopping_forces_bare_end_think(self):
        """Without early_stopping_tokens, hf_processor forces bare </think>."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=2)
        input_ids = [[100, 5, 6]]
        scores = [[1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][101] == 0.0

    def test_hf_batch_independence_with_early_stopping(self):
        """Each batch element independently derives forcing_index."""
        proc = ThinkingBudgetProcessor.from_token_ids(
            100, 101, budget=2, early_stopping_tokens=[50, 51, 101]
        )
        input_ids = [
            [100, 5, 6],  # at budget, no suffix emitted yet → force 50
            [100, 5, 6, 50],  # at budget, 1 suffix token emitted → force 51
        ]
        scores = [[1.0] * 200, [1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][50] == 0.0
        assert result[1][51] == 0.0


class TestV1EarlyStopping:
    """Tests for early stopping text in V1 processor."""

    def _make_v1_class(self):
        """Create V1 processor class with a mocked AdapterLogitsProcessor base."""
        import sys

        class MockAdapterLogitsProcessor:
            pass

        mock_module = MagicMock()
        mock_module.AdapterLogitsProcessor = MockAdapterLogitsProcessor
        sys.modules["vllm.v1.sample.logits_processor"] = mock_module

        from llenvs.inference.thinking import _build_v1_thinking_processor_class

        cls = _build_v1_thinking_processor_class()
        del sys.modules["vllm.v1.sample.logits_processor"]
        return cls

    def test_validate_params_accepts_early_stopping_text(self):
        """validate_params accepts valid thinking_early_stopping_text string."""
        cls = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {
            "thinking_budget": 512,
            "thinking_early_stopping_text": "\n\nAnswer now.\n</think>\n\n",
        }
        cls.validate_params(params)

    def test_validate_params_rejects_invalid_early_stopping_type(self):
        """validate_params rejects non-string thinking_early_stopping_text."""
        cls = self._make_v1_class()
        params = MagicMock()
        params.extra_args = {"thinking_early_stopping_text": 123}
        with pytest.raises(ValueError, match="must be a string"):
            cls.validate_params(params)

    def test_new_req_passes_early_stopping_tokens(self):
        """new_req_logits_processor creates processor with early_stopping_tokens."""
        cls = self._make_v1_class()
        instance = cls.__new__(cls)
        instance._available = True
        instance._think_id = 100
        instance._end_think_id = 101

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [50, 51, 101]
        instance._tokenizer = mock_tokenizer

        params = MagicMock()
        params.extra_args = {
            "thinking_budget": 10,
            "thinking_early_stopping_text": "answer now</think>",
        }
        result = instance.new_req_logits_processor(params)
        assert callable(result)


class TestPerBlockBudget:
    """Tests for per_block vs shared (default) budget modes."""

    def test_default_is_shared(self):
        """Default per_block is False (shared budget)."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=10)
        assert proc._per_block is False

    def test_per_block_via_constructor(self):
        """per_block is stored via main constructor."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=10, per_block=True)
        assert proc._per_block is True

    def test_per_block_via_from_token_ids(self):
        """per_block is stored via from_token_ids."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=10, per_block=True)
        assert proc._per_block is True

    def test_shared_accumulates_across_blocks(self):
        """Shared mode: count accumulates across closed blocks."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=5)
        # Block 1: 3 tokens, Block 2: 2 tokens → total 5 = at budget
        logits = [1.0] * 200
        result = proc.vllm_processor([100, 5, 6, 7, 101, 100, 8, 9], logits)
        assert result[101] == 0.0  # budget exhausted
        assert result[0] == float("-inf")

    def test_per_block_resets_across_blocks(self):
        """Per-block mode: count resets, second block under budget."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=5, per_block=True)
        # Block 1: 3 tokens, Block 2: 2 tokens → only 2 in current block
        logits = [1.0] * 200
        result = proc.vllm_processor([100, 5, 6, 7, 101, 100, 8, 9], logits)
        assert result[0] == 1.0  # not forced — under per-block budget

    def test_shared_forces_immediate_close_after_exhaustion(self):
        """Shared mode: new block after exhaustion is immediately forced closed."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=3)
        # Block 1: 3 tokens (exhausts budget), forced close, new block opens
        tokens = [100, 5, 6, 7, 101, 100]
        logits = [1.0] * 200
        result = proc.vllm_processor(tokens, logits)
        # Budget already at 3, new <think> block → immediately forced
        assert result[101] == 0.0
        assert result[0] == float("-inf")

    def test_shared_via_hf_processor(self):
        """Shared budget works through hf_processor."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=5)
        # Block 1: 3 tokens, Block 2: 2 tokens → total 5 = at budget
        input_ids = [[100, 5, 6, 7, 101, 100, 8, 9]]
        scores = [[1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][101] == 0.0

    def test_per_block_via_hf_processor(self):
        """Per-block budget works through hf_processor."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=5, per_block=True)
        input_ids = [[100, 5, 6, 7, 101, 100, 8, 9]]
        scores = [[1.0] * 200]
        result = proc.hf_processor(input_ids, scores)
        assert result[0][0] == 1.0  # not forced

    def test_shared_stateful_matches_stateless(self):
        """Stateful (vllm) and stateless (_count_thinking) agree in shared mode."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100)
        tokens = [100, 5, 6, 101, 100, 7, 8, 9]
        for i in range(len(tokens)):
            proc.vllm_processor(tokens[: i + 1], [1.0] * 200)
        in_thinking, count = proc._count_thinking(tokens)
        assert proc._in_thinking == in_thinking
        assert proc._count == count

    def test_per_block_stateful_matches_stateless(self):
        """Stateful (vllm) and stateless (_count_thinking) agree in per-block mode."""
        proc = ThinkingBudgetProcessor.from_token_ids(100, 101, budget=100, per_block=True)
        tokens = [100, 5, 6, 101, 100, 7, 8, 9]
        for i in range(len(tokens)):
            proc.vllm_processor(tokens[: i + 1], [1.0] * 200)
        in_thinking, count = proc._count_thinking(tokens)
        assert proc._in_thinking == in_thinking
        assert proc._count == count
