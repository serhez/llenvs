"""Tests for ThinkingBudgetProcessor."""

import pytest
from unittest.mock import MagicMock

from llenvs.inference.thinking import ThinkingBudgetProcessor


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
        tok.encode.side_effect = lambda s, add_special_tokens=True: {
            "<think>": [100],
            "</think>": [101],
        }[s]
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
        """Returns (False, 0) after a closed thinking block."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64)
        # <think>=100, tokens, </think>=101
        in_thinking, count = proc._count_thinking([100, 5, 6, 101])
        assert in_thinking is False
        assert count == 0

    def test_second_thinking_block(self):
        """Counter resets for a second thinking block."""
        tok = _make_tokenizer()
        proc = ThinkingBudgetProcessor(tok, budget=64)
        # First block: <think> 5 6 </think>, second block: <think> 7
        in_thinking, count = proc._count_thinking([100, 5, 6, 101, 100, 7])
        assert in_thinking is True
        assert count == 1


class TestBudgetEnforcement:
    """Tests for _apply_budget logit masking."""

    def _make_logits(self, vocab_size=200):
        """Create a simple list-based logits tensor mock."""
        import array

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
