"""Thinking budget logits processor for models with <think>...</think> blocks.

Provides a stateless logits processor that caps the number of tokens a model
can generate inside ``<think>...</think>`` reasoning blocks. When the budget
is exhausted, all logits are masked to ``-inf`` except the ``</think>`` token,
forcing the model to close the thinking block.
"""

from typing import Any


class ThinkingBudgetProcessor:
    """Stateless logits processor that caps thinking tokens.

    Works with any model whose tokenizer contains ``<think>`` and ``</think>``
    as single tokens (e.g. Qwen3). The processor is fully stateless — it
    derives thinking state by scanning the full token history on each call,
    making it safe for shared-processor batching in both vLLM and HuggingFace.

    Args:
        tokenizer: A tokenizer with ``get_vocab()``, ``convert_tokens_to_ids()``,
            or ``encode()`` methods.
        budget: Maximum number of tokens allowed inside each ``<think>`` block.
            Once reached, the model is forced to emit ``</think>``.
    """

    def __init__(self, tokenizer: Any, budget: int) -> None:
        self._budget = budget
        self._think_id = self._resolve_token(tokenizer, "<think>")
        self._end_think_id = self._resolve_token(tokenizer, "</think>")

    @staticmethod
    def _resolve_token(tokenizer: Any, token: str) -> int:
        """Resolve a special token string to its integer ID.

        Tries three strategies in order:
        1. Look up in ``get_vocab()``
        2. Call ``convert_tokens_to_ids()``
        3. Call ``encode()`` and require exactly one token
        """
        # Strategy 1: vocab lookup
        vocab = tokenizer.get_vocab()
        if token in vocab:
            return vocab[token]

        # Strategy 2: convert_tokens_to_ids
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None:
                return token_id

        # Strategy 3: encode fallback
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
            if len(ids) > 1:
                raise ValueError(
                    f"Token '{token}' encodes to multiple IDs ({ids}); "
                    f"it must be a single token in the tokenizer vocabulary."
                )

        raise ValueError(
            f"Cannot resolve '{token}' to a single token ID. "
            f"The tokenizer must have {token} as a dedicated token."
        )

    def _count_thinking(self, token_ids: list[int] | tuple[int, ...]) -> tuple[bool, int]:
        """Derive thinking state from the full token history.

        Returns:
            A tuple of (in_thinking, count) where *in_thinking* is whether the
            last open ``<think>`` has not been closed, and *count* is the number
            of tokens since that ``<think>`` (excluding the ``<think>`` token
            itself).
        """
        in_thinking = False
        count = 0
        for tid in token_ids:
            if tid == self._think_id:
                in_thinking = True
                count = 0
            elif tid == self._end_think_id:
                in_thinking = False
                count = 0
            elif in_thinking:
                count += 1
        return in_thinking, count

    def _apply_budget(self, in_thinking: bool, count: int, logits: Any) -> Any:
        """Mask logits if the thinking budget is exceeded.

        When ``in_thinking`` and ``count >= budget``, sets all logits to
        ``-inf`` except the ``</think>`` token (set to ``0.0``).

        Returns the (possibly modified) logits object.
        """
        if not in_thinking or count < self._budget:
            return logits

        # Budget exceeded — force </think>
        for i in range(len(logits)):
            logits[i] = float("-inf")
        logits[self._end_think_id] = 0.0
        return logits

    def vllm_processor(self, token_ids: list[int], logits: Any) -> Any:
        """vLLM logits processor signature: ``(list[int], Tensor) -> Tensor``.

        Safe to share across sequences in a batch — each call receives the
        token history for a single sequence.
        """
        in_thinking, count = self._count_thinking(token_ids)
        return self._apply_budget(in_thinking, count, logits)

    def hf_processor(self, input_ids: Any, scores: Any) -> Any:
        """HuggingFace logits processor signature: ``(Tensor[batch, seq], Tensor[batch, vocab]) -> Tensor``.

        Iterates batch elements independently, applying the budget to each.
        """
        for i in range(len(input_ids)):
            seq = input_ids[i]
            # Support both list-of-lists and tensor-like
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            in_thinking, count = self._count_thinking(seq)
            self._apply_budget(in_thinking, count, scores[i])
        return scores
