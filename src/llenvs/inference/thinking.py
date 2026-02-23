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

    @classmethod
    def from_token_ids(
        cls, think_id: int, end_think_id: int, budget: int
    ) -> "ThinkingBudgetProcessor":
        """Create a processor with pre-resolved token IDs.

        Useful when token IDs have already been resolved (e.g. at engine
        init time) and you want to avoid re-resolving per request.

        Args:
            think_id: Token ID for ``<think>``.
            end_think_id: Token ID for ``</think>``.
            budget: Maximum tokens allowed inside each thinking block.
        """
        instance = cls.__new__(cls)
        instance._budget = budget
        instance._think_id = think_id
        instance._end_think_id = end_think_id
        return instance

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


def make_v1_thinking_processor_class() -> type | None:
    """Create a vLLM V1-compatible thinking budget processor class.

    Returns a class subclassing ``AdapterLogitsProcessor`` that can be
    registered at ``LLM()`` init time. Per-request budgets are passed via
    ``SamplingParams.extra_args["thinking_budget"]``.

    Returns:
        The processor class, or ``None`` if the vLLM V1 API is not available
        (e.g. vLLM <0.8 or not installed).
    """
    try:
        from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    except ImportError:
        return None

    class V1ThinkingBudgetProcessor(AdapterLogitsProcessor):
        """V1-compatible thinking budget processor.

        Registered once at engine init. Resolves ``<think>``/``</think>``
        token IDs from the model's tokenizer. Per-request budgets are read
        from ``params.extra_args["thinking_budget"]``.
        """

        def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool = False) -> None:
            super().__init__(vllm_config, device, is_pin_memory)
            self._available = False
            self._think_id = -1
            self._end_think_id = -1

            try:
                model_config = vllm_config.model_config
                model_name = model_config.model

                # Try vLLM's tokenizer utility first, fall back to transformers
                tokenizer = None
                try:
                    from vllm.transformers_utils.tokenizer import get_tokenizer

                    tokenizer = get_tokenizer(model_name)
                except Exception:
                    pass

                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                self._think_id = ThinkingBudgetProcessor._resolve_token(tokenizer, "<think>")
                self._end_think_id = ThinkingBudgetProcessor._resolve_token(tokenizer, "</think>")
                self._available = True
            except Exception:
                # Model doesn't have <think>/<think> tokens — that's fine,
                # we just can't enforce budgets
                pass

        @classmethod
        def validate_params(cls, params: Any) -> None:
            """Validate extra_args for thinking_budget type."""
            extra = getattr(params, "extra_args", None) or {}
            budget = extra.get("thinking_budget")
            if budget is not None and not isinstance(budget, int):
                raise ValueError(f"thinking_budget must be an integer, got {type(budget).__name__}")

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(self, params: Any) -> Any:
            """Return a per-request logits processor callable, or None."""
            extra = getattr(params, "extra_args", None) or {}
            budget = extra.get("thinking_budget")
            if budget is None:
                return None

            if not self._available:
                raise ValueError(
                    "thinking_budget requested but <think>/<think> tokens "
                    "could not be resolved from the model's tokenizer."
                )

            proc = ThinkingBudgetProcessor.from_token_ids(
                self._think_id, self._end_think_id, int(budget)
            )
            return proc.vllm_processor

    return V1ThinkingBudgetProcessor
