"""Thinking budget logits processor for models with <think>...</think> blocks.

Provides a logits processor that caps the number of tokens a model can generate
inside ``<think>...</think>`` reasoning blocks. When the budget is exhausted,
the processor forces a sequence of early-stopping tokens (by default a
natural-language suffix from ``DEFAULT_EARLY_STOPPING_SUFFIX``) which transitions
the model from reasoning to answering.

The ``vllm_processor`` path is stateful (O(1) per token) since processor
instances are per-request in vLLM. The ``hf_processor`` path remains stateless
(scans full history) for batch safety in HuggingFace.
"""

from typing import Any

DEFAULT_EARLY_STOPPING_SUFFIX = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

_UNSET = object()


class ThinkingBudgetProcessor:
    """Logits processor that caps thinking tokens.

    Works with any model whose tokenizer contains ``<think>`` and ``</think>``
    as single tokens (e.g. Qwen3).

    The ``vllm_processor`` method is stateful — it tracks thinking state
    incrementally for O(1) per-token cost. The ``hf_processor`` method is
    stateless — it scans the full token history each call for batch safety.

    When the thinking budget is exhausted, the processor forces a sequence
    of early-stopping tokens. By default this is ``DEFAULT_EARLY_STOPPING_SUFFIX``
    (a natural-language suffix ending with ``</think>``). Pass
    ``early_stopping_text=None`` to force bare ``</think>`` instead.

    Args:
        tokenizer: A tokenizer with ``get_vocab()``, ``convert_tokens_to_ids()``,
            or ``encode()`` methods.
        budget: Maximum number of thinking tokens allowed. By default (shared
            mode), this is cumulative across all ``<think>`` blocks. With
            ``per_block=True``, each block gets its own independent budget.
        soft_budget_ratio: If set, begin boosting ``</think>`` logit at
            ``ratio * budget`` tokens.
        early_stopping_text: Text to force when budget is exhausted. Encoded
            at init time. Defaults to ``DEFAULT_EARLY_STOPPING_SUFFIX``. Pass
            ``None`` explicitly to force bare ``</think>`` instead.
        per_block: If ``True``, each ``<think>`` block gets its own independent
            budget (counter resets on each ``<think>``). If ``False`` (default),
            the budget is shared across all thinking blocks.
    """

    def __init__(
        self,
        tokenizer: Any,
        budget: int,
        soft_budget_ratio: float | None = None,
        early_stopping_text: str | None = _UNSET,
        per_block: bool = False,
    ) -> None:
        self._budget = budget
        self._soft_budget_ratio = soft_budget_ratio
        self._per_block = per_block
        self._think_id = self._resolve_token(tokenizer, "<think>")
        self._end_think_id = self._resolve_token(tokenizer, "</think>")

        # Early stopping — default to DEFAULT_EARLY_STOPPING_SUFFIX
        if early_stopping_text is _UNSET:
            early_stopping_text = DEFAULT_EARLY_STOPPING_SUFFIX
        if early_stopping_text is not None:
            self._early_stopping_tokens = self._resolve_early_stopping_tokens(
                tokenizer, early_stopping_text
            )
        else:
            self._early_stopping_tokens = None

        # Stateful tracking for vllm_processor
        self._in_thinking = False
        self._count = 0
        self._forcing_index = 0
        self._processed_count = 0

    @classmethod
    def from_token_ids(
        cls,
        think_id: int,
        end_think_id: int,
        budget: int,
        soft_budget_ratio: float | None = None,
        early_stopping_tokens: list[int] | None = None,
        per_block: bool = False,
    ) -> "ThinkingBudgetProcessor":
        """Create a processor with pre-resolved token IDs.

        Useful when token IDs have already been resolved (e.g. at engine
        init time) and you want to avoid re-resolving per request.

        Args:
            think_id: Token ID for ``<think>``.
            end_think_id: Token ID for ``</think>``.
            budget: Maximum tokens allowed inside each thinking block.
            soft_budget_ratio: If set, begin boosting ``</think>`` logit
                at ``ratio * budget`` tokens (linear ramp to max ~5.0).
            early_stopping_tokens: Pre-encoded token IDs to force when budget
                is exhausted. ``None`` forces bare ``</think>``.
            per_block: If ``True``, each ``<think>`` block gets its own
                independent budget. If ``False`` (default), the budget is
                shared across all thinking blocks.
        """
        instance = cls.__new__(cls)
        instance._budget = budget
        instance._soft_budget_ratio = soft_budget_ratio
        instance._per_block = per_block
        instance._think_id = think_id
        instance._end_think_id = end_think_id
        instance._early_stopping_tokens = early_stopping_tokens
        instance._in_thinking = False
        instance._count = 0
        instance._forcing_index = 0
        instance._processed_count = 0
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

    @staticmethod
    def _resolve_early_stopping_tokens(tokenizer: Any, text: str) -> list[int]:
        """Encode early stopping text into token IDs.

        Args:
            tokenizer: A tokenizer with an ``encode()`` method.
            text: The early stopping text to encode.

        Returns:
            List of token IDs.
        """
        return tokenizer.encode(text, add_special_tokens=False)

    def _update_thinking(self, token_id: int) -> None:
        """Update stateful thinking tracking with a single new token.

        Used by ``vllm_processor`` for O(1) per-token cost.
        """
        if token_id == self._think_id:
            self._in_thinking = True
            if self._per_block:
                self._count = 0
            self._forcing_index = 0
        elif token_id == self._end_think_id:
            self._in_thinking = False
            if self._per_block:
                self._count = 0
        elif self._in_thinking:
            self._count += 1

    def _count_thinking(self, token_ids: list[int] | tuple[int, ...]) -> tuple[bool, int]:
        """Derive thinking state from the full token history.

        Returns:
            A tuple of (in_thinking, count) where *in_thinking* is whether the
            last open ``<think>`` has not been closed, and *count* is the number
            of thinking tokens. In shared mode (default), this is cumulative
            across all blocks; in per-block mode, it resets on each ``<think>``.
        """
        in_thinking = False
        count = 0
        for tid in token_ids:
            if tid == self._think_id:
                in_thinking = True
                if self._per_block:
                    count = 0
            elif tid == self._end_think_id:
                in_thinking = False
                if self._per_block:
                    count = 0
            elif in_thinking:
                count += 1
        return in_thinking, count

    def _apply_budget(self, in_thinking: bool, count: int, logits: Any) -> Any:
        """Mask or boost logits based on thinking budget.

        When ``in_thinking`` and ``count >= budget``, sets all logits to
        ``-inf`` except the ``</think>`` token (set to ``0.0``).

        When ``soft_budget_ratio`` is set and ``count`` is between
        ``ratio * budget`` and ``budget``, boosts the ``</think>`` logit
        with a linear ramp (max boost ~5.0) to encourage natural closing.

        Returns the (possibly modified) logits object.
        """
        if not in_thinking:
            return logits

        # Hard cutoff — force early stopping sequence or bare </think>
        if count >= self._budget:
            if self._early_stopping_tokens is not None and self._forcing_index < len(
                self._early_stopping_tokens
            ):
                forced_id = self._early_stopping_tokens[self._forcing_index]
                self._forcing_index += 1
            else:
                forced_id = self._end_think_id

            if hasattr(logits, "fill_"):
                logits.fill_(float("-inf"))
                logits[forced_id] = 0.0
            else:
                for i in range(len(logits)):
                    logits[i] = float("-inf")
                logits[forced_id] = 0.0
            return logits

        # Soft transition — boost </think> logit
        if self._soft_budget_ratio is not None and self._budget > 0:
            threshold = int(self._soft_budget_ratio * self._budget)
            if count >= threshold:
                ramp_length = self._budget - threshold
                if ramp_length > 0:
                    progress = (count - threshold) / ramp_length
                    boost = progress * 5.0
                    logits[self._end_think_id] = logits[self._end_think_id] + boost

        return logits

    def vllm_processor(self, token_ids: list[int], logits: Any) -> Any:
        """vLLM logits processor signature: ``(list[int], Tensor) -> Tensor``.

        Uses stateful tracking for O(1) per-token cost. Processes only
        new tokens since the last call. Each processor instance is
        per-request in vLLM, so statefulness is safe.
        """
        for i in range(self._processed_count, len(token_ids)):
            self._update_thinking(token_ids[i])
        self._processed_count = len(token_ids)
        return self._apply_budget(self._in_thinking, self._count, logits)

    def hf_processor(self, input_ids: Any, scores: Any) -> Any:
        """HuggingFace logits processor signature: ``(Tensor[batch, seq], Tensor[batch, vocab]) -> Tensor``.

        Iterates batch elements independently, applying the budget to each.
        Stateless — derives all state from token history for batch safety.
        """
        for i in range(len(input_ids)):
            seq = input_ids[i]
            # Support both list-of-lists and tensor-like
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            in_thinking, count = self._count_thinking(seq)

            # Derive forcing_index from tail of sequence for early stopping
            if in_thinking and count >= self._budget and self._early_stopping_tokens is not None:
                self._forcing_index = self._derive_forcing_index(seq)

            self._apply_budget(in_thinking, count, scores[i])
        return scores

    def _derive_forcing_index(self, token_ids: list[int] | tuple[int, ...]) -> int:
        """Derive how many early stopping tokens have already been emitted.

        Checks how many tokens at the tail of ``token_ids`` match a prefix
        of ``_early_stopping_tokens``. O(k) where k = len(early_stopping_tokens).
        """
        es = self._early_stopping_tokens
        if not es:
            return 0

        # Check the longest prefix of es that matches the tail of token_ids
        max_match = min(len(es), len(token_ids))
        for length in range(max_match, 0, -1):
            tail = token_ids[-length:]
            prefix = es[:length]
            if list(tail) == list(prefix):
                return length
        return 0


def make_v1_thinking_processor_class() -> type | None:
    """Create a vLLM V1-compatible thinking budget processor class.

    Returns a class subclassing ``AdapterLogitsProcessor`` that can be
    registered at ``LLM()`` init time. Per-request budgets are passed via
    ``SamplingParams.extra_args["thinking_budget"]``.

    The returned class has ``__module__`` and ``__qualname__`` rewritten to
    match the module-level ``V1ThinkingBudgetProcessor`` attribute, so pickle
    can resolve it by qualified name. This is required for vLLM's spawn-based
    multiprocessing, which pickles ``logits_processors`` to the EngineCore
    subprocess.

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
            self._tokenizer = None

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
                self._tokenizer = tokenizer
                self._available = True
            except Exception:
                # Model doesn't have <think>/<think> tokens — that's fine,
                # we just can't enforce budgets
                pass

        @classmethod
        def validate_params(cls, params: Any) -> None:
            """Validate extra_args for thinking_budget, soft_ratio, and early_stopping types."""
            extra = getattr(params, "extra_args", None) or {}
            budget = extra.get("thinking_budget")
            if budget is not None and not isinstance(budget, int):
                raise ValueError(f"thinking_budget must be an integer, got {type(budget).__name__}")
            soft_ratio = extra.get("thinking_budget_soft_ratio")
            if soft_ratio is not None:
                if not isinstance(soft_ratio, (int, float)):
                    raise ValueError(
                        f"thinking_budget_soft_ratio must be a float, got {type(soft_ratio).__name__}"
                    )
                if not (0 < soft_ratio < 1):
                    raise ValueError(
                        "thinking_budget_soft_ratio must be between 0 and 1 (exclusive)"
                    )
            early_stopping = extra.get("thinking_early_stopping_text")
            if early_stopping is not None and not isinstance(early_stopping, str):
                raise ValueError(
                    f"thinking_early_stopping_text must be a string, got {type(early_stopping).__name__}"
                )
            per_block = extra.get("thinking_budget_per_block")
            if per_block is not None and not isinstance(per_block, bool):
                raise ValueError(
                    f"thinking_budget_per_block must be a bool, got {type(per_block).__name__}"
                )

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

            soft_ratio = extra.get("thinking_budget_soft_ratio")
            per_block = extra.get("thinking_budget_per_block", False)

            # Resolve early stopping tokens — default to DEFAULT_EARLY_STOPPING_SUFFIX
            _absent = object()
            early_stopping_text = extra.get("thinking_early_stopping_text", _absent)
            if early_stopping_text is _absent:
                early_stopping_text = DEFAULT_EARLY_STOPPING_SUFFIX
            early_stopping_tokens = None
            if early_stopping_text is not None:
                early_stopping_tokens = ThinkingBudgetProcessor._resolve_early_stopping_tokens(
                    self._tokenizer, early_stopping_text
                )

            proc = ThinkingBudgetProcessor.from_token_ids(
                self._think_id,
                self._end_think_id,
                int(budget),
                soft_budget_ratio=float(soft_ratio) if soft_ratio is not None else None,
                early_stopping_tokens=early_stopping_tokens,
                per_block=bool(per_block),
            )
            return proc.vllm_processor

    # Rewrite qualified name so pickle can resolve the class via
    # ``getattr(llenvs.inference.thinking, "V1ThinkingBudgetProcessor")``
    # rather than the closure-local path it would otherwise carry.
    V1ThinkingBudgetProcessor.__module__ = __name__
    V1ThinkingBudgetProcessor.__qualname__ = "V1ThinkingBudgetProcessor"
    return V1ThinkingBudgetProcessor


# Eagerly create the class at module import time when vLLM is available. This
# is what makes pickle work across process boundaries: both parent and child
# (spawn) see ``V1ThinkingBudgetProcessor`` as a real module-level attribute.
# ``None`` when vLLM V1 isn't installed — same as the factory return value.
V1ThinkingBudgetProcessor = make_v1_thinking_processor_class()
