"""Shared helpers for continuation scoring across vLLM backends.

Continuation scoring computes, for a fixed ``(messages, continuation)`` pair,
the model's per-token log-probability of the continuation conditioned on the
prompt. Both the in-process :class:`~llenvs.inference.backends.vllm.VLLMBackend`
and the container-hosted
:class:`~llenvs.inference.backends.vllm_singularity.SingularityVLLMBackend`
prepare their requests identically (apply the chat template, encode, glue the
continuation on as raw text) and differ only in *how* they obtain vLLM
``prompt_logprobs`` — a resident engine call versus an HTTP request.

This module holds the parts that don't depend on that transport:
- :func:`build_scoring_inputs` — prompt preparation (tokenizer-driven).
- :func:`parse_prompt_logprobs_http` — parse the OpenAI-compatible HTTP
  ``prompt_logprobs`` shape into a :class:`ScoringResult`.

It imports only protocol dataclasses (no ``vllm``/``transformers``), so the
Singularity backend — which runs on hosts without a local vLLM — can import it
freely.
"""

from __future__ import annotations

from typing import Any

from llenvs.inference.protocol import (
    ChatMessage,
    LogprobsNotReturnedError,
    ScoringResult,
    TokenScore,
)


def build_scoring_inputs(
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any],
    messages_batch: list[list[ChatMessage]],
    continuations: list[str],
) -> tuple[list[str], list[int], list[list[int]], dict[int, ScoringResult]]:
    """Prepare per-item scoring inputs from chat messages + continuations.

    For each pair, the prompt is rendered with ``add_generation_prompt=True``
    and the continuation is appended as raw text (matching how a model would
    emit it as its next response). Items with an empty continuation, or whose
    continuation adds no tokens, are pre-resolved to an empty
    :class:`ScoringResult` and excluded from the scorable set.

    Returns ``(full_texts, prompt_lengths, full_token_ids, empty_results)``:
    parallel lists over the *scorable* items (in input order), plus a mapping
    from original batch index to the empty result for skipped items. These are
    exactly the locals both backends need — the in-process backend feeds
    ``full_texts`` to the engine; the HTTP backend feeds ``full_token_ids`` to
    the server; both use ``prompt_lengths`` to locate the continuation span.
    """
    if len(messages_batch) != len(continuations):
        raise ValueError("messages_batch and continuations must have equal length")

    full_texts: list[str] = []
    prompt_lengths: list[int] = []
    full_token_ids: list[list[int]] = []
    empty_results: dict[int, ScoringResult] = {}

    for index, (messages, continuation) in enumerate(zip(messages_batch, continuations)):
        prompt_text = tokenizer.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        # ``apply_chat_template(tokenize=False)`` has already emitted the
        # model's BOS/control tokens. Letting ``encode`` add special tokens a
        # second time can prepend a duplicate BOS (notably with Gemma under
        # older Transformers releases), shifting the continuation span.
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if not continuation:
            empty_results[index] = ScoringResult(
                token_scores=(), prompt_tokens=len(prompt_ids), scored_tokens=0
            )
            continue

        full_text = prompt_text + continuation
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) <= len(prompt_ids):
            empty_results[index] = ScoringResult(
                token_scores=(), prompt_tokens=len(prompt_ids), scored_tokens=0
            )
            continue

        full_texts.append(full_text)
        prompt_lengths.append(len(prompt_ids))
        full_token_ids.append(full_ids)

    return full_texts, prompt_lengths, full_token_ids, empty_results


def _entry_field(entry: Any, key: str) -> Any:
    """Read a field from a prompt-logprob entry (dict from JSON, or object)."""
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def parse_prompt_logprobs_http(
    *,
    prompt_len: int,
    full_ids: list[int],
    prompt_logprobs: Any,
    model_name: str,
) -> ScoringResult:
    """Extract continuation-token logprobs from an HTTP ``prompt_logprobs`` list.

    The vLLM OpenAI-compatible server returns, per prompt position, either
    ``null`` (position 0) or a JSON object mapping the (string) token id to
    ``{"logprob", "rank", "decoded_token"}``. We read positions
    ``[prompt_len, len(full_ids))`` — the continuation span — looking up each
    realized token id. ``TokenScore.log_probs_all`` is left ``None`` (the
    scoring methods only consume the realized token's ``logprob``).
    """
    if len(full_ids) <= prompt_len:
        return ScoringResult(token_scores=(), prompt_tokens=prompt_len, scored_tokens=0)
    if prompt_logprobs is None:
        raise LogprobsNotReturnedError(
            "vLLM server returned no prompt logprobs during continuation "
            f"scoring for model {model_name!r}.",
            backend_name="SingularityVLLMBackend",
            model_name=model_name,
        )

    token_scores: list[TokenScore] = []
    for pos in range(prompt_len, len(full_ids)):
        token_id = full_ids[pos]
        try:
            position_logprobs = prompt_logprobs[pos]
        except (IndexError, KeyError) as exc:
            raise LogprobsNotReturnedError(
                "vLLM server returned too few prompt logprob positions during "
                f"continuation scoring for model {model_name!r}.",
                backend_name="SingularityVLLMBackend",
                model_name=model_name,
            ) from exc

        entry = None
        if isinstance(position_logprobs, dict):
            entry = position_logprobs.get(str(token_id))
            if entry is None:
                entry = position_logprobs.get(token_id)
        if entry is None:
            raise LogprobsNotReturnedError(
                "vLLM server prompt logprobs did not include the continuation "
                f"token id {token_id} at position {pos} for model {model_name!r}.",
                backend_name="SingularityVLLMBackend",
                model_name=model_name,
            )

        decoded = _entry_field(entry, "decoded_token")
        token_scores.append(
            TokenScore(
                token=decoded if decoded is not None else "",
                token_id=int(token_id),
                logprob=float(_entry_field(entry, "logprob")),
            )
        )

    return ScoringResult(
        token_scores=tuple(token_scores),
        prompt_tokens=prompt_len,
        scored_tokens=len(token_scores),
    )
