"""Reasoning controls and non-destructive views of local sampled tokens."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from llenvs.inference.protocol import GenerationResult


def local_chat_messages(messages, family):
    """Translate common reasoning fields into the configured template's schema."""
    values = [message.to_dict() for message in messages]
    if family == "qwen3.8":
        for value in values:
            if value.get("reasoning_details"):
                raise ValueError("Qwen local templates cannot render opaque reasoning_details")
            if "reasoning" in value:
                value["reasoning_content"] = value.pop("reasoning")
    return values


def local_template_kwargs(
    family: str | None,
    effort: str | None,
    defaults: Mapping[str, Any],
    requested: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = deepcopy(dict(defaults))
    requested = dict(requested or {})
    if effort is not None:
        if {"enable_thinking", "reasoning_effort"} & requested.keys():
            raise ValueError("reasoning_effort conflicts with per-request template controls")
        if family == "qwen3.8":
            if effort not in {"none", "low", "medium", "xhigh"}:
                raise ValueError(f"Unsupported {family} reasoning_effort: {effort!r}")
            result["enable_thinking"] = effort != "none"
            result.pop("reasoning_effort", None)
            if effort != "none":
                result["reasoning_effort"] = effort
        elif effort == "none" and "enable_thinking" in defaults:
            result["enable_thinking"] = False
        else:
            raise ValueError("reasoning_effort needs a configured local model family")
    result.update(deepcopy(requested))
    return result


def reasoning_view(
    token_ids: Sequence[int],
    *,
    close_id: int,
    decode: Callable[[Sequence[int]], str],
    open_id: int | None = None,
) -> dict[str, Any]:
    """Extract answer coordinates without changing sampled IDs or probabilities.

    The caller must establish that generation began inside a reasoning block.
    Missing closure means a truncated block, never an answer inferred from text.
    """
    raw = list(token_ids)
    start = 1 if raw and open_id is not None and raw[0] == open_id else 0
    try:
        boundary = raw.index(close_id, start)
    except ValueError:
        boundary = len(raw)
    closed = boundary < len(raw)
    answer_start = boundary + 1 if closed else len(raw)
    body, answer = raw[start:boundary], raw[answer_start:]
    return dict(
        raw_token_ids=raw,
        reasoning_ids=body,
        answer_ids=answer,
        reasoning=decode(body),
        answer=decode(answer),
        answer_span=[answer_start, len(raw)],
        reasoning_span=[start, boundary],
        reasoning_truncated=not closed,
        reasoning_tokens=len(body),
        reasoning_tokens_source="token_ids",
    )


def extract_local_reasoning(result: GenerationResult, ids, tokenizer, *, reasoning_enabled):
    """Return an answer view while keeping the full native sequence in metadata."""
    raw_metadata = {
        "raw_text": result.text,
        "raw_token_ids": list(ids),
        "raw_token_logprobs": [asdict(lp) for lp in result.token_logprobs or ()],
    }
    if not reasoning_enabled:
        return replace(result, metadata={**result.metadata, **raw_metadata})
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if len(opening) != 1 or len(closing) != 1:
        raise ValueError("Reasoning extraction requires verified single-token boundaries")
    view = reasoning_view(
        ids,
        close_id=closing[0],
        open_id=opening[0],
        decode=lambda values: tokenizer.decode(values, skip_special_tokens=True),
    )
    start, end = view["answer_span"]
    observed = view["reasoning_tokens"]
    if result.reasoning_tokens is not None:
        view.update(
            reasoning_tokens=result.reasoning_tokens,
            reasoning_tokens_source="provider",
            reasoning_token_ids_count=observed,
        )
    return replace(
        result,
        text=view["answer"],
        reasoning_tokens=view["reasoning_tokens"],
        token_logprobs=result.token_logprobs[start:end]
        if result.token_logprobs is not None
        else None,
        metadata={**result.metadata, **raw_metadata, **view},
    )
