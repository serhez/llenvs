"""Token-preserving reasoning and answer phases with explicit supplied spans."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

from llenvs.inference.protocol import GenerationResult, SamplingParams, StopReason
from llenvs.inference.reasoning import local_chat_messages
from llenvs.inference.records import USAGE_FIELDS, sum_known


@dataclass(frozen=True)
class TokenPhase:
    """An immutable sampled sequence in one actual inference context."""

    token_ids: tuple[int, ...]
    result: GenerationResult


def controlled_chat(backend, messages, params, *, generate=None, observer=None, index=0):
    """Render a configured local family's initial context exactly once."""
    if getattr(backend, "_reasoning_family", None) != "qwen3.8":
        raise ValueError(
            "after_reasoning_text requires a configured, verified local reasoning family"
        )
    if any(message.images or message.content_blocks for message in messages):
        raise ValueError("Inserted text currently requires text-only messages")
    tokenizer = backend.tokenizer
    template_kwargs = backend._chat_template_kwargs_for(params)
    prompt = tokenizer.apply_chat_template(
        local_chat_messages(messages, backend._reasoning_family),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if len(closing) != 1:
        raise ValueError("Configured reasoning closure must encode as a single token")
    if observer is not None:
        observer.begin_phases(index)

        def generate_phase(ids, budget, **options):
            return observer.run_phase(
                index,
                options["phase_name"],
                {"prompt_token_ids": list(ids), "budget": budget, "params": params, **options},
                lambda child: backend._generate_tokens(
                    ids, params, budget, **options, observer=child
                ),
                backend.reparse_token_receipt,
            )

        generate = generate_phase
    return generate_controlled(
        prompt,
        params,
        generate=generate
        or (lambda ids, budget, **kwargs: backend._generate_tokens(ids, params, budget, **kwargs)),
        suffix_ids=tokenizer.encode(params.after_reasoning_text or "", add_special_tokens=False),
        close_ids=closing,
        separator_ids=tokenizer.encode("\n\n", add_special_tokens=False),
        max_context=backend.capabilities.max_context_length,
        reasoning_enabled=template_kwargs.get("enable_thinking", True),
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )


def generate_controlled(
    initial_prompt_ids: Sequence[int],
    params: SamplingParams,
    *,
    generate: Callable,
    suffix_ids: Sequence[int],
    close_ids: Sequence[int],
    separator_ids: Sequence[int],
    max_context: int,
    decode: Callable,
    reasoning_enabled: bool | None = None,
) -> GenerationResult:
    """Generate free reasoning, then an optionally constrained answer.

    ``generate`` accepts exact prompt IDs and a sampled-token budget. Its stable
    ``phase_name`` is suitable for a child journal key. It must retain stop IDs in
    the returned sequence and must never retokenize a sampled reply.
    """
    if params.thinking_budget_per_block or params.thinking_budget_soft_ratio is not None:
        raise ValueError("Inserted text does not support per-block or soft thinking budgets")
    if params.max_tokens is None or len(close_ids) != 1 or max_context is None:
        raise ValueError(
            "Insertion needs an explicit token budget, context limit and single verified closing ID"
        )
    if params.thinking_budget is not None and params.thinking_budget <= 0:
        raise ValueError("thinking_budget must be positive")
    prompt = list(initial_prompt_ids)
    suffix, close, separator = list(suffix_ids), list(close_ids), list(separator_ids)
    phases = []
    remaining = params.max_tokens
    closing_index = None
    if reasoning_enabled is None:
        reasoning_enabled = params.reasoning_effort != "none"
    if not reasoning_enabled:
        try:
            boundary = len(prompt) - 1 - prompt[::-1].index(close[0])
        except ValueError as exc:
            raise ValueError("Disabled-reasoning template has no verified closing marker") from exc
        if prompt[boundary + 1 :] != separator:
            raise ValueError("Disabled-reasoning template has an unverified answer separator")
        answer_prompt = prompt[:boundary] + suffix + close + separator
        reasoning = []
        closure_source = "template"
    else:
        budget = min(remaining, params.thinking_budget or remaining)
        first = generate(prompt, budget, phase_name="reasoning", stop_id=close[0], schema=None)
        phases.append(first)
        sampled = list(first.token_ids)
        if len(sampled) > budget:
            raise ValueError("Reasoning phase exceeded its sampled-token budget")
        remaining -= len(sampled)
        if close[0] in sampled:
            closing_index = sampled.index(close[0])
            if closing_index != len(sampled) - 1:
                raise ValueError("Reasoning phase did not stop at its closing token")
            reasoning = sampled[:closing_index]
            closure_source = "sampled"
        elif (
            params.thinking_budget is not None
            and len(sampled) == budget
            and first.result.finish_reason == StopReason.MAX_TOKENS
        ):
            reasoning = sampled
            closure_source = "thinking_budget"
        else:
            return replace(
                first.result,
                text="",
                token_logprobs=(),
                reasoning_tokens=first.result.reasoning_tokens
                if first.result.reasoning_tokens is not None
                else len(sampled),
                metadata={
                    **first.result.metadata,
                    "raw_token_ids": sampled,
                    "raw_token_logprobs": [asdict(lp) for lp in first.result.token_logprobs or ()],
                    "reasoning": decode(sampled),
                    "reasoning_truncated": True,
                    "reasoning_tokens_source": "provider"
                    if first.result.reasoning_tokens is not None
                    else "token_ids",
                    "after_reasoning_text_applied": False,
                },
            )
        answer_prompt = prompt + reasoning + suffix + close + separator
    if remaining == 0:
        first_result = phases[0].result
        return replace(
            first_result,
            text="",
            finish_reason=StopReason.MAX_TOKENS,
            token_logprobs=(),
            reasoning_tokens=first_result.reasoning_tokens
            if first_result.reasoning_tokens is not None
            else len(reasoning),
            metadata={
                **first_result.metadata,
                "reasoning_truncated": True,
                "raw_token_ids": list(phases[0].token_ids),
                "after_reasoning_text_applied": False,
            },
        )
    if len(answer_prompt) + remaining > max_context:
        raise ValueError("Inserted answer prompt and remaining budget exceed the context limit")
    answer = generate(
        answer_prompt, remaining, phase_name="answer", stop_id=None, schema=params.response_schema
    )
    if len(answer.token_ids) > remaining:
        raise ValueError("Answer phase exceeded the remaining sampled-token budget")
    phases.append(answer)
    counts = {
        name: sum_known(getattr(phase.result, name) for phase in phases) for name in USAGE_FIELDS
    }
    reported = phases[0].result.reasoning_tokens if len(phases) == 2 else None
    counts["reasoning_tokens"] = (reported if reported is not None else len(reasoning)) + (
        answer.result.reasoning_tokens or 0
    )
    records = [
        {
            "token_ids": list(phase.token_ids),
            "token_logprobs": [asdict(lp) for lp in phase.result.token_logprobs or ()],
            "finish_reason": phase.result.finish_reason.name,
            "usage": {name: getattr(phase.result, name) for name in USAGE_FIELDS},
            "metadata": phase.result.metadata,
        }
        for phase in phases
    ]
    requests = [phase.result.metadata.get("request", {}) for phase in phases]
    supplied_start = len(answer_prompt) - len(suffix) - len(close) - len(separator)
    return replace(
        answer.result,
        text=decode(answer.token_ids),
        **counts,
        metadata={
            **answer.result.metadata,
            "phases": records,
            "requests": requests,
            "request": requests[-1],
            "reasoning": decode(reasoning),
            "reasoning_truncated": False,
            "reasoning_tokens_source": "provider" if reported is not None else "token_ids",
            "after_reasoning_text_applied": True,
            "response_schema_applied": params.response_schema is not None,
            "phase_two_prompt_ids": answer_prompt,
            "context_transformation": {
                "sampled_closing_index": closing_index,
                "closure_source": closure_source,
                "supplied_suffix_span": [supplied_start, supplied_start + len(suffix)],
                "supplied_closing_span": [
                    supplied_start + len(suffix),
                    supplied_start + len(suffix) + len(close),
                ],
                "append_only_sampled_trajectory": False,
            },
        },
    )
