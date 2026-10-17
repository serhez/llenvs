"""Two-phase formatting preserves sampled IDs and explicit context changes."""

import pytest

from llenvs.inference.controlled import TokenPhase, generate_controlled
from llenvs.inference.protocol import GenerationResult, SamplingParams, StopReason, TokenLogprob


def phase(ids, reason=StopReason.STOP_SEQUENCE):
    return TokenPhase(
        tuple(ids),
        GenerationResult(
            finish_reason=reason,
            prompt_tokens=4,
            completion_tokens=len(ids),
            token_logprobs=tuple(TokenLogprob(str(i), i, -0.1) for i in ids),
            metadata={"request": {"max_tokens": 9}},
        ),
    )


def execute(outputs, *, params=None, prompt=(1, 2), context=100):
    seen = []

    def generate(ids, budget, *, phase_name, stop_id, schema):
        seen.append((list(ids), budget, phase_name, stop_id, schema))
        return outputs[len(seen) - 1]

    result = generate_controlled(
        prompt,
        params or SamplingParams(max_tokens=8, after_reasoning_text="suffix"),
        generate=generate,
        suffix_ids=(80, 81),
        close_ids=(99,),
        separator_ids=(10,),
        max_context=context,
        decode=lambda ids: ",".join(map(str, ids)),
    )
    return result, seen


def test_sampled_closure_is_retained_but_moved_to_a_supplied_phase_two_span():
    first = phase([4, 5, 99])
    result, seen = execute([first, phase([6, 7])])
    assert seen[0] == ([1, 2], 8, "reasoning", 99, None)
    assert seen[1][:4] == ([1, 2, 4, 5, 80, 81, 99, 10], 5, "answer", None)
    assert result.text == "6,7"
    assert result.metadata["phases"][0]["token_ids"] == [4, 5, 99]
    assert result.metadata["phases"][0]["token_logprobs"][-1]["token_id"] == 99
    assert result.token_logprobs == phase([6, 7]).result.token_logprobs
    assert result.prompt_tokens == 8 and result.completion_tokens == 5
    assert result.metadata["context_transformation"]["sampled_closing_index"] == 2
    assert result.metadata["phase_two_prompt_ids"] == seen[1][0]
    assert first.token_ids == (4, 5, 99)


def test_total_budget_exhausted_during_reasoning_never_grants_an_answer_budget():
    result, seen = execute(
        [phase([4, 5], StopReason.MAX_TOKENS)],
        params=SamplingParams(max_tokens=2, after_reasoning_text="suffix"),
    )
    assert len(seen) == 1 and result.text == ""
    assert result.finish_reason == StopReason.MAX_TOKENS
    assert result.metadata["reasoning_truncated"]


def test_explicit_thinking_budget_can_force_a_supplied_closure():
    result, seen = execute(
        [phase([4, 5], StopReason.MAX_TOKENS), phase([6])],
        params=SamplingParams(max_tokens=8, thinking_budget=2, after_reasoning_text="suffix"),
    )
    assert seen[0][1] == 2 and seen[1][1] == 6
    assert result.metadata["context_transformation"]["closure_source"] == "thinking_budget"


@pytest.mark.parametrize("stop", [StopReason.END_OF_TEXT, StopReason.STOP_SEQUENCE])
def test_eos_or_caller_stop_before_boundary_does_not_authorize_phase_two(stop):
    result, seen = execute([phase([4], stop)])
    assert len(seen) == 1 and result.text == ""


def test_disabled_reasoning_inserts_before_template_supplied_closure():
    result, seen = execute(
        [phase([6])],
        prompt=(1, 99, 10),
        params=SamplingParams(max_tokens=8, reasoning_effort="none", after_reasoning_text="suffix"),
    )
    assert len(seen) == 1
    assert seen[0][0] == [1, 80, 81, 99, 10]
    assert seen[0][2] == "answer" and result.text == "6"


def test_context_overflow_is_rejected_before_dispatching_answer():
    outputs = [phase([4, 5, 99])]
    with pytest.raises(ValueError, match="context"):
        execute(outputs, context=8)


@pytest.mark.parametrize(
    "options", [{"thinking_budget_per_block": True}, {"thinking_budget_soft_ratio": 0.8}]
)
def test_incompatible_budget_controls_raise_before_phase_one(options):
    with pytest.raises(ValueError):
        execute([], params=SamplingParams(after_reasoning_text="suffix", **options))


def test_reported_reasoning_usage_takes_precedence_over_local_count():
    from dataclasses import replace

    first = phase([4, 5, 99])
    first = replace(first, result=replace(first.result, reasoning_tokens=3))
    result, _ = execute([first, phase([6])])
    assert result.reasoning_tokens == 3
    assert result.metadata["reasoning_tokens_source"] == "provider"


def test_inherited_disabled_template_skips_reasoning_phase():
    from types import SimpleNamespace

    from llenvs.inference.controlled import controlled_chat

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 99, 10]

        def encode(self, text, **kwargs):
            return [99] if text == "</think>" else [10] if text == "\n\n" else [80]

        def decode(self, ids, **kwargs):
            return "answer"

    seen = []

    def generate(ids, budget, **options):
        seen.append((ids, options["phase_name"]))
        return phase([6])

    backend = SimpleNamespace(
        _reasoning_family="qwen3.8",
        tokenizer=Tokenizer(),
        _chat_template_kwargs_for=lambda params: {"enable_thinking": False},
        capabilities=SimpleNamespace(max_context_length=100),
    )
    result = controlled_chat(
        backend, [], SamplingParams(max_tokens=8, after_reasoning_text="suffix"), generate=generate
    )
    assert seen == [([1, 80, 99, 10], "answer")]
    assert result.text == "answer"
