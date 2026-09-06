"""Tests for the shared continuation-scoring helpers (tokenizer-agnostic).

These back the refactor that lets both ``VLLMBackend`` (in-process) and
``SingularityVLLMBackend`` (HTTP) share prompt preparation, and cover the
HTTP-shaped ``prompt_logprobs`` parser used by the Singularity backend.
"""

from __future__ import annotations

import pytest

from llenvs.inference.protocol import (
    ChatMessage,
    LogprobsNotReturnedError,
    ScoringResult,
    TokenScore,
)
from llenvs.inference.scoring_utils import (
    build_scoring_inputs,
    parse_prompt_logprobs_http,
)


class _CharTokenizer:
    """Deterministic fake: prompt rendering is fixed, encode = code points.

    ``apply_chat_template`` returns a constant ``"PROMPT|"`` regardless of the
    conversation (7 code points) and records the kwargs it was called with so
    tests can assert chat-template forwarding.
    """

    PROMPT = "PROMPT|"

    def __init__(self) -> None:
        self.template_kwargs_seen: list[dict] = []
        self.encode_kwargs_seen: list[dict] = []

    def apply_chat_template(self, conversation, *, tokenize=False,
                            add_generation_prompt=True, **kwargs) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.template_kwargs_seen.append(kwargs)
        return self.PROMPT

    def encode(self, text: str, **kwargs) -> list[int]:
        self.encode_kwargs_seen.append(kwargs)
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


class _FixedLenTokenizer(_CharTokenizer):
    """encode() ignores the continuation so full_ids <= prompt_ids (degenerate)."""

    def encode(self, text: str, **kwargs) -> list[int]:
        self.encode_kwargs_seen.append(kwargs)
        return [ord(c) for c in self.PROMPT]


def _msgs(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


class TestBuildScoringInputs:
    def test_normal_prompt_and_continuation(self):
        tok = _CharTokenizer()
        full_texts, prompt_lengths, full_token_ids, empty = build_scoring_inputs(
            tok, {}, [_msgs("hi")], ["AB"],
        )
        assert full_texts == ["PROMPT|AB"]
        assert prompt_lengths == [len("PROMPT|")]
        assert full_token_ids == [[ord(c) for c in "PROMPT|AB"]]
        assert empty == {}

    def test_chat_template_kwargs_forwarded(self):
        tok = _CharTokenizer()
        build_scoring_inputs(tok, {"enable_thinking": False}, [_msgs("hi")], ["A"])
        assert tok.template_kwargs_seen == [{"enable_thinking": False}]

    def test_rendered_chat_text_is_encoded_without_adding_special_tokens(self):
        """The chat template already emits BOS/control tokens.

        Asking ``encode`` to add them again can prepend a second BOS under
        older Transformers releases and shift every continuation-token span.
        """
        tok = _CharTokenizer()
        build_scoring_inputs(tok, {}, [_msgs("hi")], ["AB"])
        assert tok.encode_kwargs_seen == [
            {"add_special_tokens": False},
            {"add_special_tokens": False},
        ]

    def test_empty_continuation_yields_empty_result_and_is_not_scored(self):
        tok = _CharTokenizer()
        full_texts, prompt_lengths, full_token_ids, empty = build_scoring_inputs(
            tok, {}, [_msgs("a"), _msgs("b")], ["", "AB"],
        )
        # Only the second item is scorable.
        assert full_texts == ["PROMPT|AB"]
        assert prompt_lengths == [len("PROMPT|")]
        assert list(empty.keys()) == [0]
        assert empty[0] == ScoringResult(
            token_scores=(), prompt_tokens=len("PROMPT|"), scored_tokens=0,
        )

    def test_degenerate_continuation_yields_empty_result(self):
        tok = _FixedLenTokenizer()
        full_texts, _, _, empty = build_scoring_inputs(
            tok, {}, [_msgs("a")], ["AB"],
        )
        assert full_texts == []
        assert empty[0].scored_tokens == 0

    def test_length_mismatch_raises(self):
        tok = _CharTokenizer()
        with pytest.raises(ValueError, match="equal length"):
            build_scoring_inputs(tok, {}, [_msgs("a")], ["x", "y"])

    def test_empty_batch(self):
        tok = _CharTokenizer()
        assert build_scoring_inputs(tok, {}, [], []) == ([], [], [], {})


def _http_plps(full_ids: list[int], *, logprob_at):
    """Build an HTTP-shaped prompt_logprobs list (string token-id keys)."""
    plps: list = [None]  # position 0 has no preceding context
    for pos in range(1, len(full_ids)):
        tid = full_ids[pos]
        plps.append({
            str(tid): {
                "logprob": logprob_at(pos),
                "decoded_token": chr(tid),
                "rank": 1,
            }
        })
    return plps


class TestParsePromptLogprobsHttp:
    def test_extracts_continuation_span(self):
        full_ids = [ord(c) for c in "PROMPT|AB"]
        prompt_len = len("PROMPT|")
        plps = _http_plps(full_ids, logprob_at=lambda pos: round(-0.1 * pos, 4))

        result = parse_prompt_logprobs_http(
            prompt_len=prompt_len, full_ids=full_ids,
            prompt_logprobs=plps, model_name="m",
        )
        assert isinstance(result, ScoringResult)
        assert result.prompt_tokens == prompt_len
        assert result.scored_tokens == 2
        assert [ts.logprob for ts in result.token_scores] == [-0.7, -0.8]
        assert [ts.token_id for ts in result.token_scores] == [ord("A"), ord("B")]
        assert [ts.token for ts in result.token_scores] == ["A", "B"]
        assert all(ts.log_probs_all is None for ts in result.token_scores)

    def test_no_continuation_tokens_returns_empty(self):
        full_ids = [ord(c) for c in "PROMPT|"]
        result = parse_prompt_logprobs_http(
            prompt_len=len(full_ids), full_ids=full_ids,
            prompt_logprobs=[None] * len(full_ids), model_name="m",
        )
        assert result.token_scores == ()
        assert result.scored_tokens == 0

    def test_none_prompt_logprobs_raises(self):
        full_ids = [ord(c) for c in "PROMPT|AB"]
        with pytest.raises(LogprobsNotReturnedError):
            parse_prompt_logprobs_http(
                prompt_len=len("PROMPT|"), full_ids=full_ids,
                prompt_logprobs=None, model_name="m",
            )

    def test_too_few_positions_raises(self):
        full_ids = [ord(c) for c in "PROMPT|AB"]
        # Truncated list: missing the last position.
        plps = _http_plps(full_ids, logprob_at=lambda pos: -0.1)[:-1]
        with pytest.raises(LogprobsNotReturnedError):
            parse_prompt_logprobs_http(
                prompt_len=len("PROMPT|"), full_ids=full_ids,
                prompt_logprobs=plps, model_name="m",
            )

    def test_realized_token_absent_raises(self):
        full_ids = [ord(c) for c in "PROMPT|AB"]
        plps = _http_plps(full_ids, logprob_at=lambda pos: -0.1)
        # Corrupt the last position so the realized token id is missing.
        plps[-1] = {"99999": {"logprob": -1.0, "decoded_token": "?", "rank": 1}}
        with pytest.raises(LogprobsNotReturnedError):
            parse_prompt_logprobs_http(
                prompt_len=len("PROMPT|"), full_ids=full_ids,
                prompt_logprobs=plps, model_name="m",
            )
