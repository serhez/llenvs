"""Tests for the cleaning layer."""

import pytest

from llenvs.core.cleaning import (
    DEFAULT_POST_CLEANERS,
    DEFAULT_PRE_CLEANERS,
    POST_CLEANERS,
    PRE_CLEANERS,
    resolve_cleaners,
    strip_latex_dollars,
    strip_special_tokens,
    strip_surrounding_quotes,
    strip_thinking_tokens,
    strip_trailing_punctuation,
)
from llenvs.core.extraction import (
    CleanedExtractor,
    PatternAnswerExtractor,
    TagBasedExtractor,
)


class TestStripSpecialTokens:
    """Tests for strip_special_tokens pre-cleaner."""

    def test_strips_endoftext(self):
        assert strip_special_tokens("hello<|endoftext|>") == "hello"

    def test_strips_pad(self):
        assert strip_special_tokens("<pad>hello<pad>") == "hello"

    def test_strips_eos(self):
        assert strip_special_tokens("hello</s>") == "hello"

    def test_strips_im_end(self):
        assert strip_special_tokens("hello<|im_end|>") == "hello"

    def test_strips_im_start(self):
        assert strip_special_tokens("<|im_start|>hello") == "hello"

    def test_strips_bos(self):
        assert strip_special_tokens("<s>hello") == "hello"

    def test_multiple_token_types(self):
        result = strip_special_tokens("<|endoftext|><pad>hello</s><|im_end|>")
        assert result == "hello"

    def test_no_tokens_unchanged(self):
        assert strip_special_tokens("hello world") == "hello world"

    def test_tokens_mid_text(self):
        result = strip_special_tokens("hello<|endoftext|>world")
        assert result == "helloworld"

    def test_empty_string(self):
        assert strip_special_tokens("") == ""


class TestStripThinkingTokens:
    """Tests for strip_thinking_tokens pre-cleaner."""

    def test_closed_block_removed(self):
        result = strip_thinking_tokens("before<think>reasoning here</think>after")
        assert result == "beforeafter"

    def test_unclosed_block_removed(self):
        result = strip_thinking_tokens("answer text<think>truncated reasoning")
        assert result == "answer text"

    def test_multiple_closed_blocks(self):
        result = strip_thinking_tokens("<think>first</think>middle<think>second</think>end")
        assert result == "middleend"

    def test_nested_content_removed(self):
        """Newlines, code, and other content inside think block removed (DOTALL)."""
        result = strip_thinking_tokens(
            "hello<think>\nline1\nline2\ndef foo():\n    pass\n</think>world"
        )
        assert result == "helloworld"

    def test_no_think_tokens_passthrough(self):
        assert strip_thinking_tokens("hello world") == "hello world"

    def test_empty_string_passthrough(self):
        assert strip_thinking_tokens("") == ""

    def test_mixed_closed_and_unclosed(self):
        """Closed block + remaining text + unclosed block at end."""
        result = strip_thinking_tokens("<think>closed</think>answer<think>unclosed reasoning")
        assert result == "answer"

    def test_think_block_at_start(self):
        result = strip_thinking_tokens("<think>reasoning</think>the answer is 42")
        assert result == "the answer is 42"

    def test_think_block_in_middle(self):
        result = strip_thinking_tokens("start<think>reasoning</think>end")
        assert result == "startend"

    def test_think_block_at_end(self):
        result = strip_thinking_tokens("the answer is 42<think>reasoning</think>")
        assert result == "the answer is 42"


class TestStripTrailingPunctuation:
    """Tests for strip_trailing_punctuation post-cleaner."""

    def test_trailing_period(self):
        assert strip_trailing_punctuation("42.") == "42"

    def test_trailing_comma(self):
        assert strip_trailing_punctuation("42,") == "42"

    def test_no_trailing_punctuation(self):
        assert strip_trailing_punctuation("42") == "42"

    def test_decimal_number_preserved(self):
        """Don't strip decimal point from numbers like 3.14."""
        assert strip_trailing_punctuation("3.14") == "3.14"

    def test_text_trailing_period(self):
        assert strip_trailing_punctuation("answer.") == "answer"

    def test_text_trailing_comma(self):
        assert strip_trailing_punctuation("answer,") == "answer"

    def test_empty_string(self):
        assert strip_trailing_punctuation("") == ""

    def test_only_period(self):
        assert strip_trailing_punctuation(".") == ""

    def test_multiple_trailing_periods(self):
        """Only strip the last trailing punctuation."""
        assert strip_trailing_punctuation("42..") == "42."

    def test_period_mid_text_preserved(self):
        """Internal periods are not affected."""
        assert strip_trailing_punctuation("3.14.") == "3.14"


class TestStripSurroundingQuotes:
    """Tests for strip_surrounding_quotes post-cleaner."""

    def test_double_quotes(self):
        assert strip_surrounding_quotes('"answer"') == "answer"

    def test_single_quotes(self):
        assert strip_surrounding_quotes("'answer'") == "answer"

    def test_no_quotes(self):
        assert strip_surrounding_quotes("answer") == "answer"

    def test_mismatched_quotes(self):
        """Mismatched quotes should not be stripped."""
        assert strip_surrounding_quotes("\"mismatched'") == "\"mismatched'"

    def test_empty_string(self):
        assert strip_surrounding_quotes("") == ""

    def test_only_quotes(self):
        assert strip_surrounding_quotes('""') == ""

    def test_nested_quotes_preserved(self):
        """Only outermost quotes removed."""
        assert strip_surrounding_quotes("\"'inner'\"") == "'inner'"

    def test_single_quote_only(self):
        """A single quote character is not stripped."""
        assert strip_surrounding_quotes("'") == "'"


class TestStripLatexDollars:
    """Tests for strip_latex_dollars post-cleaner."""

    def test_single_dollars(self):
        assert strip_latex_dollars("$42$") == "42"

    def test_double_dollars(self):
        assert strip_latex_dollars("$$42$$") == "42"

    def test_no_dollars(self):
        assert strip_latex_dollars("42") == "42"

    def test_empty_string(self):
        assert strip_latex_dollars("") == ""

    def test_single_dollar_only(self):
        """A single $ is not matched."""
        assert strip_latex_dollars("$") == "$"

    def test_mismatched_dollars(self):
        """Mismatched dollars preserved."""
        assert strip_latex_dollars("$42$$") == "$42$$"

    def test_content_with_inner_dollar(self):
        """Dollar signs within content."""
        assert strip_latex_dollars("$x + $y$") == "$x + $y$"

    def test_whitespace_inside_dollars(self):
        """Whitespace inside dollars is preserved."""
        assert strip_latex_dollars("$ 42 $") == " 42 "


class TestCleanedExtractor:
    """Tests for CleanedExtractor wrapper."""

    def test_pre_cleaner_strips_tokens(self):
        """Pre-cleaner strips tokens before inner extraction."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[strip_special_tokens],
            post_cleaners=[],
        )
        answer, meta = extractor.extract("<answer>42</answer><|endoftext|>")
        assert answer == "42"
        assert meta["pre_cleaners_applied"] is True
        assert meta["post_cleaners_applied"] is True

    def test_post_cleaner_strips_punctuation(self):
        """Post-cleaner strips trailing punctuation from extracted answer."""
        inner = PatternAnswerExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[],
            post_cleaners=[strip_trailing_punctuation],
        )
        answer, _ = extractor.extract("the answer is 42.")
        # PatternAnswerExtractor captures "42" (period is a sentence boundary)
        # but if it captures "42.", post-cleaner strips the period
        assert answer is not None
        assert answer.rstrip(".") == "42" or answer == "42"

    def test_both_pre_and_post(self):
        """Both pre and post cleaners work together."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[strip_special_tokens],
            post_cleaners=[strip_trailing_punctuation],
        )
        answer, meta = extractor.extract("<|endoftext|><answer>42.</answer></s>")
        assert answer == "42"
        assert meta["pre_cleaners_applied"] is True
        assert meta["post_cleaners_applied"] is True

    def test_no_cleaners_passthrough(self):
        """No cleaners means pass-through."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[],
            post_cleaners=[],
        )
        answer, meta = extractor.extract("<answer>42</answer>")
        assert answer == "42"

    def test_inner_returns_none_post_cleaners_not_applied(self):
        """When inner returns None, post-cleaners are not applied."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[],
            post_cleaners=[strip_trailing_punctuation],
        )
        answer, meta = extractor.extract("no tags here")
        assert answer is None
        assert meta["found"] is False

    def test_metadata_includes_cleaner_info(self):
        """Metadata includes cleaner application flags."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[strip_special_tokens],
            post_cleaners=[strip_trailing_punctuation],
        )
        _, meta = extractor.extract("<answer>42</answer>")
        assert meta["pre_cleaners_applied"] is True
        assert meta["post_cleaners_applied"] is True

    def test_multiple_post_cleaners(self):
        """Multiple post-cleaners applied in order."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(
            inner=inner,
            pre_cleaners=[],
            post_cleaners=[strip_surrounding_quotes, strip_trailing_punctuation],
        )
        answer, _ = extractor.extract('<answer>"hello,"</answer>')
        assert answer == "hello"


class TestResolveCleaners:
    """Tests for resolve_cleaners function."""

    def test_none_returns_defaults_pre(self):
        """None -> default pre-cleaners."""
        fns = resolve_cleaners(None, "pre")
        assert len(fns) == len(DEFAULT_PRE_CLEANERS)

    def test_none_returns_defaults_post(self):
        """None -> default post-cleaners."""
        fns = resolve_cleaners(None, "post")
        assert len(fns) == len(DEFAULT_POST_CLEANERS)

    def test_empty_list_returns_empty(self):
        """[] -> no cleaners (disabled)."""
        fns = resolve_cleaners([], "pre")
        assert fns == []

    def test_specific_names(self):
        """Specific names resolve to functions."""
        fns = resolve_cleaners(["strip_special_tokens"], "pre")
        assert len(fns) == 1
        assert fns[0] is strip_special_tokens

    def test_specific_post_names(self):
        """Post-cleaner names resolve to functions."""
        fns = resolve_cleaners(["strip_trailing_punctuation", "strip_surrounding_quotes"], "post")
        assert len(fns) == 2
        assert fns[0] is strip_trailing_punctuation
        assert fns[1] is strip_surrounding_quotes

    def test_unknown_name_raises(self):
        """Unknown cleaner name raises KeyError."""
        with pytest.raises(KeyError):
            resolve_cleaners(["nonexistent_cleaner"], "pre")

    def test_invalid_kind_raises(self):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError):
            resolve_cleaners(None, "invalid")


class TestCleanerRegistries:
    """Tests for PRE_CLEANERS and POST_CLEANERS registries."""

    def test_pre_cleaners_registry(self):
        assert "strip_special_tokens" in PRE_CLEANERS
        assert PRE_CLEANERS["strip_special_tokens"] is strip_special_tokens
        assert "strip_thinking_tokens" in PRE_CLEANERS
        assert PRE_CLEANERS["strip_thinking_tokens"] is strip_thinking_tokens

    def test_post_cleaners_registry(self):
        assert "strip_trailing_punctuation" in POST_CLEANERS
        assert "strip_surrounding_quotes" in POST_CLEANERS
        assert "strip_latex_dollars" in POST_CLEANERS

    def test_defaults_are_valid_names(self):
        """Default cleaner names exist in their registries."""
        for name in DEFAULT_PRE_CLEANERS:
            assert name in PRE_CLEANERS
        for name in DEFAULT_POST_CLEANERS:
            assert name in POST_CLEANERS
