"""Tests for answer extraction."""

import pytest
from llenvs.core.extraction import (
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
    CompositeExtractor,
    RawGenerationExtractor,
    BoxedExtractor,
    NumericExtractor,
    LastLineExtractor,
    CodeBlockExtractor,
    PatternAnswerExtractor,
    CleanedExtractor,
    NativeExtractor,
)
from llenvs.core.cleaning import strip_special_tokens


class TestTagBasedExtractor:
    """Tests for TagBasedExtractor."""

    def test_basic_extraction(self):
        """Test basic tag extraction."""
        extractor = TagBasedExtractor(tag_name="answer")
        answer, meta = extractor.extract("The result is <answer>42</answer>.")

        assert answer == "42"
        assert meta["found"] is True
        assert meta["tag_name"] == "answer"

    def test_custom_tag_name(self):
        """Test extraction with custom tag name."""
        extractor = TagBasedExtractor(tag_name="solution")
        answer, meta = extractor.extract("Here: <solution>correct</solution>")

        assert answer == "correct"
        assert meta["tag_name"] == "solution"

    def test_multiline_content(self):
        """Test extraction of multiline content."""
        extractor = TagBasedExtractor()
        text = """
        Let me think...
        <answer>
        Line 1
        Line 2
        </answer>
        """
        answer, meta = extractor.extract(text)

        assert "Line 1" in answer
        assert "Line 2" in answer

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped by default."""
        extractor = TagBasedExtractor()
        answer, _ = extractor.extract("<answer>  42  </answer>")
        assert answer == "42"

    def test_no_whitespace_stripping(self):
        """Test disabling whitespace stripping."""
        extractor = TagBasedExtractor(strip_whitespace=False)
        answer, _ = extractor.extract("<answer>  42  </answer>")
        assert answer == "  42  "

    def test_multiple_tags_uses_last(self):
        """Test that last tag is used when multiple present."""
        extractor = TagBasedExtractor()
        text = "First: <answer>wrong</answer> Second: <answer>correct</answer>"
        answer, meta = extractor.extract(text)

        assert answer == "correct"
        assert meta["num_matches"] == 2

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        extractor = TagBasedExtractor()
        answer, _ = extractor.extract("<ANSWER>42</ANSWER>")
        assert answer == "42"

        answer, _ = extractor.extract("<Answer>42</Answer>")
        assert answer == "42"

    def test_no_match(self):
        """Test when no tag is found."""
        extractor = TagBasedExtractor()
        answer, meta = extractor.extract("No tags here")

        assert answer is None
        assert meta["found"] is False

    def test_empty_tag(self):
        """Test extraction of empty tag content."""
        extractor = TagBasedExtractor()
        answer, meta = extractor.extract("<answer></answer>")

        assert answer == ""
        assert meta["found"] is True

    def test_special_characters_in_tag_name(self):
        """Test tag name with special regex characters."""
        extractor = TagBasedExtractor(tag_name="my.tag")
        answer, _ = extractor.extract("<my.tag>value</my.tag>")
        assert answer == "value"


class TestRegexExtractor:
    """Tests for RegexExtractor."""

    def test_basic_extraction(self):
        """Test basic regex extraction."""
        extractor = RegexExtractor(pattern=r"answer:\s*(\d+)")
        answer, meta = extractor.extract("The answer: 42")

        assert answer == "42"
        assert meta["found"] is True

    def test_custom_group_index(self):
        """Test extraction with different group index."""
        extractor = RegexExtractor(
            pattern=r"(\w+):\s*(\d+)",
            group_index=2,
        )
        answer, _ = extractor.extract("value: 123")
        assert answer == "123"

    def test_case_insensitive(self):
        """Test case insensitive regex."""
        import re

        extractor = RegexExtractor(
            pattern=r"ANSWER:\s*(\w+)",
            flags=re.IGNORECASE,
        )
        answer, _ = extractor.extract("answer: test")
        assert answer == "test"

    def test_multiple_matches_uses_last(self):
        """Test that last match is used."""
        extractor = RegexExtractor(pattern=r"num=(\d+)")
        answer, meta = extractor.extract("num=1, num=2, num=3")

        assert answer == "3"
        assert meta["num_matches"] == 3

    def test_no_match(self):
        """Test when pattern doesn't match."""
        extractor = RegexExtractor(pattern=r"xyz(\d+)")
        answer, meta = extractor.extract("no match here")

        assert answer is None
        assert meta["found"] is False

    def test_invalid_group_index(self):
        """Test with invalid group index."""
        extractor = RegexExtractor(pattern=r"(\d+)", group_index=5)
        answer, meta = extractor.extract("123")

        assert answer is None
        assert "error" in meta


class TestGSM8KExtractor:
    """Tests for GSM8KExtractor."""

    def test_basic_extraction(self):
        """Test basic GSM8K format."""
        extractor = GSM8KExtractor()
        answer, meta = extractor.extract("So the total is #### 42")

        assert answer == "42"
        assert meta["found"] is True
        assert meta["format"] == "gsm8k"

    def test_with_dollar_sign(self):
        """Test extraction with dollar sign."""
        extractor = GSM8KExtractor()
        answer, _ = extractor.extract("The cost is #### $150")
        assert answer == "150"

    def test_with_commas(self):
        """Test extraction with comma separators."""
        extractor = GSM8KExtractor()
        answer, _ = extractor.extract("Population: #### 1,234,567")
        assert answer == "1234567"  # Commas removed

    def test_decimal(self):
        """Test extraction of decimal numbers."""
        extractor = GSM8KExtractor()
        answer, _ = extractor.extract("Average: #### 3.14")
        assert answer == "3.14"

    def test_no_space_after_hashes(self):
        """Test extraction without space after ####."""
        extractor = GSM8KExtractor()
        answer, _ = extractor.extract("Result: ####42")
        assert answer == "42"

    def test_no_match(self):
        """Test when GSM8K format not found."""
        extractor = GSM8KExtractor()
        answer, meta = extractor.extract("The answer is 42")

        assert answer is None
        assert meta["found"] is False


class TestMultipleChoiceExtractor:
    """Tests for MultipleChoiceExtractor."""

    def test_answer_colon_format(self):
        """Test 'Answer: X' format."""
        extractor = MultipleChoiceExtractor()
        answer, _ = extractor.extract("I think Answer: B")
        assert answer == "B"

    def test_parentheses_format(self):
        """Test '(X)' format."""
        extractor = MultipleChoiceExtractor()
        answer, _ = extractor.extract("The correct choice is (C)")
        assert answer == "C"

    def test_choice_colon_format(self):
        """Test 'Choice: X' format."""
        extractor = MultipleChoiceExtractor()
        answer, _ = extractor.extract("My choice: A")
        assert answer == "A"

    def test_case_insensitive_by_default(self):
        """Test case insensitive matching."""
        extractor = MultipleChoiceExtractor()
        answer, _ = extractor.extract("answer: b")
        assert answer == "B"  # Normalized to uppercase

    def test_custom_choices(self):
        """Test custom choice set."""
        extractor = MultipleChoiceExtractor(choices="123456")
        answer, _ = extractor.extract("Answer: 3")
        assert answer == "3"

    def test_no_match(self):
        """Test when no choice found."""
        extractor = MultipleChoiceExtractor()
        answer, meta = extractor.extract("I don't know the answer")

        assert answer is None
        assert meta["found"] is False

    def test_invalid_choice_ignored(self):
        """Test that choices outside the set are ignored."""
        extractor = MultipleChoiceExtractor(choices="ABC")
        answer, _ = extractor.extract("Answer: D")
        # Should not match D since it's not in choices
        assert answer is None


class TestCompositeExtractor:
    """Tests for CompositeExtractor."""

    def test_first_match_wins(self):
        """Test that first successful extractor is used."""
        extractor = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                GSM8KExtractor(),
            ]
        )
        answer, meta = extractor.extract("<answer>tag_answer</answer> #### 42")

        assert answer == "tag_answer"
        assert meta["extractor_type"] == "TagBasedExtractor"
        assert meta["extractor_index"] == 0

    def test_fallback_to_second(self):
        """Test falling back to second extractor."""
        extractor = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                GSM8KExtractor(),
            ]
        )
        answer, meta = extractor.extract("The answer is #### 42")

        assert answer == "42"
        assert meta["extractor_type"] == "GSM8KExtractor"
        assert meta["extractor_index"] == 1

    def test_all_fail(self):
        """Test when all extractors fail."""
        extractor = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                GSM8KExtractor(),
            ]
        )
        answer, meta = extractor.extract("No structured answer here")

        assert answer is None
        assert meta["found"] is False
        assert "tried_extractors" in meta

    def test_empty_extractors(self):
        """Test with no extractors."""
        extractor = CompositeExtractor(extractors=[])
        answer, meta = extractor.extract("anything")

        assert answer is None


class TestRawGenerationExtractor:
    """Tests for RawGenerationExtractor."""

    def test_returns_full_response(self):
        """Test that full response is returned."""
        extractor = RawGenerationExtractor()
        text = "This is the full response"
        answer, meta = extractor.extract(text)

        assert answer == text
        assert meta["found"] is True
        assert meta["is_full_response"] is True

    def test_whitespace_stripping(self):
        """Test whitespace stripping."""
        extractor = RawGenerationExtractor()
        answer, _ = extractor.extract("  spaced  ")
        assert answer == "spaced"

    def test_no_whitespace_stripping(self):
        """Test disabling whitespace stripping."""
        extractor = RawGenerationExtractor(strip_whitespace=False)
        answer, _ = extractor.extract("  spaced  ")
        assert answer == "  spaced  "

    def test_in_composite(self):
        """Test as final fallback in composite."""
        extractor = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                RawGenerationExtractor(),
            ]
        )
        answer, meta = extractor.extract("No tags, just text")

        assert answer == "No tags, just text"
        assert meta["extractor_type"] == "RawGenerationExtractor"


# =========================================================================
# New extractor tests
# =========================================================================


class TestBoxedExtractor:
    """Tests for BoxedExtractor."""

    def test_basic_extraction(self):
        r"""Test basic \boxed{42} extraction."""
        extractor = BoxedExtractor()
        answer, meta = extractor.extract(r"The answer is \boxed{42}")

        assert answer == "42"
        assert meta["found"] is True

    def test_nested_braces(self):
        r"""Test nested braces: \boxed{x^{2}+1}."""
        extractor = BoxedExtractor()
        answer, _ = extractor.extract(r"Therefore \boxed{x^{2}+1}")
        assert answer == "x^{2}+1"

    def test_deeply_nested_braces(self):
        r"""Test deeply nested braces."""
        extractor = BoxedExtractor()
        answer, _ = extractor.extract(r"\boxed{\frac{a^{2}}{b^{3}}}")
        assert answer == r"\frac{a^{2}}{b^{3}}"

    def test_multiple_boxed_takes_last(self):
        r"""Test multiple \boxed{}, takes last."""
        extractor = BoxedExtractor()
        answer, meta = extractor.extract(
            r"First \boxed{wrong}, then \boxed{correct}"
        )
        assert answer == "correct"
        assert meta["num_matches"] == 2

    def test_no_match(self):
        """Test when no boxed found."""
        extractor = BoxedExtractor()
        answer, meta = extractor.extract("No boxed here")
        assert answer is None
        assert meta["found"] is False

    def test_space_before_brace(self):
        r"""Test \boxed {42} with space before brace."""
        extractor = BoxedExtractor()
        answer, _ = extractor.extract(r"\boxed {42}")
        assert answer == "42"

    def test_whitespace_inside_stripped(self):
        r"""Test whitespace inside \boxed{ 42 } is stripped."""
        extractor = BoxedExtractor()
        answer, _ = extractor.extract(r"\boxed{ 42 }")
        assert answer == "42"

    def test_empty_boxed(self):
        r"""Test \boxed{} returns empty string."""
        extractor = BoxedExtractor()
        answer, meta = extractor.extract(r"\boxed{}")
        assert answer == ""
        assert meta["found"] is True

    def test_unbalanced_braces_returns_none(self):
        r"""Test unbalanced braces return None."""
        extractor = BoxedExtractor()
        answer, _ = extractor.extract(r"\boxed{unclosed")
        assert answer is None


class TestNumericExtractor:
    """Tests for NumericExtractor."""

    def test_integer(self):
        """Test integer extraction."""
        extractor = NumericExtractor()
        answer, meta = extractor.extract("The answer is 42")
        assert answer == "42"
        assert meta["found"] is True

    def test_decimal_with_dot(self):
        """Test decimal with dot."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("Pi is approximately 3.14")
        assert answer == "3.14"

    def test_comma_separated_thousands(self):
        """Test comma-separated thousands: 1,234,567 -> 1234567."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("Population: 1,234,567")
        assert answer == "1234567"

    def test_negative(self):
        """Test negative numbers."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("Temperature: -42")
        assert answer == "-42"

    def test_multiple_numbers_takes_last(self):
        """Test multiple numbers, takes last."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("3 + 4 = 7")
        assert answer == "7"

    def test_no_numbers_returns_none(self):
        """Test no numbers returns None."""
        extractor = NumericExtractor()
        answer, meta = extractor.extract("No numbers here")
        assert answer is None
        assert meta["found"] is False

    def test_mixed_with_text(self):
        """Test extraction from mixed text."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("approximately 3.14159 units")
        assert answer == "3.14159"

    def test_negative_decimal(self):
        """Test negative decimal."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("The result is -3.14")
        assert answer == "-3.14"


class TestLastLineExtractor:
    """Tests for LastLineExtractor."""

    def test_single_line(self):
        """Test single line extraction."""
        extractor = LastLineExtractor()
        answer, meta = extractor.extract("Just one line")
        assert answer == "Just one line"
        assert meta["found"] is True

    def test_multiple_lines_takes_last_nonempty(self):
        """Test multiple lines, takes last non-empty."""
        extractor = LastLineExtractor()
        answer, _ = extractor.extract("Line 1\nLine 2\nLine 3")
        assert answer == "Line 3"

    def test_trailing_whitespace_and_newlines(self):
        """Test trailing whitespace/newlines stripped."""
        extractor = LastLineExtractor()
        answer, _ = extractor.extract("Line 1\nLine 2\n\n  \n")
        assert answer == "Line 2"

    def test_all_empty_lines_returns_none(self):
        """Test all empty lines returns None."""
        extractor = LastLineExtractor()
        answer, meta = extractor.extract("\n\n  \n  \n")
        assert answer is None
        assert meta["found"] is False

    def test_empty_string(self):
        """Test empty string returns None."""
        extractor = LastLineExtractor()
        answer, meta = extractor.extract("")
        assert answer is None
        assert meta["found"] is False


class TestCodeBlockExtractor:
    """Tests for CodeBlockExtractor."""

    def test_basic_fence(self):
        """Test basic code fence extraction."""
        extractor = CodeBlockExtractor()
        answer, meta = extractor.extract("Here:\n```\ncode here\n```")
        assert answer == "code here"
        assert meta["found"] is True

    def test_with_language_tag(self):
        """Test extraction with language tag."""
        extractor = CodeBlockExtractor()
        answer, _ = extractor.extract("```python\nprint('hello')\n```")
        assert answer == "print('hello')"

    def test_language_filter(self):
        """Test language filter: only extract python blocks."""
        extractor = CodeBlockExtractor(language="python")
        text = "```javascript\nconsole.log('hi')\n```\n\n```python\nprint('hello')\n```"
        answer, meta = extractor.extract(text)
        assert answer == "print('hello')"
        assert meta["language"] == "python"

    def test_language_filter_no_match(self):
        """Test language filter when no matching block exists."""
        extractor = CodeBlockExtractor(language="python")
        answer, meta = extractor.extract("```javascript\nconsole.log('hi')\n```")
        assert answer is None
        assert meta["found"] is False

    def test_multiple_fences_takes_last(self):
        """Test multiple fences, takes last."""
        extractor = CodeBlockExtractor()
        text = "```\nfirst\n```\n\n```\nsecond\n```"
        answer, meta = extractor.extract(text)
        assert answer == "second"
        assert meta["num_matches"] == 2

    def test_no_fence_returns_none(self):
        """Test no fence returns None."""
        extractor = CodeBlockExtractor()
        answer, meta = extractor.extract("No code blocks here")
        assert answer is None
        assert meta["found"] is False

    def test_multiline_code(self):
        """Test multiline code in fence."""
        extractor = CodeBlockExtractor()
        text = "```python\ndef foo():\n    return 42\n```"
        answer, _ = extractor.extract(text)
        assert answer == "def foo():\n    return 42"


class TestPatternAnswerExtractor:
    """Tests for PatternAnswerExtractor."""

    def test_the_answer_is(self):
        """Test 'the answer is X' pattern."""
        extractor = PatternAnswerExtractor()
        answer, meta = extractor.extract("So the answer is 42")
        assert answer == "42"
        assert meta["found"] is True

    def test_therefore(self):
        """Test 'therefore, X' pattern."""
        extractor = PatternAnswerExtractor()
        answer, _ = extractor.extract("Therefore, 42.")
        assert answer is not None
        assert "42" in answer

    def test_equals(self):
        """Test '= X' pattern."""
        extractor = PatternAnswerExtractor()
        answer, _ = extractor.extract("x = 7")
        assert answer is not None
        assert "7" in answer

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        extractor = PatternAnswerExtractor()
        answer, _ = extractor.extract("The Answer Is 42")
        assert answer == "42"

    def test_multiple_patterns_takes_last(self):
        """Test multiple matches, takes last."""
        extractor = PatternAnswerExtractor()
        answer, _ = extractor.extract(
            "The answer is 10. But actually the answer is 42"
        )
        assert answer == "42"

    def test_no_match_returns_none(self):
        """Test no match returns None."""
        extractor = PatternAnswerExtractor()
        answer, meta = extractor.extract("No patterns here at all")
        assert answer is None
        assert meta["found"] is False

    def test_custom_patterns(self):
        """Test custom pattern list."""
        extractor = PatternAnswerExtractor(patterns=[r"result:\s*"])
        answer, _ = extractor.extract("The result: 42")
        assert answer == "42"


class TestCleanedExtractorInExtraction:
    """Tests for CleanedExtractor (token stripping via cleaning layer)."""

    def test_strips_endoftext(self):
        """Test stripping <|endoftext|> before inner extraction."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract(
            "<answer>42</answer><|endoftext|><|endoftext|>"
        )
        assert answer == "42"

    def test_strips_pad_tokens(self):
        """Test stripping <pad> tokens."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract("<pad><pad><answer>42</answer><pad>")
        assert answer == "42"

    def test_strips_eos_token(self):
        """Test stripping </s> tokens."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract("<answer>42</answer></s>")
        assert answer == "42"

    def test_strips_im_end(self):
        """Test stripping <|im_end|> tokens."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract("<answer>42</answer><|im_end|>")
        assert answer == "42"

    def test_multiple_token_types_stripped(self):
        """Test multiple token types stripped at once."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract(
            "<|endoftext|><pad><answer>42</answer></s><|im_end|>"
        )
        assert answer == "42"

    def test_no_special_tokens_works_fine(self):
        """Test works correctly when no special tokens present."""
        inner = TagBasedExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract("<answer>42</answer>")
        assert answer == "42"

    def test_wraps_any_inner_extractor(self):
        """Test wrapping a different inner extractor."""
        inner = GSM8KExtractor()
        extractor = CleanedExtractor(inner=inner, pre_cleaners=[strip_special_tokens])
        answer, _ = extractor.extract("#### 42<|endoftext|>")
        assert answer == "42"


class TestNativeExtractor:
    """Tests for NativeExtractor."""

    def test_basic_usage(self):
        """Test wrapping a simple function."""

        def my_extract(text):
            return text.upper()

        extractor = NativeExtractor(fn=my_extract, name="test")
        answer, meta = extractor.extract("hello")
        assert answer == "HELLO"
        assert meta["found"] is True
        assert meta["native_name"] == "test"

    def test_returns_none(self):
        """Test when native function returns None."""

        def my_extract(text):
            return None

        extractor = NativeExtractor(fn=my_extract, name="test")
        answer, meta = extractor.extract("hello")
        assert answer is None
        assert meta["found"] is False

    def test_returns_empty_string(self):
        """Test when native function returns empty string."""

        def my_extract(text):
            return ""

        extractor = NativeExtractor(fn=my_extract, name="test")
        answer, meta = extractor.extract("hello")
        # Empty string is a valid extraction
        assert answer == ""
        assert meta["found"] is True


# =========================================================================
# Robustness / edge case tests
# =========================================================================


class TestExtractionRobustness:
    """Edge case and robustness tests."""

    def test_padding_tokens_with_tag(self):
        """Response with padding tokens interspersed."""
        extractor = CleanedExtractor(
            inner=TagBasedExtractor(), pre_cleaners=[strip_special_tokens]
        )
        answer, _ = extractor.extract(
            "<answer>42</answer><|endoftext|><|endoftext|>"
        )
        assert answer == "42"

    def test_empty_response(self):
        """Empty response returns None for all extractors."""
        for ext in [
            BoxedExtractor(),
            NumericExtractor(),
            LastLineExtractor(),
            CodeBlockExtractor(),
            PatternAnswerExtractor(),
        ]:
            answer, _ = ext.extract("")
            assert answer is None, f"{type(ext).__name__} should return None for empty"

    def test_unicode_content(self):
        """Unicode content in answers."""
        extractor = TagBasedExtractor()
        answer, _ = extractor.extract("<answer>π ≈ 3.14159</answer>")
        assert answer == "π ≈ 3.14159"

    def test_newlines_inside_tags(self):
        """Newlines inside tags."""
        extractor = TagBasedExtractor()
        answer, _ = extractor.extract("<answer>\n42\n</answer>")
        assert answer == "42"

    def test_boxed_in_long_response(self):
        """Boxed extraction in a long response."""
        prefix = "Let me think step by step.\n" * 50
        extractor = BoxedExtractor()
        answer, _ = extractor.extract(prefix + r"\boxed{42}")
        assert answer == "42"

    def test_numeric_with_trailing_period(self):
        """Numeric extraction doesn't include trailing period as part of number."""
        extractor = NumericExtractor()
        answer, _ = extractor.extract("The answer is 42.")
        assert answer == "42"


# =========================================================================
# Compositional pipeline tests
# =========================================================================


class TestCompositeExtractorChains:
    """Tests for extractor chains (compositional pipelines)."""

    def test_tag_succeeds_others_skipped(self):
        """Chain: tag succeeds, pattern and numeric skipped."""
        chain = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                PatternAnswerExtractor(),
                NumericExtractor(),
                RawGenerationExtractor(),
            ]
        )
        answer, meta = chain.extract(
            "<answer>42</answer> the answer is 99. 100"
        )
        assert answer == "42"
        assert meta["extractor_type"] == "TagBasedExtractor"

    def test_tag_fails_pattern_fails_numeric_succeeds(self):
        """Chain: tag fails, pattern fails, numeric succeeds."""
        chain = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                PatternAnswerExtractor(),
                NumericExtractor(),
            ]
        )
        answer, meta = chain.extract("The result: 42")
        assert answer == "42"
        assert meta["extractor_type"] == "NumericExtractor"

    def test_all_fail_returns_none(self):
        """Chain: all fail, returns None (no fallback)."""
        chain = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                PatternAnswerExtractor(),
            ]
        )
        answer, meta = chain.extract("No structured answer here at all")
        assert answer is None
        assert meta["found"] is False

    def test_metadata_tracks_which_succeeded(self):
        """Metadata tracks which extractor succeeded."""
        chain = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                BoxedExtractor(),
                NumericExtractor(),
            ]
        )
        answer, meta = chain.extract(r"Therefore \boxed{7}")
        assert answer == "7"
        assert meta["extractor_type"] == "BoxedExtractor"
        assert meta["extractor_index"] == 1

    def test_cleaned_extractor_in_chain(self):
        """CleanedExtractor wrapping a chain member."""
        chain = CompositeExtractor(
            extractors=[
                CleanedExtractor(
                    inner=TagBasedExtractor(),
                    pre_cleaners=[strip_special_tokens],
                ),
                NumericExtractor(),
            ]
        )
        answer, _ = chain.extract("<answer>42</answer><|endoftext|>")
        assert answer == "42"
