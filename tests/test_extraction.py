"""Tests for answer extraction."""

import pytest
from llenvs.core.extraction import (
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
    CompositeExtractor,
    FallbackExtractor,
)


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


class TestFallbackExtractor:
    """Tests for FallbackExtractor."""

    def test_returns_full_response(self):
        """Test that full response is returned."""
        extractor = FallbackExtractor()
        text = "This is the full response"
        answer, meta = extractor.extract(text)

        assert answer == text
        assert meta["found"] is True
        assert meta["is_full_response"] is True

    def test_whitespace_stripping(self):
        """Test whitespace stripping."""
        extractor = FallbackExtractor()
        answer, _ = extractor.extract("  spaced  ")
        assert answer == "spaced"

    def test_no_whitespace_stripping(self):
        """Test disabling whitespace stripping."""
        extractor = FallbackExtractor(strip_whitespace=False)
        answer, _ = extractor.extract("  spaced  ")
        assert answer == "  spaced  "

    def test_in_composite(self):
        """Test as final fallback in composite."""
        extractor = CompositeExtractor(
            extractors=[
                TagBasedExtractor(),
                FallbackExtractor(),
            ]
        )
        answer, meta = extractor.extract("No tags, just text")

        assert answer == "No tags, just text"
        assert meta["extractor_type"] == "FallbackExtractor"
