"""Tests for text segmentation and segmented environments."""

import json

import pytest
from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import Signal, SignalBundle, RewardType
from llenvs.core.segmentation import (
    Segmenter,
    SentenceSegmenter,
    LineSegmenter,
    PatternSegmenter,
    CompositeSegmenter,
    LLMSegmenter,
    TokenSegmenter,
    default_segment_parser,
)
from llenvs.core.segmented_environment import (
    SegmentedEnvironment,
    SegmentedHidden,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
)


class TestSentenceSegmenter:
    """Tests for SentenceSegmenter."""

    def test_basic_sentences(self):
        """Test splitting basic sentences."""
        segmenter = SentenceSegmenter()
        text = "First sentence. Second sentence. Third sentence."

        segments = segmenter.segment(text)

        assert len(segments) == 3
        assert segments[0] == "First sentence."
        assert segments[1] == "Second sentence."
        assert segments[2] == "Third sentence."

    def test_question_and_exclamation(self):
        """Test splitting on ? and !"""
        segmenter = SentenceSegmenter()
        text = "What is this? It's amazing! Yes, it is."

        segments = segmenter.segment(text)

        assert len(segments) == 3
        assert segments[0] == "What is this?"
        assert segments[1] == "It's amazing!"
        assert segments[2] == "Yes, it is."

    def test_abbreviations(self):
        """Test that common abbreviations don't cause splits."""
        segmenter = SentenceSegmenter()
        text = "Dr. Smith went to the store. Mr. Jones stayed home."

        segments = segmenter.segment(text)

        assert len(segments) == 2
        assert "Dr. Smith" in segments[0]
        assert "Mr. Jones" in segments[1]

    def test_empty_text(self):
        """Test empty text returns empty list."""
        segmenter = SentenceSegmenter()

        assert segmenter.segment("") == []
        assert segmenter.segment("   ") == []

    def test_single_sentence(self):
        """Test single sentence without trailing punctuation."""
        segmenter = SentenceSegmenter()
        text = "This is a single sentence"

        segments = segmenter.segment(text)

        assert len(segments) == 1
        assert segments[0] == "This is a single sentence"

    def test_find_boundary_basic(self):
        """Test find_boundary finds first sentence end."""
        segmenter = SentenceSegmenter()
        text = "First sentence. Second sentence."

        boundary = segmenter.find_boundary(text)

        assert boundary == 16  # After ". "

    def test_find_boundary_no_boundary(self):
        """Test find_boundary returns None when no boundary."""
        segmenter = SentenceSegmenter()
        text = "No boundary here"

        assert segmenter.find_boundary(text) is None

    def test_find_boundary_abbreviation(self):
        """Test find_boundary skips abbreviations."""
        segmenter = SentenceSegmenter()
        text = "Dr. Smith is here. Next sentence."

        boundary = segmenter.find_boundary(text)

        # Should find boundary after "here." not after "Dr."
        assert boundary > 4


class TestLineSegmenter:
    """Tests for LineSegmenter."""

    def test_single_newline(self):
        """Test splitting on single newlines."""
        segmenter = LineSegmenter(delimiter="\n")
        text = "Line one\nLine two\nLine three"

        segments = segmenter.segment(text)

        assert len(segments) == 3
        assert segments[0] == "Line one"
        assert segments[1] == "Line two"
        assert segments[2] == "Line three"

    def test_double_newline(self):
        """Test splitting on double newlines (paragraphs)."""
        segmenter = LineSegmenter(delimiter="\n\n")
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        segments = segmenter.segment(text)

        assert len(segments) == 3

    def test_empty_lines_excluded(self):
        """Test that empty lines are excluded."""
        segmenter = LineSegmenter()
        text = "Line one\n\n\nLine two"

        segments = segmenter.segment(text)

        assert len(segments) == 2

    def test_find_boundary(self):
        """Test find_boundary finds first newline."""
        segmenter = LineSegmenter()
        text = "First line\nSecond line"

        boundary = segmenter.find_boundary(text)

        assert boundary == 11  # After "First line\n"


class TestPatternSegmenter:
    """Tests for PatternSegmenter."""

    def test_numbered_steps(self):
        """Test splitting on numbered steps."""
        segmenter = PatternSegmenter()
        # Use cleaner text that separates numbered steps clearly
        text = "Introduction. 1. First part. 2. Second part. 3. Third part."

        segments = segmenter.segment(text)

        assert len(segments) == 4
        assert segments[0] == "Introduction."
        assert segments[1].startswith("1.")
        assert segments[2].startswith("2.")
        assert segments[3].startswith("3.")

    def test_step_prefix(self):
        """Test splitting on 'Step N:' format."""
        segmenter = PatternSegmenter()
        text = "Step 1: Do this Step 2: Do that Step 3: Done"

        segments = segmenter.segment(text)

        assert len(segments) == 3

    def test_transition_words(self):
        """Test splitting on transition words."""
        segmenter = PatternSegmenter()
        # Use clear text without numbers that could match patterns
        text = "We computed the sum. Therefore, the answer is eight."

        segments = segmenter.segment(text)

        assert len(segments) == 2
        assert "Therefore" in segments[1]

    def test_ordinal_words(self):
        """Test splitting on First, Second, etc."""
        segmenter = PatternSegmenter()
        text = "First, we add the numbers. Second, we subtract. Finally, we multiply."

        segments = segmenter.segment(text)

        assert len(segments) == 3

    def test_no_patterns(self):
        """Test text with no patterns returns single segment."""
        segmenter = PatternSegmenter()
        text = "Just some regular text without any step markers."

        segments = segmenter.segment(text)

        assert len(segments) == 1

    def test_find_boundary(self):
        """Test find_boundary finds first pattern."""
        segmenter = PatternSegmenter()
        text = "Introduction 1. First step 2. Second step"

        boundary = segmenter.find_boundary(text)

        assert boundary == 13  # Before "1. First"


class TestCompositeSegmenter:
    """Tests for CompositeSegmenter."""

    def test_combine_sentence_and_line(self):
        """Test combining sentence and line segmenters."""
        segmenter = CompositeSegmenter(segmenters=(LineSegmenter(), SentenceSegmenter()))
        text = "Line one. Line one continued.\nLine two. Line two end."

        segments = segmenter.segment(text)

        assert len(segments) == 4

    def test_find_boundary_earliest(self):
        """Test find_boundary returns earliest boundary."""
        segmenter = CompositeSegmenter(segmenters=(LineSegmenter(), SentenceSegmenter()))
        text = "Short.\nLonger sentence here."

        boundary = segmenter.find_boundary(text)

        # Sentence boundary (7) comes before newline boundary
        assert boundary == 7


class MockTokenizer:
    """Character-level tokenizer for testing: one token per character."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, tokens: list[int]) -> str:
        # Since tokens are just indices 0..N-1, length of tokens = length of text
        # We need the original text to decode, but this mock just returns
        # a string of the right length. Tests pass the original text through segment().
        return "x" * len(tokens)


class _CharTokenizer:
    """Character-level tokenizer that preserves text through encode/decode."""

    def __init__(self) -> None:
        self._text: str = ""

    def encode(self, text: str) -> list[int]:
        self._text = text
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        # Decode prefix: use stored text sliced to token length
        return self._text[: len(tokens)]


class TestTokenSegmenter:
    """Tests for TokenSegmenter."""

    def test_basic_segmentation(self):
        """Test splitting into expected chunks."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=4)
        text = "abcdefghij"  # 10 chars -> chunks of 4, 4, 2

        segments = segmenter.segment(text)

        assert len(segments) == 3
        assert segments[0] == "abcd"
        assert segments[1] == "efgh"
        assert segments[2] == "ij"

    def test_exact_reconstruction(self):
        """Test that joining segments exactly reconstructs original text."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=3)
        text = "Hello, world! This is a test."

        segments = segmenter.segment(text)

        assert "".join(segments) == text

    def test_text_shorter_than_token_size(self):
        """Test text shorter than token_size returns single segment."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=100)
        text = "short"

        segments = segmenter.segment(text)

        assert segments == ["short"]

    def test_text_exactly_token_size(self):
        """Test text exactly token_size returns single segment."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=5)
        text = "exact"

        segments = segmenter.segment(text)

        assert segments == ["exact"]

    def test_empty_text(self):
        """Test empty text returns empty list."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=4)

        assert segmenter.segment("") == []

    def test_find_boundary_basic(self):
        """Test find_boundary returns correct character index."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=4)
        text = "abcdefgh"  # 8 chars, boundary at char 4

        boundary = segmenter.find_boundary(text)

        assert boundary == 4

    def test_find_boundary_no_boundary(self):
        """Test find_boundary returns None when text fits in one chunk."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=100)
        text = "short text"

        assert segmenter.find_boundary(text) is None

    def test_find_boundary_empty(self):
        """Test find_boundary returns None for empty text."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=4)

        assert segmenter.find_boundary("") is None

    def test_protocol_compliance(self):
        """Test TokenSegmenter implements Segmenter protocol."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=4)

        assert isinstance(segmenter, Segmenter)

    def test_various_chunk_sizes_one(self):
        """Test token_size=1 produces one segment per character."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=1)
        text = "abcd"

        segments = segmenter.segment(text)

        assert len(segments) == 4
        assert segments == ["a", "b", "c", "d"]
        assert "".join(segments) == text

    def test_various_chunk_sizes_large(self):
        """Test token_size=8 with short text."""
        tokenizer = _CharTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=8)
        text = "abcdefghijklmnop"  # 16 chars -> 2 chunks of 8

        segments = segmenter.segment(text)

        assert len(segments) == 2
        assert segments[0] == "abcdefgh"
        assert segments[1] == "ijklmnop"
        assert "".join(segments) == text


class TestSegmentedHidden:
    """Tests for SegmentedHidden."""

    def test_creation(self):
        """Test creating segmented hidden state."""
        base_hidden = {"expected_answer": "42"}
        hidden = SegmentedHidden(
            base_hidden=base_hidden,
            accumulated_text="Step 1.",
            segment_index=1,
            segments=("Step 1.",),
            total_segments=3,
        )

        assert hidden.base_hidden == base_hidden
        assert hidden.accumulated_text == "Step 1."
        assert hidden.segment_index == 1
        assert len(hidden.segments) == 1
        assert hidden.total_segments == 3

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = SegmentedHidden(base_hidden={})
        with pytest.raises(AttributeError):
            hidden.accumulated_text = "new text"  # type: ignore


class TestSegmentedEnvironment:
    """Tests for SegmentedEnvironment."""

    def test_creation(self, mock_dataset):
        """Test creating segmented environment."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        segmenter = SentenceSegmenter()

        env = SegmentedEnvironment(base_env, segmenter)

        assert env.spec.is_multi_turn is True
        assert "segmented" in env.spec.name
        assert env.segmenter is segmenter
        assert env.base_env is base_env

    def test_spec_properties(self, mock_dataset):
        """Test segmented environment spec."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        spec = env.spec
        assert spec.is_multi_turn is True
        assert spec.max_steps is None  # Variable based on response
        assert "base_environment" in spec.metadata

    def test_reset(self, mock_dataset):
        """Test environment reset."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.hidden, SegmentedHidden)
        assert state.hidden.accumulated_text == ""
        assert state.hidden.segment_index == 0
        assert state.hidden.segments == ()
        assert state.metadata.is_terminal is False

    def test_step_intermediate(self, mock_dataset):
        """Test intermediate step (not final segment)."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        # Set total_segments to indicate we're in replay mode with more segments coming
        state = State(
            observation=state.observation,
            hidden=SegmentedHidden(
                base_hidden=state.hidden.base_hidden,
                accumulated_text="",
                segment_index=0,
                segments=(),
                total_segments=3,  # 3 segments total, so first 2 are intermediate
            ),
            metadata=state.metadata,
        )

        result = env.step(state, Action(text="First segment. "))

        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is False
        assert result.next_state.hidden.segment_index == 1
        assert result.next_state.hidden.accumulated_text == "First segment. "
        assert result.info["is_intermediate"] is True

    def test_step_final(self, mock_dataset):
        """Test final step triggers underlying env."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        # Set up state for final segment (segment_index=1 means we've done 1 segment,
        # and total_segments=2 means this next step will be the final one)
        state = State(
            observation=state.observation,
            hidden=SegmentedHidden(
                base_hidden=state.hidden.base_hidden,
                accumulated_text="<answer>",
                segment_index=1,  # Already processed 1 segment
                segments=("<answer>",),
                total_segments=2,  # 2 total, so next step (index becomes 2) finalizes
            ),
            metadata=state.metadata,
        )

        result = env.step(state, Action(text="4</answer>"))

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True
        assert result.info["is_intermediate"] is False

    def test_finalize(self, mock_dataset):
        """Test explicit finalization."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        # Add some accumulated text
        state = State(
            observation=state.observation,
            hidden=SegmentedHidden(
                base_hidden=state.hidden.base_hidden,
                accumulated_text="<answer>4</answer>",
                segment_index=1,
                segments=("<answer>4</answer>",),
                total_segments=None,  # Unknown in generation mode
            ),
            metadata=state.metadata,
        )

        result = env.finalize(state)

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_replay_full_response(self, mock_dataset):
        """Test replay with a full response."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        full_response = "First, I need to add. The result is 4. <answer>4</answer>"
        results = env.replay(state, full_response)

        # Should have multiple steps
        assert len(results) >= 2

        # Last result should be terminal
        assert results[-1].next_state.metadata.is_terminal is True
        assert results[-1].terminated is True

        # Intermediate results should not be terminal
        for result in results[:-1]:
            assert result.terminated is False

    def test_replay_correct_answer(self, mock_dataset):
        """Test replay gives correct reward for right answer."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        # Task 0 expects answer "4"
        state, _ = env.reset(options={"task_index": 0})

        full_response = "Let me calculate. The answer is <answer>4</answer>"
        results = env.replay(state, full_response)

        # Final result should have correctness reward
        final_result = results[-1]
        correctness = final_result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.reward == 1.0

    def test_replay_incorrect_answer(self, mock_dataset):
        """Test replay gives zero reward for wrong answer."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        full_response = "Let me calculate. The answer is <answer>5</answer>"
        results = env.replay(state, full_response)

        final_result = results[-1]
        correctness = final_result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.reward == 0.0

    def test_replay_empty_response(self, mock_dataset):
        """Test replay handles empty response."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        results = env.replay(state, "")

        assert len(results) == 1
        assert results[0].terminated is True

    def test_replay_single_segment(self, mock_dataset):
        """Test replay with response that doesn't split."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        # No sentence boundary
        full_response = "<answer>4</answer>"
        results = env.replay(state, full_response)

        assert len(results) == 1
        assert results[0].terminated is True

    def test_len(self, mock_dataset):
        """Test __len__ delegates to underlying env."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        assert len(env) == len(mock_dataset)

    def test_replay_with_pattern_segmenter(self, mock_dataset):
        """Test replay with PatternSegmenter for numbered steps."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, PatternSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        # Use clear step markers without embedded numbers that could confuse the pattern
        full_response = "Step 1: Add the numbers Step 2: Get result Step 3: <answer>4</answer>"
        results = env.replay(state, full_response)

        assert len(results) == 3

    def test_state_immutability(self, mock_dataset):
        """Test that step doesn't mutate input state."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, SentenceSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        # Set up state with total_segments
        state = State(
            observation=state.observation,
            hidden=SegmentedHidden(
                base_hidden=state.hidden.base_hidden,
                accumulated_text="",
                segment_index=0,
                segments=(),
                total_segments=2,
            ),
            metadata=state.metadata,
        )

        original_accumulated = state.hidden.accumulated_text
        original_index = state.hidden.segment_index

        env.step(state, Action(text="Segment one. "))

        # Original state should be unchanged
        assert state.hidden.accumulated_text == original_accumulated
        assert state.hidden.segment_index == original_index


class TestSegmentedEnvironmentWithLineSegmenter:
    """Tests for SegmentedEnvironment with LineSegmenter."""

    def test_replay_line_by_line(self, mock_dataset):
        """Test replay segments by lines."""
        from llenvs.adapters.reasoning_gym import ReasoningGymEnvironment

        base_env = ReasoningGymEnvironment(dataset=mock_dataset)
        env = SegmentedEnvironment(base_env, LineSegmenter())

        state, _ = env.reset(options={"task_index": 0})

        full_response = "Line 1: Think about it\nLine 2: Calculate\nLine 3: <answer>4</answer>"
        results = env.replay(state, full_response)

        assert len(results) == 3


class TestSegmenterProtocol:
    """Tests for Segmenter protocol compliance."""

    def test_sentence_segmenter_is_segmenter(self):
        """Test SentenceSegmenter implements Segmenter."""
        segmenter = SentenceSegmenter()
        assert isinstance(segmenter, Segmenter)

    def test_line_segmenter_is_segmenter(self):
        """Test LineSegmenter implements Segmenter."""
        segmenter = LineSegmenter()
        assert isinstance(segmenter, Segmenter)

    def test_pattern_segmenter_is_segmenter(self):
        """Test PatternSegmenter implements Segmenter."""
        segmenter = PatternSegmenter()
        assert isinstance(segmenter, Segmenter)

    def test_composite_segmenter_is_segmenter(self):
        """Test CompositeSegmenter implements Segmenter."""
        segmenter = CompositeSegmenter(segmenters=(SentenceSegmenter(),))
        assert isinstance(segmenter, Segmenter)

    def test_llm_segmenter_is_segmenter(self):
        """Test LLMSegmenter implements Segmenter."""
        backend = _MockLLMBackend(response='["text"]')
        segmenter = LLMSegmenter(backend=backend)
        assert isinstance(segmenter, Segmenter)

    def test_token_segmenter_is_segmenter(self):
        """Test TokenSegmenter implements Segmenter."""
        segmenter = TokenSegmenter(tokenizer=_CharTokenizer(), token_size=4)
        assert isinstance(segmenter, Segmenter)


# ---------------------------------------------------------------------------
# Mock backend for LLMSegmenter tests
# ---------------------------------------------------------------------------


class _MockLLMBackend(ModelBackend):
    """Minimal ModelBackend that returns a canned response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[ChatMessage] | None = None
        self.last_params: SamplingParams | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True)

    @property
    def model_name(self) -> str:
        return "mock"

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        return [GenerationResult(text=self._response) for _ in prompts]

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        self.last_messages = messages
        self.last_params = params
        return GenerationResult(text=self._response)


# ---------------------------------------------------------------------------
# TestLLMSegmenter
# ---------------------------------------------------------------------------


class TestLLMSegmenter:
    """Tests for LLMSegmenter."""

    def test_basic_segmentation(self):
        """Mock backend returns valid JSON segments, verify correct mapping."""
        original = "First, I add the numbers. Then I get the result."
        llm_response = json.dumps(["First, I add the numbers.", "Then I get the result."])
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend)

        segments = segmenter.segment(original)

        assert len(segments) == 2
        assert "First, I add the numbers." in segments[0]
        assert "Then I get the result." in segments[1]
        assert "".join(segments) == original

    def test_empty_text(self):
        """Empty text returns []."""
        backend = _MockLLMBackend(response="[]")
        segmenter = LLMSegmenter(backend=backend)

        assert segmenter.segment("") == []

    def test_whitespace_tolerance(self):
        """LLM segments have trimmed whitespace, greedy match still works."""
        original = "  Hello world.  Goodbye world.  "
        # LLM trims, but greedy match should recover positions in original
        llm_response = json.dumps(["Hello world.", "Goodbye world."])
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend)

        segments = segmenter.segment(original)

        assert len(segments) == 2
        assert "Hello world." in segments[0]
        assert "Goodbye world." in segments[1]
        # Segments should reconstruct back to original
        assert "".join(segments) == original

    def test_malformed_json_fallback(self):
        """Invalid JSON returns [original_text]."""
        original = "Some text here."
        backend = _MockLLMBackend(response="not json at all")
        segmenter = LLMSegmenter(backend=backend)

        segments = segmenter.segment(original)

        assert segments == [original]

    def test_markdown_code_block_stripping(self):
        """JSON wrapped in triple backticks still parses."""
        original = "Step one. Step two."
        llm_response = '```json\n["Step one.", "Step two."]\n```'
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend)

        segments = segmenter.segment(original)

        assert len(segments) == 2
        assert "".join(segments) == original

    def test_json_embedded_in_text(self):
        """Parser finds [...] within surrounding prose."""
        original = "Alpha. Beta."
        llm_response = 'Here are the segments:\n["Alpha.", "Beta."]\nDone.'
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend)

        segments = segmenter.segment(original)

        assert len(segments) == 2
        assert "".join(segments) == original

    def test_custom_prompt_template(self):
        """Custom template with {raw_generation} is used."""
        original = "Hello."
        llm_response = json.dumps(["Hello."])
        backend = _MockLLMBackend(response=llm_response)
        custom_template = "CUSTOM: {raw_generation}"
        segmenter = LLMSegmenter(backend=backend, prompt_template=custom_template)

        segmenter.segment(original)

        # Verify the backend received the formatted custom template
        assert backend.last_messages is not None
        user_msgs = [m for m in backend.last_messages if m.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "CUSTOM: Hello."

    def test_custom_parser(self):
        """Custom parser callable is invoked with (original, llm_response)."""
        original = "Some text."
        llm_response = "anything"
        backend = _MockLLMBackend(response=llm_response)

        calls: list[tuple[str, str]] = []

        def my_parser(orig: str, resp: str) -> list[str]:
            calls.append((orig, resp))
            return [orig]

        segmenter = LLMSegmenter(backend=backend, parser=my_parser)
        segments = segmenter.segment(original)

        assert calls == [(original, llm_response)]
        assert segments == [original]

    def test_custom_system_prompt(self):
        """Verify system message in chat messages."""
        original = "Text."
        llm_response = json.dumps(["Text."])
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend, system_prompt="Custom system prompt")

        segmenter.segment(original)

        assert backend.last_messages is not None
        system_msgs = [m for m in backend.last_messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Custom system prompt"

    def test_no_system_prompt(self):
        """system_prompt=None, only user message sent."""
        original = "Text."
        llm_response = json.dumps(["Text."])
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend, system_prompt=None)

        segmenter.segment(original)

        assert backend.last_messages is not None
        system_msgs = [m for m in backend.last_messages if m.role == "system"]
        assert len(system_msgs) == 0

    def test_find_boundary_raises(self):
        """find_boundary raises NotImplementedError."""
        backend = _MockLLMBackend(response="[]")
        segmenter = LLMSegmenter(backend=backend)

        with pytest.raises(NotImplementedError):
            segmenter.find_boundary("some text")

    def test_protocol_compliance(self):
        """isinstance(segmenter, Segmenter) is True."""
        backend = _MockLLMBackend(response="[]")
        segmenter = LLMSegmenter(backend=backend)

        assert isinstance(segmenter, Segmenter)

    def test_segments_reconstruct_original(self):
        """''.join(segments) == original for well-formed inputs."""
        original = "First sentence. Second sentence. Third sentence."
        llm_response = json.dumps(["First sentence.", "Second sentence.", "Third sentence."])
        backend = _MockLLMBackend(response=llm_response)
        segmenter = LLMSegmenter(backend=backend)

        segments = segmenter.segment(original)

        assert "".join(segments) == original


# ---------------------------------------------------------------------------
# TestDefaultSegmentParser
# ---------------------------------------------------------------------------


class TestDefaultSegmentParser:
    """Unit tests for default_segment_parser."""

    def test_exact_match_segments(self):
        """Segments that exactly match substrings."""
        original = "Hello world. Goodbye world."
        llm_response = json.dumps(["Hello world.", "Goodbye world."])

        segments = default_segment_parser(original, llm_response)

        assert len(segments) == 2
        assert "Hello world." in segments[0]
        assert "Goodbye world." in segments[1]

    def test_concatenation_invariant(self):
        """Joined segments equal original."""
        original = "Alpha. Beta. Gamma."
        llm_response = json.dumps(["Alpha.", "Beta.", "Gamma."])

        segments = default_segment_parser(original, llm_response)

        assert "".join(segments) == original

    def test_code_block_stripping(self):
        """Markdown fences removed before parsing."""
        original = "One. Two."
        llm_response = '```\n["One.", "Two."]\n```'

        segments = default_segment_parser(original, llm_response)

        assert len(segments) == 2
        assert "".join(segments) == original

    def test_invalid_json_fallback(self):
        """Returns [original] for invalid JSON."""
        original = "Some text."

        segments = default_segment_parser(original, "{{broken")

        assert segments == [original]

    def test_non_string_array_fallback(self):
        """Returns [original] for non-string array."""
        original = "Some text."

        segments = default_segment_parser(original, "[1, 2, 3]")

        assert segments == [original]

    def test_empty_segments(self):
        """Returns [] for empty original."""
        segments = default_segment_parser("", "[]")

        assert segments == []
