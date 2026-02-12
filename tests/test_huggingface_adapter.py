"""Tests for the HuggingFace adapter."""

import pytest
from typing import Any

from llenvs.core.state import Observation, Action
from llenvs.core.reward import RewardType
from llenvs.core.extraction import TagBasedExtractor, RegexExtractor
from llenvs.adapters.huggingface import (
    HuggingFaceEnvironment,
    HuggingFaceHidden,
    HuggingFaceAdapter,
    HuggingFaceCorrectnessReward,
    normalize_numeric,
    score_exact_match,
    score_numeric,
    score_numeric_tolerance,
    DATASET_PRESETS,
)
from llenvs.core.reward import FormatReward
from llenvs.core.registry import EnvironmentRegistry


class MockHFDataset:
    """Mock HuggingFace Dataset for testing."""

    def __init__(self, entries: list[dict[str, Any]] | None = None):
        self.entries = entries or [
            {
                "problem": "What is 2 + 2?",
                "solution": "We have $2 + 2 = \\boxed{4}$.",
                "level": "Level 1",
                "type": "Algebra",
            },
            {
                "problem": "What is 3 * 3?",
                "solution": "We compute $3 \\times 3 = \\boxed{9}$.",
                "level": "Level 1",
                "type": "Algebra",
            },
            {
                "problem": "What is 10 / 2?",
                "solution": "Division gives us $\\boxed{5}$.",
                "level": "Level 2",
                "type": "Algebra",
            },
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.entries[index]

    def shuffle(self, seed: int = None):
        """Mock shuffle (returns self for simplicity)."""
        return self

    def select(self, indices):
        """Mock select."""
        return MockHFDataset([self.entries[i] for i in indices])


@pytest.fixture
def mock_hf_dataset() -> MockHFDataset:
    """Create mock HuggingFace dataset."""
    return MockHFDataset()


@pytest.fixture
def mock_aime_dataset() -> MockHFDataset:
    """Create mock AIME-style dataset with direct numeric answers."""
    return MockHFDataset([
        {
            "problem": "Find the sum of 1 + 2 + 3 + ... + 10.",
            "answer": "55",
            "problem_idx": 1,
            "problem_type": ["Algebra"],
        },
        {
            "problem": "How many factors does 12 have?",
            "answer": "6",
            "problem_idx": 2,
            "problem_type": ["Number Theory"],
        },
    ])



class TestNormalization:
    """Tests for answer normalization."""

    def test_normalize_integer(self):
        """Test normalizing integer."""
        assert normalize_numeric("42") == "42"

    def test_normalize_float_as_int(self):
        """Test float that equals integer."""
        assert normalize_numeric("42.0") == "42"

    def test_normalize_float(self):
        """Test normalizing float."""
        assert normalize_numeric("3.14") == "3.14"

    def test_normalize_with_commas(self):
        """Test normalizing number with commas."""
        assert normalize_numeric("1,234") == "1234"

    def test_normalize_not_numeric(self):
        """Test non-numeric returns None."""
        assert normalize_numeric("abc") is None


class TestScoring:
    """Tests for scoring functions."""

    def test_exact_match_equal(self):
        """Test exact match for equal strings."""
        assert score_exact_match("hello", "hello") == 1.0

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        assert score_exact_match("Hello", "hello") == 1.0

    def test_exact_match_different(self):
        """Test exact match for different strings."""
        assert score_exact_match("hello", "world") == 0.0

    def test_numeric_equal(self):
        """Test numeric scoring for equal numbers."""
        assert score_numeric("42", "42") == 1.0

    def test_numeric_float_int_equal(self):
        """Test numeric scoring for float/int equivalence."""
        assert score_numeric("42.0", "42") == 1.0

    def test_numeric_different(self):
        """Test numeric scoring for different numbers."""
        assert score_numeric("42", "43") == 0.0

    def test_numeric_tolerance_within(self):
        """Test numeric tolerance scoring within tolerance."""
        assert score_numeric_tolerance("1.000001", "1.0") == 1.0

    def test_numeric_tolerance_outside(self):
        """Test numeric tolerance scoring outside tolerance."""
        assert score_numeric_tolerance("1.1", "1.0", rtol=0.01) == 0.0


class TestHuggingFaceEnvironment:
    """Tests for HuggingFaceEnvironment."""

    def test_creation(self, mock_hf_dataset):
        """Test environment creation."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
            question_column="problem",
            answer_column="solution",
            ground_truth_extractor="boxed",
            scoring="numeric",
        )

        assert env.spec.name == "test/dataset"
        assert env.spec.adapter == "huggingface"
        assert env.spec.max_steps == 1
        assert len(env) == 3

    def test_spec_metadata(self, mock_hf_dataset):
        """Test environment spec metadata."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
        )
        spec = env.spec

        assert spec.metadata["dataset_size"] == 3
        assert spec.metadata["split"] == "test"
        assert spec.observation_type == Observation
        assert spec.action_type == Action
        assert spec.supports_branching is True

    def test_reset(self, mock_hf_dataset):
        """Test environment reset."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
            metadata_columns=["level", "type"],
        )
        state, info = env.reset(options={"task_index": 0})

        # Check observation
        assert isinstance(state.observation, Observation)
        assert "2 + 2" in state.observation.prompt

        # Check hidden state
        assert isinstance(state.hidden, HuggingFaceHidden)
        assert state.hidden.expected_answer == "4"
        assert state.hidden.task_index == 0
        assert state.hidden.dataset_name == "test/dataset"

        # Check metadata
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

        # Check info includes metadata columns
        assert info["task_index"] == 0
        assert "level" in info or "level" in state.metadata.info

    def test_reset_requires_task_index(self, mock_hf_dataset):
        """Test that reset requires task_index."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
        )

        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_reset_validates_task_index(self, mock_hf_dataset):
        """Test task_index bounds checking."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
        )

        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 100})

    def test_step_correct_answer(self, mock_hf_dataset):
        """Test step with correct answer."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
        )
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="The answer is <answer>4</answer>")
        result = env.step(state, action)

        # Check termination
        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

        # Check correctness reward (native-only default)
        correctness = result.rewards.by_name("correctness")
        assert correctness is not None
        assert correctness.value == 1.0

        # No format reward by default
        assert result.rewards.by_name("format") is None

    def test_step_incorrect_answer(self, mock_hf_dataset):
        """Test step with incorrect answer."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
        )
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="The answer is <answer>5</answer>")
        result = env.step(state, action)

        correctness = result.rewards.by_name("correctness")
        assert correctness.value == 0.0

    def test_step_no_answer_extracted(self, mock_hf_dataset):
        """Test step when no answer can be extracted."""
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
        )
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="I don't know")
        result = env.step(state, action)

        correctness = result.rewards.by_name("correctness")
        assert correctness.value == 0.0

    def test_custom_extractor(self, mock_hf_dataset):
        """Test with custom extractor."""
        extractor = RegexExtractor(pattern=r"answer is (\d+)")
        env = HuggingFaceEnvironment(
            dataset=mock_hf_dataset,
            dataset_name="test/dataset",
            split="test",
            answer_extractor=extractor,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="The answer is 4"))

        assert result.info["extracted_answer"] == "4"

    def test_direct_answer_column(self, mock_aime_dataset):
        """Test with direct answer column (like AIME)."""
        env = HuggingFaceEnvironment(
            dataset=mock_aime_dataset,
            dataset_name="aime/test",
            split="test",
            question_column="problem",
            answer_column="answer",
            ground_truth_extractor="direct",
            scoring="numeric",
        )

        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.expected_answer == "55"

        result = env.step(state, Action(text="<answer>55</answer>"))
        correctness = result.rewards.by_name("correctness")
        assert correctness.value == 1.0


class TestHuggingFaceAdapter:
    """Tests for HuggingFaceAdapter."""

    def test_adapter_name(self):
        """Test adapter name property."""
        adapter = HuggingFaceAdapter()
        assert adapter.name == "huggingface"

    def test_list_environments(self):
        """Test list_environments returns curated list."""
        adapter = HuggingFaceAdapter()
        envs = adapter.list_environments()

        assert isinstance(envs, list)
        assert len(envs) > 0
        assert "HuggingFaceH4/aime_2024" in envs
        assert "gsm8k" in envs

    def test_get_environment_info(self):
        """Test get_environment_info returns metadata."""
        adapter = HuggingFaceAdapter()
        info = adapter.get_environment_info("hendrycks/competition_math")

        assert info["name"] == "hendrycks/competition_math"
        assert info["adapter"] == "huggingface"
        assert "url" in info


class TestDatasetPresets:
    """Tests for dataset presets."""

    def test_aime_2024_preset_exists(self):
        """Test AIME 2024 dataset preset exists."""
        assert "HuggingFaceH4/aime_2024" in DATASET_PRESETS

    def test_aime_2024_preset_config(self):
        """Test AIME 2024 preset has correct config."""
        preset = DATASET_PRESETS["HuggingFaceH4/aime_2024"]
        assert preset["question_column"] == "problem"
        assert preset["answer_column"] == "answer"
        assert preset["ground_truth_extractor"] == "direct"
        assert preset["scoring"] == "numeric"

    def test_aime_historical_preset_exists(self):
        """Test historical AIME preset exists."""
        assert "di-zhang-fdu/AIME_1983_2024" in DATASET_PRESETS

    def test_gsm8k_preset_exists(self):
        """Test GSM8K preset exists."""
        assert "gsm8k" in DATASET_PRESETS

    def test_gsm8k_preset_config(self):
        """Test GSM8K preset has correct config."""
        preset = DATASET_PRESETS["gsm8k"]
        assert preset["question_column"] == "question"
        assert preset["answer_column"] == "answer"
        assert preset["ground_truth_extractor"] == "numeric"
        assert preset["scoring"] == "numeric"


class TestEnvironmentRegistryIntegration:
    """Tests for EnvironmentRegistry integration."""

    def test_register_huggingface_adapter(self, mock_hf_dataset, monkeypatch):
        """Test HuggingFace adapter registration."""
        registry = EnvironmentRegistry()

        # Mock the datasets library check
        def mock_get_datasets():
            class MockDatasets:
                pass
            return MockDatasets()

        adapter = HuggingFaceAdapter()
        monkeypatch.setattr(adapter, "_get_datasets_library", mock_get_datasets)

        registry.register_adapter(adapter)
        assert "huggingface" in registry.list_adapters()

    def test_get_adapter(self, monkeypatch):
        """Test getting HuggingFace adapter."""
        registry = EnvironmentRegistry()

        def mock_get_datasets():
            class MockDatasets:
                pass
            return MockDatasets()

        adapter = HuggingFaceAdapter()
        monkeypatch.setattr(adapter, "_get_datasets_library", mock_get_datasets)

        registry.register_adapter(adapter)
        retrieved = registry.get_adapter("huggingface")
        assert retrieved is adapter


class TestHuggingFaceHidden:
    """Tests for HuggingFaceHidden."""

    def test_creation(self):
        """Test hidden state creation."""
        entry = {"problem": "Q", "solution": "A"}
        hidden = HuggingFaceHidden(
            entry=entry,
            expected_answer="42",
            task_index=5,
            dataset_name="test/dataset",
            split="test",
        )

        assert hidden.entry == entry
        assert hidden.expected_answer == "42"
        assert hidden.task_index == 5
        assert hidden.dataset_name == "test/dataset"
        assert hidden.split == "test"

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = HuggingFaceHidden(
            entry={},
            expected_answer="A",
            task_index=0,
            dataset_name="test",
            split="test",
        )
        with pytest.raises(AttributeError):
            hidden.expected_answer = "B"  # type: ignore
