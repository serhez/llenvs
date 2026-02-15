"""HuggingFace datasets adapter - wraps HF datasets as MDP environments.

Provides access to thousands of datasets on the HuggingFace Hub through
a common interface. Most HF datasets are single-turn (question -> answer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
import uuid

if TYPE_CHECKING:
    from llenvs.inference.prompts import PromptTemplate

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import SignalBundle, Signal, RewardType, RewardFunction
from llenvs.core.environment import Environment, StepResult, EnvironmentSpec
from llenvs.core.extraction import (
    AnswerExtractor,
    TagBasedExtractor,
    BoxedExtractor,
    NumericExtractor,
    LastLineExtractor,
    RawGenerationExtractor,
)


# Type alias for scoring functions
ScoringFunction = Callable[[str, str], float]


def normalize_numeric(value: str) -> str | None:
    """Normalize a numeric string for comparison.

    Handles integers, floats, removes trailing zeros.

    Args:
        value: String that may contain a number.

    Returns:
        Normalized numeric string or None if not numeric.
    """
    try:
        # Try to parse as float
        num = float(value.replace(',', ''))
        # If it's an integer, return as int string
        if num == int(num):
            return str(int(num))
        # Otherwise return float with reasonable precision
        return f"{num:.10g}"
    except (ValueError, TypeError):
        return None


def score_exact_match(predicted: str, expected: str) -> float:
    """Score based on exact string match (case-insensitive).

    Args:
        predicted: Model's predicted answer.
        expected: Expected answer.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def score_numeric(predicted: str, expected: str) -> float:
    """Score based on numeric equivalence.

    Args:
        predicted: Model's predicted answer.
        expected: Expected answer.

    Returns:
        1.0 if numerically equivalent, 0.0 otherwise.
    """
    pred_norm = normalize_numeric(predicted)
    exp_norm = normalize_numeric(expected)

    if pred_norm is None or exp_norm is None:
        return 0.0

    return 1.0 if pred_norm == exp_norm else 0.0


def score_numeric_tolerance(predicted: str, expected: str, rtol: float = 1e-5) -> float:
    """Score based on numeric equivalence with tolerance.

    Args:
        predicted: Model's predicted answer.
        expected: Expected answer.
        rtol: Relative tolerance for comparison.

    Returns:
        1.0 if within tolerance, 0.0 otherwise.
    """
    try:
        pred_val = float(predicted.replace(',', ''))
        exp_val = float(expected.replace(',', ''))

        if exp_val == 0:
            return 1.0 if abs(pred_val) < rtol else 0.0

        return 1.0 if abs(pred_val - exp_val) / abs(exp_val) < rtol else 0.0
    except (ValueError, TypeError):
        return 0.0


# Built-in ground truth extraction strategies (for dataset answer columns)
GROUND_TRUTH_EXTRACTORS: dict[str, AnswerExtractor] = {
    "boxed": BoxedExtractor(),
    "numeric": NumericExtractor(),
    "last_line": LastLineExtractor(),
    "direct": RawGenerationExtractor(),
}

# Built-in scoring functions
SCORING_FUNCTIONS: dict[str, ScoringFunction] = {
    "exact": score_exact_match,
    "numeric": score_numeric,
    "numeric_tolerance": score_numeric_tolerance,
}


@dataclass(frozen=True)
class HuggingFaceHidden:
    """Hidden state for HuggingFace dataset environments.

    Attributes:
        entry: The original dataset entry dict.
        expected_answer: The expected answer string.
        task_index: Index in the dataset.
        dataset_name: Full HuggingFace dataset name.
        split: Dataset split this came from.
    """
    entry: dict[str, Any]
    expected_answer: str
    task_index: int
    dataset_name: str
    split: str


@dataclass
class HuggingFaceCorrectnessReward:
    """Reward function for HuggingFace dataset correctness."""

    _name: str = "correctness"
    _reward_type: RewardType = RewardType.OUTCOME

    def __init__(
        self,
        answer_extractor: AnswerExtractor,
        scoring_fn: ScoringFunction,
    ) -> None:
        """Initialize reward function.

        Args:
            answer_extractor: Extractor for model responses.
            scoring_fn: Function to score predicted vs expected.
        """
        self._answer_extractor = answer_extractor
        self._scoring_fn = scoring_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[HuggingFaceHidden],
        action: Action,
        next_state: State[HuggingFaceHidden],
    ) -> Signal:
        """Compute correctness reward."""
        # Extract answer from model response
        extracted, extraction_meta = self._answer_extractor.extract(action.text)

        if extracted is None:
            return Signal(
                name=self.name,
                reward_type=self.reward_type,
                reward=0.0,
                metadata={"extracted": None, "extraction": extraction_meta},
            )

        # Get expected answer
        expected = state.hidden.expected_answer

        # Score
        score = self._scoring_fn(extracted, expected)

        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=float(score),
            metadata={
                "extracted": extracted,
                "expected": expected,
                "extraction": extraction_meta,
            },
        )


class HuggingFaceEnvironment:
    """MDP wrapper for HuggingFace datasets.

    Converts HuggingFace datasets to the Environment protocol.
    Each episode corresponds to a single example from the dataset.

    Attributes:
        dataset: The HuggingFace Dataset object.
        question_column: Column name for questions/problems.
        answer_column: Column name for answers/solutions.
    """

    def __init__(
        self,
        dataset: Any,  # datasets.Dataset
        dataset_name: str,
        split: str,
        question_column: str = "problem",
        answer_column: str = "solution",
        ground_truth_extractor: AnswerExtractor | str = "boxed",
        scoring: str | ScoringFunction = "numeric",
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        metadata_columns: list[str] | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            dataset: HuggingFace Dataset object.
            dataset_name: Full dataset name (e.g., "hendrycks/competition_math").
            split: Dataset split (e.g., "train", "test").
            question_column: Column containing the question/problem.
            answer_column: Column containing the answer/solution.
            ground_truth_extractor: Extractor for the dataset answer column.
                Either a string key ("boxed", "numeric", "last_line", "direct")
                or an AnswerExtractor instance.
            scoring: How to score answers. Either a string key
                ("exact", "numeric", "numeric_tolerance") or a custom function.
            answer_extractor: Extractor for model responses. Defaults to TagBasedExtractor.
            extra_rewards: Additional reward functions appended after native rewards.
            metadata_columns: Additional columns to include in state metadata.
        """
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._split = split
        self._question_column = question_column
        self._answer_column = answer_column
        self._metadata_columns = metadata_columns or []

        # Set up ground truth extractor
        if isinstance(ground_truth_extractor, str):
            if ground_truth_extractor not in GROUND_TRUTH_EXTRACTORS:
                raise ValueError(
                    f"Unknown ground_truth_extractor: {ground_truth_extractor}. "
                    f"Available: {list(GROUND_TRUTH_EXTRACTORS.keys())}"
                )
            self._ground_truth_extractor = GROUND_TRUTH_EXTRACTORS[ground_truth_extractor]
        else:
            self._ground_truth_extractor = ground_truth_extractor

        # Set up scoring
        if isinstance(scoring, str):
            if scoring not in SCORING_FUNCTIONS:
                raise ValueError(
                    f"Unknown scoring: {scoring}. "
                    f"Available: {list(SCORING_FUNCTIONS.keys())}"
                )
            self._scoring_fn = SCORING_FUNCTIONS[scoring]
        else:
            self._scoring_fn = scoring

        # Set up extractor for model responses
        self._answer_extractor = answer_extractor or TagBasedExtractor()

        # Build reward functions
        self._native_rewards: tuple[RewardFunction, ...] = (
            HuggingFaceCorrectnessReward(
                answer_extractor=self._answer_extractor,
                scoring_fn=self._scoring_fn,
            ),
        )
        self._extra_rewards = extra_rewards

    @property
    def prompts(self) -> dict[str, str]:
        """Single-turn environment has no configurable prompts."""
        return {}

    @property
    def available_tools(self) -> tuple:
        """No tools available for HuggingFace dataset environments."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name=self._dataset_name,
            adapter="huggingface",
            max_steps=1,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=False,
            pure_step=True,
            metadata={
                "dataset_size": len(self._dataset),
                "split": self._split,
                "question_column": self._question_column,
                "answer_column": self._answer_column,
            },
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[HuggingFaceHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    @property
    def dataset(self) -> Any:
        """Access the underlying HuggingFace dataset."""
        return self._dataset

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HuggingFaceHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed (not used, dataset is deterministic by index).
            options: Must contain "task_index" to select which task.
                Optional "episode_id" to override the generated ID.

        Returns:
            Tuple of (initial_state, info_dict).

        Raises:
            ValueError: If task_index is not provided or out of bounds.
        """
        options = options or {}

        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._dataset):
            raise ValueError(
                f"task_index {task_index} out of bounds [0, {len(self._dataset)})"
            )

        # Get the dataset entry
        entry = self._dataset[task_index]

        # Extract question
        if self._question_column not in entry:
            raise ValueError(
                f"Question column '{self._question_column}' not found in dataset. "
                f"Available columns: {list(entry.keys())}"
            )
        question = entry[self._question_column]

        # Extract and process expected answer
        if self._answer_column not in entry:
            raise ValueError(
                f"Answer column '{self._answer_column}' not found in dataset. "
                f"Available columns: {list(entry.keys())}"
            )
        raw_answer = entry[self._answer_column]

        # Apply ground truth extraction to get final answer
        expected_answer, _ = self._ground_truth_extractor.extract(str(raw_answer))
        if expected_answer is None:
            # Fall back to raw answer if extraction fails
            expected_answer = str(raw_answer).strip()

        # Create observation
        observation = Observation(prompt=question)

        # Create hidden state
        hidden = HuggingFaceHidden(
            entry=dict(entry),
            expected_answer=expected_answer,
            task_index=task_index,
            dataset_name=self._dataset_name,
            split=self._split,
        )

        # Create metadata
        episode_id = options.get("episode_id", str(uuid.uuid4()))
        info_dict: dict[str, Any] = {"task_index": task_index}

        # Add metadata columns
        for col in self._metadata_columns:
            if col in entry:
                info_dict[col] = entry[col]

        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info=info_dict,
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)

        info = {
            "task_index": task_index,
            "dataset_name": self._dataset_name,
            "split": self._split,
            "question": question,
            **{col: entry.get(col) for col in self._metadata_columns if col in entry},
        }

        return state, info

    def step(
        self,
        state: State[HuggingFaceHidden],
        action: Action,
    ) -> StepResult[HuggingFaceHidden]:
        """Take an action (model response) and return result.

        For HuggingFace datasets, a single step always terminates the episode.

        Args:
            state: Current state.
            action: Model's response.

        Returns:
            StepResult with next state, rewards, and done flag.
        """
        # Compute rewards
        rewards = self.compute_rewards(state, action, state)

        # Create terminal state
        next_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=True,
            info={
                **state.metadata.info,
                "response": action.text,
            },
        )

        next_state = State(
            observation=state.observation,
            hidden=state.hidden,
            metadata=next_metadata,
        )

        # Extract answer for info
        extracted, extraction_meta = self._answer_extractor.extract(action.text)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=True,
            truncated=False,
            info={
                "extracted_answer": extracted,
                "expected_answer": state.hidden.expected_answer,
                "extraction_metadata": extraction_meta,
            },
        )

    def compute_rewards(
        self,
        state: State[HuggingFaceHidden],
        action: Action,
        next_state: State[HuggingFaceHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def __len__(self) -> int:
        """Number of examples in the dataset."""
        return len(self._dataset)


class HuggingFaceAdapter:
    """Adapter for HuggingFace datasets.

    Provides access to datasets on the HuggingFace Hub through
    the common Adapter interface.

    Example:
        adapter = HuggingFaceAdapter()
        env = adapter.get_environment(
            "hendrycks/competition_math",
            split="test",
            question_column="problem",
            answer_column="solution",
            ground_truth_extractor="boxed",
            scoring="numeric",
        )
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialize the adapter.

        Args:
            cache_dir: Optional custom cache directory for datasets.
        """
        self._cache_dir = cache_dir
        self._loaded_datasets: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "huggingface"

    def _get_datasets_library(self) -> Any:
        """Import and return the datasets module."""
        try:
            import datasets
            return datasets
        except ImportError as e:
            raise ImportError(
                "The 'datasets' library is required for HuggingFaceAdapter. "
                "Install with: pip install datasets"
            ) from e

    def list_environments(self) -> list[str]:
        """List commonly used dataset names.

        Note: HuggingFace has thousands of datasets. This returns
        a curated list of popular math/reasoning datasets that have
        been tested with this adapter.

        Returns:
            List of popular dataset names.
        """
        return [
            # Math datasets with presets
            "HuggingFaceH4/aime_2024",
            "MathArena/aime_2025",
            "di-zhang-fdu/AIME_1983_2024",
            "gsm8k",
            # Other popular datasets (may need custom column config)
            "allenai/ai2_arc",
            "Rowan/hellaswag",
            "cais/mmlu",
            "truthful_qa",
            "trivia_qa",
        ]

    def get_environment(
        self,
        name: str,
        split: str = "test",
        subset: str | None = None,
        question_column: str = "problem",
        answer_column: str = "solution",
        ground_truth_extractor: AnswerExtractor | str = "boxed",
        scoring: str | ScoringFunction = "numeric",
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        metadata_columns: list[str] | None = None,
        size: int | None = None,
        seed: int | None = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> HuggingFaceEnvironment:
        """Create an environment from a HuggingFace dataset.

        Args:
            name: Dataset name (e.g., "hendrycks/competition_math").
            split: Dataset split to use (e.g., "train", "test").
            subset: Dataset subset/config name if applicable.
            question_column: Column containing questions.
            answer_column: Column containing answers.
            ground_truth_extractor: Extractor for the dataset answer column.
            scoring: How to score predicted vs expected answers.
            answer_extractor: Extractor for model responses.
            extra_rewards: Additional reward functions appended after native rewards.
            metadata_columns: Additional columns to include in metadata.
            size: Limit dataset to first N examples.
            seed: Random seed for shuffling (if size is set).
            streaming: Whether to use streaming mode.
            trust_remote_code: Whether to trust remote code in datasets.
            **kwargs: Additional arguments passed to load_dataset.

        Returns:
            Configured HuggingFaceEnvironment.

        Raises:
            ImportError: If datasets library is not installed.
            ValueError: If dataset or columns not found.
        """
        datasets = self._get_datasets_library()

        # Build cache key
        cache_key = f"{name}:{subset or 'default'}:{split}"

        # Load dataset
        if cache_key not in self._loaded_datasets:
            load_kwargs: dict[str, Any] = {
                "split": split,
                "trust_remote_code": trust_remote_code,
                **kwargs,
            }

            if subset:
                load_kwargs["name"] = subset

            if self._cache_dir:
                load_kwargs["cache_dir"] = self._cache_dir

            if streaming:
                load_kwargs["streaming"] = True

            dataset = datasets.load_dataset(name, **load_kwargs)

            # Handle streaming datasets differently
            if not streaming:
                self._loaded_datasets[cache_key] = dataset
        else:
            dataset = self._loaded_datasets[cache_key]

        # Apply size limit if specified
        if size is not None and not streaming:
            if seed is not None:
                dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(min(size, len(dataset))))

        return HuggingFaceEnvironment(
            dataset=dataset,
            dataset_name=name,
            split=split,
            question_column=question_column,
            answer_column=answer_column,
            ground_truth_extractor=ground_truth_extractor,
            scoring=scoring,
            answer_extractor=answer_extractor,
            extra_rewards=extra_rewards,
            metadata_columns=metadata_columns,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """HuggingFace datasets are raw, no default system prompt."""
        return None

    def get_prompt_template(self, name: str) -> "PromptTemplate | None":
        """Return a prompt template based on dataset presets.

        Math-related datasets get the math template; others return None.

        Args:
            name: Dataset name.

        Returns:
            A PromptTemplate or None.
        """
        from llenvs.inference.prompts import TEMPLATE_REGISTRY

        if name not in DATASET_PRESETS:
            return None

        preset = DATASET_PRESETS[name]
        scoring = preset.get("scoring", "")
        ground_truth_extractor = preset.get("ground_truth_extractor", "")

        if scoring == "numeric" or ground_truth_extractor in ("boxed", "numeric"):
            return TEMPLATE_REGISTRY.get("math")

        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """HuggingFace does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a dataset without loading it.

        Args:
            name: Dataset name.

        Returns:
            Dictionary with dataset metadata.
        """
        return {
            "name": name,
            "adapter": self.name,
            "type": "single_turn",
            "description": f"HuggingFace dataset: {name}",
            "url": f"https://huggingface.co/datasets/{name}",
        }

    def get_dataset_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a dataset from HuggingFace.

        Args:
            name: Dataset name.

        Returns:
            Dataset info from HuggingFace Hub.
        """
        try:
            from huggingface_hub import dataset_info
            info = dataset_info(name)
            return {
                "name": name,
                "description": info.description,
                "citation": info.citation,
                "license": info.license,
                "tags": info.tags,
                "downloads": info.downloads,
            }
        except Exception:
            return self.get_environment_info(name)


# Preset configurations for common datasets
DATASET_PRESETS: dict[str, dict[str, Any]] = {
    # AIME datasets
    "HuggingFaceH4/aime_2024": {
        "split": "train",  # This dataset only has train split
        "question_column": "problem",
        "answer_column": "answer",
        "ground_truth_extractor": "direct",
        "scoring": "numeric",
        "metadata_columns": ["id", "year", "url"],
    },
    "MathArena/aime_2025": {
        "split": "train",
        "question_column": "problem",
        "answer_column": "answer",
        "ground_truth_extractor": "direct",
        "scoring": "numeric",
    },
    "di-zhang-fdu/AIME_1983_2024": {
        "split": "train",
        "question_column": "Question",
        "answer_column": "Answer",
        "ground_truth_extractor": "direct",
        "scoring": "numeric",
        "metadata_columns": ["Year", "Problem Number"],
    },
    # GSM8K
    "gsm8k": {
        "subset": "main",
        "question_column": "question",
        "answer_column": "answer",
        "ground_truth_extractor": "numeric",
        "scoring": "numeric",
    },
    # OpenAI simple-evals MATH (if available)
    "openai/gsm8k": {
        "question_column": "question",
        "answer_column": "answer",
        "ground_truth_extractor": "numeric",
        "scoring": "numeric",
    },
}
