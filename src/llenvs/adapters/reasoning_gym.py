"""ReasoningGym adapter - wraps reasoning-gym datasets as MDP environments.

Reasoning-gym tasks are single-turn (question -> answer), but the MDP
interface generalizes to multi-turn for future adapters.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.extraction import AnswerExtractor, TagBasedExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata


@dataclass(frozen=True)
class ReasoningGymHidden:
    """Hidden state for reasoning-gym environments.

    Contains the original dataset entry and expected answer
    needed for reward computation.

    Attributes:
        entry: The original dataset entry dict.
        expected_answer: The expected answer string.
        task_index: Index in the dataset.
        dataset_name: Name of the reasoning-gym dataset.
    """

    entry: dict[str, Any]
    expected_answer: str
    task_index: int
    dataset_name: str


@dataclass
class CorrectnessRewardFunction:
    """Reward function based on reasoning-gym's built-in scoring."""

    _name: str = "correctness"
    _reward_type: RewardType = RewardType.OUTCOME

    def __init__(self, dataset: Any, answer_extractor: AnswerExtractor) -> None:
        """Initialize with dataset and answer extractor.

        Args:
            dataset: reasoning-gym ProceduralDataset instance.
            answer_extractor: Extractor to parse model responses.
        """
        self._dataset = dataset
        self._answer_extractor = answer_extractor

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[ReasoningGymHidden],
        action: Action,
        next_state: State[ReasoningGymHidden],
    ) -> Signal:
        """Compute correctness reward using reasoning-gym's scoring."""
        extracted, extraction_meta = self._answer_extractor.extract(action.text)

        if extracted is None:
            return Signal(
                name=self.name,
                reward_type=self.reward_type,
                reward=0.0,
                metadata={"extracted": None, "extraction": extraction_meta},
            )

        # Use reasoning-gym's built-in scoring
        score = self._dataset.score_answer(
            answer=extracted,
            entry=state.hidden.entry,
        )

        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=float(score),
            metadata={
                "extracted": extracted,
                "expected": state.hidden.expected_answer,
                "extraction": extraction_meta,
            },
        )


class ReasoningGymEnvironment:
    """MDP wrapper for reasoning-gym ProceduralDataset.

    Converts reasoning-gym's dataset interface to the Environment protocol.
    Each episode corresponds to a single question from the dataset.

    Attributes:
        dataset: The underlying reasoning-gym ProceduralDataset.
        answer_extractor: AnswerExtractor for parsing model responses.
    """

    def __init__(
        self,
        dataset: Any,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        """Initialize the environment.

        Args:
            dataset: reasoning-gym ProceduralDataset instance.
            answer_extractor: Extractor for parsing model responses.
                Defaults to TagBasedExtractor with "answer" tag.
            extra_rewards: Additional reward functions appended after native rewards.
        """
        self._dataset = dataset
        self._answer_extractor = answer_extractor or TagBasedExtractor()

        # Build reward functions
        self._native_rewards: tuple[RewardFunction, ...] = (
            CorrectnessRewardFunction(dataset, self._answer_extractor),
        )
        self._extra_rewards = extra_rewards

        # Get dataset name from config if available
        self._dataset_name = getattr(dataset, "name", "reasoning_gym")

    @property
    def prompts(self) -> dict[str, str]:
        """Single-turn environment has no configurable prompts."""
        return {}

    @property
    def available_tools(self) -> tuple:
        """No tools available in reasoning-gym environments."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name=self._dataset_name,
            adapter="reasoning_gym",
            max_steps=1,  # Single-turn environment
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=False,
            pure_step=True,
            metadata={
                "dataset_size": len(self._dataset),
                "dataset_name": self._dataset_name,
            },
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[ReasoningGymHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[ReasoningGymHidden], dict[str, Any]]:
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
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._dataset)})")

        # Get the dataset entry
        entry = self._dataset[task_index]

        # Create observation (what model sees)
        observation = Observation(
            prompt=entry["question"],
            task=ObservationContent(text=entry["question"]),
        )

        # Create hidden state (for reward computation)
        hidden = ReasoningGymHidden(
            entry=entry,
            expected_answer=str(entry["answer"]),
            task_index=task_index,
            dataset_name=self._dataset_name,
        )

        # Create metadata
        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)

        info = {
            "task_index": task_index,
            "dataset_name": self._dataset_name,
            "question": entry["question"],
        }

        return state, info

    def step(
        self,
        state: State[ReasoningGymHidden],
        action: Action,
    ) -> StepResult[ReasoningGymHidden]:
        """Take an action (model response) and return result.

        For reasoning-gym tasks, a single step always terminates the episode.

        Args:
            state: Current state.
            action: Model's response.

        Returns:
            StepResult with next state, rewards, and done flag.
        """
        # Compute rewards
        rewards = self.compute_rewards(state, action, state)

        # Create terminal state (same observation/hidden, updated metadata)
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
            terminated=True,  # Single-turn, always terminates
            truncated=False,
            extracted_action=extracted,
            resolved_action=extracted,
            info={
                "extracted_answer": extracted,
                "expected_answer": state.hidden.expected_answer,
                "extraction_metadata": extraction_meta,
            },
        )

    def compute_rewards(
        self,
        state: State[ReasoningGymHidden],
        action: Action,
        next_state: State[ReasoningGymHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action (unused for single-turn).

        Returns:
            SignalBundle with correctness and optionally format signals.
        """
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def __len__(self) -> int:
        """Number of tasks in the dataset."""
        return len(self._dataset)


class ReasoningGymAdapter:
    """Adapter for the reasoning-gym library.

    Provides access to all environments available in reasoning-gym
    through the common Adapter interface.

    Example:
        adapter = ReasoningGymAdapter()
        envs = adapter.list_environments()  # ["sudoku", "leg_counting", ...]
        env = adapter.get_environment("sudoku", size=100, seed=42)
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "reasoning_gym"

    def _get_reasoning_gym(self) -> Any:
        """Import and return the reasoning_gym module."""
        try:
            import reasoning_gym

            return reasoning_gym
        except ImportError as e:
            raise ImportError(
                "reasoning-gym is required for ReasoningGymAdapter. "
                "Install with: pip install reasoning-gym"
            ) from e

    def list_environments(self) -> list[str]:
        """List all available environment names from reasoning-gym.

        Returns:
            List of dataset names that can be passed to get_environment().
        """
        rg = self._get_reasoning_gym()

        # reasoning-gym provides a registry of datasets
        if hasattr(rg, "list_datasets"):
            return rg.list_datasets()
        elif hasattr(rg, "DATASETS"):
            return list(rg.DATASETS.keys())
        elif hasattr(rg, "registry") and hasattr(rg.registry, "list_datasets"):
            return rg.registry.list_datasets()
        else:
            # Fallback: return known common datasets
            return [
                "leg_counting",
                "sudoku",
                "simple_arithmetic",
                "chain_sum",
                "polynomial_equations",
            ]

    def get_environment(
        self,
        name: str,
        size: int | None = None,
        seed: int | None = None,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> ReasoningGymEnvironment:
        """Create an environment by name.

        Args:
            name: Environment/dataset name (e.g., "sudoku", "leg_counting").
            size: Number of samples to include.
            seed: Random seed for dataset generation.
            answer_extractor: AnswerExtractor for parsing responses.
            extra_rewards: Additional reward functions appended after native rewards.
            **kwargs: Additional arguments passed to reasoning-gym.

        Returns:
            Configured ReasoningGymEnvironment.

        Raises:
            ValueError: If the environment name is not recognized.
            ImportError: If reasoning-gym is not installed.
        """
        rg = self._get_reasoning_gym()

        # Create the dataset
        create_kwargs: dict[str, Any] = {**kwargs}
        if size is not None:
            create_kwargs["size"] = size
        if seed is not None:
            create_kwargs["seed"] = seed

        dataset = rg.create_dataset(name, **create_kwargs)

        return ReasoningGymEnvironment(
            dataset=dataset,
            answer_extractor=answer_extractor,
            extra_rewards=extra_rewards,
        )

    def get_native_answer_extractor(self, task_name: str) -> AnswerExtractor:
        """Return reasoning-gym's native extraction function.

        reasoning-gym provides a single generic extractor for all tasks
        via reasoning_gym.utils.extract_answer.

        Args:
            task_name: Task name (unused - reasoning-gym has one extractor).

        Returns:
            NativeExtractor wrapping reasoning_gym.utils.extract_answer.
        """
        from llenvs.core.extraction import NativeExtractor

        self._get_reasoning_gym()
        from reasoning_gym.utils import extract_answer as rg_extract

        def _extract(text: str) -> str | None:
            return rg_extract(text, tag_name="answer", strip=True)

        return NativeExtractor(fn=_extract, name="reasoning_gym")

    def get_default_system_prompt(self, name: str) -> None:
        """reasoning-gym questions are self-contained, no default system prompt."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """reasoning-gym questions are self-contained, no wrapping needed."""
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about an environment without creating it.

        Args:
            name: Environment name.

        Returns:
            Dictionary with environment metadata.
        """
        # reasoning-gym doesn't provide rich metadata, return basic info
        return {
            "name": name,
            "adapter": self.name,
            "type": "single_turn",
            "description": f"reasoning-gym dataset: {name}",
        }
