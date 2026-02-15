"""LMRL-Gym adapter — wraps LMRL-Gym text-based RL environments.

LMRL-Gym provides text-based RL environments for language models including
maze navigation, Wordle, chess, twenty questions, guess the city,
car dealer negotiation, and text navigation. Agents interact through
free-form text, receiving text observations and scalar rewards.

Reference: https://github.com/abdulhaim/LMRL-Gym
"""

from dataclasses import dataclass
from typing import Any
import uuid

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import SignalBundle, Signal, RewardType, RewardFunction
from llenvs.core.environment import StepResult, EnvironmentSpec, _StateContinuityTracker


@dataclass(frozen=True)
class _LMRLText:
    """Minimal Text-compatible object for constructing LMRL-Gym actions.

    LMRL-Gym environments access ``.text`` and ``.is_action`` via attribute
    access, so this simple frozen dataclass is fully compatible.
    """

    text: str
    is_action: bool


LMRL_PRESETS: dict[str, dict[str, Any]] = {
    "wordle": {
        "max_steps": 6,
        "description": "Wordle word guessing game (6 guesses, 5-letter words)",
    },
    "chess": {
        "max_steps": 400,
        "description": "Chess against Stockfish engine",
    },
    "chess:endgame": {
        "max_steps": 400,
        "description": "Chess endgame (King+Queen vs King)",
    },
    "maze:double_t": {
        "max_steps": 100,
        "description": "Double-T maze navigation",
    },
    "maze:umaze": {
        "max_steps": 100,
        "description": "U-maze navigation",
    },
    "twenty_questions": {
        "max_steps": 20,
        "description": "Twenty Questions guessing game (requires oracle)",
    },
    "guess_city": {
        "max_steps": 20,
        "description": "Guess the city game (requires oracle)",
    },
    "car_dealer": {
        "max_steps": 50,
        "description": "Car dealer negotiation (requires buyer policy)",
    },
    "text_nav": {
        "max_steps": 100,
        "description": "TextWorld-based text navigation",
    },
}


@dataclass(frozen=True)
class LMRLHidden:
    """Hidden state for LMRL-Gym environment.

    Attributes:
        env_name: Environment identifier.
        episode_step: Current step within the episode.
        last_action: The last action taken.
        cumulative_reward: Cumulative reward so far.
        text_history: Raw TextHistory from the LMRL-Gym environment,
            stored as a tuple of Text-compatible objects.
    """

    env_name: str
    episode_step: int
    last_action: str | None
    cumulative_reward: float
    text_history: tuple


@dataclass
class LMRLReward:
    """Reward function for LMRL-Gym environments.

    Uses the native reward signal from the TextEnv.step() call.
    Intermediate steps produce STEP rewards with the per-step reward;
    terminal steps produce OUTCOME rewards with the cumulative reward.
    """

    _name: str = "lmrl_reward"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.STEP

    def compute(
        self,
        state: State[LMRLHidden],
        action: Action,
        next_state: State[LMRLHidden],
    ) -> Signal:
        """Compute reward from LMRL-Gym's native reward signal."""
        info = next_state.metadata.info
        is_terminal = next_state.metadata.is_terminal

        if is_terminal:
            cumulative = info.get("cumulative_reward", 0.0)
            return Signal(
                name=self.name,
                reward_type=RewardType.OUTCOME,
                reward=cumulative,
                metadata={"source": "lmrl", "cumulative_reward": cumulative},
            )
        else:
            step_reward = info.get("step_reward", 0.0)
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=step_reward,
                metadata={"source": "lmrl", "step_reward": step_reward},
            )


class LMRLEnvironment:
    """MDP wrapper for LMRL-Gym TextEnv environments.

    Wraps the LMRL-Gym TextEnv protocol into the llenvs Environment
    interface. LMRL-Gym environments are multi-turn text-based RL
    environments where agents interact through free-form text.

    The TextEnv protocol expects:
        - ``reset(seed, options) -> TextHistory``
        - ``step(text_history) -> (TextHistory, float, bool)``

    where TextHistory is a tuple of Text objects (with ``.text`` and
    ``.is_action`` attributes).

    Example:
        >>> env = LMRLEnvironment(text_env=my_text_env, env_name="wordle")
        >>> state, _ = env.reset(seed=42)
        >>> result = env.step(state, Action(text="crane"))
    """

    def __init__(
        self,
        text_env: Any,
        env_name: str = "lmrl",
        max_steps: int = 100,
        num_tasks: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        """Initialize LMRL-Gym environment wrapper.

        Args:
            text_env: An LMRL-Gym TextEnv instance (or any object
                following the TextEnv protocol).
            env_name: Human-readable name for this environment.
            max_steps: Maximum steps per episode before truncation.
            num_tasks: If set, enables task indexing and ``__len__``.
                Task indices are mapped to seeds.
            extra_rewards: Additional reward functions appended after
                native rewards.
        """
        self._text_env = text_env
        self._env_name = env_name
        self._max_steps = max_steps
        self._num_tasks = num_tasks
        self._native_rewards: tuple[RewardFunction, ...] = (LMRLReward(),)
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

    def __len__(self) -> int:
        if self._num_tasks is None:
            raise TypeError(
                f"LMRLEnvironment '{self._env_name}' has no fixed task count. "
                f"Pass num_tasks= to enable __len__."
            )
        return self._num_tasks

    @property
    def prompts(self) -> dict[str, str]:
        """No configurable prompts — LMRL-Gym manages its own text."""
        return {}

    @property
    def available_tools(self) -> tuple:
        """No tools available in LMRL-Gym environments."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name=self._env_name,
            adapter="lmrl",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=self._num_tasks is not None,
            supports_len=self._num_tasks is not None,
            supports_seed=True,
            pure_step=False,
            metadata={
                "description": f"LMRL-Gym {self._env_name} environment",
            },
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[LMRLHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def _extract_observation(self, text_history: tuple) -> str:
        """Extract observation text from a full TextHistory.

        Concatenates all non-action texts in the history.
        """
        obs_parts = [t.text for t in text_history if not t.is_action]
        return "".join(obs_parts).strip() if obs_parts else ""

    def _extract_last_observation(
        self, text_history: tuple, prev_len: int
    ) -> str:
        """Extract the latest observation added after stepping.

        Only looks at texts appended after the previous history length.
        """
        new_texts = text_history[prev_len:]
        obs_parts = [t.text for t in new_texts if not t.is_action]
        return "".join(obs_parts).strip() if obs_parts else ""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[LMRLHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed for the episode.
            options: Environment-specific options.
                - task_index: Select specific task (mapped to seed).

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index")

        if task_index is not None and self._num_tasks is not None:
            if task_index < 0 or task_index >= self._num_tasks:
                raise IndexError(
                    f"task_index {task_index} out of range "
                    f"[0, {self._num_tasks})"
                )
            # Use task_index as seed if no explicit seed provided
            if seed is None:
                seed = task_index

        text_history = self._text_env.reset(seed=seed)
        obs_text = self._extract_observation(text_history)

        hidden = LMRLHidden(
            env_name=self._env_name,
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            text_history=tuple(text_history),
        )

        observation = Observation(prompt=obs_text)

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "env_name": self._env_name,
                "seed": seed,
                "done": False,
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "env_name": self._env_name,
            "seed": seed,
        }

    def step(
        self,
        state: State[LMRLHidden],
        action: Action,
    ) -> StepResult[LMRLHidden]:
        """Take an action from the given state.

        Constructs a TextHistory with the action appended and calls
        the underlying TextEnv.step().

        Args:
            state: Current state.
            action: Action to take (free-form text).

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        self._state_tracker.validate(state, "LMRLEnvironment")

        # Build TextHistory with action appended
        action_text = _LMRLText(text=action.text or "", is_action=True)
        text_history = state.hidden.text_history + (action_text,)
        prev_len = len(text_history)

        # Step the LMRL-Gym environment
        new_history, step_reward, done = self._text_env.step(text_history)

        # Extract new observation
        obs_text = self._extract_last_observation(new_history, prev_len)

        # Compute cumulative reward
        cumulative_reward = state.hidden.cumulative_reward + step_reward

        # Check termination/truncation
        next_step = state.hidden.episode_step + 1
        terminated = bool(done)
        truncated = next_step >= self._max_steps and not terminated

        new_hidden = LMRLHidden(
            env_name=self._env_name,
            episode_step=next_step,
            last_action=action.text,
            cumulative_reward=cumulative_reward,
            text_history=tuple(new_history),
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": obs_text},
        )
        new_observation = Observation(
            prompt=state.observation.prompt, messages=new_messages
        )

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                **state.metadata.info,
                "step_reward": step_reward,
                "cumulative_reward": cumulative_reward,
                "done": done,
                "last_action": action.text,
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        # Compute rewards
        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "step_reward": step_reward,
                "cumulative_reward": cumulative_reward,
                "done": done,
                "action": action.text,
            },
        )

    def compute_rewards(
        self,
        state: State[LMRLHidden],
        action: Action,
        next_state: State[LMRLHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Close the underlying TextEnv if it supports close()."""
        if hasattr(self._text_env, "close"):
            self._text_env.close()


def _create_text_env(name: str, **kwargs: Any) -> Any:
    """Create a TextEnv instance from a preset name.

    Args:
        name: Preset name (e.g., "wordle", "chess").
        **kwargs: Additional arguments passed to the environment constructor.

    Returns:
        A TextEnv-compatible instance.

    Raises:
        ImportError: If required modules are not installed.
        ValueError: If the environment requires manual setup.
    """
    if name == "chess":
        return _create_chess(**kwargs)
    elif name == "chess:endgame":
        return _create_chess_endgame(**kwargs)
    elif name in ("twenty_questions", "guess_city", "car_dealer"):
        raise ValueError(
            f"The '{name}' environment requires an external model/policy. "
            f"Create the TextEnv manually and pass it via text_env=."
        )
    elif name == "text_nav":
        raise ValueError(
            "The 'text_nav' environment requires a custom TextWorld fork. "
            "Create the TextEnv manually and pass it via text_env=."
        )
    else:
        raise ValueError(
            f"No auto-creation factory for '{name}'. "
            f"Create the TextEnv manually and pass it via text_env=."
        )


def _create_chess(**kwargs: Any) -> Any:
    """Create a Chess TextEnv."""
    try:
        from llm_rl_scripts.chess.env.env import FenChessHistoryEnv
    except ImportError as e:
        raise ImportError(
            "Chess environment requires LMRL-Gym chess module. "
            "Ensure LMRL-Gym is installed and python-chess + stockfish are available."
        ) from e

    return FenChessHistoryEnv(
        max_moves=kwargs.get("max_moves", 400),
        from_position=kwargs.get("from_position"),
        random_opponent=kwargs.get("random_opponent", False),
    )


def _create_chess_endgame(**kwargs: Any) -> Any:
    """Create a Chess endgame TextEnv."""
    try:
        from llm_rl_scripts.chess.env.env import FenChessHistoryEnv
        from llm_rl_scripts.chess.env.env import large_piece_random_endgame
    except ImportError as e:
        raise ImportError(
            "Chess endgame environment requires LMRL-Gym chess module. "
            "Ensure LMRL-Gym is installed and python-chess + stockfish are available."
        ) from e

    pieces = kwargs.get("pieces", ["K", "Q", "k"])
    position = large_piece_random_endgame(pieces)

    return FenChessHistoryEnv(
        max_moves=kwargs.get("max_moves", 400),
        from_position=position,
        random_opponent=kwargs.get("random_opponent", False),
    )


class LMRLAdapter:
    """Adapter for LMRL-Gym text-based RL environments.

    LMRL-Gym provides 8 text-based environments for language model RL:
    maze navigation, Wordle, chess, twenty questions, guess the city,
    car dealer negotiation, and text navigation.

    Requires the LMRL-Gym package:
        pip install git+https://github.com/abdulhaim/LMRL-Gym.git

    The simplest way to use this adapter is to create a TextEnv instance
    from LMRL-Gym and pass it directly:

        >>> from llm_rl_scripts.wordle.env.env import WordleEnvironment
        >>> adapter = LMRLAdapter()
        >>> env = adapter.get_environment("wordle", text_env=wordle_env)

    Some environments (e.g., chess) can be auto-created from presets.
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "lmrl"

    def _get_lmrl(self) -> Any:
        """Import and return the LLM_RL.environment module."""
        try:
            import LLM_RL.environment

            return LLM_RL.environment
        except ImportError as e:
            raise ImportError(
                "LMRL-Gym is required for LMRLAdapter. "
                "Install with: pip install git+https://github.com/abdulhaim/LMRL-Gym.git\n"
                "See: https://github.com/abdulhaim/LMRL-Gym"
            ) from e

    def list_environments(self) -> list[str]:
        """List available environment presets.

        Returns:
            List of environment IDs for known LMRL-Gym environments.
        """
        return [f"lmrl:{name}" for name in sorted(LMRL_PRESETS.keys())]

    def get_environment(
        self,
        name: str = "lmrl",
        *,
        text_env: Any = None,
        max_steps: int | None = None,
        num_tasks: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> LMRLEnvironment:
        """Create an LMRL-Gym environment.

        Args:
            name: Environment ID. Format: "lmrl:{env_name}" or just
                the env name (e.g., "wordle", "chess").
            text_env: A pre-created TextEnv instance. If provided,
                this takes priority over auto-creation from presets.
            max_steps: Maximum steps per episode. If not provided, uses
                the preset default or 100.
            num_tasks: If set, enables task indexing (task_index is
                mapped to seed).
            extra_rewards: Additional reward functions.
            **kwargs: Passed to the preset factory (e.g., max_moves
                for chess, from_position for endgames).

        Returns:
            Configured LMRLEnvironment.

        Raises:
            ImportError: If LMRL-Gym is not installed.
            ValueError: If environment cannot be auto-created.
        """
        # Parse environment name
        clean_name = name.split(":", 1)[1] if ":" in name else name
        # Handle nested preset names like "chess:endgame"
        if clean_name.startswith("lmrl:"):
            clean_name = clean_name[5:]

        if text_env is None:
            # Try to auto-create from preset
            text_env = _create_text_env(clean_name, **kwargs)

        # Resolve max_steps from preset or default
        if max_steps is None:
            preset = LMRL_PRESETS.get(clean_name, {})
            max_steps = preset.get("max_steps", 100)

        return LMRLEnvironment(
            text_env=text_env,
            env_name=clean_name,
            max_steps=max_steps,
            num_tasks=num_tasks,
            extra_rewards=extra_rewards,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """LMRL-Gym environments provide their own context."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """LMRL-Gym manages prompts internally."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """LMRL-Gym does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str = "lmrl") -> dict[str, Any]:
        """Get metadata about the environment.

        Args:
            name: Environment ID.

        Returns:
            Dictionary with environment metadata.
        """
        clean_name = name.split(":", 1)[1] if ":" in name else name
        preset = LMRL_PRESETS.get(clean_name, {})

        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "description": preset.get(
                "description",
                "LMRL-Gym text-based RL environment",
            ),
            "environments": list(LMRL_PRESETS.keys()),
            "reference": "https://github.com/abdulhaim/LMRL-Gym",
        }
