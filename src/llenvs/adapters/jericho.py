"""Jericho adapter - wraps classic interactive fiction games.

Jericho (Microsoft) provides a Python interface to 50+ classic text adventure
games (Zork, Hitchhiker's Guide, Detective, etc.) via a Z-Machine emulator.
Agents interact through free-form text commands and receive score-based rewards.

Reference: https://github.com/microsoft/jericho
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata

DEFAULT_JERICHO_PROMPTS: dict[str, str] = {
    "valid_actions_prefix": "Valid actions:",
}


def _game_name_from_path(path: str) -> str:
    """Extract game name from a ROM file path.

    Args:
        path: Path to a game ROM file (e.g., "/games/zork1.z5").

    Returns:
        The game name (stem without extension).
    """
    return Path(path).stem


def _list_bundled_games() -> dict[str, str]:
    """List all bundled games available in jericho's package data.

    Returns:
        Dict mapping game name to ROM file path.
    """
    import jericho

    games_dir = Path(jericho.__file__).parent / "games"
    if not games_dir.is_dir():
        return {}

    result: dict[str, str] = {}
    for f in sorted(games_dir.iterdir()):
        if f.suffix in (".z1", ".z2", ".z3", ".z4", ".z5", ".z6", ".z7", ".z8"):
            result[f.stem] = str(f)
    return result


@dataclass(frozen=True)
class JerichoHidden:
    """Hidden state for Jericho environment.

    Attributes:
        task_index: Index of the current task (game).
        game_name: Game identifier (e.g., "zork1").
        game_file: Path to the ROM file.
        episode_step: Current step within the episode.
        last_action: The last action taken.
        score: Current cumulative score.
        max_score: Maximum achievable score.
        moves: Move counter from Jericho.
        valid_actions: Currently valid actions.
    """

    task_index: int
    game_name: str
    game_file: str
    episode_step: int
    last_action: str | None
    score: int
    max_score: int
    moves: int
    valid_actions: tuple[str, ...]


@dataclass
class JerichoReward:
    """Reward function for Jericho based on game score.

    Intermediate steps produce STEP rewards with the score delta.
    Terminal steps produce OUTCOME rewards with the normalized score
    (score / max_score).
    """

    _name: str = "game_score"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.STEP

    def compute(
        self,
        state: State[JerichoHidden],
        action: Action,
        next_state: State[JerichoHidden],
    ) -> Signal:
        """Compute reward from Jericho's native score signal."""
        info = next_state.metadata.info
        is_terminal = next_state.metadata.is_terminal

        if is_terminal:
            score = info.get("score", 0)
            max_score = info.get("max_score", 0)
            normalized = score / max_score if max_score > 0 else 0.0
            return Signal(
                name=self.name,
                reward_type=RewardType.OUTCOME,
                reward=normalized,
                metadata={"source": "jericho", "score": score, "max_score": max_score},
            )
        else:
            score_delta = info.get("score_delta", 0)
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=score_delta,
                metadata={"source": "jericho", "score_delta": score_delta},
            )


class JerichoEnvironment:
    """MDP wrapper for Jericho interactive fiction environments.

    Jericho is a multi-turn environment where agents play classic text
    adventure games by issuing free-form text commands.

    Actions are free-form text commands:
    - open mailbox: Interact with objects
    - go north/south/east/west: Navigate
    - take {obj}: Pick up an object
    - look: Look around
    - inventory: Check held items

    Example:
        >>> env = JerichoEnvironment(game_files=(...,), game_names=(...,))
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> print(state.observation.prompt)
        # Shows initial game text

        >>> action = Action(text="open mailbox")
        >>> result = env.step(state, action)
    """

    def __init__(
        self,
        game_files: tuple[str, ...],
        game_names: tuple[str, ...],
        max_steps: int = 100,
        include_valid_actions: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
    ) -> None:
        """Initialize Jericho environment wrapper.

        Args:
            game_files: Tuple of game ROM file paths.
            game_names: Tuple of game names corresponding to game_files.
            max_steps: Maximum steps per episode before truncation.
            include_valid_actions: Whether to append valid actions to
                each observation. Defaults to False (wrapper fidelity).
            extra_rewards: Additional reward functions appended after
                native rewards.
            prompts: Override default prompt components. Keys:
                valid_actions_prefix.
        """
        self._game_files = game_files
        self._game_names = game_names
        self._max_steps = max_steps
        self._include_valid_actions = include_valid_actions
        self._native_rewards: tuple[RewardFunction, ...] = (JerichoReward(),)
        self._extra_rewards = extra_rewards
        self._prompts = {**DEFAULT_JERICHO_PROMPTS}
        if prompts:
            self._prompts.update(prompts)
        self._state_tracker = _StateContinuityTracker()

        # Current FrotzEnv instance (re-created per task)
        self._frotz_env: Any = None
        self._prev_score: int = 0

    def __len__(self) -> int:
        return len(self._game_files)

    @property
    def prompts(self) -> dict[str, str]:
        """Named prompt components used for building observations."""
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        """No tools available in Jericho environments."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name="jericho",
            adapter="jericho",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=True,
            pure_step=False,
            metadata={
                "num_games": len(self._game_files),
                "description": "Classic interactive fiction text adventure",
            },
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[JerichoHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def _init_game(self, game_file: str) -> tuple[str, dict[str, Any]]:
        """Initialize a single game via Jericho's FrotzEnv.

        Args:
            game_file: Path to the game ROM file.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        import jericho

        if self._frotz_env is not None:
            self._frotz_env.close()

        self._frotz_env = jericho.FrotzEnv(game_file)
        obs = self._frotz_env.reset()

        info = {
            "score": self._frotz_env.get_score(),
            "max_score": self._frotz_env.get_max_score(),
            "moves": self._frotz_env.get_moves(),
            "done": False,
        }

        return obs, info

    def _build_observation_prompt(
        self,
        raw_obs: str,
        valid_actions: tuple[str, ...],
    ) -> str:
        """Build the full observation prompt for the model.

        Args:
            raw_obs: Raw observation from Jericho.
            valid_actions: Currently valid actions.

        Returns:
            Formatted observation string.
        """
        parts = [raw_obs]

        if self._include_valid_actions and valid_actions:
            prefix = self._prompts["valid_actions_prefix"]
            parts.append("")
            parts.append(prefix)
            for cmd in valid_actions:
                parts.append(f"  - {cmd}")

        return "\n".join(parts)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[JerichoHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed for the game.
            options: Environment-specific options.
                - task_index: Select specific game (0-indexed).

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index", 0)

        if task_index < 0 or task_index >= len(self._game_files):
            raise IndexError(f"task_index {task_index} out of range [0, {len(self._game_files)})")

        game_file = self._game_files[task_index]
        game_name = self._game_names[task_index]

        # Initialize the game
        raw_obs, init_info = self._init_game(game_file)

        # Seed if requested
        if seed is not None:
            self._frotz_env.seed(seed)

        # Get valid actions
        valid_actions = tuple(self._frotz_env.get_valid_actions())

        # Track score
        self._prev_score = init_info["score"]

        # Build observation
        obs_prompt = self._build_observation_prompt(raw_obs, valid_actions)

        # Task = synthetic description (static); State = game text (dynamic)
        task_text = (
            f"You are playing {game_name}, a classic interactive fiction game. "
            f"Use text commands to explore, interact with objects, and solve puzzles."
        )

        hidden = JerichoHidden(
            task_index=task_index,
            game_name=game_name,
            game_file=game_file,
            episode_step=0,
            last_action=None,
            score=init_info["score"],
            max_score=init_info["max_score"],
            moves=init_info["moves"],
            valid_actions=valid_actions,
        )

        observation = Observation(
            prompt=obs_prompt,
            task=ObservationContent(text=task_text),
            state=ObservationContent(text=obs_prompt),
        )

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "task_index": task_index,
                "game_name": game_name,
                "game_file": game_file,
                "score": init_info["score"],
                "max_score": init_info["max_score"],
                "done": False,
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "game_name": game_name,
            "game_file": game_file,
            "score": init_info["score"],
            "max_score": init_info["max_score"],
        }

    def step(
        self,
        state: State[JerichoHidden],
        action: Action,
    ) -> StepResult[JerichoHidden]:
        """Take an action from the given state.

        Args:
            state: Current state.
            action: Action to take (free-form text command).

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        self._state_tracker.validate(state, "JerichoEnvironment")

        # Step Jericho environment
        raw_obs, reward, done, info = self._frotz_env.step(action.text)

        # Get current score and valid actions
        current_score = self._frotz_env.get_score()
        max_score = self._frotz_env.get_max_score()
        moves = self._frotz_env.get_moves()
        score_delta = current_score - self._prev_score
        self._prev_score = current_score

        valid_actions = tuple(self._frotz_env.get_valid_actions())

        # Check termination/truncation
        next_step = state.hidden.episode_step + 1
        terminated = bool(done)
        truncated = next_step >= self._max_steps and not terminated

        # Build next observation
        obs_prompt = self._build_observation_prompt(raw_obs, valid_actions)

        new_hidden = JerichoHidden(
            task_index=state.hidden.task_index,
            game_name=state.hidden.game_name,
            game_file=state.hidden.game_file,
            episode_step=next_step,
            last_action=action.text,
            score=current_score,
            max_score=max_score,
            moves=moves,
            valid_actions=valid_actions,
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": obs_prompt},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=obs_prompt),
        )

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                **state.metadata.info,
                "score": current_score,
                "max_score": max_score,
                "score_delta": score_delta,
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
                "score": current_score,
                "max_score": max_score,
                "score_delta": score_delta,
                "done": done,
                "action": action.text,
                "valid_actions": valid_actions,
            },
        )

    def compute_rewards(
        self,
        state: State[JerichoHidden],
        action: Action,
        next_state: State[JerichoHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Close the underlying FrotzEnv."""
        if self._frotz_env is not None:
            self._frotz_env.close()
            self._frotz_env = None


class JerichoAdapter:
    """Adapter for Jericho interactive fiction environments.

    Jericho provides access to 50+ classic text adventure games via a
    Z-Machine emulator. Agents interact through free-form text commands.

    Requires the jericho package: pip install jericho

    Example:
        >>> adapter = JerichoAdapter()
        >>> env = adapter.get_environment("jericho:zork1")
        >>> state, _ = env.reset(options={"task_index": 0})
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "jericho"

    def _get_jericho(self) -> Any:
        """Import and return the jericho module."""
        try:
            import jericho

            return jericho
        except ImportError as e:
            raise ImportError(
                "Jericho is required for JerichoAdapter. "
                "Install with: pip install jericho\n"
                "See: https://github.com/microsoft/jericho"
            ) from e

    def list_environments(self) -> list[str]:
        """List available environment variants.

        Returns:
            List of environment IDs (one per bundled game).
        """
        self._get_jericho()
        bundled = _list_bundled_games()
        return [f"jericho:{name}" for name in sorted(bundled.keys())]

    def get_environment(
        self,
        name: str = "jericho",
        games: list[str] | None = None,
        game_files: list[str] | None = None,
        max_steps: int = 100,
        include_valid_actions: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> JerichoEnvironment:
        """Create a Jericho environment.

        Args:
            name: Environment ID. Format: "jericho" (all games) or
                "jericho:{game_name}" (specific game).
            games: Game names to include (e.g., ["zork1", "detective"]).
                Overrides name parsing.
            game_files: Direct ROM file paths. Overrides both name
                and games parameters.
            max_steps: Maximum steps per episode.
            include_valid_actions: Whether to include valid actions
                in observations. Defaults to False (wrapper fidelity).
            extra_rewards: Additional reward functions.
            prompts: Override default prompt components.
            **kwargs: Additional arguments (unused).

        Returns:
            Configured JerichoEnvironment.

        Raises:
            ImportError: If jericho is not installed.
            ValueError: If a game name is unknown.
        """
        self._get_jericho()

        if game_files is not None:
            # Direct file paths provided
            resolved_files = tuple(game_files)
            resolved_names = tuple(_game_name_from_path(f) for f in game_files)
        else:
            bundled = _list_bundled_games()

            if games is not None:
                # Specific game names provided
                resolved_files_list = []
                resolved_names_list = []
                for g in games:
                    if g not in bundled:
                        raise ValueError(
                            f"Unknown game: {g!r}. Available: {sorted(bundled.keys())}"
                        )
                    resolved_files_list.append(bundled[g])
                    resolved_names_list.append(g)
                resolved_files = tuple(resolved_files_list)
                resolved_names = tuple(resolved_names_list)
            elif ":" in name:
                # Parse game name from environment ID
                game_name = name.split(":", 1)[1]
                if game_name not in bundled:
                    raise ValueError(
                        f"Unknown game: {game_name!r}. Available: {sorted(bundled.keys())}"
                    )
                resolved_files = (bundled[game_name],)
                resolved_names = (game_name,)
            else:
                # All bundled games
                resolved_files = tuple(bundled[k] for k in sorted(bundled.keys()))
                resolved_names = tuple(sorted(bundled.keys()))

        return JerichoEnvironment(
            game_files=resolved_files,
            game_names=resolved_names,
            max_steps=max_steps,
            include_valid_actions=include_valid_actions,
            extra_rewards=extra_rewards,
            prompts=prompts,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """Jericho games provide their own context."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """Jericho manages multi-turn prompts internally."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """Jericho does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str = "jericho") -> dict[str, Any]:
        """Get metadata about the environment.

        Args:
            name: Environment ID.

        Returns:
            Dictionary with environment metadata.
        """
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "description": (
                "Jericho: Classic interactive fiction text adventures. "
                "Agent plays games like Zork via free-form text commands."
            ),
            "actions": [
                "open/close {obj}",
                "take/drop {obj}",
                "go north/south/east/west/up/down",
                "look",
                "inventory",
                "examine {obj}",
                "put {obj} in/on {obj}",
                "turn on/off {obj}",
            ],
            "reference": "https://github.com/microsoft/jericho",
        }
