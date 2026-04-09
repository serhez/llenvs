"""AlfWorld adapter - wraps AlfWorld household environments.

AlfWorld (Shridhar et al., ICLR 2021) provides text-based interactive
environments for household tasks, grounded in ALFRED's visual world.
Agents navigate rooms and manipulate objects using text commands.

Reference: https://github.com/alfworld/alfworld
"""

import os
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.extraction import AnswerExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)

ALFWORLD_TASK_TYPES: dict[int, str] = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

_TASK_TYPE_NAMES: set[str] = set(ALFWORLD_TASK_TYPES.values())

DEFAULT_ALFWORLD_PROMPTS: dict[str, str] = {
    "objective_prefix": "Objective: {objective}",
    "admissible_commands_prefix": "Admissible commands:",
}

_SPLIT_DATA_KEYS: dict[str, str] = {
    "train": "data_path",
    "eval_in_distribution": "eval_id_data_path",
    "eval_out_of_distribution": "eval_ood_data_path",
}


def _extract_objective(observation: str) -> str:
    """Extract the task objective from AlfWorld's initial observation.

    AlfWorld initial observations contain "Your task is to: <objective>".

    Args:
        observation: The raw initial observation text.

    Returns:
        The extracted objective, or empty string if not found.
    """
    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation)
    if match:
        return match.group(1).strip()
    return ""


def _extract_task_type(game_file: str) -> str:
    """Extract the task type from a game file path.

    Game files contain paths like:
    ``pick_and_place_simple-Mug-001/...``

    Args:
        game_file: The game file path.

    Returns:
        The task type string, or "unknown" if not parseable.
    """
    for task_type_name in _TASK_TYPE_NAMES:
        if task_type_name in game_file:
            return task_type_name
    return "unknown"


def _unbatch_admissible_commands(raw: Any) -> tuple[str, ...]:
    """Handle both batched and flat admissible-command formats.

    TextWorld's batched gym wrapper returns ``[["cmd1", "cmd2"]]``.
    The single-game wrapper (``batch_size=None``) already unbatches and
    returns ``["cmd1", "cmd2"]``.  Both formats must produce the same
    ``tuple[str, ...]`` result.

    Args:
        raw: Raw admissible_commands value from TextWorld infos.

    Returns:
        Tuple of command strings.
    """
    if not isinstance(raw, (list, tuple)) or not raw:
        return ()
    if isinstance(raw[0], (list, tuple)):
        return tuple(raw[0])  # batched: [["cmd1", "cmd2"]]
    return tuple(raw)  # flat: ["cmd1", "cmd2"]


@dataclass(frozen=True)
class AlfWorldHidden:
    """Hidden state for AlfWorld environment.

    Attributes:
        task_index: Index of the current task (game file).
        task_type: Task type (e.g. "pick_and_place_simple").
        objective: The extracted task objective.
        game_file: Path to the current game file.
        episode_step: Current step within the episode.
        last_action: The last action taken.
        admissible_commands: Currently valid commands.
        trajectory: Sequence of actions taken to reach this state.
    """

    task_index: int
    task_type: str
    objective: str
    game_file: str
    episode_step: int
    last_action: str | None
    admissible_commands: tuple[str, ...]
    trajectory: tuple[str, ...] = ()


@dataclass
class AlfWorldReward:
    """Reward function for AlfWorld based on task completion.

    AlfWorld provides binary sparse reward: 1.0 if the task is completed
    (won), 0.0 otherwise. Reward is only meaningful at episode end.
    """

    _name: str = "task_completion"
    _reward_type: RewardType = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[AlfWorldHidden],
        action: Action,
        next_state: State[AlfWorldHidden],
    ) -> Signal:
        """Compute reward from AlfWorld's native task completion signal."""
        won = next_state.metadata.info.get("won", False)
        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=1.0 if won else 0.0,
            metadata={"source": "alfworld"},
        )


class AlfWorldEnvironment:
    """MDP wrapper for AlfWorld household environments.

    AlfWorld is a multi-turn environment where agents navigate household
    rooms and manipulate objects to complete tasks like "put a clean mug
    on the desk".

    Actions are free-form text commands:
    - go to {recep}: Navigate to a receptacle
    - take {obj} from {recep}: Pick up an object
    - put {obj} in/on {recep}: Place an object
    - open/close {recep}: Open/close a container
    - use {obj}: Use an object (e.g., lamp)
    - clean/heat/cool {obj} with {recep}: Transform an object
    - examine {obj/recep}: Look at something
    - inventory: Check held items
    - look: Look around

    Example:
        >>> env = AlfWorldEnvironment(game_files=(...,), config={...})
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> print(state.observation.prompt)
        # Shows room description with objective

        >>> action = Action(text="go to shelf 1")
        >>> result = env.step(state, action)
    """

    def __init__(
        self,
        game_files: tuple[str, ...],
        config: dict[str, Any],
        max_steps: int = 50,
        include_admissible_commands: bool = True,
        include_objective_in_obs: bool = True,
        visual: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        answer_extractor: AnswerExtractor | None = None,
    ) -> None:
        """Initialize AlfWorld environment wrapper.

        Args:
            game_files: Tuple of game file paths from AlfWorld.
            config: AlfWorld config dict for re-initialization.
            max_steps: Maximum steps per episode before truncation.
            include_admissible_commands: Whether to append admissible
                commands to each observation.
            include_objective_in_obs: Whether to prepend the task
                objective to each observation.
            visual: Enable AI2-THOR visual rendering. When True,
                observations include ego-centric RGB frames as
                ``ImageContent`` alongside text. Requires ``ai2thor``.
            extra_rewards: Additional reward functions appended after
                native rewards.
            prompts: Override default prompt components. Keys:
                objective_prefix, admissible_commands_prefix.
            answer_extractor: Extractor applied to raw action text before
                passing the command to TextWorld. When set, the extracted
                command is used for both the TextWorld step and the
                conversation history, keeping the history clean of
                reasoning tokens.
        """
        self._game_files = game_files
        self._config = config
        self._max_steps = max_steps
        self._include_admissible_commands = include_admissible_commands
        self._include_objective_in_obs = include_objective_in_obs
        self._visual = visual
        self._native_rewards: tuple[RewardFunction, ...] = (AlfWorldReward(),)
        self._extra_rewards = extra_rewards
        self._prompts = {**DEFAULT_ALFWORLD_PROMPTS}
        if prompts:
            self._prompts.update(prompts)
        self._answer_extractor = answer_extractor

        # Cache registered env_ids to avoid TextWorld registry leak
        self._env_id_cache: dict[str, str] = {}

    @property
    def answer_extractor(self):
        """The extractor used to parse agent responses in ``step()``."""
        return self._answer_extractor

    @answer_extractor.setter
    def answer_extractor(self, value):
        self._answer_extractor = value

    def __len__(self) -> int:
        return len(self._game_files)

    @property
    def prompts(self) -> dict[str, str]:
        """Named prompt components used for building observations."""
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        """No tools available in AlfWorld environments."""
        return ()

    def _frame_to_image(self, frame: Any) -> ImageContent:
        """Convert an RGB frame (numpy array) to ImageContent.

        Args:
            frame: RGB array of shape (H, W, 3), dtype uint8.

        Returns:
            Base64-encoded PNG as ImageContent.
        """
        import base64
        import io

        import numpy as np
        from PIL import Image

        arr = np.asarray(frame, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return ImageContent(data=data, media_type="image/png")

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        desc = (
            "Visual household task environment (AI2-THOR)"
            if self._visual
            else "Text-based household task environment"
        )
        metadata: dict[str, Any] = {
            "num_games": len(self._game_files),
            "description": desc,
        }
        if self._visual:
            metadata["visual"] = True
        return EnvironmentSpec(
            name="alfworld",
            adapter="alfworld",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=True,
            metadata=metadata,
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[AlfWorldHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def _init_game(
        self, game_file: str
    ) -> tuple[Any, str, dict[str, Any], tuple[ImageContent, ...]]:
        """Create a fresh game environment instance.

        Registers the game file on first encounter (cached), then creates
        a new gym env and resets it. Caller is responsible for closing
        the returned gym env.

        Args:
            game_file: Path to the game file.

        Returns:
            Tuple of (gym_env, initial_observation_text, info_dict, images).
        """
        import textworld.gym

        if game_file not in self._env_id_cache:
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

            request_infos = textworld.EnvInfos(
                won=True,
                admissible_commands=True,
            )

            env_id = textworld.gym.register_games(
                [game_file],
                request_infos=request_infos,
                max_episode_steps=self._max_steps,
                wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
            )
            self._env_id_cache[game_file] = env_id

        gym_env = textworld.gym.make(self._env_id_cache[game_file])
        obs, infos = gym_env.reset()

        # textworld returns batched results (lists)
        raw_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
        info = {}
        if isinstance(infos, dict):
            for k, v in infos.items():
                if k == "admissible_commands":
                    info[k] = _unbatch_admissible_commands(v)
                elif isinstance(v, (list, tuple)):
                    info[k] = v[0]
                else:
                    info[k] = v

        # Extract text and images from observation
        images: tuple[ImageContent, ...] = ()
        if self._visual and isinstance(raw_obs, dict):
            raw_text = raw_obs.get("text", str(raw_obs))
            frame = raw_obs.get("rgb")
            if frame is not None:
                images = (self._frame_to_image(frame),)
            raw_obs = raw_text
        elif isinstance(raw_obs, dict):
            raw_obs = raw_obs.get("text", str(raw_obs))

        return gym_env, raw_obs, info, images

    def _build_observation_prompt(
        self,
        raw_obs: str,
        objective: str,
        admissible_commands: tuple[str, ...],
    ) -> str:
        """Build the full observation prompt for the model.

        Args:
            raw_obs: Raw observation from AlfWorld.
            objective: The task objective.
            admissible_commands: Currently valid commands.

        Returns:
            Formatted observation string.
        """
        parts = []

        if self._include_objective_in_obs and objective:
            prefix = self._prompts["objective_prefix"]
            parts.append(prefix.format(objective=objective))
            parts.append("")

        parts.append(raw_obs)

        if self._include_admissible_commands and admissible_commands:
            commands_prefix = self._prompts["admissible_commands_prefix"]
            parts.append("")
            parts.append(commands_prefix)
            for cmd in admissible_commands:
                parts.append(f"  - {cmd}")

        return "\n".join(parts)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[AlfWorldHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Ignored (AlfWorld games are deterministic per game file).
            options: Environment-specific options.
                - task_index: Select specific game file (0-indexed).

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index", 0)

        if task_index < 0 or task_index >= len(self._game_files):
            raise IndexError(f"task_index {task_index} out of range [0, {len(self._game_files)})")

        game_file = self._game_files[task_index]

        # Initialize the game
        gym_env, raw_obs, init_info, images = self._init_game(game_file)
        gym_env.close()

        # Extract objective and task type
        objective = _extract_objective(raw_obs)
        task_type = _extract_task_type(game_file)

        # Extract admissible commands
        admissible_commands = tuple(init_info.get("admissible_commands", ()))

        # Build observation
        obs_prompt = self._build_observation_prompt(raw_obs, objective, admissible_commands)

        hidden = AlfWorldHidden(
            task_index=task_index,
            task_type=task_type,
            objective=objective,
            game_file=game_file,
            episode_step=0,
            last_action=None,
            admissible_commands=admissible_commands,
            trajectory=(),
        )

        # Task = objective (static); State = room description + commands (dynamic)
        task_text = ""
        if objective:
            prefix = self._prompts["objective_prefix"]
            task_text = prefix.format(objective=objective)

        state_parts = [raw_obs]
        if self._include_admissible_commands and admissible_commands:
            commands_prefix = self._prompts["admissible_commands_prefix"]
            state_parts.append("")
            state_parts.append(commands_prefix)
            for cmd in admissible_commands:
                state_parts.append(f"  - {cmd}")
        state_text = "\n".join(state_parts)

        observation = Observation(
            prompt=obs_prompt,
            images=images,
            task=ObservationContent(text=task_text),
            state=ObservationContent(text=state_text, images=images),
        )

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "task_index": task_index,
                "task_type": task_type,
                "objective": objective,
                "game_file": game_file,
                "won": False,
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)

        return state, {
            "task_index": task_index,
            "task_type": task_type,
            "objective": objective,
            "game_file": game_file,
        }

    def _text_for_history(self, raw_text: str, extracted_cmd: str | None) -> str:
        """Return the text to store as the assistant turn in conversation history.

        Uses the extracted command when available. On extraction failure, applies
        the extractor's pre-cleaners (e.g. ``strip_thinking_tokens``) to the raw
        text so that reasoning tokens never appear in the history.

        Args:
            raw_text: The original ``action.text``.
            extracted_cmd: Result of extraction, or ``None`` if extraction failed.

        Returns:
            Cleaned text suitable for the conversation history.
        """
        if extracted_cmd is not None:
            return extracted_cmd
        if self._answer_extractor is None:
            return raw_text
        from llenvs.core.extraction import CleanedExtractor
        if isinstance(self._answer_extractor, CleanedExtractor):
            cleaned = raw_text
            for cleaner in self._answer_extractor.pre_cleaners:
                cleaned = cleaner(cleaned)
            return cleaned
        return raw_text

    def step(
        self,
        state: State[AlfWorldHidden],
        action: Action,
    ) -> StepResult[AlfWorldHidden]:
        """Take an action from the given state.

        Args:
            state: Current state.
            action: Action to take (free-form text command).

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        action_text = action.text or ""

        # Extract clean command for TextWorld (strips reasoning tokens etc.)
        extracted_cmd: str | None = None
        if self._answer_extractor is not None and action.text:
            extracted_cmd, _ = self._answer_extractor.extract(action.text)
        cmd_for_env = extracted_cmd or action_text

        # Create fresh env and replay trajectory to reach current state
        gym_env, _, _, _ = self._init_game(state.hidden.game_file)
        for cmd in state.hidden.trajectory:
            gym_env.step(cmd)

        # Apply new action
        obs, scores, dones, infos = gym_env.step(cmd_for_env)
        gym_env.close()

        # Unbatch results
        raw_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
        won = False
        admissible_commands: tuple[str, ...] = ()

        if isinstance(infos, dict):
            won_val = infos.get("won", False)
            won = won_val[0] if isinstance(won_val, (list, tuple)) else won_val
            ac_val = infos.get("admissible_commands", ())
            admissible_commands = _unbatch_admissible_commands(ac_val)

        # Extract text and images from observation
        images: tuple[ImageContent, ...] = ()
        if self._visual and isinstance(raw_obs, dict):
            obs_text = raw_obs.get("text", str(raw_obs))
            frame = raw_obs.get("rgb")
            if frame is not None:
                images = (self._frame_to_image(frame),)
        elif isinstance(raw_obs, dict):
            obs_text = raw_obs.get("text", str(raw_obs))
        else:
            obs_text = raw_obs

        # Check truncation
        next_step = state.hidden.episode_step + 1
        terminated = bool(won)
        truncated = next_step >= self._max_steps and not terminated

        # Build next observation
        obs_prompt = self._build_observation_prompt(
            obs_text, state.hidden.objective, admissible_commands
        )

        new_hidden = AlfWorldHidden(
            task_index=state.hidden.task_index,
            task_type=state.hidden.task_type,
            objective=state.hidden.objective,
            game_file=state.hidden.game_file,
            episode_step=next_step,
            last_action=cmd_for_env,
            admissible_commands=admissible_commands,
            trajectory=(*state.hidden.trajectory, cmd_for_env),
        )

        user_msg: dict[str, Any] = {"role": "user", "content": obs_prompt}
        if images:
            user_msg["images"] = [
                {"data": img.data, "media_type": img.media_type} for img in images
            ]

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": self._text_for_history(action_text, extracted_cmd)},
            user_msg,
        )
        state_parts = [obs_text]
        if self._include_admissible_commands and admissible_commands:
            commands_prefix = self._prompts["admissible_commands_prefix"]
            state_parts.append("")
            state_parts.append(commands_prefix)
            for cmd in admissible_commands:
                state_parts.append(f"  - {cmd}")
        state_text = "\n".join(state_parts)

        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            images=images,
            task=state.observation.task,
            state=ObservationContent(text=state_text, images=images),
        )

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                **state.metadata.info,
                "won": won,
                "last_action": action_text,
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        # Compute rewards
        rewards = self.compute_rewards(state, action, next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            extracted_action=extracted_cmd,
            resolved_action=extracted_cmd,
            info={
                "won": won,
                "action": cmd_for_env,
                "admissible_commands": admissible_commands,
            },
        )

    def compute_rewards(
        self,
        state: State[AlfWorldHidden],
        action: Action,
        next_state: State[AlfWorldHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Clean up cached registrations."""
        self._env_id_cache.clear()


class AlfWorldAdapter:
    """Adapter for AlfWorld household environments.

    AlfWorld provides text-based interactive household tasks grounded in
    ALFRED's visual world. Agents navigate rooms and manipulate objects.

    Requires the alfworld package: pip install alfworld

    Example:
        >>> adapter = AlfWorldAdapter()
        >>> env = adapter.get_environment("alfworld:eval_out_of_distribution")
        >>> state, _ = env.reset(options={"task_index": 0})
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "alfworld"

    def _get_alfworld(self) -> Any:
        """Import and return the alfworld module."""
        try:
            import alfworld
            import alfworld.agents.environment

            return alfworld, alfworld.agents.environment
        except ImportError as e:
            raise ImportError(
                "AlfWorld is required for AlfWorldAdapter. "
                "Install with: pip install alfworld\n"
                "See: https://github.com/alfworld/alfworld"
            ) from e

    def _build_default_config(self, alfworld_mod: Any) -> dict[str, Any]:
        """Build a default ALFWorld config from packaged constants."""
        data_root = str(alfworld_mod.ALFWORLD_DATA).rstrip("/")
        dataset_root = f"{data_root}/json_2.1.1"

        return {
            "dataset": {
                "data_path": f"{dataset_root}/train",
                "eval_id_data_path": f"{dataset_root}/valid_seen",
                "eval_ood_data_path": f"{dataset_root}/valid_unseen",
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": str(alfworld_mod.ALFRED_PDDL_PATH),
                "grammar": str(alfworld_mod.ALFRED_TWL2_PATH),
            },
            "env": {
                "type": "AlfredTWEnv",
                "domain_randomization": False,
                "task_types": list(ALFWORLD_TASK_TYPES.keys()),
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
            },
            "general": {
                "training_method": "dagger",
            },
            "dagger": {
                "training": {
                    "max_nb_steps_per_episode": 50,
                }
            },
            "rl": {
                "training": {
                    "max_nb_steps_per_episode": 50,
                }
            },
        }

    def _merge_config(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override values into a config dict copy."""
        merged = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def _resolve_config(
        self,
        alfworld_mod: Any,
        *,
        config: dict[str, Any] | None,
        config_path: str | None,
    ) -> dict[str, Any]:
        """Resolve adapter config with stable defaults and overrides."""
        resolved_config = self._build_default_config(alfworld_mod)

        if config is not None:
            resolved_config = self._merge_config(resolved_config, config)

        if config_path is not None:
            import yaml

            with open(config_path) as f:
                file_config = yaml.safe_load(f) or {}
            resolved_config = self._merge_config(resolved_config, file_config)

        return resolved_config

    def _get_textworld_env_class(self, alfworld_env_mod: Any) -> Any:
        """Resolve the upstream text-only ALFWorld environment class."""
        get_environment = getattr(alfworld_env_mod, "get_environment", None)
        if callable(get_environment):
            return get_environment("AlfredTWEnv")

        try:
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

            return AlfredTWEnv
        except ImportError as e:
            raise ImportError(
                "Could not resolve ALFWorld text environment class. Expected "
                "alfworld.agents.environment.get_environment('AlfredTWEnv') "
                "or alfworld.agents.environment.alfred_tw_env.AlfredTWEnv."
            ) from e

    def list_environments(self) -> list[str]:
        """List available environment variants.

        Returns:
            List of environment IDs (split variants).
        """
        return [
            "alfworld:train",
            "alfworld:eval_in_distribution",
            "alfworld:eval_out_of_distribution",
        ]

    def get_environment(
        self,
        name: str = "alfworld:eval_out_of_distribution",
        config: dict[str, Any] | None = None,
        config_path: str | None = None,
        split: str | None = None,
        task_types: list[int] | None = None,
        max_steps: int = 50,
        include_admissible_commands: bool = True,
        include_objective_in_obs: bool = True,
        visual: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        answer_extractor: AnswerExtractor | None = None,
        **kwargs: Any,
    ) -> AlfWorldEnvironment:
        """Create an AlfWorld environment.

        Args:
            name: Environment ID. Format: "alfworld:{split}".
            config: AlfWorld config dict (overrides default config).
            config_path: Path to AlfWorld config YAML. Takes priority
                over config dict.
            split: Data split. Parsed from name if not given. One of
                "train", "eval_in_distribution", "eval_out_of_distribution".
            task_types: Filter to specific task type IDs (1-6).
                See ALFWORLD_TASK_TYPES for mapping.
            max_steps: Maximum steps per episode.
            include_admissible_commands: Whether to include admissible
                commands in observations.
            include_objective_in_obs: Whether to prepend objective
                to observations.
            visual: Enable AI2-THOR visual rendering. When True,
                uses ``AlfredThorEnv`` for 3D rendering and includes
                RGB frames as ``ImageContent`` in observations.
                Requires ``ai2thor`` (GPU + OpenGL).
            extra_rewards: Additional reward functions.
            prompts: Override default prompt components.
            answer_extractor: Extractor applied to raw action text before
                passing the command to TextWorld.
            **kwargs: Additional arguments (unused).

        Returns:
            Configured AlfWorldEnvironment.

        Raises:
            ImportError: If alfworld is not installed.
            ValueError: If split or task_types are invalid.
        """
        alfworld_mod, alfworld_env_mod = self._get_alfworld()

        # Resolve split from name
        if split is None:
            if ":" in name:
                split = name.split(":", 1)[1]
            else:
                split = "eval_out_of_distribution"

        valid_splits = {"train", "eval_in_distribution", "eval_out_of_distribution"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split!r}. Must be one of: {valid_splits}")

        resolved_config = self._resolve_config(
            alfworld_mod,
            config=config,
            config_path=config_path,
        )

        # Set the split
        resolved_config.setdefault("env", {})
        resolved_config["env"]["train_eval"] = split

        # Validate data directory exists before constructing the env
        data_key = _SPLIT_DATA_KEYS[split]
        data_path = resolved_config.get("dataset", {}).get(data_key, "")
        if not os.path.isdir(data_path):
            alfworld_data = getattr(alfworld_mod, "ALFWORLD_DATA", "unknown")
            raise ValueError(
                f"ALFWorld data directory does not exist: {data_path}\n"
                f"ALFWORLD_DATA is set to: {alfworld_data}\n"
                "Run `alfworld-download` to fetch game data (~3.5 GB)."
            )

        # Load ALFWorld text environment to get game files
        tw_env_cls = self._get_textworld_env_class(alfworld_env_mod)
        tw_env = tw_env_cls(resolved_config, train_eval=split)
        game_files = tuple(tw_env.game_files)

        # Filter by task types if specified
        if task_types is not None:
            valid_type_ids = set(ALFWORLD_TASK_TYPES.keys())
            invalid = set(task_types) - valid_type_ids
            if invalid:
                raise ValueError(
                    f"Invalid task type IDs: {invalid}. Valid IDs: {sorted(valid_type_ids)}"
                )
            type_names = {ALFWORLD_TASK_TYPES[tid] for tid in task_types}
            game_files = tuple(gf for gf in game_files if _extract_task_type(gf) in type_names)

        if not game_files:
            details = f"split={split!r}"
            if task_types is not None:
                details += f", task_types={task_types!r}"
            raise ValueError(
                "No ALFWorld games found for the requested configuration "
                f"({details}). Scanned data path: {data_path}. "
                "This usually means the split has no matching games "
                "or the game-file layout does not match the expected task-type naming."
            )

        return AlfWorldEnvironment(
            game_files=game_files,
            config=resolved_config,
            max_steps=max_steps,
            include_admissible_commands=include_admissible_commands,
            include_objective_in_obs=include_objective_in_obs,
            visual=visual,
            extra_rewards=extra_rewards,
            prompts=prompts,
            answer_extractor=answer_extractor,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """AlfWorld observations include built-in context."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """AlfWorld manages multi-turn prompts internally."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """AlfWorld does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str = "alfworld") -> dict[str, Any]:
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
                "AlfWorld: Text-based household tasks grounded in ALFRED. "
                "Agent navigates rooms and manipulates objects to complete tasks."
            ),
            "task_types": dict(ALFWORLD_TASK_TYPES),
            "actions": [
                "go to {recep}",
                "take {obj} from {recep}",
                "put {obj} in/on {recep}",
                "open/close {recep}",
                "use {obj}",
                "clean/heat/cool {obj} with {recep}",
                "examine {obj/recep}",
                "inventory",
                "look",
            ],
            "reference": "https://github.com/alfworld/alfworld",
        }
