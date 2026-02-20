"""Gymnasium adapter - wraps gymnasium environments for text-based LLM agents.

Bridges the gap between gymnasium's numeric observations/actions and LLM
text interfaces via ObservationMapper and ActionMapper protocols. Built-in
auto mappers handle common space types; users provide custom implementations
for complex environments.

Third-party gymnasium-compatible envs (e.g., Gym4Real) work out of the box
via ``gymnasium.make()``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.extraction import AnswerExtractor, RawGenerationExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)

# =============================================================================
# Mapper Protocols
# =============================================================================


@runtime_checkable
class ObservationMapper(Protocol):
    """Convert gym observations to text."""

    def map(self, obs: Any, info: dict[str, Any]) -> str:
        """Convert gym observation to text."""
        ...

    def describe(self) -> str:
        """Describe the observation space for the initial prompt."""
        ...


@runtime_checkable
class MultimodalObservationMapper(Protocol):
    """Convert gym observations to text + images.

    Unlike ObservationMapper which returns only text, this protocol
    returns both text and ImageContent objects for multimodal observations
    (e.g., pixel/image-based gymnasium environments).

    Implementors must set ``_multimodal = True`` as a class attribute
    to distinguish from text-only ObservationMapper.
    """

    _multimodal: bool

    def map(self, obs: Any, info: dict[str, Any]) -> tuple[str, tuple[ImageContent, ...]]:
        """Convert gym observation to text and images.

        Args:
            obs: Raw gymnasium observation.
            info: Step info dict from gymnasium.

        Returns:
            Tuple of (text_description, image_contents).
        """
        ...

    def describe(self) -> str:
        """Describe the observation space for the initial prompt."""
        ...


@runtime_checkable
class ActionMapper(Protocol):
    """Convert extracted action text to a gymnasium action."""

    def map(self, text: str) -> Any:
        """Convert extracted action text to a gymnasium action."""
        ...

    def describe(self) -> str:
        """Describe valid actions for the initial prompt."""
        ...


# =============================================================================
# Auto Observation Mapper
# =============================================================================


class AutoObservationMapper:
    """Automatically maps gymnasium observations to text based on space type.

    Supports Discrete, Box (1D), MultiDiscrete, MultiBinary, Text, Dict,
    and Tuple spaces. Box spaces with ndim >= 2 raise ValueError at
    construction (use a custom mapper for image observations).

    Args:
        space: The gymnasium observation space.
        labels: Optional mapping from dimension index/key to human name.
        precision: Decimal places for floating-point values.
    """

    def __init__(
        self,
        space: Any,
        labels: dict[int | str, str] | None = None,
        precision: int = 2,
    ) -> None:
        import gymnasium.spaces as spaces

        self._space = space
        self._labels = labels or {}
        self._precision = precision
        self._spaces = spaces

        # Validate: reject 2D+ Box at construction
        if isinstance(space, spaces.Box) and len(space.shape) >= 2:
            raise ValueError(
                f"AutoObservationMapper does not support Box spaces with "
                f"ndim >= 2 (got shape {space.shape}). Use a custom mapper "
                f"(e.g., GridObservationMapper for grid worlds)."
            )

    def map(self, obs: Any, info: dict[str, Any]) -> str:
        spaces = self._spaces
        return self._map_space(self._space, obs)

    def _map_space(self, space: Any, obs: Any) -> str:
        spaces = self._spaces

        if isinstance(space, spaces.Text):
            return str(obs)

        if isinstance(space, spaces.Discrete):
            label = self._labels.get(obs, str(obs))
            return f"State: {label}"

        if isinstance(space, spaces.Box):
            return self._map_box(obs)

        if isinstance(space, spaces.MultiDiscrete):
            return self._map_array(obs, "MultiDiscrete")

        if isinstance(space, spaces.MultiBinary):
            return self._map_array(obs, "MultiBinary")

        if isinstance(space, spaces.Dict):
            parts = []
            for key, subspace in space.items():
                val = obs[key]
                sub_mapper = AutoObservationMapper(subspace, precision=self._precision)
                rendered = sub_mapper._map_space(subspace, val)
                parts.append(f"{key}: {rendered}")
            return "\n".join(parts)

        if isinstance(space, spaces.Tuple):
            parts = []
            for i, subspace in enumerate(space.spaces):
                val = obs[i]
                sub_mapper = AutoObservationMapper(subspace, precision=self._precision)
                rendered = sub_mapper._map_space(subspace, val)
                parts.append(f"[{i}]: {rendered}")
            return "\n".join(parts)

        # Fallback
        return str(obs)

    def _map_box(self, obs: Any) -> str:
        obs = np.asarray(obs).flatten()
        ndim = len(obs)

        if ndim > 20:
            return (
                f"[{ndim}-dim vector, "
                f"min={obs.min():.{self._precision}f}, "
                f"max={obs.max():.{self._precision}f}]"
            )

        parts = []
        for i, val in enumerate(obs):
            label = self._labels.get(i, f"dim_{i}")
            parts.append(f"{label}: {val:.{self._precision}f}")
        return ", ".join(parts)

    def _map_array(self, obs: Any, space_type: str) -> str:
        obs = np.asarray(obs)
        parts = []
        for i, val in enumerate(obs):
            label = self._labels.get(i, f"dim_{i}")
            parts.append(f"{label}: {val}")
        return ", ".join(parts)

    def describe(self) -> str:
        spaces = self._spaces

        if isinstance(self._space, spaces.Text):
            return "Text observation."

        if isinstance(self._space, spaces.Discrete):
            n = self._space.n
            if self._labels:
                vals = ", ".join(f"{k}: {v}" for k, v in sorted(self._labels.items()))
                return f"Discrete state (one of {n} values): {vals}"
            return f"Discrete state with {n} possible values (0 to {n - 1})."

        if isinstance(self._space, spaces.Box):
            shape = self._space.shape
            ndim = shape[0] if len(shape) == 1 else int(np.prod(shape))
            if self._labels:
                dims = ", ".join(self._labels.get(i, f"dim_{i}") for i in range(min(ndim, 20)))
                return f"Continuous vector ({ndim} dims): {dims}"
            return f"Continuous vector with {ndim} dimensions."

        if isinstance(self._space, spaces.MultiDiscrete):
            return f"MultiDiscrete with dimensions: {list(self._space.nvec)}"

        if isinstance(self._space, spaces.MultiBinary):
            return f"MultiBinary with {self._space.n} bits."

        if isinstance(self._space, spaces.Dict):
            parts = []
            for key, subspace in self._space.items():
                sub = AutoObservationMapper(subspace, precision=self._precision)
                parts.append(f"  {key}: {sub.describe()}")
            return "Dict observation:\n" + "\n".join(parts)

        if isinstance(self._space, spaces.Tuple):
            parts = []
            for i, subspace in enumerate(self._space.spaces):
                sub = AutoObservationMapper(subspace, precision=self._precision)
                parts.append(f"  [{i}]: {sub.describe()}")
            return "Tuple observation:\n" + "\n".join(parts)

        return f"Observation space: {type(self._space).__name__}"


# =============================================================================
# Auto Action Mapper
# =============================================================================


class AutoActionMapper:
    """Automatically maps text actions to gymnasium action values.

    Supports Discrete (by number or name), Box (comma/whitespace-separated
    floats, clipped to bounds), MultiDiscrete, MultiBinary, and Text spaces.

    Args:
        space: The gymnasium action space.
        action_names: Optional mapping from action index to human name
            (Discrete spaces only).
    """

    def __init__(
        self,
        space: Any,
        action_names: dict[int, str] | None = None,
    ) -> None:
        import gymnasium.spaces as spaces

        self._space = space
        self._action_names = action_names or {}
        self._spaces = spaces

        # Build reverse lookup for name -> index (case-insensitive)
        self._name_to_index: dict[str, int] = {}
        for idx, name in self._action_names.items():
            self._name_to_index[name.lower()] = idx

    def map(self, text: str) -> Any:
        spaces = self._spaces
        text = text.strip()

        if isinstance(self._space, spaces.Text):
            return text

        if isinstance(self._space, spaces.Discrete):
            return self._map_discrete(text)

        if isinstance(self._space, spaces.Box):
            return self._map_box(text)

        if isinstance(self._space, spaces.MultiDiscrete):
            return self._map_multi_discrete(text)

        if isinstance(self._space, spaces.MultiBinary):
            return self._map_multi_binary(text)

        raise ValueError(f"Unsupported action space: {type(self._space).__name__}")

    def _map_discrete(self, text: str) -> int:
        # Try integer parse first
        try:
            val = int(text)
            if val < self._space.start or val >= self._space.start + self._space.n:
                raise ValueError(
                    f"Action {val} out of range "
                    f"[{self._space.start}, {self._space.start + self._space.n - 1}]."
                )
            return val
        except ValueError as e:
            if "out of range" in str(e):
                raise

        # Try name match (case-insensitive)
        lower = text.lower()
        if lower in self._name_to_index:
            return self._name_to_index[lower]

        if self._action_names:
            valid = ", ".join(f"{i}: {n}" for i, n in sorted(self._action_names.items()))
            raise ValueError(
                f"Invalid action '{text}'. Expected a number "
                f"(0-{self._space.n - 1}) or one of: {valid}"
            )
        raise ValueError(
            f"Invalid action '{text}'. Expected a number in range [0, {self._space.n - 1}]."
        )

    def _parse_values(self, text: str) -> list[str]:
        """Split text by comma or whitespace into individual values."""
        # First try comma split, then whitespace
        if "," in text:
            return [v.strip() for v in text.split(",") if v.strip()]
        return text.split()

    def _map_box(self, text: str) -> np.ndarray:
        expected = int(np.prod(self._space.shape))
        parts = self._parse_values(text)
        if len(parts) != expected:
            raise ValueError(f"Expected {expected} values for Box action, got {len(parts)}.")
        try:
            values = np.array([float(v) for v in parts], dtype=self._space.dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse action values: {e}") from e

        # Clip to finite bounds
        low = self._space.low.flatten()
        high = self._space.high.flatten()
        for i in range(len(values)):
            if np.isfinite(low[i]):
                values[i] = max(values[i], low[i])
            if np.isfinite(high[i]):
                values[i] = min(values[i], high[i])

        return values.reshape(self._space.shape)

    def _map_multi_discrete(self, text: str) -> np.ndarray:
        expected = len(self._space.nvec)
        parts = self._parse_values(text)
        if len(parts) != expected:
            raise ValueError(
                f"Expected {expected} values for MultiDiscrete action, got {len(parts)}."
            )
        try:
            values = np.array([int(v) for v in parts], dtype=self._space.dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse action values: {e}") from e

        # Validate ranges
        for i, (val, nv) in enumerate(zip(values, self._space.nvec)):
            if val < 0 or val >= nv:
                raise ValueError(f"Action value {val} at index {i} out of range [0, {nv - 1}].")
        return values

    def _map_multi_binary(self, text: str) -> np.ndarray:
        expected = self._space.n
        parts = self._parse_values(text)
        if len(parts) != expected:
            raise ValueError(f"Expected {expected} binary values, got {len(parts)}.")
        try:
            values = np.array([int(v) for v in parts], dtype=self._space.dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse binary values: {e}") from e

        for val in values:
            if val not in (0, 1):
                raise ValueError(f"MultiBinary values must be 0 or 1, got {val}.")
        return values

    def describe(self) -> str:
        spaces = self._spaces

        if isinstance(self._space, spaces.Text):
            return "Enter a text action."

        if isinstance(self._space, spaces.Discrete):
            if self._action_names:
                entries = "\n".join(
                    f"  {i}: {name}" for i, name in sorted(self._action_names.items())
                )
                return f"Choose one action by name or number:\n{entries}"
            return (
                f"Enter an integer action in range "
                f"[{self._space.start}, {self._space.start + self._space.n - 1}]."
            )

        if isinstance(self._space, spaces.Box):
            shape = self._space.shape
            ndim = int(np.prod(shape))
            return (
                f"Enter {ndim} comma-separated float values. "
                f"Bounds: [{self._space.low.min():.2f}, {self._space.high.max():.2f}]."
            )

        if isinstance(self._space, spaces.MultiDiscrete):
            ranges = ", ".join(f"[0,{n - 1}]" for n in self._space.nvec)
            return f"Enter {len(self._space.nvec)} integers. Ranges: {ranges}."

        if isinstance(self._space, spaces.MultiBinary):
            return f"Enter {self._space.n} binary values (0 or 1)."

        return f"Action space: {type(self._space).__name__}"


# =============================================================================
# Grid Observation Mapper
# =============================================================================


class GridObservationMapper:
    """Maps 2D/3D grid observations to ASCII text.

    Converts matrix observations into character-based grid representations
    using a value-to-character mapping. Values are matched to the nearest
    key in the value map.

    Args:
        value_map: Maps float/int observation values to display characters.
        labels: Optional legend labels for context.
    """

    def __init__(
        self,
        value_map: dict[float, str],
        labels: dict[str, str] | None = None,
    ) -> None:
        self._value_map = value_map
        self._labels = labels or {}
        self._sorted_keys = sorted(value_map.keys())

    def _nearest_char(self, val: float) -> str:
        """Find the character for the nearest key in value_map."""
        best_key = min(self._sorted_keys, key=lambda k: abs(k - val))
        return self._value_map[best_key]

    def map(self, obs: Any, info: dict[str, Any]) -> str:
        grid = np.asarray(obs)

        # Handle 3D (channels, H, W) -> squeeze or take first channel
        if grid.ndim == 3:
            if grid.shape[0] == 1:
                grid = grid[0]
            else:
                # Take the first channel
                grid = grid[0]

        if grid.ndim != 2:
            return str(obs)

        rows = []
        for row in grid:
            line = "".join(self._nearest_char(val) for val in row)
            rows.append(line)

        result = "\n".join(rows)

        if self._labels:
            legend = " | ".join(f"{k}={v}" for k, v in self._labels.items())
            result = f"{result}\n\nLegend: {legend}"

        return result

    def describe(self) -> str:
        legend = ", ".join(f"{char} = {val}" for val, char in sorted(self._value_map.items()))
        return f"ASCII grid map. Characters: {legend}"


# =============================================================================
# FrozenLake Observation Mapper
# =============================================================================


class FrozenLakeObservationMapper:
    """Maps FrozenLake discrete-position observations to ASCII grid text.

    FrozenLake's observation is ``Discrete(16)`` (or 64 for 8x8) — an integer
    representing the agent's position. This mapper renders the full grid with
    the agent's position marked.

    Args:
        desc: Grid description — list of strings (e.g., ``["SFFF", "FHFH", ...]``)
            or numpy byte array from ``env.unwrapped.desc``.
        agent_char: Character to mark the agent's position.
    """

    def __init__(
        self,
        desc: Any,
        agent_char: str = "@",
    ) -> None:
        # Convert numpy byte array to list of strings
        if hasattr(desc, "dtype"):
            self._rows = [
                "".join(c.decode() if isinstance(c, bytes) else str(c) for c in row) for row in desc
            ]
        else:
            self._rows = [str(row) for row in desc]
        self._nrows = len(self._rows)
        self._ncols = len(self._rows[0]) if self._rows else 0
        self._agent_char = agent_char

    def map(self, obs: Any, info: dict[str, Any]) -> str:
        """Render the grid with the agent at the given position.

        Args:
            obs: Integer position (row-major index).
            info: Step info dict.

        Returns:
            ASCII grid with agent marked and space-separated characters.
        """
        pos = int(obs)
        row = pos // self._ncols
        col = pos % self._ncols

        lines = []
        for r, row_str in enumerate(self._rows):
            chars = []
            for c, ch in enumerate(row_str):
                if r == row and c == col:
                    chars.append(self._agent_char)
                else:
                    chars.append(ch)
            lines.append(" ".join(chars))
        return "\n".join(lines)

    def describe(self) -> str:
        """Describe the grid layout with a tile legend."""
        # Render static grid (no agent)
        lines = []
        for row_str in self._rows:
            lines.append(" ".join(row_str))
        grid_text = "\n".join(lines)

        legend = (
            "Tiles: S = start, F = frozen (safe), H = hole (fall = lose), G = goal (win). "
            f"The agent's position is marked with '{self._agent_char}'. "
            f"Grid size: {self._nrows}x{self._ncols}."
        )
        return f"{grid_text}\n\n{legend}"


# =============================================================================
# Image Observation Mapper
# =============================================================================


class ImageObservationMapper:
    """Renders gymnasium pixel observations as ImageContent.

    Converts numpy RGB/grayscale/RGBA arrays to base64-encoded images
    and attaches them to the Observation alongside a text description.

    Args:
        text_description: Text to accompany the image observation.
        format: Image format ("png" or "jpeg").
        quality: JPEG quality (1-100), only used for JPEG.
    """

    _multimodal = True

    def __init__(
        self,
        text_description: str = "Current visual observation:",
        format: str = "png",
        quality: int = 85,
    ) -> None:
        self._text_description = text_description
        self._format = format.lower()
        self._quality = quality

    def map(self, obs: Any, info: dict[str, Any]) -> tuple[str, tuple[ImageContent, ...]]:
        """Convert numpy array observation to text and ImageContent.

        Args:
            obs: Numpy array observation (H,W), (H,W,3), or (H,W,4).
            info: Step info dict.

        Returns:
            Tuple of (text_description, (ImageContent,)).
        """
        import base64
        import io

        from PIL import Image as PILImage

        arr = np.asarray(obs)

        # Handle different array shapes
        if arr.ndim == 2:
            # Grayscale (H, W) -> convert to RGB
            pil_img = PILImage.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA (H, W, 4)
            pil_img = PILImage.fromarray(arr, mode="RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # RGB (H, W, 3)
            pil_img = PILImage.fromarray(arr, mode="RGB")
        else:
            # Fallback: try to interpret as RGB
            pil_img = PILImage.fromarray(arr)

        # Encode to bytes
        buffer = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        if self._format == "jpeg":
            pil_img = pil_img.convert("RGB")  # JPEG doesn't support alpha
            save_kwargs["quality"] = self._quality
            media_type = "image/jpeg"
        else:
            media_type = "image/png"

        pil_img.save(buffer, format=self._format.upper(), **save_kwargs)
        b64_data = base64.b64encode(buffer.getvalue()).decode()

        img_content = ImageContent(data=b64_data, media_type=media_type)
        return self._text_description, (img_content,)

    def describe(self) -> str:
        """Describe the visual observation space."""
        return "Visual observation rendered as an image."


# =============================================================================
# Hidden State
# =============================================================================


@dataclass(frozen=True)
class GymnasiumHidden:
    """Hidden state for gymnasium environments.

    Attributes:
        task_index: Current task index (or None if not indexed).
        seed: Seed used for this episode.
        episode_step: Current step within the episode.
        last_action: The last action text (before extraction).
        raw_observation: Raw gymnasium observation value.
        gym_reward: Cumulative gymnasium reward for this episode.
    """

    task_index: int | None
    seed: int | None
    episode_step: int
    last_action: str | None
    raw_observation: Any
    gym_reward: float


# =============================================================================
# Reward
# =============================================================================


@dataclass
class GymnasiumReward:
    """Reward function wrapping gymnasium's native step reward.

    Uses STEP type for intermediate rewards, OUTCOME for terminal.
    """

    _name: str = "gym_reward"
    _reward_type: RewardType = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[GymnasiumHidden],
        action: Action,
        next_state: State[GymnasiumHidden],
    ) -> Signal:
        """Compute reward from gymnasium's native reward."""
        gym_reward = next_state.metadata.info.get("gym_reward", 0.0)

        reward_type = self.reward_type if next_state.metadata.is_terminal else RewardType.STEP

        return Signal(
            name=self.name,
            reward_type=reward_type,
            reward=float(gym_reward),
            metadata={"source": "gymnasium"},
        )


# =============================================================================
# Environment
# =============================================================================


class GymnasiumEnvironment:
    """MDP wrapper for gymnasium environments.

    Bridges gymnasium's numeric observation/action interface with text-based
    LLM agents. Uses ObservationMapper to render observations as text and
    ActionMapper to parse text actions into gymnasium values.

    Non-pure (stateful) — uses ``_StateContinuityTracker`` to enforce
    sequential stepping.

    Args:
        gym_env: The underlying gymnasium environment.
        env_id: Optional identifier for this environment.
        observation_mapper: Custom observation-to-text mapper.
        action_mapper: Custom text-to-action mapper.
        answer_extractor: Extractor for model responses (step 1 of action pipeline).
        observation_labels: Shorthand labels for AutoObservationMapper.
        action_names: Shorthand names for AutoActionMapper.
        max_steps: Maximum steps per episode.
        seeds: List of seeds for indexed task access.
        num_tasks: Number of tasks (alternative to seeds for ``__len__``).
        use_ansi_render: Use gym's ANSI render output for observations.
        extra_rewards: Additional reward functions.
        prompts: Custom prompt components.

    Example:
        >>> import gymnasium
        >>> gym_env = gymnasium.make("CartPole-v1")
        >>> env = GymnasiumEnvironment(
        ...     gym_env=gym_env,
        ...     action_names={0: "left", 1: "right"},
        ...     num_tasks=100,
        ... )
        >>> state, _ = env.reset(seed=42)
        >>> result = env.step(state, Action(text="left"))
    """

    def __init__(
        self,
        gym_env: Any,
        env_id: str = "",
        observation_mapper: ObservationMapper | None = None,
        action_mapper: ActionMapper | None = None,
        answer_extractor: AnswerExtractor | None = None,
        observation_labels: dict[int | str, str] | None = None,
        action_names: dict[int, str] | None = None,
        max_steps: int | None = None,
        seeds: list[int] | None = None,
        num_tasks: int | None = None,
        use_ansi_render: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
    ) -> None:
        self._gym_env = gym_env
        self._env_id = env_id or getattr(getattr(gym_env, "spec", None), "id", "gymnasium")
        self._use_ansi_render = use_ansi_render
        self._seeds = seeds
        self._num_tasks = num_tasks
        self._answer_extractor = answer_extractor or RawGenerationExtractor()

        # Resolve observation mapper
        if observation_mapper is not None:
            self._observation_mapper = observation_mapper
        else:
            self._observation_mapper = AutoObservationMapper(
                gym_env.observation_space,
                labels=observation_labels,
            )

        # Resolve action mapper
        if action_mapper is not None:
            self._action_mapper = action_mapper
        else:
            self._action_mapper = AutoActionMapper(
                gym_env.action_space,
                action_names=action_names,
            )

        # Resolve max_steps
        if max_steps is not None:
            self._max_steps = max_steps
        else:
            spec = getattr(gym_env, "spec", None)
            self._max_steps = getattr(spec, "max_episode_steps", None)

        # Rewards
        self._native_rewards: tuple[RewardFunction, ...] = (GymnasiumReward(),)
        self._extra_rewards = extra_rewards

        # Prompts
        self._prompts = dict(prompts) if prompts else {}

        # State continuity tracking (non-pure env)
        self._state_tracker = _StateContinuityTracker()

    def __len__(self) -> int:
        if self._seeds is not None:
            return len(self._seeds)
        if self._num_tasks is not None:
            return self._num_tasks
        raise TypeError(
            f"{type(self).__name__} has no defined length. "
            f"Pass seeds= or num_tasks= to enable __len__."
        )

    @property
    def prompts(self) -> dict[str, str]:
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_id,
            adapter="gymnasium",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            pure_step=False,
            supports_seed=True,
            metadata={"env_id": self._env_id},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction[GymnasiumHidden], ...]:
        return self._native_rewards + self._extra_rewards

    def _resolve_seed(self, seed: int | None, task_index: int) -> int | None:
        """Resolve the seed for this episode."""
        if seed is not None:
            return seed
        if self._seeds is not None and 0 <= task_index < len(self._seeds):
            return self._seeds[task_index]
        return None

    def _render_observation(
        self, raw_obs: Any, info: dict[str, Any]
    ) -> tuple[str, tuple[ImageContent, ...]]:
        """Render an observation to text (and optionally images).

        Returns:
            Tuple of (text, images). images is empty for text-only mappers.
        """
        if self._use_ansi_render:
            rendered = self._gym_env.render()
            if rendered is not None:
                return str(rendered), ()
        if isinstance(self._observation_mapper, MultimodalObservationMapper):
            return self._observation_mapper.map(raw_obs, info)
        return self._observation_mapper.map(raw_obs, info), ()

    def _build_task_description(self) -> str:
        """Build the static task description (no step-specific observation)."""
        parts = []

        # Environment description
        desc = self._prompts.get("description", "")
        if desc:
            parts.append(f"You are interacting with the {self._env_id} gymnasium environment.")
            parts.append(desc)
        else:
            parts.append(f"You are interacting with the {self._env_id} gymnasium environment.")

        parts.append("")

        # Observation space description
        parts.append("Observation space:")
        parts.append(self._observation_mapper.describe())
        parts.append("")

        # Action space description
        parts.append("Action space:")
        parts.append(self._action_mapper.describe())

        return "\n".join(parts)

    def _build_error_step(
        self,
        state: State[GymnasiumHidden],
        action: Action,
        error_msg: str,
    ) -> StepResult[GymnasiumHidden]:
        """Build a StepResult for an invalid/failed action (wasted turn)."""
        next_step = state.hidden.episode_step + 1
        meta_step = state.metadata.step + 1

        # Check truncation
        truncated = self._max_steps is not None and next_step >= self._max_steps

        new_hidden = GymnasiumHidden(
            task_index=state.hidden.task_index,
            seed=state.hidden.seed,
            episode_step=next_step,
            last_action=action.text,
            raw_observation=state.hidden.raw_observation,
            gym_reward=state.hidden.gym_reward,
        )

        error_text = f"[Step {next_step}] Error: {error_msg}"
        state_content = ObservationContent(text=error_text)

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": error_text},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=state_content,
        )

        new_metadata = StateMetadata(
            step=meta_step,
            episode_id=state.metadata.episode_id,
            is_terminal=truncated,
            info={**state.metadata.info, "gym_reward": 0.0, "error": error_msg},
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=False,
            truncated=truncated,
            info={"error": error_msg},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[GymnasiumHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Environment-specific options.
                - task_index: Select specific task (resolves seed from seeds list).
                - episode_id: Override the generated episode ID.

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index", 0)
        resolved_seed = self._resolve_seed(seed, task_index)

        # Reset gymnasium env
        reset_kwargs: dict[str, Any] = {}
        if resolved_seed is not None:
            reset_kwargs["seed"] = resolved_seed
        raw_obs, info = self._gym_env.reset(**reset_kwargs)

        # Render observation
        obs_text, obs_images = self._render_observation(raw_obs, info)

        # Build structured observation components
        task_desc = self._build_task_description()
        task_content = ObservationContent(text=task_desc)
        state_text = f"[Step 0]\n{obs_text}"
        state_content = ObservationContent(text=state_text, images=obs_images)

        # Legacy prompt = task description (for backward compat)
        prompt = task_desc

        # Legacy messages: step-0 observation as first user message
        initial_messages: tuple[dict[str, Any], ...] = ()
        step_msg: dict[str, Any] = {"role": "user", "content": state_text}
        if obs_images:
            step_msg["images"] = [
                {"data": img.data, "media_type": img.media_type} for img in obs_images
            ]
        initial_messages = (step_msg,)

        hidden = GymnasiumHidden(
            task_index=task_index,
            seed=resolved_seed,
            episode_step=0,
            last_action=None,
            raw_observation=raw_obs,
            gym_reward=0.0,
        )

        observation = Observation(
            prompt=prompt,
            messages=initial_messages,
            images=obs_images,
            task=task_content,
            state=state_content,
        )

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "task_index": task_index,
                "seed": resolved_seed,
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {"task_index": task_index, "seed": resolved_seed}

    def step(
        self,
        state: State[GymnasiumHidden],
        action: Action,
    ) -> StepResult[GymnasiumHidden]:
        """Take an action from the given state.

        Two-step action pipeline:
        1. Extract action text from LLM response using answer_extractor
        2. Map extracted text to gymnasium action using action_mapper

        Invalid actions waste a turn (step counter advances, error as
        observation, no gymnasium step).

        Args:
            state: Current state.
            action: Action to take (model's response text).

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        self._state_tracker.validate(state, "GymnasiumEnvironment")

        # Step 1: Extract action text
        extracted, extraction_meta = self._answer_extractor.extract(action.text or "")
        if extracted is None:
            result = self._build_error_step(
                state, action, "Could not extract action from response."
            )
            self._state_tracker.track(result.next_state)
            return result

        # Step 2: Map to gymnasium action
        try:
            gym_action = self._action_mapper.map(extracted)
        except ValueError as e:
            result = self._build_error_step(state, action, str(e))
            self._state_tracker.track(result.next_state)
            return result

        # Step the gymnasium environment
        raw_obs, reward, terminated, truncated_gym, info = self._gym_env.step(gym_action)

        next_step = state.hidden.episode_step + 1
        cumulative_reward = state.hidden.gym_reward + reward

        # Check our own truncation
        truncated = truncated_gym or (self._max_steps is not None and next_step >= self._max_steps)

        # Render observation
        obs_text, obs_images = self._render_observation(raw_obs, info)

        new_hidden = GymnasiumHidden(
            task_index=state.hidden.task_index,
            seed=state.hidden.seed,
            episode_step=next_step,
            last_action=action.text,
            raw_observation=raw_obs,
            gym_reward=cumulative_reward,
        )

        state_text = f"[Step {next_step}]\n{obs_text}"
        state_content = ObservationContent(text=state_text, images=obs_images)

        user_msg: dict[str, Any] = {"role": "user", "content": state_text}
        if obs_images:
            user_msg["images"] = [
                {"data": img.data, "media_type": img.media_type} for img in obs_images
            ]

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            user_msg,
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            images=obs_images,
            task=state.observation.task,
            state=state_content,
        )

        is_terminal = terminated or truncated
        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=is_terminal,
            info={
                **state.metadata.info,
                "gym_reward": reward,
                "gym_info": info,
                "extracted_action": extracted,
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "gym_reward": reward,
                "gym_info": info,
                "extracted_action": extracted,
                "extraction_metadata": extraction_meta,
            },
        )

    def compute_rewards(
        self,
        state: State[GymnasiumHidden],
        action: Action,
        next_state: State[GymnasiumHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# =============================================================================
# Presets
# =============================================================================


GYMNASIUM_PRESETS: dict[str, dict[str, Any]] = {
    # FrozenLake environments
    "frozen_lake": {
        "description": "Navigate a frozen lake from start (S) to goal (G) without falling into holes (H). The surface is slippery.",
        "env_id": "FrozenLake-v1",
        "action_names": {0: "left", 1: "down", 2: "right", 3: "up"},
        "max_steps": 100,
    },
    "frozen_lake/8x8": {
        "description": "Navigate a frozen lake (8x8) from start (S) to goal (G) without falling into holes (H). The surface is slippery.",
        "env_id": "FrozenLake-v1",
        "action_names": {0: "left", 1: "down", 2: "right", 3: "up"},
        "max_steps": 200,
        "make_kwargs": {"map_name": "8x8"},
    },
    # Gym4Real environments
    "gym4real/dam-v0": {
        "description": "Minimize unmet demand while avoiding floods by controlling water release.",
        "env_id": "gym4real/dam-v0",
    },
    "gym4real/elevator-v0": {
        "description": "Minimize passenger waiting time by controlling an elevator.",
        "env_id": "gym4real/elevator-v0",
        "action_names": {0: "up", 1: "down", 2: "open doors"},
    },
    "gym4real/microgrid-v0": {
        "description": "Maximize energy revenue and minimize battery degradation in a microgrid.",
        "env_id": "gym4real/microgrid-v0",
    },
    "gym4real/TradingEnv-v0": {
        "description": "Maximize P&L net of transaction costs in a trading environment.",
        "env_id": "gym4real/TradingEnv-v0",
        "action_names": {0: "short", 1: "flat", 2: "long"},
    },
    "gym4real/wds-v0": {
        "description": "Maximize demand satisfaction and minimize overflow in a water distribution system.",
        "env_id": "gym4real/wds-v0",
    },
    # MarsExplorer
    "mars_explorer": {
        "description": "Explore unknown terrain and maximize coverage on Mars.",
        "env_id": "mars_explorer-v1",
        "action_names": {0: "right", 1: "left", 2: "up", 3: "down"},
        "observation_mapper": GridObservationMapper(
            value_map={0.0: ".", 0.3: " ", 0.6: "R", 1.0: "#"},
        ),
    },
    # Atari environments (requires gymnasium[atari] + ale-py, VLM backend)
    "atari/breakout": {
        "description": "Atari Breakout — break bricks with a ball and paddle. Pixel observations (requires VLM).",
        "env_id": "ALE/Breakout-v5",
        "observation_mapper": ImageObservationMapper(),
        "action_names": {0: "noop", 1: "fire", 2: "right", 3: "left"},
        "max_steps": 2000,
    },
    "atari/pong": {
        "description": "Atari Pong — table tennis against AI opponent. Pixel observations (requires VLM).",
        "env_id": "ALE/Pong-v5",
        "observation_mapper": ImageObservationMapper(),
        "action_names": {
            0: "noop",
            1: "fire",
            2: "right",
            3: "left",
            4: "right+fire",
            5: "left+fire",
        },
        "max_steps": 2000,
    },
    "atari/space_invaders": {
        "description": "Atari Space Invaders — defend Earth from descending aliens. Pixel observations (requires VLM).",
        "env_id": "ALE/SpaceInvaders-v5",
        "observation_mapper": ImageObservationMapper(),
        "action_names": {
            0: "noop",
            1: "fire",
            2: "right",
            3: "left",
            4: "right+fire",
            5: "left+fire",
        },
        "max_steps": 2000,
    },
    "atari/ms_pacman": {
        "description": "Atari Ms. Pac-Man — navigate mazes, eat pellets, avoid ghosts. Pixel observations (requires VLM).",
        "env_id": "ALE/MsPacman-v5",
        "observation_mapper": ImageObservationMapper(),
        "action_names": {
            0: "noop",
            1: "up",
            2: "right",
            3: "left",
            4: "down",
            5: "up+right",
            6: "up+left",
            7: "down+right",
            8: "down+left",
        },
        "max_steps": 2000,
    },
    "atari/freeway": {
        "description": "Atari Freeway — cross a busy highway without getting hit. Pixel observations (requires VLM).",
        "env_id": "ALE/Freeway-v5",
        "observation_mapper": ImageObservationMapper(),
        "action_names": {0: "noop", 1: "up", 2: "down"},
        "max_steps": 2000,
    },
}


# =============================================================================
# Adapter
# =============================================================================


class GymnasiumAdapter:
    """Adapter for gymnasium environments.

    Provides access to any gymnasium-compatible environment through a text
    interface suitable for LLM agents. Includes presets for popular
    environments (Gym4Real, MarsExplorer) and supports arbitrary envs
    via ``gymnasium.make()``.

    Example:
        >>> adapter = GymnasiumAdapter()
        >>> env = adapter.get_environment("CartPole-v1", num_tasks=100)
        >>> state, _ = env.reset(seed=42)
    """

    @property
    def name(self) -> str:
        return "gymnasium"

    def _get_gymnasium(self) -> Any:
        """Import and return the gymnasium module."""
        try:
            import gymnasium

            return gymnasium
        except ImportError as e:
            raise ImportError(
                "gymnasium is required for GymnasiumAdapter. Install with: pip install gymnasium"
            ) from e

    def list_environments(self) -> list[str]:
        """List preset environment names.

        Returns:
            Sorted list of preset IDs.
        """
        return sorted(GYMNASIUM_PRESETS.keys())

    def get_environment(
        self,
        name: str,
        *,
        gym_env: Any | None = None,
        observation_mapper: ObservationMapper | None = None,
        action_mapper: ActionMapper | None = None,
        answer_extractor: AnswerExtractor | None = None,
        observation_labels: dict[int | str, str] | None = None,
        action_names: dict[int, str] | None = None,
        max_steps: int | None = None,
        seeds: list[int] | None = None,
        num_tasks: int | None = None,
        use_ansi_render: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        render_mode: str | None = None,
        **make_kwargs: Any,
    ) -> GymnasiumEnvironment:
        """Create a gymnasium environment.

        If ``gym_env`` is provided, uses it directly. Otherwise calls
        ``gymnasium.make(name)`` (merging preset config if available).

        Args:
            name: Environment name (preset key or gymnasium env ID).
            gym_env: Pre-created gymnasium environment instance.
            observation_mapper: Custom observation mapper.
            action_mapper: Custom action mapper.
            answer_extractor: Extractor for model responses.
            observation_labels: Labels for AutoObservationMapper.
            action_names: Names for AutoActionMapper.
            max_steps: Maximum steps per episode.
            seeds: Seed list for indexed task access.
            num_tasks: Number of tasks for ``__len__``.
            use_ansi_render: Use gym's ANSI render for observations.
            extra_rewards: Additional reward functions.
            prompts: Custom prompt components.
            render_mode: Gymnasium render mode (auto ``"ansi"`` when
                ``use_ansi_render=True``).
            **make_kwargs: Extra arguments to ``gymnasium.make()``.

        Returns:
            Configured GymnasiumEnvironment.
        """
        # Merge preset config
        preset = GYMNASIUM_PRESETS.get(name, {})

        # Resolve env_id (preset may have different gym env_id)
        env_id = preset.get("env_id", name)

        # Merge: explicit params override preset
        if observation_mapper is None:
            observation_mapper = preset.get("observation_mapper")
        if action_mapper is None:
            action_mapper = preset.get("action_mapper")
        if action_names is None:
            action_names = preset.get("action_names")
        if observation_labels is None:
            observation_labels = preset.get("observation_labels")
        if max_steps is None:
            max_steps = preset.get("max_steps")

        # Build prompts from preset + explicit
        merged_prompts: dict[str, str] = {}
        if "description" in preset:
            merged_prompts["description"] = preset["description"]
        if prompts:
            merged_prompts.update(prompts)

        # Create gym env if not provided
        if gym_env is None:
            gymnasium = self._get_gymnasium()
            # Merge make_kwargs: preset defaults + user overrides
            make_kwargs_final: dict[str, Any] = dict(preset.get("make_kwargs", {}))
            make_kwargs_final.update(make_kwargs)
            if use_ansi_render and render_mode is None:
                make_kwargs_final["render_mode"] = "ansi"
            elif render_mode is not None:
                make_kwargs_final["render_mode"] = render_mode
            gym_env = gymnasium.make(env_id, **make_kwargs_final)

        # Auto-create FrozenLake observation mapper when no explicit mapper
        if observation_mapper is None and env_id == "FrozenLake-v1":
            unwrapped = getattr(gym_env, "unwrapped", gym_env)
            desc = getattr(unwrapped, "desc", None)
            if desc is not None:
                observation_mapper = FrozenLakeObservationMapper(desc=desc)

        return GymnasiumEnvironment(
            gym_env=gym_env,
            env_id=env_id,
            observation_mapper=observation_mapper,
            action_mapper=action_mapper,
            answer_extractor=answer_extractor,
            observation_labels=observation_labels,
            action_names=action_names,
            max_steps=max_steps,
            seeds=seeds,
            num_tasks=num_tasks,
            use_ansi_render=use_ansi_render,
            extra_rewards=extra_rewards,
            prompts=merged_prompts or None,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """Gymnasium environments build prompts internally."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """Gymnasium environments build prompts internally."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """Gymnasium does not provide native answer extraction."""
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about an environment.

        Args:
            name: Environment name.

        Returns:
            Dictionary with environment metadata.
        """
        preset = GYMNASIUM_PRESETS.get(name, {})
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "description": preset.get(
                "description",
                f"Gymnasium environment: {name}",
            ),
        }
