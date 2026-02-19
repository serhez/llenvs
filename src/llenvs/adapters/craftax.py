"""Craftax adapter — wraps Craftax JAX-based survival environments for LLM agents.

Craftax (ICML 2024) is a fast, procedurally-generated survival benchmark
combining Crafter and NetHack mechanics. Uses the Gymnax API (pure functional
JAX), not standard Gymnasium. Four variants: Full/Classic x Symbolic/Pixels.

Key design: ``pure_step=True`` — Craftax is pure functional (immutable JAX
pytrees), enabling zero-cost DirectStrategy branching.
"""

from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.extraction import AnswerExtractor, RawGenerationExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, ImageContent, Observation, State, StateMetadata

# =============================================================================
# Actions
# =============================================================================

# Classic: 17 actions
CLASSIC_ACTIONS: dict[int, str] = {
    0: "noop",
    1: "left",
    2: "right",
    3: "up",
    4: "down",
    5: "do",
    6: "sleep",
    7: "place_stone",
    8: "place_table",
    9: "place_furnace",
    10: "place_plant",
    11: "make_wood_pickaxe",
    12: "make_stone_pickaxe",
    13: "make_iron_pickaxe",
    14: "make_wood_sword",
    15: "make_stone_sword",
    16: "make_iron_sword",
}

# Full: 43 actions (classic + additional crafting/combat/movement)
FULL_ACTIONS: dict[int, str] = {
    **CLASSIC_ACTIONS,
    17: "make_diamond_pickaxe",
    18: "make_diamond_sword",
    19: "make_iron_armour",
    20: "make_diamond_armour",
    21: "make_arrow",
    22: "make_torch",
    23: "make_ruby",
    24: "descend",
    25: "ascend",
    26: "enchant_sword",
    27: "enchant_armour",
    28: "make_sapphire",
    29: "make_bow",
    30: "shoot_up",
    31: "shoot_down",
    32: "shoot_left",
    33: "shoot_right",
    34: "place_torch",
    35: "drink_potion_red",
    36: "drink_potion_green",
    37: "drink_potion_blue",
    38: "drink_potion_pink",
    39: "drink_potion_cyan",
    40: "drink_potion_yellow",
    41: "read_book",
    42: "use_wand",
}


class CraftaxActionMapper:
    """Maps text actions to Craftax discrete action indices.

    Accepts integer strings (``"5"``) or action names (``"craft_sword"``,
    case-insensitive).

    Args:
        is_classic: Whether this is Classic (17 actions) or Full (43 actions).
    """

    def __init__(self, is_classic: bool = True) -> None:
        self._actions = CLASSIC_ACTIONS if is_classic else FULL_ACTIONS
        self._name_to_index: dict[str, int] = {
            name.lower(): idx for idx, name in self._actions.items()
        }

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def map(self, text: str) -> int:
        text = text.strip()

        # Try integer
        try:
            val = int(text)
            if val < 0 or val >= self.num_actions:
                raise ValueError(f"Action {val} out of range [0, {self.num_actions - 1}].")
            return val
        except ValueError as e:
            if "out of range" in str(e):
                raise

        # Try name (case-insensitive)
        lower = text.lower()
        if lower in self._name_to_index:
            return self._name_to_index[lower]

        valid = ", ".join(f"{i}: {n}" for i, n in sorted(self._actions.items()))
        raise ValueError(
            f"Invalid action '{text}'. Expected a number "
            f"(0-{self.num_actions - 1}) or one of: {valid}"
        )

    def describe(self) -> str:
        entries = "\n".join(f"  {i}: {name}" for i, name in sorted(self._actions.items()))
        return f"Choose one action by name or number:\n{entries}"


# =============================================================================
# Observation rendering
# =============================================================================


def _render_symbolic(obs: np.ndarray, is_classic: bool) -> str:
    """Parse the flat symbolic observation array into structured text."""
    obs = np.asarray(obs).flatten()

    parts = []
    parts.append("=== Craftax Observation ===")

    # The symbolic observation is a flat array encoding the visible map,
    # inventory, intrinsics, etc. Layout differs between Classic and Full.
    if is_classic:
        # Classic: 1345-dim
        # Rough layout: map (9*11*7=693), inventory (12), intrinsics (5+),
        # direction (4), misc
        map_end = 693
        if len(obs) > map_end:
            inv_start = map_end
            inv_end = inv_start + 12
            if inv_end <= len(obs):
                inv = obs[inv_start:inv_end].astype(int)
                inv_labels = [
                    "wood",
                    "stone",
                    "coal",
                    "iron",
                    "diamond",
                    "sapling",
                    "wood_pickaxe",
                    "stone_pickaxe",
                    "iron_pickaxe",
                    "wood_sword",
                    "stone_sword",
                    "iron_sword",
                ]
                inv_parts = [
                    f"  {inv_labels[i]}: {inv[i]}" for i in range(len(inv_labels)) if i < len(inv)
                ]
                parts.append("Inventory:")
                parts.extend(inv_parts)

            # Intrinsics after inventory
            intr_start = inv_end
            intr_end = intr_start + 5
            if intr_end <= len(obs):
                intr = obs[intr_start:intr_end]
                parts.append(f"Health: {intr[0]:.1f}")
                parts.append(f"Food: {intr[1]:.1f}")
                parts.append(f"Drink: {intr[2]:.1f}")
                parts.append(f"Energy: {intr[3]:.1f}")
                parts.append(f"Mana: {intr[4]:.1f}")
    else:
        # Full: 8268-dim — similar but larger map and more inventory slots
        map_end = 7623  # 9*11*77
        if len(obs) > map_end:
            inv_start = map_end
            inv_end = inv_start + 24
            if inv_end <= len(obs):
                inv = obs[inv_start:inv_end].astype(int)
                inv_labels = [
                    "wood",
                    "stone",
                    "coal",
                    "iron",
                    "diamond",
                    "sapling",
                    "wood_pickaxe",
                    "stone_pickaxe",
                    "iron_pickaxe",
                    "diamond_pickaxe",
                    "wood_sword",
                    "stone_sword",
                    "iron_sword",
                    "diamond_sword",
                    "arrow",
                    "torch",
                    "ruby",
                    "sapphire",
                    "bow",
                    "iron_armour",
                    "diamond_armour",
                    "book",
                    "potion",
                    "wand",
                ]
                inv_parts = [
                    f"  {inv_labels[i]}: {inv[i]}" for i in range(min(len(inv_labels), len(inv)))
                ]
                parts.append("Inventory:")
                parts.extend(inv_parts)

            intr_start = inv_end if len(obs) > map_end + 24 else map_end + 24
            intr_end = intr_start + 5
            if intr_end <= len(obs):
                intr = obs[intr_start:intr_end]
                parts.append(f"Health: {intr[0]:.1f}")
                parts.append(f"Food: {intr[1]:.1f}")
                parts.append(f"Drink: {intr[2]:.1f}")
                parts.append(f"Energy: {intr[3]:.1f}")
                parts.append(f"Mana: {intr[4]:.1f}")

    return "\n".join(parts)


def _render_pixels_to_image(obs: np.ndarray) -> ImageContent:
    """Convert pixel observation (HxWx3 uint8) to base64 PNG ImageContent."""
    try:
        from PIL import Image
    except ImportError:
        # Fallback: raw numpy to PNG via minimal approach
        return _numpy_to_png_content(obs)

    img = Image.fromarray(np.asarray(obs).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return ImageContent(data=data, media_type="image/png")


def _numpy_to_png_content(obs: np.ndarray) -> ImageContent:
    """Minimal PNG encoding without PIL."""
    import struct
    import zlib

    arr = np.asarray(obs).astype(np.uint8)
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1

    # Build raw image data with filter byte
    raw = b""
    for row in range(h):
        raw += b"\x00"  # no filter
        if channels == 1:
            raw += arr[row].tobytes()
        else:
            raw += arr[row].tobytes()

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    color_type = 2 if channels == 3 else (6 if channels == 4 else 0)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", zlib.compress(raw))
    png += _chunk(b"IEND", b"")

    data = base64.b64encode(png).decode("ascii")
    return ImageContent(data=data, media_type="image/png")


# =============================================================================
# Hidden State
# =============================================================================


@dataclass(frozen=True)
class CraftaxHidden:
    """Hidden state for Craftax environments.

    Attributes:
        task_index: Current task index.
        seed: Seed used for this episode.
        episode_step: Current step within the episode.
        last_action: The last action text.
        craftax_state: The JAX EnvState pytree.
        rng_key: Current JAX PRNG key.
        cumulative_reward: Cumulative reward for this episode.
        achievements: JAX boolean array of achievements.
        is_classic: Whether this is Classic or Full variant.
    """

    task_index: int
    seed: int | None
    episode_step: int
    last_action: str | None
    craftax_state: Any
    rng_key: Any
    cumulative_reward: float
    achievements: Any
    is_classic: bool


# =============================================================================
# Rewards
# =============================================================================


@dataclass
class CraftaxReward:
    """Reward function wrapping Craftax's native step reward.

    Uses STEP type for intermediate rewards, OUTCOME for terminal.
    """

    _name: str = "craftax_reward"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[CraftaxHidden],
        action: Action,
        next_state: State[CraftaxHidden],
    ) -> Signal:
        reward = next_state.metadata.info.get("craftax_reward", 0.0)
        reward_type = RewardType.OUTCOME if next_state.metadata.is_terminal else RewardType.STEP
        return Signal(
            name=self.name,
            reward_type=reward_type,
            reward=float(reward),
            metadata={"source": "craftax"},
        )


@dataclass
class CraftaxAchievementReward:
    """Per-achievement reward signals.

    Produces a signal for each new achievement unlocked in a step.
    Intended as extra_rewards, not as a primary reward.
    """

    _name: str = "craftax_achievement"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.STEP

    def compute(
        self,
        state: State[CraftaxHidden],
        action: Action,
        next_state: State[CraftaxHidden],
    ) -> Signal:
        new_achievements = next_state.metadata.info.get("new_achievements", [])
        num_new = len(new_achievements)
        return Signal(
            name=self.name,
            reward_type=RewardType.STEP,
            reward=float(num_new),
            metadata={
                "new_achievements": new_achievements,
                "num_new": num_new,
            },
        )


# =============================================================================
# Environment
# =============================================================================


class CraftaxEnvironment:
    """MDP wrapper for Craftax environments.

    Pure step (``pure_step=True``) — Craftax is pure functional JAX,
    enabling zero-cost DirectStrategy branching.

    Args:
        craftax_env: The Craftax Gymnax environment instance.
        is_classic: Whether this is Classic (True) or Full (False).
        observation_mode: ``"text"``, ``"symbolic"``, or ``"pixels"``.
        max_steps: Maximum steps per episode.
        num_tasks: Number of tasks for ``__len__``.
        answer_extractor: Extractor for model responses.
        extra_rewards: Additional reward functions.
        prompts: Custom prompt components.
        _jax_random: Override for jax.random (for testing without JAX).
        _text_renderer: Override for text rendering function (for testing).
    """

    def __init__(
        self,
        craftax_env: Any,
        is_classic: bool = True,
        observation_mode: str = "symbolic",
        max_steps: int | None = None,
        num_tasks: int | None = None,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        _jax_random: Any = None,
        _text_renderer: Any = None,
    ) -> None:
        self._craftax_env = craftax_env
        self._is_classic = is_classic
        self._observation_mode = observation_mode
        self._num_tasks = num_tasks
        self._answer_extractor = answer_extractor or RawGenerationExtractor()
        self._action_mapper = CraftaxActionMapper(is_classic=is_classic)
        self._native_rewards: tuple[RewardFunction, ...] = (CraftaxReward(),)
        self._extra_rewards = extra_rewards
        self._prompts = dict(prompts) if prompts else {}
        self._text_renderer = _text_renderer

        # Resolve max_steps
        if max_steps is not None:
            self._max_steps = max_steps
        else:
            params = craftax_env.default_params()
            self._max_steps = getattr(params, "max_steps_in_episode", 1000)

        # JAX random module (injectable for testing)
        if _jax_random is not None:
            self._jax_random = _jax_random
        else:
            try:
                import jax

                self._jax_random = jax.random
            except ImportError:
                self._jax_random = None

    def __len__(self) -> int:
        if self._num_tasks is not None:
            return self._num_tasks
        raise TypeError(
            f"{type(self).__name__} has no defined length. Pass num_tasks= to enable __len__."
        )

    @property
    def prompts(self) -> dict[str, str]:
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        variant = "craftax-classic" if self._is_classic else "craftax"
        return EnvironmentSpec(
            name=variant,
            adapter="craftax",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            pure_step=True,
            supports_seed=True,
            metadata={
                "is_classic": self._is_classic,
                "observation_mode": self._observation_mode,
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction[CraftaxHidden], ...]:
        return self._native_rewards + self._extra_rewards

    def _render_observation(
        self, raw_obs: Any, craftax_state: Any
    ) -> tuple[str, tuple[ImageContent, ...]]:
        """Render observation based on mode. Returns (text, images)."""
        if self._observation_mode == "pixels":
            img = _render_pixels_to_image(raw_obs)
            return "[Visual observation — see attached image]", (img,)
        elif self._observation_mode == "text" and self._text_renderer is not None:
            text = self._text_renderer(craftax_state)
            return text, ()
        else:
            # symbolic mode (default)
            text = _render_symbolic(raw_obs, self._is_classic)
            return text, ()

    def _build_initial_prompt(self, obs_text: str) -> str:
        variant = "Craftax Classic" if self._is_classic else "Craftax"
        parts = []

        desc = self._prompts.get("description", "")
        if desc:
            parts.append(desc)
        else:
            parts.append(
                f"You are playing {variant}, an open-ended survival game. "
                f"Explore, gather resources, craft tools and weapons, "
                f"fight monsters, and descend into dungeons."
            )

        parts.append("")
        parts.append("Action space:")
        parts.append(self._action_mapper.describe())
        parts.append("")
        parts.append("[Step 0]")
        parts.append(obs_text)

        return "\n".join(parts)

    def _build_error_step(
        self,
        state: State[CraftaxHidden],
        action: Action,
        error_msg: str,
    ) -> StepResult[CraftaxHidden]:
        """Build a StepResult for an invalid action (wasted turn)."""
        next_step = state.hidden.episode_step + 1
        truncated = self._max_steps is not None and next_step >= self._max_steps

        new_hidden = CraftaxHidden(
            task_index=state.hidden.task_index,
            seed=state.hidden.seed,
            episode_step=next_step,
            last_action=action.text,
            craftax_state=state.hidden.craftax_state,
            rng_key=state.hidden.rng_key,
            cumulative_reward=state.hidden.cumulative_reward,
            achievements=state.hidden.achievements,
            is_classic=state.hidden.is_classic,
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": f"[Step {next_step}] Error: {error_msg}"},
        )
        new_observation = Observation(prompt=state.observation.prompt, messages=new_messages)

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=truncated,
            info={**state.metadata.info, "craftax_reward": 0.0, "error": error_msg},
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
    ) -> tuple[State[CraftaxHidden], dict[str, Any]]:
        options = options or {}
        task_index = options.get("task_index", 0)

        # Resolve seed
        resolved_seed = seed if seed is not None else task_index * 7919 + 42

        # Create JAX PRNG key
        rng_key = self._jax_random.PRNGKey(resolved_seed)

        # Reset Craftax env
        params = self._craftax_env.default_params()
        raw_obs, craftax_state = self._craftax_env.reset(rng_key, params)

        # Render observation
        obs_text, images = self._render_observation(raw_obs, craftax_state)
        prompt = self._build_initial_prompt(obs_text)

        # Get initial achievements
        achievements = getattr(craftax_state, "achievements", np.zeros(22, dtype=bool))

        hidden = CraftaxHidden(
            task_index=task_index,
            seed=resolved_seed,
            episode_step=0,
            last_action=None,
            craftax_state=craftax_state,
            rng_key=rng_key,
            cumulative_reward=0.0,
            achievements=np.asarray(achievements),
            is_classic=self._is_classic,
        )

        observation = Observation(prompt=prompt, images=images)

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={"task_index": task_index, "seed": resolved_seed},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        return state, {"task_index": task_index, "seed": resolved_seed}

    def step(
        self,
        state: State[CraftaxHidden],
        action: Action,
    ) -> StepResult[CraftaxHidden]:
        # Extract action text
        extracted, extraction_meta = self._answer_extractor.extract(action.text or "")
        if extracted is None:
            return self._build_error_step(state, action, "Could not extract action from response.")

        # Map to Craftax action
        try:
            action_idx = self._action_mapper.map(extracted)
        except ValueError as e:
            return self._build_error_step(state, action, str(e))

        # Split RNG key
        keys = self._jax_random.split(state.hidden.rng_key)
        step_key = keys[0]
        next_rng_key = keys[1]

        # Step Craftax env
        params = self._craftax_env.default_params()
        raw_obs, new_craftax_state, reward, done, info = self._craftax_env.step(
            step_key, state.hidden.craftax_state, action_idx, params
        )

        next_step = state.hidden.episode_step + 1
        cumulative_reward = state.hidden.cumulative_reward + float(reward)

        # Detect new achievements
        new_achievements_arr = getattr(new_craftax_state, "achievements", np.zeros(22, dtype=bool))
        new_achievements_arr = np.asarray(new_achievements_arr)
        old_achievements = np.asarray(state.hidden.achievements)
        new_achievement_indices = []
        if old_achievements is not None and new_achievements_arr is not None:
            for i in range(min(len(old_achievements), len(new_achievements_arr))):
                if new_achievements_arr[i] and not old_achievements[i]:
                    new_achievement_indices.append(i)

        # Check truncation
        truncated = self._max_steps is not None and next_step >= self._max_steps
        terminated = bool(done)

        # Render observation
        obs_text, images = self._render_observation(raw_obs, new_craftax_state)

        new_hidden = CraftaxHidden(
            task_index=state.hidden.task_index,
            seed=state.hidden.seed,
            episode_step=next_step,
            last_action=action.text,
            craftax_state=new_craftax_state,
            rng_key=next_rng_key,
            cumulative_reward=cumulative_reward,
            achievements=new_achievements_arr,
            is_classic=state.hidden.is_classic,
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": f"[Step {next_step}]\n{obs_text}"},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            images=images,
        )

        is_terminal = terminated or truncated
        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=is_terminal,
            info={
                **state.metadata.info,
                "craftax_reward": float(reward),
                "new_achievements": new_achievement_indices,
                "extracted_action": extracted,
            },
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
            terminated=terminated,
            truncated=truncated,
            info={
                "craftax_reward": float(reward),
                "new_achievements": new_achievement_indices,
                "extracted_action": extracted,
                "extraction_metadata": extraction_meta,
            },
        )

    def compute_rewards(
        self,
        state: State[CraftaxHidden],
        action: Action,
        next_state: State[CraftaxHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# =============================================================================
# Presets
# =============================================================================

CRAFTAX_PRESETS: dict[str, dict[str, Any]] = {
    "craftax": {
        "is_classic": False,
        "observation_mode": "text",
        "description": "Full Craftax — open-ended survival with crafting, combat, dungeons, and magic.",
    },
    "craftax-classic": {
        "is_classic": True,
        "observation_mode": "symbolic",
        "description": "Craftax Classic — streamlined survival with crafting and combat.",
    },
    "craftax-pixels": {
        "is_classic": False,
        "observation_mode": "pixels",
        "description": "Full Craftax with pixel observations (requires VLM).",
    },
    "craftax-classic-pixels": {
        "is_classic": True,
        "observation_mode": "pixels",
        "description": "Craftax Classic with pixel observations (requires VLM).",
    },
}


# =============================================================================
# Adapter
# =============================================================================


# Lazy jax import sentinel
jax: Any = None


def _get_craftax() -> Any:
    """Import and return the craftax module."""
    try:
        import craftax

        return craftax
    except ImportError as e:
        raise ImportError(
            "craftax is required for CraftaxAdapter. "
            "Install with: pip install craftax (requires JAX)"
        ) from e


class CraftaxAdapter:
    """Adapter for Craftax environments.

    Provides access to Craftax Full and Classic variants through a text/image
    interface suitable for LLM agents.
    """

    @property
    def name(self) -> str:
        return "craftax"

    def _get_craftax(self) -> Any:
        return _get_craftax()

    def list_environments(self) -> list[str]:
        return sorted(CRAFTAX_PRESETS.keys())

    def get_environment(
        self,
        name: str,
        *,
        observation_mode: str | None = None,
        max_steps: int | None = None,
        num_tasks: int | None = None,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> CraftaxEnvironment:
        """Create a Craftax environment.

        Args:
            name: Preset name or "craftax"/"craftax-classic".
            observation_mode: Override observation mode.
            max_steps: Override max steps.
            num_tasks: Number of tasks for ``__len__``.
            answer_extractor: Custom answer extractor.
            extra_rewards: Additional reward functions.
            prompts: Custom prompt components.
            **kwargs: Additional arguments.

        Returns:
            Configured CraftaxEnvironment.
        """
        self._get_craftax()

        # Merge preset
        preset = CRAFTAX_PRESETS.get(name, CRAFTAX_PRESETS.get("craftax", {}))
        is_classic = preset.get("is_classic", False)
        obs_mode = observation_mode or preset.get("observation_mode", "symbolic")

        # Build prompts
        merged_prompts: dict[str, str] = {}
        if "description" in preset:
            merged_prompts["description"] = preset["description"]
        if prompts:
            merged_prompts.update(prompts)

        # Create Craftax env
        if is_classic:
            from craftax.craftax_classic.envs.craftax_symbolic_env import (
                CraftaxClassicSymbolicEnv,
            )

            craftax_env = CraftaxClassicSymbolicEnv()
        else:
            if obs_mode == "pixels":
                from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

                craftax_env = CraftaxPixelsEnv()
            else:
                from craftax.craftax.envs.craftax_symbolic_env import (
                    CraftaxSymbolicEnv,
                )

                craftax_env = CraftaxSymbolicEnv()

        # Get text renderer for Full text mode
        text_renderer = None
        if obs_mode == "text" and not is_classic:
            try:
                from craftax.craftax.renderer import render_craftax_text

                text_renderer = render_craftax_text
            except ImportError:
                pass

        return CraftaxEnvironment(
            craftax_env=craftax_env,
            is_classic=is_classic,
            observation_mode=obs_mode,
            max_steps=max_steps,
            num_tasks=num_tasks,
            answer_extractor=answer_extractor,
            extra_rewards=extra_rewards,
            prompts=merged_prompts or None,
            _text_renderer=text_renderer,
        )

    def get_default_system_prompt(self, name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        preset = CRAFTAX_PRESETS.get(name, {})
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "description": preset.get(
                "description",
                f"Craftax environment: {name}",
            ),
        }
