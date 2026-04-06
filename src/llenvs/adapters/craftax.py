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

from llenvs.core.environment import EnvironmentSpec, StepResult, format_action_error
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


def _truncate_for_error(text: str, max_len: int = 100) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


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
    5: "interact",
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
            f"Invalid action '{_truncate_for_error(text)}'. Expected a number "
            f"(0-{self.num_actions - 1}) or one of: {valid}"
        )

    # Display-only annotations for actions that need clarification.
    _ACTION_HINTS: dict[str, str] = {
        "interact": (
            "a multi-purpose action applied to the tile you're facing, "
            "the effect depends on what's there (e.g., tree\u2192chop, "
            "ore\u2192mine, zombie\u2192attack, water\u2192drink, "
            "plants\u2192harvest, etc.)"
        ),
    }

    def describe(self) -> str:
        lines = []
        for _, name in sorted(self._actions.items()):
            hint = self._ACTION_HINTS.get(name)
            if hint:
                lines.append(f"  {name} — {hint}")
            else:
                lines.append(f"  {name}")
        return f"Choose one of the following actions:\n" + "\n".join(lines)

    def format_action(self, action: int) -> str:
        return self._actions.get(action, str(action))


# =============================================================================
# Observation rendering
# =============================================================================

# Classic BlockType value → single-character grid symbol.
# All terrain is lowercase/symbols; entities (mobs, player) are uppercase.
# Source: craftax/craftax_classic/constants.py:25-42
_CLASSIC_BLOCK_CHARS: dict[int, str] = {
    0: "?",   # INVALID
    1: "#",   # OUT_OF_BOUNDS
    2: ".",   # GRASS
    3: "~",   # WATER
    4: "s",   # STONE
    5: "t",   # TREE
    6: "w",   # WOOD
    7: "=",   # PATH
    8: "c",   # COAL
    9: "i",   # IRON
    10: "d",  # DIAMOND
    11: "&",  # CRAFTING_TABLE
    12: "f",  # FURNACE
    13: ":",  # SAND
    14: "!",  # LAVA
    15: "p",  # PLANT
    16: "r",  # RIPE_PLANT
}

# Classic direction value → name.
# Source: craftax/craftax_classic/constants.py:66-72, values 1-4
_CLASSIC_DIRECTION_NAMES: dict[int, str] = {
    1: "left",
    2: "right",
    3: "up",
    4: "down",
}

# Classic OBS_DIM.
# Source: craftax/craftax_classic/constants.py:13
_CLASSIC_OBS_ROWS = 7
_CLASSIC_OBS_COLS = 9


def render_craftax_classic_text(state: Any) -> str:
    """Render a Craftax Classic state as a compact ASCII-grid observation.

    Reads directly from the Classic ``EnvState`` (not the flat symbolic array).
    Produces a human-/LLM-readable grid with terrain, mobs, inventory, and vitals.

    Source layout: ``craftax/craftax_classic/renderer.py`` (symbolic renderer)
    and ``craftax/craftax_classic/envs/craftax_state.py`` (state fields).
    """
    game_map = np.asarray(state.map)
    pr, pc = int(state.player_position[0]), int(state.player_position[1])

    # Pad the map so edge players see OUT_OF_BOUNDS (#) beyond map borders.
    pad = max(_CLASSIC_OBS_ROWS, _CLASSIC_OBS_COLS)
    padded = np.pad(game_map, pad, constant_values=1)  # 1 = OUT_OF_BOUNDS

    # Extract visible 7×9 window centered on player (in padded coordinates).
    r0 = pr + pad - _CLASSIC_OBS_ROWS // 2
    c0 = pc + pad - _CLASSIC_OBS_COLS // 2
    view = padded[r0:r0 + _CLASSIC_OBS_ROWS, c0:c0 + _CLASSIC_OBS_COLS]

    # Build character grid from block types.
    grid = [[_CLASSIC_BLOCK_CHARS.get(int(view[r, c]), "?")
             for c in range(_CLASSIC_OBS_COLS)]
            for r in range(_CLASSIC_OBS_ROWS)]

    # Overlay mobs. Each mob group has .position (N×2) and .mask (N,).
    mob_groups = [
        (state.zombies, "Z"),
        (state.cows, "C"),
        (state.skeletons, "K"),
        (state.arrows, "A"),
    ]
    for mobs, char in mob_groups:
        positions = np.asarray(mobs.position)
        masks = np.asarray(mobs.mask)
        for idx in range(len(masks)):
            if not masks[idx]:
                continue
            mr, mc = int(positions[idx, 0]), int(positions[idx, 1])
            # Convert to grid-local coordinates.
            gr = mr - pr + _CLASSIC_OBS_ROWS // 2
            gc = mc - pc + _CLASSIC_OBS_COLS // 2
            if 0 <= gr < _CLASSIC_OBS_ROWS and 0 <= gc < _CLASSIC_OBS_COLS:
                grid[gr][gc] = char

    # Place player at center.
    grid[_CLASSIC_OBS_ROWS // 2][_CLASSIC_OBS_COLS // 2] = "@"

    # --- Format output ---
    parts: list[str] = []

    # Direction header.
    direction = _CLASSIC_DIRECTION_NAMES.get(int(state.player_direction), "unknown")
    parts.append(f"Nearby (7x9 view, you=@ facing {direction}):")

    # Grid rows — 2-char-wide columns for uniform spacing.
    for row in grid:
        parts.append("  " + " ".join(row))

    # Legend (always shown).
    parts.append("")
    parts.append(
        "Terrain: .=grass ~=water t=tree c=coal s=stone"
    )
    parts.append(
        "         d=diamond i=iron w=wood ==path :=sand"
    )
    parts.append(
        "         !=lava f=furnace &=table p=plant r=ripe"
    )
    parts.append(
        "         #=border"
    )
    parts.append("Entities: Z=zombie C=cow K=skeleton A=arrow")

    # Inventory — only non-zero items.
    inv = state.inventory
    inv_fields = [
        ("wood", inv.wood), ("stone", inv.stone), ("coal", inv.coal),
        ("iron", inv.iron), ("diamond", inv.diamond), ("sapling", inv.sapling),
        ("wood_pickaxe", inv.wood_pickaxe), ("stone_pickaxe", inv.stone_pickaxe),
        ("iron_pickaxe", inv.iron_pickaxe), ("wood_sword", inv.wood_sword),
        ("stone_sword", inv.stone_sword), ("iron_sword", inv.iron_sword),
    ]
    non_zero = [(name, int(val)) for name, val in inv_fields if int(val) > 0]
    parts.append("")
    if non_zero:
        items_str = ", ".join(f"{name} {val}" for name, val in non_zero)
        parts.append(f"Inventory: {items_str}")
    else:
        parts.append("Inventory: (empty)")

    # Vitals.
    parts.append(
        f"Health: {int(state.player_health)}  "
        f"Food: {int(state.player_food)}  "
        f"Drink: {int(state.player_drink)}  "
        f"Energy: {int(state.player_energy)}"
    )

    # Light.
    light = "day" if float(state.light_level) > 0.5 else "night"
    parts.append(f"Light: {light}")

    # Sleeping (only shown when true).
    if bool(state.is_sleeping):
        parts.append("Status: sleeping")

    return "\n".join(parts)


def _render_symbolic(obs: np.ndarray, is_classic: bool) -> str:
    """Parse the flat symbolic observation array into structured text.

    The observation is a flat array produced by Craftax's ``render_craftax_symbolic``.
    Layout verified against ``craftax/craftax_classic/renderer.py`` (Classic) and
    ``craftax/craftax/renderer.py`` (Full).
    """
    obs = np.asarray(obs).flatten()

    parts: list[str] = []
    parts.append("=== Craftax Observation ===")

    if is_classic:
        _render_symbolic_classic(obs, parts)
    else:
        _render_symbolic_full(obs, parts)

    return "\n".join(parts)


def _render_symbolic_classic(obs: np.ndarray, parts: list[str]) -> None:
    """Classic: 1345-dim.

    Layout: map (7*9*21=1323) | inventory (12) | intrinsics (4) |
            direction (4) | misc (2).
    Normalization: inventory = count/10, intrinsics = value/10.
    """
    map_end = 1323  # 7 * 9 * (17 BlockTypes + 4 mob channels)

    if len(obs) <= map_end:
        return

    # Inventory: 12 items, each normalized as count / 10.
    inv_start = map_end
    inv_end = inv_start + 12
    if inv_end <= len(obs):
        inv = obs[inv_start:inv_end]
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
        parts.append("Inventory:")
        for i, label in enumerate(inv_labels):
            parts.append(f"  {label}: {round(inv[i] * 10)}")

    # Intrinsics: 4 values (health, food, drink, energy). No mana in Classic.
    # Each normalized as value / 10.
    intr_start = inv_end
    intr_end = intr_start + 4
    if intr_end <= len(obs):
        intr = obs[intr_start:intr_end]
        parts.append(f"Health: {round(intr[0] * 10)}")
        parts.append(f"Food: {round(intr[1] * 10)}")
        parts.append(f"Drink: {round(intr[2] * 10)}")
        parts.append(f"Energy: {round(intr[3] * 10)}")


def _render_symbolic_full(obs: np.ndarray, parts: list[str]) -> None:
    """Full: 8268-dim.

    Layout: map (9*11*83=8217) | inventory (16) | potions (6) |
            intrinsics (9) | direction (4) | armour (4) |
            armour_enchantments (4) | special (8).
    """
    map_end = 8217  # 9 * 11 * (37 BlockTypes + 5 ItemTypes + 40 mob channels + 1 light)

    if len(obs) <= map_end:
        return

    # --- Inventory: 16 items with mixed normalization ---
    inv_start = map_end
    inv_end = inv_start + 16
    if inv_end <= len(obs):
        inv = obs[inv_start:inv_end]
        # Items 0-9: sqrt(count)/10  → count = round((val*10)^2)
        # Item 10 (books): count/2   → count = round(val*2)
        # Items 11-12 (pickaxe, sword): level/4 → level = round(val*4)
        # Items 13-15 (sword_ench, bow_ench, bow): raw → round(val)
        sqrt_labels = [
            "wood", "stone", "coal", "iron", "diamond",
            "sapphire", "ruby", "sapling", "torches", "arrows",
        ]
        parts.append("Inventory:")
        for i, label in enumerate(sqrt_labels):
            parts.append(f"  {label}: {round((inv[i] * 10) ** 2)}")
        parts.append(f"  books: {round(inv[10] * 2)}")
        parts.append(f"  pickaxe: {round(inv[11] * 4)}")
        parts.append(f"  sword: {round(inv[12] * 4)}")
        parts.append(f"  sword_enchantment: {round(inv[13])}")
        parts.append(f"  bow_enchantment: {round(inv[14])}")
        parts.append(f"  bow: {round(inv[15])}")

    # --- Potions: 6 types, each sqrt(count)/10 ---
    pot_start = inv_end
    pot_end = pot_start + 6
    if pot_end <= len(obs):
        pot = obs[pot_start:pot_end]
        pot_labels = [
            "red_potion", "green_potion", "blue_potion",
            "pink_potion", "cyan_potion", "yellow_potion",
        ]
        parts.append("Potions:")
        for i, label in enumerate(pot_labels):
            parts.append(f"  {label}: {round((pot[i] * 10) ** 2)}")

    # --- Intrinsics: 9 values, each value/10 ---
    intr_start = pot_end
    intr_end = intr_start + 9
    if intr_end <= len(obs):
        intr = obs[intr_start:intr_end]
        intr_labels = [
            "Health", "Food", "Drink", "Energy", "Mana",
            "XP", "Dexterity", "Strength", "Intelligence",
        ]
        for i, label in enumerate(intr_labels):
            parts.append(f"{label}: {round(intr[i] * 10)}")

    # --- Direction (4), then Armour (4), Armour enchantments (4), Special (8) ---
    dir_start = intr_end
    dir_end = dir_start + 4

    armour_start = dir_end
    armour_end = armour_start + 4
    ench_start = armour_end
    ench_end = ench_start + 4

    if armour_end <= len(obs):
        armour = obs[armour_start:armour_end]
        armour_labels = ["helmet", "chestplate", "leggings", "boots"]
        ench = obs[ench_start:ench_end] if ench_end <= len(obs) else np.zeros(4)
        ench_names = {0: "none", 1: "fire", 2: "ice"}
        parts.append("Armour:")
        for i, label in enumerate(armour_labels):
            level = round(armour[i] * 2)
            ench_val = round(ench[i]) if i < len(ench) else 0
            ench_name = ench_names.get(ench_val, "none")
            parts.append(f"  {label}: {level} (enchantment: {ench_name})")

    special_start = ench_end
    special_end = special_start + 8
    if special_end <= len(obs):
        sp = obs[special_start:special_end]
        parts.append(f"Floor: {round(sp[5] * 10)}")
        if sp[6] > 0.5:
            parts.append("Ladder: open")
        if sp[7] > 0.5:
            parts.append("Boss: vulnerable")
        if sp[3] > 0.5:
            parts.append("Learned: fireball")
        if sp[4] > 0.5:
            parts.append("Learned: iceball")


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
    last_obs_text: str = ""


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
        invalid_action_text: str | None = "[invalid action]",
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
        self._invalid_action_text = invalid_action_text
        self._text_renderer = _text_renderer

        # Resolve max_steps
        if max_steps is not None:
            self._max_steps = max_steps
        else:
            params = craftax_env.default_params
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
    def task_description(self) -> str:
        """Static task description (game rules + action space), suitable for system prompt."""
        return self._build_task_description()

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

    def _build_task_description(self) -> str:
        """Build the static task description (no step-specific observation)."""
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
        parts.append(
            "Each turn you see a 7x9 view of your surroundings and your status. "
            "The world extends beyond what you can see."
        )
        parts.append("")
        parts.append("Action space (same actions are available every turn):")
        parts.append(self._action_mapper.describe())

        return "\n".join(parts)

    def _build_error_step(
        self,
        state: State[CraftaxHidden],
        action: Action,
        error_msg: str,
        *,
        assistant_content_override: str | None = None,
        extracted_action: str | None = None,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> StepResult[CraftaxHidden]:
        """Build a StepResult for an invalid action (wasted turn).

        Args:
            state: Current state.
            action: The failed action.
            error_msg: Error description for the observation.
            assistant_content_override: If set, use this instead of
                ``action.text`` as the assistant message content in history.
            extracted_action: Extracted action text (None if extraction failed).
            extraction_metadata: Metadata from the extraction step.
        """
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
            last_obs_text=state.hidden.last_obs_text,
        )

        error_text = format_action_error(
            error_msg,
            current_state=state.hidden.last_obs_text or None,
            action_hint=self._action_mapper.describe(),
        )
        state_content = ObservationContent(text=error_text)

        assistant_content = (
            assistant_content_override
            if assistant_content_override is not None
            else (action.text or "")
        )
        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": error_text},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=state_content,
        )

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

        error_info: dict[str, Any] = {"error": error_msg}
        if extraction_metadata is not None:
            error_info["extraction_metadata"] = extraction_metadata

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=False,
            truncated=truncated,
            extracted_action=extracted_action,
            info=error_info,
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
        params = self._craftax_env.default_params
        raw_obs, craftax_state = self._craftax_env.reset(rng_key, params)

        # Render observation
        obs_text, images = self._render_observation(raw_obs, craftax_state)

        # Task content = max-turns notice (static game description + action
        # space lives in the system prompt, provided via ``task_description``).
        # State content = dynamic per-step observation (map, HUD, inventory).
        # At step 0 both are emitted as separate user messages and coalesced
        # by the runner into one; on later steps the task message carries only
        # the max-turns text while the state message has the current observation.
        if self._max_steps is not None:
            task_text = f"You have a maximum of {self._max_steps} turns."
        else:
            task_text = ""
        task_content = ObservationContent(text=task_text)
        state_content = ObservationContent(
            text=obs_text,
            images=images,
            data={"episode_step": 0, "cumulative_reward": 0.0},
        )

        prompt = self._build_task_description() + "\n\n" + obs_text

        step_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            step_msg["images"] = [
                {"data": img.data, "media_type": img.media_type} for img in images
            ]
        initial_messages: tuple[dict[str, Any], ...] = (step_msg,)

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
            last_obs_text=obs_text,
        )

        observation = Observation(
            prompt=prompt,
            messages=initial_messages,
            images=images,
            task=task_content,
            state=state_content,
        )

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
            return self._build_error_step(
                state,
                action,
                "Could not extract action from response.",
                assistant_content_override=self._invalid_action_text,
                extraction_metadata=extraction_meta,
            )

        # Map to Craftax action
        try:
            action_idx = self._action_mapper.map(extracted)
        except ValueError as e:
            return self._build_error_step(
                state,
                action,
                str(e),
                assistant_content_override=self._invalid_action_text,
                extracted_action=extracted,
                extraction_metadata=extraction_meta,
            )

        # Split RNG key
        keys = self._jax_random.split(state.hidden.rng_key)
        step_key = keys[0]
        next_rng_key = keys[1]

        # Step Craftax env
        params = self._craftax_env.default_params
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
            last_obs_text=obs_text,
        )

        state_text = obs_text
        state_content = ObservationContent(
            text=state_text,
            images=images,
            data={"episode_step": next_step, "cumulative_reward": cumulative_reward},
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": state_text},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            images=images,
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

        resolved_str = self._action_mapper.format_action(action_idx)
        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            extracted_action=extracted,
            resolved_action=resolved_str,
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
        "observation_mode": "text",
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
        invalid_action_text: str | None = "[invalid action]",
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

        # Get text renderer for text mode
        text_renderer = None
        if obs_mode == "text":
            if is_classic:
                text_renderer = render_craftax_classic_text
            else:
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
            invalid_action_text=invalid_action_text,
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
