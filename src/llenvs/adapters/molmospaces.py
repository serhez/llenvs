"""MolmoSpaces adapter — wraps MolmoSpaces robot manipulation benchmark as MDP environments.

MolmoSpaces (AI2, 2026) is a large-scale robot manipulation and navigation
research platform built on MuJoCo. It provides 8 benchmark tasks (pick, place,
navigate, open/close, etc.) with multi-view cameras, proprioceptive sensors,
and continuous robot control via IK solvers and controllers.

Key design: MolmoSpaces operates at physics-step frequency (~50Hz) with
continuous control. LLMs/VLMs cannot operate at this speed. We provide
**temporal abstraction**: each llenvs ``step()`` = one high-level VLM decision,
executed over many physics steps by a low-level controller. Tool-based
environment where the VLM uses function-calling tools (``move_end_effector``,
``grasp``, ``release``, ``navigate_to``) that internally delegate to
MolmoSpaces' IK solver and controller.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from llenvs.core.environment import (
    EnvironmentSpec,
    StepResult,
    _StateContinuityTracker,
)
from llenvs.core.reward import (
    RewardFunction,
    RewardType,
    Signal,
    SignalBundle,
)
from llenvs.core.state import Action, ImageContent, Observation, State, StateMetadata
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

MOLMOSPACES_TASKS: tuple[str, ...] = (
    "navigate-to",
    "pick",
    "pick-and-place",
    "pick-and-place-next-to",
    "pick-and-place-color",
    "open",
    "close",
    "open-door",
)

MOLMOSPACES_PRESETS: dict[str, dict[str, Any]] = {
    "navigate-to": {
        "max_steps": 20,
        "description": "Navigate the robot base to a target location in the scene.",
    },
    "pick": {
        "max_steps": 30,
        "description": "Pick up a specified object from the scene.",
    },
    "pick-and-place": {
        "max_steps": 40,
        "description": "Pick up an object and place it at a target location.",
    },
    "pick-and-place-next-to": {
        "max_steps": 40,
        "description": "Pick up an object and place it next to another object.",
    },
    "pick-and-place-color": {
        "max_steps": 40,
        "description": "Pick up an object of a specific color and place it at a target.",
    },
    "open": {
        "max_steps": 30,
        "description": "Open a container or drawer in the scene.",
    },
    "close": {
        "max_steps": 30,
        "description": "Close a container or drawer in the scene.",
    },
    "open-door": {
        "max_steps": 30,
        "description": "Open a door by manipulating its handle.",
    },
}


# ── Image rendering ──────────────────────────────────────────────


def _render_pixels_to_image(
    arr: np.ndarray,
    image_format: str = "png",
    image_quality: int = 85,
    is_depth: bool = False,
) -> ImageContent:
    """Convert a pixel array to base64 ImageContent.

    Args:
        arr: HxWx3 uint8 RGB array or HxW float32 depth array.
        image_format: ``"png"`` or ``"jpeg"``.
        image_quality: JPEG quality (1-100), ignored for PNG.
        is_depth: If True, apply colormap to depth array before encoding.

    Returns:
        ImageContent with base64-encoded image data.
    """
    try:
        from PIL import Image
    except ImportError:
        return _numpy_to_png_content(arr, is_depth=is_depth)

    if is_depth:
        # Normalize depth to 0-255 and apply grayscale
        depth = np.asarray(arr).astype(np.float32)
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint8 = (depth * 255).astype(np.uint8)
        img = Image.fromarray(depth_uint8, mode="L").convert("RGB")
    else:
        img = Image.fromarray(np.asarray(arr).astype(np.uint8))

    buf = io.BytesIO()
    fmt = image_format.upper()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=image_quality)
        media_type = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        media_type = "image/png"

    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return ImageContent(data=data, media_type=media_type)


def _numpy_to_png_content(arr: np.ndarray, is_depth: bool = False) -> ImageContent:
    """Minimal PNG encoding without PIL."""
    import struct
    import zlib

    if is_depth:
        depth = np.asarray(arr).astype(np.float32)
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        arr_uint8 = (depth * 255).astype(np.uint8)
        # Convert to 3-channel
        arr_uint8 = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
    else:
        arr_uint8 = np.asarray(arr).astype(np.uint8)

    h, w = arr_uint8.shape[:2]
    channels = arr_uint8.shape[2] if arr_uint8.ndim == 3 else 1

    raw = b""
    for row in range(h):
        raw += b"\x00"
        raw += arr_uint8[row].tobytes()

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


# ── Tool definitions ─────────────────────────────────────────────


def _build_tools(
    is_mobile: bool = False,
    controller_type: str = "ik",
) -> tuple[ToolDefinition, ...]:
    """Build tool definitions based on robot capabilities.

    Args:
        is_mobile: Whether the robot has a mobile base (e.g. RBY1).
        controller_type: ``"ik"`` or ``"joint"``.

    Returns:
        Tuple of ToolDefinition objects.
    """
    tools: list[ToolDefinition] = []

    # move_end_effector — always available
    tools.append(
        ToolDefinition(
            name="move_end_effector",
            description=(
                "Move the robot's end-effector to a target position in 3D space. "
                "Optionally specify orientation as a quaternion (qw, qx, qy, qz). "
                "The IK solver computes joint positions and the controller executes "
                "the motion over multiple physics steps."
            ),
            parameters=(
                ToolParameter(
                    name="x",
                    type=ToolParameterType.NUMBER,
                    description="Target X position in meters.",
                    required=True,
                ),
                ToolParameter(
                    name="y",
                    type=ToolParameterType.NUMBER,
                    description="Target Y position in meters.",
                    required=True,
                ),
                ToolParameter(
                    name="z",
                    type=ToolParameterType.NUMBER,
                    description="Target Z position in meters.",
                    required=True,
                ),
                ToolParameter(
                    name="qw",
                    type=ToolParameterType.NUMBER,
                    description="Quaternion W component for orientation.",
                    required=False,
                ),
                ToolParameter(
                    name="qx",
                    type=ToolParameterType.NUMBER,
                    description="Quaternion X component for orientation.",
                    required=False,
                ),
                ToolParameter(
                    name="qy",
                    type=ToolParameterType.NUMBER,
                    description="Quaternion Y component for orientation.",
                    required=False,
                ),
                ToolParameter(
                    name="qz",
                    type=ToolParameterType.NUMBER,
                    description="Quaternion Z component for orientation.",
                    required=False,
                ),
            ),
        )
    )

    # grasp
    tools.append(
        ToolDefinition(
            name="grasp",
            description="Close the gripper to grasp an object at the current end-effector position.",
            parameters=(),
        )
    )

    # release
    tools.append(
        ToolDefinition(
            name="release",
            description="Open the gripper to release a held object.",
            parameters=(),
        )
    )

    # navigate_to — only for mobile robots
    if is_mobile:
        tools.append(
            ToolDefinition(
                name="navigate_to",
                description=(
                    "Move the robot base to a target 2D position. "
                    "Only available for mobile robots (e.g. RBY1)."
                ),
                parameters=(
                    ToolParameter(
                        name="x",
                        type=ToolParameterType.NUMBER,
                        description="Target X position for the base.",
                        required=True,
                    ),
                    ToolParameter(
                        name="y",
                        type=ToolParameterType.NUMBER,
                        description="Target Y position for the base.",
                        required=True,
                    ),
                ),
            )
        )

    # set_joints — only for joint-based controllers
    if controller_type == "joint":
        tools.append(
            ToolDefinition(
                name="set_joints",
                description=(
                    "Directly set target joint positions. Expert mode — bypasses IK solver. "
                    "Provide an array of joint angles in radians."
                ),
                parameters=(
                    ToolParameter(
                        name="positions",
                        type=ToolParameterType.ARRAY,
                        description="Array of target joint positions in radians.",
                        required=True,
                    ),
                ),
            )
        )

    return tuple(tools)


# ── Hidden state ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MolmoSpacesHidden:
    """Hidden state for MolmoSpaces environments.

    Attributes:
        task_index: Index into the task list.
        task_name: Name of the current task.
        episode_step: Current high-level step in the episode.
        physics_steps: Total physics steps executed this episode.
        last_action: Name of the last tool action taken.
        ee_position: End-effector position (x, y, z).
        ee_orientation: End-effector orientation quaternion (w, x, y, z).
        gripper_state: Current gripper state ("open" or "closed").
        joint_positions: Current joint positions.
        cumulative_reward: Cumulative reward for this episode.
        task_instruction: Natural language task instruction.
        success: Whether the task was completed successfully.
    """

    task_index: int
    task_name: str
    episode_step: int
    physics_steps: int
    last_action: str | None
    ee_position: tuple[float, ...]
    ee_orientation: tuple[float, ...]
    gripper_state: str
    joint_positions: tuple[float, ...]
    cumulative_reward: float
    task_instruction: str
    success: bool


# ── Tool executor ────────────────────────────────────────────────


class MolmoSpacesToolExecutor:
    """Executes tool calls by delegating to MolmoSpaces' IK solver and controller.

    Each tool call translates to many physics steps, providing temporal
    abstraction between the VLM's high-level decisions and the robot's
    low-level control.
    """

    def __init__(
        self,
        molmospaces_env: Any,
        robot: Any,
        ik_solver: Any,
        controller: Any,
        max_physics_steps_per_action: int = 200,
    ) -> None:
        self._env = molmospaces_env
        self._robot = robot
        self._ik_solver = ik_solver
        self._controller = controller
        self._max_physics_steps_per_action = max_physics_steps_per_action

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        dispatch = {
            "move_end_effector": self._execute_move_ee,
            "grasp": self._execute_grasp,
            "release": self._execute_release,
            "navigate_to": self._execute_navigate,
            "set_joints": self._execute_set_joints,
        }
        handler = dispatch.get(call.name)
        if handler is None:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"Unknown tool: {call.name}",
            )
        try:
            return handler(call)
        except Exception as e:
            logger.warning(f"MolmoSpaces tool {call.name} failed: {e}")
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=str(e),
            )

    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls sequentially."""
        return tuple(self.execute(call) for call in calls)

    def _execute_move_ee(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        target_pos = np.array([float(args["x"]), float(args["y"]), float(args["z"])])

        target_orient = None
        if "qw" in args and args["qw"] is not None:
            target_orient = np.array(
                [
                    float(args["qw"]),
                    float(args["qx"]),
                    float(args["qy"]),
                    float(args["qz"]),
                ]
            )

        # Solve IK
        ik_result = self._ik_solver.solve(target_pos, target_orient)
        if not ik_result.success:
            return ToolResult.from_error(
                call_id=call.id,
                tool_name=call.name,
                error_message=f"IK solver failed: {ik_result.error}",
            )

        # Step controller until convergence
        ctrl_result = self._controller.step_until_converged(
            ik_result.joint_positions, max_steps=self._max_physics_steps_per_action
        )

        # Update robot state
        self._robot.ee_position = target_pos
        if target_orient is not None:
            self._robot.ee_orientation = target_orient
        self._robot.joint_positions = ik_result.joint_positions

        output = json.dumps(
            {
                "physics_steps": ctrl_result["steps"],
                "converged": ctrl_result["converged"],
                "final_position": target_pos.tolist(),
            }
        )
        return ToolResult.success(call_id=call.id, tool_name=call.name, output=output)

    def _execute_grasp(self, call: ToolCall) -> ToolResult:
        result = self._robot.gripper.close()
        output = json.dumps(
            {
                "gripper_state": result["state"],
                "force": result.get("force", 0.0),
                "physics_steps": 1,
            }
        )
        return ToolResult.success(call_id=call.id, tool_name=call.name, output=output)

    def _execute_release(self, call: ToolCall) -> ToolResult:
        result = self._robot.gripper.open()
        output = json.dumps(
            {
                "gripper_state": result["state"],
                "force": result.get("force", 0.0),
                "physics_steps": 1,
            }
        )
        return ToolResult.success(call_id=call.id, tool_name=call.name, output=output)

    def _execute_navigate(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        target = np.array([float(args["x"]), float(args["y"])])

        # Step controller for base movement
        ctrl_result = self._controller.step_until_converged(
            target, max_steps=self._max_physics_steps_per_action
        )

        self._robot.base_position = target

        output = json.dumps(
            {
                "physics_steps": ctrl_result["steps"],
                "converged": ctrl_result["converged"],
                "final_position": target.tolist(),
            }
        )
        return ToolResult.success(call_id=call.id, tool_name=call.name, output=output)

    def _execute_set_joints(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        positions = np.array(args["positions"], dtype=np.float64)

        ctrl_result = self._controller.step_until_converged(
            positions, max_steps=self._max_physics_steps_per_action
        )

        self._robot.joint_positions = positions

        output = json.dumps(
            {
                "physics_steps": ctrl_result["steps"],
                "converged": ctrl_result["converged"],
            }
        )
        return ToolResult.success(call_id=call.id, tool_name=call.name, output=output)


# ── Reward functions ─────────────────────────────────────────────


@dataclass
class MolmoSpacesReward:
    """Primary reward from MolmoSpaces' ``task.get_reward()``.

    Uses STEP type for intermediate rewards (dense shaping),
    OUTCOME type for terminal steps.
    """

    _name: str = "molmospaces"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> Signal:
        is_terminal = next_state.metadata.is_terminal
        reward = next_state.metadata.info.get("molmospaces_reward", 0.0)
        reward_type = RewardType.OUTCOME if is_terminal else RewardType.STEP

        return Signal(
            name=self.name,
            reward_type=reward_type,
            reward=float(reward),
            metadata={"source": "molmospaces"},
        )


@dataclass
class MolmoSpacesSuccessReward:
    """Binary success reward from ``task.judge_success()``.

    OUTCOME type, only meaningful at terminal steps. Non-terminal steps
    return None reward. Available via ``extra_rewards`` (opt-in per
    wrapper fidelity).
    """

    _name: str = "molmospaces_success"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> Signal:
        is_terminal = next_state.metadata.is_terminal

        if not is_terminal:
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=None,
                metadata={"is_terminal": False},
            )

        success = getattr(next_state.hidden, "success", False)
        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=1.0 if success else 0.0,
            metadata={"is_terminal": True, "success": success},
        )


# ── Environment ──────────────────────────────────────────────────


class MolmoSpacesEnvironment(BaseToolEnvironment[MolmoSpacesHidden]):
    """MDP wrapper for MolmoSpaces robot manipulation benchmark.

    Tool-based environment where each ``step()`` is a high-level VLM
    decision executed over many physics steps by MolmoSpaces' low-level
    controller. Multi-image observations from configurable cameras.
    """

    def __init__(
        self,
        molmospaces_env: Any,
        tasks: list[Any],
        robot: Any,
        ik_solver: Any,
        controller: Any,
        cameras: tuple[str, ...] = ("front", "wrist"),
        image_format: str = "jpeg",
        image_quality: int = 80,
        include_proprioception: bool = True,
        include_depth: bool = False,
        max_steps: int | None = None,
        max_physics_steps_per_action: int = 200,
        is_mobile: bool = False,
        controller_type: str = "ik",
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._molmospaces_env = molmospaces_env
        self._tasks = tasks
        self._robot = robot
        self._ik_solver = ik_solver
        self._controller = controller
        self._cameras = cameras
        self._image_format = image_format
        self._image_quality = image_quality
        self._include_proprioception = include_proprioception
        self._include_depth = include_depth
        self._max_steps = max_steps
        self._max_physics_steps_per_action = max_physics_steps_per_action
        self._is_mobile = is_mobile
        self._controller_type = controller_type

        # Build tools based on robot capabilities
        self._tools = _build_tools(is_mobile=is_mobile, controller_type=controller_type)

        # Create tool executor
        self._executor = MolmoSpacesToolExecutor(
            molmospaces_env=molmospaces_env,
            robot=robot,
            ik_solver=ik_solver,
            controller=controller,
            max_physics_steps_per_action=max_physics_steps_per_action,
        )

        self._native_rewards: tuple[RewardFunction, ...] = (
            MolmoSpacesReward(),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="molmospaces",
            adapter="molmospaces",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=True,
            pure_step=False,
            metadata={
                "cameras": self._cameras,
                "is_mobile": self._is_mobile,
                "controller_type": self._controller_type,
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def __len__(self) -> int:
        return len(self._tasks)

    def _render_images(self) -> tuple[ImageContent, ...]:
        """Render camera images from the current MolmoSpaces state."""
        images: list[ImageContent] = []

        # RGB images
        rgb_imgs = self._molmospaces_env.get_observation_images(self._cameras)
        for cam_name in self._cameras:
            if cam_name in rgb_imgs:
                img = _render_pixels_to_image(
                    rgb_imgs[cam_name],
                    image_format=self._image_format,
                    image_quality=self._image_quality,
                )
                images.append(img)

        # Depth images
        if self._include_depth:
            depth_imgs = self._molmospaces_env.get_depth_images(self._cameras)
            for cam_name in self._cameras:
                if cam_name in depth_imgs:
                    img = _render_pixels_to_image(
                        depth_imgs[cam_name],
                        image_format="png",
                        is_depth=True,
                    )
                    images.append(img)

        return tuple(images)

    def _format_proprioception(self) -> str:
        """Format proprioceptive state as text."""
        proprio = self._robot.get_proprioception()
        parts = ["Robot state:"]
        if "ee_position" in proprio:
            pos = proprio["ee_position"]
            parts.append(f"  End-effector position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        if "ee_orientation" in proprio:
            ori = proprio["ee_orientation"]
            parts.append(
                f"  End-effector orientation: ({ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f})"
            )
        if "gripper_state" in proprio:
            parts.append(f"  Gripper: {proprio['gripper_state']}")
        if "joint_positions" in proprio:
            joints = proprio["joint_positions"]
            joints_str = ", ".join(f"{j:.3f}" for j in joints)
            parts.append(f"  Joint positions: [{joints_str}]")
        return "\n".join(parts)

    def _build_initial_prompt(self, task: Any) -> str:
        """Build the initial observation prompt."""
        parts = []
        instruction = getattr(task, "instruction", "")
        parts.append(f"Task: {instruction}")
        parts.append("")
        parts.append("[Visual observations are attached as images]")

        if self._include_proprioception:
            parts.append("")
            parts.append(self._format_proprioception())

        return "\n".join(parts)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[MolmoSpacesHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        task = self._tasks[task_index]

        # Reset MolmoSpaces environment
        self._molmospaces_env.reset(seed=seed)

        # Reset task
        task_info = task.reset(seed=seed)
        instruction = task_info.get("instruction", getattr(task, "instruction", ""))

        # Render images
        images = self._render_images()

        # Build initial prompt
        prompt = self._build_initial_prompt(task)

        # Read robot state
        proprio = self._robot.get_proprioception()

        hidden = MolmoSpacesHidden(
            task_index=task_index,
            task_name=getattr(task, "name", str(task_index)),
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=tuple(proprio.get("ee_position", [0.0, 0.0, 0.0])),
            ee_orientation=tuple(proprio.get("ee_orientation", [1.0, 0.0, 0.0, 0.0])),
            gripper_state=proprio.get("gripper_state", "open"),
            joint_positions=tuple(proprio.get("joint_positions", [])),
            cumulative_reward=0.0,
            task_instruction=instruction,
            success=False,
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        observation = Observation(
            prompt=prompt,
            images=images,
            available_tools=self._tools,
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        info: dict[str, Any] = {
            "task_index": task_index,
            "task_name": hidden.task_name,
            "instruction": instruction,
            "num_tools": len(self._tools),
        }

        return state, info

    def step(
        self,
        state: State[MolmoSpacesHidden],
        action: Action,
    ) -> StepResult[MolmoSpacesHidden]:
        self._state_tracker.validate(state, "MolmoSpacesEnvironment")

        next_step = state.hidden.episode_step + 1
        terminated = False
        truncated = False
        tool_results: list[ToolResult] = []
        physics_steps_this_step = 0

        if action.has_tool_calls:
            for tc in action.tool_calls:
                # Validate against known tools
                validation_error = self._validate_tool_call(tc)
                if validation_error is not None:
                    tool_results.append(validation_error)
                    continue

                # Execute through our executor
                result = self._executor.execute(tc)
                tool_results.append(result)

                # Count physics steps from result
                if result.is_success:
                    try:
                        output = (
                            json.loads(result.output)
                            if isinstance(result.output, str)
                            else result.output
                        )
                        physics_steps_this_step += output.get("physics_steps", 0)
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass

        total_physics_steps = state.hidden.physics_steps + physics_steps_this_step

        # Check task termination
        task = self._tasks[state.hidden.task_index]
        task_done = task.is_done()
        task_reward = task.get_reward()
        task_success = task.judge_success()
        cumulative_reward = state.hidden.cumulative_reward + task_reward

        if task_done:
            terminated = True

        # Check max_steps truncation
        if not terminated and self._max_steps is not None and next_step >= self._max_steps:
            truncated = True

        # Render new images
        images = self._render_images()

        # Build observation
        if tool_results:
            next_obs = self._build_next_observation(
                current_obs=state.observation,
                action=action,
                tool_results=tuple(tool_results),
            )
            # Add images to the observation
            next_obs = Observation(
                prompt=next_obs.prompt,
                messages=next_obs.messages,
                tool_results=next_obs.tool_results,
                available_tools=next_obs.available_tools,
                images=images,
            )
        else:
            next_obs = Observation(
                prompt=state.observation.prompt,
                messages=state.observation.messages,
                available_tools=self._tools,
                images=images,
            )

        # Add proprioception if enabled
        if self._include_proprioception:
            proprio_text = self._format_proprioception()
            # Append as a user message
            messages = list(next_obs.messages)
            messages.append({"role": "user", "content": f"[Step {next_step}]\n{proprio_text}"})
            next_obs = Observation(
                prompt=next_obs.prompt,
                messages=tuple(messages),
                tool_results=next_obs.tool_results,
                available_tools=next_obs.available_tools,
                images=next_obs.images,
            )

        # Read current robot state
        proprio = self._robot.get_proprioception()

        next_hidden = MolmoSpacesHidden(
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            episode_step=next_step,
            physics_steps=total_physics_steps,
            last_action=action.tool_calls[0].name if action.has_tool_calls else action.text,
            ee_position=tuple(proprio.get("ee_position", [0.0, 0.0, 0.0])),
            ee_orientation=tuple(proprio.get("ee_orientation", [1.0, 0.0, 0.0, 0.0])),
            gripper_state=proprio.get("gripper_state", "open"),
            joint_positions=tuple(proprio.get("joint_positions", [])),
            cumulative_reward=cumulative_reward,
            task_instruction=state.hidden.task_instruction,
            success=task_success,
        )

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                **state.metadata.info,
                "episode_step": next_step,
                "molmospaces_reward": task_reward,
                "physics_steps": total_physics_steps,
            },
        )

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "tool_results": tuple(tool_results),
                "episode_step": next_step,
                "physics_steps": total_physics_steps,
                "task_reward": task_reward,
                "task_success": task_success,
            },
        )

    def compute_rewards(
        self,
        state: State[MolmoSpacesHidden],
        action: Action,
        next_state: State[MolmoSpacesHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# ── Adapter ──────────────────────────────────────────────────────


class MolmoSpacesAdapter:
    """Adapter for the MolmoSpaces robot manipulation benchmark.

    Wraps MolmoSpaces' multi-view, multi-step robot manipulation tasks
    as tool-based environments for VLM agents.
    """

    @property
    def name(self) -> str:
        return "molmospaces"

    def _get_molmospaces(self) -> Any:
        try:
            import molmospaces

            return molmospaces
        except ImportError as e:
            raise ImportError(
                "molmospaces is required for MolmoSpacesAdapter. "
                "Install with: pip install git+https://github.com/allenai/molmospaces.git"
            ) from e

    @staticmethod
    def _parse_task_name(name: str) -> str:
        """Extract task name from environment name like 'molmospaces:pick'."""
        parts = name.split(":")
        if len(parts) >= 2:
            return parts[1]
        return ""

    def list_environments(self) -> list[str]:
        return [f"molmospaces:{task}" for task in MOLMOSPACES_TASKS]

    def get_environment(
        self,
        name: str,
        molmospaces_env: Any | None = None,
        tasks: list[Any] | None = None,
        robot: Any | None = None,
        ik_solver: Any | None = None,
        controller: Any | None = None,
        cameras: tuple[str, ...] = ("front", "wrist"),
        image_format: str = "jpeg",
        image_quality: int = 80,
        include_proprioception: bool = True,
        include_depth: bool = False,
        max_steps: int | None = None,
        max_physics_steps_per_action: int = 200,
        is_mobile: bool = False,
        controller_type: str = "ik",
        robot_type: str = "franka",
        benchmark_path: str | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> MolmoSpacesEnvironment:
        """Create a MolmoSpaces environment.

        Args:
            name: Environment name (e.g., "molmospaces:pick").
            molmospaces_env: Pre-created MolmoSpaces environment.
            tasks: Pre-loaded task list.
            robot: Pre-created robot instance.
            ik_solver: Pre-created IK solver.
            controller: Pre-created controller.
            cameras: Camera names to render.
            image_format: Image encoding format ("jpeg" or "png").
            image_quality: JPEG quality (1-100).
            include_proprioception: Include proprioceptive state in text.
            include_depth: Include depth images.
            max_steps: Maximum high-level steps per episode.
            max_physics_steps_per_action: Physics budget per tool call.
            is_mobile: Whether robot has mobile base.
            controller_type: "ik" or "joint".
            robot_type: Robot model ("franka" or "rby1").
            benchmark_path: Path to benchmark data.
            extra_rewards: Additional reward functions.
            **kwargs: Additional arguments.

        Returns:
            Configured MolmoSpacesEnvironment.
        """
        task_name = self._parse_task_name(name)

        # Resolve max_steps from preset if not provided
        if max_steps is None:
            preset = MOLMOSPACES_PRESETS.get(task_name, {})
            max_steps = preset.get("max_steps")

        # Auto-detect mobility from robot_type
        if robot_type == "rby1":
            is_mobile = True

        # If objects not provided, load from MolmoSpaces library
        if molmospaces_env is None or tasks is None or robot is None:
            ms = self._get_molmospaces()
            if molmospaces_env is None:
                molmospaces_env = ms.make_env(
                    task=task_name,
                    robot_type=robot_type,
                    benchmark_path=benchmark_path,
                )
            if tasks is None:
                tasks = ms.load_tasks(task_name, benchmark_path=benchmark_path)
            if robot is None:
                robot = molmospaces_env.robot
            if ik_solver is None:
                ik_solver = molmospaces_env.ik_solver
            if controller is None:
                controller = molmospaces_env.controller

        return MolmoSpacesEnvironment(
            molmospaces_env=molmospaces_env,
            tasks=tasks,
            robot=robot,
            ik_solver=ik_solver,
            controller=controller,
            cameras=cameras,
            image_format=image_format,
            image_quality=image_quality,
            include_proprioception=include_proprioception,
            include_depth=include_depth,
            max_steps=max_steps,
            max_physics_steps_per_action=max_physics_steps_per_action,
            is_mobile=is_mobile,
            controller_type=controller_type,
            extra_rewards=extra_rewards,
        )

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        task_name = self._parse_task_name(name)
        preset = MOLMOSPACES_PRESETS.get(task_name, {})
        return {
            "name": name,
            "adapter": self.name,
            "task": task_name,
            "description": preset.get(
                "description",
                f"MolmoSpaces robot manipulation task: {task_name}",
            ),
            "tasks": list(MOLMOSPACES_TASKS),
        }
