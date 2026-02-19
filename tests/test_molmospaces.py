"""Tests for the MolmoSpaces adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from llenvs.adapters.molmospaces import (
    MOLMOSPACES_PRESETS,
    MOLMOSPACES_TASKS,
    MolmoSpacesAdapter,
    MolmoSpacesEnvironment,
    MolmoSpacesHidden,
    MolmoSpacesReward,
    MolmoSpacesSuccessReward,
    MolmoSpacesToolExecutor,
    _render_pixels_to_image,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, ImageContent, State, StateMetadata
from llenvs.core.tools import ToolCall

# ── Mock objects ─────────────────────────────────────────────────


@dataclass
class MockIKResult:
    """Mock IK solver result."""

    joint_positions: np.ndarray
    success: bool = True
    error: str | None = None


class MockIKSolver:
    """Mock IK solver that returns preconfigured joint positions."""

    def __init__(self, success: bool = True) -> None:
        self._success = success

    def solve(
        self, target_position: np.ndarray, target_orientation: np.ndarray | None = None
    ) -> MockIKResult:
        if not self._success:
            return MockIKResult(
                joint_positions=np.zeros(7), success=False, error="IK failed to converge"
            )
        return MockIKResult(joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))


class MockController:
    """Mock low-level controller that tracks step counts."""

    def __init__(self, steps_to_converge: int = 10) -> None:
        self._steps_to_converge = steps_to_converge
        self.total_steps = 0

    def step_until_converged(
        self, target_joints: np.ndarray, max_steps: int = 200
    ) -> dict[str, Any]:
        steps = min(self._steps_to_converge, max_steps)
        self.total_steps += steps
        return {"steps": steps, "converged": steps < max_steps}


class MockGripper:
    """Mock gripper with open/close."""

    def __init__(self) -> None:
        self.state = "open"

    def close(self) -> dict[str, Any]:
        self.state = "closed"
        return {"state": "closed", "force": 5.0}

    def open(self) -> dict[str, Any]:
        self.state = "open"
        return {"state": "open", "force": 0.0}


class MockCamera:
    """Mock camera that produces synthetic images."""

    def __init__(self, name: str, width: int = 64, height: int = 64) -> None:
        self.name = name
        self.width = width
        self.height = height

    def render(self) -> np.ndarray:
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def render_depth(self) -> np.ndarray:
        return np.random.rand(self.height, self.width).astype(np.float32)


class MockRobot:
    """Mock robot with EE state, gripper, and joint positions."""

    def __init__(self, is_mobile: bool = False) -> None:
        self.ee_position = np.array([0.5, 0.0, 0.3])
        self.ee_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
        self.joint_positions = np.zeros(7)
        self.gripper = MockGripper()
        self.is_mobile = is_mobile
        self.base_position = np.array([0.0, 0.0])

    def get_proprioception(self) -> dict[str, Any]:
        return {
            "ee_position": self.ee_position.tolist(),
            "ee_orientation": self.ee_orientation.tolist(),
            "gripper_state": self.gripper.state,
            "joint_positions": self.joint_positions.tolist(),
        }


class MockTask:
    """Mock MolmoSpaces task."""

    def __init__(
        self,
        name: str = "pick",
        instruction: str = "Pick up the red cube.",
        reward: float = 0.0,
        success: bool = False,
        done: bool = False,
    ) -> None:
        self.name = name
        self.instruction = instruction
        self._reward = reward
        self._success = success
        self._done = done

    def get_reward(self) -> float:
        return self._reward

    def judge_success(self) -> bool:
        return self._success

    def is_done(self) -> bool:
        return self._done

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._reward = 0.0
        self._success = False
        self._done = False
        return {"instruction": self.instruction}


class MockMolmoSpacesEnv:
    """Mock MolmoSpaces environment wrapping task, robot, cameras, etc."""

    def __init__(
        self,
        task: MockTask | None = None,
        robot: MockRobot | None = None,
        cameras: dict[str, MockCamera] | None = None,
        ik_solver: MockIKSolver | None = None,
        controller: MockController | None = None,
    ) -> None:
        self.task = task or MockTask()
        self.robot = robot or MockRobot()
        self.cameras = cameras or {
            "front": MockCamera("front"),
            "wrist": MockCamera("wrist"),
        }
        self.ik_solver = ik_solver or MockIKSolver()
        self.controller = controller or MockController()
        self.physics_steps = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.physics_steps = 0
        info = self.task.reset(seed=seed)
        return info

    def get_observation_images(self, camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
        result = {}
        for name in camera_names:
            if name in self.cameras:
                result[name] = self.cameras[name].render()
        return result

    def get_depth_images(self, camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
        result = {}
        for name in camera_names:
            if name in self.cameras:
                result[name] = self.cameras[name].render_depth()
        return result


# ── TestMolmoSpacesHidden ────────────────────────────────────────


class TestMolmoSpacesHidden:
    def test_frozen(self) -> None:
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.5, 0.0, 0.3),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(0.0,) * 7,
            cumulative_reward=0.0,
            task_instruction="Pick up the red cube.",
            success=False,
        )
        with pytest.raises(AttributeError):
            hidden.task_index = 1  # type: ignore[misc]

    def test_fields(self) -> None:
        hidden = MolmoSpacesHidden(
            task_index=3,
            task_name="navigate-to",
            episode_step=5,
            physics_steps=100,
            last_action="move_end_effector",
            ee_position=(0.5, 0.1, 0.3),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="closed",
            joint_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            cumulative_reward=0.5,
            task_instruction="Navigate to the table.",
            success=False,
        )
        assert hidden.task_index == 3
        assert hidden.task_name == "navigate-to"
        assert hidden.episode_step == 5
        assert hidden.physics_steps == 100
        assert hidden.last_action == "move_end_effector"
        assert hidden.gripper_state == "closed"
        assert hidden.cumulative_reward == 0.5
        assert hidden.success is False

    def test_defaults(self) -> None:
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="",
            success=False,
        )
        assert hidden.last_action is None
        assert hidden.cumulative_reward == 0.0
        assert hidden.success is False

    def test_equality(self) -> None:
        kwargs: dict[str, Any] = dict(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="Pick it.",
            success=False,
        )
        h1 = MolmoSpacesHidden(**kwargs)
        h2 = MolmoSpacesHidden(**kwargs)
        assert h1 == h2

    def test_different_success(self) -> None:
        kwargs: dict[str, Any] = dict(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="Pick it.",
            success=False,
        )
        h1 = MolmoSpacesHidden(**kwargs)
        h2 = MolmoSpacesHidden(**{**kwargs, "success": True})
        assert h1 != h2


# ── TestMolmoSpacesToolDefinitions ──────────────────────────────


class TestMolmoSpacesToolDefinitions:
    def _make_env(
        self, is_mobile: bool = False, controller_type: str = "ik"
    ) -> MolmoSpacesEnvironment:
        robot = MockRobot(is_mobile=is_mobile)
        ms_env = MockMolmoSpacesEnv(robot=robot)
        return MolmoSpacesEnvironment(
            molmospaces_env=ms_env,
            tasks=[MockTask()],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            is_mobile=is_mobile,
            controller_type=controller_type,
        )

    def test_base_tools_present(self) -> None:
        env = self._make_env()
        tool_names = {t.name for t in env.available_tools}
        assert "move_end_effector" in tool_names
        assert "grasp" in tool_names
        assert "release" in tool_names

    def test_no_navigate_for_stationary(self) -> None:
        env = self._make_env(is_mobile=False)
        tool_names = {t.name for t in env.available_tools}
        assert "navigate_to" not in tool_names

    def test_navigate_for_mobile(self) -> None:
        env = self._make_env(is_mobile=True)
        tool_names = {t.name for t in env.available_tools}
        assert "navigate_to" in tool_names

    def test_no_set_joints_for_ik(self) -> None:
        env = self._make_env(controller_type="ik")
        tool_names = {t.name for t in env.available_tools}
        assert "set_joints" not in tool_names

    def test_set_joints_for_joint_controller(self) -> None:
        env = self._make_env(controller_type="joint")
        tool_names = {t.name for t in env.available_tools}
        assert "set_joints" in tool_names

    def test_move_ee_has_parameters(self) -> None:
        env = self._make_env()
        move_tool = next(t for t in env.available_tools if t.name == "move_end_effector")
        param_names = {p.name for p in move_tool.parameters}
        assert "x" in param_names
        assert "y" in param_names
        assert "z" in param_names

    def test_grasp_no_required_parameters(self) -> None:
        env = self._make_env()
        grasp_tool = next(t for t in env.available_tools if t.name == "grasp")
        required = [p for p in grasp_tool.parameters if p.required]
        assert len(required) == 0

    def test_tool_descriptions_not_empty(self) -> None:
        env = self._make_env(is_mobile=True, controller_type="joint")
        for tool in env.available_tools:
            assert tool.description, f"Tool {tool.name} has no description"


# ── TestMolmoSpacesToolExecutor ──────────────────────────────────


class TestMolmoSpacesToolExecutor:
    def _make_executor(
        self,
        ik_success: bool = True,
        steps_to_converge: int = 10,
        is_mobile: bool = False,
    ) -> tuple[MolmoSpacesToolExecutor, MockMolmoSpacesEnv]:
        robot = MockRobot(is_mobile=is_mobile)
        ik_solver = MockIKSolver(success=ik_success)
        controller = MockController(steps_to_converge=steps_to_converge)
        ms_env = MockMolmoSpacesEnv(robot=robot, ik_solver=ik_solver, controller=controller)
        executor = MolmoSpacesToolExecutor(
            molmospaces_env=ms_env,
            robot=robot,
            ik_solver=ik_solver,
            controller=controller,
            max_physics_steps_per_action=200,
        )
        return executor, ms_env

    def test_move_ee_success(self) -> None:
        executor, _ = self._make_executor()
        call = ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3})
        result = executor.execute(call)
        assert result.is_success
        output = json.loads(result.output) if isinstance(result.output, str) else result.output
        assert "physics_steps" in output

    def test_move_ee_ik_failure(self) -> None:
        executor, _ = self._make_executor(ik_success=False)
        call = ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3})
        result = executor.execute(call)
        assert result.is_error
        assert "IK" in (result.error or "")

    def test_move_ee_with_orientation(self) -> None:
        executor, _ = self._make_executor()
        call = ToolCall(
            id="c1",
            name="move_end_effector",
            arguments={"x": 0.5, "y": 0.1, "z": 0.3, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        )
        result = executor.execute(call)
        assert result.is_success

    def test_grasp(self) -> None:
        executor, ms_env = self._make_executor()
        call = ToolCall(id="c1", name="grasp", arguments={})
        result = executor.execute(call)
        assert result.is_success
        assert ms_env.robot.gripper.state == "closed"

    def test_release(self) -> None:
        executor, ms_env = self._make_executor()
        # Close first
        ms_env.robot.gripper.close()
        call = ToolCall(id="c1", name="release", arguments={})
        result = executor.execute(call)
        assert result.is_success
        assert ms_env.robot.gripper.state == "open"

    def test_navigate_to(self) -> None:
        executor, _ = self._make_executor(is_mobile=True)
        call = ToolCall(id="c1", name="navigate_to", arguments={"x": 1.0, "y": 2.0})
        result = executor.execute(call)
        assert result.is_success

    def test_set_joints(self) -> None:
        executor, _ = self._make_executor()
        call = ToolCall(
            id="c1",
            name="set_joints",
            arguments={"positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
        )
        result = executor.execute(call)
        assert result.is_success

    def test_physics_step_counting(self) -> None:
        executor, ms_env = self._make_executor(steps_to_converge=25)
        call = ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3})
        result = executor.execute(call)
        assert result.is_success
        output = json.loads(result.output) if isinstance(result.output, str) else result.output
        assert output["physics_steps"] == 25

    def test_physics_step_budget_exceeded(self) -> None:
        executor, _ = self._make_executor(steps_to_converge=500)
        executor._max_physics_steps_per_action = 100
        call = ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3})
        result = executor.execute(call)
        # Should still succeed but with truncated steps
        assert result.is_success
        output = json.loads(result.output) if isinstance(result.output, str) else result.output
        assert output["physics_steps"] <= 100

    def test_unknown_tool(self) -> None:
        executor, _ = self._make_executor()
        call = ToolCall(id="c1", name="fly_away", arguments={})
        result = executor.execute(call)
        assert result.is_error

    def test_execute_batch(self) -> None:
        executor, _ = self._make_executor()
        calls = (
            ToolCall(id="c1", name="grasp", arguments={}),
            ToolCall(id="c2", name="release", arguments={}),
        )
        results = executor.execute_batch(calls)
        assert len(results) == 2
        assert all(r.is_success for r in results)


# ── TestMolmoSpacesReward ────────────────────────────────────────


class TestMolmoSpacesReward:
    def _make_states(
        self,
        is_terminal: bool = False,
        reward: float = 0.5,
        cumulative_reward: float = 1.0,
        success: bool = False,
    ) -> tuple[State, Action, State]:
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=1,
            physics_steps=10,
            last_action=None,
            ee_position=(0.5, 0.0, 0.3),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=cumulative_reward,
            task_instruction="Pick up the cube.",
            success=success,
        )
        from llenvs.core.state import Observation

        obs = Observation(prompt="test")
        state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test", is_terminal=False),
        )
        action = Action(text="test")
        next_state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(
                step=1,
                episode_id="test",
                is_terminal=is_terminal,
                info={"molmospaces_reward": reward},
            ),
        )
        return state, action, next_state

    def test_reward_name(self) -> None:
        reward = MolmoSpacesReward()
        assert reward.name == "molmospaces"

    def test_reward_type(self) -> None:
        reward = MolmoSpacesReward()
        assert reward.reward_type == RewardType.OUTCOME

    def test_step_reward(self) -> None:
        reward_fn = MolmoSpacesReward()
        state, action, next_state = self._make_states(is_terminal=False, reward=0.3)
        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward == 0.3

    def test_outcome_reward(self) -> None:
        reward_fn = MolmoSpacesReward()
        state, action, next_state = self._make_states(is_terminal=True, reward=0.8)
        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.8

    def test_zero_reward_on_missing_info(self) -> None:
        reward_fn = MolmoSpacesReward()
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="",
            success=False,
        )
        from llenvs.core.state import Observation

        obs = Observation(prompt="test")
        state = State(
            observation=obs, hidden=hidden, metadata=StateMetadata(step=0, episode_id="t")
        )
        action = Action(text="test")
        next_state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="t", is_terminal=True, info={}),
        )
        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 0.0


class TestMolmoSpacesSuccessReward:
    def test_name(self) -> None:
        reward = MolmoSpacesSuccessReward()
        assert reward.name == "molmospaces_success"

    def test_reward_type(self) -> None:
        reward = MolmoSpacesSuccessReward()
        assert reward.reward_type == RewardType.OUTCOME

    def test_non_terminal_returns_none(self) -> None:
        reward_fn = MolmoSpacesSuccessReward()
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="",
            success=False,
        )
        from llenvs.core.state import Observation

        obs = Observation(prompt="test")
        state = State(
            observation=obs, hidden=hidden, metadata=StateMetadata(step=0, episode_id="t")
        )
        action = Action(text="test")
        next_state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="t", is_terminal=False),
        )
        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward is None
        assert signal.reward_type == RewardType.STEP

    def test_terminal_success(self) -> None:
        reward_fn = MolmoSpacesSuccessReward()
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="",
            success=True,
        )
        from llenvs.core.state import Observation

        obs = Observation(prompt="test")
        state = State(
            observation=obs, hidden=hidden, metadata=StateMetadata(step=0, episode_id="t")
        )
        action = Action(text="test")
        next_state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="t", is_terminal=True),
        )
        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 1.0
        assert signal.reward_type == RewardType.OUTCOME

    def test_terminal_failure(self) -> None:
        reward_fn = MolmoSpacesSuccessReward()
        hidden = MolmoSpacesHidden(
            task_index=0,
            task_name="pick",
            episode_step=0,
            physics_steps=0,
            last_action=None,
            ee_position=(0.0, 0.0, 0.0),
            ee_orientation=(1.0, 0.0, 0.0, 0.0),
            gripper_state="open",
            joint_positions=(),
            cumulative_reward=0.0,
            task_instruction="",
            success=False,
        )
        from llenvs.core.state import Observation

        obs = Observation(prompt="test")
        state = State(
            observation=obs, hidden=hidden, metadata=StateMetadata(step=0, episode_id="t")
        )
        action = Action(text="test")
        next_state = State(
            observation=obs,
            hidden=hidden,
            metadata=StateMetadata(step=1, episode_id="t", is_terminal=True),
        )
        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward == 0.0
        assert signal.reward_type == RewardType.OUTCOME


# ── TestMolmoSpacesObservationRendering ──────────────────────────


class TestMolmoSpacesObservationRendering:
    def test_render_pixels_to_image_rgb(self) -> None:
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = _render_pixels_to_image(arr, image_format="png")
        assert isinstance(img, ImageContent)
        assert img.media_type == "image/png"
        assert len(img.data) > 0

    def test_render_pixels_to_image_jpeg(self) -> None:
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = _render_pixels_to_image(arr, image_format="jpeg")
        assert isinstance(img, ImageContent)
        assert img.media_type == "image/jpeg"

    def test_render_depth_to_image(self) -> None:
        arr = np.random.rand(32, 32).astype(np.float32)
        img = _render_pixels_to_image(arr, image_format="png", is_depth=True)
        assert isinstance(img, ImageContent)
        assert img.media_type == "image/png"

    def test_multi_camera_observation(self) -> None:
        ms_env = MockMolmoSpacesEnv()
        robot = ms_env.robot
        env = MolmoSpacesEnvironment(
            molmospaces_env=ms_env,
            tasks=[MockTask()],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            cameras=("front", "wrist"),
        )
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.observation.images) == 2

    def test_single_camera_observation(self) -> None:
        ms_env = MockMolmoSpacesEnv()
        robot = ms_env.robot
        env = MolmoSpacesEnvironment(
            molmospaces_env=ms_env,
            tasks=[MockTask()],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            cameras=("front",),
        )
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.observation.images) == 1

    def test_depth_images_included(self) -> None:
        ms_env = MockMolmoSpacesEnv()
        robot = ms_env.robot
        env = MolmoSpacesEnvironment(
            molmospaces_env=ms_env,
            tasks=[MockTask()],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            cameras=("front",),
            include_depth=True,
        )
        state, _ = env.reset(options={"task_index": 0})
        # 1 RGB + 1 depth = 2 images
        assert len(state.observation.images) == 2


# ── TestMolmoSpacesEnvironment ───────────────────────────────────


class TestMolmoSpacesEnvironment:
    def _make_env(
        self,
        max_steps: int | None = 50,
        is_mobile: bool = False,
        controller_type: str = "ik",
        include_proprioception: bool = True,
        include_depth: bool = False,
        tasks: list[MockTask] | None = None,
    ) -> MolmoSpacesEnvironment:
        if tasks is None:
            tasks = [MockTask(name="pick", instruction="Pick up the cube.")]
        robot = MockRobot(is_mobile=is_mobile)
        ms_env = MockMolmoSpacesEnv(robot=robot)
        return MolmoSpacesEnvironment(
            molmospaces_env=ms_env,
            tasks=tasks,
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            max_steps=max_steps,
            is_mobile=is_mobile,
            controller_type=controller_type,
            include_proprioception=include_proprioception,
            include_depth=include_depth,
        )

    def test_spec_name(self) -> None:
        env = self._make_env()
        assert "molmospaces" in env.spec.name

    def test_spec_adapter(self) -> None:
        env = self._make_env()
        assert env.spec.adapter == "molmospaces"

    def test_spec_multi_turn(self) -> None:
        env = self._make_env()
        assert env.spec.is_multi_turn is True

    def test_spec_not_pure_step(self) -> None:
        env = self._make_env()
        assert env.spec.pure_step is False

    def test_len(self) -> None:
        tasks = [MockTask(name=f"task_{i}") for i in range(5)]
        env = self._make_env(tasks=tasks)
        assert len(env) == 5

    def test_reset_requires_task_index(self) -> None:
        env = self._make_env()
        with pytest.raises(ValueError, match="task_index"):
            env.reset()

    def test_reset_bounds_check(self) -> None:
        env = self._make_env()
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 10})

    def test_reset_returns_state(self) -> None:
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})
        assert isinstance(state, State)
        assert state.hidden.task_index == 0
        assert state.hidden.task_name == "pick"
        assert state.hidden.episode_step == 0

    def test_reset_includes_task_instruction(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert "Pick up the cube" in state.observation.prompt

    def test_reset_has_tools(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.observation.available_tools) > 0

    def test_reset_has_images(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert len(state.observation.images) >= 1

    def test_reset_info(self) -> None:
        env = self._make_env()
        _, info = env.reset(options={"task_index": 0})
        assert "task_index" in info
        assert info["task_index"] == 0

    def test_step_with_tool_call(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="grasp", arguments={})
        action = Action(tool_calls=(call,))
        result = env.step(state, action)
        assert result.next_state.hidden.episode_step == 1

    def test_step_increments_physics_steps(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3})
        action = Action(tool_calls=(call,))
        result = env.step(state, action)
        assert result.next_state.hidden.physics_steps > 0

    def test_step_tool_results_in_info(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="grasp", arguments={})
        action = Action(tool_calls=(call,))
        result = env.step(state, action)
        assert "tool_results" in result.info

    def test_step_state_continuity(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="grasp", arguments={})
        action = Action(tool_calls=(call,))
        env.step(state, action)
        # Trying to step with old state should fail
        call2 = ToolCall(id="c2", name="release", arguments={})
        action2 = Action(tool_calls=(call2,))
        with pytest.raises(NotImplementedError):
            env.step(state, action2)

    def test_truncation_at_max_steps(self) -> None:
        env = self._make_env(max_steps=2)
        state, _ = env.reset(options={"task_index": 0})
        # Step 1
        call = ToolCall(id="c1", name="grasp", arguments={})
        result = env.step(state, Action(tool_calls=(call,)))
        assert not result.done
        # Step 2 — should truncate
        call2 = ToolCall(id="c2", name="release", arguments={})
        result2 = env.step(result.next_state, Action(tool_calls=(call2,)))
        assert result2.truncated is True

    def test_termination_on_task_done(self) -> None:
        task = MockTask(name="pick")
        env = self._make_env(tasks=[task])
        state, _ = env.reset(options={"task_index": 0})
        # Mark task as done
        task._done = True
        task._success = True
        task._reward = 1.0
        call = ToolCall(id="c1", name="grasp", arguments={})
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.terminated is True

    def test_proprioception_in_prompt(self) -> None:
        env = self._make_env(include_proprioception=True)
        state, _ = env.reset(options={"task_index": 0})
        # Proprioception info should be in the prompt
        assert (
            "ee_position" in state.observation.prompt.lower()
            or "end-effector" in state.observation.prompt.lower()
            or "position" in state.observation.prompt.lower()
        )

    def test_no_proprioception(self) -> None:
        env = self._make_env(include_proprioception=False)
        state, _ = env.reset(options={"task_index": 0})
        # Should not contain detailed proprioceptive info
        assert "joint_positions" not in state.observation.prompt.lower()

    def test_reward_functions_include_native(self) -> None:
        env = self._make_env()
        names = [r.name for r in env.reward_functions]
        assert "molmospaces" in names

    def test_reward_computation(self) -> None:
        task = MockTask(name="pick")
        task._reward = 0.5
        env = self._make_env(tasks=[task])
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="grasp", arguments={})
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.rewards is not None
        molmo_signal = result.rewards.by_name("molmospaces")
        assert molmo_signal is not None


# ── TestMolmoSpacesPresets ───────────────────────────────────────


class TestMolmoSpacesPresets:
    def test_all_tasks_present(self) -> None:
        expected = {
            "navigate-to",
            "pick",
            "pick-and-place",
            "pick-and-place-next-to",
            "pick-and-place-color",
            "open",
            "close",
            "open-door",
        }
        assert expected == set(MOLMOSPACES_PRESETS.keys())

    def test_presets_have_max_steps(self) -> None:
        for name, preset in MOLMOSPACES_PRESETS.items():
            assert "max_steps" in preset, f"Preset {name} missing max_steps"

    def test_presets_have_description(self) -> None:
        for name, preset in MOLMOSPACES_PRESETS.items():
            assert "description" in preset, f"Preset {name} missing description"

    def test_tasks_list(self) -> None:
        assert isinstance(MOLMOSPACES_TASKS, tuple)
        assert len(MOLMOSPACES_TASKS) == 8

    def test_preset_values_are_reasonable(self) -> None:
        for name, preset in MOLMOSPACES_PRESETS.items():
            assert preset["max_steps"] > 0
            assert isinstance(preset["description"], str)
            assert len(preset["description"]) > 10


# ── TestMolmoSpacesAdapter ───────────────────────────────────────


class TestMolmoSpacesAdapter:
    def test_adapter_name(self) -> None:
        adapter = MolmoSpacesAdapter()
        assert adapter.name == "molmospaces"

    def test_list_environments(self) -> None:
        adapter = MolmoSpacesAdapter()
        envs = adapter.list_environments()
        assert isinstance(envs, list)
        assert len(envs) > 0
        # Should include task names prefixed with adapter name
        assert any("molmospaces:" in e for e in envs)

    def test_list_includes_all_presets(self) -> None:
        adapter = MolmoSpacesAdapter()
        envs = adapter.list_environments()
        for task_name in MOLMOSPACES_PRESETS:
            assert f"molmospaces:{task_name}" in envs

    def test_lazy_import_error(self) -> None:
        adapter = MolmoSpacesAdapter()
        with pytest.raises(ImportError):
            adapter._get_molmospaces()

    def test_get_native_answer_extractor(self) -> None:
        adapter = MolmoSpacesAdapter()
        assert adapter.get_native_answer_extractor("anything") is None

    def test_get_prompt_template(self) -> None:
        adapter = MolmoSpacesAdapter()
        assert adapter.get_prompt_template("anything") is None

    def test_get_environment_info(self) -> None:
        adapter = MolmoSpacesAdapter()
        info = adapter.get_environment_info("molmospaces:pick")
        assert info["adapter"] == "molmospaces"
        assert "pick" in info["name"]

    def test_parse_task_name(self) -> None:
        adapter = MolmoSpacesAdapter()
        assert adapter._parse_task_name("molmospaces:pick") == "pick"
        assert adapter._parse_task_name("molmospaces:open-door") == "open-door"

    def test_get_environment_with_mock(self) -> None:
        """Test that get_environment works when MolmoSpaces objects are provided."""
        adapter = MolmoSpacesAdapter()
        robot = MockRobot()
        ms_env = MockMolmoSpacesEnv(robot=robot)
        env = adapter.get_environment(
            "molmospaces:pick",
            molmospaces_env=ms_env,
            tasks=[MockTask()],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
        )
        assert isinstance(env, MolmoSpacesEnvironment)

    def test_get_environment_with_robot_types(self) -> None:
        adapter = MolmoSpacesAdapter()
        robot = MockRobot(is_mobile=True)
        ms_env = MockMolmoSpacesEnv(robot=robot)
        env = adapter.get_environment(
            "molmospaces:navigate-to",
            molmospaces_env=ms_env,
            tasks=[MockTask(name="navigate-to")],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            is_mobile=True,
        )
        tool_names = {t.name for t in env.available_tools}
        assert "navigate_to" in tool_names


# ── TestMolmoSpacesIntegration ───────────────────────────────────


class TestMolmoSpacesIntegration:
    def _make_env(self) -> MolmoSpacesEnvironment:
        robot = MockRobot()
        ms_env = MockMolmoSpacesEnv(robot=robot)
        return MolmoSpacesEnvironment(
            molmospaces_env=ms_env,
            tasks=[MockTask(name="pick", instruction="Pick up the red cube.")],
            robot=robot,
            ik_solver=ms_env.ik_solver,
            controller=ms_env.controller,
            max_steps=10,
        )

    def test_full_episode_flow(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        # Move to object
        call1 = ToolCall(
            id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3}
        )
        result1 = env.step(state, Action(tool_calls=(call1,)))
        assert not result1.done

        # Grasp
        call2 = ToolCall(id="c2", name="grasp", arguments={})
        result2 = env.step(result1.next_state, Action(tool_calls=(call2,)))
        assert not result2.done

        # Move to place
        call3 = ToolCall(
            id="c3", name="move_end_effector", arguments={"x": 0.0, "y": 0.5, "z": 0.3}
        )
        result3 = env.step(result2.next_state, Action(tool_calls=(call3,)))
        assert result3.next_state.hidden.episode_step == 3

    def test_multiple_tool_calls_in_single_action(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        calls = (
            ToolCall(id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3}),
            ToolCall(id="c2", name="grasp", arguments={}),
        )
        result = env.step(state, Action(tool_calls=calls))
        assert "tool_results" in result.info
        assert len(result.info["tool_results"]) == 2

    def test_invalid_tool_returns_error(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="nonexistent_tool", arguments={})
        result = env.step(state, Action(tool_calls=(call,)))
        assert result.info["tool_results"][0].is_error

    def test_mixed_valid_invalid_tools(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        calls = (
            ToolCall(id="c1", name="grasp", arguments={}),
            ToolCall(id="c2", name="nonexistent", arguments={}),
        )
        result = env.step(state, Action(tool_calls=calls))
        results = result.info["tool_results"]
        assert results[0].is_success
        assert results[1].is_error

    def test_episode_accumulates_physics_steps(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        call1 = ToolCall(
            id="c1", name="move_end_effector", arguments={"x": 0.5, "y": 0.1, "z": 0.3}
        )
        result1 = env.step(state, Action(tool_calls=(call1,)))
        steps1 = result1.next_state.hidden.physics_steps

        call2 = ToolCall(
            id="c2", name="move_end_effector", arguments={"x": 0.0, "y": 0.5, "z": 0.3}
        )
        result2 = env.step(result1.next_state, Action(tool_calls=(call2,)))
        steps2 = result2.next_state.hidden.physics_steps

        assert steps2 >= steps1

    def test_images_updated_after_step(self) -> None:
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        call = ToolCall(id="c1", name="grasp", arguments={})
        result = env.step(state, Action(tool_calls=(call,)))
        # Images should be present in the next observation
        assert len(result.next_state.observation.images) >= 1
