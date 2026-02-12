"""Tests for branching strategies and BranchManager."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType
from llenvs.core.state import Action, Observation, State, StateMetadata


# ---------------------------------------------------------------------------
# Mock environments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockHidden:
    answer: str
    task_index: int = 0


class PureFunctionEnv:
    """Branchable (pure-function) environment for testing DirectStrategy."""

    def __init__(self, tasks: list[dict[str, str]] | None = None) -> None:
        self._tasks = tasks or [
            {"prompt": "What is 1+1?", "answer": "2"},
            {"prompt": "What is 2+2?", "answer": "4"},
            {"prompt": "What is 3+3?", "answer": "6"},
        ]

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="pure_mock",
            adapter="test",
            pure_step=True,
            supports_seed=True,
            supports_task_index=True,
        )

    @property
    def reward_functions(self) -> tuple:
        return ()

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    def __len__(self) -> int:
        return len(self._tasks)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[MockHidden], dict[str, Any]]:
        task_index = (options or {}).get("task_index", 0)
        task = self._tasks[task_index]
        state = State(
            observation=Observation(prompt=task["prompt"]),
            hidden=MockHidden(answer=task["answer"], task_index=task_index),
            metadata=StateMetadata(
                step=0, episode_id=f"ep-{task_index}-{seed or 0}"
            ),
        )
        return state, {"task_index": task_index}

    def step(
        self, state: State[MockHidden], action: Action
    ) -> StepResult[MockHidden]:
        correct = action.text == state.hidden.answer
        next_state = State(
            observation=Observation(
                prompt=f"You said: {action.text}. {'Correct!' if correct else 'Wrong.'}"
            ),
            hidden=state.hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=True,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=RewardBundle.single(
                1.0 if correct else 0.0, "correct", RewardType.OUTCOME
            ),
            terminated=True,
        )

    def compute_rewards(
        self, state: State[MockHidden], action: Action, next_state: State[MockHidden]
    ) -> RewardBundle:
        correct = action.text == state.hidden.answer
        return RewardBundle.single(
            1.0 if correct else 0.0, "correct", RewardType.OUTCOME
        )


class MutableEnv:
    """Non-branchable environment with mutable internal state for testing ActionReplay."""

    def __init__(self) -> None:
        self._counter = 0
        self._last_seed: int | None = None
        self._last_task_index: int = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="mutable_mock",
            adapter="test",
            pure_step=False,
            supports_seed=True,
            supports_task_index=True,
        )

    @property
    def reward_functions(self) -> tuple:
        return ()

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    def __len__(self) -> int:
        return 3

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[MockHidden], dict[str, Any]]:
        task_index = (options or {}).get("task_index", 0)
        self._counter = 0
        self._last_seed = seed
        self._last_task_index = task_index
        state = State(
            observation=Observation(prompt=f"Task {task_index}, counter=0"),
            hidden=MockHidden(answer="done", task_index=task_index),
            metadata=StateMetadata(
                step=0, episode_id=f"ep-{task_index}-{seed or 0}"
            ),
        )
        return state, {"task_index": task_index}

    def step(
        self, state: State[MockHidden], action: Action
    ) -> StepResult[MockHidden]:
        self._counter += 1
        next_state = State(
            observation=Observation(
                prompt=f"Step {self._counter}, action={action.text}"
            ),
            hidden=state.hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=RewardBundle(signals=()),
        )

    def compute_rewards(
        self, state: State[MockHidden], action: Action, next_state: State[MockHidden]
    ) -> RewardBundle:
        return RewardBundle(signals=())


class NoBranchNoSeedEnv:
    """Environment that supports neither branching nor seeds."""

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="no_branch_no_seed",
            adapter="test",
            pure_step=False,
            supports_seed=False,
            supports_task_index=False,
        )

    @property
    def reward_functions(self) -> tuple:
        return ()

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State, dict[str, Any]]:
        return (
            State(
                observation=Observation(prompt="test"),
                hidden=None,
                metadata=StateMetadata(step=0, episode_id="ep-0"),
            ),
            {},
        )

    def step(self, state: State, action: Action) -> StepResult:
        return StepResult(
            next_state=state.with_metadata(step=state.metadata.step + 1),
            rewards=RewardBundle(signals=()),
        )

    def compute_rewards(
        self, state: State, action: Action, next_state: State
    ) -> RewardBundle:
        return RewardBundle(signals=())


# ---------------------------------------------------------------------------
# DirectStrategy tests
# ---------------------------------------------------------------------------


class TestDirectStrategy:
    """Tests for DirectStrategy (pure-function environments)."""

    def test_can_branch_true_for_branching_env(self):
        from llenvs.core.branching import DirectStrategy

        strategy = DirectStrategy()
        env = PureFunctionEnv()
        assert strategy.can_branch(env) is True

    def test_can_branch_false_for_non_branching_env(self):
        from llenvs.core.branching import DirectStrategy

        strategy = DirectStrategy()
        env = MutableEnv()
        assert strategy.can_branch(env) is False

    def test_checkpoint_and_branch_returns_same_env(self):
        from llenvs.core.branching import DirectStrategy

        strategy = DirectStrategy()
        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})
        branch = strategy.create_branch(handle)

        assert branch.environment is env  # Same instance for pure-function
        assert branch.state is state

    def test_multi_branch_from_same_checkpoint(self):
        from llenvs.core.branching import DirectStrategy

        strategy = DirectStrategy()
        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})

        branches = [strategy.create_branch(handle) for _ in range(5)]
        # All should reference same env and state
        for b in branches:
            assert b.environment is env
            assert b.state is state

    def test_branches_step_independently(self):
        from llenvs.core.branching import DirectStrategy

        strategy = DirectStrategy()
        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})

        b1 = strategy.create_branch(handle)
        b2 = strategy.create_branch(handle)

        r1 = b1.environment.step(b1.state, Action.from_text("2"))
        r2 = b2.environment.step(b2.state, Action.from_text("wrong"))

        assert "Correct!" in r1.next_state.observation.prompt
        assert "Wrong." in r2.next_state.observation.prompt

    def test_release_checkpoint_is_noop(self):
        from llenvs.core.branching import DirectStrategy

        strategy = DirectStrategy()
        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})
        # Should not raise
        strategy.release_checkpoint(handle)

    def test_name(self):
        from llenvs.core.branching import DirectStrategy

        assert DirectStrategy().name == "direct"


# ---------------------------------------------------------------------------
# ActionReplayStrategy tests
# ---------------------------------------------------------------------------


class TestActionReplayStrategy:
    """Tests for ActionReplayStrategy (deterministic envs)."""

    def test_can_branch_true_for_seed_env(self):
        from llenvs.core.branching import ActionReplayStrategy

        strategy = ActionReplayStrategy(env_factory=MutableEnv)
        assert strategy.can_branch(MutableEnv()) is True

    def test_can_branch_false_for_no_seed_env(self):
        from llenvs.core.branching import ActionReplayStrategy

        strategy = ActionReplayStrategy(env_factory=NoBranchNoSeedEnv)
        assert strategy.can_branch(NoBranchNoSeedEnv()) is False

    def test_checkpoint_at_initial_state_and_branch(self):
        from llenvs.core.branching import ActionReplayStrategy

        strategy = ActionReplayStrategy(env_factory=MutableEnv)
        env = MutableEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(
            env, state, actions=(), reset_options={"seed": 42, "task_index": 0}
        )
        branch = strategy.create_branch(handle)

        # Branch env should be a fresh instance
        assert branch.environment is not env
        # But state should be equivalent (same observation after replay of 0 actions)
        assert branch.state.observation.prompt == state.observation.prompt
        assert branch.state.metadata.step == 0

    def test_checkpoint_at_step_n_replays_actions(self):
        from llenvs.core.branching import ActionReplayStrategy

        strategy = ActionReplayStrategy(env_factory=MutableEnv)
        env = MutableEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        # Take 3 steps
        actions = []
        for i in range(3):
            action = Action.from_text(f"action-{i}")
            actions.append(action)
            result = env.step(state, action)
            state = result.next_state

        # Checkpoint at step 3
        handle = strategy.create_checkpoint(
            env,
            state,
            actions=tuple(actions),
            reset_options={"seed": 42, "task_index": 0},
        )
        branch = strategy.create_branch(handle)

        # Branched state should match step 3
        assert branch.state.metadata.step == 3
        assert branch.state.observation.prompt == state.observation.prompt

    def test_multi_branch_each_gets_independent_env(self):
        from llenvs.core.branching import ActionReplayStrategy

        strategy = ActionReplayStrategy(env_factory=MutableEnv)
        env = MutableEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        action = Action.from_text("step-1")
        result = env.step(state, action)
        state = result.next_state

        handle = strategy.create_checkpoint(
            env,
            state,
            actions=(action,),
            reset_options={"seed": 42, "task_index": 0},
        )

        branches = [strategy.create_branch(handle) for _ in range(3)]
        envs = [b.environment for b in branches]

        # All distinct instances
        assert len(set(id(e) for e in envs)) == 3

        # Each can step independently
        results = []
        for b in branches:
            r = b.environment.step(b.state, Action.from_text("diverge"))
            results.append(r)

        # All should produce the same step number
        for r in results:
            assert r.next_state.metadata.step == 2

    def test_name(self):
        from llenvs.core.branching import ActionReplayStrategy

        assert ActionReplayStrategy(env_factory=MutableEnv).name == "action_replay"


# ---------------------------------------------------------------------------
# resolve_strategy tests
# ---------------------------------------------------------------------------


class TestResolveStrategy:
    """Tests for resolve_strategy auto-resolution."""

    def test_resolves_direct_for_branching_env(self):
        from llenvs.core.branching import DirectStrategy, resolve_strategy

        env = PureFunctionEnv()
        strategy = resolve_strategy(env)
        assert isinstance(strategy, DirectStrategy)

    def test_resolves_process_fork_for_non_branching_env_on_unix(self):
        from llenvs.core.branching import resolve_strategy

        if sys.platform == "win32":
            pytest.skip("ProcessForkStrategy not available on Windows")

        env = MutableEnv()
        strategy = resolve_strategy(env)
        from llenvs.core.branching import ProcessForkStrategy
        assert isinstance(strategy, ProcessForkStrategy)

    def test_resolves_action_replay_when_explicitly_requested(self):
        from llenvs.core.branching import ActionReplayStrategy, resolve_strategy

        env = MutableEnv()
        strategy = resolve_strategy(env, preference="action_replay", env_factory=MutableEnv)
        assert isinstance(strategy, ActionReplayStrategy)

    def test_preference_override(self):
        from llenvs.core.branching import ActionReplayStrategy, resolve_strategy

        env = PureFunctionEnv()
        # Even though direct is best, user requests action_replay
        strategy = resolve_strategy(
            env, preference="action_replay", env_factory=PureFunctionEnv
        )
        assert isinstance(strategy, ActionReplayStrategy)

    def test_preference_direct(self):
        from llenvs.core.branching import DirectStrategy, resolve_strategy

        env = PureFunctionEnv()
        strategy = resolve_strategy(env, preference="direct")
        assert isinstance(strategy, DirectStrategy)

    def test_preference_invalid_raises(self):
        from llenvs.core.branching import resolve_strategy

        env = PureFunctionEnv()
        with pytest.raises(ValueError, match="Unknown branching strategy"):
            resolve_strategy(env, preference="nonexistent")

    def test_preference_action_replay_without_factory_raises(self):
        from llenvs.core.branching import resolve_strategy

        env = MutableEnv()
        with pytest.raises(ValueError, match="env_factory"):
            resolve_strategy(env, preference="action_replay")


# ---------------------------------------------------------------------------
# BranchManager tests
# ---------------------------------------------------------------------------


class TestBranchManager:
    """Tests for BranchManager user-facing API."""

    def test_create_with_auto_resolution(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        mgr = BranchManager.create(env)
        assert mgr is not None

    def test_checkpoint_and_branch(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        with BranchManager.create(env) as mgr:
            mgr.checkpoint("step0", state, actions=(), reset_options={})

            branch_env, branch_state = mgr.branch("step0")
            assert branch_state is state

            result = branch_env.step(branch_state, Action.from_text("2"))
            assert result.terminated

    def test_branch_unknown_checkpoint_raises(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        with BranchManager.create(env) as mgr:
            with pytest.raises(KeyError, match="no_such_checkpoint"):
                mgr.branch("no_such_checkpoint")

    def test_release_checkpoint(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        with BranchManager.create(env) as mgr:
            mgr.checkpoint("s0", state, actions=(), reset_options={})
            mgr.release("s0")

            with pytest.raises(KeyError, match="s0"):
                mgr.branch("s0")

    def test_release_unknown_checkpoint_raises(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        with BranchManager.create(env) as mgr:
            with pytest.raises(KeyError, match="nope"):
                mgr.release("nope")

    def test_context_manager_cleanup(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        mgr = BranchManager.create(env)
        mgr.checkpoint("s0", state, actions=(), reset_options={})

        mgr.__enter__()
        mgr.__exit__(None, None, None)

        # After exit, all checkpoints should be released
        with pytest.raises(KeyError):
            mgr.branch("s0")

    def test_close_releases_all_checkpoints(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        mgr = BranchManager.create(env)
        mgr.checkpoint("a", state, actions=(), reset_options={})
        mgr.checkpoint("b", state, actions=(), reset_options={})
        mgr.close()

        with pytest.raises(KeyError):
            mgr.branch("a")
        with pytest.raises(KeyError):
            mgr.branch("b")

    def test_multiple_checkpoints(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state0, _ = env.reset(seed=42, options={"task_index": 0})

        with BranchManager.create(env) as mgr:
            mgr.checkpoint("step0", state0, actions=(), reset_options={})

            result = env.step(state0, Action.from_text("2"))
            state1 = result.next_state
            mgr.checkpoint(
                "step1", state1,
                actions=(Action.from_text("2"),),
                reset_options={},
            )

            _, s0 = mgr.branch("step0")
            assert s0.metadata.step == 0

            _, s1 = mgr.branch("step1")
            assert s1.metadata.step == 1

    def test_branch_returns_tuple(self):
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        with BranchManager.create(env) as mgr:
            mgr.checkpoint("s", state, actions=(), reset_options={})
            result = mgr.branch("s")
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_recursive_branching(self):
        """Branch from a branch — nested BranchManagers."""
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        with BranchManager.create(env) as mgr:
            mgr.checkpoint("root", state, actions=(), reset_options={})

            b_env, b_state = mgr.branch("root")
            result = b_env.step(b_state, Action.from_text("2"))

            # Create a nested BranchManager on the branched env
            with BranchManager.create(b_env) as sub_mgr:
                sub_mgr.checkpoint(
                    "child",
                    result.next_state,
                    actions=(Action.from_text("2"),),
                    reset_options={},
                )

                sub_env, sub_state = sub_mgr.branch("child")
                assert sub_state.metadata.step == 1

    def test_create_with_explicit_strategy(self):
        from llenvs.core.branching import BranchManager, DirectStrategy

        env = PureFunctionEnv()
        mgr = BranchManager.create(env, strategy="direct")
        assert isinstance(mgr._strategy, DirectStrategy)

    def test_create_with_action_replay_strategy(self):
        from llenvs.core.branching import BranchManager, ActionReplayStrategy

        env = MutableEnv()
        mgr = BranchManager.create(env, strategy="action_replay", env_factory=MutableEnv)
        assert isinstance(mgr._strategy, ActionReplayStrategy)

    def test_action_replay_through_branch_manager(self):
        """End-to-end test: mutable env + action replay + branching."""
        from llenvs.core.branching import BranchManager

        env = MutableEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        # Take 2 steps
        actions = []
        for i in range(2):
            action = Action.from_text(f"step-{i}")
            actions.append(action)
            result = env.step(state, action)
            state = result.next_state

        with BranchManager.create(env, env_factory=MutableEnv) as mgr:
            mgr.checkpoint(
                "after_2_steps",
                state,
                actions=tuple(actions),
                reset_options={"seed": 42, "task_index": 0},
            )

            # Branch and take a different action
            b_env, b_state = mgr.branch("after_2_steps")
            assert b_state.metadata.step == 2

            r = b_env.step(b_state, Action.from_text("diverge"))
            assert r.next_state.metadata.step == 3

    def test_overwrite_checkpoint(self):
        """Checkpointing the same name again should overwrite."""
        from llenvs.core.branching import BranchManager

        env = PureFunctionEnv()
        state0, _ = env.reset(seed=42, options={"task_index": 0})
        result = env.step(state0, Action.from_text("2"))
        state1 = result.next_state

        with BranchManager.create(env) as mgr:
            mgr.checkpoint("x", state0, actions=(), reset_options={})
            _, s = mgr.branch("x")
            assert s.metadata.step == 0

            # Overwrite with state at step 1
            mgr.checkpoint(
                "x", state1,
                actions=(Action.from_text("2"),),
                reset_options={},
            )
            _, s = mgr.branch("x")
            assert s.metadata.step == 1


# ---------------------------------------------------------------------------
# CheckpointHandle / BranchHandle tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for frozen dataclasses."""

    def test_checkpoint_handle_frozen(self):
        from llenvs.core.branching import CheckpointHandle

        state = State(
            observation=Observation(prompt="test"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="ep-0"),
        )
        handle = CheckpointHandle(checkpoint_id="cp-1", state=state)
        with pytest.raises(AttributeError):
            handle.checkpoint_id = "changed"  # type: ignore

    def test_branch_handle_frozen(self):
        from llenvs.core.branching import BranchHandle

        env = PureFunctionEnv()
        state = State(
            observation=Observation(prompt="test"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="ep-0"),
        )
        handle = BranchHandle(environment=env, state=state)
        assert handle.resource_id == ""
        with pytest.raises(AttributeError):
            handle.state = state  # type: ignore


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExports:
    """Verify that branching types are properly exported."""

    def test_core_exports(self):
        from llenvs.core import BranchManager, BranchHandle, BranchingStrategy

        assert BranchManager is not None
        assert BranchHandle is not None
        assert BranchingStrategy is not None

    def test_top_level_exports(self):
        from llenvs import BranchManager

        assert BranchManager is not None


# ---------------------------------------------------------------------------
# Phase 2: ProcessForkStrategy + Server /fork tests
# ---------------------------------------------------------------------------

# Shared mock with mutable state for fork testing


@dataclass(frozen=True)
class _ForkMockHidden:
    answer: str
    task_index: int = 0


class _MutableCounterEnv:
    """Environment with mutable state that tracks internal counter.

    This simulates a mutable backend (like WebShop/AgentGym) where the
    internal state cannot be captured purely from the State object.
    """

    def __init__(self) -> None:
        self._counter = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="mutable_counter",
            adapter="test",
            max_steps=10,
            pure_step=False,
            supports_seed=True,
            supports_task_index=True,
        )

    @property
    def reward_functions(self) -> tuple:
        return ()

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    def __len__(self) -> int:
        return 3

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[_ForkMockHidden], dict[str, Any]]:
        task_index = (options or {}).get("task_index", 0)
        self._counter = 0
        state = State(
            observation=Observation(prompt=f"task={task_index}, counter=0"),
            hidden=_ForkMockHidden(answer="done", task_index=task_index),
            metadata=StateMetadata(step=0, episode_id=f"ep-{task_index}-{seed or 0}"),
        )
        return state, {"task_index": task_index}

    def step(
        self, state: State[_ForkMockHidden], action: Action
    ) -> StepResult[_ForkMockHidden]:
        self._counter += 1
        next_state = State(
            observation=Observation(
                prompt=f"counter={self._counter}, action={action.text}"
            ),
            hidden=state.hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=RewardBundle(signals=()),
        )

    def compute_rewards(
        self, state: State[_ForkMockHidden], action: Action, next_state: State[_ForkMockHidden]
    ) -> RewardBundle:
        return RewardBundle(signals=())


def _start_env_server(env, host="127.0.0.1", port=0):
    """Start an EnvironmentServer in a thread, return (url, http_server)."""
    from http.server import HTTPServer
    from http.client import HTTPConnection
    from llenvs.container.server import EnvironmentHandler

    handler_class = type(
        "BoundHandler",
        (EnvironmentHandler,),
        {"environment": env, "hidden_type": None},
    )
    http_server = HTTPServer((host, port), handler_class)
    actual_port = http_server.server_address[1]

    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()

    # Wait for ready
    for _ in range(50):
        try:
            conn = HTTPConnection(host, actual_port, timeout=1)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                break
        except Exception:
            time.sleep(0.05)

    return f"http://{host}:{actual_port}", http_server


def _http_request(url, method, path, body=None):
    """Simple HTTP helper returning (status, parsed_json)."""
    from http.client import HTTPConnection
    from urllib.parse import urlparse

    parsed = urlparse(url)
    conn = HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    headers = {}
    body_bytes = None
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body_bytes))
    conn.request(method, path, body=body_bytes, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    status = resp.status
    conn.close()
    return status, data


# ---------------------------------------------------------------------------
# Server /fork endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
class TestServerForkEndpoint:
    """Tests for POST /fork on EnvironmentServer."""

    def _setup_server(self):
        """Create a mutable env, start a server, return (url, server, env)."""
        env = _MutableCounterEnv()
        url, srv = _start_env_server(env)
        return url, srv, env

    def test_fork_creates_healthy_child(self):
        url, srv, env = self._setup_server()
        try:
            # Reset to capture hidden_type
            _http_request(url, "POST", "/reset", {})

            # Fork
            status, data = _http_request(url, "POST", "/fork", {})
            assert status == 200
            assert "url" in data
            assert "pid" in data
            child_url = data["url"]
            child_pid = data["pid"]

            try:
                # Child should be healthy
                s, d = _http_request(child_url, "GET", "/health")
                assert s == 200
                assert d["status"] == "ok"
            finally:
                os.kill(child_pid, signal.SIGTERM)
                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError:
                    pass
        finally:
            srv.shutdown()

    def test_fork_child_has_same_state(self):
        """Forked child preserves environment state from before fork."""
        url, srv, env = self._setup_server()
        try:
            # Reset and step on parent
            _, reset_data = _http_request(url, "POST", "/reset", {})
            state = reset_data["state"]
            _, step_data = _http_request(
                url, "POST", "/step",
                {"state": state, "action": {"text": "action-0", "tool_calls": []}},
            )
            state_after_step = step_data["next_state"]

            # Fork after the step
            _, fork_data = _http_request(url, "POST", "/fork", {})
            child_url = fork_data["url"]
            child_pid = fork_data["pid"]

            try:
                # Step on the child — its internal counter should be at 1
                # (inherited from parent), so next step increments to 2
                _, child_step = _http_request(
                    child_url, "POST", "/step",
                    {"state": state_after_step, "action": {"text": "child-action", "tool_calls": []}},
                )
                # Child counter was at 1, now 2
                assert "counter=2" in child_step["next_state"]["observation"]["prompt"]
            finally:
                os.kill(child_pid, signal.SIGTERM)
                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError:
                    pass
        finally:
            srv.shutdown()

    def test_fork_branches_are_independent(self):
        """Two forks from same state produce independent environments."""
        url, srv, env = self._setup_server()
        pids = []
        try:
            # Reset
            _, reset_data = _http_request(url, "POST", "/reset", {})
            state = reset_data["state"]

            # Fork twice
            _, fork1 = _http_request(url, "POST", "/fork", {})
            _, fork2 = _http_request(url, "POST", "/fork", {})
            url1, pid1 = fork1["url"], fork1["pid"]
            url2, pid2 = fork2["url"], fork2["pid"]
            pids.extend([pid1, pid2])

            # Step on fork1
            _, r1 = _http_request(
                url1, "POST", "/step",
                {"state": state, "action": {"text": "A", "tool_calls": []}},
            )
            # Step on fork2
            _, r2 = _http_request(
                url2, "POST", "/step",
                {"state": state, "action": {"text": "B", "tool_calls": []}},
            )

            # They should have different actions in their observations
            assert "action=A" in r1["next_state"]["observation"]["prompt"]
            assert "action=B" in r2["next_state"]["observation"]["prompt"]
        finally:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
            srv.shutdown()


# ---------------------------------------------------------------------------
# ContainerEnvironment.fork() tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
class TestContainerEnvironmentFork:
    """Tests for ContainerEnvironment.fork() method."""

    def test_fork_returns_url_and_pid(self):
        from llenvs.container.client import ContainerEnvironment

        env = _MutableCounterEnv()
        url, srv = _start_env_server(env)
        client = ContainerEnvironment(url=url)
        pids = []
        try:
            # Reset first
            client.reset()

            fork_url, fork_pid = client.fork()
            pids.append(fork_pid)

            assert fork_url.startswith("http://")
            assert isinstance(fork_pid, int)
            assert fork_pid > 0
        finally:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
            client.close()
            srv.shutdown()


# ---------------------------------------------------------------------------
# ProcessForkStrategy tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
class TestProcessForkStrategy:
    """Tests for ProcessForkStrategy."""

    def test_name(self):
        from llenvs.core.branching import ProcessForkStrategy

        assert ProcessForkStrategy().name == "process_fork"

    def test_can_branch_any_env(self):
        """ProcessForkStrategy can branch any environment."""
        from llenvs.core.branching import ProcessForkStrategy

        strategy = ProcessForkStrategy()
        assert strategy.can_branch(PureFunctionEnv()) is True
        assert strategy.can_branch(MutableEnv()) is True
        assert strategy.can_branch(NoBranchNoSeedEnv()) is True

    def test_checkpoint_and_branch_lifecycle(self):
        from llenvs.core.branching import ProcessForkStrategy

        strategy = ProcessForkStrategy()
        env = _MutableCounterEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        # Step to build up internal state
        r = env.step(state, Action.from_text("step-0"))
        state = r.next_state

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})
        try:
            branch = strategy.create_branch(handle)

            # Branch env should be a ContainerEnvironment (proxy)
            from llenvs.container.client import ContainerEnvironment
            assert isinstance(branch.environment, ContainerEnvironment)

            # Step on the branch
            r = branch.environment.step(branch.state, Action.from_text("branch-action"))
            # Counter was at 1 from parent, now 2
            assert "counter=2" in r.next_state.observation.prompt
        finally:
            strategy.release_checkpoint(handle)

    def test_multi_branch_produces_independent_envs(self):
        from llenvs.core.branching import ProcessForkStrategy

        strategy = ProcessForkStrategy()
        env = _MutableCounterEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})
        try:
            b1 = strategy.create_branch(handle)
            b2 = strategy.create_branch(handle)

            # Step each differently
            r1 = b1.environment.step(b1.state, Action.from_text("A"))
            r2 = b2.environment.step(b2.state, Action.from_text("B"))

            assert "action=A" in r1.next_state.observation.prompt
            assert "action=B" in r2.next_state.observation.prompt
        finally:
            strategy.release_checkpoint(handle)

    def test_release_cleans_up_processes(self):
        from llenvs.core.branching import ProcessForkStrategy

        strategy = ProcessForkStrategy()
        env = _MutableCounterEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        handle = strategy.create_checkpoint(env, state, actions=(), reset_options={})
        branch = strategy.create_branch(handle)
        branch_pid = int(branch.resource_id) if branch.resource_id else None

        # Release should kill all forked processes
        strategy.release_checkpoint(handle)

        if branch_pid:
            # Give process time to exit
            time.sleep(0.2)
            try:
                os.kill(branch_pid, 0)  # Check if alive
                pytest.fail("Forked process should have been killed")
            except ProcessLookupError:
                pass  # Expected — process was cleaned up

    def test_through_branch_manager(self):
        """End-to-end: BranchManager with process_fork strategy."""
        from llenvs.core.branching import BranchManager

        env = _MutableCounterEnv()
        state, _ = env.reset(seed=42, options={"task_index": 0})

        with BranchManager.create(env, strategy="process_fork") as mgr:
            mgr.checkpoint("s0", state, actions=(), reset_options={})

            b_env, b_state = mgr.branch("s0")
            r = b_env.step(b_state, Action.from_text("hello"))
            assert r.next_state.metadata.step == 1
