"""Tests for TrajectoryRunner run-from-state methods.

Tests run_from_state, run_batch_from_states, run_from_state_action,
run_batch_from_state_actions, and the public build_messages method.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.core.trajectory import Trajectory
from llenvs.evaluation.runner import TrajectoryRunner
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
)


# ---------------------------------------------------------------------------
# Fake environment: deterministic single-turn number-guessing game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeHidden:
    target: int
    task_index: int


class _FakeEnvironment:
    """Deterministic number-guessing environment.

    Agent receives "Guess the number (1-10)." and must respond with the
    correct number.  Reward is 1.0 if correct, 0.0 otherwise.  Episode
    always terminates after one step.
    """

    def __init__(self, tasks: list[int] | None = None) -> None:
        self._tasks = tasks or [3, 7, 1, 5, 9]

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="fake_number_guess",
            max_steps=1,
            observation_type=Observation,
            action_type=Action,
        )

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def reward_functions(self) -> tuple:
        return ()

    @property
    def available_tools(self) -> tuple:
        return ()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[_FakeHidden], dict[str, Any]]:
        options = options or {}
        task_index = options.get("task_index", 0)
        target = self._tasks[task_index]
        episode_id = options.get("episode_id", str(uuid.uuid4()))

        observation = Observation(
            prompt="Guess the number (1-10). The answer is a single number.",
            task=ObservationContent(text="Guess the number (1-10). The answer is a single number."),
        )
        hidden = _FakeHidden(target=target, task_index=task_index)
        metadata = StateMetadata(step=0, episode_id=episode_id)
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        return state, {"task_index": task_index, "target": target}

    def step(
        self,
        state: State[_FakeHidden],
        action: Action,
    ) -> StepResult[_FakeHidden]:
        match = re.search(r"\d+", action.text or "")
        guess = int(match.group()) if match else None
        resolved = str(guess) if guess is not None else None
        correct = guess == state.hidden.target
        reward_value = 1.0 if correct else 0.0

        rewards = SignalBundle(
            signals=(
                Signal(name="correctness", reward_type=RewardType.OUTCOME, reward=reward_value),
            )
        )

        feedback = (
            f"You guessed {guess}. "
            f"{'Correct!' if correct else f'Wrong! The answer was {state.hidden.target}.'}"
        )
        next_obs = Observation(
            prompt=state.observation.prompt,
            messages=(
                {"role": "assistant", "content": resolved or action.text or ""},
                {"role": "user", "content": feedback},
            ),
            task=state.observation.task,
            state=ObservationContent(text=feedback),
        )
        next_state = State(
            observation=next_obs,
            hidden=state.hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=True,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=True,
            extracted_action=resolved,
            resolved_action=resolved,
        )

    def compute_rewards(
        self,
        state: State[_FakeHidden],
        action: Action,
        next_state: State[_FakeHidden],
    ) -> SignalBundle:
        match = re.search(r"\d+", action.text or "")
        guess = int(match.group()) if match else None
        correct = guess == state.hidden.target
        return SignalBundle(
            signals=(
                Signal(
                    name="correctness",
                    reward_type=RewardType.OUTCOME,
                    reward=1.0 if correct else 0.0,
                ),
            )
        )


# ---------------------------------------------------------------------------
# Fake backend: deterministic text generator
# ---------------------------------------------------------------------------


class _FakeBackend(ModelBackend):
    """Deterministic backend that returns a fixed response."""

    def __init__(self, response: str = "5") -> None:
        self._response = response

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True)

    @property
    def model_name(self) -> str:
        return "fake-backend"

    def generate(self, prompts: list[str], params: SamplingParams) -> list[GenerationResult]:
        return [
            GenerationResult(text=self._response, finish_reason=StopReason.END_OF_TEXT)
            for _ in prompts
        ]

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        return GenerationResult(text=self._response, finish_reason=StopReason.END_OF_TEXT)

    def generate_chat_batch(
        self, messages_batch: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        return [self.generate_chat(msgs, params) for msgs in messages_batch]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    env: _FakeEnvironment | None = None,
    backend: _FakeBackend | None = None,
    system_prompt: str | None = None,
) -> TrajectoryRunner:
    return TrajectoryRunner(
        environment=env or _FakeEnvironment(),
        backend=backend or _FakeBackend(),
        sampling_params=SamplingParams(),
        system_prompt=system_prompt,
    )


# ===========================================================================
# build_messages (public API)
# ===========================================================================


class TestBuildMessages:
    """Tests for the public build_messages method."""

    def test_returns_messages_for_initial_state(self):
        """build_messages produces valid chat messages from the initial state."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})
        trajectory = Trajectory.create(state)

        messages = runner.build_messages(state, trajectory=trajectory)
        assert len(messages) >= 1
        assert messages[0].role == "user"

    def test_includes_system_prompt_when_set(self):
        """System prompt appears first when configured."""
        runner = _make_runner(system_prompt="You are a helpful assistant.")
        state, _ = runner.environment.reset(options={"task_index": 0})
        trajectory = Trajectory.create(state)

        messages = runner.build_messages(state, trajectory=trajectory)
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful assistant."

    def test_legacy_mode_without_trajectory(self):
        """Without a trajectory, uses legacy message building."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})

        messages = runner.build_messages(state)
        assert len(messages) >= 1
        assert messages[0].role == "user"


# ===========================================================================
# run_from_state
# ===========================================================================


class TestRunFromState:
    """Tests for TrajectoryRunner.run_from_state."""

    def test_completes_single_step_env(self):
        """Single-step environment produces a trajectory with one transition."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})

        traj = runner.run_from_state(state)
        assert len(traj) == 1
        assert traj.transitions[-1].next_state.metadata.is_terminal

    def test_correct_guess_gives_reward(self):
        """Backend that guesses correctly produces reward 1.0."""
        env = _FakeEnvironment(tasks=[5])
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        state, _ = env.reset(options={"task_index": 0})

        traj = runner.run_from_state(state)
        assert traj.total_reward == 1.0

    def test_wrong_guess_gives_zero_reward(self):
        """Backend that guesses wrong produces reward 0.0."""
        env = _FakeEnvironment(tasks=[3])
        runner = _make_runner(env=env, backend=_FakeBackend("9"))
        state, _ = env.reset(options={"task_index": 0})

        traj = runner.run_from_state(state)
        assert traj.total_reward == 0.0

    def test_does_not_step_terminal_state(self):
        """Starting from an already terminal state returns empty trajectory."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})
        # Step once to get a terminal state
        result = runner.environment.step(state, Action(text="5"))
        terminal = result.next_state

        traj = runner.run_from_state(terminal)
        assert len(traj) == 0

    def test_respects_max_steps(self):
        """max_steps limits the number of new steps in the rollout."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})

        traj = runner.run_from_state(state, max_steps=1)
        assert len(traj) <= 1

    def test_preserves_state_metadata(self):
        """Trajectory preserves the initial state's metadata (step, episode_id)."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})

        traj = runner.run_from_state(state)
        assert traj.initial_state.metadata.step == state.metadata.step
        assert traj.initial_state.metadata.episode_id == state.metadata.episode_id


# ===========================================================================
# run_from_state_action
# ===========================================================================


class TestRunFromStateAction:
    """Tests for TrajectoryRunner.run_from_state_action."""

    def test_forces_first_action(self):
        """The forced action is used as the first step, not the backend."""
        env = _FakeEnvironment(tasks=[3])
        # Backend would say "9" (wrong), but we force "3" (correct)
        runner = _make_runner(env=env, backend=_FakeBackend("9"))
        state, _ = env.reset(options={"task_index": 0})

        traj = runner.run_from_state_action(state, Action(text="3"))
        assert len(traj) == 1
        assert traj.transitions[0].action.text == "3"
        assert traj.total_reward == 1.0

    def test_forced_wrong_action(self):
        """Forcing a wrong action produces reward 0.0."""
        env = _FakeEnvironment(tasks=[3])
        runner = _make_runner(env=env, backend=_FakeBackend("3"))
        state, _ = env.reset(options={"task_index": 0})

        traj = runner.run_from_state_action(state, Action(text="9"))
        assert traj.total_reward == 0.0

    def test_does_not_step_terminal_state(self):
        """Starting from terminal state returns empty trajectory (ignores forced action)."""
        runner = _make_runner()
        state, _ = runner.environment.reset(options={"task_index": 0})
        result = runner.environment.step(state, Action(text="5"))
        terminal = result.next_state

        traj = runner.run_from_state_action(terminal, Action(text="anything"))
        assert len(traj) == 0


# ===========================================================================
# run_batch_from_states
# ===========================================================================


class TestRunBatchFromStates:
    """Tests for TrajectoryRunner.run_batch_from_states."""

    def test_batch_produces_correct_number_of_trajectories(self):
        """Output has the same number of trajectories as input states."""
        env = _FakeEnvironment()
        runner = _make_runner(env=env)
        states = [env.reset(options={"task_index": i})[0] for i in range(3)]

        trajs = runner.run_batch_from_states(states)
        assert len(trajs) == 3

    def test_correct_backend_all_succeed(self):
        """Backend guessing correctly gives reward 1.0 on matching tasks."""
        env = _FakeEnvironment(tasks=[5, 5, 5])
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        states = [env.reset(options={"task_index": i})[0] for i in range(3)]

        trajs = runner.run_batch_from_states(states)
        assert all(t.total_reward == 1.0 for t in trajs)

    def test_wrong_backend_all_fail(self):
        """Backend guessing wrong gives reward 0.0."""
        env = _FakeEnvironment(tasks=[3, 7, 1])
        runner = _make_runner(env=env, backend=_FakeBackend("99"))
        states = [env.reset(options={"task_index": i})[0] for i in range(3)]

        trajs = runner.run_batch_from_states(states)
        assert all(t.total_reward == 0.0 for t in trajs)

    def test_terminal_states_produce_empty_trajectories(self):
        """Rollouts from terminal states have zero transitions."""
        env = _FakeEnvironment(tasks=[5])
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        state, _ = env.reset(options={"task_index": 0})
        terminal = env.step(state, Action(text="5")).next_state

        trajs = runner.run_batch_from_states([terminal, terminal])
        assert all(len(t) == 0 for t in trajs)

    def test_empty_input_returns_empty(self):
        """Empty input returns empty output."""
        runner = _make_runner()
        trajs = runner.run_batch_from_states([])
        assert trajs == []

    def test_batch_size_chunking(self):
        """Chunking with batch_size < len(states) produces the same results."""
        env = _FakeEnvironment(tasks=[5, 5, 5, 5])
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        states = [env.reset(options={"task_index": i})[0] for i in range(4)]

        trajs = runner.run_batch_from_states(states, batch_size=2)
        assert len(trajs) == 4
        assert all(t.total_reward == 1.0 for t in trajs)

    def test_preserves_order(self):
        """Output trajectories are in the same order as input states."""
        env = _FakeEnvironment(tasks=[5, 3])  # first succeeds, second fails
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        states = [env.reset(options={"task_index": i})[0] for i in range(2)]

        trajs = runner.run_batch_from_states(states)
        assert trajs[0].total_reward == 1.0  # task 0: target=5, guess=5
        assert trajs[1].total_reward == 0.0  # task 1: target=3, guess=5

    def test_progress_callback_reports_completion(self):
        """Progress callback reports non-decreasing completion counts."""
        env = _FakeEnvironment(tasks=[5, 5, 5])
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        states = [env.reset(options={"task_index": i})[0] for i in range(3)]

        reports: list[tuple[int, int]] = []
        trajs = runner.run_batch_from_states(
            states,
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        assert len(trajs) == 3
        assert reports
        assert reports[-1] == (3, 3)
        assert [c for c, _ in reports] == sorted(c for c, _ in reports)

    def test_progress_callback_batch_size_uses_global_offsets(self):
        """Chunked run_batch_from_states reports global offsets."""
        env = _FakeEnvironment(tasks=[5, 5, 5, 5])
        runner = _make_runner(env=env, backend=_FakeBackend("5"))
        states = [env.reset(options={"task_index": i})[0] for i in range(4)]

        reports: list[tuple[int, int]] = []
        trajs = runner.run_batch_from_states(
            states,
            batch_size=2,
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        assert len(trajs) == 4
        assert reports[-1] == (4, 4)
        assert all(total == 4 for _, total in reports)


# ===========================================================================
# run_batch_from_state_actions
# ===========================================================================


class TestRunBatchFromStateActions:
    """Tests for TrajectoryRunner.run_batch_from_state_actions."""

    def test_forced_correct_actions(self):
        """Forcing correct actions gives reward 1.0."""
        env = _FakeEnvironment(tasks=[3, 7])
        runner = _make_runner(env=env, backend=_FakeBackend("99"))
        states = [env.reset(options={"task_index": i})[0] for i in range(2)]
        actions = [Action(text="3"), Action(text="7")]

        trajs = runner.run_batch_from_state_actions(states, actions)
        assert all(t.total_reward == 1.0 for t in trajs)

    def test_forced_wrong_actions(self):
        """Forcing wrong actions gives reward 0.0."""
        env = _FakeEnvironment(tasks=[3, 7])
        runner = _make_runner(env=env, backend=_FakeBackend("3"))
        states = [env.reset(options={"task_index": i})[0] for i in range(2)]
        actions = [Action(text="99"), Action(text="99")]

        trajs = runner.run_batch_from_state_actions(states, actions)
        assert all(t.total_reward == 0.0 for t in trajs)

    def test_mismatched_lengths_raises(self):
        """States and actions must have the same length."""
        env = _FakeEnvironment(tasks=[3])
        runner = _make_runner(env=env)
        states = [env.reset(options={"task_index": 0})[0]]
        actions = [Action(text="3"), Action(text="7")]  # too many

        with pytest.raises(ValueError):
            runner.run_batch_from_state_actions(states, actions)

    def test_batch_size_chunking(self):
        """Chunking preserves forced actions across chunks."""
        env = _FakeEnvironment(tasks=[3, 7, 1])
        runner = _make_runner(env=env, backend=_FakeBackend("99"))
        states = [env.reset(options={"task_index": i})[0] for i in range(3)]
        actions = [Action(text="3"), Action(text="7"), Action(text="1")]

        trajs = runner.run_batch_from_state_actions(states, actions, batch_size=2)
        assert len(trajs) == 3
        assert all(t.total_reward == 1.0 for t in trajs)

    def test_progress_callback_reports_completion(self):
        """Forced-action batch runner reports progress."""
        env = _FakeEnvironment(tasks=[3, 7])
        runner = _make_runner(env=env, backend=_FakeBackend("99"))
        states = [env.reset(options={"task_index": i})[0] for i in range(2)]
        actions = [Action(text="3"), Action(text="7")]

        reports: list[tuple[int, int]] = []
        trajs = runner.run_batch_from_state_actions(
            states,
            actions,
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        assert len(trajs) == 2
        assert reports
        assert reports[-1] == (2, 2)

    def test_progress_callback_batch_size_uses_global_offsets(self):
        """Chunked forced-action batch runner reports global offsets."""
        env = _FakeEnvironment(tasks=[3, 7, 1])
        runner = _make_runner(env=env, backend=_FakeBackend("99"))
        states = [env.reset(options={"task_index": i})[0] for i in range(3)]
        actions = [Action(text="3"), Action(text="7"), Action(text="1")]

        reports: list[tuple[int, int]] = []
        trajs = runner.run_batch_from_state_actions(
            states,
            actions,
            batch_size=2,
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        assert len(trajs) == 3
        assert reports[-1] == (3, 3)
        assert all(total == 3 for _, total in reports)
