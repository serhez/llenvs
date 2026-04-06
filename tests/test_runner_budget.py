"""Tests for TrajectoryRunner with PromptBudget integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from llenvs.core.reward import SignalBundle
from llenvs.core.state import (
    Action,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.evaluation.history import HistoryEntry, PromptBudget
from llenvs.evaluation.runner import TrajectoryRunner
from llenvs.inference.protocol import ChatMessage, SamplingParams


def _make_state(
    task: ObservationContent | None = None,
    state: ObservationContent | None = None,
    step: int = 0,
) -> State:
    return State(
        observation=Observation(
            prompt="",
            messages=(),
            task=task,
            state=state,
        ),
        hidden=None,
        metadata=StateMetadata(
            step=step,
            episode_id="test",
        ),
    )


def _simple_estimator(text: str) -> int:
    """1 token per character."""
    return len(text)


class TestRunnerWithPromptBudget:
    """Tests for TrajectoryRunner using PromptBudget instead of history_fn."""

    def _make_runner(self, system_prompt=None, prompt_budget=None, history_fn=None):
        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        kwargs = dict(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
        )
        if prompt_budget is not None:
            kwargs["prompt_budget"] = prompt_budget
        if history_fn is not None:
            kwargs["history_fn"] = history_fn

        return TrajectoryRunner(**kwargs)

    def test_build_history_receives_available_tokens(self) -> None:
        """Verify build_history callback receives correct available tokens."""
        received_args: list[tuple] = []

        def mock_build_history(entries, available_tokens):
            received_args.append((entries, available_tokens))
            return []  # return no messages

        budget = PromptBudget(
            max_prompt_tokens=1000,
            estimate_tokens=_simple_estimator,
            build_history=mock_build_history,
        )
        runner = self._make_runner(system_prompt="System.", prompt_budget=budget)

        task = ObservationContent(text="Task description.")
        state_content = ObservationContent(text="Current state.")
        initial_state = _make_state(task=task, state=state_content, step=1)

        # Add a transition so there's history
        prev_state = _make_state(task=task, state=ObservationContent(text="Prev."), step=0)
        trajectory = Trajectory.create(prev_state)
        trajectory.add_transition(
            Transition(
                state=prev_state,
                action=Action(text="action1"),
                next_state=initial_state,
                rewards=SignalBundle(()),
            )
        )

        runner._build_structured_messages(initial_state, trajectory)

        assert len(received_args) == 1
        entries, available = received_args[0]
        assert len(entries) == 1
        assert entries[0].action_text == "action1"

        # available should be: max_prompt_tokens - (system + task + state + initial_obs + overheads)
        system_cost = len("System.") + 5  # content + overhead
        task_cost = len("Task description.") + 5
        state_cost = len("Current state.") + 5
        initial_obs_cost = len("Prev.") + 5  # initial obs != task → reserved
        expected_available = 1000 - system_cost - task_cost - state_cost - initial_obs_cost
        assert available == expected_available

    def test_ample_budget_large_available(self) -> None:
        """When budget is huge, build_history gets a large available_tokens."""
        received_available: list[int] = []

        def mock_build_history(entries, available_tokens):
            received_available.append(available_tokens)
            return []

        budget = PromptBudget(
            max_prompt_tokens=100000,
            estimate_tokens=_simple_estimator,
            build_history=mock_build_history,
        )
        runner = self._make_runner(prompt_budget=budget)

        task = ObservationContent(text="Short task.")
        state_content = ObservationContent(text="Short state.")
        state = _make_state(task=task, state=state_content, step=1)
        prev = _make_state(task=task, state=ObservationContent(text="S."), step=0)
        traj = Trajectory.create(prev)
        traj.add_transition(Transition(
            state=prev, action=Action(text="a"), next_state=state,
            rewards=SignalBundle(()),
        ))

        runner._build_structured_messages(state, traj)

        assert len(received_available) == 1
        assert received_available[0] > 99000

    def test_large_system_prompt_shrinks_budget(self) -> None:
        """A large system prompt reduces available tokens for history."""
        received_available: list[int] = []

        def mock_build_history(entries, available_tokens):
            received_available.append(available_tokens)
            return []

        budget = PromptBudget(
            max_prompt_tokens=500,
            estimate_tokens=_simple_estimator,
            build_history=mock_build_history,
        )
        big_system = "X" * 300
        runner = self._make_runner(system_prompt=big_system, prompt_budget=budget)

        task = ObservationContent(text="Task.")
        state_content = ObservationContent(text="State.")
        state = _make_state(task=task, state=state_content, step=1)
        prev = _make_state(task=task, state=ObservationContent(text="P."), step=0)
        traj = Trajectory.create(prev)
        traj.add_transition(Transition(
            state=prev, action=Action(text="a"), next_state=state,
            rewards=SignalBundle(()),
        ))

        runner._build_structured_messages(state, traj)

        assert len(received_available) == 1
        # 500 - 300 (system) - 5 (task) - 6 (state) - 15 (overheads) ≈ 174
        assert received_available[0] < 200

    def test_prompt_budget_takes_precedence_over_history_fn(self) -> None:
        """When both prompt_budget and history_fn are set, budget wins."""
        budget_called = []
        history_fn_called = []

        def mock_build_history(entries, available_tokens):
            budget_called.append(True)
            return []

        def mock_history_fn(entries):
            history_fn_called.append(True)
            return []

        budget = PromptBudget(
            max_prompt_tokens=1000,
            estimate_tokens=_simple_estimator,
            build_history=mock_build_history,
        )
        runner = self._make_runner(
            prompt_budget=budget,
            history_fn=mock_history_fn,
        )

        task = ObservationContent(text="Task.")
        state = _make_state(task=task, state=ObservationContent(text="S."), step=1)
        prev = _make_state(task=task, state=ObservationContent(text="P."), step=0)
        traj = Trajectory.create(prev)
        traj.add_transition(Transition(
            state=prev, action=Action(text="a"), next_state=state,
            rewards=SignalBundle(()),
        ))

        runner._build_structured_messages(state, traj)

        assert len(budget_called) == 1
        assert len(history_fn_called) == 0

    def test_no_budget_uses_history_fn(self) -> None:
        """Without prompt_budget, history_fn is used as before."""
        history_fn_called = []

        def mock_history_fn(entries):
            history_fn_called.append(True)
            return []

        runner = self._make_runner(history_fn=mock_history_fn)

        task = ObservationContent(text="Task.")
        state = _make_state(task=task, state=ObservationContent(text="S."), step=1)
        prev = _make_state(task=task, state=ObservationContent(text="P."), step=0)
        traj = Trajectory.create(prev)
        traj.add_transition(Transition(
            state=prev, action=Action(text="a"), next_state=state,
            rewards=SignalBundle(()),
        ))

        runner._build_structured_messages(state, traj)

        assert len(history_fn_called) == 1

    def test_available_tokens_floor_at_zero(self) -> None:
        """Available tokens should never go negative."""
        received_available: list[int] = []

        def mock_build_history(entries, available_tokens):
            received_available.append(available_tokens)
            return []

        budget = PromptBudget(
            max_prompt_tokens=10,  # Very small
            estimate_tokens=_simple_estimator,
            build_history=mock_build_history,
        )
        # System prompt alone exceeds budget
        runner = self._make_runner(
            system_prompt="X" * 100,
            prompt_budget=budget,
        )

        task = ObservationContent(text="Task.")
        state = _make_state(task=task, state=ObservationContent(text="S."), step=1)
        prev = _make_state(task=task, state=ObservationContent(text="P."), step=0)
        traj = Trajectory.create(prev)
        traj.add_transition(Transition(
            state=prev, action=Action(text="a"), next_state=state,
            rewards=SignalBundle(()),
        ))

        runner._build_structured_messages(state, traj)

        assert len(received_available) == 1
        assert received_available[0] == 0

    def test_legacy_messages_use_prompt_budget(self) -> None:
        """Text-only legacy histories should also respect PromptBudget."""
        received_args: list[tuple[list[HistoryEntry], int]] = []

        def mock_build_history(entries, available_tokens):
            received_args.append((entries, available_tokens))
            return []

        budget = PromptBudget(
            max_prompt_tokens=1000,
            estimate_tokens=_simple_estimator,
            build_history=mock_build_history,
        )
        runner = self._make_runner(system_prompt="System.", prompt_budget=budget)

        state = State(
            observation=Observation(
                prompt="Task prompt.",
                messages=(
                    {"role": "assistant", "content": "action1"},
                    {"role": "user", "content": "obs1"},
                    {"role": "assistant", "content": "action2"},
                    {"role": "user", "content": "current observation"},
                ),
            ),
            hidden=None,
            metadata=StateMetadata(
                step=2,
                episode_id="test",
            ),
        )
        trajectory = Trajectory.create(state)

        runner._build_messages(state, trajectory=trajectory)

        assert len(received_args) == 1
        entries, available = received_args[0]
        assert [(e.action_text, e.observation_text) for e in entries] == [
            ("action1", "obs1"),
            ("action2", ""),
        ]

        system_cost = len("System.") + 5
        prompt_cost = len("Task prompt.") + 5
        current_cost = len("current observation") + 5
        expected_available = 1000 - system_cost - prompt_cost - current_cost
        assert available == expected_available


class TestCurrentObservationTruncation:
    """Tests for current-observation truncation in structured and legacy modes."""

    def _make_runner(self, system_prompt=None, prompt_budget=None):
        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        kwargs = dict(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
        )
        if prompt_budget is not None:
            kwargs["prompt_budget"] = prompt_budget

        return TrajectoryRunner(**kwargs)

    def test_structured_truncates_current_obs_after_history_exhausted(self) -> None:
        """When history is empty and current obs exceeds budget, truncate it."""
        budget = PromptBudget(
            max_prompt_tokens=100,
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
            min_current_observation_chars=20,
        )
        runner = self._make_runner(system_prompt="Sys", prompt_budget=budget)

        task = ObservationContent(text="Task")
        # Large current state that won't fit in budget of 100 tokens
        big_state = ObservationContent(text="X" * 200)
        state = _make_state(task=task, state=big_state, step=0)
        traj = Trajectory.create(state)

        messages = runner._build_structured_messages(state, traj)

        # Find the current state message (last user message)
        current_state_msg = [m for m in messages if m.role == "user"][-1]
        # Should be truncated — shorter than original but at least min floor
        assert len(current_state_msg.content) < 200
        assert "omitted" in current_state_msg.content

    def test_structured_no_truncation_when_none(self) -> None:
        """When min_current_observation_chars is None, current obs is not truncated."""
        budget = PromptBudget(
            max_prompt_tokens=100,
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
            min_current_observation_chars=None,
        )
        runner = self._make_runner(system_prompt="Sys", prompt_budget=budget)

        task = ObservationContent(text="Task")
        big_state = ObservationContent(text="X" * 200)
        state = _make_state(task=task, state=big_state, step=0)
        traj = Trajectory.create(state)

        messages = runner._build_structured_messages(state, traj)

        current_state_msg = [m for m in messages if m.role == "user"][-1]
        # Should NOT be truncated
        assert current_state_msg.content == "X" * 200

    def test_structured_no_truncation_when_budget_sufficient(self) -> None:
        """Current obs not truncated when budget is ample."""
        budget = PromptBudget(
            max_prompt_tokens=10000,
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
            min_current_observation_chars=20,
        )
        runner = self._make_runner(prompt_budget=budget)

        task = ObservationContent(text="Task")
        state_content = ObservationContent(text="X" * 200)
        state = _make_state(task=task, state=state_content, step=0)
        traj = Trajectory.create(state)

        messages = runner._build_structured_messages(state, traj)

        current_state_msg = [m for m in messages if m.role == "user"][-1]
        assert current_state_msg.content == "X" * 200

    def test_structured_respects_floor(self) -> None:
        """Truncated current observation is at least min_current_observation_chars."""
        floor = 50
        budget = PromptBudget(
            max_prompt_tokens=30,  # Very tight — can't even fit the floor
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
            min_current_observation_chars=floor,
        )
        runner = self._make_runner(system_prompt="S", prompt_budget=budget)

        task = ObservationContent(text="T")
        big_state = ObservationContent(text="X" * 500)
        state = _make_state(task=task, state=big_state, step=0)
        traj = Trajectory.create(state)

        messages = runner._build_structured_messages(state, traj)

        current_state_msg = [m for m in messages if m.role == "user"][-1]
        # The floor is respected even if prompt is still over budget
        # (vLLM handles the final check)
        assert len(current_state_msg.content) < 500
        # The truncated text should preserve at least floor/2 chars from
        # each end plus the "[... N chars omitted ...]" marker
        assert "omitted" in current_state_msg.content

    def test_legacy_truncates_current_obs(self) -> None:
        """Legacy text-chat mode truncates final user message as current obs."""
        budget = PromptBudget(
            max_prompt_tokens=80,
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
            min_current_observation_chars=20,
        )
        runner = self._make_runner(system_prompt="Sys", prompt_budget=budget)

        state = State(
            observation=Observation(
                prompt="Task.",
                messages=(
                    {"role": "assistant", "content": "act"},
                    {"role": "user", "content": "X" * 200},
                ),
            ),
            hidden=None,
            metadata=StateMetadata(step=1, episode_id="test"),
        )
        traj = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=traj)

        # The last user message should be truncated
        last_user = [m for m in messages if m.role == "user"][-1]
        assert len(last_user.content) < 200
        assert "omitted" in last_user.content

    def test_legacy_no_truncation_when_none(self) -> None:
        """Legacy mode: no truncation when min_current_observation_chars is None."""
        budget = PromptBudget(
            max_prompt_tokens=80,
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
            min_current_observation_chars=None,
        )
        runner = self._make_runner(system_prompt="Sys", prompt_budget=budget)

        state = State(
            observation=Observation(
                prompt="Task.",
                messages=(
                    {"role": "assistant", "content": "act"},
                    {"role": "user", "content": "X" * 200},
                ),
            ),
            hidden=None,
            metadata=StateMetadata(step=1, episode_id="test"),
        )
        traj = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=traj)

        last_user = [m for m in messages if m.role == "user"][-1]
        # Full observation should be present (not truncated)
        assert "X" * 200 in last_user.content
        assert "omitted" not in last_user.content


# =============================================================================
# Initial observation injection with PromptBudget
# =============================================================================


class TestBudgetInitialObservation:
    """Tests for initial observation injection in the prompt_budget path."""

    def _make_runner(self, system_prompt=None, prompt_budget=None):
        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        return TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
            prompt_budget=prompt_budget,
        )

    def test_budget_initial_obs_injected_when_history_nonempty(self) -> None:
        """Initial obs appears in output when build_history returns content."""
        def mock_build(entries, available):
            return [
                ChatMessage(role="assistant", content="act"),
                ChatMessage(role="user", content="obs"),
            ]

        budget = PromptBudget(
            max_prompt_tokens=10000,
            estimate_tokens=_simple_estimator,
            build_history=mock_build,
        )
        runner = self._make_runner(prompt_budget=budget)

        task = ObservationContent(text="Task.")
        s0 = _make_state(task=task, state=ObservationContent(text="Initial."), step=0)
        s1 = _make_state(task=task, state=ObservationContent(text="Current."), step=1)

        traj = Trajectory.create(s0)
        traj.add_transition(Transition(
            state=s0, action=Action(text="act"), next_state=s1,
            rewards=SignalBundle(()),
        ))

        # Use _build_messages to get coalesced output
        messages = runner._build_messages(s1, trajectory=traj)
        first_user = messages[0]
        assert first_user.role == "user"
        assert "Task." in first_user.content
        assert "Initial." in first_user.content

    def test_budget_initial_obs_skipped_when_history_empty(self) -> None:
        """Initial obs NOT injected when build_history returns empty."""
        budget = PromptBudget(
            max_prompt_tokens=10000,
            estimate_tokens=_simple_estimator,
            build_history=lambda entries, available: [],
        )
        runner = self._make_runner(prompt_budget=budget)

        task = ObservationContent(text="Task.")
        s0 = _make_state(task=task, state=ObservationContent(text="Initial."), step=0)
        s1 = _make_state(task=task, state=ObservationContent(text="Current."), step=1)

        traj = Trajectory.create(s0)
        traj.add_transition(Transition(
            state=s0, action=Action(text="act"), next_state=s1,
            rewards=SignalBundle(()),
        ))

        messages = runner._build_structured_messages(s1, traj)
        first_user = messages[0]
        assert "Initial." not in first_user.content

    def test_budget_reserves_space_for_initial_obs(self) -> None:
        """Available tokens passed to build_history account for initial obs."""
        received: list[int] = []

        def mock_build(entries, available):
            received.append(available)
            return []

        budget = PromptBudget(
            max_prompt_tokens=1000,
            estimate_tokens=_simple_estimator,
            build_history=mock_build,
        )
        runner = self._make_runner(system_prompt="Sys.", prompt_budget=budget)

        task = ObservationContent(text="Task.")
        s0 = _make_state(task=task, state=ObservationContent(text="Init."), step=0)
        s1 = _make_state(task=task, state=ObservationContent(text="Curr."), step=1)

        traj = Trajectory.create(s0)
        traj.add_transition(Transition(
            state=s0, action=Action(text="a"), next_state=s1,
            rewards=SignalBundle(()),
        ))

        runner._build_structured_messages(s1, traj)

        # Budget: 1000 - sys("Sys." + 5) - task("Task." + 5) - state("Curr." + 5) - init("Init." + 5)
        sys_cost = len("Sys.") + 5
        task_cost = len("Task.") + 5
        state_cost = len("Curr.") + 5
        init_cost = len("Init.") + 5
        expected = 1000 - sys_cost - task_cost - state_cost - init_cost
        assert received[0] == expected
