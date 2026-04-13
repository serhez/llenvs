"""Tests for runner message building: _coalesce_messages and structured mode."""

from __future__ import annotations

import pytest

from llenvs.core.reward import SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.evaluation.history import HistoryEntry, no_history
from llenvs.evaluation.runner import (
    MultiEvalEntry,
    TrajectoryRunner,
    _coalesce_messages,
    _task_index_for_state,
    run_multi_evaluation,
)
from llenvs.inference.protocol import ChatMessage, GenerationResult, StopReason

# =============================================================================
# _coalesce_messages tests
# =============================================================================


class TestCoalesceMessages:
    """Tests for the _coalesce_messages helper."""

    def test_empty_list(self):
        assert _coalesce_messages([]) == []

    def test_no_consecutive_same_role(self):
        msgs = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="bye"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 3

    def test_merges_consecutive_user_messages(self):
        msgs = [
            ChatMessage(role="user", content="part1"),
            ChatMessage(role="user", content="part2"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 1
        assert result[0].content == "part1\n\npart2"
        assert result[0].role == "user"

    def test_merges_consecutive_assistant_messages(self):
        msgs = [
            ChatMessage(role="assistant", content="thought"),
            ChatMessage(role="assistant", content="answer"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 1
        assert result[0].content == "thought\n\nanswer"

    def test_preserves_system_messages(self):
        """System messages are never merged."""
        msgs = [
            ChatMessage(role="system", content="sys1"),
            ChatMessage(role="system", content="sys2"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 2

    def test_preserves_tool_messages(self):
        """Tool messages are never merged."""
        msgs = [
            ChatMessage(role="tool", content="result1", tool_call_id="1"),
            ChatMessage(role="tool", content="result2", tool_call_id="2"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 2

    def test_does_not_merge_tool_call_messages(self):
        """Assistant messages with tool_calls are not merged."""
        from llenvs.core.tools import ToolCall

        tc = ToolCall(id="1", name="fn", arguments={})
        msgs = [
            ChatMessage(role="assistant", content="text", tool_calls=(tc,)),
            ChatMessage(role="assistant", content="more"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 2

    def test_concatenates_images(self):
        img1 = ImageContent(data="abc", media_type="image/png")
        img2 = ImageContent(data="def", media_type="image/png")
        msgs = [
            ChatMessage(role="user", content="look", images=(img1,)),
            ChatMessage(role="user", content="again", images=(img2,)),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 1
        assert len(result[0].images) == 2
        assert result[0].images[0].data == "abc"
        assert result[0].images[1].data == "def"

    def test_complex_sequence(self):
        """User-user-assistant-user should become user-assistant-user."""
        msgs = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
            ChatMessage(role="assistant", content="c"),
            ChatMessage(role="user", content="d"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 3
        assert result[0].content == "a\n\nb"
        assert result[1].content == "c"
        assert result[2].content == "d"

    def test_three_consecutive(self):
        msgs = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
            ChatMessage(role="user", content="c"),
        ]
        result = _coalesce_messages(msgs)
        assert len(result) == 1
        assert result[0].content == "a\n\nb\n\nc"


# =============================================================================
# Structured message building tests
# =============================================================================


def _make_state(
    prompt: str = "",
    messages: tuple = (),
    task: ObservationContent | None = None,
    state: ObservationContent | None = None,
    step: int = 0,
    is_terminal: bool = False,
) -> State:
    return State(
        observation=Observation(
            prompt=prompt,
            messages=messages,
            task=task,
            state=state,
        ),
        hidden=None,
        metadata=StateMetadata(
            step=step,
            episode_id="test",
            is_terminal=is_terminal,
        ),
    )


class TestStructuredMessageBuilding:
    """Tests for TrajectoryRunner structured message building."""

    def _make_runner(self, system_prompt=None):
        """Create a minimal TrajectoryRunner with a mock backend."""
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner
        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        return TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
        )

    def test_structured_mode_with_task(self):
        """When task is set and trajectory provided, use structured mode."""
        runner = self._make_runner()

        task = ObservationContent(text="Solve the puzzle.")
        state_content = ObservationContent(text="You see a door.")
        initial_state = _make_state(
            prompt="Solve the puzzle.",
            task=task,
            state=state_content,
        )
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        # Should have: user (task+state coalesced)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Solve the puzzle." in messages[0].content
        assert "You see a door." in messages[0].content

    def test_structured_mode_with_transitions(self):
        """Structured mode reconstructs history from trajectory transitions."""
        runner = self._make_runner()

        task = ObservationContent(text="Navigate the maze.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Room 2")

        s0 = _make_state(prompt="Navigate the maze.", task=task, state=state0)
        s1 = _make_state(prompt="Navigate the maze.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="go north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        # user(task + initial_obs coalesced) + assistant(go north) + user(state1)
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert "Navigate the maze." in messages[0].content
        assert "Start" in messages[0].content
        assert messages[1].role == "assistant"
        assert messages[1].content == "go north"
        assert messages[2].role == "user"
        assert "Room 2" in messages[2].content

    def test_structured_mode_with_system_prompt(self):
        """System prompt is included in structured mode."""
        runner = self._make_runner(system_prompt="You are helpful.")

        task = ObservationContent(text="Task here.")
        state_content = ObservationContent(text="Obs.")
        initial_state = _make_state(prompt="Task here.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful."
        # user message should be coalesced task+state
        assert messages[1].role == "user"
        assert "Task here." in messages[1].content

    def test_legacy_mode_without_task(self):
        """When task is None, use legacy mode."""
        runner = self._make_runner()

        state = _make_state(prompt="What is 2+2?")
        trajectory = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=trajectory)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "What is 2+2?"

    def test_legacy_mode_without_trajectory(self):
        """When trajectory is None, use legacy mode even with task set."""
        runner = self._make_runner()

        task = ObservationContent(text="Task.")
        state = _make_state(prompt="prompt", task=task)

        messages = runner._build_messages(state, trajectory=None)
        assert len(messages) == 1
        assert messages[0].content == "prompt"

    def test_structured_coalesces_task_and_step0(self):
        """Task message and step-0 user message get coalesced."""
        runner = self._make_runner()

        task = ObservationContent(text="Explore the dungeon.")
        state0 = ObservationContent(text="You are in a dark room.")

        initial_state = _make_state(prompt="Explore the dungeon.", task=task, state=state0)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        # Task + state0 coalesced into one user message
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Explore the dungeon." in messages[0].content
        assert "You are in a dark room." in messages[0].content

    def test_structured_skips_duplicate_initial_state_when_same_as_task(self):
        """Step 0 should not repeat identical task/state text."""
        runner = self._make_runner()

        task = ObservationContent(text="Repeat me once.")
        initial_state = _make_state(
            prompt="Repeat me once.",
            task=task,
            state=ObservationContent(text="Repeat me once."),
        )
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Repeat me once."


# =============================================================================
# History function integration tests
# =============================================================================


class TestHistoryFnIntegration:
    """Tests for history_fn parameter in structured message building."""

    def _make_runner(self, system_prompt=None, history_fn=None, include_reasoning_in_history=False):
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner
        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        return TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
            history_fn=history_fn,
            include_reasoning_in_history=include_reasoning_in_history,
        )

    def _build_trajectory_with_transitions(self):
        """Build a trajectory with 2 transitions for testing."""
        task = ObservationContent(text="Navigate the maze.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Room 2")
        state2 = ObservationContent(text="Room 3")

        s0 = _make_state(prompt="Navigate the maze.", task=task, state=state0)
        s1 = _make_state(prompt="Navigate the maze.", task=task, state=state1, step=1)
        s2 = _make_state(prompt="Navigate the maze.", task=task, state=state2, step=2)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="go north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )
        trajectory.add_transition(
            Transition(
                state=s1,
                action=Action(text="go east"),
                next_state=s2,
                rewards=SignalBundle(signals=()),
            )
        )
        return trajectory, s2

    def test_default_history_fn_is_full(self):
        """Default behavior includes full history."""
        runner = self._make_runner()
        trajectory, current_state = self._build_trajectory_with_transitions()

        messages = runner._build_messages(current_state, trajectory=trajectory)
        # user(task + initial_obs) + assistant + user + assistant + user = 5
        assert len(messages) == 5
        assert messages[0].role == "user"
        assert "Navigate the maze." in messages[0].content
        assert "Start" in messages[0].content
        assert messages[1].role == "assistant"
        assert messages[1].content == "go north"
        assert messages[2].role == "user"
        assert "Room 2" in messages[2].content
        assert messages[3].role == "assistant"
        assert messages[3].content == "go east"
        assert messages[4].role == "user"
        assert "Room 3" in messages[4].content

    def test_no_history_fn(self):
        """no_history drops all prior turns, showing only task + current state."""
        runner = self._make_runner(history_fn=no_history)
        trajectory, current_state = self._build_trajectory_with_transitions()

        messages = runner._build_messages(current_state, trajectory=trajectory)
        # task(user) + current_state(user) = coalesced into 1
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Navigate the maze." in messages[0].content
        assert "Room 3" in messages[0].content

    def test_custom_history_fn(self):
        """Custom history_fn controls which entries appear."""

        def only_last(entries: list[HistoryEntry]) -> list[ChatMessage]:
            if not entries:
                return []
            e = entries[-1]
            return [
                ChatMessage(role="assistant", content=e.action_text),
                ChatMessage(role="user", content=e.observation_text),
            ]

        runner = self._make_runner(history_fn=only_last)
        trajectory, current_state = self._build_trajectory_with_transitions()

        messages = runner._build_messages(current_state, trajectory=trajectory)
        # user(task + initial_obs) + assistant(go east) + user(coalesced empty + Room 3) = 3
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert "Navigate the maze." in messages[0].content
        assert "Start" in messages[0].content
        assert messages[1].role == "assistant"
        assert messages[1].content == "go east"
        assert messages[2].role == "user"
        assert "Room 3" in messages[2].content

    def test_no_history_with_system_prompt(self):
        """no_history + system prompt yields system + task+current coalesced."""
        runner = self._make_runner(system_prompt="Be helpful.", history_fn=no_history)
        trajectory, current_state = self._build_trajectory_with_transitions()

        messages = runner._build_messages(current_state, trajectory=trajectory)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful."
        assert messages[1].role == "user"
        assert "Navigate the maze." in messages[1].content
        assert "Room 3" in messages[1].content

    def test_history_fn_applies_in_legacy_mode_for_text_history(self):
        """Legacy text-only histories should respect history_fn shaping."""
        runner = self._make_runner(history_fn=no_history)
        state = _make_state(
            prompt="What is 2+2?",
            messages=(
                {"role": "assistant", "content": "Let me calculate."},
                {"role": "user", "content": "Still thinking..."},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "Please provide just the answer."},
            ),
            step=2,
        )
        trajectory = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=trajectory)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "What is 2+2?" in messages[0].content
        assert "Please provide just the answer." in messages[0].content
        assert "Let me calculate." not in messages[0].content
        assert "Still thinking..." not in messages[0].content


class TestIncludeReasoningInHistory:
    """Tests for include_reasoning_in_history parameter."""

    def _make_runner(self, include_reasoning_in_history=False, history_fn=None):
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner
        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        return TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            include_reasoning_in_history=include_reasoning_in_history,
            history_fn=history_fn,
        )

    def test_default_strips_reasoning(self):
        """Default (False): uses extracted_action from step info."""
        runner = self._make_runner(include_reasoning_in_history=False)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="Let me think...\n\ngo north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={"step": {"extracted_action": "go north"}},
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        # Find the assistant message
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "go north"  # stripped

    def test_include_reasoning_shows_full_text(self):
        """When True: uses full action.text including reasoning."""
        runner = self._make_runner(include_reasoning_in_history=True)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="Let me think...\n\ngo north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={"step": {"extracted_action": "go north"}},
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "Let me think...\n\ngo north"

    def test_fallback_when_no_extracted_action(self):
        """When extracted_action missing, falls back to full action.text."""
        runner = self._make_runner(include_reasoning_in_history=False)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="go north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={},  # no step info
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "go north"

    def test_fallback_when_step_info_has_no_extracted_action(self):
        """When step info exists but has no extracted_action key."""
        runner = self._make_runner(include_reasoning_in_history=False)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="go north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={"step": {"gym_reward": 0.5}},  # step info but no extracted_action
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "go north"

    def test_fallback_strips_thinking_tokens(self):
        """When no extracted_action and include_reasoning=False, strip <think> from fallback."""
        runner = self._make_runner(include_reasoning_in_history=False)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="<think>long reasoning here</think>\ngo north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={"step": {"error": "invalid action"}},  # error step, no extracted_action
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert "<think>" not in assistant_msgs[0].content
        assert "go north" in assistant_msgs[0].content

    def test_fallback_strips_unclosed_thinking_tokens(self):
        """Unclosed <think> blocks (truncation) are also stripped in fallback."""
        runner = self._make_runner(include_reasoning_in_history=False)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        # Simulate truncation: thinking consumed entire budget, no closing tag
        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="<think>very long reasoning that consumed entire budget"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={},  # no step info at all
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert "<think>" not in assistant_msgs[0].content
        assert "very long reasoning" not in assistant_msgs[0].content

    def test_fallback_no_stripping_when_include_reasoning_true(self):
        """When include_reasoning=True, thinking tokens are preserved even in fallback."""
        runner = self._make_runner(include_reasoning_in_history=True)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="<think>reasoning</think>\ngo north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={"step": {"error": "invalid action"}},
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert "<think>" in assistant_msgs[0].content

    def test_extracted_answer_also_works(self):
        """extracted_answer (single-turn naming) is also checked."""
        runner = self._make_runner(include_reasoning_in_history=False)

        task = ObservationContent(text="Solve it.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Next")

        s0 = _make_state(prompt="Solve it.", task=task, state=state0)
        s1 = _make_state(prompt="Solve it.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="<think>hmm</think>\n42"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
                info={"step": {"extracted_answer": "42"}},
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "42"


# =============================================================================
# TurnInfoConfig injection tests
# =============================================================================


class TestTurnInfoInjection:
    """Tests for TurnInfoConfig injection in structured message building."""

    def _make_runner(self, turn_info=None, system_prompt=None, max_steps=10):
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner
        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = max_steps
        mock_backend = MagicMock()

        return TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
            turn_info=turn_info,
        )

    def test_turn_info_disabled_by_default(self):
        """No turn info when turn_info is None (default)."""
        runner = self._make_runner(turn_info=None)

        task = ObservationContent(text="Solve the puzzle.")
        state_content = ObservationContent(text="You see a door.")
        initial_state = _make_state(prompt="Solve the puzzle.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        content = messages[0].content
        assert "[Turn" not in content
        assert "maximum of" not in content

    def test_turn_info_true_shorthand(self):
        """turn_info=True enables default TurnInfoConfig."""
        runner = self._make_runner(turn_info=True)

        task = ObservationContent(text="Solve the puzzle.")
        state_content = ObservationContent(text="You see a door.")
        initial_state = _make_state(prompt="Solve the puzzle.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        # Task + state coalesced into one user message
        assert len(messages) == 1
        content = messages[0].content
        # Task suffix should appear
        assert "maximum of 10 turns" in content
        # State prefix should appear
        assert "[Turn 1/10]" in content

    def test_turn_info_custom_config(self):
        """Custom TurnInfoConfig formats are used."""
        from llenvs.evaluation.runner import TurnInfoConfig

        tic = TurnInfoConfig(
            task_suffix="\n\n(Max {max_steps} steps)",
            state_prefix="Step {turn}: ",
        )
        runner = self._make_runner(turn_info=tic, max_steps=5)

        task = ObservationContent(text="Navigate.")
        state_content = ObservationContent(text="Room 1")
        initial_state = _make_state(prompt="Navigate.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        content = messages[0].content
        assert "(Max 5 steps)" in content
        assert "Step 1: Room 1" in content

    def test_task_suffix_with_max_steps(self):
        """Task suffix uses max_steps when available."""
        runner = self._make_runner(turn_info=True, max_steps=20)

        task = ObservationContent(text="Task.")
        state_content = ObservationContent(text="Obs.")
        initial_state = _make_state(prompt="Task.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        content = messages[0].content
        assert "maximum of 20 turns" in content

    def test_task_suffix_without_max_steps(self):
        """Task suffix uses no_max variant when max_steps is None."""
        runner = self._make_runner(turn_info=True, max_steps=None)

        task = ObservationContent(text="Task.")
        state_content = ObservationContent(text="Obs.")
        initial_state = _make_state(prompt="Task.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        content = messages[0].content
        # Default task_suffix_no_max is empty, so no suffix
        assert "maximum of" not in content
        # But state prefix should still appear (no_max variant)
        assert "[Turn 1]" in content

    def test_state_prefix_step_numbering(self):
        """State prefix uses 1-indexed turn number from metadata.step."""
        runner = self._make_runner(turn_info=True, max_steps=10)

        task = ObservationContent(text="Navigate.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Room 2")

        s0 = _make_state(prompt="Navigate.", task=task, state=state0)
        s1 = _make_state(prompt="Navigate.", task=task, state=state1, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="go north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        # Initial obs gets [Turn 1/10], current state gets [Turn 2/10]
        first_user = messages[0].content
        assert "[Turn 1/10]" in first_user
        assert "Start" in first_user
        user_msgs = [m for m in messages if m.role == "user"]
        last_user = user_msgs[-1].content
        assert "[Turn 2/10]" in last_user

    def test_state_prefix_no_max_steps(self):
        """State prefix uses no_max variant when spec.max_steps is None."""
        runner = self._make_runner(turn_info=True, max_steps=None)

        task = ObservationContent(text="Navigate.")
        state_content = ObservationContent(text="Obs.")
        state = _make_state(prompt="Navigate.", task=task, state=state_content, step=2)
        trajectory = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=trajectory)
        content = messages[0].content
        assert "[Turn 3]" in content
        assert "[Turn 3/" not in content

    def test_turn_info_history_unaffected(self):
        """History entries are NOT modified by turn info."""
        runner = self._make_runner(turn_info=True, max_steps=10)

        task = ObservationContent(text="Navigate.")
        state0 = ObservationContent(text="Start")
        state1 = ObservationContent(text="Room 2")
        state2 = ObservationContent(text="Room 3")

        s0 = _make_state(prompt="Navigate.", task=task, state=state0)
        s1 = _make_state(prompt="Navigate.", task=task, state=state1, step=1)
        s2 = _make_state(prompt="Navigate.", task=task, state=state2, step=2)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0,
                action=Action(text="go north"),
                next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )
        trajectory.add_transition(
            Transition(
                state=s1,
                action=Action(text="go east"),
                next_state=s2,
                rewards=SignalBundle(signals=()),
            )
        )

        messages = runner._build_messages(s2, trajectory=trajectory)
        # Messages: user(task + [Turn 1/10] initial_obs) + assistant + user(Room 2) + assistant + user([Turn 3/10] Room 3)
        assert len(messages) == 5
        # First user message has task + initial obs with turn prefix
        assert "Start" in messages[0].content
        assert "[Turn 1/10]" in messages[0].content
        # The intermediate history observation at messages[2] should be raw
        assert messages[2].content == "Room 2"
        # Only the current state (last user) gets the prefix
        assert "[Turn 3/10]" in messages[4].content

    def test_turn_info_false_disables(self):
        """turn_info=False is equivalent to None (disabled)."""
        runner = self._make_runner(turn_info=False)

        task = ObservationContent(text="Task.")
        state_content = ObservationContent(text="Obs.")
        initial_state = _make_state(prompt="Task.", task=task, state=state_content)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        content = messages[0].content
        assert "[Turn" not in content

    def test_turn_info_legacy_mode_no_effect(self):
        """Turn info has no effect in legacy mode (no task field)."""
        runner = self._make_runner(turn_info=True)

        state = _make_state(prompt="What is 2+2?")
        trajectory = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=trajectory)
        assert len(messages) == 1
        assert messages[0].content == "What is 2+2?"
        assert "[Turn" not in messages[0].content


# =============================================================================
# Initial observation injection tests
# =============================================================================


class TestInitialObservation:
    """Tests for injecting the initial observation into the prompt at step 1+."""

    def _make_runner(self, system_prompt=None, history_fn=None, turn_info=None, max_steps=10):
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner
        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = max_steps
        mock_backend = MagicMock()

        kwargs = dict(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
        )
        if history_fn is not None:
            kwargs["history_fn"] = history_fn
        if turn_info is not None:
            kwargs["turn_info"] = turn_info
        return TrajectoryRunner(**kwargs)

    def _build_one_transition(self, task_text="Task.", state0_text="Initial room.",
                               state1_text="Room 2.", task_images=(), state0_images=()):
        """Build a trajectory with 1 transition (step 0 → step 1)."""
        task = ObservationContent(text=task_text, images=task_images)
        s0_content = ObservationContent(text=state0_text, images=state0_images)
        s1_content = ObservationContent(text=state1_text)

        s0 = _make_state(prompt=task_text, task=task, state=s0_content, step=0)
        s1 = _make_state(prompt=task_text, task=task, state=s1_content, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0, action=Action(text="go north"), next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )
        return trajectory, s1

    def test_initial_obs_visible_with_full_history(self):
        """At step 1 with full history, initial observation is in first user message."""
        runner = self._make_runner()
        trajectory, current = self._build_one_transition(
            task_text="Objective: find key.", state0_text="Kitchen.", state1_text="Hallway.",
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        # user(task + initial_obs) + assistant(go north) + user(current)
        assert len(messages) == 3
        assert "Objective: find key." in messages[0].content
        assert "Kitchen." in messages[0].content
        assert messages[1].content == "go north"
        assert "Hallway." in messages[2].content

    def test_initial_obs_not_shown_at_step0(self):
        """At step 0, no injection — initial obs IS the current state."""
        runner = self._make_runner()
        task = ObservationContent(text="Task.")
        state0 = ObservationContent(text="Initial room.")
        s0 = _make_state(prompt="Task.", task=task, state=state0, step=0)
        trajectory = Trajectory.create(s0)

        messages = runner._build_messages(s0, trajectory=trajectory)
        assert len(messages) == 1
        assert "Task." in messages[0].content
        assert "Initial room." in messages[0].content
        # Should appear only once
        assert messages[0].content.count("Initial room.") == 1

    def test_initial_obs_not_shown_with_no_history(self):
        """With no_history, initial observation is NOT injected."""
        runner = self._make_runner(history_fn=no_history)
        trajectory, current = self._build_one_transition(
            task_text="Task.", state0_text="Initial room.", state1_text="Room 2.",
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        # Only task + current state (coalesced)
        assert len(messages) == 1
        assert "Task." in messages[0].content
        assert "Room 2." in messages[0].content
        assert "Initial room." not in messages[0].content

    def test_initial_obs_dedup_when_same_as_task(self):
        """Harbor scenario: task == state at step 0 → no duplication at step 1."""
        runner = self._make_runner()
        trajectory, current = self._build_one_transition(
            task_text="Install flask.", state0_text="Install flask.", state1_text="Output.",
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        assert len(messages) == 3
        # First user message should contain task only once
        assert messages[0].content.count("Install flask.") == 1

    def test_initial_obs_dedup_with_whitespace(self):
        """Dedup works with whitespace differences."""
        runner = self._make_runner()
        trajectory, current = self._build_one_transition(
            task_text="Install flask.\n", state0_text="Install flask. ", state1_text="Out.",
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        # Should dedup despite trailing whitespace differences
        assert messages[0].content.count("Install flask.") == 1

    def test_initial_obs_with_turn_info_prefix(self):
        """When TurnInfoConfig is active, initial obs gets [Turn 1/N] prefix."""
        runner = self._make_runner(turn_info=True, max_steps=10)
        trajectory, current = self._build_one_transition(
            task_text="Task.", state0_text="Start.", state1_text="Room 2.",
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        first_user = messages[0].content
        assert "[Turn 1/10]" in first_user
        assert "Start." in first_user

    def test_initial_obs_turn_info_mid_trajectory_restore(self):
        """When restoring from mid-trajectory, initial obs uses actual turn number."""
        runner = self._make_runner(turn_info=True, max_steps=50)
        task = ObservationContent(text="Task.")
        s0_content = ObservationContent(text="Restored state.")
        s1_content = ObservationContent(text="Next state.")

        # Simulate restoring from step 8 (turn 9)
        s0 = _make_state(prompt="Task.", task=task, state=s0_content, step=8)
        s1 = _make_state(prompt="Task.", task=task, state=s1_content, step=9)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0, action=Action(text="go north"), next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        first_user = messages[0].content
        assert "[Turn 9/50]" in first_user
        assert "[Turn 1/" not in first_user
        assert "Restored state." in first_user

    def test_initial_obs_with_images(self):
        """Images from initial state are carried into the injected message."""
        img = ImageContent(data="abc", media_type="image/png")
        runner = self._make_runner()
        trajectory, current = self._build_one_transition(
            task_text="Task.", state0_text="Visual.", state1_text="Next.",
            state0_images=(img,),
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        assert img in messages[0].images

    def test_initial_obs_with_empty_state_text(self):
        """When initial state text is empty, no injection occurs."""
        runner = self._make_runner()
        task = ObservationContent(text="Task.")
        s0_content = ObservationContent(text="")
        s1_content = ObservationContent(text="Room 2.")

        s0 = _make_state(prompt="Task.", task=task, state=s0_content, step=0)
        s1 = _make_state(prompt="Task.", task=task, state=s1_content, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0, action=Action(text="go"), next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        # Only task + go + current
        assert len(messages) == 3
        assert messages[0].content == "Task."

    def test_initial_obs_when_no_state_field(self):
        """When initial observation has no state field, no injection."""
        runner = self._make_runner()
        task = ObservationContent(text="Task.")
        s1_content = ObservationContent(text="Room 2.")

        s0 = _make_state(prompt="Task.", task=task, state=None, step=0)
        s1 = _make_state(prompt="Task.", task=task, state=s1_content, step=1)

        trajectory = Trajectory.create(s0)
        trajectory.add_transition(
            Transition(
                state=s0, action=Action(text="go"), next_state=s1,
                rewards=SignalBundle(signals=()),
            )
        )

        messages = runner._build_messages(s1, trajectory=trajectory)
        assert messages[0].content == "Task."

    def test_initial_obs_shown_with_last_n_history(self):
        """Initial obs appears even when last_n_history drops early turns."""
        from llenvs.evaluation.history import last_n_history

        runner = self._make_runner(history_fn=last_n_history(1))

        task = ObservationContent(text="Task.")
        s0 = ObservationContent(text="Initial.")
        s1 = ObservationContent(text="Mid.")
        s2 = ObservationContent(text="Current.")

        st0 = _make_state(prompt="Task.", task=task, state=s0, step=0)
        st1 = _make_state(prompt="Task.", task=task, state=s1, step=1)
        st2 = _make_state(prompt="Task.", task=task, state=s2, step=2)

        trajectory = Trajectory.create(st0)
        trajectory.add_transition(Transition(
            state=st0, action=Action(text="a1"), next_state=st1,
            rewards=SignalBundle(signals=()),
        ))
        trajectory.add_transition(Transition(
            state=st1, action=Action(text="a2"), next_state=st2,
            rewards=SignalBundle(signals=()),
        ))

        messages = runner._build_messages(st2, trajectory=trajectory)
        # last_n_history(1) keeps only last entry (a2), but initial obs
        # is injected because history is non-empty.
        assert "Initial." in messages[0].content
        assert "Task." in messages[0].content

    def test_initial_obs_with_empty_task_text(self):
        """Jericho/Craftax scenario: task.text="" but state has content."""
        runner = self._make_runner()
        trajectory, current = self._build_one_transition(
            task_text="", state0_text="Game opening.", state1_text="Room 2.",
        )

        messages = runner._build_messages(current, trajectory=trajectory)
        # Empty task coalesced with initial obs
        assert "Game opening." in messages[0].content


# =============================================================================
# _task_index_for_state tests
# =============================================================================


class TestTaskIndexForState:
    def _make_state(self, *, info=None, hidden=None):
        return State(
            observation=Observation(prompt="obs"),
            hidden=hidden,
            metadata=StateMetadata(
                step=0,
                episode_id="test",
                is_terminal=False,
                info=info or {},
            ),
        )

    def test_extracts_from_metadata_info(self):
        state = self._make_state(info={"task_index": 7})
        assert _task_index_for_state(state) == 7

    def test_extracts_from_hidden_task_index(self):
        hidden = type("Hidden", (), {"task_index": 3})()
        state = self._make_state(hidden=hidden)
        assert _task_index_for_state(state) == 3

    def test_defaults_to_zero(self):
        state = self._make_state()
        assert _task_index_for_state(state) == 0

    def test_metadata_info_takes_precedence(self):
        hidden = type("Hidden", (), {"task_index": 3})()
        state = self._make_state(info={"task_index": 5}, hidden=hidden)
        assert _task_index_for_state(state) == 5

    def test_non_int_task_index_falls_through(self):
        state = self._make_state(info={"task_index": "not-an-int"})
        assert _task_index_for_state(state) == 0


# =============================================================================
# system_prompt_fn tests
# =============================================================================


class TestSystemPromptFn:
    def _make_env(self):
        from unittest.mock import MagicMock
        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        return mock_env

    def _make_backend(self):
        from unittest.mock import MagicMock
        return MagicMock()

    def test_mutually_exclusive_with_system_prompt(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TrajectoryRunner(
                environment=self._make_env(),
                backend=self._make_backend(),
                system_prompt="static",
                system_prompt_fn=lambda s, t: "dynamic",
            )

    def test_system_prompt_fn_resolves_per_state(self):
        runner = TrajectoryRunner(
            environment=self._make_env(),
            backend=self._make_backend(),
            system_prompt_fn=lambda state, task_index: f"prompt-{task_index}",
        )
        state = State(
            observation=Observation(prompt="obs"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="t", is_terminal=False, info={}),
        )
        assert runner._resolve_system_prompt(state, 0) == "prompt-0"
        assert runner._resolve_system_prompt(state, 5) == "prompt-5"

    def test_static_system_prompt_ignores_task_index(self):
        runner = TrajectoryRunner(
            environment=self._make_env(),
            backend=self._make_backend(),
            system_prompt="static-prompt",
        )
        state = State(
            observation=Observation(prompt="obs"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="t", is_terminal=False, info={}),
        )
        assert runner._resolve_system_prompt(state, 42) == "static-prompt"

    def test_heterogeneous_prompts_in_legacy_messages(self):
        prompts = {0: "Game A prompt", 1: "Game B prompt"}
        runner = TrajectoryRunner(
            environment=self._make_env(),
            backend=self._make_backend(),
            system_prompt_fn=lambda state, task_index: prompts.get(task_index, "default"),
        )
        state = State(
            observation=Observation(prompt="obs"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="t", is_terminal=False, info={}),
        )
        msgs_a = runner._build_messages(state, task_index=0)
        msgs_b = runner._build_messages(state, task_index=1)
        assert msgs_a[0].role == "system"
        assert msgs_a[0].content == "Game A prompt"
        assert msgs_b[0].role == "system"
        assert msgs_b[0].content == "Game B prompt"

    def test_none_system_prompt_fn_returns_none(self):
        runner = TrajectoryRunner(
            environment=self._make_env(),
            backend=self._make_backend(),
            system_prompt_fn=lambda state, task_index: None,
        )
        state = State(
            observation=Observation(prompt="obs"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="t", is_terminal=False, info={}),
        )
        msgs = runner._build_messages(state, task_index=0)
        assert msgs[0].role != "system"

    def test_run_multi_evaluation_threads_task_index_to_prompt_fn(self):
        from unittest.mock import MagicMock

        backend = MagicMock()
        captured_batches: list[list[ChatMessage]] = []

        def _generate_chat_batch(messages_batch, params):
            captured_batches.extend(messages_batch)
            return [
                GenerationResult(text="look", finish_reason=StopReason.END_OF_TEXT)
                for _ in messages_batch
            ]

        backend.generate_chat_batch.side_effect = _generate_chat_batch

        env = MagicMock()
        env.spec.max_steps = 1
        env.spec.name = "test-env"

        def _reset(*, options):
            task_index = options["task_index"]
            return (
                State(
                    observation=Observation(prompt=f"obs-{task_index}"),
                    hidden=None,
                    metadata=StateMetadata(
                        step=0,
                        episode_id=f"ep-{task_index}",
                        is_terminal=False,
                        info={"task_index": task_index},
                    ),
                ),
                {},
            )

        def _step(state, action):
            next_state = State(
                observation=Observation(prompt="done"),
                hidden=None,
                metadata=StateMetadata(
                    step=1,
                    episode_id=state.metadata.episode_id,
                    is_terminal=True,
                    info=state.metadata.info,
                ),
            )
            return MagicMock(
                next_state=next_state,
                rewards=SignalBundle(signals=()),
                extracted_action=None,
                resolved_action=None,
                info={},
                done=True,
            )

        env.reset.side_effect = _reset
        env.step.side_effect = _step

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            system_prompt_fn=lambda state, task_index: f"prompt-{task_index}",
        )

        run_multi_evaluation([MultiEvalEntry(runner=runner, task_indices=[7])])

        assert len(captured_batches) == 1
        assert captured_batches[0][0].role == "system"
        assert captured_batches[0][0].content == "prompt-7"

    def test_run_batch_from_states_threads_task_index_to_prompt_fn(self):
        from unittest.mock import MagicMock

        backend = MagicMock()
        captured_batches: list[list[ChatMessage]] = []

        def _generate_chat_batch(messages_batch, params):
            captured_batches.extend(messages_batch)
            return [
                GenerationResult(text="look", finish_reason=StopReason.END_OF_TEXT)
                for _ in messages_batch
            ]

        backend.generate_chat_batch.side_effect = _generate_chat_batch

        env = MagicMock()
        env.spec.max_steps = 1
        env.spec.pure_step = True

        def _step(state, action):
            del action
            return MagicMock(
                next_state=State(
                    observation=Observation(prompt="done"),
                    hidden=None,
                    metadata=StateMetadata(
                        step=1,
                        episode_id=state.metadata.episode_id,
                        is_terminal=True,
                        info={},
                    ),
                ),
                rewards=SignalBundle(signals=()),
                extracted_action=None,
                resolved_action=None,
                info={},
                done=True,
            )

        env.step.side_effect = _step

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            system_prompt_fn=lambda state, task_index: f"prompt-{task_index}",
        )

        states = [
            State(
                observation=Observation(prompt="obs-a"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id="ep-a",
                    is_terminal=False,
                    info={"task_index": 3},
                ),
            ),
            State(
                observation=Observation(prompt="obs-b"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id="ep-b",
                    is_terminal=False,
                    info={"task_index": 9},
                ),
            ),
        ]

        runner.run_batch_from_states(states, max_steps=1)

        assert len(captured_batches) == 2
        assert [prompt[0].content for prompt in captured_batches] == [
            "prompt-3",
            "prompt-9",
        ]

    def test_run_batch_from_state_actions_uses_original_task_index_for_prompt_fn(self):
        from unittest.mock import MagicMock

        backend = MagicMock()
        captured_batches: list[list[ChatMessage]] = []

        def _generate_chat_batch(messages_batch, params):
            captured_batches.extend(messages_batch)
            return [
                GenerationResult(text="look", finish_reason=StopReason.END_OF_TEXT)
                for _ in messages_batch
            ]

        backend.generate_chat_batch.side_effect = _generate_chat_batch

        env = MagicMock()
        env.spec.max_steps = 2
        env.spec.pure_step = True
        forced_call_count = [0]

        def _step(state, action):
            del action
            forced_call_count[0] += 1
            if forced_call_count[0] <= 2:
                return MagicMock(
                    next_state=State(
                        observation=Observation(prompt=f"mid-{forced_call_count[0]}"),
                        hidden=None,
                        metadata=StateMetadata(
                            step=1,
                            episode_id=state.metadata.episode_id,
                            is_terminal=False,
                            info={},
                        ),
                    ),
                    rewards=SignalBundle(signals=()),
                    extracted_action=None,
                    resolved_action=None,
                    info={},
                    done=False,
                )
            return MagicMock(
                next_state=State(
                    observation=Observation(prompt="done"),
                    hidden=None,
                    metadata=StateMetadata(
                        step=2,
                        episode_id=state.metadata.episode_id,
                        is_terminal=True,
                        info={},
                    ),
                ),
                rewards=SignalBundle(signals=()),
                extracted_action=None,
                resolved_action=None,
                info={},
                done=True,
            )

        env.step.side_effect = _step

        runner = TrajectoryRunner(
            environment=env,
            backend=backend,
            system_prompt_fn=lambda state, task_index: f"prompt-{task_index}",
        )

        states = [
            State(
                observation=Observation(prompt="obs-a"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id="ep-a",
                    is_terminal=False,
                    info={"task_index": 11},
                ),
            ),
            State(
                observation=Observation(prompt="obs-b"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id="ep-b",
                    is_terminal=False,
                    info={"task_index": 17},
                ),
            ),
        ]

        runner.run_batch_from_state_actions(
            states,
            [Action(text="force-a"), Action(text="force-b")],
            max_steps=2,
        )

        assert len(captured_batches) == 2
        assert [prompt[0].content for prompt in captured_batches] == [
            "prompt-11",
            "prompt-17",
        ]

    def test_multi_instance_run_batch_from_states_threads_task_index_to_prompt_fn(self):
        from unittest.mock import MagicMock

        backend = MagicMock()
        captured_batches: list[list[ChatMessage]] = []

        def _generate_chat_batch(messages_batch, params):
            captured_batches.extend(messages_batch)
            return [
                GenerationResult(text="look", finish_reason=StopReason.END_OF_TEXT)
                for _ in messages_batch
            ]

        backend.generate_chat_batch.side_effect = _generate_chat_batch

        primary_env = MagicMock()
        primary_env.spec.max_steps = 1
        primary_env.spec.pure_step = False

        created_envs: list[MagicMock] = []

        def _factory():
            env = MagicMock()
            env.close = MagicMock()

            def _step(state, action):
                del action
                return MagicMock(
                    next_state=State(
                        observation=Observation(prompt="done"),
                        hidden=None,
                        metadata=StateMetadata(
                            step=1,
                            episode_id=state.metadata.episode_id,
                            is_terminal=True,
                            info={},
                        ),
                    ),
                    rewards=SignalBundle(signals=()),
                    extracted_action=None,
                    resolved_action=None,
                    info={},
                    done=True,
                )

            env.step.side_effect = _step
            created_envs.append(env)
            return env

        def _restore(env, state):
            del env
            return State(
                observation=Observation(prompt=f"restored-{state.metadata.episode_id}"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id=state.metadata.episode_id,
                    is_terminal=False,
                    info={},
                ),
            )

        runner = TrajectoryRunner(
            environment=primary_env,
            backend=backend,
            system_prompt_fn=lambda state, task_index: f"prompt-{task_index}",
            env_factory=_factory,
            restore_fn=_restore,
        )

        states = [
            State(
                observation=Observation(prompt="obs-a"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id="ep-a",
                    is_terminal=False,
                    info={"task_index": 2},
                ),
            ),
            State(
                observation=Observation(prompt="obs-b"),
                hidden=None,
                metadata=StateMetadata(
                    step=0,
                    episode_id="ep-b",
                    is_terminal=False,
                    info={"task_index": 4},
                ),
            ),
        ]

        runner.run_batch_from_states(states, max_steps=1)

        assert len(captured_batches) == 2
        assert [prompt[0].content for prompt in captured_batches] == [
            "prompt-2",
            "prompt-4",
        ]
        assert len(created_envs) == 2


# =============================================================================
# _prior_history_from_messages unit tests
# =============================================================================


class TestPriorHistoryFromMessages:
    """Tests for the _prior_history_from_messages helper."""

    def _make_runner(self):
        from unittest.mock import MagicMock

        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_backend = MagicMock()

        return TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
        )

    def test_empty_messages(self):
        """Empty messages → ([], [])."""
        runner = self._make_runner()
        result = runner._prior_history_from_messages((), split_final_user=False)
        assert result == ([], [])

    def test_one_assistant_user_pair(self):
        """2 messages (1 assistant/user pair) → no prefix, 1 history entry."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "You see a door."},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is not None
        prefix_msgs, entries = result
        assert prefix_msgs == []
        assert len(entries) == 1
        assert entries[0].action_text == "go north"
        assert entries[0].observation_text == "You see a door."

    def test_three_assistant_user_pairs(self):
        """6 messages (3 pairs) → 3 history entries with monotonic steps."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "o2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "o3"},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is not None
        prefix_msgs, entries = result
        assert prefix_msgs == []
        assert len(entries) == 3
        assert entries[0].action_text == "a1"
        assert entries[0].observation_text == "o1"
        assert entries[2].action_text == "a3"
        assert entries[2].observation_text == "o3"
        # Monotonic step numbers starting at 1
        assert entries[0].step == 1
        assert entries[1].step == 2
        assert entries[2].step == 3

    def test_trailing_assistant(self):
        """Trailing assistant → final entry with empty observation."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is not None
        _, entries = result
        assert len(entries) == 2
        assert entries[1].action_text == "a2"
        assert entries[1].observation_text == ""

    def test_tool_calls_returns_none(self):
        """Messages with tool calls → None."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "using tool", "tool_calls": [{"id": "1"}]},
            {"role": "user", "content": "tool result"},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is None

    def test_unexpected_role_returns_none(self):
        """Unexpected role (e.g. 'tool') → None."""
        runner = self._make_runner()
        msgs = (
            {"role": "tool", "content": "tool result"},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is None

    def test_leading_user_message_craftax(self):
        """Leading user message (craftax pattern) → in prefix_messages."""
        runner = self._make_runner()
        msgs = (
            {"role": "user", "content": "Initial game observation"},
            {"role": "assistant", "content": "look around"},
            {"role": "user", "content": "You see a tree."},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is not None
        prefix_msgs, entries = result
        assert len(prefix_msgs) == 1
        assert prefix_msgs[0].role == "user"
        assert prefix_msgs[0].content == "Initial game observation"
        assert len(entries) == 1
        assert entries[0].action_text == "look around"
        assert entries[0].observation_text == "You see a tree."

    def test_split_final_user_true(self):
        """split_final_user=True: final user excluded, assistant becomes empty-obs entry."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "current obs"},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=True)
        assert result is not None
        _, entries = result
        assert len(entries) == 2
        assert entries[0].action_text == "a1"
        assert entries[0].observation_text == "o1"
        # Second entry: a2 with empty obs (the final user was split off)
        assert entries[1].action_text == "a2"
        assert entries[1].observation_text == ""

    def test_split_final_user_noop_when_trailing_assistant(self):
        """split_final_user=True: no effect when messages end with assistant."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"},
        )
        result_split = runner._prior_history_from_messages(msgs, split_final_user=True)
        result_nosplit = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result_split is not None
        assert result_nosplit is not None
        _, entries_split = result_split
        _, entries_nosplit = result_nosplit
        assert len(entries_split) == len(entries_nosplit)
        assert entries_split[1].observation_text == entries_nosplit[1].observation_text

    def test_messages_with_images(self):
        """User messages with images → entries with populated observation_images."""
        runner = self._make_runner()
        msgs = (
            {"role": "assistant", "content": "look"},
            {"role": "user", "content": "You see this.", "images": [
                {"data": "imgdata", "media_type": "image/png"},
            ]},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is not None
        _, entries = result
        assert len(entries) == 1
        assert len(entries[0].observation_images) == 1
        assert entries[0].observation_images[0].data == "imgdata"

    def test_multiple_leading_user_messages(self):
        """Multiple leading user messages before first assistant."""
        runner = self._make_runner()
        msgs = (
            {"role": "user", "content": "System context"},
            {"role": "user", "content": "Game rules"},
            {"role": "assistant", "content": "play"},
            {"role": "user", "content": "Result"},
        )
        result = runner._prior_history_from_messages(msgs, split_final_user=False)
        assert result is not None
        prefix_msgs, entries = result
        assert len(prefix_msgs) == 2
        assert prefix_msgs[0].content == "System context"
        assert prefix_msgs[1].content == "Game rules"
        assert len(entries) == 1
        assert entries[0].action_text == "play"


# =============================================================================
# Structured messages with prior history reconstruction
# =============================================================================


class TestStructuredMessagesWithPriorHistory:
    """Integration tests for _build_structured_messages with mid-episode state restoration."""

    def _make_runner(self, system_prompt=None, history_fn=None, format_reminder=None,
                     prompt_budget=None, max_steps=10):
        from unittest.mock import MagicMock

        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = max_steps
        mock_backend = MagicMock()

        kwargs = dict(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
            system_prompt=system_prompt,
        )
        if history_fn is not None:
            kwargs["history_fn"] = history_fn
        if format_reminder is not None:
            kwargs["format_reminder"] = format_reminder
        if prompt_budget is not None:
            kwargs["prompt_budget"] = prompt_budget
        return TrajectoryRunner(**kwargs)

    def _mid_episode_state(self, step=3, task_text="Task.", state_text="Current obs.",
                           messages=()):
        """Create a mid-episode state with accumulated messages."""
        task = ObservationContent(text=task_text)
        state_content = ObservationContent(text=state_text)
        return _make_state(
            prompt=task_text,
            task=task,
            state=state_content,
            step=step,
            messages=messages,
        )

    # -- Zero transitions (run_batch_from_states shape) --

    def test_zero_transitions_prior_history_present(self):
        """0 transitions from mid-episode state → prior history present, current obs once."""
        prior_messages = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "Room 2"},
            {"role": "assistant", "content": "look"},
            {"role": "user", "content": "Current obs."},
        )
        restored_state = self._mid_episode_state(step=2, messages=prior_messages)
        trajectory = Trajectory.create(restored_state)

        runner = self._make_runner()
        messages = runner._build_messages(restored_state, trajectory=trajectory)

        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0].content == "go north"
        assert assistant_msgs[1].content == "look"
        # Current observation appears exactly once
        full_text = "\n".join(m.content for m in messages)
        assert full_text.count("Current obs.") == 1

    def test_zero_transitions_no_history_drops_prior(self):
        """0 transitions + no_history → current obs still appears, prior history dropped."""
        prior_messages = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "Room 2"},
            {"role": "assistant", "content": "look"},
            {"role": "user", "content": "Current obs."},
        )
        restored_state = self._mid_episode_state(step=2, messages=prior_messages)
        trajectory = Trajectory.create(restored_state)

        runner = self._make_runner(history_fn=no_history)
        messages = runner._build_messages(restored_state, trajectory=trajectory)

        full_text = "\n".join(m.content for m in messages)
        assert "go north" not in full_text
        assert "Room 2" not in full_text
        assert "Current obs." in full_text

    def test_zero_transitions_prompt_budget_preserves_current(self):
        """0 transitions + prompt_budget → current obs survives even with tight budget."""
        prior_messages = (
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "Current obs."},
        )
        restored_state = self._mid_episode_state(step=2, messages=prior_messages)
        trajectory = Trajectory.create(restored_state)

        from llenvs.evaluation.history import PromptBudget

        budget = PromptBudget(
            max_prompt_tokens=50,  # very tight
            estimate_tokens=lambda s: len(s) // 4,
            build_history=lambda entries, available: [],  # drop all history
        )
        runner = self._make_runner(prompt_budget=budget)
        messages = runner._build_messages(restored_state, trajectory=trajectory)

        full_text = "\n".join(m.content for m in messages)
        assert "Current obs." in full_text

    def test_zero_transitions_format_reminder_on_current(self):
        """0 transitions + format_reminder → reminder on current obs, not on task."""
        prior_messages = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "Room 2"},
            {"role": "assistant", "content": "look"},
            {"role": "user", "content": "Current obs."},
        )
        restored_state = self._mid_episode_state(step=2, messages=prior_messages)
        trajectory = Trajectory.create(restored_state)

        runner = self._make_runner(format_reminder="REMINDER: use tags")
        messages = runner._build_messages(restored_state, trajectory=trajectory)

        user_msgs = [m for m in messages if m.role == "user"]
        last_user = user_msgs[-1].content
        assert "REMINDER: use tags" in last_user
        assert "Current obs." in last_user
        assert "REMINDER: use tags" not in messages[0].content or messages[0].role == "system"

    # -- One transition (run_batch_from_state_actions shape) --

    def test_one_transition_prior_history(self):
        """1 transition from mid-episode: prior history + new transition present."""
        prior_messages = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "Room 2"},
        )
        restored_state = self._mid_episode_state(
            step=1, state_text="Room 2", messages=prior_messages,
        )

        task = ObservationContent(text="Task.")
        new_state = _make_state(
            prompt="Task.", task=task,
            state=ObservationContent(text="Room 3"), step=2,
        )
        trajectory = Trajectory.create(restored_state)
        trajectory.add_transition(
            Transition(
                state=restored_state,
                action=Action(text="go east"),
                next_state=new_state,
                rewards=SignalBundle(signals=()),
            )
        )

        runner = self._make_runner()
        messages = runner._build_messages(new_state, trajectory=trajectory)

        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0].content == "go north"
        assert assistant_msgs[1].content == "go east"
        full_text = "\n".join(m.content for m in messages)
        assert "Room 3" in full_text

    # -- Many new transitions --

    def test_many_transitions_prior_history_persists(self):
        """Start from step 3, add 5 transitions → prior history still present."""
        prior_messages = (
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "o0"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "o2"},
        )
        restored_state = self._mid_episode_state(
            step=3, state_text="o2", messages=prior_messages,
        )

        task = ObservationContent(text="Task.")
        trajectory = Trajectory.create(restored_state)
        prev_state = restored_state
        for i in range(5):
            next_state = _make_state(
                prompt="Task.", task=task,
                state=ObservationContent(text=f"new_obs_{i}"),
                step=4 + i,
            )
            trajectory.add_transition(
                Transition(
                    state=prev_state,
                    action=Action(text=f"new_action_{i}"),
                    next_state=next_state,
                    rewards=SignalBundle(signals=()),
                )
            )
            prev_state = next_state

        runner = self._make_runner()
        messages = runner._build_messages(prev_state, trajectory=trajectory)

        full_text = "\n".join(m.content for m in messages)
        assert "a0" in full_text
        assert "a1" in full_text
        assert "a2" in full_text
        for i in range(5):
            assert f"new_action_{i}" in full_text
        assert "new_obs_4" in full_text

    # -- Edge cases --

    def test_step0_no_reconstruction(self):
        """Step 0 state → no prior reconstruction (normal behavior)."""
        state = _make_state(
            prompt="Task.",
            task=ObservationContent(text="Task."),
            state=ObservationContent(text="Initial."),
            step=0,
            messages=(
                {"role": "assistant", "content": "should not appear"},
                {"role": "user", "content": "ignored"},
            ),
        )
        trajectory = Trajectory.create(state)

        runner = self._make_runner()
        messages = runner._build_messages(state, trajectory=trajectory)

        full_text = "\n".join(m.content for m in messages)
        assert "should not appear" not in full_text
        assert "Initial." in full_text

    def test_mid_episode_empty_messages_no_reconstruction(self):
        """State at step 5 with empty messages → no prior entries."""
        state = self._mid_episode_state(step=5, messages=())
        trajectory = Trajectory.create(state)

        runner = self._make_runner()
        messages = runner._build_messages(state, trajectory=trajectory)

        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) >= 1
        full_text = "\n".join(m.content for m in messages)
        assert "Current obs." in full_text

    def test_last_n_history_with_reconstruction(self):
        """last_n_history(2) → only last 2 from combined prior + suffix history."""
        from llenvs.evaluation.history import last_n_history

        prior_messages = (
            {"role": "assistant", "content": "old_action"},
            {"role": "user", "content": "old_obs"},
            {"role": "assistant", "content": "recent_action"},
            {"role": "user", "content": "recent_obs"},
        )
        restored_state = self._mid_episode_state(
            step=2, state_text="recent_obs", messages=prior_messages,
        )

        task = ObservationContent(text="Task.")
        new_state = _make_state(
            prompt="Task.", task=task,
            state=ObservationContent(text="newest"), step=3,
        )
        trajectory = Trajectory.create(restored_state)
        trajectory.add_transition(
            Transition(
                state=restored_state,
                action=Action(text="new_action"),
                next_state=new_state,
                rewards=SignalBundle(signals=()),
            )
        )

        runner = self._make_runner(history_fn=last_n_history(2))
        messages = runner._build_messages(new_state, trajectory=trajectory)

        full_text = "\n".join(m.content for m in messages)
        assert "recent_action" in full_text
        assert "new_action" in full_text
        assert "old_action" not in full_text

    def test_craftax_leading_user_message_preserved(self):
        """Craftax-style leading user message → preserved via reconstructed_prefix_msgs."""
        prior_messages = (
            {"role": "user", "content": "Initial game state from reset"},
            {"role": "assistant", "content": "explore"},
            {"role": "user", "content": "You found a tree."},
            {"role": "assistant", "content": "chop tree"},
            {"role": "user", "content": "Current obs."},
        )
        restored_state = self._mid_episode_state(
            step=2, state_text="Current obs.", messages=prior_messages,
        )
        trajectory = Trajectory.create(restored_state)

        runner = self._make_runner()
        messages = runner._build_messages(restored_state, trajectory=trajectory)

        full_text = "\n".join(m.content for m in messages)
        assert "Initial game state from reset" in full_text
        assert "explore" in full_text
        assert "You found a tree." in full_text

    # -- Runner path tests --

    def test_run_batch_from_states_includes_prior_history(self):
        """run_batch_from_states from mid-episode → prompts include prior history."""
        from unittest.mock import MagicMock

        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_env.spec.name = "test-env"

        prior_messages = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "Room 2"},
        )
        task = ObservationContent(text="Task.")
        state_content = ObservationContent(text="Room 2")
        restored_state = State(
            observation=Observation(
                prompt="Task.",
                messages=prior_messages,
                task=task,
                state=state_content,
            ),
            hidden=None,
            metadata=StateMetadata(step=1, episode_id="test", is_terminal=False),
        )

        captured_prompts = []

        def _gen_batch(msgs_batch, params):
            captured_prompts.extend(msgs_batch)
            return [
                GenerationResult(text="done", finish_reason=StopReason.END_OF_TEXT)
                for _ in msgs_batch
            ]

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = _gen_batch

        terminal = State(
            observation=Observation(
                prompt="Task.",
                task=task,
                state=ObservationContent(text="Done."),
            ),
            hidden=None,
            metadata=StateMetadata(step=2, episode_id="test", is_terminal=True),
        )
        mock_env.step.return_value = MagicMock(
            next_state=terminal,
            rewards=SignalBundle(signals=()),
            extracted_action=None,
            resolved_action=None,
            info={},
            done=True,
        )

        runner = TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
        )
        runner.run_batch_from_states([restored_state], max_steps=1)

        assert len(captured_prompts) >= 1
        first_prompt = captured_prompts[0]
        prompt_text = "\n".join(m.content for m in first_prompt)
        assert "go north" in prompt_text

    def test_run_batch_from_state_actions_includes_prior_history(self):
        """run_batch_from_state_actions from mid-episode → prior history in prompts."""
        from unittest.mock import MagicMock

        from llenvs.inference.protocol import SamplingParams

        mock_env = MagicMock()
        mock_env.spec.max_steps = 10
        mock_env.spec.name = "test-env"

        prior_messages = (
            {"role": "assistant", "content": "go north"},
            {"role": "user", "content": "Room 2"},
        )
        task = ObservationContent(text="Task.")
        state_content = ObservationContent(text="Room 2")
        restored_state = State(
            observation=Observation(
                prompt="Task.",
                messages=prior_messages,
                task=task,
                state=state_content,
            ),
            hidden=None,
            metadata=StateMetadata(step=1, episode_id="test", is_terminal=False),
        )

        after_action = State(
            observation=Observation(
                prompt="Task.",
                task=task,
                state=ObservationContent(text="Room 3"),
            ),
            hidden=None,
            metadata=StateMetadata(step=2, episode_id="test", is_terminal=False),
        )
        terminal = State(
            observation=Observation(
                prompt="Task.",
                task=task,
                state=ObservationContent(text="Done."),
            ),
            hidden=None,
            metadata=StateMetadata(step=3, episode_id="test", is_terminal=True),
        )

        step_results = [
            MagicMock(
                next_state=after_action,
                rewards=SignalBundle(signals=()),
                extracted_action=None,
                resolved_action=None,
                info={},
                done=False,
            ),
            MagicMock(
                next_state=terminal,
                rewards=SignalBundle(signals=()),
                extracted_action=None,
                resolved_action=None,
                info={},
                done=True,
            ),
        ]
        mock_env.step.side_effect = step_results

        captured_prompts = []

        def _gen_batch(msgs_batch, params):
            captured_prompts.extend(msgs_batch)
            return [
                GenerationResult(text="done", finish_reason=StopReason.END_OF_TEXT)
                for _ in msgs_batch
            ]

        mock_backend = MagicMock()
        mock_backend.generate_chat_batch.side_effect = _gen_batch

        runner = TrajectoryRunner(
            environment=mock_env,
            backend=mock_backend,
            sampling_params=SamplingParams(),
        )
        runner.run_batch_from_state_actions(
            [restored_state],
            [Action(text="go east")],
            max_steps=2,
        )

        assert len(captured_prompts) >= 1
        prompt_text = "\n".join(m.content for m in captured_prompts[0])
        assert "go north" in prompt_text
