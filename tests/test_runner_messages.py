"""Tests for runner message building: _coalesce_messages and structured mode."""

from __future__ import annotations

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
from llenvs.evaluation.runner import _coalesce_messages
from llenvs.inference.protocol import ChatMessage

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
        state_content = ObservationContent(text="[Step 0]\nYou see a door.")
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
        assert "Step 0" in messages[0].content

    def test_structured_mode_with_transitions(self):
        """Structured mode reconstructs history from trajectory transitions."""
        runner = self._make_runner()

        task = ObservationContent(text="Navigate the maze.")
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nRoom 2")

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
        # system(0) + user(task) + assistant(go north) + user(state1) -> coalesced
        assert len(messages) == 3
        assert messages[0].role == "user"  # task (coalesced with nothing since next is assistant)
        assert messages[1].role == "assistant"
        assert messages[1].content == "go north"
        assert messages[2].role == "user"
        assert "Step 1" in messages[2].content

    def test_structured_mode_with_system_prompt(self):
        """System prompt is included in structured mode."""
        runner = self._make_runner(system_prompt="You are helpful.")

        task = ObservationContent(text="Task here.")
        state_content = ObservationContent(text="[Step 0]\nObs.")
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
        state0 = ObservationContent(text="[Step 0]\nYou are in a dark room.")

        initial_state = _make_state(prompt="Explore the dungeon.", task=task, state=state0)
        trajectory = Trajectory.create(initial_state)

        messages = runner._build_messages(initial_state, trajectory=trajectory)
        # Task + state0 coalesced into one user message
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Explore the dungeon." in messages[0].content
        assert "Step 0" in messages[0].content


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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nRoom 2")
        state2 = ObservationContent(text="[Step 2]\nRoom 3")

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
        # task + (assistant + user) * 2 transitions = 5, coalesced task+step0
        # task(user) + go_north(assistant) + step1(user) + go_east(assistant) + step2(user)
        assert len(messages) == 5
        assert messages[0].role == "user"  # task (step0 coalesced)
        assert messages[1].role == "assistant"
        assert messages[1].content == "go north"
        assert messages[2].role == "user"
        assert "Step 1" in messages[2].content
        assert messages[3].role == "assistant"
        assert messages[3].content == "go east"
        assert messages[4].role == "user"
        assert "Step 2" in messages[4].content

    def test_no_history_fn(self):
        """no_history drops all prior turns, showing only task + current state."""
        runner = self._make_runner(history_fn=no_history)
        trajectory, current_state = self._build_trajectory_with_transitions()

        messages = runner._build_messages(current_state, trajectory=trajectory)
        # task(user) + current_state(user) = coalesced into 1
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Navigate the maze." in messages[0].content
        assert "Step 2" in messages[0].content

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
        # task(user) + go_east(assistant) + step1(user from history_fn) + step2(user) = 4
        # But step1 and step2 are both user, so coalesced to 3
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert "Navigate the maze." in messages[0].content
        assert messages[1].role == "assistant"
        assert messages[1].content == "go east"
        assert messages[2].role == "user"
        assert "Step 2" in messages[2].content

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
        assert "Step 2" in messages[1].content

    def test_history_fn_not_used_in_legacy_mode(self):
        """history_fn is ignored when task is None (legacy mode)."""
        runner = self._make_runner(history_fn=no_history)
        state = _make_state(prompt="What is 2+2?")
        trajectory = Trajectory.create(state)

        messages = runner._build_messages(state, trajectory=trajectory)
        assert len(messages) == 1
        assert messages[0].content == "What is 2+2?"


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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
        state0 = ObservationContent(text="[Step 0]\nStart")
        state1 = ObservationContent(text="[Step 1]\nNext")

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
