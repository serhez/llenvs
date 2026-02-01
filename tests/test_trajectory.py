"""Tests for trajectory tracking with checkpointing and branching."""

import pytest
from llenvs.core.state import State, StateMetadata, TextObservation, TextAction
from llenvs.core.reward import RewardBundle, RewardSignal, RewardType
from llenvs.core.trajectory import Trajectory, Transition, Checkpoint


class TestTransition:
    """Tests for Transition."""

    def test_creation(self, sample_state, sample_action, terminal_state, sample_reward_bundle):
        """Test basic transition creation."""
        transition = Transition(
            state=sample_state,
            action=sample_action,
            next_state=terminal_state,
            rewards=sample_reward_bundle,
            info={"step_info": "test"},
        )

        assert transition.state == sample_state
        assert transition.action == sample_action
        assert transition.next_state == terminal_state
        assert transition.rewards == sample_reward_bundle
        assert transition.info["step_info"] == "test"

    def test_immutability(self, sample_transition):
        """Test that transition is frozen."""
        with pytest.raises(AttributeError):
            sample_transition.state = None  # type: ignore


class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_creation(self, sample_state):
        """Test basic checkpoint creation."""
        checkpoint = Checkpoint(
            name="test_checkpoint",
            trajectory_id="traj-001",
            step_index=5,
            state=sample_state,
        )

        assert checkpoint.name == "test_checkpoint"
        assert checkpoint.trajectory_id == "traj-001"
        assert checkpoint.step_index == 5
        assert checkpoint.state == sample_state

    def test_immutability(self, sample_state):
        """Test that checkpoint is frozen."""
        checkpoint = Checkpoint(
            name="test",
            trajectory_id="traj",
            step_index=0,
            state=sample_state,
        )
        with pytest.raises(AttributeError):
            checkpoint.name = "changed"  # type: ignore


class TestTrajectory:
    """Tests for Trajectory."""

    def test_create(self, sample_state):
        """Test trajectory creation."""
        trajectory = Trajectory.create(sample_state)

        assert trajectory.initial_state == sample_state
        assert trajectory.episode_id == sample_state.metadata.episode_id
        assert len(trajectory) == 0
        assert trajectory.current_state == sample_state

    def test_add_transition(self, sample_trajectory, sample_transition):
        """Test adding transitions."""
        sample_trajectory.add_transition(sample_transition)

        assert len(sample_trajectory) == 1
        assert sample_trajectory.transitions[0] == sample_transition
        assert sample_trajectory.current_state == sample_transition.next_state

    def test_multiple_transitions(self, sample_state):
        """Test adding multiple transitions."""
        trajectory = Trajectory.create(sample_state)

        # Create multiple transitions
        for i in range(3):
            current = trajectory.current_state
            next_state = current.with_metadata(step=i + 1)
            transition = Transition(
                state=current,
                action=TextAction(text=f"action_{i}"),
                next_state=next_state,
                rewards=RewardBundle.single(value=0.5),
                info={},
            )
            trajectory.add_transition(transition)

        assert len(trajectory) == 3
        assert trajectory.current_state.metadata.step == 3

    def test_state_at(self, sample_state):
        """Test accessing states by index."""
        trajectory = Trajectory.create(sample_state)

        # Add some transitions
        states = [sample_state]
        for i in range(3):
            current = trajectory.current_state
            next_state = current.with_metadata(step=i + 1)
            states.append(next_state)
            transition = Transition(
                state=current,
                action=TextAction(text="action"),
                next_state=next_state,
                rewards=RewardBundle.empty(),
                info={},
            )
            trajectory.add_transition(transition)

        # Check state_at
        assert trajectory.state_at(0) == states[0]  # Initial
        assert trajectory.state_at(1) == states[1]  # After first action
        assert trajectory.state_at(3) == states[3]  # After third action

    def test_state_at_bounds(self, sample_trajectory):
        """Test state_at with invalid indices."""
        with pytest.raises(IndexError):
            sample_trajectory.state_at(-1)

        with pytest.raises(IndexError):
            sample_trajectory.state_at(100)

    def test_total_reward(self, sample_state):
        """Test total reward computation."""
        trajectory = Trajectory.create(sample_state)

        rewards = [0.5, 1.0, 0.25]
        for i, r in enumerate(rewards):
            current = trajectory.current_state
            next_state = current.with_metadata(step=i + 1)
            transition = Transition(
                state=current,
                action=TextAction(text="action"),
                next_state=next_state,
                rewards=RewardBundle.single(value=r),
                info={},
            )
            trajectory.add_transition(transition)

        assert trajectory.total_reward == pytest.approx(1.75)

    def test_is_terminal(self, sample_state):
        """Test terminal state detection."""
        trajectory = Trajectory.create(sample_state)
        assert not trajectory.is_terminal

        # Add non-terminal transition
        next_state = sample_state.with_metadata(step=1, is_terminal=False)
        trajectory.add_transition(
            Transition(
                state=sample_state,
                action=TextAction(text="action"),
                next_state=next_state,
                rewards=RewardBundle.empty(),
                info={},
            )
        )
        assert not trajectory.is_terminal

        # Add terminal transition
        terminal = next_state.with_metadata(step=2, is_terminal=True)
        trajectory.add_transition(
            Transition(
                state=next_state,
                action=TextAction(text="final"),
                next_state=terminal,
                rewards=RewardBundle.empty(),
                info={},
            )
        )
        assert trajectory.is_terminal

    def test_transitions_immutable(self, sample_trajectory, sample_transition):
        """Test that transitions property returns immutable tuple."""
        sample_trajectory.add_transition(sample_transition)

        transitions = sample_trajectory.transitions
        assert isinstance(transitions, tuple)

    def test_checkpoint_basic(self, sample_trajectory, sample_transition):
        """Test basic checkpointing."""
        sample_trajectory.add_transition(sample_transition)

        checkpoint = sample_trajectory.checkpoint("checkpoint_1")

        assert checkpoint.name == "checkpoint_1"
        assert checkpoint.trajectory_id == sample_trajectory.episode_id
        assert checkpoint.step_index == 1
        assert checkpoint.state == sample_trajectory.current_state

    def test_checkpoint_duplicate_name(self, sample_trajectory):
        """Test that duplicate checkpoint names raise error."""
        sample_trajectory.checkpoint("my_checkpoint")

        with pytest.raises(ValueError, match="already exists"):
            sample_trajectory.checkpoint("my_checkpoint")

    def test_branch_basic(self, sample_state):
        """Test basic branching."""
        trajectory = Trajectory.create(sample_state)

        # Add some transitions
        for i in range(3):
            current = trajectory.current_state
            next_state = current.with_metadata(step=i + 1)
            trajectory.add_transition(
                Transition(
                    state=current,
                    action=TextAction(text=f"action_{i}"),
                    next_state=next_state,
                    rewards=RewardBundle.single(value=1.0),
                    info={},
                )
            )

        # Checkpoint after step 2
        trajectory.checkpoint("branch_point")

        # Add more to original
        trajectory.add_transition(
            Transition(
                state=trajectory.current_state,
                action=TextAction(text="original_path"),
                next_state=trajectory.current_state.with_metadata(step=4),
                rewards=RewardBundle.single(value=0.5),
                info={},
            )
        )

        # Branch from checkpoint
        branched = trajectory.branch("branch_point")

        # Branched has history up to checkpoint
        assert len(branched) == 3  # Steps 0-2
        assert branched.current_state.metadata.step == 3

        # Original continues
        assert len(trajectory) == 4
        assert trajectory.current_state.metadata.step == 4

        # Branched has different episode_id
        assert branched.episode_id != trajectory.episode_id

    def test_branch_independent(self, sample_state):
        """Test that branched trajectories are independent."""
        trajectory = Trajectory.create(sample_state)

        # Setup
        next_state = sample_state.with_metadata(step=1)
        trajectory.add_transition(
            Transition(
                state=sample_state,
                action=TextAction(text="step1"),
                next_state=next_state,
                rewards=RewardBundle.single(value=1.0),
                info={},
            )
        )
        trajectory.checkpoint("cp")

        # Branch
        branched = trajectory.branch("cp")

        # Add different actions to each
        trajectory.add_transition(
            Transition(
                state=trajectory.current_state,
                action=TextAction(text="original_action"),
                next_state=trajectory.current_state.with_metadata(step=2),
                rewards=RewardBundle.single(value=0.0),
                info={},
            )
        )

        branched.add_transition(
            Transition(
                state=branched.current_state,
                action=TextAction(text="branched_action"),
                next_state=branched.current_state.with_metadata(step=2),
                rewards=RewardBundle.single(value=1.0),
                info={},
            )
        )

        # Verify independence
        assert len(trajectory) == 2
        assert len(branched) == 2
        assert trajectory.transitions[-1].action.text == "original_action"
        assert branched.transitions[-1].action.text == "branched_action"
        assert trajectory.total_reward == pytest.approx(1.0)
        assert branched.total_reward == pytest.approx(2.0)

    def test_branch_nonexistent_checkpoint(self, sample_trajectory):
        """Test branching from non-existent checkpoint."""
        with pytest.raises(KeyError, match="not found"):
            sample_trajectory.branch("nonexistent")

    def test_checkpoints_property(self, sample_trajectory):
        """Test checkpoints property."""
        sample_trajectory.checkpoint("cp1")
        sample_trajectory.checkpoint("cp2")

        checkpoints = sample_trajectory.checkpoints
        assert "cp1" in checkpoints
        assert "cp2" in checkpoints
        assert len(checkpoints) == 2
