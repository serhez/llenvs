"""Tests for the action fields on StepResult and Transition.

Fields: extracted_action, resolved_action.
"""

import json

import pytest

from llenvs.core.environment import StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, ImageContent, Observation, ObservationContent, State, StateMetadata
from llenvs.core.trajectory import Trajectory, Transition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(step: int = 0, prompt: str = "test", **kwargs):
    return State(
        observation=Observation(prompt=prompt),
        hidden=None,
        metadata=StateMetadata(step=step, episode_id="ep", **kwargs),
    )


def _empty_bundle():
    return SignalBundle(signals=())


# ===========================================================================
# Core dataclasses — all three fields
# ===========================================================================


class TestStepResultActionFields:
    """extracted_action, resolved_action on StepResult."""

    def test_all_default_none(self):
        sr = StepResult(
            next_state=_make_state(),
            rewards=_empty_bundle(),
        )
        assert sr.extracted_action is None
        assert sr.resolved_action is None

    def test_explicit_values(self):
        sr = StepResult(
            next_state=_make_state(),
            rewards=_empty_bundle(),
            extracted_action="0",
            resolved_action="left",
        )
        assert sr.extracted_action == "0"
        assert sr.resolved_action == "left"

    def test_frozen(self):
        sr = StepResult(
            next_state=_make_state(),
            rewards=_empty_bundle(),
            extracted_action="y",
            resolved_action="z",
        )
        with pytest.raises(AttributeError):
            sr.extracted_action = "b"  # type: ignore
        with pytest.raises(AttributeError):
            sr.resolved_action = "c"  # type: ignore

    def test_extraction_failure(self):
        """Extraction failure: both fields None."""
        sr = StepResult(
            next_state=_make_state(),
            rewards=_empty_bundle(),
        )
        assert sr.extracted_action is None
        assert sr.resolved_action is None

    def test_mapping_failure(self):
        """Mapping failure: extracted set, resolved None."""
        sr = StepResult(
            next_state=_make_state(),
            rewards=_empty_bundle(),
            extracted_action="999",
        )
        assert sr.extracted_action == "999"
        assert sr.resolved_action is None


class TestTransitionActionFields:
    """extracted_action, resolved_action on Transition."""

    def test_all_default_none(self):
        t = Transition(
            state=_make_state(0),
            action=Action(text="a"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
        )
        assert t.extracted_action is None
        assert t.resolved_action is None

    def test_explicit_values(self):
        t = Transition(
            state=_make_state(0),
            action=Action(text="a"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            extracted_action="extracted",
            resolved_action="resolved",
        )
        assert t.extracted_action == "extracted"
        assert t.resolved_action == "resolved"

    def test_frozen(self):
        t = Transition(
            state=_make_state(0),
            action=Action(text="a"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            extracted_action="y",
            resolved_action="z",
        )
        with pytest.raises(AttributeError):
            t.extracted_action = "b"  # type: ignore


# ===========================================================================
# strip_images preserves all three fields
# ===========================================================================


class TestStripImagesPreservesActionFields:
    def test_preserves_all_fields(self):
        state = State(
            observation=Observation(
                prompt="start",
                state=ObservationContent(text="", images=(ImageContent(data="img"),)),
            ),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="ep"),
        )
        traj = Trajectory.create(state)

        t = Transition(
            state=state,
            action=Action(text="full response"),
            next_state=State(
                observation=Observation(
                    prompt="next",
                    state=ObservationContent(text="", images=(ImageContent(data="img2"),)),
                ),
                hidden=None,
                metadata=StateMetadata(step=1, episode_id="ep"),
            ),
            rewards=_empty_bundle(),
            extracted_action="extracted",
            resolved_action="resolved",
        )
        traj.add_transition(t)

        stripped = traj.strip_images()
        st = stripped.transitions[0]
        assert st.extracted_action == "extracted"
        assert st.resolved_action == "resolved"
        # Images should be removed
        assert st.state.observation.get_images().all == ()

    def test_none_fields_preserved(self):
        state = _make_state(0)
        traj = Trajectory.create(state)
        t = Transition(
            state=state,
            action=Action(text="a"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
        )
        traj.add_transition(t)

        stripped = traj.strip_images()
        st = stripped.transitions[0]
        assert st.extracted_action is None
        assert st.resolved_action is None


# ===========================================================================
# Container serialization
# ===========================================================================


class TestSerializationActionFields:
    def test_round_trip_all_fields(self):
        from llenvs.container.serialization import (
            deserialize_step_result,
            serialize_step_result,
        )

        sr = StepResult(
            next_state=_make_state(1, is_terminal=True),
            rewards=SignalBundle.single(1.0, "correct", RewardType.OUTCOME),
            terminated=True,
            extracted_action="42",
            resolved_action="42",
            info={"extracted_answer": "42"},
        )
        data = serialize_step_result(sr)
        assert data["extracted_action"] == "42"
        assert data["resolved_action"] == "42"

        restored = deserialize_step_result(data)
        assert restored.extracted_action == "42"
        assert restored.resolved_action == "42"

    def test_round_trip_none_fields(self):
        from llenvs.container.serialization import (
            deserialize_step_result,
            serialize_step_result,
        )

        sr = StepResult(
            next_state=_make_state(1),
            rewards=_empty_bundle(),
        )
        data = serialize_step_result(sr)
        assert "extracted_action" not in data
        assert "resolved_action" not in data

        restored = deserialize_step_result(data)
        assert restored.extracted_action is None
        assert restored.resolved_action is None

    def test_round_trip_partial_fields(self):
        """Mapping failure: extracted set, resolved None."""
        from llenvs.container.serialization import (
            deserialize_step_result,
            serialize_step_result,
        )

        sr = StepResult(
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            extracted_action="bad",
        )
        data = serialize_step_result(sr)
        assert data["extracted_action"] == "bad"
        assert "resolved_action" not in data

        restored = deserialize_step_result(data)
        assert restored.extracted_action == "bad"
        assert restored.resolved_action is None

    def test_json_round_trip(self):
        from llenvs.container.serialization import (
            deserialize_step_result,
            serialize_step_result,
        )

        sr = StepResult(
            next_state=_make_state(1),
            rewards=SignalBundle.single(0.5),
            extracted_action="answer",
            resolved_action="answer",
        )
        data = serialize_step_result(sr)
        json_str = json.dumps(data)
        restored = deserialize_step_result(json.loads(json_str))
        assert restored.extracted_action == "answer"
        assert restored.resolved_action == "answer"


# ===========================================================================
# Adapter tests: all three fields set correctly
# ===========================================================================


class TestReasoningGymActionFields:
    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.reasoning_gym")

    def test_step_sets_all_fields(self, import_adapter):
        mod = import_adapter
        from unittest.mock import MagicMock

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__getitem__ = MagicMock(
            return_value={"question": "What is 1+1?", "answer": "2", "metadata": {}}
        )
        mock_dataset.score_answer = MagicMock(return_value=1.0)
        mock_dataset.dataset_name = "test"

        from llenvs.core.extraction import BoxedExtractor

        env = mod.ReasoningGymEnvironment(
            dataset=mock_dataset,
            answer_extractor=BoxedExtractor(),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="The answer is \\boxed{2}"))
        assert result.extracted_action == "2"
        assert result.resolved_action == "2"

    def test_no_match_all_fields(self, import_adapter):
        mod = import_adapter
        from unittest.mock import MagicMock

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__getitem__ = MagicMock(
            return_value={"question": "Q", "answer": "A", "metadata": {}}
        )
        mock_dataset.score_answer = MagicMock(return_value=0.0)
        mock_dataset.dataset_name = "test"

        from llenvs.core.extraction import BoxedExtractor

        env = mod.ReasoningGymEnvironment(
            dataset=mock_dataset,
            answer_extractor=BoxedExtractor(),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="no boxed answer here"))
        assert result.extracted_action is None
        assert result.resolved_action is None


class TestHuggingFaceActionFields:
    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.huggingface")

    def test_step_sets_all_fields(self, import_adapter):
        mod = import_adapter
        from unittest.mock import MagicMock

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__getitem__ = MagicMock(return_value={"question": "Q", "answer": "42"})
        mock_dataset.column_names = ["question", "answer"]
        mock_dataset.features = {"question": None, "answer": None}

        from llenvs.core.extraction import BoxedExtractor

        env = mod.HuggingFaceEnvironment(
            dataset=mock_dataset,
            dataset_name="test/dataset",
            split="test",
            question_column="question",
            answer_column="answer",
            answer_extractor=BoxedExtractor(),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="\\boxed{42}"))
        assert result.extracted_action == "42"
        assert result.resolved_action == "42"


class TestGymnasiumActionFields:
    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.gymnasium")

    def test_success_all_fields(self, import_adapter):
        """Successful step: both fields set, resolved uses format_action."""
        gymnasium = pytest.importorskip("gymnasium")
        gym_env = gymnasium.make("FrozenLake-v1", is_slippery=False)
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            action_names={0: "left", 1: "down", 2: "right", 3: "up"},
            num_tasks=1,
        )
        state, _ = env.reset(seed=0)
        result = env.step(state, Action(text="down"))
        assert result.extracted_action == "down"
        assert result.resolved_action == "down"

    def test_extraction_failure(self, import_adapter):
        """Extraction fails: both fields None, metadata in info."""
        from llenvs.core.extraction import TagBasedExtractor
        from tests.test_gymnasium_adapter import MockDiscrete, MockGymEnv

        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            answer_extractor=TagBasedExtractor(tag_name="action"),
            num_tasks=1,
        )
        state, _ = env.reset()
        result = env.step(state, Action(text="no tag here"))
        assert result.extracted_action is None
        assert result.resolved_action is None
        assert "extraction_metadata" in result.info

    def test_mapping_failure(self, import_adapter):
        """Extraction OK but mapping fails: extracted set, resolved None."""
        from llenvs.core.extraction import TagBasedExtractor
        from tests.test_gymnasium_adapter import MockDiscrete, MockGymEnv

        gym_env = MockGymEnv(
            obs_space=MockDiscrete(4),
            action_space=MockDiscrete(3),
        )
        env = import_adapter.GymnasiumEnvironment(
            gym_env=gym_env,
            answer_extractor=TagBasedExtractor(tag_name="action"),
            num_tasks=1,
        )
        state, _ = env.reset()
        result = env.step(state, Action(text="<action>999</action>"))
        assert result.extracted_action == "999"
        assert result.resolved_action is None
        assert "extraction_metadata" in result.info


class TestIterativeActionFields:
    def test_all_fields_set(self):
        from unittest.mock import MagicMock

        from llenvs.adapters.iterative import IterativeEnvironment

        inner_env = MagicMock()
        inner_env.spec = MagicMock()
        inner_env.spec.name = "test"
        inner_env.spec.adapter = "mock"
        inner_env.spec.max_steps = 1
        inner_env.spec.metadata = {}
        inner_env.reward_functions = ()
        inner_env.available_tools = ()
        inner_env.prompts = {}

        mock_hidden = MagicMock()
        mock_hidden.expected_answer = "42"
        mock_state = State(
            observation=Observation(prompt="Solve: 1+1"),
            hidden=mock_hidden,
            metadata=StateMetadata(step=0, episode_id="ep"),
        )
        inner_env.reset.return_value = (mock_state, {})
        inner_env.step.return_value = StepResult(
            next_state=State(
                observation=Observation(prompt="done"),
                hidden=mock_hidden,
                metadata=StateMetadata(step=1, episode_id="ep", is_terminal=True),
            ),
            rewards=SignalBundle(
                signals=(Signal(name="correctness", reward_type=RewardType.OUTCOME, reward=1.0),)
            ),
            terminated=True,
        )
        inner_env.compute_rewards.return_value = SignalBundle(
            signals=(Signal(name="correctness", reward_type=RewardType.OUTCOME, reward=1.0),)
        )

        env = IterativeEnvironment(
            inner=inner_env,
            max_turns=3,
            solved_threshold=1.0,
        )
        state, _ = env.reset(options={"task_index": 0})
        action_text = "My submission"
        result = env.step(state, Action(text=action_text))

        assert result.extracted_action == result.info["submission"]
        assert result.resolved_action == result.info["submission"]


class TestVerifiersActionFields:
    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.verifiers")

    def test_with_extractor(self, import_adapter):
        from unittest.mock import MagicMock

        from llenvs.core.extraction import BoxedExtractor

        mod = import_adapter
        mock_env = MagicMock()
        mock_env.dataset = [{"input": "Q", "target": "42"}]
        mock_env.__len__ = MagicMock(return_value=1)
        mock_env.rubric = []

        env = mod.VerifiersSingleTurnEnvironment(
            vf_env=mock_env,
            answer_extractor=BoxedExtractor(),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="\\boxed{42}"))
        assert result.extracted_action == "42"
        assert result.resolved_action == "42"

    def test_without_extractor(self, import_adapter):
        from unittest.mock import MagicMock

        mod = import_adapter
        mock_env = MagicMock()
        mock_env.dataset = [{"input": "Q", "target": "42"}]
        mock_env.__len__ = MagicMock(return_value=1)
        mock_env.rubric = []

        env = mod.VerifiersSingleTurnEnvironment(
            vf_env=mock_env,
            answer_extractor=None,
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="some response"))
        assert result.extracted_action is None
        assert result.resolved_action is None


# ===========================================================================
# Runner _resolve_action_text
# ===========================================================================


class TestResolveActionText:
    """Tests for TrajectoryRunner._resolve_action_text priority."""

    def _make_runner(self):
        from unittest.mock import MagicMock

        from llenvs.evaluation.runner import TrajectoryRunner

        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.environment = MagicMock()
        runner.backend = MagicMock()
        runner.sampling_params = MagicMock()
        runner.include_reasoning_in_history = False
        runner.system_prompt = None
        runner.history_fn = None
        runner.tool_call_parser = None
        runner.log = None
        return runner

    def test_prefers_extracted_action(self):
        """extracted_action is highest priority (strips reasoning even on mapping failure)."""
        runner = self._make_runner()
        t = Transition(
            state=_make_state(0),
            action=Action(text="<think>long reasoning</think>The answer is 42"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            extracted_action="42",
            resolved_action="left",
        )
        assert runner._resolve_action_text(t) == "42"

    def test_falls_back_to_resolved_action(self):
        """When extracted is None, falls back to resolved_action."""
        runner = self._make_runner()
        t = Transition(
            state=_make_state(0),
            action=Action(text="<think>reasoning</think>answer"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            resolved_action="answer",
        )
        assert runner._resolve_action_text(t) == "answer"

    def test_falls_back_to_step_info(self):
        runner = self._make_runner()
        t = Transition(
            state=_make_state(0),
            action=Action(text="<think>reasoning</think>answer"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            # both extracted and resolved are None
            info={"step": {"extracted_answer": "answer"}},
        )
        assert runner._resolve_action_text(t) == "answer"

    def test_include_reasoning_returns_full_text(self):
        runner = self._make_runner()
        runner.include_reasoning_in_history = True
        t = Transition(
            state=_make_state(0),
            action=Action(text="<think>reasoning</think>42"),
            next_state=_make_state(1),
            rewards=_empty_bundle(),
            extracted_action="42",
            resolved_action="42",
        )
        assert runner._resolve_action_text(t) == "<think>reasoning</think>42"


# ===========================================================================
# Craftax invalid_action_text + action fields
# ===========================================================================


class TestCraftaxInvalidActionText:
    """Tests for invalid_action_text on CraftaxEnvironment."""

    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.craftax")

    def _make_env(self, import_adapter, invalid_action_text="[invalid action]", **kwargs):
        """Create a CraftaxEnvironment with mock JAX."""
        from unittest.mock import MagicMock

        import numpy as np

        mock_craftax = MagicMock()
        mock_craftax.default_params.return_value = MagicMock(max_steps_in_episode=1000)
        mock_state = MagicMock()
        mock_state.achievements = np.zeros(22, dtype=bool)
        mock_craftax.reset.return_value = (np.zeros(10), mock_state)
        mock_craftax.step.return_value = (
            np.zeros(10),
            mock_state,
            1.0,
            False,
            {},
        )

        mock_jax_random = MagicMock()
        mock_jax_random.PRNGKey.return_value = MagicMock()
        mock_jax_random.split.return_value = [MagicMock(), MagicMock()]

        from llenvs.core.extraction import TagBasedExtractor

        return import_adapter.CraftaxEnvironment(
            craftax_env=mock_craftax,
            is_classic=True,
            observation_mode="symbolic",
            num_tasks=1,
            answer_extractor=TagBasedExtractor(tag_name="action"),
            invalid_action_text=invalid_action_text,
            _jax_random=mock_jax_random,
            **kwargs,
        )

    def test_default_placeholder_in_messages(self, import_adapter):
        env = self._make_env(import_adapter)
        state, _ = env.reset()
        result = env.step(state, Action(text="no tag here"))
        assistant_msg = result.next_state.observation.messages[-2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "[invalid action]"

    def test_custom_placeholder(self, import_adapter):
        env = self._make_env(import_adapter, invalid_action_text="[bad]")
        state, _ = env.reset()
        result = env.step(state, Action(text="no tag here"))
        assistant_msg = result.next_state.observation.messages[-2]
        assert assistant_msg["content"] == "[bad]"

    def test_none_preserves_original(self, import_adapter):
        env = self._make_env(import_adapter, invalid_action_text=None)
        state, _ = env.reset()
        result = env.step(state, Action(text="raw model output"))
        assistant_msg = result.next_state.observation.messages[-2]
        assert assistant_msg["content"] == "raw model output"

    def test_extraction_failure_fields(self, import_adapter):
        """Extraction failure: both fields None."""
        env = self._make_env(import_adapter)
        state, _ = env.reset()
        result = env.step(state, Action(text="no tag here"))
        assert result.extracted_action is None
        assert result.resolved_action is None
        assert "extraction_metadata" in result.info

    def test_mapping_failure_fields(self, import_adapter):
        """Mapping failure: extracted set, resolved None."""
        env = self._make_env(import_adapter)
        state, _ = env.reset()
        result = env.step(state, Action(text="<action>999</action>"))
        assert result.extracted_action == "999"
        assert result.resolved_action is None
        assert "extraction_metadata" in result.info

    def test_success_all_fields(self, import_adapter):
        """Success: both fields set, resolved uses format_action."""
        env = self._make_env(import_adapter)
        state, _ = env.reset()
        result = env.step(state, Action(text="<action>left</action>"))
        assert result.extracted_action == "left"
        # format_action on CraftaxActionMapper: index 5 = "do" in classic, but "left" maps to index
        # which then gets formatted back via format_action
        assert result.resolved_action is not None


# ===========================================================================
# format_action tests
# ===========================================================================


class TestAutoActionMapperFormatAction:
    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.gymnasium")

    def test_discrete_with_names(self, import_adapter):
        import gymnasium.spaces as spaces

        mapper = import_adapter.AutoActionMapper(
            spaces.Discrete(3),
            action_names={0: "left", 1: "right", 2: "up"},
        )
        assert mapper.format_action(0) == "left"
        assert mapper.format_action(1) == "right"
        assert mapper.format_action(2) == "up"

    def test_discrete_without_names(self, import_adapter):
        import gymnasium.spaces as spaces

        mapper = import_adapter.AutoActionMapper(spaces.Discrete(5))
        assert mapper.format_action(0) == "0"
        assert mapper.format_action(4) == "4"

    def test_box(self, import_adapter):
        import gymnasium.spaces as spaces
        import numpy as np

        mapper = import_adapter.AutoActionMapper(spaces.Box(low=-1, high=1, shape=(2,)))
        result = mapper.format_action(np.array([0.5, -0.3]))
        assert "0.5" in result
        assert "-0.3" in result

    def test_multi_discrete(self, import_adapter):
        import gymnasium.spaces as spaces
        import numpy as np

        mapper = import_adapter.AutoActionMapper(spaces.MultiDiscrete([3, 4]))
        result = mapper.format_action(np.array([1, 2]))
        assert result == "1, 2"

    def test_multi_binary(self, import_adapter):
        import gymnasium.spaces as spaces
        import numpy as np

        mapper = import_adapter.AutoActionMapper(spaces.MultiBinary(3))
        result = mapper.format_action(np.array([1, 0, 1]))
        assert result == "1, 0, 1"

    def test_text(self, import_adapter):
        import gymnasium.spaces as spaces

        mapper = import_adapter.AutoActionMapper(spaces.Text(min_length=1, max_length=10))
        assert mapper.format_action("hello") == "hello"


class TestCraftaxActionMapperFormatAction:
    @pytest.fixture
    def import_adapter(self):
        return pytest.importorskip("llenvs.adapters.craftax")

    def test_known_action(self, import_adapter):
        mapper = import_adapter.CraftaxActionMapper(is_classic=True)
        assert mapper.format_action(0) == "noop"
        assert mapper.format_action(5) == "do"

    def test_unknown_index(self, import_adapter):
        mapper = import_adapter.CraftaxActionMapper(is_classic=True)
        assert mapper.format_action(999) == "999"
