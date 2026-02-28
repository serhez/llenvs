"""Tests for iterative refinement environment."""

from llenvs.adapters.iterative import (
    IterativeEnvironment,
    IterativeHidden,
    IterativeTask,
)
from llenvs.core.environment import StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyReward:
    """Reward that always returns a fixed value with feedback."""

    def __init__(self, reward=0.5, feedback="needs work", name="test_reward"):
        self._name = name
        self._reward_type = RewardType.OUTCOME
        self._reward = reward
        self._feedback = feedback

    @property
    def name(self):
        return self._name

    @property
    def reward_type(self):
        return self._reward_type

    def compute(self, state, action, next_state):
        return Signal(
            name=self._name,
            reward_type=self._reward_type,
            reward=self._reward,
            feedback=self._feedback,
        )


class PerfectReward:
    """Reward that returns 1.0 to trigger solved termination."""

    _name = "perfect"
    _reward_type = RewardType.OUTCOME

    @property
    def name(self):
        return self._name

    @property
    def reward_type(self):
        return self._reward_type

    def compute(self, state, action, next_state):
        return Signal(
            name=self._name,
            reward_type=self._reward_type,
            reward=1.0,
            feedback="Perfect!",
        )


# ---------------------------------------------------------------------------
# IterativeTask
# ---------------------------------------------------------------------------


class TestIterativeTask:
    def test_creation(self):
        task = IterativeTask(prompt="Solve x + 1 = 2", ground_truth="1")
        assert task.prompt == "Solve x + 1 = 2"
        assert task.ground_truth == "1"
        assert task.test_code == ""

    def test_with_test_code(self):
        task = IterativeTask(
            prompt="Write add()",
            test_code="assert add(1,2) == 3",
        )
        assert task.test_code == "assert add(1,2) == 3"


# ---------------------------------------------------------------------------
# IterativeHidden
# ---------------------------------------------------------------------------


class TestIterativeHidden:
    def test_ground_truth_from_inner_hidden(self):
        class Inner:
            expected_answer = "42"

        hidden = IterativeHidden(
            task_index=0,
            inner_hidden=Inner(),
            task_prompt="test",
            turn=0,
            submissions=(),
            feedback_history=(),
            max_turns=3,
        )
        assert hidden.ground_truth == "42"

    def test_ground_truth_from_task(self):
        task = IterativeTask(prompt="test", ground_truth="the answer")
        hidden = IterativeHidden(
            task_index=0,
            inner_hidden=task,
            task_prompt="test",
            turn=0,
            submissions=(),
            feedback_history=(),
            max_turns=3,
        )
        assert hidden.ground_truth == "the answer"

    def test_ground_truth_fallback(self):
        hidden = IterativeHidden(
            task_index=0,
            inner_hidden=object(),
            task_prompt="test",
            turn=0,
            submissions=(),
            feedback_history=(),
            max_turns=3,
        )
        assert hidden.ground_truth == ""

    def test_proxy_to_inner(self):
        class Inner:
            custom_attr = "hello"

        hidden = IterativeHidden(
            task_index=0,
            inner_hidden=Inner(),
            task_prompt="test",
            turn=0,
            submissions=(),
            feedback_history=(),
            max_turns=3,
        )
        assert hidden.custom_attr == "hello"


# ---------------------------------------------------------------------------
# IterativeEnvironment — standalone tasks
# ---------------------------------------------------------------------------


class TestIterativeEnvironmentStandalone:
    """Tests with direct task list (no inner environment)."""

    def _make_env(self, **kwargs):
        tasks = (
            IterativeTask(prompt="Solve: 2 + 2", ground_truth="4"),
            IterativeTask(prompt="Solve: 3 + 3", ground_truth="6"),
        )
        defaults = dict(
            tasks=tasks,
            max_turns=3,
            extra_rewards=(DummyReward(),),
        )
        defaults.update(kwargs)
        return IterativeEnvironment(**defaults)

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 0})

        assert state.observation.prompt is not None
        assert "Solve: 2 + 2" in state.observation.prompt
        assert not state.metadata.is_terminal
        assert state.hidden.turn == 0
        assert state.hidden.max_turns == 3

        # Structured observation: task set on reset
        obs = state.observation
        assert isinstance(obs.task, ObservationContent)
        assert obs.task.text == obs.prompt

    def test_len(self):
        env = self._make_env()
        assert len(env) == 2

    def test_spec(self):
        env = self._make_env()
        spec = env.spec
        assert spec.is_multi_turn
        assert spec.pure_step
        assert spec.max_steps == 3

    def test_step_returns_feedback(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="My answer is 4"))

        assert isinstance(result, StepResult)
        assert result.rewards.feedback_texts()
        assert result.next_state.hidden.turn == 1
        assert len(result.next_state.hidden.submissions) == 1

        # Structured observation: task carried forward, state updated on step
        next_obs = result.next_state.observation
        assert next_obs.task is not None
        assert next_obs.task.text == state.observation.prompt  # task stays as initial prompt
        assert isinstance(next_obs.state, ObservationContent)
        assert next_obs.state.text == next_obs.prompt  # state reflects feedback observation

    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=2)
        state, _ = env.reset(options={"task_index": 0})

        # Turn 1
        result = env.step(state, Action(text="answer 1"))
        assert not result.done

        # Turn 2 (max)
        result = env.step(result.next_state, Action(text="answer 2"))
        assert result.done
        assert result.truncated

    def test_early_submit(self):
        env = self._make_env(submit_keyword="SUBMIT")
        state, _ = env.reset(options={"task_index": 0})

        # Submit immediately
        result = env.step(state, Action(text="SUBMIT my final answer is 4"))
        assert result.done
        assert result.terminated

    def test_submit_keyword_none_disables(self):
        env = self._make_env(submit_keyword=None)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="SUBMIT my answer"))
        assert not result.done  # SUBMIT has no effect

    def test_solved_termination(self):
        env = self._make_env(
            extra_rewards=(PerfectReward(),),
            solved_threshold=1.0,
        )
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="4"))
        assert result.terminated

    def test_feedback_in_observation(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="my answer"))
        obs = result.next_state.observation
        assert "Feedback" in obs.prompt or "feedback" in obs.prompt.lower()

    def test_history_tracking(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        r1 = env.step(state, Action(text="first attempt"))
        r2 = env.step(r1.next_state, Action(text="second attempt"))

        assert len(r2.next_state.hidden.submissions) == 2
        assert r2.next_state.hidden.submissions[0] == "first attempt"
        assert r2.next_state.hidden.submissions[1] == "second attempt"

    def test_include_history_false(self):
        env = self._make_env(include_history=False)
        state, _ = env.reset(options={"task_index": 0})

        r1 = env.step(state, Action(text="first"))
        r2 = env.step(r1.next_state, Action(text="second"))

        # Without history, observation should not mention prior submissions
        obs_text = r2.next_state.observation.prompt
        assert "first" not in obs_text.lower()

    def test_messages_appended(self):
        env = self._make_env()
        state, _ = env.reset(options={"task_index": 0})

        r1 = env.step(state, Action(text="attempt 1"))
        # Should have messages for chat runners
        assert len(r1.next_state.observation.messages) > 0

    def test_prompts_property(self):
        env = self._make_env()
        prompts = env.prompts
        assert "initial" in prompts
        assert "feedback" in prompts

    def test_custom_prompts(self):
        custom = {"initial": "CUSTOM: {task}"}
        env = self._make_env(prompts=custom)
        state, _ = env.reset(options={"task_index": 0})
        assert "CUSTOM:" in state.observation.prompt


# ---------------------------------------------------------------------------
# IterativeEnvironment — with inner environment
# ---------------------------------------------------------------------------


class MockInnerEnv:
    """Mock single-turn environment for wrapping."""

    def __init__(self, reward_value=0.5):
        self._reward_value = reward_value

    @property
    def prompts(self):
        return {}

    @property
    def available_tools(self):
        return ()

    @property
    def spec(self):
        from llenvs.core.environment import EnvironmentSpec

        return EnvironmentSpec(name="mock_inner", adapter="mock", pure_step=True)

    @property
    def reward_functions(self):
        return (DummyReward(reward=self._reward_value, name="inner_reward"),)

    def __len__(self):
        return 2

    def reset(self, *, seed=None, options=None):
        options = options or {}
        task_index = options.get("task_index", 0)

        class InnerHidden:
            expected_answer = "42"

        obs = Observation(prompt=f"Task {task_index}: solve this")
        state = State(
            observation=obs,
            hidden=InnerHidden(),
            metadata=StateMetadata(step=0, episode_id="inner"),
        )
        return state, {"task_index": task_index}

    def step(self, state, action):
        return StepResult(
            next_state=state,
            rewards=self.compute_rewards(state, action, state),
            terminated=True,
        )

    def compute_rewards(self, state, action, next_state):
        signals = tuple(rf.compute(state, action, next_state) for rf in self.reward_functions)
        return SignalBundle(signals=signals)


class TestIterativeEnvironmentWithInner:
    def test_wrap_inner(self):
        inner = MockInnerEnv()
        env = IterativeEnvironment(inner=inner, max_turns=3)

        assert len(env) == 2  # delegates to inner

    def test_reset_uses_inner(self):
        inner = MockInnerEnv()
        env = IterativeEnvironment(inner=inner, max_turns=3)
        state, info = env.reset(options={"task_index": 0})

        assert "Task 0" in state.observation.prompt

    def test_step_evaluates_via_inner(self):
        inner = MockInnerEnv(reward_value=0.8)
        env = IterativeEnvironment(
            inner=inner,
            max_turns=3,
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="my answer is 42"))

        # Should have inner reward signal
        inner_sig = result.rewards.by_name("inner_reward")
        assert inner_sig is not None
        assert inner_sig.reward == 0.8

    def test_inner_plus_extra_rewards(self):
        inner = MockInnerEnv(reward_value=0.5)
        extra = DummyReward(reward=0.3, name="extra_judge")
        env = IterativeEnvironment(
            inner=inner,
            max_turns=3,
            extra_rewards=(extra,),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="answer"))

        assert result.rewards.by_name("inner_reward") is not None
        assert result.rewards.by_name("extra_judge") is not None

    def test_reward_functions_property(self):
        inner = MockInnerEnv()
        extra = DummyReward(name="extra")
        env = IterativeEnvironment(inner=inner, extra_rewards=(extra,))
        rf = env.reward_functions
        names = {r.name for r in rf}
        assert "inner_reward" in names
        assert "extra" in names


# ---------------------------------------------------------------------------
# Submission extractor
# ---------------------------------------------------------------------------


class TestSubmissionExtraction:
    def test_raw_extraction_default(self):
        """By default, the full response is used as the submission."""
        tasks = (IterativeTask(prompt="test"),)
        env = IterativeEnvironment(
            tasks=tasks,
            max_turns=2,
            extra_rewards=(DummyReward(),),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="my full response"))

        assert result.next_state.hidden.submissions[-1] == "my full response"

    def test_custom_submission_extractor(self):
        """Custom extractor can parse the answer."""
        from llenvs.core.extraction import TagBasedExtractor

        tasks = (IterativeTask(prompt="test"),)
        env = IterativeEnvironment(
            tasks=tasks,
            max_turns=2,
            submission_extractor=TagBasedExtractor(),
            extra_rewards=(DummyReward(),),
        )
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="thinking... <answer>42</answer>"))

        assert result.next_state.hidden.submissions[-1] == "42"
