"""Tests for LLM-as-a-judge reward function."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenvs.core.reward import RewardType, Signal
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.inference.protocol import GenerationResult, SamplingParams, StopReason

# ---------------------------------------------------------------------------
# extract_judge_score
# ---------------------------------------------------------------------------


class TestExtractJudgeScore:
    """Tests for extract_judge_score default parser."""

    def test_bracket_format_integer(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("The answer is [[7]]") == 7.0

    def test_bracket_format_float(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("Rating: [[7.5]]") == 7.5

    def test_bracket_multiple_last_wins(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("First [[3]] then [[8]]") == 8.0

    def test_score_colon_fallback(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("Score: 6") == 6.0

    def test_rating_colon_fallback(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("Rating: 9") == 9.0

    def test_no_score_returns_none(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("No numeric score here") is None

    def test_empty_string_returns_none(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("") is None

    def test_bracket_preferred_over_fallback(self):
        """If both bracket and fallback patterns exist, bracket wins."""
        from llenvs.core.judge import extract_judge_score

        # Last bracket match wins
        assert extract_judge_score("Score: 3\n[[9]]") == 9.0

    def test_fallback_float(self):
        from llenvs.core.judge import extract_judge_score

        assert extract_judge_score("Score: 7.5") == 7.5


# ---------------------------------------------------------------------------
# _gather_judge_context
# ---------------------------------------------------------------------------


class TestGatherJudgeContext:
    """Tests for _gather_judge_context helper."""

    def _make_state(self, prompt="What is 2+2?", hidden=None):
        return State(
            observation=Observation(prompt=prompt),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="ep-1"),
        )

    def test_basic_context(self):
        from llenvs.core.judge import _gather_judge_context

        state = self._make_state()
        action = Action.from_text("The answer is 4.")
        ctx = _gather_judge_context(state, action, state)

        assert ctx["question"] == "What is 2+2?"
        assert ctx["response"] == "The answer is 4."
        assert ctx["ground_truth"] == ""

    def test_expected_answer_attribute(self):
        from llenvs.core.judge import _gather_judge_context

        hidden = SimpleNamespace(expected_answer="4")
        state = self._make_state(hidden=hidden)
        action = Action.from_text("4")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "4"

    def test_answer_attribute(self):
        from llenvs.core.judge import _gather_judge_context

        hidden = SimpleNamespace(answer="42")
        state = self._make_state(hidden=hidden)
        action = Action.from_text("42")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "42"

    def test_ground_truth_attribute(self):
        from llenvs.core.judge import _gather_judge_context

        hidden = SimpleNamespace(ground_truth="yes")
        state = self._make_state(hidden=hidden)
        action = Action.from_text("yes")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "yes"

    def test_target_attribute(self):
        from llenvs.core.judge import _gather_judge_context

        hidden = SimpleNamespace(target="Paris")
        state = self._make_state(hidden=hidden)
        action = Action.from_text("Paris")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "Paris"

    def test_dict_hidden(self):
        from llenvs.core.judge import _gather_judge_context

        hidden = {"expected_answer": "hello"}
        state = self._make_state(hidden=hidden)
        action = Action.from_text("hello")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "hello"

    def test_none_hidden(self):
        from llenvs.core.judge import _gather_judge_context

        state = self._make_state(hidden=None)
        action = Action.from_text("test")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == ""

    def test_none_action_text(self):
        from llenvs.core.judge import _gather_judge_context

        state = self._make_state()
        action = Action(text=None)
        ctx = _gather_judge_context(state, action, state)
        assert ctx["response"] == ""

    def test_attribute_priority(self):
        """expected_answer takes priority over answer."""
        from llenvs.core.judge import _gather_judge_context

        hidden = SimpleNamespace(expected_answer="first", answer="second")
        state = self._make_state(hidden=hidden)
        action = Action.from_text("x")
        ctx = _gather_judge_context(state, action, state)
        assert ctx["ground_truth"] == "first"


# ---------------------------------------------------------------------------
# JudgePromptTemplate + JUDGE_TEMPLATES
# ---------------------------------------------------------------------------


class TestJudgePromptTemplate:
    """Tests for JudgePromptTemplate and built-in templates."""

    def test_custom_template(self):
        from llenvs.core.judge import JudgePromptTemplate

        t = JudgePromptTemplate(template="Rate: {response}", name="mine")
        assert t.template == "Rate: {response}"
        assert t.name == "mine"
        assert t.score_range == (1.0, 10.0)
        assert t.system_prompt is None

    def test_correctness_template_exists(self):
        from llenvs.core.judge import JUDGE_TEMPLATES

        t = JUDGE_TEMPLATES["correctness"]
        assert "{question}" in t.template
        assert "{response}" in t.template
        assert "{ground_truth}" in t.template
        assert "[[" in t.template  # asks for bracket format

    def test_helpfulness_template_exists(self):
        from llenvs.core.judge import JUDGE_TEMPLATES

        t = JUDGE_TEMPLATES["helpfulness"]
        assert "{question}" in t.template
        assert "{response}" in t.template

    def test_safety_template_exists(self):
        from llenvs.core.judge import JUDGE_TEMPLATES

        t = JUDGE_TEMPLATES["safety"]
        assert "{question}" in t.template
        assert "{response}" in t.template

    def test_unknown_template_keyerror(self):
        from llenvs.core.judge import JUDGE_TEMPLATES

        with pytest.raises(KeyError):
            JUDGE_TEMPLATES["nonexistent"]

    def test_template_formatting(self):
        from llenvs.core.judge import JUDGE_TEMPLATES

        t = JUDGE_TEMPLATES["correctness"]
        formatted = t.template.format(
            question="What is 1+1?",
            response="2",
            ground_truth="2",
        )
        assert "What is 1+1?" in formatted
        assert "2" in formatted


# ---------------------------------------------------------------------------
# JudgeReward
# ---------------------------------------------------------------------------


class TestJudgeReward:
    """Tests for JudgeReward.compute()."""

    def _make_backend(
        self, text="Rating: The response is good. [[8]]", prompt_tokens=10, completion_tokens=50
    ):
        backend = MagicMock()
        backend.generate_chat.return_value = GenerationResult(
            text=text,
            finish_reason=StopReason.END_OF_TEXT,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return backend

    def _make_state(self, prompt="What is 2+2?", hidden=None):
        return State(
            observation=Observation(prompt=prompt),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="ep-1"),
        )

    def test_normal_score_normalized(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="Good answer. [[8]]")
        reward = JudgeReward(backend=backend, template="correctness")

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        assert isinstance(signal, Signal)
        assert signal.name == "judge"
        assert signal.reward_type == RewardType.OUTCOME
        # 8 on 1-10 scale → (8-1)/(10-1) = 7/9 ≈ 0.778
        assert signal.reward == pytest.approx(7.0 / 9.0)
        assert signal.weight == 1.0

    def test_unparseable_returns_default(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="I cannot rate this.")
        reward = JudgeReward(backend=backend, template="correctness", default_score=0.0)

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        assert signal.reward == 0.0
        assert signal.metadata is not None
        assert "error" in signal.metadata

    def test_backend_exception_returns_error_signal(self):
        from llenvs.core.judge import JudgeReward

        backend = MagicMock()
        backend.generate_chat.side_effect = RuntimeError("API timeout")
        reward = JudgeReward(backend=backend, template="correctness", default_score=0.0)

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        assert signal.reward == 0.0
        assert signal.metadata is not None
        assert "error" in signal.metadata
        assert "API timeout" in signal.metadata["error"]

    def test_normalize_false_returns_raw(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[8]]")
        reward = JudgeReward(backend=backend, template="correctness", normalize=False)

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        assert signal.reward == 8.0

    def test_weight_propagation(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[7]]")
        reward = JudgeReward(backend=backend, template="correctness", weight=0.5)

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        assert signal.weight == 0.5

    def test_metadata_contents(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(
            text="Excellent! [[9]]", prompt_tokens=15, completion_tokens=40
        )
        reward = JudgeReward(backend=backend, template="correctness")

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        assert signal.metadata is not None
        assert signal.metadata["raw_score"] == 9.0
        assert signal.metadata["judge_response"] == "Excellent! [[9]]"
        assert signal.metadata["prompt_tokens"] == 15
        assert signal.metadata["completion_tokens"] == 40

    def test_custom_score_extractor(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="SCORE=5")

        def custom_extractor(text: str) -> float | None:
            if "SCORE=" in text:
                return float(text.split("SCORE=")[1].strip())
            return None

        reward = JudgeReward(
            backend=backend,
            template="correctness",
            score_extractor=custom_extractor,
        )

        state = self._make_state()
        action = Action.from_text("4")
        signal = reward.compute(state, action, state)

        # 5 on 1-10 scale → (5-1)/(10-1) = 4/9
        assert signal.reward == pytest.approx(4.0 / 9.0)

    def test_custom_template_string(self):
        """Pass a literal template string instead of built-in name."""
        from llenvs.core.judge import JudgePromptTemplate, JudgeReward

        backend = self._make_backend(text="[[6]]")
        template = JudgePromptTemplate(
            template="Rate this: {response}",
            score_range=(1.0, 5.0),
        )
        reward = JudgeReward(backend=backend, template=template)

        state = self._make_state()
        action = Action.from_text("hello")
        signal = reward.compute(state, action, state)

        # 6 on 1-5 scale → (6-1)/(5-1) = 5/4 = 1.25, clamped to 1.0
        assert signal.reward == pytest.approx(1.0)

    def test_name_and_reward_type(self):
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[5]]")
        reward = JudgeReward(
            backend=backend,
            template="correctness",
            name="quality_judge",
            reward_type=RewardType.PROCESS,
        )

        assert reward.name == "quality_judge"
        assert reward.reward_type == RewardType.PROCESS

    def test_sampling_params_passed(self):
        """Verify sampling params are forwarded to backend."""
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[7]]")
        params = SamplingParams(temperature=0.0, max_tokens=256)
        reward = JudgeReward(backend=backend, template="correctness", sampling_params=params)

        state = self._make_state()
        action = Action.from_text("4")
        reward.compute(state, action, state)

        call_args = backend.generate_chat.call_args
        assert call_args[0][1] == params  # second positional arg

    def test_default_sampling_params(self):
        """When no sampling_params given, uses sensible defaults."""
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[7]]")
        reward = JudgeReward(backend=backend, template="correctness")

        state = self._make_state()
        action = Action.from_text("4")
        reward.compute(state, action, state)

        call_args = backend.generate_chat.call_args
        params = call_args[0][1]
        assert params.temperature == 0.0
        assert params.max_tokens == 512

    def test_system_prompt_in_messages(self):
        """Templates with system_prompt produce a system message."""
        from llenvs.core.judge import JudgePromptTemplate, JudgeReward

        backend = self._make_backend(text="[[7]]")
        template = JudgePromptTemplate(
            template="Rate: {response}",
            system_prompt="You are a strict judge.",
        )
        reward = JudgeReward(backend=backend, template=template)

        state = self._make_state()
        action = Action.from_text("hello")
        reward.compute(state, action, state)

        messages = backend.generate_chat.call_args[0][0]
        assert messages[0].role == "system"
        assert messages[0].content == "You are a strict judge."
        assert messages[1].role == "user"

    def test_no_system_prompt(self):
        """Templates without system_prompt produce only user message."""
        from llenvs.core.judge import JudgePromptTemplate, JudgeReward

        backend = self._make_backend(text="[[7]]")
        template = JudgePromptTemplate(template="Rate: {response}")
        reward = JudgeReward(backend=backend, template=template)

        state = self._make_state()
        action = Action.from_text("hello")
        reward.compute(state, action, state)

        messages = backend.generate_chat.call_args[0][0]
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_score_at_range_minimum(self):
        """Score at lower bound normalizes to 0."""
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[1]]")
        reward = JudgeReward(backend=backend, template="correctness")

        state = self._make_state()
        action = Action.from_text("wrong")
        signal = reward.compute(state, action, state)

        assert signal.reward == pytest.approx(0.0)

    def test_score_at_range_maximum(self):
        """Score at upper bound normalizes to 1."""
        from llenvs.core.judge import JudgeReward

        backend = self._make_backend(text="[[10]]")
        reward = JudgeReward(backend=backend, template="correctness")

        state = self._make_state()
        action = Action.from_text("perfect")
        signal = reward.compute(state, action, state)

        assert signal.reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Config: JudgeConfig
# ---------------------------------------------------------------------------


class TestJudgeConfig:
    """Tests for JudgeConfig parsing via EvalConfig.from_dict/to_dict."""

    def _base_eval_dict(self):
        return {
            "environments": [
                {"name": "test_env", "adapter": "reasoning_gym"},
            ],
            "model": {"backend": "openai", "model": "gpt-4o"},
        }

    def test_eval_level_judge(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
            "template": "correctness",
        }
        config = EvalConfig.from_dict(data)
        assert config.judge is not None
        assert not isinstance(config.judge, list)
        assert config.judge.model.backend == "openai"
        assert config.judge.model.model == "gpt-4o-mini"
        assert config.judge.template == "correctness"

    def test_env_level_judge(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["environments"][0]["judge"] = {
            "model": {"backend": "anthropic", "model": "claude-sonnet-4-20250514"},
            "template": "helpfulness",
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.judge is not None
        assert not isinstance(env.judge, list)
        assert env.judge.model.backend == "anthropic"
        assert env.judge.template == "helpfulness"

    def test_multiple_judges_list(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["environments"][0]["judge"] = [
            {
                "model": {"backend": "openai", "model": "gpt-4o"},
                "template": "correctness",
                "name": "judge_correct",
                "weight": 0.5,
            },
            {
                "model": {"backend": "openai", "model": "gpt-4o"},
                "template": "safety",
                "name": "judge_safety",
                "weight": 0.5,
            },
        ]
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert isinstance(env.judge, list)
        assert len(env.judge) == 2
        assert env.judge[0].name == "judge_correct"
        assert env.judge[1].name == "judge_safety"

    def test_judge_config_defaults(self):
        from llenvs.core.config import EvalConfig, JudgeConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
        }
        config = EvalConfig.from_dict(data)
        j = config.judge
        assert isinstance(j, JudgeConfig)
        assert j.template == "correctness"
        assert j.name == "judge"
        assert j.weight == 1.0
        assert j.normalize is True
        assert j.reward_type == "outcome"
        assert j.score_range == (1.0, 10.0)

    def test_judge_to_dict_round_trip(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
            "template": "safety",
            "weight": 0.7,
        }
        config = EvalConfig.from_dict(data)
        out = config.to_dict()
        assert "judge" in out
        assert out["judge"]["template"] == "safety"
        assert out["judge"]["weight"] == 0.7

    def test_env_judge_to_dict_round_trip(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["environments"][0]["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o"},
            "template": "helpfulness",
        }
        config = EvalConfig.from_dict(data)
        out = config.to_dict()
        env_dict = out["environments"][0]
        assert "judge" in env_dict
        assert env_dict["judge"]["template"] == "helpfulness"

    def test_judge_with_inference_config(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
            "inference": {"temperature": 0.1, "max_tokens": 256},
        }
        config = EvalConfig.from_dict(data)
        assert config.judge.inference is not None
        assert config.judge.inference.temperature == 0.1
        assert config.judge.inference.max_tokens == 256

    def test_judge_with_custom_template_literal(self):
        """A template containing { is treated as literal."""
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
            "template": "Rate this response: {response}\nScore: [[score]]",
        }
        config = EvalConfig.from_dict(data)
        assert "{response}" in config.judge.template

    def test_no_judge_field(self):
        """When judge is not in config, fields are None."""
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        config = EvalConfig.from_dict(data)
        assert config.judge is None
        assert config.environments[0].judge is None

    def test_judge_score_range(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
            "score_range": [1, 5],
        }
        config = EvalConfig.from_dict(data)
        assert config.judge.score_range == (1.0, 5.0)

    def test_judge_system_prompt_override(self):
        from llenvs.core.config import EvalConfig

        data = self._base_eval_dict()
        data["judge"] = {
            "model": {"backend": "openai", "model": "gpt-4o-mini"},
            "system_prompt": "Be very strict.",
        }
        config = EvalConfig.from_dict(data)
        assert config.judge.system_prompt == "Be very strict."
