"""Configured training judges use strict parsing, weighting and owned backends."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llenvs.core.config import BackendFactory, JudgeConfig, ModelConfig
from llenvs.core.environment import StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.state import ImageContent, Observation, ObservationContent, State, StateMetadata
from llenvs.inference.protocol import GenerationResult, StopReason
from tests.test_skyrl_resources import run_async


@pytest.fixture
def judges():
    return importlib.import_module("llenvs.integrations.skyrl._judges")


@pytest.fixture
def context():
    before = State(
        Observation("Question", state=ObservationContent(text="Before")),
        hidden={"expected_answer": "gold"},
        metadata=StateMetadata(0, "episode"),
    )
    after = replace(
        before, observation=Observation("Question", state=ObservationContent(text="Transition"))
    )
    return before, StepResult(after, SignalBundle.single(2), terminated=True)


@pytest.fixture
def backend(monkeypatch):
    backend = SimpleNamespace(
        generate_chat=Mock(
            return_value=GenerationResult(text="[[7]]", finish_reason=StopReason.END_OF_TEXT)
        ),
        close=Mock(),
    )
    factory = Mock(return_value=backend)
    monkeypatch.setattr(BackendFactory, "create", factory)
    return backend, factory


@run_async
async def test_normalized_weighted_judge_sees_raw_reply_transition_and_gold(
    judges, context, backend
):
    model, factory = backend
    config = JudgeConfig(ModelConfig(backend="openai", model="external"), name="extra", weight=0.5)
    before, transition = context
    request = judges.decision_request(
        before, "RAW <tool_call>reply</tool_call>", transition, use_images=False
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = judges.JudgePool((config,), executor)
        factory.assert_not_called()
        result = await pool.score(request, existing_names={"reward"})
        again = await pool.score(request, existing_names=set())
        await pool.aclose()
        await pool.aclose()
    assert result == again and result.total == pytest.approx(1 / 3)
    assert result.signals[0].reward == pytest.approx(2 / 3)
    messages, params = model.generate_chat.call_args.args
    assert all(value in messages[-1].content for value in ("RAW <tool_call>", "Transition", "gold"))
    assert params.temperature == 0 and params.max_tokens == 512
    assert not messages[-1].images
    factory.assert_called_once_with(config.model)
    model.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["parse", "backend", "error_metadata", "overflow", "truncated"])
@run_async
async def test_explicit_judge_failure_never_becomes_fallback_zero(
    judges, context, backend, failure
):
    model, _ = backend
    if failure == "backend":
        model.generate_chat.side_effect = RuntimeError("request failed")
    elif failure == "parse":
        model.generate_chat.return_value = GenerationResult(text="no score")
    elif failure == "error_metadata":
        model.generate_chat.return_value = GenerationResult(
            text="[[7]]", metadata={"error": "failed"}
        )
    elif failure == "truncated":
        model.generate_chat.return_value = GenerationResult(
            text="[[7]]", finish_reason=StopReason.MAX_TOKENS
        )
    else:
        model.generate_chat.return_value = GenerationResult(text="[[" + "9" * 400 + "]]")
    before, transition = context
    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = judges.JudgePool((JudgeConfig(ModelConfig(model="external")),), executor)
        try:
            with pytest.raises((RuntimeError, ValueError)):
                await pool.score(
                    judges.decision_request(before, "reply", transition, use_images=False),
                    existing_names=set(),
                )
        finally:
            await pool.aclose()
    model.close.assert_called_once()


@run_async
async def test_duplicate_extra_judge_is_rejected_before_another_call(judges, context, backend):
    model, factory = backend
    config = JudgeConfig(ModelConfig(model="external"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match="duplicate"):
            judges.JudgePool((config, config), executor)
        pool = judges.JudgePool((config,), executor)
        try:
            with pytest.raises(ValueError, match="already"):
                await pool.score(
                    judges.decision_request(context[0], "reply", context[1], use_images=False),
                    existing_names={"judge"},
                )
        finally:
            await pool.aclose()
    factory.assert_not_called()
    model.generate_chat.assert_not_called()


def test_judge_images_are_opt_in_ordered_and_do_not_change_policy_context(judges, context):
    before, transition = context
    task, current, next_image = [ImageContent(data=data) for data in ("eA==", "eQ==", "eg==")]
    before = replace(
        before,
        observation=replace(
            before.observation,
            task=ObservationContent(images=(task,)),
            state=ObservationContent(text="Before", images=(current,)),
        ),
    )
    after = replace(
        transition.next_state,
        observation=replace(
            transition.next_state.observation,
            state=ObservationContent(text="Transition", images=(next_image,)),
        ),
    )
    transition = replace(transition, next_state=after)
    text = judges.decision_request(before, "reply", transition, use_images=False)
    vision = judges.decision_request(before, "reply", transition, use_images=True)
    assert not text.images
    assert [image.data for image in vision.images] == [task.data, current.data, next_image.data]
    assert "base64" not in text.response
    assert before.observation.get_images().state[0].data == current.data


def test_episode_request_contains_only_included_transcript_and_named_horizon(judges, context):
    before, transition = context
    transcript = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "feedback"},
        {"role": "assistant", "content": "last"},
    ]
    request = judges.episode_request(
        before, transcript, transition, end_reason="decision_limit", use_images=False
    )
    assert all(
        value in request.response
        for value in ("first", "feedback", "last", "decision_limit", "Transition")
    )
    assert request.ground_truth == "gold"
    assert len(transcript) == 4


def test_overflowing_judge_normalization_range_is_rejected_before_backend(judges, backend):
    _, factory = backend
    config = JudgeConfig(ModelConfig(model="external"), score_range=(-1e308, 1e308))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match="finite"):
            judges.JudgePool((config,), executor)
    factory.assert_not_called()


@run_async
async def test_invalid_judge_image_is_rejected_before_backend(judges, context, backend):
    _, factory = backend
    request = judges.decision_request(context[0], "reply", context[1], use_images=False)
    request = replace(request, images=(ImageContent("not base64"),))
    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = judges.JudgePool((JudgeConfig(ModelConfig(model="external")),), executor)
        try:
            with pytest.raises(ValueError, match="base64"):
                await pool.score(request, existing_names=set())
        finally:
            await pool.aclose()
    factory.assert_not_called()
