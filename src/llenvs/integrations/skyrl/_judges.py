"""Strict opt-in judging; no fallback scores and no policy-history mutation."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from llenvs.core.config import BackendFactory, JudgeConfig, create_sampling_params
from llenvs.core.environment import StepResult
from llenvs.core.judge import (
    JUDGE_TEMPLATES,
    JudgePromptTemplate,
    _gather_judge_context,
    extract_judge_score,
)
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import Action, ImageContent, State
from llenvs.inference.protocol import ChatMessage, SamplingParams, StopReason
from llenvs.integrations.skyrl._checks import finite_number, identifier
from llenvs.integrations.skyrl._episode import feedback_text, reward_total
from llenvs.integrations.skyrl._resources import AsyncEpisode
from llenvs.integrations.skyrl.data import _close_environment, _validate_image_url


@dataclass(frozen=True)
class JudgeRequest:
    question: str
    response: str
    ground_truth: str
    images: tuple[ImageContent, ...] = ()


def _outcome(transition: StepResult[Any]) -> str:
    return (
        f"Environment transition:\n{feedback_text(transition.next_state)}\n"
        f"terminated={transition.terminated}, truncated={transition.truncated}"
    )


def decision_request(
    state: State[Any], raw_reply: str, transition: StepResult[Any], *, use_images: bool
) -> JudgeRequest:
    context = _gather_judge_context(state, Action.from_text(raw_reply), transition.next_state)
    return JudgeRequest(
        question=context["question"],
        response=f"{raw_reply}\n\n{_outcome(transition)}",
        ground_truth=context["ground_truth"],
        images=(
            *state.observation.get_images().all,
            *transition.next_state.observation.get_images().all,
        )
        if use_images
        else (),
    )


def episode_request(
    initial_state: State[Any],
    transcript: list[dict[str, Any]],
    transition: StepResult[Any],
    *,
    end_reason: str,
    use_images: bool,
) -> JudgeRequest:
    text_messages, images = [], []
    for message in transcript:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(part["text"])
                else:
                    url = part["image_url"]["url"]
                    _validate_image_url(url)
                    parts.append("[image]" if use_images else "[image withheld from judge]")
                    if use_images:
                        media_type, data = url[5:].split(";base64,", 1)
                        images.append(ImageContent(data=data, media_type=media_type))
            content = "\n".join(parts)
        text_messages.append(
            {key: (content if key == "content" else value) for key, value in message.items()}
        )
    context = _gather_judge_context(initial_state, Action.from_text(""), transition.next_state)
    return JudgeRequest(
        question=context["question"],
        ground_truth=context["ground_truth"],
        response=f"Included transcript:\n{json.dumps(text_messages, ensure_ascii=False)}\n\n{_outcome(transition)}\nIncluded horizon: {end_reason}",
        images=(*images, *transition.next_state.observation.get_images().all) if use_images else (),
    )


class _Judge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        identifier(config.name, "judge.name")
        finite_number(config.weight, "judge.weight")
        if len(config.score_range) != 2:
            raise ValueError("judge.score_range requires two values")
        lo, hi = (finite_number(value, "judge.score_range") for value in config.score_range)
        finite_number(hi - lo, "judge score range width")
        if hi < lo or not isinstance(config.normalize, bool):
            raise ValueError("invalid judge score range or normalize setting")
        self.reward_type = RewardType[config.reward_type.upper()]
        base = JUDGE_TEMPLATES.get(config.template, JudgePromptTemplate(config.template))
        self.template = JudgePromptTemplate(
            base.template,
            base.name,
            (lo, hi),
            config.system_prompt if config.system_prompt is not None else base.system_prompt,
        )
        # Validate template variables before constructing a model resource.
        self.template.template.format(question="", response="", ground_truth="")
        self.params = (
            create_sampling_params(config.inference)
            if config.inference
            else SamplingParams(temperature=0, max_tokens=512)
        )
        self.backend: Any = None

    def score(self, request: JudgeRequest) -> Signal:
        for image in request.images:
            _validate_image_url(f"data:{image.media_type};base64,{image.data}")
        if self.backend is None:
            self.backend = BackendFactory.create(self.config.model)
        prompt = self.template.template.format(
            question=request.question, response=request.response, ground_truth=request.ground_truth
        )
        messages = []
        if self.template.system_prompt:
            messages.append(ChatMessage(role="system", content=self.template.system_prompt))
        messages.append(ChatMessage(role="user", content=prompt, images=request.images))
        result = self.backend.generate_chat(messages, self.params)
        if (
            result.finish_reason in (StopReason.ERROR, StopReason.MAX_TOKENS)
            or "error" in result.metadata
        ):
            raise ValueError("configured judge failed or returned a truncated response")
        value = raw = extract_judge_score(result.text or "")
        if raw is None:
            raise ValueError("configured judge response has no parseable score")
        finite_number(raw, "judge raw score")
        if self.config.normalize:
            lo, hi = self.template.score_range
            centered = finite_number(raw - lo, "centered judge score")
            value = finite_number(
                centered / (hi - lo) if hi != lo else 0.0, "normalized judge score"
            )
            value = max(0.0, min(1.0, value))
        signal = Signal(
            self.config.name,
            self.reward_type,
            value,
            feedback=result.text,
            metadata={
                "raw_score": raw,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            weight=self.config.weight,
        )
        reward_total(SignalBundle((signal,)))
        return signal

    def close(self) -> None:
        backend, self.backend = self.backend, None
        if backend is not None:
            _close_environment(backend)


class JudgePool:
    """One fixed backend per judge, serialized on the environment executor."""

    def __init__(self, configs: tuple[JudgeConfig, ...], executor: ThreadPoolExecutor) -> None:
        self.names = {config.name for config in configs}
        if len(self.names) != len(configs):
            raise ValueError("duplicate configured judge names")
        self._owners = [
            (AsyncEpisode(_Judge(config), executor), asyncio.Lock()) for config in configs
        ]

    async def score(self, request: JudgeRequest, *, existing_names: set[str]) -> SignalBundle:
        if self.names & existing_names:
            raise ValueError("configured judge is already present in the environment reward bundle")
        signals = []
        for owner, lock in self._owners:
            async with lock:
                signals.append(await owner.call(owner.episode.score, request))
        return SignalBundle(tuple(signals))

    async def aclose(self) -> None:
        results = await asyncio.gather(
            *(owner.aclose() for owner, _ in self._owners), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("judge cleanup failed", errors)
