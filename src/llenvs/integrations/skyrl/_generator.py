"""Complete text episodes through a token-in/token-out inference client.

This driver component has no training loop or native tensor-layout logic. The
native wrapper supplies resolved sampling parameters, metrics, and its ABC.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llenvs.core.reward import SignalBundle
from llenvs.integrations.skyrl._checks import finite_number, identifier, integer, token_ids
from llenvs.integrations.skyrl._config import SelectedEnvironment
from llenvs.integrations.skyrl._episode import Episode, feedback_message, reward_total
from llenvs.integrations.skyrl._judges import JudgePool, decision_request, episode_request
from llenvs.integrations.skyrl._rendering import TextRenderer
from llenvs.integrations.skyrl._resources import CLEANUP_TIMEOUT, AsyncEpisode, ScorerOwner
from llenvs.integrations.skyrl._trace import validate_generation
from llenvs.integrations.skyrl._validation import (
    TOKEN_ESTIMATOR,
    TURN_ESTIMATOR,
    validate_sampling_request,
)
from llenvs.integrations.skyrl.scoring import GenerationScoreInput, score_generation


class EpisodeGenerator:
    def __init__(
        self,
        cfg: Any,
        selected: SelectedEnvironment,
        tokenizer: Any,
        engine: Any,
        *,
        train_sampling_params: dict[str, Any],
        logprobs_mode: str,
        model_context_length: int,
        vocab_size: int,
        provenance: dict[str, Any],
        policy_model_name: str | None = None,
    ) -> None:
        cfg, selected = copy.deepcopy(cfg), copy.deepcopy(selected)
        self.cfg, self.selected, self.engine = cfg, selected, engine
        self.policy_model_name = policy_model_name or engine.model_name
        self.provenance = copy.deepcopy(provenance)
        self.train_sampling_params = copy.deepcopy(train_sampling_params)
        # These are resolved engine/model facts, not invented native config
        # fields or guesses from tokenizer size / positional limits.
        self.logprobs_mode = identifier(logprobs_mode, "resolved logprobs_mode")
        self.vocab_size = integer(vocab_size, "model vocabulary size", minimum=1)
        self.estimator = cfg.trainer.algorithm.advantage_estimator
        if self.estimator not in ("grpo", TURN_ESTIMATOR, TOKEN_ESTIMATOR):
            raise ValueError("unsupported advantage_estimator")
        if cfg.llenvs.token_scorer is not None and self.estimator != TOKEN_ESTIMATOR:
            raise ValueError("token scorer requires llenvs_token_rtg")
        for name in (
            "batched",
            "vision_language_generator",
            "step_wise_trajectories",
            "merge_stepwise_output",
        ):
            if getattr(cfg.generator, name):
                raise ValueError(f"{name} is not supported by the text episode generator")
        if (
            not cfg.generator.use_conversation_multi_turn
            or cfg.generator.chat_template.name_or_path is not None
        ):
            raise ValueError("text episodes require the tokenizer's fixed-base multi-turn template")
        if self.estimator != "grpo" and cfg.generator.zero_reward_on_non_stop:
            raise ValueError("zero_reward_on_non_stop is incompatible with custom credit")
        self.max_turns = integer(cfg.generator.max_turns, "max_turns", minimum=1)
        self.max_input = integer(cfg.generator.max_input_length, "max_input_length", minimum=1)
        self.max_prompt = integer(cfg.trainer.max_prompt_length, "max_prompt_length", minimum=1)
        self.max_sequence = integer(model_context_length, "model context length", minimum=1)
        # algorithm.max_seq_len is a loss normalizer/native warning threshold,
        # not a generation cap. Changing it must not change sampled episodes.
        self.renderer = TextRenderer(
            tokenizer,
            vocab_size=self.vocab_size,
            chat_template_kwargs=cfg.generator.chat_template_kwargs,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=integer(
                cfg.environment.skyrl_gym.max_env_workers, "max_env_workers", minimum=1
            ),
            thread_name_prefix="llenvs-skyrl-env",
        )
        self._capacity = asyncio.Semaphore(
            integer(cfg.llenvs.max_active_episodes, "max_active_episodes", minimum=1)
        )
        self._scorer = ScorerOwner(cfg.llenvs.token_scorer) if cfg.llenvs.token_scorer else None
        self._judges = JudgePool(selected.judges, self._executor)
        self._episodes: set[asyncio.Task[Any]] = set()
        self._cleanup_errors: list[Exception] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing: asyncio.Task[None] | None = None

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("generator cannot move between driver event loops")

    async def generate(self, input_batch: dict[str, Any]) -> dict[str, Any]:
        self._check_loop()
        if self._closing is not None:
            raise RuntimeError("generator is closed")
        try:
            phase = input_batch["batch_metadata"].training_phase
            if phase not in ("train", "eval"):
                raise ValueError("invalid batch training_phase")
            count = len(input_batch["prompts"])
            samples = integer(
                getattr(
                    self.cfg.generator,
                    "n_samples_per_prompt" if phase == "train" else "eval_n_samples_per_prompt",
                ),
                "samples per prompt",
                minimum=1,
            )
            if not count or count % samples:
                raise ValueError("generator input must contain complete prompt groups")
            for name in ("env_classes", "env_extras", "trajectory_ids"):
                if not isinstance(input_batch.get(name), list) or len(input_batch[name]) != count:
                    raise ValueError(f"{name} must have one entry per trajectory")
            parameters = copy.deepcopy(
                input_batch.get("sampling_params") or self.train_sampling_params
            )
            if phase == "eval" and input_batch.get("sampling_params") is None:
                raise ValueError("evaluation requires native-resolved eval sampling parameters")
            batch_nonce = uuid.uuid4().hex
            version = getattr(self.engine, "weight_version", None)
            cache_salt = (
                f"{self.policy_model_name}@{version}"
                if self.cfg.generator.use_cache_salt and version is not None
                else None
            )
            jobs = []
            scheduled = time.monotonic()
            for offset in range(0, count, samples):
                identities = input_batch["trajectory_ids"][offset : offset + samples]
                for identity in identities:
                    integer(identity.repetition_id, "repetition_id")
                if len({identity.instance_id for identity in identities}) != 1 or {
                    identity.repetition_id for identity in identities
                } != set(range(samples)):
                    raise ValueError("native input contains an incomplete or mixed prompt group")
                for index in range(offset, offset + samples):
                    extras = input_batch["env_extras"][index]
                    identity = input_batch["trajectory_ids"][index]
                    if not isinstance(extras, dict) or set(extras) != {"data_source", "llenvs"}:
                        raise ValueError("unexpected environment extras")
                    if extras["llenvs"]["task_id"] != identity.instance_id:
                        raise ValueError("native trajectory UID differs from prepared task_id")
                    row = {
                        "prompt": input_batch["prompts"][index],
                        "env_class": input_batch["env_classes"][index],
                        **extras,
                    }
                    # Construct the lightweight validated owner before allocating resources.
                    episode = Episode(self.selected, row)
                    task = asyncio.create_task(
                        self._episode(
                            episode,
                            identity,
                            f"{batch_nonce}:{offset // samples}",
                            parameters,
                            phase,
                            cache_salt,
                            scheduled,
                        )
                    )
                    self._episodes.add(task)
                    task.add_done_callback(self._episodes.discard)
                    jobs.append(task)
            rows = await asyncio.gather(*jobs)
            output = {name: [row[name] for row in rows] for name in rows[0]}
            output["trajectory_ids"] = list(input_batch["trajectory_ids"])
            if parameters.get("logprobs") is None:
                output["rollout_logprobs"] = None
            output["trajectory_time_splits"] = {
                key: [row["trajectory_time_splits"][key] for row in rows]
                for key in rows[0]["trajectory_time_splits"]
            }
            output["rollout_metrics"] = {
                "llenvs/mean_episode_seconds": sum(output["trajectory_generation_times"]) / count
            }
            return output
        except BaseException:
            await self.aclose()
            raise

    async def _episode(
        self,
        episode: Episode,
        identity: Any,
        group: str,
        parameters: dict[str, Any],
        phase: str,
        cache_salt: str | None,
        scheduled: float,
    ) -> dict[str, Any]:
        async with self._capacity:
            owner = AsyncEpisode(episode, self._executor)
            session = f"{group}:{identity.repetition_id}"
            timings = {
                "queue": time.monotonic() - scheduled,
                "llm": 0.0,
                "env": 0.0,
                "scoring": 0.0,
            }
            try:
                started = time.monotonic()
                conditioning = await owner.call(episode.open)
                initial_state = episode.state
                if initial_state is None:
                    raise RuntimeError("environment reset returned no state")
                native_reward_names = await owner.call(episode.reward_names)
                if self._judges.names & native_reward_names:
                    raise ValueError("configured judge is already declared by the environment")
                timings["env"] += time.monotonic() - started
                prompt = self.renderer.initial(conditioning)
                if (
                    len(prompt) > min(self.max_input, self.max_prompt)
                    or len(prompt) >= self.max_sequence
                ):
                    raise ValueError(
                        "initial prompt exceeds an actual input/sequence budget; no sampled action"
                    )
                trace = list(prompt)
                response, rewards, masks, logprobs, spans, components = [], [], [], [], [], []
                input_ids = list(prompt)
                scalar = 0.0
                for turn in range(self.max_turns):
                    request_params = dict(parameters)
                    request_params["max_tokens"] = min(
                        integer(parameters.get("max_tokens"), "max_tokens", minimum=1),
                        self.max_sequence - len(input_ids),
                    )
                    validate_sampling_request(
                        request_params,
                        contract=self.cfg.llenvs.sampling_contract,
                        phase=phase,
                        logprobs_mode=self.logprobs_mode,
                    )
                    request = {
                        "prompt_token_ids": [list(input_ids)],
                        "session_ids": [session],
                        "sampling_params": request_params,
                        "cache_salt": cache_salt,
                    }
                    before_call = tuple(input_ids)
                    provenance = {
                        **self.provenance,
                        "weight_version_at_request_start": getattr(
                            self.engine, "weight_version", None
                        ),
                    }
                    started = time.monotonic()
                    result = await self.engine.generate(request, model=self.policy_model_name)
                    timings["llm"] += time.monotonic() - started
                    if request["prompt_token_ids"] != [list(before_call)]:
                        raise ValueError("inference client mutated recorded request token IDs")
                    for name in ("responses", "response_ids", "stop_reasons"):
                        if not isinstance(result.get(name), list) or len(result[name]) != 1:
                            raise ValueError(f"inference {name} must contain one response")
                    sampled, raw, stop = (
                        result["response_ids"][0],
                        result["responses"][0],
                        result["stop_reasons"][0],
                    )
                    token_ids(sampled, "sampled output")
                    if (
                        not sampled
                        or len(sampled) > request_params["max_tokens"]
                        or any(token >= self.renderer.vocab_size for token in sampled)
                    ):
                        raise ValueError(
                            "sampled output is empty, outside vocabulary, or exceeds request budget"
                        )
                    if not isinstance(raw, str) or stop not in ("stop", "length"):
                        raise ValueError("unexpected inference text/stop reason")
                    probabilities = None
                    if request_params.get("logprobs") is not None:
                        supplied = result.get("response_logprobs")
                        if not isinstance(supplied, list) or len(supplied) != 1:
                            raise ValueError("missing chosen-token rollout logprobs")
                        probabilities = supplied[0]
                        validate_generation(
                            expected_prefix=trace,
                            actual_input_ids=before_call,
                            output_ids=sampled,
                            rollout_logprobs=probabilities,
                            vocab_size=self.renderer.vocab_size,
                        )
                    elif input_ids[: len(trace)] != trace:
                        raise ValueError("actual input does not preserve recorded token prefix")
                    generation_id = f"{session}:{turn}"
                    dense = [0.0] * len(sampled)
                    if self._scorer is not None:
                        started = time.monotonic()
                        scorer = await self._scorer.get()
                        scored = await score_generation(
                            scorer,
                            GenerationScoreInput(
                                occurrence_id=session,
                                instance_id=identity.instance_id,
                                repetition_id=identity.repetition_id,
                                generation_id=generation_id,
                                input_ids=before_call,
                                output_ids=sampled,
                                conditioning=conditioning,
                                provenance=provenance,
                            ),
                            weight=self.cfg.llenvs.token_scorer.weight,
                        )
                        dense = list(scored.rewards)
                        timings["scoring"] += time.monotonic() - started
                    before_transition = episode.state
                    if before_transition is None:
                        raise RuntimeError("environment lost its current state")
                    started = time.monotonic()
                    transition = await owner.call(episode.step, raw)
                    timings["env"] += time.monotonic() - started
                    native_reward_names.update(signal.name for signal in transition.rewards.signals)
                    signals = transition.rewards
                    if self.selected.judges and self.cfg.llenvs.judge_timing == "decision":
                        started = time.monotonic()
                        extra = await self._judges.score(
                            decision_request(
                                before_transition,
                                raw,
                                transition,
                                use_images=self.cfg.llenvs.judge_use_images,
                            ),
                            existing_names=native_reward_names,
                        )
                        timings["scoring"] += time.monotonic() - started
                        signals = SignalBundle((*signals.signals, *extra.signals))
                    decision_reward = reward_total(signals)
                    scalar = finite_number(scalar + decision_reward, "episode scalar reward")
                    dense[-1] = finite_number(
                        dense[-1] + decision_reward, "combined endpoint reward"
                    )
                    gap = input_ids[len(trace) :]
                    response.extend(gap)
                    rewards.extend([0.0] * len(gap))
                    masks.extend([0] * len(gap))
                    logprobs.extend([0.0] * len(gap))
                    start = len(response)
                    response.extend(sampled)
                    rewards.extend(dense)
                    masks.extend([1] * len(sampled))
                    if probabilities is not None:
                        logprobs.extend(probabilities)
                    spans.append(
                        {
                            "generation_id": generation_id,
                            "start": start,
                            "end": len(response),
                            "sampled_count": len(sampled),
                        }
                    )
                    components.append(
                        [
                            {"name": signal.name, "reward": signal.reward, "weight": signal.weight}
                            for signal in signals.signals
                        ]
                    )
                    trace = [*input_ids, *sampled]
                    conditioning.append({"role": "assistant", "content": raw})
                    if transition.done:
                        end_reason = (
                            "environment_terminated"
                            if transition.terminated
                            else "environment_truncated"
                        )
                        break
                    if stop == "length":
                        end_reason = "generation_limit"
                        break
                    if turn + 1 == self.max_turns:
                        end_reason = "decision_limit"
                        stop = "length"
                        break
                    observation = feedback_message(transition.next_state)
                    context_prefix = list(trace)
                    stop_strings = parameters.get("stop")
                    if (
                        stop_strings
                        and self.cfg.generator.append_eos_token_after_stop_str_in_multi_turn
                        and raw.endswith(tuple(stop_strings))
                        and trace[-1] != self.renderer.eos_id
                    ):
                        context_prefix.append(self.renderer.eos_id)
                    next_ids = self.renderer.extend(context_prefix, observation)
                    if len(next_ids) > self.max_input or len(next_ids) >= self.max_sequence:
                        end_reason = (
                            "context_limit" if len(next_ids) > self.max_input else "sequence_limit"
                        )
                        stop = "length"
                        break
                    conditioning.append(observation)
                    input_ids = next_ids
                if self.selected.judges and self.cfg.llenvs.judge_timing == "episode":
                    started = time.monotonic()
                    extra = await self._judges.score(
                        episode_request(
                            initial_state,
                            conditioning,
                            transition,
                            end_reason=end_reason,
                            use_images=self.cfg.llenvs.judge_use_images,
                        ),
                        existing_names=native_reward_names,
                    )
                    timings["scoring"] += time.monotonic() - started
                    value = reward_total(extra)
                    scalar = finite_number(scalar + value, "episode scalar reward")
                    rewards[-1] = finite_number(
                        rewards[-1] + value, "combined episode endpoint reward"
                    )
                    components[-1].extend(
                        {"name": signal.name, "reward": signal.reward, "weight": signal.weight}
                        for signal in extra.signals
                    )
                metrics: dict[str, Any] = {
                    "llenvs/end_reason": end_reason,
                    "llenvs/reward_components": components,
                }
                if self.estimator != "grpo":
                    metrics["llenvs/attribution"] = {
                        "instance_id": group,
                        "repetition_id": identity.repetition_id,
                        "response_length": len(response),
                        "sampled_spans": spans,
                    }
                if self.cfg.generator.apply_overlong_filtering and stop != "stop":
                    masks = [0] * len(masks)
                if (
                    self.estimator == "grpo"
                    and self.cfg.generator.zero_reward_on_non_stop
                    and stop != "stop"
                ):
                    scalar = 0.0
                return {
                    "prompt_token_ids": prompt,
                    "response_ids": response,
                    "rewards": scalar if self.estimator == "grpo" else rewards,
                    "loss_masks": masks,
                    "rollout_logprobs": logprobs,
                    "stop_reasons": stop,
                    "env_metrics": metrics,
                    "trajectory_generation_times": time.monotonic() - scheduled,
                    "trajectory_time_splits": timings,
                }
            finally:
                results = await asyncio.gather(
                    owner.aclose(),
                    asyncio.wait_for(self.engine.finish_session(session), CLEANUP_TIMEOUT),
                    return_exceptions=True,
                )
                errors = [result for result in results if isinstance(result, Exception)]
                if errors:
                    failure = ExceptionGroup("episode cleanup failed", errors)
                    # Done callbacks remove episodes from the live set before
                    # another failed call may start close. Retain cleanup-only
                    # diagnostics independently of that scheduling order.
                    self._cleanup_errors.append(failure)
                    raise failure

    async def _close(self) -> None:
        tasks = list(self._episodes)
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        try:
            if tasks:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)),
                    CLEANUP_TIMEOUT + 1,
                )
            if self._cleanup_errors:
                errors, self._cleanup_errors = self._cleanup_errors, []
                raise ExceptionGroup("generator episode cleanup failed", errors)
        finally:
            try:
                closing = [self._judges.aclose()]
                if self._scorer is not None:
                    closing.append(self._scorer.aclose())
                results = await asyncio.gather(*closing, return_exceptions=True)
                errors = [result for result in results if isinstance(result, Exception)]
                if errors:
                    raise ExceptionGroup("generator scorer/judge cleanup failed", errors)
            finally:
                self._executor.shutdown(wait=False, cancel_futures=False)

    async def aclose(self) -> None:
        self._check_loop()
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)
