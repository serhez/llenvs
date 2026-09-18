"""Complete text episode contracts using a recorded-token inference double."""

import asyncio
import copy
import importlib
import json
from dataclasses import replace
from types import SimpleNamespace as Namespace
from unittest.mock import AsyncMock, Mock

import pytest

from llenvs.core.config import (
    BackendFactory,
    EnvironmentFactory,
    EvalConfig,
    JudgeConfig,
    ModelConfig,
)
from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Observation, ObservationContent, State, StateMetadata
from llenvs.inference.protocol import GenerationResult
from llenvs.integrations.skyrl._config import LlenvsConfig, SelectedEnvironment, TokenScorerConfig
from llenvs.integrations.skyrl.data import _export_config, export_prompt_data
from llenvs.integrations.skyrl.scoring import GenerationTokenRewards
from tests.test_skyrl_rendering import Tokenizer
from tests.test_skyrl_resources import run_async


@pytest.fixture
def setup(monkeypatch, tmp_path):
    module = importlib.import_module("llenvs.integrations.skyrl._generator")
    config = EvalConfig.from_dict(
        {"environments": [{"name": "fixture", "adapter": "test", "seed": 42}]}
    )
    environment, prompt, fingerprint = _export_config(config, None)
    selected = SelectedEnvironment(tmp_path / "env.yaml", environment, prompt, fingerprint, ())
    events, created = [], []

    class Env:
        spec = EnvironmentSpec("fixture", "test", is_multi_turn=True)

        def __init__(self):
            self.closed = False
            self.turn = 0
            created.append(self)

        def __len__(self):
            return 1

        def reset(self, *, options):
            assert options == {"task_index": 0}
            return State(
                Observation("Task"),
                hidden=None,
                metadata=StateMetadata(step=0, episode_id="fixture"),
            ), {}

        def step(self, state, action):
            self.turn += 1
            events.append(("step", self, action.text))
            return StepResult(
                replace(
                    state, observation=Observation("Task", state=ObservationContent(text="Next"))
                ),
                SignalBundle.single(self.turn),
                terminated=self.turn == 2,
                extracted_action="parsed",
                resolved_action="resolved",
            )

        def close(self):
            assert not self.closed
            self.closed = True

    monkeypatch.setattr(EnvironmentFactory, "create", lambda cfg: Env())
    path = tmp_path / "tasks.jsonl"
    export_prompt_data(config, path)
    row = json.loads(path.read_text())
    created.clear()
    cfg = Namespace(
        llenvs=LlenvsConfig(max_active_episodes=2),
        environment=Namespace(skyrl_gym=Namespace(max_env_workers=2)),
        trainer=Namespace(
            max_prompt_length=64,
            algorithm=Namespace(advantage_estimator="llenvs_turn_grpo", max_seq_len=256),
        ),
        generator=Namespace(
            n_samples_per_prompt=2,
            eval_n_samples_per_prompt=1,
            max_turns=3,
            max_input_length=128,
            chat_template_kwargs={},
            chat_template=Namespace(name_or_path=None),
            use_conversation_multi_turn=True,
            batched=False,
            vision_language_generator=False,
            step_wise_trajectories=False,
            merge_stepwise_output=False,
            inference_engine=Namespace(engine_init_kwargs={}),
            append_eos_token_after_stop_str_in_multi_turn=True,
            use_cache_salt=True,
            apply_overlong_filtering=False,
            zero_reward_on_non_stop=False,
        ),
    )
    cfg.llenvs.sampling_contract = "unmodified"
    params = {"max_tokens": 8, "logprobs": 0, "min_tokens": 0, "temperature": 1.0}
    calls, session_turns = [], {}

    class Engine:
        weight_version = 7
        model_name = "policy"
        finish_session = AsyncMock()

        async def generate(self, request, *, model):
            assert model == "policy"
            calls.append(copy.deepcopy(request))
            session = request["session_ids"][0]
            turn = session_turns.get(session, 0)
            session_turns[session] = turn + 1
            await asyncio.sleep(0)
            return {
                "responses": ["first" if turn == 0 else "second"],
                "response_ids": [[20 + turn, 0]],
                "response_logprobs": [[-0.2, -0.3]],
                "stop_reasons": ["stop"],
            }

    engine = Engine()
    tokenizer = Tokenizer()

    def batch(phase="train"):
        n = 2 if phase == "train" else 1
        return {
            "prompts": [row["prompt"]] * n,
            "env_classes": ["llenvs"] * n,
            "env_extras": [
                {"data_source": row["data_source"], "llenvs": copy.deepcopy(row["llenvs"])}
                for _ in range(n)
            ],
            "trajectory_ids": [
                Namespace(instance_id=row["llenvs"]["task_id"], repetition_id=i) for i in range(n)
            ],
            "batch_metadata": Namespace(global_step=0, training_phase=phase),
        }

    def generator(*, model_context_length=256):
        return module.EpisodeGenerator(
            cfg,
            selected,
            tokenizer,
            engine,
            train_sampling_params=params,
            logprobs_mode="raw_logprobs",
            model_context_length=model_context_length,
            vocab_size=512,
            provenance={"model": "model-revision", "tokenizer": "tokenizer-revision"},
        )

    return Namespace(
        module=module,
        cfg=cfg,
        selected=selected,
        params=params,
        engine=engine,
        tokenizer=tokenizer,
        created=created,
        calls=calls,
        events=events,
        batch=batch,
        generator=generator,
    )


@run_async
async def test_two_turn_trace_has_exact_gaps_endpoints_and_native_id_order(setup):
    generator = setup.generator()
    batch = setup.batch()
    original = copy.deepcopy(batch)
    try:
        result = await generator.generate(batch)
    finally:
        await generator.aclose()
    assert batch == original
    assert len(setup.created) == 2 and all(env.closed and env.turn == 2 for env in setup.created)
    assert result["trajectory_ids"] == batch["trajectory_ids"]
    assert len(setup.calls) == 4
    for i, response in enumerate(result["response_ids"]):
        assert response == [20, 0, 5, 1, 3, *map(ord, "Next"), 0, 5, 1, 4, 21, 0]
        assert result["loss_masks"][i] == [1, 1] + [0] * 11 + [1, 1]
        assert result["rewards"][i] == [0, 1] + [0] * 11 + [0, 2]
        assert result["rollout_logprobs"][i] == [-0.2, -0.3] + [0] * 11 + [-0.2, -0.3]
        ledger = result["env_metrics"][i]["llenvs/attribution"]
        assert [(s["start"], s["end"]) for s in ledger["sampled_spans"]] == [(0, 2), (13, 15)]
    assert all(call["cache_salt"] == "policy@7" for call in setup.calls)
    assert len({call["session_ids"][0] for call in setup.calls}) == 2
    assert setup.engine.finish_session.await_count == 2


@run_async
async def test_same_task_new_generate_call_has_fresh_sessions_and_group_identity(setup):
    generator = setup.generator()
    try:
        first, second = await asyncio.gather(
            generator.generate(setup.batch()), generator.generate(setup.batch())
        )
    finally:
        await generator.aclose()
    a = {m["llenvs/attribution"]["instance_id"] for m in first["env_metrics"]}
    b = {m["llenvs/attribution"]["instance_id"] for m in second["env_metrics"]}
    assert len(a) == len(b) == 1 and a.isdisjoint(b)
    assert len({call["session_ids"][0] for call in setup.calls}) == 4
    assert first["trajectory_ids"] == second["trajectory_ids"]


@run_async
async def test_token_scorer_runs_before_each_transition_and_weight_applies_once(setup, monkeypatch):
    requests = []

    class Scorer:
        reward_semantics = "prefix_causal_additive"

        async def __call__(self, generation):
            requests.append(generation)
            setup.events.append(("score", generation.occurrence_id, generation.output_ids))
            assert all(message.get("content") != "second" for message in generation.conditioning)
            return GenerationTokenRewards(
                occurrence_id=generation.occurrence_id,
                generation_id=generation.generation_id,
                rewards=[0.5, 0],
            )

        aclose = AsyncMock()

    scorer = Scorer()
    factory = Mock(return_value=scorer)
    resources = importlib.import_module("llenvs.integrations.skyrl._resources")
    monkeypatch.setattr(resources, "resolve_factory", lambda _: factory)
    setup.cfg.trainer.algorithm.advantage_estimator = "llenvs_token_rtg"
    setup.cfg.llenvs.token_scorer = TokenScorerConfig("fixture:factory", "v1", weight=2)
    setup.cfg.llenvs.max_active_episodes = 1
    generator = setup.generator()
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert [event[0] for event in setup.events] == ["score", "step"] * 4
    assert len(requests) == 4
    assert result["rewards"][0] == [1, 1] + [0] * 11 + [1, 2]
    assert requests[1].input_ids[-2:] == (1, 4)
    factory.assert_called_once_with()
    scorer.aclose.assert_awaited_once_with()


@pytest.mark.parametrize("cap", ["turn", "context", "sequence"])
@run_async
async def test_declared_cap_keeps_prior_credit_and_omits_unused_observation(setup, cap):
    if cap == "turn":
        setup.cfg.generator.max_turns = 1
    elif cap == "context":
        setup.cfg.generator.max_input_length = 12
    generator = setup.generator(model_context_length=12 if cap == "sequence" else 256)
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert result["response_ids"] == [[20, 0], [20, 0]]
    assert result["rewards"] == [[0, 1], [0, 1]]
    assert all("limit" in metrics["llenvs/end_reason"] for metrics in result["env_metrics"])
    assert all(env.closed and env.turn == 1 for env in setup.created)


@pytest.mark.parametrize(
    "damage", ["empty", "ragged", "missing_logprobs", "sentinel", "request_mutation"]
)
@run_async
async def test_protocol_error_aborts_group_and_closes_all_owned_resources(setup, damage):
    original = setup.engine.generate

    async def generate(request, *, model):
        result = await original(request, model=model)
        if damage == "empty":
            result["response_ids"][0] = []
            result["response_logprobs"][0] = []
        elif damage == "ragged":
            result["response_logprobs"][0].pop()
        elif damage == "missing_logprobs":
            result["response_logprobs"] = None
        elif damage == "sentinel":
            result["response_logprobs"][0][0] = -9999
        else:
            request["prompt_token_ids"][0][0] = 255
        return result

    setup.engine.generate = generate
    generator = setup.generator()
    with pytest.raises(ValueError):
        await generator.generate(setup.batch())
    assert all(env.closed for env in setup.created)
    assert not generator._episodes
    with pytest.raises(RuntimeError, match="closed"):
        await generator.generate(setup.batch())
    await generator.aclose()


@run_async
async def test_close_does_not_reclassify_an_ordinary_episode_failure_as_cleanup(setup):
    generator = setup.generator()
    entered = asyncio.Event()

    async def episode():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ValueError("ordinary episode failure") from None

    task = asyncio.create_task(episode())
    generator._episodes.add(task)
    await entered.wait()
    await generator.aclose()
    assert isinstance(task.exception(), ValueError)


@run_async
async def test_close_retains_cleanup_errors_after_finished_tasks_leave_live_set(setup):
    import time

    from llenvs.integrations.skyrl._episode import Episode

    async def finish_session(session):
        raise RuntimeError("finished sibling session cleanup failed")

    setup.engine.finish_session = finish_session
    generator = setup.generator()
    batch = setup.batch()
    row = {"prompt": batch["prompts"][0], "env_class": "llenvs", **batch["env_extras"][0]}
    task = asyncio.create_task(
        generator._episode(
            Episode(setup.selected, row),
            batch["trajectory_ids"][0],
            "group",
            setup.params,
            "train",
            "cache-salt",
            time.monotonic(),
        )
    )
    generator._episodes.add(task)
    task.add_done_callback(generator._episodes.discard)
    with pytest.raises(ExceptionGroup, match="episode cleanup failed"):
        await task
    assert not generator._episodes
    with pytest.raises(ExceptionGroup, match="generator episode cleanup failed"):
        await generator.aclose()


@run_async
async def test_native_scalar_and_eval_sampling_are_not_turned_into_custom_credit(setup):
    setup.cfg.trainer.algorithm.advantage_estimator = "grpo"
    setup.cfg.llenvs.sampling_contract = "native"
    setup.params["temperature"] = 0.7
    generator = setup.generator()
    batch = setup.batch("eval")
    batch["sampling_params"] = {"max_tokens": 8, "temperature": 0, "logprobs": None}
    original = copy.deepcopy(batch["sampling_params"])
    try:
        result = await generator.generate(batch)
    finally:
        await generator.aclose()
    assert result["rewards"] == [3.0]
    assert result["rollout_logprobs"] is None
    assert all(call["sampling_params"] == original for call in setup.calls)
    assert all("llenvs/attribution" not in row for row in result["env_metrics"])


@pytest.mark.parametrize(
    "timing,calls,rewards", [("decision", 4, (1.5, 2.5)), ("episode", 2, (1, 2.5))]
)
@run_async
async def test_configured_judge_timing_places_scores_at_sampled_endpoints(
    setup, monkeypatch, timing, calls, rewards
):
    backend = Namespace(
        generate_chat=Mock(return_value=GenerationResult(text="[[2]]")), close=Mock()
    )
    monkeypatch.setattr(BackendFactory, "create", lambda _: backend)
    setup.selected = replace(
        setup.selected,
        judges=(
            JudgeConfig(ModelConfig(backend="openai", model="judge"), normalize=False, weight=0.25),
        ),
    )
    setup.cfg.llenvs.judge_timing = timing
    generator = setup.module.EpisodeGenerator(
        setup.cfg,
        setup.selected,
        setup.tokenizer,
        setup.engine,
        train_sampling_params=setup.params,
        logprobs_mode="raw_logprobs",
        model_context_length=256,
        vocab_size=512,
        provenance={"model": "model", "tokenizer": "tokenizer"},
    )
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert result["rewards"][0] == [0, rewards[0]] + [0] * 11 + [0, rewards[1]]
    assert backend.generate_chat.call_count == calls
    backend.close.assert_called_once()
    if timing == "episode":
        prompt = backend.generate_chat.call_args.args[0][-1].content
        assert all(text in prompt for text in ("first", "second", "environment_terminated"))


@run_async
async def test_initial_budget_failure_closes_environment_without_any_inference(setup):
    setup.cfg.trainer.max_prompt_length = 1
    generator = setup.generator()
    with pytest.raises(ValueError, match="no sampled action"):
        await generator.generate(setup.batch())
    assert not setup.calls and all(env.closed for env in setup.created)


@run_async
async def test_overlong_mask_does_not_erase_rewards_or_sampled_membership(setup):
    setup.cfg.generator.max_turns = 1
    setup.cfg.generator.apply_overlong_filtering = True
    generator = setup.generator()
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert result["loss_masks"] == [[0, 0], [0, 0]]
    assert result["rewards"] == [[0, 1], [0, 1]]
    assert all(
        row["llenvs/attribution"]["sampled_spans"][0]["sampled_count"] == 2
        for row in result["env_metrics"]
    )


@run_async
async def test_inserted_eos_is_context_not_a_sampled_token_or_reward_target(setup):
    setup.params["stop"] = ["END"]
    original = setup.engine.generate

    async def generate(request, *, model):
        result = await original(request, model=model)
        if result["responses"] == ["first"]:
            result["responses"] = ["firstEND"]
            result["response_ids"] = [[20]]
            result["response_logprobs"] = [[-0.2]]
        return result

    setup.engine.generate = generate
    generator = setup.generator()
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert result["response_ids"][0][:3] == [20, 0, 5]
    assert result["loss_masks"][0][:3] == [1, 0, 0]
    assert result["rewards"][0][:3] == [1, 0, 0]
    assert result["rollout_logprobs"][0][:3] == [-0.2, 0, 0]
    assert result["env_metrics"][0]["llenvs/attribution"]["sampled_spans"][0]["end"] == 1


@run_async
async def test_live_episode_bound_is_shared_by_concurrent_generate_calls(setup):
    original = setup.engine.generate
    peaks = []

    async def generate(request, *, model):
        peaks.append(sum(not env.closed for env in setup.created))
        await asyncio.sleep(0.001)
        return await original(request, model=model)

    setup.engine.generate = generate
    generator = setup.generator()
    try:
        results = await asyncio.gather(*(generator.generate(setup.batch()) for _ in range(4)))
    finally:
        await generator.aclose()
    assert len(results) == 4 and len(setup.created) == 8
    assert max(peaks) == 2 and all(env.closed for env in setup.created)


@run_async
async def test_concurrent_failure_and_cancellation_drain_without_deadlock(setup):
    entered = asyncio.Event()
    calls = 0

    async def generate(request, *, model):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            raise ValueError("fatal inference protocol error")
        await entered.wait()
        await asyncio.Event().wait()

    setup.engine.generate = generate
    generator = setup.generator()
    results = await asyncio.wait_for(
        asyncio.gather(
            generator.generate(setup.batch()),
            generator.generate(setup.batch()),
            return_exceptions=True,
        ),
        2,
    )
    assert any(isinstance(result, ValueError) for result in results)
    assert all(isinstance(result, BaseException) for result in results)
    assert not generator._episodes and all(env.closed for env in setup.created)
    await generator.aclose()


@run_async
async def test_weight_changes_do_not_relabel_old_rollout_logprobs(setup):
    original = setup.engine.generate

    async def generate(request, *, model):
        result = await original(request, model=model)
        if result["responses"] == ["second"]:
            result["response_logprobs"] = [[-0.8, -0.9]]
        setup.engine.weight_version += 1
        return result

    setup.engine.generate = generate
    generator = setup.generator()
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert all(row == [-0.2, -0.3] + [0] * 11 + [-0.8, -0.9] for row in result["rollout_logprobs"])
    assert all(call["cache_salt"] == "policy@7" for call in setup.calls)
    assert setup.engine.weight_version > 7  # A batch salt is not a per-token policy version.


@run_async
async def test_config_mutation_after_construction_does_not_change_queued_recipe(setup):
    generator = setup.generator()
    setup.cfg.generator.max_turns = 1
    setup.params["temperature"] = 0
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert all(
        len(row["llenvs/attribution"]["sampled_spans"]) == 2 for row in result["env_metrics"]
    )
    assert all(call["sampling_params"]["temperature"] == 1 for call in setup.calls)


@pytest.mark.parametrize("normalizer", [None, 1, 12])
@run_async
async def test_native_loss_normalizer_cannot_shorten_the_rollout(setup, normalizer):
    setup.cfg.trainer.algorithm.max_seq_len = normalizer
    generator = setup.generator()
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert generator.max_sequence == 256
    assert all(
        len(row["llenvs/attribution"]["sampled_spans"]) == 2 for row in result["env_metrics"]
    )


@run_async
async def test_model_vocabulary_can_exceed_tokenizer_vocabulary(setup):
    original = setup.engine.generate

    async def generate(request, *, model):
        result = await original(request, model=model)
        result["response_ids"] = [[300, 0]]  # Model vocab=512, tokenizer len=256.
        result["responses"] = [""]
        return result

    setup.engine.generate = generate
    generator = setup.generator()
    try:
        result = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert all(row.count(300) == 2 for row in result["response_ids"])
    assert any(300 in request["prompt_token_ids"][0] for request in setup.calls)
