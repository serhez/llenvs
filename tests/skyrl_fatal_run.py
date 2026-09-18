"""CPU-only fatal-path child. Imports are harmless; main intentionally can os._exit."""

import asyncio
import inspect
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as Namespace


async def exercise(scenario, directory, emit, monkeypatch):
    from llenvs.core.config import EnvironmentFactory
    from llenvs.integrations.skyrl import _generator, _resources
    from llenvs.integrations.skyrl._config import TokenScorerConfig
    from llenvs.integrations.skyrl.scoring import GenerationTokenRewards
    from tests.skyrl_source import definitions, driver_namespace
    from tests.test_skyrl_generator import setup

    fixture = setup.__wrapped__(monkeypatch, directory)
    fixture.cfg.llenvs.max_active_episodes = 4
    fixture.cfg.environment.skyrl_gym.max_env_workers = 4
    fixture.cfg.environment.env_class = "llenvs"
    fixture.cfg.generator.inference_engine.backend = "vllm"
    fixture.cfg.generator.sampling_params = Namespace()
    fixture.cfg.trainer.algorithm.advantage_estimator = "llenvs_token_rtg"
    fixture.cfg.llenvs.token_scorer = TokenScorerConfig("fixture:scorer", "v1")
    # Exercise real timeout code with a short test-only deadline. An operation
    # that never returns must remain visibly unclosed when native os._exit runs.
    monkeypatch.setattr(_resources, "CLEANUP_TIMEOUT", 0.2)
    monkeypatch.setattr(_generator, "CLEANUP_TIMEOUT", 0.2)
    loop = asyncio.get_running_loop()
    started, release = threading.Event(), threading.Event()
    other_closed = asyncio.Event()
    all_requests_started = asyncio.Event()
    original_factory = EnvironmentFactory.create

    def create(config):
        env = original_factory(config)
        index = fixture.created.index(env)
        emit("env_created", resource=index)
        original_close, original_step = env.close, env.step

        def close():
            original_close()
            emit("env_closed", resource=index)
            if index == 1:
                loop.call_soon_threadsafe(other_closed.set)

        def step(*args):
            if scenario.startswith("blocked"):
                if index == 0:
                    emit("step_started")
                    started.set()
                    release.wait()  # The owned child is forcibly exited in the timeout case.
                    emit("step_finished")
                else:
                    assert started.wait(2), "blocked fixture did not start"
                    raise ValueError("fixture step failure")
            return original_step(*args)

        env.close, env.step = close, step
        return env

    monkeypatch.setattr(EnvironmentFactory, "create", create)

    class Scorer:
        reward_semantics = "prefix_causal_additive"

        def __init__(self):
            emit("scorer_created")

        async def __call__(self, generation):
            return GenerationTokenRewards(
                occurrence_id=generation.occurrence_id,
                generation_id=generation.generation_id,
                rewards=[0.0] * len(generation.output_ids),
            )

        async def aclose(self):
            emit("scorer_closed")

    monkeypatch.setattr(_resources, "resolve_factory", lambda _: Scorer)
    requests = 0
    original_generate = fixture.engine.generate

    async def generate(request, *, model):
        nonlocal requests
        requests += 1
        if requests == (4 if scenario == "concurrent" else 2):
            all_requests_started.set()
        if scenario.startswith("blocked"):
            return await original_generate(request, model=model)
        await all_requests_started.wait()
        if scenario in ("outside_generator", "cancellation"):
            await asyncio.Event().wait()
        emit("inference_failed", resource=request["session_ids"][0])
        raise ValueError("fixture inference failure")

    async def finish_session(session):
        emit("session_finished", resource=session)
        if scenario == "session_failure":
            raise RuntimeError("fixture session release failure")

    fixture.engine.generate, fixture.engine.finish_session = generate, finish_session
    generator = fixture.generator()
    await (
        generator._scorer.get()
    )  # Real ScorerOwner owns this fixture even if inference fails first.

    class Logger:
        def error(self, message):
            emit("native_error", message=message)
            print(message, file=sys.stderr, flush=True)

    ns = driver_namespace(monkeypatch)
    ns.update(
        asyncio=asyncio,
        dataclass=dataclass,
        inspect=inspect,
        os=os,
        sys=sys,
        time=time,
        traceback=traceback,
        logger=Logger(),
    )
    definitions(
        "skyrl/train/fully_async_trainer.py",
        {"_RolloutStat", "_AsyncStalenessManager", "GeneratedOutputGroup"},
        ns,
    )
    definitions(
        "skyrl/train/fully_async_trainer.py",
        {"_run_generate_for_a_group_loop"},
        ns,
        owner="FullyAsyncRayPPOTrainer",
    )
    # Task preparation/sampling are data fixtures, not claims about the native
    # loader. The native failure/cancellation body and staleness manager are real.
    ns["prepare_generator_input"] = lambda *args: (fixture.batch(), ["task", "task"])
    ns["get_sampling_params_for_backend"] = lambda *args: fixture.params
    pulls = 0

    async def next_data():
        nonlocal pulls
        index, pulls = pulls, pulls + 1
        if scenario == "outside_generator" and index == 1:
            await all_requests_started.wait()
            raise RuntimeError("fixture native dataloader failure")
        return [{"fixture": True}]

    manager = ns["_AsyncStalenessManager"](2, 1, 1)
    trainer = Namespace(
        cfg=fixture.cfg,
        global_step=1,
        generator=generator,
        async_train_dataloader=Namespace(get_next_non_consumed_data=next_data),
        _staleness_manager=manager,
    )
    tasks = [
        asyncio.create_task(ns["_run_generate_for_a_group_loop"](trainer, asyncio.Queue()))
        for _ in range(2 if scenario in ("concurrent", "outside_generator") else 1)
    ]
    try:
        if scenario == "cancellation":
            await all_requests_started.wait()
            tasks[0].cancel()
        elif scenario == "blocked_release":
            await other_closed.wait()
            emit("step_released")
            release.set()
        await asyncio.gather(*tasks)
        if scenario != "cancellation":
            emit("returned_from_native_worker")
    finally:
        emit("outer_finally")  # Must not run after a real native hard exit.
    emit("cancelled_cleanly", running=manager._stat.running)


def main():
    if os.environ.get("LLENVS_SKYRL_FATAL_TEST") != "1" or not __debug__:
        raise RuntimeError("requires explicit CPU fatal-test opt-in and enabled assertions")
    scenario, target = sys.argv[1:]
    directory = Path(target)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("requires an existing owned absolute directory")
    import pytest

    lock = threading.Lock()
    with (directory / "events.jsonl").open("x", buffering=1) as stream:

        def emit(event, **details):
            with lock:
                stream.write(json.dumps({"event": event, **details}) + "\n")
                stream.flush()

        with pytest.MonkeyPatch.context() as monkeypatch:
            asyncio.run(exercise(scenario, directory, emit, monkeypatch))


if __name__ == "__main__":
    main()
