"""Cancellation-safe ownership for blocking episodes and loop-bound scorers."""

import asyncio
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llenvs.integrations.skyrl._config import TokenScorerConfig


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@pytest.fixture
def resources():
    return importlib.import_module("llenvs.integrations.skyrl._resources")


@run_async
async def test_cancelled_step_finishes_before_close_and_does_not_block_loop(resources):
    started, release = threading.Event(), threading.Event()
    events = []

    def step():
        events.append("step")
        started.set()
        assert release.wait(2)
        events.append("finished")

    with ThreadPoolExecutor(max_workers=2) as executor:
        episode = SimpleNamespace(step=step, close=lambda: events.append("close"))
        owner = resources.AsyncEpisode(episode, executor)
        task = asyncio.create_task(owner.call(episode.step))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        closing = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0)
        assert "close" not in events
        release.set()
        await closing
        await owner.aclose()
    assert events == ["step", "finished", "close"]


@run_async
async def test_cleanup_deadline_reports_limit_but_keeps_deferred_close(resources, monkeypatch):
    monkeypatch.setattr(resources, "CLEANUP_TIMEOUT", 0.01)
    started, release = threading.Event(), threading.Event()
    closed = threading.Event()

    def step():
        started.set()
        assert release.wait(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = resources.AsyncEpisode(SimpleNamespace(close=closed.set), executor)
        task = asyncio.create_task(owner.call(step))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            with pytest.raises(RuntimeError, match="cleanup deadline"):
                await owner.aclose()
            assert not closed.is_set()
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 1)


@run_async
async def test_episode_failure_does_not_skip_close_or_swallow_close_failure(resources):
    env = SimpleNamespace(close=Mock(side_effect=RuntimeError("cleanup failed")))
    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = resources.AsyncEpisode(env, executor)
        with pytest.raises(RuntimeError, match="step failed"):
            await owner.call(Mock(side_effect=RuntimeError("step failed")))
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await owner.aclose()
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await owner.aclose()
    env.close.assert_called_once()


@run_async
async def test_scorer_constructed_once_on_calling_loop_and_closed_there(resources, monkeypatch):
    loop = asyncio.get_running_loop()
    events = []

    class Scorer:
        reward_semantics = "prefix_causal_additive"

        async def __call__(self, generation):
            return generation

        async def aclose(self):
            assert asyncio.get_running_loop() is loop
            events.append("close")

    async def factory(**kwargs):
        assert kwargs == {"key": "value"}
        assert asyncio.get_running_loop() is loop
        events.append("create")
        await asyncio.sleep(0)
        return Scorer()

    monkeypatch.setattr(resources, "resolve_factory", lambda path: factory)
    owner = resources.ScorerOwner(TokenScorerConfig("fixture:factory", "v1", {"key": "value"}))
    assert events == []
    scorers = await asyncio.gather(*(owner.get() for _ in range(8)))
    assert all(s is scorers[0] for s in scorers)
    await owner.aclose()
    await owner.aclose()
    assert events == ["create", "close"]
    with pytest.raises(RuntimeError, match="closed"):
        await owner.get()


@run_async
async def test_cancelled_factory_waiter_does_not_lose_resource(resources, monkeypatch):
    release = asyncio.Event()
    closed = []

    class Scorer:
        reward_semantics = "prefix_causal_additive"

        async def __call__(self, value):
            return value

        async def aclose(self):
            closed.append(True)

    async def factory():
        await release.wait()
        return Scorer()

    monkeypatch.setattr(resources, "resolve_factory", lambda path: factory)
    owner = resources.ScorerOwner(TokenScorerConfig("fixture:factory", "v1"))
    getting = asyncio.create_task(owner.get())
    await asyncio.sleep(0)
    getting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await getting
    release.set()
    await owner.aclose()
    assert closed == [True]


@run_async
async def test_invalid_scorer_is_closed_not_retried_or_substituted(resources, monkeypatch):
    closed = []

    class Invalid:
        reward_semantics = "value_estimate"

        async def __call__(self, value):
            return value

        async def aclose(self):
            closed.append(True)

    factory = Mock(return_value=Invalid())
    monkeypatch.setattr(resources, "resolve_factory", lambda path: factory)
    owner = resources.ScorerOwner(TokenScorerConfig("fixture:factory", "v1"))
    for _ in range(2):
        with pytest.raises(ValueError, match="prefix_causal_additive"):
            await owner.get()
    await owner.aclose()
    assert closed == [True]
    factory.assert_called_once()
