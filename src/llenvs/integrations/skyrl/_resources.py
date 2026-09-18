"""Driver-loop ownership and bounded cleanup, without changing native scheduling."""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from llenvs.integrations.skyrl._config import TokenScorerConfig, resolve_factory

CLEANUP_TIMEOUT = 10.0


class AsyncEpisode:
    """Serial calls on the shared native-sized environment executor.

    Cancellation cannot stop a blocking operation. Cleanup is queued behind it
    and retained even after the async deadline expires. External operation
    timeouts/leases remain necessary; hard process death cannot be covered.
    """

    def __init__(self, episode: Any, executor: ThreadPoolExecutor) -> None:
        self.episode = episode
        self._executor = executor
        self._pending: Future[Any] | None = None
        self._closing: Future[Any] | None = None

    async def call(self, method: Callable[..., Any], *args: Any) -> Any:
        if self._closing is not None:
            raise RuntimeError("episode is closed")
        if self._pending is not None and not self._pending.done():
            raise RuntimeError("concurrent operations on one episode are not supported")
        self._pending = self._executor.submit(method, *args)
        return await asyncio.shield(asyncio.wrap_future(self._pending))

    async def aclose(self) -> None:
        if self._closing is None:
            pending = self._pending

            def close_after_pending() -> None:
                if pending is not None:
                    try:
                        pending.result()
                    except BaseException:
                        # The original call owns its failure. It cannot bypass cleanup.
                        pass
                self.episode.close()

            self._closing = self._executor.submit(close_after_pending)
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(self._closing)), CLEANUP_TIMEOUT
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "episode cleanup deadline exceeded; deferred close remains queued, "
                "but this adapter requires operation timeouts/external resource leases"
            ) from exc


class ScorerOwner:
    """One fixed scorer constructed and closed on its actual driver event loop."""

    def __init__(self, config: TokenScorerConfig) -> None:
        self._config = copy.deepcopy(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._initializing: asyncio.Task[Any] | None = None
        self._closing: asyncio.Task[None] | None = None
        self._scorer: Any = None

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("scorer owner cannot move between event loops")

    async def _create(self) -> Any:
        scorer = resolve_factory(self._config.factory)(**self._config.kwargs)
        if inspect.isawaitable(scorer):
            scorer = await scorer
        self._scorer = scorer
        if (
            not callable(scorer)
            or getattr(scorer, "reward_semantics", None) != "prefix_causal_additive"
        ):
            raise ValueError("scorer must be callable with prefix_causal_additive reward semantics")
        return scorer

    async def get(self) -> Any:
        self._check_loop()
        if self._closing is not None:
            raise RuntimeError("scorer owner is closed")
        if self._initializing is None:
            self._initializing = asyncio.create_task(self._create())
        return await asyncio.shield(self._initializing)

    async def _close(self) -> None:
        if self._initializing is not None:
            try:
                await self._initializing
            except BaseException:
                # The get() caller receives the original initialization error.
                pass
        scorer, self._scorer = self._scorer, None
        close = getattr(scorer, "aclose", None)
        if close is not None:
            result = close()
            if not inspect.isawaitable(result):
                raise TypeError("resource-owning token scorers must provide async aclose()")
            await result

    async def aclose(self) -> None:
        self._check_loop()
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        try:
            await asyncio.wait_for(asyncio.shield(self._closing), CLEANUP_TIMEOUT)
        except TimeoutError as exc:
            raise RuntimeError("token scorer cleanup deadline exceeded") from exc
