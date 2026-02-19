"""Utilities for running async code synchronously.

Shared by adapters that wrap async-first libraries (Aviary, verifiers, MARE).
"""

from __future__ import annotations

import asyncio
from typing import Any


def run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    If no event loop is running, uses ``asyncio.run()``. If called from
    within a running event loop, falls back to running in a separate
    thread via ``ThreadPoolExecutor`` to avoid ``RuntimeError``.

    Args:
        coro: An awaitable coroutine.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
