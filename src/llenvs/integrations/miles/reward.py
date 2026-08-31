"""Custom reward model for miles (``--custom-rm-path``).

Launch flag::

    --custom-rm-path llenvs.integrations.miles.reward.reward_func

Scoring per sample:

1. ``sample.metadata["reward"]`` short-circuits — the agent function already
   computed the episode reward (multi-turn path). Under session server v2 with
   the default postprocessor the RM is skipped entirely; this branch covers
   the merge/v1 path.
2. Otherwise the sample is scored against the environment via a cached
   ``Scorer`` using ``sample.metadata["task_index"]`` (single-turn RLVR path).
   The environment's extractors and reward functions run unchanged.

Scoring is serialized with a lock: the cached Scorer holds one shared
environment whose reset/step pairs must not interleave.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from llenvs.integrations.miles import config as miles_config

logger = logging.getLogger(__name__)

_score_lock = threading.Lock()


def _score_sample(sample: Any) -> float:
    metadata = getattr(sample, "metadata", None) or {}
    if "reward" in metadata:
        return float(metadata["reward"])
    if "task_index" not in metadata:
        raise ValueError(
            "Sample metadata has neither 'reward' nor 'task_index'. The miles reward "
            "function needs a task_index to score against the environment; export "
            "prompt data with llenvs.integrations.miles.data so rows carry it."
        )
    try:
        scorer = miles_config.get_scorer(metadata)
    except TypeError:
        logger.error(
            "Cannot score task %s: the selected environment is multi-turn, which the "
            "single-turn Scorer rejects. Multi-turn episodes must be rewarded by the "
            "agent function (metadata['reward']); returning 0.0.",
            metadata.get("task_index"),
        )
        return 0.0
    with _score_lock:
        result = scorer.score(int(metadata["task_index"]), sample.response or "")
    return float(result.total)


async def reward_func(args: Any, samples: Any, **kwargs: Any) -> float | list[float]:
    """Score one sample or a batch (miles calls both ways; v2 always batches).

    ``args`` may be ``None`` (miles debug replay).
    """
    if isinstance(samples, list):
        return [await asyncio.to_thread(_score_sample, s) for s in samples]
    return await asyncio.to_thread(_score_sample, samples)
