"""The ``llenvs-env`` taskset: one verifiers v1 task per llenvs task.

``LLEnvsTaskset`` reads an llenvs ``EvalConfig`` YAML, selects one of its
environments, and yields an ``LLEnvsTask`` per task index (the prompt and
ground truth come from ``DatasetProvider``). Single-turn environments score
offline through the cached ``Scorer`` (``llenvs_total``); multi-turn
environments are played by ``LLEnvsEnv`` (see ``llenvs_env.env``), which
records the episode's rewards itself and leaves ``llenvs_total`` at 0.0.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem
from llenvs_env import _config, _relay
from llenvs_env._vf import parse_message, vf

__all__ = ["LLEnvsTasksetConfig", "LLEnvsData", "LLEnvsTask", "LLEnvsTaskset"]


class LLEnvsTasksetConfig(vf.TasksetConfig):
    """``--env.taskset.*`` knobs for the ``llenvs-env`` taskset."""

    config: Path | None = None
    """Path to the llenvs ``EvalConfig`` YAML (environments, extractors, prompts).
    Required to load tasks; optional here because the verifiers CLI instantiates
    the narrowed config from the taskset id alone before reading the flags."""
    env_name: str | None = None
    """Which ``environments[]`` entry to use; required when the YAML lists several."""
    num_tasks: int | None = Field(None, ge=1)
    """Keep only the first ``num_tasks`` tasks (after ``shuffle_seed``)."""
    shuffle_seed: int | None = None
    """Shuffle task indices with this seed before ``num_tasks`` applies."""


class LLEnvsData(vf.TaskData):
    """Wire data of one llenvs task."""

    config_path: str
    """Resolved path of the llenvs config the task came from."""
    env_name: str
    """The selected llenvs environment's config name."""
    task_index: int
    """Task index inside the llenvs environment."""
    multi_turn: bool
    """Whether the environment is multi-turn (played by ``LLEnvsEnv``)."""
    answer: str | None = None
    """Ground truth when the environment exposes one (single-turn only)."""
    info: dict[str, Any] = Field(default_factory=dict)
    """JSON-serializable reset metadata (``episode_id`` excluded)."""


class LLEnvsTask(vf.Task[LLEnvsData, vf.State, vf.TaskConfig]):
    """A verifiers task wrapping one llenvs task index."""

    @property
    def key(self) -> str:
        return f"llenvs:{self.data.env_name}:{self.data.task_index}"

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        """A single-turn task is over after the first sampled reply."""
        return not self.data.multi_turn and trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def llenvs_total(self, trace: vf.Trace) -> float:
        """The llenvs weighted reward total for a single-turn reply.

        On a relayed episode ``LLEnvsEnv`` has already recorded the per-signal
        rewards (and left ``trace.info["llenvs"]``), so this contributes 0.0 and
        ``trace.reward`` stays equal to the llenvs total.
        """
        if "llenvs" in trace.info:
            return 0.0
        if self.data.multi_turn:
            raise RuntimeError(
                f"Task {self.key} belongs to a multi-turn llenvs environment, which cannot be "
                "scored from a single reply; run it under the llenvs-env Env "
                "(the taskset's bundled Env, or --env.id llenvs-env)."
            )
        scorer = _config.get_scorer(self.data.config_path, self.data.env_name)
        result = await asyncio.to_thread(scorer.score, self.data.task_index, trace.last_reply)
        trace.record_metrics({f"llenvs/{name}": value for name, value in result.signals.items()})
        return result.total


class LLEnvsTaskset(vf.Taskset[LLEnvsTask, LLEnvsTasksetConfig]):
    """Yields one ``LLEnvsTask`` per task index of the selected llenvs environment."""

    def load(self) -> Iterable[LLEnvsTask]:
        cfg = self.config
        if cfg.config is None:
            raise ValueError(
                "llenvs-env needs the llenvs config path: set --env.taskset.config <eval.yaml> "
                "(TOML: env.taskset.config)."
            )
        eval_cfg = _config.load_eval_config(cfg.config)
        env_cfg = _config.select_environment(eval_cfg, cfg.env_name)
        system_prompt = _config.resolve_config_system_prompt(eval_cfg, env_cfg)
        config_path = _config.config_key(cfg.config)
        env = _config.create_environment(cfg.config, cfg.env_name)
        try:
            provider = DatasetProvider(env)
            multi_turn = bool(env.spec.is_multi_turn)
            indices = list(range(len(provider)))
            if cfg.shuffle_seed is not None:
                random.Random(cfg.shuffle_seed).shuffle(indices)
            if cfg.num_tasks is not None:
                indices = indices[: cfg.num_tasks]
            for idx, task_index in enumerate(indices):
                data = _task_data(
                    provider[task_index],
                    idx=idx,
                    env_name=env_cfg.name,
                    config_path=config_path,
                    multi_turn=multi_turn,
                    system_prompt=system_prompt,
                )
                yield LLEnvsTask(data, cfg.task)
        finally:
            _relay.close_environment(env)


def _task_data(
    item: TaskItem,
    *,
    idx: int,
    env_name: str,
    config_path: str,
    multi_turn: bool,
    system_prompt: str | None,
) -> LLEnvsData:
    if item.images:
        raise NotImplementedError(
            f"Task {env_name}#{item.task_index} contains images; the llenvs-env relay is "
            "text-only, so vision environments are not supported."
        )
    prompt: str | vf.Messages
    if item.messages:
        prompt = [parse_message(m) for m in _relay.history_messages(item.prompt, item.messages)]
    else:
        prompt = item.prompt
    parts = [system_prompt] if system_prompt else []
    if item.available_tools:
        parts.append(_relay.default_parser().format_tools(tuple(item.available_tools)))
    return LLEnvsData(
        idx=idx,
        name=f"{env_name}#{item.task_index}",
        prompt=prompt,
        system_prompt="\n\n".join(parts) or None,
        config_path=config_path,
        env_name=env_name,
        task_index=item.task_index,
        multi_turn=multi_turn,
        answer=item.ground_truth,
        info=_json_safe(item.metadata),
    )


def _json_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reset metadata without ``episode_id`` and without non-JSON values."""
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "episode_id":
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        out[key] = value
    return out
