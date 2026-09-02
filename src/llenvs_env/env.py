"""The ``llenvs-env`` Env: relays a multi-turn llenvs episode through a verifiers seat.

``LLEnvsEnv.run(task, agents)`` creates a fresh llenvs environment, resets it
to the task's index, and alternates policy replies (``interaction.turn``)
with ``env.step``. The framework sends the task prompt (a prompted task opens
with a bare ``turn()``); the relay feeds back the environment's observation
text, or Hermes ``<tool_response>`` blocks when the action carried tool calls.

Rewards: each step's ``SignalBundle`` is recorded per signal under
``llenvs/<name>`` (native weight when constant across turns; weight-0 signals
become metrics), and the per-turn totals land in ``trace.info["llenvs"]`` —
the turn-credit channel. ``trace.reward`` equals the sum of the per-turn
llenvs totals (``LLEnvsTask.llenvs_total`` contributes 0.0 on this path).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from llenvs.core.reward import SignalBundle
from llenvs_env import _config, _relay
from llenvs_env._vf import vf
from llenvs_env.taskset import LLEnvsTask

__all__ = ["LLEnvsEnvConfig", "LLEnvsEnv"]

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 100
"""Step cap when neither the env config nor the llenvs spec sets one."""


class LLEnvsEnvConfig(vf.EnvConfig):
    """The run's ``[env]`` block: one policy seat plus the relay's step cap."""

    agent: vf.AgentConfig = vf.AgentConfig()
    """The policy seat (``--env.agent.*``)."""
    max_steps: int | None = Field(None, ge=1)
    """Environment steps per episode; defaults to the llenvs spec's ``max_steps``, else 100."""


class LLEnvsEnv(vf.Env[LLEnvsEnvConfig]):
    """Plays an ``LLEnvsTask`` against its llenvs environment, one step per policy turn."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, LLEnvsTask):
            raise TypeError(
                f"{type(self).__name__} relays LLEnvsTask tasks only, got "
                f"{type(task).__name__}; pair it with the llenvs-env taskset."
            )
        data = task.data
        env = await asyncio.to_thread(_config.create_environment, data.config_path, data.env_name)
        try:
            await self._play(task, agents, env)
        finally:
            await asyncio.to_thread(_relay.close_environment, env)

    async def _play(self, task: LLEnvsTask, agents: vf.Agents, env: Any) -> None:
        data = task.data
        state, _ = await asyncio.to_thread(env.reset, options={"task_index": data.task_index})
        _relay.refuse_images(state)
        max_steps = self.config.max_steps or env.spec.max_steps or DEFAULT_MAX_STEPS
        episode = _EpisodeRecord(task_index=data.task_index, env_name=data.env_name)
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn()
            while not segment.terminated:
                reply = segment.last_reply
                if not reply:
                    episode.stop = "empty_action"
                    break
                action = _relay.action_from_reply(reply, state.observation.available_tools)
                result = await asyncio.to_thread(env.step, state, action)
                state = result.next_state
                episode.record(result.rewards)
                if result.terminated:
                    episode.stop = "env_terminated"
                    break
                if result.truncated:
                    episode.stop = "env_truncated"
                    break
                if episode.env_steps >= max_steps:
                    episode.stop = "max_steps"
                    break
                _relay.refuse_images(state)
                segment = await interaction.turn(_relay.feedback_text(state))
            episode.commit(interaction.trace)


@dataclass
class _EpisodeRecord:
    """Per-episode reward bookkeeping, committed to the trace once the loop ends."""

    task_index: int
    env_name: str
    stop: str | None = None
    turn_rewards: list[float] = field(default_factory=list)
    turn_signals: list[dict[str, float]] = field(default_factory=list)
    signal_weights: dict[str, float | None] = field(default_factory=dict)
    _raw_sums: dict[str, float] = field(default_factory=dict)
    _weighted_sums: dict[str, float] = field(default_factory=dict)

    @property
    def env_steps(self) -> int:
        return len(self.turn_rewards)

    def record(self, bundle: SignalBundle) -> None:
        self.turn_rewards.append(bundle.total)
        signals: dict[str, float] = {}
        for signal in bundle.signals:
            if signal.reward is None:
                continue
            signals[signal.name] = signals.get(signal.name, 0.0) + signal.reward
            self._raw_sums[signal.name] = self._raw_sums.get(signal.name, 0.0) + signal.reward
            self._weighted_sums[signal.name] = (
                self._weighted_sums.get(signal.name, 0.0) + signal.reward * signal.weight
            )
            if signal.name not in self.signal_weights:
                self.signal_weights[signal.name] = signal.weight
            elif self.signal_weights[signal.name] != signal.weight:
                self.signal_weights[signal.name] = None  # weight varied across turns
        self.turn_signals.append(signals)

    def commit(self, trace: vf.Trace) -> None:
        for name, raw_sum in self._raw_sums.items():
            key = f"llenvs/{name}"
            weight = self.signal_weights[name]
            if weight is None:
                logger.warning(
                    "Signal %r changed weight across turns of %s#%d; recording its weighted "
                    "sum under %r at weight 1.0",
                    name,
                    self.env_name,
                    self.task_index,
                    key,
                )
                trace.record_reward(key, self._weighted_sums[name], 1.0)
            elif weight == 0.0:
                trace.record_metric(key, raw_sum)
            else:
                trace.record_reward(key, raw_sum, weight)
        trace.record_metric("llenvs/env_steps", self.env_steps)
        trace.info["llenvs"] = {
            "turn_rewards": list(self.turn_rewards),
            "turn_signals": [dict(s) for s in self.turn_signals],
            "signal_weights": dict(self.signal_weights),
            "env_steps": self.env_steps,
            "stop": self.stop,
            "task_index": self.task_index,
            "env_name": self.env_name,
        }
        if self.stop is not None:
            trace.stop(self.stop)
