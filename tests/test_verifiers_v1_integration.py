"""Integration tests for the verifiers_v1 adapter against a real installation.

Dormant when verifiers (with the v1 API) is not installed — the unit suite
in tests/test_verifiers_v1.py covers the adapter on fakes. With verifiers
present, these register tiny echo taskset plugins in ``sys.modules`` (a
taskset id resolves to an installed module exporting its classes via
``__all__``) and drive the real loader/trace/scoring path end to end.
"""

from __future__ import annotations

import sys
import types

import pytest

vf = pytest.importorskip("verifiers.v1")

from llenvs.adapters.verifiers_v1 import (  # noqa: E402
    VerifiersV1Adapter,
    VerifiersV1MultiTurnEnvironment,
    VerifiersV1SingleTurnEnvironment,
)
from llenvs.core.state import Action  # noqa: E402

# ── Echo taskset plugins ────────────────────────────────────────────


class EchoData(vf.TaskData):
    answer: str = ""


class EchoTask(vf.Task[EchoData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward
    async def exact(self, trace: vf.Trace) -> float:
        return 1.0 if trace.last_reply == trace.task.data.answer else 0.0


class RelayTask(vf.Task[EchoData]):
    """Echo task without the single-turn stop, for the multi-turn relay."""

    @vf.reward
    async def exact(self, trace: vf.Trace) -> float:
        return 1.0 if trace.last_reply == trace.task.data.answer else 0.0


class EchoConfig(vf.TasksetConfig):
    num_tasks: int = 3


class EchoTaskset(vf.Taskset[EchoTask, EchoConfig]):
    def load(self):
        for i in range(self.config.num_tasks):
            yield EchoTask(
                EchoData(
                    idx=i,
                    prompt=f"Echo the word: word-{i}",
                    system_prompt="Echo exactly.",
                    answer=f"word-{i}",
                )
            )


class EchoRelayTaskset(vf.Taskset[RelayTask, EchoConfig]):
    def load(self):
        for i in range(self.config.num_tasks):
            yield RelayTask(EchoData(idx=i, prompt=f"Echo the word: word-{i}", answer=f"word-{i}"))


class EchoEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()


class EchoRelayEnv(vf.Env[EchoEnvConfig]):
    """Two-turn relay: echo the first reply back, then end the exchange."""

    async def run(self, task, agents):
        async with agents.solver.interaction(task) as interaction:
            segment = await interaction.turn()
            await interaction.turn(f"you said: {segment.last_reply}")


@pytest.fixture(autouse=True, scope="module")
def _install_echo_plugins():
    single = types.ModuleType("llenvs_echo_taskset")
    single.EchoTaskset = EchoTaskset
    single.__all__ = ["EchoTaskset"]
    relay = types.ModuleType("llenvs_echo_relay")
    relay.EchoRelayTaskset = EchoRelayTaskset
    relay.EchoRelayEnv = EchoRelayEnv
    relay.__all__ = ["EchoRelayTaskset", "EchoRelayEnv"]
    sys.modules["llenvs_echo_taskset"] = single
    sys.modules["llenvs_echo_relay"] = relay
    yield
    sys.modules.pop("llenvs_echo_taskset", None)
    sys.modules.pop("llenvs_echo_relay", None)


# ── Tests ───────────────────────────────────────────────────────────


class TestSingleTurnIntegration:
    def test_end_to_end_correct_answer(self):
        env = VerifiersV1Adapter().get_environment("llenvs-echo-taskset")

        assert isinstance(env, VerifiersV1SingleTurnEnvironment)
        assert len(env) == 3
        state, info = env.reset(options={"task_index": 1})
        assert state.observation.prompt == "Echo the word: word-1"
        assert info["system_prompt"] == "Echo exactly."

        result = env.step(state, Action.from_text("word-1"))

        assert result.terminated is True
        assert [s.name for s in result.rewards.signals] == ["exact"]
        assert result.rewards.total == pytest.approx(1.0)

    def test_end_to_end_wrong_answer(self):
        env = VerifiersV1Adapter().get_environment("llenvs-echo-taskset")
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("wrong"))

        assert result.rewards.total == pytest.approx(0.0)

    def test_size_and_seed(self):
        env = VerifiersV1Adapter().get_environment("llenvs-echo-taskset", size=2, seed=7)

        assert len(env) == 2
        keys = set()
        for i in range(2):
            _, info = env.reset(options={"task_index": i})
            keys.add(info["task_key"])
        assert len(keys) == 2

    def test_taskset_params(self):
        env = VerifiersV1Adapter().get_environment(
            "llenvs-echo-taskset", taskset_params={"num_tasks": 5}
        )

        assert len(env) == 5


class TestMultiTurnIntegration:
    def test_relay_end_to_end(self):
        env = VerifiersV1Adapter().get_environment("llenvs-echo-relay", step_timeout=30.0)

        assert isinstance(env, VerifiersV1MultiTurnEnvironment)
        try:
            state, _ = env.reset(options={"task_index": 0})
            assert state.observation.prompt == "Echo the word: word-0"

            first = env.step(state, Action.from_text("hello"))
            assert first.terminated is False
            assert first.next_state.observation.state is not None
            assert first.next_state.observation.state.text == "you said: hello"

            second = env.step(first.next_state, Action.from_text("word-0"))
        finally:
            env.close()

        assert second.terminated is True
        assert second.rewards.total == pytest.approx(1.0)
        assert second.info["stop_condition"] == "user_closed"
        assert second.info["episode_ok"] is True

    def test_trace_stop_hook_ends_episode(self):
        """The single-turn @vf.stop refuses the relay's second turn offline."""
        relay = sys.modules["llenvs_echo_relay"]
        stopped = types.ModuleType("llenvs_echo_stopped")
        stopped.EchoTaskset = EchoTaskset  # carries the single_turn stop
        stopped.EchoRelayEnv = relay.EchoRelayEnv
        stopped.__all__ = ["EchoTaskset", "EchoRelayEnv"]
        sys.modules["llenvs_echo_stopped"] = stopped
        try:
            env = VerifiersV1Adapter().get_environment("llenvs-echo-stopped", step_timeout=30.0)
            try:
                state, _ = env.reset(options={"task_index": 0})
                result = env.step(state, Action.from_text("word-0"))
            finally:
                env.close()
        finally:
            sys.modules.pop("llenvs_echo_stopped", None)

        assert result.terminated is True
        assert result.info["stop_condition"] == "single_turn"
        assert result.rewards.total == pytest.approx(1.0)


class TestAdapterIntegration:
    def test_list_environments_includes_builtin_tasksets(self):
        ids = VerifiersV1Adapter().list_environments()

        assert "openenv" in ids
