"""Tests for the verifiers v1 adapter.

verifiers is not installed in the test environment. All v1 access in the
adapter goes through the ``_V1Handle`` seam, so these tests build fake v1
modules mirroring the duck-typed surface (plain classes rather than
MagicMocks because the v1 surface is richer: traces, graph commits,
async scoring).
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import logging
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from llenvs.core.reward import RewardType, Signal
from llenvs.core.state import Action
from llenvs.inference import SamplingParams

# ── Fake verifiers v1 surface ───────────────────────────────────────


@dataclass
class FakeReward:
    """Mirror of verifiers.v1 Reward entries in ``trace.rewards``."""

    score: float | None
    weight: float = 1.0


class FakeMessage:
    role = ""

    def __init__(self, content: Any = None, **kwargs: Any) -> None:
        self.content = content
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeSystemMessage(FakeMessage):
    role = "system"


class FakeUserMessage(FakeMessage):
    role = "user"


class FakeAssistantMessage(FakeMessage):
    role = "assistant"

    def __init__(self, content: Any = None, tool_calls: Any = None, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self.tool_calls = tool_calls


@dataclass
class FakeNode:
    message: Any
    sampled: bool = False


@dataclass
class FakeTraceTask:
    type: str
    data: Any
    key: str = ""
    hash: str = ""


class FakeAgentConfig:
    def __init__(self, max_turns: int | None = None, sampling: Any = None) -> None:
        self.max_turns = max_turns
        self.sampling = sampling


@dataclass
class FakeAgentInfo:
    config: Any
    name: str = ""
    trainable: bool = True


class FakeTrace:
    """Message-graph trace mirroring the surface the adapter touches."""

    def __init__(self, *, task: Any, agent: Any, state: Any = None) -> None:
        self.task = task
        self.agent = agent
        self.state = state
        self.nodes: list[FakeNode] = []
        self.rewards: dict[str, FakeReward | None] = {}
        self.metrics: dict[str, float] = {}
        self.info: dict[str, Any] = {}
        self.stop_condition: str | None = None
        self.root_reply: str | None = None
        self.is_completed = False
        self.ok = True

    @property
    def num_turns(self) -> int:
        return sum(1 for n in self.nodes if n.sampled)

    @property
    def last_reply(self) -> str:
        if self.root_reply:
            return self.root_reply.strip()
        for node in reversed(self.nodes):
            if node.sampled and getattr(node.message, "role", "") == "assistant":
                return (node.message.content or "").strip()
        return ""

    def stop(self, condition: str) -> None:
        if self.stop_condition is None:
            self.stop_condition = condition

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
        self.rewards[name] = FakeReward(value, weight)

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value


@dataclass
class FakeResponse:
    id: str = "resp-0"
    created: int = 0
    model: str = ""
    message: Any = None
    finish_reason: str | None = "stop"
    tokens: Any = None


@dataclass
class FakeSegment:
    messages: list[Any] = field(default_factory=list)
    root_reply: str | None = None
    terminated: bool = False

    @property
    def last_reply(self) -> str:
        if self.root_reply:
            return self.root_reply.strip()
        for msg in reversed(self.messages):
            if getattr(msg, "role", "") == "assistant":
                return (msg.content or "").strip()
        return ""


class FakePendingTurn:
    def __init__(self, trace: FakeTrace, prompt: list[Any]) -> None:
        self._trace = trace
        self._prompt = prompt

    def commit(self, response: FakeResponse, tools: Any = None) -> int:
        for msg in self._prompt:
            self._trace.nodes.append(FakeNode(message=msg, sampled=False))
        self._trace.nodes.append(FakeNode(message=response.message, sampled=True))
        return len(self._trace.nodes) - 1


class FakeGraph:
    @staticmethod
    def prepare_turn(trace: FakeTrace, prompt: list[Any]) -> FakePendingTurn:
        return FakePendingTurn(trace, prompt)


@dataclass
class FakeTaskData:
    prompt: Any = None
    system_prompt: str | None = None
    image: Any = None


@dataclass
class FakeEnvInfo:
    id: str = ""


class FakeEpisode:
    def __init__(self, *, env: Any = None, task: Any = None) -> None:
        self.env = env
        self.task = task
        self.ok = False
        self.errors: list[Any] = []
        self.traces: list[Any] = []


class FakeSingleAgentEnv:
    """Marker for the default single-agent env class (the routing sentinel)."""


class FakeTask:
    """Behavior-class stand-in; ``score()`` records the scripted rewards."""

    def __init__(
        self,
        data: FakeTaskData,
        *,
        rewards: dict[str, tuple[float | None, float]] | None = None,
        metrics: dict[str, float] | None = None,
        trace_info: dict[str, Any] | None = None,
        reward_hooks: tuple[Any, ...] = (),
        metric_hooks: tuple[Any, ...] = (),
        stop_hooks: tuple[Any, ...] = (),
        key: str = "task-key",
        task_hash: str = "task-hash",
        config: Any = None,
    ) -> None:
        self.data = data
        self.key = key
        self.hash = task_hash
        self.config = config
        self._rewards = rewards or {}
        self._metrics = metrics or {}
        self._trace_info = trace_info or {}
        self._reward_hooks = list(reward_hooks)
        self._metric_hooks = list(metric_hooks)
        self._stop_hooks = list(stop_hooks)
        self.scored_traces: list[FakeTrace] = []

    def hooks(self, attr: str) -> list[Any]:
        return {
            "reward": self._reward_hooks,
            "metric": self._metric_hooks,
            "stop": self._stop_hooks,
        }.get(attr, [])

    def toolsets(self, config: Any) -> list[Any]:
        return []

    async def setup(self, trace: Any, runtime: Any) -> None:
        pass

    async def finalize(self, trace: Any, runtime: Any) -> None:
        pass

    async def score(self, trace: FakeTrace, runtime: Any = None) -> None:
        self.scored_traces.append(trace)
        for name, (value, weight) in self._rewards.items():
            if value is None:
                trace.rewards[name] = None
            else:
                trace.record_reward(name, value, weight)
        for name, value in self._metrics.items():
            trace.record_metric(name, value)
        trace.info.update(self._trace_info)


def _fake_parse_message(msg: Any) -> Any:
    if not isinstance(msg, dict):
        return msg
    role_cls = {
        "system": FakeSystemMessage,
        "user": FakeUserMessage,
        "assistant": FakeAssistantMessage,
    }[msg["role"]]
    return role_cls(content=msg.get("content"))


def _fake_hook_boundary(fn: Any, allow_trace: bool = False) -> Any:
    """Boundary is scripted per-hook via a ``_boundary`` attribute."""
    return getattr(fn, "_boundary", FakeTrace)


def _fake_discover_decorated(obj: Any, attr: str) -> list[Any]:
    return list(getattr(obj, f"_{attr}_hooks", []))


def _make_fake_v1(loaders: Any = None) -> Any:
    from llenvs.adapters.verifiers_v1 import _V1Handle

    vf = SimpleNamespace(
        Trace=FakeTrace,
        TraceTask=FakeTraceTask,
        AgentInfo=FakeAgentInfo,
        AgentConfig=FakeAgentConfig,
        Task=FakeTask,
        Segment=FakeSegment,
        Response=FakeResponse,
        SystemMessage=FakeSystemMessage,
        UserMessage=FakeUserMessage,
        AssistantMessage=FakeAssistantMessage,
        Episode=FakeEpisode,
        EnvInfo=FakeEnvInfo,
        SingleAgentEnv=FakeSingleAgentEnv,
    )
    return _V1Handle(
        vf=vf,
        graph=FakeGraph,
        parse_message=_fake_parse_message,
        hook_boundary=_fake_hook_boundary,
        loaders=loaders or SimpleNamespace(),
        state_cls=lambda task_type: dict,
        discover_decorated=_fake_discover_decorated,
    )


# ── Generic stubs ───────────────────────────────────────────────────


class StubExtractor:
    """Minimal AnswerExtractor: extracts the last token of the response."""

    def extract(self, text: str | None) -> tuple[str | None, dict[str, Any]]:
        if not text:
            return None, {"method": "stub"}
        return text.split()[-1], {"method": "stub"}


class StubExtraReward:
    _name = "extra"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.FORMAT

    def compute(self, state: Any, action: Any, next_state: Any) -> Signal:
        return Signal(name=self._name, reward_type=RewardType.FORMAT, reward=0.25)


def _make_single_turn(tasks: list[FakeTask], **kwargs: Any) -> Any:
    from llenvs.adapters.verifiers_v1 import VerifiersV1SingleTurnEnvironment

    return VerifiersV1SingleTurnEnvironment(_make_fake_v1(), "fake-taskset", list(tasks), **kwargs)


def _simple_task(**kwargs: Any) -> FakeTask:
    data = FakeTaskData(prompt="What is 2+2?", system_prompt="Be terse.")
    return FakeTask(data, **kwargs)


class ScriptedProvider:
    """Reply provider stub: records each call's (trace, prompt_messages)."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[list[Any]] = []
        self.traces: list[Any] = []

    async def __call__(self, trace: Any, prompt_messages: list[Any]) -> str:
        self.traces.append(trace)
        self.calls.append(list(prompt_messages))
        if self._replies:
            return self._replies.pop(0)
        return f"reply-{len(self.calls)}"


class FakeChatBackend:
    """ModelBackend stand-in recording generate_chat calls + executing thread."""

    def __init__(self, reply: str | None = "env reply") -> None:
        self.calls: list[tuple[list[Any], Any, threading.Thread]] = []
        self._reply = reply

    def generate_chat(self, messages: Any, sampling_params: Any = None) -> Any:
        self.calls.append((list(messages), sampling_params, threading.current_thread()))
        return SimpleNamespace(text=self._reply)


def _make_stub_agent(
    *,
    role: str = "agent",
    config: FakeAgentConfig | None = None,
    policy: Any = None,
    env_llm: Any = None,
    max_turns: int | None = None,
    episode_traces: list[Any] | None = None,
    v1: Any = None,
) -> Any:
    from llenvs.adapters.verifiers_v1 import _StubAgent

    return _StubAgent(
        v1 or _make_fake_v1(),
        role,
        config or FakeAgentConfig(),
        [] if episode_traces is None else episode_traces,
        policy_reply=policy or ScriptedProvider(),
        env_llm_reply=env_llm,
        max_turns=max_turns,
    )


def _make_interaction(task: FakeTask, **agent_kwargs: Any) -> Any:
    from llenvs.adapters.verifiers_v1 import _StubInteraction

    v1 = agent_kwargs.pop("v1", None) or _make_fake_v1()
    agent = _make_stub_agent(v1=v1, **agent_kwargs)
    return _StubInteraction(v1, agent, task)


# ── Chunk 1: probe + registration ───────────────────────────────────


class TestProbe:
    def test_import_error_when_verifiers_missing(self):
        from llenvs.adapters.verifiers_v1 import VerifiersV1Adapter

        adapter = VerifiersV1Adapter()
        with pytest.raises(ImportError, match="verifiers"):
            adapter._get_verifiers_v1()

    def test_adapter_name(self):
        from llenvs.adapters.verifiers_v1 import VerifiersV1Adapter

        assert VerifiersV1Adapter().name == "verifiers_v1"

    def test_adapter_satisfies_protocol(self):
        from llenvs.adapters.verifiers_v1 import VerifiersV1Adapter
        from llenvs.core.adapter import Adapter

        assert isinstance(VerifiersV1Adapter(), Adapter)

    def test_exported_from_adapters_package(self):
        from llenvs.adapters import VerifiersV1Adapter  # noqa: F401


class TestRegistration:
    @pytest.fixture
    def verifiers_v1_unregistered(self):
        from llenvs.core.registry import environment_registry

        try:
            original = environment_registry.get_adapter("verifiers_v1")
        except KeyError:
            original = None
        environment_registry.unregister_adapter("verifiers_v1")
        yield
        environment_registry.unregister_adapter("verifiers_v1")
        if original is not None:
            environment_registry.register_adapter(original)

    def test_not_registered_without_verifiers(self, verifiers_v1_unregistered):
        """The real probe fails in this venv, so registration skips the adapter."""
        from llenvs.adapters import _register_adapters
        from llenvs.core.registry import environment_registry

        _register_adapters()
        assert "verifiers_v1" not in environment_registry.list_adapters()

    def test_registered_with_stubbed_probe(self, monkeypatch, verifiers_v1_unregistered):
        from llenvs.adapters import VerifiersV1Adapter, _register_adapters
        from llenvs.core.registry import environment_registry

        monkeypatch.setattr(VerifiersV1Adapter, "_get_verifiers_v1", lambda self: object())
        _register_adapters()
        assert "verifiers_v1" in environment_registry.list_adapters()


# ── Chunk 2: single-turn environment ────────────────────────────────


class TestSingleTurnReset:
    def test_requires_task_index(self):
        env = _make_single_turn([_simple_task()])
        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_task_index_bounds(self):
        env = _make_single_turn([_simple_task()])
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 5})

    def test_str_prompt_observation(self):
        env = _make_single_turn([_simple_task()])
        state, info = env.reset(options={"task_index": 0})

        assert state.observation.prompt == "What is 2+2?"
        assert state.observation.task is not None
        assert state.observation.task.text == "What is 2+2?"
        assert state.observation.messages == ()
        assert info["task_index"] == 0
        assert info["taskset_id"] == "fake-taskset"
        assert info["task_key"] == "task-key"
        assert info["system_prompt"] == "Be terse."
        assert state.metadata.step == 0
        assert not state.metadata.is_terminal

    def test_messages_prompt_split(self):
        """First user message becomes the prompt; the rest replay as messages."""
        data = FakeTaskData(
            prompt=[
                FakeUserMessage("Turn 1"),
                FakeAssistantMessage("Reply 1"),
                FakeUserMessage("Turn 2"),
            ]
        )
        env = _make_single_turn([FakeTask(data)])
        state, _ = env.reset(options={"task_index": 0})

        assert state.observation.prompt == "Turn 1"
        assert state.observation.messages == (
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
        )

    def test_messages_prompt_system_extracted(self):
        """A system message inside the prompt wins over data.system_prompt."""
        data = FakeTaskData(
            prompt=[FakeSystemMessage("From row"), FakeUserMessage("Q")],
            system_prompt="From task",
        )
        env = _make_single_turn([FakeTask(data)])
        state, info = env.reset(options={"task_index": 0})

        assert state.observation.prompt == "Q"
        assert info["system_prompt"] == "From row"

    def test_promptless_task_refused(self):
        env = _make_single_turn([FakeTask(FakeTaskData(prompt=None))])
        with pytest.raises(ValueError, match="prompt"):
            env.reset(options={"task_index": 0})

    def test_episode_id_option_honored(self):
        env = _make_single_turn([_simple_task()])
        state, _ = env.reset(options={"task_index": 0, "episode_id": "ep-1"})
        assert state.metadata.episode_id == "ep-1"


class TestSingleTurnStep:
    def test_terminal_flags(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("4"))

        assert result.terminated is True
        assert result.truncated is False
        assert result.next_state.metadata.is_terminal
        assert result.next_state.metadata.step == 1

    def test_commit_shape(self):
        """The scored trace holds system+prompt unsampled, assistant sampled."""
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action.from_text("The answer is 4"))

        assert len(task.scored_traces) == 1
        trace = task.scored_traces[0]
        roles = [node.message.role for node in trace.nodes]
        sampled = [node.sampled for node in trace.nodes]
        assert roles == ["system", "user", "assistant"]
        assert sampled == [False, False, True]
        assert trace.nodes[-1].message.content == "The answer is 4"
        assert trace.stop_condition == "agent_completed"
        assert trace.agent.trainable is True

    def test_per_name_reward_signals(self):
        task = _simple_task(rewards={"acc": (0.8, 1.0), "fmt": (0.5, 0.25)})
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("4"))

        acc = result.rewards.by_name("acc")
        fmt = result.rewards.by_name("fmt")
        assert acc.reward == 0.8 and acc.weight == 1.0
        assert fmt.reward == 0.5 and fmt.weight == 0.25
        assert acc.reward_type == RewardType.OUTCOME
        # SignalBundle.total reproduces the verifiers weighted total exactly.
        assert result.rewards.total == pytest.approx(0.8 * 1.0 + 0.5 * 0.25)

    def test_unscored_reward_maps_to_none(self):
        """A seeded-but-unscored verifiers reward becomes a reward-less Signal."""
        task = _simple_task(rewards={"acc": (None, 1.0)})
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("4"))

        assert result.rewards.by_name("acc").reward is None
        assert result.rewards.total == 0.0

    def test_metrics_and_trace_info_in_info(self):
        task = _simple_task(
            rewards={"acc": (1.0, 1.0)},
            metrics={"turns": 1.0},
            trace_info={"note": "hi"},
        )
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("4"))

        assert result.info["verifiers_metrics"] == {"turns": 1.0}
        assert result.info["trace_info"] == {"note": "hi"}
        assert result.info["stop_condition"] == "agent_completed"
        assert result.next_state.metadata.info["verifiers_metrics"] == {"turns": 1.0}

    def test_runtime_requiring_hooks_surfaced(self):
        async def needs_rt(trace, runtime):  # mandatory runtime → skipped offline
            pass

        async def offline_ok(trace, runtime=None):
            pass

        task = _simple_task(reward_hooks=(needs_rt, offline_ok))
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("4"))

        assert result.info["runtime_skipped_signals"] == ["needs_rt"]

    def test_extra_rewards_after_native(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        env = _make_single_turn([task], extra_rewards=(StubExtraReward(),))
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("4"))

        assert [s.name for s in result.rewards.signals] == ["acc", "extra"]
        assert result.rewards.total == pytest.approx(1.25)

    def test_extractor_plumbed(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        env = _make_single_turn([task], answer_extractor=StubExtractor())
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action.from_text("answer is 4"))

        assert result.extracted_action == "4"
        assert result.info["extracted_answer"] == "4"
        assert result.info["extraction_metadata"] == {"method": "stub"}

    def test_pure_step_repeatable(self):
        """step() scores a fresh trace each call — same state twice is fine."""
        task = _simple_task(rewards={"acc": (0.5, 1.0)})
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        first = env.step(state, Action.from_text("4"))
        second = env.step(state, Action.from_text("4"))

        assert len(task.scored_traces) == 2
        assert first.rewards.total == second.rewards.total == 0.5

    def test_setup_override_warns(self, caplog):
        class TaskWithSetup(FakeTask):
            async def setup(self, trace, runtime):
                pass

        task = TaskWithSetup(FakeTaskData(prompt="Q"), rewards={"acc": (1.0, 1.0)})
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        with caplog.at_level(logging.WARNING):
            env.step(state, Action.from_text("4"))

        assert "setup" in caplog.text

    def test_tool_call_action_refused(self):
        from llenvs.core.tools import ToolCall

        task = _simple_task()
        env = _make_single_turn([task])
        state, _ = env.reset(options={"task_index": 0})
        action = Action.from_tool_call(ToolCall(id="c1", name="t", arguments={}))
        with pytest.raises(ValueError, match="tool"):
            env.step(state, action)


class TestSingleTurnSpec:
    def test_spec(self):
        env = _make_single_turn([_simple_task(), _simple_task()])
        spec = env.spec

        assert spec.name == "fake-taskset"
        assert spec.adapter == "verifiers_v1"
        assert spec.max_steps == 1
        assert spec.is_multi_turn is False
        assert spec.pure_step is True
        assert spec.supports_seed is False
        assert spec.metadata["dataset_size"] == 2

    def test_len(self):
        env = _make_single_turn([_simple_task(), _simple_task(), _simple_task()])
        assert len(env) == 3

    def test_prompts_and_tools_empty(self):
        env = _make_single_turn([_simple_task()])
        assert env.prompts == {}
        assert env.available_tools == ()

    def test_reward_functions_native_first(self):
        extra = StubExtraReward()
        env = _make_single_turn([_simple_task()], extra_rewards=(extra,))
        fns = env.reward_functions
        assert len(fns) == 2
        assert fns[-1] is extra
        # Native fn satisfies the RewardFunction protocol surface.
        assert fns[0].name == "verifiers_v1"
        assert isinstance(fns[0].reward_type, RewardType)

    def test_answer_extractor_settable(self):
        env = _make_single_turn([_simple_task()])
        assert env.answer_extractor is None
        extractor = StubExtractor()
        env.answer_extractor = extractor
        assert env.answer_extractor is extractor


# ── Chunk 3: stub agents / interactions ─────────────────────────────


class TestStubInteractionTurnRules:
    """The turn() contract, replicated from verifiers Interaction._turn."""

    def test_prompted_bare_turn_takes_opening_reply(self):
        provider = ScriptedProvider("The answer is 4")
        interaction = _make_interaction(_simple_task(), policy=provider)
        segment = asyncio.run(interaction.turn())

        assert segment.terminated is False
        assert segment.last_reply == "The answer is 4"
        # The provider saw the full first-turn prompt: system + task prompt.
        assert [m.role for m in provider.calls[0]] == ["system", "user"]
        assert provider.calls[0][1].content == "What is 2+2?"
        # Committed: prompt unsampled, assistant sampled.
        roles = [n.message.role for n in interaction.trace.nodes]
        sampled = [n.sampled for n in interaction.trace.nodes]
        assert roles == ["system", "user", "assistant"]
        assert sampled == [False, False, True]

    def test_prompted_first_turn_with_message_refused(self):
        interaction = _make_interaction(_simple_task())
        with pytest.raises(ValueError, match="prompt opens this exchange"):
            asyncio.run(interaction.turn("hi"))

    def test_promptless_first_turn_requires_message(self):
        interaction = _make_interaction(FakeTask(FakeTaskData(prompt=None)))
        with pytest.raises(ValueError, match="nothing to run a turn on"):
            asyncio.run(interaction.turn())

    def test_promptless_message_turn(self):
        task = FakeTask(FakeTaskData(prompt=None, system_prompt="Sys"))
        interaction = _make_interaction(task, policy=ScriptedProvider("hello"))
        segment = asyncio.run(interaction.turn("Open the game"))

        assert segment.last_reply == "hello"
        roles = [n.message.role for n in interaction.trace.nodes]
        assert roles == ["system", "user", "assistant"]
        assert interaction.trace.nodes[1].message.content == "Open the game"

    def test_second_bare_turn_refused(self):
        interaction = _make_interaction(_simple_task())

        async def exchange():
            await interaction.turn()
            await interaction.turn()

        with pytest.raises(ValueError, match="nothing to run a turn on"):
            asyncio.run(exchange())

    def test_dict_messages_normalized(self):
        task = FakeTask(FakeTaskData(prompt=None))
        interaction = _make_interaction(task)
        asyncio.run(interaction.turn([{"role": "user", "content": "hi"}]))

        user_node = interaction.trace.nodes[0]
        assert isinstance(user_node.message, FakeUserMessage)
        assert user_node.message.content == "hi"

    def test_multi_turn_commits_accumulate(self):
        """Later turns commit only the new user messages, never the prompt again."""
        provider = ScriptedProvider()
        interaction = _make_interaction(_simple_task(), policy=provider)

        async def exchange():
            await interaction.turn()
            return await interaction.turn("And 3+3?")

        segment = asyncio.run(exchange())
        roles = [n.message.role for n in interaction.trace.nodes]
        assert roles == ["system", "user", "assistant", "user", "assistant"]
        assert [m.role for m in provider.calls[1]] == ["user"]
        assert segment.last_reply == "reply-2"
        assert interaction.trace.num_turns == 2

    def test_turn_after_close_raises(self):
        interaction = _make_interaction(_simple_task())

        async def exchange():
            await interaction.close()
            await interaction.turn()

        with pytest.raises(RuntimeError, match="closed"):
            asyncio.run(exchange())

    def test_turn_after_over_raises(self):
        interaction = _make_interaction(_simple_task(), max_turns=0)

        async def exchange():
            first = await interaction.turn()
            assert first.terminated is True
            await interaction.turn()

        with pytest.raises(RuntimeError, match="the exchange is over"):
            asyncio.run(exchange())

    def test_root_reply_reset_before_provider(self):
        """A stale root_reply from a prior segment never leaks into the next."""

        class RootReplyObservingProvider(ScriptedProvider):
            observed: list[str | None] = []

            async def __call__(self, trace: Any, prompt_messages: list[Any]) -> str:
                type(self).observed.append(trace.root_reply)
                return await super().__call__(trace, prompt_messages)

        interaction = _make_interaction(_simple_task(), policy=RootReplyObservingProvider())
        interaction.trace.root_reply = "stale"
        segment = asyncio.run(interaction.turn())

        assert RootReplyObservingProvider.observed == [None]
        assert segment.root_reply is None

    def test_refused_turn_preserves_root_reply(self):
        interaction = _make_interaction(_simple_task(), max_turns=0)
        interaction.trace.root_reply = "prior"
        segment = asyncio.run(interaction.turn())

        assert segment.terminated is True
        assert interaction.trace.root_reply == "prior"


class TestStubInteractionLimits:
    def test_max_turns_refusal(self):
        """The refused turn commits nothing and never calls the provider."""
        provider = ScriptedProvider()
        interaction = _make_interaction(_simple_task(), policy=provider, max_turns=1)

        async def exchange():
            await interaction.turn()
            return await interaction.turn("again")

        segment = asyncio.run(exchange())
        assert segment.terminated is True
        assert segment.messages == []
        assert interaction.trace.stop_condition == "max_turns"
        assert len(provider.calls) == 1
        assert [n.message.role for n in interaction.trace.nodes] == [
            "system",
            "user",
            "assistant",
        ]

    def test_config_max_turns_fallback(self):
        config = FakeAgentConfig(max_turns=1)
        interaction = _make_interaction(_simple_task(), config=config)

        async def exchange():
            await interaction.turn()
            return await interaction.turn("again")

        segment = asyncio.run(exchange())
        assert segment.terminated is True
        assert interaction.trace.stop_condition == "max_turns"

    def test_explicit_max_turns_wins_over_config(self):
        config = FakeAgentConfig(max_turns=5)
        interaction = _make_interaction(_simple_task(), config=config, max_turns=1)

        async def exchange():
            await interaction.turn()
            return await interaction.turn("again")

        segment = asyncio.run(exchange())
        assert segment.terminated is True
        assert interaction.trace.stop_condition == "max_turns"


class TestStubInteractionStops:
    def test_trace_boundary_stop_fires(self):
        def stops_after_first(trace):
            return trace.num_turns >= 1

        task = _simple_task(stop_hooks=(stops_after_first,))
        interaction = _make_interaction(task)

        async def exchange():
            await interaction.turn()
            return await interaction.turn("again")

        segment = asyncio.run(exchange())
        assert segment.terminated is True
        assert interaction.trace.stop_condition == "stops_after_first"

    def test_async_stop_hook_awaited(self):
        async def async_stop(trace):
            return trace.num_turns >= 1

        task = _simple_task(stop_hooks=(async_stop,))
        interaction = _make_interaction(task)

        async def exchange():
            await interaction.turn()
            return await interaction.turn("again")

        segment = asyncio.run(exchange())
        assert segment.terminated is True
        assert interaction.trace.stop_condition == "async_stop"

    def test_non_trace_boundary_stop_ignored(self):
        """Request/Response-boundary stops need interception — never called offline."""

        def request_stop(request):
            raise AssertionError("Request-boundary stop must not run offline")

        request_stop._boundary = "request"
        task = _simple_task(stop_hooks=(request_stop,))
        interaction = _make_interaction(task)
        segment = asyncio.run(interaction.turn())

        assert segment.terminated is False

    def test_non_bool_stop_raises(self):
        def bad_stop(trace):
            return 1

        task = _simple_task(stop_hooks=(bad_stop,))
        interaction = _make_interaction(task)
        with pytest.raises(RuntimeError, match="must return bool"):
            asyncio.run(interaction.turn())

    def test_false_stop_continues(self):
        def never_stop(trace):
            return False

        task = _simple_task(stop_hooks=(never_stop,))
        interaction = _make_interaction(task)
        segment = asyncio.run(interaction.turn())

        assert segment.terminated is False
        assert interaction.trace.stop_condition is None


class TestStubInteractionClose:
    def test_close_scores_and_completes(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        interaction = _make_interaction(task)

        async def exchange():
            await interaction.turn()
            return await interaction.close()

        trace = asyncio.run(exchange())
        assert task.scored_traces == [trace]
        assert trace.stop_condition == "user_closed"
        assert trace.is_completed is True
        assert trace.ok is True
        assert trace.rewards["acc"].score == 1.0

    def test_close_idempotent(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        interaction = _make_interaction(task)

        async def exchange():
            first = await interaction.close()
            second = await interaction.close()
            return first, second

        first, second = asyncio.run(exchange())
        assert first is second
        assert len(task.scored_traces) == 1

    def test_close_keeps_earlier_stop(self):
        task = _simple_task()
        interaction = _make_interaction(task)

        async def exchange():
            await interaction.turn()
            interaction.trace.stop("agent_completed")
            return await interaction.close()

        trace = asyncio.run(exchange())
        assert trace.stop_condition == "agent_completed"

    def test_failed_close_skips_scoring(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        interaction = _make_interaction(task)

        async def exchange():
            await interaction.turn()
            interaction.fail(RuntimeError("boom"))
            return await interaction.close()

        trace = asyncio.run(exchange())
        assert task.scored_traces == []
        assert trace.ok is False
        assert trace.is_completed is True


class TestStubAgentInteractionContext:
    def test_context_closes_and_appends(self):
        episode_traces: list[Any] = []
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        agent = _make_stub_agent(episode_traces=episode_traces)

        async def exchange():
            async with agent.interaction(task) as interaction:
                await interaction.turn()
                return interaction.trace

        trace = asyncio.run(exchange())
        assert trace.is_completed is True
        assert task.scored_traces == [trace]
        assert episode_traces == [trace]

    def test_on_trace_fired_at_mint(self):
        seen: list[tuple[Any, int]] = []
        task = _simple_task()
        agent = _make_stub_agent()

        async def exchange():
            async with agent.interaction(
                task, on_trace=lambda t: seen.append((t, t.num_turns))
            ) as interaction:
                await interaction.turn()
                return interaction.trace

        trace = asyncio.run(exchange())
        assert seen == [(trace, 0)]

    def test_exception_marks_failed_and_reraises(self):
        episode_traces: list[Any] = []
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        agent = _make_stub_agent(episode_traces=episode_traces)

        async def exchange():
            async with agent.interaction(task) as interaction:
                await interaction.turn()
                raise RuntimeError("env crashed")

        with pytest.raises(RuntimeError, match="env crashed"):
            asyncio.run(exchange())

        # The failed rollout still completes and joins the episode, unscored.
        assert len(episode_traces) == 1
        trace = episode_traces[0]
        assert trace.ok is False
        assert trace.is_completed is True
        assert task.scored_traces == []

    def test_agent_identity_stamped_on_trace(self):
        task = _simple_task()
        config = FakeAgentConfig(max_turns=7)
        agent = _make_stub_agent(role="user_sim", config=config)
        agent.trainable = False

        async def exchange():
            async with agent.interaction(task) as interaction:
                return interaction.trace

        trace = asyncio.run(exchange())
        assert trace.agent.name == "user_sim"
        assert trace.agent.trainable is False
        assert trace.agent.config is config
        assert trace.task.key == task.key
        assert trace.task.hash == task.hash

    def test_borrowed_runtime_refused(self):
        agent = _make_stub_agent()

        async def exchange():
            async with agent.interaction(_simple_task(), runtime=object()):
                pass

        with pytest.raises(NotImplementedError, match="runtime"):
            asyncio.run(exchange())

    def test_borrowed_tools_refused(self):
        agent = _make_stub_agent()

        async def exchange():
            async with agent.interaction(_simple_task(), tools={"srv": object()}):
                pass

        with pytest.raises(NotImplementedError, match="tools"):
            asyncio.run(exchange())

    def test_provision_refused(self):
        agent = _make_stub_agent()

        async def provision():
            async with agent.provision(_simple_task()):
                pass

        with pytest.raises(NotImplementedError, match="provision"):
            asyncio.run(provision())


class TestStubAgentRun:
    def test_run_prompted_single_segment(self):
        episode_traces: list[Any] = []
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        agent = _make_stub_agent(policy=ScriptedProvider("4"), episode_traces=episode_traces)

        trace = asyncio.run(agent.run(task))
        assert trace.num_turns == 1
        assert trace.stop_condition == "agent_completed"
        assert trace.is_completed is True
        assert task.scored_traces == [trace]
        assert episode_traces == [trace]

    def test_run_promptless_raises(self):
        agent = _make_stub_agent()
        with pytest.raises(ValueError, match="nothing to run a turn on"):
            asyncio.run(agent.run(FakeTask(FakeTaskData(prompt=None))))

    def test_run_refused_turn_keeps_limit_stop(self):
        agent = _make_stub_agent(max_turns=0)
        trace = asyncio.run(agent.run(_simple_task()))

        assert trace.num_turns == 0
        assert trace.stop_condition == "max_turns"


class TestStubAgentProviders:
    def test_trainable_flip_switches_provider(self):
        policy = ScriptedProvider("from policy")
        env_llm = ScriptedProvider("from env_llm")
        task = FakeTask(FakeTaskData(prompt=None))
        agent = _make_stub_agent(policy=policy, env_llm=env_llm)

        async def exchange():
            from llenvs.adapters.verifiers_v1 import _StubInteraction

            interaction = _StubInteraction(agent.v1, agent, task)
            first = await interaction.turn("turn one")
            agent.trainable = False
            second = await interaction.turn("turn two")
            return first, second

        first, second = asyncio.run(exchange())
        assert first.last_reply == "from policy"
        assert second.last_reply == "from env_llm"
        assert len(policy.calls) == 1
        assert len(env_llm.calls) == 1

    def test_untrainable_without_env_llm_raises(self):
        agent = _make_stub_agent(env_llm=None)
        agent.trainable = False
        interaction_task = _simple_task()

        from llenvs.adapters.verifiers_v1 import _StubInteraction

        interaction = _StubInteraction(agent.v1, agent, interaction_task)
        with pytest.raises(RuntimeError, match="env_llm"):
            asyncio.run(interaction.turn())


class TestStubAgents:
    def test_getattr_iter_len(self):
        from llenvs.adapters.verifiers_v1 import _StubAgents

        alice = _make_stub_agent(role="alice")
        bob = _make_stub_agent(role="bob")
        agents = _StubAgents({"alice": alice, "bob": bob})

        assert agents.alice is alice
        assert agents.bob is bob
        assert list(agents) == [alice, bob]
        assert len(agents) == 2

    def test_unknown_role_lists_available(self):
        from llenvs.adapters.verifiers_v1 import _StubAgents

        agents = _StubAgents({"alice": _make_stub_agent(role="alice")})
        with pytest.raises(AttributeError, match="alice"):
            agents.unknown


class TestEnvLLMProvider:
    def _make_trace(self, *, system_prompt: str | None = "Sys", sampling: Any = None) -> FakeTrace:
        data = FakeTaskData(prompt="Q", system_prompt=system_prompt)
        return FakeTrace(
            task=FakeTraceTask(type="FakeTask", data=data),
            agent=FakeAgentInfo(config=FakeAgentConfig(sampling=sampling)),
        )

    def _provider(self, backend: FakeChatBackend, **kwargs: Any) -> Any:
        from llenvs.adapters.verifiers_v1 import _env_llm_reply_provider

        return _env_llm_reply_provider(backend, SamplingParams(), **kwargs)

    def test_conversation_built_from_trace(self):
        backend = FakeChatBackend("env says hi")
        provider = self._provider(backend)
        trace = self._make_trace()
        trace.nodes = [
            FakeNode(FakeSystemMessage("Sys")),
            FakeNode(FakeUserMessage("Q")),
            FakeNode(FakeAssistantMessage("A1"), sampled=True),
        ]
        pending = [FakeUserMessage("Follow-up")]

        reply = asyncio.run(provider(trace, pending))
        assert reply == "env says hi"
        messages = backend.calls[0][0]
        assert [(m.role, m.content) for m in messages] == [
            ("system", "Sys"),
            ("user", "Q"),
            ("assistant", "A1"),
            ("user", "Follow-up"),
        ]

    def test_fallback_system_prompt(self):
        backend = FakeChatBackend()
        provider = self._provider(backend, fallback_system_prompt="Fallback sys")
        trace = self._make_trace(system_prompt=None)

        asyncio.run(provider(trace, [FakeUserMessage("hi")]))
        messages = backend.calls[0][0]
        assert (messages[0].role, messages[0].content) == ("system", "Fallback sys")

    def test_task_system_prompt_wins_over_fallback(self):
        backend = FakeChatBackend()
        provider = self._provider(backend, fallback_system_prompt="Fallback sys")
        trace = self._make_trace(system_prompt="Task sys")

        asyncio.run(provider(trace, [FakeUserMessage("hi")]))
        assert backend.calls[0][0][0].content == "Task sys"

    def test_no_system_message_when_neither(self):
        backend = FakeChatBackend()
        provider = self._provider(backend)
        trace = self._make_trace(system_prompt=None)

        asyncio.run(provider(trace, [FakeUserMessage("hi")]))
        assert [m.role for m in backend.calls[0][0]] == ["user"]

    def test_generation_runs_in_worker_thread(self):
        backend = FakeChatBackend()
        provider = self._provider(backend)
        trace = self._make_trace()

        asyncio.run(provider(trace, [FakeUserMessage("hi")]))
        assert backend.calls[0][2] is not threading.current_thread()

    def test_none_text_becomes_empty_reply(self):
        backend = FakeChatBackend(reply=None)
        provider = self._provider(backend)
        trace = self._make_trace()

        reply = asyncio.run(provider(trace, [FakeUserMessage("hi")]))
        assert reply == ""

    def test_agent_sampling_merged(self):
        backend = FakeChatBackend()
        provider = self._provider(backend)
        sampling = SimpleNamespace(temperature=0.7, top_p=None, max_tokens=99)
        trace = self._make_trace(sampling=sampling)

        asyncio.run(provider(trace, [FakeUserMessage("hi")]))
        params = backend.calls[0][1]
        assert params.temperature == 0.7
        assert params.max_tokens == 99
        assert params.top_p == SamplingParams().top_p


class TestMergeSamplingParams:
    def test_no_sampling_returns_base(self):
        from llenvs.adapters.verifiers_v1 import _merge_sampling_params

        base = SamplingParams(temperature=0.3)
        merged = _merge_sampling_params(base, FakeAgentConfig(sampling=None))
        assert merged == base

    def test_overrides_only_set_fields(self):
        from llenvs.adapters.verifiers_v1 import _merge_sampling_params

        base = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=512)
        sampling = SimpleNamespace(temperature=1.0, top_p=None, max_tokens=None)
        merged = _merge_sampling_params(base, FakeAgentConfig(sampling=sampling))

        assert merged.temperature == 1.0
        assert merged.top_p == 0.9
        assert merged.max_tokens == 512
        # The base is never mutated.
        assert base.temperature == 0.3


# ── Chunk 4: episode bridge ─────────────────────────────────────────


class RelayEnv:
    """Prompted two-turn relay: echo the reply back once, then end."""

    def __init__(self) -> None:
        self.finalize_episodes: list[Any] = []

    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        async with agents.solver.interaction(task) as interaction:
            first = await interaction.turn()
            await interaction.turn(f"you said: {first.last_reply}")

    async def finalize(self, task: Any, episode: Any) -> None:
        self.finalize_episodes.append(episode)


class OpenerEnv:
    """Promptless env: opens the conversation itself, one turn."""

    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        async with agents.solver.interaction(task) as interaction:
            await interaction.turn("env opener")

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class NpcOnlyEnv:
    """Runs only an untrainable seat — the policy is never consulted."""

    async def setup(self, agents: Any) -> None:
        agents.user.trainable = False

    async def run(self, task: Any, agents: Any) -> None:
        await agents.user.run(task)

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class BonusFinalizeEnv(RelayEnv):
    """Records a cross-agent reward on the finished episode's trace."""

    async def finalize(self, task: Any, episode: Any) -> None:
        await super().finalize(task, episode)
        episode.traces[0].record_reward("bonus", 0.5)


class ExplodingEnv:
    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        raise ValueError("env exploded")

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class NoAgentEnv:
    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        pass

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class ConcurrentTurnsEnv:
    """Opens a second policy turn while the first is still awaiting a reply."""

    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        first = asyncio.create_task(self._one_turn(agents.solver, task))
        await asyncio.sleep(0.05)  # let the first turn park on its policy future
        try:
            async with agents.solver.interaction(task) as second:
                await second.turn()
        finally:
            first.cancel()

    @staticmethod
    async def _one_turn(agent: Any, task: Any) -> None:
        async with agent.interaction(task) as interaction:
            await interaction.turn()

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class StalledEnv:
    """Never produces a policy turn nor finishes."""

    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        await asyncio.Event().wait()

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


def _make_bridge(env: Any, task: FakeTask, **kwargs: Any) -> Any:
    from llenvs.adapters.verifiers_v1 import _EpisodeBridge

    kwargs.setdefault("agent_specs", {"solver": FakeAgentConfig()})
    kwargs.setdefault("step_timeout", 5.0)
    kwargs.setdefault("env_id", "fake-env")
    return _EpisodeBridge(_make_fake_v1(), env, task, kwargs.pop("agent_specs"), **kwargs)


class TestEpisodeBridgeFirstObservation:
    def test_prompted_first_obs(self):
        bridge = _make_bridge(RelayEnv(), _simple_task())
        try:
            kind, turn = bridge.start()
        finally:
            bridge.shutdown()

        assert kind == "obs"
        assert turn.messages == (
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "What is 2+2?"},
        )
        assert turn.num_turns == 0
        assert turn.agent_name == "solver"

    def test_promptless_first_obs(self):
        task = FakeTask(FakeTaskData(prompt=None))
        bridge = _make_bridge(OpenerEnv(), task)
        try:
            kind, turn = bridge.start()
        finally:
            bridge.shutdown()

        assert kind == "obs"
        assert turn.messages == ({"role": "user", "content": "env opener"},)

    def test_done_without_policy_turn(self):
        """An untrainable-only episode finishes without consulting the policy."""
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        bridge = _make_bridge(
            NpcOnlyEnv(),
            task,
            agent_specs={"user": FakeAgentConfig()},
            env_llm_reply=ScriptedProvider("npc reply"),
        )
        try:
            kind, result = bridge.start()
        finally:
            bridge.shutdown()

        assert kind == "done"
        assert result.trace_results is None
        assert result.episode_ok is True
        assert result.all_trace_results == (("user", (("acc", 1.0, 1.0),)),)


class TestEpisodeBridgeRelay:
    def test_submit_advances_relay(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        env = RelayEnv()
        bridge = _make_bridge(env, task)
        try:
            kind1, _ = bridge.start()
            kind2, turn2 = bridge.submit("A1")
            kind3, result = bridge.submit("A2")
        finally:
            bridge.shutdown()

        assert (kind1, kind2, kind3) == ("obs", "obs", "done")
        assert turn2.messages == ({"role": "user", "content": "you said: A1"},)
        assert turn2.num_turns == 1
        assert result.num_turns == 2
        assert result.stop_condition == "user_closed"
        assert result.trace_ok is True
        assert len(task.scored_traces) == 1
        assert env.finalize_episodes[0].traces == task.scored_traces

    def test_done_carries_frozen_rewards(self):
        task = _simple_task(
            rewards={"acc": (0.8, 1.0), "fmt": (0.5, 0.25)},
            metrics={"turns": 2.0},
            trace_info={"note": "hi"},
        )
        bridge = _make_bridge(RelayEnv(), task)
        try:
            bridge.start()
            bridge.submit("A1")
            kind, result = bridge.submit("A2")
        finally:
            bridge.shutdown()

        assert kind == "done"
        assert result.trace_results == (("acc", 0.8, 1.0), ("fmt", 0.5, 0.25))
        assert result.metrics == {"turns": 2.0}
        assert result.trace_info == {"note": "hi"}
        assert result.episode_ok is True

    def test_finalize_reward_lands_in_results(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        bridge = _make_bridge(BonusFinalizeEnv(), task)
        try:
            bridge.start()
            bridge.submit("A1")
            _, result = bridge.submit("A2")
        finally:
            bridge.shutdown()

        assert ("bonus", 0.5, 1.0) in result.trace_results


class TestEpisodeBridgeFailures:
    def test_env_error_posts_error_event(self):
        bridge = _make_bridge(ExplodingEnv(), _simple_task())
        try:
            kind, error = bridge.start()
        finally:
            bridge.shutdown()

        assert kind == "error"
        assert isinstance(error, ValueError)
        assert "env exploded" in str(error)

    def test_ran_no_agent_error(self):
        bridge = _make_bridge(NoAgentEnv(), _simple_task())
        try:
            kind, error = bridge.start()
        finally:
            bridge.shutdown()

        assert kind == "error"
        assert "ran no agent" in str(error)

    def test_concurrent_trainable_turns_error(self):
        bridge = _make_bridge(ConcurrentTurnsEnv(), _simple_task())
        try:
            kind1, _ = bridge.start()
            # Read the next event WITHOUT resolving the parked first turn —
            # the env opens its second turn while the first is still parked.
            kind2, error = bridge._next_event()
        finally:
            bridge.shutdown()

        assert kind1 == "obs"
        assert kind2 == "error"
        assert "concurrent" in str(error)

    def test_step_timeout(self):
        bridge = _make_bridge(StalledEnv(), _simple_task(), step_timeout=0.2)
        try:
            with pytest.raises(TimeoutError, match="no progress"):
                bridge.start()
        finally:
            bridge.shutdown()

        assert bridge.running is False


class TestEpisodeBridgeShutdown:
    def test_shutdown_mid_episode(self):
        """Abandonment unwinds silently: no events, no scoring, reclaimed thread."""
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        bridge = _make_bridge(RelayEnv(), task)
        kind, _ = bridge.start()
        assert kind == "obs"
        assert bridge.running is True

        bridge.shutdown()
        assert bridge.running is False
        assert bridge._events.empty()
        assert task.scored_traces == []
        bridge.shutdown()  # idempotent

    def test_new_bridge_after_shutdown(self):
        task = _simple_task(rewards={"acc": (1.0, 1.0)})
        env = RelayEnv()
        first = _make_bridge(env, task)
        first.start()
        first.shutdown()

        second = _make_bridge(env, task)
        try:
            kind, _ = second.start()
            _, turn2 = second.submit("B1")
            kind3, result = second.submit("B2")
        finally:
            second.shutdown()

        assert kind == "obs"
        assert turn2.messages == ({"role": "user", "content": "you said: B1"},)
        assert kind3 == "done"
        assert result.trace_results == (("acc", 1.0, 1.0),)


# ── Chunk 5: multi-turn environment ─────────────────────────────────


class TwoSeatEnv:
    """User-sim shape: an untrainable user opens, the trainable solver replies."""

    async def setup(self, agents: Any) -> None:
        agents.user.trainable = False

    async def run(self, task: Any, agents: Any) -> None:
        async with agents.user.interaction(task) as user_side:
            opener = await user_side.turn("start the conversation")
            async with agents.solver.interaction(task) as solver_side:
                await solver_side.turn(opener.last_reply)

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class SkipsPolicyEnv(TwoSeatEnv):
    """Finishes the episode without ever consulting the trainable seat."""

    async def run(self, task: Any, agents: Any) -> None:
        await agents.user.run(task)


class LoopingEnv:
    """Relays forever; only an external limit ends the episode."""

    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        async with agents.solver.interaction(task) as interaction:
            segment = await interaction.turn()
            while not segment.terminated:
                segment = await interaction.turn(f"again: {segment.last_reply}")

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


class MidStepExplodingEnv:
    """Raises after the first policy turn resolves."""

    async def setup(self, agents: Any) -> None:
        pass

    async def run(self, task: Any, agents: Any) -> None:
        async with agents.solver.interaction(task) as interaction:
            await interaction.turn()
            raise ValueError("mid boom")

    async def finalize(self, task: Any, episode: Any) -> None:
        pass


def _make_multi_turn(env: Any, tasks: list[FakeTask], **kwargs: Any) -> Any:
    from llenvs.adapters.verifiers_v1 import VerifiersV1MultiTurnEnvironment

    kwargs.setdefault("agent_specs", {"solver": FakeAgentConfig()})
    kwargs.setdefault("step_timeout", 5.0)
    return VerifiersV1MultiTurnEnvironment(
        _make_fake_v1(),
        "fake-taskset",
        env,
        list(tasks),
        kwargs.pop("agent_specs"),
        **kwargs,
    )


class TestMultiTurnConstruction:
    def test_two_trainable_seats_rejected(self):
        with pytest.raises(NotImplementedError, match="trainable"):
            _make_multi_turn(
                RelayEnv(),
                [_simple_task()],
                agent_specs={"solver": FakeAgentConfig(), "critic": FakeAgentConfig()},
            )

    def test_zero_trainable_seats_rejected(self):
        with pytest.raises(ValueError, match="trainable"):
            _make_multi_turn(
                NpcOnlyEnv(),
                [_simple_task()],
                agent_specs={"user": FakeAgentConfig()},
                env_llm=FakeChatBackend("npc"),
            )

    def test_untrainable_seat_requires_env_llm(self):
        with pytest.raises(ValueError, match="env_llm"):
            _make_multi_turn(
                TwoSeatEnv(),
                [FakeTask(FakeTaskData(prompt=None))],
                agent_specs={"solver": FakeAgentConfig(), "user": FakeAgentConfig()},
            )

    def test_env_llm_enables_untrainable_seats(self):
        env = _make_multi_turn(
            TwoSeatEnv(),
            [FakeTask(FakeTaskData(prompt=None))],
            agent_specs={"solver": FakeAgentConfig(), "user": FakeAgentConfig()},
            env_llm=FakeChatBackend("What is the answer?"),
        )
        assert env.spec.is_multi_turn is True


class TestMultiTurnReset:
    def test_reset_requires_task_index(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        with pytest.raises(ValueError, match="task_index"):
            env.reset()

    def test_reset_bounds(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 3})

    def test_first_observation(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        try:
            state, info = env.reset(options={"task_index": 0})
        finally:
            env.close()

        assert state.observation.prompt == "What is 2+2?"
        assert state.observation.messages == ()
        assert state.observation.task is not None
        assert state.observation.task.text == "What is 2+2?"
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert state.hidden.task_index == 0
        assert state.hidden.trace_results is None
        assert info["task_index"] == 0
        assert info["taskset_id"] == "fake-taskset"
        assert info["task_key"] == "task-key"
        assert info["system_prompt"] == "Be terse."

    def test_episode_id_honored(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        try:
            state, _ = env.reset(options={"task_index": 0, "episode_id": "ep-42"})
        finally:
            env.close()

        assert state.metadata.episode_id == "ep-42"

    def test_reset_error_event(self):
        env = _make_multi_turn(ExplodingEnv(), [_simple_task()])
        with pytest.raises(RuntimeError, match="env exploded") as excinfo:
            env.reset(options={"task_index": 0})

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_done_before_policy_turn(self):
        env = _make_multi_turn(
            SkipsPolicyEnv(),
            [_simple_task(rewards={"acc": (1.0, 1.0)})],
            agent_specs={"solver": FakeAgentConfig(), "user": FakeAgentConfig()},
            env_llm=FakeChatBackend("npc says"),
        )
        with pytest.raises(RuntimeError, match="without consulting"):
            env.reset(options={"task_index": 0})

    def test_reset_replaces_previous_episode(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        try:
            env.reset(options={"task_index": 0})
            first_bridge = env._bridge
            env.reset(options={"task_index": 0})

            assert first_bridge.running is False
        finally:
            env.close()


class TestMultiTurnStep:
    def test_relay_non_terminal_step(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task(rewards={"acc": (1.0, 1.0)})])
        try:
            state, _ = env.reset(options={"task_index": 0})
            result = env.step(state, Action.from_text("A1"))
        finally:
            env.close()

        assert result.terminated is False
        assert result.truncated is False
        assert result.rewards.signals == ()
        obs = result.next_state.observation
        assert obs.prompt == "What is 2+2?"
        assert obs.task is not None
        assert obs.task.text == "What is 2+2?"
        assert obs.messages == (
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "you said: A1"},
        )
        assert obs.state is not None
        assert obs.state.text == "you said: A1"
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.trace_results is None

    def test_relay_terminal_step(self):
        task = _simple_task(
            rewards={"acc": (1.0, 1.0)},
            metrics={"turns": 2.0},
            trace_info={"note": "hi"},
        )
        env = _make_multi_turn(RelayEnv(), [task])
        try:
            state0, _ = env.reset(options={"task_index": 0})
            state1 = env.step(state0, Action.from_text("A1")).next_state
            result = env.step(state1, Action.from_text("A2"))
        finally:
            env.close()

        assert result.terminated is True
        assert result.truncated is False
        assert [(s.name, s.reward, s.weight) for s in result.rewards.signals] == [("acc", 1.0, 1.0)]
        assert result.rewards.total == pytest.approx(1.0)
        assert result.next_state.metadata.is_terminal is True
        assert result.next_state.hidden.trace_results == (("acc", 1.0, 1.0),)
        assert result.next_state.observation.messages[-1] == {
            "role": "assistant",
            "content": "A2",
        }
        # RelayEnv closes the interaction itself -> "user_closed" (an env
        # that ends via Agent.run()'s bare episode gets "agent_completed").
        assert result.info["stop_condition"] == "user_closed"
        assert result.info["verifiers_metrics"] == {"turns": 2.0}
        assert result.info["trace_info"] == {"note": "hi"}
        assert result.info["runtime_skipped_signals"] == []
        assert result.info["episode_ok"] is True
        assert result.info["all_trace_rewards"] == (("solver", (("acc", 1.0, 1.0),)),)

    def test_truncated_on_max_steps(self):
        env = _make_multi_turn(
            LoopingEnv(),
            [_simple_task(rewards={"acc": (1.0, 1.0)})],
            max_steps=2,
        )
        try:
            state0, _ = env.reset(options={"task_index": 0})
            first = env.step(state0, Action.from_text("A1"))
            second = env.step(first.next_state, Action.from_text("A2"))
        finally:
            env.close()

        assert first.terminated is False
        assert first.truncated is False
        assert second.truncated is True
        assert second.terminated is False
        assert second.info["stop_condition"] == "max_turns"

    def test_stale_state_rejected(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        try:
            state0, _ = env.reset(options={"task_index": 0})
            env.step(state0, Action.from_text("A1"))
            with pytest.raises(NotImplementedError):
                env.step(state0, Action.from_text("A1-again"))
        finally:
            env.close()

    def test_step_after_terminal_rejected(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task(rewards={"acc": (1.0, 1.0)})])
        try:
            state0, _ = env.reset(options={"task_index": 0})
            state1 = env.step(state0, Action.from_text("A1")).next_state
            terminal = env.step(state1, Action.from_text("A2")).next_state
            with pytest.raises(ValueError, match="terminal"):
                env.step(terminal, Action.from_text("A3"))
        finally:
            env.close()

    def test_tool_call_action_refused(self):
        from llenvs.core.tools import ToolCall

        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        try:
            state, _ = env.reset(options={"task_index": 0})
            action = Action.from_tool_call(ToolCall(id="c1", name="t", arguments={}))
            with pytest.raises(ValueError, match="tool"):
                env.step(state, action)
        finally:
            env.close()

    def test_mid_episode_env_error(self):
        env = _make_multi_turn(MidStepExplodingEnv(), [_simple_task()])
        try:
            state0, _ = env.reset(options={"task_index": 0})
            with pytest.raises(RuntimeError, match="mid boom"):
                env.step(state0, Action.from_text("A1"))
        finally:
            env.close()

    def test_env_llm_drives_untrainable_seat(self):
        backend = FakeChatBackend("What is the answer?")
        task = FakeTask(FakeTaskData(prompt=None), rewards={"acc": (1.0, 1.0)})
        env = _make_multi_turn(
            TwoSeatEnv(),
            [task],
            agent_specs={"solver": FakeAgentConfig(), "user": FakeAgentConfig()},
            env_llm=backend,
        )
        try:
            state, _ = env.reset(options={"task_index": 0})
            result = env.step(state, Action.from_text("42"))
        finally:
            env.close()

        assert state.observation.prompt == "What is the answer?"
        assert result.terminated is True
        messages, _params, _thread = backend.calls[0]
        assert [(m.role, m.content) for m in messages] == [("user", "start the conversation")]
        assert dict(result.info["all_trace_rewards"]).keys() == {"solver", "user"}
        # The trainable-seat view is the solver's trace.
        assert result.next_state.hidden.trace_results == (("acc", 1.0, 1.0),)

    def test_extra_rewards_appended(self):
        env = _make_multi_turn(
            RelayEnv(),
            [_simple_task(rewards={"acc": (1.0, 1.0)})],
            extra_rewards=(StubExtraReward(),),
        )
        try:
            state0, _ = env.reset(options={"task_index": 0})
            state1 = env.step(state0, Action.from_text("A1")).next_state
            result = env.step(state1, Action.from_text("A2"))
        finally:
            env.close()

        assert [s.name for s in result.rewards.signals] == ["acc", "extra"]
        assert result.rewards.total == pytest.approx(1.25)

    def test_runtime_skipped_in_info(self):
        async def needs_rt(trace: Any, runtime: Any) -> float:
            return 0.0

        task = _simple_task(rewards={"acc": (1.0, 1.0)}, reward_hooks=(needs_rt,))
        env = _make_multi_turn(RelayEnv(), [task])
        try:
            state0, _ = env.reset(options={"task_index": 0})
            state1 = env.step(state0, Action.from_text("A1")).next_state
            result = env.step(state1, Action.from_text("A2"))
        finally:
            env.close()

        assert result.info["runtime_skipped_signals"] == ["needs_rt"]


class TestMultiTurnSpecAndLifecycle:
    def test_spec_and_len(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task(), _simple_task()], max_steps=5)
        spec = env.spec

        assert spec.name == "fake-taskset"
        assert spec.adapter == "verifiers_v1"
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is False
        assert spec.max_steps == 5
        assert len(env) == 2
        assert env.prompts == {}
        assert env.available_tools == ()
        assert env.reward_functions[0].name == "verifiers_v1"

    def test_close_shuts_down_bridge(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        env.reset(options={"task_index": 0})
        bridge = env._bridge

        env.close()

        assert bridge.running is False
        env.close()  # idempotent
        env2 = _make_multi_turn(RelayEnv(), [_simple_task()])
        env2.close()  # safe before any reset

    def test_finalizer_shuts_down_bridge(self):
        env = _make_multi_turn(RelayEnv(), [_simple_task()])
        env.reset(options={"task_index": 0})
        bridge = env._bridge

        del env
        gc.collect()

        assert bridge.running is False


# ── Chunk 5: adapter routing ────────────────────────────────────────


class FakeTasksetConfig:
    """Plain taskset config recording the kwargs the adapter passed."""

    def __init__(self, **kwargs: Any) -> None:
        self.raw = kwargs
        self.id = kwargs.get("id", "")
        self.task = kwargs.get("task")


class FakeV1Taskset:
    """Iterable taskset with head/shuffle views mirroring the v1 surface."""

    INFINITE = False

    def __init__(self, tasks: Any) -> None:
        self._tasks = tasks
        self.config: Any = None
        self.head_calls: list[int] = []
        self.shuffle_seeds: list[int | None] = []

    def __iter__(self) -> Any:
        source = self._tasks() if callable(self._tasks) else self._tasks
        yield from source

    def _view(self, tasks: list[Any]) -> FakeV1Taskset:
        view = FakeV1Taskset(tasks)
        view.config = self.config
        view.head_calls = self.head_calls
        view.shuffle_seeds = self.shuffle_seeds
        return view

    def head(self, n: int) -> FakeV1Taskset:
        self.head_calls.append(n)
        return self._view(list(itertools.islice(iter(self), n)))

    def shuffle(self, seed: int | None = None) -> FakeV1Taskset:
        if self.INFINITE:
            raise ValueError("bound it first with head(num_tasks)")
        self.shuffle_seeds.append(seed)
        return self._view(list(reversed(list(self))))

    @classmethod
    def toolsets(cls, config: Any) -> list[Any]:
        return []

    @classmethod
    def task_type(cls) -> type:
        return FakeTask


class ToolsetTaskset(FakeV1Taskset):
    @classmethod
    def toolsets(cls, config: Any) -> list[Any]:
        return [object()]


class ContainerTask(FakeTask):
    NEEDS_CONTAINER = True


class ContainerTaskset(FakeV1Taskset):
    @classmethod
    def task_type(cls) -> type:
        return ContainerTask


class FakeEnvConfig:
    """Env config with one declared seat, mirroring pydantic's model_fields."""

    model_fields: ClassVar[dict[str, Any]] = {
        "id": SimpleNamespace(default=""),
        "solver": SimpleNamespace(default=FakeAgentConfig()),
    }

    def __init__(self) -> None:
        self.solver = FakeAgentConfig()


class FakeCustomEnv(RelayEnv):
    """Custom Env subclass shape: constructed with its resolved config."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config


def _routing_adapter(
    monkeypatch: Any,
    taskset: FakeV1Taskset,
    *,
    env_cls: type | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Adapter whose v1 handle routes through scripted fake loaders."""
    from llenvs.adapters.verifiers_v1 import VerifiersV1Adapter

    record: dict[str, Any] = {}

    def environment_class(taskset_id: str, env_id: str = "") -> type:
        record["environment_class"] = (taskset_id, env_id)
        return env_cls or FakeSingleAgentEnv

    def taskset_config_type(taskset_id: str) -> type:
        record["taskset_config_type"] = taskset_id
        return FakeTasksetConfig

    def load_taskset(config: Any) -> FakeV1Taskset:
        taskset.config = config
        record["taskset_config"] = config
        return taskset

    def resolve_env_config(data: Any) -> FakeEnvConfig:
        record["env_config_data"] = data
        return FakeEnvConfig()

    loaders = SimpleNamespace(
        environment_class=environment_class,
        taskset_config_type=taskset_config_type,
        load_taskset=load_taskset,
        resolve_env_config=resolve_env_config,
    )
    handle = _make_fake_v1(loaders=loaders)
    monkeypatch.setattr(VerifiersV1Adapter, "_get_verifiers_v1", lambda self: handle)
    return VerifiersV1Adapter(), record


class TestGetEnvironmentRouting:
    def test_single_agent_routes_single_turn(self, monkeypatch):
        from llenvs.adapters.verifiers_v1 import VerifiersV1SingleTurnEnvironment

        taskset = FakeV1Taskset([_simple_task(), _simple_task()])
        adapter, record = _routing_adapter(monkeypatch, taskset)
        env = adapter.get_environment("fake-taskset")

        assert isinstance(env, VerifiersV1SingleTurnEnvironment)
        assert record["environment_class"] == ("fake-taskset", "")
        assert record["taskset_config"].id == "fake-taskset"
        assert len(env) == 2
        assert env.spec.name == "fake-taskset"

    def test_single_turn_kwargs_forwarded(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task(rewards={"acc": (1.0, 1.0)})])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        extractor = StubExtractor()
        env = adapter.get_environment(
            "fake-taskset",
            answer_extractor=extractor,
            extra_rewards=(StubExtraReward(),),
        )

        assert env.answer_extractor is extractor
        assert [fn.name for fn in env.reward_functions] == ["verifiers_v1", "extra"]

    def test_custom_env_routes_multi_turn(self, monkeypatch):
        from llenvs.adapters.verifiers_v1 import VerifiersV1MultiTurnEnvironment

        taskset = FakeV1Taskset([_simple_task(rewards={"acc": (1.0, 1.0)})])
        adapter, record = _routing_adapter(monkeypatch, taskset, env_cls=FakeCustomEnv)
        env = adapter.get_environment("fake-taskset", step_timeout=5.0)

        assert isinstance(env, VerifiersV1MultiTurnEnvironment)
        assert record["env_config_data"] == {"id": "", "taskset": {"id": "fake-taskset"}}
        try:
            state, _ = env.reset(options={"task_index": 0})
            first = env.step(state, Action.from_text("A1"))
            second = env.step(first.next_state, Action.from_text("A2"))
        finally:
            env.close()
        assert second.terminated is True
        assert second.rewards.total == pytest.approx(1.0)

    def test_name_parsing_env_plus_taskset(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task()])
        adapter, record = _routing_adapter(monkeypatch, taskset, env_cls=FakeCustomEnv)
        env = adapter.get_environment("fake-env+fake-taskset", step_timeout=5.0)
        try:
            assert record["environment_class"] == ("fake-taskset", "fake-env")
            assert record["env_config_data"]["id"] == "fake-env"
            assert env.spec.name == "fake-env+fake-taskset"
        finally:
            env.close()

    def test_env_id_kwarg_wins(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task()])
        adapter, record = _routing_adapter(monkeypatch, taskset, env_cls=FakeCustomEnv)
        env = adapter.get_environment("fake-taskset", env_id="other-env", step_timeout=5.0)
        try:
            assert record["environment_class"] == ("fake-taskset", "other-env")
            assert env.spec.name == "other-env+fake-taskset"
        finally:
            env.close()

    def test_params_passthrough(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task()])
        adapter, record = _routing_adapter(monkeypatch, taskset, env_cls=FakeCustomEnv)
        env = adapter.get_environment(
            "fake-taskset",
            taskset_params={"split": "dev"},
            task_params={"judges": []},
            env_params={"timeout": 60},
            step_timeout=5.0,
        )
        try:
            expected_taskset = {
                "id": "fake-taskset",
                "split": "dev",
                "task": {"judges": []},
            }
            assert record["taskset_config"].raw == expected_taskset
            assert record["env_config_data"]["timeout"] == 60
            assert record["env_config_data"]["taskset"] == expected_taskset
        finally:
            env.close()

    def test_max_steps_forwarded(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task()])
        adapter, _ = _routing_adapter(monkeypatch, taskset, env_cls=FakeCustomEnv)
        env = adapter.get_environment("fake-taskset", max_steps=3, step_timeout=5.0)
        try:
            assert env.spec.max_steps == 3
        finally:
            env.close()


class TestGetEnvironmentMaterialization:
    def test_infinite_requires_size(self, monkeypatch):
        taskset = FakeV1Taskset(lambda: (_simple_task() for _ in itertools.count()))
        taskset.INFINITE = True
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(ValueError, match="size"):
            adapter.get_environment("fake-taskset")

    def test_infinite_with_size(self, monkeypatch):
        taskset = FakeV1Taskset(lambda: (_simple_task() for _ in itertools.count()))
        taskset.INFINITE = True
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        env = adapter.get_environment("fake-taskset", size=3)

        assert len(env) == 3
        assert taskset.head_calls == [3]
        assert env.spec.metadata["infinite"] is True

    def test_size_truncates_finite(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task(), _simple_task(), _simple_task()])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        env = adapter.get_environment("fake-taskset", size=2)

        assert len(env) == 2
        assert env.spec.metadata["infinite"] is False

    def test_seed_shuffles(self, monkeypatch):
        first = FakeTask(FakeTaskData(prompt="Q1"), key="k1")
        second = FakeTask(FakeTaskData(prompt="Q2"), key="k2")
        taskset = FakeV1Taskset([first, second])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        env = adapter.get_environment("fake-taskset", seed=7)

        assert taskset.shuffle_seeds == [7]
        _, info = env.reset(options={"task_index": 0})
        assert info["task_key"] == "k2"  # the fake shuffle reverses

    def test_no_seed_no_shuffle(self, monkeypatch):
        taskset = FakeV1Taskset([_simple_task()])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        adapter.get_environment("fake-taskset")

        assert taskset.shuffle_seeds == []

    def test_empty_taskset_rejected(self, monkeypatch):
        taskset = FakeV1Taskset([])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(ValueError, match="no tasks"):
            adapter.get_environment("fake-taskset")


class TestGetEnvironmentRefusals:
    def test_container_tasks_refused(self, monkeypatch):
        taskset = ContainerTaskset([_simple_task()])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(NotImplementedError, match="container"):
            adapter.get_environment("fake-taskset")

    def test_toolsets_refused(self, monkeypatch):
        taskset = ToolsetTaskset([_simple_task()])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(NotImplementedError, match="toolset"):
            adapter.get_environment("fake-taskset")

    def test_intercept_hooks_refused(self, monkeypatch):
        task = _simple_task()
        task._intercept_hooks = [lambda request: request]
        taskset = FakeV1Taskset([task])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(NotImplementedError, match="intercept"):
            adapter.get_environment("fake-taskset")

    def test_non_trace_stop_refused(self, monkeypatch):
        async def request_stop(request: Any) -> bool:
            return False

        request_stop._boundary = "request"
        task = _simple_task(stop_hooks=(request_stop,))
        taskset = FakeV1Taskset([task])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(NotImplementedError, match="boundary"):
            adapter.get_environment("fake-taskset")

    def test_image_tasks_refused(self, monkeypatch):
        task = FakeTask(FakeTaskData(prompt="Q", image=object()))
        taskset = FakeV1Taskset([task])
        adapter, _ = _routing_adapter(monkeypatch, taskset)
        with pytest.raises(NotImplementedError, match="image"):
            adapter.get_environment("fake-taskset")
