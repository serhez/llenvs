"""Shared stubs for the ``llenvs_env`` plugin tests.

Requires verifiers v1: import this module only after
``pytest.importorskip("verifiers.v1")``.

Two stub families live here:

- ``MockRelayEnv`` / ``MockRelayAdapter``: a scripted multi-turn llenvs
  environment registered under the adapter name ``llenvs_env_test`` so YAML
  configs can reach it through ``EnvironmentFactory``. It records reset/step
  thread ids, actions, and close() so tests can assert the relay's discipline.
- ``StubAgents`` / ``StubAgent`` / ``StubInteraction``: the sanctioned
  ``Env.run(task, agents)`` seam, replaying scripted policy replies as
  token-free trace commits (the same commit form the real rollout uses).
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import verifiers.v1 as vf
from verifiers.v1 import graph as vf_graph
from verifiers.v1.dialects.chat import parse_message

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.registry import environment_registry
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.tools import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)

ADAPTER_NAME = "llenvs_env_test"

LOOKUP_TOOL = ToolDefinition(
    name="lookup",
    description="Look something up.",
    parameters=(ToolParameter(name="q", type=ToolParameterType.STRING, description="Query."),),
)

IMAGE = ImageContent(data="aGk=", media_type="image/png")


# ---------------------------------------------------------------------------
# Mock llenvs environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelayHidden:
    task_index: int
    step: int


class MockRelayEnv:
    """Scripted multi-turn environment.

    Every step yields ``progress`` (0.5, weight 1.0 — or 2.0 on even steps when
    ``varying_weights``) and ``format`` (1.0 when the action has text, weight
    0.0). ``none_signal`` adds a reward-less ``hint`` signal. The episode
    terminates at ``total_steps``; ``truncate_at`` truncates instead at that
    step; ``fail_on_step`` raises there.
    """

    instances: ClassVar[list[MockRelayEnv]] = []

    def __init__(
        self,
        *,
        num_tasks: int = 3,
        total_steps: int = 2,
        max_steps: int = 5,
        tools: bool = False,
        images: bool = False,
        history: bool = False,
        fail_on_step: int | None = None,
        truncate_at: int | None = None,
        varying_weights: bool = False,
        none_signal: bool = False,
        **_ignored: Any,
    ) -> None:
        self._num_tasks = num_tasks
        self._total_steps = total_steps
        self._max_steps = max_steps
        self._tools = tools
        self._images = images
        self._history = history
        self._fail_on_step = fail_on_step
        self._truncate_at = truncate_at
        self._varying_weights = varying_weights
        self._none_signal = none_signal
        self.actions: list[Action] = []
        self.threads: list[int] = []
        self.closed = False
        MockRelayEnv.instances.append(self)

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="relay", adapter=ADAPTER_NAME, max_steps=self._max_steps, is_multi_turn=True
        )

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        return (LOOKUP_TOOL,) if self._tools else ()

    @property
    def reward_functions(self) -> tuple:
        return ()

    def __len__(self) -> int:
        return self._num_tasks

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[RelayHidden], dict[str, Any]]:
        self.threads.append(threading.get_ident())
        idx = (options or {}).get("task_index", 0)
        if not 0 <= idx < self._num_tasks:
            raise IndexError(f"task_index {idx} out of range")
        messages: tuple[dict[str, Any], ...] = ()
        if self._history:
            messages = (
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": "Go."},
            )
        observation = Observation(
            prompt=f"Task {idx}: take {self._total_steps} steps.",
            messages=messages,
            available_tools=self.available_tools,
            state=ObservationContent(text="Ready.", images=(IMAGE,) if self._images else ()),
        )
        state = State(
            observation=observation,
            hidden=RelayHidden(task_index=idx, step=0),
            metadata=StateMetadata(step=0, episode_id=f"ep-{idx}"),
        )
        return state, {"task_index": idx}

    def step(self, state: State[RelayHidden], action: Action) -> StepResult[RelayHidden]:
        self.threads.append(threading.get_ident())
        self.actions.append(action)
        step = state.hidden.step + 1
        if self._fail_on_step == step:
            raise RuntimeError("env exploded")
        weight = 2.0 if self._varying_weights and step % 2 == 0 else 1.0
        signals = [
            Signal(name="progress", reward_type=RewardType.STEP, reward=0.5, weight=weight),
            Signal(
                name="format",
                reward_type=RewardType.STEP,
                reward=1.0 if action.text else 0.0,
                weight=0.0,
            ),
        ]
        if self._none_signal:
            signals.append(Signal(name="hint", reward_type=RewardType.STEP, reward=None))
        tool_results = tuple(
            ToolResult.success(call_id=tc.id, tool_name=tc.name, output={"value": 42})
            for tc in action.tool_calls
        )
        terminated = step >= self._total_steps
        truncated = self._truncate_at == step and not terminated
        observation = Observation(
            prompt=state.observation.prompt,
            tool_results=tool_results,
            available_tools=self.available_tools,
            state=ObservationContent(text=f"Step {step} done."),
        )
        next_state = State(
            observation=observation,
            hidden=RelayHidden(task_index=state.hidden.task_index, step=step),
            metadata=StateMetadata(
                step=step,
                episode_id=state.metadata.episode_id,
                is_terminal=terminated or truncated,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=SignalBundle(signals=tuple(signals)),
            terminated=terminated,
            truncated=truncated,
            info={},
        )

    def compute_rewards(self, state: Any, action: Any, next_state: Any) -> SignalBundle:
        return SignalBundle.empty()

    def close(self) -> None:
        self.closed = True


class MockRelayAdapter:
    @property
    def name(self) -> str:
        return ADAPTER_NAME

    def list_environments(self) -> list[str]:
        return ["relay"]

    def get_environment(self, name: str, **kwargs: Any) -> MockRelayEnv:
        return MockRelayEnv(**kwargs)

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None


def ensure_registered() -> None:
    """Register the mock adapter once per process."""
    if ADAPTER_NAME not in environment_registry.list_adapters():
        environment_registry.register_adapter(MockRelayAdapter())


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def mint_trace(task: vf.Task) -> vf.Trace:
    """A token-free trace for ``task``, as the rollout would mint it."""
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
    )


def opening_messages(data: vf.TaskData) -> list[Any]:
    """System prompt (if any) plus the task prompt, as the first turn's inputs."""
    messages: list[Any] = []
    if data.system_prompt:
        messages.append(vf.SystemMessage(content=data.system_prompt))
    if isinstance(data.prompt, str):
        messages.append(vf.UserMessage(content=data.prompt))
    elif data.prompt is not None:
        messages.extend(data.prompt)
    return messages


def commit_reply(trace: vf.Trace, inputs: list[Any], text: str) -> None:
    """Commit one turn: ``inputs`` plus a sampled assistant reply, without tokens."""
    response = vf.Response(
        id=uuid.uuid4().hex,
        created=int(time.time()),
        model="stub-policy",
        message=vf.AssistantMessage(content=text),
        finish_reason="stop",
    )
    vf_graph.prepare_turn(trace, inputs).commit(response)


# ---------------------------------------------------------------------------
# Stub Agents / Agent / Interaction
# ---------------------------------------------------------------------------


class StubInteraction:
    """Scripted stand-in for ``verifiers.v1.Interaction``.

    Replays ``replies`` one per ``turn()``; once exhausted, the next ``turn()``
    returns a terminated segment (the framework ending the run). ``turns``
    records what the env sent (``None`` for a bare opening turn).
    """

    def __init__(self, task: vf.Task, replies: list[str]) -> None:
        self.task = task
        self.trace = mint_trace(task)
        self._replies = list(replies)
        self.turns: list[Any] = []
        self._started = False
        self._over = False

    async def turn(self, message: Any = None) -> vf.Segment:
        if self._over:
            raise RuntimeError("the exchange is over (the run ended); read interaction.trace")
        self.turns.append(message)
        data = self.task.data
        prompted = not self._started and data.prompt is not None
        if message is None and not prompted:
            raise ValueError("nothing to run a turn on")
        if message is not None and prompted:
            raise ValueError("the task's prompt opens this exchange")
        inputs: list[Any] = []
        if not self._started:
            inputs.extend(opening_messages(data))
        if isinstance(message, str):
            inputs.append(vf.UserMessage(content=message))
        elif message is not None:
            inputs.extend(parse_message(m) if isinstance(m, dict) else m for m in message)
        self._started = True
        if not self._replies:
            self._over = True
            return vf.Segment(messages=[], terminated=True)
        reply = self._replies.pop(0)
        commit_reply(self.trace, inputs, reply)
        return vf.Segment(messages=[vf.AssistantMessage(content=reply)])


class StubAgent:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.trainable = True
        self.interactions: list[StubInteraction] = []
        self.completed: list[vf.Trace] = []

    @asynccontextmanager
    async def interaction(self, task: vf.Task, **_: Any):
        interaction = StubInteraction(task, self.replies)
        self.interactions.append(interaction)
        yield interaction
        # Reached only when the env's run() exits cleanly: close the exchange
        # the way the rollout does — stop, score offline, complete.
        trace = interaction.trace
        if trace.stop_condition is None:
            trace.stop("user_closed")
        await task.score(trace, None)
        trace.is_completed = True
        trace.ok = True
        self.completed.append(trace)

    async def run(self, task: vf.Task) -> vf.Trace:
        raise AssertionError("the relay must drive the seat through interaction()")


class StubAgents:
    def __init__(self, replies: list[str]) -> None:
        self.agent = StubAgent(replies)

    def __iter__(self):
        return iter([self.agent])

    def __len__(self) -> int:
        return 1
