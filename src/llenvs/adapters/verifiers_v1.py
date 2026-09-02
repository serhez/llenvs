"""Verifiers v1 adapter — wraps verifiers-v1 tasksets and envs as MDP environments.

Single-turn tasksets (``SingleAgentEnv``-routed) need no rollout
infrastructure: each llenvs step commits the assistant reply into a
token-free message trace and scores it offline via
``Task.score(trace, runtime=None)`` — a supported verifiers mode in which
runtime-requiring signals are skipped. Multi-turn v1 environments own their
episode loop (``Env.run``); they are inverted into the reset/step protocol
via a thread-per-episode bridge.

All verifiers access goes through the ``_V1Handle`` seam: the adapter
imports nothing from verifiers at module import time, ``verifiers.v1`` is
imported directly (never root v0 attributes, which trigger legacy logging
side effects), and tests fake the whole surface through the handle.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

from llenvs.core.async_utils import run_async as _run_async
from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.extraction import AnswerExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.inference.protocol import ChatMessage, SamplingParams

logger = logging.getLogger(__name__)


# ── v1 access seam ──────────────────────────────────────────────────


@dataclass(frozen=True)
class _V1Handle:
    """Lazily-imported verifiers.v1 touchpoints.

    Every verifiers symbol the adapter uses is reached through this handle,
    making it the single seam tests fake and the single place to absorb
    upstream renames (verifiers v1 has no stability contract).

    Attributes:
        vf: The ``verifiers.v1`` module (Trace, Task, Segment, messages, ...).
        graph: ``verifiers.v1.graph`` (prepare_turn/commit).
        parse_message: ``verifiers.v1.dialects.chat.parse_message``.
        hook_boundary: ``verifiers.v1.session.hook_boundary``.
        loaders: ``verifiers.v1.utils.loaders`` (plugin id resolution).
        state_cls: ``verifiers.v1.state.state_cls`` (task-typed trace state).
        discover_decorated: ``verifiers.v1.utils.decorators.discover_decorated``.
    """

    vf: Any
    graph: Any
    parse_message: Any
    hook_boundary: Any
    loaders: Any
    state_cls: Any
    discover_decorated: Any


def _load_v1_handle() -> _V1Handle:
    """Import verifiers.v1 and build the access handle."""
    try:
        import verifiers.v1 as vf
        from verifiers.v1 import graph
        from verifiers.v1.dialects.chat import parse_message
        from verifiers.v1.session import hook_boundary
        from verifiers.v1.state import state_cls
        from verifiers.v1.utils import loaders
        from verifiers.v1.utils.decorators import discover_decorated
    except ImportError as e:
        raise ImportError(
            "verifiers with the v1 API is required for VerifiersV1Adapter. "
            'Install with: pip install "llenvs[verifiers]"'
        ) from e
    return _V1Handle(
        vf=vf,
        graph=graph,
        parse_message=parse_message,
        hook_boundary=hook_boundary,
        loaders=loaders,
        state_cls=state_cls,
        discover_decorated=discover_decorated,
    )


# ── Helpers ─────────────────────────────────────────────────────────


def _content_text(content: Any) -> str:
    """Flatten v1 MessageContent (str or content-part list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _observation_from_prompt(data: Any) -> tuple[Observation, str | None]:
    """Build an llenvs Observation from ``TaskData.prompt``.

    The runner emits ``[system?] + [user: obs.prompt] + replay(obs.messages)``,
    so a Messages prompt maps to: first user message -> ``obs.prompt``,
    remaining messages -> ``obs.messages``. A system message inside the
    prompt is extracted and returned (it wins over ``data.system_prompt``,
    mirroring the legacy adapter).

    Returns:
        Tuple of (observation, prompt-embedded system prompt or None).
    """
    prompt = data.prompt
    if isinstance(prompt, str):
        return (
            Observation(prompt=prompt, task=ObservationContent(text=prompt)),
            None,
        )

    row_system: str | None = None
    first_user: str | None = None
    remaining: list[dict[str, Any]] = []
    for msg in prompt:
        role = getattr(msg, "role", "user")
        text = _content_text(msg.content)
        if role == "system" and first_user is None and row_system is None:
            row_system = text
            continue
        if role == "user" and first_user is None:
            first_user = text
            continue
        remaining.append({"role": role, "content": text})

    if first_user is None:
        raise ValueError("verifiers v1 Messages prompt must contain a user message")

    observation = Observation(
        prompt=first_user,
        task=ObservationContent(text=first_user),
        messages=tuple(remaining),
    )
    return observation, row_system


def _prompt_message_list(v1: _V1Handle, data: Any) -> list[Any]:
    """Build the v1 message list committed as the trainable turn's prompt.

    Mirrors what a real rollout request contains: the task's system prompt
    (harnesses append it) followed by the task prompt verbatim.
    """
    messages: list[Any] = []
    if getattr(data, "system_prompt", None):
        messages.append(v1.vf.SystemMessage(content=data.system_prompt))
    prompt = data.prompt
    if isinstance(prompt, str):
        messages.append(v1.vf.UserMessage(content=prompt))
    else:
        for msg in prompt:
            messages.append(v1.parse_message(msg) if isinstance(msg, dict) else msg)
    return messages


def _requires_runtime(fn: Any) -> bool:
    """Whether a task hook has a mandatory ``runtime`` parameter.

    Reproduces verifiers' ``requires_runtime``: such hooks are skipped by
    ``Task.score(trace, runtime=None)``.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    param = sig.parameters.get("runtime")
    return param is not None and param.default is inspect.Parameter.empty


def _runtime_skipped_signals(task: Any) -> list[str]:
    """Names of reward/metric hooks that offline scoring skips."""
    skipped: list[str] = []
    for attr in ("reward", "metric"):
        for fn in task.hooks(attr):
            if _requires_runtime(fn):
                skipped.append(fn.__name__)
    return skipped


def _overridden_runtime_hooks(v1: _V1Handle, task: Any) -> list[str]:
    """Task lifecycle hooks (setup/finalize) overridden from the base Task.

    These require a Runtime and never run in offline scoring.
    """
    base = v1.vf.Task
    return [
        name
        for name in ("setup", "finalize")
        if getattr(type(task), name, None) is not getattr(base, name, None)
    ]


def _freeze_trace_rewards(trace: Any) -> tuple[tuple[str, float | None, float], ...]:
    """Freeze ``trace.rewards`` into (name, score, weight) tuples.

    Seeded-but-unscored entries (value ``None``) are kept as reward-less
    rows so they surface as feedback-less Signals.
    """
    frozen: list[tuple[str, float | None, float]] = []
    for name, reward in trace.rewards.items():
        if reward is None:
            frozen.append((name, None, 1.0))
        else:
            frozen.append((name, reward.score, reward.weight))
    return tuple(frozen)


def _mint_trace(
    v1: _V1Handle,
    task: Any,
    *,
    name: str = "agent",
    trainable: bool = True,
    config: Any = None,
) -> Any:
    """Mint a token-free trace for one task, mirroring the rollout's mint.

    Seat identity (name/trainable) is stamped at mint, as the episode
    agent's trace watch does.
    """
    vf = v1.vf
    return vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(
            config=config if config is not None else vf.AgentConfig(),
            name=name,
            trainable=trainable,
        ),
        state=v1.state_cls(type(task))(),
    )


def _commit_reply(v1: _V1Handle, trace: Any, prompt_messages: list[Any], text: str) -> None:
    """Commit one relay turn: prompt messages + the assistant reply."""
    response = v1.vf.Response(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model="llenvs-policy",
        message=v1.vf.AssistantMessage(content=text),
        finish_reason="stop",
    )
    v1.graph.prepare_turn(trace, prompt_messages).commit(response)


# ── Offline episode stubs ───────────────────────────────────────────
#
# Multi-turn v1 envs drive their episode through four surfaces: Agents
# (seat lookup), Agent (run/interaction/trainable), Interaction
# (trace/turn/close), and Segment. The classes below replicate those
# surfaces offline — including the per-turn refusal check that verifiers
# runs behind its HTTP interception server (RolloutSession.refused),
# which the Rollout itself never enforces.

ReplyProvider = Callable[[Any, list[Any]], Awaitable[str]]
"""Produces one assistant reply for (trace, pending prompt messages)."""


def _trace_boundary_stops(v1: _V1Handle, task: Any) -> list[Any]:
    """The task's stop hooks enforceable offline: those with a Trace boundary.

    Request/Response-boundary stops need the interception server and never
    run here (mirrors the rollout's boundary split).
    """
    return [
        fn for fn in task.hooks("stop") if v1.hook_boundary(fn, allow_trace=True) is v1.vf.Trace
    ]


def _merge_sampling_params(base: SamplingParams, agent_config: Any) -> SamplingParams:
    """Overlay a seat's ``AgentConfig.sampling`` onto llenvs sampling params.

    Only the fields both sides share (temperature, top_p, max_tokens) are
    merged; unset (None) values keep the base.
    """
    sampling = getattr(agent_config, "sampling", None)
    if sampling is None:
        return base
    overrides = {
        name: value
        for name in ("temperature", "top_p", "max_tokens")
        if (value := getattr(sampling, name, None)) is not None
    }
    return replace(base, **overrides) if overrides else base


def _seat_chat_message(message: Any) -> ChatMessage | None:
    """Map one v1 message to a ChatMessage from the seat's point of view.

    System messages are dropped (the seat's system prompt is prepended
    fresh); anything that is not the seat's own assistant reply — user
    turns, tool results — arrives as user input.
    """
    role = getattr(message, "role", "user")
    if role == "system":
        return None
    return ChatMessage(
        role="assistant" if role == "assistant" else "user",
        content=_content_text(message.content),
    )


def _env_llm_reply_provider(
    backend: Any,
    sampling_params: SamplingParams,
    fallback_system_prompt: str | None = None,
) -> ReplyProvider:
    """Build the reply provider driving an untrainable seat with ``env_llm``.

    The conversation is rebuilt from the seat's own trace plus the turn's
    pending prompt messages; the seat task's native ``system_prompt`` wins
    over the adapter-level fallback. Generation runs in a worker thread so
    a synchronous backend never blocks the episode loop.
    """

    async def provider(trace: Any, prompt_messages: list[Any]) -> str:
        system = getattr(trace.task.data, "system_prompt", None) or fallback_system_prompt
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        for node in trace.nodes:
            if (chat_message := _seat_chat_message(node.message)) is not None:
                messages.append(chat_message)
        for message in prompt_messages:
            if (chat_message := _seat_chat_message(message)) is not None:
                messages.append(chat_message)
        params = _merge_sampling_params(sampling_params, trace.agent.config)
        result = await asyncio.to_thread(backend.generate_chat, messages, params)
        return result.text or ""

    return provider


class _StubInteraction:
    """Offline replica of ``verifiers.v1.Interaction``.

    Replicates the exact turn contract: the prompted/bare-turn rule
    matrix, the per-turn refusal check (max_turns, then Trace-boundary
    ``@stop`` hooks — a refused turn commits nothing and returns a
    terminated Segment), segment-scoped ``root_reply`` hygiene, and close
    semantics (stop as ``user_closed``, offline scoring unless failed,
    ``is_completed``/``ok`` stamps). The trace is live from the moment
    the interaction exists.
    """

    def __init__(self, v1: _V1Handle, agent: _StubAgent, task: Any) -> None:
        self._v1 = v1
        self._agent = agent
        self._task = task
        self.trace = _mint_trace(
            v1,
            task,
            name=agent.role,
            trainable=agent.trainable,
            config=agent.config,
        )
        self.closed = False
        self._failed: Exception | None = None
        self._over = False
        self._started = False
        self._lock = asyncio.Lock()

    async def turn(self, message: Any = None) -> Any:
        """Send one user turn (or take a prompted task's opening reply)."""
        async with self._lock:
            return await self._turn(message)

    async def _turn(self, message: Any) -> Any:
        vf = self._v1.vf
        trace = self.trace
        if self.closed:
            raise RuntimeError("this interaction is closed")
        if self._over:
            raise RuntimeError("the exchange is over (the run ended); read interaction.trace")
        prompted = not self._started and self._task.data.prompt is not None
        if message is None and not prompted:
            raise ValueError(
                "nothing to run a turn on: a bare turn() takes a prompted task's "
                "opening reply; this exchange takes its next user message"
            )
        if message is not None and prompted:
            raise ValueError(
                "the task's prompt opens this exchange: take its first reply "
                "with a bare turn() before answering"
            )
        if isinstance(message, str):
            messages: list[Any] | None = [vf.UserMessage(content=message)]
        elif message is not None:
            messages = [self._v1.parse_message(m) if isinstance(m, dict) else m for m in message]
        else:
            messages = None
        first = not self._started
        self._started = True

        if (refusal := await self._refused()) is not None:
            trace.stop(refusal)
            self._over = True
            return vf.Segment(messages=[], terminated=True)

        if prompted:
            prompt_messages = _prompt_message_list(self._v1, self._task.data)
        else:
            prompt_messages = list(messages or [])
            if first and getattr(self._task.data, "system_prompt", None):
                prompt_messages.insert(0, vf.SystemMessage(content=self._task.data.system_prompt))

        nodes_before = len(trace.nodes)
        trace.root_reply = None
        reply = await self._agent.reply_provider()(trace, prompt_messages)
        _commit_reply(self._v1, trace, prompt_messages, reply)
        # Offline commits never produce tool messages, so the segment is
        # exactly the sampled nodes this turn added.
        segment_messages = [node.message for node in trace.nodes[nodes_before:] if node.sampled]
        return vf.Segment(messages=segment_messages, root_reply=trace.root_reply)

    async def _refused(self) -> str | None:
        """The pre-turn refusal check (``RolloutSession.refused`` offline).

        Only ``max_turns`` is enforceable: token limits are inert on
        token-free traces. Stop hooks are called with the trace directly —
        a Trace-boundary hook has exactly one Trace parameter by
        construction, so this matches verifiers' annotation-driven invoke.
        """
        max_turns = (
            self._agent.max_turns
            if self._agent.max_turns is not None
            else getattr(self._agent.config, "max_turns", None)
        )
        if max_turns is not None and self.trace.num_turns >= max_turns:
            return "max_turns"
        for stop in _trace_boundary_stops(self._v1, self._task):
            result = stop(self.trace)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, bool):
                raise RuntimeError(f"@stop must return bool, got {type(result).__name__}")
            if result:
                return stop.__name__
        return None

    def fail(self, error: Exception) -> None:
        """Record a rollout failure; close() then skips scoring and stamps ok=False."""
        if self._failed is None:
            self._failed = error

    def abort(self) -> None:
        """Abandon without completing: the trace never joins the episode.

        The cancellation path (a BaseException unwinding the interaction
        context) must not await scoring mid-teardown; marking the
        interaction closed makes the context's finally a no-op.
        """
        self.closed = True

    async def close(self) -> Any:
        """End the exchange: score offline unless failed; idempotent."""
        async with self._lock:
            if self.closed:
                return self.trace
            self.closed = True
            if self._failed is None:
                self.trace.stop("user_closed")
                await self._task.score(self.trace, None)
            self.trace.is_completed = True
            self.trace.ok = self._failed is None
            return self.trace


class _StubAgent:
    """Offline replica of a verifiers episode seat.

    ``trainable`` is mutable — env ``setup()`` flips untrainable seats
    (``agents.user.trainable = False``) before any interaction runs. The
    reply provider is resolved per turn from the flag: trainable seats are
    driven by the llenvs policy, untrainable seats by ``env_llm``.
    """

    def __init__(
        self,
        v1: _V1Handle,
        role: str,
        config: Any,
        episode_traces: list[Any],
        *,
        policy_reply: ReplyProvider,
        env_llm_reply: ReplyProvider | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.v1 = v1
        self.role = role
        self.config = config
        self.trainable = True
        self.policy_reply = policy_reply
        self.env_llm_reply = env_llm_reply
        self.max_turns = max_turns
        self._episode_traces = episode_traces

    def reply_provider(self) -> ReplyProvider:
        """The provider for this seat's next reply, per the trainable flag."""
        if self.trainable:
            return self.policy_reply
        if self.env_llm_reply is None:
            raise RuntimeError(
                f"seat {self.role!r} is untrainable but the environment was "
                "created without env_llm; pass env_llm= (and optionally "
                "sampling_params/system_prompt) to drive non-policy seats"
            )
        return self.env_llm_reply

    @asynccontextmanager
    async def interaction(
        self,
        task: Any,
        *,
        runtime: Any = None,
        tools: Any = None,
        on_trace: Callable[[Any], None] | None = None,
    ) -> AsyncIterator[_StubInteraction]:
        """Interact with this seat turn-by-turn; the caller is the run's user.

        Leaving the context closes the exchange and appends the completed
        trace to the episode — a failed exchange included, unscored with
        ``ok=False`` (real episode-agent semantics).
        """
        if runtime is not None:
            raise NotImplementedError(
                "borrowed runtime resources require verifiers' interception "
                "server; the offline verifiers_v1 bridge cannot use them"
            )
        if tools:
            raise NotImplementedError(
                "shared tools require verifiers' interception server; the "
                "offline verifiers_v1 bridge cannot use them"
            )
        interaction = _StubInteraction(self.v1, self, task)
        if on_trace is not None:
            on_trace(interaction.trace)
        try:
            yield interaction
        except Exception as e:
            interaction.fail(e)
            raise
        except BaseException:
            # Cancellation (episode abandonment/teardown): abort without
            # completing — real rollouts do the same, and awaiting scoring
            # mid-cancellation would be interrupted anyway.
            interaction.abort()
            raise
        finally:
            trace = interaction.trace if interaction.closed else await interaction.close()
            if trace.is_completed:
                self._episode_traces.append(trace)

    async def run(self, task: Any, *, on_trace: Callable[[Any], None] | None = None) -> Any:
        """One-segment episode: the task's opening reply, then agent_completed."""
        async with self.interaction(task, on_trace=on_trace) as interaction:
            segment = await interaction.turn()
            if not segment.terminated:
                interaction.trace.stop("agent_completed")
        return interaction.trace

    def provision(self, task: Any = None) -> Any:
        raise NotImplementedError(
            "provision() requires a real verifiers runtime; the offline "
            "verifiers_v1 bridge cannot provision boxes"
        )


class _StubAgents:
    """Offline replica of ``verifiers.v1.Agents``: seat access by role name."""

    def __init__(self, agents: dict[str, _StubAgent]) -> None:
        self._agents = dict(agents)

    def __getattr__(self, role: str) -> _StubAgent:
        try:
            return self._agents[role]
        except KeyError:
            raise AttributeError(
                f"no agent {role!r}; available seats: {sorted(self._agents)}"
            ) from None

    def __iter__(self) -> Any:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)


# ── Episode bridge ──────────────────────────────────────────────────


class _EpisodeAbandoned(Exception):  # noqa: N818 - control-flow signal, not an error
    """Injected into a parked policy turn when its episode is abandoned."""


@dataclass(frozen=True)
class _PolicyTurn:
    """Frozen snapshot of one policy turn request crossing the thread boundary.

    Attributes:
        messages: This turn's prompt messages as ``{"role", "content"}`` dicts.
        num_turns: The seat's sampled turns before this reply.
        agent_name: The seat requesting the reply.
    """

    messages: tuple[dict[str, str], ...]
    num_turns: int
    agent_name: str


@dataclass(frozen=True)
class _EpisodeResult:
    """Frozen episode outcome crossing the thread boundary.

    ``trace_results``/``metrics``/``trace_info``/``stop_condition``/
    ``trace_ok``/``num_turns`` describe the trainable seat's trace (the
    last one, when the env ran it more than once); ``all_trace_results``
    carries every seat's rewards as ``(agent_name, rewards)`` rows.
    """

    trace_results: tuple[tuple[str, float | None, float], ...] | None
    metrics: dict[str, float]
    trace_info: dict[str, Any]
    stop_condition: str | None
    trace_ok: bool
    num_turns: int
    episode_ok: bool
    all_trace_results: tuple[tuple[str, tuple[tuple[str, float | None, float], ...]], ...]


class _EpisodeBridge:
    """Runs one verifiers episode on a daemon thread, inverted into sync steps.

    The driver replicates ``Env.run_episode`` offline: mint an Episode,
    build stub agents, ``setup`` -> ``run`` -> ran-no-agent check ->
    ``finalize`` -> ``ok`` stamp. Trainable seats' replies come from a
    provider that posts a frozen ``_PolicyTurn`` to the event queue and
    parks on an asyncio future; the sync side resolves it via
    ``call_soon_threadsafe``. Event payloads are plain dicts/tuples — no
    live v1 objects cross the thread boundary.

    Events: ``("obs", _PolicyTurn)``, ``("done", _EpisodeResult)``,
    ``("error", Exception)``. ``shutdown()`` abandons the episode: a
    parked policy turn gets ``_EpisodeAbandoned`` injected (unwinding
    through the failed-close path, unscored), a busy driver is cancelled.
    A hung synchronous call inside the driver (an env_llm request in its
    worker thread) cannot be force-killed — the daemon thread is left
    behind and the process exits around it.
    """

    def __init__(
        self,
        v1: _V1Handle,
        env: Any,
        task: Any,
        agent_specs: dict[str, Any],
        *,
        env_id: str = "",
        env_llm_reply: ReplyProvider | None = None,
        max_turns: int | None = None,
        step_timeout: float | None = None,
    ) -> None:
        self._v1 = v1
        self._env = env
        self._task = task
        self._agent_specs = dict(agent_specs)
        self._env_id = env_id
        self._env_llm_reply = env_llm_reply
        self._max_turns = max_turns
        self._step_timeout = step_timeout
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._driver_task: asyncio.Task[None] | None = None
        self._pending: asyncio.Future[str] | None = None
        self._abandoned = False

    @property
    def running(self) -> bool:
        """Whether the episode thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> tuple[str, Any]:
        """Spawn the episode thread and return the first event."""
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"verifiers-v1-episode-{id(self):x}",
            daemon=True,
        )
        self._thread.start()
        return self._next_event()

    def submit(self, action_text: str) -> tuple[str, Any]:
        """Resolve the parked policy turn with the action; return the next event."""
        loop = self._loop

        def _resolve() -> None:
            if self._pending is not None and not self._pending.done():
                self._pending.set_result(action_text)

        if loop is not None:
            try:
                loop.call_soon_threadsafe(_resolve)
            except RuntimeError:
                pass  # the loop already closed (episode over); the queue has the event
        return self._next_event()

    def shutdown(self) -> None:
        """Abandon the episode and reclaim the thread; idempotent."""
        if self._abandoned:
            return
        self._abandoned = True
        loop = self._loop

        def _abort() -> None:
            if self._pending is not None and not self._pending.done():
                self._pending.set_exception(_EpisodeAbandoned())
            elif self._driver_task is not None:
                self._driver_task.cancel()

        if loop is not None and self.running:
            try:
                loop.call_soon_threadsafe(_abort)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _next_event(self) -> tuple[str, Any]:
        try:
            return self._events.get(timeout=self._step_timeout)
        except queue.Empty:
            raise TimeoutError(
                f"verifiers_v1 episode made no progress within "
                f"{self._step_timeout:g}s (step_timeout)"
            ) from None

    # -- episode thread -------------------------------------------------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._driver())
        except BaseException:
            if not self._abandoned:
                raise

    async def _driver(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._driver_task = asyncio.current_task()
        vf = self._v1.vf
        try:
            episode = vf.Episode(
                env=vf.EnvInfo(id=self._env_id),
                task=vf.TraceTask(
                    type=type(self._task).__name__,
                    data=self._task.data,
                    key=self._task.key,
                    hash=self._task.hash,
                ),
            )
            agents = _StubAgents(
                {
                    role: _StubAgent(
                        self._v1,
                        role,
                        config,
                        episode.traces,
                        policy_reply=self._policy_reply,
                        env_llm_reply=self._env_llm_reply,
                        max_turns=self._max_turns,
                    )
                    for role, config in self._agent_specs.items()
                }
            )
            await self._env.setup(agents)
            await self._env.run(self._task, agents)
            if not episode.traces:
                raise ValueError(
                    f"{type(self._env).__name__}.run() ran no agent — every "
                    "episode must carry at least one run"
                )
            await self._env.finalize(self._task, episode)
            episode.ok = all(t.ok for t in episode.traces)
            self._post(("done", self._freeze_result(episode)))
        except _EpisodeAbandoned:
            pass  # shutdown() unwound the episode; exit silently
        except Exception as e:
            self._post(("error", e))

    async def _policy_reply(self, trace: Any, prompt_messages: list[Any]) -> str:
        """The trainable seats' reply provider: hand off to the sync side."""
        if self._abandoned:
            raise _EpisodeAbandoned()
        if self._pending is not None:
            raise RuntimeError(
                "concurrent trainable turns are not supported: the verifiers_v1 "
                "bridge serves one policy turn at a time"
            )
        assert self._loop is not None
        future: asyncio.Future[str] = self._loop.create_future()
        self._pending = future
        self._post(
            (
                "obs",
                _PolicyTurn(
                    messages=tuple(
                        {
                            "role": getattr(m, "role", "user"),
                            "content": _content_text(m.content),
                        }
                        for m in prompt_messages
                    ),
                    num_turns=trace.num_turns,
                    agent_name=trace.agent.name,
                ),
            )
        )
        try:
            return await future
        finally:
            self._pending = None

    def _post(self, event: tuple[str, Any]) -> None:
        if not self._abandoned:
            self._events.put(event)

    def _freeze_result(self, episode: Any) -> _EpisodeResult:
        trainable = [t for t in episode.traces if t.agent.trainable]
        trace = trainable[-1] if trainable else None
        return _EpisodeResult(
            trace_results=_freeze_trace_rewards(trace) if trace is not None else None,
            metrics=dict(trace.metrics) if trace is not None else {},
            trace_info=dict(trace.info) if trace is not None else {},
            stop_condition=trace.stop_condition if trace is not None else None,
            trace_ok=bool(trace.ok) if trace is not None else True,
            num_turns=trace.num_turns if trace is not None else 0,
            episode_ok=bool(episode.ok),
            all_trace_results=tuple(
                (t.agent.name, _freeze_trace_rewards(t)) for t in episode.traces
            ),
        )


# ── Hidden states ───────────────────────────────────────────────────


@dataclass(frozen=True)
class VerifiersV1Hidden:
    """Hidden state for single-turn verifiers-v1 environments.

    Attributes:
        taskset_id: The verifiers taskset identifier.
        task_index: Index into the materialized task list.
        task_key: Durable task identity (``Task.key``).
        trace_results: Frozen (name, score, weight) reward tuples from the
            scored trace; None until the terminal step.
    """

    taskset_id: str
    task_index: int
    task_key: str
    trace_results: tuple[tuple[str, float | None, float], ...] | None = None


# ── Native reward function ──────────────────────────────────────────


@dataclass
class VerifiersV1TraceRewards:
    """Native reward function exposing scored trace rewards as Signals.

    ``signals()`` expands to one Signal per named verifiers reward (each
    with its native weight, so ``SignalBundle.total`` reproduces the
    verifiers weighted total exactly). ``compute()`` returns the aggregate
    as a single Signal for direct probing.
    """

    _name: str = "verifiers_v1"
    _reward_type: RewardType = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def signals(self, next_state: State[Any]) -> tuple[Signal, ...]:
        """One Signal per named trace reward; reward-less STEP signal pre-terminal."""
        results = getattr(next_state.hidden, "trace_results", None)
        if results is None or not next_state.metadata.is_terminal:
            return (Signal(name=self._name, reward_type=RewardType.STEP, reward=None),)
        signals = []
        for name, score, weight in results:
            metadata: dict[str, Any] = {"source": "verifiers_v1"}
            if score is None:
                metadata["unscored"] = True
            signals.append(
                Signal(
                    name=name,
                    reward_type=RewardType.OUTCOME,
                    reward=score,
                    weight=weight,
                    metadata=metadata,
                )
            )
        return tuple(signals)

    def compute(
        self,
        state: State[Any],
        action: Any,
        next_state: State[Any],
    ) -> Signal:
        """Aggregate weighted total as a single Signal (direct probing)."""
        results = getattr(next_state.hidden, "trace_results", None) or ()
        total = sum(score * weight for _, score, weight in results if score is not None)
        return Signal(name=self._name, reward_type=self._reward_type, reward=total)


# ── Single-turn environment ─────────────────────────────────────────


class VerifiersV1SingleTurnEnvironment:
    """MDP wrapper for SingleAgentEnv-routed verifiers-v1 tasksets.

    Each episode is one dataset task: reset() presents the task prompt,
    step() commits the assistant reply into a fresh token-free trace and
    scores it offline (``Task.score(trace, runtime=None)``). Pure-step:
    every step mints and scores its own trace, so any reset state can be
    stepped any number of times.
    """

    def __init__(
        self,
        v1: _V1Handle,
        taskset_id: str,
        tasks: list[Any],
        *,
        infinite: bool = False,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._v1 = v1
        self._taskset_id = taskset_id
        self._tasks = tasks
        self._infinite = infinite
        self._answer_extractor = answer_extractor
        self._native = VerifiersV1TraceRewards()
        self._extra_rewards = extra_rewards
        self._warned_runtime_hooks = False

    @property
    def answer_extractor(self) -> AnswerExtractor | None:
        """The extractor used to parse agent responses in ``step()``."""
        return self._answer_extractor

    @answer_extractor.setter
    def answer_extractor(self, value: AnswerExtractor | None) -> None:
        self._answer_extractor = value

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._taskset_id,
            adapter="verifiers_v1",
            max_steps=1,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=False,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=True,
            metadata={
                "taskset_id": self._taskset_id,
                "dataset_size": len(self._tasks),
                "infinite": self._infinite,
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return (self._native, *self._extra_rewards)

    def __len__(self) -> int:
        return len(self._tasks)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[VerifiersV1Hidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        task = self._tasks[task_index]
        if task.data.prompt is None:
            raise ValueError(
                f"task {task.key!r} has no prompt: promptless tasks require an "
                "env that opens the conversation and are not supported by the "
                "single-turn wrapper"
            )

        observation, row_system_prompt = _observation_from_prompt(task.data)
        system_prompt = row_system_prompt or getattr(task.data, "system_prompt", None)

        hidden = VerifiersV1Hidden(
            taskset_id=self._taskset_id,
            task_index=task_index,
            task_key=task.key,
        )
        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)

        info: dict[str, Any] = {
            "task_index": task_index,
            "taskset_id": self._taskset_id,
            "task_key": task.key,
            "system_prompt": system_prompt,
        }
        return state, info

    def step(
        self,
        state: State[VerifiersV1Hidden],
        action: Action,
    ) -> StepResult[VerifiersV1Hidden]:
        if action.tool_calls:
            raise ValueError(
                "verifiers_v1 environments are chat relays: tool-call actions "
                "are not supported (send the reply as text)"
            )

        task = self._tasks[state.hidden.task_index]
        self._warn_runtime_hooks(task)

        trace = _mint_trace(self._v1, task)
        _commit_reply(self._v1, trace, _prompt_message_list(self._v1, task.data), action.text or "")
        trace.stop("agent_completed")
        _run_async(task.score(trace, None))

        skipped = _runtime_skipped_signals(task)
        hidden = replace(state.hidden, trace_results=_freeze_trace_rewards(trace))
        next_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=True,
            info={
                **state.metadata.info,
                "response": action.text,
                "verifiers_metrics": dict(trace.metrics),
                "stop_condition": trace.stop_condition,
            },
        )
        next_state = State(
            observation=state.observation,
            hidden=hidden,
            metadata=next_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)

        info: dict[str, Any] = {
            "stop_condition": trace.stop_condition,
            "verifiers_metrics": dict(trace.metrics),
            "trace_info": dict(trace.info),
            "runtime_skipped_signals": skipped,
        }

        extracted: str | None = None
        if self._answer_extractor is not None:
            extracted, extraction_meta = self._answer_extractor.extract(action.text or "")
            info["extracted_answer"] = extracted
            info["extraction_metadata"] = extraction_meta

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=True,
            truncated=False,
            extracted_action=extracted,
            resolved_action=extracted,
            info=info,
        )

    def compute_rewards(
        self,
        state: State[VerifiersV1Hidden],
        action: Action,
        next_state: State[VerifiersV1Hidden],
    ) -> SignalBundle:
        signals = list(self._native.signals(next_state))
        for reward_fn in self._extra_rewards:
            signals.append(reward_fn.compute(state, action, next_state))
        return SignalBundle(signals=tuple(signals))

    def _warn_runtime_hooks(self, task: Any) -> None:
        if self._warned_runtime_hooks:
            return
        overridden = _overridden_runtime_hooks(self._v1, task)
        if overridden:
            logger.warning(
                "verifiers_v1: task %s overrides %s, which require a Runtime "
                "and are skipped in offline scoring",
                type(task).__name__,
                ", ".join(overridden),
            )
            self._warned_runtime_hooks = True


# ── Multi-turn environment ──────────────────────────────────────────

_TRUNCATING_STOPS = frozenset(
    {"max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens"}
)


async def _probe_policy_reply(trace: Any, prompt_messages: list[Any]) -> str:
    raise RuntimeError("probe agents never take a turn")


def _probe_seat_trainability(
    v1: _V1Handle, env: Any, agent_specs: dict[str, Any]
) -> dict[str, bool]:
    """Run ``env.setup()`` on throwaway agents to observe seat trainability.

    Envs flip untrainable seats in ``setup()`` (``agents.user.trainable =
    False``); probing once at construction lets the wrapper fail fast on
    unsupported seat layouts instead of mid-episode.
    """
    agents = _StubAgents(
        {
            role: _StubAgent(v1, role, config, [], policy_reply=_probe_policy_reply)
            for role, config in agent_specs.items()
        }
    )
    _run_async(env.setup(agents))
    return {agent.role: agent.trainable for agent in agents}


def _split_first_turn(
    messages: tuple[dict[str, str], ...],
) -> tuple[str | None, str | None, list[dict[str, str]]]:
    """Split the first policy turn into (system_prompt, prompt, remainder).

    Mirrors the runner's message layout: a leading system message is the
    episode system prompt, the first user message becomes ``obs.prompt``,
    everything else replays via ``obs.messages``.
    """
    system: str | None = None
    prompt: str | None = None
    remainder: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system" and system is None and prompt is None:
            system = message["content"]
            continue
        if message["role"] == "user" and prompt is None:
            prompt = message["content"]
            continue
        remainder.append(message)
    return system, prompt, remainder


def _shutdown_bridges(holder: list[_EpisodeBridge]) -> None:
    while holder:
        holder.pop().shutdown()


class VerifiersV1MultiTurnEnvironment:
    """Relay wrapper for custom verifiers-v1 Envs (multi-turn, multi-seat).

    Each episode runs the env's own ``run()`` loop on a daemon thread via
    ``_EpisodeBridge``; ``step()`` feeds the policy's reply to the parked
    trainable turn and returns the next one. Exactly one seat may remain
    trainable after ``setup()``; untrainable seats are driven by
    ``env_llm``. Rewards are terminal-only: the trainable seat's scored
    trace expands to per-name Signals, with every seat's rewards surfaced
    in ``info["all_trace_rewards"]``.

    Episodes are stateful (``pure_step=False``): only the latest state can
    be stepped. ``close()`` abandons any in-flight episode; ``reset()`` and
    a GC finalizer call it implicitly.
    """

    def __init__(
        self,
        v1: _V1Handle,
        name: str,
        env: Any,
        tasks: list[Any],
        agent_specs: dict[str, Any],
        *,
        env_llm: Any = None,
        sampling_params: SamplingParams | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        step_timeout: float | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        env_id: str = "",
        infinite: bool = False,
    ) -> None:
        self._v1 = v1
        self._name = name
        self._env = env
        self._tasks = tasks
        self._agent_specs = dict(agent_specs)
        self._env_id = env_id
        self._max_steps = max_steps
        self._step_timeout = step_timeout
        self._native = VerifiersV1TraceRewards()
        self._extra_rewards = extra_rewards
        self._infinite = infinite

        trainability = _probe_seat_trainability(v1, env, self._agent_specs)
        trainable = sorted(role for role, flag in trainability.items() if flag)
        untrainable = sorted(role for role, flag in trainability.items() if not flag)
        if not trainable:
            raise ValueError(
                f"verifiers_v1 env {name!r} leaves no trainable seat after "
                f"setup() (seats: {sorted(trainability)}); there is nothing to train"
            )
        if len(trainable) > 1:
            raise NotImplementedError(
                f"verifiers_v1 env {name!r} has {len(trainable)} trainable seats "
                f"({trainable}); the adapter supports exactly one"
            )
        if untrainable and env_llm is None:
            raise ValueError(
                f"verifiers_v1 env {name!r} has untrainable seats ({untrainable}) "
                "that need a model to drive them; pass env_llm= (and optionally "
                "sampling_params/system_prompt)"
            )
        self._env_llm_reply = (
            _env_llm_reply_provider(env_llm, sampling_params or SamplingParams(), system_prompt)
            if env_llm is not None
            else None
        )

        self._history: list[dict[str, str]] = []
        self._tracker = _StateContinuityTracker()
        # The finalizer must not reference self, so the live bridge sits in
        # a holder both sides share; close() drains it.
        self._bridge_holder: list[_EpisodeBridge] = []
        self._finalizer = weakref.finalize(self, _shutdown_bridges, self._bridge_holder)

    @property
    def _bridge(self) -> _EpisodeBridge | None:
        return self._bridge_holder[0] if self._bridge_holder else None

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._name,
            adapter="verifiers_v1",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={
                "taskset_id": self._name,
                "dataset_size": len(self._tasks),
                "infinite": self._infinite,
                "seats": sorted(self._agent_specs),
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return (self._native, *self._extra_rewards)

    def __len__(self) -> int:
        return len(self._tasks)

    def close(self) -> None:
        """Abandon any in-flight episode and reclaim its thread; idempotent."""
        _shutdown_bridges(self._bridge_holder)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[VerifiersV1Hidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        self.close()
        task = self._tasks[task_index]
        bridge = _EpisodeBridge(
            self._v1,
            self._env,
            task,
            self._agent_specs,
            env_id=self._env_id or self._name,
            env_llm_reply=self._env_llm_reply,
            max_turns=self._max_steps,
            step_timeout=self._step_timeout,
        )
        self._bridge_holder.append(bridge)
        kind, payload = bridge.start()
        if kind == "error":
            raise RuntimeError(f"verifiers_v1 episode failed: {payload}") from payload
        if kind == "done":
            raise RuntimeError(
                f"verifiers_v1 episode finished without consulting the policy: "
                f"{type(self._env).__name__}.run() never took a trainable turn"
            )

        system_prompt, prompt, remainder = _split_first_turn(payload.messages)
        if prompt is None:
            raise ValueError(
                f"verifiers_v1 env {self._name!r} opened the policy turn "
                "without a user message; the relay wrapper needs one for "
                "the observation prompt"
            )
        self._history = list(remainder)
        observation = Observation(
            prompt=prompt,
            task=ObservationContent(text=prompt),
            messages=tuple(remainder),
        )
        hidden = VerifiersV1Hidden(
            taskset_id=self._name,
            task_index=task_index,
            task_key=task.key,
        )
        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )
        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._tracker.track(state)

        info: dict[str, Any] = {
            "task_index": task_index,
            "taskset_id": self._name,
            "task_key": task.key,
            "system_prompt": system_prompt,
        }
        return state, info

    def step(
        self,
        state: State[VerifiersV1Hidden],
        action: Action,
    ) -> StepResult[VerifiersV1Hidden]:
        if state.metadata.is_terminal:
            raise ValueError(
                "cannot step a terminal state: the verifiers_v1 episode is "
                "over; call reset() to start a new one"
            )
        self._tracker.validate(state, "VerifiersV1MultiTurnEnvironment")
        if action.tool_calls:
            raise ValueError(
                "verifiers_v1 environments are chat relays: tool-call actions "
                "are not supported (send the reply as text)"
            )
        bridge = self._bridge
        if bridge is None:
            raise ValueError("no episode in flight; call reset() first")

        text = action.text or ""
        kind, payload = bridge.submit(text)
        if kind == "error":
            raise RuntimeError(f"verifiers_v1 episode failed: {payload}") from payload

        self._history.append({"role": "assistant", "content": text})
        if kind == "obs":
            return self._observation_step(state, payload)
        return self._terminal_step(state, action, payload)

    def _observation_step(
        self,
        state: State[VerifiersV1Hidden],
        turn: _PolicyTurn,
    ) -> StepResult[VerifiersV1Hidden]:
        new_messages = [m for m in turn.messages if m["role"] != "system"]
        self._history.extend(new_messages)
        state_text = new_messages[-1]["content"] if new_messages else None
        observation = Observation(
            prompt=state.observation.prompt,
            task=state.observation.task,
            messages=tuple(self._history),
            state=ObservationContent(text=state_text) if state_text is not None else None,
        )
        next_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=False,
            info=dict(state.metadata.info),
        )
        next_state = State(observation=observation, hidden=state.hidden, metadata=next_metadata)
        self._tracker.track(next_state)
        return StepResult(
            next_state=next_state,
            rewards=SignalBundle.empty(),
            terminated=False,
            truncated=False,
        )

    def _terminal_step(
        self,
        state: State[VerifiersV1Hidden],
        action: Action,
        result: _EpisodeResult,
    ) -> StepResult[VerifiersV1Hidden]:
        task = self._tasks[state.hidden.task_index]
        truncated = result.stop_condition in _TRUNCATING_STOPS
        observation = Observation(
            prompt=state.observation.prompt,
            task=state.observation.task,
            messages=tuple(self._history),
            state=state.observation.state,
        )
        hidden = replace(state.hidden, trace_results=result.trace_results)
        next_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=True,
            info={
                **state.metadata.info,
                "stop_condition": result.stop_condition,
                "verifiers_metrics": dict(result.metrics),
            },
        )
        next_state = State(observation=observation, hidden=hidden, metadata=next_metadata)
        self._tracker.track(next_state)
        rewards = self.compute_rewards(state, action, next_state)
        info: dict[str, Any] = {
            "stop_condition": result.stop_condition,
            "verifiers_metrics": dict(result.metrics),
            "trace_info": dict(result.trace_info),
            "runtime_skipped_signals": _runtime_skipped_signals(task),
            "episode_ok": result.episode_ok,
            "all_trace_rewards": result.all_trace_results,
        }
        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=not truncated,
            truncated=truncated,
            info=info,
        )

    def compute_rewards(
        self,
        state: State[VerifiersV1Hidden],
        action: Action,
        next_state: State[VerifiersV1Hidden],
    ) -> SignalBundle:
        signals = list(self._native.signals(next_state))
        for reward_fn in self._extra_rewards:
            signals.append(reward_fn.compute(state, action, next_state))
        return SignalBundle(signals=tuple(signals))


# ── Adapter ─────────────────────────────────────────────────────────


class VerifiersV1Adapter:
    """Adapter for verifiers v1 tasksets and environments.

    Requires a verifiers release that ships the ``verifiers.v1`` API.
    The legacy v0 API is served by the sibling ``VerifiersAdapter``; both
    surfaces ship in the same installed ``verifiers`` package.
    """

    @property
    def name(self) -> str:
        return "verifiers_v1"

    def _get_verifiers_v1(self) -> _V1Handle:
        """Import verifiers.v1 (registration probe + lazy import point)."""
        return _load_v1_handle()

    def list_environments(self) -> list[str]:
        """List the taskset ids shipped inside verifiers itself.

        Third-party taskset plugins are installed packages and cannot be
        enumerated; pass their id directly to ``get_environment()``.
        """
        self._get_verifiers_v1()
        import pkgutil

        import verifiers.v1.tasksets as tasksets_pkg

        return sorted(
            module.name.replace("_", "-") for module in pkgutil.iter_modules(tasksets_pkg.__path__)
        )

    def get_environment(
        self,
        name: str,
        *,
        env_id: str = "",
        taskset_params: dict[str, Any] | None = None,
        task_params: dict[str, Any] | None = None,
        env_params: dict[str, Any] | None = None,
        env_llm: Any = None,
        sampling_params: SamplingParams | None = None,
        system_prompt: str | None = None,
        size: int | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        step_timeout: float | None = None,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Build an environment for a verifiers-v1 taskset (or env+taskset pair).

        ``name`` is a taskset id, optionally paired as ``"<env_id>+<taskset_id>"``
        (verifiers' own env-id format); an explicit ``env_id`` kwarg wins over
        the name's prefix. Plain tasksets (``SingleAgentEnv``-routed) become
        single-turn environments; a custom Env class — the taskset's exported
        one or an explicitly paired one — runs through the multi-turn bridge.

        Args:
            name: Taskset id or ``"<env_id>+<taskset_id>"`` pair.
            env_id: Explicit env plugin id (overrides the name's prefix).
            taskset_params: Extra fields for the taskset's config.
            task_params: Task config fields, nested under the taskset config's
                ``task`` field (e.g. ``{"judges": []}`` disables judges).
            env_params: Extra fields for the env's config (multi-turn only).
            env_llm: Backend driving untrainable seats (multi-turn only).
            sampling_params: Sampling for ``env_llm`` (seat configs overlay it).
            system_prompt: Fallback system prompt for ``env_llm`` seats whose
                task carries none.
            size: Materialize at most this many tasks (required for INFINITE
                tasksets, which must be bounded for indexed access).
            seed: Opt-in ``Taskset.shuffle(seed)`` before materializing.
            max_steps: Cap on policy turns per episode (maps to the offline
                max_turns refusal check; exceeding it truncates).
            step_timeout: Seconds one ``reset()``/``step()`` may wait for the
                episode thread before raising TimeoutError.
            answer_extractor: Answer extractor (single-turn only).
            extra_rewards: Additional llenvs reward functions, appended after
                the native trace rewards.
        """
        v1 = self._get_verifiers_v1()

        parsed_env_id, _, taskset_id = name.rpartition("+")
        env_id = env_id or parsed_env_id
        env_cls = v1.loaders.environment_class(taskset_id, env_id)

        taskset_data: dict[str, Any] = {"id": taskset_id, **(taskset_params or {})}
        if task_params is not None:
            taskset_data["task"] = task_params
        config_cls = v1.loaders.taskset_config_type(taskset_id)
        taskset = v1.loaders.load_taskset(config_cls(**taskset_data))

        task_cls = type(taskset).task_type()
        if getattr(task_cls, "NEEDS_CONTAINER", False):
            raise NotImplementedError(
                f"taskset {taskset_id!r} tasks need a container runtime "
                "(NEEDS_CONTAINER); the offline verifiers_v1 adapter cannot "
                "provision containers"
            )
        if type(taskset).toolsets(taskset.config):
            raise NotImplementedError(
                f"taskset {taskset_id!r} declares toolsets, which run behind "
                "verifiers' interception server; the offline verifiers_v1 "
                "adapter cannot serve them"
            )

        infinite = bool(getattr(taskset, "INFINITE", False))
        if infinite and size is None:
            raise ValueError(
                f"taskset {taskset_id!r} is infinite; pass size= to bound it "
                "(the adapter materializes tasks for indexed access)"
            )
        if size is not None:
            taskset = taskset.head(size)
        if seed is not None:
            taskset = taskset.shuffle(seed)
        tasks = list(taskset)
        if not tasks:
            raise ValueError(f"taskset {taskset_id!r} yielded no tasks")

        self._refuse_unsupported_tasks(v1, taskset_id, tasks)

        if env_cls is v1.vf.SingleAgentEnv:
            return VerifiersV1SingleTurnEnvironment(
                v1,
                taskset_id,
                tasks,
                infinite=infinite,
                answer_extractor=answer_extractor,
                extra_rewards=extra_rewards,
            )

        env_config = v1.loaders.resolve_env_config(
            {"id": env_id, "taskset": taskset_data, **(env_params or {})}
        )
        env = env_cls(env_config)
        agent_specs = {
            field_name: getattr(env_config, field_name)
            for field_name, model_field in type(env_config).model_fields.items()
            if isinstance(model_field.default, v1.vf.AgentConfig)
        }
        return VerifiersV1MultiTurnEnvironment(
            v1,
            f"{env_id}+{taskset_id}" if env_id else taskset_id,
            env,
            tasks,
            agent_specs,
            env_llm=env_llm,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            max_steps=max_steps,
            step_timeout=step_timeout,
            extra_rewards=extra_rewards,
            env_id=env_id,
            infinite=infinite,
        )

    @staticmethod
    def _refuse_unsupported_tasks(v1: _V1Handle, taskset_id: str, tasks: list[Any]) -> None:
        """Refuse interception-dependent and non-text task features loudly."""
        first = tasks[0]
        if v1.discover_decorated(first, "intercept"):
            raise NotImplementedError(
                f"taskset {taskset_id!r} tasks declare @intercept hooks, which "
                "require verifiers' interception server; the offline "
                "verifiers_v1 adapter cannot run them"
            )
        non_trace_stops = sorted(
            fn.__name__
            for fn in first.hooks("stop")
            if v1.hook_boundary(fn, allow_trace=True) is not v1.vf.Trace
        )
        if non_trace_stops:
            raise NotImplementedError(
                f"taskset {taskset_id!r} has Request/Response-boundary @stop "
                f"hooks ({non_trace_stops}), which require verifiers' "
                "interception server; only Trace-boundary stops run offline"
            )
        if any(getattr(task.data, "image", None) is not None for task in tasks):
            raise NotImplementedError(
                f"taskset {taskset_id!r} tasks carry images; the verifiers_v1 adapter is text-only"
            )

    def get_native_answer_extractor(self, task_name: str) -> AnswerExtractor | None:
        return None

    def get_default_system_prompt(self, name: str) -> str | None:
        return None

    def get_prompt_template(self, name: str) -> Any:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        return {"name": name, "adapter": self.name}
