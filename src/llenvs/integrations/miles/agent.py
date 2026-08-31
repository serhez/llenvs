"""miles agent function (``--custom-agent-function-path``).

Launch flags::

    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path llenvs.integrations.miles.agent.run
    --use-session-server v2

miles hands ``run`` a per-episode session ``base_url``; every chat-completions
request against it is recorded token-exact (TITO) as the episode's trainable
trajectory. The agent drives ``Environment.reset/step`` between requests.

Hard rules on this path:

- **Verbatim, append-only message history.** Assistant messages are echoed
  back exactly as returned (``message.model_dump(exclude_none=True)``,
  reasoning fields included); env feedback is appended, never rewritten.
  Anything else silently branches the v2 session tree into junk samples.
- **Text-only.** Image observations raise (use a vision-capable
  generate-function integration instead).
- **One fresh environment per episode**, created and cleaned up inside
  ``run`` — llenvs environments enforce state continuity and are sync, so
  ``reset``/``step`` run under ``asyncio.to_thread``.

The returned dict carries the Tier-0 scalar ``reward`` (sum of transition
totals, equal to ``Trajectory.total_reward``) plus per-turn ``reward_events``
keyed by chat-completion ``response_id`` for the v2 postprocessor's
span join (see ``postprocess``). Misconfigurations (missing ``task_index``,
vision env, judge/env-LLM pointed at the session server) raise so the miles
wrapper marks the sample ABORTED and ``check_no_aborted`` regenerates it —
failing visibly instead of training on garbage.

The whole episode runs under a wall-clock timeout,
``LLENVS_MILES_EPISODE_TIMEOUT`` seconds (default 3600).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from llenvs.core.state import Action, State
from llenvs.core.tools import ToolCall
from llenvs.inference.protocol import ChatMessage
from llenvs.integrations.miles import config as miles_config

logger = logging.getLogger(__name__)

EPISODE_TIMEOUT_ENV_VAR = "LLENVS_MILES_EPISODE_TIMEOUT"
DEFAULT_EPISODE_TIMEOUT = 3600.0


class MilesIntegrationError(RuntimeError):
    """Misconfiguration or unsupported feature — abort the sample loudly."""


def _client_factory(base_url: str) -> Any:
    """Build the session client. Module-level seam so tests can inject a
    mock transport."""
    from openai import AsyncOpenAI

    # No retries: the session server is local, and its 400/409 responses are
    # deterministic episode-ending signals that a retry would only mask.
    return AsyncOpenAI(base_url=f"{base_url}/v1", api_key="EMPTY", max_retries=0)


def _refuse_images(state: State[Any]) -> None:
    if state.observation.get_images():
        raise MilesIntegrationError(
            "Observation contains images; the miles TITO session path is text-only, "
            "so vision environments are not supported by this agent function."
        )


def build_initial_messages(
    state: State[Any], *, system_prompt: str | None = None
) -> list[dict[str, Any]]:
    """Build the episode's opening messages from a freshly-reset state.

    Mirrors the runner's raw legacy message shape ([system?] + user prompt +
    replay of ``obs.messages``) without any history shaping — the TITO path
    requires the history verbatim.
    """
    _refuse_images(state)
    obs = state.observation
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt).to_dict())
    messages.append(ChatMessage(role="user", content=obs.prompt).to_dict())

    for msg in obs.messages:
        role = msg.get("role", "user")
        if role == "assistant":
            tool_calls = tuple(
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in msg.get("tool_calls", [])
            )
            if tool_calls:
                messages.append(
                    ChatMessage(
                        role="assistant", content=msg.get("content"), tool_calls=tool_calls
                    ).to_dict()
                )
            else:
                messages.append(
                    ChatMessage(role="assistant", content=msg.get("content", "")).to_dict()
                )
        elif role == "tool":
            messages.append(
                ChatMessage(
                    role="tool",
                    content=msg.get("content", ""),
                    tool_call_id=msg.get("tool_call_id"),
                    name=msg.get("name"),
                ).to_dict()
            )
        else:
            messages.append(ChatMessage(role="user", content=msg.get("content", "")).to_dict())

    return messages


def _action_from_message(message: Any) -> Action:
    """Convert an OpenAI chat-completion message into an llenvs Action."""
    tool_calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        try:
            arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            arguments = None
        if not isinstance(arguments, dict):
            logger.warning(
                "Malformed tool-call arguments for %s: %r; substituting {}",
                tc.function.name,
                tc.function.arguments,
            )
            arguments = {}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    if tool_calls:
        return Action(text=message.content, tool_calls=tuple(tool_calls))
    return Action.from_text(message.content or "")


def _tool_result_content(result: Any) -> str:
    output = result.output
    content = json.dumps(output) if isinstance(output, dict) else str(output)
    if not content:
        content = result.error or "(no output)"
    return content


def _feedback_messages(state: State[Any]) -> list[dict[str, Any]]:
    """Messages to append after an env step (tool results or user feedback)."""
    obs = state.observation
    if obs.tool_results:
        return [
            {
                "role": "tool",
                "tool_call_id": r.call_id,
                "name": r.tool_name,
                "content": _tool_result_content(r),
            }
            for r in obs.tool_results
        ]
    if obs.state is not None and obs.state.text:
        content: str = obs.state.text
    elif obs.messages and obs.messages[-1].get("role") == "user":
        content = obs.messages[-1].get("content", "")
    else:
        content = obs.prompt
    return [{"role": "user", "content": content}]


def _close_environment(env: Any) -> None:
    """Best-effort cleanup — the Environment protocol has no close()."""
    for method_name in ("close", "shutdown"):
        method = getattr(env, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                logger.warning("Environment %s() failed during cleanup", method_name)
            return


def _result(
    *,
    reward: float,
    exit_status: str,
    env_truncated: bool = False,
    num_steps: int = 0,
    turn_rewards: list[float] | None = None,
    reward_events: list[dict[str, Any]] | None = None,
    signals: dict[str, float] | None = None,
    total_tool_time: float = 0.0,
) -> dict[str, Any]:
    return {
        "reward": float(reward),
        "exit_status": exit_status,
        "env_truncated": env_truncated,
        "num_steps": num_steps,
        "turn_rewards": turn_rewards or [],
        "reward_events": reward_events or [],
        "signals": signals or {},
        "agent_metrics": {"total_tool_time": total_tool_time},
    }


async def _run_episode(
    base_url: str, request_kwargs: dict[str, Any], md: dict[str, Any]
) -> dict[str, Any]:
    from openai import APIStatusError

    client = _client_factory(base_url)
    env: Any = None
    try:
        env = await asyncio.to_thread(miles_config.create_environment, md)
        state, _ = await asyncio.to_thread(env.reset, options={"task_index": int(md["task_index"])})
        _refuse_images(state)
        system_prompt = miles_config.resolve_system_prompt_for(md)
        messages = build_initial_messages(state, system_prompt=system_prompt)
        max_steps = int(md.get("max_steps") or env.spec.max_steps or 100)

        turn_rewards: list[float] = []
        reward_events: list[dict[str, Any]] = []
        signals: dict[str, float] = {}
        total_tool_time = 0.0
        env_truncated = False
        exit_status = "max_steps"
        num_steps = 0

        for _ in range(max_steps):
            tools = [t.to_openai_schema() for t in state.observation.available_tools]
            try:
                completion = await client.chat.completions.create(
                    model="llenvs",
                    messages=messages,
                    extra_body=request_kwargs or None,
                    **({"tools": tools} if tools else {}),
                )
            except APIStatusError as exc:
                # 400 = context overflow; 409 = extending a truncated v2 node.
                if exc.status_code in (400, 409):
                    exit_status = "context_overflow"
                    break
                raise
            message = completion.choices[0].message
            # Verbatim echo — reasoning/extra fields included (see module docstring).
            messages.append(message.model_dump(exclude_none=True))
            action = _action_from_message(message)

            step_start = time.monotonic()
            try:
                result = await asyncio.to_thread(env.step, state, action)
            except Exception:
                logger.exception("Environment step failed (task_index=%s)", md["task_index"])
                return _result(
                    reward=0.0,
                    exit_status="env_error",
                    env_truncated=env_truncated,
                    num_steps=num_steps,
                    turn_rewards=turn_rewards,
                    reward_events=reward_events,
                    signals=signals,
                    total_tool_time=total_tool_time,
                )
            total_tool_time += time.monotonic() - step_start
            num_steps += 1

            transition_total = result.rewards.total
            turn_rewards.append(transition_total)
            for signal in result.rewards.signals:
                if signal.reward is not None:
                    signals[signal.name] = signals.get(signal.name, 0.0) + signal.reward
            reward_events.append(
                {
                    "response_id": completion.id,
                    "value": transition_total,
                    "signals": {s.name: s.reward for s in result.rewards.signals},
                }
            )

            state = result.next_state
            _refuse_images(state)
            env_truncated = env_truncated or result.truncated
            if result.terminated or result.truncated:
                exit_status = "completed"
                break
            messages.extend(_feedback_messages(state))

        return _result(
            reward=sum(turn_rewards),
            exit_status=exit_status,
            env_truncated=env_truncated,
            num_steps=num_steps,
            turn_rewards=turn_rewards,
            reward_events=reward_events,
            signals=signals,
            total_tool_time=total_tool_time,
        )
    finally:
        try:
            if env is not None:
                _close_environment(env)
        finally:
            await client.close()


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run one environment episode against the TITO session at ``base_url``.

    ``prompt`` (the prompt-data row's messages) is unused: the environment is
    the source of truth, and the exported row was built from the same
    ``reset`` observation. ``metadata`` must carry ``task_index``; optional
    keys: ``max_steps``, ``llenvs_config``, ``llenvs_env_name``.

    Returns the miles agent dict, or ``None`` on unexpected internal errors
    (miles marks the sample ABORTED).
    """
    md = dict(metadata or {})
    if "task_index" not in md:
        raise ValueError(
            "metadata['task_index'] is required to reset the environment to the "
            "sample's task. Export prompt data with llenvs.integrations.miles.data "
            "so every row carries it."
        )
    miles_config.ensure_isolated_from_session(base_url, md)

    timeout = float(os.environ.get(EPISODE_TIMEOUT_ENV_VAR, str(DEFAULT_EPISODE_TIMEOUT)))
    try:
        return await asyncio.wait_for(_run_episode(base_url, request_kwargs or {}, md), timeout)
    except TimeoutError:
        logger.error("Episode timed out after %.0fs (task_index=%s)", timeout, md["task_index"])
        return _result(reward=0.0, exit_status="timeout")
    except MilesIntegrationError:
        raise
    except Exception:
        logger.exception(
            "Unexpected error in miles agent episode (task_index=%s)", md["task_index"]
        )
        return None


async def abort(args: Any) -> None:
    """Rollout-teardown hook (``run``'s sibling in the agent-function contract).

    A no-op: every episode creates and cleans up its own environment inside
    ``run``. Reserved for future environment pooling.
    """
    return None
