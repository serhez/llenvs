"""Tests for the miles agent function (``--custom-agent-function-path``).

The agent drives ``Environment.reset/step`` against the TITO session server's
OpenAI-compatible endpoint. Tests replace the endpoint with an in-process
``httpx.MockTransport`` (via the ``agent._client_factory`` seam) that serves
scripted completions and records every request body, so the exact wire
behavior — verbatim message echo, tool schemas, failure handling — is pinned
without a real server.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardType, Signal, SignalBundle
from llenvs.core.state import (
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
from llenvs.integrations.miles import agent as miles_agent
from llenvs.integrations.miles import config as miles_config

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

BASE_URL = "http://localhost:7999/sessions/test-episode"

# ---------------------------------------------------------------------------
# Fake session server
# ---------------------------------------------------------------------------


class FakeSession:
    """Scripted OpenAI-compatible chat endpoint backed by httpx.MockTransport.

    ``replies`` entries are either assistant-message payload dicts or an int
    HTTP status code (served as an error response). Request JSON bodies are
    recorded in ``requests``.
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self._counter = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content.decode()))
        reply = self.replies.pop(0) if self.replies else {"role": "assistant", "content": ""}
        if isinstance(reply, int):
            return httpx.Response(
                reply,
                json={"error": {"message": "boom", "type": "invalid_request_error"}},
            )
        self._counter += 1
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{self._counter}",
                "object": "chat.completion",
                "created": 0,
                "model": "llenvs",
                "choices": [{"index": 0, "message": reply, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


@pytest.fixture()
def fake_session(monkeypatch):
    """Install scripted replies; returns the FakeSession for assertions."""

    def _install(replies: list[Any]) -> FakeSession:
        session = FakeSession(replies)

        def factory(base_url: str):
            return openai.AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="EMPTY",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(session.handler)),
            )

        monkeypatch.setattr(miles_agent, "_client_factory", factory)
        return session

    return _install


@pytest.fixture()
def patched_config(monkeypatch):
    """Bypass YAML discovery: serve a given env factory + system prompt."""

    def _install(env_factory, system_prompt: str | None = None) -> None:
        monkeypatch.setattr(miles_config, "create_environment", lambda md=None: env_factory())
        monkeypatch.setattr(
            miles_config, "resolve_system_prompt_for", lambda md=None: system_prompt
        )
        monkeypatch.setattr(miles_config, "ensure_isolated_from_session", lambda url, md=None: None)

    return _install


def _run(**kwargs: Any) -> dict[str, Any] | None:
    defaults: dict[str, Any] = {
        "base_url": BASE_URL,
        "prompt": [{"role": "user", "content": "unused"}],
        "metadata": {"task_index": 0},
    }
    defaults.update(kwargs)
    return asyncio.run(miles_agent.run(**defaults))


# ---------------------------------------------------------------------------
# Mock environments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Hidden:
    task_index: int


def _bundle(value: float) -> SignalBundle:
    return SignalBundle(
        signals=(Signal(reward=value, name="step_reward", reward_type=RewardType.OUTCOME),)
    )


class MockMultiTurnEnv:
    """Text env: reward n on step n, terminates after ``total_steps`` steps."""

    def __init__(self, total_steps: int = 2) -> None:
        self.total_steps = total_steps
        self.reset_options: dict[str, Any] | None = None
        self.step_actions: list[Any] = []
        self.step_sleep = 0.0
        self.truncate_on_step: int | None = None

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock-mt", adapter="mock", max_steps=8, is_multi_turn=True)

    def reset(self, *, seed=None, options=None):
        self.reset_options = dict(options or {})
        idx = self.reset_options.get("task_index", 0)
        state = State(
            observation=Observation(prompt=f"Task {idx}: do things."),
            hidden=_Hidden(idx),
            metadata=StateMetadata(step=0, episode_id=f"ep{idx}"),
        )
        return state, {}

    def step(self, state, action):
        if self.step_sleep:
            time.sleep(self.step_sleep)
        self.step_actions.append(action)
        n = state.metadata.step + 1
        terminated = n >= self.total_steps
        truncated = self.truncate_on_step is not None and n >= self.truncate_on_step
        next_state = State(
            observation=Observation(
                prompt=state.observation.prompt,
                state=ObservationContent(text=f"obs after step {n}"),
            ),
            hidden=state.hidden,
            metadata=StateMetadata(
                step=n,
                episode_id=state.metadata.episode_id,
                is_terminal=terminated or truncated,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=_bundle(float(n)),
            terminated=terminated,
            truncated=truncated,
            info={},
        )


LOOKUP_TOOL = ToolDefinition(
    name="lookup",
    description="Look something up.",
    parameters=(ToolParameter(name="q", type=ToolParameterType.STRING, description="Query."),),
)


class MockToolEnv(MockMultiTurnEnv):
    """Tool env: echoes tool calls back as tool_results."""

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        return (
            State(
                observation=Observation(
                    prompt=state.observation.prompt, available_tools=(LOOKUP_TOOL,)
                ),
                hidden=state.hidden,
                metadata=state.metadata,
            ),
            info,
        )

    def step(self, state, action):
        result = super().step(state, action)
        tool_results = tuple(
            ToolResult.success(call_id=tc.id, tool_name=tc.name, output={"value": 42})
            for tc in action.tool_calls
        )
        next_state = State(
            observation=Observation(
                prompt=result.next_state.observation.prompt,
                tool_results=tool_results,
                available_tools=(LOOKUP_TOOL,),
            ),
            hidden=result.next_state.hidden,
            metadata=result.next_state.metadata,
        )
        return StepResult(
            next_state=next_state,
            rewards=result.rewards,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )


class MockVisionEnv(MockMultiTurnEnv):
    """Env whose observations carry images — must be refused on the TITO path."""

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        img = ImageContent(data="aGk=", media_type="image/png")
        return (
            State(
                observation=Observation(
                    prompt=state.observation.prompt,
                    state=ObservationContent(text="look", images=(img,)),
                ),
                hidden=state.hidden,
                metadata=state.metadata,
            ),
            info,
        )


class MockErrorEnv(MockMultiTurnEnv):
    def step(self, state, action):
        raise RuntimeError("env exploded")


# ---------------------------------------------------------------------------
# build_initial_messages
# ---------------------------------------------------------------------------


class TestBuildInitialMessages:
    def _state(self, **obs_kwargs: Any) -> State[Any]:
        obs_kwargs.setdefault("prompt", "Solve it.")
        return State(
            observation=Observation(**obs_kwargs),
            hidden=_Hidden(0),
            metadata=StateMetadata(step=0, episode_id="ep0"),
        )

    def test_system_and_user(self):
        messages = miles_agent.build_initial_messages(self._state(), system_prompt="Be brief.")
        assert messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Solve it."},
        ]

    def test_no_system_prompt(self):
        messages = miles_agent.build_initial_messages(self._state())
        assert messages == [{"role": "user", "content": "Solve it."}]

    def test_obs_messages_replayed(self):
        state = self._state(
            messages=(
                {"role": "assistant", "content": "first try"},
                {"role": "user", "content": "try again"},
            )
        )
        messages = miles_agent.build_initial_messages(state)
        assert messages == [
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": "first try"},
            {"role": "user", "content": "try again"},
        ]

    def test_tool_history_replayed(self):
        state = self._state(
            messages=(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
                },
                {"role": "tool", "content": "42", "tool_call_id": "c1", "name": "lookup"},
            )
        )
        messages = miles_agent.build_initial_messages(state)
        assistant = messages[1]
        assert assistant["tool_calls"][0]["id"] == "c1"
        assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
        tool = messages[2]
        assert tool == {"role": "tool", "content": "42", "tool_call_id": "c1", "name": "lookup"}

    def test_images_raise(self):
        img = ImageContent(data="aGk=", media_type="image/png")
        state = self._state(state=ObservationContent(text="look", images=(img,)))
        with pytest.raises(RuntimeError, match="image"):
            miles_agent.build_initial_messages(state)


# ---------------------------------------------------------------------------
# Agent loop — text envs
# ---------------------------------------------------------------------------


class TestAgentLoopText:
    def test_contract_keys(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}])
        patched_config(MockMultiTurnEnv)
        result = _run()
        assert result is not None
        assert {
            "reward",
            "exit_status",
            "env_truncated",
            "num_steps",
            "turn_rewards",
            "reward_events",
            "signals",
            "agent_metrics",
        } <= set(result)

    def test_reward_is_sum_of_transition_totals(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}])
        patched_config(MockMultiTurnEnv)
        result = _run()
        assert result["reward"] == 3.0  # 1.0 + 2.0
        assert result["turn_rewards"] == [1.0, 2.0]
        assert result["exit_status"] == "completed"
        assert result["num_steps"] == 2
        assert result["signals"] == {"step_reward": 3.0}

    def test_verbatim_assistant_echo(self, fake_session, patched_config):
        """Extra fields like reasoning_content must survive the echo byte-identical
        — anything else silently branches the v2 session tree."""
        session = fake_session(
            [
                {
                    "role": "assistant",
                    "content": "step one",
                    "reasoning_content": "thinking hard",
                },
                {"role": "assistant", "content": "step two"},
            ]
        )
        patched_config(MockMultiTurnEnv)
        _run()
        second_request = session.requests[1]["messages"]
        echoed = [m for m in second_request if m["role"] == "assistant"][-1]
        assert echoed["content"] == "step one"
        assert echoed["reasoning_content"] == "thinking hard"

    def test_system_prompt_and_task_prompt_sent(self, fake_session, patched_config):
        session = fake_session([{"role": "assistant", "content": "a"}])
        patched_config(lambda: MockMultiTurnEnv(total_steps=1), system_prompt="Be brief.")
        _run()
        first = session.requests[0]["messages"]
        assert first[0] == {"role": "system", "content": "Be brief."}
        assert first[1] == {"role": "user", "content": "Task 0: do things."}

    def test_feedback_appended_as_user_message(self, fake_session, patched_config):
        session = fake_session(
            [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        patched_config(MockMultiTurnEnv)
        _run()
        assert session.requests[1]["messages"][-1] == {
            "role": "user",
            "content": "obs after step 1",
        }

    def test_reward_events_match_completion_ids(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}])
        patched_config(MockMultiTurnEnv)
        result = _run()
        assert [e["response_id"] for e in result["reward_events"]] == [
            "chatcmpl-1",
            "chatcmpl-2",
        ]
        assert [e["value"] for e in result["reward_events"]] == [1.0, 2.0]
        assert result["reward_events"][0]["signals"] == {"step_reward": 1.0}

    def test_max_steps_from_metadata(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "x"}] * 5)
        patched_config(lambda: MockMultiTurnEnv(total_steps=10))
        result = _run(metadata={"task_index": 0, "max_steps": 2})
        assert result["num_steps"] == 2
        assert result["exit_status"] == "max_steps"

    def test_max_steps_defaults_to_spec(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "x"}] * 20)
        patched_config(lambda: MockMultiTurnEnv(total_steps=100))  # spec.max_steps == 8
        result = _run()
        assert result["num_steps"] == 8
        assert result["exit_status"] == "max_steps"

    def test_request_kwargs_forwarded(self, fake_session, patched_config):
        session = fake_session([{"role": "assistant", "content": "a"}])
        patched_config(lambda: MockMultiTurnEnv(total_steps=1))
        _run(request_kwargs={"temperature": 0.7, "top_p": 0.9})
        assert session.requests[0]["temperature"] == 0.7
        assert session.requests[0]["top_p"] == 0.9

    def test_task_index_reaches_env_reset(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}])
        env = MockMultiTurnEnv(total_steps=1)
        patched_config(lambda: env)
        _run(metadata={"task_index": 3})
        assert env.reset_options == {"task_index": 3}

    def test_fresh_env_per_episode(self, fake_session, patched_config):
        created: list[MockMultiTurnEnv] = []

        def factory():
            env = MockMultiTurnEnv(total_steps=1)
            created.append(env)
            return env

        patched_config(factory)
        fake_session([{"role": "assistant", "content": "a"}])
        _run()
        fake_session([{"role": "assistant", "content": "a"}])
        _run()
        assert len(created) == 2

    def test_empty_content_becomes_empty_text_action(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": ""}])
        env = MockMultiTurnEnv(total_steps=1)
        patched_config(lambda: env)
        _run()
        assert env.step_actions[0].text == ""
        assert not env.step_actions[0].tool_calls


# ---------------------------------------------------------------------------
# Agent loop — tool envs
# ---------------------------------------------------------------------------


def _tool_call_reply(arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": arguments},
            }
        ],
    }


class TestAgentLoopTools:
    def test_tool_schema_sent_every_request(self, fake_session, patched_config):
        session = fake_session(
            [_tool_call_reply('{"q": "x"}'), {"role": "assistant", "content": "done"}]
        )
        patched_config(MockToolEnv)
        _run()
        for request in session.requests:
            assert request["tools"][0]["function"]["name"] == "lookup"

    def test_tool_calls_become_action(self, fake_session, patched_config):
        fake_session([_tool_call_reply('{"q": "x"}'), {"role": "assistant", "content": "done"}])
        env = MockToolEnv()
        patched_config(lambda: env)
        _run()
        action = env.step_actions[0]
        assert action.tool_calls[0].name == "lookup"
        assert action.tool_calls[0].arguments == {"q": "x"}
        assert action.tool_calls[0].id == "call_1"

    def test_tool_results_fed_back_as_tool_messages(self, fake_session, patched_config):
        session = fake_session(
            [_tool_call_reply('{"q": "x"}'), {"role": "assistant", "content": "done"}]
        )
        patched_config(MockToolEnv)
        _run()
        last = session.requests[1]["messages"][-1]
        assert last["role"] == "tool"
        assert last["tool_call_id"] == "call_1"
        assert json.loads(last["content"]) == {"value": 42}

    def test_malformed_arguments_fall_back_to_empty(self, fake_session, patched_config):
        fake_session([_tool_call_reply("not json"), {"role": "assistant", "content": "done"}])
        env = MockToolEnv()
        patched_config(lambda: env)
        _run()
        assert env.step_actions[0].tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# Failure taxonomy & guards
# ---------------------------------------------------------------------------


class TestAgentFailureTaxonomy:
    def test_missing_task_index_raises(self, fake_session, patched_config):
        fake_session([])
        patched_config(MockMultiTurnEnv)
        with pytest.raises(ValueError, match="task_index"):
            _run(metadata={})

    def test_vision_env_refused(self, fake_session, patched_config):
        fake_session([])
        patched_config(MockVisionEnv)
        with pytest.raises(RuntimeError, match="image"):
            _run()

    def test_isolation_guard_enforced(self, fake_session, patched_config, monkeypatch):
        fake_session([])
        patched_config(MockMultiTurnEnv)

        def guard(url, md=None):
            raise ValueError("judge targets the session endpoint")

        monkeypatch.setattr(miles_config, "ensure_isolated_from_session", guard)
        with pytest.raises(ValueError, match="session"):
            _run()

    def test_http_400_is_context_overflow(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}, 400])
        patched_config(lambda: MockMultiTurnEnv(total_steps=5))
        result = _run()
        assert result["exit_status"] == "context_overflow"
        assert result["reward"] == 1.0  # keeps the partial sum
        assert result["num_steps"] == 1

    def test_http_409_is_context_overflow(self, fake_session, patched_config):
        fake_session([409])
        patched_config(MockMultiTurnEnv)
        result = _run()
        assert result["exit_status"] == "context_overflow"
        assert result["num_steps"] == 0

    def test_env_error_zeroes_reward(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}])
        patched_config(MockErrorEnv)
        result = _run()
        assert result["exit_status"] == "env_error"
        assert result["reward"] == 0.0

    def test_episode_timeout(self, fake_session, patched_config, monkeypatch):
        monkeypatch.setenv("LLENVS_MILES_EPISODE_TIMEOUT", "0.1")
        fake_session([{"role": "assistant", "content": "a"}] * 3)
        env = MockMultiTurnEnv(total_steps=3)
        env.step_sleep = 0.5
        patched_config(lambda: env)
        result = _run()
        assert result["exit_status"] == "timeout"
        assert result["reward"] == 0.0

    def test_total_tool_time_measured(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}])
        env = MockMultiTurnEnv(total_steps=1)
        env.step_sleep = 0.02
        patched_config(lambda: env)
        result = _run()
        assert result["agent_metrics"]["total_tool_time"] >= 0.02

    def test_env_truncation_flag(self, fake_session, patched_config):
        fake_session([{"role": "assistant", "content": "a"}])
        env = MockMultiTurnEnv(total_steps=5)
        env.truncate_on_step = 1
        patched_config(lambda: env)
        result = _run()
        assert result["env_truncated"] is True
        assert result["exit_status"] == "completed"


class TestAbort:
    def test_abort_is_a_noop(self):
        assert asyncio.run(miles_agent.abort(None)) is None
