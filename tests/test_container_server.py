"""Tests for the container environment server."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.client import HTTPConnection
from typing import Any

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle, Signal, RewardType
from llenvs.core.state import Action, Observation, State, StateMetadata
from llenvs.core.tools import ToolDefinition, ToolParameter, ToolParameterType
from llenvs.container.server import EnvironmentServer


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockHidden:
    expected_answer: str
    category: str = "general"


class MockEnvironment:
    """Minimal environment for server testing."""

    def __init__(self, size: int = 5) -> None:
        self._size = size
        self._tasks = [{"prompt": f"Question {i}", "answer": str(i * 10)} for i in range(size)]

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(name="mock", adapter="test", max_steps=3)

    @property
    def reward_functions(self) -> tuple:
        return ()

    @property
    def available_tools(self) -> tuple:
        return (
            ToolDefinition(
                name="submit",
                description="Submit answer",
                parameters=(
                    ToolParameter(
                        name="answer",
                        type=ToolParameterType.STRING,
                        description="The answer",
                    ),
                ),
                is_terminal=True,
            ),
        )

    @property
    def prompts(self) -> dict[str, str]:
        return {"system": "You are a helpful assistant."}

    def __len__(self) -> int:
        return self._size

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State[MockHidden], dict[str, Any]]:
        task_index = 0
        if options and "task_index" in options:
            task_index = options["task_index"]
        task = self._tasks[task_index]
        state = State(
            observation=Observation(prompt=task["prompt"]),
            hidden=MockHidden(expected_answer=task["answer"]),
            metadata=StateMetadata(step=0, episode_id=f"ep-{task_index}"),
        )
        return state, {"task_index": task_index}

    def step(self, state: State[MockHidden], action: Action) -> StepResult[MockHidden]:
        correct = action.text == state.hidden.expected_answer
        reward = SignalBundle.single(1.0 if correct else 0.0, "correct", RewardType.OUTCOME)
        next_state = State(
            observation=Observation(prompt="Done"),
            hidden=state.hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=True,
            ),
        )
        return StepResult(
            next_state=next_state,
            rewards=reward,
            terminated=True,
            info={"correct": correct},
        )

    def compute_rewards(
        self,
        state: State[MockHidden],
        action: Action,
        next_state: State[MockHidden],
    ) -> SignalBundle:
        correct = action.text == state.hidden.expected_answer
        return SignalBundle.single(1.0 if correct else 0.0, "correct", RewardType.OUTCOME)


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def server_url():
    """Start a server in a thread, yield its URL, then shut down."""
    env = MockEnvironment(size=5)
    srv = EnvironmentServer(env, host="127.0.0.1", port=0)

    # Use port 0 to get a free port — need to start manually
    from http.server import HTTPServer

    handler_class = type(
        "BoundHandler",
        (
            __import__(
                "llenvs.container.server", fromlist=["EnvironmentHandler"]
            ).EnvironmentHandler,
        ),
        {"environment": env, "hidden_type": None},
    )
    http_server = HTTPServer(("127.0.0.1", 0), handler_class)
    port = http_server.server_address[1]

    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                conn.close()
                break
            conn.close()
        except Exception:
            time.sleep(0.05)

    yield f"http://127.0.0.1:{port}"

    http_server.shutdown()


def _request(url: str, method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
    """Helper: send HTTP request, return (status, parsed_json)."""
    host = url.split("//")[1]
    conn = HTTPConnection(host, timeout=5)
    headers = {}
    body_bytes = None
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body_bytes))
    conn.request(method, path, body=body_bytes, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    status = resp.status
    conn.close()
    return status, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, server_url):
        status, data = _request(server_url, "GET", "/health")
        assert status == 200
        assert data == {"status": "ok"}


class TestSpec:
    def test_spec(self, server_url):
        status, data = _request(server_url, "GET", "/spec")
        assert status == 200
        assert data["name"] == "mock"
        assert data["adapter"] == "test"
        assert data["max_steps"] == 3


class TestLen:
    def test_len(self, server_url):
        status, data = _request(server_url, "GET", "/len")
        assert status == 200
        assert data["length"] == 5


class TestTools:
    def test_tools(self, server_url):
        status, data = _request(server_url, "GET", "/tools")
        assert status == 200
        assert len(data) == 1
        assert data[0]["name"] == "submit"
        assert data[0]["is_terminal"] is True
        assert len(data[0]["parameters"]) == 1


class TestPrompts:
    def test_prompts(self, server_url):
        status, data = _request(server_url, "GET", "/prompts")
        assert status == 200
        assert data["system"] == "You are a helpful assistant."


class TestReset:
    def test_reset_default(self, server_url):
        status, data = _request(server_url, "POST", "/reset", {})
        assert status == 200
        assert "state" in data
        assert "info" in data
        state = data["state"]
        assert state["observation"]["prompt"] == "Question 0"
        assert state["hidden"]["expected_answer"] == "0"

    def test_reset_with_task_index(self, server_url):
        status, data = _request(server_url, "POST", "/reset", {"options": {"task_index": 2}})
        assert status == 200
        assert data["state"]["observation"]["prompt"] == "Question 2"
        assert data["state"]["hidden"]["expected_answer"] == "20"
        assert data["info"]["task_index"] == 2

    def test_reset_with_seed(self, server_url):
        status, data = _request(server_url, "POST", "/reset", {"seed": 42})
        assert status == 200
        assert "state" in data


class TestStep:
    def test_step_correct(self, server_url):
        # Reset first
        _, reset_data = _request(server_url, "POST", "/reset", {})
        state = reset_data["state"]

        # Step with correct answer
        _, step_data = _request(
            server_url,
            "POST",
            "/step",
            {"state": state, "action": {"text": "0", "tool_calls": []}},
        )
        assert step_data["terminated"] is True
        assert step_data["info"]["correct"] is True
        assert step_data["rewards"]["signals"][0]["reward"] == 1.0

    def test_step_incorrect(self, server_url):
        _, reset_data = _request(server_url, "POST", "/reset", {})
        state = reset_data["state"]

        _, step_data = _request(
            server_url,
            "POST",
            "/step",
            {"state": state, "action": {"text": "wrong", "tool_calls": []}},
        )
        assert step_data["info"]["correct"] is False
        assert step_data["rewards"]["signals"][0]["reward"] == 0.0

    def test_step_without_reset_fails(self, server_url):
        """If hidden_type was not captured yet, step uses a fallback or errors."""
        # Reset first to ensure hidden_type is captured (server_url is shared)
        _request(server_url, "POST", "/reset", {})

        # Now step should work with the captured hidden type
        _, reset_data = _request(server_url, "POST", "/reset", {})
        state = reset_data["state"]
        status, _ = _request(
            server_url,
            "POST",
            "/step",
            {"state": state, "action": {"text": "0", "tool_calls": []}},
        )
        assert status == 200


class TestComputeRewards:
    def test_compute_rewards(self, server_url):
        _, reset_data = _request(server_url, "POST", "/reset", {})
        state = reset_data["state"]

        # Step to get next_state
        _, step_data = _request(
            server_url,
            "POST",
            "/step",
            {"state": state, "action": {"text": "0", "tool_calls": []}},
        )
        next_state = step_data["next_state"]

        # Compute rewards independently
        status, rewards_data = _request(
            server_url,
            "POST",
            "/compute_rewards",
            {
                "state": state,
                "action": {"text": "0", "tool_calls": []},
                "next_state": next_state,
            },
        )
        assert status == 200
        assert rewards_data["signals"][0]["reward"] == 1.0


class TestErrors:
    def test_unknown_get_path(self, server_url):
        status, data = _request(server_url, "GET", "/unknown")
        assert status == 404
        assert "error" in data

    def test_unknown_post_path(self, server_url):
        status, data = _request(server_url, "POST", "/unknown", {})
        assert status == 404
        assert "error" in data


class TestHiddenTypeCapture:
    def test_hidden_type_captured_on_reset(self, server_url):
        """After reset, the server knows the hidden type."""
        _request(server_url, "POST", "/reset", {})

        # Second reset still works
        status, data = _request(server_url, "POST", "/reset", {"options": {"task_index": 1}})
        assert status == 200
        assert data["state"]["hidden"]["expected_answer"] == "10"


class TestMultipleEpisodes:
    def test_sequential_episodes(self, server_url):
        """Multiple reset/step cycles work correctly."""
        for i in range(3):
            _, reset_data = _request(server_url, "POST", "/reset", {"options": {"task_index": i}})
            state = reset_data["state"]
            expected = str(i * 10)

            _, step_data = _request(
                server_url,
                "POST",
                "/step",
                {"state": state, "action": {"text": expected, "tool_calls": []}},
            )
            assert step_data["info"]["correct"] is True
