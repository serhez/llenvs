"""Tests for the container environment client (proxy)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llenvs.container.client import ContainerEnvironment, ContainerEnvironmentError
from llenvs.container.serialization import OpaqueHidden
from llenvs.core.environment import EnvironmentSpec
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Action, State

# ---------------------------------------------------------------------------
# Mock HTTP server
# ---------------------------------------------------------------------------


MOCK_SPEC = {
    "name": "mock",
    "adapter": "test",
    "max_steps": 5,
    "is_multi_turn": False,
    "supports_task_index": True,
    "supports_len": True,
    "supports_seed": True,
    "metadata": {},
}

MOCK_TOOLS = [
    {
        "name": "submit",
        "description": "Submit answer",
        "parameters": [
            {"name": "answer", "type": "string", "description": "The answer", "required": True}
        ],
        "is_terminal": True,
    }
]

MOCK_PROMPTS = {"system": "Be helpful."}


def _make_state_dict(prompt: str = "What is 2+2?", answer: str = "4", step: int = 0):
    return {
        "observation": {
            "prompt": prompt,
            "messages": [],
            "tool_results": [],
            "available_tools": [],
        },
        "hidden": {"expected_answer": answer, "category": "math"},
        "metadata": {
            "step": step,
            "episode_id": "ep-1",
            "is_terminal": False,
            "info": {},
        },
    }


class MockHandler(BaseHTTPRequestHandler):
    """Simple mock handler for testing the client."""

    def log_message(self, format, *args):
        pass  # Suppress output

    def do_GET(self):
        routes = {
            "/health": lambda: {"status": "ok"},
            "/spec": lambda: MOCK_SPEC,
            "/len": lambda: {"length": 10},
            "/tools": lambda: MOCK_TOOLS,
            "/prompts": lambda: MOCK_PROMPTS,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._send(404, {"error": {"type": "NotFound", "message": f"Unknown: {self.path}"}})
            return
        self._send(200, handler())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        body = json.loads(raw)

        if self.path == "/reset":
            state = _make_state_dict()
            self._send(200, {"state": state, "info": {"task_index": 0}})
        elif self.path == "/step":
            action_text = body.get("action", {}).get("text", "")
            correct = action_text == "4"
            next_state = _make_state_dict(prompt="Done", step=1)
            next_state["metadata"]["is_terminal"] = True
            self._send(
                200,
                {
                    "next_state": next_state,
                    "rewards": {
                        "signals": [
                            {
                                "reward": 1.0 if correct else 0.0,
                                "name": "correct",
                                "reward_type": "OUTCOME",
                                "metadata": None,
                                "weight": 1.0,
                            }
                        ]
                    },
                    "terminated": True,
                    "truncated": False,
                    "info": {"correct": correct},
                },
            )
        elif self.path == "/compute_rewards":
            action_text = body.get("action", {}).get("text", "")
            correct = action_text == "4"
            self._send(
                200,
                {
                    "signals": [
                        {
                            "reward": 1.0 if correct else 0.0,
                            "name": "correct",
                            "reward_type": "OUTCOME",
                            "metadata": None,
                            "weight": 1.0,
                        }
                    ]
                },
            )
        elif self.path == "/error":
            self._send(
                500,
                {
                    "error": {
                        "type": "ValueError",
                        "message": "Something went wrong",
                        "traceback": "Traceback ...",
                    }
                },
            )
        else:
            self._send(404, {"error": {"type": "NotFound", "message": "Unknown path"}})

    def _send(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def mock_server():
    """Start a mock HTTP server in a thread."""
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Wait for server readiness
    import http.client

    for _ in range(50):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            break
        except Exception:
            time.sleep(0.05)

    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def client(mock_server):
    """Create a ContainerEnvironment connected to the mock server."""
    env = ContainerEnvironment(mock_server, timeout=5.0)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpec:
    def test_spec(self, client):
        spec = client.spec
        assert isinstance(spec, EnvironmentSpec)
        assert spec.name == "mock"
        assert spec.adapter == "test"
        assert spec.max_steps == 5

    def test_spec_cached(self, client):
        spec1 = client.spec
        spec2 = client.spec
        assert spec1 is spec2


class TestLen:
    def test_len(self, client):
        assert len(client) == 10

    def test_len_cached(self, client):
        _ = len(client)
        # Clear connection to prove caching
        client._conn = None
        assert len(client) == 10  # Uses cached value


class TestTools:
    def test_available_tools(self, client):
        tools = client.available_tools
        assert len(tools) == 1
        assert tools[0].name == "submit"
        assert tools[0].is_terminal is True
        assert len(tools[0].parameters) == 1

    def test_tools_cached(self, client):
        tools1 = client.available_tools
        tools2 = client.available_tools
        assert tools1 is tools2


class TestPrompts:
    def test_prompts(self, client):
        prompts = client.prompts
        assert prompts == {"system": "Be helpful."}

    def test_prompts_cached(self, client):
        p1 = client.prompts
        p2 = client.prompts
        assert p1 is p2


class TestRewardFunctions:
    def test_empty(self, client):
        assert client.reward_functions == ()


class TestReset:
    def test_reset(self, client):
        state, info = client.reset()
        assert isinstance(state, State)
        assert isinstance(state.hidden, OpaqueHidden)
        assert state.observation.prompt == "What is 2+2?"
        assert state.hidden.expected_answer == "4"
        assert info["task_index"] == 0

    def test_reset_with_options(self, client):
        state, info = client.reset(options={"task_index": 3})
        assert isinstance(state, State)

    def test_reset_with_seed(self, client):
        state, info = client.reset(seed=42)
        assert isinstance(state, State)


class TestStep:
    def test_step_correct(self, client):
        state, _ = client.reset()
        action = Action.from_text("4")
        result = client.step(state, action)
        assert result.terminated is True
        assert result.info["correct"] is True
        assert result.rewards.total == 1.0

    def test_step_incorrect(self, client):
        state, _ = client.reset()
        action = Action.from_text("wrong")
        result = client.step(state, action)
        assert result.terminated is True
        assert result.info["correct"] is False
        assert result.rewards.total == 0.0

    def test_step_returns_opaque_hidden(self, client):
        state, _ = client.reset()
        action = Action.from_text("4")
        result = client.step(state, action)
        assert isinstance(result.next_state.hidden, OpaqueHidden)


class TestComputeRewards:
    def test_compute_rewards(self, client):
        state, _ = client.reset()
        action = Action.from_text("4")
        result = client.step(state, action)
        rewards = client.compute_rewards(state, action, result.next_state)
        assert isinstance(rewards, SignalBundle)
        assert rewards.total == 1.0


class TestErrorHandling:
    def test_server_error(self, mock_server):
        """Server 500 errors are raised as ContainerEnvironmentError."""
        env = ContainerEnvironment(mock_server, timeout=5.0)
        with pytest.raises(ContainerEnvironmentError, match="Something went wrong"):
            env._request("POST", "/error", {})
        env.close()

    def test_not_found(self, mock_server):
        env = ContainerEnvironment(mock_server, timeout=5.0)
        with pytest.raises(ContainerEnvironmentError, match="Unknown"):
            env._request("GET", "/nonexistent")
        env.close()


class TestReconnection:
    def test_reconnect_on_closed_connection(self, mock_server):
        """Client reconnects if connection was closed."""
        env = ContainerEnvironment(mock_server, timeout=5.0)
        # Make a successful request
        spec = env.spec
        assert spec.name == "mock"

        # Forcibly close the connection
        env._conn.close()
        env._conn = None

        # Next request should reconnect automatically
        state, _ = env.reset()
        assert state.observation.prompt == "What is 2+2?"
        env.close()


class TestClose:
    def test_close(self, client):
        client.close()
        assert client._conn is None

    def test_close_idempotent(self, client):
        client.close()
        client.close()  # Should not raise
