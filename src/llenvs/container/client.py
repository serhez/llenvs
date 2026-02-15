"""Container environment proxy that delegates to a remote server over HTTP.

``ContainerEnvironment`` implements the ``Environment`` protocol, sending all
calls to an ``EnvironmentServer`` over JSON/HTTP.  Uses stdlib
``http.client.HTTPConnection`` for persistent connections.
"""

from __future__ import annotations

import http.client
import json
from typing import Any
from urllib.parse import urlparse

from llenvs.container.serialization import (
    OpaqueHidden,
    deserialize_env_spec,
    deserialize_reward_bundle,
    deserialize_state,
    deserialize_step_result,
    deserialize_tool_definition,
    serialize_action,
    serialize_state,
)
from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Action, State


class ContainerEnvironmentError(Exception):
    """Error from a containerized environment server."""


class ContainerEnvironment:
    """Proxy that delegates to an environment running in a container/subprocess.

    Implements the ``Environment[OpaqueHidden]`` protocol.  All reward
    computation happens server-side; ``reward_functions`` returns ``()``.

    Args:
        url: Base URL of the environment server (e.g. ``http://localhost:9123``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout
        parsed = urlparse(url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._conn: http.client.HTTPConnection | None = None
        # Cached properties
        self._spec: EnvironmentSpec | None = None
        self._length: int | None = None
        self._tools: tuple | None = None
        self._prompts: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _get_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        return self._conn

    def _request(self, method: str, path: str, body: dict | list | None = None) -> Any:
        """Send an HTTP request and return parsed JSON response.

        Reconnects once on connection errors.
        """
        for attempt in range(2):
            try:
                conn = self._get_conn()
                headers: dict[str, str] = {}
                body_bytes: bytes | None = None
                if body is not None:
                    body_bytes = json.dumps(body).encode("utf-8")
                    headers["Content-Type"] = "application/json"
                    headers["Content-Length"] = str(len(body_bytes))
                conn.request(method, path, body=body_bytes, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                data = json.loads(raw.decode("utf-8"))

                if resp.status >= 400:
                    error = data.get("error", {})
                    msg = error.get("message", str(data))
                    error_type = error.get("type", "ServerError")
                    tb = error.get("traceback", "")
                    full_msg = f"{error_type}: {msg}"
                    if tb:
                        full_msg += f"\n\nServer traceback:\n{tb}"
                    raise ContainerEnvironmentError(full_msg)

                return data
            except (ConnectionError, OSError, http.client.HTTPException):
                if attempt == 0:
                    # Reconnect on first failure
                    self._conn = None
                    continue
                raise

        raise ContainerEnvironmentError("Failed to connect after retry")

    # ------------------------------------------------------------------
    # Environment protocol
    # ------------------------------------------------------------------

    @property
    def spec(self) -> EnvironmentSpec:
        if self._spec is None:
            data = self._request("GET", "/spec")
            self._spec = deserialize_env_spec(data)
        return self._spec

    @property
    def reward_functions(self) -> tuple:
        # Rewards are computed server-side
        return ()

    @property
    def available_tools(self) -> tuple:
        if self._tools is None:
            data = self._request("GET", "/tools")
            self._tools = tuple(deserialize_tool_definition(d) for d in data)
        return self._tools

    @property
    def prompts(self) -> dict[str, str]:
        if self._prompts is None:
            self._prompts = self._request("GET", "/prompts")
        return self._prompts

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[OpaqueHidden], dict[str, Any]]:
        body: dict[str, Any] = {}
        if seed is not None:
            body["seed"] = seed
        if options is not None:
            body["options"] = options
        data = self._request("POST", "/reset", body)
        state = deserialize_state(data["state"])
        return state, data["info"]

    def step(
        self,
        state: State[OpaqueHidden],
        action: Action,
    ) -> StepResult[OpaqueHidden]:
        body = {
            "state": serialize_state(state),
            "action": serialize_action(action),
        }
        data = self._request("POST", "/step", body)
        return deserialize_step_result(data)

    def compute_rewards(
        self,
        state: State[OpaqueHidden],
        action: Action,
        next_state: State[OpaqueHidden],
    ) -> SignalBundle:
        body = {
            "state": serialize_state(state),
            "action": serialize_action(action),
            "next_state": serialize_state(next_state),
        }
        data = self._request("POST", "/compute_rewards", body)
        return deserialize_reward_bundle(data)

    def __len__(self) -> int:
        if self._length is None:
            data = self._request("GET", "/len")
            self._length = data["length"]
        return self._length

    def fork(self) -> tuple[str, int]:
        """Fork the remote server process, creating an independent copy.

        Returns:
            Tuple of (child_url, child_pid).
        """
        data = self._request("POST", "/fork", {})
        return data["url"], data["pid"]

    def close(self) -> None:
        """Close the HTTP connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
