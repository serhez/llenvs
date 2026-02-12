"""HTTP server wrapping any Environment instance.

Exposes the Environment protocol over JSON endpoints. Uses stdlib
``http.server`` — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from llenvs.container.serialization import (
    deserialize_action,
    deserialize_state_typed,
    serialize_env_spec,
    serialize_reward_bundle,
    serialize_state,
    serialize_step_result,
    serialize_tool_definition,
)

logger = logging.getLogger(__name__)


class EnvironmentHandler(BaseHTTPRequestHandler):
    """HTTP request handler for environment protocol methods."""

    # Assigned by EnvironmentServer
    environment: Any = None
    hidden_type: type | None = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format, *args)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        routes = {
            "/health": self._handle_health,
            "/spec": self._handle_spec,
            "/len": self._handle_len,
            "/tools": self._handle_tools,
            "/prompts": self._handle_prompts,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._send_error(404, "Not found", f"Unknown path: {self.path}")
            return
        try:
            handler()
        except Exception as e:
            self._send_exception(e)

    def do_POST(self) -> None:
        routes = {
            "/reset": self._handle_reset,
            "/step": self._handle_step,
            "/compute_rewards": self._handle_compute_rewards,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._send_error(404, "Not found", f"Unknown path: {self.path}")
            return
        try:
            body = self._read_body()
            handler(body)
        except Exception as e:
            self._send_exception(e)

    # ------------------------------------------------------------------
    # GET handlers
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        self._send_json({"status": "ok"})

    def _handle_spec(self) -> None:
        spec = self.environment.spec
        self._send_json(serialize_env_spec(spec))

    def _handle_len(self) -> None:
        length = len(self.environment)
        self._send_json({"length": length})

    def _handle_tools(self) -> None:
        tools = self.environment.available_tools
        self._send_json([serialize_tool_definition(t) for t in tools])

    def _handle_prompts(self) -> None:
        prompts = self.environment.prompts
        self._send_json(prompts)

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------

    def _handle_reset(self, body: dict[str, Any]) -> None:
        seed = body.get("seed")
        options = body.get("options")

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options

        state, info = self.environment.reset(**kwargs)

        # Capture hidden type from first reset
        cls = type(self)
        if cls.hidden_type is None:
            cls.hidden_type = type(state.hidden)

        self._send_json({
            "state": serialize_state(state),
            "info": info,
        })

    def _handle_step(self, body: dict[str, Any]) -> None:
        cls = type(self)
        if cls.hidden_type is None:
            raise RuntimeError("Must call /reset before /step")

        state = deserialize_state_typed(body["state"], cls.hidden_type)
        action = deserialize_action(body["action"])
        result = self.environment.step(state, action)
        self._send_json(serialize_step_result(result))

    def _handle_compute_rewards(self, body: dict[str, Any]) -> None:
        cls = type(self)
        if cls.hidden_type is None:
            raise RuntimeError("Must call /reset before /compute_rewards")

        state = deserialize_state_typed(body["state"], cls.hidden_type)
        action = deserialize_action(body["action"])
        next_state = deserialize_state_typed(body["next_state"], cls.hidden_type)
        rewards = self.environment.compute_rewards(state, action, next_state)
        self._send_json(serialize_reward_bundle(rewards))

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _read_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, error_type: str, message: str) -> None:
        self._send_json(
            {"error": {"type": error_type, "message": message}},
            status=status,
        )

    def _send_exception(self, exc: Exception) -> None:
        tb = traceback.format_exc()
        logger.error("Handler error: %s\n%s", exc, tb)
        self._send_json(
            {
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb,
                }
            },
            status=500,
        )


class EnvironmentServer:
    """HTTP server wrapping an Environment for container use.

    Args:
        environment: The environment instance to serve.
        host: Host to bind to.
        port: Port to bind to.
    """

    def __init__(
        self,
        environment: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self._environment = environment
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None

    def start(self) -> None:
        """Start the server (blocking)."""
        # Create a handler class bound to this environment
        handler_class = type(
            "BoundHandler",
            (EnvironmentHandler,),
            {"environment": self._environment, "hidden_type": None},
        )
        self._server = HTTPServer((self._host, self._port), handler_class)
        logger.info("Environment server listening on %s:%d", self._host, self._port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.shutdown()

    @property
    def address(self) -> tuple[str, int]:
        return (self._host, self._port)


def run_server_from_config(config_json: str, port: int = 8080) -> None:
    """Create an environment from JSON config and run the server.

    Args:
        config_json: JSON string of EnvironmentConfig fields.
        port: Port to listen on.
    """
    import json as _json

    from llenvs.core.config import EnvironmentConfig, EnvironmentFactory

    data = _json.loads(config_json)
    config = EnvironmentConfig(
        name=data["name"],
        adapter=data.get("adapter", "reasoning_gym"),
        size=data.get("size"),
        seed=data.get("seed"),
        answer_extractor=data.get("answer_extractor", "tag_based"),
        answer_extractor_config=data.get("answer_extractor_config", {}),
        answer_extractors=data.get("answer_extractors"),
        pre_cleaners=data.get("pre_cleaners"),
        post_cleaners=data.get("post_cleaners"),
        prompt_template=data.get("prompt_template"),
        system_prompt=data.get("system_prompt"),
        prompts=data.get("prompts"),
        params=data.get("params", {}),
    )
    env = EnvironmentFactory.create(config)
    server = EnvironmentServer(env, port=port)
    server.start()
