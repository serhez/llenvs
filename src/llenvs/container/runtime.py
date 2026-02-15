"""Container runtimes for managing environment server processes.

Two implementations:

- ``DockerRuntime``: Manages a Docker container via the ``docker`` CLI.
- ``ProcessRuntime``: Runs the server as a local subprocess.
"""

from __future__ import annotations

import atexit
import http.client
import json
import logging
import socket
import subprocess
import sys
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ContainerRuntime(Protocol):
    """Protocol for container/process runtimes."""

    def start(self) -> str:
        """Start the environment server. Returns base URL."""
        ...

    def stop(self) -> None:
        """Stop the environment server and clean up."""
        ...

    def is_running(self) -> bool: ...

    def logs(self) -> str: ...


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(host: str, port: int, timeout: float) -> None:
    """Poll GET /health until the server responds or timeout."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return
        except Exception as e:
            last_error = e
        time.sleep(0.1)
    msg = f"Server on {host}:{port} did not become healthy within {timeout}s"
    if last_error:
        msg += f" (last error: {last_error})"
    raise TimeoutError(msg)


class DockerRuntime:
    """Manages a Docker container running an environment server.

    Args:
        image: Docker image name.
        config_json: JSON string of EnvironmentConfig fields.
        port: Host port to bind (None = auto-select).
        env_vars: Environment variables for the container.
        volumes: Volume mounts (host_path -> container_path).
        timeout: Max seconds to wait for server startup.
        docker_command: Path to docker CLI.
    """

    def __init__(
        self,
        image: str,
        config_json: str,
        *,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        timeout: float = 60.0,
        docker_command: str = "docker",
    ) -> None:
        self._image = image
        self._config_json = config_json
        self._port = port
        self._env_vars = env_vars or {}
        self._volumes = volumes or {}
        self._timeout = timeout
        self._docker = docker_command
        self._container_id: str | None = None
        self._host_port: int | None = None

    def start(self) -> str:
        self._host_port = self._port or _find_free_port()

        cmd = [
            self._docker,
            "run",
            "-d",
            "--rm",
            "-p",
            f"{self._host_port}:8080",
        ]
        for key, value in self._env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        for host_path, container_path in self._volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
        cmd.extend(
            [
                self._image,
                "python",
                "-m",
                "llenvs.container",
                "--config",
                self._config_json,
                "--port",
                "8080",
            ]
        )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Docker run failed (exit {result.returncode}): {result.stderr}")
        self._container_id = result.stdout.strip()
        logger.info("Started container %s on port %d", self._container_id[:12], self._host_port)

        # Register cleanup
        atexit.register(self.stop)

        _wait_for_health("127.0.0.1", self._host_port, self._timeout)
        return f"http://127.0.0.1:{self._host_port}"

    def stop(self) -> None:
        if self._container_id is None:
            return
        cid = self._container_id
        self._container_id = None
        try:
            subprocess.run(
                [self._docker, "stop", cid],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as e:
            logger.warning("Failed to stop container %s: %s", cid[:12], e)

    def is_running(self) -> bool:
        if self._container_id is None:
            return False
        try:
            result = subprocess.run(
                [self._docker, "inspect", "-f", "{{.State.Running}}", self._container_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    def logs(self) -> str:
        if self._container_id is None:
            return ""
        try:
            result = subprocess.run(
                [self._docker, "logs", self._container_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {e}"


class ProcessRuntime:
    """Runs the environment server as a local subprocess.

    Lighter than Docker — uses the same Python environment.

    Args:
        config_json: JSON string of EnvironmentConfig fields.
        port: Host port to bind (None = auto-select).
        timeout: Max seconds to wait for server startup.
        python: Path to the Python interpreter.
    """

    def __init__(
        self,
        config_json: str,
        *,
        port: int | None = None,
        timeout: float = 30.0,
        python: str | None = None,
    ) -> None:
        self._config_json = config_json
        self._port = port
        self._timeout = timeout
        self._python = python or sys.executable
        self._process: subprocess.Popen | None = None
        self._host_port: int | None = None

    def start(self) -> str:
        self._host_port = self._port or _find_free_port()

        cmd = [
            self._python,
            "-m",
            "llenvs.container",
            "--config",
            self._config_json,
            "--port",
            str(self._host_port),
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Started process PID %d on port %d", self._process.pid, self._host_port)

        atexit.register(self.stop)

        try:
            _wait_for_health("127.0.0.1", self._host_port, self._timeout)
        except TimeoutError:
            # Collect process output for debugging
            stderr_output = ""
            if self._process.poll() is not None:
                _, stderr_bytes = self._process.communicate(timeout=2)
                stderr_output = stderr_bytes.decode("utf-8", errors="replace")
            self.stop()
            raise TimeoutError(
                f"Process server did not start within {self._timeout}s. Stderr: {stderr_output}"
            )
        return f"http://127.0.0.1:{self._host_port}"

    def stop(self) -> None:
        if self._process is None:
            return
        proc = self._process
        self._process = None
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        except Exception as e:
            logger.warning("Failed to stop process %d: %s", proc.pid, e)

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def logs(self) -> str:
        if self._process is None:
            return ""
        # Can't read from pipes without blocking if process is still running
        if self._process.poll() is not None:
            stdout, stderr = self._process.communicate(timeout=2)
            return stdout.decode("utf-8", errors="replace") + stderr.decode(
                "utf-8", errors="replace"
            )
        return "(process still running — logs not available)"
