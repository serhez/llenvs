"""Tests for container runtimes (DockerRuntime, ProcessRuntime)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llenvs.container.runtime import (
    DockerRuntime,
    ProcessRuntime,
    _find_free_port,
    _wait_for_health,
)


# ---------------------------------------------------------------------------
# _find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_int(self):
        port = _find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_unique_ports(self):
        ports = {_find_free_port() for _ in range(5)}
        # Should get at least 2 unique ports (OS may reuse quickly)
        assert len(ports) >= 2


# ---------------------------------------------------------------------------
# _wait_for_health
# ---------------------------------------------------------------------------


class TestWaitForHealth:
    def test_success(self):
        """Health check succeeds with a real server."""

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                body = b'{"status":"ok"}'
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_health("127.0.0.1", port, timeout=5.0)
        finally:
            server.shutdown()

    def test_timeout(self):
        """Times out when no server is listening."""
        port = _find_free_port()
        with pytest.raises(TimeoutError, match="did not become healthy"):
            _wait_for_health("127.0.0.1", port, timeout=0.3)


# ---------------------------------------------------------------------------
# DockerRuntime (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDockerRuntime:
    def test_start_builds_correct_command(self):
        rt = DockerRuntime(
            image="llenvs-test:latest",
            config_json='{"name":"sudoku"}',
            port=9999,
            env_vars={"KEY": "val"},
            volumes={"/host/data": "/data"},
            docker_command="docker",
        )

        with (
            patch("llenvs.container.runtime.subprocess.run") as mock_run,
            patch("llenvs.container.runtime._wait_for_health"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
            url = rt.start()

        assert url == "http://127.0.0.1:9999"
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "run" in call_args
        assert "-d" in call_args
        assert "--rm" in call_args
        assert "-p" in call_args
        assert "9999:8080" in call_args
        assert "-e" in call_args
        assert "KEY=val" in call_args
        assert "-v" in call_args
        assert "/host/data:/data" in call_args
        assert "llenvs-test:latest" in call_args

    def test_start_auto_port(self):
        rt = DockerRuntime(
            image="test:latest",
            config_json='{"name":"test"}',
        )
        with (
            patch("llenvs.container.runtime.subprocess.run") as mock_run,
            patch("llenvs.container.runtime._wait_for_health"),
            patch("llenvs.container.runtime._find_free_port", return_value=12345),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="container123\n", stderr="")
            url = rt.start()

        assert "12345" in url

    def test_start_failure(self):
        rt = DockerRuntime(image="bad:latest", config_json="{}")
        with patch("llenvs.container.runtime.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="image not found")
            with pytest.raises(RuntimeError, match="Docker run failed"):
                rt.start()

    def test_stop(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        rt._container_id = "abc123"
        with patch("llenvs.container.runtime.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rt.stop()
        assert rt._container_id is None
        mock_run.assert_called_once()

    def test_stop_idempotent(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        rt.stop()  # No container_id, should not raise

    def test_is_running_true(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        rt._container_id = "abc123"
        with patch("llenvs.container.runtime.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="true\n", stderr="")
            assert rt.is_running() is True

    def test_is_running_false(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        rt._container_id = "abc123"
        with patch("llenvs.container.runtime.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="false\n", stderr="")
            assert rt.is_running() is False

    def test_is_running_no_container(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        assert rt.is_running() is False

    def test_logs(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        rt._container_id = "abc123"
        with patch("llenvs.container.runtime.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="server started\n", stderr="warn\n"
            )
            logs = rt.logs()
        assert "server started" in logs

    def test_logs_no_container(self):
        rt = DockerRuntime(image="test:latest", config_json="{}")
        assert rt.logs() == ""


# ---------------------------------------------------------------------------
# ProcessRuntime (mocked subprocess)
# ---------------------------------------------------------------------------


class TestProcessRuntime:
    def test_start_builds_correct_command(self):
        rt = ProcessRuntime(
            config_json='{"name":"sudoku"}',
            port=8888,
            python="/usr/bin/python3",
        )

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with (
            patch(
                "llenvs.container.runtime.subprocess.Popen", return_value=mock_proc
            ) as mock_popen,
            patch("llenvs.container.runtime._wait_for_health"),
        ):
            url = rt.start()

        assert url == "http://127.0.0.1:8888"
        call_args = mock_popen.call_args[0][0]
        assert "/usr/bin/python3" in call_args
        assert "-m" in call_args
        assert "llenvs.container" in call_args
        assert "--config" in call_args
        assert '{"name":"sudoku"}' in call_args
        assert "--port" in call_args
        assert "8888" in call_args

    def test_start_auto_port(self):
        rt = ProcessRuntime(config_json='{"name":"test"}')

        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.poll.return_value = None

        with (
            patch("llenvs.container.runtime.subprocess.Popen", return_value=mock_proc),
            patch("llenvs.container.runtime._wait_for_health"),
            patch("llenvs.container.runtime._find_free_port", return_value=54321),
        ):
            url = rt.start()

        assert "54321" in url

    def test_stop(self):
        rt = ProcessRuntime(config_json="{}")
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        rt._process = mock_proc

        rt.stop()
        mock_proc.terminate.assert_called_once()
        assert rt._process is None

    def test_stop_kills_on_timeout(self):
        rt = ProcessRuntime(config_json="{}")
        mock_proc = MagicMock()
        mock_proc.terminate.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
        rt._process = mock_proc

        rt.stop()
        mock_proc.kill.assert_called_once()

    def test_stop_idempotent(self):
        rt = ProcessRuntime(config_json="{}")
        rt.stop()  # No process, should not raise

    def test_is_running_true(self):
        rt = ProcessRuntime(config_json="{}")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        rt._process = mock_proc
        assert rt.is_running() is True

    def test_is_running_false_exited(self):
        rt = ProcessRuntime(config_json="{}")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Exited
        rt._process = mock_proc
        assert rt.is_running() is False

    def test_is_running_no_process(self):
        rt = ProcessRuntime(config_json="{}")
        assert rt.is_running() is False

    def test_logs_after_exit(self):
        rt = ProcessRuntime(config_json="{}")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.communicate.return_value = (b"output\n", b"error\n")
        rt._process = mock_proc
        logs = rt.logs()
        assert "output" in logs
        assert "error" in logs

    def test_logs_while_running(self):
        rt = ProcessRuntime(config_json="{}")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        rt._process = mock_proc
        logs = rt.logs()
        assert "still running" in logs

    def test_logs_no_process(self):
        rt = ProcessRuntime(config_json="{}")
        assert rt.logs() == ""

    def test_start_timeout_collects_stderr(self):
        rt = ProcessRuntime(config_json="{}", port=1, timeout=0.3)

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.poll.return_value = 1  # Already exited
        mock_proc.communicate.return_value = (b"", b"import error\n")

        with (
            patch("llenvs.container.runtime.subprocess.Popen", return_value=mock_proc),
            patch("llenvs.container.runtime._wait_for_health", side_effect=TimeoutError("timeout")),
        ):
            with pytest.raises(TimeoutError, match="import error"):
                rt.start()
