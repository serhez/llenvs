"""Owned loopback-only native vLLM-SR d3 worker; no managed frontend."""

import atexit
import os
import signal
import socket
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from llenvs.inference.backends.api import _AsyncRunner, _notify, _run_concurrent
from llenvs.inference.decisions.protocol import (
    DecisionBackend,
    DecisionCapabilities,
    DecisionResult,
    normalize_probabilities,
)
from llenvs.inference.protocol import BackendProcessExitedError
from llenvs.inference.records import (
    credential_free,
    native_runtime_identity,
    normalize_usage,
    sanitized_argv,
)

D3_REVISION = "156a6d9330277b7e1a6571a1d881b516e7e1ea15"
RUNTIME_REVISION = "72ce0e55f47a5f5b632719a28232308c57c78bc5"


class D3Backend(DecisionBackend):
    def __init__(
        self,
        *,
        executable="vllm-srun",
        model="vllm-sr/d3",
        revision=D3_REVISION,
        device="cuda:0",
        port=None,
        startup_timeout=600,
        shutdown_timeout=20,
        max_concurrency=64,
        log_path,
        engine_args=(),
    ):
        import httpx

        self._model, self._revision = model, revision
        self._device = device
        self._max_concurrency = max_concurrency
        self._shutdown_timeout = shutdown_timeout
        self._proc = None
        self._closed = False
        runtime = native_runtime_identity(str(executable), "vllm-srun")
        if ((runtime.get("source") or {}).get("vcs_info") or {}).get(
            "commit_id"
        ) != RUNTIME_REVISION:
            raise ValueError("d3 requires the pinned native vLLM-SR source revision")
        self._runtime_artifact = runtime
        self._runner = _AsyncRunner()
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("d3 requires an immutable 40-hex model revision")
        if any(
            str(arg).split("=")[0] in {"--host", "--port", "--revision", "--model", "--device"}
            for arg in engine_args
        ):
            raise ValueError("engine_args cannot replace owned launch controls")
        if port is None:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        self._base_url = f"http://127.0.0.1:{port}"
        argv = [
            str(executable),
            "serve",
            model,
            "--revision",
            revision,
            "--device",
            device,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *engine_args,
        ]
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("ab")
        self._client = httpx.Client(base_url=self._base_url, timeout=30, trust_env=False)
        self._async_client = httpx.AsyncClient(
            base_url=self._base_url, timeout=600, trust_env=False
        )
        self._provenance = {
            "argv": sanitized_argv(argv),
            "selected_device": device,
            "log_path": str(log_path.resolve()),
            "host": "127.0.0.1",
            "port": port,
            "container_digest": None,
            "started_at": datetime.now(UTC).isoformat(),
        }
        try:
            self._proc = subprocess.Popen(
                argv, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True
            )
            atexit.register(self.close)
            self._provenance["pid"] = self._proc.pid
            deadline = time.monotonic() + startup_timeout
            startup_health = None

            def startup_failure(error_type, message):
                reason = (startup_health or {}).get("body", {}).get("reason")
                detail = f": {reason}" if reason else ""
                error = error_type(f"{message}{detail}; see {log_path}")
                error.startup_health = deepcopy(startup_health)
                raise error

            while True:
                self._check_process()
                try:
                    health = self._client.get("/health")
                    try:
                        health_data = health.json()
                    except ValueError:
                        if health.status_code == 200:
                            raise
                        health_data = None
                    if isinstance(health_data, dict) and (
                        "api_version" in health_data or health.status_code == 200
                    ):
                        version = health_data.get("api_version")
                        if not isinstance(version, str) or version.split(".")[0] != "3":
                            raise ValueError("d3 runtime must implement API major version 3")
                        startup_health = credential_free(
                            {"status_code": health.status_code, "body": health_data}
                        )
                        if health_data.get("status") == "failed":
                            startup_failure(RuntimeError, "d3 worker failed during startup")
                    if health.status_code == 200:
                        if not isinstance(health_data, dict):
                            raise ValueError("Invalid d3 runtime health response")
                        response = self._client.get("/v1/models")
                        response.raise_for_status()
                        self._runtime = self._validate_models(response.json())
                        break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                if time.monotonic() >= deadline:
                    startup_failure(TimeoutError, "d3 startup timed out")
                time.sleep(0.2)
        except BaseException:
            self.close()
            raise

    def _validate_models(self, response):
        if response.get("api_version", "").split(".")[0] != "3":
            raise ValueError("Unsupported d3 runtime API")
        matches = [
            card
            for card in response["data"]
            if card["id"] == self._model or card.get("repo") == self._model
        ]
        if len(matches) != 1:
            raise ValueError("Worker did not report the requested d3 model")
        card = matches[0]
        if card.get("revision") != self._revision or not card.get("ready"):
            raise ValueError("d3 worker revision/readiness mismatch")
        if "choice" not in card.get("question_types", []):
            raise ValueError("d3 worker does not support choice questions")
        if card.get("family") != "decision3" or card.get("numerics") not in {
            "exact",
            "approximate",
        }:
            raise ValueError("d3 worker reported an incompatible family or numerical contract")
        limits = card.get("limits") or {}
        if (
            limits.get("min_options") != 2
            or type(limits.get("max_options")) is not int
            or not 2 <= limits["max_options"] <= 255
        ):
            raise ValueError("d3 worker did not report valid choice limits")
        engine = card.get("engine_args") or {}
        if (
            not {
                "max_num_seqs",
                "max_num_batched_tokens",
                "batch_window_ms",
                "dtype",
                "enable_shared_context",
            }
            <= engine.keys()
        ):
            raise ValueError("d3 worker did not report effective engine arguments")
        if (card.get("golden") or {}).get("status") not in {"matched", "unverified"}:
            raise ValueError("d3 worker reported a failed or incomplete native golden check")
        if (
            not card.get("model_sha256")
            or not card.get("manifest_sha256")
            or not card.get("dtype")
            or not card.get("numerics")
        ):
            raise ValueError("d3 worker did not report verified model/numerical identity")
        self._served_model = card["id"]
        self._runtime_card = deepcopy(card)
        return deepcopy(response)

    def _check_process(self):
        if self._proc is None or self._proc.poll() is not None:
            raise BackendProcessExitedError("Owned d3 worker exited")

    @property
    def capabilities(self):
        return DecisionCapabilities(
            version_pinned=True,
            deterministic=False,
            batch_invariant=False,
            max_options=getattr(self, "_runtime_card", {})
            .get("limits", {})
            .get("max_options", 255),
        )

    @property
    def model_name(self):
        return self._model

    def descriptor(self):
        runtime = deepcopy(self._runtime_card)
        runtime.pop("device", None)
        artifact = deepcopy(self._runtime_artifact)
        hardware = artifact.get("hardware", {})
        if self._device.startswith(("cuda:", "rocm:")) and hardware.get("devices"):
            hardware["devices"] = [hardware["devices"][int(self._device.split(":", 1)[1])]]
        return {
            "identity": {
                "backend": "native_d3",
                "protocol_version": 3,
                "service": "owned_loopback",
                "model": self._model,
                "revision": self._revision,
                "runtime_source_revision": RUNTIME_REVISION,
                "runtime_artifact": credential_free(artifact),
                "runtime": {
                    "api_version": self._runtime["api_version"],
                    "data": [credential_free(runtime)],
                },
                "container_digest": None,
                "max_concurrency": self._max_concurrency,
            },
            "provenance": deepcopy(self._provenance),
        }

    def _request_kwargs(self, request):
        if len(request.options) > self.capabilities.max_options:
            raise ValueError("Decision exceeds the worker's option limit")
        return {
            "model": getattr(self, "_served_model", self._model),
            "state": deepcopy(request.context),
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": request.question,
                    "criteria": {option.value: option.description for option in request.options},
                }
            },
            "options": {"return_meta": True},
        }

    def reparse_receipt(self, receipt):
        request, response = receipt["request"], receipt["response"]
        if hasattr(self, "_runtime_card"):
            meta = response.get("meta") or {}
            card = self._runtime_card
            if response.get("model") != self._served_model or any(
                meta.get(name) != expected
                for name, expected in {
                    "model_id": card.get("repo") or self._served_model,
                    "revision": self._revision,
                    "model_sha256": card["model_sha256"],
                    "numerics": card["numerics"],
                }.items()
            ):
                raise ValueError("d3 response identity differs from the verified worker")
        if set(response.get("answers", {})) != {"decision"}:
            raise ValueError("d3 must return exactly the named answer")
        answer = response["answers"]["decision"]
        if answer.get("error") or answer.get("type") != "choice":
            raise ValueError(f"d3 failed the question: {answer}")
        probabilities, normalization = normalize_probabilities(
            request, answer["probabilities"].items()
        )
        return DecisionResult(
            probabilities,
            tuple(option.value for option in request.options),
            response["model"],
            deepcopy(response),
            **normalize_usage(response.get("usage")),
            metadata={
                **receipt.get("metadata", {}),
                **normalization,
                "runtime_meta": deepcopy(response.get("meta")),
                "raw_usage": deepcopy(response.get("usage")),
            },
        )

    async def _decide(self, index, request, observer):
        self._check_process()
        await _notify(observer, "started", index, 0)
        started = time.monotonic()
        try:
            response = await self._async_client.post(
                "/v1/systemone", json=self._request_kwargs(request)
            )
            response.raise_for_status()
            receipt = {
                "request": request,
                "response": response.json(),
                "metadata": {
                    "elapsed_seconds": time.monotonic() - started,
                    "endpoint": "/v1/systemone",
                    "server_timing": response.headers.get("server-timing"),
                },
            }
            await _notify(observer, "received", index, receipt)
            result = self.reparse_receipt(receipt)
            await _notify(observer, "completed", index, result)
            return result
        except Exception as exc:
            await _notify(observer, "failed", index, exc)
            raise

    def decide_batch(self, requests, *, observer=None):
        for request in requests:
            self._request_kwargs(request)
        return _run_concurrent(
            lambda item: self._decide(item[0], item[1], observer),
            list(enumerate(requests)),
            self._max_concurrency,
            return_partial=True,
            runner=self._runner,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    os.killpg(self._proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    self._proc.wait(timeout=self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self._proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self._proc.wait(timeout=self._shutdown_timeout)
        finally:
            self._client.close()
            self._runner.close(self._async_client.aclose())
            self._log.close()
