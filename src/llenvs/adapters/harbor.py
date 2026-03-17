"""Harbor adapter — wraps Harbor containerized evaluation environments.

Harbor (Laude Institute) is a generic framework for containerized agent
evaluation. It manages Docker containers, task discovery via a JSON registry,
and verification (test scripts produce binary pass/fail rewards).

By wrapping Harbor (not individual benchmarks), this adapter provides access
to Terminal-Bench, aider-polyglot, swe-bench, and other datasets through a
single interface.

Dual-mode design:
- **Text mode** (``HarborEnvironment``): Agent sends shell commands as text,
  receives stdout/stderr. Submit via keyword in action text.
- **Tool mode** (``HarborToolEnvironment``): Agent uses structured tool calls
  (``execute_command``, ``read_file``, ``write_file``, ``submit``).

Reference: https://github.com/laude-institute/harbor
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llenvs.core.async_utils import run_async
from llenvs.core.environment import (
    EnvironmentSpec,
    StepResult,
    _StateContinuityTracker,
)
from llenvs.core.reward import (
    RewardFunction,
    RewardType,
    Signal,
    SignalBundle,
)
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)

logger = logging.getLogger(__name__)


def _run_with_timeout(coro: Any, timeout: int | None, label: str) -> Any:
    if timeout is None:
        return run_async(coro)
    try:
        return run_async(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout}s") from exc

# ── Tool definitions (for tool mode) ────────────────────────────

HARBOR_EXECUTE_COMMAND_TOOL = ToolDefinition(
    name="execute_command",
    description="Run a shell command in the container.",
    parameters=(
        ToolParameter(
            name="command",
            type=ToolParameterType.STRING,
            description="Shell command to execute",
        ),
        ToolParameter(
            name="cwd",
            type=ToolParameterType.STRING,
            description="Working directory (default: /)",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type=ToolParameterType.INTEGER,
            description="Timeout in seconds (default: 120)",
            required=False,
        ),
    ),
)

HARBOR_READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read file contents from the container.",
    parameters=(
        ToolParameter(
            name="path",
            type=ToolParameterType.STRING,
            description="Absolute file path to read",
        ),
    ),
)

HARBOR_WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write file contents to the container.",
    parameters=(
        ToolParameter(
            name="path",
            type=ToolParameterType.STRING,
            description="Absolute file path to write",
        ),
        ToolParameter(
            name="content",
            type=ToolParameterType.STRING,
            description="Content to write to the file",
        ),
    ),
)

HARBOR_SUBMIT_TOOL = ToolDefinition(
    name="submit",
    description="Signal task completion and trigger verification.",
    parameters=(),
    is_terminal=True,
)

HARBOR_TOOLS: tuple[ToolDefinition, ...] = (
    HARBOR_EXECUTE_COMMAND_TOOL,
    HARBOR_READ_FILE_TOOL,
    HARBOR_WRITE_FILE_TOOL,
    HARBOR_SUBMIT_TOOL,
)


# ── Hidden state ────────────────────────────────────────────────


@dataclass(frozen=True)
class HarborHidden:
    """Hidden state for Harbor environments.

    Attributes:
        task_index: Index into the task list.
        task_name: The Harbor task identifier.
        instruction: Task instruction text.
        episode_step: Current step in the episode.
        last_action: Text of the last action taken.
        trajectory: Command history (frozen tuple).
    """

    task_index: int
    task_name: str
    instruction: str
    episode_step: int
    last_action: str | None = None
    trajectory: tuple[str, ...] = ()


# ── Reward function ─────────────────────────────────────────────


@dataclass
class HarborReward:
    """Native reward function reading Harbor's verifier result.

    Non-terminal steps return STEP signal with None reward.
    Terminal steps return OUTCOME signal with the verifier reward
    (read from ``next_state.metadata.info["reward"]``).
    """

    _name: str = "harbor"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[Any],
        action: Action,
        next_state: State[Any],
    ) -> Signal:
        is_terminal = next_state.metadata.is_terminal

        if not is_terminal:
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=None,
                metadata={"is_terminal": False},
            )

        reward = next_state.metadata.info.get("reward", 0.0)
        return Signal(
            name=self.name,
            reward_type=RewardType.OUTCOME,
            reward=float(reward),
            metadata={"is_terminal": True},
        )


@dataclass(frozen=True)
class _HarborAPI:
    registry_client_factory: Any
    task_client: Any
    task_class: Any
    task_paths_class: Any
    environment_factory: Any
    environment_type_enum: Any
    trial_paths_class: Any
    verifier_class: Any


@dataclass(frozen=True)
class _CLIResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass(frozen=True)
class _PodmanVolumeMount:
    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class _PodmanHealthcheck:
    test: str | tuple[str, ...] | None = None
    interval_sec: float = 1.0
    timeout_sec: float = 30.0
    retries: int = 30
    start_period_sec: float = 0.0


@dataclass(frozen=True)
class _PodmanServiceSpec:
    name: str
    image: str | None
    build_context: Path | None
    dockerfile: Path | None
    command: str | tuple[str, ...] | None
    entrypoint: str | tuple[str, ...] | None
    environment: tuple[tuple[str, str], ...]
    working_dir: str | None
    volumes: tuple[_PodmanVolumeMount, ...]
    depends_on: tuple[str, ...]
    healthcheck: _PodmanHealthcheck | None = None


# ── Helpers ─────────────────────────────────────────────────────


def _format_exec_result(result: Any) -> str:
    """Format an exec result as observation text.

    Shows stdout always, stderr with [stderr] prefix when non-empty.
    When both empty, shows exit code.
    """
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return_code = getattr(result, "return_code", 0)

    if not stdout and not stderr:
        return f"[exit code: {return_code}]"

    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr] {stderr}")
    return "\n".join(parts)


def _run_verifier(
    verifier_factory: Any,
    task: Any,
    harbor_env: Any,
) -> dict[str, float]:
    """Run the verifier and return rewards dict."""
    verifier = verifier_factory(task, harbor_env)
    result = run_async(verifier.verify())
    return result.rewards


def _normalize_container_name(name: str) -> str:
    normalized = name.lower().replace(".", "-")
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in normalized)


_COMPOSE_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _parse_compose_duration(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if text.isdigit():
        return float(text)

    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text):
        matched = True
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
        total += float(amount) * scale
    if matched:
        return total

    raise ValueError(f"Unsupported compose duration: {value!r}")


def _topological_service_order(services: dict[str, _PodmanServiceSpec]) -> list[str]:
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Cyclic compose dependency involving {name!r}")
        visiting.add(name)
        spec = services[name]
        for dep in spec.depends_on:
            if dep not in services:
                raise ValueError(f"Compose service {name!r} depends on unknown service {dep!r}")
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for service_name in services:
        visit(service_name)
    return order


def _compose_shell_command(
    value: str | tuple[str, ...] | None,
    *,
    entrypoint_present: bool = False,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return list(value)
    if entrypoint_present:
        return [value]
    return ["sh", "-lc", value]


class PodmanHPCEnvironment:
    """Local Harbor-compatible runtime using ``podman-hpc``.

    This is a Harbor-facing environment object with the methods the llenvs
    Harbor adapter and Harbor verifiers rely on. It supports Harbor's default
    single-container tasks and a constrained task-local ``docker-compose.yaml``
    subset centered on a ``main`` service plus sidecars.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        logger: logging.Logger | None = None,
        *,
        podman_command: str = "podman-hpc",
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.environment_dir = Path(environment_dir)
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.logger = logger or logging.getLogger(__name__)
        self._podman = podman_command
        self._container_name = _normalize_container_name(session_id)
        self._image_name = f"hb__{_normalize_container_name(environment_name)}"
        self._started = False
        self.is_mounted = False

        self._dockerfile_path = self.environment_dir / "Dockerfile"
        self._compose_path = self.environment_dir / "docker-compose.yaml"
        self._network_name = f"{self._container_name}-net"
        self._volume_root = Path(self.trial_paths.trial_dir) / "compose-volumes"
        self._compose_services: dict[str, _PodmanServiceSpec] = {}
        self._service_order: tuple[str, ...] = ()
        self._service_container_names: dict[str, str] = {}
        self._main_container_name = self._container_name
        self._validate_definition()

    def _validate_definition(self) -> None:
        if self._compose_path.exists():
            self._compose_services = self._parse_compose_definition()
            if "main" not in self._compose_services:
                raise ValueError("Compose environments must define a 'main' service")
            self._service_order = tuple(_topological_service_order(self._compose_services))
            self._service_container_names = {
                name: _normalize_container_name(f"{self.session_id}-{name}")
                for name in self._compose_services
            }
            self._main_container_name = self._service_container_names["main"]
            return

        docker_image = getattr(self.task_env_config, "docker_image", None)
        if not self._dockerfile_path.exists() and not docker_image:
            raise FileNotFoundError(
                f"{self._dockerfile_path} not found and task_env_config.docker_image is unset."
            )

    async def _run_podman_command(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> _CLIResult:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.environment_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        try:
            if timeout_sec is not None:
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            else:
                stdout_b, stderr_b = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                stdout_b, stderr_b = await process.communicate()
            raise RuntimeError(f"podman-hpc command timed out after {timeout_sec}s")

        result = _CLIResult(
            stdout=stdout_b.decode(errors="replace").strip() if stdout_b else "",
            stderr=stderr_b.decode(errors="replace").strip() if stderr_b else "",
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"podman-hpc command failed (exit {result.return_code}): "
                f"{' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    def _docker_image_source(self) -> str:
        image = getattr(self.task_env_config, "docker_image", None)
        if image is None:
            raise ValueError("docker_image is required for migrate-based startup")
        return image if "://" in image else f"docker://{image}"

    def _compose_image_source(self, image: str) -> str:
        return image if "://" in image else f"docker://{image}"

    def _runtime_env_vars(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "MAIN_IMAGE_NAME": self._image_name,
                "CONTEXT_DIR": str(self.environment_dir.resolve()),
                "TEST_DIR": "/tests",
                "HOST_VERIFIER_LOGS_PATH": str(Path(self.trial_paths.verifier_dir).resolve()),
                "HOST_AGENT_LOGS_PATH": str(Path(self.trial_paths.agent_dir).resolve()),
                "ENV_VERIFIER_LOGS_PATH": "/logs/verifier",
                "ENV_AGENT_LOGS_PATH": "/logs/agent",
                "CPUS": str(getattr(self.task_env_config, "cpus", 1)),
                "MEMORY": f"{getattr(self.task_env_config, 'memory_mb', 1024)}M",
            }
        )
        docker_image = getattr(self.task_env_config, "docker_image", None)
        if docker_image is not None:
            env["PREBUILT_IMAGE_NAME"] = str(docker_image)
        return env

    def _interpolate_compose_value(self, value: Any) -> Any:
        if isinstance(value, str):
            def repl(match: re.Match[str]) -> str:
                name = match.group(1)
                default = match.group(2)
                return self._runtime_env_vars().get(name, default or "")

            return _COMPOSE_VAR_PATTERN.sub(repl, value)
        if isinstance(value, list):
            return [self._interpolate_compose_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._interpolate_compose_value(item)
                for key, item in value.items()
            }
        return value

    def _normalize_environment(self, raw_env: Any) -> tuple[tuple[str, str], ...]:
        if raw_env is None:
            return ()
        if isinstance(raw_env, dict):
            return tuple(
                (str(key), "" if value is None else str(value))
                for key, value in raw_env.items()
            )
        if isinstance(raw_env, list):
            pairs: list[tuple[str, str]] = []
            for item in raw_env:
                if not isinstance(item, str):
                    raise NotImplementedError("Compose environment list entries must be strings")
                key, sep, value = item.partition("=")
                pairs.append((key, value if sep else ""))
            return tuple(pairs)
        raise NotImplementedError("Unsupported compose environment format")

    def _resolve_volume_source(self, source: str, *, named: bool) -> str:
        if named:
            host_path = self._volume_root / source
            host_path.mkdir(parents=True, exist_ok=True)
            return str(host_path)
        host_path = Path(source)
        if not host_path.is_absolute():
            host_path = (self.environment_dir / host_path).resolve()
        return str(host_path)

    def _parse_volume_mount(self, raw_volume: Any) -> _PodmanVolumeMount:
        if isinstance(raw_volume, str):
            parts = raw_volume.split(":")
            if len(parts) < 2:
                raise NotImplementedError("Compose volume entries must include source and target")
            source, target = parts[0], parts[1]
            mode = parts[2] if len(parts) > 2 else ""
            read_only = "ro" in mode.split(",")
            named = not source.startswith(("/", ".", "~"))
            return _PodmanVolumeMount(
                source=self._resolve_volume_source(source, named=named),
                target=target,
                read_only=read_only,
            )

        if isinstance(raw_volume, dict):
            volume_type = raw_volume.get("type", "volume")
            if volume_type not in {"bind", "volume"}:
                raise NotImplementedError(
                    f"Unsupported compose volume type: {volume_type!r}"
                )
            source = raw_volume.get("source")
            target = raw_volume.get("target")
            if not source or not target:
                raise NotImplementedError("Compose volume mappings require source and target")
            return _PodmanVolumeMount(
                source=self._resolve_volume_source(
                    str(source),
                    named=volume_type == "volume",
                ),
                target=str(target),
                read_only=bool(raw_volume.get("read_only", False)),
            )

        raise NotImplementedError("Unsupported compose volume format")

    def _parse_healthcheck(self, raw_healthcheck: Any) -> _PodmanHealthcheck | None:
        if raw_healthcheck in (None, False):
            return None
        if not isinstance(raw_healthcheck, dict):
            raise NotImplementedError("Unsupported compose healthcheck format")
        if raw_healthcheck.get("disable") is True:
            return None
        test = raw_healthcheck.get("test")
        if isinstance(test, list):
            normalized_test: str | tuple[str, ...] | None = tuple(str(part) for part in test)
        elif test is None or isinstance(test, str):
            normalized_test = test
        else:
            raise NotImplementedError("Unsupported compose healthcheck.test format")
        return _PodmanHealthcheck(
            test=normalized_test,
            interval_sec=_parse_compose_duration(raw_healthcheck.get("interval"), 1.0),
            timeout_sec=_parse_compose_duration(raw_healthcheck.get("timeout"), 30.0),
            retries=int(raw_healthcheck.get("retries", 30)),
            start_period_sec=_parse_compose_duration(raw_healthcheck.get("start_period"), 0.0),
        )

    def _parse_compose_definition(self) -> dict[str, _PodmanServiceSpec]:
        data = yaml.safe_load(self._compose_path.read_text()) or {}
        data = self._interpolate_compose_value(data)
        if not isinstance(data, dict):
            raise ValueError("docker-compose.yaml must define a mapping")

        top_level_networks = data.get("networks")
        if top_level_networks:
            raise NotImplementedError("Compose networks are not supported by podman-hpc runtime")

        top_level_volumes = data.get("volumes", {})
        if not isinstance(top_level_volumes, dict):
            raise ValueError("Top-level compose volumes must be a mapping")
        for name, cfg in top_level_volumes.items():
            if not cfg:
                continue
            if not isinstance(cfg, dict):
                raise NotImplementedError("Unsupported top-level compose volume configuration")
            if cfg.get("external"):
                raise NotImplementedError(
                    f"External compose volume {name!r} is not supported"
                )

        raw_services = data.get("services")
        if not isinstance(raw_services, dict) or not raw_services:
            raise ValueError("docker-compose.yaml must define at least one service")

        services: dict[str, _PodmanServiceSpec] = {}
        for name, raw_service in raw_services.items():
            if not isinstance(raw_service, dict):
                raise ValueError(f"Compose service {name!r} must be a mapping")
            unsupported_keys = {"ports", "networks", "secrets", "configs", "profiles", "devices"}
            present_unsupported = unsupported_keys.intersection(raw_service)
            if present_unsupported:
                raise NotImplementedError(
                    f"Unsupported compose fields for service {name!r}: "
                    + ", ".join(sorted(present_unsupported))
                )

            build_context: Path | None = None
            dockerfile: Path | None = None
            build = raw_service.get("build")
            if isinstance(build, str):
                build_context = (self.environment_dir / build).resolve()
            elif isinstance(build, dict):
                unsupported_build_keys = set(build).difference({"context", "dockerfile"})
                if unsupported_build_keys:
                    raise NotImplementedError(
                        f"Unsupported compose build fields for service {name!r}: "
                        + ", ".join(sorted(unsupported_build_keys))
                    )
                build_context = (self.environment_dir / build.get("context", ".")).resolve()
                dockerfile_name = build.get("dockerfile")
                if dockerfile_name is not None:
                    dockerfile = (build_context / str(dockerfile_name)).resolve()
            elif build is not None:
                raise NotImplementedError("Unsupported compose build format")

            image = raw_service.get("image")
            if build_context is None and image is None:
                raise ValueError(
                    f"Compose service {name!r} must define either image or build"
                )

            command = raw_service.get("command")
            if isinstance(command, list):
                normalized_command: str | tuple[str, ...] | None = tuple(
                    str(part) for part in command
                )
            else:
                normalized_command = None if command is None else str(command)

            entrypoint = raw_service.get("entrypoint")
            if isinstance(entrypoint, list):
                normalized_entrypoint: str | tuple[str, ...] | None = tuple(
                    str(part) for part in entrypoint
                )
            else:
                normalized_entrypoint = None if entrypoint is None else str(entrypoint)

            depends_on_raw = raw_service.get("depends_on", ())
            if isinstance(depends_on_raw, dict):
                depends_on = tuple(str(dep_name) for dep_name in depends_on_raw)
            elif isinstance(depends_on_raw, list):
                depends_on = tuple(str(dep_name) for dep_name in depends_on_raw)
            elif depends_on_raw in (None, ()):
                depends_on = ()
            else:
                raise NotImplementedError("Unsupported compose depends_on format")

            services[name] = _PodmanServiceSpec(
                name=name,
                image=None if image is None else str(image),
                build_context=build_context,
                dockerfile=dockerfile,
                command=normalized_command,
                entrypoint=normalized_entrypoint,
                environment=self._normalize_environment(raw_service.get("environment")),
                working_dir=(
                    None
                    if raw_service.get("working_dir") is None
                    else str(raw_service.get("working_dir"))
                ),
                volumes=tuple(
                    self._parse_volume_mount(volume)
                    for volume in raw_service.get("volumes", ())
                ),
                depends_on=depends_on,
                healthcheck=self._parse_healthcheck(raw_service.get("healthcheck")),
            )
        return services

    def _service_image_name(self, service_name: str) -> str:
        return f"{self._image_name}__{_normalize_container_name(service_name)}"

    async def _prepare_service_image(
        self,
        service: _PodmanServiceSpec,
        *,
        force_build: bool,
    ) -> str:
        if service.build_context is not None:
            build_cmd = [
                self._podman,
                "build",
                "-t",
                self._service_image_name(service.name),
            ]
            if service.dockerfile is not None:
                build_cmd.extend(["-f", str(service.dockerfile)])
            build_cmd.append(str(service.build_context))
            await self._run_podman_command(build_cmd)
            return self._service_image_name(service.name)

        if service.image is None:
            raise ValueError(f"Compose service {service.name!r} has no runnable image")
        if not force_build:
            await self._run_podman_command(
                [self._podman, "migrate", self._compose_image_source(service.image)],
            )
        return service.image

    def _build_service_run_command(
        self,
        service: _PodmanServiceSpec,
        image_ref: str,
    ) -> list[str]:
        cmd = [
            self._podman,
            "run",
            "-d",
            "--name",
            self._service_container_names[service.name],
            "--network",
            self._network_name,
            "--network-alias",
            service.name,
            "--cpus",
            str(getattr(self.task_env_config, "cpus", 1)),
            "--memory",
            f"{getattr(self.task_env_config, 'memory_mb', 1024)}M",
        ]
        if service.working_dir:
            cmd.extend(["-w", service.working_dir])
        for key, value in service.environment:
            cmd.extend(["-e", f"{key}={value}"])
        for volume in service.volumes:
            suffix = ":ro" if volume.read_only else ""
            cmd.extend(["-v", f"{volume.source}:{volume.target}{suffix}"])
        if service.entrypoint is not None:
            entrypoint_value = (
                json.dumps(list(service.entrypoint))
                if isinstance(service.entrypoint, tuple)
                else service.entrypoint
            )
            cmd.extend(["--entrypoint", entrypoint_value])
        cmd.append(image_ref)
        cmd.extend(
            _compose_shell_command(
                service.command,
                entrypoint_present=service.entrypoint is not None,
            )
        )
        return cmd

    def _healthcheck_command(self, service_name: str) -> str | None:
        healthcheck = self._compose_services[service_name].healthcheck
        if healthcheck is None or healthcheck.test is None:
            return None
        if isinstance(healthcheck.test, str):
            return healthcheck.test
        if not healthcheck.test:
            return None
        head, *tail = healthcheck.test
        upper_head = head.upper()
        if upper_head == "NONE":
            return None
        if upper_head == "CMD":
            return shlex.join(tail)
        if upper_head == "CMD-SHELL":
            return " ".join(tail)
        return shlex.join(list(healthcheck.test))

    async def _wait_for_service_health(self, service_name: str) -> None:
        healthcheck = self._compose_services[service_name].healthcheck
        command = self._healthcheck_command(service_name)
        if healthcheck is None or command is None:
            return
        if healthcheck.start_period_sec > 0:
            await asyncio.sleep(healthcheck.start_period_sec)
        attempts = max(1, healthcheck.retries)
        for attempt in range(attempts):
            result = await self.exec_service(
                service_name,
                command,
                timeout_sec=max(1, int(healthcheck.timeout_sec)),
            )
            if result.return_code == 0:
                return
            if attempt < attempts - 1:
                await asyncio.sleep(healthcheck.interval_sec)
        raise RuntimeError(
            f"Compose service {service_name!r} failed healthcheck after {attempts} attempts"
        )

    async def exec_service(
        self,
        service_name: str,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> _CLIResult:
        if service_name not in self._service_container_names:
            raise ValueError(f"Unknown compose service: {service_name}")
        cmd = [self._podman, "exec"]
        if cwd:
            cmd.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._service_container_names[service_name], "bash", "-lc", command])
        return await self._run_podman_command(cmd, check=False, timeout_sec=timeout_sec)

    async def _bootstrap_runtime_dirs(self) -> None:
        if self._compose_services:
            await self.exec_service("main", "mkdir -p /logs/agent /logs/verifier")
            return
        await self._run_podman_command(
            [
                self._podman,
                "exec",
                self._main_container_name,
                "bash",
                "-lc",
                "mkdir -p /logs/agent /logs/verifier",
            ],
            check=False,
        )

    async def _start_compose(self, force_build: bool) -> None:
        network_cmd = [self._podman, "network", "create"]
        if getattr(self.task_env_config, "allow_internet", True) is False:
            network_cmd.append("--internal")
        network_cmd.append(self._network_name)
        await self._run_podman_command(network_cmd)

        image_refs: dict[str, str] = {}
        for service_name in self._service_order:
            image_refs[service_name] = await self._prepare_service_image(
                self._compose_services[service_name],
                force_build=force_build,
            )

        for service_name in self._service_order:
            await self._run_podman_command(
                self._build_service_run_command(
                    self._compose_services[service_name],
                    image_refs[service_name],
                )
            )
            if self._compose_services[service_name].healthcheck is not None:
                await self._wait_for_service_health(service_name)

        await self._bootstrap_runtime_dirs()
        self._started = True

    async def start(self, force_build: bool = False) -> None:
        if self._compose_services:
            await self._start_compose(force_build=force_build)
            return

        docker_image = getattr(self.task_env_config, "docker_image", None)
        if docker_image and not force_build:
            await self._run_podman_command(
                [self._podman, "migrate", self._docker_image_source()],
            )
            image_ref = docker_image
        else:
            await self._run_podman_command(
                [self._podman, "build", "-t", self._image_name, str(self.environment_dir)],
            )
            image_ref = self._image_name

        run_cmd = [
            self._podman,
            "run",
            "-d",
            "--name",
            self._container_name,
        ]
        if getattr(self.task_env_config, "allow_internet", True) is False:
            run_cmd.extend(["--network", "none"])
        run_cmd.extend([image_ref, "bash", "-lc", "while true; do sleep 3600; done"])
        await self._run_podman_command(run_cmd)
        await self._bootstrap_runtime_dirs()
        self._started = True

    async def stop(self, delete: bool = True) -> None:
        if not self._started:
            return
        try:
            if self._compose_services:
                verb = ["rm", "-f"] if delete else ["stop"]
                for service_name in reversed(self._service_order):
                    await self._run_podman_command(
                        [
                            self._podman,
                            *verb,
                            self._service_container_names[service_name],
                        ],
                        check=False,
                    )
                if delete:
                    await self._run_podman_command(
                        [self._podman, "network", "rm", self._network_name],
                        check=False,
                    )
            else:
                cmd = [self._podman, "rm", "-f", self._container_name]
                if not delete:
                    cmd = [self._podman, "stop", self._container_name]
                await self._run_podman_command(cmd, check=False)
        finally:
            self._started = False

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> _CLIResult:
        if not self._started:
            raise RuntimeError("podman-hpc environment has not been started")

        cmd = [self._podman, "exec"]
        if cwd:
            cmd.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._main_container_name, "bash", "-lc", command])
        return await self._run_podman_command(cmd, check=False, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", str(source_path), f"{self._main_container_name}:{target_path}"]
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", str(source_dir), f"{self._main_container_name}:{target_dir}"]
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", f"{self._main_container_name}:{source_path}", str(target_path)]
        )

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._run_podman_command(
            [self._podman, "cp", f"{self._main_container_name}:{source_dir}", str(target_dir)]
        )


# ── Text-mode environment ───────────────────────────────────────


class HarborEnvironment:
    """Text-based MDP wrapper for Harbor containerized environments.

    Agents send shell commands as ``Action(text="ls -la")`` and receive
    stdout/stderr as observation text. Termination occurs via a submit
    keyword in the action text or truncation at ``max_steps``.

    Example:
        >>> env = HarborEnvironment(tasks=tasks, harbor_env_factory=factory, ...)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> result = env.step(state, Action(text="ls"))
        >>> result = env.step(result.next_state, Action(text="SUBMIT"))
    """

    def __init__(
        self,
        tasks: tuple[Any, ...],
        harbor_env_factory: Any,
        verifier_factory: Any | None = None,
        *,
        dataset_name: str = "terminal-bench",
        max_steps: int = 30,
        submit_keyword: str = "SUBMIT",
        verify_on_truncation: bool = True,
        start_timeout: int | None = 120,
        exec_timeout: int = 120,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._tasks = tasks
        self._harbor_env_factory = harbor_env_factory
        self._verifier_factory = verifier_factory
        self._dataset_name = dataset_name
        self._max_steps = max_steps
        self._submit_keyword = submit_keyword
        self._verify_on_truncation = verify_on_truncation
        self._start_timeout = start_timeout
        self._exec_timeout = exec_timeout

        self._native_rewards: tuple[RewardFunction, ...] = (HarborReward(),)
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

        self._harbor_env: Any = None
        self._current_task: Any = None

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=f"harbor:{self._dataset_name}",
            adapter="harbor",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={"dataset": self._dataset_name},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HarborHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        # Stop previous container if running
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass

        task = self._tasks[task_index]
        self._current_task = task

        # Create and start container
        self._harbor_env = self._harbor_env_factory(task)
        _run_with_timeout(
            self._harbor_env.start(force_build=False),
            self._start_timeout,
            "Harbor container start",
        )

        instruction = getattr(task, "instruction", str(task))

        hidden = HarborHidden(
            task_index=task_index,
            task_name=getattr(task, "name", str(task_index)),
            instruction=instruction,
            episode_step=0,
        )

        observation = Observation(
            prompt=instruction,
            task=ObservationContent(text=instruction),
            state=ObservationContent(text=instruction),
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "task_name": hidden.task_name,
        }

    def step(
        self,
        state: State[HarborHidden],
        action: Action,
    ) -> StepResult[HarborHidden]:
        self._state_tracker.validate(state, "HarborEnvironment")

        next_step = state.hidden.episode_step + 1
        action_text = action.text or ""
        terminated = False
        truncated = False

        # Check for submit keyword
        if self._submit_keyword in action_text:
            terminated = True

        # Execute command in container (even for submit, to maintain trajectory)
        if not terminated:
            exec_result = run_async(
                self._harbor_env.exec(action_text, timeout_sec=self._exec_timeout)
            )
            obs_text = _format_exec_result(exec_result)
        else:
            obs_text = "Submitting for verification..."

        # Check truncation
        if not terminated and next_step >= self._max_steps:
            truncated = True

        # Run verifier at terminal
        reward_value: float | None = None
        if terminated or (truncated and self._verify_on_truncation):
            if self._verifier_factory is not None:
                try:
                    rewards = _run_verifier(
                        self._verifier_factory, self._current_task, self._harbor_env
                    )
                    reward_value = rewards.get("reward", 0.0)
                except Exception as e:
                    logger.warning(f"Verifier failed: {e}")
                    reward_value = 0.0

        # Build next hidden
        next_hidden = HarborHidden(
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            instruction=state.hidden.instruction,
            episode_step=next_step,
            last_action=action_text,
            trajectory=state.hidden.trajectory + (action_text,),
        )

        # Build messages
        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action_text},
            {"role": "user", "content": obs_text},
        )

        next_obs = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=obs_text),
        )

        info: dict[str, Any] = {
            **state.metadata.info,
            "episode_step": next_step,
        }
        if reward_value is not None:
            info["reward"] = reward_value

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info=info,
        )

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "episode_step": next_step,
                "observation": obs_text,
            },
        )

    def compute_rewards(
        self,
        state: State[HarborHidden],
        action: Action,
        next_state: State[HarborHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Stop the running container."""
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass
            self._harbor_env = None


# ── Tool-mode environment ───────────────────────────────────────


class HarborToolEnvironment(BaseToolEnvironment[HarborHidden]):
    """Tool-based MDP wrapper for Harbor containerized environments.

    Agents use structured tool calls (``execute_command``, ``read_file``,
    ``write_file``, ``submit``) instead of free-form text commands.
    Inherits tool validation, message building, and monitoring rewards
    from ``BaseToolEnvironment``.

    Example:
        >>> env = HarborToolEnvironment(tasks=tasks, harbor_env_factory=factory, ...)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> call = ToolCall(id="1", name="execute_command", arguments={"command": "ls"})
        >>> result = env.step(state, Action(tool_calls=(call,)))
    """

    def __init__(
        self,
        tasks: tuple[Any, ...],
        harbor_env_factory: Any,
        verifier_factory: Any | None = None,
        *,
        dataset_name: str = "terminal-bench",
        max_steps: int = 30,
        verify_on_truncation: bool = True,
        start_timeout: int | None = 120,
        exec_timeout: int = 120,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._tasks = tasks
        self._harbor_env_factory = harbor_env_factory
        self._verifier_factory = verifier_factory
        self._dataset_name = dataset_name
        self._max_steps = max_steps
        self._verify_on_truncation = verify_on_truncation
        self._start_timeout = start_timeout
        self._exec_timeout = exec_timeout

        self._tools = HARBOR_TOOLS
        self._executor = None  # Not used — we handle execution directly

        self._native_rewards: tuple[RewardFunction, ...] = (
            HarborReward(),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

        self._harbor_env: Any = None
        self._current_task: Any = None

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=f"harbor:{self._dataset_name}",
            adapter="harbor",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={"dataset": self._dataset_name, "tool_mode": True},
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[HarborHidden], dict[str, Any]]:
        options = options or {}
        if "task_index" not in options:
            raise ValueError("options must contain 'task_index'")

        task_index = options["task_index"]
        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        # Stop previous container
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass

        task = self._tasks[task_index]
        self._current_task = task

        # Create and start container
        self._harbor_env = self._harbor_env_factory(task)
        _run_with_timeout(
            self._harbor_env.start(force_build=False),
            self._start_timeout,
            "Harbor container start",
        )

        instruction = getattr(task, "instruction", str(task))

        hidden = HarborHidden(
            task_index=task_index,
            task_name=getattr(task, "name", str(task_index)),
            instruction=instruction,
            episode_step=0,
        )

        observation = Observation(
            prompt=instruction,
            available_tools=self._tools,
            task=ObservationContent(text=instruction),
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "task_name": hidden.task_name,
        }

    def _execute_tool_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call against the Harbor container."""
        if call.name == "execute_command":
            command = call.arguments.get("command", "")
            cwd = call.arguments.get("cwd")
            timeout = call.arguments.get("timeout", self._exec_timeout)

            if cwd:
                command = f"cd {shlex.quote(cwd)} && {command}"

            result = run_async(self._harbor_env.exec(command, timeout_sec=timeout))
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=_format_exec_result(result),
            )

        elif call.name == "read_file":
            path = call.arguments.get("path", "")
            result = run_async(
                self._harbor_env.exec(f"cat {shlex.quote(path)}", timeout_sec=self._exec_timeout)
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            if stderr and not stdout:
                return ToolResult.from_error(
                    call_id=call.id,
                    tool_name=call.name,
                    error_message=stderr,
                )
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output=stdout,
            )

        elif call.name == "write_file":
            path = call.arguments.get("path", "")
            content = call.arguments.get("content", "")
            # Use heredoc to write content safely
            eof_marker = "_LLENVS_EOF_"
            cmd = f"cat > {shlex.quote(path)} << '{eof_marker}'\n{content}\n{eof_marker}"
            result = run_async(self._harbor_env.exec(cmd, timeout_sec=self._exec_timeout))
            stderr = getattr(result, "stderr", "") or ""
            if stderr:
                return ToolResult.from_error(
                    call_id=call.id,
                    tool_name=call.name,
                    error_message=stderr,
                )
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output="File written successfully.",
            )

        elif call.name == "submit":
            return ToolResult.success(
                call_id=call.id,
                tool_name=call.name,
                output="Submitting for verification...",
            )

        return ToolResult.from_error(
            call_id=call.id,
            tool_name=call.name,
            error_message=f"Unknown tool: {call.name}",
        )

    def step(
        self,
        state: State[HarborHidden],
        action: Action,
    ) -> StepResult[HarborHidden]:
        self._state_tracker.validate(state, "HarborToolEnvironment")

        next_step = state.hidden.episode_step + 1
        terminated = False
        truncated = False
        tool_results: list[ToolResult] = []

        if action.has_tool_calls:
            for tc in action.tool_calls:
                validation_error = self._validate_tool_call(tc)
                if validation_error is not None:
                    tool_results.append(validation_error)
                    continue

                try:
                    result = self._execute_tool_call(tc)
                    tool_results.append(result)
                except Exception as e:
                    logger.warning(f"Harbor tool call {tc.name} failed: {e}")
                    tool_results.append(
                        ToolResult.from_error(
                            call_id=tc.id,
                            tool_name=tc.name,
                            error_message=str(e),
                        )
                    )

            # Check for terminal tools
            terminated = self._check_terminal_tools(action.tool_calls)

        # Check truncation
        if not terminated and next_step >= self._max_steps:
            truncated = True

        # Run verifier at terminal
        reward_value: float | None = None
        if terminated or (truncated and self._verify_on_truncation):
            if self._verifier_factory is not None:
                try:
                    rewards = _run_verifier(
                        self._verifier_factory, self._current_task, self._harbor_env
                    )
                    reward_value = rewards.get("reward", 0.0)
                except Exception as e:
                    logger.warning(f"Verifier failed: {e}")
                    reward_value = 0.0

        # Build next observation via BaseToolEnvironment helper
        state_text = "\n".join(
            str(tr.output) if tr.is_success else str(tr.error) for tr in tool_results
        )
        next_obs = self._build_next_observation(
            current_obs=state.observation,
            action=action,
            tool_results=tuple(tool_results),
            state_content=ObservationContent(text=state_text) if state_text else None,
        )

        # Build action text for trajectory tracking
        action_text = action.text
        if action.has_tool_calls:
            parts = []
            for tc in action.tool_calls:
                if tc.name == "execute_command":
                    parts.append(tc.arguments.get("command", tc.name))
                else:
                    parts.append(tc.name)
            action_text = "; ".join(parts)

        next_hidden = HarborHidden(
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            instruction=state.hidden.instruction,
            episode_step=next_step,
            last_action=action_text,
            trajectory=state.hidden.trajectory + ((action_text,) if action_text else ()),
        )

        info: dict[str, Any] = {
            **state.metadata.info,
            "episode_step": next_step,
        }
        if reward_value is not None:
            info["reward"] = reward_value

        next_metadata = StateMetadata(
            step=next_step,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info=info,
        )

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        rewards = self._compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "tool_results": tuple(tool_results),
                "episode_step": next_step,
            },
        )

    def _compute_rewards(
        self,
        state: State[HarborHidden],
        action: Action,
        next_state: State[HarborHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Stop the running container."""
        if self._harbor_env is not None:
            try:
                run_async(self._harbor_env.stop(delete=True))
            except Exception:
                pass
            self._harbor_env = None


# ── Adapter ─────────────────────────────────────────────────────


class HarborAdapter:
    """Adapter for Harbor containerized evaluation environments.

    Harbor is a generic framework for containerized agent evaluation
    managing Docker containers, task discovery, and verification.
    Datasets include Terminal-Bench, aider-polyglot, swe-bench, etc.
    """

    @property
    def name(self) -> str:
        return "harbor"

    def _get_harbor_api(self) -> Any:
        """Import and return Harbor API handles."""
        try:
            from harbor.environments.factory import EnvironmentFactory
            from harbor.models.environment_type import EnvironmentType
            from harbor.models.task.paths import TaskPaths
            from harbor.models.task.task import Task
            from harbor.models.trial.paths import TrialPaths
            from harbor.registry.client.factory import RegistryClientFactory
            from harbor.tasks.client import TaskClient
            from harbor.verifier.verifier import Verifier

            return _HarborAPI(
                registry_client_factory=RegistryClientFactory,
                task_client=TaskClient,
                task_class=Task,
                task_paths_class=TaskPaths,
                environment_factory=EnvironmentFactory,
                environment_type_enum=EnvironmentType,
                trial_paths_class=TrialPaths,
                verifier_class=Verifier,
            )
        except ImportError as e:
            raise ImportError(
                "harbor is required for HarborAdapter. "
                "Install with: pip install harbor\n"
                "See: https://github.com/laude-institute/harbor"
            ) from e

    @staticmethod
    def _parse_name(name: str) -> tuple[str, str | None]:
        """Parse dataset name and optional version.

        Format: ``"dataset@version"`` or just ``"dataset"``.

        Returns:
            Tuple of (dataset_name, version_or_none).
        """
        if "@" in name:
            dataset, version = name.split("@", 1)
            return dataset, version
        return name, None

    def list_environments(self) -> list[str]:
        """List available datasets from Harbor's registry.

        Returns:
            Sorted list of dataset identifiers.

        Raises:
            ImportError: If harbor is not installed.
        """
        api = self._get_harbor_api()
        client = api.registry_client_factory.create()
        datasets = client.get_datasets()
        names = {f"{dataset.name}@{dataset.version}" for dataset in datasets}
        return sorted(names)

    def get_environment(
        self,
        name: str = "terminal-bench@2.0",
        tasks: tuple[Any, ...] | None = None,
        env_factory: Any | None = None,
        verify_factory: Any | None = None,
        dataset_path: str | None = None,
        environment_type: str = "docker",
        tool_mode: bool = False,
        max_steps: int = 30,
        submit_keyword: str = "SUBMIT",
        start_timeout: int | None = 120,
        exec_timeout: int = 120,
        verify_on_truncation: bool = True,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> HarborEnvironment | HarborToolEnvironment:
        """Create a Harbor environment.

        Args:
            name: Dataset name with optional version (e.g., "terminal-bench@2.0").
            tasks: Pre-loaded tuple of Harbor Task objects. If None, loaded
                from Harbor's registry or ``dataset_path``.
            env_factory: Callable ``(task) -> BaseEnvironment`` creating
                Harbor container environments. If None, built from harbor library.
            verify_factory: Callable ``(task, env) -> Verifier``. If None,
                built from harbor library.
            dataset_path: Local path to dataset directory. Used when tasks
                and factories are not provided.
            environment_type: Harbor environment type (docker, daytona, etc.).
            tool_mode: If True, returns ``HarborToolEnvironment`` with structured
                tool calls. If False (default), returns ``HarborEnvironment``
                with text-based commands.
            max_steps: Maximum steps per episode.
            submit_keyword: Text mode only — keyword triggering submission.
            start_timeout: Timeout (seconds) for container start/reset.
            exec_timeout: Per-command timeout in seconds.
            verify_on_truncation: Run verifier when truncating at max_steps.
            extra_rewards: Additional reward functions.
            **kwargs: Passed to Harbor constructors.

        Returns:
            HarborEnvironment or HarborToolEnvironment.
        """
        dataset_name, _version = self._parse_name(name)

        # Load tasks and create factories from Harbor if not provided
        if tasks is None or env_factory is None or verify_factory is None:
            api = self._get_harbor_api()

            if tasks is None:
                if dataset_path is not None:
                    tasks = self._load_tasks_from_path(api, dataset_path)
                else:
                    tasks = self._load_tasks_from_registry(api, dataset_name, _version)

            if env_factory is None:

                def build_harbor_env(task: Any) -> Any:
                    if environment_type == "podman-hpc":
                        return self._create_local_environment(api, task, environment_type, **kwargs)
                    return self._create_harbor_environment(api, task, environment_type, **kwargs)

                env_factory = build_harbor_env

            if verify_factory is None:

                def build_verifier(task: Any, env: Any) -> Any:
                    return api.verifier_class(
                        task=task,
                        trial_paths=env.trial_paths,
                        environment=env,
                        logger=logger,
                    )

                verify_factory = build_verifier

        if tool_mode:
            return HarborToolEnvironment(
                tasks=tasks,
                harbor_env_factory=env_factory,
                verifier_factory=verify_factory,
                dataset_name=dataset_name,
                max_steps=max_steps,
                verify_on_truncation=verify_on_truncation,
                start_timeout=start_timeout,
                exec_timeout=exec_timeout,
                extra_rewards=extra_rewards,
            )

        return HarborEnvironment(
            tasks=tasks,
            harbor_env_factory=env_factory,
            verifier_factory=verify_factory,
            dataset_name=dataset_name,
            max_steps=max_steps,
            submit_keyword=submit_keyword,
            verify_on_truncation=verify_on_truncation,
            start_timeout=start_timeout,
            exec_timeout=exec_timeout,
            extra_rewards=extra_rewards,
        )

    def get_default_system_prompt(self, name: str) -> str:
        """Return a terminal-agent system prompt."""
        return (
            "You are an AI agent with access to a Linux terminal. "
            "Execute commands to complete the task described below. "
            "Work step by step, checking the output of each command "
            "before proceeding. When you have completed the task, "
            "submit your work for verification."
        )

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        dataset_name, version = self._parse_name(name)
        return {
            "name": name,
            "adapter": self.name,
            "dataset": dataset_name,
            "version": version,
            "description": f"Harbor containerized environment ({dataset_name})",
        }

    @staticmethod
    def _load_tasks_from_path(api: "_HarborAPI", dataset_path: str) -> tuple[Any, ...]:
        tasks_root = Path(dataset_path)
        if not tasks_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        tasks: list[Any] = []
        for entry in sorted(tasks_root.iterdir()):
            if not entry.is_dir():
                continue
            task_paths = api.task_paths_class(entry)
            if task_paths.is_valid():
                tasks.append(api.task_class(entry))

        if not tasks:
            raise ValueError(
                "No valid Harbor tasks found in dataset_path. "
                "Expected task directories with instruction.md, task.toml, and environment/."
            )

        return tuple(sorted(tasks, key=lambda t: t.name))

    @staticmethod
    def _load_tasks_from_registry(
        api: "_HarborAPI", dataset_name: str, version: str | None
    ) -> tuple[Any, ...]:
        client = api.registry_client_factory.create()
        spec = client.get_dataset_spec(dataset_name, version=version)
        task_ids = [task.to_source_task_id() for task in spec.tasks]
        task_dirs = api.task_client().download_tasks(task_ids=task_ids)
        tasks = [api.task_class(task_dir=task_dir) for task_dir in task_dirs]
        return tuple(sorted(tasks, key=lambda t: t.name))

    @staticmethod
    def _create_harbor_environment(
        api: "_HarborAPI", task: Any, environment_type: str, **kwargs: Any
    ) -> Any:
        trial_paths = api.trial_paths_class(trial_dir=Path("trials") / str(uuid.uuid4()))
        trial_paths.mkdir()
        env_type = api.environment_type_enum(environment_type)
        env = api.environment_factory.create_environment(
            type=env_type,
            environment_dir=task.paths.environment_dir,
            environment_name=task.name,
            session_id=str(uuid.uuid4()),
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            **kwargs,
        )
        env.trial_paths = trial_paths
        return env

    @staticmethod
    def _create_local_environment(
        api: "_HarborAPI",
        task: Any,
        environment_type: str,
        **kwargs: Any,
    ) -> Any:
        trial_paths = api.trial_paths_class(trial_dir=Path("trials") / str(uuid.uuid4()))
        trial_paths.mkdir()

        if environment_type == "podman-hpc":
            env = PodmanHPCEnvironment(
                environment_dir=Path(task.paths.environment_dir),
                environment_name=task.name,
                session_id=str(uuid.uuid4()),
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                logger=logger,
                **kwargs,
            )
            env.trial_paths = trial_paths
            return env

        raise ValueError(f"Unsupported local Harbor environment type: {environment_type}")


# ── Restore / replay utilities ──────────────────────────────────


def harbor_restore(
    env: HarborEnvironment,
    state: State[HarborHidden],
) -> State[HarborHidden]:
    """Restore a Harbor env to a saved state by replaying the trajectory prefix.

    Resets to the original task via ``task_index``, then replays each command
    from ``state.hidden.trajectory``. Validates task name to guard against
    index drift across dataset versions.

    Args:
        env: A fresh ``HarborEnvironment`` instance (new container).
        state: The target state whose ``hidden.trajectory`` is replayed.

    Returns:
        The restored state after replaying all commands.

    Raises:
        ValueError: If the task name at the given index doesn't match
            the expected task name from the saved state.
    """
    current, info = env.reset(options={"task_index": state.hidden.task_index})

    # Validate task identity
    if state.hidden.task_name and info.get("task_name"):
        if state.hidden.task_name != info["task_name"]:
            raise ValueError(
                f"Task name mismatch: expected {state.hidden.task_name!r}, "
                f"got {info['task_name']!r} at index {state.hidden.task_index}. "
                f"Dataset version may have changed."
            )

    for cmd in state.hidden.trajectory:
        result = env.step(current, Action(text=cmd))
        current = result.next_state

    return current


def validate_replay_consistency(
    env_factory: Callable[[], HarborEnvironment],
    task_index: int,
    trajectory: tuple[str, ...],
    probe_commands: tuple[str, ...] = (
        "find /app /home /etc -type f 2>/dev/null | sort | md5sum",
        "dpkg -l 2>/dev/null | awk '{print $2, $3}' | md5sum",
    ),
    reference_probes: dict[str, str] | None = None,
    num_trials: int = 3,
) -> dict[str, Any]:
    """Test whether replaying a trajectory produces consistent container state.

    Two validation modes:

    1. **Self-consistency** (``reference_probes=None``): checks that multiple
       replays produce the same state as each other.
    2. **Live-vs-restored** (``reference_probes`` provided): checks that
       restored state matches probe outputs captured from the live env
       during original data collection.

    Args:
        env_factory: Creates a fresh ``HarborEnvironment`` instance.
        task_index: Task index to reset to.
        trajectory: Commands to replay.
        probe_commands: Commands to run after replay to fingerprint state.
        reference_probes: Optional mapping of probe command → expected stdout
            from the live container. Enables live-vs-restored comparison.
        num_trials: Number of independent replay trials.

    Returns:
        Dict with keys:
            ``consistent`` (bool): All trials match each other.
            ``matches_reference`` (bool | None): Whether probes match stored
                live probes (None if ``reference_probes`` not provided).
            ``probe_outputs`` (list[dict[str, str]]): Probe results per trial.
            ``divergence_details`` (list[str]): Description of any differences.
    """
    trial_outputs: list[dict[str, str]] = []

    for _trial in range(num_trials):
        env = env_factory()
        try:
            # Reset and replay
            current, _info = env.reset(options={"task_index": task_index})
            for cmd in trajectory:
                result = env.step(current, Action(text=cmd))
                current = result.next_state

            # Run probe commands
            probes: dict[str, str] = {}
            for probe_cmd in probe_commands:
                probe_result = env.step(current, Action(text=probe_cmd))
                obs = probe_result.next_state.observation
                probes[probe_cmd] = obs.state.text if obs.state else ""
                current = probe_result.next_state

            trial_outputs.append(probes)
        finally:
            env.close()

    # Self-consistency: all trials must match the first
    divergence_details: list[str] = []
    consistent = True
    if trial_outputs:
        baseline = trial_outputs[0]
        for i, trial in enumerate(trial_outputs[1:], 1):
            for cmd in probe_commands:
                if trial.get(cmd) != baseline.get(cmd):
                    consistent = False
                    divergence_details.append(f"Trial {i} diverges from trial 0 on probe: {cmd!r}")

    # Live-vs-restored comparison
    matches_reference: bool | None = None
    if reference_probes is not None and trial_outputs:
        matches_reference = True
        baseline = trial_outputs[0]
        for cmd, expected in reference_probes.items():
            actual = baseline.get(cmd, "")
            if actual != expected:
                matches_reference = False
                divergence_details.append(
                    f"Reference mismatch on probe {cmd!r}: expected {expected!r}, got {actual!r}"
                )

    return {
        "consistent": consistent,
        "matches_reference": matches_reference,
        "probe_outputs": trial_outputs,
        "divergence_details": divergence_details,
    }
