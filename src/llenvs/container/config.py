"""Configuration for running environments in containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContainerConfig:
    """Configuration for running an environment in a container.

    Attributes:
        runtime: Runtime type ("docker" or "process").
        image: Docker image name (required for docker runtime, ignored for process).
        port: Host port to bind (None = auto-select free port).
        timeout: Max seconds to wait for server startup.
        env_vars: Environment variables to pass to the container.
        volumes: Volume mounts (host_path -> container_path).
        docker_command: Path to docker CLI.
    """

    runtime: str = "docker"
    image: str | None = None
    port: int | None = None
    timeout: float = 60.0
    env_vars: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    docker_command: str = "docker"
