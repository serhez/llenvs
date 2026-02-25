"""Container support for running environments in isolated processes or Docker containers."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenvs.container.client import ContainerEnvironment
    from llenvs.core.config import EnvironmentConfig


def create_container_environment(config: EnvironmentConfig) -> ContainerEnvironment:
    """Create a containerized environment from config.

    Starts a runtime (Docker or subprocess), waits for the server to be
    healthy, and returns a ``ContainerEnvironment`` proxy.

    Args:
        config: Environment config with ``container`` set.

    Returns:
        A ``ContainerEnvironment`` connected to the running server.

    Raises:
        ValueError: If ``config.container`` is ``None`` or runtime is unknown.
    """
    from llenvs.container.client import ContainerEnvironment as _ContainerEnvironment
    from llenvs.container.runtime import DockerRuntime, ProcessRuntime

    container_config = config.container
    if container_config is None:
        raise ValueError("config.container must be set")

    # Strip container config for server-side creation (prevents recursion)
    server_config = replace(config, container=None)

    # Serialize config for the server process
    config_dict: dict[str, Any] = {
        "name": server_config.name,
        "adapter": server_config.adapter,
        "size": server_config.size,
        "seed": server_config.seed,
        "answer_extractor": server_config.answer_extractor,
        "answer_extractor_config": server_config.answer_extractor_config,
        "params": server_config.params,
    }
    if server_config.answer_extractors is not None:
        config_dict["answer_extractors"] = server_config.answer_extractors
    if server_config.pre_cleaners is not None:
        config_dict["pre_cleaners"] = server_config.pre_cleaners
    if server_config.post_cleaners is not None:
        config_dict["post_cleaners"] = server_config.post_cleaners
    if server_config.prompt_template is not None:
        config_dict["prompt_template"] = server_config.prompt_template
    if server_config.system_prompt is not None:
        config_dict["system_prompt"] = server_config.system_prompt
    if server_config.prompts is not None:
        config_dict["prompts"] = server_config.prompts

    config_json = json.dumps(config_dict)

    runtime: DockerRuntime | ProcessRuntime
    if container_config.runtime == "docker":
        if container_config.image is None:
            raise ValueError("Docker runtime requires 'image' to be set")
        runtime = DockerRuntime(
            image=container_config.image,
            config_json=config_json,
            port=container_config.port,
            env_vars=container_config.env_vars,
            volumes=container_config.volumes,
            timeout=container_config.timeout,
            docker_command=container_config.docker_command,
        )
    elif container_config.runtime == "process":
        runtime = ProcessRuntime(
            config_json=config_json,
            port=container_config.port,
            timeout=container_config.timeout,
        )
    else:
        raise ValueError(f"Unknown container runtime: {container_config.runtime}")

    url = runtime.start()
    env = _ContainerEnvironment(url=url, timeout=container_config.timeout)
    env._runtime = runtime  # type: ignore[attr-defined]  # Hold reference for cleanup
    return env
