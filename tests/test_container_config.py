"""Tests for container configuration and factory integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llenvs.container.config import ContainerConfig
from llenvs.core.config import EnvironmentConfig, EnvironmentFactory, EvalConfig


# ---------------------------------------------------------------------------
# ContainerConfig
# ---------------------------------------------------------------------------


class TestContainerConfig:
    def test_defaults(self):
        cc = ContainerConfig()
        assert cc.runtime == "docker"
        assert cc.image is None
        assert cc.port is None
        assert cc.timeout == 60.0
        assert cc.env_vars == {}
        assert cc.volumes == {}
        assert cc.docker_command == "docker"

    def test_all_fields(self):
        cc = ContainerConfig(
            runtime="process",
            image="test:latest",
            port=9090,
            timeout=120.0,
            env_vars={"KEY": "val"},
            volumes={"/host": "/container"},
            docker_command="/usr/local/bin/docker",
        )
        assert cc.runtime == "process"
        assert cc.image == "test:latest"
        assert cc.port == 9090


# ---------------------------------------------------------------------------
# EnvironmentConfig with container
# ---------------------------------------------------------------------------


class TestEnvironmentConfigContainer:
    def test_default_none(self):
        ec = EnvironmentConfig(name="test")
        assert ec.container is None

    def test_with_container(self):
        ec = EnvironmentConfig(
            name="test",
            container=ContainerConfig(runtime="process"),
        )
        assert ec.container is not None
        assert ec.container.runtime == "process"


# ---------------------------------------------------------------------------
# YAML / dict round-trip
# ---------------------------------------------------------------------------


class TestYamlParsing:
    def test_from_dict_with_container(self):
        data = {
            "environments": [
                {
                    "name": "sudoku",
                    "adapter": "reasoning_gym",
                    "size": 100,
                    "container": {
                        "runtime": "docker",
                        "image": "llenvs-rg:latest",
                        "timeout": 120,
                        "env_vars": {"CACHE": "/tmp"},
                        "volumes": {"/host/data": "/data"},
                    },
                }
            ],
            "model": {"backend": "openai", "model": "gpt-4"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.container is not None
        assert env.container.runtime == "docker"
        assert env.container.image == "llenvs-rg:latest"
        assert env.container.timeout == 120.0
        assert env.container.env_vars == {"CACHE": "/tmp"}
        assert env.container.volumes == {"/host/data": "/data"}

    def test_from_dict_process_runtime(self):
        data = {
            "environments": [
                {
                    "name": "test",
                    "container": {"runtime": "process"},
                }
            ],
            "model": {"model": "x"},
        }
        config = EvalConfig.from_dict(data)
        env = config.environments[0]
        assert env.container is not None
        assert env.container.runtime == "process"
        assert env.container.image is None

    def test_from_dict_no_container(self):
        data = {
            "environments": [{"name": "test"}],
            "model": {"model": "x"},
        }
        config = EvalConfig.from_dict(data)
        assert config.environments[0].container is None

    def test_to_dict_with_container(self):
        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    container=ContainerConfig(
                        runtime="docker",
                        image="test:latest",
                        port=9090,
                        env_vars={"K": "V"},
                    ),
                )
            ],
            model=__import__("llenvs.core.config", fromlist=["ModelConfig"]).ModelConfig(),
        )
        d = config.to_dict()
        env_d = d["environments"][0]
        assert "container" in env_d
        assert env_d["container"]["runtime"] == "docker"
        assert env_d["container"]["image"] == "test:latest"
        assert env_d["container"]["port"] == 9090
        assert env_d["container"]["env_vars"] == {"K": "V"}

    def test_to_dict_no_container(self):
        config = EvalConfig(
            environments=[EnvironmentConfig(name="test")],
            model=__import__("llenvs.core.config", fromlist=["ModelConfig"]).ModelConfig(),
        )
        d = config.to_dict()
        assert "container" not in d["environments"][0]

    def test_to_dict_process_minimal(self):
        """Process runtime with defaults omits optional fields."""
        config = EvalConfig(
            environments=[
                EnvironmentConfig(
                    name="test",
                    container=ContainerConfig(runtime="process"),
                )
            ],
            model=__import__("llenvs.core.config", fromlist=["ModelConfig"]).ModelConfig(),
        )
        d = config.to_dict()
        c = d["environments"][0]["container"]
        assert c["runtime"] == "process"
        assert "image" not in c  # None is omitted
        assert "port" not in c
        assert "timeout" not in c  # 60.0 is default, omitted
        assert "env_vars" not in c  # Empty dict omitted
        assert "docker_command" not in c  # "docker" is default

    def test_round_trip(self):
        data = {
            "environments": [
                {
                    "name": "test",
                    "adapter": "reasoning_gym",
                    "container": {
                        "runtime": "docker",
                        "image": "img:v1",
                        "timeout": 30,
                    },
                }
            ],
            "model": {"model": "x"},
        }
        config = EvalConfig.from_dict(data)
        d = config.to_dict()
        config2 = EvalConfig.from_dict(d)
        assert config2.environments[0].container.runtime == "docker"
        assert config2.environments[0].container.image == "img:v1"
        assert config2.environments[0].container.timeout == 30.0


# ---------------------------------------------------------------------------
# EnvironmentFactory.create() with container
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    def test_factory_delegates_to_create_container(self):
        """Factory calls create_container_environment when container is set."""
        config = EnvironmentConfig(
            name="test",
            container=ContainerConfig(runtime="process"),
        )
        mock_env = MagicMock()
        with patch(
            "llenvs.container.create_container_environment",
            return_value=mock_env,
        ) as mock_create:
            result = EnvironmentFactory.create(config)

        mock_create.assert_called_once_with(config)
        assert result is mock_env

    def test_factory_normal_without_container(self):
        """Without container, factory creates environment normally."""
        config = EnvironmentConfig(name="leg_counting", adapter="reasoning_gym", size=5)
        # This should work normally without container
        env = EnvironmentFactory.create(config)
        assert env is not None

    def test_create_container_unknown_runtime(self):
        """Unknown runtime raises ValueError."""
        from llenvs.container import create_container_environment

        config = EnvironmentConfig(
            name="test",
            container=ContainerConfig(runtime="unknown"),
        )
        with pytest.raises(ValueError, match="Unknown container runtime"):
            create_container_environment(config)

    def test_create_container_docker_no_image(self):
        """Docker runtime without image raises ValueError."""
        from llenvs.container import create_container_environment

        config = EnvironmentConfig(
            name="test",
            container=ContainerConfig(runtime="docker", image=None),
        )
        with pytest.raises(ValueError, match="image"):
            create_container_environment(config)

    def test_create_container_no_config(self):
        """container=None raises ValueError."""
        from llenvs.container import create_container_environment

        config = EnvironmentConfig(name="test")
        with pytest.raises(ValueError, match="container must be set"):
            create_container_environment(config)
