"""Tests for the plugin's bundled default harness.

Requires verifiers v1 (skipped otherwise). verifiers resolves an unpinned seat
(``harness=None``) to the taskset's bundled harness when the taskset module
exports one, else to the built-in ``bash`` coding-agent harness. llenvs tasks
are text relays, so the plugin bundles a tool-less chat-loop harness; a seat
pinned through ``--env.agent.harness.id`` still wins.
"""

# ruff: noqa: E402, I001  (imports follow the importorskip guard)
from __future__ import annotations

import pytest

vf = pytest.importorskip("verifiers.v1")

import llenvs_env
from llenvs_env import LLEnvsHarness
from llenvs_env.env import LLEnvsEnvConfig
from llenvs_env.taskset import LLEnvsTasksetConfig
from verifiers.v1.harnesses.null import NullHarness, NullHarnessConfig
from verifiers.v1.utils import loaders

TASKSET_ID = "llenvs-env"


class TestBundledHarness:
    def test_plugin_exports_the_harness(self):
        assert "LLEnvsHarness" in llenvs_env.__all__
        assert loaders.harness_class(TASKSET_ID) is LLEnvsHarness

    def test_is_a_tool_less_chat_loop(self):
        assert issubclass(LLEnvsHarness, NullHarness)
        assert LLEnvsHarness.EXECUTES_CODE is False
        assert LLEnvsHarness.NEEDS_CONTAINER is False
        assert LLEnvsHarness.APPENDS_SYSTEM_PROMPT is True

    def test_taskset_default_harness_is_the_bundled_one(self):
        assert loaders.default_harness_id(TASKSET_ID) == TASKSET_ID
        cfg = loaders.harness_config_type(TASKSET_ID)(id=TASKSET_ID)
        assert isinstance(cfg, NullHarnessConfig)
        assert isinstance(loaders.load_harness(cfg), LLEnvsHarness)


class TestSeatResolution:
    def test_unpinned_relay_seat_resolves_to_bundled_harness(self):
        config = LLEnvsEnvConfig(taskset=LLEnvsTasksetConfig(id=TASKSET_ID))
        assert config.agent.harness is None  # unpinned in the config itself
        harnesses = config.agent_harnesses()
        assert list(harnesses) == ["agent"]
        assert isinstance(harnesses["agent"], NullHarnessConfig)
        assert harnesses["agent"].id == TASKSET_ID

    def test_unpinned_single_agent_seat_resolves_to_bundled_harness(self):
        # `--env.id single-agent` (single-turn tasks) gets the same default.
        config_type = vf.env_config_type(TASKSET_ID, "single-agent")
        config = config_type(id="single-agent", taskset=LLEnvsTasksetConfig(id=TASKSET_ID))
        harness = config.agent_harnesses()["agent"]
        assert isinstance(harness, NullHarnessConfig)
        assert harness.id == TASKSET_ID

    def test_pinned_seat_wins(self):
        config = LLEnvsEnvConfig(
            taskset=LLEnvsTasksetConfig(id=TASKSET_ID),
            agent=vf.AgentConfig(harness={"id": "bash"}),
        )
        harness = config.agent_harnesses()["agent"]
        assert harness.id == "bash"
        assert not isinstance(harness, NullHarnessConfig)

    def test_pinned_null_still_works(self):
        config = LLEnvsEnvConfig(
            taskset=LLEnvsTasksetConfig(id=TASKSET_ID),
            agent=vf.AgentConfig(harness={"id": "null"}),
        )
        harness = config.agent_harnesses()["agent"]
        assert isinstance(harness, NullHarnessConfig)
        assert harness.id == "null"
