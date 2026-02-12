"""Tests for the AgentGym adapter."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenvs.adapters.agentgym import (
    AgentGymAdapter,
    AgentGymEnvironment,
    AgentGymHidden,
    AgentGymReward,
    ENV_REGISTRY,
    _ServerManager,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, Observation


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class _StepOutput:
    """Mimics agentenv StepOutput."""
    state: Any
    reward: float
    done: bool


def _make_mock_client(
    num_tasks: int = 10,
    initial_obs: str = "You are in a room.",
    step_obs: str = "You moved north.",
    step_reward: float = 0.0,
    step_done: bool = False,
) -> MagicMock:
    """Create a mock BaseEnvClient with controlled behaviour."""
    client = MagicMock()
    client.__len__ = MagicMock(return_value=num_tasks)

    # observe() returns the current observation text
    client.observe.return_value = initial_obs

    # reset() returns None (side-effect only in agentenv)
    client.reset.return_value = None

    # step() returns a StepOutput-like object
    step_output = _StepOutput(state=step_obs, reward=step_reward, done=step_done)
    client.step.return_value = step_output

    return client


# ---------------------------------------------------------------------------
# ENV_REGISTRY
# ---------------------------------------------------------------------------

class TestEnvRegistry:
    def test_all_environments_present(self):
        expected = {
            "webshop", "alfworld", "babyai", "maze", "wordle",
            "sciworld", "sqlgym", "textcraft", "webarena",
            "searchqa", "movie", "weather", "academia", "todo", "sheet",
        }
        assert set(ENV_REGISTRY.keys()) == expected

    def test_registry_values_are_tuples(self):
        for name, value in ENV_REGISTRY.items():
            assert isinstance(value, tuple), f"{name} should map to a tuple"
            assert len(value) == 2, f"{name} tuple should have 2 elements"


# ---------------------------------------------------------------------------
# AgentGymHidden
# ---------------------------------------------------------------------------

class TestAgentGymHidden:
    def test_frozen(self):
        hidden = AgentGymHidden(task_index=0, env_name="maze", episode_step=1, last_action=None)
        with pytest.raises(AttributeError):
            hidden.task_index = 5  # type: ignore[misc]

    def test_fields(self):
        hidden = AgentGymHidden(task_index=3, env_name="wordle", episode_step=2, last_action="go north")
        assert hidden.task_index == 3
        assert hidden.env_name == "wordle"
        assert hidden.episode_step == 2
        assert hidden.last_action == "go north"


# ---------------------------------------------------------------------------
# AgentGymReward
# ---------------------------------------------------------------------------

class TestAgentGymReward:
    def test_name(self):
        reward = AgentGymReward()
        assert reward.name == "agentgym_native"

    def test_intermediate_step_reward_type(self):
        reward = AgentGymReward()
        # non-terminal state → STEP
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = False
        next_state.metadata.info = {"agentgym_reward": 0.5}

        signal = reward.compute(state, action, next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.value == 0.5

    def test_terminal_step_reward_type(self):
        reward = AgentGymReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = True
        next_state.metadata.info = {"agentgym_reward": 1.0}

        signal = reward.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.value == 1.0

    def test_missing_reward_defaults_to_zero(self):
        reward = AgentGymReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = False
        next_state.metadata.info = {}

        signal = reward.compute(state, action, next_state)
        assert signal.value == 0.0


# ---------------------------------------------------------------------------
# AgentGymEnvironment
# ---------------------------------------------------------------------------

class TestAgentGymEnvironment:
    def _make_env(self, client=None, max_steps=20, **kwargs):
        if client is None:
            client = _make_mock_client()
        return AgentGymEnvironment(client=client, env_name="maze", max_steps=max_steps, **kwargs)

    # -- spec -----------------------------------------------------------------

    def test_spec_name(self):
        env = self._make_env()
        assert env.spec.name == "maze"

    def test_spec_adapter(self):
        env = self._make_env()
        assert env.spec.adapter == "agentgym"

    def test_spec_is_multi_turn(self):
        env = self._make_env()
        assert env.spec.is_multi_turn is True

    def test_spec_max_steps(self):
        env = self._make_env(max_steps=10)
        assert env.spec.max_steps == 10

    def test_spec_metadata_dataset_size(self):
        client = _make_mock_client(num_tasks=42)
        env = self._make_env(client=client)
        assert env.spec.metadata["dataset_size"] == 42

    def test_spec_metadata_action_format(self):
        client = _make_mock_client()
        env = AgentGymEnvironment(
            client=client, env_name="maze", action_format="function_calling",
        )
        assert env.spec.metadata["action_format"] == "function_calling"

    def test_spec_pure_step_false(self):
        env = self._make_env()
        assert env.spec.pure_step is False

    # -- stale-state detection ------------------------------------------------

    def test_step_raises_on_stale_state(self):
        """Replaying the initial state after a step raises NotImplementedError."""
        client = _make_mock_client(step_done=False)
        env = self._make_env(client=client)
        state_0, _ = env.reset(options={"task_index": 0})

        result = env.step(state_0, Action(text="go north"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, Action(text="go north"))

    # -- __len__ --------------------------------------------------------------

    def test_len(self):
        client = _make_mock_client(num_tasks=25)
        env = self._make_env(client=client)
        assert len(env) == 25

    def test_len_default_client(self):
        env = self._make_env()
        assert len(env) == 10  # _make_mock_client default

    # -- properties -----------------------------------------------------------

    def test_available_tools_empty(self):
        env = self._make_env()
        assert env.available_tools == ()

    def test_prompts_default_empty(self):
        env = self._make_env()
        assert env.prompts == {}

    def test_prompts_custom(self):
        env = self._make_env(prompts={"hint": "Do something"})
        assert env.prompts == {"hint": "Do something"}

    def test_reward_functions_native_only(self):
        env = self._make_env()
        assert len(env.reward_functions) == 1
        assert isinstance(env.reward_functions[0], AgentGymReward)

    def test_reward_functions_with_extras(self):
        extra = MagicMock()
        env = self._make_env(extra_rewards=(extra,))
        assert len(env.reward_functions) == 2
        assert env.reward_functions[1] is extra

    # -- reset ----------------------------------------------------------------

    def test_reset_returns_state_and_info(self):
        env = self._make_env()
        state, info = env.reset(options={"task_index": 3})
        assert state.observation.prompt == "You are in a room."
        assert state.hidden.task_index == 3
        assert state.hidden.env_name == "maze"
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 3

    def test_reset_calls_client(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        env.reset(options={"task_index": 5})
        client.reset.assert_called_once_with(5)
        client.observe.assert_called_once()

    def test_reset_default_task_index(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        env.reset()
        client.reset.assert_called_once_with(0)

    def test_reset_task_index_out_of_bounds_high(self):
        client = _make_mock_client(num_tasks=5)
        env = self._make_env(client=client)
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": 5})

    def test_reset_task_index_out_of_bounds_negative(self):
        client = _make_mock_client(num_tasks=5)
        env = self._make_env(client=client)
        with pytest.raises(ValueError, match="out of bounds"):
            env.reset(options={"task_index": -1})

    def test_reset_task_index_boundary_valid(self):
        client = _make_mock_client(num_tasks=5)
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 4})
        assert state.hidden.task_index == 4

    # -- observe coercion -----------------------------------------------------

    def test_reset_coerces_str_observation(self):
        client = _make_mock_client(initial_obs="plain text")
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.prompt == "plain text"

    def test_reset_coerces_dict_observation_with_key(self):
        client = _make_mock_client()
        client.observe.return_value = {"observation": "from dict", "extra": 123}
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.prompt == "from dict"

    def test_reset_coerces_dict_observation_without_key(self):
        client = _make_mock_client()
        client.observe.return_value = {"some_key": "some_val"}
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        assert state.observation.prompt == str({"some_key": "some_val"})

    def test_step_coerces_dict_state(self):
        client = _make_mock_client()
        client.step.return_value = _StepOutput(
            state={"observation": "dict obs"}, reward=0.0, done=False,
        )
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.next_state.observation.prompt == "dict obs"

    def test_step_coerces_non_string_state(self):
        client = _make_mock_client()
        client.step.return_value = _StepOutput(
            state=42, reward=0.0, done=False,
        )
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.next_state.observation.prompt == "42"

    # -- action_format --------------------------------------------------------

    def test_action_format_default(self):
        env = self._make_env()
        assert env.spec.metadata["action_format"] == "react"

    def test_action_format_custom(self):
        client = _make_mock_client()
        env = AgentGymEnvironment(
            client=client, env_name="maze", action_format="code_as_action",
        )
        assert env.spec.metadata["action_format"] == "code_as_action"

    # -- step -----------------------------------------------------------------

    def test_step_returns_step_result(self):
        client = _make_mock_client(step_obs="You see a door.", step_reward=0.5, step_done=False)
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="go north")
        result = env.step(state, action)

        assert result.next_state.observation.prompt == "You see a door."
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "go north"
        assert result.terminated is False
        assert result.truncated is False
        assert result.info["agentgym_reward"] == 0.5

    def test_step_calls_client(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="go north")
        env.step(state, action)
        client.step.assert_called_once_with("go north")

    def test_step_terminal(self):
        client = _make_mock_client(step_reward=1.0, step_done=True)
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="submit")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.truncated is False
        assert result.next_state.metadata.is_terminal is True

    def test_step_truncation_at_max_steps(self):
        client = _make_mock_client(step_done=False)
        env = self._make_env(client=client, max_steps=2)
        state, _ = env.reset(options={"task_index": 0})

        # Step 1: episode_step goes to 1, max_steps=2 so not truncated yet
        action = Action(text="action1")
        result = env.step(state, action)
        assert result.truncated is False

        # Step 2: episode_step goes to 2, 2 >= 2 so truncated
        result2 = env.step(result.next_state, Action(text="action2"))
        assert result2.truncated is True
        assert result2.terminated is False
        assert result2.next_state.metadata.is_terminal is True

    def test_step_rewards_computed(self):
        client = _make_mock_client(step_reward=0.75, step_done=False)
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="act")
        result = env.step(state, action)

        assert len(result.rewards.signals) == 1
        assert result.rewards.signals[0].value == 0.75
        assert result.rewards.signals[0].reward_type == RewardType.STEP

    def test_step_terminal_reward_type(self):
        client = _make_mock_client(step_reward=1.0, step_done=True)
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="done"))
        assert result.rewards.signals[0].reward_type == RewardType.OUTCOME

    def test_step_increments_metadata_step(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        assert state.metadata.step == 0

        result = env.step(state, Action(text="a"))
        assert result.next_state.metadata.step == 1

        result2 = env.step(result.next_state, Action(text="b"))
        assert result2.next_state.metadata.step == 2

    def test_episode_id_preserved_across_steps(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        eid = state.metadata.episode_id

        result = env.step(state, Action(text="a"))
        assert result.next_state.metadata.episode_id == eid

    # -- compute_rewards directly ---------------------------------------------

    def test_compute_rewards(self):
        env = self._make_env()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = True
        next_state.metadata.info = {"agentgym_reward": 0.9}

        bundle = env.compute_rewards(state, action, next_state)
        assert len(bundle.signals) == 1
        assert bundle.signals[0].value == 0.9

    # -- Phase 2: reset return capture ----------------------------------------

    def test_reset_captures_dict_return(self):
        client = _make_mock_client()
        client.reset.return_value = {"goal": "find the exit", "difficulty": 3}
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        assert info["reset_goal"] == "find the exit"
        assert info["reset_difficulty"] == 3
        assert state.metadata.info["reset_goal"] == "find the exit"

    def test_reset_captures_none_return(self):
        client = _make_mock_client()
        client.reset.return_value = None
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        # No reset_ keys in info
        assert not any(k.startswith("reset_") for k in info)

    def test_reset_captures_list_return(self):
        client = _make_mock_client()
        client.reset.return_value = [{"obs": "start"}, {"extra": True}]
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        assert info["reset_obs"] == "start"

    # -- Phase 2: client info -------------------------------------------------

    def test_reset_reads_client_info(self):
        client = _make_mock_client()
        client.info = {"admissible_commands": ["go north", "look"], "observation": "dup"}
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        # "observation" is skipped to avoid redundancy
        assert info["client_admissible_commands"] == ["go north", "look"]
        assert "client_observation" not in info

    def test_step_reads_client_info(self):
        client = _make_mock_client()
        client.info = {"score": 5}
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.info["client_score"] == 5

    def test_client_without_info_attr(self):
        client = _make_mock_client()
        # MagicMock has no real .info by default — use spec to remove it
        del client.info
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        assert not any(k.startswith("client_") for k in info)

    # -- Phase 2: enriched info dicts -----------------------------------------

    def test_reset_info_has_action_format(self):
        env = self._make_env()
        _, info = env.reset(options={"task_index": 0})
        assert info["action_format"] == "react"

    def test_reset_info_has_dataset_size(self):
        client = _make_mock_client(num_tasks=50)
        env = self._make_env(client=client)
        _, info = env.reset(options={"task_index": 0})
        assert info["dataset_size"] == 50

    def test_step_info_has_truncated(self):
        client = _make_mock_client(step_done=False)
        env = self._make_env(client=client, max_steps=3)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.info["truncated"] is False

    def test_step_info_has_episode_step(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.info["episode_step"] == 1

    def test_step_info_has_env_name(self):
        client = _make_mock_client()
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.info["env_name"] == "maze"

    # -- Phase 3: available actions -------------------------------------------

    def test_hidden_available_actions_default(self):
        hidden = AgentGymHidden(
            task_index=0, env_name="maze", episode_step=0, last_action=None,
        )
        assert hidden.available_actions == ()

    def test_reset_extracts_available_actions(self):
        client = _make_mock_client()
        client.info = {"available_actions": ["go north", "look", "take key"]}
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        assert state.hidden.available_actions == ("go north", "look", "take key")
        assert info["available_actions"] == ("go north", "look", "take key")

    def test_step_extracts_available_actions(self):
        client = _make_mock_client()
        client.info = {"available_actions": ["open door"]}
        env = self._make_env(client=client)
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="act"))
        assert result.next_state.hidden.available_actions == ("open door",)
        assert result.info["available_actions"] == ("open door",)

    def test_no_available_actions_when_absent(self):
        client = _make_mock_client()
        # No .info attribute
        del client.info
        env = self._make_env(client=client)
        state, info = env.reset(options={"task_index": 0})
        assert state.hidden.available_actions == ()
        assert "available_actions" not in info


# ---------------------------------------------------------------------------
# AgentGymAdapter - conversation prompts
# ---------------------------------------------------------------------------

class TestAgentGymAdapter:
    def test_name(self):
        adapter = AgentGymAdapter()
        assert adapter.name == "agentgym"

    def test_list_environments(self):
        adapter = AgentGymAdapter()
        envs = adapter.list_environments()
        assert len(envs) == 15
        assert "maze" in envs
        assert "webshop" in envs
        assert "alfworld" in envs

    def test_get_native_answer_extractor_returns_none(self):
        adapter = AgentGymAdapter()
        assert adapter.get_native_answer_extractor("maze") is None

    def test_get_default_system_prompt_returns_none(self):
        adapter = AgentGymAdapter()
        assert adapter.get_default_system_prompt("maze") is None

    def test_get_prompt_template_returns_none(self):
        adapter = AgentGymAdapter()
        assert adapter.get_prompt_template("maze") is None

    def test_get_environment_info(self):
        adapter = AgentGymAdapter()
        info = adapter.get_environment_info("maze")
        assert info["name"] == "maze"
        assert info["adapter"] == "agentgym"
        assert info["type"] == "multi_turn"

    @patch("llenvs.adapters.agentgym._ServerManager.get_or_start")
    def test_get_environment_creates_env(self, mock_get_or_start):
        mock_get_or_start.return_value = "http://localhost:12345"

        mock_client_class = MagicMock()
        mock_client_instance = _make_mock_client()
        mock_client_class.return_value = mock_client_instance

        adapter = AgentGymAdapter()
        with patch.object(adapter, "_get_agentenv"), \
             patch.object(adapter, "_resolve_client_class", return_value=mock_client_class):
            env = adapter.get_environment("maze", max_steps=10)

        assert isinstance(env, AgentGymEnvironment)
        assert env.spec.name == "maze"
        assert env.spec.max_steps == 10
        mock_get_or_start.assert_called_once()

    @patch("llenvs.adapters.agentgym._ServerManager.get_or_start")
    def test_get_environment_passes_action_format(self, mock_get_or_start):
        mock_get_or_start.return_value = "http://localhost:12345"

        mock_client_class = MagicMock()
        mock_client_instance = _make_mock_client()
        mock_client_class.return_value = mock_client_instance

        adapter = AgentGymAdapter()
        with patch.object(adapter, "_get_agentenv"), \
             patch.object(adapter, "_resolve_client_class", return_value=mock_client_class):
            env = adapter.get_environment("maze", action_format="function_calling")

        assert env.spec.metadata["action_format"] == "function_calling"

    @patch("llenvs.adapters.agentgym._ServerManager.get_or_start")
    def test_get_environment_extracts_conversation_prompts(self, mock_get_or_start):
        mock_get_or_start.return_value = "http://localhost:12345"

        mock_client_class = MagicMock()
        mock_client_instance = _make_mock_client()
        # Simulate _conversation_start attribute (agentenv pattern)
        mock_client_instance._conversation_start = {
            "react": [
                ("human", "You are an agent. Use ReAct format."),
                ("gpt", "OK, I will use ReAct format."),
            ],
        }
        mock_client_class.return_value = mock_client_instance

        adapter = AgentGymAdapter()
        with patch.object(adapter, "_get_agentenv"), \
             patch.object(adapter, "_resolve_client_class", return_value=mock_client_class):
            env = adapter.get_environment("maze", action_format="react")

        assert env.prompts["system_prompt"] == "You are an agent. Use ReAct format."
        assert env.prompts["assistant_ack"] == "OK, I will use ReAct format."

    @patch("llenvs.adapters.agentgym._ServerManager.get_or_start")
    def test_conversation_prompts_user_override(self, mock_get_or_start):
        mock_get_or_start.return_value = "http://localhost:12345"

        mock_client_class = MagicMock()
        mock_client_instance = _make_mock_client()
        mock_client_instance._conversation_start = {
            "react": [("human", "Auto prompt")],
        }
        mock_client_class.return_value = mock_client_instance

        adapter = AgentGymAdapter()
        with patch.object(adapter, "_get_agentenv"), \
             patch.object(adapter, "_resolve_client_class", return_value=mock_client_class):
            env = adapter.get_environment(
                "maze",
                prompts={"system_prompt": "My custom prompt"},
            )

        # User override wins
        assert env.prompts["system_prompt"] == "My custom prompt"

    @patch("llenvs.adapters.agentgym._ServerManager.get_or_start")
    def test_no_conversation_start_attr(self, mock_get_or_start):
        mock_get_or_start.return_value = "http://localhost:12345"

        mock_client_class = MagicMock()
        mock_client_instance = _make_mock_client()
        # Remove _conversation_start (not all clients have it)
        del mock_client_instance._conversation_start
        mock_client_class.return_value = mock_client_instance

        adapter = AgentGymAdapter()
        with patch.object(adapter, "_get_agentenv"), \
             patch.object(adapter, "_resolve_client_class", return_value=mock_client_class):
            env = adapter.get_environment("maze")

        assert env.prompts == {}

    @patch("llenvs.adapters.agentgym._ServerManager.get_or_start")
    def test_conversation_start_fallback_format(self, mock_get_or_start):
        mock_get_or_start.return_value = "http://localhost:12345"

        mock_client_class = MagicMock()
        mock_client_instance = _make_mock_client()
        # Only has "react" format, but we ask for "function_calling"
        mock_client_instance._conversation_start = {
            "react": [("human", "React prompt")],
        }
        mock_client_class.return_value = mock_client_instance

        adapter = AgentGymAdapter()
        with patch.object(adapter, "_get_agentenv"), \
             patch.object(adapter, "_resolve_client_class", return_value=mock_client_class):
            env = adapter.get_environment("maze", action_format="function_calling")

        # Falls back to first available format
        assert env.prompts["system_prompt"] == "React prompt"

    def test_get_agentenv_import_error(self):
        adapter = AgentGymAdapter()
        with patch.dict("sys.modules", {"agentenv": None}):
            with pytest.raises(ImportError):
                adapter._get_agentenv()


# ---------------------------------------------------------------------------
# _ServerManager
# ---------------------------------------------------------------------------

class TestServerManager:
    def test_returns_existing_url_if_env_server_base_provided(self):
        url = _ServerManager.get_or_start("maze", env_server_base="http://my-server:8000")
        assert url == "http://my-server:8000"

    @patch("llenvs.adapters.agentgym._ServerManager._start_server")
    def test_starts_server_when_not_running(self, mock_start):
        mock_start.return_value = "http://localhost:9999"
        # Clear any cached servers
        _ServerManager._servers.pop("testenv_unique", None)

        url = _ServerManager.get_or_start("testenv_unique")
        assert url == "http://localhost:9999"
        mock_start.assert_called_once_with("testenv_unique")

        # Cleanup
        _ServerManager._servers.pop("testenv_unique", None)

    @patch("llenvs.adapters.agentgym._ServerManager._start_server")
    def test_reuses_running_server(self, mock_start):
        # Pre-populate a running server
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        _ServerManager._servers["reuse_test"] = (mock_proc, 7777)

        url = _ServerManager.get_or_start("reuse_test")
        assert url == "http://localhost:7777"
        mock_start.assert_not_called()

        # Cleanup
        _ServerManager._servers.pop("reuse_test", None)
