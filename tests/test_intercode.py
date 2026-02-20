"""Tests for the InterCode adapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenvs.core.reward import RewardType
from llenvs.core.state import Action, Observation, ObservationContent

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockDataLoader:
    """Mock InterCode data_loader for testing.

    Simulates a dataset with tasks that have 'query' and 'gold' fields.
    """

    def __init__(self, tasks: list[dict[str, str]] | None = None):
        self._tasks = tasks or [
            {"query": "List all files in /home", "gold": "ls /home"},
            {"query": "Find the number of lines in file.txt", "gold": "wc -l file.txt"},
            {"query": "Create a directory called test", "gold": "mkdir test"},
            {"query": "Show disk usage", "gold": "df -h"},
            {"query": "Print current directory", "gold": "pwd"},
        ]

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self._tasks[index]

    def __iter__(self):
        return iter(self._tasks)


class MockInterCodeEnv:
    """Mock InterCode environment for testing.

    Simulates InterCode's reset(index) → obs, step(action) → (obs, reward, done, info)
    API with a simple state machine.
    """

    def __init__(self, data_path: str = "", image_name: str = ""):
        self.data_loader = MockDataLoader()
        self._step_count = 0
        self._done = False
        self._current_query = ""
        self._current_gold = ""
        self._reward = 0.0
        self._closed = False

    def reset(self, index: int | None = None) -> str:
        self._step_count = 0
        self._done = False
        self._reward = 0.0

        if index is not None and 0 <= index < len(self.data_loader):
            task = self.data_loader[index]
            self._current_query = task["query"]
            self._current_gold = task["gold"]
        else:
            self._current_query = "Default task"
            self._current_gold = "default"

        return f"Task: {self._current_query}"

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._step_count += 1

        if action.strip().lower().startswith("submit"):
            self._done = True
            # Simulate scoring: 1.0 if gold matches, 0.0 otherwise
            submitted = action.replace("submit", "", 1).strip()
            if submitted == self._current_gold:
                self._reward = 1.0
            else:
                self._reward = 0.0
            return (
                "Submitted.",
                self._reward,
                True,
                {"reward": self._reward, "action": action},
            )

        if action == "error_command":
            return (
                "Error: command not found",
                0.0,
                False,
                {"error": True, "action": action},
            )

        # Normal execution
        obs = f"Output of: {action}"
        return obs, 0.0, False, {"action": action}

    def close(self) -> None:
        self._closed = True


def _make_data_loader(num_tasks: int = 5) -> MockDataLoader:
    """Create a MockDataLoader with a specified number of tasks."""
    tasks = [{"query": f"Task {i}", "gold": f"solution_{i}"} for i in range(num_tasks)]
    return MockDataLoader(tasks)


def _make_ic_env(num_tasks: int = 5) -> MockInterCodeEnv:
    """Create a MockInterCodeEnv with the given number of tasks."""
    ic_env = MockInterCodeEnv()
    ic_env.data_loader = _make_data_loader(num_tasks)
    return ic_env


# ---------------------------------------------------------------------------
# Hidden state tests
# ---------------------------------------------------------------------------


class TestInterCodeHidden:
    """Tests for InterCodeHidden state."""

    def test_creation(self):
        from llenvs.adapters.intercode import InterCodeHidden

        hidden = InterCodeHidden(
            task_index=0,
            env_type="bash",
            query="List files",
            gold="ls",
            episode_step=2,
            last_action="ls -la",
            cumulative_reward=0.0,
            trajectory=("cd /home", "ls -la"),
        )

        assert hidden.task_index == 0
        assert hidden.env_type == "bash"
        assert hidden.query == "List files"
        assert hidden.gold == "ls"
        assert hidden.episode_step == 2
        assert hidden.last_action == "ls -la"
        assert hidden.cumulative_reward == 0.0
        assert hidden.trajectory == ("cd /home", "ls -la")

    def test_immutability(self):
        from llenvs.adapters.intercode import InterCodeHidden

        hidden = InterCodeHidden(
            task_index=0,
            env_type="bash",
            query="Q",
            gold="A",
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            trajectory=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore

    def test_defaults(self):
        from llenvs.adapters.intercode import InterCodeHidden

        hidden = InterCodeHidden(
            task_index=0,
            env_type="sql",
            query="SELECT 1",
            gold="1",
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            trajectory=(),
        )
        assert hidden.last_action is None
        assert hidden.trajectory == ()
        assert hidden.cumulative_reward == 0.0


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


class TestInterCodeReward:
    """Tests for InterCodeReward."""

    def test_reward_name(self):
        from llenvs.adapters.intercode import InterCodeReward

        reward_fn = InterCodeReward()
        assert reward_fn.name == "intercode"

    def test_intermediate_reward_none(self):
        """Non-terminal steps produce STEP rewards with reward=None."""
        from llenvs.adapters.intercode import InterCodeReward

        reward_fn = InterCodeReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = False
        next_state.metadata.info = {"reward": 0.0}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.STEP
        assert signal.reward is None

    def test_terminal_reward_from_info(self):
        """Terminal steps produce OUTCOME rewards with reward from info dict."""
        from llenvs.adapters.intercode import InterCodeReward

        reward_fn = InterCodeReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = True
        next_state.metadata.info = {"reward": 1.0}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 1.0

    def test_terminal_reward_zero(self):
        """Terminal with incorrect answer gives reward=0.0."""
        from llenvs.adapters.intercode import InterCodeReward

        reward_fn = InterCodeReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = True
        next_state.metadata.info = {"reward": 0.0}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0

    def test_terminal_partial_reward(self):
        """Terminal with partial score gives intermediate reward value."""
        from llenvs.adapters.intercode import InterCodeReward

        reward_fn = InterCodeReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = True
        next_state.metadata.info = {"reward": 0.5}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.5

    def test_terminal_no_reward_in_info(self):
        """Terminal with no reward in info defaults to 0.0."""
        from llenvs.adapters.intercode import InterCodeReward

        reward_fn = InterCodeReward()
        state = MagicMock()
        action = MagicMock()
        next_state = MagicMock()
        next_state.metadata.is_terminal = True
        next_state.metadata.info = {}

        signal = reward_fn.compute(state, action, next_state)
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


def _make_env(ic_env: MockInterCodeEnv | None = None, **kwargs) -> Any:
    """Create an InterCodeEnvironment with a mock InterCode env."""
    from llenvs.adapters.intercode import InterCodeEnvironment

    if ic_env is None:
        ic_env = _make_ic_env()
    return InterCodeEnvironment(intercode_env=ic_env, **kwargs)


class TestInterCodeEnvironment:
    """Tests for InterCodeEnvironment."""

    def test_spec(self):
        env = _make_env()
        spec = env.spec
        assert spec.name == "intercode"
        assert spec.adapter == "intercode"
        assert spec.is_multi_turn is True
        assert spec.pure_step is False
        assert spec.supports_task_index is True
        assert spec.supports_len is True
        assert spec.supports_seed is False

    def test_spec_max_steps(self):
        env = _make_env(max_steps=15)
        assert env.spec.max_steps == 15

    def test_spec_default_max_steps(self):
        env = _make_env()
        assert env.spec.max_steps == 10

    def test_len(self):
        env = _make_env()
        assert len(env) == 5

    def test_len_custom(self):
        ic_env = _make_ic_env(num_tasks=10)
        env = _make_env(ic_env=ic_env)
        assert len(env) == 10

    def test_available_tools_empty(self):
        env = _make_env()
        assert env.available_tools == ()

    def test_reward_functions(self):
        env = _make_env()
        rfs = env.reward_functions
        assert len(rfs) == 1
        assert rfs[0].name == "intercode"

    def test_reward_functions_with_extra(self):
        extra = MagicMock()
        env = _make_env(extra_rewards=(extra,))
        rfs = env.reward_functions
        assert len(rfs) == 2
        assert rfs[1] is extra

    def test_reset(self):
        env = _make_env()
        state, info = env.reset(options={"task_index": 0})

        assert isinstance(state.observation, Observation)
        assert state.hidden.task_index == 0
        assert state.hidden.env_type == "bash"
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None
        assert state.hidden.cumulative_reward == 0.0
        assert state.hidden.trajectory == ()

        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False
        assert info["task_index"] == 0

    def test_reset_prompt_from_observation(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        # The prompt should contain the InterCode reset observation
        assert len(state.observation.prompt) > 0

    def test_reset_query_in_hidden(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.query == "Task 0"

    def test_reset_gold_in_hidden(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.gold == "solution_0"

    def test_reset_requires_task_index(self):
        env = _make_env()
        with pytest.raises(ValueError, match="task_index"):
            env.reset(options={})

    def test_reset_validates_bounds(self):
        env = _make_env()
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": 100})

    def test_reset_negative_index(self):
        env = _make_env()
        with pytest.raises(IndexError, match="out of range"):
            env.reset(options={"task_index": -1})

    def test_reset_custom_episode_id(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0, "episode_id": "custom-ep"})
        assert state.metadata.episode_id == "custom-ep"

    def test_step_basic(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="ls -la")
        result = env.step(state, action)

        assert result.terminated is False
        assert result.truncated is False
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "ls -la"
        assert result.next_state.metadata.step == 1
        assert result.next_state.metadata.is_terminal is False

    def test_step_submit(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="submit solution_0")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

    def test_step_submit_reward_in_info(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="submit solution_0")
        result = env.step(state, action)

        assert "reward" in result.next_state.metadata.info

    def test_step_error_output(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        action = Action(text="error_command")
        result = env.step(state, action)

        assert result.terminated is False
        # Error output should be in the observation
        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "Error" in last_msg

    def test_truncation(self):
        env = _make_env(max_steps=2)
        state, _ = env.reset(options={"task_index": 0})

        # Step 1
        result = env.step(state, Action(text="ls"))
        assert result.truncated is False

        # Step 2 — truncation
        result = env.step(result.next_state, Action(text="pwd"))
        assert result.truncated is True
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is True

    def test_terminated_at_max_steps_not_truncated(self):
        """If the env terminates on the last possible step, terminated=True, truncated=False."""
        env = _make_env(max_steps=1)
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="submit solution_0"))
        assert result.terminated is True
        assert result.truncated is False

    def test_messages_accumulate(self):
        env = _make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})
        initial_prompt = state.observation.prompt

        result = env.step(state, Action(text="ls"))
        state = result.next_state
        assert len(state.observation.messages) == 2
        assert state.observation.messages[0] == {"role": "assistant", "content": "ls"}
        assert state.observation.prompt == initial_prompt

        result = env.step(state, Action(text="pwd"))
        state = result.next_state
        assert len(state.observation.messages) == 4
        assert state.observation.messages[2] == {"role": "assistant", "content": "pwd"}
        assert state.observation.prompt == initial_prompt

    def test_state_continuity_rejects_stale(self):
        env = _make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})
        env.step(state, Action(text="ls"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state, Action(text="pwd"))

    def test_trajectory_accumulates(self):
        env = _make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.trajectory == ()

        result = env.step(state, Action(text="ls"))
        assert result.next_state.hidden.trajectory == ("ls",)

        result = env.step(result.next_state, Action(text="pwd"))
        assert result.next_state.hidden.trajectory == ("ls", "pwd")

    def test_cumulative_reward_tracking(self):
        env = _make_env(max_steps=10)
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.cumulative_reward == 0.0

        # Non-submit step: reward is 0.0
        result = env.step(state, Action(text="ls"))
        assert result.next_state.hidden.cumulative_reward == 0.0

        # Submit step: reward is 1.0
        result = env.step(result.next_state, Action(text="submit solution_0"))
        assert result.next_state.hidden.cumulative_reward == 1.0

    def test_close(self):
        ic_env = _make_ic_env()
        env = _make_env(ic_env=ic_env)
        env.reset(options={"task_index": 0})
        env.close()
        assert ic_env._closed is True

    def test_close_without_reset(self):
        env = _make_env()
        env.close()  # Should not raise

    def test_compute_rewards_directly(self):
        from llenvs.core.state import State, StateMetadata

        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        next_state = State(
            observation=state.observation,
            hidden=state.hidden,
            metadata=StateMetadata(
                step=1,
                episode_id=state.metadata.episode_id,
                is_terminal=True,
                info={"reward": 1.0},
            ),
        )
        rewards = env.compute_rewards(state, Action(text="submit"), next_state)
        assert len(rewards.signals) >= 1

    def test_env_type_stored(self):
        env = _make_env(env_type="sql")
        state, _ = env.reset(options={"task_index": 0})
        assert state.hidden.env_type == "sql"

    def test_step_info_contains_action(self):
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="ls -la"))
        assert "action" in result.info

    def test_prompts_empty(self):
        env = _make_env()
        assert env.prompts == {}

    def test_spec_metadata(self):
        env = _make_env()
        assert "env_type" in env.spec.metadata


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestInterCodeAdapter:
    """Tests for InterCodeAdapter."""

    def test_name(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        assert adapter.name == "intercode"

    def test_list_environments(self):
        from llenvs.adapters.intercode import INTERCODE_PRESETS, InterCodeAdapter

        adapter = InterCodeAdapter()
        envs = adapter.list_environments()
        for preset_name in INTERCODE_PRESETS:
            assert f"intercode:{preset_name}" in envs

    def test_import_error(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        with pytest.raises(ImportError, match="InterCode"):
            adapter._get_intercode()

    def test_get_environment_with_intercode_env(self):
        from llenvs.adapters.intercode import InterCodeAdapter, InterCodeEnvironment

        adapter = InterCodeAdapter()
        ic_env = _make_ic_env()
        env = adapter.get_environment("intercode:bash", intercode_env=ic_env)
        assert isinstance(env, InterCodeEnvironment)
        assert len(env) == 5

    def test_get_environment_preset_resolution(self, monkeypatch):
        """Adapter resolves preset name to create env with correct env_type."""
        from llenvs.adapters.intercode import InterCodeAdapter, InterCodeEnvironment

        adapter = InterCodeAdapter()

        # Mock the import
        mock_intercode = MagicMock()
        mock_ic_class = MagicMock(return_value=_make_ic_env())
        mock_intercode.BashEnv = mock_ic_class
        monkeypatch.setattr(adapter, "_get_intercode", lambda: mock_intercode)

        env = adapter.get_environment("intercode:bash", data_path="/some/path")
        assert isinstance(env, InterCodeEnvironment)

    def test_get_environment_unknown_preset(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        ic_env = _make_ic_env()
        with pytest.raises(ValueError, match="Unknown"):
            adapter.get_environment("intercode:unknown_type", intercode_env=ic_env)

    def test_get_environment_no_env_no_path(self, monkeypatch):
        """Without intercode_env or data_path, error is raised."""
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        mock_intercode = MagicMock()
        monkeypatch.setattr(adapter, "_get_intercode", lambda: mock_intercode)

        with pytest.raises(ValueError, match="intercode_env.*data_path"):
            adapter.get_environment("intercode:bash")

    def test_max_steps_passed_through(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        ic_env = _make_ic_env()
        env = adapter.get_environment("intercode:bash", intercode_env=ic_env, max_steps=20)
        assert env._max_steps == 20

    def test_get_native_answer_extractor(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        assert adapter.get_native_answer_extractor("intercode") is None

    def test_get_prompt_template(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        assert adapter.get_prompt_template("intercode") is None

    def test_get_default_system_prompt(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        assert adapter.get_default_system_prompt("intercode") is None

    def test_get_environment_info(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        info = adapter.get_environment_info("intercode")
        assert info["name"] == "intercode"
        assert info["adapter"] == "intercode"
        assert "env_types" in info


# ---------------------------------------------------------------------------
# Full episode integration tests
# ---------------------------------------------------------------------------


class TestInterCodeFullEpisode:
    """Integration tests for multi-step episodes."""

    def test_full_bash_episode(self):
        """Full bash episode: explore, then submit."""
        env = _make_env(max_steps=10)
        state, info = env.reset(options={"task_index": 0})

        assert state.metadata.step == 0

        # task and state are set on reset
        assert isinstance(state.observation.task, ObservationContent)
        assert state.observation.task.text == state.observation.prompt
        assert isinstance(state.observation.state, ObservationContent)
        assert state.observation.state.text == state.observation.prompt
        reset_task = state.observation.task

        # Explore
        actions = ["ls -la", "cat file.txt", "pwd"]
        for i, act_text in enumerate(actions):
            result = env.step(state, Action(text=act_text))
            state = result.next_state
            assert state.metadata.step == i + 1
            assert state.hidden.episode_step == i + 1
            assert state.hidden.last_action == act_text
            assert result.terminated is False

            # task carried forward, state updated to step obs
            assert state.observation.task is reset_task
            assert isinstance(state.observation.state, ObservationContent)
            step_obs = state.observation.messages[-1]["content"]
            assert state.observation.state.text == step_obs

        # Submit
        result = env.step(state, Action(text="submit solution_0"))
        state = result.next_state
        assert result.terminated is True
        assert state.metadata.is_terminal is True

        # task still carried forward on terminal step
        assert state.observation.task is reset_task

        # Check OUTCOME reward
        signal = result.rewards.by_name("intercode")
        assert signal is not None
        assert signal.reward_type == RewardType.OUTCOME

    def test_full_sql_episode(self):
        """Full SQL episode."""
        env = _make_env(env_type="sql", max_steps=10)
        state, _ = env.reset(options={"task_index": 0})

        # Query
        result = env.step(state, Action(text="SELECT * FROM table"))
        assert result.terminated is False

        # Submit
        result = env.step(result.next_state, Action(text="submit solution_0"))
        assert result.terminated is True

    def test_truncation_episode(self):
        """Episode truncated at max_steps."""
        env = _make_env(max_steps=3)
        state, _ = env.reset(options={"task_index": 0})

        for act in ["ls", "pwd", "whoami"]:
            result = env.step(state, Action(text=act))
            state = result.next_state

        assert result.truncated is True
        assert result.terminated is False

        signal = result.rewards.by_name("intercode")
        assert signal.reward_type == RewardType.OUTCOME

    def test_immediate_submit(self):
        """Submit on first step."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="submit solution_0"))
        assert result.terminated is True
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.trajectory == ("submit solution_0",)

    def test_incorrect_submit(self):
        """Submit with wrong answer gives low reward."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="submit wrong_answer"))
        assert result.terminated is True

        signal = result.rewards.by_name("intercode")
        assert signal.reward_type == RewardType.OUTCOME
        assert signal.reward == 0.0

    def test_messages_on_terminal(self):
        """Messages are properly accumulated even on terminal step."""
        env = _make_env()
        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="submit solution_0"))

        assert result.terminated is True
        assert len(result.next_state.observation.messages) == 2
        assert result.next_state.observation.messages[0] == {
            "role": "assistant",
            "content": "submit solution_0",
        }

    def test_different_task_indices(self):
        """Resetting with different task indices selects different tasks."""
        env = _make_env(max_steps=10)
        state0, _ = env.reset(options={"task_index": 0})
        assert state0.hidden.query == "Task 0"
        assert state0.hidden.gold == "solution_0"

        state1, _ = env.reset(options={"task_index": 1})
        assert state1.hidden.query == "Task 1"
        assert state1.hidden.gold == "solution_1"


# ---------------------------------------------------------------------------
# Presets tests
# ---------------------------------------------------------------------------


class TestInterCodePresets:
    """Tests for INTERCODE_PRESETS constant."""

    def test_presets_defined(self):
        from llenvs.adapters.intercode import INTERCODE_PRESETS

        assert isinstance(INTERCODE_PRESETS, dict)
        assert len(INTERCODE_PRESETS) >= 3
        assert "bash" in INTERCODE_PRESETS
        assert "sql" in INTERCODE_PRESETS

    def test_preset_keys(self):
        from llenvs.adapters.intercode import INTERCODE_PRESETS

        for name, preset in INTERCODE_PRESETS.items():
            assert "env_class" in preset
            assert "module" in preset


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestInterCodeRegistration:
    """Tests for adapter registration."""

    def test_adapter_registers_when_available(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        with patch.object(adapter, "_get_intercode") as mock_get:
            mock_get.return_value = MagicMock()
            adapter._get_intercode()
            mock_get.assert_called_once()

    def test_adapter_skipped_when_not_installed(self):
        from llenvs.adapters.intercode import InterCodeAdapter

        adapter = InterCodeAdapter()
        with patch.object(adapter, "_get_intercode", side_effect=ImportError("no intercode")):
            with pytest.raises(ImportError):
                adapter._get_intercode()
