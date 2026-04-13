"""Tests for the OpenApps adapter."""

from typing import Any

import pytest

from llenvs.adapters.open_apps import (
    OPEN_APPS_MODULES,
    OPEN_APPS_TASKS,
    OpenAppsAdapter,
    OpenAppsEnvironment,
    OpenAppsHidden,
    OpenAppsReward,
    _get_app_state,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, Observation, ObservationContent


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockTask:
    """Mock OpenApps Task matching the Task ABC interface."""

    def __init__(
        self,
        goal: str = "Add 'Call Mom' to my todo list.",
        task_id: str = "mock_task_123",
        complete: bool = False,
    ):
        self.goal = goal
        self.task_id = task_id
        self._complete = complete

    def check_if_task_is_complete(
        self, initial_state: dict, current_state: dict
    ) -> bool:
        return self._complete


class MockBrowserGymEnv:
    """Mock BrowserGym environment for testing.

    Simulates the gymnasium-style interface returned by browsergym:
    - reset() -> (obs_dict, info)
    - step(action) -> (obs_dict, reward, terminated, truncated, info)
    """

    def __init__(self, goal: str = "Add 'Call Mom' to my todo list."):
        self._goal = goal
        self._step_count = 0
        self._done = False

    def _make_obs(self, content: str = "Welcome page") -> dict[str, Any]:
        """Build a BrowserGym-style observation dict.

        Returns an ``axtree_object`` dict (as real BrowserGym does) that
        ``flatten_axtree_to_str`` can process, plus ``extra_element_properties``.
        """
        axtree_object = {
            "nodes": [
                {
                    "nodeId": "1",
                    "role": {"value": "RootWebArea"},
                    "name": {"value": "OpenApps"},
                    "childIds": ["2", "3", "4", "5", "6", "7"],
                    "properties": [],
                },
                {
                    "nodeId": "2",
                    "role": {"value": "heading"},
                    "name": {"value": "OpenApps Dashboard"},
                    "childIds": [],
                    "browsergym_id": "10",
                    "properties": [],
                },
                {
                    "nodeId": "3",
                    "role": {"value": "link"},
                    "name": {"value": "Calendar"},
                    "childIds": [],
                    "browsergym_id": "11",
                    "properties": [],
                },
                {
                    "nodeId": "4",
                    "role": {"value": "link"},
                    "name": {"value": "Todo"},
                    "childIds": [],
                    "browsergym_id": "12",
                    "properties": [],
                },
                {
                    "nodeId": "5",
                    "role": {"value": "link"},
                    "name": {"value": "Messenger"},
                    "childIds": [],
                    "browsergym_id": "13",
                    "properties": [],
                },
                {
                    "nodeId": "6",
                    "role": {"value": "link"},
                    "name": {"value": "Map"},
                    "childIds": [],
                    "browsergym_id": "14",
                    "properties": [],
                },
                {
                    "nodeId": "7",
                    "role": {"value": "StaticText"},
                    "name": {"value": content},
                    "childIds": [],
                    "properties": [],
                },
            ]
        }
        extra_properties = {
            "10": {"visibility": 1.0, "bbox": [0, 0, 100, 30], "clickable": False, "set_of_marks": False},
            "11": {"visibility": 1.0, "bbox": [0, 30, 100, 20], "clickable": True, "set_of_marks": False},
            "12": {"visibility": 1.0, "bbox": [0, 50, 100, 20], "clickable": True, "set_of_marks": False},
            "13": {"visibility": 1.0, "bbox": [0, 70, 100, 20], "clickable": True, "set_of_marks": False},
            "14": {"visibility": 1.0, "bbox": [0, 90, 100, 20], "clickable": True, "set_of_marks": False},
        }
        return {
            "axtree_object": axtree_object,
            "extra_element_properties": extra_properties,
            "screenshot": None,
        }

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_count = 0
        self._done = False
        return self._make_obs("Welcome page"), {"task_id": "mock"}

    def step(
        self, action: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        if action and "click" in action.lower() and "todo" in action.lower():
            obs = self._make_obs("Todo App - Add a new task")
            return obs, 0.0, False, False, {}

        if action and "fill" in action.lower() and "call mom" in action.lower():
            obs = self._make_obs("Todo App - Task 'Call Mom' added")
            return obs, 0.0, False, False, {}

        if action and "click" in action.lower() and "submit" in action.lower():
            obs = self._make_obs("Todo App - Task submitted successfully")
            self._done = True
            return obs, 1.0, True, False, {}

        # Default: no-op
        obs = self._make_obs(f"Page after action: {action}")
        return obs, 0.0, False, False, {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_http(monkeypatch):
    """Stub out _get_app_state so tests don't need a running server."""
    monkeypatch.setattr(
        "llenvs.adapters.open_apps._get_app_state",
        lambda url: {},
    )


@pytest.fixture
def mock_task() -> MockTask:
    return MockTask()


@pytest.fixture
def mock_task_complete() -> MockTask:
    return MockTask(complete=True)


@pytest.fixture
def mock_browsergym_env() -> MockBrowserGymEnv:
    return MockBrowserGymEnv()


def _make_env(
    mock_task: MockTask,
    mock_browsergym_env: MockBrowserGymEnv,
    *,
    task_name: str = "add_call_mom_to_my_todo",
    max_steps: int = 10,
    use_screenshot: bool = False,
    screenshot_with_som: bool = False,
    omit_axtree_text: bool = False,
) -> OpenAppsEnvironment:
    """Build a single-task OpenAppsEnvironment from mock components.

    Wraps the mocks in a one-element ``task_names`` + a trivial factory
    that always returns the same (task, browsergym) pair.  Used by the
    legacy fixtures whose tests don't care about task switching.
    """
    return OpenAppsEnvironment(
        task_names=(task_name,),
        task_factory=lambda _name: (mock_task, mock_browsergym_env),
        base_url="http://localhost:5001",
        max_steps=max_steps,
        use_screenshot=use_screenshot,
        screenshot_with_som=screenshot_with_som,
        omit_axtree_text=omit_axtree_text,
    )


@pytest.fixture
def env(mock_browsergym_env, mock_task) -> OpenAppsEnvironment:
    """Standard environment with task-incomplete mock."""
    return _make_env(mock_task, mock_browsergym_env)


@pytest.fixture
def env_complete(mock_browsergym_env, mock_task_complete) -> OpenAppsEnvironment:
    """Environment where the task is always complete (for reward testing)."""
    return _make_env(mock_task_complete, mock_browsergym_env)


# ---------------------------------------------------------------------------
# Tests: OpenAppsHidden
# ---------------------------------------------------------------------------


class TestOpenAppsHidden:
    """Tests for the hidden state dataclass."""

    def test_creation(self):
        hidden = OpenAppsHidden(
            task_name="add_call_mom_to_my_todo",
            goal="Add 'Call Mom' to my todo list.",
            task_index=0,
            episode_step=0,
            initial_app_state={"todo": [], "calendar": []},
            last_action=None,
            base_url="http://localhost:5001",
        )
        assert hidden.task_name == "add_call_mom_to_my_todo"
        assert hidden.goal == "Add 'Call Mom' to my todo list."
        assert hidden.task_index == 0
        assert hidden.episode_step == 0
        assert hidden.last_action is None
        assert hidden.base_url == "http://localhost:5001"

    def test_immutability(self):
        hidden = OpenAppsHidden(
            task_name="test",
            goal="test goal",
            task_index=0,
            episode_step=0,
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore

    def test_defaults(self):
        hidden = OpenAppsHidden(
            task_name="test",
            goal="test",
            task_index=0,
            episode_step=0,
        )
        assert hidden.initial_app_state == {}
        assert hidden.last_action is None
        assert hidden.base_url == ""


# ---------------------------------------------------------------------------
# Tests: OpenAppsReward
# ---------------------------------------------------------------------------


class TestOpenAppsReward:
    """Tests for the reward function."""

    def test_name(self):
        reward_fn = OpenAppsReward()
        assert reward_fn.name == "task_completion"

    def test_reward_type(self):
        reward_fn = OpenAppsReward()
        assert reward_fn.reward_type == RewardType.OUTCOME


# ---------------------------------------------------------------------------
# Tests: OpenAppsEnvironment
# ---------------------------------------------------------------------------


class TestOpenAppsEnvironment:
    """Tests for the MDP environment wrapper."""

    def test_spec(self, env):
        assert env.spec.name == "add_call_mom_to_my_todo"
        assert env.spec.adapter == "open_apps"
        assert env.spec.is_multi_turn is True
        assert env.spec.max_steps == 10
        assert env.spec.pure_step is False

    def test_available_tools_empty(self, env):
        assert env.available_tools == ()

    def test_reward_functions(self, env):
        rfs = env.reward_functions
        assert len(rfs) >= 1
        assert rfs[0].name == "task_completion"

    def test_reset(self, env):
        state, info = env.reset(options={"task_index": 0})

        # Observation
        assert isinstance(state.observation, Observation)
        assert "Goal:" in state.observation.prompt
        assert "Call Mom" in state.observation.prompt

        # Hidden
        assert isinstance(state.hidden, OpenAppsHidden)
        assert state.hidden.task_name == "add_call_mom_to_my_todo"
        assert state.hidden.episode_step == 0
        assert state.hidden.last_action is None

        # Metadata
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

        # Info
        assert info["task_name"] == "add_call_mom_to_my_todo"
        assert "Call Mom" in info["goal"]

    def test_reset_observation_structure(self, env):
        state, _ = env.reset()

        # task = static goal, state = dynamic axtree content
        assert isinstance(state.observation.task, ObservationContent)
        assert "Goal:" in state.observation.task.text
        assert isinstance(state.observation.state, ObservationContent)
        assert "heading" in state.observation.state.text

        # No messages on reset
        assert state.observation.messages == ()

    def test_step(self, env):
        state, _ = env.reset()

        action = Action(text="click('todo_link')")
        result = env.step(state, action)

        assert result.terminated is False
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "click('todo_link')"
        assert result.next_state.metadata.step == 1

    def test_step_raises_on_stale_state(self, env):
        state_0, _ = env.reset()

        env.step(state_0, Action(text="click('x')"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, Action(text="click('x')"))

    def test_truncation(self, mock_browsergym_env, mock_task):
        env = _make_env(mock_task, mock_browsergym_env, task_name="test", max_steps=2)
        state, _ = env.reset()

        result1 = env.step(state, Action(text="click('a')"))
        assert result1.truncated is False

        result2 = env.step(result1.next_state, Action(text="click('b')"))
        assert result2.truncated is True
        assert result2.next_state.metadata.is_terminal is True

    def test_task_completion_terminates(self, env_complete, monkeypatch):
        """When task.check_if_task_is_complete returns True, episode terminates."""
        monkeypatch.setattr(
            "llenvs.adapters.open_apps._get_app_state",
            lambda url: {"todo": [{"title": "Call Mom", "done": None}]},
        )
        state, _ = env_complete.reset()

        result = env_complete.step(state, Action(text="click('submit')"))

        assert result.terminated is True
        assert result.info["task_complete"] is True
        assert result.info["open_apps_reward"] == 1.0

    def test_reward_zero_when_incomplete(self, env):
        """When task is not complete, reward is 0."""
        state, _ = env.reset()
        result = env.step(state, Action(text="click('x')"))

        reward_signal = result.rewards.by_name("task_completion")
        assert reward_signal is not None
        assert reward_signal.reward == 0.0

    def test_reward_one_when_complete(self, env_complete):
        """When task is complete, reward is 1.0."""
        state, _ = env_complete.reset()
        result = env_complete.step(state, Action(text="click('x')"))

        reward_signal = result.rewards.by_name("task_completion")
        assert reward_signal is not None
        assert reward_signal.reward == 1.0

    def test_compute_rewards(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="noop"))

        signals = result.rewards.signals
        assert len(signals) >= 1
        assert signals[0].name == "task_completion"
        assert signals[0].reward_type == RewardType.OUTCOME


# ---------------------------------------------------------------------------
# Tests: Message history
# ---------------------------------------------------------------------------


class TestOpenAppsMessageHistory:
    """Tests for message history accumulation."""

    def test_empty_on_reset(self, env):
        state, _ = env.reset()
        assert state.observation.messages == ()

    def test_accumulates(self, env):
        state, _ = env.reset()
        initial_prompt = state.observation.prompt

        # Turn 1
        result1 = env.step(state, Action(text="click('todo')"))
        s1 = result1.next_state
        assert len(s1.observation.messages) == 2
        assert s1.observation.messages[0] == {
            "role": "assistant",
            "content": "click('todo')",
        }
        assert s1.observation.messages[1]["role"] == "user"
        assert s1.observation.prompt == initial_prompt

        # Turn 2
        result2 = env.step(s1, Action(text="fill('input', 'Call Mom')"))
        s2 = result2.next_state
        assert len(s2.observation.messages) == 4
        assert s2.observation.messages[2] == {
            "role": "assistant",
            "content": "fill('input', 'Call Mom')",
        }
        assert s2.observation.prompt == initial_prompt


# ---------------------------------------------------------------------------
# Tests: Multi-step episode
# ---------------------------------------------------------------------------


class TestOpenAppsMultiStepEpisode:
    """Integration tests for multi-step episodes."""

    def test_full_todo_episode(self, monkeypatch, mock_browsergym_env):
        """Test a complete episode: navigate → fill → submit."""
        call_count = {"n": 0}

        def mock_app_state(url):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return {"todo": []}
            # After submit, task appears
            return {"todo": [{"title": "Call Mom", "done": None}]}

        monkeypatch.setattr(
            "llenvs.adapters.open_apps._get_app_state",
            mock_app_state,
        )

        task = MockTask(complete=False)
        env = _make_env(task, mock_browsergym_env)

        state, info = env.reset()
        assert state.metadata.step == 0
        assert "Call Mom" in info["goal"]

        # Step 1: click todo link
        result1 = env.step(state, Action(text="click('todo_link')"))
        assert not result1.terminated
        assert result1.next_state.metadata.step == 1

        # Step 2: fill input
        result2 = env.step(
            result1.next_state,
            Action(text="fill('task_input', 'Call Mom')"),
        )
        assert not result2.terminated
        assert result2.next_state.metadata.step == 2

        # Step 3: click submit → BrowserGym returns terminated=True
        # Also make the task mock say complete
        task._complete = True
        result3 = env.step(
            result2.next_state,
            Action(text="click('submit_button')"),
        )
        assert result3.terminated is True
        assert result3.info["task_complete"] is True
        assert result3.info["open_apps_reward"] == 1.0
        assert result3.next_state.metadata.step == 3

        # Message history has 6 entries (3 turns × 2)
        assert len(result3.next_state.observation.messages) == 6

    def test_episode_truncates_without_completion(self, mock_browsergym_env, mock_task):
        env = _make_env(mock_task, mock_browsergym_env, task_name="test", max_steps=3)

        state, _ = env.reset()
        r1 = env.step(state, Action(text="a"))
        r2 = env.step(r1.next_state, Action(text="b"))
        r3 = env.step(r2.next_state, Action(text="c"))

        assert r3.truncated is True
        assert r3.terminated is False
        assert r3.info["task_complete"] is False


# ---------------------------------------------------------------------------
# Tests: OpenAppsAdapter
# ---------------------------------------------------------------------------


class TestOpenAppsAdapter:
    """Tests for the adapter class (no real dependencies needed)."""

    def test_adapter_name(self):
        adapter = OpenAppsAdapter()
        assert adapter.name == "open_apps"

    def test_list_environments(self):
        adapter = OpenAppsAdapter()
        envs = adapter.list_environments()
        assert isinstance(envs, list)
        assert len(envs) == len(OPEN_APPS_TASKS)
        assert "add_call_mom_to_my_todo" in envs
        assert "add_meeting_with_dennis" in envs
        assert "save_paris_to_my_favorite_places" in envs

    def test_get_environment_info(self):
        adapter = OpenAppsAdapter()
        info = adapter.get_environment_info("add_call_mom_to_my_todo")
        assert info["name"] == "add_call_mom_to_my_todo"
        assert info["adapter"] == "open_apps"
        assert info["type"] == "multi_turn"
        assert "apps" in info
        assert set(info["apps"]) == set(OPEN_APPS_MODULES)

    def test_get_default_system_prompt(self):
        adapter = OpenAppsAdapter()
        prompt = adapter.get_default_system_prompt("add_call_mom_to_my_todo")
        assert prompt is not None
        assert "web application" in prompt.lower()
        assert "browser" in prompt.lower()

    def test_get_prompt_template_none(self):
        adapter = OpenAppsAdapter()
        assert adapter.get_prompt_template("test") is None

    def test_get_native_answer_extractor_none(self):
        adapter = OpenAppsAdapter()
        assert adapter.get_native_answer_extractor("test") is None


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_open_apps_tasks_not_empty(self):
        assert len(OPEN_APPS_TASKS) > 0

    def test_open_apps_modules_not_empty(self):
        assert len(OPEN_APPS_MODULES) > 0
        assert "calendar" in OPEN_APPS_MODULES
        assert "todo" in OPEN_APPS_MODULES
        assert "messenger" in OPEN_APPS_MODULES
        assert "map" in OPEN_APPS_MODULES


# ---------------------------------------------------------------------------
# Tests: task_index switching via task_factory
# ---------------------------------------------------------------------------


class TestOpenAppsTaskIndexSwitching:
    """reset(options={'task_index': N}) should swap to OPEN_APPS_TASKS[N]."""

    @staticmethod
    def _make_factory(calls: list[str]):
        """Build a factory that records every (mock_task, mock_env) it makes."""
        def factory(task_name: str) -> tuple[MockTask, MockBrowserGymEnv]:
            calls.append(task_name)
            return MockTask(goal=f"goal for {task_name}"), MockBrowserGymEnv()
        return factory

    def _build_env(
        self,
        calls: list[str],
        *,
        task_names: tuple[str, ...] = OPEN_APPS_TASKS,
        initial_task_index: int = 0,
    ) -> OpenAppsEnvironment:
        return OpenAppsEnvironment(
            task_names=task_names,
            task_factory=self._make_factory(calls),
            base_url="http://localhost:5001",
            max_steps=10,
            initial_task_index=initial_task_index,
        )

    def test_initial_task_built_eagerly(self):
        calls: list[str] = []
        env = self._build_env(calls, initial_task_index=0)
        assert env._task_name == OPEN_APPS_TASKS[0]
        # Factory invoked once for the initial task
        assert calls == [OPEN_APPS_TASKS[0]]
        # __len__ exposes the full collection
        assert len(env) == len(OPEN_APPS_TASKS)
        # spec advertises task_index support
        assert env.spec.supports_task_index is True
        assert env.spec.supports_len is True
        assert env.spec.metadata["task_names"] == list(OPEN_APPS_TASKS)

    def test_switches_to_target_task_on_reset(self):
        calls: list[str] = []
        env = self._build_env(calls, initial_task_index=0)

        target_name = OPEN_APPS_TASKS[4]
        state, _ = env.reset(options={"task_index": 4})

        assert env._task_name == target_name
        assert state.hidden.task_name == target_name
        assert state.hidden.task_index == 4
        # Factory invoked once for the initial task and once for the target
        assert calls == [OPEN_APPS_TASKS[0], target_name]

    def test_no_swap_when_target_matches_current(self):
        calls: list[str] = []
        env = self._build_env(calls, initial_task_index=2)
        # Reset to the same index — no extra factory call
        env.reset(options={"task_index": 2})
        assert env._task_name == OPEN_APPS_TASKS[2]
        assert calls == [OPEN_APPS_TASKS[2]]  # only the eager initial build

    def test_caches_proxies_per_task_name(self):
        """Switching back to a previously-built task should reuse the cached env."""
        calls: list[str] = []
        env = self._build_env(calls, initial_task_index=0)

        env.reset(options={"task_index": 1})  # build task 1
        env.reset(options={"task_index": 0})  # back to task 0 (cached)
        env.reset(options={"task_index": 1})  # back to task 1 (cached)

        # Factory called exactly twice: once eagerly for task 0, once for task 1
        assert calls == [OPEN_APPS_TASKS[0], OPEN_APPS_TASKS[1]]

    def test_index_modulo_wraps(self):
        calls: list[str] = []
        env = self._build_env(calls, initial_task_index=0)

        # task_index = 8 should wrap to OPEN_APPS_TASKS[0] (no swap)
        env.reset(options={"task_index": 8})
        assert env._task_name == OPEN_APPS_TASKS[0]
        assert calls == [OPEN_APPS_TASKS[0]]  # only eager build

        # task_index = 9 should wrap to OPEN_APPS_TASKS[1]
        env.reset(options={"task_index": 9})
        assert env._task_name == OPEN_APPS_TASKS[1]
        assert calls == [OPEN_APPS_TASKS[0], OPEN_APPS_TASKS[1]]

    def test_round_robin_8_tasks(self):
        """Driving task_index 0..7 should produce 8 distinct task names."""
        calls: list[str] = []
        env = self._build_env(calls, initial_task_index=0)

        seen_names = []
        for i in range(len(OPEN_APPS_TASKS)):
            state, _ = env.reset(options={"task_index": i})
            seen_names.append(state.hidden.task_name)

        assert seen_names == list(OPEN_APPS_TASKS)
        # Factory called once per task (initial eager + 7 lazy), each cached
        assert calls == list(OPEN_APPS_TASKS)


# ---------------------------------------------------------------------------
# Tests: vision-only mode (omit_axtree_text + screenshot_with_som)
# ---------------------------------------------------------------------------


class TestOpenAppsVisionOnly:
    """Flags that support a true vision-only actor condition."""

    def test_omit_axtree_text_replaces_state_text_with_stub(
        self, mock_task, mock_browsergym_env
    ):
        env = _make_env(mock_task, mock_browsergym_env, omit_axtree_text=True)
        state, _ = env.reset()
        text = state.observation.state.text
        assert "accessibility tree omitted" in text
        assert "Set-of-Marks" in text
        # Sanity: real axtree content from MockBrowserGymEnv should NOT appear
        assert "OpenApps Dashboard" not in text

    def test_omit_axtree_text_persists_through_step(
        self, mock_task, mock_browsergym_env
    ):
        env = _make_env(mock_task, mock_browsergym_env, omit_axtree_text=True)
        state, _ = env.reset()
        result = env.step(state, Action(text="click('11')"))
        assert "accessibility tree omitted" in result.next_state.observation.state.text

    def test_default_keeps_axtree_text(self, mock_task, mock_browsergym_env):
        """Without omit_axtree_text, the flattened a11y tree is shown."""
        env = _make_env(mock_task, mock_browsergym_env, omit_axtree_text=False)
        state, _ = env.reset()
        text = state.observation.state.text
        assert "accessibility tree omitted" not in text
        # MockBrowserGymEnv axtree contains "OpenApps Dashboard"
        assert "OpenApps Dashboard" in text

    def test_screenshot_with_som_calls_overlay(
        self, mock_task, mock_browsergym_env, monkeypatch
    ):
        """When screenshot_with_som=True, overlay_som is invoked on the screenshot."""
        import numpy as np

        # Provide a real numpy screenshot in the mock obs so the vision
        # path actually runs.
        original_make_obs = mock_browsergym_env._make_obs

        def make_obs_with_screenshot(*args, **kwargs):
            obs = original_make_obs(*args, **kwargs)
            obs["screenshot"] = np.zeros((720, 1280, 3), dtype=np.uint8)
            return obs

        monkeypatch.setattr(
            mock_browsergym_env, "_make_obs", make_obs_with_screenshot
        )

        # Spy on browsergym.utils.obs.overlay_som from within the adapter.
        calls: list[int] = []
        from browsergym.utils import obs as bg_obs

        def fake_overlay_som(screenshot, extra_props, **kw):
            calls.append(len(extra_props))
            # Return the input unchanged so encoding succeeds.
            return np.asarray(screenshot)

        monkeypatch.setattr(bg_obs, "overlay_som", fake_overlay_som)

        env = _make_env(
            mock_task,
            mock_browsergym_env,
            use_screenshot=True,
            screenshot_with_som=True,
        )
        state, _ = env.reset()

        # overlay_som called at least once with the mock's extra_props
        assert len(calls) >= 1
        # Image is still attached to the observation
        assert state.observation.state.images
        assert state.observation.state.images[0].media_type == "image/png"

    def test_screenshot_without_som_skips_overlay(
        self, mock_task, mock_browsergym_env, monkeypatch
    ):
        import numpy as np

        original_make_obs = mock_browsergym_env._make_obs

        def make_obs_with_screenshot(*args, **kwargs):
            obs = original_make_obs(*args, **kwargs)
            obs["screenshot"] = np.zeros((720, 1280, 3), dtype=np.uint8)
            return obs

        monkeypatch.setattr(
            mock_browsergym_env, "_make_obs", make_obs_with_screenshot
        )

        from browsergym.utils import obs as bg_obs

        called = []
        monkeypatch.setattr(
            bg_obs, "overlay_som", lambda *a, **kw: called.append(1) or a[0]
        )

        env = _make_env(
            mock_task,
            mock_browsergym_env,
            use_screenshot=True,
            screenshot_with_som=False,
        )
        env.reset()
        assert called == []  # overlay never invoked
