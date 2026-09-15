"""Explicit task-local replay must reproduce rewards, not just annotate states.

Synthetic app data and comparator; no browser, network or real dataset is used.
Native behavior remains the default. The scoped rule uses native task targets
and comparison, masking irrelevant apps to TARGET values before comparison.
"""

import copy
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from llenvs.adapters import open_apps as oa
from llenvs.core.state import Action, Observation

INITIAL = {
    "calendar": [{"title": "Deadline"}, {"title": "Keep"}],
    "todo": [{"title": "Water plants", "done": False}],
    "map": [{"name": "Park", "coords": [40, -73]}],
    "messenger": [{"user": "Bob", "messages": []}],
}
TASK = "remove_wacv_abstract_deadline"


class RemoveDeadline:
    goal = "Remove the deadline"
    task_id = "synthetic-deadline"

    def get_target_state(self, initial):
        target = copy.deepcopy(initial)
        target["calendar"] = [{"title": "Keep"}]
        return target

    def check_if_task_is_complete(self, initial, current):
        return self.get_target_state(initial) == current


@pytest.fixture
def build_environment(monkeypatch):
    live = copy.deepcopy(INITIAL)
    next_data = copy.deepcopy(INITIAL)

    class Browser:
        def reset(self):
            live.clear()
            live.update(copy.deepcopy(INITIAL))
            return {}, {}

        def step(self, action):
            assert action == "click('delete')"
            live.clear()
            live.update(copy.deepcopy(next_data))
            return {}, 0.0, False, False, {}

        def close(self):
            pass

    class Comparison:
        def __init__(self, target, current):
            self.target, self.current = target, current

        def compare(self):
            same = self.target == self.current
            # Model upstream normalizers' nested mutation without corrupting
            # the replay source or the current live app data.
            self.target["todo"].clear()
            self.current["todo"].clear()
            return same

    for name in ("open_apps", "open_apps.tasks", "open_apps.tasks.tasks"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys.modules["open_apps.tasks.tasks"], "AppStateComparison", Comparison, raising=False
    )
    monkeypatch.setattr(oa, "_get_app_state", lambda _: copy.deepcopy(live))
    monkeypatch.setattr(
        oa.OpenAppsEnvironment, "_build_observation", lambda *a, **k: Observation(prompt="Calendar")
    )
    environments = []

    def build(*, scope=None, task=None, names=(TASK,)):
        options = {} if scope is None else {"reward_scope": scope}
        env = oa.OpenAppsEnvironment(
            task_names=names,
            task_factory=lambda _: (task or RemoveDeadline(), Browser()),
            base_url="http://unused.invalid",
            max_steps=45,
            **options,
        )
        environments.append(env)
        return env, next_data

    yield build
    for env in environments:
        env.close()


@pytest.mark.parametrize("scope", [None, "native", "task_local"])
@pytest.mark.parametrize(
    "case", ["completed", "completed_other_app_changed", "incomplete", "wrong_event"]
)
def test_actual_step_obeys_explicit_scope_and_preserves_source(build_environment, scope, case):
    env, current = build_environment(scope=scope)
    if case.startswith("completed"):
        current["calendar"] = [{"title": "Keep"}]
    elif case == "wrong_event":
        current["calendar"] = [{"title": "Deadline"}]
    if case == "completed_other_app_changed":
        current["map"] = []
    state, _ = env.reset()
    original = copy.deepcopy(state)
    result = env.step(state, Action(text="click('delete')"))
    expected = case == "completed" or (
        case == "completed_other_app_changed" and scope == "task_local"
    )
    assert result.rewards.by_name("task_completion").reward == float(expected)
    assert result.terminated is expected
    assert result.next_state.metadata.is_terminal is expected
    assert result.truncated is False
    assert state == original
    if scope == "task_local":
        for info in (result.info, result.next_state.metadata.info):
            assert info["reward_scope"] == "task_local"
            assert info["reward_relevant_apps"] == ("calendar",)
            assert info["task_complete"] is expected
    else:
        # Native opt-in and the default are observationally identical; do not
        # alter existing serialized states just to add new annotations.
        assert "reward_scope" not in result.next_state.metadata.info


@pytest.mark.parametrize("scope", ["guess", "full_state", True])
def test_invalid_scope_rejected_before_browser_creation(scope):
    factory = Mock(side_effect=AssertionError("must not start browser"))
    with pytest.raises(ValueError, match="(?i)scope"):
        oa.OpenAppsEnvironment((TASK,), factory, "http://unused.invalid", reward_scope=scope)
    factory.assert_not_called()


def test_unknown_task_does_not_silently_fall_back_to_native_reward(build_environment):
    with pytest.raises(ValueError, match="(?i)(task|scope)"):
        build_environment(scope="task_local", names=("unknown-task",))


def test_missing_relevant_app_does_not_silently_fall_back(build_environment):
    env, current = build_environment(scope="task_local")
    current.pop("calendar")
    state, _ = env.reset()
    with pytest.raises(ValueError, match="(?i)(calendar|missing|app)"):
        env.step(state, Action(text="click('delete')"))


def test_comparator_exceptions_propagate(build_environment, monkeypatch):
    env, _ = build_environment(scope="task_local")
    state, _ = env.reset()
    comparison = sys.modules["open_apps.tasks.tasks"].AppStateComparison
    monkeypatch.setattr(comparison, "compare", Mock(side_effect=RuntimeError("comparison failed")))
    with pytest.raises(RuntimeError, match="comparison failed"):
        env.step(state, Action(text="click('delete')"))


def test_mask_uses_target_not_initial_values_for_irrelevant_apps(build_environment):
    class NormalizedTarget(RemoveDeadline):
        def get_target_state(self, initial):
            target = super().get_target_state(initial)
            target["todo"] = []
            return target

    env, current = build_environment(scope="task_local", task=NormalizedTarget())
    current["calendar"] = [{"title": "Keep"}]
    state, _ = env.reset()
    assert env.step(state, Action(text="click('delete')")).terminated is True
