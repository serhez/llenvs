"""Reference annotations must not spend an action or alter the actor's state."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from llenvs.adapters.alfworld import AlfWorldEnvironment
from llenvs.adapters import alfworld as adapter_module
from llenvs.core.state import Action
from tests.test_alfworld import MockAlfWorldGymEnv, _install_fake_textworld_modules


@pytest.mark.parametrize("expert_type", ["planner", "handcoded"])
def test_requested_expert_type_reaches_the_real_constructor_parameter(monkeypatch, expert_type):
    calls, _, _ = _install_fake_textworld_modules(monkeypatch)

    class Expert:
        # Match upstream: its first positional argument is NOT expert_type.
        def __init__(self, env=None, expert_type="handcoded"):
            self.env = env
            self.expert_type = expert_type

    module = sys.modules["alfworld.agents.environment.alfred_tw_env"]
    monkeypatch.setattr(module, "AlfredExpert", Expert, raising=False)
    env = AlfWorldEnvironment(
        game_files=("pick_and_place_simple-Mug-001/game.tw",),
        config={"env": {"expert_type": expert_type}},
        expert_plan=True,
    )
    try:
        env.reset()
        expert = next(w for w in calls[0]["wrappers"] if isinstance(w, Expert))
        assert expert.expert_type == expert_type
        assert expert.env is None
    finally:
        env.close()


class _PlannedGame(MockAlfWorldGymEnv):
    def __init__(self, game_file):
        super().__init__(game_file)
        self.actions = []

    def reset(self):
        observations, infos = super().reset()
        infos["extra.expert_plan"] = [["go to shelf 1", "take mug 1 from shelf 1"]]
        return observations, infos

    def step(self, action):
        self.actions.append(action)
        observations, rewards, done, infos = super().step(action)
        infos["extra.expert_plan"] = [["take mug 1 from shelf 1"]]
        return observations, rewards, done, infos


@pytest.fixture
def planned_env(monkeypatch):
    _, _, instances = _install_fake_textworld_modules(monkeypatch, env_factory=_PlannedGame)
    module = sys.modules["alfworld.agents.environment.alfred_tw_env"]
    # This fixture tests refresh separately from the constructor regression.
    monkeypatch.setattr(module, "AlfredExpert", lambda *args, **kwargs: object(), raising=False)
    env = AlfWorldEnvironment(
        game_files=("pick_and_place_simple-Mug-001/game.tw",),
        config={"env": {"expert_type": "planner"}},
        expert_plan=True,
        expose_expert_plan_in_obs=True,
        max_steps=40,
    )
    yield env, instances
    env.close()


def _without_expert(state):
    text = state.observation.state.text
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith("[expert_plan_next:")
    )
    return replace(
        state,
        hidden=replace(state.hidden, expert_plan=None),
        observation=replace(state.observation, state=replace(state.observation.state, text=text)),
    )


@pytest.mark.parametrize("prefix", [(), ("go to shelf 1",)])
def test_refresh_replays_only_the_prefix_and_preserves_unannotated_input(planned_env, prefix):
    env, instances = planned_env
    state, _ = env.reset()
    for action in prefix:
        state = env.step(state, Action(text=action)).next_state
    original = _without_expert(state)
    game = next(iter(instances.values()))
    game.actions.clear()

    refreshed = env.refresh_expert_plan(original)

    assert game.actions == list(prefix), (
        "Refreshing must not execute look or the next expert action"
    )
    assert refreshed.metadata == original.metadata
    assert replace(refreshed.hidden, expert_plan=None) == original.hidden
    assert original.hidden.expert_plan is None
    assert "[expert_plan_next:" not in original.observation.state.text
    expected = "take mug 1 from shelf 1" if prefix else "go to shelf 1"
    assert refreshed.hidden.expert_plan[0] == expected
    assert f"[expert_plan_next: {expected}]" in refreshed.observation.state.text
    assert _without_expert(refreshed) == original
    game.actions.clear()
    assert env.refresh_expert_plan(original) == refreshed
    assert game.actions == list(prefix)


@pytest.mark.parametrize("terminal,step", [(True, 2), (False, 40)])
def test_refresh_does_not_consult_planner_after_episode_ends(
    planned_env, monkeypatch, terminal, step
):
    env, _ = planned_env
    state, _ = env.reset()
    state = replace(
        state,
        hidden=replace(state.hidden, episode_step=step),
        metadata=replace(state.metadata, step=step, is_terminal=terminal),
    )
    monkeypatch.setattr(
        env, "_init_game", lambda *_: pytest.fail("Ended episodes must not be replayed")
    )
    assert env.refresh_expert_plan(state) == state


def test_deferred_planner_does_not_request_search_during_load_or_replay(monkeypatch):
    _install_fake_textworld_modules(monkeypatch)
    module = sys.modules["alfworld.agents.environment.alfred_tw_env"]

    class NativeExpert:
        def __init__(self, env=None, expert_type="handcoded"):
            self.expert_type = expert_type
            self.request_infos = SimpleNamespace(policy_commands=False)
            self.state = {}

        def load(self, filename):
            self.request_infos.policy_commands = True

    monkeypatch.setattr(module, "AlfredExpert", NativeExpert, raising=False)
    expert = adapter_module._deferred_planner_expert()
    expert.load("game")
    expert._gather_infos()
    assert expert.expert_type == "planner"
    assert expert.request_infos.policy_commands is False
    assert expert.state["extra.expert_plan"] == []
