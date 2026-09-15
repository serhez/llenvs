"""Contracts for opt-in recovery of incomplete browser observations.

No real browser, network or dataset access. Model the captured head-only DOM:
the screenshot exists, but there are no visible accessibility-tree nodes.
"""

import copy
import threading
from types import SimpleNamespace

import pytest

from llenvs.adapters import open_apps as oa


def observation(kind="complete", *, screenshot=True):
    properties = {
        "1": {"visibility": 0.0, "bbox": None, "clickable": False, "set_of_marks": False},
        "2": {"visibility": 1.0, "bbox": [0, 0, 50, 20], "clickable": True, "set_of_marks": True},
    }
    nodes = [
        {"nodeId": "0", "role": {"value": "RootWebArea"}, "childIds": ["1"]},
        {
            "nodeId": "1",
            "browsergym_id": "2",
            "role": {"value": "button"},
            "name": {"value": "Calendar"},
        },
    ]
    if kind == "head_only":
        nodes = nodes[:1]
        properties.pop("2")
    elif kind == "missing_properties":
        properties = {}
    elif kind == "invisible":
        properties["2"]["visibility"] = 0.0
    elif kind == "noninteractive":
        nodes[1]["role"]["value"] = "heading"
        properties["2"].update(clickable=False, set_of_marks=False)
    raw = {
        "url": "http://localhost/calendar",
        "last_action": "click('30')",
        "last_action_error": "",
        "axtree_object": {"nodes": nodes},
        "extra_element_properties": properties,
    }
    if screenshot:
        raw["screenshot"] = "unchanged screenshot sentinel"
    return raw


class Browser:
    def __init__(self, raw, refreshed=()):
        self.raw = raw
        self.refreshed = list(refreshed)
        self.events = []
        self.page = SimpleNamespace(wait_for_load_state=self.wait)
        self.unwrapped = self
        self.info = {"opaque": object()}
        self.reset_error = self.step_error = self.refresh_error = self.wait_error = None

    def record(self, operation, *args):
        self.events.append((operation, threading.get_ident(), args))

    def reset(self, **kwargs):
        self.record("reset", kwargs)
        if self.reset_error:
            raise self.reset_error
        return self.raw, self.info

    def step(self, action):
        self.record("step", action)
        if self.step_error:
            raise self.step_error
        return self.raw, 0.75, True, False, self.info

    def wait(self, state, *, timeout):
        self.record("wait", state, timeout)
        assert state == "domcontentloaded" and 0 < timeout <= 10000
        if self.wait_error:
            raise self.wait_error

    def _get_obs(self):
        self.record("observe")
        if self.refresh_error:
            raise self.refresh_error
        return self.refreshed.pop(0) if self.refreshed else self.raw

    def close(self):
        self.record("close")


@pytest.fixture
def factory(monkeypatch):
    monkeypatch.setattr(oa, "_patch_browsergym_thread_local_pw", lambda: None)
    proxies = []

    def create(browser, **kwargs):
        proxy = oa._BrowserGymProxy(lambda: browser, **kwargs)
        proxies.append(proxy)
        return proxy

    yield create
    for proxy in proxies:
        proxy.close()
        assert not proxy._thread.is_alive()


@pytest.mark.parametrize("method", ["reset", "step"])
def test_native_default_does_not_reobserve_or_change_returned_data(factory, method):
    browser = Browser(observation("head_only"), [observation()])
    proxy = factory(browser)
    result = proxy.reset() if method == "reset" else proxy.step("click('30')")
    assert result[0] is browser.raw and result[-1] is browser.info
    assert [event[0] for event in browser.events] == [method]


@pytest.mark.parametrize("method", ["reset", "step"])
@pytest.mark.parametrize("kind", ["head_only", "missing_properties", "invisible"])
def test_recovery_rereads_observation_without_repeating_action_or_reset(factory, method, kind):
    raw, fresh = observation(kind), observation()
    before = copy.deepcopy(raw)
    browser = Browser(raw, [fresh])
    proxy = factory(browser, recover_observation=True)
    result = proxy.reset(seed=42) if method == "reset" else proxy.step("click('30')")
    assert result[0] is fresh and result[-1] is browser.info
    if method == "step":
        assert result[1:4] == (0.75, True, False)
        assert browser.events[0][2] == ("click('30')",)
    else:
        assert browser.events[0][2] == ({"seed": 42},)
    assert raw == before  # Do not patch the original response into looking valid.
    assert [event[0] for event in browser.events] == [method, "wait", "observe"]
    assert {event[1] for event in browser.events} == {proxy._thread.ident}
    assert proxy._thread.ident != threading.get_ident()


@pytest.mark.parametrize(
    "kind,screenshot", [("complete", True), ("complete", False), ("noninteractive", True)]
)
def test_valid_observation_needs_no_extra_read_or_wait(factory, kind, screenshot):
    browser = Browser(observation(kind, screenshot=screenshot))
    proxy = factory(browser, recover_observation=True)
    result = proxy.step("click('30')")
    assert result[0] is browser.raw
    assert [event[0] for event in browser.events] == ["step"]


def test_recovery_can_wait_for_a_second_observation_without_repeating_action(factory):
    fresh = observation()
    browser = Browser(observation("head_only"), [observation("head_only"), fresh])
    proxy = factory(browser, recover_observation=True)
    assert proxy.step("click('30')")[0] is fresh
    assert [event[0] for event in browser.events] == ["step", "wait", "observe", "wait", "observe"]


def test_persistent_incomplete_observation_stops_after_three_reads_and_invalidates_proxy(factory):
    browser = Browser(observation("head_only"))
    proxy = factory(browser, recover_observation=True)
    with pytest.raises(RuntimeError, match="(?i)observation"):
        proxy.step("click('30')")
    events = list(browser.events)
    assert [event[0] for event in events].count("step") == 1
    assert [event[0] for event in events].count("observe") == 3
    with pytest.raises(RuntimeError, match="(?i)(unusable|observation)"):
        proxy.step("must not run")
    assert browser.events == events


@pytest.mark.parametrize("method", ["reset", "step"])
def test_operation_errors_are_not_retried_as_observation_failures(factory, method):
    browser = Browser(observation("head_only"), [observation()])
    setattr(browser, method + "_error", ValueError("original operation failed"))
    proxy = factory(browser, recover_observation=True)
    with pytest.raises(ValueError, match="original operation"):
        proxy.reset() if method == "reset" else proxy.step("click('30')")
    assert [event[0] for event in browser.events] == [method]


def test_unexpected_refresh_error_is_not_silently_swallowed(factory):
    browser = Browser(observation("head_only"))
    browser.refresh_error = ValueError("browser closed")
    proxy = factory(browser, recover_observation=True)
    with pytest.raises(ValueError, match="browser closed"):
        proxy.step("click('30')")
    assert [event[0] for event in browser.events] == ["step", "wait", "observe"]
    with pytest.raises(RuntimeError, match="(?i)(unusable|observation)"):
        proxy.step("must not run")


@pytest.mark.parametrize("error", [TimeoutError("page still loading"), ValueError("page closed")])
def test_load_wait_errors_stop_instead_of_capturing_an_unready_page(factory, error):
    browser = Browser(observation("head_only"), [observation()])
    browser.wait_error = error
    proxy = factory(browser, recover_observation=True)
    with pytest.raises(type(error), match=str(error)):
        proxy.step("click('30')")
    assert [event[0] for event in browser.events] == ["step", "wait"]
    with pytest.raises(RuntimeError, match="(?i)(unusable|observation)"):
        proxy.step("must not run")


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_invalid_recovery_option_rejected_before_starting_browser(monkeypatch, value):
    def forbidden_thread(*args, **kwargs):
        raise AssertionError("Do not start a browser for an invalid option")

    monkeypatch.setattr(oa.threading, "Thread", forbidden_thread)
    with pytest.raises(ValueError, match="recover_observation"):
        oa._BrowserGymProxy(lambda: None, recover_observation=value)
