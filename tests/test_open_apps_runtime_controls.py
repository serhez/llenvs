"""Offline contracts for opt-in OpenApps replay controls.

No real browser, app server, dataset, model, or network connection is used.
"""

import base64
import io
import queue
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from llenvs.adapters import open_apps as oa


class FakeBrowserEnv:
    def reset(self):
        return {"dom_txt": "Calendar"}, {}

    def step(self, action):
        return action

    def close(self):
        pass


@pytest.fixture
def proxy_factory(monkeypatch):
    monkeypatch.setattr(oa, "_patch_browsergym_thread_local_pw", lambda: None)
    proxies = []

    def create(**kwargs):
        proxy = oa._BrowserGymProxy(FakeBrowserEnv, **kwargs)
        proxies.append(proxy)
        return proxy

    yield create
    for proxy in proxies:
        proxy.close()
        proxy._thread.join(timeout=1)
        assert not proxy._thread.is_alive()


@pytest.mark.parametrize(
    "options, expected", [({}, 60), ({"call_timeout": 180}, 180), ({"call_timeout": 2.5}, 2.5)]
)
def test_proxy_uses_configured_queue_timeout(proxy_factory, monkeypatch, options, expected):
    proxy = proxy_factory(**options)
    receive = Mock(wraps=proxy._res_q.get)
    with monkeypatch.context() as patch:
        patch.setattr(proxy._res_q, "get", receive)
        assert proxy.step("click('12')") == "click('12')"
    assert receive.call_args.kwargs["timeout"] == expected


def test_timeout_override_is_per_proxy_not_global(proxy_factory, monkeypatch):
    custom = proxy_factory(call_timeout=180)
    native = proxy_factory()
    for proxy, expected in ((custom, 180), (native, 60), (custom, 180)):
        receive = Mock(wraps=proxy._res_q.get)
        with monkeypatch.context() as patch:
            patch.setattr(proxy._res_q, "get", receive)
            assert proxy.step("next") == "next"
        assert receive.call_args.kwargs["timeout"] == expected


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "180"])
def test_invalid_timeout_rejected_before_thread_start(monkeypatch, value):
    thread = Mock(side_effect=AssertionError("worker must not start"))
    monkeypatch.setattr(oa.threading, "Thread", thread)
    with pytest.raises(ValueError, match="timeout"):
        oa._BrowserGymProxy(FakeBrowserEnv, call_timeout=value)
    thread.assert_not_called()


def test_timeout_invalidates_proxy_instead_of_reusing_late_response(proxy_factory, monkeypatch):
    proxy = proxy_factory()
    # The second response belongs to the FIRST operation, not to the next one.
    receive = Mock(side_effect=[queue.Empty, ("ok", "late first response")])
    send = Mock(wraps=proxy._cmd_q.put)
    with monkeypatch.context() as patch:
        patch.setattr(proxy._res_q, "get", receive)
        patch.setattr(proxy._cmd_q, "put", send)
        with pytest.raises(TimeoutError):
            proxy.step("first")
        with pytest.raises(RuntimeError, match="(?i)(timeout|timed.out|unusable|invalid)"):
            proxy.step("second")
        assert send.call_count == 1
        assert receive.call_count == 1


def test_returned_action_exception_is_not_a_queue_timeout(proxy_factory, monkeypatch):
    proxy = proxy_factory()
    receive = Mock(side_effect=[("err", ValueError("bad action")), ("ok", "next")])
    with monkeypatch.context() as patch:
        patch.setattr(proxy._res_q, "get", receive)
        with pytest.raises(ValueError, match="bad action"):
            proxy.step("first")
        assert proxy.step("second") == "next"


@pytest.fixture
def fake_adapter(tmp_path, monkeypatch):
    """Exercise get_environment's real plumbing without importing OpenApps."""
    for name in (
        "open_apps",
        "open_apps.tasks",
        "open_apps.tasks.tasks",
        "open_apps.tasks.add_tasks_to_browsergym",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    tasks = sys.modules["open_apps.tasks.tasks"]
    for name in (
        "AddEventTask",
        "AddToDoTask",
        "MarkToDoDoneTask",
        "RemoveEventTask",
        "SavePlaceTask",
        "SendMessageTask",
        "Task",
    ):
        setattr(tasks, name, type(name, (), {}))
    registration = sys.modules["open_apps.tasks.add_tasks_to_browsergym"]
    registration.register_tasks_with_browsergym = Mock()

    config = tmp_path / "config/tasks/all_tasks.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("first: {}\nsecond: {}\n")
    import gymnasium
    import yaml

    # Configuration loading and task construction are external boundaries here;
    # these plumbing tests must also run without the optional OpenApps extras.
    hydra = ModuleType("hydra")
    hydra.__path__ = []
    hydra.utils = ModuleType("hydra.utils")
    hydra.utils.instantiate = lambda *a, **kw: SimpleNamespace(
        task_id="fake", goal="Calendar"
    )
    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace(load=yaml.safe_load)
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra.utils)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    make = Mock(return_value=FakeBrowserEnv())
    monkeypatch.setattr(gymnasium, "make", make)
    proxy = Mock(side_effect=lambda factory, **kw: factory())
    monkeypatch.setattr(oa, "_BrowserGymProxy", proxy)
    monkeypatch.setattr(oa, "_get_app_state", lambda url: {})
    adapter = oa.OpenAppsAdapter(open_apps_path=str(tmp_path))
    monkeypatch.setattr(adapter, "_get_browsergym", lambda: None)
    return adapter, make, proxy


def test_adapter_consumes_controls_for_initial_and_lazy_tasks(fake_adapter):
    adapter, make, proxy = fake_adapter
    env = adapter.get_environment(
        "first",
        task_names=("first", "second"),
        base_url="http://unused.invalid",
        browsergym_call_timeout=180,
        browser_scale_factor=1,
        viewport={"width": 1280, "height": 720},
        timeout=1234,
    )
    try:
        env.reset(options={"task_index": 1})
        assert make.call_count == proxy.call_count == 2
        for call in make.call_args_list:
            assert call.kwargs == {
                "task_kwargs": {"base_url": "http://unused.invalid"},
                "headless": True,
                "viewport": {"width": 1280, "height": 720},
                "timeout": 1234,
            }
        for call in proxy.call_args_list:
            assert call.kwargs["call_timeout"] == 180
            assert call.kwargs["browser_scale_factor"] == 1
    finally:
        env.close()


def test_reward_scope_is_consumed_by_adapter_not_forwarded_to_browser(fake_adapter):
    adapter, make, _ = fake_adapter
    env = adapter.get_environment(
        "first",
        task_names=("first", "second"),
        base_url="http://unused.invalid",
        reward_scope="native",
    )
    try:
        env.reset(options={"task_index": 1})
        assert env._reward_scope == "native"
        assert len(make.call_args_list) == 2
        assert all("reward_scope" not in call.kwargs for call in make.call_args_list)
    finally:
        env.close()


def test_observation_recovery_option_is_consumed_by_adapter_for_initial_and_lazy_tasks(
    fake_adapter,
):
    adapter, make, proxy = fake_adapter
    env = adapter.get_environment(
        "first",
        task_names=("first", "second"),
        base_url="http://unused.invalid",
        recover_observation=True,
    )
    try:
        env.reset(options={"task_index": 1})
        assert proxy.call_count == make.call_count == 2
        assert all(call.kwargs["recover_observation"] is True for call in proxy.call_args_list)
        assert all("recover_observation" not in call.kwargs for call in make.call_args_list)
    finally:
        env.close()


def test_invalid_recovery_option_stops_before_runtime_import_or_start(monkeypatch):
    adapter = oa.OpenAppsAdapter(open_apps_path="/unused")
    start = Mock(side_effect=AssertionError("must not import/start runtime"))
    monkeypatch.setattr(adapter, "_get_open_apps", start)
    with pytest.raises(ValueError, match="recover_observation"):
        adapter.get_environment("remove_wacv_abstract_deadline", recover_observation="yes")
    start.assert_not_called()


def test_invalid_reward_scope_stops_before_runtime_import_or_start(monkeypatch):
    adapter = oa.OpenAppsAdapter(open_apps_path="/not-used")
    load = Mock(side_effect=AssertionError("runtime must not load"))
    monkeypatch.setattr(adapter, "_get_open_apps", load)
    with pytest.raises(ValueError, match="(?i)scope"):
        adapter.get_environment("first", reward_scope="guess")
    load.assert_not_called()


def test_external_server_cannot_silently_ignore_reference_time(fake_adapter):
    adapter, make, proxy = fake_adapter
    with pytest.raises(ValueError, match="(?i)(reference.time|base_url|external)"):
        adapter.get_environment(
            "first",
            task_names=("first",),
            base_url="http://unused.invalid",
            reference_time="2001-02-03T04:05:06+02:00",
        )
    make.assert_not_called()
    proxy.assert_not_called()


def test_managed_server_gets_reference_time_but_browser_does_not(fake_adapter, monkeypatch):
    adapter, make, _ = fake_adapter
    server = SimpleNamespace(url="http://unused.invalid", stop=Mock())
    ensure = Mock(return_value=server)
    monkeypatch.setattr(adapter, "_ensure_server", ensure)
    timestamp = "2001-02-03T04:05:06+02:00"
    env = adapter.get_environment("first", task_names=("first",), reference_time=timestamp)
    try:
        assert ensure.call_args.kwargs["reference_time"] == timestamp
        assert "reference_time" not in make.call_args.kwargs
    finally:
        env.close()


@pytest.mark.parametrize("other_time", [None, "2001-03-03T04:05:06+02:00"])
def test_live_server_reuse_rejects_clock_change_without_stopping_it(
    tmp_path, monkeypatch, other_time
):
    server_type = oa._OpenAppsServer
    monkeypatch.setattr(server_type, "_pick_port", staticmethod(lambda start: 5001))
    monkeypatch.setattr(server_type, "is_running", property(lambda self: True))
    start, stop = Mock(), Mock()
    monkeypatch.setattr(server_type, "start", start)
    monkeypatch.setattr(server_type, "stop", stop)
    build = Mock(wraps=server_type)
    monkeypatch.setattr(oa, "_OpenAppsServer", build)
    adapter = oa.OpenAppsAdapter(open_apps_path=str(tmp_path))
    timestamp = "2001-02-03T04:05:06+02:00"
    try:
        first = adapter._ensure_server(reference_time=timestamp)
        assert adapter._ensure_server(reference_time=timestamp) is first
        with pytest.raises(ValueError, match="(?i)(clock|reference.time)"):
            adapter._ensure_server(reference_time=other_time)
        assert adapter._server is first
        assert build.call_count == start.call_count == 1
        stop.assert_not_called()
    finally:
        adapter.stop_server()


@pytest.fixture
def browser_proxy_factory(monkeypatch):
    """Fake only Playwright's launch boundary, retaining owner-thread plumbing."""
    import browsergym.core
    import browsergym.core.chat
    import browsergym.core.env
    import playwright.sync_api

    # Register restoration before the production patch changes these bindings.
    for module in (browsergym.core, browsergym.core.env, browsergym.core.chat):
        monkeypatch.setattr(module, "_get_global_playwright", module._get_global_playwright)
    monkeypatch.setattr(oa, "_pw_patched", False)
    launches = []

    def launch(**kwargs):
        launches.append(deepcopy(kwargs))
        return kwargs

    monkeypatch.setattr(
        playwright.sync_api,
        "sync_playwright",
        lambda: SimpleNamespace(
            start=lambda: SimpleNamespace(chromium=SimpleNamespace(launch=launch))
        ),
    )
    proxies = []

    def create(*, browser_scale_factor=None, args=None):
        launch_args = ["--window-size=1280,720"] if args is None else args

        def factory():
            pw = browsergym.core.env._get_global_playwright()
            # BrowserGym already binds args; adding it again through kwargs fails.
            browser = pw.chromium.launch(args=launch_args, headless=True)
            env = FakeBrowserEnv()
            env.launch_options = browser
            return env

        options = (
            {} if browser_scale_factor is None else {"browser_scale_factor": browser_scale_factor}
        )
        proxy = oa._BrowserGymProxy(factory, **options)
        proxies.append(proxy)
        return proxy

    yield create, launches
    for proxy in proxies:
        proxy.close()
        proxy._thread.join(timeout=1)
        assert not proxy._thread.is_alive()


def test_browser_scale_is_opt_in_and_does_not_leak_between_owner_threads(browser_proxy_factory):
    create, launches = browser_proxy_factory
    original_args = ["--window-size=1280,720", "--disable-dev-shm-usage"]
    create(browser_scale_factor=1, args=original_args)
    create(args=original_args)
    create(browser_scale_factor=2, args=original_args)
    assert launches[0]["args"] == original_args + ["--force-device-scale-factor=1"]
    assert launches[1] == {"args": original_args, "headless": True}
    assert launches[2]["args"] == original_args + ["--force-device-scale-factor=2"]
    assert original_args == ["--window-size=1280,720", "--disable-dev-shm-usage"]


def test_native_browser_launch_is_unchanged_without_scale_override(browser_proxy_factory):
    create, launches = browser_proxy_factory
    create()
    assert launches == [{"args": ["--window-size=1280,720"], "headless": True}]


def test_scale_is_applied_before_screenshot_and_som(browser_proxy_factory, monkeypatch):
    import browsergym.utils.obs
    import numpy as np
    from PIL import Image

    create, launches = browser_proxy_factory
    create(browser_scale_factor=1)
    # Model the independently observed CDP behavior: native Mac capture is 2x.
    scale = 1 if "--force-device-scale-factor=1" in launches[0]["args"] else 2
    pixels = np.zeros((720 * scale, 1280 * scale, 3), dtype=np.uint8)
    pixels[50 * scale, 100 * scale] = [255, 0, 0]
    saved_pixels = pixels.copy()
    props = {"12": {"bbox": [100, 50, 10, 10], "set_of_marks": True}}
    saved_props = deepcopy(props)

    def overlay(image, extra):
        assert image.shape == (720, 1280, 3)
        assert image[50, 100].tolist() == [255, 0, 0]
        assert extra == saved_props
        result = image.copy()
        result[50, 100] = [0, 255, 0]
        return result

    monkeypatch.setattr(browsergym.utils.obs, "overlay_som", overlay)
    env = oa.OpenAppsEnvironment(
        task_names=("fake",),
        task_factory=lambda name: (SimpleNamespace(goal="Calendar"), FakeBrowserEnv()),
        base_url="http://unused.invalid",
        use_screenshot=True,
        screenshot_with_som=True,
    )
    try:
        obs = env._build_observation(
            {"dom_txt": "Calendar", "screenshot": pixels, "extra_element_properties": props},
            "Calendar",
        )
        image = Image.open(io.BytesIO(base64.b64decode(obs.state.images[0].data)))
        assert image.size == (1280, 720)
        assert image.getpixel((100, 50)) == (0, 255, 0)
        assert np.array_equal(pixels, saved_pixels)
        assert props == saved_props
    finally:
        env.close()
