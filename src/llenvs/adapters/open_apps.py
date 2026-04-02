"""OpenApps adapter – wraps Meta's OpenApps UI-agent benchmark.

OpenApps provides synthetic web applications (calendar, todo, messenger,
map, code-editor, online-shop) served via FastHTML. Agents interact through
BrowserGym which exposes accessibility-tree observations and accepts
Playwright-style browser actions (click, type, scroll).

Reference: https://github.com/facebookresearch/OpenApps
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)


# ---------------------------------------------------------------------------
# Task names shipped with OpenApps
# ---------------------------------------------------------------------------

OPEN_APPS_TASKS: list[str] = [
    "add_meeting_with_dennis",
    "add_christmas_shopping_event",
    "add_paper_reading_meeting_with_einstein",
    "remove_wacv_abstract_deadline",
    "add_call_mom_to_my_todo",
    "mark_water_plants_as_done",
    "message_bob_to_meet",
    "save_paris_to_my_favorite_places",
]

# Available application modules in OpenApps
OPEN_APPS_MODULES: list[str] = [
    "calendar",
    "todo",
    "messenger",
    "map",
    "codeeditor",
    "onlineshop",
]


# ---------------------------------------------------------------------------
# Hidden state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAppsHidden:
    """Hidden state for OpenApps environments.

    Attributes:
        task_name: Name of the task being executed.
        goal: Natural-language goal string shown to the agent.
        task_index: Index for sequencing resets.
        episode_step: Current step within the episode.
        initial_app_state: Snapshot of app state at episode start (for reward).
        last_action: Previous browser action text.
        base_url: URL of the running OpenApps web server.
    """

    task_name: str
    goal: str
    task_index: int
    episode_step: int
    initial_app_state: dict[str, Any] = field(default_factory=dict)
    last_action: str | None = None
    base_url: str = ""
    trajectory: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


@dataclass
class OpenAppsReward:
    """Reward based on OpenApps task completion.

    Uses the Task's ``check_if_task_is_complete`` method which compares
    the initial application state against the current state to determine
    whether the agent has achieved the goal.
    """

    _name: str = "task_completion"
    _reward_type: RewardType = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[OpenAppsHidden],
        action: Action,
        next_state: State[OpenAppsHidden],
    ) -> Signal:
        """Compute reward from the info dict populated during step."""
        reward = next_state.metadata.info.get("open_apps_reward", 0.0)
        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=float(reward),
            metadata={"source": "open_apps"},
        )


# ---------------------------------------------------------------------------
# Helper: fetch app state via HTTP
# ---------------------------------------------------------------------------


def _get_app_state(base_url: str) -> dict[str, Any]:
    """Fetch the current application state from the running OpenApps server.

    Returns a dict with keys: todo, calendar, map, messenger, codeeditor,
    online_shop.
    """
    import requests

    state: dict[str, Any] = {}
    endpoints = {
        "todo": "/todo_all",
        "calendar": "/calendar_all",
        "map": "/maps/landmarks",
        "messenger": "/messages_all",
        "codeeditor": "/codeeditor_all",
    }
    for key, path in endpoints.items():
        try:
            resp = requests.get(base_url + path, timeout=10)
            resp.raise_for_status()
            state[key] = resp.json()
        except Exception:
            state[key] = []

    try:
        resp = requests.get(base_url + "/onlineshop_all", timeout=10)
        resp.raise_for_status()
        state["online_shop"] = resp.json()
    except Exception:
        state["online_shop"] = []

    return state


# ---------------------------------------------------------------------------
# Thread-safe BrowserGym proxy
# ---------------------------------------------------------------------------

_pw_patch_lock = threading.Lock()
_pw_patched = False


def _patch_browsergym_thread_local_pw() -> None:
    """Monkey-patch BrowserGym to use thread-local Playwright instances.

    BrowserGym stores a single module-level ``_PLAYWRIGHT`` global.  When
    multiple threads each own a BrowserGym env, they all share that global
    and Playwright's greenlet-based sync API crashes on cross-thread access.

    This replaces ``_get_global_playwright`` with a version backed by
    ``threading.local()`` so every thread that creates a BrowserGym env
    gets its own Playwright connection.  The patch is applied once and is
    safe to call from any thread.
    """
    global _pw_patched
    with _pw_patch_lock:
        if _pw_patched:
            return
        _pw_patched = True

    import browsergym.core
    import browsergym.core.chat
    import browsergym.core.env

    _tls = threading.local()

    def _thread_local_get_pw() -> Any:
        pw = getattr(_tls, "pw", None)
        if pw is None:
            import playwright.sync_api

            pw = playwright.sync_api.sync_playwright().start()
            _tls.pw = pw
        return pw

    # Patch every module that imports the accessor so their local
    # reference also points to the thread-local version.
    browsergym.core._get_global_playwright = _thread_local_get_pw
    browsergym.core.env._get_global_playwright = _thread_local_get_pw
    browsergym.core.chat._get_global_playwright = _thread_local_get_pw


class _BrowserGymProxy:
    """Proxy that pins a BrowserGym env to a dedicated owner thread.

    Playwright's sync API uses greenlets that are bound to the OS thread
    that created the browser.  The value-bench runner dispatches
    ``env.step()`` from a ``ThreadPoolExecutor``, which would violate this
    constraint.

    Each proxy spawns **its own daemon thread**, creates the env on that
    thread (including a thread-local Playwright instance), and forwards
    ``reset`` / ``step`` / ``close`` calls through a command queue.
    Multiple proxies can run in parallel — each on its own thread with its
    own Playwright — giving true concurrent browser execution.
    """

    _CALL_TIMEOUT = 300  # seconds per operation

    def __init__(self, env_factory: Any) -> None:
        self._cmd_q: queue.Queue = queue.Queue()
        self._res_q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(env_factory,), daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=180):
            raise RuntimeError("BrowserGym proxy thread failed to start in 180s")
        if self._error is not None:
            raise RuntimeError(
                f"BrowserGym proxy thread died during startup: {self._error}"
            )

    # -- internal event loop (runs on the owner thread) --------------------

    def _run(self, env_factory: Any) -> None:
        import asyncio
        import logging

        logger = logging.getLogger("llenvs.adapters.open_apps.proxy")

        # Each thread needs its own event loop for Playwright internals.
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # Ensure BrowserGym uses thread-local Playwright instances.
        _patch_browsergym_thread_local_pw()

        try:
            env = env_factory()
        except Exception as exc:
            logger.error("Proxy thread: env_factory failed: %s", exc)
            self._error = exc
            self._ready.set()
            return

        self._ready.set()

        try:
            while True:
                cmd, args, kwargs = self._cmd_q.get()
                if cmd == "__stop__":
                    self._res_q.put(None)
                    break
                try:
                    result = getattr(env, cmd)(*args, **kwargs)
                    self._res_q.put(("ok", result))
                except Exception as exc:
                    self._res_q.put(("err", exc))
        except Exception as exc:
            logger.error("Proxy thread died: %s", exc)
        finally:
            try:
                env.close()
            except Exception:
                pass

    # -- public forwarding methods (called from any thread) ----------------

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("BrowserGym proxy thread is dead")
        self._cmd_q.put((method, args, kwargs))
        try:
            tag, result = self._res_q.get(timeout=self._CALL_TIMEOUT)
        except queue.Empty:
            raise TimeoutError(
                f"BrowserGym proxy: '{method}' timed out after "
                f"{self._CALL_TIMEOUT}s"
            ) from None
        if tag == "err":
            raise result
        return result

    def reset(self, **kwargs: Any) -> Any:
        return self._call("reset", **kwargs)

    def step(self, action: Any) -> Any:
        return self._call("step", action)

    def close(self) -> None:
        if self._thread.is_alive():
            self._cmd_q.put(("__stop__", (), {}))
            try:
                self._res_q.get(timeout=30)
            except queue.Empty:
                pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class OpenAppsEnvironment:
    """MDP wrapper for OpenApps browser-based tasks.

    OpenApps is a multi-turn environment where agents interact with
    synthetic web applications via BrowserGym.  Observations consist
    of an accessibility-tree text representation (and optionally a
    screenshot).  Actions are BrowserGym action strings such as
    ``click('element_id')``, ``fill('element_id', 'text')``, etc.

    Example::

        >>> adapter = OpenAppsAdapter()
        >>> env = adapter.get_environment("add_call_mom_to_my_todo")
        >>> state, _ = env.reset()
        >>> action = Action(text="click('42')")
        >>> result = env.step(state, action)
    """

    def __init__(
        self,
        browsergym_env: Any,
        task: Any,
        task_name: str,
        base_url: str,
        max_steps: int = 30,
        use_screenshot: bool = False,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        self._env = browsergym_env
        self._task = task
        self._task_name = task_name
        self._base_url = base_url
        self._max_steps = max_steps
        self._use_screenshot = use_screenshot

        self._native_rewards: tuple[RewardFunction, ...] = (OpenAppsReward(),)
        self._extra_rewards = extra_rewards

        self._state_tracker = _StateContinuityTracker()
        self._current_task_index = 0

    # -- Protocol properties ------------------------------------------------

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._task_name,
            adapter="open_apps",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            metadata={
                "base_url": self._base_url,
                "description": f"OpenApps task: {self._task_name}",
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction[OpenAppsHidden], ...]:
        return self._native_rewards + self._extra_rewards

    # -- Helpers ------------------------------------------------------------

    def _build_observation(
        self,
        raw_obs: dict[str, Any],
        goal: str,
        previous: Observation | None = None,
        action: Action | None = None,
    ) -> Observation:
        """Build an Observation from a BrowserGym observation dict.

        The ``prompt`` field is set once at reset (goal + initial axtree)
        and carried forward unchanged so that it stays constant across
        steps, matching the convention used by other multi-turn adapters.
        Dynamic per-step content goes into ``state`` and ``messages``.
        """
        # BrowserGym returns axtree_object (dict), not a string.
        # Use flatten_axtree_to_str to convert to a readable text format.
        from browsergym.utils.obs import flatten_axtree_to_str

        axtree_obj = raw_obs.get("axtree_object")
        extra_props = raw_obs.get("extra_element_properties", {})
        if axtree_obj is not None:
            obs_text = flatten_axtree_to_str(
                axtree_obj,
                extra_properties=extra_props,
                with_clickable=True,
                with_visible=True,
                filter_visible_only=True,
                filter_with_bid_only=True,
            )
        else:
            obs_text = raw_obs.get("dom_txt", "")

        images: tuple[ImageContent, ...] = ()
        if self._use_screenshot and "screenshot" in raw_obs:
            screenshot = raw_obs["screenshot"]
            if screenshot is not None:
                images = (ImageContent(data=screenshot, media_type="image/png"),)

        task_content: ObservationContent
        state_content = ObservationContent(text=obs_text, images=images)

        if previous is None:
            # Initial reset: build prompt and task from scratch
            task_content = ObservationContent(text=f"Goal: {goal}")
            prompt = f"Goal: {goal}\n\n{obs_text}"
            messages: tuple[dict[str, Any], ...] = ()
        else:
            # Subsequent step: carry forward prompt and task from reset
            task_content = previous.task  # type: ignore[assignment]
            prompt = previous.prompt
            messages = tuple(previous.messages) + (
                {"role": "assistant", "content": action.text or "" if action else ""},
                {"role": "user", "content": obs_text},
            )

        return Observation(
            prompt=prompt,
            messages=messages,
            task=task_content,
            state=state_content,
            images=images,
        )

    # -- MDP interface ------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[OpenAppsHidden], dict[str, Any]]:
        """Reset the environment and return the initial state."""
        options = options or {}
        task_index = options.get("task_index", self._current_task_index)
        self._current_task_index = task_index

        raw_obs, info = self._env.reset()

        goal = self._task.goal
        initial_app_state = _get_app_state(self._base_url)

        hidden = OpenAppsHidden(
            task_name=self._task_name,
            goal=goal,
            task_index=task_index,
            episode_step=0,
            initial_app_state=initial_app_state,
            last_action=None,
            base_url=self._base_url,
        )

        observation = self._build_observation(raw_obs, goal)

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "task_index": task_index,
                "task_name": self._task_name,
                "goal": goal,
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "task_name": self._task_name,
            "goal": goal,
        }

    def step(
        self,
        state: State[OpenAppsHidden],
        action: Action,
    ) -> StepResult[OpenAppsHidden]:
        """Take a browser action and return the next state."""
        self._state_tracker.validate(state, "OpenAppsEnvironment")

        raw_obs, reward, terminated, truncated, info = self._env.step(action.text)

        next_step = state.hidden.episode_step + 1

        # Compute task-level reward via state comparison
        current_app_state = _get_app_state(self._base_url)
        task_complete = self._task.check_if_task_is_complete(
            state.hidden.initial_app_state, current_app_state
        )
        task_reward = 1.0 if task_complete else 0.0

        # Check truncation
        if next_step >= self._max_steps and not terminated:
            truncated = True

        new_hidden = OpenAppsHidden(
            task_name=state.hidden.task_name,
            goal=state.hidden.goal,
            task_index=state.hidden.task_index,
            episode_step=next_step,
            initial_app_state=state.hidden.initial_app_state,
            last_action=action.text,
            base_url=state.hidden.base_url,
            trajectory=state.hidden.trajectory + (action.text or "",),
        )

        observation = self._build_observation(
            raw_obs,
            state.hidden.goal,
            previous=state.observation,
            action=action,
        )

        is_terminal = terminated or truncated or task_complete

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=is_terminal,
            info={
                **state.metadata.info,
                "open_apps_reward": task_reward,
                "browsergym_reward": reward,
                "task_complete": task_complete,
                "last_action": action.text,
            },
        )

        next_state = State(
            observation=observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated or task_complete,
            truncated=truncated,
            info={
                "open_apps_reward": task_reward,
                "browsergym_reward": reward,
                "task_complete": task_complete,
                "action": action.text,
            },
        )

    def compute_rewards(
        self,
        state: State[OpenAppsHidden],
        action: Action,
        next_state: State[OpenAppsHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


def open_apps_restore(
    env: OpenAppsEnvironment,
    state: State[OpenAppsHidden],
) -> State[OpenAppsHidden]:
    """Restore an OpenApps env to a saved state by replaying the trajectory.

    Resets to the original task via ``task_index``, then replays each action
    from ``state.hidden.trajectory``.  Validates task name to guard against
    index drift across dataset versions.

    Args:
        env: A fresh ``OpenAppsEnvironment`` instance.
        state: The target state whose ``hidden.trajectory`` is replayed.

    Returns:
        The restored state after replaying all actions.

    Raises:
        ValueError: If the task name at the given index doesn't match
            the expected task name from the saved state.
    """
    current, info = env.reset(
        options={
            "task_index": state.hidden.task_index,
            "episode_id": state.metadata.episode_id,
        }
    )

    # Validate task identity
    if state.hidden.task_name and info.get("task_name"):
        if state.hidden.task_name != info["task_name"]:
            raise ValueError(
                f"Task name mismatch: expected {state.hidden.task_name!r}, "
                f"got {info['task_name']!r} at index {state.hidden.task_index}. "
                f"Dataset version may have changed."
            )

    for action_text in state.hidden.trajectory:
        result = env.step(current, Action(text=action_text))
        current = result.next_state

    return current


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


class _OpenAppsServer:
    """Manages the OpenApps FastHTML web-server subprocess.

    Launches the server on a free port and waits for it to become ready.
    Call :meth:`stop` to tear down the process.
    """

    def __init__(
        self,
        open_apps_path: str,
        config_overrides: dict[str, Any] | None = None,
        start_port: int = 5001,
        startup_timeout: float = 120.0,
    ) -> None:
        self._open_apps_path = open_apps_path
        self._config_overrides = config_overrides or {}
        self._startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None

        self.port = self._pick_port(start_port)
        self.host = "localhost"
        self.url = f"http://{self.host}:{self.port}"

    @staticmethod
    def _pick_port(start: int = 5001) -> int:
        """Find a free TCP port."""
        import socket

        for port in range(start, start + 1000):
            if port in (5060, 5061):
                continue  # unsafe for Chrome
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", port))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return port
                except OSError:
                    continue
        # Fallback: OS picks any free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def start(self) -> None:
        """Launch the OpenApps web server."""
        import shutil

        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                "'uv' is required to launch OpenApps but was not found on PATH"
            )

        cmd = [uv, "run", "launch.py"]
        for key, value in self._config_overrides.items():
            cmd.append(f"{key}={value}")

        self._process = subprocess.Popen(
            cmd,
            cwd=self._open_apps_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Block until the server responds to HTTP requests.

        The OpenApps launcher picks its own free port, so we read stdout
        to discover the actual port (looking for the ``Uvicorn running on``
        line) and update :pyattr:`port` / :pyattr:`url` accordingly.
        """
        import re
        import selectors
        import urllib.request

        assert self._process is not None
        sel = selectors.DefaultSelector()
        sel.register(self._process.stdout, selectors.EVENT_READ)

        deadline = time.monotonic() + self._startup_timeout
        port_found = False
        collected_output: list[str] = []

        while time.monotonic() < deadline:
            # Check if process died
            if self._process.poll() is not None:
                remaining = self._process.stdout.read().decode(errors="replace")
                collected_output.append(remaining)
                raise RuntimeError(
                    "OpenApps server process exited unexpectedly "
                    f"(code {self._process.returncode}).\n"
                    f"Output:\n{''.join(collected_output)}"
                )

            # Read available output (non-blocking via selector)
            ready = sel.select(timeout=2)
            if ready:
                data = self._process.stdout.read1(8192).decode(errors="replace")  # type: ignore[attr-defined]
                if data:
                    collected_output.append(data)
                    # Look for port in output, e.g.
                    #   "Using port 5002 for the web app"
                    #   "Uvicorn running on http://localhost:5002"
                    if not port_found:
                        m = re.search(r"Using port (\d+)", data)
                        if m:
                            self.port = int(m.group(1))
                            self.url = f"http://{self.host}:{self.port}"
                            port_found = True

            # Once we know the port, try connecting
            if port_found:
                try:
                    resp = urllib.request.urlopen(self.url, timeout=5)
                    if resp.status == 200:
                        sel.close()
                        return
                except Exception:
                    pass

        sel.close()
        raise TimeoutError(
            f"OpenApps server did not start within {self._startup_timeout}s "
            f"at {self.url}\n"
            f"Output:\n{''.join(collected_output)}"
        )

    def stop(self) -> None:
        """Terminate the server subprocess."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenAppsAdapter:
    """Adapter for the OpenApps UI-agent benchmark.

    OpenApps provides synthetic web applications for evaluating digital
    agents on UI tasks such as adding calendar events, sending messages,
    managing todos, and saving map locations.

    Requirements:
        - ``open_apps`` package (clone and install from source:
          ``git clone https://github.com/facebookresearch/OpenApps && pip install -e OpenApps``)
        - ``browsergym`` (``pip install browsergym``)
        - Playwright browsers (``playwright install chromium``)

    The adapter manages an OpenApps web-server subprocess and creates
    BrowserGym environments pointing at that server.

    Example::

        >>> adapter = OpenAppsAdapter()
        >>> env = adapter.get_environment("add_call_mom_to_my_todo")
        >>> state, _ = env.reset()
    """

    def __init__(self, open_apps_path: str | None = None) -> None:
        """Initialise the adapter.

        Args:
            open_apps_path: Path to the cloned OpenApps repository.
                If *None*, resolved from the ``open_apps`` package location.
        """
        self._open_apps_path = open_apps_path
        self._server: _OpenAppsServer | None = None
        self._server_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "open_apps"

    # -- Library access -----------------------------------------------------

    def _get_open_apps(self) -> Any:
        """Import and return the open_apps module."""
        try:
            import open_apps  # noqa: F401

            return open_apps
        except ImportError as e:
            raise ImportError(
                "OpenApps is required for OpenAppsAdapter. "
                "Install from source: git clone https://github.com/facebookresearch/OpenApps && pip install -e OpenApps\n"
                "Or clone: https://github.com/facebookresearch/OpenApps"
            ) from e

    def _get_browsergym(self) -> Any:
        """Import and return browsergym."""
        try:
            import browsergym  # noqa: F401

            return browsergym
        except ImportError as e:
            raise ImportError(
                "BrowserGym is required for OpenAppsAdapter. "
                "Install with: pip install browsergym"
            ) from e

    def _resolve_open_apps_path(self) -> str:
        """Resolve the path to the OpenApps repo."""
        if self._open_apps_path is not None:
            return self._open_apps_path
        oa = self._get_open_apps()
        import pathlib

        # open_apps package lives at <repo>/src/open_apps
        pkg_dir = pathlib.Path(oa.__file__).resolve().parent
        return str(pkg_dir.parent.parent)

    # -- Server management --------------------------------------------------

    def _ensure_server(
        self,
        config_overrides: dict[str, Any] | None = None,
    ) -> _OpenAppsServer:
        """Start the OpenApps server if it isn't already running.

        Thread-safe: multiple proxy threads may call get_environment
        concurrently on a shared adapter instance.
        """
        with self._server_lock:
            if self._server is not None and self._server.is_running:
                return self._server
            path = self._resolve_open_apps_path()
            self._server = _OpenAppsServer(path, config_overrides=config_overrides)
            self._server.start()
            return self._server

    def stop_server(self) -> None:
        """Stop the managed OpenApps server (if running)."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    # -- Adapter protocol ---------------------------------------------------

    def list_environments(self) -> list[str]:
        """List available task names."""
        return list(OPEN_APPS_TASKS)

    def get_environment(
        self,
        name: str,
        max_steps: int = 30,
        use_screenshot: bool = False,
        base_url: str | None = None,
        headless: bool = True,
        extra_rewards: tuple[RewardFunction, ...] = (),
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OpenAppsEnvironment:
        """Create an OpenApps environment for the given task.

        Args:
            name: Task name (one of :data:`OPEN_APPS_TASKS`).
            max_steps: Maximum browser interactions per episode.
            use_screenshot: Include screenshots in observations.
            base_url: URL of an already-running OpenApps server.
                If *None*, the adapter will launch one automatically.
            headless: Run the browser in headless mode.
            extra_rewards: Additional reward functions.
            config_overrides: Hydra config overrides for the server.
            **kwargs: Extra keyword arguments forwarded to the task.

        Returns:
            Configured :class:`OpenAppsEnvironment`.
        """
        self._get_open_apps()
        self._get_browsergym()

        from open_apps.tasks.tasks import (
            AddEventTask,
            AddToDoTask,
            MarkToDoDoneTask,
            RemoveEventTask,
            SavePlaceTask,
            SendMessageTask,
            Task,
        )
        from open_apps.tasks.add_tasks_to_browsergym import register_tasks_with_browsergym

        import hydra
        from omegaconf import OmegaConf

        # Start server if needed
        if base_url is None:
            server = self._ensure_server(config_overrides)
            base_url = server.url

        # Load task config via Hydra
        task_configs_path = self._resolve_open_apps_path() + "/config/tasks/all_tasks.yaml"
        with open(task_configs_path) as f:
            all_tasks_cfg = OmegaConf.load(f)

        if name not in all_tasks_cfg:
            raise ValueError(
                f"Unknown task '{name}'. Available: {list(all_tasks_cfg.keys())}"
            )

        task_cfg = all_tasks_cfg[name]
        task: Task = hydra.utils.instantiate(task_cfg)

        # Register with BrowserGym
        register_tasks_with_browsergym(tasks=[task])

        # Create the BrowserGym env on a dedicated proxy thread.
        # Playwright's sync API binds greenlets to the creating thread,
        # so each env gets its own thread with its own Playwright instance.
        # Multiple envs run truly in parallel on separate threads.
        import gymnasium as gym

        task_id = task.task_id
        _base_url = base_url
        _headless = headless
        _extra_kw = dict(kwargs)

        def _make_browsergym_env() -> Any:
            return gym.make(
                f"browsergym/{task_id}",
                task_kwargs={"base_url": _base_url},
                headless=_headless,
                **_extra_kw,
            )

        browsergym_env = _BrowserGymProxy(_make_browsergym_env)

        return OpenAppsEnvironment(
            browsergym_env=browsergym_env,
            task=task,
            task_name=name,
            base_url=base_url,
            max_steps=max_steps,
            use_screenshot=use_screenshot,
            extra_rewards=extra_rewards,
        )

    def get_default_system_prompt(self, name: str) -> str | None:
        """Return a default system prompt for browser-based interaction."""
        return (
            "You are an AI agent interacting with a web application. "
            "You can see the page's accessibility tree and take browser "
            "actions such as click, fill, scroll, etc. Complete the given "
            "task by interacting with the UI elements."
        )

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a task without creating the environment."""
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "description": f"OpenApps UI task: {name}",
            "apps": OPEN_APPS_MODULES,
            "reference": "https://github.com/facebookresearch/OpenApps",
        }
