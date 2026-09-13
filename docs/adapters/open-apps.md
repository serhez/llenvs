# OpenApps

OpenApps provides browser-based calendar, todo, messenger, and map tasks.
The adapter uses BrowserGym and an OpenApps app server. Install the app source,
its runtime dependencies, BrowserGym, and a compatible Playwright Chromium
separately; the adapter does not bundle them.

```python
from llenvs.adapters.open_apps import OpenAppsAdapter

adapter = OpenAppsAdapter(open_apps_path="/path/to/OpenApps")
env = adapter.get_environment(
    "add_call_mom_to_my_todo",
    max_steps=30,
    use_screenshot=True,
    screenshot_with_som=True,
    viewport={"width": 1280, "height": 720},
)
try:
    state, info = env.reset()
finally:
    env.close()
```

Each task's browser lives on its own owner thread. The environment caches these
browsers when switching `task_index`. Reset and step calls on a browser are
serialized. `base_url` can point to an existing server; otherwise the adapter
starts and manages one. `config_overrides` supplies Hydra overrides to that server.
Managed servers continue draining stdout/stderr after readiness, retaining only
a bounded output tail so a full pipe cannot block app requests. Stopping the
server also stops its output reader and closes the pipe.

Browser reset does **not** restore the app server's database. For independent
episodes or branches, create a fresh adapter-managed server and environment,
then use `open_apps_restore` to replay the saved prefix. Reusing an adapter or
an existing `base_url` shares app mutations even if a new browser is created.
Validate restored app data as well as visible text before accepting a replay.

## Reward scope

`reward_scope="native"` is the default: task completion uses OpenApps' native
whole-app-state comparison. `reward_scope="task_local"` explicitly restricts the
comparison to the task's relevant app (calendar, todo, messenger or map).
The adapter builds the native target state, then replaces its relevant app with
the current app data before applying the native comparator. Unrelated apps remain
at their target values **for comparison only**; displayed app data is unchanged.

Task-local completion determines the task reward and terminal flag, and records
`reward_scope="task_local"` and `reward_relevant_apps` in step info and next-state
metadata. Native mode does not add these annotations. Unsupported task names,
missing relevant app data and comparator errors fail explicitly in task-local
mode, rather than silently switching to another reward rule.

Choose this option explicitly when restoring data collected under task-local
rewards. Saved annotations do not automatically select it. Changing reward scope
can change termination as well as rewards, so validate it against saved transitions
before using a replay to regenerate observations or reference labels.

## Replay controls

All controls below are arguments to `OpenAppsAdapter.get_environment`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `browsergym_call_timeout` | `60` | Seconds to wait for each browser operation. Must be finite and positive. |
| `browser_scale_factor` | `None` | Optional positive Chromium device scale; `None` preserves native scaling. |
| `reference_time` | `None` | Optional ISO timestamp with an explicit timezone for the managed app clock. |
| `recover_observation` | `False` | Opt-in bounded re-reading when raw accessibility data has no visible BID-bearing node. |

The timeout belongs to the browser proxy, not BrowserGym's own operation timeout
or the app-server startup deadline. After it expires, the proxy refuses further
reset/step calls: a late result must not be mistaken for a later action's result.
Close and recreate the environment. Close waits up to 30 seconds for the owner
thread; it cannot forcibly interrupt a stuck browser operation. Deferred cleanup
runs when the operation returns.

`recover_observation=True` checks each reset/step observation before formatting
its text and screenshot. A screenshot can depict a finished page even when the
accessibility tree was captured before the page loaded. If no non-ignored node
has a BrowserGym ID with visibility at least 0.5, the proxy waits up to 10 seconds
for `domcontentloaded` and takes a fresh observation on the browser's owner
thread. It allows at most three fresh reads; a load timeout or browser error
propagates immediately. Recovery remains inside `browsergym_call_timeout`.

Recovery does not repeat the action, reset the environment, inject a no-op, or
recompute the original BrowserGym rewards/terminal flags/info. Healthy responses
need no extra wait or read. Visible headings are sufficient; neither clickable
elements nor screenshots are required. The check uses raw accessibility data
even when `omit_axtree_text=True` hides it from the actor. Genuinely blank pages
are rejected by this opt-in mode. With the default `False`, observations pass
through unchanged.

If recovery fails, that browser proxy refuses further operations. Close and
recreate the environment: the browser may already have executed the action even
though no usable observation was returned. This incompleteness check does not
prove a nonempty observation is correct or that all page assets have loaded.
Dataset replay must still validate text, action IDs, state identity and rewards.

`browser_scale_factor=1` requests CSS-sized Chromium screenshots on HiDPI hosts.
It merges the device-scale flag with Chromium's existing launch arguments on the
owner thread, before screenshots and Set-of-Marks overlays are produced. It does
not resize saved images. Conflicting device-scale flags are rejected.

For example, `reference_time="2001-02-03T04:05:06+02:00"` fixes the
calendar/messenger/map app datetime bindings to that instant; its offset defines
the app's local time. System time, browser JavaScript time, task objects in the
parent process, and request timers are not changed. The override is installed
only in the managed server's child process, without editing app source files.
The clocked server uses the calling Python interpreter and does not synchronize
packages. That interpreter must already have the required app dependencies.
Without a reference-time override, the native `uv run launch.py` launch path is
used.

A reference-time override cannot control an external `base_url`. An adapter
also refuses to change the clock of its running shared server. Use a separate
adapter/server for a different clock, rather than changing a server in use.

These controls do not guarantee exact replay across app/browser versions or
operating systems. Fonts, scrollbars, animations, and other rendering details
can still differ. Validate saved states, action identifiers, app data, and image
geometry before treating regenerated observations as equivalent.
