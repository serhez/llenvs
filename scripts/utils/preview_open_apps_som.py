"""Preview Set-of-Marks overlay rendering on an OpenApps screenshot.

Spins up an OpenApps env, grabs a raw BrowserGym observation (after
the home-screen fade-in settle), then renders the screenshot both
plain and with browsergym.utils.obs.overlay_som applied.  Saves both
PNGs side by side so we can eyeball whether SoM is legible enough to
support a vision-only actor.

Run from the llenvs root:
    uv run python scripts/utils/preview_open_apps_som.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
from browsergym.utils.obs import overlay_som

from llenvs.adapters.open_apps import OpenAppsAdapter

OUTPUT_DIR = Path("data/open_apps_som_preview")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = OpenAppsAdapter()
    env = adapter.get_environment(
        name="add_call_mom_to_my_todo",
        max_steps=20,
        use_screenshot=True,
        viewport={"width": 1280, "height": 720},
    )

    # Walk through a few steps so we can render SoM on different screens:
    # - the welcome page after reset
    # - the OpenTodos page after clicking the OpenTodos card
    # - the OpenTodos page after focusing the New Todo textbox
    state, _ = env.reset()

    def grab_raw():
        """Re-fetch the latest BrowserGym observation via a noop()."""
        # _env is the _BrowserGymProxy; .step("noop(0)") returns a fresh
        # observation without changing the page state.
        raw_obs, _r, _term, _trunc, _info = env._env.step("noop(0)")
        return raw_obs

    def save_pair(name: str, raw_obs: dict) -> None:
        screenshot = raw_obs["screenshot"]
        if not isinstance(screenshot, np.ndarray):
            screenshot = np.asarray(screenshot)
        plain_path = OUTPUT_DIR / f"{name}_plain.png"
        PIL.Image.fromarray(screenshot).save(plain_path)

        extra_props = raw_obs.get("extra_element_properties", {})
        som_arr = overlay_som(screenshot, extra_props)
        som_path = OUTPUT_DIR / f"{name}_som.png"
        PIL.Image.fromarray(som_arr).save(som_path)

        n_marked = sum(
            1
            for p in extra_props.values()
            if p.get("set_of_marks") and p.get("bbox")
        )
        print(f"  {name}: {n_marked} SoM-tagged elements → {som_path}")

    print("== Step 0: welcome page ==")
    save_pair("step0_welcome", grab_raw())

    print("== Step 1: clicking OpenTodos card ==")
    # Find the OpenTodos link's bid by scanning extra_props for the right name
    raw = grab_raw()
    todos_bid = None
    for bid, props in raw.get("extra_element_properties", {}).items():
        if props.get("set_of_marks"):
            # We don't have node names here; just try clicking the first
            # SoM-tagged element near the top-left card and let the
            # accessibility tree guide us.
            pass
    # Easier: look at the axtree for "OpenTodos" name → bid mapping.
    from browsergym.utils.obs import flatten_axtree_to_str

    axtree_text = flatten_axtree_to_str(
        raw["axtree_object"],
        extra_properties=raw.get("extra_element_properties", {}),
        with_clickable=True,
        with_visible=True,
        filter_visible_only=True,
        filter_with_bid_only=True,
    )
    # crude parse: lines like "[N] link 'OpenTodos'"
    for line in axtree_text.splitlines():
        if "OpenTodos" in line and line.strip().startswith("["):
            bid = line.strip().split("]", 1)[0].lstrip("[")
            todos_bid = bid
            break

    if todos_bid is None:
        print("  (could not find OpenTodos bid — skipping)")
    else:
        print(f"  clicking bid {todos_bid}")
        env._env.step(f"click('{todos_bid}')")
        save_pair("step1_open_todos", grab_raw())

    print("== Step 2: clicking the New Todo textbox ==")
    raw = grab_raw()
    axtree_text = flatten_axtree_to_str(
        raw["axtree_object"],
        extra_properties=raw.get("extra_element_properties", {}),
        with_clickable=True,
        with_visible=True,
        filter_visible_only=True,
        filter_with_bid_only=True,
    )
    textbox_bid = None
    for line in axtree_text.splitlines():
        if "textbox" in line and "New Todo" in line and line.strip().startswith("["):
            textbox_bid = line.strip().split("]", 1)[0].lstrip("[")
            break
    if textbox_bid is None:
        print("  (could not find New Todo textbox bid — skipping)")
    else:
        env._env.step(f"click('{textbox_bid}')")
        save_pair("step2_textbox_focused", grab_raw())

    print()
    print(f"Saved previews to {OUTPUT_DIR.resolve()}")
    print("Compare *_plain.png vs *_som.png to see whether the bid")
    print("overlays are legible at 1280x720.")


if __name__ == "__main__":
    main()
