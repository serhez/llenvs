"""Explore OpenApps with screenshots enabled.

Runs a short trajectory with use_screenshot=True, saves each step's
screenshot to disk, and prints a summary of the image data flowing
through the observation pipeline.

Run from the llenvs root:
    uv run python scripts/utils/explore_open_apps_vision.py
"""

from __future__ import annotations

import base64
from pathlib import Path

from llenvs.adapters.open_apps import OpenAppsAdapter, _get_app_state
from llenvs.core.state import Action


OUTPUT_DIR = Path("data/open_apps_vision_explore")


def save_screenshot(img_data: str, name: str) -> Path:
    """Decode base64 screenshot and save as PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    path.write_bytes(base64.b64decode(img_data))
    return path


def print_image_info(label: str, obs) -> None:
    """Print image info from an observation's structured fields."""
    imgs = obs.get_images()
    print(f"\n  [{label}] Images:")
    print(f"    task images:  {len(imgs.task)}")
    print(f"    state images: {len(imgs.state)}")
    for i, img in enumerate(imgs.state):
        size_kb = len(img.data) * 3 / 4 / 1024  # approx decoded size
        print(f"    state[{i}]: media_type={img.media_type}, ~{size_kb:.0f} KB")


def main():
    adapter = OpenAppsAdapter()

    task_name = "add_call_mom_to_my_todo"
    print(f"Creating environment: {task_name} (use_screenshot=True)")
    env = adapter.get_environment(
        task_name,
        max_steps=15,
        headless=True,
        use_screenshot=True,
        viewport={"width": 1280, "height": 720},
    )

    # ---- Reset ----
    print("\n" + "=" * 60)
    print("RESET")
    print("=" * 60)
    state, info = env.reset()

    print(f"Goal: {state.hidden.goal}")
    print(f"AXTree (first 500 chars): {state.observation.state.text[:500]}")
    print_image_info("reset", state.observation)

    if state.observation.state.images:
        path = save_screenshot(state.observation.state.images[0].data, "step0_reset")
        print(f"  -> Saved: {path}")

    # ---- Step 1: Click on OpenTodos link ----
    print("\n" + "=" * 60)
    print("STEP 1 - click OpenTodos link")
    print("=" * 60)
    action1 = Action(text="click('23')")
    print(f"Action: {action1.text}")
    result1 = env.step(state, action1)
    print(f"Reward: {result1.rewards.total}, Terminated: {result1.terminated}")
    print_image_info("step1", result1.next_state.observation)

    if result1.next_state.observation.state.images:
        path = save_screenshot(result1.next_state.observation.state.images[0].data, "step1_click_todos")
        print(f"  -> Saved: {path}")

    # ---- Step 2: Fill in "Call Mom" ----
    print("\n" + "=" * 60)
    print("STEP 2 - fill 'Call Mom'")
    print("=" * 60)
    axtree2 = result1.next_state.observation.state.text
    for line in axtree2.split("\n"):
        if any(kw in line.lower() for kw in ("textbox", "input", "button", "add")):
            print(f"  >> {line.strip()}")

    action2 = Action(text="fill('23', 'Call Mom')")
    print(f"Action: {action2.text}")
    result2 = env.step(result1.next_state, action2)
    print(f"Reward: {result2.rewards.total}, Terminated: {result2.terminated}")
    print_image_info("step2", result2.next_state.observation)

    if result2.next_state.observation.state.images:
        path = save_screenshot(result2.next_state.observation.state.images[0].data, "step2_fill")
        print(f"  -> Saved: {path}")

    # ---- Step 3: Click Add button ----
    print("\n" + "=" * 60)
    print("STEP 3 - click Add button")
    print("=" * 60)
    action3 = Action(text="click('24')")
    print(f"Action: {action3.text}")
    result3 = env.step(result2.next_state, action3)
    print(f"Reward: {result3.rewards.total}, Terminated: {result3.terminated}")
    print_image_info("step3", result3.next_state.observation)

    if result3.next_state.observation.state.images:
        path = save_screenshot(result3.next_state.observation.state.images[0].data, "step3_add")
        print(f"  -> Saved: {path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nScreenshots saved to: {OUTPUT_DIR.resolve()}")

    final_state = _get_app_state(env._base_url)
    print("\nFinal app state (todos):")
    for todo in final_state.get("todo", []):
        print(f"  {'[x]' if todo.get('done') else '[ ]'} {todo['title']}")

    print("\nReward signals:")
    for sig in result3.rewards.signals:
        print(f"  {sig.name}: reward={sig.reward}")

    # ---- Verify pipeline ----
    print("\n" + "=" * 60)
    print("PIPELINE VERIFICATION")
    print("=" * 60)
    obs = result3.next_state.observation
    imgs = obs.get_images()
    print(f"obs.state is not None: {obs.state is not None}")
    print(f"obs.state.images count: {len(obs.state.images) if obs.state else 0}")
    print(f"get_images().state count: {len(imgs.state)}")
    print(f"get_images().task count: {len(imgs.task)}")
    if imgs.state:
        print(f"First state image source tag: {imgs.state[0].source}")
        print(f"First state image media_type: {imgs.state[0].media_type}")
        print(f"First state image data length: {len(imgs.state[0].data)} chars (base64)")

    # ---- Cleanup ----
    print("\n" + "=" * 60)
    print("CLEANUP")
    adapter.stop_server()
    print("Server stopped.")


if __name__ == "__main__":
    main()
