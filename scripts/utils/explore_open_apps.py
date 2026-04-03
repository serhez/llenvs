"""Explore the OpenApps adapter end-to-end (real server + BrowserGym).

Run from the llenvs root:
    python scripts/explore_open_apps.py
"""

from pprint import pprint

from llenvs.adapters.open_apps import OpenAppsAdapter, _get_app_state
from llenvs.core.state import Action


def main():
    adapter = OpenAppsAdapter()

    print("=" * 60)
    print("Available tasks:")
    for t in adapter.list_environments():
        print(f"  - {t}")
    print()

    task_name = "add_call_mom_to_my_todo"
    print(f"Creating environment: {task_name}")
    env = adapter.get_environment(
        task_name,
        max_steps=15,
        headless=True,
        use_screenshot=False,
    )

    # ── Reset ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESET")
    print("=" * 60)
    state, info = env.reset()

    print(f"\nGoal: {state.hidden.goal}")
    print(f"\nAXTree:")
    print(state.observation.state.text[:2000])

    # ── Step 1: Click on OpenTodos link (bid=23) ─────────────
    print("\n" + "=" * 60)
    print("STEP 1 — click OpenTodos link")
    print("=" * 60)
    action1 = Action(text="click('23')")
    print(f"Action: {action1.text}")
    result1 = env.step(state, action1)
    print(f"Reward: {result1.rewards.total}, Terminated: {result1.terminated}")
    print(f"\nAXTree after click:")
    print(result1.next_state.observation.state.text[:2000])

    # ── Step 2: type "Call Mom" into the input ───────────────
    # Look at the AXTree to find the input field bid
    print("\n" + "=" * 60)
    print("STEP 2 — type 'Call Mom' into input")
    print("=" * 60)
    # Find the input field from the AXTree — we'll inspect what bids are on the todo page
    axtree2 = result1.next_state.observation.state.text
    # Print lines containing 'textbox' or 'input' to find the right bid
    for line in axtree2.split("\n"):
        if any(kw in line.lower() for kw in ("textbox", "input", "button", "add")):
            print(f"  >> {line.strip()}")

    # Textbox is bid 23, Add button is bid 24 (from AXTree above)
    action2 = Action(text="fill('23', 'Call Mom')")
    print(f"\nAction: {action2.text}")
    result2 = env.step(result1.next_state, action2)
    print(f"Reward: {result2.rewards.total}, Terminated: {result2.terminated}")
    print(f"\nAXTree after fill:")
    print(result2.next_state.observation.state.text[:2000])

    # ── Step 3: click Add button ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — click Add button")
    print("=" * 60)
    axtree3 = result2.next_state.observation.state.text
    for line in axtree3.split("\n"):
        if any(kw in line.lower() for kw in ("button", "add", "submit")):
            print(f"  >> {line.strip()}")

    action3 = Action(text="click('24')")  # Add button
    print(f"\nAction: {action3.text}")
    result3 = env.step(result2.next_state, action3)
    print(f"Reward: {result3.rewards.total}, Terminated: {result3.terminated}")
    print(f"\nAXTree after add:")
    print(result3.next_state.observation.state.text[:2000])

    # ── Check app state ──────────────────────────────────────
    print("\n--- Final app state (todos) ---")
    final_state = _get_app_state(env._base_url)
    for todo in final_state.get("todo", []):
        print(f"  {'[x]' if todo.get('done') else '[ ]'} {todo['title']}")

    print("\n--- Reward signals ---")
    for sig in result3.rewards.signals:
        print(f"  {sig.name}: reward={sig.reward}")

    # ── Cleanup ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLEANUP")
    adapter.stop_server()
    print("Server stopped.")


if __name__ == "__main__":
    main()
