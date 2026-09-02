"""The plugin's bundled default harness: a tool-less chat loop.

verifiers resolves an unpinned seat (``harness=None``) to the taskset's bundled
harness when the taskset module exports a ``Harness`` subclass, else to the
built-in ``bash`` coding-agent harness (bash and edit tools inside a sandbox).
llenvs tasks are text relays — the environment drives every turn through
``Interaction.turn`` and parses tools from the reply text — so the policy needs
exactly one model call per turn and no sandbox. Bundling a null-harness subclass
makes that the default for every env that runs the ``llenvs-env`` taskset,
including ``--env.id single-agent``; a seat pinned through
``--env.agent.harness.id`` still wins.
"""

from __future__ import annotations

from verifiers.v1.harnesses.null import NullHarness


class LLEnvsHarness(NullHarness):
    """Tool-less chat loop (one model call per turn); the llenvs relay's default."""
