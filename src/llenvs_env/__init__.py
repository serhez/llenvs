"""verifiers v1 plugin exposing llenvs environments as the ``llenvs-env`` taskset.

The v1 plugin loader maps the taskset id ``llenvs-env`` to this module and
resolves the names in ``__all__`` with ``getattr``: exactly one ``Taskset``
subclass plus (optionally) one ``Env`` subclass. The exported ``LLEnvsEnv``
therefore pairs with ``LLEnvsTaskset`` automatically unless a run names
another env id.

The exports are resolved lazily so that the verifiers-free halves of this
package (``_config``, ``_relay``) stay importable without verifiers installed;
the install-hint guard lives in ``llenvs_env._vf`` and fires on first access.

Import hygiene: this package imports ``verifiers.v1`` and ``llenvs``; ``llenvs``
never imports this package.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LLEnvsTaskset", "LLEnvsEnv"]


def __getattr__(name: str) -> Any:
    if name == "LLEnvsTaskset":
        from llenvs_env.taskset import LLEnvsTaskset

        return LLEnvsTaskset
    if name == "LLEnvsEnv":
        from llenvs_env.env import LLEnvsEnv

        return LLEnvsEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
