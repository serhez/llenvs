"""Child-process launcher for OpenApps with an optional archived app clock.

Only the app modules' datetime bindings are replaced. System time, networking
timeouts, and third-party library clocks remain live. No source files are edited.
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from datetime import UTC, datetime, tzinfo
from pathlib import Path

from llenvs.adapters.open_apps import _parse_reference_time


def _set_app_clock(reference: datetime) -> None:
    """Install a fixed clock in the known app modules, inside the child only."""

    class AppDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return reference.replace(tzinfo=None) if tz is None else reference.astimezone(tz)

        @classmethod
        def today(cls) -> datetime:
            return cls.now()

        @classmethod
        def utcnow(cls) -> datetime:
            return reference.astimezone(UTC).replace(tzinfo=None)

    for app in ("calendar_app", "messenger_app", "map_app"):
        module = importlib.import_module(f"open_apps.apps.{app}.main")
        if getattr(module, "datetime", None) is not datetime:
            raise RuntimeError(
                f"Unexpected datetime binding in {module.__name__}; cannot set app clock"
            )
        module.datetime = AppDateTime


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-apps-path", required=True, type=Path)
    parser.add_argument("--reference-time", type=_parse_reference_time)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    source = args.open_apps_path.resolve()
    launcher = source / "launch.py"
    if not launcher.is_file():
        parser.error(f"OpenApps launcher does not exist: {launcher}")
    # OpenApps uses both open_apps.* and src.open_apps.* imports.
    sys.path[:0] = [str(source / "src"), str(source)]
    if args.reference_time is not None:
        _set_app_clock(args.reference_time)
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    sys.argv = [str(launcher), *overrides]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
