"""Archived app-clock contracts using a synthetic launcher in local subprocesses.

Only temporary test files are created. The real OpenApps checkout and datasets
are not loaded or modified, and no HTTP server or browser is started.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from llenvs.adapters import open_apps as oa

BOOTSTRAP = "llenvs.adapters._open_apps_server"
APP_NAMES = ("calendar_app", "messenger_app", "map_app")


@pytest.fixture
def synthetic_openapps(tmp_path):
    source = tmp_path / "OpenApps with spaces"
    apps = source / "src/open_apps/apps"
    for name in APP_NAMES:
        module = apps / name / "main.py"
        module.parent.mkdir(parents=True)
        module.write_text(
            "from datetime import datetime, timezone\n"
            "def read_clock():\n"
            "    return {'local': datetime.now().isoformat(),\n"
            "            'utc': datetime.now(timezone.utc).isoformat(),\n"
            "            'today': datetime.today().isoformat()}\n"
        )
    for package in (source / "src/open_apps", apps, *(apps / name for name in APP_NAMES)):
        (package / "__init__.py").write_text("")
    (source / "launch.py").write_text(
        "import importlib, json, sys, time\n"
        "from datetime import datetime, timezone\n"
        f"names = {APP_NAMES!r}\n"
        "modules = [importlib.import_module('open_apps.apps.' + n + '.main') for n in names]\n"
        "start = time.monotonic()\n"
        "time.sleep(0.002)\n"
        "print(json.dumps({'apps': [m.read_clock() for m in modules],\n"
        "    'system_now': datetime.now(timezone.utc).timestamp(),\n"
        "    'wall_time': time.time(), 'elapsed': time.monotonic() - start,\n"
        "    'argv': sys.argv[1:], 'python': sys.executable}))\n"
    )
    return source


def _source_hashes(source):
    return {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.py")
    }


def _run_launcher(source, reference_time=None, overrides=()):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(__file__).resolve().parents[1] / "src"),
            str(source / "src"),
            env.get("PYTHONPATH", ""),
        )
    )
    command = [sys.executable, "-m", BOOTSTRAP, "--open-apps-path", str(source)]
    if reference_time is not None:
        command += ["--reference-time", reference_time]
    command += ["--", *overrides]
    return subprocess.run(command, cwd=source, env=env, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("reference", ["2001-02-03T04:05:06+02:00", "2002-07-09T00:15:16-03:00"])
def test_archived_clock_only_affects_app_bindings(synthetic_openapps, reference):
    source = synthetic_openapps
    before = _source_hashes(source)
    parent_datetime = datetime
    lower = time.time()
    result = _run_launcher(source, reference)
    upper = time.time()
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    expected = datetime.fromisoformat(reference)
    for clock in data["apps"]:
        assert clock["local"] == expected.replace(tzinfo=None).isoformat()
        assert clock["today"] == clock["local"]
        assert clock["utc"] == expected.astimezone(UTC).isoformat()
    assert lower <= data["system_now"] <= upper
    assert lower <= data["wall_time"] <= upper
    assert data["elapsed"] > 0
    assert datetime is parent_datetime
    assert lower <= datetime.now(UTC).timestamp() <= time.time()
    assert _source_hashes(source) == before


def test_clock_defaults_to_live_time_and_preserves_literal_overrides(synthetic_openapps):
    overrides = ["seed=42", "logs_dir=a directory with spaces", "note=$(do-not-execute)"]
    before = _source_hashes(synthetic_openapps)
    lower = time.time()
    result = _run_launcher(synthetic_openapps, overrides=overrides)
    upper = time.time()
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    for clock in data["apps"]:
        assert lower <= datetime.fromisoformat(clock["utc"]).timestamp() <= upper
    assert data["argv"] == overrides
    assert data["python"] == sys.executable
    assert _source_hashes(synthetic_openapps) == before


@pytest.mark.parametrize("reference", ["not-a-date", "2001-02-03", "2001-02-03T04:05:06"])
def test_invalid_or_timezone_free_clock_rejected_before_launch(synthetic_openapps, reference):
    before = _source_hashes(synthetic_openapps)
    result = _run_launcher(synthetic_openapps, reference)
    assert result.returncode != 0
    # A missing module is not evidence of correct validation.
    assert "No module named" not in result.stderr
    assert "reference" in result.stderr.lower()
    assert result.stdout == ""
    assert _source_hashes(synthetic_openapps) == before


def test_clocked_server_uses_isolated_child_and_passes_overrides_literally(tmp_path, monkeypatch):
    monkeypatch.setattr(oa._OpenAppsServer, "_pick_port", staticmethod(lambda start: 5001))
    monkeypatch.setattr(oa._OpenAppsServer, "_wait_until_ready", lambda self: None)
    process = Mock()
    popen = Mock(return_value=process)
    monkeypatch.setattr(oa.subprocess, "Popen", popen)
    env_before = dict(os.environ)
    overrides = {"seed": 42, "logs_dir": str(tmp_path / "logs with spaces")}
    original_overrides = dict(overrides)
    timestamp = "2001-02-03T04:05:06+02:00"
    server = oa._OpenAppsServer(str(tmp_path), config_overrides=overrides, reference_time=timestamp)
    try:
        server.start()
        command = popen.call_args.args[0]
        assert command[:3] == [sys.executable, "-m", BOOTSTRAP]
        assert command[command.index("--reference-time") + 1] == timestamp
        assert command[command.index("--open-apps-path") + 1] == str(tmp_path)
        assert command[command.index("--") + 1 :] == [f"{k}={v}" for k, v in overrides.items()]
        assert not popen.call_args.kwargs.get("shell", False)
        assert popen.call_args.kwargs["start_new_session"] is True
        assert os.environ == env_before
        assert overrides == original_overrides
    finally:
        server.stop()
