"""Actual native hard exits composed with production cleanup in owned CPU children."""

import json
import os
import signal
import subprocess
import sys

import pytest

from tests.skyrl_source import REPO, source


@pytest.mark.parametrize(
    "scenario",
    [
        "failure",
        "concurrent",
        "blocked_release",
        "blocked_timeout",
        "session_failure",
        "cancellation",
        "outside_generator",
    ],
)
def test_native_exit_and_production_resource_cleanup(tmp_path, scenario):
    source("skyrl/train/fully_async_trainer.py")  # Apply default-skip/explicit-invalid rules here.
    log = tmp_path / "child.log"
    with log.open("x") as stream:
        with subprocess.Popen(
            [sys.executable, "-m", "tests.skyrl_fatal_run", scenario, str(tmp_path)],
            cwd=REPO,
            env={**os.environ, "LLENVS_SKYRL_FATAL_TEST": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                pytest.fail(f"fatal-path child hung; inspect {log}")
    output = log.read_text()
    assert code == (0 if scenario == "cancellation" else 1), output
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    created = {e["resource"] for e in events if e["event"] == "env_created"}
    closed = [e["resource"] for e in events if e["event"] == "env_closed"]
    sessions = [e["resource"] for e in events if e["event"] == "session_finished"]
    assert len(created) == (4 if scenario == "concurrent" else 2), output
    assert len(closed) == len(set(closed)), "resource closed twice"
    assert len(sessions) == len(set(sessions)), "session finished twice"
    assert "scorer_created" in names
    if scenario == "outside_generator":
        assert not closed and not sessions and "scorer_closed" not in names
    else:
        assert len(sessions) == len(created)
        assert names.count("scorer_closed") == 1
        assert set(closed) == created - ({0} if scenario == "blocked_timeout" else set())
    if scenario == "cancellation":
        assert "native_error" not in names
        assert events[-1] == {"event": "cancelled_cleanly", "running": 0}
        assert "outer_finally" in names
    else:
        assert "native_error" in names and "Traceback:" in output
        assert "outer_finally" not in names and "returned_from_native_worker" not in names
        first_error = names.index("native_error")
        assert all(i < first_error for i, name in enumerate(names) if name == "env_closed")
        if scenario != "outside_generator":
            assert names.index("scorer_closed") < first_error
    if scenario == "blocked_release":
        assert (
            names.index("step_started")
            < names.index("step_released")
            < names.index("step_finished")
        )
        closing = next(
            i for i, e in enumerate(events) if e == {"event": "env_closed", "resource": 0}
        )
        assert names.index("step_finished") < closing < names.index("native_error")
    if scenario == "blocked_timeout":
        assert "cleanup deadline exceeded" in output
        assert "step_finished" not in names
    if scenario == "session_failure":
        assert "fixture session release failure" in output
    if scenario == "concurrent":
        assert names.count("inference_failed") >= 2
