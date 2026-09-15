"""A ready app server must not hang when its stdout pipe fills.

Only a tiny synthetic local child is launched; no HTTP server, browser, installed
OpenApps application, network access or dependency synchronization is involved.
"""

import subprocess
import sys
import time

from llenvs.adapters import open_apps as oa


def test_ready_server_continues_draining_stdout_and_stops_its_reader(tmp_path, monkeypatch):
    marker = tmp_path / "child-finished-output"
    real_popen = subprocess.Popen
    children = []
    code = (
        "import os, pathlib, sys, time\n"
        "print('ready', flush=True)\n"
        "chunk = b'x' * 8192\n"
        "for _ in range(256): os.write(1, chunk)\n"
        "pathlib.Path(sys.argv[1]).write_text('drained')\n"
        "time.sleep(30)\n"
    )

    def spawn(*args, **kwargs):
        child = real_popen([sys.executable, "-u", "-c", code, str(marker)], **kwargs)
        children.append(child)
        return child

    def ready(server):
        assert server._process.stdout.readline() == b"ready\n"

    monkeypatch.setattr(oa.subprocess, "Popen", spawn)
    monkeypatch.setattr(oa._OpenAppsServer, "_pick_port", staticmethod(lambda _: 5001))
    monkeypatch.setattr(oa._OpenAppsServer, "_wait_until_ready", ready)
    server = oa._OpenAppsServer(str(tmp_path), reference_time="2001-02-03T04:05:06+02:00")
    try:
        server.start()
        deadline = time.monotonic() + 2.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "Ready server blocks after filling its undrained stdout pipe"
        assert marker.read_text() == "drained"
    finally:
        server.stop()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    assert not server.is_running
    assert all(child.poll() is not None and child.stdout.closed for child in children)
