"""Launch ordering uses a native-runtime double; never initializes Ray here."""

import importlib
import sys
from types import SimpleNamespace as Namespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def launch(monkeypatch):
    entry = importlib.import_module("llenvs.integrations.skyrl.entrypoint")
    events = []
    cfg = Namespace(llenvs=Namespace(check_only=True))
    prepared = Namespace(identity={"recipe": "hash"})
    runtime = Namespace(
        LlenvsSkyRLTrainConfig=Namespace(from_cli_overrides=Mock(return_value=cfg)),
        prepare=Mock(side_effect=lambda _cfg: events.append("prepare") or prepared),
        launch=Mock(side_effect=lambda *args: events.append("launch")),
    )
    monkeypatch.setitem(sys.modules, "llenvs.integrations.skyrl._native", runtime)
    return Namespace(entry=entry, runtime=runtime, cfg=cfg, events=events)


def test_check_only_finishes_before_native_launch(launch, capsys):
    launch.entry.main(["llenvs.check_only=true"])
    assert launch.events == ["prepare"]
    launch.runtime.launch.assert_not_called()
    assert "no Ray" in capsys.readouterr().out


def test_real_launch_uses_the_preflight_identity(launch):
    launch.cfg.llenvs.check_only = False
    launch.entry.main(["llenvs.check_only=false"])
    assert launch.events == ["prepare", "launch"]
    launch.runtime.launch.assert_called_once_with(launch.cfg, {"recipe": "hash"})


def test_preflight_failure_cannot_fall_through_to_launch(launch):
    launch.runtime.prepare.side_effect = ValueError("unsafe recipe")
    with pytest.raises(ValueError, match="unsafe recipe"):
        launch.entry.main([])
    launch.runtime.launch.assert_not_called()


def test_help_does_not_import_native_stack(monkeypatch, capsys):
    entry = importlib.import_module("llenvs.integrations.skyrl.entrypoint")
    monkeypatch.setattr(
        importlib, "import_module", Mock(side_effect=AssertionError("native import"))
    )
    entry.main(["--help"])
    assert "key.path=value" in capsys.readouterr().out
