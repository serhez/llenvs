"""Tests for optional-adapter registration robustness.

``_register_adapters()`` runs at ``import llenvs`` time and probes each
optional adapter by importing its third-party stack. A probe that fails for
*any* reason — not just a missing package — must skip that adapter instead of
breaking the import. (Real-world case: pyjnius raises ``RuntimeError`` when no
JVM is present, which made ``import llenvs`` impossible on machines without
Java even though WebShop was not being used.)
"""

import logging

import pytest

from llenvs.adapters import WebShopAdapter, _register_adapters
from llenvs.core.registry import environment_registry


@pytest.fixture
def webshop_unregistered():
    """Clear any import-time webshop registration; restore it afterwards."""
    try:
        original = environment_registry.get_adapter("webshop")
    except KeyError:
        original = None
    environment_registry.unregister_adapter("webshop")
    yield
    environment_registry.unregister_adapter("webshop")
    if original is not None:
        environment_registry.register_adapter(original)


def _probe_raising(exc: Exception):
    def probe(self):
        raise exc

    return probe


def test_probe_runtime_error_skips_adapter(monkeypatch, webshop_unregistered):
    """A non-ImportError probe failure must not propagate out of registration."""
    monkeypatch.setattr(
        WebShopAdapter, "_get_webshop", _probe_raising(RuntimeError("no libjvm.so"))
    )
    _register_adapters()  # must not raise
    assert "webshop" not in environment_registry.list_adapters()


def test_probe_import_error_skips_adapter(monkeypatch, webshop_unregistered):
    monkeypatch.setattr(
        WebShopAdapter, "_get_webshop", _probe_raising(ImportError("not installed"))
    )
    _register_adapters()
    assert "webshop" not in environment_registry.list_adapters()


def test_probe_failure_is_logged(monkeypatch, webshop_unregistered, caplog):
    monkeypatch.setattr(
        WebShopAdapter, "_get_webshop", _probe_raising(RuntimeError("no libjvm.so"))
    )
    with caplog.at_level(logging.DEBUG, logger="llenvs.adapters"):
        _register_adapters()
    assert "no libjvm.so" in caplog.text
    assert "WebShop" in caplog.text


def test_successful_probe_registers_adapter(monkeypatch, webshop_unregistered):
    monkeypatch.setattr(WebShopAdapter, "_get_webshop", lambda self: object())
    _register_adapters()
    assert "webshop" in environment_registry.list_adapters()
