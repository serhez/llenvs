"""Single import point for the verifiers v1 API with an install hint.

``taskset`` and ``env`` import ``vf`` (and ``parse_message``) from here;
``_config`` and ``_relay`` never touch verifiers so they can be tested
without it.
"""

try:
    import verifiers.v1 as vf
    from verifiers.v1.dialects.chat import parse_message
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "llenvs_env requires the verifiers v1 API (verifiers>=0.3.1). "
        "Install with: uv pip install -e '.[verifiers]'"
    ) from e

__all__ = ["vf", "parse_message"]
