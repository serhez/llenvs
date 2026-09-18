"""Immutable, local run compatibility identity; no trajectory/checkpoint state."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from llenvs.integrations.skyrl.data import _canonical_json

_SECTIONS = ("datasets", "environment", "rewards", "execution", "models", "runtime")
_NAME = "llenvs-run.json"


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def tree_hash(root: Path) -> str:
    """Hash ordered relative filenames and contents, including model symlink targets.

    Bytecode and tool caches are not source/model identity. Directory symlinks
    are rejected rather than silently omitting their contents.
    """
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("identity root must be a directory")
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".cache"} for part in relative.parts):
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError("identity tree cannot contain directory symlinks")
        if path.is_dir() or path.suffix == ".pyc":
            continue
        if not path.is_file():
            raise ValueError("identity tree contains a missing/non-regular file")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append((relative.as_posix(), digest))
    if not entries:
        raise ValueError("identity tree cannot be empty")
    return content_hash(entries)


def run_identity(
    *, datasets: Any, environment: Any, rewards: Any, execution: Any, models: Any, runtime: Any
) -> dict[str, Any]:
    """Hash semantic sections, keeping task content and configuration out of logs.

    Callers own complete recipe resolution. Forwarded credential values must
    never be supplied here, including as part of a runtime environment dump.
    """
    sections = dict(
        datasets=datasets,
        environment=environment,
        rewards=rewards,
        execution=execution,
        models=models,
        runtime=runtime,
    )
    return {"schema_version": 1, **{name: content_hash(value) for name, value in sections.items()}}


def _validate(identity: Any) -> None:
    if (
        not isinstance(identity, dict)
        or set(identity) != {"schema_version", *_SECTIONS}
        or type(identity["schema_version"]) is not int
        or identity["schema_version"] != 1
        or any(
            not isinstance(identity[key], str) or not re.fullmatch("[0-9a-f]{64}", identity[key])
            for key in _SECTIONS
        )
    ):
        raise ValueError("invalid llenvs run manifest schema")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in run manifest")
        result[key] = value
    return result


def _match(path: Path, identity: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError("run manifest cannot be a symlink")
    if not path.is_file():
        raise ValueError("existing checkpoint run is missing its llenvs run manifest")
    if path.stat().st_size > 4096:
        raise ValueError("invalid oversized run manifest")
    try:
        stored = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid llenvs run manifest JSON") from error
    _validate(stored)
    changed = [name for name in identity if identity[name] != stored[name]]
    if changed:
        raise ValueError(f"run manifest mismatch in: {', '.join(changed)}; use a new run")


def local_path(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be a local absolute path")
    return path.resolve()


def check_run(
    checkpoint_root: str | Path,
    identity: dict[str, Any],
    *,
    resume_mode: str | None,
    resume_path: str | Path | None = None,
    write: bool = False,
) -> None:
    """Validate before native allocation; publish once only for a real launch.

    ``latest`` retains native fresh-run behavior for an empty directory.
    ``from_path`` checks the manifest next to the selected global_step_N.
    This neither selects the latest checkpoint nor loads any native state.
    """
    _validate(identity)
    if resume_mode not in (None, "none", "latest", "from_path"):
        raise ValueError("unsupported resume_mode")
    root = local_path(checkpoint_root, "trainer.ckpt_path")
    path = root / _NAME
    if resume_mode == "from_path":
        if resume_path is None:
            raise ValueError("from_path requires trainer.resume_path")
        source = local_path(resume_path, "trainer.resume_path")
        if not source.is_dir() or not re.fullmatch(r"global_step_\d+", source.name):
            raise ValueError("resume_path must be an existing global_step_N directory")
        _match(source.parent / _NAME, identity)
    elif resume_path is not None:
        raise ValueError("resume_path is only meaningful with resume_mode=from_path")
    if path.exists() or path.is_symlink():
        if resume_mode in (None, "none"):
            raise FileExistsError("fresh run cannot overwrite an existing run manifest")
        _match(path, identity)
        return
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("nonempty checkpoint directory is missing its llenvs run manifest")
    if not write:
        return
    root.mkdir(parents=True, exist_ok=True)
    # Atomic exclusive publication avoids partial JSON or replacing a racing run.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=root, prefix=".llenvs-run-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_canonical_json(identity) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()
