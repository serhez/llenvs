"""Filesystem ownership and durable atomic journal writes."""

import fcntl
import hashlib
import os
import shutil
import uuid
from pathlib import Path


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class FilesystemJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def acquire(self, fingerprint):
        from ..store import CheckpointInUseError

        (self.root / ".locks").mkdir(parents=True, exist_ok=True)
        handle = (self.root / ".locks" / (fingerprint + ".lock")).open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise CheckpointInUseError(f"Run {fingerprint} is already open") from exc
        return _FilesystemRun(self.root / fingerprint, handle)


class _FilesystemRun:
    max_concurrency = 1

    def __init__(self, run_dir, handle):
        self.run_dir = run_dir
        self.location = str(run_dir)
        self.existing = run_dir.exists()
        self._handle = handle

    def read(self, name):
        try:
            return (self.run_dir / name).read_bytes()
        except FileNotFoundError:
            return None

    def pin(self, name):
        pass

    def unpin(self, name):
        pass

    def write(self, name, payload):
        _atomic_write_bytes(self.run_dir / name, payload)

    def write_many(self, records):
        records = list(records)
        errors = []
        for name, payload in records:
            try:
                self.write(name, payload)
                errors.append(None)
            except BaseException as exc:
                errors.extend([exc] * (len(records) - len(errors)))
                break
        return errors

    def names(self, prefix):
        for path in (self.run_dir / prefix).rglob("*"):
            if path.is_file():
                yield path.relative_to(self.run_dir).as_posix()

    def write_blob(self, name, chunks):
        directory = self.run_dir / "blobs"
        directory.mkdir(exist_ok=True)
        temporary = directory / ("." + name + "." + uuid.uuid4().hex + ".tmp")
        digest, size = hashlib.sha256(), 0
        try:
            with temporary.open("xb") as handle:
                for chunk in [chunks] if isinstance(chunks, bytes) else chunks:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / name)
            _fsync_directory(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return {"name": name, "sha256": digest.hexdigest(), "size": size}

    def blob_path(self, entry, *, verify):
        from llenvs.inference.records import file_digest

        from ..store import CheckpointCorruptionError

        path = self.run_dir / "blobs" / entry["name"]
        if verify and (
            path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]
        ):
            raise CheckpointCorruptionError(
                f"Retained Batch file checksum mismatch: {entry['name']}"
            )
        return path

    def delete(self):
        shutil.rmtree(self.run_dir)

    def close(self):
        self._handle.close()
