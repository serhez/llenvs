"""Storage boundary for an exclusively owned call journal.

Names are relative POSIX paths. Serialization and call transitions belong to
CallStore; implementations provide durable bytes, discovery and ownership.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable


class RunStorage(Protocol):
    location: str
    run_dir: Path | None
    existing: bool

    def read(self, name: str) -> bytes | None: ...
    def write(self, name: str, payload: bytes) -> None: ...
    def names(self, prefix: str) -> Iterator[str]: ...
    def write_blob(self, name: str, chunks: bytes | Iterable[bytes]) -> dict: ...
    def blob_path(self, entry: dict, *, verify: bool) -> Path: ...
    def delete(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class Journal(Protocol):
    def acquire(self, fingerprint: str) -> RunStorage: ...
