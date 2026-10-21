"""Storage boundary for an exclusively owned call journal.

Names are relative POSIX paths. Serialization and call transitions belong to
CallStore; implementations provide durable bytes, discovery and ownership.
"""

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Protocol, runtime_checkable


class RunStorage(Protocol):
    location: str
    run_dir: Path | None
    existing: bool
    max_concurrency: int

    def read(self, name: str) -> bytes | None: ...
    def write(self, name: str, payload: bytes) -> None: ...
    def submit_writes(self, records: Iterable[tuple[str, bytes]]) -> Iterator[Future[None]]:
        """Yield each accepted write's durable future in input order.

        Accepted futures remain valid if further admission raises. Maintenance
        waits for them even after this iterator returns or raises.
        """
        ...

    def pin(self, name: str) -> None: ...
    def unpin(self, name: str) -> None: ...
    def names(self, prefix: str) -> Iterator[str]: ...
    def write_blob(self, name: str, chunks: bytes | Iterable[bytes]) -> dict: ...
    def blob_path(self, entry: dict, *, verify: bool) -> Path: ...
    def delete(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class Journal(Protocol):
    def acquire(self, fingerprint: str) -> RunStorage: ...
