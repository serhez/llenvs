"""Native GCS journal with conditional writes and confirmed-termination recovery.

Ownership never expires. Recovery requires the caller to establish that the old
worker has terminated; checking a separate owner object is not distributed
fencing against a paused worker or its outstanding network requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from importlib import import_module
from pathlib import Path

from ..store import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointInUseError,
    _canonical_json,
    _run_identity,
)

_ATTEMPTS = 3
_TIMEOUT = 30


def _transient(error):
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout

    return isinstance(
        error, (ConnectionError, TimeoutError, RequestsConnectionError, Timeout)
    ) or getattr(error, "code", None) in {
        408,
        429,
        500,
        502,
        503,
        504,
    }


class _GenerationConflictError(CheckpointError):
    pass


class _Objects:
    def __init__(self, client, bucket):
        self.client = client
        self.bucket = client.bucket(bucket)
        self.observed = OrderedDict()

    def remember(self, name, generation):
        self.observed[name] = generation
        self.observed.move_to_end(name)
        while len(self.observed) > 256:
            self.observed.popitem(last=False)

    def read(self, name):
        for attempt in range(_ATTEMPTS):
            blob = self.bucket.blob(name)
            try:
                try:
                    blob.reload(retry=None, timeout=_TIMEOUT)
                except Exception as exc:
                    if getattr(exc, "code", None) == 404:
                        self.remember(name, 0)
                        return None, 0
                    raise
                generation = int(blob.generation)
                data = blob.download_as_bytes(
                    if_generation_match=generation, retry=None, timeout=_TIMEOUT
                )
                self.remember(name, generation)
                return data, generation
            except Exception as exc:
                if attempt + 1 == _ATTEMPTS or not (
                    _transient(exc) or getattr(exc, "code", None) in {404, 412}
                ):
                    raise

    def generation(self, name):
        if name not in self.observed:
            for attempt in range(_ATTEMPTS):
                blob = self.bucket.blob(name)
                try:
                    blob.reload(retry=None, timeout=_TIMEOUT)
                    self.remember(name, int(blob.generation))
                    break
                except Exception as exc:
                    if getattr(exc, "code", None) == 404:
                        self.remember(name, 0)
                        break
                    if attempt + 1 == _ATTEMPTS or not _transient(exc):
                        raise
        return self.observed[name]

    def commit(self, name, expected, upload, reconcile):
        """Retry one operation, reconciling its exact committed content first."""
        for attempt in range(_ATTEMPTS):
            blob = self.bucket.blob(name, chunk_size=8 * 1024 * 1024)
            try:
                upload(blob, expected)
                generation = int(blob.generation)
                self.remember(name, generation)
                return generation
            except Exception as exc:
                if not (_transient(exc) or getattr(exc, "code", None) == 412):
                    raise
                generation, matches = reconcile()
                if matches:
                    self.remember(name, generation)
                    return generation
                if generation != expected:
                    raise _GenerationConflictError(f"Another operation changed {name}") from exc
                if attempt + 1 == _ATTEMPTS:
                    raise CheckpointError(f"Cannot confirm durable write: {name}") from exc

    def write(self, name, payload, expected):
        def reconcile():
            actual, generation = self.read(name)
            return generation, actual == payload

        return self.commit(
            name,
            expected,
            lambda blob, generation: blob.upload_from_string(
                payload, if_generation_match=generation, retry=None, timeout=_TIMEOUT
            ),
            reconcile,
        )

    def delete(self, name, expected):
        for attempt in range(_ATTEMPTS):
            try:
                self.bucket.blob(name).delete(
                    if_generation_match=expected, retry=None, timeout=_TIMEOUT
                )
                self.remember(name, 0)
                return
            except Exception as exc:
                if not (_transient(exc) or getattr(exc, "code", None) in {404, 412}):
                    raise
                actual, generation = self.read(name)
                if actual is None:
                    self.remember(name, 0)
                    return
                if generation != expected:
                    raise _GenerationConflictError(f"Another operation changed {name}") from exc
                if attempt + 1 == _ATTEMPTS:
                    raise CheckpointError(f"Cannot confirm deletion: {name}") from exc


class GCSJournal:
    """GCS location passed to ``CallStore.open`` instead of a filesystem path.

    Install ``llenvs[gcs]``. Supply an authorized Storage client or let the SDK
    use its native authentication. ``cache_dir`` holds disposable Batch files;
    absent an explicit directory, each open store owns a temporary directory.
    The bucket must already exist. Use a private, trusted prefix without an
    external expiration policy for journal objects.
    """

    def __init__(self, *, bucket: str, prefix: str, client=None, cache_dir=None):
        if not bucket or "/" in bucket:
            raise ValueError("bucket must be a bucket name")
        prefix = prefix.strip("/")
        if any(part in {"", ".", ".."} for part in prefix.split("/")):
            raise ValueError("prefix must be a nonempty object prefix")
        if client is None:
            try:
                client = import_module("google.cloud.storage").Client()
            except ImportError as exc:
                raise ImportError("GCS journals require the llenvs[gcs] extra") from exc
        self.bucket, self.prefix = bucket, prefix
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._objects = _Objects(client, bucket)

    def _owner_name(self, fingerprint):
        return f"{self.prefix}/.owners/{fingerprint}.json"

    def _owner(self, fingerprint):
        data, generation = self._objects.read(self._owner_name(fingerprint))
        if data is None:
            return None, generation
        try:
            value = json.loads(data)
            if value["run"] != fingerprint or value["schema"] != 1:
                raise ValueError("owner identity mismatch")
            if value["owner"] is not None and not isinstance(value["owner"], str):
                raise ValueError("invalid owner token")
            return value, generation
        except Exception as exc:
            raise CheckpointCorruptionError("Invalid GCS ownership record") from exc

    def _set_owner(self, fingerprint, state, generation):
        payload = (_canonical_json({**state, "operation": uuid.uuid4().hex}) + "\n").encode()
        return self._objects.write(self._owner_name(fingerprint), payload, generation)

    def current_owner(self, *, run_descriptor):
        fingerprint, _ = _run_identity(run_descriptor)
        state, _ = self._owner(fingerprint)
        return state["owner"] if state else None

    def acknowledge_owner_termination(
        self, *, run_descriptor, expected_owner, reason: str, confirmed_terminated: bool
    ):
        """Release exactly one terminated owner, retaining the caller's assertion.

        The caller must first confirm termination through its orchestrator. This
        does not authorize resubmission of any call with an unknown outcome.
        """
        if confirmed_terminated is not True or not reason.strip() or not expected_owner:
            raise ValueError("A reason, expected owner and confirmed termination are required")
        fingerprint, _ = _run_identity(run_descriptor)
        state, generation = self._owner(fingerprint)
        if state is None or state["owner"] != expected_owner:
            raise CheckpointError("Termination confirmation does not name the current owner")
        # The assertion is durable before ownership can be released. A crash
        # between these objects leaves the old owner in place, never a new draw.
        event = {"owner": expected_owner, "reason": reason, "confirmed_terminated": True}
        self._objects.write(
            f"{self.prefix}/.recoveries/{fingerprint}/{uuid.uuid4().hex}.json",
            _canonical_json(event).encode(),
            0,
        )
        self._set_owner(fingerprint, {**state, "owner": None}, generation)

    def acquire(self, fingerprint):
        previous, generation = self._owner(fingerprint)
        if previous is not None and previous["owner"] is not None:
            raise CheckpointInUseError(
                f"Run {fingerprint} has an owner; termination recovery required"
            )
        state = {
            "schema": 1,
            "run": fingerprint,
            "owner": uuid.uuid4().hex,
            "initialized": previous.get("initialized", False) if previous else False,
        }
        try:
            generation = self._set_owner(fingerprint, state, generation)
        except _GenerationConflictError as exc:
            raise CheckpointInUseError(f"Run {fingerprint} was acquired by another owner") from exc
        return _GCSRun(self, fingerprint, state, generation)


class _GCSRun:
    run_dir = None

    def __init__(self, journal, fingerprint, owner, generation):
        self.journal, self.fingerprint = journal, fingerprint
        self.owner, self.owner_generation = owner, generation
        self.existing = owner["initialized"]
        self.prefix = f"{journal.prefix}/{fingerprint}/"
        self.location = f"gs://{journal.bucket}/{self.prefix}"
        self._objects = journal._objects
        self._mutex = threading.RLock()
        self._closed = False
        self._temporary = None
        if journal.cache_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="llenvs-gcs-")
            self.cache_dir = Path(self._temporary.name)
        else:
            self.cache_dir = journal.cache_dir / hashlib.sha256(self.location.encode()).hexdigest()

    def _check_owner(self):
        if self._closed:
            raise CheckpointError("GCS journal is closed")
        current, generation = self.journal._owner(self.fingerprint)
        if (
            current is None
            or current["owner"] != self.owner["owner"]
            or generation != self.owner_generation
        ):
            raise CheckpointError("GCS journal ownership changed")

    def _set_initialized(self, initialized):
        state = {**self.owner, "initialized": initialized}
        self.owner_generation = self.journal._set_owner(
            self.fingerprint, state, self.owner_generation
        )
        self.owner = state

    def read(self, name):
        data, _ = self._objects.read(self.prefix + name)
        if data is None:
            return None
        header, payload = data.split(b"\n", 1)
        if json.loads(header)["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"GCS object checksum mismatch: {name}")
        return payload

    def write(self, name, payload):
        with self._mutex:
            self._check_owner()
            key = self.prefix + name
            expected = self._objects.generation(key)
            header = _canonical_json(
                {"operation": uuid.uuid4().hex, "sha256": hashlib.sha256(payload).hexdigest()}
            ).encode()
            self._objects.write(key, header + b"\n" + payload, expected)
            if name == "manifest.json" and not self.owner["initialized"]:
                self._set_initialized(True)

    def names(self, prefix):
        for blob in self._objects.client.list_blobs(
            self.journal.bucket, prefix=self.prefix + prefix, retry=None, timeout=_TIMEOUT
        ):
            yield blob.name[len(self.prefix) :]

    def _download(self, name, path):
        for attempt in range(_ATTEMPTS):
            blob = self._objects.bucket.blob(name, chunk_size=8 * 1024 * 1024)
            try:
                blob.reload(retry=None, timeout=_TIMEOUT)
                generation = int(blob.generation)
                blob.download_to_filename(
                    path, if_generation_match=generation, retry=None, timeout=_TIMEOUT
                )
                return generation
            except Exception as exc:
                if attempt + 1 == _ATTEMPTS or not (
                    _transient(exc) or getattr(exc, "code", None) in {404, 412}
                ):
                    raise

    def write_blob(self, name, chunks):
        from llenvs.inference.records import file_digest

        with self._mutex:
            self._check_owner()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary:
                path = Path(temporary) / "upload"
                with path.open("wb") as stream:
                    for chunk in [chunks] if isinstance(chunks, bytes) else chunks:
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                key = self.prefix + "blobs/" + name
                expected = self._objects.generation(key)

                def reconcile():
                    observed = Path(temporary) / "observed"
                    try:
                        generation = self._download(key, observed)
                    except Exception as exc:
                        if getattr(exc, "code", None) == 404:
                            return 0, False
                        raise
                    return generation, (
                        observed.stat().st_size == size
                        and file_digest(observed) == digest.hexdigest()
                    )

                self._objects.commit(
                    key,
                    expected,
                    lambda blob, generation: blob.upload_from_filename(
                        path, if_generation_match=generation, retry=None, timeout=_TIMEOUT
                    ),
                    reconcile,
                )
                os.replace(path, self.cache_dir / name)
            return {"name": name, "sha256": digest.hexdigest(), "size": size}

    def blob_path(self, entry, *, verify):
        from llenvs.inference.records import file_digest

        with self._mutex:
            self._check_owner()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            destination = self.cache_dir / entry["name"]
            # Cache contents are disposable. Always materialize authoritative
            # bytes; a cached file must not hide a missing or changed object.
            with tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary:
                path = Path(temporary) / "download"
                self._download(self.prefix + "blobs/" + entry["name"], path)
                if verify and (
                    path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]
                ):
                    raise CheckpointCorruptionError("Retained Batch file checksum mismatch")
                os.replace(path, destination)
            return destination

    def delete(self):
        with self._mutex:
            self._check_owner()
            for name in self.names(""):
                key = self.prefix + name
                self._objects.delete(key, self._objects.generation(key))
            self._set_initialized(False)

    def close(self):
        with self._mutex:
            if self._closed:
                return
            try:
                self._check_owner()
                self.journal._set_owner(
                    self.fingerprint, {**self.owner, "owner": None}, self.owner_generation
                )
            finally:
                self._closed = True
                if self._temporary is not None:
                    self._temporary.cleanup()
