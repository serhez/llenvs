"""Checksummed call journal with shared replay and provenance semantics.

Pickles are trusted journal artifacts. Checksums detect accidental corruption.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import enum
import hashlib
import json
import math
import pickle
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from llenvs.inference.decisions.protocol import DecisionRequest
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    PartialBatchError,
    StopReason,
    encode_backend_error,
    restore_backend_error,
)

from .storage import Journal
from .writer import GroupWriter, MaintenanceGate, operation


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalize(value: Any) -> Any:
    """Convert supported values to a deterministic JSON-compatible form."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if callable(value):
        raise TypeError("Cannot build a stable checkpoint identity for a callable policy option")
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {
            "__bytes__": hashlib.sha256(value).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, enum.Enum):
        return {
            "__enum__": _qualified_type(value),
            "name": value.name,
            "value": _normalize(value.value),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Absent reasoning, and a decision's absent cache settings, must not
        # change the identity of ordinary saved calls.
        return {
            "__dataclass__": _qualified_type(value),
            "fields": {
                field.name: _normalize(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if not (
                    isinstance(value, ChatMessage)
                    and field.name in {"reasoning", "reasoning_details"}
                    and not getattr(value, field.name)
                )
                and not (
                    isinstance(value, DecisionRequest)
                    and field.name in {"cache_mode", "cache_key"}
                    and getattr(value, field.name) is None
                )
            },
        }
    if isinstance(value, Mapping):
        normalized_items = [[_normalize(key), _normalize(inner)] for key, inner in value.items()]
        normalized_items.sort(key=lambda item: _canonical_json(item[0]))
        return {"__mapping__": normalized_items}
    if isinstance(value, tuple):
        return {"__tuple__": [_normalize(item) for item in value]}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalize(item) for item in value]
        items.sort(key=_canonical_json)
        return {"__set__": items}
    if hasattr(value, "__dict__"):
        return {
            "__object__": _qualified_type(value),
            "fields": _normalize(vars(value)),
        }
    raise TypeError(
        f"Cannot build a stable prediction checkpoint identity for {_qualified_type(value)}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    # Model inputs may legitimately contain credential-shaped field names.
    # Transport credentials are excluded before hashing the private request.
    normalized = _normalize(value)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _encode_pickle(value: Any) -> bytes:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return pickle.dumps(
        {"payload_sha256": hashlib.sha256(payload).hexdigest(), "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _decode_pickle(data: bytes) -> Any:
    envelope = pickle.loads(data)
    payload = envelope["payload"]
    if hashlib.sha256(payload).hexdigest() != envelope["payload_sha256"]:
        raise ValueError("Checksum mismatch")
    return pickle.loads(payload)


def _run_identity(run_descriptor):
    from llenvs.inference.records import credential_free

    descriptor = _normalize(credential_free(run_descriptor))
    return _fingerprint({"schema": 1, "run": descriptor}), descriptor


class CheckpointError(RuntimeError):
    """Fatal journal failure; no further requests may be dispatched."""


class CheckpointCorruptionError(CheckpointError):
    """Saved records cannot be trusted for automatic replay."""


class CheckpointInUseError(CheckpointError):
    """Another process owns this run."""


class UnknownOutcomeError(CheckpointError):
    """Dispatch may have succeeded without a durable receipt."""

    def __init__(self, key: str):
        self.key = key
        self.metadata: dict[str, Any] = {}
        super().__init__(f"Outcome unknown for {key}; explicit recovery decision required")


class _MissingRecordError(KeyError):
    """Storage confirmed absence, rather than an error decoding retained bytes."""


class _RecordIndex(MutableMapping):
    """Keep active reservations and a bounded cache; completed draws live on disk."""

    def __init__(self, store):
        self.store = store
        self.volatile = {}
        self.cache = OrderedDict()

    def path(self, key):
        return f"calls/{key[:2]}/{key}.pkl"

    def __getitem__(self, key):
        with self.store._key_lock(key):
            with self.store._mutex:
                if key in self.volatile:
                    return self.volatile[key]
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
            try:
                record = self.store._load(self.path(key))
            except _MissingRecordError:
                raise KeyError(key) from None
            except Exception as exc:
                self.store._fail(exc)
                raise CheckpointCorruptionError(f"Invalid retained call {key}") from exc
            with self.store._mutex:
                self.retain(key, record)
                return record

    def __setitem__(self, key, value):
        with self.store._mutex:
            self.volatile[key] = value

    def retain(self, key, value):
        if key in self.store._inflight or key in self.store._pins:
            self.cache.pop(key, None)
            self.volatile[key] = value
            return
        self.volatile.pop(key, None)
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > max(64, self.store._storage.max_concurrency * 2):
            self.cache.popitem(last=False)

    def __delitem__(self, key):
        raise TypeError("Retained records can only be removed through mark_complete(delete=True)")

    def __iter__(self):
        with self.store._mutex:
            active = set(self.volatile)
        yield from active
        for name in self.store._storage.names("calls/"):
            if name.endswith(".pkl") and PurePosixPath(name).stem not in active:
                yield PurePosixPath(name).stem

    def __len__(self):
        return sum(1 for _ in self)


def _validate_result(result: Any) -> None:
    nested = getattr(result, "result", None)
    if isinstance(nested, GenerationResult):
        _validate_result(nested)
    metadata = getattr(result, "metadata", {})
    if (
        isinstance(result, BaseException)
        or (isinstance(result, GenerationResult) and result.finish_reason == StopReason.ERROR)
        or any(metadata.get(key) is not None for key in ("error", "provider_error"))
        or any(
            str(metadata.get(key, "")).lower() == "error"
            for key in ("finish_reason", "native_finish_reason")
        )
    ):
        raise ValueError("Explicit provider failure cannot be cached as a successful outcome")


def _failure(error: BaseException) -> dict[str, Any]:
    return encode_backend_error(error)


class CallStore:
    """One exclusively owned run with short journal locks and bounded writes."""

    @classmethod
    def open(
        cls,
        root: str | Path | Journal,
        *,
        run_descriptor: Mapping[str, Any],
        validation: Literal["full", "lazy"] = "full",
    ) -> CallStore:
        from .storage.filesystem import FilesystemJournal

        if validation not in {"full", "lazy"}:
            raise ValueError("validation must be 'full' or 'lazy'")
        fingerprint, descriptor = _run_identity(run_descriptor)
        if isinstance(root, Journal):
            journal = root
        else:
            root = Path(root)
            journal = FilesystemJournal(root)
        storage = journal.acquire(fingerprint)
        try:
            return cls(root, fingerprint, descriptor, storage, validation=validation)
        except BaseException:
            storage.close()
            raise

    def __init__(self, root, fingerprint, descriptor, storage, *, validation):
        self.root, self.fingerprint = root, fingerprint
        self._storage = storage
        self.run_dir = storage.run_dir
        self._mutex = threading.RLock()
        self._gate = MaintenanceGate()
        self._key_locks = {}
        self._pins = {}
        self._batch_lock = threading.RLock()
        self._closed = False
        self._deleted = False
        self._poison: BaseException | None = None
        self._occurrences: dict[str, int] = {}
        self._inflight: dict[str, Future] = {}
        self._records = _RecordIndex(self)
        self._journals: dict[str, dict] = {}
        self._intent_submissions = {}
        self._reader = ThreadPoolExecutor(
            max_workers=storage.max_concurrency, thread_name_prefix="llenvs-read"
        )
        manifest = {"schema": 1, "fingerprint": fingerprint, "run_descriptor": descriptor}
        try:
            if storage.existing:
                if json.loads(storage.read("manifest.json")) != manifest:
                    raise ValueError("manifest mismatch")
                if validation == "full":
                    for _ in self._parallel(
                        self._load,
                        (name for name in storage.names("calls/") if name.endswith(".pkl")),
                    ):
                        pass
                for name in storage.names("lifecycle/"):
                    if not name.endswith(".pkl"):
                        continue
                    event = _decode_pickle(storage.read(name))
                    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                        raise ValueError("Invalid lifecycle event")
                for name in storage.names("journals/"):
                    if not name.endswith(".pkl"):
                        continue
                    journal = _decode_pickle(storage.read(name))
                    if journal["run"] != fingerprint or journal["id"] != PurePosixPath(name).stem:
                        raise ValueError("submission journal identity mismatch")
                    self._journals[journal["id"]] = journal
                    if validation == "full":
                        for blob in [journal["input_blob"], *journal.get("raw_files", {}).values()]:
                            self.blob_path(blob)

                        def check_key(key):
                            if self._records[key]["request"].get("scope") != "batch":
                                raise ValueError("Submission references a non-Batch call")

                        for _ in self._parallel(check_key, journal["keys"]):
                            pass
                    if journal["state"] == "intent":
                        for key in journal["keys"]:
                            self._intent_submissions[key] = journal["id"]
            else:
                storage.write("manifest.json", (_canonical_json(manifest) + "\n").encode())
        except Exception as exc:
            self._reader.shutdown(wait=True)
            raise CheckpointCorruptionError(
                f"Invalid call store {storage.location}: {exc}"
            ) from exc
        self._commits = GroupWriter(storage, self._fail_waiters)
        self._writer = ThreadPoolExecutor(
            max_workers=self._commits.capacity, thread_name_prefix="llenvs-observer"
        )
        self._write_slots = threading.BoundedSemaphore(self._commits.capacity)

    @contextmanager
    def _key_lock(self, key):
        with self._gate.shared():
            with self._mutex:
                entry = self._key_locks.setdefault(key, [threading.RLock(), 0])
                entry[1] += 1
            try:
                with entry[0]:
                    yield
            finally:
                with self._mutex:
                    entry[1] -= 1
                    if not entry[1]:
                        self._key_locks.pop(key)

    def _parallel(self, function, values):
        """Bound discovery, futures and decoded objects independently of store size."""
        with self._gate.shared():
            values = iter(values)
            while chunk := list(islice(values, self._storage.max_concurrency)):
                if self._storage.max_concurrency == 1:
                    yield from map(function, chunk)
                else:
                    yield from self._reader.map(function, chunk)

    def _fail(self, error):
        with self._mutex:
            self._poison = self._poison or error
        commits = getattr(self, "_commits", None)
        if commits is None:
            # Startup validation can fail before the writer exists.
            self._fail_waiters(error)
        else:
            # Do not wait under the metadata lock: admitted completions need it.
            commits.fail(error)

    def _fail_waiters(self, error):
        with self._mutex:
            self._poison = self._poison or error
            for future in self._inflight.values():
                if not future.done():
                    future.set_exception(CheckpointError("Call store failed to persist progress"))

    def _release(self, key, future=None):
        with self._mutex:
            if future is None or self._inflight.get(key) is future:
                if self._inflight.pop(key, None) is not None:
                    self._storage.unpin(self._records.path(key))
                if (
                    key in self._records.volatile
                    and self._records.volatile[key]["state"] != "reserved"
                ):
                    self._records.retain(key, self._records.volatile[key])

    @contextmanager
    def _hold_records(self):
        keys = set()

        def hold(key):
            with self._mutex:
                if key not in keys:
                    keys.add(key)
                    self._pins[key] = self._pins.get(key, 0) + 1
                    self._storage.pin(self._records.path(key))
                    if key in self._records.cache:
                        self._records.volatile[key] = self._records.cache.pop(key)

        with self._gate.shared():
            try:
                yield hold
            finally:
                with self._mutex:
                    for key in keys:
                        self._storage.unpin(self._records.path(key))
                        self._pins[key] -= 1
                        if not self._pins[key]:
                            self._pins.pop(key)
                            if key in self._records.volatile:
                                self._records.retain(key, self._records.volatile[key])

    def _load(self, path):
        data = self._storage.read(path)
        if data is None:
            raise _MissingRecordError(path)
        record = _decode_pickle(data)
        if record["key"] != PurePosixPath(path).stem or record["run"] != self.fingerprint:
            raise ValueError("record identity mismatch")
        if path != self._records.path(record["key"]):
            raise ValueError("record shard mismatch")
        if _fingerprint(record["request"]) != record["key"]:
            raise ValueError("request identity mismatch")
        if record["state"] not in {
            "reserved",
            "started",
            "received",
            "completed",
            "failed",
            "pending",
            "phased",
        }:
            raise ValueError("invalid journal state")
        if record["state"] == "completed":
            _validate_result(record["result"])
        return record

    def _check(self):
        if self._closed:
            raise CheckpointError("Call store is closed")
        if self._deleted:
            raise CheckpointError("Call store was deleted")
        error = self._poison
        commits = getattr(self, "_commits", None)
        if error is None and commits is not None:
            error = commits.error
        if error is not None:
            raise CheckpointError("Call store is poisoned by a storage failure") from error

    def reserve(self, request: dict, repeat_id) -> tuple[str, Future, bool]:
        return self._reserve_prepared(self._prepare_request(request, repeat_id))

    def reserve_many(self, requests):
        prepared = (self._prepare_request(request, repeat) for request, repeat in requests)
        return list(self._parallel(self._reserve_prepared, prepared))

    def _reserve_entry(self, prepared):
        with self._key_lock(prepared[0]):
            key, future, owner = self._reserve_prepared(prepared)
            provenance = self._provenance(key, self._result_source(key, future, owner))
            return key, future, owner, provenance

    def _reserve_entries(self, requests):
        if len(requests) == 1:
            return [self._reserve_entry(self._prepare_request(*requests[0]))]
        prepared = (self._prepare_request(request, repeat) for request, repeat in requests)
        return list(self._parallel(self._reserve_entry, prepared))

    def _prepare_request(self, request, repeat_id):
        with self._mutex:
            self._check()
            request = dict(request)
            params = request.get("arguments", {}).get("params")
            from llenvs.inference.protocol import SamplingParams

            if isinstance(params, SamplingParams):
                from llenvs.inference.records import credential_free

                extra = {
                    key: value
                    for key, value in params.extra.items()
                    if key
                    not in {
                        "callbacks",
                        "callback",
                        "logger_fn",
                        "http_client",
                        "client",
                        "async_client",
                    }
                }
                for name in ("extra_headers", "headers"):
                    if name in extra:
                        extra[name] = credential_free(extra[name])
                request["arguments"] = {
                    **request["arguments"],
                    "params": dataclasses.replace(params, extra=extra),
                }
            request = copy.deepcopy(request)
            base = _fingerprint(request)
            if repeat_id is None:
                occurrence = self._occurrences.get(base, 0)
                self._occurrences[base] = occurrence + 1
                request["repeat"] = {"occurrence": occurrence}
            else:
                if type(repeat_id) not in (str, int):
                    raise TypeError("repeat_id must be an int, str or None")
                request["repeat"] = {"explicit": repeat_id}
            key = _fingerprint(request)
            return key, request

    def _reserve_prepared(self, prepared):
        key, request = prepared
        with self._key_lock(key):
            with self._mutex:
                self._check()
                if key in self._inflight:
                    return key, self._inflight[key], False
                future = Future()
                self._inflight[key] = future
                self._storage.pin(self._records.path(key))
            record = self._records.get(key)
            with self._mutex:
                self._check()
                if record is not None:
                    if record["state"] == "completed":
                        future.set_result(copy.deepcopy(record["result"]))
                        return key, future, False
                    if record["state"] in {"started", "pending"}:
                        future.set_exception(UnknownOutcomeError(key))
                        return key, future, False
                    if record["state"] == "failed":
                        future.set_exception(restore_backend_error(record["error"]))
                        return key, future, False
                else:
                    self._records[key] = {
                        "key": key,
                        "run": self.fingerprint,
                        "request": request,
                        "state": "reserved",
                        "attempts": [],
                    }
            return key, future, True

    def update(self, key, state, **values):
        with self._key_lock(key):
            self._check()
            record = copy.deepcopy(self._records[key])
            record.update(values, state=state)
            if state == "completed":
                _validate_result(record["result"])

            def committed():
                with self._mutex:
                    self._records.retain(key, record)
                    future = self._inflight.get(key)
                    if state == "completed" and future is not None and not future.done():
                        future.set_result(copy.deepcopy(record["result"]))

            try:
                self._commits.submit(
                    self._records.path(key), _encode_pickle(record), committed
                ).result()
            except Exception as exc:
                self._fail(exc)
                raise CheckpointError(f"Cannot persist {state} for {key}") from exc

    def update_many(self, updates):
        """Wait for every bounded update, retaining confirmed partial progress."""

        def update(entry):
            key, state, values = entry
            self.update(key, state, **values)

        self._apply_many(update, updates)

    @operation
    def _apply_many(self, function, values):
        error = None

        def invoke(value):
            try:
                return function(value)
            finally:
                self._write_slots.release()

        for chunk in self._chunks(values):
            futures = []
            for value in chunk:
                self._write_slots.acquire()
                futures.append(self._writer.submit(invoke, value))
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error

    def _chunks(self, values):
        values = iter(values)
        while chunk := list(islice(values, self._commits.capacity)):
            yield chunk

    def _result_source(self, key, future, owner):
        if owner:
            return "receipt" if self._records[key]["state"] == "received" else "backend"
        return "store" if future.done() else "shared"

    def _result(self, outcome, provenance):
        """Detach invocation provenance without changing the retained outcome."""
        result = copy.deepcopy(outcome)
        target = getattr(result, "result", result)
        target.metadata["persistence"] = dict(provenance)
        return result

    def _provenance(self, key, source):
        previous_attempts = len(self._records[key]["attempts"])
        return {
            "run_fingerprint": self.fingerprint,
            "call_key": key,
            "source": source,
            "previous_attempts": previous_attempts,
        }

    def _error(self, error, provenance):
        """Keep native exception semantics and detach per-invocation metadata."""
        try:
            result = copy.copy(error)
        except TypeError:
            # SDK exceptions often require keyword-only response/request objects.
            # Bypass __init__; keep those objects without copying live clients.
            result = BaseException.__new__(type(error), *error.args)
            result.__dict__.update(vars(error))
        result.args = error.args
        result.__traceback__ = error.__traceback__
        result.__cause__ = error.__cause__
        result.__context__ = error.__context__
        result.__suppress_context__ = error.__suppress_context__
        result.metadata = {
            **getattr(error, "metadata", {}),
            "persistence": dict(provenance),
        }
        return result

    def write_journal(self, journal: dict):
        """Commit a submission manifest, including exact upload and output bytes."""
        with self._key_lock("journal/" + journal["id"]):
            self._check()
            identifier = journal["id"]
            if not identifier.isalnum() or journal["run"] != self.fingerprint:
                raise ValueError("Invalid journal identity")
            try:
                self._storage.write(f"journals/{identifier}.pkl", _encode_pickle(journal))
            except Exception as exc:
                self._fail(exc)
                raise CheckpointError("Cannot persist submission journal") from exc
            with self._mutex:
                self._journals[identifier] = copy.deepcopy(journal)
                for key in journal["keys"]:
                    if journal["state"] == "intent":
                        self._intent_submissions[key] = identifier
                    elif self._intent_submissions.get(key) == identifier:
                        self._intent_submissions.pop(key)

    @operation
    def write_blob(self, name, chunks):
        """Stream exact upload/output bytes into an atomic, checksummed artifact.

        A failure of ``chunks`` itself, such as a provider stream that breaks
        off, is raised unchanged and leaves the store open; only a failure to
        persist the bytes poisons it.
        """
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in name
        ):
            raise ValueError("Invalid blob name")
        self._check()
        failed = []

        def source():
            try:
                yield from [chunks] if isinstance(chunks, bytes) else chunks
            except Exception as exc:
                failed.append(exc)
                raise

        try:
            return self._storage.write_blob(name, source())
        except Exception as exc:
            if failed and exc is failed[0]:
                raise
            self._fail(exc)
            raise CheckpointError("Cannot persist exact Batch file bytes") from exc

    @operation
    def blob_path(self, entry, *, verify=True):
        name = entry["name"]
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in name
        ):
            raise CheckpointCorruptionError("Invalid retained blob name")
        self._check()
        try:
            return self._storage.blob_path(entry, verify=verify)
        except Exception as exc:
            self._fail(exc)
            raise CheckpointCorruptionError(f"Invalid retained Batch file: {name}") from exc

    async def update_async(self, key, state, **values):
        await asyncio.to_thread(self._write_slots.acquire)
        try:
            await asyncio.wrap_future(self._writer.submit(self.update, key, state, **values))
        finally:
            self._write_slots.release()

    def acknowledge_resubmission(
        self, key: str, *, reason: str, accept_duplicate_expenditure: bool
    ):
        """Record a caller decision before making an uncertain draw eligible again."""
        if not accept_duplicate_expenditure or not reason.strip():
            raise ValueError(
                "A reason and explicit acceptance of duplicate expenditure are required"
            )
        with self._key_lock(key):
            self._check()
            record = self._records[key]
            if record["state"] not in {"started", "failed"}:
                raise ValueError("Only unresolved or definitely failed calls can be resubmitted")
            history = record.get("recovery_decisions", []) + [
                {"reason": reason, "duplicate_expenditure_accepted": True}
            ]
            self.update(
                key,
                "reserved",
                recovery_decisions=history,
                attempts=record["attempts"]
                + [{k: v for k, v in record.items() if k not in {"request", "attempts"}}],
                error=None,
                receipt=None,
                outcome_unknown=False,
                submission=None,
            )
            self._release(key)

    @operation
    def lifecycle(self, event: Mapping[str, Any]):
        """Persist an independent provenance event without changing request identity."""
        self._check()
        try:
            self._storage.write(f"lifecycle/{uuid.uuid4().hex}.pkl", _encode_pickle(dict(event)))
        except Exception as exc:
            self._fail(exc)
            raise CheckpointError("Cannot persist lifecycle event") from exc

    def mark_complete(self, *, delete: bool = False):
        with self._gate.exclusive():
            self._check()
            if any(
                record["state"] not in {"completed", "failed"} for record in self._records.values()
            ):
                raise UnknownOutcomeError("run has unresolved requests")
            try:
                self._storage.write(
                    "complete.json", _canonical_json({"retained": not delete}).encode()
                )
                if delete:
                    self._storage.delete()
                    self._deleted = True
            except Exception as exc:
                self._poison = exc
                raise CheckpointError("Cannot persist run completion") from exc

    def close(self):
        if not self._closed:
            self._writer.shutdown(wait=True)
            with self._gate.exclusive():
                if not self._closed:
                    self._commits.close()
                    self._reader.shutdown(wait=True)
                    self._closed = True
                    for key, future in list(self._inflight.items()):
                        if not future.done():
                            future.set_exception(CheckpointError("Call store is closed"))
                        self._release(key, future)
                    self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _Observer:
    def __init__(self, store, keys):
        self.store, self.keys = store, keys
        self.provenance = [store._provenance(key, "backend") for key in keys]

    def started(self, index, attempt):
        key = self.keys[index]
        record = self.store._records[key]
        history = record["attempts"]
        if record["state"] != "reserved":
            history = history + [
                {
                    k: copy.deepcopy(v)
                    for k, v in record.items()
                    if k in {"state", "attempt", "error", "receipt"}
                }
            ]
        self.store.update(
            key, "started", attempt=attempt, attempts=history, error=None, receipt=None
        )
        self.store._check()

    def received(self, index, receipt):
        self.store.update(self.keys[index], "received", receipt=copy.deepcopy(receipt))

    def begin_phases(self, index):
        self.store.update(self.keys[index], "phased")

    def run_phase(self, index, name, arguments, delegate, reparse):
        key, future, owner, provenance = self.store._reserve_entry(
            self.store._prepare_request(
                {"parent": self.keys[index], "phase": name, "arguments": arguments}, 0
            )
        )
        if owner:
            child = _Observer(self.store, [key])
            try:
                record = self.store._records[key]
                if record["state"] == "received":
                    outcome = reparse(record["receipt"])
                else:
                    outcome = delegate(child)
                if not future.done():
                    child.completed(0, outcome)
            except BaseException as exc:
                if not future.done():
                    future.set_exception(
                        UnknownOutcomeError(key)
                        if self.store._records[key]["state"] == "started"
                        else exc
                    )
        try:
            return self.store._result(future.result(), provenance)
        except UnknownOutcomeError as exc:
            raise self.store._error(exc, provenance) from exc
        finally:
            if future.done():
                self.store._release(key, future)

    def completed(self, index, outcome):
        self.store.update(self.keys[index], "completed", result=outcome)

    def failed(self, index, error):
        key = self.keys[index]
        record = self.store._records[key]
        # Keep the receipt available for local parsing even if the parser failed.
        if record["state"] == "received":
            from llenvs.inference.protocol import (
                LogprobsNotReturnedError,
                MalformedResponseError,
                RefusedByPolicyError,
            )

            self.store.update(
                key,
                "failed"
                if isinstance(
                    error, (RefusedByPolicyError, MalformedResponseError, LogprobsNotReturnedError)
                )
                else "received",
                error=_failure(error),
            )
            return
        if record["state"] != "received":
            status = getattr(error, "status_code", None) or getattr(
                getattr(error, "response", None), "status_code", None
            )
            if isinstance(status, int):
                self.store.update(key, "failed", error=_failure(error))
            elif record["state"] == "started":
                self.store.update(key, "started", error=_failure(error), outcome_unknown=True)
                failure = self.store._error(UnknownOutcomeError(key), self.provenance[index])
                future = self.store._inflight.get(key)
                if future is not None and not future.done():
                    future.set_exception(failure)
                raise failure from error

    async def notify(self, method, index, value):
        # The bounded writer keeps fsync off the HTTP event loop.
        await asyncio.to_thread(self.store._write_slots.acquire)
        try:
            await asyncio.wrap_future(
                self.store._writer.submit(getattr(self, method), index, value)
            )
        finally:
            self.store._write_slots.release()


class PersistentBackend:
    """Replay individual draws independently of batch grouping and ordering.

    Implicit occurrences require the original call order. Explicit repeat IDs
    identify draws across subsets, regrouping, and duplicate batch positions.
    """

    def __init__(
        self,
        backend,
        store: CallStore,
        *,
        checkpoint_granularity="request",
        scope="default",
        persist=None,
    ):
        if checkpoint_granularity not in {"request", "batch"}:
            raise ValueError("Unknown checkpoint_granularity")
        if (
            checkpoint_granularity == "request"
            and not backend.capabilities.supports_request_progress
        ):
            raise ValueError("Request checkpoint granularity needs native per-request progress")
        self._backend, self._store = backend, store
        self._closed = False
        self._instance_id = uuid.uuid4().hex
        self._granularity, self._scope, self._persist = checkpoint_granularity, scope, persist
        for name in ("_client", "_async_client"):
            client = getattr(backend, name, None)
            if client is not None and getattr(client, "max_retries", 0) != 0:
                raise ValueError("Persisted calls require SDK max_retries=0")
        if hasattr(backend, "_num_retries") and backend._num_retries != 0:
            raise ValueError("Persisted LiteLLM calls require num_retries=0")
        from datetime import UTC, datetime

        store.lifecycle(
            {
                "event": "backend_instance",
                "instance_id": self._instance_id,
                "observed_at": datetime.now(UTC).isoformat(),
                **backend.descriptor(),
            }
        )

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def close(self):
        if self._closed:
            return
        from datetime import UTC, datetime

        try:
            close = getattr(self._backend, "close", None)
            if close is not None:
                close()
        except Exception as exc:
            self._store.lifecycle(
                {
                    "event": "backend_stop_failed",
                    "instance_id": self._instance_id,
                    "error": _failure(exc),
                    "observed_at": datetime.now(UTC).isoformat(),
                }
            )
            raise
        else:
            self._closed = True
            self._store.lifecycle(
                {
                    "event": "backend_stopped",
                    "instance_id": self._instance_id,
                    "stopped_at": datetime.now(UTC).isoformat(),
                }
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def generate_chat(self, messages, params, *, repeat_id=None):
        return self.generate_chat_batch([messages], params, repeat_ids=[repeat_id])[0]

    def generate(self, prompts, params, *, repeat_ids=None):
        return self._batch(
            "completion",
            prompts,
            {"params": params},
            repeat_ids,
            lambda batch, observer: self._backend.generate(
                batch, params, **({"observer": observer} if observer else {})
            ),
        )

    def generate_single(self, prompt, params, *, repeat_id=None):
        return self.generate([prompt], params, repeat_ids=[repeat_id])[0]

    def generate_with_logprobs(self, prompts, params, *, repeat_ids=None):
        return self.generate(
            prompts, dataclasses.replace(params, logprobs=True), repeat_ids=repeat_ids
        )

    def generate_with_tools(self, messages, tools, params, tool_choice="auto", *, repeat_id=None):
        return self.generate_with_tools_batch(
            [messages], tools, params, tool_choice, repeat_ids=[repeat_id]
        )[0]

    def generate_with_tools_batch(
        self, messages_batch, tools, params, tool_choice="auto", *, repeat_ids=None
    ):
        return self._batch(
            "tools",
            messages_batch,
            {"params": params, "tools": tools, "tool_choice": tool_choice},
            repeat_ids,
            lambda batch, observer: self._backend.generate_with_tools_batch(
                batch, tools, params, tool_choice, **({"observer": observer} if observer else {})
            ),
        )

    def _batch_only(self):
        if self._granularity != "batch":
            raise ValueError("This operation requires explicit checkpoint_granularity='batch'")

    def continue_from_prefix(self, prefix, params, num_continuations=1, *, repeat_ids=None):
        self._batch_only()
        return self._batch(
            "prefix",
            [prefix] * num_continuations,
            {"params": params},
            repeat_ids,
            lambda batch, observer: self._backend.continue_from_prefix(prefix, params, len(batch)),
        )

    def generate_chat_with_prefix(self, messages, assistant_prefix, params, *, repeat_id=None):
        return self.generate_chat_with_prefix_batch(
            [messages], [assistant_prefix], params, repeat_ids=[repeat_id]
        )[0]

    def generate_chat_with_prefix_batch(
        self, messages_batch, assistant_prefixes, params, *, repeat_ids=None
    ):
        self._batch_only()
        if len(messages_batch) != len(assistant_prefixes):
            raise ValueError("assistant_prefixes must align with messages_batch")
        return self._batch(
            "chat_prefix",
            list(zip(messages_batch, assistant_prefixes, strict=True)),
            {"params": params},
            repeat_ids,
            lambda batch, observer: self._backend.generate_chat_with_prefix_batch(
                [entry[0] for entry in batch], [entry[1] for entry in batch], params
            ),
        )

    def score_chat(self, messages, continuation, *, repeat_id=None):
        return self.score_chat_batch([messages], [continuation], repeat_ids=[repeat_id])[0]

    def score_chat_batch(self, messages_batch, continuations, *, repeat_ids=None):
        self._batch_only()
        if len(messages_batch) != len(continuations):
            raise ValueError("continuations must align with messages_batch")
        return self._batch(
            "score",
            list(zip(messages_batch, continuations, strict=True)),
            {},
            repeat_ids,
            lambda batch, observer: self._backend.score_chat_batch(
                [entry[0] for entry in batch], [entry[1] for entry in batch]
            ),
        )

    def decide(self, request, *, repeat_id=None):
        return self.decide_batch([request], repeat_ids=[repeat_id])[0]

    def generate_chat_batch(self, messages_batch, params, *, repeat_ids=None):
        return self._batch(
            "chat",
            messages_batch,
            {"params": params},
            repeat_ids,
            lambda batch, observer: self._backend.generate_chat_batch(
                batch, params, **({"observer": observer} if observer else {})
            ),
        )

    def decide_batch(self, requests, *, repeat_ids=None):
        return self._batch(
            "decision",
            requests,
            {},
            repeat_ids,
            lambda batch, observer: self._backend.decide_batch(batch, observer=observer),
        )

    def _batch(self, operation, batch, arguments, repeat_ids, delegate):
        if "params" in arguments and hasattr(self._backend, "validate_sampling"):
            self._backend.validate_sampling(arguments["params"])
        if self._persist is not None and not self._persist(self._scope):
            return delegate(batch, None)
        repeats = [None] * len(batch) if repeat_ids is None else list(repeat_ids)
        if len(repeats) != len(batch):
            raise ValueError("repeat_ids must align with the requests")
        entries, missing, keys = [], [], []
        requests = [
            (
                {
                    "backend": self._backend.descriptor()["identity"],
                    "scope": self._scope,
                    "operation": operation,
                    "arguments": {"input": request, **arguments},
                },
                repeat,
            )
            for request, repeat in zip(batch, repeats, strict=True)
        ]
        reservations = self._store._reserve_entries(requests)
        owned_futures = {}
        for request, (key, future, owner, provenance) in zip(batch, reservations, strict=True):
            entries.append((key, future, provenance))
            if owner:
                record = self._store._records[key]
                if record["state"] == "received":
                    try:
                        result = self._backend.reparse_receipt(
                            record["receipt"],
                            **({"params": arguments["params"]} if "params" in arguments else {}),
                        )
                        self._store.update(key, "completed", result=result)
                    except BaseException as exc:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    missing.append(request)
                    keys.append(key)
                    owned_futures[key] = future
        if missing:
            observer = _Observer(self._store, keys)
            phase_progress = (
                self._granularity == "batch"
                and operation == "chat"
                and arguments["params"].after_reasoning_text is not None
                and hasattr(self._backend, "reparse_token_receipt")
            )
            try:
                if self._granularity == "batch" and not phase_progress:
                    for index in range(len(keys)):
                        observer.started(index, 0)
                results = delegate(
                    missing, observer if self._granularity == "request" or phase_progress else None
                )
                if len(results) != len(missing):
                    raise ValueError("Backend returned the wrong number of results")
                for index, result in enumerate(results):
                    if not owned_futures[keys[index]].done():
                        observer.completed(index, result)
            except BaseException as exc:
                if isinstance(exc, CheckpointError):
                    if isinstance(exc, UnknownOutcomeError):
                        parent = (
                            self._store._records.get(exc.key, {}).get("request", {}).get("parent")
                        )
                        with self._store._mutex:
                            future = self._store._inflight.get(parent)
                            if parent in keys and future is not None:
                                if not future.done():
                                    future.set_exception(exc)
                                self._store._release(parent, future)
                    raise
                failures = (
                    exc.failures
                    if isinstance(exc, PartialBatchError)
                    else dict.fromkeys(range(len(keys)), exc)
                )
                for index, key in enumerate(keys):
                    future = owned_futures[key]
                    if not future.done():
                        if index not in failures and isinstance(exc, PartialBatchError):
                            observer.completed(index, exc.results[index])
                        else:
                            observer.failed(index, failures[index])
                            future.set_exception(failures[index])
        outcomes, failures = [], {}
        for index, (key, future, provenance) in enumerate(entries):
            try:
                outcomes.append(self._store._result(future.result(), provenance))
            except UnknownOutcomeError as exc:
                if exc.key != key and "persistence" in exc.metadata:
                    # A joined parent reports the child reservation's history.
                    provenance = {**exc.metadata["persistence"], "source": provenance["source"]}
                raise self._store._error(exc, provenance) from exc
            except CheckpointError:
                raise
            except BaseException as exc:
                error = self._store._error(exc, provenance)
                outcomes.append(error)
                failures[index] = error
            finally:
                if future.done():
                    self._store._release(key, future)
        if failures:
            raise PartialBatchError(outcomes, failures)
        return outcomes
