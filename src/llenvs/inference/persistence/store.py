"""Checksummed call journal, generalized from QVal's retained call store.

Pickles are trusted local artifacts. Checksums detect accidental corruption.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import enum
import fcntl
import hashlib
import json
import math
import os
import pickle
import shutil
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    PartialBatchError,
    StopReason,
    encode_backend_error,
    restore_backend_error,
)


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
        # Absent reasoning must not change the identity of ordinary saved calls.
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


def _fsync_directory(path: Path) -> None:
    """Commit the directory entry; a failure invalidates the durability claim."""
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_write_pickle(path: Path, value: Any) -> None:
    record_payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "payload_sha256": hashlib.sha256(record_payload).hexdigest(),
        "payload": record_payload,
    }
    _atomic_write_bytes(
        path,
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL),
    )


def _read_checked_pickle(path: Path) -> Any:
    envelope = pickle.loads(path.read_bytes())
    payload = envelope["payload"]
    if hashlib.sha256(payload).hexdigest() != envelope["payload_sha256"]:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return pickle.loads(payload)


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
        super().__init__(f"Outcome unknown for {key}; explicit recovery decision required")


class _RecordIndex(MutableMapping):
    """Keep active reservations and a bounded cache; completed draws live on disk."""

    def __init__(self, store):
        self.store = store
        self.volatile = {}
        self.cache = OrderedDict()

    def path(self, key):
        return self.store.run_dir / "calls" / key[:2] / (key + ".pkl")

    def __getitem__(self, key):
        with self.store._mutex:
            if key in self.volatile:
                return self.volatile[key]
            if key not in self.cache:
                path = self.path(key)
                if not path.exists():
                    raise KeyError(key)
                try:
                    self[key] = self.store._load(path)
                except Exception as exc:
                    self.store._poison = exc
                    raise CheckpointCorruptionError(f"Invalid retained call {key}") from exc
            self.cache.move_to_end(key)
            return self.cache[key]

    def __setitem__(self, key, value):
        with self.store._mutex:
            if not self.path(key).exists():
                self.volatile[key] = value
                return
            self.volatile.pop(key, None)
            self.cache[key] = value
            self.cache.move_to_end(key)
            while len(self.cache) > 64:
                self.cache.popitem(last=False)

    def __delitem__(self, key):
        raise TypeError("Retained records can only be removed through mark_complete(delete=True)")

    def __iter__(self):
        yield from tuple(self.volatile)
        for path in (self.store.run_dir / "calls").rglob("*.pkl"):
            yield path.stem

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
    def open(cls, root: str | Path, *, run_descriptor: Mapping[str, Any]) -> CallStore:
        from llenvs.inference.records import credential_free

        descriptor = _normalize(credential_free(run_descriptor))
        fingerprint = _fingerprint({"schema": 1, "run": descriptor})
        root = Path(root)
        (root / ".locks").mkdir(parents=True, exist_ok=True)
        handle = (root / ".locks" / (fingerprint + ".lock")).open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise CheckpointInUseError(f"Run {fingerprint} is already open") from exc
        try:
            return cls(root, fingerprint, descriptor, handle)
        except BaseException:
            handle.close()
            raise

    def __init__(self, root, fingerprint, descriptor, handle):
        self.root, self.fingerprint = root, fingerprint
        self.run_dir = root / fingerprint
        self._handle = handle
        self._mutex = threading.RLock()
        self._batch_lock = threading.RLock()
        self._closed = False
        self._poison: BaseException | None = None
        self._occurrences: dict[str, int] = {}
        self._inflight: dict[str, Future] = {}
        self._records = _RecordIndex(self)
        self._journals: dict[str, dict] = {}
        self._intent_submissions = {}
        manifest = {"schema": 1, "fingerprint": fingerprint, "run_descriptor": descriptor}
        path = self.run_dir / "manifest.json"
        try:
            if self.run_dir.exists():
                if json.loads(path.read_text()) != manifest:
                    raise ValueError("manifest mismatch")
                for record_path in (self.run_dir / "calls").rglob("*.pkl"):
                    self._load(record_path)
                for lifecycle_path in (self.run_dir / "lifecycle").glob("*.pkl"):
                    event = _read_checked_pickle(lifecycle_path)
                    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                        raise ValueError("Invalid lifecycle event")
                for journal_path in (self.run_dir / "journals").glob("*.pkl"):
                    journal = _read_checked_pickle(journal_path)
                    if journal["run"] != fingerprint or journal["id"] != journal_path.stem:
                        raise ValueError("submission journal identity mismatch")
                    self._journals[journal["id"]] = journal
                    for blob in [journal["input_blob"], *journal.get("raw_files", {}).values()]:
                        self.blob_path(blob)
                    for key in journal["keys"]:
                        record = self._records[key]
                        if record["request"].get("scope") != "batch":
                            raise ValueError("Submission references a non-Batch call")
                        if journal["state"] == "intent":
                            self._intent_submissions[key] = journal["id"]
            else:
                _atomic_write_json(path, manifest)
        except Exception as exc:
            raise CheckpointCorruptionError(f"Invalid call store {self.run_dir}: {exc}") from exc
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llenvs-journal")
        self._write_slots = threading.BoundedSemaphore(64)

    def _load(self, path):
        record = _read_checked_pickle(path)
        if record["key"] != path.stem or record["run"] != self.fingerprint:
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
        if self._poison is not None:
            raise CheckpointError("Call store is poisoned by a storage failure") from self._poison

    def reserve(self, request: dict, repeat_id) -> tuple[str, Future, bool]:
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
            if key in self._inflight:
                return key, self._inflight[key], False
            future = Future()
            self._inflight[key] = future
            record = self._records.get(key)
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
        with self._mutex:
            self._check()
            record = copy.deepcopy(self._records[key])
            record.update(values, state=state)
            if state == "completed":
                _validate_result(record["result"])
            try:
                _atomic_write_pickle(self._records.path(key), record)
            except Exception as exc:
                self._poison = exc
                raise CheckpointError(f"Cannot persist {state} for {key}") from exc
            self._records[key] = record
            future = self._inflight.get(key)
            if state == "completed" and future is not None and not future.done():
                future.set_result(copy.deepcopy(record["result"]))

    def _result_source(self, key, future, owner):
        if owner:
            return "receipt" if self._records[key]["state"] == "received" else "backend"
        return "store" if future.done() else "shared"

    def _result(self, key, outcome, source):
        """Detach invocation provenance without changing the retained outcome."""
        result = copy.deepcopy(outcome)
        target = getattr(result, "result", result)
        target.metadata["persistence"] = {
            "run_fingerprint": self.fingerprint,
            "call_key": key,
            "source": source,
        }
        return result

    def write_journal(self, journal: dict):
        """Commit a submission manifest, including exact upload and output bytes."""
        with self._mutex:
            self._check()
            identifier = journal["id"]
            if not identifier.isalnum() or journal["run"] != self.fingerprint:
                raise ValueError("Invalid journal identity")
            try:
                _atomic_write_pickle(self.run_dir / "journals" / (identifier + ".pkl"), journal)
            except Exception as exc:
                self._poison = exc
                raise CheckpointError("Cannot persist submission journal") from exc
            self._journals[identifier] = copy.deepcopy(journal)
            for key in journal["keys"]:
                if journal["state"] == "intent":
                    self._intent_submissions[key] = identifier
                elif self._intent_submissions.get(key) == identifier:
                    self._intent_submissions.pop(key)

    def write_blob(self, name, chunks):
        """Stream exact upload/output bytes into an atomic, checksummed artifact."""
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in name
        ):
            raise ValueError("Invalid blob name")
        self._check()
        directory = self.run_dir / "blobs"
        directory.mkdir(exist_ok=True)
        destination = directory / name
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
            os.replace(temporary, destination)
            _fsync_directory(directory)
        except Exception as exc:
            self._poison = exc
            raise CheckpointError("Cannot persist exact Batch file bytes") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return {"name": name, "sha256": digest.hexdigest(), "size": size}

    def blob_path(self, entry, *, verify=True):
        name = entry["name"]
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in name
        ):
            raise CheckpointCorruptionError("Invalid retained blob name")
        path = self.run_dir / "blobs" / name
        if verify:
            from llenvs.inference.records import file_digest

            if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
                raise CheckpointCorruptionError(f"Retained Batch file checksum mismatch: {name}")
        return path

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
        with self._mutex:
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
            self._inflight.pop(key, None)

    def lifecycle(self, event: Mapping[str, Any]):
        """Persist an independent provenance event without changing request identity."""
        with self._mutex:
            self._check()
            try:
                _atomic_write_pickle(
                    self.run_dir / "lifecycle" / (uuid.uuid4().hex + ".pkl"), dict(event)
                )
            except Exception as exc:
                self._poison = exc
                raise CheckpointError("Cannot persist lifecycle event") from exc

    def mark_complete(self, *, delete: bool = False):
        with self._mutex:
            self._check()
            if any(
                record["state"] not in {"completed", "failed"} for record in self._records.values()
            ):
                raise UnknownOutcomeError("run has unresolved requests")
            try:
                _atomic_write_json(self.run_dir / "complete.json", {"retained": not delete})
            except Exception as exc:
                self._poison = exc
                raise CheckpointError("Cannot persist run completion") from exc
            if delete:
                shutil.rmtree(self.run_dir)

    def close(self):
        if not self._closed:
            self._writer.shutdown(wait=True)
            self._closed = True
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _Observer:
    def __init__(self, store, keys):
        self.store, self.keys = store, keys

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

    def received(self, index, receipt):
        self.store.update(self.keys[index], "received", receipt=copy.deepcopy(receipt))

    def begin_phases(self, index):
        self.store.update(self.keys[index], "phased")

    def run_phase(self, index, name, arguments, delegate, reparse):
        with self.store._mutex:
            key, future, owner = self.store.reserve(
                {"parent": self.keys[index], "phase": name, "arguments": arguments}, 0
            )
            source = self.store._result_source(key, future, owner)
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
            return self.store._result(key, future.result(), source)
        finally:
            with self.store._mutex:
                if future.done() and self.store._inflight.get(key) is future:
                    self.store._inflight.pop(key)

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
                raise UnknownOutcomeError(key) from error

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
        for request, repeat in zip(batch, repeats, strict=True):
            with self._store._mutex:
                key, future, owner = self._store.reserve(
                    {
                        "backend": self._backend.descriptor()["identity"],
                        "scope": self._scope,
                        "operation": operation,
                        "arguments": {"input": request, **arguments},
                    },
                    repeat,
                )
                source = self._store._result_source(key, future, owner)
            entries.append((key, future, source))
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
                        future.set_exception(exc)
                else:
                    missing.append(request)
                    keys.append(key)
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
                    if not self._store._inflight[keys[index]].done():
                        observer.completed(index, result)
            except BaseException as exc:
                if isinstance(exc, CheckpointError):
                    raise
                failures = (
                    exc.failures
                    if isinstance(exc, PartialBatchError)
                    else dict.fromkeys(range(len(keys)), exc)
                )
                for index, key in enumerate(keys):
                    future = self._store._inflight[key]
                    if not future.done():
                        if index not in failures and isinstance(exc, PartialBatchError):
                            observer.completed(index, exc.results[index])
                        else:
                            observer.failed(index, failures[index])
                            future.set_exception(failures[index])
        outcomes, failures = [], {}
        for index, (key, future, source) in enumerate(entries):
            try:
                outcomes.append(self._store._result(key, future.result(), source))
            except CheckpointError:
                raise
            except BaseException as exc:
                outcomes.append(exc)
                failures[index] = exc
            finally:
                with self._store._mutex:
                    if future.done() and self._store._inflight.get(key) is future:
                        self._store._inflight.pop(key)
        if failures:
            raise PartialBatchError(outcomes, failures)
        return outcomes
