"""Retained, reconciled Chat Completions Batch submissions."""

import json
import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from llenvs.inference.backends.api import _normalize_provider_error
from llenvs.inference.persistence import CheckpointError, UnknownOutcomeError
from llenvs.inference.protocol import (
    LogprobsNotReturnedError,
    MalformedResponseError,
    PartialBatchError,
    RefusedByPolicyError,
    encode_backend_error,
    restore_backend_error,
)
from llenvs.inference.records import plain


@dataclass(frozen=True)
class BatchHandle:
    run_fingerprint: str
    submissions: tuple[str, ...]
    keys: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(value["run_fingerprint"], tuple(value["submissions"]), tuple(value["keys"]))


@dataclass
class BatchCollection:
    results: list
    failures: dict[int, BaseException] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)


class BatchUnfinished(RuntimeError):  # noqa: N818
    """A terminal Batch left a requested slot without a durable output."""


class BatchStopped(RuntimeError):  # noqa: N818
    """Cooperative stop with recoverable submission and collection progress."""

    def __init__(
        self, handle: BatchHandle | None = None, submission_states: dict[str, str] | None = None
    ):
        super().__init__("Batch operation stopped at a safe point")
        self.handle = handle
        self.submission_states = dict(submission_states or {})
        self.partial_collection: BatchCollection | None = None


def _streamed(response):
    """A streamed body's bytes. A stream that breaks off is raised as the SDK
    raises a connection lost before the body, so callers handle both alike."""
    from openai import APIConnectionError

    try:
        yield from response.iter_bytes()
    except Exception as exc:
        raise APIConnectionError(request=getattr(response, "http_request", None)) from exc


class OpenAIBatchBackend:
    """Submit and collect under the shared store's exclusive process ownership.

    A failed create is reconciled by exact metadata and input-file identity. No
    listing match remains uncertain; collection never creates a replacement.

    ``poll_timeout`` bounds, in seconds, each request that retrieves or lists
    batches; ``None`` keeps the client's timeout. Polls only read, so a stopped
    collection whose poll stalls ends within it. Uploads, creates and downloads
    keep the client's timeout.
    """

    def __init__(
        self,
        backend,
        store,
        *,
        max_file_requests=50_000,
        max_file_bytes=200_000_000,
        poll_interval=10,
        poll_timeout=None,
    ):
        if backend._api_flavor != "openai" or backend._client.max_retries != 0:
            raise ValueError("Batch requires native OpenAI and SDK max_retries=0")
        if poll_timeout is not None and (
            isinstance(poll_timeout, bool)
            or not isinstance(poll_timeout, (int, float))
            or not math.isfinite(poll_timeout)
            or poll_timeout <= 0
        ):
            raise ValueError("poll_timeout must be finite and positive")
        if not 1 <= max_file_requests <= 50_000 or not 1 <= max_file_bytes <= 200_000_000:
            raise ValueError("Batch file limits exceed the provider contract")
        self.backend, self.store = backend, store
        self.client = backend._client
        self.max_file_requests, self.max_file_bytes = max_file_requests, max_file_bytes
        self.poll_interval = poll_interval
        self._polls = {} if poll_timeout is None else {"timeout": poll_timeout}

    def descriptor(self):
        descriptor = self.backend.descriptor()
        descriptor["identity"].update(
            backend="openai_batch", endpoint_path="/v1/chat/completions", pricing_tier="batch"
        )
        return descriptor

    def _write(self, manifest):
        self.store.write_journal(manifest)

    def _submission_states(self, handle):
        return (
            {
                identifier: self.store._journals[identifier]["state"]
                for identifier in handle.submissions
            }
            if handle is not None
            else {}
        )

    def _stop(self, stop, handle=None):
        if stop is not None and stop.is_set():
            if callable(handle):
                handle = handle()
            raise BatchStopped(handle, self._submission_states(handle))

    def _check_client(self):
        is_closed = getattr(self.client, "is_closed", None)
        if is_closed is not None and is_closed():
            raise RuntimeError("OpenAI client is closed; Batch create was not dispatched")

    def _unknown_submission(self, manifest):
        error = UnknownOutcomeError(manifest["id"])
        error.metadata = {
            "persistence_calls": [self.store._provenance(key, "batch") for key in manifest["keys"]]
        }
        return error

    @staticmethod
    def _line(key, body):
        return (
            json.dumps(
                {"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body": body},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()

    def _prepare(self, chunk, params):
        identifier = uuid.uuid4().hex
        self.store.update_many((key, "reserved", {}) for key, _, _ in chunk)
        blob = self.store.write_blob(identifier + "_input", (entry[1] for entry in chunk))
        manifest = {
            "kind": "openai_batch",
            "run": self.store.fingerprint,
            "id": identifier,
            "keys": [entry[0] for entry in chunk],
            "input_blob": blob,
            "sha256": blob["sha256"],
            "state": "intent",
            "endpoint": "/v1/chat/completions",
            "model": self.backend.model_name,
            "params": params,
            "request_metadata": chunk[0][2],
            "input_file_id": None,
            "batch_id": None,
            "raw_files": {},
        }
        self._write(manifest)
        self.store.update_many(
            (key, "reserved", {"submission": identifier}) for key in manifest["keys"]
        )
        return identifier

    def submit_chat_batch(self, messages_batch, params, *, repeat_ids, stop=None):
        """Submit eligible calls; a cooperative stop raises ``BatchStopped``."""
        self._stop(stop)
        with self.store._batch_lock, self.store._hold_records() as hold:
            return self._submit_chat_batch(
                messages_batch, params, repeat_ids=repeat_ids, hold=hold, stop=stop
            )

    def acknowledge_resubmission(self, submission_id, *, reason, accept_duplicate_expenditure):
        """Record authorization; only a subsequent explicit submit buys another draw."""
        if not accept_duplicate_expenditure or not reason.strip():
            raise ValueError(
                "A reason and explicit acceptance of duplicate expenditure are required"
            )
        with self.store._batch_lock:
            self.store._check()
            manifest = deepcopy(self.store._journals[submission_id])
            if manifest["state"] not in {"creating", "submission_unknown", "duplicate_submissions"}:
                raise ValueError(
                    "Only uncertain Batch creation outcomes can be authorized for resubmission"
                )
            manifest.setdefault("recovery_decisions", []).append(
                {"reason": reason, "duplicate_expenditure_accepted": True}
            )
            manifest["state"] = "resubmission_authorized"
            self._write(manifest)

    def _submit_chat_batch(self, messages_batch, params, *, repeat_ids, hold, stop):
        if len(repeat_ids) != len(messages_batch) or any(
            type(value) not in (str, int) for value in repeat_ids
        ):
            raise ValueError("Batch requires aligned explicit repeat_ids")
        identity = self.descriptor()["identity"]
        # Validate every serialized request before reserving or uploading anything.
        for messages in messages_batch:
            self._stop(stop)
            kwargs = self.backend._request_kwargs(messages, params)
            if {"extra_headers", "extra_query", "timeout"} & kwargs.keys():
                raise ValueError("Batch does not support per-request transport options")
            body = self.backend._serialized_body(kwargs)
            if len(self._line("0" * 64, body)) > self.max_file_bytes:
                raise ValueError("One serialized Batch request exceeds the file byte limit")
        keys, submissions, chunks, chunk, length = [], [], [], [], 0
        known_keys = set()
        missing_keys = set()

        def progress():
            identifiers = tuple(dict.fromkeys([*submissions, *chunks]))
            retained = known_keys | {
                key
                for identifier in identifiers
                for key in self.store._journals[identifier]["keys"]
            }
            retained_keys = tuple(key for key in keys if key in retained)
            return (
                BatchHandle(self.store.fingerprint, identifiers, retained_keys)
                if retained_keys
                else None
            )

        def reservation_requests():
            for messages, repeat in zip(messages_batch, repeat_ids, strict=True):
                if stop is not None and stop.is_set():
                    return
                yield (
                    {
                        "backend": identity,
                        "scope": "batch",
                        "operation": "chat",
                        "arguments": {"input": messages, "params": params},
                    },
                    repeat,
                )

        reservations = self.store.reserve_many(reservation_requests())
        try:
            keys = [key for key, _, _ in reservations]
            for key in keys:
                hold(key)
            self._stop(stop, progress)
            for messages, (key, _, owner) in zip(messages_batch, reservations, strict=True):
                self._stop(stop, progress)
                record = self.store._records[key]
                self.store._release(key)
                if record["state"] == "completed":
                    known_keys.add(key)
                existing = record.get("submission") or self.store._intent_submissions.get(key)
                authorized = (
                    existing
                    and self.store._journals[existing]["state"] == "resubmission_authorized"
                )
                entry = (record.get("receipt") or {}).get("batch_entry") or {}
                if (
                    existing
                    and record["state"] != "completed"
                    and (
                        authorized
                        or record["state"] == "failed"
                        and self.store._journals[existing]["state"] == "terminal"
                        and entry.get("response") is None
                        and (entry.get("error") or {}).get("code") == "batch_expired"
                    )
                ):
                    history = record["attempts"] + [
                        {
                            "outcome_unknown": bool(authorized),
                            **{
                                name: deepcopy(record[name])
                                for name in ("state", "error", "receipt", "submission")
                                if name in record
                            },
                        }
                    ]
                    self.store.update(
                        key, "reserved", attempts=history, error=None, receipt=None, submission=None
                    )
                    existing = None
                    owner = True
                if existing:
                    if record.get("submission") is None:
                        self.store.update(key, "reserved", submission=existing)
                    if existing not in submissions:
                        submissions.append(existing)
                elif (owner or record["state"] == "reserved") and key not in missing_keys:
                    missing_keys.add(key)
                    kwargs = self.backend._request_kwargs(messages, params)
                    metadata = self.backend._request_metadata(kwargs, params)
                    line = self._line(key, self.backend._serialized_body(kwargs))
                    if chunk and (
                        len(chunk) >= self.max_file_requests
                        or length + len(line) > self.max_file_bytes
                    ):
                        chunks.append(self._prepare(chunk, params))
                        chunk, length = [], 0
                        self._stop(stop, progress)
                    chunk.append((key, line, metadata))
                    length += len(line)
            if chunk:
                self._stop(stop, progress)
                chunks.append(self._prepare(chunk, params))
            submissions = list(dict.fromkeys([*submissions, *chunks]))
            for identifier in submissions:
                self._stop(stop, progress)
                manifest = deepcopy(self.store._journals[identifier])
                if manifest["state"] == "intent":
                    self._submit(manifest, **({"stop": stop} if stop is not None else {}))
            self._stop(stop, progress)
            return BatchHandle(self.store.fingerprint, tuple(submissions), tuple(keys))
        except BatchStopped as error:
            error.handle = progress()
            error.submission_states = self._submission_states(error.handle)
            raise
        finally:
            for key, future, _ in reservations:
                self.store._release(key, future)

    def _submit(self, manifest, *, stop=None):
        if manifest["state"] != "intent":
            raise ValueError("Only an undispatched intent may be submitted")
        handle = BatchHandle(self.store.fingerprint, (manifest["id"],), tuple(manifest["keys"]))
        self._stop(stop, handle)
        if manifest["input_file_id"] is None:
            with self.store.blob_path(manifest["input_blob"]).open("rb") as stream:
                self._stop(stop)
                uploaded = self.client.files.create(
                    file=("requests.jsonl", stream, "application/jsonl"), purpose="batch"
                )
            manifest["input_file_id"] = uploaded.id
            self._write(manifest)
        self._stop(stop)
        self.store.update_many(
            (key, "pending", {"submission": manifest["id"]}) for key in manifest["keys"]
        )
        self._stop(stop)
        self._check_client()
        manifest["state"] = "creating"
        self._write(manifest)
        try:
            self._stop(stop)
            self._check_client()
        except BaseException:
            # Dispatch has not been invoked. A durable rollback keeps this
            # interruption definite; a hard kill in this gap stays uncertain.
            manifest["state"] = "intent"
            self._write(manifest)
            raise
        try:
            response = self.client.batches.create(
                input_file_id=manifest["input_file_id"],
                endpoint=manifest["endpoint"],
                completion_window="24h",
                metadata={
                    "run_fingerprint": manifest["run"],
                    "submission_uuid": manifest["id"],
                    "file_digest": manifest["sha256"],
                },
            )
        except Exception as exc:
            # Even an unsuccessful listing does not prove the create was rejected.
            raise self._unknown_submission(manifest) from exc
        manifest.update(batch_id=response.id, state="pending")
        self._write(manifest)

    def _commit_receipt(self, key, receipt, params):
        entry = receipt["batch_entry"]
        response = entry.get("response") or {}
        provider_error = entry.get("error") or (response.get("body") or {}).get("error")
        try:
            if entry.get("error") or response.get("status_code") != 200:
                error = ValueError(f"Batch request failed: {entry}")
                setattr(error, "status_code", response.get("status_code"))
                setattr(error, "provider_error", provider_error)
                raise _normalize_provider_error(error, model_name=self.backend.model_name)
            result = self.backend.reparse_receipt(receipt, params)
            self.store.update(key, "completed", result=result)
        except CheckpointError:
            raise
        except Exception as exc:
            setattr(exc, "status_code", response.get("status_code"))
            setattr(
                exc,
                "code",
                provider_error.get("code") if isinstance(provider_error, dict) else None,
            )
            setattr(exc, "attempts", deepcopy(receipt.get("metadata", {}).get("attempts")))
            encoded = encode_backend_error(exc)
            setattr(exc, "transient", encoded["transient"])
            definite = (
                bool(entry.get("error"))
                or response.get("status_code") != 200
                or isinstance(
                    exc, (RefusedByPolicyError, MalformedResponseError, LogprobsNotReturnedError)
                )
            )
            self.store.update(
                key,
                "failed" if definite else "received",
                error=encoded,
                receipt=receipt,
            )
            return exc
        return None

    def _reconcile(self, manifest, *, stop=None):
        after, matches, cursors = None, [], set()
        while True:
            self._stop(stop)
            page = self.client.batches.list(
                **({"after": after} if after else {}), limit=100, **self._polls
            )
            for item in page.data:
                raw = plain(item)
                metadata = raw.get("metadata") or {}
                if (
                    metadata.get("run_fingerprint") == manifest["run"]
                    and metadata.get("submission_uuid") == manifest["id"]
                    and metadata.get("file_digest") == manifest["sha256"]
                    and raw.get("endpoint") == manifest["endpoint"]
                    and raw.get("input_file_id") == manifest["input_file_id"]
                ):
                    matches.append(raw)
            if not page.has_more:
                break
            if not page.data:
                raise ValueError("Provider Batch listing did not advance")
            after = page.data[-1].id
            if after in cursors:
                raise ValueError("Provider Batch listing did not advance")
            cursors.add(after)
        if len(matches) > 1:
            manifest.update(state="duplicate_submissions", matches=matches)
            self._write(manifest)
            raise self._unknown_submission(manifest)
        if matches:
            manifest.update(batch_id=matches[0]["id"], state="pending")
        else:
            manifest["state"] = "submission_unknown"
        self._write(manifest)

    def _collect_once(self, handle: BatchHandle, sources, collection, *, stop=None):
        if handle.run_fingerprint != self.store.fingerprint:
            raise ValueError("Batch handle belongs to another run")
        self.store._check()
        identity = self.descriptor()["identity"]
        if len(set(handle.submissions)) != len(handle.submissions) or any(
            identifier not in self.store._journals for identifier in handle.submissions
        ):
            raise ValueError("Invalid Batch handle submission")

        def validate_key(key):
            if stop is not None and stop.is_set():
                return
            record = self.store._records.get(key)
            if (
                record is None
                or record["request"].get("scope") != "batch"
                or record["request"].get("backend") != identity
            ):
                raise ValueError("Batch handle contains an unknown or mismatched request")
            intent = self.store._intent_submissions.get(key)
            if (
                record["state"] == "reserved"
                and record.get("submission") is None
                and intent in handle.submissions
                and self.store._journals[intent]["state"] == "intent"
            ):
                self.store.update(key, "reserved", submission=intent)
                record = self.store._records[key]
            if (
                record["state"] != "completed"
                and record.get("submission") not in handle.submissions
            ):
                raise ValueError("Batch handle does not cover the request's current submission")
            sources.setdefault(
                key,
                self.store._provenance(
                    key,
                    "store"
                    if record["state"] in {"completed", "failed"}
                    else "receipt"
                    if record["state"] == "received"
                    else "batch",
                ),
            )

        for _ in self.store._parallel(validate_key, handle.keys):
            pass
        self._stop(stop, handle)
        indices = {}
        for index, key in enumerate(handle.keys):
            indices.setdefault(key, []).append(index)

        def capture(key, error=None):
            record = self.store._records[key]
            for index in indices[key]:
                if record["state"] == "completed":
                    collection.results[index] = self.store._result(record["result"], sources[key])
                    collection.failures.pop(index, None)
                elif record["state"] == "failed" or (
                    record["state"] == "received" and record.get("error")
                ):
                    collection.results[index] = None
                    collection.failures[index] = self.store._error(
                        error if error is not None else restore_backend_error(record["error"]),
                        sources[key],
                    )

        statuses = collection.statuses

        def collect_retained(key):
            if stop is not None and stop.is_set():
                return
            if sources[key]["source"] == "batch":
                return
            record = self.store._records[key]
            if record["state"] == "received":
                manifest = self.store._journals[record["submission"]]
                error = self._commit_receipt(key, record["receipt"], manifest["params"])
                capture(key, error)
            elif all(
                collection.results[index] is None and index not in collection.failures
                for index in indices[key]
            ):
                capture(key)

        self.store._apply_many(collect_retained, indices)
        for identifier in handle.submissions:
            self._stop(stop, handle)
            manifest = deepcopy(self.store._journals[identifier])
            if not set(manifest["keys"]) & set(handle.keys):
                raise ValueError("Batch handle does not match its submission")
            selected = set(manifest["keys"]) & set(handle.keys)
            if manifest["state"] == "resubmission_authorized":
                statuses[identifier] = "resubmission_authorized"
                continue
            settled = (
                {"completed", "failed", "received"}
                if manifest["state"] == "terminal"
                else {"completed"}
            )
            if not any(
                self.store._records[key]["state"] not in settled
                and self.store._records[key].get("submission") == identifier
                for key in selected
            ):
                statuses[identifier] = (manifest.get("provider_batch") or {}).get(
                    "status", "local_completed"
                )
                continue
            if manifest["input_blob"]["sha256"] != manifest["sha256"]:
                raise ValueError("Batch upload digest mismatch")
            self.store.blob_path(manifest["input_blob"])
            if manifest["state"] in {"creating", "submission_unknown"}:
                self._reconcile(manifest, **({"stop": stop} if stop is not None else {}))
            if manifest["state"] == "intent":
                self._submit(manifest, **({"stop": stop} if stop is not None else {}))
            self._stop(stop, handle)
            if manifest["batch_id"] is None:
                statuses[identifier] = manifest["state"]
                continue
            batch = manifest.get("provider_batch")
            if batch is None or batch["status"] not in {
                "completed",
                "expired",
                "cancelled",
                "failed",
            }:
                self._stop(stop, handle)
                batch = plain(self.client.batches.retrieve(manifest["batch_id"], **self._polls))
            if batch.get("id") != manifest["batch_id"] or any(
                name in batch and batch[name] != manifest[name]
                for name in ("endpoint", "input_file_id")
            ):
                raise ValueError("Retrieved Batch does not match its submission manifest")
            statuses[identifier] = batch["status"]
            manifest["provider_batch"] = batch
            if stop is not None and stop.is_set():
                self._write(manifest)
                self._stop(stop, handle)
            for kind in ("output_file_id", "error_file_id"):
                file_id = batch.get(kind)
                if file_id and file_id not in manifest["raw_files"]:
                    self._stop(stop, handle)
                    if hasattr(self.client.files, "with_streaming_response"):
                        with self.client.files.with_streaming_response.content(file_id) as response:
                            blob = self.store.write_blob(
                                identifier + "_" + file_id, _streamed(response)
                            )
                    else:
                        blob = self.store.write_blob(
                            identifier + "_" + file_id, self.client.files.content(file_id).content
                        )
                    manifest["raw_files"][file_id] = blob
                    self._write(manifest)
            self._stop(stop, handle)
            paths = [self.store.blob_path(blob) for blob in manifest["raw_files"].values()]

            def read_entries():
                for path in paths:
                    with path.open("rb") as handle:
                        for line in handle:
                            if stop is not None and stop.is_set():
                                return
                            if line.strip():
                                yield json.loads(line)

            seen, allowed = (set(), set(manifest["keys"]))
            for entry in read_entries():
                key = entry.get("custom_id")
                if key not in allowed or key in seen:
                    raise ValueError("Unknown or duplicate custom_id in Batch output")
                seen.add(key)
            self._stop(stop, handle)

            def commit_held_entry(entry):
                key = entry["custom_id"]
                if (
                    key not in selected
                    or self.store._records[key]["state"] in {"completed", "failed"}
                    or self.store._records[key].get("submission") != identifier
                ):
                    return
                response = entry.get("response") or {}
                metadata = {
                    **manifest["request_metadata"],
                    "batch_id": manifest["batch_id"],
                    "pricing_tier": "batch",
                    "endpoint": manifest["endpoint"],
                    "attempts": [
                        {
                            "batch_id": manifest["batch_id"],
                            "endpoint": manifest["endpoint"],
                            "pricing_tier": "batch",
                            "status_code": response.get("status_code"),
                            "request_id": response.get("request_id"),
                            "response_id": (response.get("body") or {}).get("id"),
                            "usage": (response.get("body") or {}).get("usage"),
                        }
                    ],
                }
                receipt = {
                    "response": response.get("body"),
                    "metadata": metadata,
                    "batch_entry": entry,
                }
                try:
                    self.store.update(key, "received", receipt=receipt)
                    error = self._commit_receipt(key, receipt, manifest["params"])
                except BaseException:
                    capture(key)
                    raise
                capture(key, error)

            def commit_entry(entry):
                if stop is not None and stop.is_set():
                    return
                if entry["custom_id"] not in selected:
                    return
                with self.store._hold_records() as hold:
                    hold(entry["custom_id"])
                    commit_held_entry(entry)

            self.store._apply_many(commit_entry, read_entries())
            self._stop(stop, handle)
            if batch["status"] in {"completed", "expired", "cancelled", "failed"}:

                def missing_entry(key):
                    if stop is not None and stop.is_set():
                        return
                    if (
                        key not in seen
                        and self.store._records[key]["state"] not in {"completed", "failed"}
                        and self.store._records[key].get("submission") == identifier
                    ):
                        error = {
                            "type": "BatchUnfinished",
                            "message": f"Batch {batch['status']} without a durable output",
                        }
                        self.store.update(key, "failed", error=error)
                        capture(key, BatchUnfinished(error["message"]))

                self.store._apply_many(missing_entry, selected)
                self._stop(stop, handle)
                manifest["state"] = "terminal"
            self._write(manifest)

    def collect(self, handle, *, wait=False, stop=None):
        """Collect named outcomes; errors expose any progress as ``partial_collection``."""
        sources = {}
        collection = BatchCollection([None] * len(handle.keys))
        try:
            while True:
                try:
                    with self.store._batch_lock:
                        self._stop(stop, handle)
                        self._collect_once(
                            handle,
                            sources,
                            collection,
                            **({"stop": stop} if stop is not None else {}),
                        )
                        self._stop(stop, handle)
                finally:
                    collection.pending = [
                        key
                        for index, key in enumerate(handle.keys)
                        if collection.results[index] is None and index not in collection.failures
                    ]
                if (
                    not wait
                    or not collection.pending
                    or any(
                        status
                        in {
                            "submission_unknown",
                            "duplicate_submissions",
                            "resubmission_authorized",
                        }
                        for status in collection.statuses.values()
                    )
                ):
                    return collection
                if stop is None:
                    time.sleep(self.poll_interval)
                elif stop.wait(self.poll_interval):
                    self._stop(stop, handle)
        except BaseException as error:
            if isinstance(error, BatchStopped):
                error.handle = handle
                error.submission_states = self._submission_states(handle)
                error.partial_collection = collection
            elif collection.failures or any(result is not None for result in collection.results):
                setattr(error, "partial_collection", collection)
            raise

    def generate_chat_batch(self, messages_batch, params, *, repeat_ids):
        collection = self.collect(
            self.submit_chat_batch(messages_batch, params, repeat_ids=repeat_ids), wait=True
        )
        if collection.pending:
            key = collection.pending[0]
            raise self.store._error(UnknownOutcomeError(key), self.store._provenance(key, "batch"))
        if collection.failures:
            raise PartialBatchError(
                [
                    collection.failures.get(index, result)
                    for index, result in enumerate(collection.results)
                ],
                collection.failures,
            )
        return collection.results
