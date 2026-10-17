"""Retained, reconciled Chat Completions Batch submissions."""

import json
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field

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


class OpenAIBatchBackend:
    """Submit and collect under the shared store's exclusive process ownership.

    A failed create is reconciled by exact metadata and input-file identity. No
    listing match remains uncertain; collection never creates a replacement.
    """

    def __init__(
        self,
        backend,
        store,
        *,
        max_file_requests=50_000,
        max_file_bytes=200_000_000,
        poll_interval=10,
    ):
        if backend._api_flavor != "openai" or backend._client.max_retries != 0:
            raise ValueError("Batch requires native OpenAI and SDK max_retries=0")
        if not 1 <= max_file_requests <= 50_000 or not 1 <= max_file_bytes <= 200_000_000:
            raise ValueError("Batch file limits exceed the provider contract")
        self.backend, self.store = backend, store
        self.client = backend._client
        self.max_file_requests, self.max_file_bytes = max_file_requests, max_file_bytes
        self.poll_interval = poll_interval

    def descriptor(self):
        descriptor = self.backend.descriptor()
        descriptor["identity"].update(
            backend="openai_batch", endpoint_path="/v1/chat/completions", pricing_tier="batch"
        )
        return descriptor

    def _write(self, manifest):
        self.store.write_journal(manifest)

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
        for key in manifest["keys"]:
            self.store.update(key, "reserved", submission=identifier)
        return identifier

    def submit_chat_batch(self, messages_batch, params, *, repeat_ids):
        with self.store._batch_lock:
            return self._submit_chat_batch(messages_batch, params, repeat_ids=repeat_ids)

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

    def _submit_chat_batch(self, messages_batch, params, *, repeat_ids):
        if len(repeat_ids) != len(messages_batch) or any(
            type(value) not in (str, int) for value in repeat_ids
        ):
            raise ValueError("Batch requires aligned explicit repeat_ids")
        identity = self.descriptor()["identity"]
        # Validate every serialized request before reserving or uploading anything.
        for messages in messages_batch:
            kwargs = self.backend._request_kwargs(messages, params)
            if {"extra_headers", "extra_query", "timeout"} & kwargs.keys():
                raise ValueError("Batch does not support per-request transport options")
            body = self.backend._serialized_body(kwargs)
            if len(self._line("0" * 64, body)) > self.max_file_bytes:
                raise ValueError("One serialized Batch request exceeds the file byte limit")
        keys, submissions, chunks, chunk, length = [], [], [], [], 0
        missing_keys = set()
        for messages, repeat in zip(messages_batch, repeat_ids, strict=True):
            key, _, owner = self.store.reserve(
                {
                    "backend": identity,
                    "scope": "batch",
                    "operation": "chat",
                    "arguments": {"input": messages, "params": params},
                },
                repeat,
            )
            keys.append(key)
            record = self.store._records[key]
            self.store._inflight.pop(key, None)
            existing = record.get("submission") or self.store._intent_submissions.get(key)
            authorized = (
                existing and self.store._journals[existing]["state"] == "resubmission_authorized"
            )
            if (
                existing
                and record["state"] != "completed"
                and (
                    authorized
                    or record["state"] == "failed"
                    and self.store._journals[existing]["state"] == "terminal"
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
                self.store.update(key, "reserved")
                missing_keys.add(key)
                kwargs = self.backend._request_kwargs(messages, params)
                metadata = self.backend._request_metadata(kwargs, params)
                line = self._line(key, self.backend._serialized_body(kwargs))
                if chunk and (
                    len(chunk) >= self.max_file_requests or length + len(line) > self.max_file_bytes
                ):
                    chunks.append(self._prepare(chunk, params))
                    chunk, length = [], 0
                chunk.append((key, line, metadata))
                length += len(line)
        if chunk:
            chunks.append(self._prepare(chunk, params))
        for identifier in chunks:
            if identifier not in submissions:
                submissions.append(identifier)
            self._submit(deepcopy(self.store._journals[identifier]))
        return BatchHandle(self.store.fingerprint, tuple(submissions), tuple(keys))

    def _submit(self, manifest):
        if manifest["state"] != "intent":
            raise ValueError("Only an undispatched intent may be submitted")
        if manifest["input_file_id"] is None:
            with self.store.blob_path(manifest["input_blob"]).open("rb") as handle:
                uploaded = self.client.files.create(
                    file=("requests.jsonl", handle, "application/jsonl"), purpose="batch"
                )
            manifest["input_file_id"] = uploaded.id
            self._write(manifest)
        manifest["state"] = "creating"
        self._write(manifest)

        for key in manifest["keys"]:
            self.store.update(key, "pending", submission=manifest["id"])
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
            raise UnknownOutcomeError(manifest["id"]) from exc
        manifest.update(batch_id=response.id, state="pending")
        self._write(manifest)

    def _commit_receipt(self, key, receipt, params):
        entry = receipt["batch_entry"]
        response = entry.get("response") or {}
        try:
            if entry.get("error") or response.get("status_code") != 200:
                raise ValueError(f"Batch request failed: {entry}")
            result = self.backend.reparse_receipt(receipt, params)
            self.store.update(key, "completed", result=result)
        except CheckpointError:
            raise
        except Exception as exc:
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
                error=encode_backend_error(exc),
                receipt=receipt,
            )
            return exc
        return None

    def _reconcile(self, manifest):
        after, matches, cursors = None, [], set()
        while True:
            page = self.client.batches.list(**({"after": after} if after else {}), limit=100)
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
            raise UnknownOutcomeError(manifest["id"])
        if matches:
            manifest.update(batch_id=matches[0]["id"], state="pending")
        else:
            manifest["state"] = "submission_unknown"
        self._write(manifest)

    def _collect_once(self, handle: BatchHandle, sources):
        if handle.run_fingerprint != self.store.fingerprint:
            raise ValueError("Batch handle belongs to another run")
        self.store._check()
        identity = self.descriptor()["identity"]
        if len(set(handle.submissions)) != len(handle.submissions) or any(
            identifier not in self.store._journals for identifier in handle.submissions
        ):
            raise ValueError("Invalid Batch handle submission")
        for key in handle.keys:
            record = self.store._records.get(key)
            if (
                record is None
                or record["request"].get("scope") != "batch"
                or record["request"].get("backend") != identity
            ):
                raise ValueError("Batch handle contains an unknown or mismatched request")
            if (
                record["state"] != "completed"
                and record.get("submission") not in handle.submissions
            ):
                raise ValueError("Batch handle does not cover the request's current submission")
            sources.setdefault(
                key,
                "store"
                if record["state"] == "completed"
                else "receipt"
                if record["state"] == "received"
                else "batch",
            )
        statuses = {}
        failures = {}
        for key in dict.fromkeys(handle.keys):
            record = self.store._records[key]
            if record["state"] == "received":
                manifest = self.store._journals[record["submission"]]
                error = self._commit_receipt(key, record["receipt"], manifest["params"])
                if error is not None:
                    failures[key] = error
        for identifier in handle.submissions:
            manifest = deepcopy(self.store._journals[identifier])
            if not set(manifest["keys"]) & set(handle.keys):
                raise ValueError("Batch handle does not match its submission")
            selected = set(manifest["keys"]) & set(handle.keys)
            if manifest["state"] == "resubmission_authorized":
                statuses[identifier] = "resubmission_authorized"
                continue
            if manifest["state"] == "terminal" or not any(
                self.store._records[key]["state"] != "completed"
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
                self._reconcile(manifest)
            if manifest["state"] == "intent":
                self._submit(manifest)
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
                batch = plain(self.client.batches.retrieve(manifest["batch_id"]))
            if batch.get("id") != manifest["batch_id"] or any(
                name in batch and batch[name] != manifest[name]
                for name in ("endpoint", "input_file_id")
            ):
                raise ValueError("Retrieved Batch does not match its submission manifest")
            statuses[identifier] = batch["status"]
            manifest["provider_batch"] = batch
            for kind in ("output_file_id", "error_file_id"):
                file_id = batch.get(kind)
                if file_id and file_id not in manifest["raw_files"]:
                    if hasattr(self.client.files, "with_streaming_response"):
                        with self.client.files.with_streaming_response.content(file_id) as response:
                            blob = self.store.write_blob(
                                identifier + "_" + file_id, response.iter_bytes()
                            )
                    else:
                        blob = self.store.write_blob(
                            identifier + "_" + file_id, self.client.files.content(file_id).content
                        )
                    manifest["raw_files"][file_id] = blob
                    self._write(manifest)
            paths = [self.store.blob_path(blob) for blob in manifest["raw_files"].values()]

            def read_entries():
                for path in paths:
                    with path.open("rb") as handle:
                        for line in handle:
                            if line.strip():
                                yield json.loads(line)

            seen, allowed = (set(), set(manifest["keys"]))
            for entry in read_entries():
                key = entry.get("custom_id")
                if key not in allowed or key in seen:
                    raise ValueError("Unknown or duplicate custom_id in Batch output")
                seen.add(key)
            for entry in read_entries():
                key = entry["custom_id"]
                if (
                    self.store._records[key]["state"] == "completed"
                    or self.store._records[key].get("submission") != identifier
                ):
                    continue
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
                self.store.update(key, "received", receipt=receipt)
                error = self._commit_receipt(key, receipt, manifest["params"])
                if error is not None:
                    failures[key] = error
            if batch["status"] in {"completed", "expired", "cancelled", "failed"}:
                for key in manifest["keys"]:
                    if (
                        key not in seen
                        and self.store._records[key]["state"] != "completed"
                        and self.store._records[key].get("submission") == identifier
                    ):
                        error = {
                            "type": "BatchUnfinished",
                            "message": f"Batch {batch['status']} without a durable output",
                        }
                        self.store.update(key, "failed", error=error)
                        failures[key] = RuntimeError(error["message"])
                manifest["state"] = "terminal"
            self._write(manifest)
        results, indexed_failures, pending = ([], {}, [])
        for index, key in enumerate(handle.keys):
            record = self.store._records[key]
            if record["state"] == "completed":
                results.append(self.store._result(key, record["result"], sources[key]))
            else:
                results.append(None)
                if (
                    record["state"] == "failed"
                    or record["state"] == "received"
                    and record.get("error")
                ):
                    indexed_failures[index] = failures.get(
                        key, restore_backend_error(record["error"])
                    )
                else:
                    pending.append(key)
        collection = BatchCollection(results, indexed_failures, pending, statuses)
        return collection

    def collect(self, handle, *, wait=False):
        sources = {}
        while True:
            with self.store._batch_lock:
                collection = self._collect_once(handle, sources)
            if (
                not wait
                or not collection.pending
                or any(
                    status
                    in {"submission_unknown", "duplicate_submissions", "resubmission_authorized"}
                    for status in collection.statuses.values()
                )
            ):
                return collection
            time.sleep(self.poll_interval)

    def generate_chat_batch(self, messages_batch, params, *, repeat_ids):
        collection = self.collect(
            self.submit_chat_batch(messages_batch, params, repeat_ids=repeat_ids), wait=True
        )
        if collection.pending:
            raise UnknownOutcomeError(collection.pending[0])
        if collection.failures:
            raise PartialBatchError(
                [
                    collection.failures.get(index, result)
                    for index, result in enumerate(collection.results)
                ],
                collection.failures,
            )
        return collection.results
