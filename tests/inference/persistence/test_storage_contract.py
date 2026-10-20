"""Call semantics must not depend on whether durable state is local or in GCS."""

import hashlib
import json
from dataclasses import replace

import pytest

from llenvs.inference.backends.openai_batch import BatchHandle, OpenAIBatchBackend
from llenvs.inference.persistence import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointInUseError,
    PersistentBackend,
    UnknownOutcomeError,
)
from llenvs.inference.protocol import (
    ChatMessage,
    PartialBatchError,
    RefusedByPolicyError,
    SamplingParams,
)
from tests.inference.test_openai_batch import Client, backend, completion, generated_failure


def _unavailable_artifact():
    raise FileNotFoundError("Synthetic artifact unavailable during record decoding")


class _UnavailableArtifact:
    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return _unavailable_artifact, ()


def messages(text="request"):
    return [ChatMessage("user", text)]


def logical_request(draw, params):
    return {
        "backend": draw.descriptor()["identity"],
        "scope": "default",
        "operation": "chat",
        "arguments": {"input": messages(), "params": params},
    }


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_completed_calls_reopen_without_dispatch_or_identity_changes(stores, draw, validation):
    params = SamplingParams(max_tokens=16)
    with stores(validation=validation) as store:
        first = PersistentBackend(draw, store).generate_chat(messages(), params, repeat_id="a")
        expected = dict(first.metadata["persistence"])
    with stores(validation=validation) as store:
        replay = PersistentBackend(draw, store).generate_chat(messages(), params, repeat_id="a")
        assert replay.metadata["persistence"] == {**expected, "source": "store"}
        assert replace(replay, metadata=first.metadata) == first
        assert draw.calls == 1


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_decoder_file_not_found_is_never_treated_as_an_absent_call(stores, draw, validation):
    parse = draw.reparse_receipt

    def with_artifact(receipt, params):
        result = parse(receipt, params)
        return replace(result, metadata={**result.metadata, "artifact": _UnavailableArtifact()})

    draw.reparse_receipt = with_artifact
    params = SamplingParams(max_tokens=16)
    with stores() as store:
        PersistentBackend(draw, store).generate_chat(messages(), params, repeat_id="a")
    assert draw.calls == 1
    if validation == "full":
        with pytest.raises(CheckpointCorruptionError):
            stores(validation=validation)
    else:
        with stores(validation=validation) as store:
            wrapper = PersistentBackend(draw, store)
            with pytest.raises(CheckpointCorruptionError):
                wrapper.generate_chat(messages(), params, repeat_id="a")
            with pytest.raises(CheckpointError):
                wrapper.generate_chat(messages("other"), params, repeat_id="b")
    assert draw.calls == 1


def test_regrouping_preserves_duplicate_keys_and_per_slot_provenance(stores, draw):
    params = SamplingParams(max_tokens=16)
    with stores() as store:
        wrapper = PersistentBackend(draw, store)
        wrapper.generate_chat(messages("old"), params, repeat_id="old")
        result = wrapper.generate_chat_batch(
            [messages("old"), messages("new"), messages("new")],
            params,
            repeat_ids=["old", "new", "new"],
        )
        evidence = [r.metadata["persistence"] for r in result]
        assert [p["source"] for p in evidence] == ["store", "backend", "shared"]
        assert evidence[1]["call_key"] == evidence[2]["call_key"] != evidence[0]["call_key"]
    with stores() as store:
        result = PersistentBackend(draw, store).generate_chat_batch(
            [messages("new"), messages("old")], params, repeat_ids=["new", "old"]
        )
        assert [r.text for r in result] == ["new", "old"]
        assert all(r.metadata["persistence"]["source"] == "store" for r in result)
        assert draw.calls == 2


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_receipt_survives_interruption_before_result_parsing(stores, draw, validation):
    params = SamplingParams(max_tokens=16)
    with stores() as store:
        key, _, owner = store.reserve(logical_request(draw, params), "a")
        assert owner
        store.update(key, "started")
        store.update(key, "received", receipt={"text": "retained", "draw": 42})
    with stores(validation=validation) as store:
        result = PersistentBackend(draw, store).generate_chat(messages(), params, repeat_id="a")
        assert result.text == "retained" and result.metadata["draw"] == 42
        assert result.metadata["persistence"]["source"] == "receipt"
        assert result.metadata["persistence"]["call_key"] == key
        assert draw.calls == 0
    with stores() as store:
        result = PersistentBackend(draw, store).generate_chat(messages(), params, repeat_id="a")
        assert result.metadata["persistence"]["source"] == "store" and draw.calls == 0


def test_started_without_receipt_stays_unknown_until_explicit_resubmission(stores, draw):
    params = SamplingParams(max_tokens=16)
    with stores() as store:
        key, _, _ = store.reserve(logical_request(draw, params), "a")
        store.update(key, "started")
    with stores() as store:
        wrapper = PersistentBackend(draw, store)
        for _ in range(2):
            with pytest.raises(UnknownOutcomeError) as caught:
                wrapper.generate_chat(messages(), params, repeat_id="a")
            assert caught.value.key == key
            assert caught.value.metadata["persistence"]["source"] == "store"
        assert draw.calls == 0
        with pytest.raises(ValueError):
            store.acknowledge_resubmission(key, reason="", accept_duplicate_expenditure=True)
        with pytest.raises(ValueError):
            store.acknowledge_resubmission(
                key, reason="recovery", accept_duplicate_expenditure=False
            )
        store.acknowledge_resubmission(
            key, reason="Synthetic explicit recovery", accept_duplicate_expenditure=True
        )
        result = wrapper.generate_chat(messages(), params, repeat_id="a")
        assert result.metadata["persistence"]["previous_attempts"] == 1
        assert result.metadata["persistence"]["source"] == "backend" and draw.calls == 1


def test_billed_failure_replay_keeps_key_source_and_attempt_history(stores, draw):
    params = SamplingParams(max_tokens=16)
    draw.failure = RefusedByPolicyError("Synthetic refusal")
    with stores() as store:
        with pytest.raises(PartialBatchError) as caught:
            PersistentBackend(draw, store).generate_chat_batch(
                [messages()], params, repeat_ids=["a"]
            )
        first = caught.value.failures[0].metadata["persistence"]
        assert first["source"] == "backend" and first["previous_attempts"] == 0
    draw.failure = None
    with stores() as store:
        wrapper = PersistentBackend(draw, store)
        with pytest.raises(PartialBatchError) as caught:
            wrapper.generate_chat_batch([messages()], params, repeat_ids=["a"])
        assert caught.value.failures[0].metadata["persistence"] == {**first, "source": "store"}
        assert draw.calls == 1
        store.acknowledge_resubmission(
            first["call_key"], reason="Retry explicitly accepted", accept_duplicate_expenditure=True
        )
        result = wrapper.generate_chat(messages(), params, repeat_id="a")
        assert result.metadata["persistence"]["previous_attempts"] == 1
        assert draw.calls == 2


def test_owners_exclude_competitors_but_different_runs_remain_independent(stores):
    first = stores()
    with pytest.raises(CheckpointInUseError):
        stores()
    with stores(descriptor={"experiment": "different"}) as other:
        assert other.fingerprint != first.fingerprint
    first.close()
    with stores() as reopened:
        assert reopened.fingerprint == first.fingerprint


def test_exact_streamed_artifacts_survive_a_fresh_open(stores):
    payload = bytes(range(256)) * 8192
    with stores() as store:
        entry = store.write_blob(
            "synthetic-output", (payload[i : i + 4096] for i in range(0, len(payload), 4096))
        )
        assert entry["size"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    with stores() as reopened:
        assert reopened.blob_path(entry).read_bytes() == payload


@pytest.mark.parametrize("outcome", ["success", "refusal", "unfinished"])
@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_batch_reopen_preserves_first_collection_and_partial_error_sources(
    stores, outcome, validation
):
    client = Client()
    failure = RuntimeError("Synthetic later-submission retrieval failure")
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=2)
        handle = api.submit_chat_batch(
            [messages(str(i)) for i in range(4)], SamplingParams(), repeat_ids=list(range(4))
        )
        assert len(handle.submissions) == 2
        rows = [completion(handle.keys[0], "first")]
        if outcome == "success":
            rows.append(completion(handle.keys[1], "second"))
        elif outcome == "refusal":
            rows.append(generated_failure(handle.keys[1], "refusal"))
        client.outputs["batch-1"] = b"\n".join(json.dumps(row).encode() for row in rows)
        client.outputs["batch-2"] = b"\n".join(
            json.dumps(completion(key, "later")).encode() for key in handle.keys[2:]
        )

        def retrieve(identifier):
            if identifier == "batch-2":
                raise failure
            from types import SimpleNamespace

            return SimpleNamespace(
                id=identifier, status="completed", output_file_id=identifier, error_file_id=None
            )

        client.batches.retrieve = retrieve
        with pytest.raises(RuntimeError) as caught:
            api.collect(handle)
        assert caught.value is failure
        partial = failure.partial_collection
        assert partial.pending == list(handle.keys[2:])
        delivered = [partial.results[0], partial.results[1] or partial.failures[1]]
        assert [value.metadata["persistence"]["source"] for value in delivered] == [
            "batch",
            "batch",
        ]
        assert all(value.metadata["persistence"]["previous_attempts"] == 0 for value in delivered)
    with stores(validation=validation) as store:
        from types import SimpleNamespace

        client.batches.retrieve = lambda identifier: SimpleNamespace(
            id=identifier, status="completed", output_file_id=identifier, error_file_id=None
        )
        api = OpenAIBatchBackend(backend(client), store)
        # Reconstruct the saved handle from its public fields, not live objects.
        retained = BatchHandle.from_dict(json.loads(json.dumps(handle.to_dict())))
        result = api.collect(retained)
        delivered = [result.results[i] or result.failures[i] for i in range(4)]
        assert [value.metadata["persistence"]["source"] for value in delivered] == [
            "store",
            "store",
            "batch",
            "batch",
        ]
        assert not result.pending
        assert len(client.creates) == 2 and len(client.uploads) == 2


def test_pending_batch_is_reused_after_reopen_and_request_regrouping(stores):
    from types import SimpleNamespace

    client = Client()
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [messages("first"), messages("second")], SamplingParams(), repeat_ids=["a", "b"]
        )
    client.batches.retrieve = lambda identifier: SimpleNamespace(
        id=identifier, status="in_progress", output_file_id=None, error_file_id=None
    )
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        rebuilt = api.submit_chat_batch(
            [messages("second"), messages("first")], SamplingParams(), repeat_ids=["b", "a"]
        )
        assert rebuilt.submissions == handle.submissions
        assert rebuilt.keys == tuple(reversed(handle.keys))
        assert len(client.creates) == len(client.uploads) == 1
        pending = api.collect(rebuilt)
        assert pending.pending == list(rebuilt.keys)
        assert pending.results == [None, None] and not pending.failures


@pytest.mark.parametrize("outcome", ["success", "refusal", "unfinished", "expired"])
def test_subset_collection_reports_each_key_as_batch_once_across_reopen(stores, outcome):
    from types import SimpleNamespace

    client = Client()
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [messages(str(i)) for i in range(5)], SamplingParams(), repeat_ids=list(range(5))
        )
        rows = [completion(key, str(i)) for i, key in enumerate(handle.keys)]
        if outcome == "refusal":
            rows[3] = generated_failure(handle.keys[3], "refusal")
        elif outcome == "unfinished":
            rows.pop(3)
        elif outcome == "expired":
            rows[3] = {
                "custom_id": handle.keys[3],
                "response": None,
                "error": {"code": "batch_expired", "message": "Not executed"},
            }
        client.outputs["output"] = b"\n".join(json.dumps(row).encode() for row in rows)
        client.batches.retrieve = lambda identifier: SimpleNamespace(
            id=identifier,
            status="expired" if outcome == "expired" else "completed",
            output_file_id="output",
            error_file_id=None,
        )
        subset = BatchHandle(handle.run_fingerprint, handle.submissions, handle.keys[:2])
        first = api.collect(subset)
        assert [r.metadata["persistence"]["source"] for r in first.results] == ["batch", "batch"]
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        remaining = BatchHandle(handle.run_fingerprint, handle.submissions, handle.keys[2:])
        first = api.collect(remaining)
        for i in range(3):
            value = first.results[i] or first.failures[i]
            assert value.metadata["persistence"]["source"] == "batch"
            assert value.metadata["persistence"]["previous_attempts"] == 0
        replay = api.collect(handle)
        for i in range(5):
            value = replay.results[i] or replay.failures[i]
            assert value.metadata["persistence"]["source"] == "store"
        assert len(client.creates) == len(client.uploads) == 1
