"""Batch creation is journaled before dispatch and reconciled by exact identity."""

import json
from types import SimpleNamespace

import pytest

from llenvs.inference.backends.api import OpenAIBackend
from llenvs.inference.backends.openai_batch import BatchHandle, OpenAIBatchBackend
from llenvs.inference.persistence import CallStore, UnknownOutcomeError
from llenvs.inference.protocol import ChatMessage, SamplingParams


class Client:
    max_retries = 0
    base_url = "https://offline.invalid/v1"

    def __init__(self):
        self.creates, self.uploads, self.outputs = [], [], {}
        self.files = SimpleNamespace(
            create=self.upload, content=lambda id: SimpleNamespace(content=self.outputs[id])
        )
        self.batches = SimpleNamespace(
            create=self.create,
            retrieve=self.retrieve,
            list=lambda **kw: SimpleNamespace(data=[], has_more=False),
        )

    def upload(self, *, file, purpose):
        assert purpose == "batch"
        self.uploads.append(file[1].read() if hasattr(file[1], "read") else file[1])
        return SimpleNamespace(id=f"file-{len(self.uploads)}")

    def create(self, **kwargs):
        self.creates.append(kwargs)
        return SimpleNamespace(id=f"batch-{len(self.creates)}")

    def retrieve(self, id):
        return SimpleNamespace(
            id=id, status="completed", output_file_id="output", error_file_id=None
        )


def backend(client):
    result = OpenAIBackend.__new__(OpenAIBackend)
    result._model = "gpt-4o"
    result._api_flavor = "openai"
    result._rate_limit_wait = 0
    result._rate_limit_max_retries = 0
    result._client = client
    return result


def completion(key, text):
    return {
        "custom_id": key,
        "response": {
            "status_code": 200,
            "body": {
                "id": key,
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        },
    }


def test_out_of_order_collection_replays_across_process_ownership(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=["a", "b"],
        )
        lines = [json.loads(line) for line in client.uploads[0].splitlines()]
        assert [line["custom_id"] for line in lines] == list(handle.keys)
        assert client.creates[0]["metadata"]["run_fingerprint"] == store.fingerprint
        serialized = json.loads(json.dumps(handle.to_dict()))
    client.outputs["output"] = (
        "\n".join(
            json.dumps(completion(key, text))
            for key, text in zip(reversed(handle.keys), ["b", "a"], strict=True)
        )
    ).encode()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        collection = api.collect(BatchHandle.from_dict(serialized))
        assert not collection.failures, collection.failures
        assert [result.text for result in collection.results] == ["a", "b"]
        assert not collection.pending and not collection.failures
        assert collection.results[0].metadata["pricing_tier"] == "batch"
        assert len(client.creates) == 1


def test_collection_identifies_each_call_and_replays_without_mutating_saved_results(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=["a", "b"],
        )
        client.outputs["output"] = b"\n".join(
            json.dumps(completion(key, text)).encode()
            for key, text in zip(handle.keys, ["a", "b"], strict=True)
        )
        first = api.collect(handle)
        for key, result in zip(handle.keys, first.results, strict=True):
            assert result.metadata["persistence"] == {
                "run_fingerprint": store.fingerprint,
                "call_key": key,
                "source": "batch",
            }
            assert result.prompt_tokens == 3 and result.completion_tokens == 1
            assert "persistence" not in store._records[key]["result"].metadata
        first.results[0].metadata["persistence"]["call_key"] = "tampered"
        first.results[0].metadata["request"]["model"] = "tampered"
        client.batches.retrieve = lambda *args: pytest.fail(
            "replayed collection contacted provider"
        )
        client.files.content = lambda *args: pytest.fail("replayed collection downloaded output")
        repeated = api.collect(handle)
        for key, result in zip(handle.keys, repeated.results, strict=True):
            assert result.metadata["persistence"] == {
                "run_fingerprint": store.fingerprint,
                "call_key": key,
                "source": "store",
            }
            assert result.metadata["request"]["model"] == "gpt-4o"
        fingerprint = store.fingerprint
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        results = OpenAIBatchBackend(backend(client), store).collect(handle).results
        assert results == repeated.results
        assert all(
            result.metadata["persistence"]["run_fingerprint"] == fingerprint for result in results
        )
        assert len(client.creates) == 1


def test_waiting_collection_preserves_newly_collected_provenance_across_polls(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, poll_interval=0)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=["a", "b"],
        )
        client.outputs["partial"] = json.dumps(completion(handle.keys[0], "a")).encode()
        client.outputs["output"] = json.dumps(completion(handle.keys[1], "b")).encode()
        responses = iter(
            [
                SimpleNamespace(
                    id="batch-1", status="in_progress", output_file_id="partial", error_file_id=None
                ),
                client.retrieve("batch-1"),
            ]
        )
        client.batches.retrieve = lambda *args: next(responses)
        results = api.collect(handle, wait=True).results
        assert [result.metadata["persistence"]["source"] for result in results] == [
            "batch",
            "batch",
        ]
        assert [result.metadata["persistence"]["call_key"] for result in results] == list(
            handle.keys
        )


def test_unknown_create_does_not_authorize_a_second_submission(tmp_path):
    client = Client()

    def uncertain(**kwargs):
        client.creates.append(kwargs)
        raise TimeoutError("lost create response")

    client.batches.create = uncertain
    messages = [[ChatMessage("user", "a")]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        with pytest.raises(UnknownOutcomeError):
            api.submit_chat_batch(messages, SamplingParams(), repeat_ids=["a"])
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(messages, SamplingParams(), repeat_ids=["a"])
        collection = api.collect(handle)
        assert collection.pending == list(handle.keys)
        assert "submission_unknown" in collection.statuses.values()
        assert len(client.creates) == 1


def test_unknown_batch_resubmission_requires_a_recorded_explicit_decision(tmp_path):
    client = Client()

    def uncertain(**kwargs):
        client.creates.append(kwargs)
        raise TimeoutError("lost create reply")

    client.batches.create = uncertain
    messages = [[ChatMessage("user", "a")]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        with pytest.raises(UnknownOutcomeError):
            api.submit_chat_batch(messages, SamplingParams(), repeat_ids=[1])
        original = api.submit_chat_batch(messages, SamplingParams(), repeat_ids=[1])
        with pytest.raises(ValueError):
            api.acknowledge_resubmission(
                original.submissions[0], reason="", accept_duplicate_expenditure=True
            )
        api.acknowledge_resubmission(
            original.submissions[0],
            reason="Provider could not recover it",
            accept_duplicate_expenditure=True,
        )
        assert len(client.creates) == 1
        client.batches.create = client.create
        retried = api.submit_chat_batch(messages, SamplingParams(), repeat_ids=[1])
        assert retried.keys == original.keys and retried.submissions != original.submissions
        assert len(client.creates) == 2
        assert (
            store._journals[original.submissions[0]]["recovery_decisions"][0][
                "duplicate_expenditure_accepted"
            ]
            is True
        )
        assert store._records[retried.keys[0]]["attempts"][0]["outcome_unknown"] is True


def test_files_split_at_actual_serialized_byte_and_request_limits(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=1)
        api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=[1, 2],
        )
        assert len(client.uploads) == 2
        assert all(len(payload.splitlines()) == 1 for payload in client.uploads)


def test_duplicate_or_unknown_output_ids_are_rejected(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        client.outputs["output"] = json.dumps(completion("unknown", "a")).encode()
        with pytest.raises(ValueError, match="custom_id"):
            api.collect(handle)


def test_retained_batch_files_stream_and_are_validated_on_reopen(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        payload = json.dumps(completion(handle.keys[0], "a")).encode()

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_bytes(self):
                yield payload[:7]
                yield payload[7:]

        client.files.with_streaming_response = SimpleNamespace(content=lambda id: Stream())
        client.files.content = lambda id: pytest.fail(
            "Batch output was buffered instead of streamed"
        )
        assert api.collect(handle).results[0].text == "a"
        retained = list((store.run_dir / "blobs").iterdir())
        assert len(retained) == 2
    retained[0].write_bytes(b"corrupt")
    from llenvs.inference.persistence import CheckpointCorruptionError

    with pytest.raises(CheckpointCorruptionError):
        CallStore.open(tmp_path, run_descriptor={})


def test_paginated_reconciliation_adopts_only_exact_submission_identity(tmp_path):
    client = Client()

    def uncertain(**kwargs):
        client.creates.append(kwargs)
        raise TimeoutError("lost create reply")

    client.batches.create = uncertain
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        with pytest.raises(UnknownOutcomeError):
            api.submit_chat_batch([[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1])
        manifest = next(iter(store._journals.values()))
        match = {
            "id": "recovered",
            "metadata": client.creates[0]["metadata"],
            "endpoint": manifest["endpoint"],
            "input_file_id": manifest["input_file_id"],
        }
        pages = []

        def listing(**kwargs):
            pages.append(kwargs)
            if len(pages) == 1:
                return SimpleNamespace(
                    data=[SimpleNamespace(id="unrelated", metadata={})], has_more=True
                )
            return SimpleNamespace(data=[SimpleNamespace(**match)], has_more=False)

        client.batches.list = listing
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="in_progress", output_file_id=None, error_file_id=None
        )
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        collection = api.collect(handle)
        assert pages[1]["after"] == "unrelated"
        assert collection.pending == list(handle.keys)
        assert next(iter(store._journals.values()))["batch_id"] == "recovered"
        assert len(client.creates) == 1


@pytest.mark.parametrize("terminal_status", ["expired", "cancelled", "failed"])
def test_partial_expiry_collects_success_before_an_explicit_failed_slot_resubmission(
    tmp_path, terminal_status
):
    client = Client()
    batch = [[ChatMessage("user", "a")], [ChatMessage("user", "b")]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=[1, 2])
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status=terminal_status, output_file_id="output", error_file_id=None
        )
        first = api.collect(handle)
        assert first.results[0].text == "a" and set(first.failures) == {1}
        assert len(client.creates) == 1
        retry = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=[1, 2])
        assert len(client.creates) == 2
        assert len(client.uploads[1].splitlines()) == 1
        assert json.loads(client.uploads[1])["custom_id"] == handle.keys[1]
        client.outputs["second-output"] = json.dumps(completion(handle.keys[1], "b")).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="completed", output_file_id="second-output", error_file_id=None
        )
        final = api.collect(retry)
        assert [result.text for result in final.results] == ["a", "b"]
        assert [result.metadata["persistence"]["source"] for result in final.results] == [
            "store",
            "batch",
        ]
        assert [result.metadata["persistence"]["call_key"] for result in final.results] == list(
            handle.keys
        )


def test_cached_batch_results_do_not_require_provider_files_or_api_access(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()
        assert api.collect(handle).results[0].text == "a"
    client.batches.retrieve = lambda id: pytest.fail("Paid outcome replay contacted the provider")
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        assert OpenAIBatchBackend(backend(client), store).collect(handle).results[0].text == "a"


def test_durable_batch_receipt_reparses_without_contacting_provider(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        source = backend(client)
        api = OpenAIBatchBackend(source, store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()
        parse = source.reparse_receipt
        source.reparse_receipt = lambda *args: (_ for _ in ()).throw(
            ValueError("interrupted parser")
        )
        assert set(api.collect(handle).failures) == {0}
        source.reparse_receipt = parse
        client.batches.retrieve = lambda id: pytest.fail(
            "Durable receipt recovery contacted the provider"
        )
        result = api.collect(handle).results[0]
        assert result.text == "a"
        assert result.metadata["persistence"] == {
            "run_fingerprint": store.fingerprint,
            "call_key": handle.keys[0],
            "source": "receipt",
        }
        assert len(client.creates) == 1


def test_utf8_byte_limit_is_inclusive_and_checked_before_upload(tmp_path):
    client = Client()
    messages = [[ChatMessage("user", "α😀")]]
    body = backend(client)._request_kwargs(messages[0], SamplingParams())
    size = len(OpenAIBatchBackend._line("0" * 64, body))
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_bytes=size - 1)
        with pytest.raises(ValueError, match="byte limit"):
            api.submit_chat_batch(messages, SamplingParams(), repeat_ids=[1])
        assert not client.uploads and not list(store._records)
        api.max_file_bytes = size
        handle = api.submit_chat_batch(messages * 2, SamplingParams(), repeat_ids=[1, 2])
        assert len(handle.submissions) == 2
        assert [len(value) for value in client.uploads] == [size, size]


def test_duplicate_outputs_fail_before_any_result_is_committed(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        line = json.dumps(completion(handle.keys[0], "a"))
        client.outputs["output"] = (line + "\n" + line).encode()
        with pytest.raises(ValueError, match="duplicate custom_id"):
            api.collect(handle)
        assert store._records[handle.keys[0]]["state"] == "pending"


def test_multiple_reconciliation_matches_never_create_again(tmp_path):
    client = Client()

    def uncertain(**kwargs):
        client.creates.append(kwargs)
        raise TimeoutError("lost")

    client.batches.create = uncertain
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        with pytest.raises(UnknownOutcomeError):
            api.submit_chat_batch([[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1])
        manifest = next(iter(store._journals.values()))
        client.batches.list = lambda **kwargs: SimpleNamespace(
            data=[
                SimpleNamespace(
                    id=identifier,
                    metadata=client.creates[0]["metadata"],
                    endpoint=manifest["endpoint"],
                    input_file_id=manifest["input_file_id"],
                )
                for identifier in ["one", "two"]
            ],
            has_more=False,
        )
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        with pytest.raises(UnknownOutcomeError):
            api.collect(handle)
        assert len(client.creates) == 1


def test_handle_must_cover_every_unfinished_key_and_its_current_submission(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        with pytest.raises(ValueError, match="handle"):
            api.collect(BatchHandle(handle.run_fingerprint, (), handle.keys))
        with pytest.raises(ValueError, match="handle"):
            api.collect(BatchHandle(handle.run_fingerprint, handle.submissions, ("f" * 64,)))
        assert len(client.creates) == 1


def test_intent_without_call_pointer_is_adopted_without_duplicate_upload(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        api._submit = lambda manifest: None
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        store.update(handle.keys[0], "reserved", submission=None)
        assert not client.uploads
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        restored = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="in_progress", output_file_id=None, error_file_id=None
        )
        result = api.collect(restored)
        assert len(client.uploads) == len(client.creates) == 1
        assert result.pending == list(handle.keys)


def _collect_in_process(root, serialized, payload, connection):
    client = Client()
    client.outputs["output"] = payload
    with CallStore.open(root, run_descriptor={}) as store:
        result = OpenAIBatchBackend(backend(client), store).collect(
            BatchHandle.from_dict(serialized)
        )
        connection.send([item.text for item in result.results])


def test_submit_and_collect_transfer_exclusive_ownership_between_processes(tmp_path):
    import multiprocessing

    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        handle = OpenAIBatchBackend(backend(client), store).submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    payload = json.dumps(completion(handle.keys[0], "a")).encode()
    process = context.Process(
        target=_collect_in_process, args=(str(tmp_path), handle.to_dict(), payload, child)
    )
    process.start()
    try:
        assert parent.poll(15), "Collector did not acknowledge the durable result"
        assert parent.recv() == ["a"]
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()


def _submission_crash_worker(root, stage, connection, release):
    def stop(manifest, **values):
        connection.send(
            {
                "run": manifest["run"],
                "uuid": manifest["id"],
                "digest": manifest["sha256"],
                "endpoint": manifest["endpoint"],
                "input_file_id": manifest["input_file_id"],
                **values,
            }
        )
        release.recv()

    class InterruptedClient(Client):
        def upload(self, **kwargs):
            result = super().upload(**kwargs)
            if stage == "uploaded":
                stop(next(iter(store._journals.values())), input_file_id=result.id)
            return result

        def create(self, **kwargs):
            result = super().create(**kwargs)
            if stage == "created":
                stop(next(iter(store._journals.values())), batch_id=result.id)
            return result

    with CallStore.open(root, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(InterruptedClient()), store)
        submit, write = api._submit, api._write
        if stage == "intent":

            def before_upload(manifest):
                stop(manifest)
                submit(manifest)

            api._submit = before_upload
        if stage == "acknowledged":

            def after_ack(manifest):
                write(manifest)
                if manifest["batch_id"] is not None:
                    stop(manifest, batch_id=manifest["batch_id"])

            api._write = after_ack
        api.submit_chat_batch([[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1])


@pytest.mark.parametrize("stage", ["intent", "uploaded", "created", "acknowledged"])
def test_process_death_at_submission_boundaries_preserves_dispatch_eligibility(tmp_path, stage):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    release, held = context.Pipe(duplex=False)
    process = context.Process(
        target=_submission_crash_worker, args=(str(tmp_path), stage, child, release)
    )
    process.start()
    try:
        assert parent.poll(15), "Submission did not reach its synchronized crash point"
        saved = parent.recv()
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()
        client = Client()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="in_progress", output_file_id=None, error_file_id=None
        )
        if stage == "created":
            client.batches.list = lambda **kwargs: SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id=saved["batch_id"],
                        endpoint=saved["endpoint"],
                        input_file_id=saved["input_file_id"],
                        metadata={
                            "run_fingerprint": saved["run"],
                            "submission_uuid": saved["uuid"],
                            "file_digest": saved["digest"],
                        },
                    )
                ],
                has_more=False,
            )
        with CallStore.open(tmp_path, run_descriptor={}) as store:
            api = OpenAIBatchBackend(backend(client), store)
            handle = api.submit_chat_batch(
                [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
            )
            assert api.collect(handle).pending == list(handle.keys)
            assert len(client.creates) == (1 if stage in {"intent", "uploaded"} else 0)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()
        release.close()
        held.close()


def _collection_crash_worker(root, connection, release):
    client = Client()
    with CallStore.open(root, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=[1, 2],
        )
        client.outputs["output"] = b"\n".join(
            json.dumps(completion(key, text)).encode()
            for key, text in zip(handle.keys, ["a", "b"], strict=True)
        )
        commit = api._commit_receipt

        def interrupted(key, receipt, params):
            result = commit(key, receipt, params)
            connection.send(handle.to_dict())
            release.recv()
            return result

        api._commit_receipt = interrupted
        api.collect(handle)


def test_process_death_during_collection_reuses_retained_files_without_api_access(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    release, held = context.Pipe(duplex=False)
    process = context.Process(target=_collection_crash_worker, args=(str(tmp_path), child, release))
    process.start()
    try:
        assert parent.poll(15), "Collector did not commit its first result"
        handle = BatchHandle.from_dict(parent.recv())
        process.kill()
        process.join(timeout=10)
        client = Client()
        client.batches.retrieve = lambda id: pytest.fail(
            "Retained completed Batch needed the provider"
        )
        with CallStore.open(tmp_path, run_descriptor={}) as store:
            result = OpenAIBatchBackend(backend(client), store).collect(handle)
            assert [item.text for item in result.results] == ["a", "b"]
            assert not client.creates
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()
        release.close()
        held.close()


def test_real_sdk_upload_create_retrieve_and_streaming_collection(tmp_path):
    from email import policy
    from email.parser import BytesParser

    import httpx
    from openai import OpenAI

    requests, state = [], {}

    def respond(request):
        path = request.url.path
        requests.append((request.method, path))
        if path == "/v1/files" and request.method == "POST":
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode()
                + request.content
            )
            part = next(
                part for part in message.iter_parts() if part.get_filename() == "requests.jsonl"
            )
            state["key"] = json.loads(part.get_payload(decode=True))["custom_id"]
            return httpx.Response(
                200, json={"id": "file-input", "object": "file", "purpose": "batch"}
            )
        if path == "/v1/batches" and request.method == "POST":
            body = json.loads(request.content)
            assert body["input_file_id"] == "file-input" and body["completion_window"] == "24h"
            state["metadata"] = body["metadata"]
            return httpx.Response(200, json={"id": "batch-one"})
        if path == "/v1/batches/batch-one":
            return httpx.Response(
                200,
                json={
                    "id": "batch-one",
                    "status": "completed",
                    "endpoint": "/v1/chat/completions",
                    "input_file_id": "file-input",
                    "output_file_id": "file-output",
                    "metadata": state["metadata"],
                },
            )
        if path == "/v1/files/file-output/content":
            return httpx.Response(200, content=json.dumps(completion(state["key"], "a")).encode())
        pytest.fail(f"Unexpected request: {request.method} {path}")

    with OpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        with CallStore.open(tmp_path, run_descriptor={}) as store:
            api = OpenAIBatchBackend(backend(client), store)
            handle = api.submit_chat_batch(
                [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
            )
            assert api.collect(handle).results[0].text == "a"
    assert requests == [
        ("POST", "/v1/files"),
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches/batch-one"),
        ("GET", "/v1/files/file-output/content"),
    ]
