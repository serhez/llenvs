"""Batch creation is journaled before dispatch and reconciled by exact identity."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from llenvs.inference.backends.api import OpenAIBackend
from llenvs.inference.backends.openai_batch import (
    BatchCollection,
    BatchHandle,
    BatchUnfinished,
    OpenAIBatchBackend,
)
from llenvs.inference.persistence import CallStore, CheckpointError, UnknownOutcomeError
from llenvs.inference.protocol import (
    ChatMessage,
    LogprobsNotReturnedError,
    MalformedResponseError,
    PartialBatchError,
    QuotaExhaustedError,
    RefusedByPolicyError,
    ReplayedBackendError,
    SamplingParams,
    encode_backend_error,
)


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


def completion(key, text, *, logprobs=False):
    entry = {
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
    if logprobs:
        entry["response"]["body"]["choices"][0]["logprobs"] = {
            "content": [{"token": text, "logprob": -0.1, "top_logprobs": []}]
        }
    return entry


def generated_failure(key, kind):
    entry = completion(key, "unusable answer")
    body = entry["response"]["body"]
    if kind == "refusal":
        body["choices"][0]["message"].update(content=None, refusal="Cannot answer")
    elif kind == "malformed":
        body["choices"] = []
    else:
        assert kind == "missing_logprobs"
    return entry


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("stage", ["retrieve", "download"])
@pytest.mark.parametrize("outcome", ["success", "refusal", "server", "expired", "unfinished"])
def test_collection_error_carries_committed_outcomes_and_reopens(
    tmp_path, validation, stage, outcome
):
    client = Client()
    error = APIConnectionError(request=httpx.Request("GET", "https://offline.invalid/batches"))
    error.__cause__ = OSError("connection lost")
    error.metadata = {"transport": "offline fixture"}
    original_args, original_cause, original_request = error.args, error.__cause__, error.request
    first_status = "expired" if outcome == "expired" else "completed"
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=2)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", str(index))] for index in range(4)],
            SamplingParams(),
            repeat_ids=list(range(4)),
        )
        assert len(handle.submissions) == 2
        entries = [completion(handle.keys[0], "0"), completion(handle.keys[1], "1")]
        if outcome == "refusal":
            entries[1] = generated_failure(handle.keys[1], "refusal")
        elif outcome == "server":
            entries[1] = {
                "custom_id": handle.keys[1],
                "response": {
                    "status_code": 503,
                    "body": {"error": {"code": "server_error", "message": "Try again"}},
                },
            }
        elif outcome == "expired":
            entries[1] = {
                "custom_id": handle.keys[1],
                "response": None,
                "error": {"code": "batch_expired", "message": "Not executed"},
            }
        elif outcome == "unfinished":
            entries.pop()
        client.outputs["batch-1"] = b"\n".join(json.dumps(entry).encode() for entry in entries)
        client.outputs["batch-2"] = b"\n".join(
            json.dumps(completion(handle.keys[index], str(index))).encode() for index in [3, 2]
        )

        def retrieve(identifier):
            if stage == "retrieve" and identifier == "batch-2":
                raise error
            return SimpleNamespace(
                id=identifier,
                status=first_status if identifier == "batch-1" else "completed",
                output_file_id=identifier,
                error_file_id=None,
            )

        def download(identifier):
            if stage == "download" and identifier == "batch-2":
                raise error
            return SimpleNamespace(content=client.outputs[identifier])

        client.batches.retrieve, client.files.content = retrieve, download
        with pytest.raises(APIConnectionError) as caught:
            api.collect(handle)
        assert caught.value is error
        assert error.args == original_args and error.__cause__ is original_cause
        assert error.request is original_request and error.metadata == {
            "transport": "offline fixture"
        }
        partial = error.partial_collection
        assert isinstance(partial, BatchCollection)
        assert [result.text if result else None for result in partial.results] == [
            "0",
            "1" if outcome == "success" else None,
            None,
            None,
        ]
        assert partial.pending == list(handle.keys[2:])
        assert set(partial.failures) == (set() if outcome == "success" else {1})
        assert partial.statuses == {
            handle.submissions[0]: first_status,
            **({handle.submissions[1]: "completed"} if stage == "download" else {}),
        }
        outcomes = [partial.results[0], partial.results[1] or partial.failures[1]]
        for key, result in zip(handle.keys[:2], outcomes, strict=True):
            assert result.metadata["persistence"] == {
                "run_fingerprint": handle.run_fingerprint,
                "call_key": key,
                "source": "batch",
                "previous_attempts": 0,
            }
        assert partial.results[0].metadata["attempts"][0]["response_id"] == handle.keys[0]
        assert partial.results[0].metadata["attempts"][0]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 1,
        }
        saved_error = encode_backend_error(partial.failures[1]) if partial.failures else None
        if outcome == "unfinished":
            assert type(partial.failures[1]) is BatchUnfinished
            assert saved_error["status_code"] is None and saved_error["attempts"] is None
        elif outcome == "refusal":
            assert isinstance(partial.failures[1], RefusedByPolicyError)
            assert saved_error["status_code"] == 200 and not saved_error["transient"]
            assert saved_error["attempts"][0]["usage"]["prompt_tokens"] == 3
        elif outcome == "server":
            assert saved_error["status_code"] == 503 and saved_error["transient"]
            assert saved_error["code"] == "server_error"
        elif outcome == "expired":
            assert saved_error["code"] == "batch_expired" and not saved_error["transient"]
        partial.results[0].metadata["attempts"][0]["usage"]["prompt_tokens"] = 999
        partial.results[0].metadata["persistence"]["source"] = "changed by caller"
        if outcome == "refusal":
            partial.failures[1].attempts[0]["usage"]["prompt_tokens"] = 999

    resumed_client = Client()

    def forbidden(*args, **kwargs):
        pytest.fail("Recovery resubmitted requests or fetched the first submission again")

    def resume_retrieve(identifier):
        assert identifier == "batch-2"
        return SimpleNamespace(
            id=identifier, status="completed", output_file_id=identifier, error_file_id=None
        )

    def resume_download(identifier):
        assert identifier == "batch-2"
        return SimpleNamespace(content=client.outputs[identifier])

    resumed_client.files.create = resumed_client.batches.create = resumed_client.batches.list = (
        forbidden
    )
    resumed_client.batches.retrieve, resumed_client.files.content = resume_retrieve, resume_download
    with CallStore.open(tmp_path, run_descriptor={}, validation=validation) as store:
        collection = OpenAIBatchBackend(backend(resumed_client), store).collect(handle)
        assert [result.text if result else None for result in collection.results] == [
            "0",
            "1" if outcome == "success" else None,
            "2",
            "3",
        ]
        assert not collection.pending and set(collection.failures) == set(partial.failures)
        for index, key in enumerate(handle.keys):
            result = collection.results[index] or collection.failures[index]
            assert result.metadata["persistence"] == {
                "run_fingerprint": handle.run_fingerprint,
                "call_key": key,
                "source": "store" if index < 2 else "batch",
                "previous_attempts": 0,
            }
        assert collection.results[0].metadata["attempts"][0]["usage"]["prompt_tokens"] == 3
        if saved_error:
            replayed = collection.failures[1]
            if isinstance(replayed, ReplayedBackendError):
                assert replayed.original_error_type == saved_error["type"]
                assert encode_backend_error(replayed) == {
                    **saved_error,
                    "type": "ReplayedBackendError",
                }
            else:
                assert encode_backend_error(replayed) == saved_error
        assert len(client.creates) == len(client.uploads) == 2


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_backend_close_preserves_committed_collection_progress(tmp_path, validation):
    client = Client()
    source = backend(client)
    entered, release = Event(), Event()
    error = APIConnectionError(request=httpx.Request("GET", "https://offline.invalid/batches"))
    client.close = release.set
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(source, store, max_file_requests=1)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=[0, 1],
        )
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()

        def retrieve(identifier):
            if identifier == "batch-2":
                entered.set()
                assert release.wait(5), "backend.close() did not release the blocked retrieval"
                raise error
            return client.retrieve(identifier)

        client.batches.retrieve = retrieve
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(api.collect, handle)
            try:
                assert entered.wait(5), "Collection did not reach the second retrieval"
                source.close()
                assert source._client is None
                with pytest.raises(APIConnectionError) as caught:
                    future.result(timeout=5)
                assert caught.value is error
                partial = error.partial_collection
                assert [result.text if result else None for result in partial.results] == [
                    "a",
                    None,
                ]
                assert partial.results[0].metadata["persistence"]["source"] == "batch"
                assert not partial.failures and partial.pending == [handle.keys[1]]
            finally:
                release.set()
    resumed = Client()
    resumed.outputs["output"] = json.dumps(completion(handle.keys[1], "b")).encode()
    with CallStore.open(tmp_path, run_descriptor={}, validation=validation) as store:
        collection = OpenAIBatchBackend(backend(resumed), store).collect(handle)
        assert [result.text for result in collection.results] == ["a", "b"]
        assert [result.metadata["persistence"]["source"] for result in collection.results] == [
            "store",
            "batch",
        ]
        assert not collection.failures and not collection.pending and not resumed.creates


@pytest.mark.parametrize("outcome", ["success", "unfinished"])
def test_collection_error_preserves_progress_from_earlier_wait_polls(tmp_path, outcome):
    client = Client()
    error = APIConnectionError(request=httpx.Request("GET", "https://offline.invalid/batches"))
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=1, poll_interval=0)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=[0, 1],
        )
        client.outputs["output"] = (
            json.dumps(completion(handle.keys[0], "a")).encode() if outcome == "success" else b""
        )
        polls = 0

        def retrieve(identifier):
            nonlocal polls
            if identifier == "batch-1":
                return client.retrieve(identifier)
            polls += 1
            if polls > 1:
                raise error
            return SimpleNamespace(
                id=identifier, status="in_progress", output_file_id=None, error_file_id=None
            )

        client.batches.retrieve = retrieve
        with pytest.raises(APIConnectionError) as caught:
            api.collect(handle, wait=True)
        assert caught.value is error and polls == 2
        partial = error.partial_collection
        if outcome == "success":
            assert partial.results[0].text == "a" and not partial.failures
            first = partial.results[0]
        else:
            assert partial.results[0] is None and set(partial.failures) == {0}
            first = partial.failures[0]
            assert type(first) is BatchUnfinished
        assert first.metadata["persistence"]["source"] == "batch"
        assert partial.results[1] is None and partial.pending == [handle.keys[1]]
        assert partial.statuses == dict(
            zip(handle.submissions, ["completed", "in_progress"], strict=True)
        )
        client.batches.retrieve = client.retrieve
        client.outputs["output"] = json.dumps(completion(handle.keys[1], "b")).encode()
        recovered = api.collect(handle, wait=True)
        assert [
            result.metadata["persistence"]["source"]
            for result in [recovered.results[0] or recovered.failures[0], recovered.results[1]]
        ] == [
            "store",
            "batch",
        ]
        assert partial.pending == [handle.keys[1]] and partial.results[1] is None
        assert first.metadata["persistence"]["source"] == "batch"


def test_collection_error_partway_through_commits_carries_results_and_failures(
    tmp_path, monkeypatch
):
    client = Client()
    error = CheckpointError("Injected failure before the third receipt was written")
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", str(index))] for index in range(3)],
            SamplingParams(),
            repeat_ids=list(range(3)),
        )
        client.outputs["output"] = b"\n".join(
            json.dumps(entry).encode()
            for entry in [
                completion(handle.keys[0], "0"),
                generated_failure(handle.keys[1], "refusal"),
                completion(handle.keys[2], "2"),
            ]
        )
        update = store.update

        def fail_third(key, state, **changes):
            if key == handle.keys[2] and state == "received":
                raise error
            return update(key, state, **changes)

        monkeypatch.setattr(store, "update", fail_third)
        with pytest.raises(CheckpointError) as caught:
            api.collect(handle)
        assert caught.value is error
        partial = error.partial_collection
        assert [result.text if result else None for result in partial.results] == ["0", None, None]
        assert set(partial.failures) == {1} and partial.pending == [handle.keys[2]]
        assert isinstance(partial.failures[1], RefusedByPolicyError)
        for result in [partial.results[0], partial.failures[1]]:
            assert result.metadata["persistence"]["source"] == "batch"
    client.batches.retrieve = client.files.content = lambda *args: pytest.fail(
        "Retained files not reused"
    )
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        recovered = OpenAIBatchBackend(backend(client), store).collect(handle)
        assert [result.text if result else None for result in recovered.results] == ["0", None, "2"]
        assert set(recovered.failures) == {1} and not recovered.pending
        assert [
            result.metadata["persistence"]["source"]
            for result in [recovered.results[0], recovered.failures[1], recovered.results[2]]
        ] == ["store", "store", "batch"]


def test_collection_error_keeps_store_receipt_and_batch_sources(tmp_path, monkeypatch):
    client = Client()
    error = APIConnectionError(request=httpx.Request("GET", "https://offline.invalid/batches"))
    source = backend(client)
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(source, store, max_file_requests=1)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", str(index))] for index in range(4)],
            SamplingParams(),
            repeat_ids=list(range(4)),
        )
        for index in range(3):
            client.outputs[f"batch-{index + 1}"] = json.dumps(
                completion(handle.keys[index], str(index))
            ).encode()

        def retrieve(identifier):
            if identifier == "batch-4":
                raise error
            return SimpleNamespace(
                id=identifier, status="completed", output_file_id=identifier, error_file_id=None
            )

        client.batches.retrieve = retrieve
        api.collect(
            BatchHandle(handle.run_fingerprint, (handle.submissions[0],), (handle.keys[0],))
        )
        reparse = source.reparse_receipt
        with monkeypatch.context() as patch:

            def fail_parse(*args):
                raise ValueError("Temporary local parser failure")

            patch.setattr(source, "reparse_receipt", fail_parse)
            failed = api.collect(
                BatchHandle(handle.run_fingerprint, (handle.submissions[1],), (handle.keys[1],))
            )
            assert isinstance(failed.failures[0], ValueError)
        assert source.reparse_receipt == reparse
        with pytest.raises(APIConnectionError):
            api.collect(handle)
        partial = error.partial_collection
        assert [result.text if result else None for result in partial.results] == [
            "0",
            "1",
            "2",
            None,
        ]
        assert [result.metadata["persistence"]["source"] for result in partial.results[:3]] == [
            "store",
            "receipt",
            "batch",
        ]
        assert all(
            result.metadata["persistence"]["previous_attempts"] == 0
            for result in partial.results[:3]
        )
        assert not partial.failures and partial.pending == [handle.keys[3]]


def test_collection_error_without_outcomes_keeps_the_original_exception(tmp_path):
    client = Client()
    error = APIConnectionError(request=httpx.Request("GET", "https://offline.invalid/batches"))
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[0]
        )

        def fail_retrieve(*args):
            raise error

        client.batches.retrieve = fail_retrieve
        with pytest.raises(APIConnectionError) as caught:
            api.collect(handle)
        assert caught.value is error and not hasattr(error, "partial_collection")


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


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("second_view", ["remaining", "all"])
def test_subset_collection_reports_each_key_as_batch_once(
    tmp_path, validation, reopen, second_view
):
    client = Client()
    batch = [[ChatMessage("user", str(index))] for index in range(5)]
    with ExitStack() as stack:
        store = stack.enter_context(
            CallStore.open(tmp_path, run_descriptor={}, validation=validation)
        )
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=list(range(5)))
        subset = api.submit_chat_batch(batch[:2], SamplingParams(), repeat_ids=[0, 1])
        client.outputs["output"] = b"\n".join(
            json.dumps(completion(original.keys[index], str(index))).encode()
            for index in reversed(range(5))
        )
        first = api.collect(subset)
        assert [result.text for result in first.results] == ["0", "1"]
        assert not first.failures and not first.pending
        assert list(first.statuses.values()) == ["completed"]
        assert [result.metadata["persistence"]["source"] for result in first.results] == [
            "batch",
            "batch",
        ]
        if reopen:
            store.close()
            store = stack.enter_context(
                CallStore.open(tmp_path, run_descriptor={}, validation=validation)
            )
            api = OpenAIBatchBackend(backend(client), store)

        def forbidden(*args, **kwargs):
            pytest.fail("Collection did not reuse retained Batch files")

        client.files.create = client.files.content = forbidden
        client.batches.create = client.batches.retrieve = client.batches.list = forbidden
        indices = [2, 3, 4] if second_view == "remaining" else list(range(5))
        rebuilt = api.submit_chat_batch(
            [batch[index] for index in indices], SamplingParams(), repeat_ids=indices
        )
        second = api.collect(rebuilt)
        assert [result.text for result in second.results] == [str(index) for index in indices]
        assert not second.failures and not second.pending
        assert second.statuses == first.statuses
        for index, result in zip(indices, second.results, strict=True):
            assert result.metadata["persistence"] == {
                "run_fingerprint": original.run_fingerprint,
                "call_key": original.keys[index],
                "source": "store" if index < 2 else "batch",
                "previous_attempts": 0,
            }
            assert result.prompt_tokens == 3 and result.completion_tokens == 1
        final = api.collect(original)
        assert [result.metadata["persistence"]["source"] for result in final.results] == [
            "store"
        ] * 5
        assert len(client.creates) == len(client.uploads) == 1


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize(
    "failure_kind", ["refusal", "malformed", "missing_logprobs", "server", "expired", "unfinished"]
)
def test_subset_collection_preserves_first_named_failure_after_reopen(
    tmp_path, validation, failure_kind
):
    client = Client()
    batch = [[ChatMessage("user", str(index))] for index in range(5)]
    params = SamplingParams(logprobs=failure_kind == "missing_logprobs")
    terminal_status = "expired" if failure_kind == "expired" else "completed"
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(batch, params, repeat_ids=list(range(5)))
        subset = api.submit_chat_batch(batch[:2], params, repeat_ids=[0, 1])
        entries = [
            completion(key, str(index), logprobs=params.logprobs)
            for index, key in enumerate(original.keys)
        ]
        failed_key = original.keys[3]
        if failure_kind == "unfinished":
            entries.pop(3)
        elif failure_kind == "expired":
            entries[3] = {
                "custom_id": failed_key,
                "response": None,
                "error": {"code": "batch_expired", "message": "Request was not executed"},
            }
        elif failure_kind == "server":
            entries[3] = {
                "custom_id": failed_key,
                "response": {
                    "status_code": 500,
                    "body": {"error": {"code": "server_error", "message": "Try again"}},
                },
            }
        else:
            entries[3] = generated_failure(failed_key, failure_kind)
        client.outputs["output"] = b"\n".join(json.dumps(entry).encode() for entry in entries)
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status=terminal_status, output_file_id="output", error_file_id=None
        )
        first = api.collect(subset)
        assert [result.text for result in first.results] == ["0", "1"]
        assert not first.failures and not first.pending

    def forbidden(*args, **kwargs):
        pytest.fail("An unreported Batch outcome caused provider access or another submission")

    client.files.create = client.files.content = forbidden
    client.batches.create = client.batches.retrieve = client.batches.list = forbidden
    with CallStore.open(tmp_path, run_descriptor={}, validation=validation) as store:
        api = OpenAIBatchBackend(backend(client), store)
        remaining = api.submit_chat_batch(batch[2:], params, repeat_ids=[2, 3, 4])
        second = api.collect(remaining)
        assert [result.text if result else None for result in second.results] == ["2", None, "4"]
        assert set(second.failures) == {1} and not second.pending
        assert list(second.statuses.values()) == [terminal_status]
        error = second.failures[1]
        outcomes = [second.results[0], error, second.results[2]]
        for key, outcome in zip(remaining.keys, outcomes, strict=True):
            assert outcome.metadata["persistence"] == {
                "run_fingerprint": original.run_fingerprint,
                "call_key": key,
                "source": "batch",
                "previous_attempts": 0,
            }
        if failure_kind == "unfinished":
            assert type(error).__name__ == "BatchUnfinished"
        elif failure_kind == "expired":
            assert error.code == "batch_expired"
            assert error.status_code is None and not error.transient
        elif failure_kind == "server":
            assert error.code == "server_error"
            assert error.status_code == 500 and error.transient
        else:
            assert isinstance(
                error,
                {
                    "refusal": RefusedByPolicyError,
                    "malformed": MalformedResponseError,
                    "missing_logprobs": LogprobsNotReturnedError,
                }[failure_kind],
            )
            assert error.status_code == 200 and not error.transient
        again = api.collect(remaining)
        assert set(again.failures) == {1} and not again.pending
        assert again.failures[1].metadata["persistence"] == {
            **error.metadata["persistence"],
            "source": "store",
        }
        assert [
            result.metadata["persistence"]["source"]
            for result in [again.results[0], again.results[2]]
        ] == ["store", "store"]
        assert (
            encode_backend_error(again.failures[1])["attempts"]
            == encode_backend_error(error)["attempts"]
        )
        if failure_kind == "unfinished":
            assert isinstance(again.failures[1], ReplayedBackendError)
            assert again.failures[1].original_error_type == "BatchUnfinished"
        assert len(client.creates) == len(client.uploads) == 1


def test_waiting_subset_leaves_other_keys_new_for_a_later_collection(tmp_path):
    client = Client()
    batch = [[ChatMessage("user", text)] for text in ["a", "b", "c"]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, poll_interval=0)
        original = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=[0, 1, 2])
        subset = api.submit_chat_batch(batch[:2], SamplingParams(), repeat_ids=[0, 1])
        client.outputs["partial"] = b"\n".join(
            json.dumps(completion(original.keys[index], ["a", "b", "c"][index])).encode()
            for index in [0, 2]
        )
        client.outputs["output"] = json.dumps(completion(original.keys[1], "b")).encode()
        responses = iter(
            [
                SimpleNamespace(
                    id="batch-1", status="in_progress", output_file_id="partial", error_file_id=None
                ),
                client.retrieve("batch-1"),
            ]
        )
        client.batches.retrieve = lambda *args: next(responses)
        first = api.collect(subset, wait=True)
        assert [result.text for result in first.results] == ["a", "b"]
        assert [result.metadata["persistence"]["source"] for result in first.results] == [
            "batch",
            "batch",
        ]
        client.batches.retrieve = lambda *args: pytest.fail("Terminal batch was retrieved again")
        remaining = api.submit_chat_batch(batch[2:], SamplingParams(), repeat_ids=[2])
        result = api.collect(remaining).results[0]
        assert result.text == "c" and result.metadata["persistence"]["source"] == "batch"
        assert api.collect(remaining).results[0].metadata["persistence"]["source"] == "store"


def test_batch_unfinished_is_exported_on_first_collection_and_plain_on_replay(tmp_path):
    from llenvs.inference.backends import BatchUnfinished
    from llenvs.inference.backends.openai_batch import BatchUnfinished as NativeBatchUnfinished

    assert BatchUnfinished is NativeBatchUnfinished
    assert issubclass(BatchUnfinished, RuntimeError)
    client = Client()
    client.outputs["output"] = b""
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[0]
        )
        first = api.collect(handle)
        assert type(first.failures[0]) is BatchUnfinished
        assert first.failures[0].metadata["persistence"]["source"] == "batch"
        saved = encode_backend_error(first.failures[0])
        assert saved["type"] == "BatchUnfinished"
        assert saved["status_code"] is None and not saved["transient"]
        assert saved["attempts"] is None
    client.batches.retrieve = lambda *args: pytest.fail("Missing-output replay contacted provider")
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        error = OpenAIBatchBackend(backend(client), store).collect(handle).failures[0]
        assert isinstance(error, ReplayedBackendError)
        assert error.original_error_type == "BatchUnfinished"
        assert error.metadata["persistence"]["source"] == "store"
        assert error.status_code is None and not error.transient
        assert error.attempts is None


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
                "previous_attempts": 0,
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
                "previous_attempts": 0,
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


def test_failed_collection_slots_identify_new_and_replayed_errors(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")], [ChatMessage("user", "b")]],
            SamplingParams(),
            repeat_ids=["a", "b"],
        )
        client.outputs["output"] = b"\n".join(
            [
                json.dumps(completion(handle.keys[0], "a")).encode(),
                json.dumps(
                    {
                        "custom_id": handle.keys[1],
                        "error": {"code": "bad_request", "message": "rejected"},
                    }
                ).encode(),
            ]
        )
        first = api.collect(handle)
        assert first.results[0].text == "a" and first.results[1] is None
        provenance = first.failures[1].metadata["persistence"]
        assert provenance == {
            "run_fingerprint": store.fingerprint,
            "call_key": handle.keys[1],
            "source": "batch",
            "previous_attempts": 0,
        }
        client.batches.retrieve = lambda *args: pytest.fail("Replayed failure contacted provider")
        replay = api.collect(handle)
        assert replay.failures[1].metadata["persistence"] == {**provenance, "source": "store"}
        assert "metadata" not in store._records[handle.keys[1]]["error"]
        assert len(client.creates) == 1
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        replay = OpenAIBatchBackend(backend(client), store).collect(handle)
        assert replay.failures[1].metadata["persistence"] == {**provenance, "source": "store"}


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize(
    "kind,status_code,code,transient",
    [
        ("http", 500, "server_error", True),
        ("http", 503, "service_unavailable", True),
        ("http", 500, None, True),
        ("http", 429, "rate_limit_exceeded", True),
        ("http", 400, "invalid_request_error", False),
        ("http", 429, "insufficient_quota", False),
        ("http", 429, "billing_hard_limit_reached", False),
        ("expired", None, "batch_expired", False),
        ("refusal", 200, None, False),
        ("malformed", 200, None, False),
        ("missing_logprobs", 200, None, False),
    ],
)
def test_failed_slots_retain_status_code_classification_and_attempts_after_reopen(
    tmp_path, validation, kind, status_code, code, transient
):
    client = Client()
    params = SamplingParams(logprobs=kind == "missing_logprobs")
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 7},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch([[ChatMessage("user", "question")]], params, repeat_ids=[0])
        key = handle.keys[0]
        if kind == "expired":
            entry = {
                "custom_id": key,
                "response": None,
                "error": {"code": code, "message": "Request was not executed before expiry"},
            }
        elif kind == "http":
            entry = {
                "custom_id": key,
                "response": {
                    "status_code": status_code,
                    "request_id": "request-1",
                    "body": {"error": {"code": code, "message": "Provider rejected request"}},
                },
                "error": None,
            }
            if status_code == 500:
                entry["response"]["body"]["usage"] = usage
        else:
            entry = generated_failure(key, kind)
            entry["response"]["request_id"] = "request-1"
            entry["response"]["body"]["usage"] = usage
        client.outputs["output"] = json.dumps(entry).encode()
        first = api.collect(handle)
        assert set(first.failures) == {0} and not first.pending
        error = first.failures[0]
        expected_attempts = [
            {
                "batch_id": "batch-1",
                "endpoint": "/v1/chat/completions",
                "pricing_tier": "batch",
                "status_code": status_code,
                "request_id": None if kind == "expired" else "request-1",
                "response_id": key
                if kind in {"refusal", "malformed", "missing_logprobs"}
                else None,
                "usage": usage if status_code in {200, 500} else None,
            }
        ]

        def check(error, source):
            assert error.status_code == status_code
            assert error.code == code
            assert error.transient is transient
            assert error.attempts == expected_attempts
            saved = encode_backend_error(error)
            assert saved["status_code"] == status_code
            assert saved["code"] == code
            assert saved["transient"] is transient
            assert saved["attempts"] == expected_attempts
            assert error.metadata["persistence"] == {
                "run_fingerprint": handle.run_fingerprint,
                "call_key": key,
                "source": source,
                "previous_attempts": 0,
            }
            if code in {"insufficient_quota", "billing_hard_limit_reached"}:
                assert isinstance(error, QuotaExhaustedError)
            typed = {
                "refusal": RefusedByPolicyError,
                "malformed": MalformedResponseError,
                "missing_logprobs": LogprobsNotReturnedError,
            }.get(kind)
            if typed is not None:
                assert isinstance(error, typed)

        check(error, "batch")
        error.attempts[0]["status_code"] = 999
        error.attempts[0]["usage"] = {"prompt_tokens": 999}
        check(api.collect(handle).failures[0], "store")

    def forbidden(*args, **kwargs):
        pytest.fail("Failure replay contacted the provider")

    client.files.create = client.files.content = forbidden
    client.batches.create = client.batches.retrieve = client.batches.list = forbidden
    with CallStore.open(tmp_path, run_descriptor={}, validation=validation) as store:
        replay = OpenAIBatchBackend(backend(client), store).collect(handle)
        check(replay.failures[0], "store")
        assert len(client.creates) == len(client.uploads) == 1


def test_unknown_create_does_not_authorize_a_second_submission(tmp_path):
    client = Client()

    def uncertain(**kwargs):
        client.creates.append(kwargs)
        raise TimeoutError("lost create response")

    client.batches.create = uncertain
    messages = [[ChatMessage("user", "a")]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        with pytest.raises(UnknownOutcomeError) as caught:
            api.submit_chat_batch(messages, SamplingParams(), repeat_ids=["a"])
        manifest = store._journals[caught.value.key]
        assert caught.value.metadata["persistence_calls"] == [
            {
                "run_fingerprint": store.fingerprint,
                "call_key": key,
                "source": "batch",
                "previous_attempts": 0,
            }
            for key in manifest["keys"]
        ]
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


@pytest.mark.parametrize("stage", ["retrieve", "stream"])
def test_a_connection_lost_while_collecting_leaves_the_store_open_for_the_next_collection(
    tmp_path, stage
):
    """A caller may collect again in the same process after a connection drops,
    whether a Batch was being retrieved or its output was streaming. A broken
    stream is raised as the SDK's connection error, like a connection lost
    before the body, and the next collection gets every outcome without
    creating, uploading or downloading again what the first one retained."""
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=2)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", str(index))] for index in range(4)],
            SamplingParams(),
            repeat_ids=list(range(4)),
        )
        outputs = {
            identifier: b"\n".join(
                json.dumps(completion(handle.keys[index], str(index))).encode() for index in indices
            )
            for identifier, indices in (("batch-1", [0, 1]), ("batch-2", [2, 3]))
        }
        lost = ConnectionResetError("connection reset by peer")
        retrieved, streamed = [], []

        def retrieve(identifier):
            retrieved.append(identifier)
            if stage == "retrieve" and retrieved == ["batch-1", "batch-2"]:
                raise APIConnectionError(
                    request=httpx.Request("GET", "https://offline.invalid/batches")
                )
            return SimpleNamespace(
                id=identifier, status="completed", output_file_id=identifier, error_file_id=None
            )

        class Stream:
            def __init__(self, identifier):
                self.identifier = identifier
                streamed.append(identifier)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_bytes(self):
                payload = outputs[self.identifier]
                yield payload[:5]
                if stage == "stream" and streamed == ["batch-1", "batch-2"]:
                    raise lost
                yield payload[5:]

        client.batches.retrieve = retrieve
        client.files.with_streaming_response = SimpleNamespace(content=Stream)
        client.files.content = lambda id: pytest.fail(
            "Batch output was buffered instead of streamed"
        )
        with pytest.raises(APIConnectionError) as caught:
            api.collect(handle)
        if stage == "stream":
            assert caught.value.__cause__ is lost
        partial = caught.value.partial_collection
        assert [result.text if result else None for result in partial.results] == [
            "0",
            "1",
            None,
            None,
        ]
        collection = api.collect(handle)
        assert [result.text for result in collection.results] == ["0", "1", "2", "3"]
        assert not collection.pending and not collection.failures
        assert len(client.creates) == 2 and len(client.uploads) == 2
        assert retrieved.count("batch-1") == 1 and streamed.count("batch-1") == 1


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


def test_partial_expiry_resubmits_only_a_request_explicitly_reported_unexecuted(tmp_path):
    client = Client()
    batch = [[ChatMessage("user", "a")], [ChatMessage("user", "b")]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=[1, 2])
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()
        client.outputs["errors"] = json.dumps(
            {
                "custom_id": handle.keys[1],
                "response": None,
                "error": {
                    "code": "batch_expired",
                    "message": "This request could not be executed before the completion window expired.",
                },
            }
        ).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="expired", output_file_id="output", error_file_id="errors"
        )
        first = api.collect(handle)
        assert first.results[0].text == "a" and set(first.failures) == {1}
        assert first.failures[1].metadata["persistence"] == {
            "run_fingerprint": store.fingerprint,
            "call_key": handle.keys[1],
            "source": "batch",
            "previous_attempts": 0,
        }
        assert len(client.creates) == 1

    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        api = OpenAIBatchBackend(backend(client), store)
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
        assert [
            result.metadata["persistence"]["previous_attempts"] for result in final.results
        ] == [0, 1]


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize(
    "kind,error_type",
    [
        ("refusal", RefusedByPolicyError),
        ("malformed", MalformedResponseError),
        ("missing_logprobs", LogprobsNotReturnedError),
    ],
)
def test_generated_failures_replay_after_restart_and_regrouping(
    tmp_path, monkeypatch, validation, kind, error_type
):
    client = Client()
    batch = [[ChatMessage("user", text)] for text in ["good", "bad", "new"]]
    params = SamplingParams(logprobs=kind == "missing_logprobs")
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(batch[:2], params, repeat_ids=["good", "bad"])
        client.outputs["output"] = b"\n".join(
            json.dumps(entry).encode()
            for entry in [
                completion(original.keys[0], "good", logprobs=params.logprobs),
                generated_failure(original.keys[1], kind),
            ]
        )
        first = api.collect(original)
        assert first.results[0].text == "good" and not first.pending
        assert isinstance(first.failures[1], error_type)
        assert first.failures[1].metadata["persistence"]["source"] == "batch"
        saved_receipt = store._records[original.keys[1]]["receipt"]
        assert saved_receipt["response"]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 1,
        }
        error_message = str(first.failures[1])

    def forbidden(*args, **kwargs):
        pytest.fail("A stored generated outcome contacted the provider")

    with CallStore.open(tmp_path, run_descriptor={}, validation=validation) as store:
        api = OpenAIBatchBackend(backend(client), store)
        with monkeypatch.context() as blocked:
            blocked.setattr(client.files, "create", forbidden)
            blocked.setattr(client.files, "content", forbidden)
            blocked.setattr(client.batches, "create", forbidden)
            blocked.setattr(client.batches, "retrieve", forbidden)
            blocked.setattr(client.batches, "list", forbidden)
            rebuilt = api.submit_chat_batch(
                [batch[1], batch[0]], params, repeat_ids=["bad", "good"]
            )
            replay = api.collect(rebuilt)
            assert rebuilt.keys == (original.keys[1], original.keys[0])
            assert replay.results[1].text == "good" and not replay.pending
            assert isinstance(replay.failures[0], error_type)
            assert str(replay.failures[0]) == error_message
            assert replay.failures[0].metadata["persistence"] == {
                "run_fingerprint": store.fingerprint,
                "call_key": original.keys[1],
                "source": "store",
                "previous_attempts": 0,
            }
            with pytest.raises(PartialBatchError) as caught:
                api.generate_chat_batch([batch[1]], params, repeat_ids=["bad"])
            assert isinstance(caught.value.failures[0], error_type)
            assert caught.value.results[0] is caught.value.failures[0]
            assert caught.value.failures[0].metadata["persistence"]["source"] == "store"

        regrouped = api.submit_chat_batch(
            [batch[1], batch[2], batch[0]], params, repeat_ids=["bad", "new", "good"]
        )
        assert len(client.creates) == len(client.uploads) == 2
        rows = [json.loads(line) for line in client.uploads[-1].splitlines()]
        assert [row["custom_id"] for row in rows] == [regrouped.keys[1]]
        client.outputs["new-output"] = json.dumps(
            completion(regrouped.keys[1], "new", logprobs=params.logprobs)
        ).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="completed", output_file_id="new-output", error_file_id=None
        )
        collected = api.collect(regrouped)
        assert [result.text if result else None for result in collected.results] == [
            None,
            "new",
            "good",
        ]
        assert set(collected.failures) == {0} and not collected.pending
        assert collected.failures[0].metadata["persistence"]["source"] == "store"
        assert collected.results[1].metadata["persistence"]["source"] == "batch"
        record = store._records[original.keys[1]]
        assert record["receipt"] == saved_receipt and record["attempts"] == []


def test_per_key_acknowledgement_survives_restart_and_resubmits_only_that_failure(tmp_path):
    client = Client()
    batch = [[ChatMessage("user", text)] for text in ["good", "retry", "keep"]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=[0, 1, 2])
        client.outputs["output"] = b"\n".join(
            json.dumps(entry).encode()
            for entry in [
                completion(original.keys[0], "good"),
                generated_failure(original.keys[1], "refusal"),
                generated_failure(original.keys[2], "refusal"),
            ]
        )
        assert set(api.collect(original).failures) == {1, 2}
        receipt = store._records[original.keys[1]]["receipt"]

    reason = "Another draw is intended for this one refusal"
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        store.acknowledge_resubmission(
            original.keys[1], reason=reason, accept_duplicate_expenditure=True
        )
        assert len(client.creates) == len(client.uploads) == 1

    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        api = OpenAIBatchBackend(backend(client), store)
        retried = api.submit_chat_batch(
            [batch[2], batch[0], batch[1]], SamplingParams(), repeat_ids=[2, 0, 1]
        )
        assert retried.keys == (original.keys[2], original.keys[0], original.keys[1])
        assert len(client.creates) == len(client.uploads) == 2
        assert [json.loads(row)["custom_id"] for row in client.uploads[-1].splitlines()] == [
            original.keys[1]
        ]
        client.outputs["retry-output"] = json.dumps(completion(original.keys[1], "retry")).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status="completed", output_file_id="retry-output", error_file_id=None
        )
        final = api.collect(retried)
        assert [result.text if result else None for result in final.results] == [
            None,
            "good",
            "retry",
        ]
        assert set(final.failures) == {0} and not final.pending
        assert final.failures[0].metadata["persistence"]["source"] == "store"
        assert final.failures[0].metadata["persistence"]["previous_attempts"] == 0
        assert final.results[1].metadata["persistence"]["previous_attempts"] == 0
        assert final.results[2].metadata["persistence"] == {
            "run_fingerprint": store.fingerprint,
            "call_key": original.keys[1],
            "source": "batch",
            "previous_attempts": 1,
        }
        record = store._records[original.keys[1]]
        assert len(record["attempts"]) == 1
        assert record["attempts"][0]["receipt"] == receipt
        assert record["attempts"][0]["submission"] == original.submissions[0]
        assert record["recovery_decisions"] == [
            {"reason": reason, "duplicate_expenditure_accepted": True}
        ]
        assert store._records[original.keys[2]]["attempts"] == []


@pytest.mark.parametrize("terminal_status", ["completed", "expired", "cancelled", "failed"])
def test_missing_terminal_output_replays_failure_without_authorizing_another_draw(
    tmp_path, terminal_status
):
    client = Client()
    batch = [[ChatMessage("user", text)] for text in ["good", "missing"]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=[0, 1])
        client.outputs["output"] = json.dumps(completion(original.keys[0], "good")).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id, status=terminal_status, output_file_id="output", error_file_id=None
        )
        first = api.collect(original)
        assert first.results[0].text == "good" and set(first.failures) == {1}
        saved = encode_backend_error(first.failures[1])
        assert saved["status_code"] is None and saved["transient"] is False
        assert saved["attempts"] is None
        assert getattr(first.failures[1], "code", None) is None
        message = str(first.failures[1])

    def forbidden(*args, **kwargs):
        pytest.fail("An absent output was treated as permission to execute again")

    client.files.create = client.files.content = forbidden
    client.batches.create = client.batches.retrieve = client.batches.list = forbidden
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        api = OpenAIBatchBackend(backend(client), store)
        rebuilt = api.submit_chat_batch([batch[1], batch[0]], SamplingParams(), repeat_ids=[1, 0])
        replay = api.collect(rebuilt)
        assert replay.results[1].text == "good" and set(replay.failures) == {0}
        assert str(replay.failures[0]) == message
        saved = encode_backend_error(replay.failures[0])
        assert saved["status_code"] is None and saved["transient"] is False
        assert saved["attempts"] is None
        assert getattr(replay.failures[0], "code", None) is None
        assert replay.failures[0].metadata["persistence"]["source"] == "store"
        assert store._records[original.keys[1]]["attempts"] == []
        assert len(client.creates) == len(client.uploads) == 1


@pytest.mark.parametrize(
    "status_code,error_code",
    [
        (400, None),
        (401, None),
        (429, None),
        (500, None),
        (None, "invalid_request"),
        (None, "rate_limit_exceeded"),
        (None, "batch_cancelled"),
        (None, "internal_error"),
        (200, "batch_expired"),
        (500, "batch_expired"),
    ],
)
def test_request_errors_are_not_automatically_assumed_unexecuted_and_retryable(
    tmp_path, status_code, error_code
):
    client = Client()
    messages = [[ChatMessage("user", "question")]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(messages, SamplingParams(), repeat_ids=[0])
        entry = completion(original.keys[0], "answer")
        entry["response"] = (
            {**entry["response"], "status_code": status_code} if status_code is not None else None
        )
        entry["error"] = {"code": error_code, "message": "Request failed"} if error_code else None
        client.outputs["output"] = json.dumps(entry).encode()
        first = api.collect(original)
        assert set(first.failures) == {0}
        message = str(first.failures[0])

    def forbidden(*args, **kwargs):
        pytest.fail("An arbitrary request error was automatically resubmitted")

    client.files.create = client.files.content = forbidden
    client.batches.create = client.batches.retrieve = client.batches.list = forbidden
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        rebuilt = api.submit_chat_batch(messages, SamplingParams(), repeat_ids=[0])
        replay = api.collect(rebuilt)
        assert set(replay.failures) == {0} and str(replay.failures[0]) == message
        assert replay.failures[0].metadata["persistence"] == {
            "run_fingerprint": store.fingerprint,
            "call_key": original.keys[0],
            "source": "store",
            "previous_attempts": 0,
        }
        assert store._records[original.keys[0]]["attempts"] == []


def test_pending_requests_rejoin_their_submission_after_restart_with_different_windows(tmp_path):
    client = Client()
    batch = [[ChatMessage("user", text)] for text in ["a", "b", "c"]]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        original = OpenAIBatchBackend(backend(client), store).submit_chat_batch(
            batch[:2], SamplingParams(), repeat_ids=["a", "b"]
        )

    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        api = OpenAIBatchBackend(backend(client), store)
        rebuilt = api.submit_chat_batch(batch[1:], SamplingParams(), repeat_ids=["b", "c"])
        assert rebuilt.keys[0] == original.keys[1]
        assert len(client.creates) == len(client.uploads) == 2
        assert [json.loads(row)["custom_id"] for row in client.uploads[-1].splitlines()] == [
            rebuilt.keys[1]
        ]
        client.outputs["first"] = b"\n".join(
            json.dumps(completion(key, text)).encode()
            for key, text in zip(original.keys, ["a", "b"], strict=True)
        )
        client.outputs["second"] = json.dumps(completion(rebuilt.keys[1], "c")).encode()
        client.batches.retrieve = lambda id: SimpleNamespace(
            id=id,
            status="completed",
            output_file_id="first" if id == "batch-1" else "second",
            error_file_id=None,
        )
        results = api.collect(rebuilt)
        assert [result.text for result in results.results] == ["b", "c"]
        assert not results.pending and not results.failures
        client.batches.retrieve = lambda id: pytest.fail("A completed sibling was fetched again")
        sibling = api.submit_chat_batch(batch[:1], SamplingParams(), repeat_ids=["a"])
        assert api.collect(sibling).results[0].text == "a"
        assert len(client.creates) == 2


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


def test_lazy_open_defers_batch_payload_validation_until_collection(tmp_path, monkeypatch):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()
        assert api.collect(handle).results[0].text == "a"
    load, blob = CallStore._load, CallStore.blob_path
    reads, blobs = [], []

    def tracked(store, path):
        reads.append(path)
        return load(store, path)

    def tracked_blob(store, entry):
        blobs.append(entry)
        return blob(store, entry)

    monkeypatch.setattr(CallStore, "_load", tracked)
    monkeypatch.setattr(CallStore, "blob_path", tracked_blob)
    client.batches.retrieve = lambda *args: pytest.fail("Replay contacted provider")
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        assert reads == [] and blobs == []
        result = OpenAIBatchBackend(backend(client), store).collect(handle).results[0]
        assert result.text == "a" and len(reads) == 1
        assert result.metadata["persistence"]["source"] == "store"
        assert len(client.creates) == 1


def test_durable_batch_receipt_reparses_without_contacting_provider(tmp_path):
    client = Client()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        source = backend(client)
        api = OpenAIBatchBackend(source, store)
        handle = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        client.outputs["output"] = json.dumps(completion(handle.keys[0], "a")).encode()
        source.reparse_receipt = lambda *args: (_ for _ in ()).throw(
            ValueError("interrupted parser")
        )
        failed = api.collect(handle)
        assert set(failed.failures) == {0}
        assert failed.failures[0].metadata["persistence"]["source"] == "batch"
        failed = api.collect(handle)
        assert failed.failures[0].metadata["persistence"]["source"] == "receipt"
        assert failed.failures[0].metadata["persistence"]["previous_attempts"] == 0

    def forbidden(*args, **kwargs):
        pytest.fail("Durable receipt recovery contacted the provider")

    client.files.create = client.files.content = forbidden
    client.batches.create = client.batches.retrieve = client.batches.list = forbidden
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        api = OpenAIBatchBackend(backend(client), store)
        rebuilt = api.submit_chat_batch(
            [[ChatMessage("user", "a")]], SamplingParams(), repeat_ids=[1]
        )
        assert rebuilt.keys == handle.keys
        result = api.collect(rebuilt).results[0]
        assert result.text == "a"
        assert result.metadata["persistence"] == {
            "run_fingerprint": store.fingerprint,
            "call_key": handle.keys[0],
            "source": "receipt",
            "previous_attempts": 0,
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


def _collect_subset_without_provider(root, serialized, connection):
    client = Client()

    def forbidden(*args, **kwargs):
        raise AssertionError("A later owner contacted the provider instead of using retained files")

    client.files.create = client.files.content = forbidden
    client.batches.create = client.batches.retrieve = client.batches.list = forbidden
    with CallStore.open(root, run_descriptor={}, validation="lazy") as store:
        collection = OpenAIBatchBackend(backend(client), store).collect(
            BatchHandle.from_dict(serialized)
        )
        assert not collection.failures and not collection.pending
        connection.send(
            {
                "texts": [item.text for item in collection.results],
                "sources": [item.metadata["persistence"]["source"] for item in collection.results],
                "previous_attempts": [
                    item.metadata["persistence"]["previous_attempts"] for item in collection.results
                ],
            }
        )
    connection.close()


def test_subset_collection_reports_new_outcomes_to_a_later_process(tmp_path):
    import multiprocessing

    client = Client()
    batch = [[ChatMessage("user", str(index))] for index in range(5)]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        api = OpenAIBatchBackend(backend(client), store)
        original = api.submit_chat_batch(batch, SamplingParams(), repeat_ids=list(range(5)))
        subset = api.submit_chat_batch(batch[:2], SamplingParams(), repeat_ids=[0, 1])
        client.outputs["output"] = b"\n".join(
            json.dumps(completion(key, str(index))).encode()
            for index, key in enumerate(original.keys)
        )
        assert [item.metadata["persistence"]["source"] for item in api.collect(subset).results] == [
            "batch",
            "batch",
        ]
    remaining = BatchHandle(original.run_fingerprint, original.submissions, original.keys[2:])
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_collect_subset_without_provider, args=(str(tmp_path), remaining.to_dict(), child)
    )
    process.start()
    child.close()
    try:
        assert parent.poll(15), "Later owner did not return retained outcomes"
        assert parent.recv() == {
            "texts": ["2", "3", "4"],
            "sources": ["batch", "batch", "batch"],
            "previous_attempts": [0, 0, 0],
        }
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
    client.batches.retrieve = lambda *args: pytest.fail("Returned outcomes contacted provider")
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        final = OpenAIBatchBackend(backend(client), store).collect(original)
        assert [item.metadata["persistence"]["source"] for item in final.results] == ["store"] * 5


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
