"""Cooperative Batch stops preserve durable work and definite non-dispatch."""

import inspect
import json
import selectors
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from llenvs.inference.backends import openai_batch
from llenvs.inference.backends.openai_batch import BatchHandle, OpenAIBatchBackend
from llenvs.inference.persistence import CallStore, UnknownOutcomeError
from llenvs.inference.protocol import ChatMessage, SamplingParams
from tests.inference.test_openai_batch import Client, backend, completion, generated_failure

RUN = {"experiment": "batch-stop"}


def requests(count):
    return [[ChatMessage("user", str(index))] for index in range(count)]


def expect_stopped(function, *args, **kwargs):
    with pytest.raises(RuntimeError) as caught:
        function(*args, **kwargs)
    assert type(caught.value) is openai_batch.BatchStopped
    return caught.value


@pytest.fixture
def stop_contract():
    def require():
        assert "stop" in inspect.signature(OpenAIBatchBackend.submit_chat_batch).parameters
        assert "stop" in inspect.signature(OpenAIBatchBackend.collect).parameters

    return require


def test_batch_stopped_is_exported_without_being_a_persisted_slot_failure():
    from llenvs.inference.backends import BatchStopped

    assert BatchStopped is openai_batch.BatchStopped
    assert issubclass(BatchStopped, RuntimeError)


@pytest.mark.parametrize("operation", ["submit", "collect"])
def test_preset_stop_makes_no_provider_request(stores, stop_contract, operation):
    stop_contract()
    client, stop = Client(), threading.Event()
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = None
        if operation == "collect":
            handle = api.submit_chat_batch(requests(2), SamplingParams(), repeat_ids=[0, 1])
        stop.set()

        def forbidden(*args, **kwargs):
            pytest.fail("A pre-stopped call accessed the provider")

        client.files.create = client.files.content = forbidden
        client.batches.create = client.batches.retrieve = client.batches.list = forbidden
        if operation == "submit":
            error = expect_stopped(
                api.submit_chat_batch, requests(2), SamplingParams(), repeat_ids=[0, 1], stop=stop
            )
            assert error.handle is None and not error.submission_states
            assert not store._journals and not store._inflight and not store._pins
        else:
            error = expect_stopped(api.collect, handle, stop=stop)
            assert error.handle == handle
            assert error.partial_collection.results == [None, None]
            assert not error.partial_collection.failures
            assert error.partial_collection.pending == list(handle.keys)
            assert set(error.submission_states.values()) == {"pending"}
        store._check()


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("boundary", ["upload", "create"])
@pytest.mark.parametrize("resume", ["submit", "collect"])
def test_stop_finishes_only_the_current_chunk_and_resumes_each_intent_once(
    stores, stop_contract, validation, boundary, resume
):
    stop_contract()
    client, stop = Client(), threading.Event()
    entered, release = threading.Event(), threading.Event()
    client.batches.retrieve = lambda identifier: SimpleNamespace(
        id=identifier, status="in_progress", output_file_id=None, error_file_id=None
    )
    original = getattr(client.files if boundary == "upload" else client.batches, "create")

    def held(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    setattr(client.files if boundary == "upload" else client.batches, "create", held)
    with stores() as store, ThreadPoolExecutor(max_workers=1) as worker:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=2)
        submitted = worker.submit(
            expect_stopped,
            api.submit_chat_batch,
            requests(6),
            SamplingParams(),
            repeat_ids=list(range(6)),
            stop=stop,
        )
        try:
            assert entered.wait(3)
            stop.set()
            release.set()
            error = submitted.result(timeout=3)
        finally:
            release.set()
        handle = error.handle
        assert len(handle.keys) == 6 and len(handle.submissions) == 3
        assert len(client.uploads) == 1
        assert len(client.creates) == (boundary == "create")
        assert [error.submission_states[i] for i in handle.submissions] == [
            "intent" if boundary == "upload" else "pending",
            "intent",
            "intent",
        ]
        assert store._journals[handle.submissions[0]]["input_file_id"] == "file-1"
        assert not store._inflight and not store._pins
    stop.clear()
    setattr(client.files if boundary == "upload" else client.batches, "create", original)
    with stores(validation=validation) as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=2)
        if resume == "submit":
            rebuilt = api.submit_chat_batch(
                requests(6), SamplingParams(), repeat_ids=list(range(6)), stop=stop
            )
            assert rebuilt == handle
        else:
            collection = api.collect(handle, stop=stop)
            assert collection.pending == list(handle.keys)
        assert len(client.uploads) == len(client.creates) == 3
        assert all(store._journals[i]["state"] == "pending" for i in handle.submissions)
        assert all(not store._records[key]["attempts"] for key in handle.keys)
        api.collect(handle, stop=stop)
        assert len(client.uploads) == len(client.creates) == 3


def test_stop_during_preparation_retains_only_prepared_work_and_releases_reservations(
    stores, stop_contract, monkeypatch
):
    stop_contract()
    client, stop = Client(), threading.Event()
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=2)
        prepare = api._prepare

        def prepare_then_stop(*args):
            identifier = prepare(*args)
            stop.set()
            return identifier

        with monkeypatch.context() as patch:
            patch.setattr(api, "_prepare", prepare_then_stop)
            error = expect_stopped(
                api.submit_chat_batch,
                requests(8),
                SamplingParams(),
                repeat_ids=list(range(8)),
                stop=stop,
            )
        assert len(error.handle.submissions) == 1 and len(error.handle.keys) == 2
        assert len(store._journals) == 1 and not client.uploads and not client.creates
        assert not store._inflight and not store._pins
        stop.clear()
        rebuilt = api.submit_chat_batch(
            requests(8), SamplingParams(), repeat_ids=list(range(8)), stop=stop
        )
        assert rebuilt.keys[:2] == error.handle.keys
        assert rebuilt.submissions[0] == error.handle.submissions[0]
        assert len(client.uploads) == len(client.creates) == 4


def test_stop_observed_after_the_creating_marker_restores_an_undispatched_intent(
    stores, stop_contract, monkeypatch
):
    stop_contract()
    client, stop = Client(), threading.Event()
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        write = store.write_journal

        def write_then_stop(manifest):
            write(manifest)
            if manifest["state"] == "creating":
                assert all(store._records[key]["state"] == "pending" for key in manifest["keys"])
                stop.set()

        with monkeypatch.context() as patch:
            patch.setattr(store, "write_journal", write_then_stop)
            error = expect_stopped(
                api.submit_chat_batch, requests(2), SamplingParams(), repeat_ids=[0, 1], stop=stop
            )
        assert not client.creates and len(client.uploads) == 1
        assert set(error.submission_states.values()) == {"intent"}
        assert all(store._journals[i]["state"] == "intent" for i in error.handle.submissions)
        stop.clear()
        api.submit_chat_batch(requests(2), SamplingParams(), repeat_ids=[0, 1], stop=stop)
        assert len(client.uploads) == len(client.creates) == 1


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("outcome", ["success", "refusal", "unfinished"])
def test_stopped_collection_carries_durable_progress_and_resumes_from_retained_files(
    stores, stop_contract, monkeypatch, validation, outcome
):
    stop_contract()
    client, stop, fired = Client(), threading.Event(), threading.Event()
    updates, downloads, retrieves = Counter(), [], []
    update = CallStore.update

    def record_and_stop(self, key, state, **values):
        answer = update(self, key, state, **values)
        if state in {"completed", "failed"}:
            updates[key] += 1
            if not fired.is_set():
                fired.set()
                stop.set()
        return answer

    monkeypatch.setattr(CallStore, "update", record_and_stop)
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store, max_file_requests=260)
        handle = api.submit_chat_batch(requests(520), SamplingParams(), repeat_ids=list(range(520)))
        for identifier in handle.submissions:
            manifest = store._journals[identifier]
            client.outputs[manifest["batch_id"]] = (
                b"\n".join(
                    json.dumps(
                        completion(key, key)
                        if outcome == "success"
                        else generated_failure(key, "refusal")
                    ).encode()
                    for key in manifest["keys"]
                )
                if outcome != "unfinished"
                else b""
            )

        def retrieve(identifier):
            retrieves.append(identifier)
            assert not stop.is_set(), "A later submission was retrieved after stop"
            return SimpleNamespace(
                id=identifier, status="completed", output_file_id=identifier, error_file_id=None
            )

        def download(identifier):
            downloads.append(identifier)
            assert not stop.is_set(), "A later file was downloaded after stop"
            return SimpleNamespace(content=client.outputs[identifier])

        client.batches.retrieve, client.files.content = retrieve, download
        error = expect_stopped(api.collect, handle, stop=stop)
        partial = error.partial_collection
        assert error.handle == handle and len(partial.results) == 520
        obtained = {
            index for index, result in enumerate(partial.results) if result is not None
        } | set(partial.failures)
        assert obtained and len(obtained) < 260
        assert partial.pending == [
            key for index, key in enumerate(handle.keys) if index not in obtained
        ]
        for index in obtained:
            value = partial.results[index] or partial.failures[index]
            assert value.metadata["persistence"] == {
                "run_fingerprint": handle.run_fingerprint,
                "call_key": handle.keys[index],
                "source": "batch",
                "previous_attempts": 0,
            }
        assert downloads == retrieves == ["batch-1"]
        store._check()
    stop.clear()
    with stores(validation=validation) as store:
        result = OpenAIBatchBackend(backend(client), store).collect(handle, stop=stop)
        assert not result.pending
        for index, key in enumerate(handle.keys):
            value = result.results[index] or result.failures[index]
            assert value.metadata["persistence"]["source"] == (
                "store" if index in obtained else "batch"
            )
            assert value.metadata["persistence"]["previous_attempts"] == 0
            assert updates[key] == 1, "A committed output was persisted twice"
        assert downloads == retrieves == ["batch-1", "batch-2"]
        assert len(client.creates) == len(client.uploads) == 2


def test_stop_wakes_a_waiting_collection_without_another_poll(stores, stop_contract, monkeypatch):
    stop_contract()
    client, stop, retrieved = Client(), threading.Event(), threading.Event()
    polls = []

    def retrieve(identifier):
        polls.append(identifier)
        retrieved.set()
        return SimpleNamespace(
            id=identifier, status="in_progress", output_file_id=None, error_file_id=None
        )

    client.batches.retrieve = retrieve

    def noninterruptible_sleep(seconds):
        raise AssertionError("Polling used a wait that cannot observe the caller's stop")

    monkeypatch.setattr(openai_batch, "time", SimpleNamespace(sleep=noninterruptible_sleep))
    with stores() as store, ThreadPoolExecutor(max_workers=1) as worker:
        api = OpenAIBatchBackend(backend(client), store, poll_interval=30)
        handle = api.submit_chat_batch(requests(1), SamplingParams(), repeat_ids=[0])
        collecting = worker.submit(expect_stopped, api.collect, handle, wait=True, stop=stop)
        try:
            assert retrieved.wait(3)
            started = time.monotonic()
            stop.set()
            error = collecting.result(timeout=1)
            assert time.monotonic() - started < 1
            assert error.partial_collection.pending == list(handle.keys) and len(polls) == 1
        finally:
            stop.set()


def test_stop_between_reconciliation_pages_keeps_creation_unknown(stores, stop_contract):
    stop_contract()
    client, stop = Client(), threading.Event()

    def uncertain(**kwargs):
        client.creates.append(kwargs)
        raise TimeoutError("Synthetic lost create acknowledgement")

    client.batches.create = uncertain
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        with pytest.raises(UnknownOutcomeError):
            api.submit_chat_batch(requests(1), SamplingParams(), repeat_ids=[0])
        handle = api.submit_chat_batch(requests(1), SamplingParams(), repeat_ids=[0])
        pages = []

        def page(**kwargs):
            pages.append(kwargs)
            stop.set()
            return SimpleNamespace(
                data=[SimpleNamespace(id="unrelated", metadata={})], has_more=True
            )

        client.batches.list = page
        error = expect_stopped(api.collect, handle, stop=stop)
        assert len(pages) == 1 and error.partial_collection.pending == list(handle.keys)
        stop.clear()
        client.batches.list = lambda **kwargs: SimpleNamespace(data=[], has_more=False)
        result = api.collect(handle, stop=stop)
        assert result.pending == list(handle.keys)
        assert set(result.statuses.values()) == {"submission_unknown"} and len(client.creates) == 1


@pytest.mark.parametrize("boundary", ["pending", "creating"])
def test_native_closed_client_before_create_leaves_a_resumable_intent(
    stores, monkeypatch, boundary
):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        assert request.url.path.endswith("/files"), "A closed client sent a create"
        return httpx.Response(
            200,
            json={
                "id": "native-file",
                "object": "file",
                "bytes": 1,
                "created_at": 0,
                "filename": "input",
                "purpose": "batch",
            },
        )

    client = OpenAI(
        api_key="synthetic-offline-key",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    with client, stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        update_many, write = store.update_many, store.write_journal

        def update_then_close(values):
            values = list(values)
            answer = update_many(values)
            if (
                boundary == "pending"
                and values
                and all(state == "pending" for _, state, _ in values)
            ):
                client.close()
            return answer

        def write_then_close(manifest):
            answer = write(manifest)
            if boundary == "creating" and manifest["state"] == "creating":
                client.close()
            return answer

        with monkeypatch.context() as patch:
            patch.setattr(store, "update_many", update_then_close)
            patch.setattr(store, "write_journal", write_then_close)
            with pytest.raises(Exception) as caught:
                api.submit_chat_batch(requests(2), SamplingParams(), repeat_ids=[0, 1])
        assert not isinstance(caught.value, UnknownOutcomeError)
        assert paths == ["/v1/files"]
        [manifest] = store._journals.values()
        assert manifest["state"] == "intent" and manifest["input_file_id"] == "native-file"
        replacement = Client()
        replacement.base_url = str(client.base_url)
        resumed = OpenAIBatchBackend(backend(replacement), store)
        assert resumed.descriptor()["identity"] == api.descriptor()["identity"]
        rebuilt = resumed.submit_chat_batch(requests(2), SamplingParams(), repeat_ids=[0, 1])
        assert rebuilt.keys == tuple(manifest["keys"])
        assert not replacement.uploads and len(replacement.creates) == 1
    client.close()


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_legacy_creating_with_partially_pending_records_remains_conservative(
    stores, monkeypatch, validation
):
    client = Client()
    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)

        def legacy_boundary(manifest, **kwargs):
            manifest.update(state="creating", input_file_id="legacy-file")
            store.write_journal(manifest)
            store.update(manifest["keys"][0], "pending", submission=manifest["id"])
            raise InterruptedError("Synthetic legacy worker stopped before dispatch")

        with monkeypatch.context() as patch:
            patch.setattr(api, "_submit", legacy_boundary)
            with pytest.raises(InterruptedError):
                api.submit_chat_batch(requests(2), SamplingParams(), repeat_ids=[0, 1])
        handle = BatchHandle(
            store.fingerprint,
            tuple(store._journals),
            tuple(next(iter(store._journals.values()))["keys"]),
        )
    with stores(validation=validation) as store:
        result = OpenAIBatchBackend(backend(client), store).collect(handle)
        assert set(result.statuses.values()) == {"submission_unknown"}
        assert result.pending == list(handle.keys) and not client.creates


@pytest.mark.parametrize("operation", ["upload", "create", "download"])
def test_stop_does_not_mask_an_operation_error_or_retry_an_uncertain_create(
    stores, stop_contract, operation
):
    stop_contract()
    client, stop = Client(), threading.Event()
    error = TimeoutError("Synthetic operation lost its response during stop")

    def fail(**kwargs):
        stop.set()
        if operation == "create":
            client.creates.append(kwargs)
        raise error

    with stores() as store:
        api = OpenAIBatchBackend(backend(client), store)
        if operation == "download":
            handle = api.submit_chat_batch(requests(1), SamplingParams(), repeat_ids=[0])
            client.files.content = lambda identifier: fail()
            with pytest.raises(TimeoutError) as caught:
                api.collect(handle, stop=stop)
            assert caught.value is error and not hasattr(error, "partial_collection")
        else:
            setattr(client.files if operation == "upload" else client.batches, "create", fail)
            with pytest.raises(
                UnknownOutcomeError if operation == "create" else TimeoutError
            ) as caught:
                api.submit_chat_batch(requests(1), SamplingParams(), repeat_ids=[0], stop=stop)
            [manifest] = store._journals.values()
            if operation == "create":
                assert caught.value.__cause__ is error and manifest["state"] == "creating"
                stop.clear()
                handle = api.submit_chat_batch(
                    requests(1), SamplingParams(), repeat_ids=[0], stop=stop
                )
                collection = api.collect(handle, stop=stop)
                assert collection.pending == list(handle.keys)
                assert set(collection.statuses.values()) == {"submission_unknown"}
                assert len(client.creates) == 1
            else:
                assert caught.value is error and manifest["state"] == "intent"
                assert not client.creates and manifest["input_file_id"] is None
        store._check()


@pytest.mark.parametrize("boundary", ["upload", "pending", "collection"])
def test_stop_and_owner_release_finish_under_ten_seconds_with_512_calls_and_100ms_io(
    gcs_root, cloud, stop_contract, monkeypatch, boundary, record_property
):
    stop_contract()
    from tests.inference.persistence.test_throughput import DelayedIO

    io = DelayedIO(cloud, monkeypatch)
    stop, entered, release = threading.Event(), threading.Event(), threading.Event()
    client, calls = Client(), []
    journal = gcs_root()
    store = CallStore.open(journal, run_descriptor=RUN, validation="lazy")
    io.delay = 0.1

    def trace(label, operation):
        def invoke(*args, **kwargs):
            calls.append((label, stop.is_set()))
            if boundary == "upload" and label == "upload":
                entered.set()
                assert release.wait(10)
            return operation(*args, **kwargs)

        return invoke

    client.files.create = trace("upload", client.files.create)
    client.files.content = trace("download", client.files.content)
    client.batches.create = trace("create", client.batches.create)
    client.batches.retrieve = trace("retrieve", client.batches.retrieve)
    client.batches.list = trace("list", client.batches.list)
    api = OpenAIBatchBackend(backend(client), store)

    def hold(kind, name, record):
        state = "completed" if boundary == "collection" else "pending"
        if kind == "put" and record and record["state"] == state:
            entered.set()
            assert release.wait(10)

    try:
        if boundary == "collection":
            handle = api.submit_chat_batch(
                requests(512), SamplingParams(), repeat_ids=list(range(512))
            )
            client.outputs["output"] = b"\n".join(
                json.dumps(completion(key, key)).encode() for key in handle.keys
            )
            io.before = hold
            operation = partial(api.collect, handle, stop=stop)
        else:
            if boundary == "pending":
                io.before = hold
            operation = partial(
                api.submit_chat_batch,
                requests(512),
                SamplingParams(),
                repeat_ids=list(range(512)),
                stop=stop,
            )
        with ThreadPoolExecutor(max_workers=1) as worker:
            running = worker.submit(expect_stopped, operation)
            try:
                assert entered.wait(10)
                if boundary == "pending":
                    assert set(manifest["state"] for manifest in store._journals.values()) == {
                        "intent"
                    }
                started = time.monotonic()
                stop.set()
                release.set()
                error = running.result(timeout=8)
                store.close()
                elapsed = time.monotonic() - started
                record_property("stop_to_close_seconds", elapsed)
                assert elapsed < 10, f"Stop-to-close took {elapsed:.3f}s"
                assert journal.current_owner(run_descriptor=RUN) is None
                assert not store._storage._objects.pinned
                assert all(not after_stop for _, after_stop in calls), calls
                assert error.handle is not None
                if boundary == "collection":
                    collection = error.partial_collection
                    assert any(result is not None for result in collection.results)
                    assert collection.pending
            finally:
                release.set()
    finally:
        release.set()
        io.before = None
        store.close()


class RetainedProvider(Client):
    """A synthetic provider's state outlives a killed journal worker."""

    def __init__(self, path, *, match=True):
        super().__init__()
        self.path, self.match = Path(path), match
        self.batches.list = self.list_batches

    def state(self):
        return (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"uploads": [], "creates": []}
        )

    def upload(self, **kwargs):
        state = self.state()
        identifier = f"file-{len(state['uploads']) + 1}"
        state["uploads"].append(identifier)
        self.path.write_text(json.dumps(state))
        return SimpleNamespace(id=identifier)

    def create(self, **kwargs):
        state = self.state()
        identifier = f"batch-{len(state['creates']) + 1}"
        state["creates"].append({"id": identifier, **kwargs})
        self.path.write_text(json.dumps(state))
        return SimpleNamespace(id=identifier)

    def retrieve(self, identifier):
        return SimpleNamespace(
            id=identifier, status="in_progress", output_file_id=None, error_file_id=None
        )

    def list_batches(self, **kwargs):
        data = [SimpleNamespace(**entry) for entry in self.state()["creates"]] if self.match else []
        return SimpleNamespace(data=data, has_more=False)


def crash_worker(provider, root, provider_path, stage, address):
    if provider == "gcs":
        from multiprocessing.managers import BaseManager

        from llenvs.inference.persistence.storage.gcs import GCSJournal
        from tests.inference.persistence.conftest import CloudClient

        BaseManager.register("storage_cloud")
        manager = BaseManager(address=tuple(address), authkey=b"synthetic-storage-test")
        manager.connect()
        journal = GCSJournal(
            bucket="synthetic-bucket",
            prefix="batch-stop-kill",
            client=CloudClient(manager.storage_cloud()),
        )
    else:
        journal = Path(root)
    store = CallStore.open(journal, run_descriptor=RUN, validation="lazy")
    client = RetainedProvider(provider_path)
    blocked = threading.Event()

    def checkpoint():
        manifest = next(iter(store._journals.values()))
        print(
            json.dumps(
                {
                    "keys": manifest["keys"],
                    "owner": journal.current_owner(run_descriptor=RUN)
                    if provider == "gcs"
                    else None,
                }
            ),
            flush=True,
        )
        blocked.wait()

    write, update, update_many, create = (
        store.write_journal,
        store.update,
        store.update_many,
        client.batches.create,
    )

    def journal_write(manifest):
        write(manifest)
        if (
            stage == "intent"
            and manifest["state"] == "intent"
            and manifest["input_file_id"] is None
            or stage == "uploaded"
            and manifest["state"] == "intent"
            and manifest["input_file_id"] is not None
            or stage == "creating"
            and manifest["state"] == "creating"
        ):
            checkpoint()

    def pending_update(key, state, **values):
        if stage == "pending_partial" and state == "pending":
            manifest = next(iter(store._journals.values()))
            if key != manifest["keys"][0]:
                blocked.wait()
            answer = update(key, state, **values)
            checkpoint()
            return answer
        return update(key, state, **values)

    def pending_updates(values):
        values = list(values)
        answer = update_many(values)
        if stage == "pending_all" and values and all(state == "pending" for _, state, _ in values):
            checkpoint()
        return answer

    def sent_create(**kwargs):
        answer = create(**kwargs)
        if stage == "sent":
            checkpoint()
        return answer

    store.write_journal, store.update, store.update_many = (
        journal_write,
        pending_update,
        pending_updates,
    )
    client.batches.create = sent_create
    OpenAIBatchBackend(backend(client), store).submit_chat_batch(
        requests(2), SamplingParams(), repeat_ids=[0, 1]
    )
    raise AssertionError("The worker did not reach its requested crash boundary")


@pytest.mark.parametrize("provider", ["filesystem", "gcs"])
@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize(
    "stage,match",
    [
        ("intent", False),
        ("uploaded", False),
        ("pending_partial", False),
        ("pending_all", False),
        ("creating", False),
        ("sent", False),
        ("sent", True),
    ],
)
def test_killed_submission_recovers_only_when_dispatch_was_provably_not_started(
    request, tmp_path, provider, validation, stage, match
):
    address, service = (
        request.getfixturevalue("cloud_server") if provider == "gcs" else (("", 0), None)
    )
    root, provider_path = tmp_path / "calls", tmp_path / "provider.json"
    arguments = [provider, str(root), str(provider_path), stage, list(address)]
    child = "import json,sys; from tests.inference.persistence.test_batch_stop import crash_worker; crash_worker(*json.loads(sys.argv[1]))"
    process = subprocess.Popen(
        [sys.executable, "-c", child, json.dumps(arguments)],
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(15), "Worker did not reach its durable crash boundary"
        line = process.stdout.readline()
        assert line, process.stderr.read()
        checkpoint = json.loads(line)
        process.kill()
        process.communicate(timeout=5)
        if provider == "gcs":
            from llenvs.inference.persistence.storage.gcs import GCSJournal
            from tests.inference.persistence.conftest import CloudClient

            journal = GCSJournal(
                bucket="synthetic-bucket", prefix="batch-stop-kill", client=CloudClient(service)
            )
            journal.acknowledge_owner_termination(
                run_descriptor=RUN,
                expected_owner=checkpoint["owner"],
                reason="Owned synthetic worker confirmed killed",
                confirmed_terminated=True,
            )
        else:
            journal = root
        client = RetainedProvider(provider_path, match=match)
        before = client.state()
        with CallStore.open(journal, run_descriptor=RUN, validation=validation) as store:
            api = OpenAIBatchBackend(backend(client), store)
            handle = BatchHandle(
                store.fingerprint, tuple(store._journals), tuple(checkpoint["keys"])
            )
            collection = api.collect(handle)
            [manifest] = store._journals.values()
            if stage in {"creating", "sent"}:
                expected = "pending" if stage == "sent" and match else "submission_unknown"
                assert manifest["state"] == expected
                assert len(client.state()["creates"]) == len(before["creates"])
            else:
                assert manifest["state"] == "pending"
                assert len(client.state()["creates"]) == 1
            assert len(client.state()["uploads"]) == 1
            assert collection.pending == list(handle.keys)
            assert all(not store._records[key]["attempts"] for key in handle.keys)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
