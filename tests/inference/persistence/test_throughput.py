"""Concurrent durable journals; all storage and model responses are synthetic."""

import asyncio
import copy
import json
import pickle
import selectors
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llenvs.inference.backends.openai_batch import OpenAIBatchBackend
from llenvs.inference.persistence import (
    CallStore,
    CheckpointError,
    PersistentBackend,
    UnknownOutcomeError,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    SamplingParams,
)
from tests.inference.test_openai_batch import Client, backend, completion

RUN = {"experiment": "journal-throughput"}


def messages(index):
    return [ChatMessage("user", str(index))]


def record_from_object(name, data):
    if "/calls/" not in name or data is None:
        return None
    envelope = pickle.loads(data.split(b"\n", 1)[1])
    return pickle.loads(envelope["payload"])


class ConcurrentDraw:
    """Independent completions using the real synchronous observer interface."""

    capabilities = BackendCapabilities(supports_request_progress=True)

    def __init__(self, *, asynchronous=False):
        self.calls = 0
        self.lock = threading.Lock()
        self.before_receipt = None
        self.asynchronous = asynchronous

    def descriptor(self):
        return {"identity": {"model": "offline-throughput"}, "provenance": {}}

    def generate_chat_batch(self, batch, params, *, observer):
        if self.asynchronous:

            async def invoke_async(index, request):
                await observer.notify("started", index, 0)
                with self.lock:
                    self.calls += 1
                receipt = {"text": request[0].content}
                await observer.notify("received", index, receipt)
                result = self.reparse_receipt(receipt, params)
                await observer.notify("completed", index, result)
                return result

            async def run():
                return await asyncio.gather(
                    *(invoke_async(index, request) for index, request in enumerate(batch))
                )

            return asyncio.run(run())

        def invoke(item):
            index, request = item
            observer.started(index, 0)
            with self.lock:
                self.calls += 1
            if self.before_receipt:
                self.before_receipt()
            receipt = {"text": request[0].content}
            observer.received(index, receipt)
            result = self.reparse_receipt(receipt, params)
            observer.completed(index, result)
            return result

        with ThreadPoolExecutor(max_workers=min(64, len(batch))) as workers:
            return list(workers.map(invoke, enumerate(batch)))

    def reparse_receipt(self, receipt, params):
        return GenerationResult(
            text=receipt["text"],
            prompt_tokens=13,
            completion_tokens=2,
            cached_prompt_tokens=5,
            reasoning_tokens=1,
            metadata={"raw_token_ids": [7, 8], "original_usage": {"input": 13, "output": 2}},
        )


class DelayedIO:
    """Latency occurs outside the service lock, allowing genuine overlapping RPCs."""

    def __init__(self, cloud, monkeypatch):
        self.cloud = cloud
        self.lock = threading.Lock()
        self.delay = 0
        self.before = None
        self.active = {"get": 0, "put": 0}
        self.peak = {"get": 0, "put": 0}
        self.owner_reads = 0
        self.records = []
        self.finished_reads = 0
        self.listed = 0
        self.discovery_limit = None
        read, mutate, list_names = cloud.read, cloud.mutate, cloud.list_names

        def request(kind, key, call, record=None):
            with self.lock:
                self.active[kind] += 1
                self.peak[kind] = max(self.peak[kind], self.active[kind])
                if kind == "get" and "/.owners/" in key[1]:
                    self.owner_reads += 1
            try:
                if self.delay:
                    time.sleep(self.delay)
                if self.before:
                    self.before(kind, key[1], record)
                result = call()
                with self.lock:
                    if kind == "get" and "/calls/" in key[1]:
                        self.finished_reads += 1
                    if record is not None:
                        self.records.append(copy.deepcopy(record))
                return result
            finally:
                with self.lock:
                    self.active[kind] -= 1

        def write(key, data, expected):
            return request(
                "put", key, lambda: mutate(key, data, expected), record_from_object(key[1], data)
            )

        def discover(bucket, prefix):
            names = list_names(bucket, prefix)
            if not prefix.endswith("calls/") or self.discovery_limit is None:
                return names

            def bounded():
                for name in names:
                    with self.lock:
                        self.listed += 1
                        assert self.listed - self.finished_reads <= self.discovery_limit, (
                            "Full validation eagerly queued the complete store"
                        )
                    yield name

            return bounded()

        monkeypatch.setattr(
            cloud,
            "read",
            lambda key, generation=None: request("get", key, lambda: read(key, generation)),
        )
        monkeypatch.setattr(cloud, "mutate", write)
        monkeypatch.setattr(cloud, "list_names", discover)

    def reset(self):
        with self.lock:
            assert not any(self.active.values())
            self.peak = {"get": 0, "put": 0}
            self.owner_reads = self.finished_reads = self.listed = 0
            self.records.clear()


@pytest.fixture
def io(cloud, gcs_root, monkeypatch):
    return DelayedIO(cloud, monkeypatch)


@pytest.fixture
def journal_store(gcs_root, io):
    opened = []

    def open_store(*, validation="lazy", journal=None):
        store = CallStore.open(
            gcs_root() if journal is None else journal, run_descriptor=RUN, validation=validation
        )
        opened.append(store)
        return store

    yield open_store
    io.before = None
    io.delay = 0
    io.cloud.fault = None
    for store in reversed(opened):
        # A failing implementation must not leave coalesced test callers stuck.
        for future in list(store._inflight.values()):
            if not future.done():
                future.set_exception(CheckpointError("Synthetic test cleanup"))
        try:
            store.close()
        except CheckpointError:
            pass


@pytest.mark.parametrize("separate_callers", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_standard_tier_overlaps_durable_writes_and_amortizes_owner_reads(
    journal_store, io, separate_callers, asynchronous
):
    draw = ConcurrentDraw(asynchronous=asynchronous)
    params = SamplingParams(max_tokens=8)
    with journal_store() as store:
        wrapper = PersistentBackend(draw, store)
        io.reset()
        io.delay = 0.01
        started = time.monotonic()
        if separate_callers:
            with ThreadPoolExecutor(max_workers=64) as callers:
                answers = list(
                    callers.map(
                        lambda index: wrapper.generate_chat(
                            messages(index), params, repeat_id=index
                        ),
                        range(64),
                    )
                )
        else:
            answers = wrapper.generate_chat_batch(
                [messages(index) for index in range(64)], params, repeat_ids=list(range(64))
            )
        elapsed = time.monotonic() - started
        assert 64 / elapsed >= 50, f"Delayed synthetic journal: {64 / elapsed:.1f} calls/s"
        assert [answer.text for answer in answers] == [str(index) for index in range(64)]
        assert draw.calls == 64 and io.peak["put"] >= 4
        assert io.owner_reads < len(io.records) / 2, "Ownership was checked for each state write"
        for answer in answers:
            assert answer.prompt_tokens == 13 and answer.cached_prompt_tokens == 5
            assert answer.reasoning_tokens == 1 and answer.metadata["raw_token_ids"] == [7, 8]
            assert answer.metadata["persistence"]["previous_attempts"] == 0


def test_batch_tier_submission_and_collection_use_parallel_durable_writes(journal_store, io):
    client = Client()
    with journal_store() as store:
        api = OpenAIBatchBackend(backend(client), store)
        io.reset()
        io.delay = 0.01
        started = time.monotonic()
        handle = api.submit_chat_batch(
            [messages(index) for index in range(64)], SamplingParams(), repeat_ids=list(range(64))
        )
        client.outputs["output"] = b"\n".join(
            json.dumps(completion(key, str(index))).encode()
            for index, key in reversed(list(enumerate(handle.keys)))
        )
        collection = api.collect(handle)
        elapsed = time.monotonic() - started
        assert 64 / elapsed >= 50, f"Delayed synthetic Batch journal: {64 / elapsed:.1f} calls/s"
        assert io.peak["put"] >= 4
        assert io.owner_reads < len(io.records) / 2
        assert not collection.pending and not collection.failures
        assert [r.text for r in collection.results] == [str(index) for index in range(64)]
        assert all(r.metadata["persistence"]["source"] == "batch" for r in collection.results)
        assert len(client.uploads) == len(client.creates) == 1


@pytest.mark.parametrize("separate_callers", [False, True])
def test_cold_lazy_replay_reads_in_parallel_without_new_inference(
    journal_store, io, separate_callers
):
    draw = ConcurrentDraw()
    params = SamplingParams(max_tokens=8)
    with journal_store() as store:
        expected = PersistentBackend(draw, store).generate_chat_batch(
            [messages(index) for index in range(64)], params, repeat_ids=list(range(64))
        )
    with journal_store() as store:
        wrapper = PersistentBackend(draw, store)
        io.reset()
        io.delay = 0.01
        started = time.monotonic()
        if separate_callers:
            with ThreadPoolExecutor(max_workers=64) as callers:
                answers = list(
                    callers.map(
                        lambda index: wrapper.generate_chat(
                            messages(index), params, repeat_id=index
                        ),
                        range(64),
                    )
                )
        else:
            answers = wrapper.generate_chat_batch(
                [messages(index) for index in range(64)], params, repeat_ids=list(range(64))
            )
        assert 64 / (time.monotonic() - started) >= 50
        assert io.peak["get"] >= 4 and draw.calls == 64
        for first, replay in zip(expected, answers, strict=True):
            assert replay.text == first.text and replay.metadata["raw_token_ids"] == [7, 8]
            assert replay.metadata["persistence"] == {
                **first.metadata["persistence"],
                "source": "store",
            }


def test_full_validation_reads_concurrently_and_keeps_discovery_bounded(journal_store, io):
    draw = ConcurrentDraw()
    with journal_store() as store:
        PersistentBackend(draw, store).generate_chat_batch(
            [messages(index) for index in range(256)], SamplingParams(), repeat_ids=list(range(256))
        )
    io.reset()
    io.delay = 0.01
    io.discovery_limit = 256
    started = time.monotonic()
    with journal_store(validation="full"):
        pass
    elapsed = time.monotonic() - started
    assert io.peak["get"] >= 4
    assert 256 / elapsed >= 100_000 / 300, "Full validation remained serial"
    assert io.finished_reads >= 256 and draw.calls == 256


@pytest.mark.parametrize("boundary", ["started", "completed"])
def test_group_acknowledgements_wait_for_their_own_durable_state(journal_store, io, boundary):
    entered, release = threading.Event(), threading.Event()
    draw = ConcurrentDraw()

    def hold(kind, name, record):
        if record and record["state"] == boundary:
            entered.set()
            assert release.wait(10)

    with journal_store() as store, ThreadPoolExecutor(max_workers=1) as caller:
        wrapper = PersistentBackend(draw, store)
        io.before = hold
        future = caller.submit(
            wrapper.generate_chat_batch,
            [messages(index) for index in range(16)],
            SamplingParams(),
            repeat_ids=list(range(16)),
        )
        try:
            assert entered.wait(5)
            assert not future.done(), "A caller was released before durability"
            if boundary == "started":
                assert draw.calls == 0, "Model dispatch preceded a durable intent"
        finally:
            release.set()
        assert len(future.result(timeout=10)) == 16


def test_each_keys_states_remain_ordered_under_concurrent_commit(journal_store, io):
    with journal_store() as store:
        io.reset()
        io.delay = 0.002
        PersistentBackend(ConcurrentDraw(), store).generate_chat_batch(
            [messages(index) for index in range(64)], SamplingParams(), repeat_ids=list(range(64))
        )
        histories = {}
        for record in io.records:
            histories.setdefault(record["key"], []).append(record["state"])
        assert len(histories) == 64
        assert all(states == ["started", "received", "completed"] for states in histories.values())


@pytest.mark.parametrize("phase", ["started", "received", "completed"])
def test_lost_ack_in_one_parallel_write_never_repeats_a_draw(journal_store, io, phase):
    draw = ConcurrentDraw()
    armed = threading.Event()

    def lose_ack(kind, name, record):
        if record and record["state"] == phase and not armed.is_set():
            armed.set()
            io.cloud.fail("after", predicate=lambda target: target == name)

    with journal_store() as store:
        io.before = lose_ack
        first = PersistentBackend(draw, store).generate_chat_batch(
            [messages(index) for index in range(32)], SamplingParams(), repeat_ids=list(range(32))
        )
    io.before = None
    with journal_store() as store:
        replay = PersistentBackend(draw, store).generate_chat_batch(
            [messages(index) for index in range(32)], SamplingParams(), repeat_ids=list(range(32))
        )
    assert draw.calls == 32 and [r.text for r in first] == [r.text for r in replay]


def test_storage_failure_wakes_an_already_coalesced_caller(journal_store, io, monkeypatch):
    draw = ConcurrentDraw()
    entered, joined, release = threading.Event(), threading.Event(), threading.Event()

    def wait_for_sibling():
        entered.set()
        assert release.wait(10)

    draw.before_receipt = wait_for_sibling
    with journal_store() as store:
        wrapper = PersistentBackend(draw, store)
        reserve = store._reserve_prepared

        def observe_reservation(*args, **kwargs):
            reserved = reserve(*args, **kwargs)
            if not reserved[2]:
                joined.set()
            return reserved

        monkeypatch.setattr(store, "_reserve_prepared", observe_reservation)
        callers = ThreadPoolExecutor(max_workers=2)
        try:
            owner = callers.submit(
                wrapper.generate_chat, messages(0), SamplingParams(), repeat_id=0
            )
            assert entered.wait(5)
            shared = callers.submit(
                wrapper.generate_chat, messages(0), SamplingParams(), repeat_id=0
            )
            assert joined.wait(5)
            io.cloud.fail("before", predicate=lambda name: "/calls/" in name, persistent=True)
            release.set()
            for future in (owner, shared):
                with pytest.raises(CheckpointError):
                    future.result(timeout=3)
            assert draw.calls == 1
        finally:
            release.set()
            for pending in list(store._inflight.values()):
                if not pending.done():
                    pending.set_exception(CheckpointError("Synthetic test cleanup"))
            callers.shutdown(wait=True)
            io.cloud.fault = None


def test_partial_group_commit_is_carried_by_a_collection_error(journal_store, io, monkeypatch):
    client = Client()
    with journal_store() as store:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [messages(index) for index in range(32)], SamplingParams(), repeat_ids=list(range(32))
        )
        client.outputs["output"] = b"\n".join(
            json.dumps(completion(key, str(index))).encode()
            for index, key in enumerate(handle.keys)
        )
        confirmed_siblings, finished = threading.Event(), set()
        lock = threading.Lock()
        update = store.update

        def boundary(key, state, **values):
            if key == handle.keys[-1] and state == "completed":
                assert confirmed_siblings.wait(5)
            update(key, state, **values)
            if key != handle.keys[-1] and state == "completed":
                with lock:
                    finished.add(key)
                    if len(finished) == len(handle.keys) - 1:
                        confirmed_siblings.set()

        # Select a real failure boundary after confirmed progress, independently
        # of which worker happens to finish first.
        monkeypatch.setattr(store, "update", boundary)

        def fail_last(kind, name, record):
            if record and record["key"] == handle.keys[-1] and record["state"] == "completed":
                raise io.cloud.error(503)

        io.before = fail_last
        with pytest.raises(CheckpointError) as caught:
            api.collect(handle)
        partial = caught.value.partial_collection
        confirmed = {record["key"] for record in io.records if record["state"] == "completed"}
        assert confirmed
        for index, key in enumerate(handle.keys):
            if key in confirmed:
                assert partial.results[index].metadata["persistence"]["source"] == "batch"
        assert partial.pending == [handle.keys[-1]]
    io.before = None
    with journal_store() as store:
        api = OpenAIBatchBackend(backend(client), store)
        replay = api.collect(handle)
        assert all(
            replay.results[index].metadata["persistence"]["source"] == "store"
            for index, key in enumerate(handle.keys)
            if key in confirmed
        )
        assert len(client.creates) == len(client.uploads) == 1


def test_changed_owner_refuses_the_next_group_before_any_new_call(journal_store, io, gcs_root):
    draw = ConcurrentDraw()
    old = journal_store()
    PersistentBackend(draw, old).generate_chat_batch(
        [messages(index) for index in range(16)], SamplingParams(), repeat_ids=list(range(16))
    )
    journal = gcs_root()
    owner = journal.current_owner(run_descriptor=RUN)
    journal.acknowledge_owner_termination(
        run_descriptor=RUN,
        expected_owner=owner,
        reason="Synthetic displaced-worker test",
        confirmed_terminated=True,
    )
    with journal_store():
        io.reset()
        with pytest.raises(CheckpointError):
            PersistentBackend(draw, old).generate_chat(
                messages("new"), SamplingParams(), repeat_id="new"
            )
        assert draw.calls == 16 and not io.records


def test_owner_taken_during_a_group_prevents_the_waiting_group(journal_store, io, gcs_root):
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    entered, release = threading.Event(), threading.Event()
    journal = GCSJournal(
        bucket="synthetic-bucket",
        prefix="ownership-groups",
        client=io.cloud.client(),
        max_concurrency=4,
    )
    old = journal_store(journal=journal)
    keys = []
    for index in range(8):
        key, _, _ = old.reserve({"synthetic": index}, index)
        keys.append(key)

    def hold_first(kind, name, record):
        if record and record["key"] == keys[0]:
            entered.set()
            assert release.wait(10)

    io.reset()
    io.before = hold_first
    with ThreadPoolExecutor(max_workers=8) as workers:
        first = workers.submit(old.update, keys[0], "started")
        try:
            assert entered.wait(5)
            waiting = [workers.submit(old.update, key, "started") for key in keys[1:]]
            owner = journal.current_owner(run_descriptor=RUN)
            journal.acknowledge_owner_termination(
                run_descriptor=RUN,
                expected_owner=owner,
                confirmed_terminated=True,
                reason="Synthetic owner change during a deliberately blocked group",
            )
            replacement = GCSJournal(
                bucket="synthetic-bucket",
                prefix="ownership-groups",
                client=io.cloud.client(),
            )
            with journal_store(journal=replacement):
                release.set()
                first.result(timeout=10)
                for future in waiting:
                    with pytest.raises(CheckpointError):
                        future.result(timeout=10)
            assert [record["key"] for record in io.records] == keys[:1]
        finally:
            release.set()


@pytest.mark.parametrize("limit", [1, 4, 16])
def test_gcs_concurrency_limit_bounds_remote_operations(io, limit):
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    journal = GCSJournal(
        bucket="synthetic-bucket",
        prefix="bounded-workers",
        client=io.cloud.client(),
        max_concurrency=limit,
    )
    with CallStore.open(journal, run_descriptor=RUN, validation="lazy") as store:
        io.reset()
        io.delay = 0.002
        PersistentBackend(ConcurrentDraw(), store).generate_chat_batch(
            [messages(index) for index in range(32)], SamplingParams(), repeat_ids=list(range(32))
        )
        assert max(io.peak.values()) <= limit
        if limit > 1:
            assert io.peak["put"] >= 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_gcs_concurrency_rejects_invalid_limits(io, limit):
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    with pytest.raises(ValueError):
        GCSJournal(
            bucket="synthetic-bucket",
            prefix="bounded-workers",
            client=io.cloud.client(),
            max_concurrency=limit,
        )


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("boundary", ["received", "completed"])
def test_kill_during_group_keeps_paid_calls_durable_or_explicitly_unknown(
    cloud_server, gcs_root, tmp_path, draw, boundary, validation
):
    from tests.inference.persistence.conftest import CloudClient

    address, service = cloud_server
    journal_type = type(gcs_root())
    journal = journal_type(
        bucket="synthetic-bucket",
        prefix="group-crash",
        client=CloudClient(service),
        cache_dir=tmp_path / "replacement",
    )
    child = r"""
import json, pickle, sys, threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import BaseManager
from llenvs.inference.persistence import CallStore
from llenvs.inference.persistence.storage.gcs import GCSJournal
from llenvs.inference.protocol import ChatMessage, GenerationResult, SamplingParams
from tests.inference.persistence.conftest import Blob, CloudClient
BaseManager.register("storage_cloud")
manager=BaseManager(address=(sys.argv[1],int(sys.argv[2])),authkey=b"synthetic-storage-test")
manager.connect()
journal=GCSJournal(bucket="synthetic-bucket",prefix="group-crash",client=CloudClient(manager.storage_cloud()))
store=CallStore.open(journal,run_descriptor={"experiment":"journal-throughput"},validation="lazy")
owner=journal.current_owner(run_descriptor={"experiment":"journal-throughput"})
keys=[]
for i in range(16):
    request={"backend":{"model":"offline-fixture"},"scope":"default","operation":"chat",
        "arguments":{"input":[ChatMessage("user",str(i))],"params":SamplingParams()}}
    key,_,_=store.reserve(request,i); keys.append(key); store.update(key,"started")
    if sys.argv[3]=="completed":store.update(key,"received",receipt={"text":str(i),"draw":i})
upload=Blob.upload_from_string
ready=threading.Event(); block=threading.Event(); lock=threading.Lock(); committed=set(); announced=False
def gated(self,data,**kwargs):
    global announced
    record=pickle.loads(pickle.loads(data.split(b"\n",1)[1])["payload"]) if "/calls/" in self.name else None
    selected=record and record["state"]==sys.argv[3]
    index=record["request"]["repeat"]["explicit"] if record else -1
    if selected and index>=4:
        assert ready.wait(15)
        with lock:
            if not announced:
                announced=True; print(json.dumps({"owner":owner,"keys":keys}),flush=True)
        block.wait()
    result=upload(self,data,**kwargs)
    if selected:
        with lock:
            committed.add(index)
            if len(committed)>=4:ready.set()
    return result
Blob.upload_from_string=gated
def update(i):
    if sys.argv[3]=="received":store.update(keys[i],"received",receipt={"text":str(i),"draw":i})
    else:store.update(keys[i],"completed",result=GenerationResult(text=str(i),metadata={"draw":i}))
with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(update,range(16)))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child, address[0], str(address[1]), boundary],
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(25), "Worker never reached its partial group write"
        line = process.stdout.readline()
        assert line, "Worker exited before its partial group write"
        checkpoint = json.loads(line)
        process.kill()
        process.communicate(timeout=10)
        journal.acknowledge_owner_termination(
            run_descriptor=RUN,
            expected_owner=checkpoint["owner"],
            reason="Owned synthetic worker is confirmed killed",
            confirmed_terminated=True,
        )
        with CallStore.open(journal, run_descriptor=RUN, validation=validation) as store:
            wrapper = PersistentBackend(draw, store)
            for index in range(16):
                try:
                    answer = wrapper.generate_chat(
                        messages(index), SamplingParams(), repeat_id=index
                    )
                except UnknownOutcomeError:
                    assert boundary == "received" and index >= 4
                else:
                    assert answer.text == str(index) and answer.metadata["draw"] == index
        assert draw.calls == 0, "Recovery repeated a potentially paid call"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.parametrize("provider", ["filesystem", "gcs"])
@pytest.mark.parametrize("delete", [False, True])
def test_finalization_serializes_with_an_active_resubmission(request, tmp_path, provider, delete):
    address = request.getfixturevalue("cloud_server")[0] if provider == "gcs" else ("", 0)
    child = r"""
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from llenvs.inference.persistence import CallStore, UnknownOutcomeError
if sys.argv[1] == "gcs":
    from multiprocessing.managers import BaseManager
    from llenvs.inference.persistence.storage.gcs import GCSJournal
    from tests.inference.persistence.conftest import CloudClient
    BaseManager.register("storage_cloud")
    manager=BaseManager(address=(sys.argv[2],int(sys.argv[3])),authkey=b"synthetic-storage-test")
    manager.connect()
    root=GCSJournal(bucket="synthetic-bucket",prefix="finalization",client=CloudClient(manager.storage_cloud()))
else: root=Path(sys.argv[4])
with CallStore.open(root, run_descriptor={}) as store:
    key,_,_=store.reserve({"synthetic":"finalization"},0)
    store.update(key,"failed",error={"type":"RuntimeError","message":"definite synthetic failure"})
    entered, release, finalizing = threading.Event(), threading.Event(), threading.Event()
    write=store._storage.write_many
    def blocked(records):
        entered.set()
        assert release.wait(5)
        return write(records)
    store._storage.write_many=blocked
    def finish():
        finalizing.set()
        try:store.mark_complete(delete=sys.argv[5]=="True")
        except UnknownOutcomeError:return
        raise AssertionError("Finalization accepted an unresolved resubmission")
    with ThreadPoolExecutor(max_workers=2) as workers:
        recovery=workers.submit(store.acknowledge_resubmission,key,reason="synthetic recovery",accept_duplicate_expenditure=True)
        assert entered.wait(5)
        final=workers.submit(finish)
        assert finalizing.wait(5)
        # Give finalization a chance to wait for the update it must serialize
        # with; no completion or performance assertion depends on this delay.
        time.sleep(0.05)
        release.set()
        recovery.result(timeout=3)
        final.result(timeout=3)
    assert store._records[key]["state"]=="reserved"
    print(json.dumps({"retained":True,"refused_unresolved":True}),flush=True)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child,
            provider,
            address[0],
            str(address[1]),
            str(tmp_path / "store"),
            str(delete),
        ],
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            pytest.fail("Finalization deadlocked with a durable resubmission")
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == {"retained": True, "refused_unresolved": True}
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
