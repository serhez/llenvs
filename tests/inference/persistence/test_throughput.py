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
from contextlib import contextmanager
from functools import partial
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


def test_single_io_worker_refuses_waiting_groups_after_takeover(journal_store, io, gcs_root):
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    entered, release = threading.Event(), threading.Event()
    journal = GCSJournal(
        bucket="synthetic-bucket",
        prefix="ownership-groups",
        client=io.cloud.client(),
        max_concurrency=1,
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
            replacement = GCSJournal(
                bucket="synthetic-bucket",
                prefix="ownership-groups",
                client=io.cloud.client(),
            )
            # Recovery uses an independent client's transport; the old client's
            # only I/O slot is deliberately occupied by the blocked upload.
            owner = replacement.current_owner(run_descriptor=RUN)
            replacement.acknowledge_owner_termination(
                run_descriptor=RUN,
                expected_owner=owner,
                confirmed_terminated=True,
                reason="Synthetic owner change during a deliberately blocked group",
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


def test_an_independent_group_finishes_while_an_earlier_upload_is_held(journal_store, io):
    entered, release, advanced = threading.Event(), threading.Event(), threading.Event()
    with journal_store() as store, ThreadPoolExecutor(max_workers=3) as callers:
        first, _, _ = store.reserve({"synthetic": "slow"}, 0)
        second, second_result, _ = store.reserve({"synthetic": "independent"}, 0)

        def hold(kind, name, record):
            if record and record["key"] == first:
                if record["state"] == "started":
                    entered.set()
                    assert release.wait(5)
                else:
                    advanced.set()

        io.before = hold
        slow = callers.submit(store.update, first, "started")
        try:
            assert entered.wait(3)
            independent = callers.submit(
                store.update, second, "completed", result=GenerationResult(text="durable")
            )
            later = callers.submit(store.update, first, "received", receipt={"text": "later"})
            independent.result(timeout=2)
            assert second_result.result(timeout=1).text == "durable"
            assert not slow.done() and not later.done() and not advanced.is_set()
        finally:
            release.set()
        slow.result(timeout=3)
        later.result(timeout=3)
        assert advanced.is_set()
        assert [r["state"] for r in io.records if r["key"] == first] == ["started", "received"]


def test_failed_group_drains_confirmed_siblings_before_poisoning_waiters(
    journal_store, io, monkeypatch
):
    first_entered, second_entered = threading.Event(), threading.Event()
    release_failure, release_success, failure_reported = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    with journal_store() as store, ThreadPoolExecutor(max_workers=3) as callers:
        first, _, _ = store.reserve({"synthetic": "failure"}, 0)
        second, result, _ = store.reserve({"synthetic": "success"}, 0)
        third, _, _ = store.reserve({"synthetic": "queued"}, 0)
        store.update(first, "started")
        store.update(second, "started")

        def gate(kind, name, record):
            if not record or record["state"] != "completed":
                return
            if record["key"] == first:
                first_entered.set()
                assert release_failure.wait(5)
                raise io.cloud.error(503)
            if record["key"] == second:
                second_entered.set()
                assert release_success.wait(5)

        write_many = store._storage.write_many

        def observe(records):
            errors = write_many(records)
            if any(error is not None for error in errors):
                failure_reported.set()
            return errors

        monkeypatch.setattr(store._storage, "write_many", observe)
        io.before = gate
        failed = callers.submit(
            store.update, first, "completed", result=GenerationResult(text="lost")
        )
        try:
            assert first_entered.wait(3)
            succeeded = callers.submit(
                store.update, second, "completed", result=GenerationResult(text="retained")
            )
            assert second_entered.wait(2), "The second group never overlapped the first"
            queued = callers.submit(store.update, third, "started")
            release_failure.set()
            assert failure_reported.wait(3)
            with pytest.raises(TimeoutError):
                failed.result(timeout=0.05)
            assert not result.done(), "A successful sibling was poisoned before its write finished"
            release_success.set()
            with pytest.raises(CheckpointError):
                failed.result(timeout=3)
            succeeded.result(timeout=3)
            assert result.result(timeout=1).text == "retained"
            with pytest.raises(CheckpointError):
                queued.result(timeout=3)
            assert not any(r["key"] == third for r in io.records)
        finally:
            release_failure.set()
            release_success.set()
    io.before = None
    with journal_store() as reopened:
        _, replay, owner = reopened.reserve({"synthetic": "success"}, 0)
        assert not owner and replay.result().text == "retained"
        _, unknown, owner = reopened.reserve({"synthetic": "failure"}, 0)
        assert not owner
        with pytest.raises(UnknownOutcomeError):
            unknown.result()


def test_two_checked_groups_can_finish_after_takeover_but_the_queued_group_cannot(
    journal_store, io, gcs_root
):
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    old = journal_store()
    keys = [old.reserve({"synthetic": index}, 0)[0] for index in range(3)]

    def gate(kind, name, record):
        if record and record["key"] in keys[:2]:
            entered[keys.index(record["key"])].set()
            assert release.wait(5)

    io.before = gate
    with ThreadPoolExecutor(max_workers=3) as callers:
        first = callers.submit(old.update, keys[0], "started")
        try:
            assert entered[0].wait(3)
            second = callers.submit(old.update, keys[1], "started")
            assert entered[1].wait(2), "The second group did not pass its owner check"
            queued = callers.submit(old.update, keys[2], "started")
            journal = gcs_root()
            journal.acknowledge_owner_termination(
                run_descriptor=RUN,
                expected_owner=journal.current_owner(run_descriptor=RUN),
                reason="Synthetic takeover while both checked groups are deliberately held",
                confirmed_terminated=True,
            )
            with journal_store():
                release.set()
                first.result(timeout=3)
                second.result(timeout=3)
                with pytest.raises(CheckpointError):
                    queued.result(timeout=3)
            assert {record["key"] for record in io.records} == set(keys[:2])
        finally:
            release.set()


def test_close_drains_both_groups_before_releasing_owner_and_generations(journal_store, io):
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    store = journal_store()
    keys = [store.reserve({"synthetic": index}, 0)[0] for index in range(2)]

    def gate(kind, name, record):
        if record and record["key"] in keys:
            entered[keys.index(record["key"])].set()
            assert release.wait(5)

    io.before = gate
    with ThreadPoolExecutor(max_workers=3) as callers:
        first = callers.submit(store.update, keys[0], "started")
        try:
            assert entered[0].wait(3)
            second = callers.submit(store.update, keys[1], "started")
            assert entered[1].wait(2), "Close must cover actual overlapping groups"
            closing = callers.submit(store.close)
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.05)
            release.set()
            first.result(timeout=3)
            second.result(timeout=3)
            closing.result(timeout=3)
            assert not store._storage._objects.pinned
        finally:
            release.set()


def test_filesystem_writes_remain_serial_with_the_parallel_coordinator(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    with (
        CallStore.open(tmp_path, run_descriptor=RUN) as store,
        ThreadPoolExecutor(max_workers=2) as callers,
    ):
        keys = [store.reserve({"synthetic": index}, 0)[0] for index in range(2)]
        write = store._storage.write

        def gate(name, data):
            if keys[0] in name:
                entered.set()
                assert release.wait(5)
            return write(name, data)

        monkeypatch.setattr(store._storage, "write", gate)
        first = callers.submit(store.update, keys[0], "started")
        try:
            assert entered.wait(3)
            second = callers.submit(store.update, keys[1], "started")
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
        finally:
            release.set()
        first.result(timeout=3)
        second.result(timeout=3)


@contextmanager
def held_call_state(store, monkeypatch, state, *, key=None):
    """Hold a real call write after its group has been admitted."""
    entered, release, keys = threading.Event(), threading.Event(), []
    method = "write" if store.run_dir is not None else "_write_record"
    write = getattr(store._storage, method)

    def hold(name, payload):
        if name.startswith("calls/"):
            record = pickle.loads(pickle.loads(payload)["payload"])
            if record["state"] == state and (key is None or record["key"] == key):
                keys.append(record["key"])
                entered.set()
                assert release.wait(10), "The test did not release the admitted write"
        return write(name, payload)

    with monkeypatch.context() as patch:
        patch.setattr(store._storage, method, hold)
        try:
            yield entered, release, keys
        finally:
            release.set()


def prepare_independent_failure(stores, kind):
    """Retain valid fixtures, then reopen so a record read is genuinely cold."""
    if kind not in {"journal", "blob_read", "record_read"}:
        return {}
    with stores() as store:
        if kind == "journal":
            api = OpenAIBatchBackend(backend(Client()), store)
            handle = api.submit_chat_batch(
                [messages("unrelated")], SamplingParams(), repeat_ids=[0]
            )
            return {"journal": copy.deepcopy(store._journals[handle.submissions[0]])}
        if kind == "blob_read":
            return {"blob": store.write_blob("retainedInput", b"synthetic retained input")}
        request = {"synthetic": "cold-record"}
        key, _, _ = store.reserve(request, 0)
        store.update(key, "completed", result=GenerationResult(text="seed"))
        return {"request": request, "key": key}


def independent_failure(store, kind, fixture, monkeypatch):
    """Fail the actual store operation at its storage boundary, without paid I/O."""
    error = OSError("Synthetic independent storage failure")
    observed = threading.Event()

    def fail(*args, **kwargs):
        observed.set()
        raise error

    if kind in {"lifecycle", "journal"}:
        write = store._storage.write
        prefix = "lifecycle/" if kind == "lifecycle" else "journals/"

        def write_or_fail(name, payload):
            return fail() if name.startswith(prefix) else write(name, payload)

        monkeypatch.setattr(store._storage, "write", write_or_fail)
        operation = (
            partial(store.lifecycle, {"event": "synthetic independent diagnostic"})
            if kind == "lifecycle"
            else partial(store.write_journal, fixture["journal"])
        )
    elif kind == "blob_write":
        monkeypatch.setattr(store._storage, "write_blob", fail)
        operation = partial(store.write_blob, "independentInput", b"synthetic input")
    elif kind == "blob_read":
        monkeypatch.setattr(store._storage, "blob_path", fail)
        operation = partial(store.blob_path, fixture["blob"])
    else:
        assert kind == "record_read"
        read = store._storage.read
        name = store._records.path(fixture["key"])
        monkeypatch.setattr(
            store._storage, "read", lambda path: fail() if path == name else read(path)
        )
        operation = partial(store.reserve, fixture["request"], 0)
    return operation, error, observed


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("kind", ["lifecycle", "journal", "blob_write", "blob_read", "record_read"])
def test_independent_failure_drains_a_confirmed_draw_before_poisoning(
    stores, monkeypatch, kind, validation
):
    fixture = prepare_independent_failure(stores, kind)
    draw, params = ConcurrentDraw(), SamplingParams(max_tokens=8)
    with stores() as store, ThreadPoolExecutor(max_workers=3) as callers:
        wrapper = PersistentBackend(draw, store)
        with held_call_state(store, monkeypatch, "completed") as (entered, release, keys):
            completed = callers.submit(wrapper.generate_chat, messages("kept"), params, repeat_id=0)
            try:
                assert entered.wait(3) and draw.calls == 1
                outcome = store._inflight[keys[0]]
                with monkeypatch.context() as fault:
                    operation, error, observed = independent_failure(store, kind, fixture, fault)
                    failing = callers.submit(operation)
                    assert observed.wait(3)
                    refused = callers.submit(
                        wrapper.generate_chat, messages("must not dispatch"), params, repeat_id=1
                    )
                    with pytest.raises(CheckpointError):
                        refused.result(timeout=3)
                    assert draw.calls == 1
                    with pytest.raises(TimeoutError):
                        outcome.result(timeout=0.05)
                    release.set()
                    with pytest.raises(CheckpointError) as caught:
                        failing.result(timeout=3)
                    assert caught.value.__cause__ is error
                    answer = completed.result(timeout=3)
                    assert answer.text == "kept"
                    assert answer.metadata["persistence"]["source"] == "backend"
                    assert answer.metadata["persistence"]["previous_attempts"] == 0
                    assert outcome.result().text == "kept"
            finally:
                release.set()
        with pytest.raises(CheckpointError):
            wrapper.generate_chat(messages("still poisoned"), params, repeat_id=2)
    with stores(validation=validation) as reopened:
        replay = PersistentBackend(draw, reopened).generate_chat(
            messages("kept"), params, repeat_id=0
        )
        assert replay.text == "kept" and draw.calls == 1
        assert replay.metadata["persistence"] == {
            **answer.metadata["persistence"],
            "source": "store",
        }


def test_serialization_failure_does_not_hide_an_admitted_completion(stores, monkeypatch):
    draw, params = ConcurrentDraw(), SamplingParams(max_tokens=8)
    with stores() as store, ThreadPoolExecutor(max_workers=3) as callers:
        bad_key, _, _ = store.reserve({"synthetic": "cannot-pickle"}, 0)
        wrapper = PersistentBackend(draw, store)
        with held_call_state(store, monkeypatch, "completed") as (entered, release, keys):
            completed = callers.submit(wrapper.generate_chat, messages("kept"), params, repeat_id=0)
            try:
                assert entered.wait(3)
                outcome = store._inflight[keys[0]]
                invalid = GenerationResult(text="invalid", metadata={"lock": threading.Lock()})
                failing = callers.submit(store.update, bad_key, "completed", result=invalid)
                try:
                    failing.result(timeout=0.05)
                except (TimeoutError, CheckpointError):
                    pass
                refused = callers.submit(
                    wrapper.generate_chat, messages("must not dispatch"), params, repeat_id=1
                )
                with pytest.raises(CheckpointError):
                    refused.result(timeout=3)
                with pytest.raises(TimeoutError):
                    outcome.result(timeout=0.05)
                release.set()
                with pytest.raises(CheckpointError) as caught:
                    failing.result(timeout=3)
                assert isinstance(caught.value.__cause__, TypeError)
                answer = completed.result(timeout=3)
                assert answer.text == "kept" and draw.calls == 1
                assert answer.metadata["persistence"]["source"] == "backend"
            finally:
                release.set()


def test_independent_failure_blocks_dispatch_from_an_admitted_intent(stores, monkeypatch):
    draw, params = ConcurrentDraw(), SamplingParams(max_tokens=8)
    with stores() as store, ThreadPoolExecutor(max_workers=2) as callers:
        wrapper = PersistentBackend(draw, store)
        with held_call_state(store, monkeypatch, "started") as (entered, release, keys):
            started = callers.submit(
                wrapper.generate_chat, messages("must not run"), params, repeat_id=0
            )
            try:
                assert entered.wait(3) and draw.calls == 0
                with monkeypatch.context() as fault:
                    operation, error, observed = independent_failure(store, "lifecycle", {}, fault)
                    failing = callers.submit(operation)
                    assert observed.wait(3)
                    try:
                        failing.result(timeout=0.05)
                    except (TimeoutError, CheckpointError):
                        pass
                    release.set()
                    with pytest.raises(CheckpointError):
                        failing.result(timeout=3)
                    with pytest.raises(CheckpointError):
                        started.result(timeout=3)
                    assert draw.calls == 0
            finally:
                release.set()
    with stores() as reopened:
        with pytest.raises(UnknownOutcomeError):
            PersistentBackend(draw, reopened).generate_chat(
                messages("must not run"), params, repeat_id=0
            )
        assert draw.calls == 0


@pytest.mark.parametrize("validation", ["full", "lazy"])
@pytest.mark.parametrize("first_outcome", ["success", "refusal"])
def test_collection_keeps_confirmed_progress_during_an_independent_failure(
    stores, monkeypatch, validation, first_outcome
):
    client, params = Client(), SamplingParams()
    with stores() as store, ThreadPoolExecutor(max_workers=3) as callers:
        api = OpenAIBatchBackend(backend(client), store)
        handle = api.submit_chat_batch(
            [messages("first"), messages("second")], params, repeat_ids=[0, 1]
        )
        first_entry = completion(handle.keys[0], "first")
        if first_outcome == "refusal":
            first_entry["response"]["body"]["choices"][0]["message"].update(
                content=None, refusal="Synthetic policy refusal"
            )
        client.outputs["output"] = b"\n".join(
            json.dumps(entry).encode()
            for entry in [first_entry, completion(handle.keys[1], "second")]
        )
        state = "completed" if first_outcome == "success" else "failed"
        second_entered, release_second = threading.Event(), threading.Event()
        update = store.update

        def hold_second(key, state, **values):
            if key == handle.keys[1]:
                second_entered.set()
                assert release_second.wait(10)
            return update(key, state, **values)

        monkeypatch.setattr(store, "update", hold_second)
        with held_call_state(store, monkeypatch, state, key=handle.keys[0]) as (
            entered,
            release,
            _,
        ):
            collecting = callers.submit(api.collect, handle)
            try:
                assert entered.wait(3)
                if store._storage.max_concurrency > 1:
                    assert second_entered.wait(3)
                with monkeypatch.context() as fault:
                    operation, error, observed = independent_failure(store, "lifecycle", {}, fault)
                    failing = callers.submit(operation)
                    assert observed.wait(3)
                    try:
                        failing.result(timeout=0.05)
                    except (TimeoutError, CheckpointError):
                        pass
                    refused = callers.submit(store.reserve, {"synthetic": "new work"}, 0)
                    with pytest.raises(CheckpointError):
                        refused.result(timeout=3)
                    release.set()
                    release_second.set()
                    with pytest.raises(CheckpointError) as outside:
                        failing.result(timeout=3)
                    assert outside.value.__cause__ is error
                    with pytest.raises(CheckpointError) as caught:
                        collecting.result(timeout=3)
                partial = caught.value.partial_collection
                assert len(partial.results) == 2 and partial.pending == [handle.keys[1]]
                first = partial.results[0] if first_outcome == "success" else partial.failures[0]
                assert first.metadata["persistence"]["source"] == "batch"
                assert first.metadata["persistence"]["previous_attempts"] == 0
                assert 1 not in partial.failures
            finally:
                release.set()
                release_second.set()
    with stores(validation=validation) as reopened:
        replay = OpenAIBatchBackend(backend(client), reopened).collect(handle)
        assert not replay.pending and replay.results[1].text == "second"
        first = replay.results[0] if first_outcome == "success" else replay.failures[0]
        assert first.metadata["persistence"]["source"] == "store"
        assert replay.results[1].metadata["persistence"]["source"] == "batch"
        assert len(client.creates) == len(client.uploads) == 1


def test_independent_storage_failure_drains_both_groups_and_refuses_the_queue(
    journal_store, io, monkeypatch
):
    with journal_store() as store, ThreadPoolExecutor(max_workers=4) as callers:
        reservations = [store.reserve({"synthetic": index}, 0) for index in range(3)]
        first_key, first_result, _ = reservations[0]
        second_key, second_result, _ = reservations[1]
        queued_key = reservations[2][0]
        with (
            held_call_state(store, monkeypatch, "completed", key=first_key) as first,
            held_call_state(store, monkeypatch, "completed", key=second_key) as second,
        ):
            first_write = callers.submit(
                store.update, first_key, "completed", result=GenerationResult(text="first")
            )
            try:
                assert first[0].wait(3)
                second_write = callers.submit(
                    store.update, second_key, "completed", result=GenerationResult(text="second")
                )
                assert second[0].wait(3)
                queued_write = callers.submit(store.update, queued_key, "started")
                io.cloud.fail(
                    "before", predicate=lambda name: "/lifecycle/" in name, persistent=True
                )
                failing = callers.submit(store.lifecycle, {"event": "synthetic failure"})
                try:
                    failing.result(timeout=0.05)
                except (TimeoutError, CheckpointError):
                    pass
                with pytest.raises(CheckpointError):
                    store.reserve({"synthetic": "refused"}, 0)
                for outcome in (first_result, second_result):
                    with pytest.raises(TimeoutError):
                        outcome.result(timeout=0.05)
                first[1].set()
                with pytest.raises(TimeoutError):
                    second_result.result(timeout=0.05)
                second[1].set()
                with pytest.raises(CheckpointError):
                    failing.result(timeout=3)
                first_write.result(timeout=3)
                second_write.result(timeout=3)
                assert first_result.result().text == "first"
                assert second_result.result().text == "second"
                with pytest.raises(CheckpointError):
                    queued_write.result(timeout=3)
                assert {record["key"] for record in io.records} == {first_key, second_key}
            finally:
                first[1].set()
                second[1].set()
                io.cloud.fault = None
