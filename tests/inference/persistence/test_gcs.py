"""Conditional object state and explicit owner recovery; no live cloud traffic."""

import json
import selectors
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from llenvs.inference.persistence import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointInUseError,
    PersistentBackend,
    UnknownOutcomeError,
)
from llenvs.inference.protocol import ChatMessage, GenerationResult, SamplingParams

RUN = {"experiment": "storage-contract"}
MESSAGES = [ChatMessage("user", "synthetic request")]


@pytest.fixture
def native_storage_client():
    storage = pytest.importorskip("google.cloud.storage")
    from google.auth.credentials import AnonymousCredentials
    from requests import Session

    client = storage.Client(
        project="synthetic-project", credentials=AnonymousCredentials(), _http=Session()
    )
    yield client
    client.close()


def test_owned_storage_client_has_a_pool_for_all_record_workers(native_storage_client, monkeypatch):
    import google.cloud.storage

    from llenvs.inference.persistence.storage.gcs import GCSJournal

    monkeypatch.setattr(google.cloud.storage, "Client", lambda: native_storage_client)
    GCSJournal(bucket="synthetic-bucket", prefix="pooled", max_concurrency=32)
    for scheme in ("http://", "https://"):
        adapter = native_storage_client._http.get_adapter(scheme + "storage.example.invalid")
        assert adapter._pool_maxsize >= 32
        assert adapter.max_retries.total == 0


def test_supplied_storage_client_keeps_its_transport_by_default(native_storage_client):
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    session = native_storage_client._http
    adapters = dict(session.adapters)
    GCSJournal(bucket="synthetic-bucket", prefix="pooled", client=native_storage_client)
    assert native_storage_client._http is session
    assert dict(session.adapters) == adapters


@pytest.mark.parametrize("size", [32, 64])
def test_explicit_pool_size_preserves_provided_auth_session_and_retry_policy(
    native_storage_client, size
):
    from requests.adapters import HTTPAdapter

    from llenvs.inference.persistence.storage.gcs import GCSJournal

    session = native_storage_client._http
    original = HTTPAdapter(pool_connections=7, pool_maxsize=10, max_retries=3, pool_block=True)
    session.mount("https://", original)
    GCSJournal(
        bucket="synthetic-bucket",
        prefix="pooled",
        client=native_storage_client,
        max_concurrency=32,
        connection_pool_size=size,
    )
    adapter = session.get_adapter("https://storage.example.invalid")
    assert native_storage_client._http is session
    assert adapter._pool_maxsize >= size
    assert adapter.max_retries == original.max_retries
    assert adapter._pool_connections == 7 and adapter._pool_block is True


def test_explicit_pool_size_never_shrinks_a_larger_existing_pool(native_storage_client):
    from requests.adapters import HTTPAdapter

    from llenvs.inference.persistence.storage.gcs import GCSJournal

    original = HTTPAdapter(pool_maxsize=256)
    native_storage_client._http.mount("https://", original)
    GCSJournal(
        bucket="synthetic-bucket",
        prefix="pooled",
        client=native_storage_client,
        max_concurrency=32,
        connection_pool_size=64,
    )
    assert native_storage_client._http.get_adapter("https://storage.example.invalid") is original


def test_pool_configuration_refuses_custom_adapters_without_replacing_them(native_storage_client):
    from requests.adapters import HTTPAdapter

    from llenvs.inference.persistence.storage.gcs import GCSJournal

    class CustomTLSAdapter(HTTPAdapter):
        pass

    session = native_storage_client._http
    session.mount("https://", CustomTLSAdapter())
    before = dict(session.adapters)
    with pytest.raises(ValueError):
        GCSJournal(
            bucket="synthetic-bucket",
            prefix="pooled",
            client=native_storage_client,
            max_concurrency=32,
            connection_pool_size=64,
        )
    assert dict(session.adapters) == before


def test_pool_configuration_never_bypasses_an_unsupported_transport_wrapper(native_storage_client):
    from types import SimpleNamespace

    from llenvs.inference.persistence.storage.gcs import GCSJournal

    session = native_storage_client._http
    wrapper = SimpleNamespace(delegate=session, close=session.close)
    native_storage_client._http_internal = wrapper
    with pytest.raises(ValueError):
        GCSJournal(
            bucket="synthetic-bucket",
            prefix="pooled",
            client=native_storage_client,
            max_concurrency=32,
            connection_pool_size=64,
        )
    assert native_storage_client._http is wrapper


@pytest.mark.parametrize("size", [0, -1, True, 1.5, 31])
def test_explicit_pool_size_is_an_integer_at_least_the_io_limit(native_storage_client, size):
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    before = dict(native_storage_client._http.adapters)
    with pytest.raises(ValueError):
        GCSJournal(
            bucket="synthetic-bucket",
            prefix="pooled",
            client=native_storage_client,
            max_concurrency=32,
            connection_pool_size=size,
        )
    assert dict(native_storage_client._http.adapters) == before


def test_new_call_uses_conditional_creation_without_a_second_metadata_read(
    gcs_stores, cloud, draw, monkeypatch
):
    from tests.inference.persistence.conftest import Blob

    reloads, writes = [], []
    reload, mutate = Blob.reload, cloud.mutate

    def note_reload(blob, **kwargs):
        reloads.append(blob.name)
        return reload(blob, **kwargs)

    def note_write(key, data, expected):
        if "/calls/" in key[1]:
            writes.append(expected)
        return mutate(key, data, expected)

    monkeypatch.setattr(Blob, "reload", note_reload)
    monkeypatch.setattr(cloud, "mutate", note_write)
    with gcs_stores() as store:
        result = PersistentBackend(draw, store).generate_chat(
            MESSAGES, SamplingParams(), repeat_id=0
        )
        assert result.text == "synthetic request" and draw.calls == 1
    assert writes[0] == 0 and all(value > 0 for value in writes[1:])
    assert not any("/calls/" in name for name in reloads)


def recover(journal, owner, **overrides):
    arguments = {
        "run_descriptor": RUN,
        "expected_owner": owner,
        "reason": "Synthetic worker termination confirmed",
        "confirmed_terminated": True,
    }
    arguments.update(overrides)
    return journal.acknowledge_owner_termination(**arguments)


def test_supplied_client_never_constructs_another_client_or_reads_credentials(
    monkeypatch, gcs_stores
):
    import google.cloud.storage

    def forbidden(*args, **kwargs):
        raise AssertionError("An explicitly supplied client must be used")

    monkeypatch.setattr(google.cloud.storage, "Client", forbidden)
    with gcs_stores():
        pass


def test_atomic_owner_creation_admits_one_of_two_independent_clients(gcs_stores, gcs_root):
    barrier = Barrier(2)

    def acquire():
        barrier.wait(timeout=5)
        try:
            return gcs_stores()
        except CheckpointInUseError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        opened = list(executor.map(lambda _: acquire(), range(2)))
    owners = [store for store in opened if store is not None]
    assert len(owners) == 1
    assert gcs_root().current_owner(run_descriptor=RUN)
    owners[0].close()
    assert gcs_root().current_owner(run_descriptor=RUN) is None


def test_elapsed_time_does_not_authorize_implicit_takeover(monkeypatch, gcs_stores, gcs_root):
    import time

    with gcs_stores():
        owner = gcs_root().current_owner(run_descriptor=RUN)
        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now + 10 * 365 * 24 * 3600)
        with pytest.raises(CheckpointInUseError):
            gcs_stores()
        assert gcs_root().current_owner(run_descriptor=RUN) == owner


@pytest.mark.parametrize("invalid", [{"reason": " "}, {"confirmed_terminated": False}])
def test_owner_recovery_requires_reason_and_explicit_termination_confirmation(
    gcs_stores, gcs_root, invalid
):
    with gcs_stores():
        journal = gcs_root()
        owner = journal.current_owner(run_descriptor=RUN)
        with pytest.raises(ValueError):
            recover(journal, owner, **invalid)
        assert journal.current_owner(run_descriptor=RUN) == owner


def test_stale_termination_confirmation_cannot_displace_a_new_owner(gcs_stores, gcs_root):
    journal = gcs_root()
    with gcs_stores():
        previous = journal.current_owner(run_descriptor=RUN)
    with gcs_stores():
        current = journal.current_owner(run_descriptor=RUN)
        assert current != previous
        with pytest.raises(CheckpointError):
            recover(journal, previous)
        assert journal.current_owner(run_descriptor=RUN) == current


def test_confirmed_termination_reopens_without_resetting_unknown_calls(
    gcs_stores, gcs_root, cloud, draw
):
    # Retaining the Python handle models the displaced owner's outstanding
    # release request. This is an offline state test, not a process-death proof.
    old = gcs_stores()
    params = SamplingParams(max_tokens=16)
    request = {
        "backend": draw.descriptor()["identity"],
        "scope": "default",
        "operation": "chat",
        "arguments": {"input": MESSAGES, "params": params},
    }
    key, _, _ = old.reserve(request, "a")
    old.update(key, "started")
    journal = gcs_root()
    displaced = journal.current_owner(run_descriptor=RUN)
    recover(journal, displaced)
    assert any(
        b"Synthetic worker termination confirmed" in obj.data for obj in cloud.objects.values()
    )
    with gcs_stores() as replacement:
        current = journal.current_owner(run_descriptor=RUN)
        assert current and current != displaced
        with pytest.raises(UnknownOutcomeError) as caught:
            PersistentBackend(draw, replacement).generate_chat(MESSAGES, params, repeat_id="a")
        assert caught.value.key == key and draw.calls == 0
        assert caught.value.metadata["persistence"]["previous_attempts"] == 0
        try:
            old.close()
        except CheckpointError:
            pass
        assert journal.current_owner(run_descriptor=RUN) == current


@pytest.mark.parametrize("phase", ["intent", "receipt", "completion"])
def test_lost_storage_ack_is_reconciled_without_another_draw(gcs_stores, cloud, draw, phase):
    params = SamplingParams(max_tokens=16)
    with gcs_stores() as store:
        wrapper = PersistentBackend(draw, store)
        if phase == "intent":
            cloud.fail("after")
        elif phase == "receipt":
            draw.before_receipt = lambda: cloud.fail("after")
        else:
            draw.before_completion = lambda: cloud.fail("after")
        first = wrapper.generate_chat(MESSAGES, params, repeat_id="a")
        assert first.text == "synthetic request" and draw.calls == 1
    draw.before_receipt = draw.before_completion = None
    with gcs_stores() as store:
        replay = PersistentBackend(draw, store).generate_chat(MESSAGES, params, repeat_id="a")
        assert replay.metadata["persistence"]["source"] == "store" and draw.calls == 1


def test_unacknowledged_intent_prevents_dispatch_and_poisons_the_handle(gcs_stores, cloud, draw):
    store = gcs_stores()
    wrapper = PersistentBackend(draw, store)
    cloud.fail("before", persistent=True)
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, SamplingParams(), repeat_id="a")
    assert draw.calls == 0
    cloud.fault = None
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, SamplingParams(), repeat_id="b")
    assert draw.calls == 0


@pytest.mark.parametrize("phase", ["receipt", "completion"])
def test_failed_persistence_after_a_draw_keeps_unknown_or_reparseable_state(
    gcs_stores, cloud, draw, phase
):
    params = SamplingParams(max_tokens=16)
    first = gcs_stores()
    wrapper = PersistentBackend(draw, first)

    def arm():
        cloud.fail("before", persistent=True)

    if phase == "receipt":
        draw.before_receipt = arm
    else:
        draw.before_completion = arm
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, params, repeat_id="a")
    assert draw.calls == 1
    cloud.fault = None
    draw.before_receipt = draw.before_completion = None
    first.close()
    with gcs_stores() as store:
        wrapper = PersistentBackend(draw, store)
        if phase == "receipt":
            with pytest.raises(UnknownOutcomeError):
                wrapper.generate_chat(MESSAGES, params, repeat_id="a")
        else:
            result = wrapper.generate_chat(MESSAGES, params, repeat_id="a")
            assert result.metadata["persistence"]["source"] == "receipt"
        assert draw.calls == 1


def outage(cloud, clock, when, seconds):
    """Fail every storage operation of one kind for ``seconds`` from now."""
    end = clock.time() + seconds
    cloud.fail(when, predicate=lambda name: clock.time() < end, persistent=True)


def test_a_write_that_succeeds_at_once_never_waits(gcs_stores, draw, storage_clock):
    with gcs_stores() as store:
        PersistentBackend(draw, store).generate_chat(MESSAGES, SamplingParams(), repeat_id="a")
    assert storage_clock.waits == []


def test_a_write_waits_out_a_storage_outage_of_a_minute_without_another_draw(
    gcs_stores, cloud, draw, storage_clock
):
    params = SamplingParams(max_tokens=16)
    with gcs_stores() as store:
        outage(cloud, storage_clock, "before", 60)
        result = PersistentBackend(draw, store).generate_chat(MESSAGES, params, repeat_id="a")
        assert result.text == "synthetic request" and draw.calls == 1
        assert storage_clock.time() >= 60
    with gcs_stores() as store:
        replay = PersistentBackend(draw, store).generate_chat(MESSAGES, params, repeat_id="a")
        assert replay.metadata["persistence"]["source"] == "store" and draw.calls == 1


def test_a_reopened_store_waits_out_a_storage_outage_of_reads(
    gcs_stores, cloud, draw, storage_clock
):
    params = SamplingParams(max_tokens=16)
    with gcs_stores() as store:
        PersistentBackend(draw, store).generate_chat(MESSAGES, params, repeat_id="a")
    outage(cloud, storage_clock, "read", 60)
    with gcs_stores() as store:
        replay = PersistentBackend(draw, store).generate_chat(MESSAGES, params, repeat_id="a")
        assert replay.metadata["persistence"]["source"] == "store" and draw.calls == 1
    assert storage_clock.time() >= 60


def test_batch_files_wait_out_a_storage_outage_on_upload_and_download(
    gcs_stores, cloud, storage_clock
):
    with gcs_stores() as store:
        outage(cloud, storage_clock, "before", 60)
        entry = store.write_blob("input-fixture", [b"synthetic batch file"])
        outage(cloud, storage_clock, "read", 60)
        assert store.blob_path(entry).read_bytes() == b"synthetic batch file"
    assert storage_clock.time() >= 120


def broken_transfer():
    """A connection that closes while a download's bytes arrive, as `requests`
    reports it through google-cloud-storage."""
    from requests.exceptions import ChunkedEncodingError
    from urllib3.exceptions import ProtocolError

    cause = ProtocolError("Connection broken: IncompleteRead(4 bytes read, 16 more expected)")
    error = ChunkedEncodingError(cause)
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    "error",
    [
        broken_transfer,
        lambda: __import__("urllib3").exceptions.ProtocolError("Connection broken"),
    ],
    ids=["requests", "urllib3"],
)
def test_a_batch_file_download_broken_mid_transfer_is_tried_again(
    gcs_stores, cloud, storage_clock, error
):
    with gcs_stores() as store:
        entry = store.write_blob("input-fixture", [b"synthetic batch file"])
        cloud.fail("download", error=error)
        assert store.blob_path(entry).read_bytes() == b"synthetic batch file"
    assert len(storage_clock.waits) == 1


def test_a_missing_object_found_while_reconciling_costs_no_wait(gcs_stores, cloud, storage_clock):
    with gcs_stores() as store:
        cloud.fail("before")
        entry = store.write_blob("input-fixture", [b"synthetic batch file"])
        assert store.blob_path(entry).read_bytes() == b"synthetic batch file"
    assert len(storage_clock.waits) == 1


def test_a_write_gives_up_on_an_outage_after_one_to_two_minutes_without_a_draw(
    gcs_stores, cloud, draw, storage_clock
):
    wrapper = PersistentBackend(draw, gcs_stores())
    cloud.fail("before", persistent=True)
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, SamplingParams(), repeat_id="a")
    assert draw.calls == 0
    assert 60 < storage_clock.time() <= 120


def test_waits_grow_from_at_most_a_second_and_never_exceed_a_minute(
    gcs_stores, cloud, draw, storage_clock
):
    wrapper = PersistentBackend(draw, gcs_stores())
    cloud.fail("before", persistent=True)
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, SamplingParams(), repeat_id="a")
    waits = storage_clock.waits
    assert len(waits) >= 6
    assert all(0 <= wait <= min(2.0**index, 60.0) for index, wait in enumerate(waits))


def test_conflicting_generation_is_not_mistaken_for_our_lost_ack(gcs_stores, cloud, draw):
    store = gcs_stores()
    wrapper = PersistentBackend(draw, store)
    cloud.conflict = True
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, SamplingParams(), repeat_id="a")
    assert draw.calls == 0
    with pytest.raises(CheckpointError):
        wrapper.generate_chat(MESSAGES, SamplingParams(), repeat_id="b")
    assert draw.calls == 0


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_changed_record_bytes_are_detected_and_never_cause_dispatch(
    gcs_stores, gcs_root, cloud, draw, validation
):
    params = SamplingParams(max_tokens=16)
    with gcs_stores() as store:
        result = PersistentBackend(draw, store).generate_chat(MESSAGES, params, repeat_id="a")
        key = result.metadata["persistence"]["call_key"]
    retained = [name for name in cloud.objects if key in name[1]]
    assert len(retained) == 1
    name = retained[0]
    cloud.mutate(name, b"invalid retained record", cloud.objects[name].generation)
    if validation == "full":
        with pytest.raises(CheckpointCorruptionError):
            gcs_stores(validation=validation)
        assert gcs_root().current_owner(run_descriptor=RUN) is None
    else:
        store = gcs_stores(validation=validation)
        wrapper = PersistentBackend(draw, store)
        with pytest.raises(CheckpointCorruptionError):
            wrapper.generate_chat(MESSAGES, params, repeat_id="a")
        with pytest.raises(CheckpointError):
            wrapper.generate_chat(MESSAGES, params, repeat_id="b")
    assert draw.calls == 1


def test_lost_ack_during_initial_open_and_close_keeps_exclusive_ownership(
    gcs_stores, gcs_root, cloud
):
    cloud.fail("after")
    first = gcs_stores()
    owner = gcs_root().current_owner(run_descriptor=RUN)
    assert owner
    with pytest.raises(CheckpointInUseError):
        gcs_stores()
    cloud.fail("after")
    first.close()
    assert gcs_root().current_owner(run_descriptor=RUN) is None
    with gcs_stores():
        assert gcs_root().current_owner(run_descriptor=RUN) != owner


def test_lost_ack_while_recovering_an_owner_does_not_require_another_recovery(
    gcs_stores, gcs_root, cloud
):
    old = gcs_stores()
    journal = gcs_root()
    owner = journal.current_owner(run_descriptor=RUN)
    names = [name for name, obj in cloud.objects.items() if owner.encode() in obj.data]
    assert len(names) == 1
    cloud.fail("after", predicate=lambda name: name == names[0][1])
    recover(journal, owner)
    with gcs_stores():
        replacement = journal.current_owner(run_descriptor=RUN)
        assert replacement and replacement != owner
        try:
            old.close()
        except CheckpointError:
            pass
        assert journal.current_owner(run_descriptor=RUN) == replacement


def test_large_blob_streams_and_cache_loss_does_not_lose_authoritative_bytes(
    gcs_stores, cloud, tmp_path
):
    block = bytes(range(256)) * 256
    with gcs_stores() as store:
        entry = store.write_blob("output-fixture", (block for _ in range(32)))
        assert store.blob_path(entry).read_bytes() == block * 32
    shutil.rmtree(tmp_path / "disposable-cache", ignore_errors=True)
    with gcs_stores() as reopened:
        assert reopened.blob_path(entry).read_bytes() == block * 32
    assert not any(kind == "string" and size >= len(block) * 32 for kind, _, size in cloud.uploads)


def test_lazy_reopen_does_not_read_or_list_completed_call_objects(
    gcs_stores, cloud, request_record
):
    keys = set()
    with gcs_stores() as store:
        for index in range(1000):
            key, _, _ = store.reserve(request_record, index)
            keys.add(key)
            store.update(key, "completed", result=GenerationResult(text="retained"))
    cloud.operations.clear()
    cloud.listed_names.clear()
    with gcs_stores(validation="lazy"):
        pass
    accessed = [name for operation, name in cloud.operations if operation == "get"]
    accessed += cloud.listed_names
    assert not any(key in name for key in keys for name in accessed)


@pytest.mark.parametrize("boundary", ["started", "received", "completed"])
def test_killed_worker_requires_confirmed_owner_recovery_and_never_redispatches(
    cloud_server, gcs_root, tmp_path, draw, boundary
):
    from tests.inference.persistence.conftest import CloudClient

    address, service = cloud_server
    journal_type = type(gcs_root())
    journal = journal_type(
        bucket="synthetic-bucket",
        prefix="crash-fixture",
        client=CloudClient(service),
        cache_dir=tmp_path / "replacement-cache",
    )
    child = r"""
import json, sys
from multiprocessing.managers import BaseManager
from pathlib import Path
from llenvs.inference.persistence import CallStore
from llenvs.inference.persistence.storage.gcs import GCSJournal
from llenvs.inference.protocol import GenerationResult, SamplingParams
from tests.inference.persistence.conftest import CloudClient
from tests.inference.persistence.test_gcs import RUN, MESSAGES
BaseManager.register("storage_cloud")
manager = BaseManager(address=(sys.argv[1], int(sys.argv[2])), authkey=b"synthetic-storage-test")
manager.connect()
journal = GCSJournal(bucket="synthetic-bucket", prefix="crash-fixture",
    client=CloudClient(manager.storage_cloud()), cache_dir=Path(sys.argv[4]))
store = CallStore.open(journal, run_descriptor=RUN, validation="lazy")
params = SamplingParams(max_tokens=16)
request = {"backend":{"model":"offline-fixture"}, "scope":"default", "operation":"chat",
    "arguments":{"input":MESSAGES, "params":params}}
key, _, _ = store.reserve(request, "a")
store.update(key, "started")
if sys.argv[3] in ("received", "completed"):
    store.update(key, "received", receipt={"text":"retained", "draw":42})
if sys.argv[3] == "completed":
    store.update(key, "completed", result=GenerationResult(text="retained", metadata={"draw":42}))
print(json.dumps({"key":key,"owner":journal.current_owner(run_descriptor=RUN)}), flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child,
            address[0],
            str(address[1]),
            boundary,
            str(tmp_path / "dead-worker-cache"),
        ],
        cwd=Path(__file__).resolve().parents[3],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(60), "Worker did not reach its durable interruption boundary"
        line = process.stdout.readline()
        assert line, "Worker exited before its durable boundary"
        receipt = json.loads(line)
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode != 0
        assert journal.current_owner(run_descriptor=RUN) == receipt["owner"]
        with pytest.raises(CheckpointInUseError):
            from llenvs.inference.persistence import CallStore

            CallStore.open(journal, run_descriptor=RUN, validation="lazy")
        recover(journal, receipt["owner"])
        from llenvs.inference.persistence import CallStore

        with CallStore.open(journal, run_descriptor=RUN, validation="lazy") as replacement:
            wrapper = PersistentBackend(draw, replacement)
            if boundary == "started":
                with pytest.raises(UnknownOutcomeError) as caught:
                    wrapper.generate_chat(MESSAGES, SamplingParams(max_tokens=16), repeat_id="a")
                assert caught.value.key == receipt["key"]
            else:
                result = wrapper.generate_chat(
                    MESSAGES, SamplingParams(max_tokens=16), repeat_id="a"
                )
                assert result.text == "retained"
                assert result.metadata["persistence"]["source"] == (
                    "receipt" if boundary == "received" else "store"
                )
            assert draw.calls == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
