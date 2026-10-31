"""Storage contract fixtures; every object and provider response is synthetic."""

import threading
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.managers import BaseManager
from types import SimpleNamespace

import pytest

from llenvs.inference.persistence import CallStore, CheckpointError
from llenvs.inference.protocol import BackendCapabilities, GenerationResult


@dataclass
class Object:
    data: bytes
    generation: int


class Cloud:
    """A shared service behind independent clients, with atomic generation checks."""

    def __init__(self):
        self.objects = {}
        self.generation = 0
        self.operations = []
        self.uploads = []
        self.listed_names = []
        self.conflict = False
        self.fault = None
        self.mutex = threading.RLock()

    def error(self, code):
        from google.api_core import exceptions

        return {
            404: exceptions.NotFound,
            412: exceptions.PreconditionFailed,
            503: exceptions.ServiceUnavailable,
        }[code]("Synthetic storage failure")

    def fail(self, when, *, predicate=lambda name: True, persistent=False, error=None):
        """Fail operations of phase ``when`` with ``error()``, or with HTTP 503
        when no error is given; the phase "download" fails a blob's bytes after
        its metadata was read."""
        self.fault = (when, predicate, persistent, error)

    def maybe_fail(self, when, name):
        if self.fault is None:
            return
        phase, predicate, persistent, error = self.fault
        if phase == when and predicate(name):
            if not persistent:
                self.fault = None
            raise error() if error is not None else self.error(503)

    def read(self, key, generation=None):
        with self.mutex:
            self.operations.append(("get", key[1]))
            self.maybe_fail("read", key[1])
            obj = self.objects.get(key)
            if obj is None or (generation is not None and obj.generation != int(generation)):
                raise self.error(404)
            return SimpleNamespace(data=obj.data, generation=obj.generation)

    def mutate(self, key, data, expected):
        assert expected is not None, "Every GCS mutation needs a generation precondition"
        with self.mutex:
            self.operations.append(("delete" if data is None else "put", key[1]))
            self.maybe_fail("before", key[1])
            if self.conflict:
                self.conflict = False
                self.generation += 1
                self.objects[key] = Object(b"a different writer's bytes", self.generation)
            previous = self.objects.get(key)
            actual = previous.generation if previous else 0
            if int(expected) != actual:
                raise self.error(412)
            if data is None:
                if previous is None:
                    raise self.error(404)
                del self.objects[key]
                result = None
            else:
                self.generation += 1
                result = Object(bytes(data), self.generation)
                self.objects[key] = result
            self.maybe_fail("after", key[1])
            return (
                SimpleNamespace(data=result.data, generation=result.generation)
                if result is not None
                else None
            )

    def client(self):
        return CloudClient(self)

    def list_names(self, bucket, prefix):
        with self.mutex:
            self.operations.append(("list", prefix))
            names = sorted(
                name for b, name in self.objects if b == bucket and name.startswith(prefix)
            )
            self.listed_names.extend(names)
            return names

    def note_upload(self, kind, name, size):
        self.uploads.append((kind, name, size))


class CloudClient:
    def __init__(self, cloud):
        self.cloud = cloud

    def bucket(self, name):
        return Bucket(self.cloud, name)

    def list_blobs(self, bucket, *, prefix="", **kwargs):
        bucket = bucket if isinstance(bucket, str) else bucket.name
        return iter(
            Blob(self.cloud, bucket, name) for name in self.cloud.list_names(bucket, prefix)
        )


class CloudManager(BaseManager):
    pass


_managed_cloud = None


def shared_cloud():
    global _managed_cloud
    if _managed_cloud is None:
        _managed_cloud = Cloud()
    return _managed_cloud


CloudManager.register("storage_cloud", shared_cloud)


class Bucket:
    def __init__(self, cloud, name):
        self.cloud, self.name = cloud, name

    def blob(self, name, *, generation=None, **kwargs):
        return Blob(self.cloud, self.name, name, generation=generation)

    def get_blob(self, name, **kwargs):
        result = self.blob(name)
        try:
            result.reload(**kwargs)
        except Exception as exc:
            if getattr(exc, "code", None) == 404:
                return None
            raise
        return result


class Blob:
    def __init__(self, cloud, bucket, name, *, generation=None):
        self.cloud, self.name = cloud, name
        self.bucket = Bucket(cloud, bucket)
        self.generation = generation
        self.size = None
        self.metadata = {}

    @property
    def key(self):
        return self.bucket.name, self.name

    def reload(self, **kwargs):
        result = self.cloud.read(self.key, self.generation)
        self.generation, self.size = result.generation, len(result.data)

    def exists(self, **kwargs):
        try:
            self.reload(**kwargs)
            return True
        except Exception as exc:
            if getattr(exc, "code", None) == 404:
                return False
            raise

    def download_as_bytes(self, *, if_generation_match=None, **kwargs):
        self.cloud.maybe_fail("download", self.name)
        result = self.cloud.read(self.key, self.generation)
        if if_generation_match is not None and result.generation != int(if_generation_match):
            raise self.cloud.error(412)
        self.generation, self.size = result.generation, len(result.data)
        return result.data

    def download_to_file(self, stream, **kwargs):
        stream.write(self.download_as_bytes(**kwargs))

    def download_to_filename(self, name, **kwargs):
        with open(name, "wb") as stream:
            self.download_to_file(stream, **kwargs)

    def _upload(self, data, expected):
        result = self.cloud.mutate(self.key, data, expected)
        self.generation, self.size = result.generation, len(result.data)

    def upload_from_string(self, data, *, if_generation_match=None, **kwargs):
        data = data.encode() if isinstance(data, str) else data
        self.cloud.note_upload("string", self.name, len(data))
        self._upload(data, if_generation_match)

    def upload_from_file(self, stream, *, if_generation_match=None, rewind=False, **kwargs):
        if rewind:
            stream.seek(0)
        chunks = iter(lambda: stream.read(64 * 1024), b"")
        data = b"".join(chunks)
        self.cloud.note_upload("file", self.name, len(data))
        self._upload(data, if_generation_match)

    def upload_from_filename(self, name, **kwargs):
        with open(name, "rb") as stream:
            self.upload_from_file(stream, **kwargs)

    def delete(self, *, if_generation_match=None, **kwargs):
        self.cloud.mutate(self.key, None, if_generation_match)


@pytest.fixture(autouse=True)
def storage_clock(monkeypatch):
    """The GCS journal's clock, which each wait between tries advances at once,
    so that a test of a long storage outage takes no real time."""
    from llenvs.inference.persistence.storage import gcs

    class Clock:
        def __init__(self):
            self.now, self.waits, self.lock = 0.0, [], threading.Lock()

        def time(self):
            with self.lock:
                return self.now

        def sleep(self, seconds):
            with self.lock:
                self.waits.append(seconds)
                self.now += seconds

    clock = Clock()
    monkeypatch.setattr(gcs, "_clock", clock.time)
    monkeypatch.setattr(gcs, "_sleep", clock.sleep)
    return clock


@pytest.fixture
def cloud():
    return Cloud()


@pytest.fixture
def gcs_root(cloud, tmp_path):
    # Import the proposed API first so an absent implementation is red, rather
    # than hidden by the optional SDK dependency in an ordinary environment.
    from llenvs.inference.persistence.storage.gcs import GCSJournal

    pytest.importorskip("google.cloud.storage")

    def root(*, max_concurrency=128):
        return GCSJournal(
            bucket="synthetic-bucket",
            prefix="journal-fixture",
            client=cloud.client(),
            cache_dir=tmp_path / "disposable-cache",
            max_concurrency=max_concurrency,
        )

    return root


@pytest.fixture
def cloud_server(gcs_root):
    # A local service outlives the killed journal worker. It contains only
    # synthetic records; its authentication marker is not a user credential.
    with CloudManager(
        address=("127.0.0.1", 0),
        authkey=b"synthetic-storage-test",
        ctx=get_context("spawn"),
    ) as manager:
        yield manager.address, manager.storage_cloud()


@pytest.fixture(params=["filesystem", "gcs"])
def stores(request, tmp_path):
    location = (
        (lambda: tmp_path / "calls")
        if request.param == "filesystem"
        else request.getfixturevalue("gcs_root")
    )
    opened = []

    def open_store(*, descriptor=None, validation="lazy"):
        store = CallStore.open(
            location(),
            run_descriptor={"experiment": "storage-contract"} if descriptor is None else descriptor,
            validation=validation,
        )
        opened.append(store)
        return store

    yield open_store
    for store in reversed(opened):
        store.close()


@pytest.fixture
def gcs_stores(gcs_root, cloud):
    opened = []

    def open_store(*, descriptor=None, validation="lazy"):
        store = CallStore.open(
            gcs_root(),
            run_descriptor={"experiment": "storage-contract"} if descriptor is None else descriptor,
            validation=validation,
        )
        opened.append(store)
        return store

    yield open_store
    cloud.fault = None
    for store in reversed(opened):
        try:
            store.close()
        except CheckpointError:
            # Fault tests deliberately poison a handle or displace its owner.
            # Dedicated tests check normal close and stale-owner release.
            pass


@pytest.fixture
def draw():
    class Backend:
        capabilities = BackendCapabilities(supports_request_progress=True)
        calls = 0
        failure = None
        before_receipt = None
        before_completion = None

        def descriptor(self):
            return {"identity": {"model": "offline-fixture"}, "provenance": {}}

        def generate_chat_batch(self, batch, params, *, observer):
            results = []
            for i, messages in enumerate(batch):
                observer.started(i, 0)
                self.calls += 1
                receipt = {"text": messages[0].content, "draw": self.calls}
                if self.before_receipt:
                    self.before_receipt()
                observer.received(i, receipt)
                if self.failure:
                    raise self.failure
                result = self.reparse_receipt(receipt, params)
                if self.before_completion:
                    self.before_completion()
                observer.completed(i, result)
                results.append(result)
            return results

        def reparse_receipt(self, receipt, params):
            return GenerationResult(text=receipt["text"], metadata={"draw": receipt["draw"]})

    return Backend()


@pytest.fixture
def request_record():
    return {
        "backend": {"model": "offline-fixture"},
        "scope": "default",
        "operation": "chat",
        "arguments": {"input": "synthetic request"},
    }
