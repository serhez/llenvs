"""Durable records advance independently inside a bounded write window."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from llenvs.inference.persistence import writer as writer_module
from llenvs.inference.persistence.writer import GroupWriter


class ControlledStorage:
    """Manually completed native writes; no clock or network determines progress."""

    def __init__(self, concurrency):
        self.max_concurrency = concurrency
        self.condition = threading.Condition()
        self.accepted = {}
        self.admitted = set()
        self.durable = {}
        self.checks = 0
        self.before_admission = lambda index: None
        self.admission_error = None
        self.cleaning_up = False

    def submit_writes(self, records):
        with self.condition:
            self.checks += 1
            index = self.checks
        self.before_admission(index)
        cohort = []
        try:
            for name, payload in records:
                future = Future()
                with self.condition:
                    assert name not in self.accepted
                    self.accepted[name] = (payload, future)
                    cohort.append(name)
                    cleanup = self.cleaning_up
                    self.condition.notify_all()
                if cleanup:
                    self.finish(name)
                yield future
                if self.admission_error is not None:
                    raise self.admission_error
        finally:
            with self.condition:
                self.admitted.update(cohort)
                self.condition.notify_all()

    def wait_for(self, *names):
        with self.condition:
            assert self.condition.wait_for(lambda: set(names) <= self.accepted.keys(), 2), (
                f"Writes were not admitted: {set(names) - self.accepted.keys()}"
            )

    def wait_for_admission(self, *names):
        with self.condition:
            assert self.condition.wait_for(lambda: set(names) <= self.admitted, 2)

    def finish(self, name, error=None):
        with self.condition:
            payload, future = self.accepted[name]
            if future.done():
                return
            if error is None:
                self.durable[name] = payload
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    def drain(self):
        with self.condition:
            self.cleaning_up = True
            names = list(self.accepted)
        for name in names:
            self.finish(name)


@contextmanager
def controlled_writer(concurrency=4):
    storage = ControlledStorage(concurrency)
    published, failures = [], []
    writer = GroupWriter(storage, lambda error: failures.append((error, list(published))))

    def submit(name, committed=None):
        def publish():
            assert storage.durable[name] == name.encode(), "Acknowledged before durability"
            if committed is not None:
                committed()
            published.append(name)

        return writer.submit(name, name.encode(), publish)

    try:
        yield storage, writer, submit, published, failures
    finally:
        storage.drain()
        writer.close()


def wait_for_error(writer, error):
    with writer._condition:
        assert writer._condition.wait_for(lambda: writer.error is error, 2)


def test_fast_record_acknowledgement_does_not_wait_for_its_cohort_straggler():
    entered, release, release_second = threading.Event(), threading.Event(), threading.Event()
    with controlled_writer() as (storage, writer, submit, published, failures):

        def check(index):
            if index == 1:
                entered.set()
                assert release.wait(5)
            elif index == 2:
                assert release_second.wait(5)

        storage.before_admission = check
        slow = submit("slow")
        try:
            assert entered.wait(2)
            submit("other")
            fast = submit("fast")
            release.set()
            storage.wait_for("slow", "fast")
            storage.finish("fast")
            fast.result(timeout=1)
            assert published == ["fast"] and not slow.done() and not failures
        finally:
            release.set()
            release_second.set()


def test_two_stragglers_do_not_stop_a_stream_of_independent_records():
    with controlled_writer() as (storage, writer, submit, published, failures):
        slow = [submit("slow-0")]
        storage.wait_for_admission("slow-0")
        slow.append(submit("slow-1"))
        storage.wait_for_admission("slow-1")
        for index in range(6):
            name = f"fast-{index}"
            fast = submit(name)
            storage.wait_for(name)
            storage.finish(name)
            fast.result(timeout=1)
        assert published == [f"fast-{index}" for index in range(6)]
        assert all(not future.done() for future in slow) and not failures
        assert storage.checks >= 3, "Successor cohorts did not check ownership"


@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_rolling_admission_never_exceeds_the_native_write_window(concurrency):
    entered, release = threading.Event(), threading.Event()
    with controlled_writer(concurrency) as (storage, writer, submit, published, failures):

        def check(index):
            entered.set()
            assert release.wait(5)

        storage.before_admission = check
        acknowledgements = [submit("record-0")]
        try:
            assert entered.wait(2)
            acknowledgements += [submit(f"record-{index}") for index in range(1, concurrency + 1)]
            release.set()
            storage.wait_for(*(f"record-{index}" for index in range(concurrency)))
            with storage.condition:
                assert not storage.condition.wait_for(
                    lambda: f"record-{concurrency}" in storage.accepted, 0.05
                ), "More records were admitted than the configured I/O window"
            storage.finish("record-0")
            acknowledgements[0].result(timeout=1)
            storage.wait_for(f"record-{concurrency}")
        finally:
            release.set()


def test_slow_successor_owner_read_does_not_block_a_durable_acknowledgement():
    entered, release = threading.Event(), threading.Event()
    with controlled_writer() as (storage, writer, submit, published, failures):
        first = submit("first")
        storage.wait_for_admission("first")

        def check(index):
            entered.set()
            assert release.wait(5)

        storage.before_admission = check
        second = submit("second")
        try:
            assert entered.wait(2)
            storage.finish("first")
            first.result(timeout=1)
            assert published == ["first"] and not second.done()
        finally:
            release.set()


@pytest.mark.parametrize("failure_kind", ["native", "callback", "external"])
def test_failure_freezes_admission_and_drains_confirmed_records_before_poison(failure_kind):
    error = OSError("Synthetic fatal journal operation")
    with controlled_writer(2) as (storage, writer, submit, published, failures):

        def fail_callback():
            raise error

        bad = submit("bad", fail_callback if failure_kind == "callback" else None)
        storage.wait_for("bad")
        good = submit("good")
        storage.wait_for("good")
        queued = submit("queued")
        if failure_kind == "external":
            writer.fail(error)
        else:
            storage.finish("bad", error if failure_kind == "native" else None)
        wait_for_error(writer, error)
        assert not failures and not bad.done() and not queued.done()
        if failure_kind == "external":
            storage.finish("bad")
        storage.finish("good")
        good.result(timeout=2)
        if failure_kind == "external":
            bad.result(timeout=2)
        else:
            with pytest.raises(OSError) as caught:
                bad.result(timeout=2)
            assert caught.value is error
        with pytest.raises(OSError) as caught:
            queued.result(timeout=2)
        assert caught.value is error and "queued" not in storage.accepted
        assert failures == [(error, published)] and "good" in published


def test_an_already_returned_record_survives_a_later_sibling_failure():
    error = OSError("Synthetic delayed failed upload")
    with controlled_writer() as (storage, writer, submit, published, failures):
        slow = submit("slow")
        storage.wait_for_admission("slow")
        fast = submit("fast")
        storage.wait_for("fast")
        storage.finish("fast")
        fast.result(timeout=1)
        storage.finish("slow", error)
        with pytest.raises(OSError) as caught:
            slow.result(timeout=2)
        assert caught.value is error
        assert fast.result() is None and published == ["fast"]
        assert failures == [(error, ["fast"])]


def test_a_failed_record_freezes_admission_while_its_cohort_straggler_is_still_held():
    entered, release, release_second = threading.Event(), threading.Event(), threading.Event()
    error = OSError("Synthetic failure beside an unfinished cohort member")
    with controlled_writer() as (storage, writer, submit, published, failures):

        def check(index):
            if index == 1:
                entered.set()
                assert release.wait(5)
            elif index == 2:
                assert release_second.wait(5)

        storage.before_admission = check
        slow = submit("slow")
        try:
            assert entered.wait(2)
            submit("other")
            bad = submit("bad")
            release.set()
            storage.wait_for("slow", "bad")
            storage.finish("bad", error)
            wait_for_error(writer, error)
            queued = submit("queued")
            assert not slow.done() and not bad.done() and not failures
            assert "queued" not in storage.accepted
            release_second.set()
            storage.drain()
            slow.result(timeout=2)
            with pytest.raises(OSError) as caught:
                bad.result(timeout=2)
            assert caught.value is error
            with pytest.raises(OSError) as caught:
                queued.result(timeout=2)
            assert caught.value is error and "slow" in published
            assert failures == [(error, published)]
        finally:
            release.set()
            release_second.set()


def test_partial_admission_failure_keeps_the_accepted_future_and_its_confirmed_outcome():
    error = OSError("Synthetic admission failure after yielding a native future")
    with controlled_writer() as (storage, writer, submit, published, failures):
        storage.admission_error = error
        accepted = submit("accepted")
        storage.wait_for("accepted")
        wait_for_error(writer, error)
        queued = submit("queued")
        assert not failures and not accepted.done()
        storage.finish("accepted")
        accepted.result(timeout=2)
        with pytest.raises(OSError) as caught:
            queued.result(timeout=2)
        assert caught.value is error
        assert published == ["accepted"] and failures == [(error, ["accepted"])]
        assert set(storage.accepted) == {"accepted"}


def test_commit_callbacks_do_not_hold_the_admission_lock():
    with (
        controlled_writer() as (storage, writer, submit, published, failures),
        ThreadPoolExecutor(max_workers=1) as worker,
    ):
        acknowledgement = submit(
            "first", lambda: worker.submit(lambda: writer.error).result(timeout=1)
        )
        storage.wait_for("first")
        storage.finish("first")
        acknowledgement.result(timeout=2)
        assert published == ["first"] and not failures


def test_close_drains_the_window_and_pending_records():
    with (
        controlled_writer(2) as (storage, writer, submit, published, failures),
        ThreadPoolExecutor(max_workers=1) as worker,
    ):
        first = submit("first")
        storage.wait_for("first")
        second = submit("second")
        storage.wait_for("second")
        third = submit("third")
        closing = worker.submit(writer.close)
        try:
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.05)
            storage.finish("first")
            first.result(timeout=1)
            storage.wait_for("third")
            storage.finish("third")
            third.result(timeout=1)
            assert not closing.done() and not second.done()
            storage.finish("second")
            second.result(timeout=1)
            closing.result(timeout=2)
            assert set(published) == {"first", "second", "third"} and not failures
        finally:
            storage.drain()


def test_pending_payload_queue_applies_backpressure_until_a_window_slot_is_released():
    with (
        controlled_writer(2) as (storage, writer, submit, published, failures),
        ThreadPoolExecutor(max_workers=1) as worker,
    ):
        submit("active-0")
        storage.wait_for("active-0")
        submit("active-1")
        storage.wait_for("active-1")
        for index in range(writer.capacity):
            submit(f"pending-{index}")
        blocked = worker.submit(submit, "overflow")
        try:
            with pytest.raises(TimeoutError):
                blocked.result(timeout=0.05)
            storage.finish("active-0")
            assert isinstance(blocked.result(timeout=2), Future)
        finally:
            storage.drain()


@pytest.fixture
def admission_clock(monkeypatch):
    """Control the admission deadline independently of thread scheduling."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        writer_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now),
        raising=False,
    )

    def advance(writer, seconds):
        with writer._condition:
            clock.now += seconds
            writer._condition.notify_all()

    clock.advance = advance
    return clock


def wait_for_open_cohort(storage, name):
    storage.wait_for(name)
    with storage.condition:
        assert not storage.condition.wait_for(lambda: name in storage.admitted, 0.02), (
            "An unfinished cohort closed as soon as its input queue became empty"
        )


@pytest.mark.parametrize("concurrency", [2, 4, 128])
def test_later_arrivals_use_the_active_bounded_owner_check(admission_clock, concurrency):
    with controlled_writer(concurrency) as (storage, writer, submit, published, failures):
        first = submit("first")
        wait_for_open_cohort(storage, "first")
        later = submit("later")
        storage.wait_for("later")
        assert storage.checks == 1, "A later arrival paid another owner-read round trip"
        assert not first.done() and not later.done() and not published
        storage.finish("later")
        later.result(timeout=1)
        assert published == ["later"] and not first.done() and not failures


def test_checked_cohort_cannot_accept_after_its_100ms_deadline(admission_clock):
    with controlled_writer() as (storage, writer, submit, published, failures):
        submit("first")
        wait_for_open_cohort(storage, "first")
        admission_clock.advance(writer, 0.099)
        submit("before-deadline")
        storage.wait_for("before-deadline")
        assert storage.checks == 1
        admission_clock.advance(writer, 0.002)
        submit("after-deadline")
        storage.wait_for("after-deadline")
        assert storage.checks == 2, "An expired cohort reused its owner observation"
        assert not published and not failures


@pytest.mark.parametrize("finished", ["first", "later"])
def test_first_native_completion_ends_cohort_without_delaying_its_ack(admission_clock, finished):
    with controlled_writer() as (storage, writer, submit, published, failures):
        acknowledgements = {"first": submit("first")}
        wait_for_open_cohort(storage, "first")
        acknowledgements["later"] = submit("later")
        storage.wait_for("later")
        assert storage.checks == 1
        storage.finish(finished)
        acknowledgements[finished].result(timeout=1)
        storage.wait_for_admission("first", "later")
        submit("successor")
        storage.wait_for("successor")
        assert storage.checks == 2 and published == [finished] and not failures
        other = "later" if finished == "first" else "first"
        assert not acknowledgements[other].done()


@pytest.mark.parametrize("concurrency", [2, 4])
def test_open_cohort_keeps_the_write_window_and_queue_limits(admission_clock, concurrency):
    with controlled_writer(concurrency) as (storage, writer, submit, published, failures):
        first = submit("record-0")
        wait_for_open_cohort(storage, "record-0")
        for index in range(1, concurrency):
            submit(f"record-{index}")
            storage.wait_for(f"record-{index}")
        assert storage.checks == 1
        queued = submit("queued")
        with storage.condition:
            assert not storage.condition.wait_for(lambda: "queued" in storage.accepted, 0.02)
        storage.finish("record-0")
        first.result(timeout=1)
        storage.wait_for("queued")
        assert storage.checks == 2 and not queued.done() and not failures


@pytest.mark.parametrize("failure_kind", ["native", "callback", "external"])
def test_idle_admission_wakes_on_failure_and_preserves_confirmed_siblings(
    admission_clock, failure_kind
):
    error = OSError("Synthetic failure while a checked cohort awaits arrivals")
    with controlled_writer() as (storage, writer, submit, published, failures):

        def fail_callback():
            raise error

        bad = submit("bad", fail_callback if failure_kind == "callback" else None)
        wait_for_open_cohort(storage, "bad")
        good = submit("good")
        storage.wait_for("good")
        assert storage.checks == 1
        if failure_kind == "external":
            writer.fail(error)
        else:
            storage.finish("bad", error if failure_kind == "native" else None)
        wait_for_error(writer, error)
        queued = submit("queued")
        storage.wait_for_admission("bad", "good")
        assert "queued" not in storage.accepted and not failures
        if failure_kind == "external":
            storage.finish("bad")
        storage.finish("good")
        good.result(timeout=1)
        if failure_kind == "external":
            bad.result(timeout=1)
        else:
            with pytest.raises(OSError) as caught:
                bad.result(timeout=1)
            assert caught.value is error
        with pytest.raises(OSError) as caught:
            queued.result(timeout=1)
        assert caught.value is error
        assert failures == [(error, published)] and "good" in published


def test_expired_checked_cohort_cannot_bypass_a_replacement_owner(admission_clock):
    error = OSError("Synthetic ownership change before successor admission")
    with controlled_writer() as (storage, writer, submit, published, failures):
        first = submit("first")
        wait_for_open_cohort(storage, "first")

        def reject_successor(index):
            assert index == 2
            raise error

        storage.before_admission = reject_successor
        admission_clock.advance(writer, 0.101)
        refused = submit("refused")
        wait_for_error(writer, error)
        assert "refused" not in storage.accepted and not failures and not refused.done()
        storage.finish("first")
        first.result(timeout=1)
        with pytest.raises(OSError) as caught:
            refused.result(timeout=1)
        assert caught.value is error
        assert published == ["first"] and failures == [(error, ["first"])]


def test_close_wakes_an_idle_cohort_but_waits_for_its_native_write(admission_clock):
    with (
        controlled_writer() as (storage, writer, submit, published, failures),
        ThreadPoolExecutor(max_workers=1) as worker,
    ):
        first = submit("first")
        wait_for_open_cohort(storage, "first")
        closing = worker.submit(writer.close)
        storage.wait_for_admission("first")
        assert not closing.done() and not first.done()
        storage.finish("first")
        first.result(timeout=1)
        closing.result(timeout=1)
        assert published == ["first"] and not failures


def test_serial_storage_never_lingers_for_another_record(admission_clock):
    with controlled_writer(1) as (storage, writer, submit, published, failures):
        first = submit("first")
        storage.wait_for_admission("first")
        storage.finish("first")
        first.result(timeout=1)
        second = submit("second")
        storage.wait_for_admission("second")
        storage.finish("second")
        second.result(timeout=1)
        assert storage.checks == 2 and published == ["first", "second"] and not failures


@pytest.mark.parametrize("validation", ["full", "lazy"])
def test_native_gcs_late_arrival_uses_one_check_and_keeps_both_intents_durable(
    gcs_root, cloud, admission_clock, monkeypatch, validation
):
    from llenvs.inference.persistence import CallStore
    from tests.inference.persistence.test_throughput import DelayedIO

    io = DelayedIO(cloud, monkeypatch)
    journal = gcs_root(max_concurrency=4)
    run = {"experiment": "bounded-admission"}
    entered = {index: threading.Event() for index in range(2)}
    release = threading.Event()
    with CallStore.open(journal, run_descriptor=run, validation="lazy") as store:
        keys = [store.reserve({"input": index}, 0)[0] for index in range(2)]
        indices = {key: index for index, key in enumerate(keys)}

        def hold(kind, name, record):
            if kind == "put" and record is not None:
                entered[indices[record["key"]]].set()
                assert release.wait(5)

        io.reset()
        io.before = hold
        with ThreadPoolExecutor(max_workers=2) as worker:
            first = worker.submit(store.update, keys[0], "started")
            try:
                assert entered[0].wait(2)
                second = worker.submit(store.update, keys[1], "started")
                assert entered[1].wait(2)
                assert io.owner_reads == 1
                assert not first.done() and not second.done()
                release.set()
                first.result(timeout=2)
                second.result(timeout=2)
            finally:
                release.set()
                io.before = None
    with CallStore.open(journal, run_descriptor=run, validation=validation) as store:
        assert all(store._records[key]["state"] == "started" for key in keys)
        assert all(not store._records[key]["attempts"] for key in keys)
