"""Durable records advance independently inside a bounded write window."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import pytest

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
        storage.wait_for("first")

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
