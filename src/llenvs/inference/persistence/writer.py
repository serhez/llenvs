"""Bounded journal admission with independent durable acknowledgements."""

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial, wraps


class MaintenanceGate:
    """Concurrent operations; maintenance waits for them and runs alone.

    Readers may start while maintenance waits: an existing reader can be
    waiting for work on another thread. Blocking that work would deadlock.
    """

    def __init__(self):
        self.condition = threading.Condition()
        self.active = 0
        self.owner = None

    @contextmanager
    def shared(self):
        with self.condition:
            self.condition.wait_for(lambda: self.owner in {None, threading.get_ident()})
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    @contextmanager
    def exclusive(self):
        with self.condition:
            self.condition.wait_for(lambda: self.owner is None and not self.active)
            self.owner = threading.get_ident()
        try:
            yield
        finally:
            with self.condition:
                self.owner = None
                self.condition.notify_all()

    def track(self, future):
        """Retain an asynchronous operation beyond its admitting shared scope."""
        with self.condition:
            self.active += 1

        def finished(_):
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

        future.add_done_callback(finished)


def operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._gate.shared():
            return method(self, *args, **kwargs)

    return guarded


class GroupWriter:
    def __init__(self, storage, failed):
        self.storage, self.failed = storage, failed
        self.capacity = storage.max_concurrency * 2 if storage.max_concurrency > 1 else 1
        self._condition = threading.Condition()
        self._pending = deque()
        self._finished = deque()
        self._admitted = deque()
        self._error = self._poisoned = None
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="llenvs-group-commit", daemon=True)
        self.thread.start()

    @property
    def error(self):
        with self._condition:
            return self._error

    def fail(self, error):
        """Freeze admission; the coordinator publishes admitted work before failing waiters."""
        with self._condition:
            self._error = self._error or error
            self._condition.notify_all()

    def submit(self, name, payload, committed):
        future = Future()
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._poisoned or len(self._pending) < self.capacity
            )
            if self._closed:
                raise RuntimeError("Journal writer is closed")
            if self._poisoned is not None:
                future.set_exception(self._poisoned)
            else:
                self._pending.append((name, payload, committed, future))
                self._condition.notify_all()
        return future

    def _admit(self, first, limit):
        group = [first]

        def records():
            with self._condition:
                if self._error is not None:
                    return
            yield first[0], first[1]
            # Ownership is checked before consumption. Reserve a bounded share
            # of the window, including arrivals during the owner read.
            while len(group) < limit:
                with self._condition:
                    if self._error is not None or not self._pending:
                        return
                    following = self._pending.popleft()
                    self._condition.notify_all()
                group.append(following)
                yield following[0], following[1]

        yielded, error = 0, None
        try:
            for future in self.storage.submit_writes(records()):
                item = group[yielded]
                yielded += 1
                future.add_done_callback(partial(self._finished_write, item))
        except BaseException as exc:
            error = exc
        with self._condition:
            if yielded < len(group):
                error = error or self._error or RuntimeError("Storage did not accept a record")
            if error is not None:
                self._error = self._error or error
            # A failed admission must not replace already yielded futures with
            # the admission error: those writes can still commit successfully.
            self._finished.extend((item, error) for item in group[yielded:])
            self._admitted.append(limit - len(group))
            self._condition.notify_all()

    def _finished_write(self, item, future):
        error = None
        try:
            future.result()
        except BaseException as exc:
            error = exc
        with self._condition:
            if error is not None:
                self._error = self._error or error
            self._finished.append((item, error))
            self._condition.notify_all()

    @staticmethod
    def _acknowledge(item, error):
        future = item[3]
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    def _run(self):
        active, admitting, held = 0, False, []
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llenvs-write-admission"
        ) as workers:
            while True:
                drained_error = None
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._finished
                            or self._admitted
                            or (
                                self._error is not None
                                and not active
                                and not admitting
                                and self._poisoned is None
                            )
                            or (
                                self._pending
                                and (
                                    self._poisoned is not None
                                    or (
                                        self._error is None
                                        and not admitting
                                        and active < self.storage.max_concurrency
                                    )
                                )
                            )
                            or (self._closed and not active and not admitting)
                        )
                    )
                    if self._finished:
                        item, error = self._finished.popleft()
                        active -= 1
                    elif self._admitted:
                        active -= self._admitted.popleft()
                        admitting = False
                        self._condition.notify_all()
                        continue
                    elif (
                        self._error is not None
                        and not active
                        and not admitting
                        and self._poisoned is None
                    ):
                        # All admitted writes have drained and every confirmed
                        # callback has run. Poisoning cannot hide sibling results.
                        self._poisoned = self._error
                        drained_error = self._poisoned
                    elif self._pending and self._poisoned is not None:
                        while self._pending:
                            self._acknowledge(self._pending.popleft(), self._poisoned)
                        self._condition.notify_all()
                        continue
                    elif (
                        self._pending
                        and self._error is None
                        and not admitting
                        and active < self.storage.max_concurrency
                    ):
                        first = self._pending.popleft()
                        limit = self.storage.max_concurrency - active
                        active += limit
                        admitting = True
                        workers.submit(self._admit, first, limit)
                        self._condition.notify_all()
                        continue
                    elif self._closed and not active and not admitting:
                        return
                if drained_error is not None:
                    # Store callbacks take the metadata lock. Never hold the
                    # coordinator lock across them: callers may check our error
                    # while already holding the store's lock.
                    self.failed(drained_error)
                    with self._condition:
                        for item, error in held:
                            self._acknowledge(item, error)
                        held.clear()
                        self._condition.notify_all()
                    continue
                if error is None:
                    try:
                        item[2]()
                    except BaseException as exc:
                        error = exc
                with self._condition:
                    if error is not None:
                        self._error = self._error or error
                    if self._error is None:
                        self._acknowledge(item, error)
                    else:
                        held.append((item, error))

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.thread.join()
