"""Bounded groups of ordered journal writes with individual acknowledgements."""

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps


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
        self._groups = 2 if storage.max_concurrency > 1 else 1
        self._condition = threading.Condition()
        self._pending = deque()
        self._finished = deque()
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

    def _write(self, first):
        group = [first]

        def records():
            yield first[0], first[1]
            # Ownership is checked before this iterator is consumed. Include
            # arrivals during that read, unless another group has failed.
            while len(group) < self.storage.max_concurrency:
                with self._condition:
                    if self._error is not None or not self._pending:
                        return
                    following = self._pending.popleft()
                    self._condition.notify_all()
                group.append(following)
                yield following[0], following[1]

        try:
            errors = self.storage.write_many(records())
        except BaseException as exc:
            errors = [exc] * len(group)
        with self._condition:
            error = next((error for error in errors if error is not None), None)
            if error is not None:
                self._error = self._error or error
            self._finished.append((group, errors))
            self._condition.notify_all()

    @staticmethod
    def _acknowledge(group, errors):
        for (_, _, _, future), error in zip(group, errors, strict=True):
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)

    def _run(self):
        active, held = 0, []
        with ThreadPoolExecutor(
            max_workers=self._groups, thread_name_prefix="llenvs-write-group"
        ) as workers:
            while True:
                drained_error = None
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._finished
                            or (self._error is not None and not active and self._poisoned is None)
                            or (
                                self._pending
                                and (
                                    self._poisoned is not None
                                    or (self._error is None and active < self._groups)
                                )
                            )
                            or (self._closed and not active)
                        )
                    )
                    if self._finished:
                        group, errors = self._finished.popleft()
                        active -= 1
                    elif self._error is not None and not active and self._poisoned is None:
                        # All admitted writes have drained and every confirmed
                        # callback has run. Poisoning cannot hide sibling results.
                        self._poisoned = self._error
                        drained_error = self._poisoned
                    elif self._pending and self._poisoned is not None:
                        group = list(self._pending)
                        self._pending.clear()
                        self._acknowledge(group, [self._poisoned] * len(group))
                        self._condition.notify_all()
                        continue
                    elif self._pending and self._error is None and active < self._groups:
                        first = self._pending.popleft()
                        active += 1
                        workers.submit(self._write, first)
                        self._condition.notify_all()
                        continue
                    elif self._closed and not active:
                        return
                if drained_error is not None:
                    # Store callbacks take the metadata lock. Never hold the
                    # coordinator lock across them: callers may check our error
                    # while already holding the store's lock.
                    self.failed(drained_error)
                    with self._condition:
                        for group, errors in held:
                            self._acknowledge(group, errors)
                        held.clear()
                        self._condition.notify_all()
                    continue
                for index, (_, _, committed, _) in enumerate(group):
                    if errors[index] is None:
                        try:
                            committed()
                        except BaseException as exc:
                            errors[index] = exc
                with self._condition:
                    error = next((error for error in errors if error is not None), None)
                    if error is not None:
                        self._error = self._error or error
                    if self._error is None:
                        self._acknowledge(group, errors)
                    else:
                        held.append((group, errors))

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.thread.join()
