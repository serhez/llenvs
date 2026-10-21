"""Bounded groups of ordered journal writes with individual acknowledgements."""

import queue
import threading
from concurrent.futures import Future
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
        self.queue = queue.Queue(self.capacity)
        self._guard = threading.Lock()
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="llenvs-group-commit", daemon=True)
        self.thread.start()

    def submit(self, name, payload, committed):
        future = Future()
        with self._guard:
            if self._closed:
                raise RuntimeError("Journal writer is closed")
            self.queue.put((name, payload, committed, future))
        return future

    def _run(self):
        poisoned = None
        while True:
            item = self.queue.get()
            if item is None:
                return
            group, stopping = [item], False

            def records():
                nonlocal stopping
                yield item[0], item[1]
                # The cloud storage checks ownership before consuming these
                # records. Include arrivals during that round trip in the group.
                while len(group) < self.capacity:
                    try:
                        following = self.queue.get_nowait()
                    except queue.Empty:
                        return
                    if following is None:
                        stopping = True
                        return
                    group.append(following)
                    yield following[0], following[1]

            try:
                errors = (
                    [poisoned] * len(group)
                    if poisoned is not None
                    else self.storage.write_many(records())
                )
            except BaseException as exc:
                errors = [exc] * len(group)
            # Publish every confirmed write before waking failed callers. Batch
            # collection can then account for a partially committed group.
            for index, (_, _, committed, _) in enumerate(group):
                if errors[index] is None:
                    try:
                        committed()
                    except BaseException as exc:
                        errors[index] = exc
            error = next((error for error in errors if error is not None), None)
            if error is not None:
                poisoned = error
                self.failed(error)
            for (_, _, _, future), error in zip(group, errors, strict=True):
                if error is None:
                    future.set_result(None)
                else:
                    future.set_exception(error)
            if stopping:
                return

    def close(self):
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self.queue.put(None)
        self.thread.join()
