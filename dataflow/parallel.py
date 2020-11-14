from typing import *

from . import Node
import asyncio
import os
import inspect
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor


class ConcurrentMapData(Node):
    def __init__(self, ds, num_concurrent=1, map_func=None, executor_type=None, buffer_factor=4, none_on_exception=False, initializer=None, initargs=()):
        if num_concurrent == 1:
            _logger.warn("You need num_concurrent > 1 to obtain acceleration.")
        assert num_concurrent > 0
        if num_concurrent >= os.cpu_count():
            _logger.warn("num_concurrent is too large")

        assert map_func is not None
        assert buffer_factor > 1

        self._ds = ds
        self._num_concurrent = num_concurrent
        self._fn = map_func
        self._exe_type = executor_type
        self._num_submit = int(buffer_factor * self._num_concurrent)
        self._none_on_exception = none_on_exception

        if self._exe_type == "thread":
            self._exe = ThreadPoolExecutor(max_workers=self._num_concurrent, initializer=initializer, initargs=initargs)
        elif self._exe_type == "process":
            self._exe = ProcessPoolExecutor(max_workers=self._num_concurrent, initializer=initializer, initargs=initargs)
        else:
            raise RuntimeError("Unknown executor type, must be 'thread' or 'process'")
        self._exhausted = False

    def __iter__(self):
        def try_fill_pending(pending: set, iterator):
            while not self._exhausted:
                if len(pending) >= self._num_submit:
                    break
                try:
                    dp = next(iterator)
                    future = self._exe.submit(self._fn, dp)
                    pending.add(future)
                except StopIteration:
                    self._exhausted = True
                    break

        done = set()
        pending = set()
        it = iter(self._ds)
        while True:
            try_fill_pending(pending, it)
            if len(pending) == 0:
                break  # raise StopIteration
            else:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for d in done:
                    if d is not None:
                        try:
                            yield d.result()
                        except Exception as e:
                            _logger.error(repr(e))
                            if self._none_on_exception:
                                yield None


class AsyncMapData(Node):

    """AsyncMapData is used for IO only"""

    def __init__(self, ds, map_func, num_async=128):
        assert map_func is not None, "map_func must be specified"
        self._ds = ds
        self._fn = map_func
        self._is_async = inspect.iscoroutinefunction(self._fn)
        self._num_async = num_async

        self._loop = asyncio.get_event_loop()
        self._exe = ThreadPoolExecutor(max_workers=512)
        # self._exe = None # use the default executor
        self._exhausted = False

    def __iter__(self):
        def try_fill_pending(pending: set, iterator):
            while not self._exhausted:
                if len(pending) >= self._num_async:
                    break
                try:
                    dp = next(iterator)
                    if self._is_async:
                        future = self._fn(dp)
                    else:
                        future = self._loop.run_in_executor(self._exe, self._fn, dp)
                    pending.add(asyncio.ensure_future(future))
                except StopIteration:
                    self._exhausted = True
                    break

        done = set()
        pending = set()
        it = iter(self._ds)
        while True:
            try_fill_pending(pending, it)
            if len(pending) == 0:
                break  # raise StopIteration
            else:
                done, pending = self._loop.run_until_complete(asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED))
                for d in done:
                    if d is not None:
                        yield d.result()

    def reset_state(self):
        super().reset_state()
        self._exhausted = False
        self._ds.reset_state()
