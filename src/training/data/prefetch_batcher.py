"""Owner-rank background prefetch thread that fans batches into N queues."""
from __future__ import annotations

import logging
import threading
import time

from .collate import collate_batch


class QueueIterator:
    """Iterate a queue (``multiprocessing.Queue`` or ``queue.Queue``)
    until a ``None`` sentinel, re-raising any exception object the
    producer put in our place.

    This is the **single read path** for every TP rank — the owner's
    prefetch worker pushes each batch into every rank's queue, and
    every rank (including the owner) pulls from its own queue with
    one of these iterators. No more owner / non-owner branching at
    the consumer side.
    """

    # ``None`` is a singleton in CPython and survives pickle round-
    # trips, so it works as the end-of-stream marker for both
    # in-process ``queue.Queue`` and cross-process ``mp.Queue``.
    _SENTINEL = None

    def __init__(self, q) -> None:
        self._q = q

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item


class PrefetchBatcher:
    """One-deep async prefetch fanning out into per-rank queues.

    Constructed **only on the TP owner rank**. Spawns one daemon
    worker that loops::

        fetch batch  ->  for q in queues: q.put(batch)  ->  repeat

    The queue ``maxsize=2`` caps in-flight batches at 2 per rank.
    When any consumer is slow, the worker blocks on that consumer's
    ``queue.put`` — correct backpressure for TP. On end-of-stream a
    ``None`` sentinel is broadcast to every queue; on exception the
    exception object itself is broadcast (each consumer's
    :class:`QueueIterator` re-raises it).

    There is no owner-side fast path: the owner reads from its own
    queue via :class:`QueueIterator` exactly like every non-owner
    rank, so the consumer code is uniform across ranks.
    """

    _SENTINEL = None

    def __init__(self, dataset, batch_size: int, queues) -> None:
        """
        Args:
            dataset: an iterable yielding raw samples.
            batch_size: number of samples to collate per output batch.
            queues: list of queue-likes (``mp.Queue`` cross-process,
                ``queue.Queue`` in-process), one per consumer rank
                **including the owner's own queue**. The worker
                broadcasts each batch into every queue here.
        """
        if not queues:
            raise ValueError("PrefetchBatcher: queues must be non-empty")
        self._batch_size = batch_size
        self._queues = list(queues)
        self._iter = iter(dataset)
        self._stop = threading.Event()
        self._log = logging.getLogger(__name__)
        self._t0 = time.monotonic()
        self._first_sample_logged = False
        self._worker = threading.Thread(
            target=self._worker_loop, name="hf-prefetch", daemon=True,
        )
        self._worker.start()

    def _fetch_one(self):
        """Pull ``batch_size`` samples from the dataset and collate.

        Returns the collated batch dict, or :attr:`_SENTINEL` if the
        dataset iterator is exhausted before any sample was read.
        """
        batch = []
        for _ in range(self._batch_size):
            try:
                batch.append(next(self._iter))
            except StopIteration:
                break
        if not batch:
            return self._SENTINEL
        if not self._first_sample_logged:
            self._log.info(
                f"prefetch: first sample after"
                f" {time.monotonic() - self._t0:.2f}s"
            )
            self._first_sample_logged = True
        return collate_batch(batch)

    def _broadcast(self, item) -> None:
        """Put ``item`` into every consumer queue. Blocking — slow
        consumers gate the worker, which is the backpressure we want."""
        for q in self._queues:
            q.put(item)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._fetch_one()
            except BaseException as e:  # noqa: BLE001 — propagate
                self._broadcast(e)
                return
            self._broadcast(result)
            if result is self._SENTINEL:
                return  # end of stream; consumers will raise StopIteration

    def close(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait briefly for it to exit.

        Worker is a daemon thread, so a stuck ``queue.put`` (e.g.
        consumer abandoned mid-stream) will be killed at interpreter
        exit even if the join times out.
        """
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
