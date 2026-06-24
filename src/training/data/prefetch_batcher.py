"""Owner-rank background prefetch thread that fans batches into N queues.

Each "batch" is produced by chunk-aligned FFD packing:
:func:`pack_chunk_aligned` consumes ``batch_size * pack_buffer_size``
input docs and emits ``batch_size`` packed rows plus a global
``cu_seqlens`` tensor. The prefetcher threads the entire packed
batch dict through the per-rank queues so every TP rank consumes
identical inputs (input-replication contract).
"""
from __future__ import annotations

import logging
import threading
import time

import torch

from .collate import pack_chunk_aligned


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

    def __init__(
        self,
        dataset,
        batch_size: int,
        queues,
        *,
        seq_len: int,
        chunk_size: int,
        pad_id: int = 0,
        eos_id: int | None = None,
        pack_buffer_size: int = 8,
    ) -> None:
        """
        Args:
            dataset: an iterable yielding raw tokenized samples
                (``{"input_ids": [...int], "labels": [...int]}``).
            batch_size: number of packed rows in each output batch.
            queues: list of queue-likes (``mp.Queue`` cross-process,
                ``queue.Queue`` in-process), one per consumer rank
                **including the owner's own queue**. The worker
                broadcasts each batch into every queue here.
            seq_len: target packed sequence length. Must be a
                multiple of ``chunk_size``.
            chunk_size: alignment granularity for doc boundaries
                inside a pack (see :func:`pack_chunk_aligned`).
            pad_id: padding token id.
            eos_id: optional EOS token id. When set, each doc is
                forced to end with this id before packing.
            pack_buffer_size: how many input docs to pull per
                packing window. Higher = denser packs at the cost
                of one window's latency. The packer produces
                ``batch_size`` rows from this buffer; oversized
                windows (more docs than fit) only emit a single
                batch and discard the rest on the next pull.
        """
        if not queues:
            raise ValueError("PrefetchBatcher: queues must be non-empty")
        if pack_buffer_size < 1:
            raise ValueError(
                f"PrefetchBatcher: pack_buffer_size must be >= 1, got"
                f" {pack_buffer_size}"
            )
        self._batch_size = batch_size
        self._pack_buffer_size = pack_buffer_size
        self._seq_len = seq_len
        self._chunk_size = chunk_size
        self._pad_id = pad_id
        self._eos_id = eos_id
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
        """Pull up to ``batch_size * pack_buffer_size`` samples from
        the dataset and pack them into a single batch dict.

        Returns the packed batch dict, or :attr:`_SENTINEL` if the
        dataset iterator is exhausted before any sample was read.
        The packed batch has shape::

            {
              "input_ids":  [B, seq_len]    long,
              "labels":     [B, seq_len]    long  (-100 at pad),
              "cu_seqlens": [total_docs+1]  long,
            }

        The "total_docs" count can vary across batches (FFD output
        is data-dependent); the model is responsible for handling
        a per-batch variable-length ``cu_seqlens``.
        """
        target = self._batch_size * self._pack_buffer_size
        batch: list[dict] = []
        for _ in range(target):
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

        # Pull the raw id lists out of each sample dict. The
        # tokenizer yields a 1D list of ints (the streaming
        # dataset's ``return_tensors=None`` path); older 1D
        # tensors are handled by ``.tolist()``.
        token_lists: list[list[int]] = []
        for s in batch:
            ids = s["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().tolist()
            else:
                ids = list(ids)
            token_lists.append(ids)

        input_ids, labels, cu_seqlens = pack_chunk_aligned(
            token_lists,
            seq_len=self._seq_len,
            chunk_size=self._chunk_size,
            batch_size=self._batch_size,
            pad_id=self._pad_id,
            eos_id=self._eos_id,
        )
        # Pin input_ids and labels so the consumer's
        # ``.to('cuda', non_blocking=True)`` can actually overlap
        # with the next forward pass. ``cu_seqlens`` is small
        # enough that pinning is not worth the alloc cost.
        if torch.cuda.is_available():
            input_ids = input_ids.pin_memory()
            labels = labels.pin_memory()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "cu_seqlens": cu_seqlens,
        }

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