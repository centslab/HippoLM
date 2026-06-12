"""Local parquet iteration via pyarrow row groups."""
from __future__ import annotations

from pathlib import Path


class LocalParquetIterable:
    """Yield examples from a local parquet shard via pyarrow row groups.

    The MS streaming reader returns dicts with the same keys as the
    parquet schema (``content`` for text, ``messages`` for SFT). This
    wrapper does the same, so the existing :class:`StreamingDataset`
    text / SFT path works unchanged.

    Iteration is row-group-by-row-group, not row-by-row, so the inner
    loop is fast even on multi-GB files.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def __iter__(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(self._path))
        # ``iter_batches`` reads one row group at a time and yields
        # an Arrow RecordBatch. Converting to pandas then to dicts is
        # cheap for our schema (1-2 text columns) and avoids the
        # per-row Python overhead of ``to_pylist``.
        for batch in pf.iter_batches(batch_size=1024):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield row.to_dict()
