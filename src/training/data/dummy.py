"""Random-tokens dataloader for smoke tests (skips streaming/tokenize)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def dummy_dataloader(batch_size: int, seq_len: int, vocab_size: int) -> DataLoader:
    """A dataloader that yields random-token batches.

    Cheap smoke test for the model + optimizer + TP code path. Each
    rank builds its own (intentionally not routed through
    :class:`PrefetchBatcher`); batches are random per rank, which is
    fine for a smoke test of the sharded forward/backward path even
    though it doesn't strictly satisfy the TP input-replication
    contract.
    """
    class _DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size: int = 1000) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int) -> dict:
            input_ids = torch.randint(0, vocab_size, (seq_len,))
            return {"input_ids": input_ids, "labels": input_ids.clone()}

    return DataLoader(_DummyDataset(), batch_size=batch_size, shuffle=True)
