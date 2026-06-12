"""Collate per-sample dicts into a batched dict with right-padding."""
from __future__ import annotations

import torch


def collate_batch(samples: list[dict]) -> dict:
    """Stack a list of per-sample dicts into a batched dict.

    Each sample is ``{"input_ids": [T], "labels": [T]}``. Lengths may
    differ, so we right-pad to the longest sequence in the batch with
    pad id 0; labels are padded with ``-100`` (PyTorch's
    ``ignore_index`` for cross-entropy loss).
    """
    pad_id = 0
    max_len = max(s["input_ids"].size(0) for s in samples)
    out_ids: list[torch.Tensor] = []
    out_labels: list[torch.Tensor] = []
    for s in samples:
        ids = s["input_ids"]
        labs = s["labels"]
        pad = max_len - ids.size(0)
        if pad:
            ids = torch.cat([ids, ids.new_full((pad,), pad_id)])
            labs = torch.cat([labs, labs.new_full((pad,), -100)])
        out_ids.append(ids)
        out_labels.append(labs)
    return {
        "input_ids": torch.stack(out_ids, dim=0),
        "labels": torch.stack(out_labels, dim=0),
    }
