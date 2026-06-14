"""Collate per-sample dicts into a batched dict with right-padding.

Also exposes :func:`ffd_pack_samples` (first-fit-decreasing packing)
to concatenate short samples into a single ``[seq_len]`` row, which
avoids the ~50% padding waste of the legacy right-pad path on text
streams of variable length.
"""
from __future__ import annotations

import torch


def ffd_pack_samples(
    samples: list[list[int]],
    seq_len: int,
    pad_id: int = 0,
    doc_sep_id: int | None = None,
) -> list[list[int]]:
    """First-fit-decreasing packing of variable-length token lists.

    Args:
        samples: per-document token lists. Each is a Python list of
            ints (token ids). The caller is responsible for any
            BOS / EOS insertion; this function only packs.
        seq_len: target packed sequence length. Every returned pack
            has length exactly ``seq_len`` (right-padded with
            ``pad_id``).
        pad_id: padding id (default 0).
        doc_sep_id: optional document-separator id. When set, a
            single ``doc_sep_id`` is inserted between adjacent
            documents inside a pack so the loss doesn't train
            across document boundaries. The separator is counted
            toward ``seq_len``, so a pack holds
            ``sum(len(d) for d in pack) + (n_docs - 1) <= seq_len``.

    Returns:
        A list of ``[seq_len]`` int lists. Each pack has length
        exactly ``seq_len``.

    Algorithm: classic first-fit-decreasing. Sort samples by length
    descending, then for each sample place it into the first pack
    that has enough room (when ``doc_sep_id`` is set, the per-doc
    overhead is ``len(d) + 1`` except for the first doc). O(N log N)
    for the sort + O(N * P) for placement, where P is the number of
    packs (typically ``N / (seq_len / mean_doc_len)``).
    """
    if not samples:
        return []
    # Reserve ``seq_len + 1`` if we need a doc separator (one extra
    # slot per pack for the very first doc's leading separator is
    # not added; the separator only appears between docs).
    def slot_cost(n: int, is_first_in_pack: bool) -> int:
        if doc_sep_id is None or is_first_in_pack:
            return n
        return n + 1

    # Sort by length descending so the longest doc gets placed
    # first (FFD's improvement over naive first-fit is the
    # descending order — packs are tighter on average).
    sorted_idx = sorted(range(len(samples)), key=lambda i: len(samples[i]), reverse=True)
    packs: list[list[int]] = []
    used: list[int] = []  # tokens used per pack (excl. doc_sep overhead)

    for i in sorted_idx:
        doc = list(samples[i])
        # If a single document is longer than ``seq_len`` (shouldn't
        # happen if the tokenize step already truncated, but be
        # defensive), truncate it. We don't split a doc across
        # packs because that would train the model on a doc with
        # the wrong attention context.
        max_doc_tokens = seq_len if doc_sep_id is None else seq_len
        if len(doc) > max_doc_tokens:
            doc = doc[:max_doc_tokens]
        placed = False
        for p_idx, pack in enumerate(packs):
            # The new doc fits if the current pack has room for
            # ``slot_cost(len(doc), is_first=False)`` additional
            # tokens (a separator is inserted before this doc if
            # the pack already has at least one doc).
            extra = slot_cost(len(doc), is_first_in_pack=len(pack) == 0)
            if used[p_idx] + extra <= seq_len:
                if doc_sep_id is not None and len(pack) > 0:
                    pack.append(doc_sep_id)
                pack.extend(doc)
                used[p_idx] = len(pack)  # actual length after extend
                placed = True
                break
        if not placed:
            # Start a new pack with this doc.
            new_pack: list[int] = list(doc)
            packs.append(new_pack)
            used.append(len(new_pack))
    # Right-pad to seq_len.
    out: list[list[int]] = []
    for pack in packs:
        if len(pack) < seq_len:
            pack = pack + [pad_id] * (seq_len - len(pack))
        out.append(pack)
    return out


def collate_batch(
    samples: list[dict],
    *,
    pin_memory: bool = True,
) -> dict:
    """Stack a list of per-sample dicts into a batched dict.

    Each sample is ``{"input_ids": [T], "labels": [T]}`` where
    ``input_ids`` / ``labels`` may be a 1D ``torch.Tensor`` (the
    legacy path) or a Python list of ints (the v0.0.2 tokenize
    path that drops the wasteful ``return_tensors='pt' +
    .squeeze(0)`` round-trip). Both are handled.

    Lengths may differ, so we right-pad to the longest sequence in
    the batch with pad id 0; labels are padded with ``-100``
    (PyTorch's ``ignore_index`` for cross-entropy loss).

    When ``pin_memory=True`` (the default) the output tensors are
    allocated on pinned host memory so the downstream
    ``.to('cuda', non_blocking=True)`` can actually overlap with
    the next forward pass. The legacy code path always returned
    non-pinned tensors and the ``non_blocking`` flag was a lie.
    """
    pad_id = 0

    def _to_tensor_1d(x):
        # Accept either a 1D tensor (legacy) or a Python list
        # (v0.0.2 tokenize path). Return a 1D ``long`` tensor on
        # CPU; the dtype is fixed so the embedding lookup can use
        # the right index dtype.
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.long).view(-1).cpu()
        return torch.as_tensor(list(x), dtype=torch.long)

    max_len = max(_to_tensor_1d(s["input_ids"]).numel() for s in samples)
    out_ids: list[torch.Tensor] = []
    out_labels: list[torch.Tensor] = []
    for s in samples:
        ids = _to_tensor_1d(s["input_ids"])
        labs = _to_tensor_1d(s["labels"])
        pad = max_len - ids.numel()
        if pad:
            ids = torch.cat([ids, ids.new_full((pad,), pad_id)])
            labs = torch.cat([labs, labs.new_full((pad,), -100)])
        out_ids.append(ids)
        out_labels.append(labs)
    if pin_memory and not torch.cuda.is_available():
        # No GPU to DMA into; skip the pinning (it costs alloc time).
        pin_memory = False
    out = {
        "input_ids": torch.stack(out_ids, dim=0),
        "labels": torch.stack(out_labels, dim=0),
    }
    if pin_memory:
        out["input_ids"] = out["input_ids"].pin_memory()
        out["labels"] = out["labels"].pin_memory()
    return out
