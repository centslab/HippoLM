"""Collate per-sample dicts into a packed batch with chunk-aligned doc
boundaries.

FFD packing is the primary mode: each tokenized doc is rounded up to
a multiple of ``chunk_size`` so the GDN2 chunkwise kernel's state
reset lands exactly at a doc boundary (the kernel's intra-chunk
computation is otherwise wrong for chunks that straddle two docs).
The packer also produces a ``cu_seqlens`` tensor that the model
threads into the GDN2 call so each doc's recurrence is independent
across the pack.

Mask policy
-----------
Three classes of label positions are set to ``-100`` (PyTorch's
``ignore_index`` for cross-entropy):

  1. Intra-doc tail pad ``[L, L_ck)``: rounds each doc up to a chunk
     multiple; these are filler tokens (the model's representation
     of position ``L-1`` — the doc's last token — is still correct
     because all ``L-1`` real tokens precede them in the recurrence).
  2. Cross-doc transition ``[L, L)`` (no length; the very first
     token of doc ``i+1``): masks the prediction from doc ``i``'s
     last context. Without the mask the model would be trained to
     predict doc ``i+1``'s first token from doc ``i``'s EOS, which
     is meaningless noise. (Auto-handled when ``L_ck > L`` because
     the position falls inside the tail pad from rule 1.)
  3. Pack trailing pad ``[sum(L_ck), S)``: filler to reach
     ``seq_len``.

The EOS at each doc's last position (``L-1``) is *not* masked: the
model is trained to predict EOS at doc end, exactly the standard
LM convention with the natural tokenizer-added EOS.

The right-pad legacy path (``collate_batch``) is preserved for the
dummy data loader; the real streaming path uses the packer.
"""
from __future__ import annotations

import torch


def pack_chunk_aligned(
    samples: list[list[int]],
    seq_len: int,
    chunk_size: int,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-fit-decreasing packing of tokenized documents into a
    batch of ``[B, seq_len]`` rows with chunk-aligned doc boundaries.

    Args:
        samples: per-document token lists. The caller is responsible
            for tokenization; this function only packs. If
            ``eos_id`` is set, each doc is forced to end with this
            id (appended if not present) so the boundary marker is
            uniform.
        seq_len: target packed sequence length (must be a multiple
            of ``chunk_size``).
        chunk_size: alignment granularity for doc boundaries. Must
            be a positive multiple of the GDN2 kernel's internal
            ``BT=64`` (the packer rounds up internally if not).
        pad_id: padding id for both intra-doc tail pad and pack
            trailing pad.
        eos_id: optional EOS token id. If set, ``samples`` are
            post-processed so each doc ends with ``eos_id``.

    Returns:
        A tuple ``(input_ids, labels, cu_seqlens)`` where:

          - ``input_ids``: ``[B, seq_len]`` long tensor.
          - ``labels``:    ``[B, seq_len]`` long tensor, ``-100`` at
            pad positions (see module docstring for the three
            classes).
          - ``cu_seqlens``: ``[total_docs + 1]`` long tensor with
            global offsets across the flattened ``[B * seq_len]``
            sequence. Pass to the GDN2 kernel (which expects ``B=1``)
            to mark each doc's ``[start, end)`` for state reset.

        When ``samples`` is empty the returned tensors are zero-
        sized along the batch / doc dimensions (``input_ids`` is
        ``[0, seq_len]``, ``cu_seqlens`` is ``[0]``).
    """
    # Defensive: chunk_size must be a multiple of the kernel's BT=64
    # so that every doc's chunk-aligned length is a multiple of BT
    # and the kernel's per-chunk computation is well-defined. Round
    # up if the caller passed something else.
    if chunk_size % 64 != 0:
        chunk_size = ((chunk_size + 63) // 64) * 64

    if seq_len % chunk_size != 0:
        raise ValueError(
            f"pack_chunk_aligned: seq_len ({seq_len}) must be a multiple"
            f" of chunk_size ({chunk_size})"
        )

    if not samples:
        return (
            torch.zeros(0, seq_len, dtype=torch.long),
            torch.full((0, seq_len), -100, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )

    # Force-EOS at the end of each doc so the boundary marker is
    # uniform and the cross-doc prediction position (the first
    # token of doc i+1) is always preceded by EOS.
    if eos_id is not None:
        samples = [
            doc if (len(doc) == 0 or doc[-1] == eos_id) else (doc + [eos_id])
            for doc in samples
        ]

    # FFD sort key: each doc's chunk count. FFD by chunk-count (not
    # by raw length) is correct here because every doc rounds up
    # to a chunk multiple regardless of where its real tokens end.
    n_chunks_per_doc = [
        max(1, (len(doc) + chunk_size - 1) // chunk_size)
        for doc in samples
    ]

    sorted_idx = sorted(
        range(len(samples)),
        key=lambda i: n_chunks_per_doc[i],
        reverse=True,
    )

    # FFD placement: each pack holds up to seq_len // chunk_size
    # chunks total (across all docs in the pack).
    max_chunks_per_pack = seq_len // chunk_size
    packs: list[tuple[list[int], int]] = []  # (doc_indices, chunks_used)
    for i in sorted_idx:
        n = n_chunks_per_doc[i]
        placed = False
        for idx, (p, used) in enumerate(packs):
            if used + n <= max_chunks_per_pack:
                p.append(i)
                packs[idx] = (p, used + n)
                placed = True
                break
        if not placed:
            packs.append(([i], n))

    n_packs = len(packs)
    input_ids = torch.full((n_packs, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((n_packs, seq_len), -100, dtype=torch.long)
    cu_seqlens_list: list[int] = [0]

    for p_idx, (pack, _chunks_used) in enumerate(packs):
        pos = 0  # token position within this pack
        for di, doc_idx in enumerate(pack):
            doc = samples[doc_idx]
            L = len(doc)
            L_ck = n_chunks_per_doc[doc_idx] * chunk_size

            doc_t = torch.as_tensor(doc, dtype=torch.long)
            input_ids[p_idx, pos:pos + L] = doc_t
            labels[p_idx, pos:pos + L] = doc_t

            # Cross-doc mask: don't train the model to predict doc
            # i+1's first token from doc i's last (EOS) context.
            # When L_ck > L this position is the start of the
            # tail pad and is already -100; when L_ck == L it's
            # the very first token of doc i+1 and must be set
            # explicitly.
            if di + 1 < len(pack):
                labels[p_idx, pos + L] = -100

            pos += L_ck
            cu_seqlens_list.append(p_idx * seq_len + pos)

        assert pos <= seq_len, (
            f"pack {p_idx} overflow: pos={pos} > seq_len={seq_len} "
            f"(should never happen — FFD guarantees placement)"
        )

    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.long)
    return input_ids, labels, cu_seqlens


def collate_batch(
    samples: list[dict],
    *,
    pin_memory: bool = True,
) -> dict:
    """Stack a list of per-sample dicts into a batched dict (right-pad).

    Lengths may differ, so we right-pad to the longest sequence in
    the batch with pad id 0; labels are padded with ``-100``
    (PyTorch's ``ignore_index`` for cross-entropy loss).

    This is the legacy right-pad path kept for the dummy-data
    smoke-test loader. The real streaming path uses
    :func:`pack_chunk_aligned` via :class:`PrefetchBatcher`.

    When ``pin_memory=True`` (the default) the output tensors are
    allocated on pinned host memory so the downstream
    ``.to('cuda', non_blocking=True)`` can actually overlap with
    the next forward pass.
    """
    pad_id = 0

    def _to_tensor_1d(x):
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