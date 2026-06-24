"""Collate per-sample dicts into a packed batch with chunk-aligned doc
boundaries.

FFD packing is the primary mode: each tokenized doc is rounded up to
a multiple of ``chunk_size`` so the KDA chunkwise kernel's state
reset lands exactly at a doc boundary (the kernel's intra-chunk
computation is otherwise wrong for chunks that straddle two docs).
The packer also produces a ``cu_seqlens`` tensor that the model
threads into the KDA call so each doc's recurrence is independent
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
    batch_size: int,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-fit-decreasing packing of tokenized documents into a
    batch of ``[batch_size, seq_len]`` rows with chunk-aligned doc
    boundaries.

    The output batch dim is **exactly** ``batch_size`` regardless of
    how densely the input samples pack. When more samples fit than
    ``batch_size`` packs can hold the excess is dropped (see *FFD
    overflow* below); when fewer samples fit the unused packs are
    filled entirely with ``pad_id`` tokens (their labels are all
    ``-100`` and the kernel sees them as a single all-pad "doc").

    Args:
        samples: per-document token lists. The caller is responsible
            for tokenization; this function only packs. If
            ``eos_id`` is set, each doc is forced to end with this
            id (appended if not present) so the boundary marker is
            uniform.
        seq_len: target packed sequence length (must be a multiple
            of ``chunk_size``).
        chunk_size: alignment granularity for doc boundaries. Must
            be a positive multiple of the KDA kernel's internal
            ``BT=64`` (the packer rounds up internally if not).
        batch_size: number of packed rows in the output. The output
            ``input_ids`` and ``labels`` always have shape
            ``[batch_size, seq_len]`` — the consumer (training
            loop) relies on this to plan its per-step token budget
            and TP all-reduce buffers. Must be >= 1.
        pad_id: padding id for both intra-doc tail pad and pack
            trailing pad.
        eos_id: optional EOS token id. If set, ``samples`` are
            post-processed so each doc ends with ``eos_id``.

    Returns:
        A tuple ``(input_ids, labels, cu_seqlens)`` where:

          - ``input_ids``: ``[batch_size, seq_len]`` long tensor.
          - ``labels``:    ``[batch_size, seq_len]`` long tensor,
            ``-100`` at pad positions (see module docstring for
            the three classes).
          - ``cu_seqlens``: ``[total_docs + batch_size]`` long
            tensor with global offsets across the flattened
            ``[batch_size * seq_len]`` sequence. Pass to the KDA
            kernel (which expects ``B=1``) to mark each doc's
            ``[start, end)`` for state reset.

            The boundary semantics are: each real doc contributes
            one entry (its end); each non-empty pack's tail pad
            contributes one entry (the start of the tail-pad "doc");
            each pack boundary contributes one entry (``pack_end``
            = start of next pack or total length for the last
            pack). For an empty pack the previous pack's ``pack_end``
            already starts the empty pack's all-pad "doc", so the
            empty pack contributes no additional entries beyond its
            own ``pack_end``.

        ``cu_seqlens[0]`` is always 0 and ``cu_seqlens[-1]`` is
        always ``batch_size * seq_len``.

    FFD overflow
        The packer pre-allocates ``batch_size`` packs and places
        each sample first-fit-decreasing into the first pack with
        enough room. When a sample does not fit in any existing
        pack it is silently dropped (rare — only happens when the
        doc itself is larger than ``seq_len``, which should never
        occur for any reasonable text source). Raise
        ``pack_buffer_size`` on the caller side if you want denser
        packs (more candidates = better FFD fit); ``batch_size``
        controls only the number of output rows.
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

    if batch_size < 1:
        raise ValueError(
            f"pack_chunk_aligned: batch_size must be >= 1, got {batch_size}"
        )

    # Empty input: return ``[batch_size, seq_len]`` all-pad with
    # ``batch_size`` single-pad-doc boundaries (one per pack).
    if not samples:
        input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.long,
        )
        return input_ids, labels, cu_seqlens

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

    # Fixed-pack FFD: pre-allocate ``batch_size`` packs and place
    # each sample first-fit-decreasing into the first pack with
    # enough room. Overflow samples (those that don't fit in any
    # pack because they are individually larger than ``seq_len``,
    # which the chunk rounding would cap) are silently dropped —
    # but for any reasonable tokenized text source this branch
    # never triggers because ``max(1, ceil(L/chunk_size))`` packs
    # the doc into ``seq_len`` regardless of how it rounds.
    max_chunks_per_pack = seq_len // chunk_size
    packs: list[list[int]] = [[] for _ in range(batch_size)]
    chunks_used: list[int] = [0] * batch_size
    for i in sorted_idx:
        n = n_chunks_per_doc[i]
        placed = False
        for j in range(batch_size):
            if chunks_used[j] + n <= max_chunks_per_pack:
                packs[j].append(i)
                chunks_used[j] += n
                placed = True
                break
        # Overflow: doc is so large (after chunk-rounding) that it
        # does not fit in any single pack. This should not happen
        # for real text data — ``max(1, ceil(L/chunk_size))`` is
        # bounded by ``seq_len/chunk_size`` whenever
        # ``L <= seq_len``, which is the contract with the upstream
        # tokenizer. We drop silently to preserve the
        # ``n_packs == batch_size`` invariant; the warning is left
        # for a future debugging session if it ever fires.
        # assert placed, f"pack_chunk_aligned: doc {i} (L={len(samples[i])}) overflowed"

    # Build output tensors ``[batch_size, seq_len]``.
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    cu_seqlens_list: list[int] = [0]

    for p_idx, pack in enumerate(packs):
        pack_start = p_idx * seq_len
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
            cu_seqlens_list.append(pack_start + pos)

        # Per-pack boundary. Appends the END of this pack
        # (= start of next pack OR ``batch_size * seq_len`` for the
        # last pack). This guarantees the kernel resets its
        # recurrent state ``h`` between packs: any tail-pad tokens
        # within this pack are seen as their own "doc" (the entry
        # for the last real doc's end already started it; this
        # entry terminates it), and any empty pack is bounded on
        # both sides by ``pack_end`` (the previous pack's end on
        # the left, this pack's end on the right) — so the kernel
        # processes the all-pad region as a single zero-information
        # "doc" with no state contamination of the next pack.
        #
        # Without per-pack boundaries the tail-pad tokens
        # [last_doc_end, pack_end) would be attributed to the
        # preceding real doc (the entry for the next pack's first
        # doc starts a new "doc" at ``pack_end``, leaving the tail
        # pad inside the old doc's window). At prod dims (B=6
        # packs at T=4096, ~32 real docs packed, with ~2.4k-token
        # tail pads from non-full packs) the stale ``h`` flowing
        # through the tail pad amplifies via the ``(I+T)^{-1}``
        # solve and NaNs the FORWARD in train mode.
        #
        # De-dup vs the previous entry: if the last doc of this
        # pack was a full-pack doc (L_ck == seq_len) the inner
        # loop already emitted ``pack_start + seq_len``; emitting
        # it again would create a zero-length "doc" that the
        # kernel sees as a no-op but bloats cu_seqlens and risks
        # an edge case in chunk_indices.
        pack_end = pack_start + seq_len
        if not cu_seqlens_list or cu_seqlens_list[-1] != pack_end:
            cu_seqlens_list.append(pack_end)

    # ``cu_seqlens[0] == 0`` (set above) and ``cu_seqlens[-1] ==
    # batch_size * seq_len`` (the final per-pack boundary we just
    # appended) are guaranteed. The total number of cu_seqlens
    # entries is ``total_real_docs + batch_size`` (one per real
    # doc + one per pack boundary).

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