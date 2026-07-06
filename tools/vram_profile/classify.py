"""Classify peak-VRAM allocations by component purpose.

Reads a ``{peak_snapshots, end}`` pickle (from
:mod:`tools.vram_profile.profile`) and walks the live
blocks, bucketing each into a coarse component class:

  - model.weight          : embed / RMSNorm / AttnRes / A_log / dt_bias
                            / o_norm / fg_first / per-layer KDA
                            projections (the model params)
  - kda.intermediates     : KDA kernel saved tensors from
                            :class:`ChunkKDAFunction` (q, k, v,
                            g_input, g_cumsum FP32, Aqk, Akk, w, u,
                            qg, kg, v_new, h) — including the
                            un-checkpointed layer 31.
  - checkpoint.input      : the per-sub-block ``torch.utils.checkpoint``
                            input tensors (one per 2-layer sub-block,
                            14 of them: 7 blocks * 2 sub-blocks)
  - attn_res.saved        : BlockAttnRes input saved tensors (one per
                            non-first block boundary; 6 stacks live
                            at peak fwd_out)
  - rms_norm.saved        : BF16 ref saves from `LayerNormFunction` /
                            `LayerNormGatedFunction` (NOT fp32 casts —
                            see :mod:`docs.vram_debugging` §3.3)
  - residual.input        : TPHippoLayer.forward's `x = x + SubLayer(x)`
                            residual-add input save
  - fused_lce.dx          : FusedLinearCE backward dx + dw saved tensor
                            (label is a misnomer — line 433 is `dw`,
                            not `dx`; see docstring)
  - workspace             : short-lived matmul/einsum/etc outputs that
                            are still alive at snapshot time
  - other                 : anything that doesn't fit the above

Caveat: ``is_residual_input`` matches ANY frame in the stack
containing ``layer.py`` + ``forward``, so KDA intermediates called
from inside a layer forward get misclassified into ``residual.input``.
For accurate KDA-last-block accounting, use the manual
re-bucketing described in ``docs/vram_debugging.md``.
"""
from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path


def human_mib(n: int) -> str:
    return f"{n / 1024**3:.3f} GiB" if n > 1024**3 else f"{n / 1024**2:.1f} MiB"


# Bucket map: list of (substr_in_innermost_frame, label).
# First match wins. Order matters: more specific first.
BUCKETS_INNER = [
    # FusedLinearCE backward dx is a single [N, V] tensor.
    ("fused_linear_cross_entropy.py:433", "fused_lce.dx"),
    ("fused_linear_cross_entropy.py:427", "fused_lce.dx"),
    # NVFP4 linear: dequantized weight view + matmul output.
    ("nvfp4_linear.py:89",                "model.ffn_nvfp4"),
    ("nvfp4.py:190",                      "model.ffn_nvfp4"),
    # KDA intra chunkwise intermediates.
    ("chunk_delta_h.py:699",              "kda.saved.last_block"),
    ("chunk_delta_h.py:700",              "kda.saved.last_block"),
    ("chunk_delta_h.py:702",              "kda.saved.last_block"),
    ("chunk_gla_fwd_o_gk.py:898",         "kda.saved.last_block"),
    ("chunk_kda_fwd_intra.py:771",        "kda.saved.last_block"),
    ("chunk_kda_fwd_intra.py:773",        "kda.saved.last_block"),
    ("chunk_kda_fwd_intra.py:836",        "kda.saved.last_block"),
    ("recompute_w_u_fwd",                 "kda.saved.last_block"),
    ("chunk_kda_fwd_h",                   "kda.saved.last_block"),
    ("chunk_kda_fwd_o",                   "kda.saved.last_block"),
    ("chunk_kda_fwd_wy",                  "kda.saved.last_block"),
    ("chunk_kda_fwd_intra",               "kda.saved.last_block"),
    ("chunk_kda_fwd_norm",                "kda.saved.last_block"),
    ("kda_gate_chunk_cumsum",             "kda.saved.last_block"),
    ("kda_gate_chunk_reducecumsum",       "kda.saved.last_block"),
    ("l2norm_fwd",                        "kda.saved.last_block"),
    ("l2norm_bwd",                        "kda.saved.last_block"),
    # RMSNorm (FusedRMSNormGated or plain RMSNorm) forward saves
    # the input in fp32 for backward — one per layer (32 of them)
    # plus 7 AttnRes-internal RMSNorms.
    ("layernorm.py:562",                  "rms_norm.saved"),
    ("fused_norm_gate.py:465",            "rms_norm.saved"),
    # Checkpoint input saves: these come from inside the
    # checkpoint wrapper, so their innermost frame is the layer
    # forward that produced them. We bucket by checking the
    # 3rd-deep frame for ``utils.checkpoint``.
    # (handled separately in classify())
    # AttnRes saved tensors (BlockAttnRes.forward input).
    ("attn_res.py:245",                   "attn_res.saved"),
    # BlockAttnRes's einsum internals (its q/k/v projection outputs).
    ("attn_res.py:268",                   "attn_res.saved"),
    ("attn_res.py:263",                   "attn_res.saved"),
    # Linear (TP ColumnParallelLinear) outputs saved for backward.
    ("tp_layers.py:297",                  "linear.saved"),
    # SwiGLU intermediate (gate, up, down projections).
    ("swiglu.py:67",                      "ffn.saved"),
    # Short conv output.
    ("ops.py:51",                         "short_conv.saved"),
    # Embed lookup partial: [B*T, H] before all-reduce.
    ("embed.py:60",                       "embed.saved"),
    # silu / softmax intermediates.
    ("functional.py:2371",                "ffn.saved"),
    ("functional.py:2133",                "ffn.saved"),
    ("functional.py:373",                 "attn_res.saved"),
]


def short_frame(frames, depth: int = 0) -> str:
    if not frames:
        return "(no frames)"
    if depth >= len(frames):
        fr = frames[-1]
    else:
        fr = frames[depth]
    return f"{Path(fr['filename']).name}:{fr['line']} {fr['name']}"


def is_checkpoint_input(frames) -> bool:
    """True if the alloc was triggered by the per-sub-block
    ``torch.utils.checkpoint.checkpoint`` call (the input tensor
    that is saved for the backward recomputation)."""
    for fr in frames:
        if "checkpoint.py" in fr["filename"] and fr["name"] in (
            "checkpoint", "forward",
        ):
            return True
    return False


def is_residual_input(frames) -> bool:
    """True if the alloc was triggered by TPHippoLayer.forward's
    residual addition ``x = x + SubLayer(x)`` (saved x for backward).
    """
    for fr in frames:
        if "layer.py" in fr["filename"] and fr["name"] == "forward":
            return True
    return False


def classify(seg: dict) -> tuple[str, str]:
    """Return (bucket, frame_label) for a segment dict."""
    blocks = seg.get("blocks", [])
    # We classify by aggregating the frames of all active blocks
    # in the segment (segments can hold multiple blocks; the
    # frames are typically per-block).
    sample_frames = None
    for blk in blocks:
        if blk.get("state") == "active_allocated":
            sample_frames = blk.get("frames", [])
            break
    if not sample_frames:
        return ("(unattributed)", "(no frames)")

    inner_label = short_frame(sample_frames, 0)

    # 1) Checkpoint input.
    if is_checkpoint_input(sample_frames):
        return ("checkpoint.input", inner_label)
    # 2) Residual addition input (TPHippoLayer's x + sublayer(x)).
    if is_residual_input(sample_frames):
        return ("residual.input", inner_label)
    # 3) Buckets by innermost-frame substring.
    for substr, label in BUCKETS_INNER:
        if substr in inner_label:
            return (label, inner_label)
    return ("other", inner_label)


def analyse(pickle_path: str, focus: str = "peak") -> None:
    with open(pickle_path, "rb") as f:
        snap = pickle.load(f)
    if isinstance(snap, dict) and "peak" in snap:
        snap = snap[focus]
    print(f"[classify:{focus}] {pickle_path}")

    segments = snap.get("segments") or []
    live_segments = [s for s in segments
                     if any(b.get("state") == "active_allocated"
                            for b in s.get("blocks", []))]
    print(f"[classify:{focus}] n_segments_total={len(segments)}, "
          f"n_live_segments={len(live_segments)}")

    # Aggregate by bucket.
    bucket_total: dict[str, int] = collections.Counter()
    bucket_count: dict[str, int] = collections.Counter()
    bucket_largest: dict[str, tuple[int, str]] = {}  # bucket -> (size, frame)
    grand_total = 0
    grand_attributed = 0
    for seg in live_segments:
        # All active blocks in this segment.
        active_blocks = [b for b in seg["blocks"]
                         if b.get("state") == "active_allocated"]
        if not active_blocks:
            continue
        # All active blocks in one segment share the same frame
        # signature (that's the segment's purpose). Bucket them
        # together.
        bucket, label = classify(seg)
        seg_active = sum(b["size"] for b in active_blocks)
        bucket_total[bucket] += seg_active
        bucket_count[bucket] += len(active_blocks)
        if bucket not in bucket_largest or seg_active > bucket_largest[bucket][0]:
            bucket_largest[bucket] = (seg_active, label)
        grand_total += seg_active
        if bucket != "(unattributed)":
            grand_attributed += seg_active

    print(f"\n=== [{focus}] Live by component bucket ===")
    print(f"  grand total: {human_mib(grand_total)}")
    print(f"  attributed:  {human_mib(grand_attributed)} "
          f"({100 * grand_attributed / max(grand_total, 1):.1f}%)")
    print()
    rows = sorted(bucket_total.items(), key=lambda x: -x[1])
    for bucket, sz in rows:
        pct = 100 * sz / max(grand_total, 1)
        n = bucket_count[bucket]
        largest = bucket_largest.get(bucket, (0, ""))
        bar = "#" * int(pct / 2)
        print(f"  {bucket:<30} {human_mib(sz):>10} ({pct:5.1f}%)  "
              f"n={n:>4}  largest={human_mib(largest[0])}  "
              f"largest_at={largest[1][:60]}{bar}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pickle")
    p.add_argument("--focus", default="peak", choices=["peak", "end"])
    args = p.parse_args()
    analyse(args.pickle, args.focus)