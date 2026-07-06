"""Analyse torch.cuda.memory._snapshot() pickle (PyTorch 2.9 schema).

Snapshot dict shape:
  - 'segments': list of segment dicts with 'blocks' (each block
    has 'state': 'active_allocated' | 'inactive' | 'free', plus
    'size' and 'frames').
  - 'allocator_history': list of alloc/free TraceEntry dicts.
  - 'allocator_settings', 'external_annotations', 'device_traces'.

Reads a pickle dumped by :mod:`tools.vram_profile.profile`.
The profile dumps ``{'peak_snapshots': [...], 'end': ...}``
(one peak snapshot per chunk's fwd_out / bwd_out / flush_out,
end = residual state at run exit). We bucket the PEAK allocations
(which is what dominates VRAM) by size bracket and innermost Python frame.
"""
from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path


def human_mib(n: int) -> str:
    return f"{human_gib(n):.3f} GiB" if n > 1024**3 else f"{n / 1024**2:.1f} MiB"


def human_gib(n: int) -> float:
    return n / 1024**3


def short_frame(frames, depth: int = 0) -> str:
    """Return ``file:line func`` for the frame at ``depth`` from
    innermost (depth=0). In the snapshot schema, frames are stored
    innermost-first (frames[0] is the Python function that called
    into C++ to allocate), so ``depth=0`` = where the alloc happened."""
    if not frames:
        return "(no frames)"
    if depth >= len(frames):
        fr = frames[-1]
    else:
        fr = frames[depth]
    return f"{Path(fr['filename']).name}:{fr['line']} {fr['name']}"


def _analyse_snap(snap: dict, tag: str) -> None:
    print(f"\n[analyze:{tag}] keys: {list(snap.keys())}")
    segments = snap.get("segments") or []
    print(f"[analyze:{tag}] n_segments: {len(segments)}")

    live_blocks = []
    for seg in segments:
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                live_blocks.append((
                    blk["size"], blk["requested_size"], blk.get("frames", []),
                ))
    live_total = sum(b[0] for b in live_blocks)
    n_with_frames = sum(1 for b in live_blocks if b[2])
    print(f"[analyze:{tag}] n_live_blocks: {len(live_blocks)} "
          f"(of which {n_with_frames} have stack frames)")
    print(f"[analyze:{tag}] total live:    {human_mib(live_total)} "
          f"({human_gib(live_total):.3f} GiB)")

    # 1) Size-bracket distribution.
    brackets = [
        (1,                64 * 1024,            "< 64 KiB"),
        (64 * 1024,        1024 * 1024,          "64 KiB - 1 MiB"),
        (1024 * 1024,      16 * 1024 * 1024,     "1 - 16 MiB"),
        (16 * 1024 * 1024, 64 * 1024 * 1024,     "16 - 64 MiB"),
        (64 * 1024 * 1024, 256 * 1024 * 1024,    "64 - 256 MiB"),
        (256 * 1024 * 1024, 1024**3,             "256 MiB - 1 GiB"),
        (1024**3,          16 * 1024**3,         ">= 1 GiB"),
    ]
    bracket_total = collections.Counter()
    bracket_count = collections.Counter()
    for sz, _, _ in live_blocks:
        for lo, hi, label in brackets:
            if lo <= sz < hi:
                bracket_total[label] += sz
                bracket_count[label] += 1
                break

    print(f"\n=== [{tag}] Live allocation size distribution ===")
    for lo, hi, label in brackets:
        n = bracket_count[label]
        sz = bracket_total[label]
        pct = 100 * sz / live_total if live_total else 0
        bar = "#" * int(pct / 2)
        print(f"  {label:>20}: {n:>6} blocks, "
              f"{human_mib(sz):>10} ({pct:5.1f}%) {bar}")

    # 2) Top-N individual allocations.
    print(f"\n=== [{tag}] Top-30 individual live allocations ===")
    top = sorted(live_blocks, key=lambda b: -b[0])[:30]
    for sz, req, frames in top:
        loc = short_frame(frames) if frames else "(no frames)"
        print(f"  {human_mib(sz):>10} (req {human_mib(req):>10})  {loc}")

    # 3) Aggregate by innermost frame.
    print(f"\n=== [{tag}] Aggregate live by innermost frame ===")
    by_innermost: dict[str, int] = collections.defaultdict(int)
    by_count: dict[str, int] = collections.defaultdict(int)
    for sz, _, frames in live_blocks:
        loc = short_frame(frames) if frames else "(no frames)"
        by_innermost[loc] += sz
        by_count[loc] += 1
    rows = sorted(by_innermost.items(), key=lambda x: -x[1])[:30]
    for name, sz in rows:
        pct = 100 * sz / live_total if live_total else 0
        n = by_count[name]
        print(f"  {human_mib(sz):>10} ({pct:5.1f}%)  n={n:>3}  {name}")

    # 4) Aggregate by 2-deep frame.
    print(f"\n=== [{tag}] Aggregate live by 2-deep frame ===")
    by_2deep: dict[tuple, int] = collections.defaultdict(int)
    for sz, _, frames in live_blocks:
        if len(frames) >= 2:
            key = (short_frame(frames, 1), short_frame(frames, 0))
        elif frames:
            key = ("(caller)", short_frame(frames, 0))
        else:
            key = ("(caller)", "(no frames)")
        by_2deep[key] += sz
    rows = sorted(by_2deep.items(), key=lambda x: -x[1])[:30]
    for (caller, callee), sz in rows:
        pct = 100 * sz / live_total if live_total else 0
        print(f"  {human_mib(sz):>10} ({pct:5.1f}%)  {caller}  ->  {callee}")

    # 5) Aggregate by 3-deep frame.
    print(f"\n=== [{tag}] Aggregate live by 3-deep frame ===")
    by_3deep: dict[tuple, int] = collections.defaultdict(int)
    for sz, _, frames in live_blocks:
        if len(frames) >= 3:
            key = (short_frame(frames, 2), short_frame(frames, 1),
                   short_frame(frames, 0))
        elif len(frames) >= 2:
            key = ("(grandcaller)", short_frame(frames, 1),
                   short_frame(frames, 0))
        elif frames:
            key = ("(grandcaller)", "(caller)", short_frame(frames, 0))
        else:
            key = ("(grandcaller)", "(caller)", "(no frames)")
        by_3deep[key] += sz
    rows = sorted(by_3deep.items(), key=lambda x: -x[1])[:30]
    for (gc, c, callee), sz in rows:
        pct = 100 * sz / live_total if live_total else 0
        print(f"  {human_mib(sz):>10} ({pct:5.1f}%)  {gc}  ->  {c}  ->  {callee}")

    # 6) Segment summary.
    print(f"\n=== [{tag}] Segment summary (n={len(segments)}) ===")
    seg_total = sum(seg.get("allocated_size", 0) for seg in segments)
    seg_active = sum(seg.get("active_size", 0) for seg in segments)
    seg_total_total = sum(seg.get("total_size", 0) for seg in segments)
    print(f"  sum(allocated_size):     {human_mib(seg_total)}")
    print(f"  sum(active_size):        {human_mib(seg_active)}")
    print(f"  sum(total_size):         {human_mib(seg_total_total)}")
    print(f"  segment types:           "
          f"{dict(collections.Counter(s.get('segment_type') for s in segments))}")


def analyse(pickle_path: str) -> None:
    with open(pickle_path, "rb") as f:
        snap = pickle.load(f)

    print(f"[analyze] {pickle_path}")
    if isinstance(snap, dict) and "peak" in snap:
        print(f"[analyze] format: peak + end (run-time captured)")
        print(f"\n{'#' * 70}")
        print(f"# PEAK SNAPSHOT (saved-tensor set alive, post first chunk fwd)")
        print(f"{'#' * 70}")
        _analyse_snap(snap["peak"], "PEAK")
        print(f"\n{'#' * 70}")
        print(f"# END-OF-RUN SNAPSHOT (residual: model weights + final grads)")
        print(f"{'#' * 70}")
        _analyse_snap(snap["end"], "END")
    else:
        _analyse_snap(snap, "SNAP")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pickle")
    args = p.parse_args()
    analyse(args.pickle)