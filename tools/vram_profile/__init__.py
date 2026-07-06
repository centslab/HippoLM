"""Peak VRAM profiling and analysis tools.

Five reusable scripts (originally developed ad-hoc, formalized
2026-07-06):

  - :func:`profile.run` — main profiler. Runs the production
    training loop with ``torch.cuda.memory._record_memory_history``
    enabled, captures per-phase snapshots, dumps a pickle.
  - :func:`analyze.run` — offline analyser. Reads the pickle,
    walks the live blocks, buckets by size bracket and innermost
    Python frame.
  - :func:`classify.run` — offline classifier. Reads the pickle,
    buckets by coarse component purpose (model weight vs
    checkpoint input vs KDA last-block cache vs FusedLinearCE
    dw/dx vs AttnRes saved vs RMSNorm saved).
  - :func:`saved_tensor_probe.run` — small-scale (H=512, T=2048)
    probe that hooks ``FunctionCtx.save_for_backward`` to capture
    every saved tensor at forward time with dtype+shape+bytes.
  - :func:`saved_tensor_probe_base.run` — base-scale (H=1536,
    T=2048) probe with the same hook for production-dim shapes.
  - :func:`vram_peak.run` — A/B harness for opt-5/1 (and similar):
    runs the model twice (with/without the candidate change) and
    compares ``torch.cuda.max_memory_allocated()`` across 3 scales.
  - :func:`poll_prod.run` — high-frequency (default 200 Hz) poll
    of ``mem_get_info()`` during a live run. Catches transient
    peaks between synchronous snapshots.
  - :func:`rmsnorm_math_check.run` — verifies that the y-based
    RMSNorm backward is bit-equivalent to the x-based one (the
    mathematical premise behind the opt-5/1 revert).

Run from the repo root::

    python -m tools.vram_profile.profile --steps 2 --n-chunks 16
    python -m tools.vram_profile.analyze /tmp/vram.pickle
    python -m tools.vram_profile.classify /tmp/vram.pickle --focus peak
    python -m tools.vram_profile.saved_tensor_probe_base
    python -m tools.vram_profile.vram_peak
    python -m tools.vram_profile.poll_prod --steps 1 --poll-hz 500

See ``docs/vram_debugging.md`` for the full methodology.
"""
from __future__ import annotations

from . import (
    analyze,
    classify,
    poll_prod,
    profile,
    rmsnorm_math_check,
    saved_tensor_probe,
    saved_tensor_probe_base,
    vram_peak,
)

__all__ = [
    "analyze",
    "classify",
    "poll_prod",
    "profile",
    "rmsnorm_math_check",
    "saved_tensor_probe",
    "saved_tensor_probe_base",
    "vram_peak",
]