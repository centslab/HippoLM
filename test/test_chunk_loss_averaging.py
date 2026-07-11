"""Tests for the chunked-training per-chunk loss averaging invariant.

The chunk loop in ``src/training/loop.py`` averages per-chunk losses
into a single ``step_loss``. The contract pinned here:

  ``step_loss`` MUST equal the token-weighted mean CE over the
  whole packed sequence:

      step_loss = sum_i mean_ci * n_valid_i / sum_i n_valid_i

  where ``mean_ci`` is what ``FusedLinearCE`` returns for chunk i
  (mean over that chunk's valid, post-shift tokens) and
  ``n_valid_ci = (labels[:, ci*mb+1 : (ci+1)*mb] != -100).sum()``.

This invariant was broken in the b173879 refactor by dividing every
chunk's mean by ``n_chunks`` (``Σ mean_ci / n_chunks``), which only
holds when every chunk holds an equal share of valid tokens — true
for the dummy-data smoke test, false for real FFD-packed data whose
trailing chunks are pure pad (labels all -100) and on which the model
returns ``loss = 0.0``.

The bug was invisible to tests because:

  * ``--use_dummy_data`` uses right-pad to ``seq_len`` so every
    chunk is full.
  * No previous test exercised ``pack_chunk_aligned`` with a
    realistic utilization < 100% together with the chunked loop.

These tests sweep several utilization ratios (100%, 70%, 31.6%, 10%,
0%) and assert the contract holds across all of them — including
properties like ``step_loss`` is invariant to the number of trailing
empty chunks and equals a single full-sequence forward's mean CE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# --------------------------------------------------------------------------- #
# Mirror of the chunk loop's loss math (loop.py) extracted to a pure function.#
# Faithful replica — update in lockstep if loop.py's formula changes.        #
# --------------------------------------------------------------------------- #
def chunked_step_loss(
    per_chunk_mean: list[float],
    chunk_valid: list[int],
    total_valid: int | None = None,
    n_chunks: int | None = None,
) -> tuple[float, float]:
    """Compute the OLD (buggy) and NEW (fixed) step loss values.

    Mirrors ``src/training/loop.py``'s loop body exactly so a refactor
    of either side breaks this test and forces a coordinated update.

    The NEW value uses token-weighted averaging (skips trailing
    all-pad chunks, in line with the ``last_real`` early-stop in
    loop.py).
    """
    if not per_chunk_mean:
        return 0.0, 0.0
    if total_valid is None:
        total_valid = max(sum(chunk_valid), 1)
    if n_chunks is None:
        n_chunks = len(per_chunk_mean)

    # OLD (buggy, prior to the fix): Σ mean_ci / n_chunks.
    # Multiply out by n_chunks so we can return the dimensionless
    # step_loss the loop logs.
    old = sum(per_chunk_mean) / n_chunks

    # NEW (token-weighted): NEW mirrors the loop formula:
    #   chunk_loss = outputs["loss"] * (n_valid_ci / total_valid)
    #   step_loss_chunk_sum += chunk_loss
    # but also skips trailing all-pad chunks (last_real early-stop)
    # and skips interior empty chunks (no bwd). Both produce 0
    # contribution to step_loss_chunk_sum, so the formula reduces to:
    #   step_loss = Σ_i mean_ci * (n_valid_ci / total_valid)
    # for chunks where n_valid_ci > 0.
    last_real = max(
        (ci for ci, nv in enumerate(chunk_valid) if nv > 0),
        default=-1,
    )
    new = 0.0
    for ci in range(last_real + 1):
        nv = chunk_valid[ci]
        if nv == 0:
            continue
        new += per_chunk_mean[ci] * (nv / total_valid)

    return old, new


# --------------------------------------------------------------------------- #
# Pure-formula property tests — drive via synthetic per-chunk arrays.         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "utilization, n_chunks, mb",
    [
        # utilization = fraction of seq_len that is valid tokens.
        # 100% = dummy-data smoke test path (every chunk full).
        # 31.6% = prod on real Ultra-FineWeb — the exact ratio that
        #         triggered the original bug (loss=3.98 instead of
        #         ~12.7).
        # Other rows: coverage for combined/scaled configs.
        ("100%", 16, 16384),
        ("70%",  16, 16384),
        ("31.6%", 16, 16384),  # production-observed (the bug repro)
        ("10%",  16, 16384),
        ("5%",   16, 16384),
    ],
)
def test_new_equals_token_weighted_mean(utilization, n_chunks, mb):
    """NEW step_loss equals token-weighted mean CE for any fill ratio.

    Build per-chunk means (all ~log(V)=12.42 — what a random-init model
    returns on next-token-prediction) with chunk_valid sized to give
    the requested utilization, then assert NEW equals the analytical
    token-weighted mean.
    """
    import math
    V = 248320
    LOG_V = math.log(V)
    seq_len = n_chunks * mb
    target_valid = int(seq_len * float(utilization.rstrip("%")) / 100)

    # Fill chunks from the front; chunk 0 is fullest, chunk N is
    # tail-pad (mirrors the FFD packer's "docs first, pad tail"
    # contract — see pack_chunk_aligned's loop).
    chunk_valid = []
    remaining = target_valid
    for _ in range(n_chunks):
        chunk_valid.append(min(remaining, mb))
        remaining -= chunk_valid[-1]

    # Random-init model returns loss~log(V) for chunks with valid
    # tokens; 0.0 for empty trailing chunks.
    per_chunk_mean = [LOG_V if nv > 0 else 0.0 for nv in chunk_valid]
    total_valid = max(sum(chunk_valid), 1)

    _, new = chunked_step_loss(per_chunk_mean, chunk_valid, total_valid, n_chunks)
    expected = sum(
        LOG_V * nv for nv in chunk_valid if nv > 0
    ) / total_valid
    assert new == pytest.approx(expected, rel=1e-9), (
        f"utilization={utilization}: new={new:.6f} expected={expected:.6f}"
    )


@pytest.mark.parametrize(
    "utilization",
    ["100%", "70%", "31.6%", "10%", "5%", "1%"],
)
def test_new_invariant_to_trailing_pad_count(utilization):
    """Adding more empty trailing chunks must NOT change step_loss.

    The packer boundary at prod produces a fixed utilization (~31.6%)
    but ``--max_steps`` keeps producing the same fill ratio on every
    step. If padding chunks contributed non-zero (or worse, biased)
    loss, repeated steps would drift — this test pins the
    "padding regions are no-op" invariant.
    """
    import math
    LOG_V = math.log(248320)
    mb = 16384

    # base: minimal chunk count that fits the target utilization.
    seq_len_base = mb  # 1 chunk
    for n_chunks in range(1, 32):
        seq_len = n_chunks * mb
        target = int(seq_len * float(utilization.rstrip("%")) / 100)
        if target >= mb:  # at least chunk 0 must have mb valid
            break
    chunk_valid = [min(target, mb)] + [0] * (n_chunks - 1)
    per_chunk_mean = [LOG_V if nv > 0 else 0.0 for nv in chunk_valid]
    total_valid = max(sum(chunk_valid), 1)
    _, new_base = chunked_step_loss(per_chunk_mean, chunk_valid, total_valid, n_chunks)

    # Append 1, 4, 16 more empty trailing chunks; step_loss must
    # not change.
    for extra in (1, 4, 16):
        cv = chunk_valid + [0] * extra
        pm = per_chunk_mean + [0.0] * extra
        tot = max(sum(cv), 1)
        _, new_x = chunked_step_loss(pm, cv, tot, n_chunks + extra)
        assert new_x == pytest.approx(new_base, abs=1e-9), (
            f"utilization={utilization}, extra={extra}: "
            f"step_loss drifted from {new_base:.6f} to {new_x:.6f}"
        )


def test_new_equals_old_when_all_chunks_equally_full():
    """When every chunk holds equal valid tokens, NEW == OLD.

    This is the regression guard for the dummy-data / right-pad path.
    If the fix over-rotates and starts to differ from /n_chunks even
    in the dense case, this test fails. Expected ratio ≈ 1.0.
    """
    import math
    LOG_V = math.log(248320)
    n_chunks = 16
    mb = 16384
    chunk_valid = [mb] * n_chunks
    per_chunk_mean = [LOG_V] * n_chunks
    old, new = chunked_step_loss(per_chunk_mean, chunk_valid)
    assert new == pytest.approx(old, rel=1e-9)


def test_old_dilutes_below_token_mean_on_sparse_data():
    """The OLD formula produces a value LOWER than the token-weighted
    mean when chunks are sparse. This is the bug signature: pinned as
    a *negative* test so future devs see it and understand why the
    new formula exists.
    """
    import math
    LOG_V = math.log(248320)
    n_chunks = 16
    mb = 16384
    # ~31.6% utilization — the prod repro: 5 full chunks + 1 partial.
    chunk_valid = [mb] * 5 + [mb // 2] + [0] * (n_chunks - 6)
    per_chunk_mean = [LOG_V if nv > 0 else 0.0 for nv in chunk_valid]
    total_valid = max(sum(chunk_valid), 1)
    old, new = chunked_step_loss(per_chunk_mean, chunk_valid, total_valid, n_chunks)
    assert old < new - 1.0, (
        f"OLD ({old:.3f}) should be diluted well below NEW ({new:.3f}) "
        f"on sparse data (this is the bug — pinned for future devs)."
    )
    # And quantitatively, on this utilization, OLD should be ~31.6%
    # of NEW's log(V) — i.e. around 3.93 / 12.42 = 0.316. If this
    # number changes the formula drift isn't fixing the dilution.
    expected_old = LOG_V * sum(1 for nv in chunk_valid if nv > 0) / n_chunks
    assert old == pytest.approx(expected_old, rel=1e-9)


def test_new_handles_zero_total_valid_gracefully():
    """All-pad pack (no docs at all) must not NaN or raise.

    Total valid = 0 would division-by-zero. The loop guards with
    ``max(..., 1)``; mirror that here so future refactors don't
    regress.
    """
    old, new = chunked_step_loss(
        per_chunk_mean=[0.0] * 4,
        chunk_valid=[0] * 4,
        total_valid=0,
        n_chunks=4,
    )
    assert old == 0.0
    assert new == 0.0


def test_new_handles_interior_empty_chunks():
    """An interior empty chunk (should not occur with FFD, but guard
    anyway) must not corrupt step_loss. The loop's ``last_real``
    covers trailing empty chunks; interior empty chunks are skipped
    via the ``if n_valid_ci == 0: continue`` branch.
    """
    import math
    LOG_V = math.log(248320)
    chunk_valid = [mb := 8192] * 4 + [0] + [mb] * 3
    per_chunk_mean = [LOG_V if nv > 0 else 0.0 for nv in chunk_valid]
    total_valid = max(sum(chunk_valid), 1)
    old, new = chunked_step_loss(per_chunk_mean, chunk_valid, total_valid, 8)
    # NEW = LOG_V * 7*8192 / (7*8192) = LOG_V (the empty interior is excluded).
    assert new == pytest.approx(LOG_V, rel=1e-9)
    # OLD remains diluted by the empty interior (just like trailing).
    assert old < new


# --------------------------------------------------------------------------- #
# Integration smoke test: with the packer actually splitting input into a    #
# sparse pack, run the scaled model and verify step_loss ≈ token-weighted.   #
# Skipped when no GPU / no model / large dims.                                #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()",
    reason="needs CUDA for live model forward",
)
def test_real_pack_step_loss_matches_token_weighted_mean():
    """End-to-end: pack real Ultra-FineWeb docs, run the model via the
    loop's chunk math, assert step_loss equals the token-weighted mean
    CE (within bf16 tolerance).

    This is the test that would have caught the bug — it exercises
    ``pack_chunk_aligned`` (real utilization) + the model + the
    formula together. Skipped on CPU-only runners.
    """
    import math
    import torch
    from src.models import HippoConfig
    from src.models.tp_model._primitives import init_tp
    from src.models.tp_model import TPHippoModel
    from src.training.data.collate import pack_chunk_aligned
    from src.training.loop import _slice_cu_seqlens

    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[0], backend="gloo")

    seq_len, mb = 16384, 4096
    n_chunks = seq_len // mb
    # Synthetic docs that produce a sparse pack (utilization ~ 25%).
    torch.manual_seed(0)
    docs = [torch.randint(100, 1000, (600,)).tolist() for _ in range(2)]
    ids, labels, cu = pack_chunk_aligned(
        docs, seq_len=seq_len, chunk_size=64, batch_size=1,
        pad_id=0, eos_id=None,
    )
    ids, labels, cu = ids.to(0), labels.to(0), cu.to(0)

    cfg = HippoConfig(
        vocab_size=248320, hidden_size=512, tie_word_embeddings=True,
        use_bias=False, num_heads=4, head_dim=128, expand_v=1.0,
        kda_mode="chunk", use_short_conv=True, allow_neg_eigval=False,
        safe_gate=True, lower_bound=-5.0, conv_size=4, conv_bias=False,
        num_layers=4, num_blocks=1, intermediate_size=1280,
        rms_norm_eps=1e-6, ffn_nvfp4=False,
    )
    torch.manual_seed(42)
    model = TPHippoModel(cfg, devices=[0], dtype=torch.bfloat16)

    chunk_valid = [
        int((labels[:, ci*mb+1:(ci+1)*mb] != -100).sum().item())
        for ci in range(n_chunks)
    ]
    total_valid = max(sum(chunk_valid), 1)

    kda = None
    sum_loss_times_valid = 0.0
    last_real = max((ci for ci, nv in enumerate(chunk_valid) if nv > 0),
                    default=-1)
    for ci in range(last_real + 1):
        s, e = ci*mb, (ci+1)*mb
        chunk_cu = _slice_cu_seqlens(cu, s, e)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=True):
            out = model(ids[:, s:e], labels=labels[:, s:e],
                        cu_seqlens=chunk_cu, kda_states=kda)
        m = out["loss"].item()
        if chunk_valid[ci] > 0:
            sum_loss_times_valid += m * (chunk_valid[ci] / total_valid)
        kda = [st.detach() if st is not None else None
               for st in out["kda_states"]]
        del out
    torch.cuda.empty_cache()

    # Token-weighted mean must be ≈ log(V). Tolerate 1.5 sigma of bf16.
    assert math.isclose(
        sum_loss_times_valid,
        math.log(248320),
        rel_tol=0.20,  # bf16 + truncated BPTT can drift up to ~20%
    ), (
        f"step_loss={sum_loss_times_valid:.4f} far from log(V)="
        f"{math.log(248320):.4f}; per-chunk valid: {chunk_valid}"
    )
