"""Regression test: TPFusedLceLoss produces the SAME loss on every rank
when TP > 1.

Background
----------
Before the fix in :mod:`src.models.ops._vendored.fla.modules.
fused_linear_cross_entropy`
(``fused_linear_cross_entropy_tp_forward`` + ``cross_entropy_kernel_tp``),
``_TiedFusedLCEFunction.forward`` called the stock
``fused_linear_cross_entropy_forward`` directly with the rank-local
embed slice as ``weight``. The stock kernel reads
``logits[b_y]`` where ``b_y`` is the GLOBAL target id and ``logits``
has shape ``[N, V // world]`` — out-of-bounds for any token whose
target lives in another rank's slice. Worse: the kernel computes
lse over the rank-local slice only, so the loss is wrong even for
tokens whose target IS in this rank's slice (missing V - V/world
contributions to the softmax denominator).

Symptoms (June 2026):
  - Sporadic mb loss = -4.8e24, -1.5e35, 1.35e25, -7.9e33, -4.3e27
  - Both signs (positive AND negative)
  - Only TP > 1; TP = 1 unaffected
  - Optimizer state stays bounded (the corrupted gradient gets the
    Inf/NaN guard in ``_run_training_loop`` and the step is skipped
    via ``found_inf=True``)

Why the bug slips through: the OOB read often lands on memory with
normal-ish values (adjacent chunk's logits, padding), so a single
run may LOOK fine. Two consecutive runs at TP=2 with the same seed
differ — see ``project_tp_loss_oob_bug.md`` for the full diagnosis.

What this test pins
-------------------
For the exact same hidden state + target + weight, after going through
the TP-aware ``fused_linear_cross_entropy_tp_forward`` kernel:

  1. **Rank parity:** rank 0's loss equals rank 1's loss (the historical
     bug produced DIFFERENT losses per rank, varying per run).
  2. **TP-2 == TP-1:** the TP=2 loss equals the reference TP=1 loss
     (computed via the stock ``fused_linear_cross_entropy_forward``)
     to bf16 noise. With the stock kernel on TP=2, these would have
     diverged by huge amounts (the OOB read returning random data).
  3. **Finite:** both ranks' losses are finite (no NaN, no Inf).
  4. **Multi-seed:** repeat across N random seeds so a flaky OOB
     read that happens to return reasonable values doesn't sneak
     past a single trial.

Why mp.spawn
------------
``fused_linear_cross_entropy_tp_forward`` calls real
``torch.distributed.all_gather`` and ``torch.distributed.all_reduce``,
which require a real distributed group with ``world_size == 2``.
Single-process mocks would not exercise the actual collectives.

The gloo backend is used (matching ``configs/test/tp2.yml``'s
``tp_sim=true, tp_size=2``): same sharding math as NCCL, only the
transport differs. Two child processes are spawned, both bound to
the same physical GPU (``cuda:0``); each calls the kernel with its
rank-specific vocab slice, then reports its loss to the parent via
a file on disk (mp.Queue would also work; file is simplest).

Run:
    python -m pytest test/test_tp_lce_rank_parity.py -v

Hardware: needs CUDA. Skips on CPU-only runners.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# --------------------------------------------------------------------------- #
# Skip gating.                                                                #
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 1


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="TP-rank-parity test needs CUDA; not available",
)


# --------------------------------------------------------------------------- #
# Child worker. Runs in a separate process (one per rank).                    #
# --------------------------------------------------------------------------- #
def _child_run(rank: int, port: int, weights_path: str, hidden_path: str,
               target_path: str, loss_path: str, world_size: int = 2) -> None:
    """Spawned child: load the shared inputs from disk, init the TP
    group, run the TP-aware loss kernel, write the loss to disk.

    All inputs were serialized by the parent BEFORE spawn so every
    rank sees the exact same hidden / target / full-vocab weight.
    Each rank only USES its slice of the weight; the kernel's
    all_gather / all_reduce ensure the loss is computed against the
    full global vocab.
    """
    import torch.distributed as dist

    # Pin device + set up the env-var-driven process group.
    torch.cuda.set_device(0)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Load shared inputs (serialized by parent).
        weight = torch.load(weights_path, map_location="cuda:0")
        hidden = torch.load(hidden_path, map_location="cuda:0")
        target = torch.load(target_path, map_location="cuda:0")

        V, H = weight.shape
        assert V % world_size == 0, (
            f"vocab_size={V} not divisible by world_size={world_size}"
        )
        vp = V // world_size
        # Rank r's vocab slice is ``weight[r*vp : (r+1)*vp]``.
        local_w = weight[rank * vp : (rank + 1) * vp].contiguous()

        from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_tp_forward,
        )

        # ``tp_group=dist.group.WORLD`` is what the kernel needs.
        loss, _dx, _dw, _db = fused_linear_cross_entropy_tp_forward(
            hidden, target, local_w, None,
            ignore_index=-100, num_chunks=8,
            tp_group=dist.group.WORLD,
            tp_rank=rank,
            tp_world=world_size,
        )
        # Serialize the scalar loss. Use float() so the parent
        # doesn't need to deserialize a CUDA scalar.
        torch.save(loss.detach().float().cpu(), loss_path)
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Reference (TP=1) — what the loss SHOULD equal at TP=2.                      #
# --------------------------------------------------------------------------- #
def _reference_loss_tp1(weight: torch.Tensor, hidden: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Compute the same loss via the stock (non-TP) kernel, using
    the FULL vocab. The TP=2 result must match this within bf16.

    Inputs may be on CPU or CUDA; we move them to CUDA inside this
    helper so the Triton kernel can access them.
    """
    from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import (
        fused_linear_cross_entropy_forward,
    )
    weight = weight.to("cuda:0", non_blocking=True)
    hidden = hidden.to("cuda:0", non_blocking=True)
    target = target.to("cuda:0", non_blocking=True)
    loss, _dx, _dw, _db = fused_linear_cross_entropy_forward(
        hidden, target, weight, None,
        ignore_index=-100, num_chunks=8,
        reduction="mean",
    )
    return loss.detach().float().cpu()


# --------------------------------------------------------------------------- #
# Tests.                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tp2_ranks_produce_same_loss(seed: int) -> None:
    """For the same inputs, rank 0's loss must equal rank 1's loss.

    This is the property that the OOB bug breaks: the stock kernel
    reads ``logits[b_y]`` with a global target id on a
    ``[N, V//world]`` logits buffer, so off-rank tokens return
    garbage that varies per rank and per microsecond (depending on
    what memory happens to sit past the buffer end). Two consecutive
    trials with the same seed would produce DIFFERENT losses per
    rank — see ``project_tp_loss_oob_bug.md``.
    """
    import torch.multiprocessing as mp

    # ---- Shape choices: kept small so the test runs in <2s. ----
    # The bug fires at any size (V=2 is enough for one rank to own
    # a token and the other to OOB-read), so we don't need prod
    # dimensions. V must be a multiple of 2.
    V, H, N = 64, 16, 32

    torch.manual_seed(seed)
    weight = torch.randn(V, H, dtype=torch.bfloat16, device="cpu")
    hidden = torch.randn(N, H, dtype=torch.bfloat16, device="cpu")
    target = torch.randint(0, V, (N,), dtype=torch.long, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        weights_path = str(tmp / "weight.pt")
        hidden_path = str(tmp / "hidden.pt")
        target_path = str(tmp / "target.pt")
        loss0_path = str(tmp / "loss0.pt")
        loss1_path = str(tmp / "loss1.pt")
        torch.save(weight, weights_path)
        torch.save(hidden, hidden_path)
        torch.save(target, target_path)

        port = 29500 + (os.getpid() % 1000)
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_child_run, args=(0, port, weights_path,
                                                 hidden_path, target_path,
                                                 loss0_path)),
            ctx.Process(target=_child_run, args=(1, port, weights_path,
                                                 hidden_path, target_path,
                                                 loss1_path)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            # 60s is generous; the test runs in <2s on a 5060 Ti.
            p.join(timeout=60)
            assert not p.is_alive(), (
                f"rank worker {p} did not finish within 60s — "
                f"likely a gloo init hang"
            )
            assert p.exitcode == 0, (
                f"rank worker {p} exited with code {p.exitcode}; "
                f"see its stderr for the kernel exception"
            )

        loss0 = torch.load(loss0_path)
        loss1 = torch.load(loss1_path)

    # (1) Finite.
    assert torch.isfinite(loss0).item(), f"rank 0 loss is not finite: {loss0.item()}"
    assert torch.isfinite(loss1).item(), f"rank 1 loss is not finite: {loss1.item()}"

    # (2) Rank parity — the property the OOB bug breaks.
    #
    # Bit-exact equality is the strongest statement: with the fix,
    # both ranks compute the SAME loss via the all_gather + all_reduce
    # contract (loss is all-reduced SUM, then /total_global — fully
    # symmetric). bf16 tolerance only because the kernel may use
    # different reduction orders on each rank; the all-reduce itself
    # is deterministic so the final loss is bit-exact in practice.
    diff_abs = (loss0 - loss1).abs().item()
    assert diff_abs < 1e-3, (
        f"rank 0 loss={loss0.item():.6e} != rank 1 loss={loss1.item():.6e} "
        f"(abs diff={diff_abs:.4e}). This is the TP-loss-OOB regression "
        f"from project_tp_loss_oob_bug.md — the stock FusedLinearCE kernel "
        f"reads logits[b_y] OOB for off-rank targets, producing per-rank "
        f"losses that differ. Did the TP-aware path in "
        f"fused_linear_cross_entropy_tp_forward regress?"
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tp2_loss_matches_tp1_reference(seed: int) -> None:
    """The TP=2 loss must equal the stock TP=1 reference loss
    (within bf16). With the OOB bug, the TP=2 loss is essentially
    random garbage — ``abs diff`` on the order of 1e3 or larger.
    """
    import torch.multiprocessing as mp

    V, H, N = 64, 16, 32

    torch.manual_seed(seed)
    weight = torch.randn(V, H, dtype=torch.bfloat16, device="cpu")
    hidden = torch.randn(N, H, dtype=torch.bfloat16, device="cpu")
    target = torch.randint(0, V, (N,), dtype=torch.long, device="cpu")

    ref_loss = _reference_loss_tp1(weight, hidden, target).item()
    assert torch.isfinite(torch.tensor(ref_loss)).item(), (
        f"TP=1 reference loss is not finite: {ref_loss} — test setup error"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        weights_path = str(tmp / "weight.pt")
        hidden_path = str(tmp / "hidden.pt")
        target_path = str(tmp / "target.pt")
        loss0_path = str(tmp / "loss0.pt")
        loss1_path = str(tmp / "loss1.pt")
        torch.save(weight, weights_path)
        torch.save(hidden, hidden_path)
        torch.save(target, target_path)

        port = 29500 + (os.getpid() % 1000)
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_child_run, args=(0, port, weights_path,
                                                 hidden_path, target_path,
                                                 loss0_path)),
            ctx.Process(target=_child_run, args=(1, port, weights_path,
                                                 hidden_path, target_path,
                                                 loss1_path)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert not p.is_alive(), (
                f"rank worker {p} did not finish within 60s"
            )
            assert p.exitcode == 0, (
                f"rank worker {p} exited with code {p.exitcode}"
            )

        loss0 = torch.load(loss0_path).item()
        loss1 = torch.load(loss1_path).item()

    # Both ranks must agree with the TP=1 reference within bf16 noise.
    # The TP=2 path computes the GLOBAL CE via all_gather lse +
    # per-rank target-remapped kernel + all-reduce loss — it should
    # be numerically equivalent to the TP=1 path (modulo bf16
    # reduction-order rounding, <1% rel error).
    for rank_label, loss in (("rank 0", loss0), ("rank 1", loss1)):
        assert torch.isfinite(torch.tensor(loss)).item(), (
            f"{rank_label} loss is not finite: {loss}"
        )
        # Absolute tolerance on the order of 1e-2 — bf16 cross-entropy
        # noise floor at V=64. The bug fires with abs diff > 1e3.
        diff_abs = abs(loss - ref_loss)
        rel_err = diff_abs / max(abs(ref_loss), 1e-6)
        assert diff_abs < 1e-1 or rel_err < 0.05, (
            f"{rank_label} TP=2 loss={loss:.6e} disagrees with TP=1 "
            f"reference={ref_loss:.6e} (abs diff={diff_abs:.4e}, "
            f"rel err={rel_err:.4%}). TP-loss-OOB regression — the "
            f"per-rank kernel is computing against the rank-local "
            f"vocab slice instead of the global vocab."
        )