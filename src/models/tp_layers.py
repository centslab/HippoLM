"""Megatron-style Tensor Parallel layers for HippoLM.

Implements ColumnParallelLinear (split output dim, no comm) and
RowParallelLinear (split input dim, all-reduce output). These are
the building blocks of column-row parallel pairs (e.g. SwiGLU FFN
and KDA Q/K/V/O projections) and of the column-parallel lm_head.

TP is implemented as a single process managing N CUDA devices. The
NCCL process group is initialized once at startup and reused for
all communication. World size is fixed at init time; the layers
read it via ``get_tp_world_size()``.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Process-group state. Set once by init_tp(); queried by every TP op.         #
# --------------------------------------------------------------------------- #
_TP_GROUP = None
_TP_WORLD_SIZE = 1
_TP_RANK = 0


def get_tp_world_size() -> int:
    return _TP_WORLD_SIZE


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_group():
    return _TP_GROUP


def init_tp(world_size: int, devices: list[int], backend: str = "nccl") -> None:
    """Initialize the single-process TP group spanning ``world_size`` GPUs.

    Uses ``torch.distributed`` with a single rank 0 process owning
    ``world_size`` devices. NCCL handles the cross-device comms
    on the production 8xV100 box; the ``gloo`` backend is used
    during development on a single-GPU box (``--tp_sim``) to
    exercise the same sharding code paths without NVLink.

    We must call ``torch.cuda.set_device(devices[rank])`` before
    touching any tensor on that device, so the rank's "default"
    stream is bound correctly. The TP layers below always use
    ``param.device`` rather than relying on a default device, so
    this matters mainly for collective ops.

    ``device_id`` is intentionally NOT passed to
    ``init_process_group`` here: only NCCL needs it, and the actual
    ``dist.init_process_group`` call lives in :func:`_train_worker`
    (which DOES pin the NCCL comm to ``gpus[rank]``). For gloo
    the call is the same minus ``device_id``.
    """
    import torch.distributed as dist

    global _TP_GROUP, _TP_WORLD_SIZE, _TP_RANK
    assert _TP_GROUP is None, "init_tp called twice"

    os.environ.setdefault("MASTER_ADDR", "localhost")
    # Pick a port that's unlikely to clash. Use the same convention
    # as torchrun defaults so external launches don't break.
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", "0")

    if not dist.is_initialized():
        # ``backend`` may be "nccl" (production) or "gloo" (TP
        # simulation on a single-GPU dev box). gloo does its comm
        # on host memory and stages CUDA tensors through it, so it
        # has no concept of ``device_id``; NCCL would require it,
        # but that path is handled in :func:`_train_worker` before
        # we get here, so this branch is a fallback for callers
        # that init from scratch.
        if backend == "nccl":
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=0,
                world_size=world_size,
                device_id=torch.device(f"cuda:{devices[0]}"),
            )
        else:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=0,
                world_size=world_size,
            )
    _TP_GROUP = dist.group.WORLD
    _TP_WORLD_SIZE = world_size
    _TP_RANK = 0  # single-process; rank is the device slot
    # NOTE: do NOT call ``set_device`` here. Each worker in the TP
    # group has already called ``set_device(gpus[rank])`` before
    # init_tp, and the NCCL process group was initialised with
    # ``device_id=cuda:{gpus[rank]}``. Pinning the current device
    # back to ``devices[0]`` (always cuda:5 in the 8xV100 layout)
    # silently desyncs every rank's ``current_device()`` from the
    # NCCL backend's device constraint, and the next all-reduce
    # on a tensor allocated via ``torch.cuda.current_device()``
    # blows up with "Tensor found on device cuda:5 but backend
    # constrained to cuda:6". The TP layers themselves always
    # reference ``param.device`` rather than the default device,
    # so we don't need this set_device call at all.


def shutdown_tp() -> None:
    import torch.distributed as dist
    global _TP_GROUP, _TP_WORLD_SIZE, _TP_RANK
    if dist.is_initialized():
        dist.destroy_process_group()
    _TP_GROUP = None
    _TP_WORLD_SIZE = 1
    _TP_RANK = 0


# --------------------------------------------------------------------------- #
# TP linear layers                                                            #
# --------------------------------------------------------------------------- #
class _AllReduce(torch.autograd.Function):
    """Forward: all-reduce. Backward: identity (sum's gradient is 1).

    Implementation note: NCCL's ``all_reduce`` is in-place (the
    output buffer is the same as the input buffer). The earlier
    ``out = x.clone()`` was a leftover from a transport that needed
    a separate output buffer; NCCL doesn't, and the clone was
    adding an extra full-tensor copy per all-reduce (which on a
    ``[B, T, H] = [4, 4096, 1024]`` tensor is 32 MB at FP16).
    In-place is safe here because the autograd graph never uses
    ``x`` again after this op (the next node consumes the
    all-reduce's output).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if _TP_WORLD_SIZE == 1:
            return x
        import torch.distributed as dist
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_TP_GROUP)
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        # d/dx sum_i x_i = 1 for each x_i, so each rank's grad is
        # passed through unchanged. (We do NOT all-reduce the grad;
        # the upstream graph will route it to the right sharded
        # inputs.)
        return grad


def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return _AllReduce.apply(x)


class _AllReduceSum(torch.autograd.Function):
    """All-reduce SUM. Backward: identity (alias of :class:`_AllReduce`).

    Kept as a separate class so the call site reads
    ``tp_all_reduce_sum`` (rather than ``tp_all_reduce``) and so the
    backward contract is documented at the function level.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if _TP_WORLD_SIZE == 1:
            return x
        import torch.distributed as dist
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_TP_GROUP)
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return grad


def tp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    return _AllReduceSum.apply(x)


class _AllReduceMax(torch.autograd.Function):
    """All-reduce MAX. Backward: straight-through to the rank that held the max.

    Forward computes ``out = max_r x_r`` across the TP group and saves
    the *local* input plus the *global* output. Backward is the
    subgradient of a max reduction: only the rank(s) that contributed
    the max value receive the upstream gradient; others get zero. Ties
    (multiple ranks with the same value as the global max) split the
    gradient among them — this is the standard straight-through
    estimator for max.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if _TP_WORLD_SIZE == 1:
            # No comm: the global max equals the local value. Save both
            # pointers (x and the same x) so backward can unpack a
            # uniform 2-tuple regardless of world size.
            ctx.save_for_backward(x, x)
            return x
        import torch.distributed as dist
        # ``_AllReduceMax`` needs the pre-reduce ``x`` for the
        # backward mask, so we still need a copy here. The copy
        # is unavoidable; the in-place trick from ``_AllReduce``
        # would clobber the saved tensor.
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.MAX, group=_TP_GROUP)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        x, out = ctx.saved_tensors
        # 1 where this rank's value was the global max, 0 elsewhere.
        mask = (x == out).to(grad.dtype)
        return grad * mask


def tp_all_reduce_max(x: torch.Tensor) -> torch.Tensor:
    return _AllReduceMax.apply(x)


def _slice(t: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    """Split ``t`` along ``dim`` into ``world`` equal pieces and return rank's.

    Falls back to identity when world == 1.
    """
    if world == 1:
        return t
    size = t.size(dim)
    assert size % world == 0, f"dim={dim} size={size} not divisible by world={world}"
    chunk = size // world
    return t.narrow(dim, rank * chunk, chunk).contiguous()


class ColumnParallelLinear(nn.Module):
    """Linear whose output dim is sharded across the TP group.

    Weight shape: ``[in_features, out_features // world]``. No
    communication on forward (output is sharded); on backward the
    input gradient is implicitly summed via the autograd graph of
    the consumer (typically a RowParallelLinear that all-reduces).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world = get_tp_world_size()
        assert out_features % self.world == 0, (
            f"out_features={out_features} not divisible by tp_world={self.world}"
        )
        self.out_features_per_partition = out_features // self.world
        # Store the full logical size so checkpoints / state_dicts look
        # like a normal Linear to a non-TP consumer.
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]; weight: [out/world, in]
        # F.linear computes x @ weight.T -> [..., out/world]
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"out_features_per_partition={self.out_features_per_partition}, "
            f"tp_world={self.world}, bias={self.bias is not None}"
        )


class RowParallelLinear(nn.Module):
    """Linear whose input dim is sharded across the TP group.

    Weight shape: ``[in_features // world, out_features]``. Forward
    is a local matmul producing a partial output, followed by an
    all-reduce. The bias is added on each rank but only the rank-0
    contribution is kept after the all-reduce to avoid double-counting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world = get_tp_world_size()
        assert in_features % self.world == 0, (
            f"in_features={in_features} not divisible by tp_world={self.world}"
        )
        self.in_features_per_partition = in_features // self.world
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition, device=device, dtype=dtype)
        )
        # Bias lives on every rank; we mask it to zero on non-zero
        # ranks so the all-reduce sums to a single effective bias.
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in/world]. Local matmul -> [..., out].
        out = F.linear(x, self.weight)
        if self.bias is not None and _TP_RANK == 0:
            out = out + self.bias
        elif self.bias is not None:
            # Add a zero bias on other ranks so the all-reduce
            # only sees the rank-0 contribution once.
            out = out + 0.0
        return tp_all_reduce(out)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"in_features_per_partition={self.in_features_per_partition}, "
            f"out_features={self.out_features}, "
            f"tp_world={self.world}, bias={self.bias is not None}"
        )


# --------------------------------------------------------------------------- #
# Helpers for inspecting how a tensor would be sharded                        #
# --------------------------------------------------------------------------- #
def partition_slice(total: int) -> tuple[int, int]:
    """Return (start, length) of this rank's shard of a 1D dim of size ``total``."""
    rank = get_tp_rank()
    world = get_tp_world_size()
    chunk = total // world
    return rank * chunk, chunk
