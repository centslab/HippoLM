"""Column-row parallel SwiGLU for the TP path.

Fused gate+up projection (one ``ColumnParallelLinear`` with
output = ``2 * intermediate_size``) + row-parallel down projection
with all-reduce. Saves 1 kernel launch per SwiGLU vs the unfused
3-projection baseline.

Precision is scheme-driven via ``config.ffn_precision``
(``HippoConfig``, see the 2026-07-21 5-scheme migration):

  * ``"w16a16"`` (default) — plain ``ColumnParallelLinear`` /
                             ``RowParallelLinear`` (BF16 GEMM).
  * ``"w8a16"``  — no W8A16 FFN kernel on sm_120 today; falls
                   back to plain BF16 TP with a logged warning.
                   Listed for completeness; not used by the
                   canonical ``base.yml`` config.
  * ``"w8a8"``   — no per-rank FP8 column/row-parallel Linear on
                   sm_120 today (the W8A8 path is single-GPU only
                   via ``FP8Linear``); falls back to plain BF16
                   TP with a logged warning.
  * ``"w4a8"``   — NVFP4 W4A8 two-pass, no TP variant implemented
                   on sm_120 today (``NVFP4ColumnParallelLinear``
                   and ``NVFP4RowParallelLinear`` are the W4A16
                   Marlin variants). Falls back to plain BF16 TP
                   with a logged warning until the W4A8 TP
                   variants ship.
  * ``"w4a16"``  — ``NVFP4ColumnParallelLinear`` /
                   ``NVFP4RowParallelLinear`` with Marlin mode-3
                   (``use_marlin=True, no_bf16_master=True`` —
                   hardcoded in the wrapper, not user-
                   controllable). The canonical prod FFN scheme.

The legacy boolean flags ``ffn_nvfp4`` / ``ffn_nvfp4_marlin`` /
``ffn_nvfp4_no_bf16_master`` were removed on 2026-07-21; the
scheme is the single source of truth.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._primitives import ColumnParallelLinear, RowParallelLinear
from src.models.ops.nvfp4_tp import NVFP4ColumnParallelLinear, NVFP4RowParallelLinear


class TPSwiGLU(nn.Module):
    """Column-row parallel SwiGLU.

    gate_proj:  hidden  -> intermediate (column parallel, sharded)
    up_proj:    hidden  -> intermediate (column parallel, sharded)
    down_proj:  intermediate (sharded) -> hidden (row parallel, all-reduce)
    """

    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()
        scheme = getattr(config, "ffn_precision", "w16a16")
        ColCls, RowCls, extra = self._resolve_classes(scheme)
        # Fused gate+up projection: one ColumnParallelLinear with
        # output = 2 * intermediate_size. The first ``intermediate``
        # output channels are the gate, the next ``intermediate``
        # are the up. This halves the matmul kernel-launch count
        # vs. two separate ColumnParallelLinear's (3 -> 2 per
        # SwiGLU). Bias: when ``use_bias`` is True the fused bias
        # is one tensor of size ``2 * intermediate_per_partition``
        # on this rank (gate half then up half); we expose it as
        # ``self.gate_up_bias`` so the forward can split it.
        col_kwargs = {"bias": config.use_bias, "device": device, "dtype": dtype}
        row_kwargs = {"bias": config.use_bias, "device": device, "dtype": dtype}
        col_kwargs.update(extra.get("col", {}))
        row_kwargs.update(extra.get("row", {}))
        self.gate_up_proj = ColCls(
            config.hidden_size, 2 * config.intermediate_size, **col_kwargs,
        )
        self.down_proj = RowCls(
            config.intermediate_size, config.hidden_size, **row_kwargs,
        )

    @staticmethod
    def _resolve_classes(scheme: str):
        """Map the FFN precision scheme to (ColCls, RowCls, extra kwargs).

        Each scheme's kernel choice is hardcoded — no per-config
        knobs. Per-rank FP8 / NVFP4 Column/Row-Parallel on sm_120
        is not implemented today, so w8a8 / w4a8 fall back to
        plain BF16 TP with a logged warning.
        """
        if scheme == "w16a16":
            return (
                ColumnParallelLinear, RowParallelLinear,
                {"col": {}, "row": {}},
            )
        if scheme == "w8a16":
            # No W8A16 FFN kernel on sm_120 today. Fall back to
            # BF16 TP with warning.
            warnings.warn(
                "ffn_precision='w8a16' is not supported on the TP path "
                "(no W8A16 FFN kernel); falling back to 'w16a16' (BF16 TP).",
                RuntimeWarning, stacklevel=2,
            )
            return (
                ColumnParallelLinear, RowParallelLinear,
                {"col": {}, "row": {}},
            )
        if scheme == "w8a8":
            # No per-rank FP8 Column/Row-Parallel on sm_120 today.
            warnings.warn(
                "ffn_precision='w8a8' is not supported on the TP path "
                "(no FP8 Column/Row-Parallel on sm_120); falling back to "
                "'w16a16' (BF16 TP).",
                RuntimeWarning, stacklevel=2,
            )
            return (
                ColumnParallelLinear, RowParallelLinear,
                {"col": {}, "row": {}},
            )
        if scheme == "w4a8":
            # No W4A8 NVFP4 Column/Row-Parallel on sm_120 today.
            # (NVFP4ColumnParallelLinear is the W4A16 Marlin
            # variant — not the W4A8 two-pass.) Fall back to BF16
            # TP until the W4A8 TP variants ship.
            warnings.warn(
                "ffn_precision='w4a8' is not supported on the TP path "
                "(no NVFP4 W4A8 Column/Row-Parallel on sm_120); falling "
                "back to 'w16a16' (BF16 TP).",
                RuntimeWarning, stacklevel=2,
            )
            return (
                ColumnParallelLinear, RowParallelLinear,
                {"col": {}, "row": {}},
            )
        if scheme == "w4a16":
            # NVFP4 mode-3 Marlin. ``use_marlin`` and
            # ``no_bf16_master`` are hardcoded — the kernel
            # choice IS the scheme.
            return (
                NVFP4ColumnParallelLinear, NVFP4RowParallelLinear,
                {
                    "col": {"use_marlin": True, "no_bf16_master": True},
                    "row": {"use_marlin": True, "no_bf16_master": True},
                },
            )
        raise AssertionError(
            f"ffn_precision must be one of 'w16a16', 'w8a16', 'w8a8', "
            f"'w4a16', 'w4a8'; got {scheme!r}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is replicated (full hidden). gate_up_proj produces
        # sharded intermediate slices on this rank: the first
        # ``inter_per_partition`` channels are the gate, the next
        # ``inter_per_partition`` are the up. down is row-parallel
        # and all-reduces back to full hidden.
        gu = self.gate_up_proj(x)
        inter_per_partition = self.gate_up_proj.out_features_per_partition // 2
        gate, up = gu.split(inter_per_partition, dim=-1)
        return self.down_proj(F.silu(gate) * up)