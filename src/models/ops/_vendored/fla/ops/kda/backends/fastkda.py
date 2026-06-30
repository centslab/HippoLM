"""FastKDABackend: in-tree Triton KDA forward.

Wire the fast (Triton) KDA kernels
(``src.models.ops._triton.fast_kda.prepare`` +
``src.models.ops._triton.fast_kda.recurrence``) into the FLA backend
dispatch. The fast path replaces the multi-kernel FLA chunk path
(prepare + recurrence) with a single fused pair — ~1.6x end-to-end
fwd speedup at prod shape (see
``docs/fast_kda_bottleneck_analysis.md``).

Constraints (enforced by the verifier):
  - K = V = 128 (hard-coded in the Triton kernels; the algorithm's
    design is tied to CHUNK=16 × K=128)
  - bf16 only (KDA contract)
  - No varlen (cu_seqlens must be None) — the Triton kernels do not
    yet support per-document boundary resets
  - No GVA: HV must equal H (the kernels index heads directly)
  - L2 norm + gate + beta-sigmoid fused in-kernel → all three
    ``*_in_kernel`` flags must be True
  - safe_gate + lower_bound set (the Neumann-series safety contract;
    see ``prepare.py`` for the math)
  - No cp_context, no return_intermediate_states
  - Inference mode only: ``not torch.is_grad_enabled()`` (the fast
    kernels are fwd-only — the bwd path stays on the FLA default)

Toggled via ``FLA_FAST_KDA=0`` env var (or by setting
``default_enable=False`` here). The fast backend takes priority over
``FlashKDABackend`` (priority 2 vs 3) so the in-tree Triton path wins
when both are eligible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._triton.fast_kda.recurrence import kda_recurrence_triton
from src.models.ops._vendored.fla.ops.backends import BaseBackend

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class FastKDABackend(BaseBackend):
    """Triton fused-KDA forward (in-tree; no external deps).

    The fast kernel pair (prepare + recurrence) replaces the vendored
    FLA chunk path's intra-solve + delta-h + chunk-o stages with two
    kernels that do the work in registers and never round-trip through
    gmem for the inter-chunk intermediates. See
    ``docs/fast_kda_bottleneck_analysis.md`` for the perf analysis
    and ``src/models/ops/_triton/fast_kda/`` for the kernel sources.
    """

    backend_type = "fastkda"
    package_name = None  # in-tree, no external dep
    env_var = "FLA_FAST_KDA"
    default_enable = True
    # Higher priority than FlashKDABackend (priority 3) so the in-tree
    # Triton path wins when both are eligible and the package is
    # available. FLA default has priority 5 (registered first).
    priority = 2

    def chunk_kda_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
        **kwargs,
    ) -> tuple[bool, str | None]:
        # Bwd path: the fast kernels are fwd-only. Training (grad
        # enabled) must fall through to the FLA default chunk path
        # which has a real bwd.
        if torch.is_grad_enabled():
            return False, "FastKDA is fwd-only; training falls through to FLA default"
        if q.dtype != torch.bfloat16:
            return False, f"FastKDA requires bfloat16, got {q.dtype}"
        if q.shape[-1] != 128:
            return False, f"FastKDA requires K=128, got {q.shape[-1]}"
        if v.shape[-1] != 128:
            return False, f"FastKDA requires V=128, got {v.shape[-1]}"
        if v.shape[2] != q.shape[2]:
            return False, f"FastKDA does not support GVA (HV={v.shape[2]} != H={q.shape[2]})"
        if not use_gate_in_kernel:
            return False, "FastKDA requires use_gate_in_kernel=True (gate is fused into prepare)"
        if not use_qk_l2norm_in_kernel:
            return False, "FastKDA requires use_qk_l2norm_in_kernel=True (L2 norm is fused into prepare)"
        if not use_beta_sigmoid_in_kernel:
            return False, "FastKDA requires use_beta_sigmoid_in_kernel=True (sigmoid is fused into prepare)"
        if not safe_gate or lower_bound is None:
            return False, "FastKDA requires safe_gate=True and a lower_bound (Neumann-series safety contract)"
        if cu_seqlens is not None:
            return False, "FastKDA does not yet support varlen (cu_seqlens)"
        if cp_context is not None:
            return False, "FastKDA does not support context parallel"
        if return_intermediate_states:
            return False, "FastKDA does not support return_intermediate_states"
        return True, None

    def chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        **kwargs,
    ):
        if scale is None:
            scale = q.shape[-1] ** -0.5

        if A_log is None:
            raise ValueError("FastKDA requires A_log (the log-space decay parameter)")

        # Workspace: prepare runs all the per-chunk work (gate
        # activation, q/k L2 norm, exp decay apply, INV via Neumann
        # series, Round-9 fold-in to Mqk_eff and K_pre). Returns 5
        # workspace tensors (Round-9 dropped the standalone INV tensor
        # since Mqk_eff and K_pre fully capture its information content
        # for the recurrence).
        ws = kda_prepare_triton(
            q=q, k=k, g=g, beta=beta,
            A_log=A_log, dt_bias=dt_bias,
            lower_bound=lower_bound, scale=scale,
        )

        # Recurrence: cross-chunk state evolution + per-token O.
        # Round-9 inputs: k_decayed (for v_residual subtraction), beta
        # (for v_residual_b scaling), K_pre (INV @ k_restored), Mqk_eff
        # (Mqk @ INV). Returns (o [B, T, H, V] bf16, h_intermediate, final_state).
        o, _h_intermediate, final_state = kda_recurrence_triton(
            k_decayed=ws["k_decayed"],
            q_decayed=ws["q_decayed"],
            K_pre=ws["K_pre"],
            g_total=ws["g_total"],
            mqk_eff=ws["Mqk_eff"],
            beta=beta,
            v=v,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

        # FLA convention: o.dtype matches q, final_state is fp32.
        return o.type_as(q), final_state
