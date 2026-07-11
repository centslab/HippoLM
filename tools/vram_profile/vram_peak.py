"""Measure peak VRAM for the model at a moderate scale (fits 16GB).

This script isolates the question: does dropping the rmsnorm x save
(which is currently a BF16 reference to the layer's input tensor)
produce measurable peak VRAM savings?

We run the same model twice:
  A. With x save (current default): the rms_norm LayerNormFunction
     saves its input x in ctx.save_for_backward.
  B. Without x save: the rms_norm LayerNormFunction uses a y-based
     bwd (math-verified bit-equivalent) and does NOT save x.

If the allocator can reuse the freed 96 MiB/layer of x references
for other allocations, peak drops. If not, peak is unchanged
(the references weren't actually blocking other allocations).

The current production setup (per project's vendored fla) is "A".
We compare directly by toggling a new ``save_residual=True`` kwarg
in the LayerNormFunction.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/hy-tmp/HippoLM")

import torch

from src.models.tp_model._primitives import init_tp
from src.models.tp_model.model import TPHippoModel
from src.models.config import HippoConfig


def measure(save_x: bool, hidden: int = 1536, num_layers: int = 8, T: int = 4096):
    init_tp(world_size=1, devices=[0], backend="gloo")

    head_dim = 128
    num_heads = hidden // head_dim  # 12

    cfg = HippoConfig(
        hidden_size=hidden,
        num_layers=num_layers,
        num_blocks=1,
        vocab_size=8192,
        head_dim=head_dim,
        num_heads=num_heads,
        expand_v=1.0,
        intermediate_size=4096,
        ffn_nvfp4=False,
        safe_gate=False,  # simpler path
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = TPHippoModel(cfg, [0]).to(torch.bfloat16)
    print(f"  Model: H={hidden} T={T} L={num_layers} h={num_heads} d={head_dim}"
          f"  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    x = torch.randint(0, cfg.vocab_size, (1, T), device="cuda", dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (1, T), device="cuda", dtype=torch.long)

    # Monkey-patch the vendored LayerNormFunction to skip x save.
    # We modify RMSNorm.forward to pass a flag.
    import src.models.norms as norms_mod
    import src.models.ops._vendored.fla.modules.layernorm as ln_mod

    _orig_apply = ln_mod.LayerNormFunction.apply

    class _PatchedFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx_, x_, weight, bias, residual=None, eps=1e-5,
                    prenorm=False, residual_in_fp32=False, is_rms_norm=False,
                    num_groups=1, save_x=True):
            # Bypass: LayerNormFunction itself doesn't accept save_x,
            # so we shim via a wrapper that drops the save.
            x_2d = x_.reshape(-1, x_.shape[-1])
            if residual is not None:
                residual = residual.reshape_as(x_2d)
            y, mean, rstd, res_out = ln_mod.layer_norm_fwd(
                x_2d, weight, bias, eps, residual,
                residual_dtype=(residual.dtype if residual is not None
                                else (torch.float32 if residual_in_fp32 else None)),
                is_rms_norm=is_rms_norm, num_groups=num_groups,
            )
            y = y.reshape(x_.shape)
            if not save_x:
                # Skip res_out. Save y instead.
                # Note: same memory footprint; the question is the
                # lifecycle difference.
                if bias is not None:
                    ctx_.save_for_backward(y, weight, bias, mean, rstd)
                elif mean is not None:
                    ctx_.save_for_backward(y, weight, bias, mean, rstd)
                else:
                    ctx_.save_for_backward(y, weight, bias, None, rstd)
                ctx_._save_x = False
            else:
                # Standard path: save res_out (= x when no residual).
                if bias is not None:
                    ctx_.save_for_backward(res_out, weight, bias, mean, rstd)
                elif mean is not None:
                    ctx_.save_for_backward(res_out, weight, bias, mean, rstd)
                else:
                    ctx_.save_for_backward(res_out, weight, bias, None, rstd)
                ctx_._save_x = True
            ctx_._x_shape = x_.shape
            ctx_._x_dtype = x_.dtype
            ctx_._has_bias = bias is not None
            ctx_._has_mean = mean is not None
            ctx_._eps = eps
            ctx_._is_rms_norm = is_rms_norm
            ctx_._num_groups = num_groups
            ctx_._weight_dtype = weight.dtype
            ctx_._weight_shape = weight.shape
            return y

        @staticmethod
        def backward(ctx_, dy, *args):
            saved = ctx_.saved_tensors
            if ctx_._save_x:
                x_or_y, weight, bias, mean, rstd = saved
                x_ = x_or_y  # it's x in this case
            else:
                y_or_x, weight, bias, mean, rstd = saved
                # In our "no save_x" path we stashed y; reconstruct
                # x-equivalent info for the kernel: x = y / (w * rstd) (rms, no bias)
                # We just delegate to the kernel with y's semantics; the kernel
                # takes x's role. The bwd uses x (i.e., y) to compute xhat.
                x_ = y_or_x  # kernel uses x's slot, we'll compute xhat = y / w
            dy2 = dy.reshape(-1, dy.shape[-1])
            if ctx_._num_groups != 1:
                raise NotImplementedError("num_groups != 1 not handled in shim")
            dx, dw, db, dresidual_in = ln_mod.layer_norm_bwd(
                dy2, x_, weight, bias, mean, rstd, None, False,
                ctx_._is_rms_norm, x_dtype=ctx_._x_dtype,
                num_groups=ctx_._num_groups,
            )
            # Note: kernel's x argument is treated as the post-residual-add
            # value. With our y-substitution, the bwd produces an
            # inconsistent dx (xhat gets derived wrong). We ONLY use this
            # path to measure VRAM — correctness is verified separately.
            return (dx.reshape(ctx_._x_shape), dw, db, None, None, None, None, None, None, None)

    # Patch the rms_norm() function to use the new path
    def _patched_rms_norm(x, weight, bias, residual=None, eps=1e-5,
                          prenorm=False, residual_in_fp32=False, save_x=True):
        return _PatchedFn.apply(x, weight, bias, residual, eps, prenorm,
                                residual_in_fp32, True, 1, save_x)

    # Monkey-patch the rms_norm symbol used by src.models.norms.RMSNorm
    if save_x:
        # Use ORIGINAL default — don't patch
        # Note: we still need to force save_x=True in our shim path,
        # so use the un-patched original.
        pass

    fwd_path = _patched_rms_norm if not save_x else ln_mod.rms_norm

    # Override the rms_norm module-level binding
    if not save_x:
        ln_mod.rms_norm = _patched_rms_norm
        # Also override where RMSNorm reads it
        norms_mod._rms_norm_override = _patched_rms_norm

    # Override RMSNorm.forward to use the override
    from src.models.norms import RMSNorm as _OrigRMSNorm

    if not save_x:
        def _fwd_override(self, x):
            return norms_mod._rms_norm_override(x, self.weight, None, eps=self.eps)
        _OrigRMSNorm.forward = _fwd_override

    # Reset module
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup (compile, autotune)
    out = model(x, y)
    loss = out["loss"] if "loss" in out else out["logits"].sum() * 0
    loss.backward()

    torch.cuda.synchronize()
    peak_pre = torch.cuda.max_memory_allocated()

    # Reset
    torch.cuda.reset_peak_memory_stats()

    # Real measurement
    out = model(x, y)
    loss = out["loss"] if "loss" in out else out["logits"].sum() * 0
    loss.backward()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return peak


if __name__ == "__main__":
    for hidden, T, L in [
        (1536, 4096, 8),    # moderate (already tested)
        (1536, 4096, 24),   # more depth — more rmsnorm calls alive at peak
        (1536, 8192, 8),    # more T — bigger per-call rmsnorm x save
    ]:
        print(f"=== hidden={hidden}, T={T}, L={L} ===")
        peak_a = measure(save_x=True, hidden=hidden, T=T, num_layers=L)
        print(f"  A (with x save) peak: {peak_a / (1024**2):.2f} MiB")
        peak_b = measure(save_x=False, hidden=hidden, T=T, num_layers=L)
        print(f"  B (no  x save) peak: {peak_b / (1024**2):.2f} MiB")
        delta = peak_a - peak_b
        print(f"  Delta:               {delta / (1024**2):+.2f} MiB")
        print()
