"""Real-activation scale-precision sweep.

Builds a tiny HippoLM model, captures intermediate activations at every
quantization site (RMSNorm out, residual output, FFN inner, KDA q/k/v,
attention output, etc.), then runs the same scale-precision sweep as
synthetic Part B but on these real tensors.

Goal: find which scale format and block size gives the lowest sig_rel
on REAL activations — the choice that matters for production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


from src.models.config import HippoConfig  # noqa: E402
from src.models.model import HippoModel  # noqa: E402


E4M3_MAX = 448.0
SCALE_TYPES = ["FP32", "FP16", "BF16", "E8M0"]


def scale_round_to_format(s: torch.Tensor, fmt: str, encoder: str = "ceil_log2") -> torch.Tensor:
    s = s.float()
    if fmt == "FP32":
        return s
    if fmt == "BF16":
        return s.to(torch.bfloat16).to(torch.float32)
    if fmt == "FP16":
        return s.to(torch.float16).to(torch.float32)
    if fmt == "E8M0":
        log2_s = torch.log2(s.clamp_min(5.877e-39))
        scale_exp = torch.ceil(log2_s).clamp(-127, 127)
        scale_u8 = (scale_exp + 127).to(torch.uint8)
        return scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    raise ValueError(fmt)


def quant_blockwise(v: torch.Tensor, block: int, scale_type: str, fp8_max: float, fp8_dtype):
    shape = v.shape
    v_flat = v.float().reshape(-1, shape[-1])  # [N, K]
    N, K = v_flat.shape
    assert K % block == 0
    v_b = v_flat.reshape(N, K // block, block)
    s_ideal = v_b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal, scale_type).reshape(N, K // block, 1)
    r = (v_b / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = r.reshape(shape).contiguous()
    s_full = s_stored.expand(N, K // block, block).reshape(N, K).reshape(*shape[:-1], shape[-1])
    return v_fp8, s_full


def quant_rowwise(v: torch.Tensor, scale_type: str, fp8_max: float, fp8_dtype):
    shape = v.shape
    v_flat = v.float().reshape(-1, shape[-1])
    N, K = v_flat.shape
    s_ideal = v_flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal, scale_type)
    r = (v_flat / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = r.reshape(shape).contiguous()
    s_full = s_stored.expand(N, K).reshape(*shape[:-1], shape[-1])
    return v_fp8, s_full


def quant_tensorwise(v: torch.Tensor, scale_type: str, fp8_max: float, fp8_dtype):
    s_ideal = v.float().abs().amax().clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal.reshape(1, 1), scale_type).reshape(1, 1)
    r = (v.float() / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return r.contiguous(), s_stored.expand_as(v)


def metrics_per_tensor(v_fp8, s, v_ref):
    """SQNR per row, then median."""
    v_dq = (v_fp8.float() * s.float())
    diff = (v_dq - v_ref.float())
    sig_pow = (v_ref.float() ** 2).mean(dim=-1).clamp_min(1e-12)
    noise_pow = (diff ** 2).mean(dim=-1).clamp_min(1e-20)
    sqnr_db = 10.0 * torch.log10(sig_pow / noise_pow)
    return sqnr_db.median().item(), sqnr_db.mean().item(), diff.abs().median().item(), diff.abs().max().item()


def run_real():
    cfg = HippoConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        num_layers=2,
        num_blocks=1,  # skip AttnRes to keep it simple
        intermediate_size=1536,
        vocab_size=1024,
        attention_precision="w16a16",
        ffn_precision="w16a16",
    )
    model = HippoModel(cfg).cuda().to(torch.bfloat16)
    model.eval()  # avoid dropout

    # Capture activations at each Linear's input
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            captured[name] = x.detach()
        return hook

    # Find all linear-like modules in the model
    hooks = []
    for name, module in model.named_modules():
        # nn.Linear / FP8Linear / NVFP4Linear* / MXFP8Linear — anything with weight+bias
        cls_name = type(module).__name__
        if cls_name in {"Linear", "FP8Linear", "NVFP4Linear", "NVFP4LinearW4A8",
                          "MXFP8Linear", "NVFP4Marlin", "NVFP4LinearTP", "_nvfp4_marlin"}:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Run forward
    B, T = 2, 256
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    with torch.no_grad():
        out = model(input_ids)

    for h in hooks:
        h.remove()

    print(f"Captured {len(captured)} activation tensors.")
    print()

    # Print amax distribution
    print(f"{'name':40s}  {'shape':20s}  {'amax':>8s}  {'mean|v|':>8s}  {'std':>8s}  {'max/mean':>8s}")
    print("-" * 100)
    for name, v in captured.items():
        v_f = v.float()
        amax = v_f.abs().amax().item()
        mean = v_f.abs().mean().item()
        std = v_f.std().item()
        ratio = amax / mean if mean > 0 else 0
        print(f"{name[:40]:40s}  {str(tuple(v.shape)):20s}  {amax:8.4f}  {mean:8.4f}  {std:8.4f}  {ratio:8.1f}")
    print()

    # Run scale-precision sweep on each captured tensor
    print("=" * 100)
    print("Scale-precision sweep on real activations")
    print("=" * 100)
    print()

    fp8_max = E4M3_MAX
    fp8_dtype = torch.float8_e4m3fn

    for name, v in captured.items():
        # Skip tiny tensors (e.g., bias)
        if v.numel() < 1024 or v.dtype != torch.bfloat16:
            continue
        v_bf = v.contiguous()
        v_ref = v_bf.float()
        K = v_bf.shape[-1]
        # Determine which block sizes are valid for this K
        valid_blocks = [b for b in [1, 16, 32, 64, 128] if K % b == 0]
        print(f"--- {name}  shape={tuple(v.shape)}  amax={v_ref.abs().amax().item():.4f} ---")

        # Per scale_type × granularity
        for scale_type in SCALE_TYPES:
            cells = []
            # tensor
            v_fp8, s = quant_tensorwise(v_bf, scale_type, fp8_max, fp8_dtype)
            sqnr_med, _, _, _ = metrics_per_tensor(v_fp8, s, v_bf)
            cells.append(f"tensor={sqnr_med:5.1f}dB")
            # row
            v_fp8, s = quant_rowwise(v_bf, scale_type, fp8_max, fp8_dtype)
            sqnr_med, _, _, _ = metrics_per_tensor(v_fp8, s, v_bf)
            cells.append(f"row={sqnr_med:5.1f}dB")
            # blocks
            for blk in valid_blocks:
                v_fp8, s = quant_blockwise(v_bf, blk, scale_type, fp8_max, fp8_dtype)
                sqnr_med, _, _, _ = metrics_per_tensor(v_fp8, s, v_bf)
                cells.append(f"b{blk}={sqnr_med:5.1f}dB")
            print(f"  {scale_type:6s}: " + "  ".join(cells))
        print()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")
    print(f"torch: {torch.__version__}")
    print()
    run_real()