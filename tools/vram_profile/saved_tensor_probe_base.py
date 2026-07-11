"""Probe what's saved by each autograd.Function at base-scale dims.

base.yml: H=1536, T=16384, 32 layers, n_blocks=8. This won't fit
in 16 GiB at full scale but we can probe with fewer layers and
infer per-call sizes.

We use H=1536, T=4096, n_layers=4 so the saved-tensor footprint
is large enough to see the actual per-call sizes.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/hy-tmp/HippoLM")

import torch

captured: list[tuple[str, tuple]] = []
_orig_save = torch.autograd.function.FunctionCtx.save_for_backward

def _wrapped_save(self, *tensors):
    cls = type(self).__name__
    summary = []
    for t in tensors:
        if t is None:
            summary.append(("None",))
        elif isinstance(t, torch.Tensor):
            summary.append((str(t.dtype).replace("torch.", ""), tuple(t.shape), t.element_size() * t.numel()))
        else:
            summary.append((type(t).__name__,))
    captured.append((cls, tuple(summary)))
    return _orig_save(self, *tensors)


torch.autograd.function.FunctionCtx.save_for_backward = _wrapped_save


def main():
    from src.models.tp_model._primitives import init_tp
    from src.models.tp_model.model import TPHippoModel

    init_tp(world_size=1, devices=[0] if torch.cuda.is_available() else ["cpu"], backend="gloo")

    from src.models.config import HippoConfig
    cfg = HippoConfig(
        hidden_size=1536, num_layers=2, num_blocks=1,
        vocab_size=8192, head_dim=128, num_heads=12,
        expand_v=1.0, intermediate_size=4096,
        ffn_nvfp4=False,
        safe_gate=True, lower_bound=-5.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    devs = [0] if torch.cuda.is_available() else ["cpu"]
    model = TPHippoModel(cfg, devs).to(torch.bfloat16)
    if device.type == "cpu":
        model = model.to(torch.float32)

    print(f"Model: H=1536, T=2048, n_layers=2")

    # B=1, T=2048 — should be safe on 16GB
    x = torch.randint(0, cfg.vocab_size, (1, 2048), device=device, dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (1, 2048), device=device, dtype=torch.long)

    captured.clear()
    out = model(x, y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\n[{len(captured)} save_for_backward calls]\n")

    by_class = {}
    for cls, summary in captured:
        by_class.setdefault(cls, []).append(summary)

    grand_total_bytes = 0
    for cls, calls in sorted(
        by_class.items(),
        key=lambda kv: -sum(sum(it[2] for it in items if len(it) == 3) for items in kv[1]),
    ):
        total_bytes = sum(sum(it[2] for it in items if len(it) == 3) for items in calls)
        grand_total_bytes += total_bytes
        first = calls[0]
        print(f"--- {cls} ({len(calls)} calls, total = {total_bytes/(1024**2):.2f} MiB) ---")
        # Print first 3 examples
        for i, items in enumerate(calls[:3]):
            print(f"  example {i}: {items}")
        # Show unique signatures
        sigs = {}
        for items in calls:
            sig = tuple((it if len(it) <= 1 else (it[0], it[1])) for it in items)
            sigs[sig] = sigs.get(sig, 0) + 1
        for sig, cnt in sorted(sigs.items(), key=lambda kv: -kv[1])[:4]:
            sizes = [it[2] for it in sig if len(it) == 3]
            shape_strs = []
            for s in sig:
                if len(s) <= 1:
                    shape_strs.append(str(s[0]))
                else:
                    shape_strs.append(f"{s[0]}({','.join(str(x) for x in s[1])})")
            print(f"    {cnt}× sig: [{','.join(shape_strs)}], sizes={[f'{s/(1024**2):.2f}MiB' for s in sizes]}")
        print()

    print(f"\nGRAND TOTAL saved across fwd: {grand_total_bytes/(1024**2):.2f} MiB\n")


if __name__ == "__main__":
    main()
