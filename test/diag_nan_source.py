"""Locate the layer that first produces non-finite values during forward.

Runs HippoModel in FP16 with deterministic input, then walks the output
of every layer and prints the first non-finite tensor.

Usage:
    CUDA_VISIBLE_DEVICES=5 python3 test/diag_nan_source.py
"""
import sys
import torch

sys.path.insert(0, "/home/wlx/HippoLM")

from src.models import HippoConfig
from src.models.model import HippoModel


def check_finite(name: str, t: torch.Tensor) -> bool:
    """Print stats and return True if tensor is fully finite."""
    if t is None:
        return True
    if not t.is_cuda:
        # CPU tensors are fine, just skip
        return True
    if t.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        # Integer / bool tensors: always finite
        return True
    finite = torch.isfinite(t)
    n = t.numel()
    n_bad = (~finite).sum().item()
    if n_bad > 0:
        abs_max = t.abs().max().item()
        print(f"  [BAD]  {name}: {n_bad}/{n} non-finite, max|.|={abs_max:.4e}")
        return False
    # only print stats for big tensors to keep output small
    if n >= 1024:
        print(f"  [ok]   {name}: shape={tuple(t.shape)} dtype={t.dtype}"
              f" max|.|={t.abs().max().item():.4e}")
    return True


def main():
    torch.manual_seed(0)
    config = HippoConfig()  # full config
    model = HippoModel(config).to("cuda")
    model = model.to(torch.float16)
    model.eval()

    B, T = 2, 256
    input_ids = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    print(f"input_ids: shape={tuple(input_ids.shape)}"
          f" min={input_ids.min().item()} max={input_ids.max().item()}")

    # Hook every layer to inspect outputs
    handles = []
    bad_layer = [None]

    def make_hook(name):
        def hook(module, inputs, output):
            if bad_layer[0] is not None:
                return  # already found the bad one
            # Output may be tuple/list; flatten
            if isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if not check_finite(f"{name}[{i}]", o):
                        bad_layer[0] = name
                        return
            else:
                if not check_finite(name, output):
                    bad_layer[0] = name
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or "RMSNorm" in type(module).__name__:
            handles.append(module.register_forward_hook(make_hook(name)))
        # Also hook the KDA module as a whole to catch its internal kernel outputs
        if "KDA" in type(module).__name__ and name.endswith("kda"):
            handles.append(module.register_forward_hook(make_hook(name)))

    print("\n--- Forward pass with hooks (per-layer) ---")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids)
    for h in handles:
        h.remove()

    print("\n--- Final output ---")
    logits = outputs["logits"]
    print(f"logits: shape={tuple(logits.shape)} dtype={logits.dtype}"
          f" max|.|={logits.abs().max().item():.4e}"
          f" nan={torch.isnan(logits).sum().item()}"
          f" inf={torch.isinf(logits).sum().item()}")

    if bad_layer[0]:
        print(f"\n*** First bad layer: {bad_layer[0]} ***")
    else:
        print("\n*** All layers produced finite outputs ***")


if __name__ == "__main__":
    main()
