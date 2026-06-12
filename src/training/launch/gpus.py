"""GPU auto-detection for TP: select the largest viable contiguous run."""
from __future__ import annotations

from typing import List


def select_tp_gpus(min_memory_mb: int = 10240) -> List[int]:
    """Select a contiguous run of 2/4/8 GPUs, each with at least
    ``min_memory_mb`` free VRAM.

    Strategy: collect the indices of all eligible GPUs (in increasing
    physical order), then look for the largest contiguous run of
    length 2, 4, or 8 and return that. 1, 3, 5, 6, 7 are
    intentionally not supported — odd counts would force uneven
    sharding and 6/7 are not power-of-two friendly. The largest
    viable prefix is preferred (8 > 4 > 2) so we don't
    under-utilize the box.

    Returns ``[]`` when no eligible GPU is found. The caller decides
    what to do with the empty result (fall back to single-GPU,
    TP-sim mode, or error out).

    Tries ``pynvml`` first; falls back to ``nvidia-smi``; falls back
    to ``torch.cuda.get_device_properties`` when neither is
    available. Honors ``CUDA_VISIBLE_DEVICES`` by clipping the
    eligible list to the visible-device count.
    """
    import torch

    if not torch.cuda.is_available():
        return []

    eligible: list[int] = []

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = info.free / (1024 ** 2)
            if free_mb >= min_memory_mb:
                eligible.append(i)
        pynvml.nvmlShutdown()
    except ImportError:
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 2:
                    idx = int(parts[0].strip())
                    free = float(parts[1].strip())
                    if free >= min_memory_mb:
                        eligible.append(idx)
        except FileNotFoundError:
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                if total >= min_memory_mb:
                    eligible.append(i)

    # Respect CUDA_VISIBLE_DEVICES if set.
    visible = torch.cuda.device_count()
    eligible = [g for g in eligible if g < visible]
    if not eligible:
        eligible = list(range(visible))

    # Find the largest contiguous run of size 2, 4, or 8.
    for target in (8, 4, 2):
        for start in range(len(eligible) - target + 1):
            window = eligible[start:start + target]
            if window[-1] - window[0] == target - 1:
                # All contiguous and in order.
                return window

    return []
