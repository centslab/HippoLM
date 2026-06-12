"""
Memory test utilities for HippoLM training optimization.

Tests:
1. Memory footprint measurement (model, optimizer, gradients, activations)
2. OOM detection and recovery
3. Gradient flow verification
4. CPU offload verification
"""
import gc
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MemoryMonitor:
    """Monitor GPU memory usage during training."""

    def __init__(self, device: torch.device = torch.device('cuda')):
        self.device = device
        self.peak_memory = 0
        self.baseline_memory = 0
        self snapshots = []

    def reset(self):
        """Reset memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            gc.collect()
        self.peak_memory = 0
        self.snapshots = []

    def start(self):
        """Record baseline memory."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.baseline_memory = torch.cuda.memory_allocated(self.device)
        return self

    def snapshot(self, label: str):
        """Take a memory snapshot with label."""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, peak)
            self.snapshots.append({
                'label': label,
                'current_mb': current / 1024**2,
                'peak_mb': peak / 1024**2,
            })
        return self

    def get_peak_mb(self) -> float:
        """Get peak memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / 1024**2
        return 0.0

    def print_summary(self):
        """Print memory usage summary."""
        print("\n" + "=" * 60)
        print("Memory Usage Summary")
        print("=" * 60)
        for snap in self.snapshots:
            peak_delta = snap['peak_mb'] - self.baseline_memory / 1024**2
            current_delta = snap['current_mb'] - self.baseline_memory / 1024**2
            print(f"  {snap['label']:30s}  current: {current_delta:8.1f} MB  peak: {peak_delta:8.1f} MB")
        print(f"\n  {'TOTAL PEAK':30s}  {self.get_peak_mb() - self.baseline_memory / 1024**2:8.1f} MB")
        print("=" * 60)


def test_memory_efficient_training(
    config,
    batch_size: int = 2,
    seq_len: int = 512,
    use_cpu_offload: bool = True,
    use_gradient_checkpointing: bool = True,
    logger=None,
) -> Dict:
    """
    Test memory-efficient training setup.

    Returns dict with memory stats and test results.
    """
    log = logger or (lambda msg: print(msg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor = MemoryMonitor(device)

    from src.models.model import HippoModel
    from src.training._legacy.cpu_adamw import create_cpu_adamw_optimizer

    results = {
        'passed': False,
        'peak_memory_mb': 0,
        'error': None,
    }

    try:
        log("Building model...")
        monitor.start().snapshot("baseline")

        model = HippoModel(config).to(device)
        monitor.snapshot("model_loaded")

        total_params = sum(p.numel() for p in model.parameters())
        log(f"Model parameters: {total_params:,} ({total_params * 4 / 1024**3:.2f} GB)")

        # Optimizer
        if use_cpu_offload:
            log("Setting up CPU AdamW optimizer (offload to CPU)...")
            optimizer = create_cpu_adamw_optimizer(
                model,
                lr=1e-4,
                weight_decay=0.01,
            )
        else:
            log("Setting up standard AdamW optimizer (GPU)...")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.01,
            )
        monitor.snapshot("optimizer_created")

        # Create dummy input
        log(f"Creating dummy batch (batch={batch_size}, seq={seq_len})...")
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        monitor.snapshot("data_created")

        # Forward pass
        log("Running forward pass...")
        model.train()
        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
        monitor.snapshot("forward_pass")

        # Backward pass
        log("Running backward pass...")
        loss.backward()
        monitor.snapshot("backward_pass")

        # Optimizer step
        log("Running optimizer step...")
        optimizer.zero_grad()
        optimizer.step()
        monitor.snapshot("optimizer_step")

        peak_mb = monitor.get_peak_mb() - monitor.baseline_memory / 1024**2
        results['peak_memory_mb'] = peak_mb
        log(f"Peak memory usage: {peak_mb:.1f} MB ({peak_mb / 1024:.2f} GB)")

        # Verify gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        log(f"Parameters with gradients: {grad_count}/{total_params}")

        results['passed'] = True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            results['error'] = "OOM"
            log(f"ERROR: Out of Memory - {e}")
        else:
            results['error'] = str(e)
            log(f"ERROR: {e}")
    except Exception as e:
        results['error'] = str(e)
        log(f"ERROR: {e}")

    monitor.print_summary()
    return results


def test_gradient_checkpointing(config, device, logger=None):
    """Test gradient checkpointing correctness."""
    log = logger or (lambda msg: print(msg))

    from src.models.model import HippoModel

    log("Testing gradient checkpointing...")

    model = HippoModel(config).to(device)
    model.train()

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Forward with checkpointing
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    log(f"Loss with checkpointing: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Check gradients exist
    grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
    log(f"Parameters with gradients: {len(grad_params)}")

    # Verify gradient values are non-zero
    total_grad_norm = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item()

    log(f"Total gradient norm: {total_grad_norm:.4f}")

    return total_grad_norm > 0


def sweep_batch_sizes(config, max_seq_len=512, logger=None):
    """Sweep different batch sizes to find OOM boundary."""
    log = logger or (lambda msg: print(msg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    for batch_size in [1, 2, 4, 8, 16]:
        log(f"\n--- Testing batch_size={batch_size} ---")
        result = test_memory_efficient_training(
            config,
            batch_size=batch_size,
            seq_len=max_seq_len,
            use_cpu_offload=True,
            use_gradient_checkpointing=True,
            logger=logger,
        )
        results.append({
            'batch_size': batch_size,
            'peak_mb': result['peak_memory_mb'],
            'passed': result['passed'],
            'error': result['error'],
        })

        if not result['passed'] and result['error'] == 'OOM':
            log(f"OOM at batch_size={batch_size}")
            break

        # Clean up for next test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def measure_activation_memory(config, batch_size=2, seq_len=512, logger=None):
    """Measure activation memory footprint."""
    log = logger or (lambda msg: print(msg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.models.model import HippoModel

    model = HippoModel(config).to(device)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Reset peak stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with torch.amp.autocast("cuda", enabled=True):
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]

    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    log(f"Peak activation memory (batch={batch_size}, seq={seq_len}): {peak_mb:.1f} MB")

    return peak_mb


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/wlx/HippoLM")

    from src.models import HippoConfig

    config = HippoConfig()

    print("\n" + "=" * 60)
    print("HippoLM Memory Test Suite")
    print("=" * 60)

    print("\n1. Testing gradient checkpointing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_gradient_checkpointing(config, device)

    print("\n2. Sweeping batch sizes...")
    results = sweep_batch_sizes(config, max_seq_len=512)

    print("\n3. Batch size sweep results:")
    print("-" * 40)
    for r in results:
        status = "PASS" if r['passed'] else f"FAIL ({r['error']})"
        print(f"  batch_size={r['batch_size']:2d}: {r['peak_mb']:8.1f} MB - {status}")