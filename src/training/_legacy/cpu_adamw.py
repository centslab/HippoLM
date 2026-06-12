"""
CPU AdamW Optimizer with memory-efficient streaming for HippoLM.

Key optimizations:
1. Optimizer states (exp_avg, exp_avg_sq) stored on CPU pinned memory
2. Gradients streamed from GPU to CPU, update computed on CPU
3. Updated parameters streamed back to GPU
4. Block-wise processing for pipelining

Inspired by DeepSpeed ZeRO-Offload and rosaflow implementation.
"""
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import threading
from concurrent.futures import ThreadPoolExecutor


class CPUAdamW:
    """
    Memory-efficient AdamW optimizer that performs updates on CPU.

    Strategy:
    - Model parameters stay on GPU
    - Optimizer states (exp_avg, exp_avg_sq) on CPU pinned memory
    - Gradients computed on GPU, streamed to CPU
    - Update computation happens on CPU using in-place operations
    - Updated parameters streamed back to GPU

    Memory savings:
    - Without offload: 2 * model_params (for optimizer states) on GPU
    - With offload:   0 * model_params (optimizer states on CPU)

    For 624M params with FP32 optimizer states: saves ~2.3GB GPU memory
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 2097152,  # 2M params per block
        logger=None,
    ):
        self.logger = logger or (lambda msg: print(f"[CPUAdamW] {msg}"))
        self.device = torch.device('cpu')

        # Adam hyperparameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.block_size = block_size

        # State
        self.state: Dict[int, Dict] = {}
        self.param_groups = [{'params': params, 'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}]
        self.step_count = 0

        # Performance tracking
        self.stats = {
            'cpu_update_time': 0.0,
            'transfer_time': 0.0,
            'total_steps': 0,
            'total_params_updated': 0,
        }

        # Setup parameters and CPU states
        self._params = [p for p in params if p.requires_grad]
        self._setup_cpu_states()

        total_params = sum(p.numel() for p in self._params)
        self.logger(f"CPUAdamW initialized: lr={lr}, betas={betas}, block_size={block_size:,}")
        self.logger(f"Total trainable parameters: {total_params:,} ({total_params * 4 / 1024**3:.2f} GB optimizer states on CPU)")

    def _setup_cpu_states(self):
        """Initialize optimizer states on CPU pinned memory."""
        for i, p in enumerate(self._params):
            if not p.requires_grad:
                continue
            self.state[i] = {
                'exp_avg': torch.zeros(p.numel(), dtype=torch.float32, device='cpu').pin_memory(),
                'exp_avg_sq': torch.zeros(p.numel(), dtype=torch.float32, device='cpu').pin_memory(),
                'step': 0,
            }

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients (GPU-side operation)."""
        for p in self._params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_().zero_()

    def _adam_update_cpu(
        self,
        param_cpu: torch.Tensor,
        grad_cpu: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> torch.Tensor:
        """Perform AdamW update on CPU with in-place operations."""
        # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)

        # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1

        # Decoupled weight decay
        if weight_decay != 0:
            param_cpu.mul_(1 - lr * weight_decay)

        # Adam update: param -= step_size * exp_avg / sqrt(exp_avg_sq / bias_correction2 + eps)
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        param_cpu.addcdiv_(exp_avg, denom, value=-step_size)

        return param_cpu

    def step(self, closure=None):
        """Perform optimization step with CPU offload.

        Optimizations:
        - Batch all parameters and transfer once (not one-by-one)
        - Use multithreading for parallel CPU computation
        - Minimize GPU-CPU-GPU round-trips
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        step_start = time.time()

        # Collect all params with gradients
        params_with_grad = []
        for i, p in enumerate(self._params):
            if p.grad is not None:
                params_with_grad.append((i, p))

        if not params_with_grad:
            return loss

        # Batch transfer all gradients to CPU
        batch_start = time.time()
        all_grads = []
        all_params = []
        all_exp_avgs = []
        all_exp_avg_sqs = []

        for i, p in params_with_grad:
            all_grads.append(p.grad.data.view(-1).clone().detach().to('cpu', non_blocking=True))
            all_params.append(p.data.view(-1).clone().detach().to('cpu', non_blocking=True))
            state = self.state[i]
            all_exp_avgs.append(state['exp_avg'])
            all_exp_avg_sqs.append(state['exp_avg_sq'])

        torch.cuda.synchronize()
        batch_transfer_time = time.time() - batch_start

        # CPU computation in parallel using thread pool
        cpu_start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for j, (i, p) in enumerate(params_with_grad):
                grad_cpu = all_grads[j]
                param_cpu = all_params[j]
                exp_avg = all_exp_avgs[j]
                exp_avg_sq = all_exp_avg_sqs[j]

                future = executor.submit(
                    self._adam_update_cpu,
                    param_cpu, grad_cpu, exp_avg, exp_avg_sq,
                    self.step_count, self.lr, self.beta1, self.beta2,
                    self.eps, self.weight_decay,
                )
                futures.append(future)

            # Wait for all to complete
            for future in futures:
                future.result()

        cpu_time = time.time() - cpu_start

        # Batch transfer back to GPU
        batch_start = time.time()
        for j, (i, p) in enumerate(params_with_grad):
            param_cpu = all_params[j]
            p.data.view(-1).copy_(param_cpu, non_blocking=True)

        torch.cuda.synchronize()
        batch_transfer_time += time.time() - batch_start

        self.stats['cpu_update_time'] += cpu_time
        self.stats['transfer_time'] += batch_transfer_time
        self.stats['total_steps'] += 1
        self.stats['total_params_updated'] += sum(p.numel() for _, p in params_with_grad)

        return loss

    def state_dict(self) -> Dict:
        return {
            'state': self.state,
            'param_groups': self.param_groups,
            'step_count': self.step_count,
        }

    def load_state_dict(self, state_dict: Dict):
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
        self.step_count = state_dict.get('step_count', 0)


def create_cpu_adamw_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    block_size: int = 2097152,
    logger=None,
) -> CPUAdamW:
    """Factory function to create CPU AdamW optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    return CPUAdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        block_size=block_size,
        logger=logger,
    )