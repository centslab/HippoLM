"""Training utilities for HippoLM.

The canonical optimizers live in :mod:`src.training.param_offload`
(:class:`CPUAdamW`, :class:`CPUMuon`, :func:`build_param_groups`,
:func:`accumulate_grads_to_cpu`, :func:`zero_cpu_grad_accum`).
The :mod:`src.training._legacy` subpackage is kept for backward
compatibility with the v0.0.0-era ``cpu_adamw.CPUAdamW`` reference
implementation; new code should import from :mod:`.param_offload`.

Checkpoint save/load is exposed as :func:`save_checkpoint` and
:func:`load_checkpoint` (see :mod:`.checkpoint`).

Per-step diagnostic logging helpers (pre-step, post-Muon,
post-AdamW) live in :mod:`.diagnostics` and consume the
:class:`OptimizerState` protocol.

The per-rank training worker (setup / run / teardown) lives in
:mod:`.loop` and is exposed as :func:`_train_worker`.
"""
from ._legacy.cpu_adamw import CPUAdamW, create_cpu_adamw_optimizer
from .checkpoint import save_checkpoint, load_checkpoint
from .diagnostics import (
    amax_cpu,
    log_pre_step_diag,
    log_post_opt_diag,
)
from .loop import _train_worker
from .param_offload import OptimizerState
from .tokenizer import load_tokenizer

__all__ = [
    'CPUAdamW', 'create_cpu_adamw_optimizer',
    'save_checkpoint', 'load_checkpoint',
    'amax_cpu', 'log_pre_step_diag', 'log_post_opt_diag',
    'OptimizerState',
    '_train_worker',
    'load_tokenizer',
]