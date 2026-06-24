"""TP simulation: collapse the GPU list and warn at the right moments.

Single-GPU dev boxes (e.g. one 5060 Ti 16G) can still exercise the
TP code paths via :func:`expand_for_tp_sim`: it returns a list
``[gpus[0]] * tp_size`` so :func:`torch.multiprocessing.spawn`
launches N child processes all bound to the same physical device,
and the user supplies ``backend="gloo"`` (instead of NCCL) for the
cross-process collectives. gloo can transport CUDA tensors (it
D2H-stages each collective), so the sharding code paths run
byte-for-byte the same; only the transport changes.

The replicated params (KDA, embed, RMSNorm, AttnRes) are
duplicated in every child process. For the default HippoConfig the
per-process replicated weight is ~900 MB in FP16; with optimizer
state and activations a 4-way sim is already tight on 16 GB, and
8-way will OOM. We warn (but do not block) when ``tp_size > 4``.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


def expand_for_tp_sim(
    gpus: List[int],
    tp_size: Optional[int],
    sim_enabled: bool,
) -> Tuple[List[int], bool]:
    """Optionally collapse the GPU list for TP simulation.

    Returns:
        ``(expanded_gpus, sim_active)``:

        - If ``sim_enabled`` is False, returns ``(list(gpus), False)``
          unchanged.
        - If ``sim_enabled`` is True and ``tp_size >= 2``, returns
          ``([gpus[0]] * tp_size, True)`` and logs a WARNING when
          ``tp_size > 4`` (likely OOM on 16 GB).
        - If ``sim_enabled`` is True but ``tp_size < 2``, logs a
          WARNING explaining that sim is being disabled and returns
          ``(list(gpus), False)``.

    Logs at WARNING level (per project decision: TP-sim warnings
    always go through the logger, never ``print(file=sys.stderr)``,
    so they end up in the run-dir log file when the entry point
    wires one up).
    """
    if not sim_enabled:
        return list(gpus), False

    if tp_size is None or tp_size < 2:
        log.warning(
            "--tp_size=%s; --tp_sim requires >= 2, disabling sim",
            tp_size,
        )
        return list(gpus), False

    if tp_size > 4:
        log.warning(
            "--tp_size=%d > 4 may OOM on 16 GB "
            "(replicated weights duplicated per process). "
            "Try --use_dummy_data first.",
            tp_size,
        )

    first_gpu = gpus[0]
    log.warning(
        "[TP-SIM] simulating TP=%d ranks on cuda:%d (gloo backend)",
        tp_size, first_gpu,
    )
    return [first_gpu] * tp_size, True
