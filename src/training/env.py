"""Runtime environment shims for the training entry point.

This module is imported **early** in the training entry point and
exists to make the local box (8x V100 / single 5060Ti dev box)
behave the same way on every run, regardless of what the user has
already exported in their shell.

What it configures (in order, all best-effort and idempotent — they
only set env vars that are not already set, except where noted):

1. **``.env`` loader** (a stdlib-only minimal loader; we don't
   depend on ``python-dotenv``). Runs first so the env vars the
   later steps can read are already populated.
2. **``HF_ENDPOINT``** — defaults to ``https://hf-mirror.com`` if
   unset, so streaming reads prefer the Chinese mirror.
3. **NCCL transport pinning** — see :func:`pin_nccl_environment`
   for the rationale. The defaults are tuned for this repo's
   8xV100-SXM2 box; users with a different topology can override
   in their shell before invoking.
4. **Datasets streaming retry budget** — see
   :func:`pin_datasets_retry_config`. The ``datasets`` library's
   defaults (20 × 60s = 1200s) cause init-time hangs on a slow
   CDN; we tighten both the retry count and the sleep intervals.
5. **Socket default timeout** — :func:`pin_socket_default_timeout`
   bounds the *unconditional* read timeout on a streaming HTTP
   connection that accepts the TCP open then stops sending bytes.
   A stalled connection would otherwise wait the OS TCP keepalive
   (default ~7200s on Linux) before failing.

Usage::

    # at the top of scripts/train.py, BEFORE any heavy import:
    from src.training.env import configure_runtime_environment
    configure_runtime_environment()
"""
from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------------------------------- #
# .env loader                                                                 #
# --------------------------------------------------------------------------- #
def apply_env_from_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into ``os.environ``.

    Stdlib-only — no python-dotenv dependency. Lines starting with
    ``#`` or without ``=`` are ignored; existing env vars are NOT
    overwritten (``setdefault`` semantics). If ``path`` is None,
    defaults to ``<repo_root>/.env``.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --------------------------------------------------------------------------- #
# HF mirror                                                                   #
# --------------------------------------------------------------------------- #
def pin_hf_endpoint(default: str = "https://hf-mirror.com") -> None:
    """Default ``HF_ENDPOINT`` to the Chinese HF mirror if unset.

    The mirror is much faster for users in CN; the env override is
    a convenience for users outside CN who prefer the canonical
    endpoint. The mirror is set *only* if no value is already in
    the environment — passing the canonical URL in the shell
    wins.
    """
    os.environ.setdefault("HF_ENDPOINT", default)


# --------------------------------------------------------------------------- #
# NCCL transport pinning                                                      #
# --------------------------------------------------------------------------- #
def pin_nccl_environment() -> None:
    """Pin the NCCL transport to the local NVLink fabric.

    On this 8xV100-SXM2 box, GPUs 0-3 belong to a vLLM TP
    deployment and GPU 4 to ComfyUI; we must not touch them.
    ``nvidia-smi topo -m`` shows the training pair (e.g. 5,6)
    connected via NVLink (NV1) so inter-GPU traffic should stay
    on the NVLink fabric, not fall through to PCIe/SYS (which
    would be ~5x slower and would also let NCCL probe the vLLM
    GPUs).

    Also disable the NCCL heartbeat watchdog: the first step
    compiles many Triton kernels (chunk_kda fwd/bwd × 32 layers,
    fused CE, fused RMSNormGated) which can hold the GIL for
    >480s, making the watchdog falsely report a hang and abort
    the job. We additionally widen the heartbeat timeout
    belt-and-braces, in case the monitoring flag is interpreted
    differently in this PyTorch build.
    """
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker,veth,br-")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")


# --------------------------------------------------------------------------- #
# Datasets streaming retry config                                             #
# --------------------------------------------------------------------------- #
def pin_datasets_retry_config() -> None:
    """Tighten the ``datasets`` streaming-read retry budget.

    The default ``MAX_RETRIES=20`` on the slowest path
    (``STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL=60s``) means a
    single bad shard can hang the init for 1200s of silent
    sleeping — that is the "init hang" we've been seeing. We
    tighten both the retries count (50, 5s total) and the
    intervals on the slow paths (503 → 1s, 429 → 2s) so transient
    CDN disconnects are recovered transparently inside the
    streaming read rather than surfacing as a 100s+ wall-clock
    stall. The HF mirror is also routed through the same
    wrapper, so this tuning helps its disconnects too.
    """
    import datasets as _datasets_for_cfg
    _datasets_for_cfg.config.STREAMING_READ_MAX_RETRIES = 50
    _datasets_for_cfg.config.STREAMING_READ_RETRY_INTERVAL = 0.1
    _datasets_for_cfg.config.STREAMING_OPEN_MAX_RETRIES = 10
    _datasets_for_cfg.config.STREAMING_OPEN_RETRY_INTERVAL = 0.5
    _datasets_for_cfg.config.STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL = 1
    _datasets_for_cfg.config.STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL = 2


# --------------------------------------------------------------------------- #
# Socket default timeout                                                      #
# --------------------------------------------------------------------------- #
def pin_socket_default_timeout(seconds: float = 60.0) -> None:
    """Bound streaming HTTP reads against silent CDN stalls.

    Without this, a CDN that accepts the TCP connection then stops
    sending bytes (silent stall, no 503, no TCP RST) leaves
    ``datasets`` / ``modelscope`` blocked in a ``recv()`` until the
    OS TCP keepalive fires (default ~7200s on Linux). Setting
    ``60s`` means a stalled connection raises a timeout exception
    after 1 minute, which the retry config above then turns into a
    transparent reconnect. Healthy chunks of this dataset arrive
    in 1-3s, so 60s is generous headroom for slow CDN hops.
    """
    import socket as _socket
    _socket.setdefaulttimeout(seconds)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def configure_runtime_environment() -> None:
    """Run all env-var shims in the right order.

    Safe to call multiple times — every step is idempotent
    (``os.environ.setdefault`` for the env-var setters;
    ``datasets.config.*`` and ``socket.setdefaulttimeout`` are
    unconditionally re-assigned to the same values, so re-calls
    are a no-op).
    """
    apply_env_from_dotenv()
    pin_hf_endpoint()
    pin_nccl_environment()
    pin_datasets_retry_config()
    pin_socket_default_timeout()
