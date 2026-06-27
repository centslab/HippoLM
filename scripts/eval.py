"""evalscope driver for HippoLM.

Usage::

    # Default: MMLU, full eval
    python scripts/eval.py --config configs/base.yml \\
        --checkpoint output/20260626_223318/checkpoints/checkpoint_step_4.pt

    # Other datasets (any evalscope benchmark name, e.g. ``mmlu``,
    # ``ceval``, ``gsm8k``, ``arc``, ``hellaswag``):
    python scripts/eval.py --config configs/base.yml \\
        --checkpoint ... --dataset ceval --limit 200

    # MMLU with a specific subset:
    python scripts/eval.py --config configs/base.yml \\
        --checkpoint ... --dataset mmlu --subset philosophy,machine_learning

    # Use a server you've already started:
    python scripts/eval.py --no-server --api-url http://gpu-box:18080/v1

What it does
------------
1. Spawns :mod:`scripts.eval_server` as a subprocess (unless
   ``--no-server``). The server loads the checkpoint and serves an
   OpenAI-compatible API on ``--host:--port``.
2. Polls ``GET /health`` until the server answers (5-min cap).
3. Builds an :class:`evalscope.config.TaskConfig` for the chosen
   benchmark and calls :func:`evalscope.run_task`.
4. Tears the server down on exit (SIGTERM, then SIGKILL after 10 s).

Default benchmark is MMLU. ``--subset`` is forwarded as
``dataset_args.<dataset>.subset_list`` so the run can target one
or more of the 57 MMLU subjects. ``--limit`` is forwarded as the
top-level ``limit`` (each subset is independently truncated to
``limit`` samples; ``None`` runs the whole split).

The script intentionally does NOT re-implement the tokenizer /
chat-template / generation logic — that all lives in
:mod:`scripts.eval_server` and is unit-testable there. The split
keeps the model-serving code orthogonal to the eval-harness glue.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Make ``scripts.*`` and ``src.*`` importable when invoked from
# any cwd. Same convention as the training entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


log = logging.getLogger("hippo.eval")


# ---------------------------------------------------------------------------
# Server lifecycle helpers.
# ---------------------------------------------------------------------------

def _wait_for_health(url: str, timeout_s: float = 300.0) -> None:
    """Block until ``GET url`` returns 200, or raise after timeout.

    evalscope is unhappy when the URL is reachable but the model
    isn't actually loaded yet (it'll 5xx on every chat call), so
    we wait for ``/health`` (200 OK only after model load) rather
    than just TCP connectivity.
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    log.info("server ready: %s", url)
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, socket.timeout) as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"server at {url} not ready after {timeout_s:.0f}s: {last_err!r}")


def _start_server(args, log_path: Path) -> subprocess.Popen:
    """Spawn ``scripts.eval_server`` and return the Popen handle.

    stdout/stderr are tee'd to ``log_path`` so any import-time
    crash (CUDA OOM during model load, missing checkpoint, etc.)
    is captured for debugging — printing the traceback into the
    parent logger is the difference between "the eval just hangs"
    and "the eval fails fast with the actual stack trace".
    """
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "eval_server.py"),
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--tokenizer", args.tokenizer,
        "--model-id", args.model_id,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.hidden_size is not None:
        cmd += ["--hidden-size", str(args.hidden_size)]
    if args.num_layers is not None:
        cmd += ["--num-layers", str(args.num_layers)]
    if args.num_blocks is not None:
        cmd += ["--num-blocks", str(args.num_blocks)]
    if args.num_heads is not None:
        cmd += ["--num-heads", str(args.num_heads)]
    if args.head_dim is not None:
        cmd += ["--head-dim", str(args.head_dim)]
    if args.intermediate_size is not None:
        cmd += ["--intermediate-size", str(args.intermediate_size)]

    log.info("starting eval server: %s", " ".join(cmd))
    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    # Attach the file handle so callers can close it on cleanup.
    proc._hippo_log_file = log_file  # type: ignore[attr-defined]
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM, wait up to 10 s, then SIGKILL.

    Mirrors the training script's teardown philosophy: be patient
    with a clean shutdown (model unloads, CUDA context released),
    but never hang the eval forever on a wedged child.
    """
    if proc.poll() is not None:
        return
    log.info("stopping eval server (pid=%d)", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        log.warning("server did not exit after SIGTERM; sending SIGKILL")
        proc.kill()
        proc.wait(timeout=5.0)
    log_file = getattr(proc, "_hippo_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# evalscope TaskConfig assembly.
# ---------------------------------------------------------------------------

def _build_dataset_args(args) -> dict:
    """Forward dataset-specific knobs (``--subset``, ``--fewshot``).

    Currently only MMLU exposes both knobs; other benchmarks can be
    added here as we wire them up. We deliberately pass the dict
    through ``getattr``-style access so an unknown dataset name
    doesn't crash — evalscope itself will raise a clearer error if
    the benchmark id isn't registered.
    """
    ds_args: dict = {}
    if args.subset:
        ds_args["subset_list"] = [s.strip() for s in args.subset.split(",") if s.strip()]
    if args.fewshot is not None and args.fewshot >= 0:
        ds_args["few_shot_num"] = int(args.fewshot)
    if not ds_args:
        return {}
    return {args.dataset: ds_args}


def _build_task_config(args, api_url: str):
    """Build the :class:`evalscope.config.TaskConfig`.

    Centralised so the same code path is exercised for the
    end-to-end smoke test and the actual benchmark run — the
    smoke test just passes ``--limit 4`` (or similar).
    """
    from evalscope.config import TaskConfig

    cfg = TaskConfig(
        model=args.model_id,
        model_id=args.model_id,
        api_url=api_url,
        api_key="EMPTY",
        datasets=[args.dataset],
        dataset_args=_build_dataset_args(args),
        dataset_dir=args.dataset_dir,
        dataset_hub=args.dataset_hub,
        limit=args.limit,
        eval_batch_size=args.batch_size,
        generation_config={
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        work_dir=args.work_dir,
        seed=args.seed,
        ignore_errors=args.ignore_errors,
        debug=args.debug,
    )
    return cfg


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evalscope benchmarks against a HippoLM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model serving ---
    p.add_argument("--config", default="configs/base.yml",
                   help="HippoConfig YAML overlay (same as train)")
    p.add_argument("--checkpoint", required=True,
                   help="path to checkpoint_step_*.pt from scripts/train.py")
    p.add_argument("--tokenizer", default="src/tokenizer")
    p.add_argument("--model-id", default="hippolm",
                   help="model id reported in /v1/models and used as the eval ``model`` field")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default=None,
                   help="device passed to eval_server (default: cuda if available else cpu)")
    # Dimension overrides; only needed if the YAML doesn't match the
    # checkpoint (e.g. eval-ing a small-config smoke checkpoint with
    # the prod YAML).
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None)

    # --- Server lifecycle ---
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--no-server", action="store_true",
                   help="don't spawn eval_server; use --api-url directly")
    p.add_argument("--api-url", default=None,
                   help="override the server URL (implies --no-server)")
    p.add_argument("--server-log", default=None,
                   help="where to write eval_server's stdout/stderr (default: <work-dir>/eval_server.log)")

    # --- evalscope knobs ---
    p.add_argument("--dataset", default="mmlu",
                   help="benchmark name (any evalscope benchmark id, e.g. mmlu, ceval, gsm8k, arc)")
    p.add_argument("--subset", default=None,
                   help="comma-separated subset list for benchmarks that have subsets (e.g. mmlu subjects)")
    p.add_argument("--fewshot", type=int, default=None,
                   help="override few-shot count (0 for zero-shot); benchmark default otherwise")
    p.add_argument("--limit", type=int, default=None,
                   help="cap each subset to N samples; default runs the full split")
    p.add_argument("--batch-size", type=int, default=8,
                   help="evalscope's eval_batch_size (concurrent chat-completions)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="max new tokens per chat completion (MMLU needs only a few)")
    p.add_argument("--dataset-dir", default="/root/.cache/modelscope/hub/datasets",
                   help="evalscope's local dataset cache")
    p.add_argument("--dataset-hub", default="modelscope", choices=["modelscope", "huggingface"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default="output/eval",
                   help="evalscope's output directory (reports, per-sample logs)")
    p.add_argument("--ignore-errors", action="store_true",
                   help="evalscope's ignore_errors flag; use when one bad sample would otherwise abort the run")
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Server URL resolution. Three modes:
    #   1. --api-url X            → use X verbatim, don't spawn.
    #   2. --no-server            → use http://host:port, don't spawn.
    #   3. (default)              → spawn eval_server, wait for /health.
    if args.api_url:
        api_url = args.api_url
        server_proc: subprocess.Popen | None = None
    elif args.no_server:
        api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
        server_proc = None
    else:
        log_path = Path(args.server_log) if args.server_log else (work_dir / "eval_server.log")
        server_proc = _start_server(args, log_path)
        health_url = f"http://{args.host}:{args.port}/health"
        api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
        try:
            _wait_for_health(health_url)
        except Exception:
            log.exception("server failed to come up; see %s", log_path)
            _stop_server(server_proc)
            return 2

    cfg = _build_task_config(args, api_url)
    log.info("running evalscope: dataset=%s limit=%s api_url=%s",
             args.dataset, args.limit, api_url)

    from evalscope import run_task

    rc = 0
    try:
        # ``run_task`` returns either a dict (single benchmark) or
        # a list of dicts (multi-benchmark). We always pass one
        # benchmark so the dict path is the only one exercised,
        # but we accept either shape for forward-compat.
        result = run_task(cfg)
        if isinstance(result, list):
            for r in result:
                log.info("benchmark %s: metrics=%s",
                         r.get("benchmark_name", "?"),
                         r.get("metrics", r))
        elif isinstance(result, dict):
            log.info("benchmark %s: metrics=%s",
                     result.get("benchmark_name", "?"),
                     result.get("metrics", result))
    except SystemExit as e:
        # evalscope calls ``sys.exit`` on certain fatal config
        # errors; map that to a non-zero exit code without losing
        # the message.
        log.error("evalscope exited with code %s", e.code)
        rc = int(e.code) if isinstance(e.code, int) else 1
    except Exception:
        log.exception("evalscope run failed")
        rc = 1
    finally:
        if server_proc is not None:
            _stop_server(server_proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
