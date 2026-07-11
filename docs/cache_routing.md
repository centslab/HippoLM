# Local cache routing

Every local cache the training stack writes or reads is routed through
one anchor: **`<repo_root>/.cache/`**. Three sibling subtrees live
under it, each owned by a different subsystem:

```
<repo_root>/.cache/
├── hippolm/datasets/        # HIPPOLM-side parquet shards
├── modelscope/              # MODELSCOPE_CACHE
└── huggingface/datasets/    # HF_HOME + HF_DATASETS_CACHE
```

This document covers:

1. **Why** we route all three caches through one anchor.
2. **Where** each cache is configured (env var / CLI flag / yml key).
3. **The timing constraint** that keeps `import datasets` from
   snapshotting `HF_DATASETS_CACHE` to the wrong value.

## Why a single anchor

The project's hardware target spans 4090 (prod), 5060 Ti 16G (dev),
and various cloud spot instances. On those boxes:

- `$HOME` is not stable across runs (containers, CI runners, shared
  multi-tenant hosts).
- RAID mount naming varies by cloud provider (`/mnt/disk1`,
  `/data/raid0`, `/hy-tmp`, ...).
- `$HOME` often has tight disk quotas.

The previous default for the HIPPOLM-side parquet cache was
`~/.cache/hippolm/datasets/`. That caused cross-environment pollution
— a CI runner's leftover cache would land in the developer's home
directory. In 2026-06-27 we moved the HIPPOLM-side default to
`<repo>/.cache/hippolm/datasets/`.

After several more days of development we observed that two
**third-party SDK caches** were still drifting back to `~/.cache/`:

- `modelscope` (the ModelScope SDK) writes its own cache to
  `~/.cache/modelscope/` (hardcoded at
  `modelscope/hub/utils/utils.py:190`:
  `default_cache_dir = Path.home().joinpath('.cache', 'modelscope')`).
- `datasets` (the HF datasets SDK) writes its cache to
  `~/.cache/huggingface/datasets/` (resolved via
  `HF_HOME` / `HF_DATASETS_CACHE` env vars, defaulting to home-relative).

The drift is silent — there is no warning, no error, no log line.
The only symptom is the wrong disk filling up.

The fix: pin **both SDK caches** to `<repo>/.cache/<vendor>/` so the
entire cache tree lives under one root the user controls via the repo
checkout location.

## Where each cache is configured

### HIPPOLM-side parquet shards (`<repo>/.cache/hippolm/datasets/`)

The files we actually download — one parquet shard per
`(ms_dataset, subset, part_index)` triple, written by
`RotatingParquetIterable` in `src/training/data/rotating.py`.

Resolution (`src/training/data/cache.py:get_cache_dir()`):

1. `HIPPOLM_CACHE_DIR` env var (explicit override).
2. `<repo>/.cache/hippolm/datasets/` default
   (`_PROJECT_CACHE_ROOT = Path(__file__).resolve().parents[3] / ...`).

CLI surface: `--cache_dir /path` (see `scripts/cli.py:build_parser`).
Yml surface: `cache_dir: /path` in `configs/base.yml`. Both wire
through to `HIPPOLM_CACHE_DIR` via module-level wiring in
`scripts/train.py` (see "Timing constraint" below).

### ModelScope SDK cache (`<repo>/.cache/modelscope/`)

The ModelScope SDK's own internal cache (metadata for `MsDataset`,
HTTP client caches, etc.). Pinned via `MODELSCOPE_CACHE`.

Resolution (`src/training/env.py:pin_cache_directories()`):

- Default: `<repo>/.cache/modelscope/`.
- Override: `export MODELSCOPE_CACHE=/path` in the env or `.env` file.

### HF datasets SDK cache (`<repo>/.cache/huggingface/datasets/`)

The HF datasets SDK's own cache (resolved file metadata, downloaded
config artifacts, etc.). Pinned via `HF_HOME` and
`HF_DATASETS_CACHE`.

Resolution (`src/training/env.py:pin_cache_directories()`):

- Default: `HF_HOME=<repo>/.cache/huggingface/`,
  `HF_DATASETS_CACHE=<repo>/.cache/huggingface/datasets/`.
- Override: `export HF_HOME=/path` or `export HF_DATASETS_CACHE=/path`
  in the env or `.env` file.

## Override precedence

`pin_cache_directories()` uses `os.environ.setdefault` semantics — an
explicit export in the caller's shell (or the repo's `.env` file)
always wins over the project-relative default. The CLI flag
`--cache_dir` does NOT route SDK caches; it only sets
`HIPPOLM_CACHE_DIR`. This is intentional:

- `--cache_dir` is the answer to "I want to move the project's own
  parquet cache off this disk" — typically a disk-space issue with
  our own downloads.
- `MODELSCOPE_CACHE` / `HF_HOME` are the answer to "I want to route
  all data traffic through one fast SSD" — typically a network or
  CDN performance issue.

If the user wants to unify everything (HIPPOLM + SDKs) to a single
external disk, they should set both: `--cache_dir /mnt/big/hippolm`
AND `export MODELSCOPE_CACHE=/mnt/big/modelscope`
AND `export HF_HOME=/mnt/big/huggingface`. Each knob does one thing.

## Timing constraint

The HF datasets SDK snapshots `HF_DATASETS_CACHE` into a module
constant (`datasets.config.HF_DATASETS_CACHE`) **at import time**.
Later `os.environ["HF_DATASETS_CACHE"] = ...` is silently ignored.
Verified empirically: `datasets.config.HF_DATASETS_CACHE` resolves to
`/root/.cache/huggingface/datasets/` if `datasets` is imported before
`HF_DATASETS_CACHE` is set, and to the pinned value if set before.

This forces a specific ordering in `scripts/train.py`:

```python
# scripts/train.py — module-level (top of file)

# 1. Parse CLI + YAML (import-light: scripts.cli only needs
#    argparse + lazy yaml).
from scripts.cli import parse_args as _full_parse_args
_args = _full_parse_args(sys.argv[1:])
if getattr(_args, "cache_dir", None):
    os.environ["HIPPOLM_CACHE_DIR"] = str(
        Path(_args.cache_dir).expanduser().resolve()
    )

# 2. Pin SDK caches via env.py BEFORE configure_runtime_environment
#    runs pin_datasets_retry_config (which imports datasets).
from src.training.env import configure_runtime_environment
configure_runtime_environment()
#    ↑ Inside: apply_env_from_dotenv → pin_hf_endpoint →
#      pin_nccl_environment → pin_cache_directories →
#      pin_datasets_retry_config (← imports datasets here, by which
#      point HF_DATASETS_CACHE is already set) →
#      pin_socket_default_timeout

# 3. NOW safe to import torch / datasets / modelscope (any of which
#    may transitively import datasets).
import torch
```

The order is enforced by
`test/test_env_cache_directories.py::test_configure_runs_pin_before_datasets_import`
which monkeypatches `pin_cache_directories` and
`pin_datasets_retry_config` to record call order.

**Do not move `parse_args` into `main()`** or the early env-var setup
becomes a `main()`-time side effect that runs AFTER `import datasets`
has already frozen `HF_DATASETS_CACHE` to `~/.cache/huggingface/datasets/`.

## File layout summary

| Cache | Env var(s) | Owner | Default path |
|---|---|---|---|
| HIPPOLM-side parquet | `HIPPOLM_CACHE_DIR` | `src/training/data/cache.py` | `<repo>/.cache/hippolm/datasets/` |
| ModelScope SDK | `MODELSCOPE_CACHE` | `modelscope` package | `<repo>/.cache/modelscope/` |
| HF datasets SDK | `HF_HOME`, `HF_DATASETS_CACHE` | `datasets` package | `<repo>/.cache/huggingface/datasets/` |

All three are pinned by the same function —
`src/training/env.py:pin_cache_directories()` — called once at startup
by `configure_runtime_environment()`.

## Testing

- `test/test_cache_dir.py` (12 cases) — HIPPOLM-side resolution
  (`HIPPOLM_CACHE_DIR` env var > default; default is absolute;
  empty env var treated as unset; idempotent dir creation).
- `test/test_env_cache_directories.py` (5 cases) — SDK pinning layer
  (default layout under `<repo>/.cache/<vendor>/`; explicit exports
  survive `setdefault`; `pin_cache_directories` runs before
  `pin_datasets_retry_config`; HIPPOLM-side var is not touched by
  the SDK pinning layer).