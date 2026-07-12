> **Rule**: [`dont-target-v100`](../.claude/rules/dont-target-v100.md) — sm_70 已 drop；改 `base.yml` 不要引入 V100 兼容路径。

# `configs/base.yml` parameter reference

Every non-obvious knob in `configs/base.yml` is documented here.
The yml itself holds only the value (and at most a one-line label)
so that running ``diff`` against it stays noise-free when you only
want to see what changed.

## Model Architecture

- `vocab_size` / `hidden_size` / `tie_word_embeddings` / `use_bias`
  — passed straight to `HippoConfig`. Vocab comes from the Qwen3.5
  tokenizer vendored under `src/tokenizer/`.

## KDA (Kimi Delta Attention)

- `kda_mode` — `chunk` for training (chunkwise recurrent state,
  lets us use the TensorCore fast path); `fused_recurrent` is the
  inference-only path and is not used by `scripts/train.py`.
- `use_short_conv` — depthwise short conv before the recurrent
  state (Kimi Delta's design). Always true on this stack.
- `allow_neg_eigval` — permits the decay term to flip sign across
  the chunk. Off in production; the safe path assumes a
  non-negative decay so the chunkwise cumsum stays in a stable
  range.
- `safe_gate` — enables the M=16 TensorCore fast path in
  `chunk_kda` (bwd intra + dhu + wy_dqkg). Requires
  `lower_bound` so the chunk-cumsummed gate activation stays in a
  numerically safe range. Recommended `lower_bound` is `-5`
  (FLA's default).
- `lower_bound` — clamp on the cumsummed log-gate value. See
  `safe_gate`.
- `conv_size` / `conv_bias` — short-conv kernel size and whether
  to add a per-channel bias. Bias off in production.

## Architecture Depth

- `num_layers` — total transformer blocks (each block contains
  one KDA + one block-attn residual aggregator + one FFN).
- `num_blocks` — number of block-level summaries that the
  AttnRes routes over. The first `num_blocks` layers include the
  block-attn aggregator; the remaining `num_layers - num_blocks`
  layers are intra-block only.

## FFN (SwiGLU)

- `intermediate_size` — SwiGLU hidden width. The linear is
  `(gate, up) → 2 × intermediate_size`; the activation gate halves
  it back.

## Normalization

- `rms_norm_eps` — epsilon for `RMSNorm`. 1e-6 matches the
  Qwen-style value.

## Training

- `seq_len` — total tokens per step (one super-long FFD-packed
  sequence). 262144 = 16 × 16384 = same total tokens/step as the
  legacy `seq_len=16384, gradient_accumulation_steps=16` config.
- `micro_batch_size` — chunk size: the super-long sequence is
  split into `seq_len // micro_batch_size` chunks of this many
  tokens each, with the KDA recurrent state carried between
  chunks (full BPTT). Default 16384 matches the legacy per-
  microbatch length so backward-compat configs continue to train
  one chunk per "old" microbatch.
- `learning_rate` / `weight_decay` — these are no longer top-level
  keys; they live inside the `optimizer:` block below (under
  `optimizer.adamw:` for the AdamW side, `optimizer.muon:` for
  the Muon side). Override either via the nested block in a
  child yml or the corresponding flat CLI flag
  (`--learning_rate`, `--weight_decay`).
- `lr_warmup_steps` / `lr_decay_steps` — WSD (warmup → stable →
  decay) schedule. The decay phase spans the last
  `lr_decay_steps` of training, regardless of `max_steps`.
- `gradient_accumulation_steps` — derived: `n_chunks = seq_len /
  micro_batch_size`. Kept as a config knob for documentation;
  the training loop ignores this value and computes `n_chunks`
  from `seq_len` and `micro_batch_size` directly. Set to any
  value — the new loop only reads `seq_len` and
  `micro_batch_size`.
- `pack_chunk_size` — chunk-aware FFD packing (see
  `src/training/data/collate.py`). Alignment granularity for doc
  boundaries inside a pack. Each doc rounds up to a multiple of
  this size so the KDA chunkwise kernel's state reset lands
  exactly at a doc boundary. `0` means "use head_dim"
  (`HippoConfig` resolves). Rounded up to a multiple of 64
  internally (the vendored KDA kernel's `BT`).
- `pack_buffer_size` — how many input docs the owner prefetcher
  accumulates per packing window (pulls `batch_size *
  pack_buffer_size` docs per window). The packer uses a
  fixed-pack FFD that ALWAYS emits exactly `batch_size` packed
  rows, so the consumer's microbatch batch dim matches
  `batch_size` by contract — overflow docs are dropped and empty
  packs are filled with `pad_id` + `-100` labels. Higher
  `pack_buffer_size` = more candidates for FFD = denser packs
  (less tail-pad per pack), at the cost of one window's prefetch
  latency. With default `B=1` and `pack_buffer_size=8` we only
  pull 8 docs per window, which under-fills a `T=16384` pack;
  raise to 32-64 if you see sparse packs in the First-batch
  heartbeat.

## Optimizer hparams (grouped by optimizer)

`weight_decay` and `lr` are per-optimizer rather than shared:

- `adamw` — 1-D params (RMSNorm / KDA `A_log` / `dt_bias` /
  BlockAttnRes query / `embed_tokens` / `lm_head` / KDA
  short-conv 3-D weights).
- `muon` — 2-D Linear weights (the dominant bulk of params on
  this stack).

Each sub-block accepts `lr` + `weight_decay` (`muon` also takes
`momentum`). Override either the flat CLI flag (`--muon_lr`,
`--muon_weight_decay`, …) or any single nested sub-key in a
child yml; flat keys win over the nested block. To disable Muon-
side weight decay entirely, set
`optimizer.muon.weight_decay: 0.0` (the default).

`adamw.beta1` / `adamw.beta2` follow the merged-accumulator
AdamW design: `m` is the raw sum of microbatch grads (no β1 EMA
across cycles), so `beta1` is stored / serialized / logged but
the `step()` algorithm only consumes `beta2` for `v = β2*v +
(1-β2)*m² + bc2`. Default values match the v0.0.x hardcoded
(0.9, 0.95) so existing checkpoints carry the same math after
a config-only refactor.

`adamw.eps` is the AdamW epsilon inside `sqrt(v/bc2) + eps`.
Default 1e-8 matches the previously-hardcoded value in
`build_param_groups`.

`max_grad_norm` — pre-clip L2 norm (TP-reduced). `<= 0`
disables clipping.

`seed` — RNG seed for replicated-param init on each device.

## Precision

Each entry sets the storage dtype for a tensor class. Supported
dtypes (as of 2026-07-12):

- `fp32`, `fp16`, `bf16` — floating point; no scaling needed.
- All other dtypes (`int8`, `int4`, `mxfp8`) were **removed**
  on 2026-07-12 after long-training runs showed quantization
  error compounding over many gradient-accumulation steps
  and destabilizing optimization. The `TensorPrecision`
  constructor rejects any other dtype at config-parse time.
  The pre-removal code is preserved on the
  `archive/int8-mxfp8-muon` branch.

Scale modes were likewise removed along with the integer
storage formats.

`activations` only accepts the floating dtypes
(`fp32`/`fp16`/`bf16`). `fp16` / `bf16` enable
`torch.amp.autocast` with that dtype; `fp32` disables autocast
and runs a pure FP32 forward.

`muon_momentum` storage. Default is `bf16`. Full-precision
only — see the removal note above. The merged-accumulator
design (`s.mom_buf` doubles as the per-mb grad accumulator)
applies; there is no separate `s.accum` / `s.mom_scale` /
`mxfp8_block_size`.

`adamw_m` / `adamw_v` — AdamW state buffers. Both default to
`bf16`; they follow the merged-accumulator design (see
`adamw.beta1` above).

## Stage

- `stage` — `pretrain` or `sft`. Selects the dataset / packing
  pipeline.

## Data

- `use_modelscope` — dual-source streaming with ModelScope
  preferred (Aliyun CDN is much faster than `hf-mirror.com`).
  On network/connection errors only, we fall back to the HF
  mirror. Set `use_modelscope=false` to force HF.
- `pretrain_multi_source` — stream 4 Ultra-FineWeb-L3 subsets
  in a 2-phase schedule (multi-style first, then QA), with
  English weighted 2x over Chinese. Phase 0:
  `en_multi * 2 + zh_multi * 1`. Phase 1: `en_qa * 2 + zh_qa *
  1`. When all 4 are exhausted, the schedule restarts from
  phase 0 (infinite stream). When false, the single-source
  `pretrain_config` path is used.
- `text_field` — JSON field inside each dataset row to feed the
  tokenizer.
- `tokenizer_path` — path to the vendored tokenizer
  (under `src/tokenizer/`).
- `shuffle` — disable shuffle: the 1000-sample shuffle buffer
  causes random shard access which multiplies per-shard latency
  and makes the first batch take minutes on either CDN.
  Sequential streaming is fine for the v0.0.0 validation run
  (1000 steps).
- `use_dummy_data` — skip streaming + tokenize; use random
  tokens. Cheap smoke test for the model + optimizer + TP code
  path.
- `cache_dir` — local cache directory for the HIPPOLM-side streamed
  parquet shards (the files our `RotatingParquetIterable` downloads).
  Wires through to `HIPPOLM_CACHE_DIR`; see `docs/cache_routing.md`
  for the three-way layout (HIPPOLM / ModelScope SDK / HF datasets
  SDK). `null` = repo-relative `.cache/hippolm/datasets/`. Override
  on the CLI via `--cache_dir /path` (the CLI flag takes precedence
  over the yml value). Note: SDK-side caches are NOT affected by this
  key — set `MODELSCOPE_CACHE` / `HF_HOME` in the env (or the repo's
  `.env`) to relocate them.

## Output

- `log_interval` — number of steps between training-log lines.
- `checkpoint_interval` — number of steps between checkpoint
  saves.
- `checkpoint_keep_last_n` — after each save, prune older
  `checkpoint_step_*.pt` files so at most N remain (newest N
  kept). `null` / `None` keeps all. Set to `0` to keep only
  the just-saved file.

## GPU

- `min_gpu_memory_mb` — fail-fast threshold when enumerating
  GPUs. Skips any device whose total memory is below this.

## TP simulation

- `tp_sim` — single-GPU dev box path: spawn N processes all
  bound to `cuda:0`, transport collectives via gloo. Verifies
  the sharding code paths without owning N physical GPUs. Much
  slower than NVLink NCCL.
- `tp_size` — number of TP ranks (when `tp_sim: true`, all
  pinned to `cuda:0`).

## Misc

- `mb_timing` — per-microbatch timing breakdown (see
  `scripts/cli.py`). `0` disables. When `> 0`, logs
  `data_ms` / `h2d_ms` / `fwd_ms` / `bwd_ms` / `sync_ms` for
  the first N microbatches of every step.
- `empty_cache_between_mb` — release the caching-allocator
  slack pool back to the driver after each microbatch. Frees
  ~4 GB of pool on the 16 GB production config at the cost of
  ~24 ms/mb (+0.7% wall-clock).
- `kda_skip_aqk_akk_saved` — skip saving the KDA per-chunk
  attention statistics `Aqk` + `Akk` in the forward pass;
  recompute them in the backward pass via
  `chunk_kda_fwd_intra`. Trades a small bwd time increase for
  ~32 MiB of saved-tensor memory per KDA layer (~1 GB at 30
  layers).

## FFN NVFP4 path

- `ffn_nvfp4` — W4A16 NVFP4 FFN (research path, disabled as
  of 2026-07-06 — opt-5/2 VRAM initiative). When `True`, every
  FFN linear stores its weight in NVFP4 packed format (E2M1 +
  FP8-e4m3fn 1x16 microblock scales). Activation stays in BF16;
  the matmul itself runs in BF16 (dequantize-on-fwd) since
  PyTorch 2.9.1's `torch._scaled_mm` NVFP4 path requires both
  A and B to be FP4. 2026-07-06: flipped to `False` — only
  the BF16 master is held on GPU, saving the per-layer
  packed_weight (uint8) + scales (fp8_e4m3fn) buffers. At
  TP=1 / 32 layers / 3 NVFP4 modules per FFN, that drop is
  ~324 MiB off the 5060 Ti 16G peak. The matmul becomes plain
  `F.linear(x, weight, bias)` (BF16 GEMM, no dequant
  round-trip). Training loss ratio BF16 / NVFP4 was 1.00x at
  100 steps per `project_w4a16_nvfp4.md`, so this is a clean
  VRAM win.
- `ffn_nvfp4_marlin` — when `True` (and `ffn_nvfp4` is also
  `True`), the FFN forward matmul uses vLLM's Marlin FP4 kernel
  (BF16 MMA + register dequant + `cp.async` double-buffered
  prefetch). ~3.5x speedup over the dequant+cuBLAS path at FFN
  shapes on sm_120 (47-49 TFLOPS). Backward stays as BF16
  matmul (STE for the quantize noise). Requires the prebuilt
  Marlin .so at `src/models/ops/cuda/lib/` (shipped in this
  repo for the 5060 Ti dev box). No-op when `ffn_nvfp4` is
  `False`. See `:mod:docs.marlin_build_pipeline` for the
  build contract and the namespace wrap notes.