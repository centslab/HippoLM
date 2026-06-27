# Triton kernel playbook

The patterns below are the ones that worked (or didn't) when
writing the KDA Triton forward / backward. Read this before
opening a new kernel file; the choice of layout / unroll strategy
is the difference between a 30-second first compile and a 15-
minute hang.

The reference for what's "good practice" in this project is the
vendored FLA library at `src/models/ops/_vendored/fla/ops/`.
The KDA kernel there is the production oracle; the patterns
below are either adopted from FLA or learned the hard way when
the FLA pattern didn't apply.

## Working patterns (adopted from FLA, verified in this repo)

### 1. Multi-kernel split, not one big kernel

FLA's chunk-KDA forward is 4-5 separate Triton kernels
(`chunk_kda_fwd_kernel_intra_sub_chunk`,
`chunk_kda_fwd_kernel_inter_solve_fused`,
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
`chunk_gla_fwd_o_gk`, `chunk_local_cumsum`,
`kda_gate_chunk_cumsum`). Backward has 4 kernels
(`chunk_kda_bwd_kernel_dAv`, `chunk_kda_bwd_kernel_dhu`,
`chunk_kda_bwd_kernel_intra`,
`chunk_kda_bwd_kernel_wy_dqkg_fused`).

- **Intermediates (Akk, v_new, h) are saved to GLOBAL tensors
  between kernels and re-loaded by the next.**
- Each kernel is ~300-600 lines and focused on one concern.
- Each compiles independently and fast.

### 2. `@triton.jit(do_not_specialize=['T'])`

FLA kernels do NOT specialize on T (sequence length). T can vary
in varlen / different chunk counts without invalidating the
cache. Our KDA kernels don't have T as a constexpr so this
pattern doesn't apply directly, but the lesson is general:
identify any per-launch parameter that varies (e.g. `is_doc_start`
0/1) and consider whether specializing on it is worth the cache
blowup.

### 3. `@triton.autotune` with explicit `key=[...]`

FLA uses `key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV']` or
similar. The autotune runs once per unique key and caches the
chosen `(num_warps, num_stages, BK, BV)` config. Use
`**autotune_cache_kwargs` to persist the autotune cache to disk
across runs.

### 4. `@triton.heuristics` for runtime decisions

```python
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
```

`IS_VARLEN` becomes a `tl.constexpr` and Triton can dead-code-
eliminate the unused branch. The kernel only specializes on the
relevant dim combinations. Other common ones: `USE_G`,
`STORE_QG`, `STORE_KG`.

### 5. `tl.make_block_ptr` with `boundary_check=(0, 1)`

```python
q_block = tl.make_block_ptr(
    base=q, shape=(B, T, H, K), strides=(stride_qb, stride_qt, stride_qh, stride_qk),
    offsets=(i_b, 0, i_h, 0), block_shape=(1, BT, 1, BK), order=(0, 1, 2, 3),
)
```

Clean block-strided loads; Triton derives the index arithmetic
from the shape/strides/offsets. Used in 100% of FLA kernels. Our
kernel uses raw `ptr + b_idx * stride_b + ...` loads because the
shape arithmetic for the `[L, L, K]` intermediates is hard to
express as block_ptrs; switch to block_ptrs wherever the shape
is rectangular.

### 6. `tl.debug_barrier()` between phases

Explicit synchronization point. FLA uses it in KDA fwd intra
sub-chunk between the diagonal-block compute and the forward-
substitution. Maps to the multi-kernel philosophy at finer
granularity.

### 7. `BK_LIST = [32, 64] if check_shared_mem() else [16, 32]`

Runtime selection of tile size based on shmem budget. Pair with
`check_shared_mem('ampere')` for the Ampere-class 96KB limit
(vs the 99KB we have on the 5060 Ti).

### 8. `num_warps=8` for K=128, `num_warps=4` for K=64

Empirically confirmed optimal at K=128. The dev box's ptxas
handles `num_warps=8` cleanly for K=128 but `num_warps=4` is
~4.5x slower.

## Anti-patterns (tried, DIDN'T help)

### A. `do_not_specialize=['is_doc_start']`

Made K=16 bwd compile time WORSE (300s timeout vs 1.2s cached,
~200s first compile). The Triton cache was wiped (kernel
signature changed), AND the runtime branch in `if is_doc_start ==
1:` may have caused register pressure. Reverted. The win of
splitting `is_doc_start` into 2 specializations (smaller per-
kernel code) outweighs the win of one shared kernel here.

### B. `tl.static_range(NS)` → `range(NS)` for the sub-tile loop

Same negative result. FLA can do `range(...)` for the K-axis sum
because the body is plain (no per-iteration index-dependent
shapes); our sub-tile body has
`ti_mask = (offs_t[:, None] == ti_offs[None, :])` which is
dynamic, so Triton can't optimize per-iteration and the runtime
branch adds register pressure. The reverse is also true: don't
use `tl.static_range` for inner loops whose iteration count
varies across calls.

### C. `tl.exp` → `tl.math.exp2(x * LOG2E)` in the masked-decay path

Tested and HURT K=128 by 34% (169ms → 227ms in the K=128 prod
bf16 tri-only bench). The explicit `* LOG2E` prevents the
compiler from constant-folding some fusions in the
masked/conditional decay paths. **Do not use exp2 in this
kernel** — keep `tl.exp`.

K=64 saw a 32% gain in a side-by-side (1.51x → 2.00x) but the
K=128 regression dominates. If the K is small and constant
across launches, exp2 is fine; if K varies, stick with `tl.exp`.

## Decision tree for a new Triton kernel

For any new Triton kernel in this project, default to:

1. **Split into 3+ focused kernels if the body exceeds ~400
   lines.** Save intermediates (T, Q_dot_K, K_alpha, p, o_first,
   v_new, grad_T, grad_Q_dot_K, grad_v_new, grad_u) to global
   tensors and re-load in the next kernel.
2. **Use `tl.make_block_ptr` with `boundary_check` for all
   block loads.** Fall back to raw `ptr + offsets` only when
   the shape is irregular (e.g. triangular sub-blocks).
3. **Use `@triton.heuristics` for runtime decisions** that
   branch on a Python `bool` (varlen, use_g, store_qg, etc.).
4. **Use `@triton.autotune` with explicit `key=[...]` over
   `(num_warps, num_stages, BK, BV)`** — never hard-code these.
5. **Specialize on `T` only if you can prove the compile time
   stays bounded.** Otherwise `do_not_specialize=['T']`.
6. **Use `tl.exp`, not `exp2`**, in masked-decay paths. The
   `* LOG2E` fusion cost is real.
7. **Use `tl.static_range` only for the small fully-unrolled
   triangular solve (range 0..BC) where per-iteration shapes
   are constant.** Otherwise `range(...)`.
8. **For K=128 head_dim, use `num_warps=8` and a fresh
   autotune cache** (don't reuse a cache warmed for K=64).

## The `autotune_cache_kwargs` pattern

```python
from src.models.ops._vendored.fla.utils import autotune_cache_kwargs

@triton.autotune(
    configs=[...],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
```

The autotune cache is persisted to
`~/.triton/cache/autotune-{hash}.json` (or wherever
`TRITON_CACHE_DIR` points). The first run on a fresh machine
takes the autotune time; subsequent runs reuse the cache. On
a CI box without `~/.triton`, the autotune runs every time —
budget for that.

## Specific anti-patterns unique to this repo

- **DO NOT use `torch.compile` on the KDA path.** The custom
  autograd Functions (`_TiedFusedLCEFunction`,
  `_ShardedEmbedLookup`, the KDA `ChunkKDAFunction`) don't
  compose with it. `torch.compile(model, mode="reduce-overhead")`
  blows up the bwd 26x. Just set
  `torch.set_float32_matmul_precision("high")` and skip compile.
- **DO NOT use `CUDA graphs` on the KDA path.** Triton
  autotune / ptxas compile takes too long to capture in 240s.
  The capture fails and the model runs the slow path.
- **DO NOT bisect `.pyc` cache for "stale source" symptoms.**
  The bug lives in the kernel's cu_seqlens handling, not the
  import cache. `find . -name __pycache__ -exec rm -rf {} \;`
  is a ritual, not a fix.
- **DO NOT propose LR/clipping/init changes to make the
  PyTorch ref "more stable".** It's already stable at the
  model's actual input scale. If a change makes it unstable,
  the input is wrong or the cu_seqlens is wrong; look there.
