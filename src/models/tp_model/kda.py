"""TP-sharded Kimi Delta Attention.

The KDA op itself is rank-local (each rank runs the full chunk
algorithm on its ``num_heads // world`` heads with a rank-local
recurrent state). Only the ``o_proj`` does an inter-rank
all-reduce.

Per-rank layout (world = TP group size):

- ``qkv_proj``: column-parallel on ``2*key_dim + value_dim``.
  Each rank: ``[2*key_dim + value_dim // world, hidden_size]``
  (see fused QKV below).
- ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d``: depthwise conv
  with per-channel filter; each rank has the channels in its
  head slice. Shape: ``[c_per_rank, 1, conv_size]``.
- ``f_proj1``: column-parallel ``head_v_dim → gate_dim``.
  The first f_proj layer (``fg_first``) is replicated because
  its output (``head_v_dim``) is a per-head bottleneck that
  is the same on every rank.
- ``g_proj1``: column-parallel ``head_v_dim → value_dim``.
  Same replicated-first-layer story as f_proj.
- ``b_proj``: column-parallel on ``num_v_heads``.
- ``A_log``: ``[num_v_heads // world]`` FP32.
- ``dt_bias``: ``[gate_dim // world]`` FP32.
- ``o_norm``: per-head RMSNorm on ``head_v_dim``; weight
  ``[head_v_dim]`` (replicated within each rank's head slice).
- ``o_proj``: row-parallel on ``value_dim``; all-reduces the
  output to ``[hidden_size]``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._primitives import ColumnParallelLinear, RowParallelLinear, get_tp_rank, get_tp_world_size


class TPKDA(nn.Module):
    """TP-sharded Kimi Delta Attention.

    Per-rank layout (world = TP group size):

    - ``qkv_proj``: column-parallel on ``2*key_dim + value_dim``.
      Each rank: ``[2*key_dim + value_dim // world, hidden_size]``
      (see fused QKV below).
    - ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d``: depthwise conv
      with per-channel filter; each rank has the channels in its
      head slice. Shape: ``[c_per_rank, 1, conv_size]``.
    - ``f_proj1``: column-parallel ``head_v_dim → gate_dim``.
      The first f_proj layer (``fg_first``) is replicated because
      its output (``head_v_dim``) is a per-head bottleneck that
      is the same on every rank.
    - ``g_proj1``: column-parallel ``head_v_dim → value_dim``.
      Same replicated-first-layer story as f_proj.
    - ``b_proj``: column-parallel on ``num_v_heads``.
    - ``A_log``: ``[num_v_heads // world]`` FP32.
    - ``dt_bias``: ``[gate_dim // world]`` FP32.
    - ``o_norm``: per-head RMSNorm on ``head_v_dim``; weight
      ``[head_v_dim]`` (replicated within each rank's head slice).
    - ``o_proj``: row-parallel on ``value_dim``; all-reduces the
      output to ``[hidden_size]``.

    The chunk-KDA kernel itself is rank-local: each rank runs the
    full chunk algorithm on its ``H // world`` heads with a
    rank-local recurrent state. No inter-rank communication is
    required inside the KDA op — only the o_proj all-reduce.
    """

    def __init__(self, config, layer_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        from src.models.ops._vendored.fla.modules import FusedRMSNormGated, ShortConvolution
        from src.models.ops._vendored.fla.ops.kda import chunk_kda

        self._chunk_kda = chunk_kda  # cache for forward

        self.world = get_tp_world_size()
        self.rank = get_tp_rank()

        h = config.hidden_size
        nh = config.num_heads
        nvh = config.num_heads  # no GVA
        d_h = config.head_dim
        d_v = int(d_h * config.expand_v)
        key_dim = nh * d_h
        value_dim = nvh * d_v
        gate_dim = nvh * d_h  # gate_dim = num_v_heads * head_k_dim

        assert nh % self.world == 0, f"num_heads={nh} not divisible by tp_world={self.world}"
        assert nvh % self.world == 0, f"num_v_heads={nvh} not divisible by tp_world={self.world}"
        assert value_dim % self.world == 0, f"value_dim={value_dim} not divisible by tp_world={self.world}"
        assert gate_dim % self.world == 0, f"gate_dim={gate_dim} not divisible by tp_world={self.world}"

        self.hpp = nh // self.world           # heads per partition (qk and v share this)
        self.key_per_partition = key_dim // self.world
        self.value_per_partition = value_dim // self.world
        self.gate_per_partition = gate_dim // self.world
        self.head_k_dim = d_h
        self.head_v_dim = d_v
        self.num_heads = nh
        self.num_v_heads = nvh
        self.hidden_size = h
        self.expand_v = config.expand_v
        self.mode = config.kda_mode
        self.use_short_conv = config.use_short_conv
        self.conv_size = config.conv_size
        self.conv_bias = config.conv_bias
        self.allow_neg_eigval = config.allow_neg_eigval
        self.safe_gate = config.safe_gate
        self.lower_bound = config.lower_bound
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        # KDA bwd saved-tensor optimization: skip saving Aqk + Akk
        # in fwd, recompute in bwd via chunk_kda_fwd_intra.
        self.skip_aqk_akk_saved = getattr(config, "kda_skip_aqk_akk_saved", False)

        # ---- Fused QKV projection (column-parallel) ---- #
        # One ColumnParallelLinear with output = 2*key_dim + value_dim.
        # Output channels: [0:key_per_partition] = Q,
        # [key_per_partition:2*key_per_partition] = K,
        # [2*key_per_partition:2*key_per_partition+value_per_partition] = V.
        # Saves 2 kernel launches per KDA layer (3 -> 1).
        self.qkv_proj = ColumnParallelLinear(
            h, 2 * key_dim + value_dim, bias=False,
            device=device, dtype=dtype,
        )

        # ---- Depthwise conv1d (per-channel, sharded) ---- #
        if config.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_per_partition, kernel_size=config.conv_size,
                bias=config.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_per_partition, kernel_size=config.conv_size,
                bias=config.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_per_partition, kernel_size=config.conv_size,
                bias=config.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)

        # ---- Fused FG first layer + split FG second layer ---- #
        # The first layer of each is ``hidden → head_v_dim`` where
        # ``head_v_dim`` is a per-head bottleneck (the same on every
        # rank), so it's a vanilla ``nn.Linear`` (replicated). We
        # fuse the two first layers into one ``nn.Linear`` with
        # output = 2*d_v (saves 1 kernel launch per KDA layer).
        #
        # The second layers are NOT fused: packing f and g into one
        # big ``ColumnParallelLinear(2*d_v, gate_dim + value_dim)``
        # requires a weight matrix with 50% zeros (the off-diagonal
        # blocks). The fused matmul would execute those zero*input
        # FMAs, doubling the FLOPs of this layer with no compute
        # benefit. So we keep f_proj[1] and g_proj[1] as separate
        # ColumnParallelLinear's. (Tested empirically in
        # test_fused_proj_bench.py: fusing FG-second makes TPKDA
        # ~5% slower.)
        self.fg_first = nn.Linear(
            h, 2 * d_v, bias=False, device=device, dtype=dtype,
        )
        self.f_proj1 = ColumnParallelLinear(
            d_v, gate_dim, bias=False, device=device, dtype=dtype,
        )
        self.g_proj1 = ColumnParallelLinear(
            d_v, value_dim, bias=True, device=device, dtype=dtype,
        )

        # ---- b_proj: column-parallel on num_v_heads ---- #
        self.b_proj = ColumnParallelLinear(h, nvh, bias=False, device=device, dtype=dtype)

        # ---- A_log, dt_bias: sharded along num_v_heads ---- #
        if config.safe_gate:
            self.A_log = nn.Parameter(
                torch.zeros(self.hpp, dtype=torch.float32, device=device)
            )
        else:
            self.A_log = nn.Parameter(
                torch.log(
                    torch.empty(self.hpp, dtype=torch.float32, device=device).uniform_(1, 16)
                )
            )
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.gate_per_partition, dtype=torch.float32, device=device) *
            (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ---- o_norm: per-head RMSNorm on head_v_dim ---- #
        # weight has shape [head_v_dim] = [d_v]; same value applied
        # to each head on this rank. Replicated within rank.
        self.o_norm = FusedRMSNormGated(
            d_v, activation="sigmoid", eps=self.norm_eps,
            device=device, dtype=dtype,
        )

        # ---- o_proj: row-parallel on value_dim ---- #
        self.o_proj = RowParallelLinear(
            value_dim, h, bias=False, device=device, dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. ``x`` is the full hidden (replicated
        across the TP group). Returns ``(output, final_state)``
        where ``output`` is the residual stream contribution
        (after the ``o_proj`` all-reduce) and ``final_state``
        is the KDA recurrent state at the last token (shape
        ``[1, hpp, head_k_dim, head_v_dim]`` in float32).

        ``cu_seqlens`` (when set) is forwarded only to
        ``ShortConvolution`` so the depthwise conv kernel's
        causal state resets at each doc boundary. The KDA
        recurrence itself sees the whole chunk as one stream
        (no per-doc reset) — this is the chunked-training
        invariant: state flows across chunks within a step.

        ``initial_state`` is the KDA recurrent state at the
        start of this chunk. ``None`` means "start from zeros"
        (the legacy / first-chunk-of-step behavior). For
        subsequent chunks in a step, pass the ``final_state``
        returned by the previous chunk's forward. State dtype
        must be float32 (the FLA kernel's contract).
        """
        from einops import rearrange

        # Packed path: fold batch dim into seq dim. The QKV /
        # FG / gate projections are just matmuls and don't care
        # about the leading-dim interpretation; only the
        # chunkwise KDA recurrence cares, and we now always
        # feed it the ``[B, T, ...]`` shape (no cu_seqlens on
        # the KDA call, so it processes the whole chunk as one
        # stream). ``cu_seqlens`` is still threaded into the
        # ShortConvolution call to keep the depthwise conv's
        # causal state from blurring across unrelated docs.
        B, T, hidden = x.shape
        x_in = x  # KDA sees [B, T, hidden] without cu_seqlens-driven flatten

        # Fused QKV: one matmul produces [B, T, 2*key+v] which
        # we split into Q / K / V per rank. Saves 2 kernel
        # launches vs. 3 separate projections.
        qkv = self.qkv_proj(x_in)
        q, k, v = qkv.split(
            [self.key_per_partition, self.key_per_partition, self.value_per_partition],
            dim=-1,
        )

        if self.use_short_conv:
            # Depthwise conv1d on each rank's channel slice.
            # The kernel expects [B, T, C] -> [B, T, C] with
            # causal padding, applied per-channel.
            # ``ShortConvolution.forward`` returns ``(y, cache)``;
            # we discard the cache because training uses the
            # chunk path (no incremental decoding). When
            # ``cu_seqlens`` is set the vendored
            # :class:`ShortConvolution` resets the causal state at
            # each doc boundary so a depthwise conv kernel does
            # not blur tokens across an unrelated document.
            q, _ = self.q_conv1d(q, cu_seqlens=cu_seqlens)
            k, _ = self.k_conv1d(k, cu_seqlens=cu_seqlens)
            v, _ = self.v_conv1d(v, cu_seqlens=cu_seqlens)
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # FG path: one nn.Linear (``fg_first``) produces the
        # concatenated f/g intermediates; two separate
        # ColumnParallelLinear's (``f_proj1``, ``g_proj1``) project
        # each to its output. We keep the second layers split
        # because fusing them would require a 50%-zero weight
        # matrix and double the FLOPs of this layer (see comment
        # in __init__).
        fg_first = self.fg_first(x_in)
        f_inter, g_inter = fg_first.chunk(2, dim=-1)
        g = self.f_proj1(f_inter)
        g_for_norm = self.g_proj1(g_inter)
        beta = self.b_proj(x_in).sigmoid()

        # Reshape to per-head: each rank has ``hpp`` heads.
        # q, k: [B, T, hpp, head_k_dim]
        # g:    [B, T, hpp, head_k_dim]  (gate_dim/world = hpp*head_k_dim)
        # v:    [B, T, hpp, head_v_dim]
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # chunk_kda: rank-local. No TP comm inside the kernel.
        # The chunked-training invariant: NO ``cu_seqlens`` here,
        # so the KDA recurrence sees the whole chunk as one stream
        # and the recurrent state can be carried across chunks
        # via ``initial_state`` / ``output_final_state=True``. Doc
        # boundaries inside the chunk are NOT resets — that signal
        # is only used by ShortConvolution above (depthwise conv
        # state) and the label-mask in the loss head.
        o, final_state = self._chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=self.A_log, dt_bias=self.dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            skip_aqk_akk_saved=self.skip_aqk_akk_saved,
        )

        # o has shape ``[B, T, hpp, head_v_dim]`` (no flatten
        # happened — KDA sees the natural ``[B, T, ...]`` shape
        # because we no longer thread ``cu_seqlens`` to it).
        # g_for_norm is already ``[B, T, hpp, head_v_dim]`` from
        # the ``g_proj1`` rearrange above.

        # o_norm: per-head RMSNorm on the last dim.
        o = self.o_norm(o, g_for_norm)
        o = rearrange(o, "b t h d -> b t (h d)")
        # o_proj: row-parallel, all-reduces to [B, T, hidden_size].
        o = self.o_proj(o)
        # Return the residual contribution AND the final KDA
        # recurrent state so the caller can carry it into the next
        # chunk. ``final_state`` is shape
        # ``[1, hpp, head_k_dim, head_v_dim]`` in float32
        # (matches ``initial_state`` contract).
        return o, final_state
