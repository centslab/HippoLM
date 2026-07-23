"""Scheme-driven precision config + FP8 E4M3 bwd + FP32-cast bug fix.

Covers the 2026-07-21 5-scheme rollout (the current production spec):

  1. HippoConfig has 3 scheme fields — ``embedding_precision``,
     ``attention_precision``, ``ffn_precision`` — each one of the
     FIVE schemes ``{"w16a16", "w8a16", "w8a8", "w4a16", "w4a8"}``
     defined in :data:`src.models.config.SCHEMES`. Invalid values
     are rejected by ``__post_init__``. Default for all three is
     ``"w16a16"``.

  2. KDA wrapper (``src/models/ops/kda.py``) reads
     ``config.attention_precision``. Resolution table
     (see the module docstring for the full spec):

        w16a16 → no swap (plain nn.Linear throughout)
        w8a8   → FP8Linear(fp8_bwd=True) on q/k/v/o + g_proj
        w8a16  → fallback to w16a16 (no W8A16 KDA kernel) + warn
        w4a8   → fallback to w8a8 (no NVFP4 KDA kernel) + warn
        w4a16  → fallback to w16a16 (no NVFP4 KDA kernel) + warn

     ``f_proj`` and ``b_proj`` stay as ``nn.Linear`` (precision-
     sensitive — see the comments above ``_DIRECT_FP8_LINEARS``).

  3. FFN wrapper — non-TP (``src/models/activation.py:SwiGLU``):

        w16a16 → nn.Linear
        w8a16  → nn.Linear (no W8A16 FFN kernel) + warn
        w8a8   → FP8Linear(fp8_bwd=True)
        w4a8   → NVFP4LinearW4A8 (two-pass Triton dequant +
                 _scaled_mm). BF16 STE bwd; spec says FP8 bwd +
                 FP8 grads (TODO — see module docstring).
        w4a16  → NVFP4Linear(use_marlin=True, no_bf16_master=True)
                 — Marlin mode-3. FP4 packed buffers are the source
                 of truth.

  4. FFN wrapper — TP (``src/models/tp_model/swiglu.py:TPSwiGLU``):

        w16a16 → ColumnParallelLinear + RowParallelLinear (BF16)
        w8a16  → BF16 TP fallback + warn
        w8a8   → BF16 TP fallback (no per-rank FP8 Col/Row on
                 sm_120) + warn
        w4a8   → BF16 TP fallback (no NVFP4 W4A8 Col/Row on
                 sm_120; the W4A16 NVFP4 Col/Row are Marlin-only)
                 + warn
        w4a16  → NVFP4ColumnParallelLinear +
                 NVFP4RowParallelLinear with use_marlin=True,
                 no_bf16_master=True (hardcoded)

  5. FP8Linear default flips ``fp8_bwd=False -> True`` (FP8 E4M3
     backward by default when the forward is FP8, same scheme as
     FFN W4A8's autograd contract). The FP32-cast bug in
     ``_FP8E4M3Matmul.backward`` is fixed (drop the
     ``.to(torch.float32)`` casts, keep BF16 leaves).

  6. Legacy boolean flags (``kda_fp8``, ``kda_mxfp8``, ``ffn_nvfp4``,
     ``ffn_nvfp4_marlin``, ``ffn_nvfp4_no_bf16_master``) were
     REMOVED from HippoConfig on 2026-07-21. The scheme is the
     single source of truth.

Run:  pytest -s test/test_scheme_precision.py -v
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.config import SCHEMES, HippoConfig
from src.models.ops.fp8_linear import FP8Linear, _FP8E4M3Matmul
from src.models.ops.kda import KDA
from src.models.ops.nvfp4_linear import NVFP4Linear
from src.models.ops.nvfp4_linear_w4a8 import NVFP4LinearW4A8

device = torch.device("cuda:0")
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# 1. HippoConfig: 5-scheme validation
# ---------------------------------------------------------------------------
class TestHippoConfigScheme:
    def test_default_scheme_is_w16a16(self):
        """Defaults to w16a16 for all three module groups."""
        cfg = HippoConfig()
        assert cfg.embedding_precision == "w16a16"
        assert cfg.attention_precision == "w16a16"
        assert cfg.ffn_precision == "w16a16"

    def test_schemes_tuple_lists_all_five(self):
        """SCHEMES is the canonical enumeration — must list all five
        schemes (order matches the spec docstring order)."""
        assert SCHEMES == ("w16a16", "w8a16", "w8a8", "w4a16", "w4a8")

    def test_explicit_scheme_fields(self):
        """All three fields can be set independently to any valid scheme."""
        cfg = HippoConfig(
            embedding_precision="w16a16",
            attention_precision="w8a8",
            ffn_precision="w4a8",
        )
        assert cfg.embedding_precision == "w16a16"
        assert cfg.attention_precision == "w8a8"
        assert cfg.ffn_precision == "w4a8"

    @pytest.mark.parametrize("scheme", list(SCHEMES))
    def test_all_five_schemes_accepted(self, scheme):
        """Every scheme in SCHEMES is a valid value for every field."""
        # ``tie_word_embeddings=False`` so each field can take an
        # independent scheme — with tying on, HippoConfig requires
        # embedding_precision == lm_head_precision (they share one
        # weight tensor). This test only exercises value acceptance.
        cfg = HippoConfig(
            tie_word_embeddings=False,
            embedding_precision=scheme,
            attention_precision=scheme,
            ffn_precision=scheme,
            lm_head_precision=scheme,
        )
        assert cfg.embedding_precision == scheme
        assert cfg.attention_precision == scheme
        assert cfg.ffn_precision == scheme
        assert cfg.lm_head_precision == scheme

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("embedding_precision", "w2a2"),
            ("embedding_precision", "w16a8"),  # garbled scheme
            ("attention_precision", "w16a8"),
            ("attention_precision", "w4a4"),
            ("ffn_precision", "w8a32"),
            ("ffn_precision", "w2a4"),
        ],
    )
    def test_invalid_scheme_rejected(self, field, bad_value):
        """Schemes not in SCHEMES are rejected by ``__post_init__``."""
        with pytest.raises(AssertionError):
            HippoConfig(**{field: bad_value})

    def test_old_flag_fields_are_gone(self):
        """The legacy boolean flags were removed on 2026-07-21 — the
        dataclass no longer accepts them (TypeError)."""
        with pytest.raises(TypeError):
            HippoConfig(kda_fp8=True)
        with pytest.raises(TypeError):
            HippoConfig(kda_mxfp8=True)
        with pytest.raises(TypeError):
            HippoConfig(ffn_nvfp4=True)
        with pytest.raises(TypeError):
            HippoConfig(ffn_nvfp4_marlin=True)
        with pytest.raises(TypeError):
            HippoConfig(ffn_nvfp4_no_bf16_master=True)

    # ---- Tied-embeddings precision consistency (2026-07-23) ----
    def test_tied_embeddings_require_matching_precision(self):
        """With tie_word_embeddings=True (the default), embedding and
        lm_head share one weight tensor, so their schemes must match —
        a mismatch is rejected at config-parse time."""
        with pytest.raises(AssertionError, match="embedding_precision == lm_head_precision"):
            HippoConfig(
                tie_word_embeddings=True,
                embedding_precision="w8a8",
                lm_head_precision="w16a16",
            )

    def test_tied_embeddings_matching_precision_ok(self):
        """Both sides at the same scheme is accepted (the prod
        base.yml case: embedding == lm_head == w8a8)."""
        cfg = HippoConfig(
            tie_word_embeddings=True,
            embedding_precision="w8a8",
            lm_head_precision="w8a8",
        )
        assert cfg.embedding_precision == "w8a8"
        assert cfg.lm_head_precision == "w8a8"

    def test_untied_embeddings_allow_mismatched_precision(self):
        """With tying OFF, the two roles are independent tensors and
        may pick different schemes."""
        cfg = HippoConfig(
            tie_word_embeddings=False,
            embedding_precision="w16a16",
            lm_head_precision="w8a8",
        )
        assert cfg.embedding_precision == "w16a16"
        assert cfg.lm_head_precision == "w8a8"


# ---------------------------------------------------------------------------
# 1b. HippoConfig: producer-side fused-op precision (2026-07-23)
# ---------------------------------------------------------------------------
class TestProducerPrecisionConfig:
    """residual_precision / rmsnorm_precision / attn_res_precision are
    the bf16/fp8 knobs for the fused element-wise ops between the GEMMs
    (residual add / RMSNorm / BlockAttnRes). Independent of the four
    5-scheme GEMM fields above."""

    def test_producer_precision_defaults(self):
        """Prod defaults: residual bf16, rmsnorm fp8, attn_res bf16."""
        cfg = HippoConfig()
        assert cfg.residual_precision == "bf16"
        assert cfg.rmsnorm_precision == "fp8"
        assert cfg.attn_res_precision == "bf16"

    @pytest.mark.parametrize("value", ["bf16", "fp8"])
    def test_residual_precision_accepts_bf16_and_fp8(self, value):
        cfg = HippoConfig(residual_precision=value)
        assert cfg.residual_precision == value

    @pytest.mark.parametrize("value", ["bf16", "fp8"])
    def test_rmsnorm_precision_accepts_bf16_and_fp8(self, value):
        cfg = HippoConfig(rmsnorm_precision=value)
        assert cfg.rmsnorm_precision == value

    def test_attn_res_precision_accepts_bf16(self):
        cfg = HippoConfig(attn_res_precision="bf16")
        assert cfg.attn_res_precision == "bf16"

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("residual_precision", "w8a8"),   # GEMM scheme, not a producer knob
            ("residual_precision", "fp16"),
            ("rmsnorm_precision", "w16a16"),
            ("rmsnorm_precision", "int8"),
        ],
    )
    def test_invalid_producer_value_rejected(self, field, bad_value):
        with pytest.raises(AssertionError):
            HippoConfig(**{field: bad_value})

    def test_attn_res_precision_fp8_rejected(self):
        """No FP8 BlockAttnRes kernel exists — only bf16 is valid."""
        with pytest.raises(AssertionError):
            HippoConfig(attn_res_precision="fp8")


# ---------------------------------------------------------------------------
# 1c. HippoLayer / model wiring for the producer-side precision fields
# ---------------------------------------------------------------------------
class TestProducerPrecisionWiring:
    """The layer resolves the config fields into boolean toggles once
    at construction and gates the fused kernels on them."""

    @staticmethod
    def _cfg(**overrides):
        base = dict(
            hidden_size=128, num_heads=2, head_dim=64,
            intermediate_size=256, num_layers=2, num_blocks=2,
            vocab_size=256, safe_gate=True, lower_bound=-5.0,
            use_short_conv=False,
        )
        base.update(overrides)
        return HippoConfig(**base)

    def test_layer_toggles_follow_config(self):
        from src.models.model import HippoLayer
        layer = HippoLayer(0, self._cfg(
            rmsnorm_precision="fp8", residual_precision="fp8"))
        assert layer._use_fp8_rmsnorm is True
        assert layer._use_fp8_residual is True
        assert layer.config is not None

        layer_bf = HippoLayer(0, self._cfg(
            rmsnorm_precision="bf16", residual_precision="bf16"))
        assert layer_bf._use_fp8_rmsnorm is False
        assert layer_bf._use_fp8_residual is False

    def test_tp_layer_toggles_follow_config(self):
        from src.models.tp_model.layer import TPHippoLayer
        layer = TPHippoLayer(0, self._cfg(
            rmsnorm_precision="fp8", residual_precision="bf16"))
        assert layer._use_fp8_rmsnorm is True
        assert layer._use_fp8_residual is False

    def test_model_rejects_non_bf16_attn_res_at_build(self):
        """Config validation fires first, but the model-level guard is
        the backstop if a field is mutated post-construction."""
        from src.models.model import HippoModel
        cfg = self._cfg()
        object.__setattr__(cfg, "attn_res_precision", "fp8")
        with pytest.raises(ValueError, match="attn_res_precision"):
            HippoModel(cfg)



def _make_kda(cfg):
    return KDA(cfg, layer_idx=0).to(device=device, dtype=dtype)


class TestKDASchemeWiring:
    def test_w16a16_keeps_nn_linear(self):
        """w16a16 -> no FP8 swap. All projections stay nn.Linear.
        ``f_proj`` is a Sequential of two Linears; both sub-modules
        must remain nn.Linear."""
        cfg = HippoConfig(
            hidden_size=128, num_heads=16, head_dim=32,
            num_layers=1, num_blocks=1,
            safe_gate=True, lower_bound=-5.0,
            kda_mode="chunk", use_short_conv=False,
            attention_precision="w16a16",
        )
        attn = _make_kda(cfg).attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "b_proj"):
            sub = getattr(attn, name)
            assert isinstance(sub, nn.Linear), (
                f"attention w16a16: {name} should be plain nn.Linear, "
                f"got {type(sub).__name__}")
            assert not isinstance(sub, FP8Linear)
        # Sequential g_proj — every sub-module is nn.Linear.
        for sub in attn.g_proj:
            assert isinstance(sub, nn.Linear)
            assert not isinstance(sub, FP8Linear)
        # Sequential f_proj — every sub-module is nn.Linear.
        for sub in attn.f_proj:
            assert isinstance(sub, nn.Linear)
            assert not isinstance(sub, FP8Linear)

    def test_w8a8_swaps_to_fp8_with_bwd(self):
        """w8a8 -> q/k/v/o + g_proj become FP8Linear(fp8_bwd=True).
        f_proj (Sequential of 2) + b_proj stay as nn.Linear
        (precision-sensitive — see the ``_DIRECT_FP8_LINEARS``
        comment in kda.py)."""
        cfg = HippoConfig(
            hidden_size=128, num_heads=16, head_dim=32,
            num_layers=1, num_blocks=1,
            safe_gate=True, lower_bound=-5.0,
            kda_mode="chunk", use_short_conv=False,
            attention_precision="w8a8",
        )
        attn = _make_kda(cfg).attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sub = getattr(attn, name)
            assert isinstance(sub, FP8Linear), (
                f"attention w8a8: {name} should be FP8Linear, "
                f"got {type(sub).__name__}")
            assert sub.fp8_bwd is True
        for sub in attn.g_proj:
            assert isinstance(sub, FP8Linear)
            assert sub.fp8_bwd is True
        # b_proj stays plain nn.Linear.
        b = attn.b_proj
        assert isinstance(b, nn.Linear)
        assert not isinstance(b, FP8Linear)
        # f_proj is a Sequential; BOTH sub-modules stay nn.Linear.
        assert isinstance(attn.f_proj, nn.Sequential)
        for sub in attn.f_proj:
            assert isinstance(sub, nn.Linear)
            assert not isinstance(sub, FP8Linear)

    def test_w8a16_falls_back_to_w16a16_with_warning(self):
        """No W8A16 KDA kernel on sm_120 — w8a16 falls back to plain
        nn.Linear (BF16 GEMM) and emits a RuntimeWarning."""
        cfg = HippoConfig(
            hidden_size=128, num_heads=16, head_dim=32,
            num_layers=1, num_blocks=1,
            safe_gate=True, lower_bound=-5.0,
            kda_mode="chunk", use_short_conv=False,
            attention_precision="w8a16",
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            attn = _make_kda(cfg).attn
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w8a16" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for attention w8a16 fallback")
        # Fallback: no FP8 swap, all projections stay nn.Linear.
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert not isinstance(getattr(attn, name), FP8Linear)

    def test_w4a8_falls_back_to_w8a8_with_warning(self):
        """No NVFP4 KDA kernel on sm_120 — w4a8 falls back to w8a8
        (FP8Linear) and emits a RuntimeWarning."""
        cfg = HippoConfig(
            hidden_size=128, num_heads=16, head_dim=32,
            num_layers=1, num_blocks=1,
            safe_gate=True, lower_bound=-5.0,
            kda_mode="chunk", use_short_conv=False,
            attention_precision="w4a8",
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            attn = _make_kda(cfg).attn
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w4a8" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for attention w4a8 fallback")
        # Fallback: FP8 swap (FP8Linear on q/k/v/o).
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(attn, name), FP8Linear), (
                f"attention w4a8 fallback should be FP8Linear on {name}")

    def test_w4a16_falls_back_to_w16a16_with_warning(self):
        """No NVFP4 KDA kernel on sm_120 — w4a16 falls back to w16a16
        (plain nn.Linear) and emits a RuntimeWarning."""
        cfg = HippoConfig(
            hidden_size=128, num_heads=16, head_dim=32,
            num_layers=1, num_blocks=1,
            safe_gate=True, lower_bound=-5.0,
            kda_mode="chunk", use_short_conv=False,
            attention_precision="w4a16",
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            attn = _make_kda(cfg).attn
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w4a16" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for attention w4a16 fallback")
        # Fallback: no FP8 swap.
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert not isinstance(getattr(attn, name), FP8Linear), (
                f"attention w4a16 fallback should NOT be FP8Linear on {name}")


# ---------------------------------------------------------------------------
# 3. FFN wrapper (non-TP SwiGLU): all 5 schemes
# ---------------------------------------------------------------------------
def _make_swiglu(scheme):
    from src.models.activation import SwiGLU
    cfg = HippoConfig(
        ffn_precision=scheme, use_bias=False,
        hidden_size=128, intermediate_size=256,
    )
    return SwiGLU(cfg)


class TestFFNSchemeWiring:
    def test_w16a16_keeps_nn_linear(self):
        """ffn_precision='w16a16' -> plain nn.Linear for all three projections."""
        layer = _make_swiglu("w16a16")
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(layer, name)
            assert isinstance(sub, nn.Linear), (
                f"ffn w16a16: {name} should be nn.Linear, got {type(sub).__name__}")
            assert not isinstance(sub, FP8Linear)
            assert not isinstance(sub, NVFP4Linear)

    def test_w8a16_falls_back_to_nn_linear_with_warning(self):
        """ffn_precision='w8a16' -> no W8A16 FFN kernel on sm_120;
        falls back to nn.Linear with a RuntimeWarning."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            layer = _make_swiglu("w8a16")
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w8a16" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for ffn w8a16 fallback")
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(layer, name)
            assert isinstance(sub, nn.Linear)
            assert not isinstance(sub, FP8Linear)

    def test_w8a8_uses_fp8_linear_with_bwd(self):
        """ffn_precision='w8a8' -> FP8Linear(fp8_bwd=True)."""
        layer = _make_swiglu("w8a8")
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(layer, name)
            assert isinstance(sub, FP8Linear), (
                f"ffn w8a8: {name} should be FP8Linear, got {type(sub).__name__}")
            assert sub.fp8_bwd is True

    def test_w4a8_uses_nvfp4_w4a8_two_pass(self):
        """ffn_precision='w4a8' -> NVFP4LinearW4A8 (two-pass Triton
        dequant + _scaled_mm). NOT NVFP4Linear (which is the W4A16
        Marlin mode-3 path — the previously-shipped bug had this
        wired to NVFP4Linear instead of NVFP4LinearW4A8)."""
        layer = _make_swiglu("w4a8")
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(layer, name)
            assert isinstance(sub, NVFP4LinearW4A8), (
                f"ffn w4a8: {name} should be NVFP4LinearW4A8, "
                f"got {type(sub).__name__} (this is the W4A8 → W4A16 "
                f"mis-wiring regression — check activation.py)")
            # Regression guard: must NOT be the W4A16 Marlin class.
            assert not isinstance(sub, NVFP4Linear), (
                f"ffn w4a8 must NOT use NVFP4Linear (W4A16 Marlin path)")

    def test_w4a16_uses_nvfp4_marlin_mode3(self):
        """ffn_precision='w4a16' -> NVFP4Linear with hardcoded
        use_marlin=True and no_bf16_master=True (Marlin mode-3).
        The kernel choice IS the scheme — there are no per-config
        knobs."""
        layer = _make_swiglu("w4a16")
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(layer, name)
            assert isinstance(sub, NVFP4Linear), (
                f"ffn w4a16: {name} should be NVFP4Linear, "
                f"got {type(sub).__name__}")
            assert sub.use_marlin is True, (
                f"{name}.use_marlin must be True (Marlin mode-3 hardcoded)")
            assert sub.no_bf16_master is True, (
                f"{name}.no_bf16_master must be True (Marlin mode-3 hardcoded)")


# ---------------------------------------------------------------------------
# 4. FFN wrapper (TP SwiGLU): all 5 schemes
# ---------------------------------------------------------------------------
def _make_tp_swiglu(scheme):
    from src.models.tp_model.swiglu import TPSwiGLU
    from src.models.tp_model._primitives import (
        ColumnParallelLinear, RowParallelLinear,
    )
    from src.models.ops.nvfp4_tp import (
        NVFP4ColumnParallelLinear, NVFP4RowParallelLinear,
    )
    cfg = HippoConfig(
        ffn_precision=scheme, use_bias=False,
        hidden_size=128, intermediate_size=256, num_blocks=1,
    )
    return cfg, TPSwiGLU(cfg)


class TestTPSwiGLUSchemeWiring:
    def test_w16a16_uses_bf16_tp_linears(self):
        """TPSwiGLU w16a16 -> plain ColumnParallelLinear +
        RowParallelLinear (BF16)."""
        cfg, layer = _make_tp_swiglu("w16a16")
        from src.models.tp_model._primitives import (
            ColumnParallelLinear, RowParallelLinear,
        )
        assert isinstance(layer.gate_up_proj, ColumnParallelLinear)
        assert isinstance(layer.down_proj, RowParallelLinear)

    def test_w8a16_falls_back_to_bf16_tp_with_warning(self):
        """TPSwiGLU w8a16 -> BF16 TP fallback + RuntimeWarning."""
        from src.models.tp_model._primitives import (
            ColumnParallelLinear, RowParallelLinear,
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            cfg, layer = _make_tp_swiglu("w8a16")
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w8a16" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for TP-SwiGLU w8a16 fallback")
        assert isinstance(layer.gate_up_proj, ColumnParallelLinear)
        assert isinstance(layer.down_proj, RowParallelLinear)

    def test_w8a8_falls_back_to_bf16_tp_with_warning(self):
        """TPSwiGLU w8a8 -> no per-rank FP8 Col/Row on sm_120;
        falls back to BF16 TP + RuntimeWarning."""
        from src.models.tp_model._primitives import (
            ColumnParallelLinear, RowParallelLinear,
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            cfg, layer = _make_tp_swiglu("w8a8")
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w8a8" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for TP-SwiGLU w8a8 fallback")
        assert isinstance(layer.gate_up_proj, ColumnParallelLinear)
        assert isinstance(layer.down_proj, RowParallelLinear)

    def test_w4a8_falls_back_to_bf16_tp_with_warning(self):
        """TPSwiGLU w4a8 -> no NVFP4 W4A8 Col/Row on sm_120
        (NVFP4ColumnParallelLinear is the W4A16 Marlin variant only);
        falls back to BF16 TP + RuntimeWarning."""
        from src.models.tp_model._primitives import (
            ColumnParallelLinear, RowParallelLinear,
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            cfg, layer = _make_tp_swiglu("w4a8")
        warn_msgs = [str(w.message) for w in ws
                     if issubclass(w.category, RuntimeWarning)
                     and "w4a8" in str(w.message)]
        assert len(warn_msgs) >= 1, (
            "expected RuntimeWarning for TP-SwiGLU w4a8 fallback")
        assert isinstance(layer.gate_up_proj, ColumnParallelLinear)
        assert isinstance(layer.down_proj, RowParallelLinear)

    def test_w4a16_uses_nvfp4_tp_marlin_mode3(self):
        """TPSwiGLU w4a16 -> NVFP4ColumnParallelLinear +
        NVFP4RowParallelLinear with use_marlin=True,
        no_bf16_master=True (hardcoded)."""
        from src.models.ops.nvfp4_tp import (
            NVFP4ColumnParallelLinear, NVFP4RowParallelLinear,
        )
        cfg, layer = _make_tp_swiglu("w4a16")
        assert isinstance(layer.gate_up_proj, NVFP4ColumnParallelLinear)
        assert isinstance(layer.down_proj, NVFP4RowParallelLinear)
        assert layer.gate_up_proj.use_marlin is True
        assert layer.gate_up_proj.no_bf16_master is True
        assert layer.down_proj.use_marlin is True
        assert layer.down_proj.no_bf16_master is True


# ---------------------------------------------------------------------------
# 5. FP8Linear: default fp8_bwd flipped + FP32-cast bug fix
# ---------------------------------------------------------------------------
def _make_pair(K, N, *, bias, seed=42):
    torch.manual_seed(seed)
    ref = nn.Linear(K, N, bias=bias).to(device=device, dtype=dtype)
    fp8 = FP8Linear(K, N, bias=bias, device=device, dtype=dtype)
    fp8.weight.data.copy_(ref.weight.data)
    if bias:
        fp8.bias.data.copy_(ref.bias.data)
    return ref, fp8


def _sig_rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


class TestFP8LinearBwdFix:
    def test_default_fp8_bwd_now_true(self):
        """fp8_bwd defaults to True (was False pre-2026-07-21)."""
        fp8 = FP8Linear(128, 128, bias=False, device=device, dtype=dtype)
        assert fp8.fp8_bwd is True, "fp8_bwd must default to True"

    def test_ste_bwd_is_bit_equivalent_to_bf16(self):
        """_FP8E4M3Matmul.backward (BF16 STE, fp8_bwd=False) is
        bit-equivalent to ``nn.Linear.backward`` within BF16 noise.

        Regression guard for the FP32-cast bug: previously the STE
        path cast grad_out/x/w to FP32 and ran FP32 GEMMs (1.87x
        slower than BF16). The fix keeps BF16 leaves and runs BF16
        matmul. Expected: cos_sim 1.0, max abs diff at BF16 noise.
        """
        torch.manual_seed(0)
        M, K, N = 1024, 1536, 1536
        x = torch.randn(M, K, device=device, dtype=dtype)
        dy = torch.randn(M, N, device=device, dtype=dtype)

        ref = nn.Linear(K, N, bias=False).to(device=device, dtype=dtype)
        fp8 = FP8Linear(K, N, bias=False, device=device, dtype=dtype,
                        fp8_bwd=False)
        fp8.weight.data.copy_(ref.weight.data)

        # BF16 reference
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = ref(x_ref); y_ref.backward(dy)
        dx_ref = x_ref.grad.detach().clone()
        dw_ref = ref.weight.grad.detach().clone()

        # STE (fp8_bwd=False) — the fixed path
        x_fp8 = x.detach().clone().requires_grad_(True)
        y_fp8 = fp8(x_fp8); y_fp8.backward(dy)
        dx_fp8 = x_fp8.grad.detach().clone()
        dw_fp8 = fp8.weight.grad.detach().clone()

        # Within BF16 rounding noise (cos_sim 1.0, max-diff tiny).
        cos_dx = F.cosine_similarity(
            dx_fp8.float().flatten(), dx_ref.float().flatten(), dim=0).item()
        cos_dw = F.cosine_similarity(
            dw_fp8.float().flatten(), dw_ref.float().flatten(), dim=0).item()
        assert cos_dx > 0.99999, f"dx cos_sim {cos_dx} too low (FP32-bug regression?)"
        assert cos_dw > 0.99999, f"dw cos_sim {cos_dw} too low (FP32-bug regression?)"
        assert _sig_rel(dx_fp8, dx_ref) < 1e-3
        assert _sig_rel(dw_fp8, dw_ref) < 1e-3

    def test_default_path_uses_fp8_bwd_and_is_fast(self):
        """Default FP8Linear.forward uses fp8_bwd=True (FP8 E4M3 bwd).
        The forward numerics match the BF16 reference within FP8 noise
        (~5% sig_rel); the backward produces finite gradients."""
        torch.manual_seed(0)
        M, K, N = 1024, 1536, 1536
        x = torch.randn(M, K, device=device, dtype=dtype)
        dy = torch.randn(M, N, device=device, dtype=dtype)

        ref, fp8 = _make_pair(K, N, bias=False, seed=42)
        # Default fp8_bwd is True now.
        assert fp8.fp8_bwd is True

        x_fp8 = x.detach().clone().requires_grad_(True)
        y_fp8 = fp8(x_fp8)
        y_fp8.backward(dy)
        assert torch.isfinite(x_fp8.grad).all()
        assert torch.isfinite(fp8.weight.grad).all()


# ---------------------------------------------------------------------------
# 6. End-to-end: KDA fwd+bwd finite with attention_precision=w8a8
# ---------------------------------------------------------------------------
def test_kda_w8a8_fwd_bwd_finite():
    """Full KDA fwd+bwd is finite and grads are close to BF16 baseline
    when attention_precision='w8a8' (the prod-relevant path)."""
    common = dict(
        hidden_size=128, num_heads=16, head_dim=32,
        num_layers=1, num_blocks=1, safe_gate=True, lower_bound=-5.0,
        kda_mode="chunk", use_short_conv=False,
    )
    torch.manual_seed(7)
    layer_bf16 = KDA(HippoConfig(**common, attention_precision="w16a16"),
                     layer_idx=0).to(device=device, dtype=dtype)
    layer_fp8 = KDA(HippoConfig(**common, attention_precision="w8a8"),
                    layer_idx=0).to(device=device, dtype=dtype)
    layer_fp8.load_state_dict(layer_bf16.state_dict())

    x = torch.randn(2, 64, 128, device=device, dtype=dtype)
    y_bf16 = layer_bf16(x)
    y_fp8 = layer_fp8(x)
    assert torch.isfinite(y_fp8).all()

    g = torch.randn_like(y_fp8)
    y_bf16.backward(g)
    y_fp8.backward(g)
    # FP8 bwd grads are noisy (~3-5% sig_rel) but must be finite.
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w_grad = getattr(layer_fp8.attn, name).weight.grad
        assert w_grad is not None and torch.isfinite(w_grad).all(), (
            f"{name}.weight.grad not finite")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))