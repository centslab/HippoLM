"""Precision configuration for HippoLM training.

Defines the per-tensor-class dtype. The yml schema in
:file:`configs/base.yml` is::

    precision:
      model_weights: { dtype: bf16 }
      gradients:     { dtype: bf16 }
      activations:   { dtype: bf16 }
      muon_momentum: { dtype: bf16 }
      adamw_m:       { dtype: bf16 }
      adamw_v:       { dtype: bf16}

The keys under each entry:

  - ``dtype``: one of ``fp32``, ``fp16``, ``bf16``.

The six tensor classes (one :class:`TensorPrecision` each):

  - ``model_weights``  — the GPU-side trainable params (FFN,
    KDA, AttnRes, embed, lm_head, RMSNorm). BF16 is the
    default (V100 has FP16 tensor cores; BF16 is also fine on
    5060Ti / H100).
  - ``gradients``      — **DEPRECATED in the merged-accumulator
    design.** Previously the dtype of the CPU pinned ``s.accum``
    grad accumulator. Now there is no separate accumulator:
    ``s.m`` (AdamW) / ``s.mom_buf`` (Muon) double as the
    accumulator AND the optimizer's first-moment / momentum
    feed. The cast target in the offload hook follows the
    accumulator's storage dtype (``adamw_m`` / ``muon_momentum``)
    directly. The field is kept in the schema for backwards
    compatibility with existing yml files.
  - ``activations``    — the dtype used for the per-layer
    forward activations under ``torch.amp.autocast``. BF16 is
    the default. Allowed values: ``fp32`` (autocast is disabled
    — pure FP32 forward), ``fp16``, ``bf16``.
  - ``muon_momentum``  — Muon's SGD momentum (the buffer that
    feeds Newton-Schulz, which now also doubles as the grad
    accumulator with mu=1 accumulation). bf16 / fp16 / fp32
    are full-precision storage with no requant. The canonical
    production value is ``bf16`` — quantized storage formats
    (``int8`` per-row, ``mxfp8`` per-block) were removed
    on 2026-07-12 after long training runs showed
    quantization-error accumulation that destabilized
    optimization. See ``docs/optimizer_layout.md`` for the
    post-removal layout.
  - ``adamw_m``        — AdamW's first moment AND the grad
    accumulator (merged). BF16 is the default (magnitude
    bounded, no precision concern).
  - ``adamw_v``        — AdamW's second moment. BF16 is the
    default (BF16's 8-bit exponent keeps ``v = g²`` from
    underflowing for typical grad magnitudes; FP16's 5-bit
    exponent would not).

Construction: prefer :meth:`PrecisionConfig.from_dict` for yml
loading. Direct dataclass construction with all 6 fields is
also valid (used by tests and by the CLI default path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class DType(str, Enum):
    """Storage precision for a tensor class.

    String-valued so that yml-sourced values (``"fp16"``,
    ``"bf16"``, etc.) parse directly without a separate
    normalization step.

    Only floating dtypes are supported as of 2026-07-12:
    ``int8`` / ``int4`` / ``mxfp8`` storage were removed
    after long-training instability was observed. See
    ``docs/optimizer_layout.md``.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def to_torch(self) -> torch.dtype:
        """Return the corresponding :class:`torch.dtype`."""
        return {
            DType.FP32: torch.float32,
            DType.FP16: torch.float16,
            DType.BF16: torch.bfloat16,
        }[self]


@dataclass
class TensorPrecision:
    """Precision spec for a single tensor class.

    Attributes:
        dtype:       Storage precision (see :class:`DType`).

    Only ``dtype`` is meaningful; the legacy ``scale`` /
    ``block_size`` fields were removed along with quantized
    storage (see module docstring). Unknown keys in the
    input yml dict are silently ignored for forward
    compatibility.
    """

    dtype: DType = DType.BF16

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            self.dtype = DType(self.dtype)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to a yml-shaped dict (for logging / tests)."""
        return {"dtype": self.dtype.value}


# --------------------------------------------------------------------------- #
# Top-level config: one TensorPrecision per tensor class.                     #
# --------------------------------------------------------------------------- #
@dataclass
class PrecisionConfig:
    """Per-tensor-class precision for the whole training run.

    Defaults match the canonical yml in :file:`configs/base.yml`:
    BF16 for everything.

    Use :meth:`from_dict` to construct from a yml payload
    (e.g. the value of the ``precision:`` key). Direct
    construction with overridden fields is also valid; tests
    and the CLI default path use that.
    """

    model_weights: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    gradients: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    activations: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    muon_momentum: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    adamw_m: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    adamw_v: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )

    @property
    def autocast_enabled(self) -> bool:
        """Whether autocast should be enabled for this config.

        FP16 / BF16 activations: autocast on (tensor-core matmul).
        FP32 activations: autocast off (pure FP32 forward; enabling
        autocast with ``dtype=torch.float32`` is a no-op, so we
        skip the autocast context entirely).
        """
        return self.activations.dtype != DType.FP32

    @property
    def autocast_dtype(self) -> "torch.dtype":
        """The ``dtype`` kwarg to pass to ``torch.amp.autocast``.

        Returns the float16/bfloat16 torch dtype derived from
        :attr:`activations`. For FP32 activations the returned
        dtype is unused (autocast is disabled); we still return
        a sensible value (``torch.float32``) so a caller that
        ignores :attr:`autocast_enabled` does not crash.
        """
        return self.activations.dtype.to_torch()

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PrecisionConfig":
        """Build from a yml dict (e.g. ``{"model_weights": {...}, ...}``).

        Missing keys fall through to the dataclass defaults. ``None``
        or empty input returns the default config unchanged.
        Unknown keys are passed through silently for forward
        compatibility (we add new tensor classes by adding fields
        to this dataclass, not by changing the schema).

        Unknown subkeys inside a known tensor class are also
        ignored (the legacy quantized-storage fields ``scale`` /
        ``block_size`` are silently dropped — a yml carrying
        them is treated as "use the default storage").
        """
        if not d:
            return cls()
        kwargs: Dict[str, Any] = {}
        for field_name in (
            "model_weights", "gradients", "activations",
            "muon_momentum", "adamw_m", "adamw_v",
        ):
            if field_name in d and isinstance(d[field_name], dict):
                tp_dict = d[field_name]
                tp = TensorPrecision(
                    dtype=tp_dict.get("dtype", "bf16"),
                )
                kwargs[field_name] = tp
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a yml-shaped dict (for logging)."""
        return {
            "model_weights": self.model_weights.to_dict(),
            "gradients": self.gradients.to_dict(),
            "activations": self.activations.to_dict(),
            "muon_momentum": self.muon_momentum.to_dict(),
            "adamw_m": self.adamw_m.to_dict(),
            "adamw_v": self.adamw_v.to_dict(),
        }
