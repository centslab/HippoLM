"""Precision configuration for HippoLM training.

Defines the per-tensor-class dtype and (for quantized types)
the scale mode. The yml schema in :file:`configs/base.yml` is::

    precision:
      model_weights: { dtype: fp16 }
      gradients:     { dtype: bf16 }
      activations:   { dtype: fp16 }
      muon_momentum: { dtype: int8, scale: per-channel }
      adamw_m:       { dtype: bf16 }
      adamw_v:       { dtype: bf16 }

The keys under each entry:

  - ``dtype``: one of ``fp32``, ``fp16``, ``bf16``, ``int8``, ``int4``.
    For ``activations``, only ``fp32``, ``fp16``, ``bf16`` are
    accepted — integer dtypes are rejected because activations
    are continuous-valued and autocast does not consume an
    int dtype anyway.
  - ``scale``: required for ``int8`` / ``int4``; one of ``no``,
    ``tensor``, ``per-channel``, ``block``. Ignored for floating
    dtypes (always stored as ``None``).
  - ``block_size``: required when ``scale == 'block'``; ignored
    otherwise.

The six tensor classes (one :class:`TensorPrecision` each):

  - ``model_weights``  — the GPU-side trainable params (FFN,
    GDN2, AttnRes, embed, lm_head, RMSNorm). FP16 is the
    default (V100 has FP16 tensor cores; BF16 is also fine on
    5060Ti / H100).
  - ``gradients``      — the CPU pinned accumulator
    (``s.accum``) into which the GPU ``.grad`` is DMA'd. BF16
    is the default (FP32-like exponent range, no overflow on
    the transfer).
  - ``activations``    — the dtype used for the per-layer
    forward activations under ``torch.amp.autocast``. FP16 is
    the default (matches the canonical model_weights setting;
    the autocast layer internally promotes matmul/conv to the
    activation dtype on tensor cores). Allowed values: ``fp32``
    (autocast is disabled — pure FP32 forward), ``fp16``,
    ``bf16``. Integer dtypes are rejected.
  - ``muon_momentum``  — Muon's SGD momentum (the buffer that
    feeds Newton-Schulz). int8 + per-channel is the default
    (halves CPU RAM vs FP16). int4 further halves but loses
    precision.
  - ``adamw_m``        — AdamW's first moment. BF16 is the
    default (magnitude bounded, no precision concern).
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
    ``"int8"``, etc.) parse directly without a separate
    normalization step.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def is_integer(self) -> bool:
        return self in (DType.INT8, DType.INT4)

    @property
    def is_floating(self) -> bool:
        return not self.is_integer

    def to_torch(self) -> torch.dtype:
        """Return the corresponding :class:`torch.dtype`.

        ``INT4`` is not a native torch dtype; the optimizer stores
        int4-packed bytes in an int8 buffer (2 elements per byte).
        :class:`CPUMuon` is responsible for the packing scheme;
        we just hand back ``int8`` so the storage tensor
        construction is uniform.
        """
        return {
            DType.FP32: torch.float32,
            DType.FP16: torch.float16,
            DType.BF16: torch.bfloat16,
            DType.INT8: torch.int8,
            DType.INT4: torch.int8,
        }[self]


class ScaleMode(str, Enum):
    """Quantization scale granularity.

    String-valued for yml compatibility.
    """

    NO = "no"
    TENSOR = "tensor"
    PER_CHANNEL = "per-channel"
    BLOCK = "block"


@dataclass
class TensorPrecision:
    """Precision spec for a single tensor class.

    Attributes:
        dtype:       Storage precision (see :class:`DType`).
        scale:       Quantization scale mode; ``None`` for fp* and
                     required for int8/int4.
        block_size:  Elements per scale block, only when
                     ``scale == ScaleMode.BLOCK``.

    Invariants enforced in :meth:`__post_init__`:

    - ``int8`` / ``int4`` must specify a ``scale``.
    - ``scale == 'block'`` must specify a positive ``block_size``.
    - Floating dtypes ignore ``scale`` and ``block_size``
      (silently cleared to ``None``).
    """

    dtype: DType = DType.BF16
    scale: Optional[ScaleMode] = None
    block_size: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            self.dtype = DType(self.dtype)
        if isinstance(self.scale, str):
            self.scale = ScaleMode(self.scale)
        if self.dtype.is_integer:
            if self.scale is None:
                raise ValueError(
                    f"precision: dtype={self.dtype.value} requires a 'scale'"
                    f" mode (one of {[s.value for s in ScaleMode]})"
                )
            if self.scale == ScaleMode.BLOCK:
                if self.block_size is None:
                    raise ValueError(
                        "precision: scale='block' requires 'block_size'"
                    )
                if self.block_size <= 0:
                    raise ValueError(
                        f"precision: block_size must be positive,"
                        f" got {self.block_size}"
                    )
        else:
            # fp* dtypes: silently drop scale / block_size to keep
            # the config self-consistent. This means a user can
            # write ``adamw_m: { dtype: bf16, scale: per-channel }``
            # without us complaining — the scale is just ignored.
            self.scale = None
            self.block_size = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to a yml-shaped dict (for logging / tests)."""
        out: Dict[str, Any] = {"dtype": self.dtype.value}
        if self.scale is not None:
            out["scale"] = self.scale.value
        if self.block_size is not None:
            out["block_size"] = self.block_size
        return out


# --------------------------------------------------------------------------- #
# Top-level config: one TensorPrecision per tensor class.                     #
# --------------------------------------------------------------------------- #
@dataclass
class PrecisionConfig:
    """Per-tensor-class precision for the whole training run.

    Defaults match the canonical yml in :file:`configs/base.yml`:
    FP16 weights, BF16 gradients, FP16 activations (autocast),
    int8 + per-channel Muon momentum, BF16 AdamW m / v.

    Use :meth:`from_dict` to construct from a yml payload
    (e.g. the value of the ``precision:`` key). Direct
    construction with overridden fields is also valid; tests
    and the CLI default path use that.
    """

    model_weights: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.FP16)
    )
    gradients: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    activations: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.FP16)
    )
    muon_momentum: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(
            dtype=DType.INT8, scale=ScaleMode.PER_CHANNEL,
        )
    )
    adamw_m: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )
    adamw_v: TensorPrecision = field(
        default_factory=lambda: TensorPrecision(dtype=DType.BF16)
    )

    def __post_init__(self) -> None:
        # Activations are continuous-valued and consumed by
        # ``torch.amp.autocast``, which only accepts fp16 / bf16
        # (fp32 is implemented by *disabling* autocast). Reject
        # integer dtypes here so a misconfigured yml fails at
        # config-parse time rather than at the first forward
        # pass.
        if self.activations.dtype.is_integer:
            raise ValueError(
                f"precision.activations: dtype={self.activations.dtype.value}"
                f" is not allowed (activations must be a floating dtype;"
                f" use one of fp32, fp16, bf16)."
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
                    scale=tp_dict.get("scale"),
                    block_size=tp_dict.get("block_size"),
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
