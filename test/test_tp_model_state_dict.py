"""Pin the contract that :class:`TPHippoModel.state_dict()` returns
every trainable parameter, not an empty dict.

Background
----------
Before this fix, ``TPHippoModel`` stored its per-device submodules
in four plain Python ``dict[int, ...]`` containers
(``replicated_per_device``, ``layers_per_device``,
``lm_head_per_device``, ``fused_lm_ce_per_device``). PyTorch's
``state_dict()`` walks ``_modules``, not Python ``__dict__``, so
plain dicts are invisible — the saved state_dict was always
empty and every checkpoint was silently lost.

The fix replaces those four containers with ``nn.ModuleDict`` so
they appear in ``_modules`` and ``state_dict()`` recursively visits
them. Side effect: state_dict keys are auto-prefixed with
``replicated_per_device.<d>.`` / ``layers_per_device.<d>.`` /
``lm_head_per_device.<d>.`` / ``fused_lm_ce_per_device.<d>.``,
which matches the heuristic in
:func:`scripts.eval_server._is_tp_state_dict` so resume-from-
checkpoint and TP eval keep working without key remapping.

These tests live here so a future refactor that accidentally
re-introduces plain dicts gets caught immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models import HippoConfig  # noqa: E402
from src.models.tp_model._primitives import init_tp, shutdown_tp  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tp_group():
    """Initialise a TP group with world=1 (single device)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for TPHippoModel construction")
    init_tp(world_size=1, devices=[0], backend="gloo")
    yield
    shutdown_tp()


def _tiny_config() -> HippoConfig:
    """Minimal config that still exercises every submodule:
    - 2 layers / 2 blocks ⇒ block_size=1
    - 2 heads, head_dim 32 ⇒ KDA q/k/v/o projections all exist
    - 2 < num_blocks * 2 means block-boundary attn_res fires
    """
    return HippoConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_blocks=2,
        num_heads=2,
        head_dim=32,
        intermediate_size=128,
    )


# --------------------------------------------------------------------------- #
# Tests.                                                                      #
# --------------------------------------------------------------------------- #
def test_tp_model_state_dict_is_nonempty(tp_group):
    """Construct a tiny TPHippoModel and verify ``model.state_dict()``
    returns more than zero tensors. Pre-fix: returned 0 because
    submodules lived in plain dicts."""
    torch.manual_seed(0)
    from src.models.tp_model import TPHippoModel  # noqa: E402

    model = TPHippoModel(_tiny_config(), devices=[0], dtype=torch.float32)
    sd = model.state_dict()

    assert len(sd) > 0, (
        f"TPHippoModel.state_dict() returned {len(sd)} tensors; "
        "expected all trainable parameters. This means checkpoint "
        "save/load silently drops every model weight."
    )


def test_tp_model_state_dict_roundtrips_through_save_load(tp_group, tmp_path):
    """Save and reload the state_dict into a fresh model; tensors must
    match. Pins the contract that the state_dict is fully recoverable."""
    torch.manual_seed(0)
    from src.models.tp_model import TPHippoModel  # noqa: E402

    model = TPHippoModel(_tiny_config(), devices=[0], dtype=torch.float32)
    sd = model.state_dict()
    assert len(sd) > 0, "state_dict must be non-empty"

    # Reload into a freshly-constructed model with matching config.
    fresh = TPHippoModel(_tiny_config(), devices=[0], dtype=torch.float32)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert len(missing) == 0, f"reload dropped keys: {missing[:5]}"
    assert len(unexpected) == 0, (
        f"reload produced unexpected keys: {unexpected[:5]}"
    )

    # And the post-reload tensor values must match exactly.
    reloaded = fresh.state_dict()
    assert set(sd.keys()) == set(reloaded.keys())
    for k in sd:
        assert torch.equal(sd[k], reloaded[k]), (
            f"value mismatch on {k} after reload"
        )


def test_tp_model_state_dict_keys_use_module_dict_prefixes(tp_group):
    """The state_dict keys must be auto-prefixed with the
    ``nn.ModuleDict`` container names. eval_server's
    ``_is_tp_state_dict`` sniff relies on these exact prefixes
    to route a TP-shaped checkpoint to the TPHippoModel loader."""
    torch.manual_seed(0)
    from src.models.tp_model import TPHippoModel  # noqa: E402

    model = TPHippoModel(_tiny_config(), devices=[0], dtype=torch.float32)
    keys = list(model.state_dict().keys())

    # At least one key per known container; the per-device index is "0"
    # for a single-rank TP group.
    expected_prefixes = (
        "replicated_per_device.0.",
        "layers_per_device.0.",
    )
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), (
            f"no state_dict key starts with {prefix!r}; "
            f"eval_server TP sniff would route this checkpoint to "
            f"the plain HippoModel loader and break resume"
        )


def test_tp_model_state_dict_uses_module_dict_not_plain_dict(tp_group):
    """Direct regression check on the bug itself: the four per-device
    containers must be ``nn.ModuleDict`` instances, not plain
    ``dict``. A future refactor that re-introduces a plain dict
    silently breaks checkpoint save/load — this test makes the
    regression loud."""
    import torch.nn as nn  # noqa: E402
    from src.models.tp_model import TPHippoModel  # noqa: E402

    model = TPHippoModel(_tiny_config(), devices=[0], dtype=torch.float32)
    for attr in (
        "replicated_per_device",
        "layers_per_device",
        "lm_head_per_device",
        "fused_lm_ce_per_device",
    ):
        container = getattr(model, attr)
        assert isinstance(container, nn.ModuleDict), (
            f"{attr} is {type(container).__name__}, expected "
            f"nn.ModuleDict. Plain dicts are invisible to "
            f"state_dict() — this is the bug we're guarding against."
        )
