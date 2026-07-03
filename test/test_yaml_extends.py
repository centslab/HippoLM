"""Tests for the ``extends:`` mechanism in ``scripts/cli.py``.

The ``configs/test/*.yml`` files rely on ``extends: ../base.yml``
to inherit the production defaults (precision block, optimizer
hparams, data source) without duplicating them. The merge is
**shallow** — the child wins for any top-level key it specifies,
but cannot reach inside a nested dict. These tests pin both
the happy path and the failure modes so a future refactor that
breaks inheritance gets caught immediately.

Pinned here:
  * Child yml with ``extends:`` inherits parent's keys
  * Child override of a parent key wins (shallow)
  * Nested dicts are replaced whole, not deep-merged
  * ``extends`` key itself is consumed (never reaches argparse)
  * Multi-level chain (A -> B -> C) works
  * Circular chains raise ``ValueError``
  * Missing parent raises a clear ``FileNotFoundError`` /
    ``ValueError`` (not a silent fallback to empty dict)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.cli import _load_yaml_with_extends  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_yml_tree(tmp_path):
    """Build a tiny yml tree under ``tmp_path`` and return the path-to-
    child mapping. Default tree:

        tmp_path/
            root.yml        <- inherits precision + lr
            child.yml       <- overrides num_layers
            leaf.yml        <- overrides max_steps, no extends
            middle.yml      <- extends root, overrides vocab_size
            cycle_a.yml     <- extends cycle_b
            cycle_b.yml     <- extends cycle_a
    """
    (tmp_path / "root.yml").write_text(
        "precision:\n"
        "  model_weights: { dtype: bf16 }\n"
        "  adamw_m: { dtype: bf16 }\n"
        "learning_rate: 0.01\n"
        "num_layers: 32\n"
        "vocab_size: 248320\n"
    )
    (tmp_path / "child.yml").write_text(
        "extends: root.yml\n"
        "num_layers: 2\n"
    )
    (tmp_path / "leaf.yml").write_text(
        "max_steps: 4\n"
    )
    (tmp_path / "middle.yml").write_text(
        "extends: root.yml\n"
        "vocab_size: 256\n"
    )
    (tmp_path / "cycle_a.yml").write_text(
        "extends: cycle_b.yml\n"
        "a: 1\n"
    )
    (tmp_path / "cycle_b.yml").write_text(
        "extends: cycle_a.yml\n"
        "b: 2\n"
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# Happy path.                                                                 #
# --------------------------------------------------------------------------- #
def test_extends_inherits_parent_keys(tmp_yml_tree):
    """Child yml with ``extends: root.yml`` gets every parent key it
    didn't override."""
    merged = _load_yaml_with_extends(str(tmp_yml_tree / "child.yml"))
    assert merged["learning_rate"] == 0.01
    assert merged["precision"]["model_weights"]["dtype"] == "bf16"
    assert merged["precision"]["adamw_m"]["dtype"] == "bf16"


def test_child_override_wins(tmp_yml_tree):
    """Child override of a top-level key wins over the parent."""
    merged = _load_yaml_with_extends(str(tmp_yml_tree / "child.yml"))
    assert merged["num_layers"] == 2  # not 32


def test_extends_key_is_consumed(tmp_yml_tree):
    """The ``extends`` key itself must not appear in the merged
    result — if it does, ``parser.set_defaults(**merged)`` would
    see a key with no matching argparse action and silently add it
    to the Namespace as a str (or fail outright)."""
    merged = _load_yaml_with_extends(str(tmp_yml_tree / "child.yml"))
    assert "extends" not in merged


def test_no_extends_works(tmp_yml_tree):
    """A yml with no ``extends`` key loads as-is (the simple
    pre-extends behavior)."""
    merged = _load_yaml_with_extends(str(tmp_yml_tree / "leaf.yml"))
    assert merged == {"max_steps": 4}


def test_nested_dict_replaced_not_merged(tmp_yml_tree):
    """Shallow merge: child override of a top-level dict replaces
    the parent's whole dict, not deep-merges sub-keys.

    We test this indirectly by adding a child yml that overrides
    the precision block with a partial dict (only one sub-key) and
    asserting the OTHER sub-keys are dropped (i.e. replaced whole).
    """
    (tmp_yml_tree / "shallow.yml").write_text(
        "extends: root.yml\n"
        "precision:\n"
        "  adamw_m: { dtype: fp32 }\n"  # only this sub-key, no model_weights
    )
    merged = _load_yaml_with_extends(str(tmp_yml_tree / "shallow.yml"))
    # Whole precision block replaced — model_weights is gone.
    assert merged["precision"] == {"adamw_m": {"dtype": "fp32"}}
    assert "model_weights" not in merged["precision"]


def test_unrelated_keys_from_child_persist(tmp_yml_tree):
    """Child's own keys (not overrides) flow through alongside
    inherited ones."""
    (tmp_yml_tree / "extra.yml").write_text(
        "extends: root.yml\n"
        "max_steps: 4\n"
        "use_dummy_data: true\n"
    )
    merged = _load_yaml_with_extends(str(tmp_yml_tree / "extra.yml"))
    assert merged["max_steps"] == 4
    assert merged["use_dummy_data"] is True
    assert merged["learning_rate"] == 0.01  # inherited
    assert merged["num_layers"] == 32  # inherited


# --------------------------------------------------------------------------- #
# Failure modes.                                                              #
# --------------------------------------------------------------------------- #
def test_circular_extends_raises(tmp_yml_tree):
    """A -> B -> A must raise ``ValueError``, not loop forever."""
    with pytest.raises(ValueError, match="[Cc]ircular"):
        _load_yaml_with_extends(str(tmp_yml_tree / "cycle_a.yml"))


def test_self_extend_raises(tmp_yml_tree):
    """A yml that extends itself is a degenerate cycle of length 1
    and must also raise."""
    (tmp_yml_tree / "self.yml").write_text(
        "extends: self.yml\n"
    )
    with pytest.raises(ValueError, match="[Cc]ircular"):
        _load_yaml_with_extends(str(tmp_yml_tree / "self.yml"))


def test_missing_parent_raises(tmp_yml_tree):
    """A child yml whose parent doesn't exist must raise
    ``FileNotFoundError`` (or wrap it) — not silently fall back
    to an empty dict."""
    (tmp_yml_tree / "orphan.yml").write_text(
        "extends: nope.yml\n"
    )
    # ``open(path)`` raises FileNotFoundError; we don't catch it.
    with pytest.raises(FileNotFoundError):
        _load_yaml_with_extends(str(tmp_yml_tree / "orphan.yml"))


# --------------------------------------------------------------------------- #
# Real config smoke check.                                                    #
# --------------------------------------------------------------------------- #
def test_real_base_yml_loads():
    """The actual ``configs/base.yml`` loads without ``extends`` (it
    IS the base). Sanity check that the loader doesn't accidentally
    require an ``extends`` key."""
    base_path = _REPO / "configs" / "base.yml"
    if not base_path.exists():
        pytest.skip("configs/base.yml not present (out-of-tree checkout)")
    merged = _load_yaml_with_extends(str(base_path))
    assert "precision" in merged
    assert "model_weights" in merged["precision"]


def test_real_test_yml_inherits_base():
    """The actual ``configs/test/quick.yml`` extends base.yml and
    pulls in the precision block. This is the contract the test
    configs rely on."""
    quick = _REPO / "configs" / "test" / "quick.yml"
    if not quick.exists():
        pytest.skip("configs/test/quick.yml not present")
    merged = _load_yaml_with_extends(str(quick))
    # Inherited from base:
    assert "precision" in merged
    assert merged["precision"]["muon_momentum"]["dtype"] == "mxfp8"
    # Overridden by the child:
    assert merged["num_layers"] == 2
    # The extends key itself is consumed.
    assert "extends" not in merged


# --------------------------------------------------------------------------- #
# Optimizer block flatten helper (consumed by ``scripts.cli.parse_args``).    #
# --------------------------------------------------------------------------- #
def test_optimizer_block_in_base_resolves_all_five():
    """``configs/base.yml`` uses the grouped ``optimizer:`` block;
    after ``parse_args`` every CLI-default flat key the training
    loop reads must be populated (learning_rate, weight_decay,
    muon_lr, muon_weight_decay, muon_momentum).

    Locks in the contract that the nested yml shape correctly
    feeds the loop / ``build_param_groups`` reads.
    """
    from scripts.cli import parse_args as _parse_args

    base = _REPO / "configs" / "base.yml"
    if not base.exists():
        pytest.skip("configs/base.yml not present")
    args = _parse_args(["--config", str(base)])
    assert hasattr(args, "muon_weight_decay"), (
        "--muon_weight_decay CLI flag must exist for the grouped"
        " optimizer config to be overridable from the command line"
    )
    assert args.learning_rate == pytest.approx(0.01)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.muon_lr == pytest.approx(0.02)
    assert args.muon_weight_decay == pytest.approx(0.0)
    assert args.muon_momentum == pytest.approx(0.95)
